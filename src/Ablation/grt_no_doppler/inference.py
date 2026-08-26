#!/usr/bin/env python3
"""Sequence-by-sequence GRT no-Doppler inference on Smoke-Eval.

Launch on one or more GPUs with Accelerate, for example:

    accelerate launch inference.py --config config.yaml

Each Smoke-Eval sequence is processed independently and saved as
``<output_dir>/<sequence>_pred.npy``.  Every file is float32 with shape
``[N, 1, H, W]`` and normalized depth clipped to ``[0, 1]``.
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file
from tqdm import tqdm

from augmentations import dequantize_depth, translate_radar
from dataloader import RiceDataset, create_rice_dataloader
from grt_model import GRTSmall


def batch_radar_to_spectrum(
    radar_amplitude: torch.Tensor, radar_phase: torch.Tensor
) -> torch.Tensor:
    """Build ``[B, 64, 8, 2, 256, 2]`` spectra from Rice tensors."""
    amp = radar_amplitude.permute(0, 1, 3, 2, 4)
    phase = radar_phase.permute(0, 1, 3, 2, 4)
    return torch.stack([amp, phase], dim=-1)


def _resolve_path(config_path: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(config_path)), value)
    )


def _validate_prediction_array(predictions: np.ndarray, sequence: str) -> None:
    if predictions.ndim != 4 or predictions.shape[1] != 1:
        raise RuntimeError(
            f"{sequence}: expected prediction shape [N, 1, H, W], "
            f"got {predictions.shape}"
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{sequence}: predictions contain NaN or Inf")
    if predictions.min() < 0.0 or predictions.max() > 1.0:
        raise RuntimeError(
            f"{sequence}: normalized predictions are outside [0, 1]: "
            f"[{predictions.min()}, {predictions.max()}]"
        )


def _merge_rank_results(
    gather_dir: str,
    sequence: str,
    num_processes: int,
) -> Dict[int, np.ndarray]:
    safe_sequence = sequence.replace("/", "_").replace("\\", "_").lower()
    merged: Dict[int, np.ndarray] = {}
    for rank in range(num_processes):
        rank_path = os.path.join(gather_dir, f"rank_{rank}_{safe_sequence}.pkl")
        with open(rank_path, "rb") as handle:
            rank_results = pickle.load(handle)
        for frame_idx, prediction in rank_results:
            merged.setdefault(int(frame_idx), prediction)
        os.remove(rank_path)
    return merged


def _save_sequence(
    output_dir: str,
    sequence: str,
    frame_predictions: Dict[int, np.ndarray],
    expected_frames: List[int],
    debug: bool,
) -> np.ndarray:
    if not frame_predictions:
        raise RuntimeError(f"{sequence}: inference produced no predictions")

    if not debug:
        missing = [frame for frame in expected_frames if frame not in frame_predictions]
        if missing:
            raise RuntimeError(
                f"{sequence}: missing {len(missing)} predictions "
                f"(first few frame indices: {missing[:5]})"
            )
        ordered_frames = expected_frames
    else:
        ordered_frames = sorted(frame_predictions)

    prediction_array = np.stack(
        [frame_predictions[frame] for frame in ordered_frames], axis=0
    ).astype(np.float32, copy=False)
    _validate_prediction_array(prediction_array, sequence)

    safe_sequence = sequence.replace("/", "_").replace("\\", "_").lower()
    output_path = os.path.join(output_dir, f"{safe_sequence}_pred.npy")
    np.save(output_path, prediction_array)
    return prediction_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRT no-Doppler inference sequence by sequence on Smoke-Eval."
    )
    parser.add_argument("--config", default="config.yaml", help="YAML config path")
    parser.add_argument("--checkpoint", default=None, help="Override inference checkpoint")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--debug", action="store_true", help="Process one batch per sequence")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    with open(cli.config, "r") as handle:
        config = yaml.safe_load(handle) or {}

    training_config = config.get("training", {})
    paths_config = config.get("paths", {})
    inference_config = config.get("inference", {})

    data_root_value = paths_config.get("test_data_root")
    if not data_root_value:
        raise ValueError("config['paths']['test_data_root'] is required")
    data_root = _resolve_path(cli.config, str(data_root_value))

    checkpoint_value = cli.checkpoint or inference_config.get("checkpoint_path")
    if not checkpoint_value:
        raise ValueError(
            "Set config['inference']['checkpoint_path'] or pass --checkpoint"
        )
    checkpoint_path = _resolve_path(cli.config, str(checkpoint_value))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_value = cli.output_dir or inference_config.get(
        "output_dir", "inference_results"
    )
    output_dir = _resolve_path(cli.config, str(output_value))
    batch_size = int(
        inference_config.get("batch_size", training_config.get("batch_size", 1))
    )
    num_workers = int(
        inference_config.get("num_workers", training_config.get("num_workers", 0))
    )
    frame_skip = int(inference_config.get("frame_skip", 1))
    mixed_precision = "fp16"

    accelerator = Accelerator(mixed_precision=mixed_precision)
    set_seed(int(training_config.get("seed", 42)))

    discovery_dataset = RiceDataset(
        root_dir=data_root,
        sequences=None,
        frame_skip=frame_skip,
        depth_in_meters=True,
        rgb_normalize=False,
    )
    sequences = discovery_dataset.sequences
    if not sequences:
        raise ValueError(f"No valid Smoke-Eval sequences found under {data_root}")
    del discovery_dataset

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Smoke-Eval: {data_root} ({len(sequences)} sequences)")
        print(f"Output: {output_dir}")
        print(
            f"Mixed precision: {mixed_precision} | "
            f"processes: {accelerator.num_processes}"
        )
    accelerator.wait_for_everyone()

    gather_dir = os.path.join(output_dir, "_gather")
    os.makedirs(gather_dir, exist_ok=True)

    model = GRTSmall()
    model.load_state_dict(load_file(checkpoint_path, device="cpu"), strict=True)
    model.eval()
    model = accelerator.prepare(model)

    for sequence_index, sequence in enumerate(sequences):
        loader = create_rice_dataloader(
            root_dir=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            frame_skip=frame_skip,
            sequences=[sequence],
            depth_in_meters=True,
            rgb_normalize=False,
            shuffle=False,
        )
        expected_frames = [
            int(frame_idx) for _, frame_idx in loader.dataset.index_map
        ]
        loader = accelerator.prepare(loader)

        local_results: List[Tuple[int, np.ndarray]] = []
        with torch.no_grad():
            progress = tqdm(
                loader,
                desc=f"[{sequence_index + 1}/{len(sequences)}] {sequence}",
                disable=not accelerator.is_local_main_process,
                dynamic_ncols=True,
                leave=False,
            )
            for batch in progress:
                radar_spectrum = batch_radar_to_spectrum(
                    batch["radar_amplitude"], batch["radar_phase"]
                )
                radar_spectrum = translate_radar(radar_spectrum)
                with accelerator.autocast():
                    occupancy_logits = model(radar_spectrum)
                    prediction = dequantize_depth(occupancy_logits).clamp_(0.0, 1.0)

                prediction_np = prediction.detach().float().cpu().numpy()
                frame_indices = batch["frame_idx"].detach().cpu().tolist()
                local_results.extend(
                    (int(frame_idx), prediction_np[index])
                    for index, frame_idx in enumerate(frame_indices)
                )
                if cli.debug:
                    break

        accelerator.wait_for_everyone()
        safe_sequence = sequence.replace("/", "_").replace("\\", "_")
        rank_path = os.path.join(
            gather_dir,
            f"rank_{accelerator.process_index}_{safe_sequence}.pkl",
        )
        with open(rank_path, "wb") as handle:
            pickle.dump(local_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            merged = _merge_rank_results(
                gather_dir, sequence, accelerator.num_processes
            )
            prediction_array = _save_sequence(
                output_dir,
                sequence,
                merged,
                expected_frames,
                cli.debug,
            )
            print(f"{sequence}: saved {prediction_array.shape}")
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if os.path.isdir(gather_dir) and not os.listdir(gather_dir):
            os.rmdir(gather_dir)
        print(f"Saved {len(sequences)} sequence predictions to: {output_dir}")


if __name__ == "__main__":
    main()
