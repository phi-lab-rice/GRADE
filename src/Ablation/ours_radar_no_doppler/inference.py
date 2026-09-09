#!/usr/bin/env python3
"""Sequence-by-sequence RadarDepth no-Doppler inference on Smoke-Eval.

Launch with ``accelerate launch inference.py --config <config.yaml>``.
Each output is ``<sequence>_pred.npy`` with float32 shape ``[N, 1, H, W]``
and normalized depth clipped to ``[0, 1]``.
"""

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from radar_depth import RadarDepth
from rice_dataset import RiceDataset


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
    predictions: Dict[int, np.ndarray],
    expected_frames: List[int],
    debug: bool,
) -> np.ndarray:
    if not predictions:
        raise RuntimeError(f"{sequence}: inference produced no predictions")

    if not debug:
        missing = [frame for frame in expected_frames if frame not in predictions]
        if missing:
            raise RuntimeError(
                f"{sequence}: missing {len(missing)} predictions "
                f"(first few frame indices: {missing[:5]})"
            )
        ordered_frames = expected_frames
    else:
        ordered_frames = sorted(predictions)

    prediction_array = np.stack(
        [predictions[frame] for frame in ordered_frames], axis=0
    ).astype(np.float32, copy=False)
    _validate_prediction_array(prediction_array, sequence)

    safe_sequence = sequence.replace("/", "_").replace("\\", "_").lower()
    np.save(
        os.path.join(output_dir, f"{safe_sequence}_pred.npy"),
        prediction_array,
    )
    return prediction_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RadarDepth no-Doppler inference on Smoke-Eval."
    )
    parser.add_argument(
        "--config",
        default="config_stage1_iq1m.yaml",
        help="YAML config path",
    )
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--debug", action="store_true", help="Process one batch per sequence")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    with open(cli.config, "r") as handle:
        config = yaml.safe_load(handle) or {}

    training_config = config.get("training", {})
    data_config = config.get("data", {})
    inference_config = config.get("inference", {})

    test_root_value = data_config.get("test_root")
    if not test_root_value:
        raise ValueError("config['data']['test_root'] is required")
    test_root = _resolve_path(cli.config, str(test_root_value))

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
        inference_config.get("num_workers", data_config.get("num_workers", 0))
    )
    frame_skip = int(inference_config.get("frame_skip", 1))
    mixed_precision = "fp16"
    scale_factor = float(data_config.get("scale_factor", 0.001))
    max_depth_m = float(data_config.get("max_depth_m", 11.2))
    depth_resolution = tuple(data_config.get("depth_resolution", [128, 256]))

    accelerator = Accelerator(mixed_precision=mixed_precision)
    set_seed(int(training_config.get("seed", 42)))

    discovery_dataset = RiceDataset(
        root_dir=test_root,
        sequences=None,
        frame_skip=frame_skip,
        scale_factor=scale_factor,
        max_depth_m=max_depth_m,
        depth_resolution=depth_resolution,
        use_rgb=False,
    )
    sequences = discovery_dataset.sequences
    if not sequences:
        raise ValueError(f"No valid Smoke-Eval sequences found under {test_root}")
    del discovery_dataset

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Smoke-Eval: {test_root} ({len(sequences)} sequences)")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Output: {output_dir}")
        print(
            f"Mixed precision: {mixed_precision} | "
            f"processes: {accelerator.num_processes}"
        )
    accelerator.wait_for_everyone()

    gather_dir = os.path.join(output_dir, "_gather")
    os.makedirs(gather_dir, exist_ok=True)

    model = RadarDepth(
        output_height=int(depth_resolution[0]),
        output_width=int(depth_resolution[1]),
    )
    model.load_state_dict(load_file(checkpoint_path, device="cpu"), strict=True)
    model.eval()
    model = accelerator.prepare(model)

    for sequence_index, sequence in enumerate(sequences):
        dataset = RiceDataset(
            root_dir=test_root,
            sequences=[sequence],
            frame_skip=frame_skip,
            scale_factor=scale_factor,
            max_depth_m=max_depth_m,
            depth_resolution=depth_resolution,
            use_rgb=False,
        )
        expected_frames = [int(frame_idx) for _, frame_idx in dataset.index_map]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(accelerator.device.type == "cuda"),
            drop_last=False,
        )
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
                with accelerator.autocast():
                    prediction = model(batch["radar"]).clamp_(0.0, 1.0)
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
