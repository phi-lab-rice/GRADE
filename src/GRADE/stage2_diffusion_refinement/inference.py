#!/usr/bin/env python3
"""Three-stage, sequence-wise Smoke-Eval inference for the retrained GRADE model.

The script runs the stages in this order:

1. RadarDepth only -> ``inference_results/ours_radar``
2. RadarDepth + diffusion -> ``inference_results/ours_diffusion``
3. RadarDepth + diffusion + ControlNet -> ``inference_results/ours_full``

Stages 2 and 3 rerun RadarDepth so their conditioning and pure-noise DDIM setup
match ``validate()`` in ``train_diffusion.py`` and
``train_control-defish.py``. Every sequence output is float32 normalized depth
with shape ``[N, 1, H, W]``.

Run the wrapper from this directory; it launches the three stage scripts with
Accelerate one after another:

    python inference.py --config control.yaml --num_processes 4

Each stage can also be launched independently:

    accelerate launch --num_processes 4 inference_radar.py --config control.yaml
    accelerate launch --num_processes 4 inference_diffusion.py --config control.yaml
    accelerate launch --num_processes 4 inference_full.py --config control.yaml
"""

import argparse
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import ControlNetModel, DDIMScheduler
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from collate_fn_helpers import to_3ch, to_neg_pos
from models.radar_depth import RadarDepth
from models.unet import SDUnet
from models.vae import VAE
from rice_dataset import RiceDataset


VARIANT_RADAR = "ours_radar"
VARIANT_DIFFUSION = "ours_diffusion"
VARIANT_FULL = "ours_full"
VALIDATION_NOISE_SEED = 42
VALIDATION_NUM_INFERENCE_STEPS = 8


def _resolve_path(config_path: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(config_dir, value))


def _safe_sequence_name(sequence: str) -> str:
    return sequence.replace("/", "_").replace("\\", "_").lower()


def _discover_sequences(smoke_eval_root: str) -> List[str]:
    root = Path(smoke_eval_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Smoke-Eval root not found: {smoke_eval_root}")

    required_files = ("radar.npy", "zed_depth.npy", "dji_rgb.npy")
    sequences = sorted(
        item.name
        for item in root.iterdir()
        if item.is_dir()
        and not item.name.startswith(".")
        and all((item / name).is_file() for name in required_files)
    )
    if not sequences:
        raise ValueError(
            f"No Smoke-Eval sequences with {required_files} found under {root}"
        )
    for sequence in sequences:
        sequence_dir = root / sequence
        frame_counts = {
            name: int(np.load(sequence_dir / name, mmap_mode="r").shape[0])
            for name in required_files
        }
        if len(set(frame_counts.values())) != 1:
            raise ValueError(
                f"{sequence}: radar/depth/RGB frame counts must match so "
                f"diffusion and ControlNet consume the same seed-42 noise stream; "
                f"got {frame_counts}"
            )
    return sequences


def _validate_prediction_array(
    predictions: np.ndarray,
    sequence: str,
    expected_count: int,
) -> None:
    if predictions.shape[0] != expected_count:
        raise RuntimeError(
            f"{sequence}: expected {expected_count} predictions, "
            f"got {predictions.shape[0]}"
        )
    if predictions.ndim != 4 or predictions.shape[1] != 1:
        raise RuntimeError(
            f"{sequence}: expected [N, 1, H, W], got {predictions.shape}"
        )
    if predictions.dtype != np.float32:
        raise RuntimeError(
            f"{sequence}: expected float32 predictions, got {predictions.dtype}"
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{sequence}: predictions contain NaN or Inf")
    if predictions.size and (predictions.min() < 0.0 or predictions.max() > 1.0):
        raise RuntimeError(
            f"{sequence}: normalized predictions are outside [0, 1]: "
            f"[{predictions.min()}, {predictions.max()}]"
        )


def _merge_rank_results(
    gather_dir: str,
    variant: str,
    sequence: str,
    num_processes: int,
) -> Dict[int, np.ndarray]:
    safe_sequence = _safe_sequence_name(sequence)
    merged: Dict[int, np.ndarray] = {}
    for rank in range(num_processes):
        rank_path = os.path.join(
            gather_dir,
            f"{variant}_rank_{rank}_{safe_sequence}.pkl",
        )
        with open(rank_path, "rb") as handle:
            rank_results = pickle.load(handle)
        for frame_idx, prediction in rank_results:
            merged.setdefault(int(frame_idx), prediction)
        os.remove(rank_path)
    return merged


def _gather_deduplicate_and_save(
    accelerator: Accelerator,
    local_results: List[Tuple[int, np.ndarray]],
    gather_dir: str,
    output_dir: str,
    variant: str,
    sequence: str,
    expected_frames: Sequence[int],
) -> Optional[np.ndarray]:
    os.makedirs(gather_dir, exist_ok=True)
    safe_sequence = _safe_sequence_name(sequence)
    rank_path = os.path.join(
        gather_dir,
        f"{variant}_rank_{accelerator.process_index}_{safe_sequence}.pkl",
    )
    with open(rank_path, "wb") as handle:
        pickle.dump(local_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    accelerator.wait_for_everyone()
    saved_array = None
    if accelerator.is_main_process:
        predictions = _merge_rank_results(
            gather_dir,
            variant,
            sequence,
            accelerator.num_processes,
        )
        expected_frames = [int(frame_idx) for frame_idx in expected_frames]
        expected_set = set(expected_frames)
        missing = [frame_idx for frame_idx in expected_frames if frame_idx not in predictions]
        unexpected = sorted(set(predictions) - expected_set)
        if missing or unexpected:
            raise RuntimeError(
                f"{sequence}: distributed inference coverage mismatch; "
                f"missing={missing[:5]} ({len(missing)} total), "
                f"unexpected={unexpected[:5]} ({len(unexpected)} total)"
            )

        saved_array = np.stack(
            [predictions[frame_idx] for frame_idx in expected_frames],
            axis=0,
        ).astype(np.float32, copy=False)
        _validate_prediction_array(saved_array, sequence, len(expected_frames))
        output_path = os.path.join(output_dir, f"{safe_sequence}_pred.npy")
        np.save(output_path, saved_array)
        print(
            f"  Saved {saved_array.shape[0]} frames, "
            f"shape={saved_array.shape} -> {output_path}"
        )

    accelerator.wait_for_everyone()
    return saved_array


def _make_sequence_dataset(
    data_root: str,
    sequence: str,
    frame_skip: int,
    num_frames: int,
    scale_factor: float,
    max_depth_m: float,
    target_height: int,
    target_width: int,
    use_rgb: bool,
) -> RiceDataset:
    return RiceDataset(
        root_dir=data_root,
        sequences=[sequence],
        frame_skip=frame_skip,
        num_frames=num_frames,
        scale_factor=scale_factor,
        max_depth_m=max_depth_m,
        depth_resolution=(target_height, target_width),
        use_rgb=use_rgb,
        rgb_resolution=(target_height, target_width),
    )


def _make_loader(
    dataset: RiceDataset,
    batch_size: int,
    num_workers: int,
    accelerator: Accelerator,
) -> DataLoader:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(accelerator.device.type == "cuda"),
        drop_last=False,
    )
    return accelerator.prepare(loader)


def _load_state_dict(
    checkpoint_path: str,
    candidate_keys: Sequence[str],
    label: str,
) -> Dict[str, torch.Tensor]:
    del candidate_keys, label
    return load_file(checkpoint_path, device="cpu")


@torch.no_grad()
def _run_radar_stage(
    sequences: Sequence[str],
    data_root: str,
    output_dir: str,
    gather_dir: str,
    radar_model: torch.nn.Module,
    accelerator: Accelerator,
    frame_skip: int,
    num_frames: int,
    batch_size: int,
    num_workers: int,
    scale_factor: float,
    max_depth_m: float,
    target_height: int,
    target_width: int,
) -> None:
    if accelerator.is_main_process:
        print(f"\nStage 1/3: {VARIANT_RADAR}")

    radar_model.eval()
    for sequence_index, sequence in enumerate(sequences):
        dataset = _make_sequence_dataset(
            data_root,
            sequence,
            frame_skip,
            num_frames,
            scale_factor,
            max_depth_m,
            target_height,
            target_width,
            use_rgb=False,
        )
        expected_frames = [int(frame_idx) for _, frame_idx in dataset.index_map]
        if not expected_frames:
            raise RuntimeError(f"{sequence}: no frames found")
        loader = _make_loader(dataset, batch_size, num_workers, accelerator)

        local_results: List[Tuple[int, np.ndarray]] = []
        progress = tqdm(
            loader,
            desc=f"[{sequence_index + 1}/{len(sequences)}] {sequence}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=False,
        )
        for batch in progress:
            radar_spectrum = batch["radar"][:, 0]
            with accelerator.autocast():
                # Preserve RadarDepth's native [B, 1, 128, 256] prediction grid.
                prediction = radar_model(radar_spectrum).clamp(0.0, 1.0)
            prediction_np = prediction.detach().float().cpu().numpy()
            frame_indices = batch["frame_idx"].detach().cpu().tolist()
            local_results.extend(
                (int(frame_idx), prediction_np[index])
                for index, frame_idx in enumerate(frame_indices)
            )

        _gather_deduplicate_and_save(
            accelerator,
            local_results,
            gather_dir,
            output_dir,
            VARIANT_RADAR,
            sequence,
            expected_frames,
        )
        del loader, dataset, local_results


@torch.no_grad()
def _prepare_refinement_inputs(
    batch: Dict[str, torch.Tensor],
    radar_model: torch.nn.Module,
    vae: VAE,
    accelerator: Accelerator,
    use_control: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Mirror prepare_inputs() in the two retrain validation paths."""
    radar_spectrum = batch["radar"][:, 0]
    metric_depth = batch["depth"][:, 0]

    with accelerator.autocast():
        radar_depth = radar_model(radar_spectrum)
    radar_depth_resized = F.interpolate(
        radar_depth,
        size=metric_depth.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    radar_depth_norm = to_neg_pos(to_3ch(radar_depth_resized))
    metric_depth_norm = to_neg_pos(to_3ch(metric_depth))
    unet_conditions = vae.encode_latent(radar_depth_norm)
    target_latents = vae.encode_latent(metric_depth_norm)

    control_pixel = None
    if use_control:
        # Matches prepare_inputs(..., apply_weather=False): clean calibrated RGB.
        control_pixel = to_neg_pos(batch["rgb"][:, 0])
    return unet_conditions, control_pixel, target_latents


@torch.no_grad()
def _run_refinement_stage(
    variant: str,
    sequences: Sequence[str],
    data_root: str,
    output_dir: str,
    gather_dir: str,
    radar_model: torch.nn.Module,
    unet: torch.nn.Module,
    controlnet: Optional[torch.nn.Module],
    vae: VAE,
    accelerator: Accelerator,
    frame_skip: int,
    num_frames: int,
    batch_size: int,
    num_workers: int,
    scale_factor: float,
    max_depth_m: float,
    target_height: int,
    target_width: int,
    num_train_timesteps: int,
) -> None:
    use_control = controlnet is not None
    if accelerator.is_main_process:
        print(f"\nStage {'3' if use_control else '2'}/3: {variant}")

    scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    if accelerator.device.type == "cpu":
        noise_generator = torch.Generator()
    else:
        noise_generator = torch.Generator(device=accelerator.device)
    noise_generator.manual_seed(VALIDATION_NOISE_SEED)

    radar_model.eval()
    unet.eval()
    if controlnet is not None:
        controlnet.eval()

    for sequence_index, sequence in enumerate(sequences):
        dataset = _make_sequence_dataset(
            data_root,
            sequence,
            frame_skip,
            num_frames,
            scale_factor,
            max_depth_m,
            target_height,
            target_width,
            use_rgb=use_control,
        )
        expected_frames = [int(frame_idx) for _, frame_idx in dataset.index_map]
        if not expected_frames:
            raise RuntimeError(f"{sequence}: no frames found")
        loader = _make_loader(dataset, batch_size, num_workers, accelerator)

        local_results: List[Tuple[int, np.ndarray]] = []
        progress = tqdm(
            loader,
            desc=f"[{sequence_index + 1}/{len(sequences)}] {sequence}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=False,
        )
        for batch in progress:
            frame_indices = [
                int(frame_idx)
                for frame_idx in batch["frame_idx"].detach().cpu().tolist()
            ]
            unet_conditions, control_pixel, target_latents = (
                _prepare_refinement_inputs(
                    batch,
                    radar_model,
                    vae,
                    accelerator,
                    use_control,
                )
            )
            batch_size_actual = target_latents.shape[0]
            latents = torch.randn(
                target_latents.shape,
                generator=noise_generator,
                device=target_latents.device,
                dtype=target_latents.dtype,
            )
            scheduler.set_timesteps(
                VALIDATION_NUM_INFERENCE_STEPS,
                device=accelerator.device,
            )
            encoder_hidden_states = torch.zeros(
                batch_size_actual,
                1,
                1024,
                device=accelerator.device,
                dtype=latents.dtype,
            )

            if controlnet is not None:
                # Matches validate() in train_control-defish.py.
                with accelerator.autocast():
                    for timestep in scheduler.timesteps:
                        scaled_latents = scheduler.scale_model_input(
                            latents, timestep
                        )
                        unet_input = torch.cat(
                            [scaled_latents, unet_conditions],
                            dim=1,
                        )
                        timesteps = torch.full(
                            (batch_size_actual,),
                            timestep,
                            device=accelerator.device,
                            dtype=torch.long,
                        )
                        down_samples, mid_sample = controlnet(
                            unet_input,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            controlnet_cond=control_pixel,
                            return_dict=False,
                        )
                        noise_prediction = unet(
                            unet_input,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            down_block_additional_residuals=down_samples,
                            mid_block_additional_residual=mid_sample,
                        ).sample
                        latents = scheduler.step(
                            model_output=noise_prediction,
                            timestep=timestep,
                            sample=latents,
                        ).prev_sample
            else:
                # Matches validate() in train_diffusion.py (no outer autocast).
                for timestep in scheduler.timesteps:
                    scaled_latents = scheduler.scale_model_input(latents, timestep)
                    unet_input = torch.cat(
                        [scaled_latents, unet_conditions],
                        dim=1,
                    )
                    timesteps = torch.full(
                        (batch_size_actual,),
                        timestep,
                        device=accelerator.device,
                        dtype=torch.long,
                    )
                    noise_prediction = unet(
                        unet_input,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample
                    latents = scheduler.step(
                        model_output=noise_prediction,
                        timestep=timestep,
                        sample=latents,
                    ).prev_sample

            prediction_3ch, _ = vae.decode_latent(latents)
            prediction = prediction_3ch.mean(dim=1, keepdim=True).clamp(0.0, 1.0)

            prediction_np = prediction.detach().float().cpu().numpy()
            local_results.extend(
                (frame_idx, prediction_np[index])
                for index, frame_idx in enumerate(frame_indices)
            )

        _gather_deduplicate_and_save(
            accelerator,
            local_results,
            gather_dir,
            output_dir,
            variant,
            sequence,
            expected_frames,
        )
        del loader, dataset, local_results


def parse_stage_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description,
    )
    parser.add_argument(
        "--config",
        default="control.yaml",
        help="Path to the retrain ControlNet YAML config",
    )
    return parser.parse_args()


def _load_runtime(
    config_path: str,
    required_checkpoints: Sequence[str],
) -> Dict[str, object]:
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle) or {}

    training = config.get("training", {})
    data = config.get("data", {})
    pretrained = config.get("pretrained", {})
    diffusion = config.get("diffusion", {})
    inference = config.get("inference", {})

    smoke_eval_value = data.get("smoke_eval_root")
    radar_value = pretrained.get("radar_model")
    unet_value = pretrained.get("unet") or pretrained.get("diffusion_unet")
    controlnet_value = pretrained.get("controlnet")
    checkpoint_values = {
        "radar": ("pretrained.radar_model", radar_value),
        "unet": ("pretrained.unet", unet_value),
        "controlnet": ("pretrained.controlnet", controlnet_value),
    }
    missing_config = []
    if not smoke_eval_value:
        missing_config.append("data.smoke_eval_root")
    for checkpoint_name in required_checkpoints:
        config_name, value = checkpoint_values[checkpoint_name]
        if not value:
            missing_config.append(config_name)
    if missing_config:
        raise ValueError(f"Missing required config values: {missing_config}")

    smoke_eval_root = _resolve_path(config_path, str(smoke_eval_value))
    resolved_checkpoints = {}
    checkpoint_labels = {
        "radar": "RadarDepth",
        "unet": "UNet",
        "controlnet": "ControlNet",
    }
    for checkpoint_name in required_checkpoints:
        _, value = checkpoint_values[checkpoint_name]
        checkpoint_path = _resolve_path(config_path, str(value))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"{checkpoint_labels[checkpoint_name]} checkpoint not found: "
                f"{checkpoint_path}"
            )
        resolved_checkpoints[checkpoint_name] = checkpoint_path

    target_height = int(data.get("resolution", {}).get("height", 288))
    target_width = int(data.get("resolution", {}).get("width", 512))
    frame_skip = int(inference.get("frame_skip", data.get("test_skip", 1)))
    if frame_skip != 1:
        raise ValueError(
            f"Final Smoke-Eval inference requires frame_skip=1, got {frame_skip}"
        )
    num_frames = int(data.get("num_frames", 1))
    scale_factor = float(data.get("scale_factor", 0.001))
    max_depth_m = float(data.get("max_depth_m", 11.2))
    batch_size = int(inference.get("batch_size", training.get("batch_size", 1)))
    num_workers = int(inference.get("num_workers", data.get("num_workers", 0)))
    mixed_precision = "fp16"
    num_train_timesteps = int(diffusion.get("num_train_timesteps", 1000))

    # Keep newly generated predictions separate from the archived cluster
    # results.  A caller may provide ``inference.output_root`` in the YAML,
    # which is particularly useful for a small reproducibility run.  Preserve
    # the historical in-source default for legacy configs that do not provide
    # an override.
    output_root_value = inference.get("output_root")
    if output_root_value:
        output_root = _resolve_path(config_path, str(output_root_value))
    else:
        output_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "inference_results",
        )
    output_dirs = {
        variant: os.path.join(output_root, variant)
        for variant in (VARIANT_RADAR, VARIANT_DIFFUSION, VARIANT_FULL)
    }
    gather_dir = os.path.join(output_root, "_gather")

    sequences = _discover_sequences(smoke_eval_root)
    return {
        "smoke_eval_root": smoke_eval_root,
        "checkpoints": resolved_checkpoints,
        "target_height": target_height,
        "target_width": target_width,
        "frame_skip": frame_skip,
        "num_frames": num_frames,
        "scale_factor": scale_factor,
        "max_depth_m": max_depth_m,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "mixed_precision": mixed_precision,
        "num_train_timesteps": num_train_timesteps,
        "output_root": output_root,
        "output_dirs": output_dirs,
        "gather_dir": gather_dir,
        "sequences": sequences,
    }


def _make_accelerator(runtime: Dict[str, object], stage: str) -> Accelerator:
    accelerator = Accelerator(mixed_precision=str(runtime["mixed_precision"]))
    set_seed(VALIDATION_NOISE_SEED)

    if accelerator.is_main_process:
        for output_dir in runtime["output_dirs"].values():
            os.makedirs(str(output_dir), exist_ok=True)
        os.makedirs(str(runtime["gather_dir"]), exist_ok=True)
        print(f"Stage: {stage}")
        print(f"Smoke-Eval: {runtime['smoke_eval_root']}")
        print(f"Sequences: {len(runtime['sequences'])}")
        print(f"Output root: {runtime['output_root']}")
        print(
            f"Mixed precision: {runtime['mixed_precision']} | "
            f"processes: {accelerator.num_processes} | "
            f"batch size: {runtime['batch_size']}"
        )
        print(
            f"Validation sampling: seed={VALIDATION_NOISE_SEED} | "
            f"DDIM steps={VALIDATION_NUM_INFERENCE_STEPS}"
        )
    accelerator.wait_for_everyone()
    return accelerator


def _remove_empty_gather_dir(
    accelerator: Accelerator,
    gather_dir: str,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if os.path.isdir(gather_dir) and not os.listdir(gather_dir):
            os.rmdir(gather_dir)


def run_radar(config_path: str) -> None:
    runtime = _load_runtime(config_path, required_checkpoints=("radar",))
    accelerator = _make_accelerator(runtime, VARIANT_RADAR)

    radar_model = RadarDepth()
    radar_state = _load_state_dict(
        runtime["checkpoints"]["radar"],
        ("model_state_dict", "state_dict"),
        "RadarDepth",
    )
    radar_model.load_state_dict(radar_state, strict=True)
    radar_model.eval()
    for parameter in radar_model.parameters():
        parameter.requires_grad = False
    radar_model = accelerator.prepare(radar_model)

    _run_radar_stage(
        runtime["sequences"],
        runtime["smoke_eval_root"],
        runtime["output_dirs"][VARIANT_RADAR],
        runtime["gather_dir"],
        radar_model,
        accelerator,
        runtime["frame_skip"],
        runtime["num_frames"],
        runtime["batch_size"],
        runtime["num_workers"],
        runtime["scale_factor"],
        runtime["max_depth_m"],
        runtime["target_height"],
        runtime["target_width"],
    )
    _remove_empty_gather_dir(accelerator, str(runtime["gather_dir"]))
    if accelerator.is_main_process:
        print(f"\n{VARIANT_RADAR} inference complete.")


def _load_radar_model(runtime: Dict[str, object]) -> torch.nn.Module:
    radar_model = RadarDepth()
    radar_state = _load_state_dict(
        runtime["checkpoints"]["radar"],
        ("model_state_dict", "state_dict"),
        "RadarDepth",
    )
    radar_model.load_state_dict(radar_state, strict=True)
    radar_model.eval()
    for parameter in radar_model.parameters():
        parameter.requires_grad = False
    return radar_model


def _load_vae_and_unet(
    runtime: Dict[str, object],
    accelerator: Accelerator,
) -> Tuple[VAE, torch.nn.Module]:

    vae = VAE(device=accelerator.device, torch_dtype=torch.float32)
    vae.vae.eval()
    for parameter in vae.vae.parameters():
        parameter.requires_grad = False

    sd_unet = SDUnet(
        use_pretrained=False,
        device=accelerator.device,
        torch_dtype=torch.float32,
    )
    unet = sd_unet.unet
    unet_state = _load_state_dict(
        runtime["checkpoints"]["unet"],
        ("unet_state_dict", "model_state_dict", "state_dict"),
        "UNet",
    )
    unet.load_state_dict(unet_state, strict=True)
    unet.eval()
    for parameter in unet.parameters():
        parameter.requires_grad = False
    return vae, unet


def run_diffusion(config_path: str) -> None:
    runtime = _load_runtime(config_path, required_checkpoints=("radar", "unet"))
    accelerator = _make_accelerator(runtime, VARIANT_DIFFUSION)
    radar_model = _load_radar_model(runtime)
    vae, unet = _load_vae_and_unet(runtime, accelerator)
    radar_model, unet = accelerator.prepare(radar_model, unet)

    _run_refinement_stage(
        VARIANT_DIFFUSION,
        runtime["sequences"],
        runtime["smoke_eval_root"],
        runtime["output_dirs"][VARIANT_DIFFUSION],
        runtime["gather_dir"],
        radar_model,
        unet,
        None,
        vae,
        accelerator,
        runtime["frame_skip"],
        runtime["num_frames"],
        runtime["batch_size"],
        runtime["num_workers"],
        runtime["scale_factor"],
        runtime["max_depth_m"],
        runtime["target_height"],
        runtime["target_width"],
        runtime["num_train_timesteps"],
    )
    _remove_empty_gather_dir(accelerator, str(runtime["gather_dir"]))
    if accelerator.is_main_process:
        print(f"\n{VARIANT_DIFFUSION} inference complete.")


def run_full(config_path: str) -> None:
    runtime = _load_runtime(
        config_path,
        required_checkpoints=("radar", "unet", "controlnet"),
    )
    accelerator = _make_accelerator(runtime, VARIANT_FULL)
    radar_model = _load_radar_model(runtime)
    vae, unet = _load_vae_and_unet(runtime, accelerator)

    controlnet = ControlNetModel.from_unet(unet)
    controlnet_state = _load_state_dict(
        runtime["checkpoints"]["controlnet"],
        ("controlnet_state_dict", "model_state_dict", "state_dict"),
        "ControlNet",
    )
    controlnet.load_state_dict(controlnet_state, strict=True)
    controlnet.eval()
    for parameter in controlnet.parameters():
        parameter.requires_grad = False

    radar_model, unet, controlnet = accelerator.prepare(
        radar_model,
        unet,
        controlnet,
    )
    _run_refinement_stage(
        VARIANT_FULL,
        runtime["sequences"],
        runtime["smoke_eval_root"],
        runtime["output_dirs"][VARIANT_FULL],
        runtime["gather_dir"],
        radar_model,
        unet,
        controlnet,
        vae,
        accelerator,
        runtime["frame_skip"],
        runtime["num_frames"],
        runtime["batch_size"],
        runtime["num_workers"],
        runtime["scale_factor"],
        runtime["max_depth_m"],
        runtime["target_height"],
        runtime["target_width"],
        runtime["num_train_timesteps"],
    )
    _remove_empty_gather_dir(accelerator, str(runtime["gather_dir"]))
    if accelerator.is_main_process:
        print(f"\n{VARIANT_FULL} inference complete.")


def _parse_wrapper_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three Smoke-Eval inference scripts sequentially."
    )
    parser.add_argument(
        "--config",
        default="control.yaml",
        help="Path to the retrain ControlNet YAML config",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of Accelerate worker processes; defaults to visible CUDA GPUs",
    )
    return parser.parse_args()


def main() -> None:
    cli = _parse_wrapper_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(cli.config)
    with open(config_path, "r") as handle:
        wrapper_config = yaml.safe_load(handle) or {}
    mixed_precision = "fp16"

    accelerate_path = shutil.which("accelerate")
    if accelerate_path is None:
        raise FileNotFoundError("The 'accelerate' executable was not found in PATH")

    num_processes = cli.num_processes
    if num_processes is None:
        num_processes = max(torch.cuda.device_count(), 1)
    if num_processes < 1:
        raise ValueError("--num_processes must be at least 1")

    stage_scripts = (
        "inference_radar.py",
        "inference_diffusion.py",
        "inference_full.py",
    )
    for stage_index, stage_script in enumerate(stage_scripts, start=1):
        command = [
            accelerate_path,
            "launch",
            "--num_processes",
            str(num_processes),
            "--num_machines",
            "1",
            "--mixed_precision",
            mixed_precision,
            "--dynamo_backend",
            "no",
            os.path.join(script_dir, stage_script),
            "--config",
            config_path,
        ]
        print(
            f"\nWrapper stage {stage_index}/{len(stage_scripts)}: "
            f"{stage_script}"
        )
        subprocess.run(command, cwd=script_dir, check=True)

    print("\nAll three inference stages completed.")


if __name__ == "__main__":
    main()
