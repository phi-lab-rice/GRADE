import argparse
import os
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from safetensors.torch import load_file
from tqdm.auto import tqdm

from dataloader import create_inference_loader
from models.model import CaFNet


DEFAULT_CONFIG = {
    # Packaged evaluation dataset.
    "base_dir": "",
    "split_json": None,
    "test_base_dir": None,
    "test_split": "train",
    "test_split_json": None,
    # Input and radar processing
    "input_height": 288,
    "input_width": 512,
    "radar_max_depth_m": 11.2,
    "max_dist_correspondence": 0.5,
    "patch_size": None,
    # Model
    "encoder": "resnet34_bts",
    "encoder_radar": "resnet18",
    "radar_input_channels": 1,
    "bts_size": 512,
    "max_depth": 11.2,
    # Runtime
    "batch_size": 8,
    # Windows uses spawn-based multiprocessing; keep the public evaluation
    # entry point portable and deterministic by default.
    "num_workers": 0,
    "seed": 42,
    "cpu": False,
    "mixed_precision": "fp16",
    "checkpoint_path": "checkpoints/cafnet_no_smoke.safetensors",
    "prediction_dir": "prediction",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run CaFNet inference on Smoke-Eval.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping (key-value pairs).")

    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)

    if not merged["test_base_dir"]:
        raise ValueError("Config must define 'test_base_dir'.")
    if not merged["checkpoint_path"]:
        raise ValueError("Config must define 'checkpoint_path'.")
    if not os.path.isfile(merged["checkpoint_path"]):
        raise FileNotFoundError(f"Checkpoint not found: {merged['checkpoint_path']}")
    if merged.get("radar_input_channels", 1) != 1:
        raise ValueError("radar_input_channels must be 1 for this setup.")

    return argparse.Namespace(**merged)


def build_model_args(args):
    return argparse.Namespace(
        encoder=args.encoder,
        encoder_radar=args.encoder_radar,
        radar_input_channels=args.radar_input_channels,
        input_height=args.input_height,
        input_width=args.input_width,
        max_depth=args.max_depth,
        bts_size=args.bts_size,
    )


def _extract_model_state(checkpoint):
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format.")


def _gather_objects(accelerator, obj):
    if accelerator.num_processes == 1:
        return [obj]
    if not dist.is_available() or not dist.is_initialized():
        return [obj]

    gathered = [None for _ in range(accelerator.num_processes)]
    dist.all_gather_object(gathered, obj)
    return gathered


def _merge_predictions(all_rank_predictions):
    merged: Dict[str, Dict[int, np.ndarray]] = {}
    for rank_dict in all_rank_predictions:
        if not rank_dict:
            continue
        for seq_name, frame_map in rank_dict.items():
            seq_slot = merged.setdefault(seq_name, {})
            for frame_idx, pred in frame_map.items():
                frame_idx = int(frame_idx)
                if frame_idx not in seq_slot:
                    seq_slot[frame_idx] = pred
    return merged


def _save_sequence_predictions(predictions, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for seq_name in sorted(predictions.keys()):
        frame_map = predictions[seq_name]
        ordered_frames = sorted(frame_map.keys())
        if not ordered_frames:
            pred_stack = np.zeros((0,), dtype=np.float32)
        else:
            pred_stack = np.stack([frame_map[k] for k in ordered_frames], axis=0).astype(
                np.float32,
                copy=False,
            )
        np.save(os.path.join(out_dir, f"{seq_name.lower()}_pred.npy"), pred_stack)


def _run_loader_inference(accelerator, model, loader, samples, save_dir, desc):
    model.eval()
    local_preds: Dict[str, Dict[int, np.ndarray]] = {}

    with torch.no_grad():
        pbar = tqdm(
            loader,
            desc=desc,
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=False,
        )
        for batch in pbar:
            sample_idx, image, depth_gt, radar, radar_gt = batch

            image = image.to(accelerator.device, non_blocking=True)
            radar = radar.to(accelerator.device, non_blocking=True)
            # Kept for parity with validation loop structure.
            _ = depth_gt.to(accelerator.device, non_blocking=True)
            _ = radar_gt.to(accelerator.device, non_blocking=True)

            focal = torch.ones((image.size(0),), device=image.device)
            _, _, _, _, depth_est, _, _ = model(image, radar, focal)

            pred_np = depth_est.detach().float().cpu().numpy()
            if pred_np.ndim == 4 and pred_np.shape[1] == 1:
                pred_np = pred_np[:, 0]

            if torch.is_tensor(sample_idx):
                sample_idx_list = sample_idx.detach().cpu().tolist()
            else:
                sample_idx_list = list(sample_idx)

            for local_i, sample_i in enumerate(sample_idx_list):
                seq_name, frame_idx = samples[int(sample_i)]
                seq_slot = local_preds.setdefault(seq_name, {})
                frame_idx = int(frame_idx)
                if frame_idx not in seq_slot:
                    seq_slot[frame_idx] = pred_np[local_i].astype(np.float32, copy=False)

    gathered = _gather_objects(accelerator, local_preds)
    if accelerator.is_main_process:
        merged = _merge_predictions(gathered)
        _save_sequence_predictions(merged, save_dir)

    accelerator.wait_for_everyone()


def main():
    cli = parse_args()
    args = load_config(cli.config)

    set_seed(args.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision in ("no", "none") else args.mixed_precision,
        cpu=args.cpu,
        kwargs_handlers=[ddp_kwargs],
    )

    test_loader = create_inference_loader(
        args,
        pin_memory=(accelerator.device.type == "cuda"),
    )
    test_samples: List = test_loader.dataset.samples

    model = CaFNet(build_model_args(args))

    model, test_loader = accelerator.prepare(model, test_loader)

    state_dict = load_file(args.checkpoint_path, device="cpu")
    accelerator.unwrap_model(model).load_state_dict(state_dict, strict=True)

    _run_loader_inference(
        accelerator=accelerator,
        model=model,
        loader=test_loader,
        samples=test_samples,
        save_dir=args.prediction_dir,
        desc="Inference",
    )

    if accelerator.is_main_process:
        print(f"Saved predictions to: {args.prediction_dir}")


if __name__ == "__main__":
    main()
