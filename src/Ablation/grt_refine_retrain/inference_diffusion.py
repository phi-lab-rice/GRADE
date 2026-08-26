#!/usr/bin/env python3
"""
Diffusion Model Inference Script

Runs DDIM sampling (8 steps, seed=42 – identical to validate() in
train_diffusion.py) on the val split (Rice test-rice sequences) and/or the
test split (Smoke-Eval sequences).

Usage:
    python inference_diffusion.py \
        --config configs/stage2.yaml \
        --val_out results/val \
        --test_out results/test

The config supplies data roots plus the GRT and default Stage-2 checkpoints.
`--checkpoint` can override the configured Stage-2 checkpoint.

Output layout (one file per sequence):
    <val_out>/<seq_name>_pred.npy   float32  (N, 1, H, W)  values in [0, 1]
    <test_out>/<seq_name>_pred.npy  float32  (N, 1, H, W)  values in [0, 1]

Multi-GPU:
    accelerate launch --num_processes 2 inference_diffusion.py ...
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDIMScheduler
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from collate_fn_helpers import to_3ch, to_neg_pos
from checkpoints import require_checkpoint
from conditioning import build_unet_input, resize_condition_depth
from models.grt_depth import GRTDepth, load_grt_checkpoint
from models.unet import SDUnet
from models.vae import VAE
from rice_dataset import RiceDataset


# ---------------------------------------------------------------------------
# Input preparation (mirrors prepare_inputs in train_spatial.py)
# ---------------------------------------------------------------------------


@torch.no_grad()
def prepare_inputs_inference(
    batch,
    radar_model,
    vae,
    accelerator,
    target_height: int = 288,
    target_width: int = 512,
):
    """
    Prepare UNet conditioning from a RiceDataset batch.

    Args:
        batch: dict with keys "radar", "depth", "rgb"  (F-dim already squeezed out by RiceDataset)
        radar_model: frozen GRT coarse-depth model
        vae: VAE wrapper
        accelerator: Accelerator instance
        target_height / target_width: spatial size fed to the UNet

    Returns:
        unet_conditions : [B, 4, H_lat, W_lat]  – radar latents encoded at target size
        metric_depth    : [B, 1, target_height, target_width]  – ground-truth depth in [0, 1]
    """
    # squeeze the leading frame dimension (num_frames=1)
    radar_spectrum = batch["radar"].squeeze(1)  # [B, 64, 8, 2, 256, 2]
    depth_01 = batch["depth"].squeeze(1)  # [B, 1, H_d, W_d]

    # Resize depth to target resolution
    if depth_01.shape[2] != target_height or depth_01.shape[3] != target_width:
        metric_depth = F.interpolate(
            depth_01,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
    else:
        metric_depth = depth_01

    # Coarse radar depth via frozen model
    with accelerator.autocast():
        radar_depth = radar_model(radar_spectrum)  # [B, 1, 64, 128]

    # Align normalized GRT depth in pixel space before VAE compression.
    radar_depth_resized = resize_condition_depth(radar_depth, metric_depth)

    # Encode aligned depths to VAE latent space.
    radar_depth_3ch = to_3ch(radar_depth_resized)  # [B, 3, H, W]
    metric_depth_3ch = to_3ch(metric_depth)  # [B, 3, H, W]

    radar_depth_norm = to_neg_pos(radar_depth_3ch)  # [-1, 1]
    metric_depth_norm = to_neg_pos(metric_depth_3ch)  # [-1, 1]

    radar_latents = vae.encode_latent(radar_depth_norm)  # [B, 4, H_l, W_l]
    target_latents = vae.encode_latent(metric_depth_norm)  # [B, 4, H_l, W_l]

    if radar_latents.shape[-2:] != target_latents.shape[-2:]:
        raise RuntimeError("Aligned GRT and target depth latents must share spatial dimensions.")

    unet_conditions = radar_latents
    return unet_conditions, metric_depth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _discover_test_sequences(smoke_eval_root: str):
    """Return sorted sequence names found under smoke_eval_root."""
    root = Path(smoke_eval_root)
    if not root.is_dir():
        raise FileNotFoundError(f"smoke_eval_root not found: {smoke_eval_root}")
    required = ["radar.npy", "zed_depth.npy"]
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and all((d / f).exists() for f in required)
    )


def _discover_val_sequences(rice_root: str):
    """Return val sequences (test-rice entries from split.json that exist on disk)."""
    split_file = Path(__file__).parent / "split.json"
    if not split_file.exists():
        raise FileNotFoundError(f"split.json not found: {split_file}")
    with open(split_file, "r") as f:
        split_config = json.load(f)
    val_seqs = split_config.get("test-rice", [])
    root = Path(rice_root)
    return sorted(s for s in val_seqs if (root / s).is_dir())


@torch.no_grad()
def _run_split(
    split_label,
    sequences,
    data_root,
    output_dir,
    unet,
    radar_model,
    vae,
    scheduler,
    accelerator,
    target_height,
    target_width,
    scale_factor,
    max_depth_m,
    num_frames,
    batch_size,
    num_workers,
    num_inference_steps,
    noise_seed,
):
    """Run DDIM inference for all sequences in a split and save per-sequence .npy files."""
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Split: {split_label}  ({len(sequences)} sequences)")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")

    gather_dir = os.path.join(output_dir, "_gather")
    os.makedirs(gather_dir, exist_ok=True)

    for seq_idx, seq_name in enumerate(sequences):
        if accelerator.is_main_process:
            print(f"\n[{seq_idx + 1}/{len(sequences)}] {seq_name}")

        seq_dataset = RiceDataset(
            root_dir=data_root,
            sequences=[seq_name],
            frame_skip=1,
            num_frames=num_frames,
            scale_factor=scale_factor,
            max_depth_m=max_depth_m,
            depth_resolution=(target_height, target_width),
            use_rgb=False,
            rgb_resolution=(target_height, target_width),
        )

        if len(seq_dataset) == 0:
            if accelerator.is_main_process:
                print(f"  [skip] No frames found.")
            continue

        seq_loader = DataLoader(
            seq_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        seq_loader = accelerator.prepare(seq_loader)

        # Fixed-seed noise generator (matches validate() in train_diffusion.py)
        if accelerator.device.type == "cpu":
            noise_generator = torch.Generator()
        else:
            noise_generator = torch.Generator(device=accelerator.device)
        noise_generator.manual_seed(noise_seed)

        seq_results = []  # list of (frame_idx, pred_np)

        pbar = tqdm(
            seq_loader,
            desc=f"  {seq_name}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in pbar:
            with accelerator.autocast():
                unet_conditions, _ = prepare_inputs_inference(
                    batch,
                    radar_model,
                    vae,
                    accelerator,
                    target_height=target_height,
                    target_width=target_width,
                )

                B = unet_conditions.shape[0]
                h, w = unet_conditions.shape[2], unet_conditions.shape[3]

                # Pure Gaussian noise start (same as validate())
                latents = torch.randn(
                    (B, 4, h, w),
                    generator=noise_generator,
                    device=unet_conditions.device,
                    dtype=unet_conditions.dtype,
                )

                scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
                encoder_hidden_states = torch.zeros(
                    B,
                    1,
                    1024,
                    device=accelerator.device,
                    dtype=latents.dtype,
                )

                for t in scheduler.timesteps:
                    scaled = scheduler.scale_model_input(latents, t)
                    unet_input = build_unet_input(scaled, unet_conditions)
                    timesteps_t = torch.full(
                        (B,), t, device=accelerator.device, dtype=torch.long
                    )
                    noise_pred = unet(
                        unet_input,
                        timesteps_t,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample
                    latents = scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=latents,
                    ).prev_sample

                pred_3ch, _ = vae.decode_latent(latents)
                pred = torch.mean(pred_3ch, dim=1, keepdim=True)  # [B, 1, H, W]

            pred_np = pred.detach().cpu().float().numpy()
            frame_idx = batch["frame_idx"]
            if torch.is_tensor(frame_idx):
                frame_idx = frame_idx.cpu().tolist()
            for i in range(pred_np.shape[0]):
                seq_results.append((int(frame_idx[i]), pred_np[i]))

        # Gather across DDP ranks via temp files
        accelerator.wait_for_everyone()
        rank = accelerator.process_index
        safe_seq = seq_name.replace("/", "_").replace("\\", "_").lower()
        rank_file = os.path.join(gather_dir, f"rank_{rank}_{safe_seq}.pkl")
        with open(rank_file, "wb") as f:
            pickle.dump(seq_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            merged = []
            for r in range(accelerator.num_processes):
                pkl_path = os.path.join(gather_dir, f"rank_{r}_{safe_seq}.pkl")
                with open(pkl_path, "rb") as f:
                    merged.extend(pickle.load(f))
                os.remove(pkl_path)

            # Deduplicate by frame_idx (keep first occurrence)
            by_idx: dict = {}
            for idx, pred_i in merged:
                if idx not in by_idx:
                    by_idx[idx] = pred_i

            items = sorted(by_idx.items(), key=lambda x: x[0])
            pred_arr = np.stack([v for _, v in items], axis=0).astype(np.float32)
            out_path = os.path.join(output_dir, f"{safe_seq}_pred.npy")
            np.save(out_path, pred_arr)
            print(
                f"  Saved {pred_arr.shape[0]} frames  shape={pred_arr.shape}  output={out_path}"
            )

        accelerator.wait_for_everyone()
        del seq_results
        torch.cuda.empty_cache()

    # Cleanup empty gather dir
    if accelerator.is_main_process:
        if os.path.isdir(gather_dir) and not os.listdir(gather_dir):
            os.rmdir(gather_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Diffusion inference on val (Rice) and/or test (Smoke-Eval) splits"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage2.yaml",
        help="YAML config containing data and checkpoint paths.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional override for inference.diffusion_checkpoint from YAML.",
    )
    parser.add_argument(
        "--val_out",
        type=str,
        default=None,
        help="Output folder for val-split predictions (Rice test-rice sequences). "
        "Omit to skip val inference.",
    )
    parser.add_argument(
        "--test_out",
        type=str,
        default=None,
        help="Output folder for test-split predictions (Smoke-Eval sequences). "
        "Omit to skip test inference.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of DDIM inference steps (default: 8, matching validate())",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.val_out is None and args.test_out is None:
        parser.error("Provide at least one of --val_out or --test_out.")

    # ------------------------------------------------------------------
    # Config – only data paths are consumed
    # ------------------------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    rice_root = config["data"].get("rice_root")
    smoke_eval_root = config["data"].get("smoke_eval_root")
    target_height = config["data"].get("resolution", {}).get("height", 288)
    target_width = config["data"].get("resolution", {}).get("width", 512)
    scale_factor = config["data"].get("scale_factor", 0.001)
    max_depth_m = config["data"].get("max_depth_m", 11.2)
    num_train_timesteps = config.get("diffusion", {}).get("num_train_timesteps", 1000)
    num_frames = config["data"].get("num_frames", 1)
    grt_checkpoint = config["pretrained"]["grt_model"]
    unet_checkpoint = args.checkpoint or config["inference"]["diffusion_checkpoint"]
    require_checkpoint(grt_checkpoint, "GRT")
    require_checkpoint(unet_checkpoint, "Stage-2 UNet")

    # ------------------------------------------------------------------
    # Accelerator
    # ------------------------------------------------------------------
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    radar_model = GRTDepth()
    load_grt_checkpoint(radar_model, grt_checkpoint, map_location=device)
    radar_model.eval()
    for p in radar_model.parameters():
        p.requires_grad = False
    radar_model.to(device)

    vae = VAE(device=device, torch_dtype=torch.float32)
    vae.vae.eval()
    for p in vae.vae.parameters():
        p.requires_grad = False

    sd_unet = SDUnet(use_pretrained=False, device=device, torch_dtype=torch.float32)
    unet = sd_unet.unet

    unet.load_state_dict(load_file(unet_checkpoint, device="cpu"), strict=True)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False
    unet.to(device)

    if accelerator.is_main_process:
        print(f"Loaded UNet checkpoint: {unet_checkpoint}")
        print(f"Target resolution: {target_width}x{target_height}")

    # ------------------------------------------------------------------
    # DDIM scheduler
    # ------------------------------------------------------------------
    scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    # ------------------------------------------------------------------
    # Accelerator wrapping
    # ------------------------------------------------------------------
    unet, radar_model = accelerator.prepare(unet, radar_model)

    common = dict(
        unet=unet,
        radar_model=radar_model,
        vae=vae,
        scheduler=scheduler,
        accelerator=accelerator,
        target_height=target_height,
        target_width=target_width,
        scale_factor=scale_factor,
        max_depth_m=max_depth_m,
        num_frames=num_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_inference_steps=args.steps,
        noise_seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Val split (Rice test-rice sequences)
    # ------------------------------------------------------------------
    if args.val_out is not None:
        if not rice_root:
            raise ValueError("config['data']['rice_root'] is required for --val_out.")
        val_seqs = _discover_val_sequences(rice_root)
        if not val_seqs:
            raise ValueError(f"No val sequences found under rice_root={rice_root}.")
        _run_split(
            split_label="val",
            sequences=val_seqs,
            data_root=rice_root,
            output_dir=args.val_out,
            **common,
        )

    # ------------------------------------------------------------------
    # Test split (Smoke-Eval sequences)
    # ------------------------------------------------------------------
    if args.test_out is not None:
        if not smoke_eval_root:
            raise ValueError(
                "config['data']['smoke_eval_root'] is required for --test_out."
            )
        test_seqs = _discover_test_sequences(smoke_eval_root)
        if not test_seqs:
            raise ValueError(
                f"No test sequences found under smoke_eval_root={smoke_eval_root}."
            )
        _run_split(
            split_label="test",
            sequences=test_seqs,
            data_root=smoke_eval_root,
            output_dir=args.test_out,
            **common,
        )

    if accelerator.is_main_process:
        print("\nInference complete.")


if __name__ == "__main__":
    main()
