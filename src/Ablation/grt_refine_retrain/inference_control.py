#!/usr/bin/env python3
"""
ControlNet Inference Script (Smoke-Eval only)

Runs DDIM sampling using the same validation settings as train_control.py:
- DDIM scheduler config matches validation loop
- 8 inference steps
- seed=42
- seed set once per split (no re-seed per sequence)

No RGB weather augmentation is applied in this script.

Usage:
    accelerate launch inference_control.py --config stage3.yaml

The YAML config supplies GRT, Stage-2 UNet, and Stage-3 ControlNet paths.
`--unet` and `--control` can override the configured refinement checkpoints.
"""

import argparse
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
from diffusers import DDIMScheduler, ControlNetModel
from safetensors.torch import load_file
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.utils.data import DataLoader

from collate_fn_helpers import to_3ch, to_neg_pos
from checkpoints import require_checkpoint
from conditioning import build_unet_input, resize_condition_depth
from models.grt_depth import GRTDepth, load_grt_checkpoint
from models.unet import SDUnet
from models.vae import VAE
from rice_dataset import RiceDataset

VALIDATE_NUM_INFERENCE_STEPS = 8
VALIDATE_NOISE_SEED = 42


# ---------------------------------------------------------------------------
# Input preparation
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
    Prepare UNet/ControlNet conditioning from a RiceDataset batch.

    Returns:
        unet_conditions : [B, 4, H_lat, W_lat]  – radar latents
        control_pixel   : [B, 3, H, W]           – RGB in [-1, 1]
        metric_depth    : [B, 1, H, W]            – GT depth in [0, 1]
    """
    radar_spectrum = batch["radar"].squeeze(1)  # [B, 64, 8, 2, 256, 2]
    depth_01 = batch["depth"].squeeze(1)  # [B, 1, H_d, W_d]
    rgb = batch["rgb"].squeeze(1)  # [B, 3, H_r, W_r] in [0, 1]

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

    # Resize RGB to target resolution
    if rgb.shape[2] != target_height or rgb.shape[3] != target_width:
        rgb = F.interpolate(
            rgb,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )

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

    # RGB as ControlNet pixel conditioning in [-1, 1]
    control_pixel = to_neg_pos(rgb)  # [B, 3, H, W]

    return unet_conditions, control_pixel, metric_depth


# ---------------------------------------------------------------------------
# Sequence discovery helpers
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

def _depth_to_xyz(depth_01, max_depth_m, fx, fy, cx, cy):
    """Back-project normalized depth [0,1] into metric XYZ coordinates."""
    _, _, h, w = depth_01.shape
    z = depth_01 * max_depth_m

    u = torch.arange(w, device=depth_01.device, dtype=depth_01.dtype)
    v = torch.arange(h, device=depth_01.device, dtype=depth_01.dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    uu = uu.unsqueeze(0).unsqueeze(0)
    vv = vv.unsqueeze(0).unsqueeze(0)

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return torch.cat([x, y, z], dim=1)


# ---------------------------------------------------------------------------
# Per-split runner
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_split(
    sequences,
    data_root,
    output_dir,
    unet,
    controlnet,
    radar_model,
    vae,
    accelerator,
    target_height,
    target_width,
    scale_factor,
    max_depth_m,
    num_frames,
    frame_skip,
    batch_size,
    num_workers,
    num_train_timesteps,
    lpips_metric,
    fx,
    fy,
    cx,
    cy,
):
    """Run DDIM inference for all sequences in a split and save per-sequence .npy files."""
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Split: smoke-eval ({len(sequences)} sequences)")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")

    # Match train_control.py validate(): fresh DDIM scheduler per split.
    scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    # Match validate(): seed once per split and do not re-seed between sequences/batches.
    if accelerator.device.type == "cpu":
        noise_generator = torch.Generator()
    else:
        noise_generator = torch.Generator(device=accelerator.device)
    noise_generator.manual_seed(VALIDATE_NOISE_SEED)

    gather_dir = os.path.join(output_dir, "_gather")
    os.makedirs(gather_dir, exist_ok=True)

    for seq_idx, seq_name in enumerate(sequences):
        if accelerator.is_main_process:
            print(f"\n[{seq_idx + 1}/{len(sequences)}] {seq_name}")

        seq_dataset = RiceDataset(
            root_dir=data_root,
            sequences=[seq_name],
            frame_skip=frame_skip,
            num_frames=num_frames,
            scale_factor=scale_factor,
            max_depth_m=max_depth_m,
            depth_resolution=(target_height, target_width),
            use_rgb=True,
            rgb_resolution=(target_height, target_width),
            dji_calibrate=True,
        )

        if len(seq_dataset) == 0:
            if accelerator.is_main_process:
                print(f"  [skip] No frames found.")
            continue

        expected_frame_indices = [
            int(frame_idx) for _, frame_idx in seq_dataset.index_map
        ]

        seq_loader = DataLoader(
            seq_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        seq_loader = accelerator.prepare(seq_loader)

        seq_results = []  # list of (frame_idx, pred_np)
        seq_l1_sum = 0.0
        seq_lpips_sum = 0.0
        seq_xyz_sum = 0.0
        seq_count = 0.0

        pbar = tqdm(
            seq_loader,
            desc=f"  {seq_name}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in pbar:
            with accelerator.autocast():
                unet_conditions, control_pixel, metric_depth = prepare_inputs_inference(
                    batch,
                    radar_model,
                    vae,
                    accelerator,
                    target_height=target_height,
                    target_width=target_width,
                )

                B = unet_conditions.shape[0]
                h, w = unet_conditions.shape[2], unet_conditions.shape[3]

                # Pure Gaussian noise start
                latents = torch.randn(
                    (B, 4, h, w),
                    generator=noise_generator,
                    device=unet_conditions.device,
                    dtype=unet_conditions.dtype,
                )

                scheduler.set_timesteps(
                    VALIDATE_NUM_INFERENCE_STEPS, device=accelerator.device
                )
                encoder_hidden_states = torch.zeros(
                    B,
                    1,
                    1024,
                    device=accelerator.device,
                    dtype=latents.dtype,
                )

                for t in scheduler.timesteps:
                    scaled = scheduler.scale_model_input(latents, t)
                    unet_input = build_unet_input(
                        scaled, unet_conditions
                    )  # [B, 8, h, w]
                    timesteps_t = torch.full(
                        (B,), t, device=accelerator.device, dtype=torch.long
                    )

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        unet_input,
                        timesteps_t,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=control_pixel,
                        return_dict=False,
                    )

                    noise_pred = unet(
                        unet_input,
                        timesteps_t,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                    latents = scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=latents,
                    ).prev_sample

                pred_3ch, _ = vae.decode_latent(latents)
                pred = torch.mean(pred_3ch, dim=1, keepdim=True)  # [B, 1, H, W]
                pred = pred.clamp_(0.0, 1.0)

            # Per-batch metrics (no augmentation path)
            B = pred.shape[0]
            l1_val = F.l1_loss(pred, metric_depth).item()

            pred_lp = torch.clamp(to_3ch(pred), 0, 1)
            gt_lp = torch.clamp(to_3ch(metric_depth), 0, 1)
            lpips_val = lpips_metric(pred_lp, gt_lp).item()

            pred_xyz = _depth_to_xyz(pred, max_depth_m, fx, fy, cx, cy)
            gt_xyz = _depth_to_xyz(metric_depth, max_depth_m, fx, fy, cx, cy)
            valid_mask = metric_depth > 0
            if torch.any(valid_mask):
                valid_mask_xyz = valid_mask.expand(-1, 3, -1, -1)
                xyz_val = F.l1_loss(pred_xyz[valid_mask_xyz], gt_xyz[valid_mask_xyz]).item()
            else:
                xyz_val = 0.0

            seq_l1_sum += l1_val * B
            seq_lpips_sum += lpips_val * B
            seq_xyz_sum += xyz_val * B
            seq_count += float(B)

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

        metrics_local = torch.tensor(
            [seq_l1_sum, seq_lpips_sum, seq_xyz_sum, seq_count],
            device=accelerator.device,
            dtype=torch.float64,
        )
        metrics_global = accelerator.reduce(metrics_local, reduction="sum")
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
            missing = [idx for idx in expected_frame_indices if idx not in by_idx]
            if missing:
                raise RuntimeError(
                    f"{seq_name}: missing {len(missing)} predictions "
                    f"(first few frame indices: {missing[:5]})"
                )
            pred_arr = np.stack([v for _, v in items], axis=0).astype(np.float32)
            if pred_arr.ndim != 4 or pred_arr.shape[1] != 1:
                raise RuntimeError(
                    f"{seq_name}: expected [N, 1, H, W], got {pred_arr.shape}"
                )
            if not np.isfinite(pred_arr).all():
                raise RuntimeError(f"{seq_name}: predictions contain NaN or Inf")
            if pred_arr.min() < 0.0 or pred_arr.max() > 1.0:
                raise RuntimeError(f"{seq_name}: predictions are outside [0, 1]")
            out_path = os.path.join(output_dir, f"{safe_seq}_pred.npy")
            np.save(out_path, pred_arr)
            count = max(float(metrics_global[3].item()), 1.0)
            seq_l1_avg = float(metrics_global[0].item() / count)
            seq_lpips_avg = float(metrics_global[1].item() / count)
            seq_xyz_avg = float(metrics_global[2].item() / count)
            print(
                f"  Saved {pred_arr.shape[0]} frames  shape={pred_arr.shape}  output={out_path}"
            )
            print(
                f"    Avg metrics: L1={seq_l1_avg:.6f}  LPIPS={seq_lpips_avg:.6f}  XYZ={seq_xyz_avg:.6f}"
            )

        accelerator.wait_for_everyone()
        del seq_results
        torch.cuda.empty_cache()

    # Cleanup empty gather dir
    if accelerator.is_main_process:
        if os.path.isdir(gather_dir) and not os.listdir(gather_dir):
            os.rmdir(gather_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ControlNet inference on Smoke-Eval with per-sequence metrics"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage3.yaml",
        help="YAML config containing data and checkpoint paths.",
    )
    parser.add_argument(
        "--unet",
        type=str,
        default=None,
        help="Optional override for inference.unet_checkpoint from YAML.",
    )
    parser.add_argument(
        "--control",
        type=str,
        default=None,
        help="Optional override for inference.controlnet_checkpoint from YAML.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional override for inference.output_dir from YAML.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    training_config = config.get("training", {})
    inference_config = config.get("inference", {})
    smoke_eval_root = data_config.get("smoke_eval_root")
    target_height = data_config.get("resolution", {}).get("height", 288)
    target_width = data_config.get("resolution", {}).get("width", 512)
    scale_factor = data_config.get("scale_factor", 0.001)
    max_depth_m = data_config.get("max_depth_m", 11.2)
    camera_intrinsics = data_config.get("camera_intrinsics", {})
    missing = [k for k in ("fx", "fy", "cx", "cy") if k not in camera_intrinsics]
    if missing:
        raise ValueError(
            f"camera_intrinsics missing keys for XYZ metric: {missing}. "
            "Expected config['data']['camera_intrinsics'] with fx, fy, cx, cy."
        )
    fx = float(camera_intrinsics["fx"])
    fy = float(camera_intrinsics["fy"])
    cx = float(camera_intrinsics["cx"])
    cy = float(camera_intrinsics["cy"])
    num_train_timesteps = config.get("diffusion", {}).get("num_train_timesteps", 1000)
    num_frames = data_config.get("num_frames", 1)
    frame_skip = int(inference_config.get("frame_skip", data_config.get("test_skip", 1)))
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else inference_config.get("batch_size", training_config.get("batch_size", 1))
    )
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else inference_config.get("num_workers", data_config.get("num_workers", 0))
    )
    mixed_precision = "fp16"
    output_dir = args.out or inference_config.get("output_dir")
    if not output_dir:
        raise ValueError("Set config['inference']['output_dir'] or pass --out")
    grt_checkpoint = config["pretrained"]["grt_model"]
    unet_checkpoint = args.unet or config["inference"]["unet_checkpoint"]
    controlnet_checkpoint = (
        args.control or config["inference"]["controlnet_checkpoint"]
    )
    require_checkpoint(grt_checkpoint, "GRT")
    require_checkpoint(unet_checkpoint, "Stage-2 UNet")
    require_checkpoint(controlnet_checkpoint, "Stage-3 ControlNet")

    # ------------------------------------------------------------------
    # Accelerator
    # ------------------------------------------------------------------
    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device
    set_seed(VALIDATE_NOISE_SEED)

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

    controlnet = ControlNetModel.from_unet(unet)
    controlnet.load_state_dict(
        load_file(controlnet_checkpoint, device="cpu"), strict=True
    )
    controlnet.eval()
    for p in controlnet.parameters():
        p.requires_grad = False
    controlnet.to(device)

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", normalize=True
    ).to(device)
    lpips_metric.eval()

    if accelerator.is_main_process:
        print(f"Loaded UNet checkpoint      : {unet_checkpoint}")
        print(f"Loaded ControlNet checkpoint: {controlnet_checkpoint}")
        print(f"Target resolution: {target_width}x{target_height}")
        print(
            f"Mixed precision: {mixed_precision} | "
            f"processes: {accelerator.num_processes}"
        )

    # ------------------------------------------------------------------
    # Accelerator wrapping
    # ------------------------------------------------------------------
    unet, controlnet, radar_model = accelerator.prepare(unet, controlnet, radar_model)

    common = dict(
        unet=unet,
        controlnet=controlnet,
        radar_model=radar_model,
        vae=vae,
        accelerator=accelerator,
        target_height=target_height,
        target_width=target_width,
        scale_factor=scale_factor,
        max_depth_m=max_depth_m,
        num_frames=num_frames,
        frame_skip=frame_skip,
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_timesteps=num_train_timesteps,
        lpips_metric=lpips_metric,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    # Smoke-Eval only
    if not smoke_eval_root:
        raise ValueError("config['data']['smoke_eval_root'] is required.")
    test_seqs = _discover_test_sequences(smoke_eval_root)
    if not test_seqs:
        raise ValueError(
            f"No test sequences found under smoke_eval_root={smoke_eval_root}."
        )
    _run_split(
        sequences=test_seqs,
        data_root=smoke_eval_root,
        output_dir=output_dir,
        **common,
    )

    if accelerator.is_main_process:
        print("\nInference complete.")


if __name__ == "__main__":
    main()
