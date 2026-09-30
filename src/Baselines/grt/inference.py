#!/usr/bin/env python3
"""
Inference Script for GRT-Small (finetuned weights)

Runs inference on all valid sequences in the configured Smoke-Eval root by
default, or on an explicit list supplied with ``--sequences``.
using weights trained by grt_finetune/train.py.
For each sequence, saves one .npy file: pred_depth.npy (dequantized predicted depth [T, 64, 128], values in [0, 1]).

Single GPU: Each frame is seen exactly once; no duplication or incompleteness.
Multi-GPU (DDP): Dataloader is sharded; each rank writes its results to a file, then
main process merges with deduplication by frame_idx (keeps first occurrence) and saves.
"""

import os
import torch
import numpy as np
import argparse
import yaml
import pickle
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from collections import defaultdict
from safetensors.torch import load_file

from grt_model import GRTSmall
from dataloader import create_rice_dataloader
from augmentations import (
    translate_radar,
    dequantize_depth,
)

def batch_radar_to_spectrum(
    radar_amplitude: torch.Tensor, radar_phase: torch.Tensor
) -> torch.Tensor:
    """Restore the GRT spectrum layout from the packaged Smoke-Eval tensors."""

    amplitude = radar_amplitude.permute(0, 1, 3, 2, 4)
    phase = radar_phase.permute(0, 1, 3, 2, 4)
    return torch.stack((amplitude, phase), dim=-1)


def main():
    parser = argparse.ArgumentParser(
        description="Run GRT inference on Smoke-Eval."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to weights-only GRT .safetensors file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Optional sequence names; default discovers all valid sequences.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode (process only 1 batch)"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    set_seed(config["training"].get("seed", 42))

    # Create output directory (all ranks so DDP gather_dir can be created)
    os.makedirs(args.output_dir, exist_ok=True)

    # Create model
    accelerator.print("Creating GRT-Small model...")
    model = GRTSmall()

    # Safetensors files contain only the model state dictionary.
    accelerator.print(f"Loading checkpoint from {args.checkpoint}")
    model.load_state_dict(load_file(args.checkpoint, device="cpu"), strict=True)

    # With ``sequences=None`` the public dataset loader discovers every valid
    # sequence under the configured Smoke-Eval root.
    accelerator.print(f"Inference sequences: {args.sequences}")
    inference_loader = create_rice_dataloader(
        root_dir=config["paths"]["data_root"],
        batch_size=config["training"]["batch_size"],
        num_workers=0,
        frame_skip=1,
        sequences=args.sequences,
        shuffle=False,
    )

    # Prepare model and dataloader
    model, inference_loader = accelerator.prepare(model, inference_loader)
    model.eval()

    # Dictionary to aggregate results by sequence: sequence_id -> list of (frame_idx, pred_depth)
    results_by_sequence = defaultdict(list)

    accelerator.print("Starting inference...")

    with torch.no_grad():
        for batch in tqdm(
            inference_loader, disable=not accelerator.is_local_main_process
        ):
            # Extract data
            rsp_data = batch_radar_to_spectrum(
                batch["radar_amplitude"], batch["radar_phase"]
            )
            sequences = batch["sequence"]
            frame_indices = batch["frame_idx"]

            # Apply radar augmentation
            rsp_data = translate_radar(rsp_data)

            # Forward pass
            occupancy_pred_logits = model(rsp_data)  # [B, 64, 128, 64]

            # Dequantize predicted occupancy to depth [B, 1, 64, 128], values in [0, 1]
            pred_depth = dequantize_depth(occupancy_pred_logits)
            pred_depth_np = (
                pred_depth.cpu().numpy().astype(np.float32)
            )  # [B, 1, 64, 128]

            # Collect results (frame_idx, pred_depth per sample)
            for i in range(len(sequences)):
                seq_id = sequences[i]
                f_idx = frame_indices[i].item()
                # Store [1, 64, 128] per frame; will stack to [T, 64, 128] when saving
                results_by_sequence[seq_id].append(
                    {
                        "frame_idx": f_idx,
                        "pred_depth": pred_depth_np[i],
                    }
                )

            if args.debug:
                break

    # Single GPU: save directly (each frame seen once, no duplication)
    # Multi-GPU: gather via files, merge with dedupe by frame_idx, then save
    if accelerator.num_processes == 1:
        if accelerator.is_main_process:
            accelerator.print("Saving results (single process)...")
            for seq_id, frames in tqdm(
                results_by_sequence.items(), desc="Saving sequences"
            ):
                frames.sort(key=lambda x: x["frame_idx"])
                pred_depth_stack = np.stack([f["pred_depth"] for f in frames], axis=0)
                pred_depth_stack = np.squeeze(pred_depth_stack, axis=1)  # [T, 64, 128]
                np.save(
                    os.path.join(args.output_dir, f"{seq_id.lower()}_pred.npy"),
                    pred_depth_stack,
                )
                accelerator.print(
                    f"  {seq_id}: saved {pred_depth_stack.shape[0]} frames"
                )
            accelerator.print(f"Processed {len(results_by_sequence)} sequences.")
            accelerator.print(f"Results saved to {args.output_dir}")
    else:
        # DDP: gather results from all ranks via files, dedupe by frame_idx, save on main
        accelerator.wait_for_everyone()
        gather_dir = os.path.join(args.output_dir, "_gather")
        os.makedirs(gather_dir, exist_ok=True)
        rank = accelerator.process_index
        rank_file = os.path.join(gather_dir, f"rank_{rank}_results.pkl")
        with open(rank_file, "wb") as f:
            pickle.dump(dict(results_by_sequence), f, protocol=pickle.HIGHEST_PROTOCOL)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            accelerator.print("Merging and deduplicating results from all ranks...")
            merged_results = defaultdict(dict)  # seq_id -> {frame_idx: pred_depth}
            for r in range(accelerator.num_processes):
                pkl_path = os.path.join(gather_dir, f"rank_{r}_results.pkl")
                with open(pkl_path, "rb") as f:
                    rank_results = pickle.load(f)
                for seq_id, frames in rank_results.items():
                    for frame_data in frames:
                        f_idx = frame_data["frame_idx"]
                        if f_idx not in merged_results[seq_id]:
                            merged_results[seq_id][f_idx] = frame_data["pred_depth"]
                os.remove(pkl_path)

            for seq_id, frame_dict in tqdm(
                merged_results.items(), desc="Saving sequences"
            ):
                sorted_items = sorted(frame_dict.items(), key=lambda x: x[0])
                pred_depth_stack = np.stack([item[1] for item in sorted_items], axis=0)
                pred_depth_stack = np.squeeze(pred_depth_stack, axis=1)  # [T, 64, 128]
                np.save(
                    os.path.join(args.output_dir, f"{seq_id.lower()}_pred.npy"),
                    pred_depth_stack,
                )
                accelerator.print(
                    f"  {seq_id}: saved {pred_depth_stack.shape[0]} frames"
                )
            if os.path.isdir(gather_dir) and not os.listdir(gather_dir):
                os.rmdir(gather_dir)
            accelerator.print(f"Processed {len(merged_results)} sequences.")
            accelerator.print(f"Results saved to {args.output_dir}")

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
