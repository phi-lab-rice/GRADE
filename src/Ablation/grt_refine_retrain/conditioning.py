"""Shared conditioning operations for Stage-2 and Stage-3 refinement."""

import torch
import torch.nn.functional as F


def resize_condition_depth(condition_depth, target_depth):
    """Resize normalized GRT depth before VAE encoding to match target depth."""
    return F.interpolate(
        condition_depth,
        size=target_depth.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def build_unet_input(noisy_latent, condition_latent):
    """Concatenate four-channel depth and GRT latents into the UNet input."""
    return torch.cat([noisy_latent, condition_latent], dim=1)
