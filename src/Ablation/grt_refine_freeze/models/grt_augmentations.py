import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize, InterpolationMode
from typing import Union
import numpy as np

AZIMUTH_RESOLUTION = 128
ELEVATION_RESOLUTION = 64

# Depth output resolution: height=64, width=128
DEPTH_TARGET_HEIGHT = 64
DEPTH_TARGET_WIDTH = 128

resize_transform = Resize(
    size=[ELEVATION_RESOLUTION, AZIMUTH_RESOLUTION],
    interpolation=InterpolationMode.BILINEAR,
    antialias=True,
)

depth_resize_transform = Resize(
    size=(DEPTH_TARGET_HEIGHT, DEPTH_TARGET_WIDTH),
    interpolation=InterpolationMode.BILINEAR,
    antialias=True,
)


def translate_radar(radar_data):
    """
    Applies normalization to radar data after batching from dataloader.
    Called before passing data into the model.

    Args:
        radar_data: Batched radar tensor from dataloader
                   Shape: [B, 64, 8, 2, 256, 2] (batch, doppler, azimuth, elevation, range, channels)
                   - Channel 0: raw amplitude values
                   - Channel 1: phase normalized to [-1, 1] (divided by π)

    Returns:
        Processed radar tensor with same shape [B, 64, 8, 2, 256, 2]
        - Channel 0: sqrt(amplitude * 1e-3) for magnitude normalization
        - Channel 1: phase * π (converted back to radians [-π, π])
    """
    radar_mag = radar_data[..., 0]  # [B, 64, 8, 2, 256] - Extract raw amplitude
    radar_phase = radar_data[..., 1]  # [B, 64, 8, 2, 256] - Extract normalized phase

    # Normalize amplitude: scale then sqrt
    radar_mag_processed = torch.sqrt(radar_mag * 1e-6)

    # Convert phase back to radians: [-1, 1] -> [-π, π]
    radar_phase_processed = radar_phase * torch.pi

    # Stack channels back together: [B, 64, 8, 2, 256, 2]
    radar_data_translated = torch.stack(
        [radar_mag_processed, radar_phase_processed], dim=-1
    )
    return radar_data_translated


def resize_depth(
    depth_map: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Process depth map from dataloader (same pipeline as denoiser/control crop_depth):
    mm -> meters, clamp [0, 11.2] m, normalize to [0, 1], resize to (64, 128) (h, w).

    Args:
        depth_map: Depth in millimeters. Torch or numpy.
            Shapes: (H, W), (B, H, W), or (B, 1, H, W).

    Returns:
        Depth in [0, 1], spatial size (64, 128). Shape [B, 64, 128] for batched input.
    """
    is_numpy = isinstance(depth_map, np.ndarray)
    if is_numpy:
        depth_map = torch.from_numpy(depth_map)

    depth_map = depth_map.float()
    original_shape = depth_map.shape

    if depth_map.dim() == 2:
        depth_map = depth_map.unsqueeze(0)  # (H, W) -> (1, H, W)
    elif depth_map.dim() == 3:
        depth_map = depth_map.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    elif depth_map.dim() != 4:
        raise ValueError(f"Unexpected depth shape: {original_shape}")

    invalid_mask = ~(torch.isfinite(depth_map) & (depth_map >= 0))
    depth_map[invalid_mask] = 0.0

    depth_map = depth_map / 1000.0  # mm -> meters
    max_depth_m = 11.2
    depth_map = torch.clamp(depth_map, min=0.0, max=max_depth_m)
    depth_map = depth_map / max_depth_m  # [0, 1]

    invalid_mask = ~torch.isfinite(depth_map)
    depth_map[invalid_mask] = 0.0

    depth_map = depth_resize_transform(depth_map)  # (..., 64, 128)
    depth_values = depth_map.squeeze(1)  # [B, 64, 128] or [1, 64, 128]

    if len(original_shape) == 2:
        depth_values = depth_values.squeeze(0)  # (64, 128)

    if is_numpy:
        depth_values = depth_values.numpy()
    return depth_values


def quantize_depth_to_occupancy(depth_values, num_range_bins=64):
    """
    Quantizes 2D depth values into 3D binary occupancy grid.

    Args:
        depth_values: Resized depth tensor
                     Shape: [B, elevation, azimuth]
                     Values: normalized to [0, 1] range
        num_range_bins: Number of range bins for quantization (default: 64)

    Returns:
        Binary 3D occupancy grid
        Shape: [B, elevation, azimuth, num_range_bins]
        Values: binary (0 or 1) indicating occupied bins
    """
    B, elevation, azimuth = depth_values.shape

    # Quantize normalized depth [0, 1] directly to range bins [0, num_range_bins-1]
    # Each bin represents 1/num_range_bins of the normalized depth range
    bin_indices = torch.floor(
        depth_values / (1.0 / num_range_bins)
    ).long()  # [B, elevation, azimuth]
    bin_indices = torch.clamp(
        bin_indices, 0, num_range_bins - 1
    )  # Handle edge case where depth_values = 1.0

    # Create binary 3D occupancy grid
    occupancy_grid = torch.zeros(
        B,
        elevation,
        azimuth,
        num_range_bins,
        dtype=torch.float32,
        device=depth_values.device,
    )  # [B, elevation, azimuth, num_range_bins]

    # Set occupied bins to 1
    # Use advanced indexing to mark the appropriate range bin for each (elevation, azimuth) cell
    batch_idx = torch.arange(B, device=depth_values.device)[:, None, None].expand(
        B, elevation, azimuth
    )
    elevation_idx = torch.arange(elevation, device=depth_values.device)[
        None, :, None
    ].expand(B, elevation, azimuth)
    azimuth_idx = torch.arange(azimuth, device=depth_values.device)[
        None, None, :
    ].expand(B, elevation, azimuth)

    occupancy_grid[batch_idx, elevation_idx, azimuth_idx, bin_indices] = 1.0

    return occupancy_grid  # [B, elevation, azimuth, num_range_bins]


def dequantize_depth(occupancy_grid):
    """
    Converts 3D binary occupancy grid back to 2D depth map.
    This is the inverse operation of quantize_depth_to_occupancy.

    Args:
        occupancy_grid: Binary 3D occupancy grid
                       Shape: [B, 64, 128, 64] (batch, elevation, azimuth, range)
                       Values: binary (0 or 1) or continuous (predicted probabilities)

    Returns:
        Reconstructed depth map
        Shape: [B, 1, 64, 128] (batch, channel, elevation, azimuth)
        Values: normalized to [0, 1] range
    """
    num_range_bins = occupancy_grid.shape[3]

    # Find the range bin with maximum value for each (elevation, azimuth) cell
    # For binary: finds the occupied bin
    # For continuous: finds the most likely bin
    bin_indices = torch.argmax(occupancy_grid, dim=3)  # [B, 64, 128]

    # Convert bin indices back to normalized depth values [0, 1]
    # Use bin center: (bin_idx + 0.5) / num_bins
    depth_values = (bin_indices.float() + 1) / num_range_bins  # [B, 64, 128]

    # Add channel dimension: [B, 64, 128] -> [B, 1, 64, 128]
    depth_map = depth_values.unsqueeze(1)  # [B, 1, 64, 128]

    return depth_map


