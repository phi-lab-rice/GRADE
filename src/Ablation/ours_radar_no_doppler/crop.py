"""
Preprocessing Module for Radar-to-Depth Training

1. Radar Scaling: Normalize radar amplitude and phase channels
   - Magnitude: sqrt scaling with configurable factor
   - Phase: Convert from [-1, 1] to [0, 1] range

2. LiDAR Depth Cropping: Crop depth frames to match camera FOV
   - Strategy: Resize (128, 256) → (192, 384) then center crop → (128, 256)
   - Effect: FOV reduced from 180°x90° to 120°x60°

3. Helper Functions: Tensor range conversions and channel utilities
"""

import torch
import numpy as np
from typing import Union
from torchvision import transforms as T
import torch.fft as tfft


## radar
radar_config = {"scale_type": "sqrt", "scale_factor": 0.001}


def translate_radar(
    radar_amplitude: torch.Tensor, radar_phase: torch.Tensor = None
) -> torch.Tensor:
    """
    Convert radar amplitude/phase to normalized 6D spectrum.

    This function merges the behavior of the previous `radar_batch_to_spectrum`
    and `translate_radar` helpers.

    Two usage patterns are supported:
        1) Separate Rice amplitude/phase inputs:
            - radar_amplitude: [B, 64, 2, 8, 256]
            - radar_phase:     [B, 64, 2, 8, 256]
           Returns normalized spectrum:
            - [B, 2, 256, 64, 8, 2]

        2) Pre-stacked spectrum input:
            - radar_amplitude: [B, 2, 256, 64, 8, 2]
            - radar_phase:     None
           In this case, the input is treated as the raw spectrum and only
           normalization is applied.
    """
    if radar_phase is not None:
        # Case 1: build spectrum from separate amp/phase (Rice format)
        # (doppler=64, elevation=2, azimuth=8, range=256)
        radar_data = torch.stack(
            [radar_amplitude, radar_phase], dim=1
        )  # (B, 2, 64, 2, 8, 256)
        # permute to [B, 2, range, doppler, azimuth, elevation]
        radar_data = radar_data.permute(0, 1, 5, 2, 4, 3)  # (B, 2, 256, 64, 8, 2)
    else:
        # Case 2: already in spectrum format
        radar_data = radar_amplitude

    # radar_data shape: [B, 2, 256, 64, 8, 2]
    radar_mag = radar_data[:, 0:1, :, :, :, :]  # [B, 1, 256, 64, 8, 2]
    radar_phase_ch = radar_data[:, 1:2, :, :, :, :]  # [B, 1, 256, 64, 8, 2]

    # scale the magnitude channel
    if radar_config["scale_type"] == "sqrt":
        radar_mag_processed = (
            torch.sqrt(radar_mag + 1e-8) * radar_config["scale_factor"]
        )
    elif radar_config["scale_type"] == "log":
        radar_mag_processed = torch.log(radar_mag + 1) * radar_config["scale_factor"]
    elif radar_config["scale_type"] == "linear":
        radar_mag_processed = radar_mag * radar_config["scale_factor"]
    else:
        raise ValueError(f"Unknown scale type: {radar_config['scale_type']}")

    # convert the phase channel to [0, 1] range
    radar_phase_processed = to_zero_one(radar_phase_ch)

    # Recombine the processed magnitude and phase
    radar_data_translated = torch.cat(
        [radar_mag_processed, radar_phase_processed], dim=1
    )  # [B, 2, 256, 64, 8, 2]

    return radar_data_translated


## lidar
def crop_depth(
    lidar_depth: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Process camera depth map from `camera_depth_mm.npy`.

    Steps:
        1) Convert from millimeters to meters.
        2) Clamp depth values to [0, 11.2] meters.
        3) Normalize to [0, 1] by dividing by 11.2 meters.
        4) Resize spatial resolution to (128, 256) to match model output.

    Input:
        - lidar_depth: depth array in millimeters
            - Can be torch.Tensor or np.ndarray
            - Supports shapes (..., 128, 256) or (..., H, W)
            - Optional channel dimension

    Output:
        - Same shape structure as input, but with spatial size (128, 256)
          and values in [0, 1].
    """
    is_numpy = isinstance(lidar_depth, np.ndarray)

    # Convert to torch if needed
    if is_numpy:
        lidar_depth = torch.from_numpy(lidar_depth)

    lidar_depth = lidar_depth.float()
    original_shape = lidar_depth.shape

    # Handle different input shapes
    if lidar_depth.dim() == 2:
        # (H, W) -> (1, H, W)
        lidar_depth = lidar_depth.unsqueeze(0)
    elif lidar_depth.dim() == 3:
        # Could be (B, H, W) or (1, H, W)
        # Add channel dim: (B, H, W) -> (B, 1, H, W)
        lidar_depth = lidar_depth.unsqueeze(1)
    elif lidar_depth.dim() == 4:
        # (B, 1, H, W) - already has channel dimension
        pass
    else:
        raise ValueError(f"Unexpected input shape: {original_shape}")

    # Handle invalid values (NaN, inf, negative) before processing
    invalid_mask = ~(torch.isfinite(lidar_depth) & (lidar_depth >= 0))
    lidar_depth[invalid_mask] = 0.0

    # Convert from millimeters to meters
    lidar_depth = lidar_depth / 1000.0

    # Clamp to [0, 11.2] meters
    max_depth_m = 11.2
    lidar_depth = torch.clamp(lidar_depth, min=0.0, max=max_depth_m)

    # Normalize to [0, 1]
    lidar_depth = lidar_depth / max_depth_m

    # Final check: ensure no NaN/inf after normalization
    invalid_mask = ~torch.isfinite(lidar_depth)
    lidar_depth[invalid_mask] = 0.0

    # Resize to (128, 256) to match model output
    resize_transform = T.Resize(
        (128, 256), interpolation=T.InterpolationMode.BILINEAR, antialias=True
    )
    resized = resize_transform(lidar_depth)

    # Restore original shape structure (except spatial size)
    if len(original_shape) == 2:
        # Remove channel dim: (1, H, W) -> (H, W)
        resized = resized.squeeze(0)
    elif len(original_shape) == 3 and original_shape[0] != 1:
        # Was (B, H, W), now (B, 1, H, W), keep as is for training
        # Don't squeeze - preserve channel dimension
        pass
    # If original was (B, 1, H, W), keep as (B, 1, H, W)

    # Convert back to numpy if needed
    if is_numpy:
        resized = resized.numpy()

    return resized


## rgb
def resize_rgb_frame(
    rgb_frame: Union[torch.Tensor, np.ndarray],
    target_size: tuple = (256, 128),
) -> Union[torch.Tensor, np.ndarray]:
    """
    Resize RGB video frame to target resolution.

    Input: (3, 1080, 1920), (1080, 1920, 3), or (B, 3, 1080, 1920) - RGB video frame
    Output: (3, 128, 256), (128, 256, 3), or (B, 3, 128, 256) - Resized frame

    Args:
        rgb_frame: RGB frame of shape (C, H, W), (H, W, C), or (B, C, H, W)
            - PyTorch: typically (3, H, W) or (B, 3, H, W)
            - NumPy: typically (H, W, 3)
        target_size: Target (width, height), default (256, 128)

    Returns:
        Resized frame with same channel order as input

    Examples:
        >>> # PyTorch (C, H, W)
        >>> frame = torch.randn(3, 1080, 1920)
        >>> resized = resize_rgb_frame(frame)
        >>> resized.shape
        torch.Size([3, 128, 256])

        >>> # Batched PyTorch (B, C, H, W)
        >>> frames = torch.randn(8, 3, 1080, 1920)
        >>> resized = resize_rgb_frame(frames)
        >>> resized.shape
        torch.Size([8, 3, 128, 256])
    """
    is_numpy = isinstance(rgb_frame, np.ndarray)
    target_w, target_h = target_size

    # Convert to torch if needed
    if is_numpy:
        rgb_frame = torch.from_numpy(rgb_frame)

    # Detect channel order and batch dimension
    # Batched: (B, C, H, W) where C=3
    # Single: (C, H, W) where C=3
    # NumPy: (H, W, C) where C=3
    if rgb_frame.dim() == 4:
        # Batched input (B, C, H, W)
        if rgb_frame.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got shape {rgb_frame.shape}")
        channels_first = True
        is_batched = True
    elif rgb_frame.dim() == 3:
        if rgb_frame.shape[0] == 3:
            # (C, H, W) format
            channels_first = True
            is_batched = False
        elif rgb_frame.shape[-1] == 3:
            # (H, W, C) format - need to permute to (C, H, W)
            channels_first = False
            is_batched = False
            rgb_frame = rgb_frame.permute(2, 0, 1)
        else:
            raise ValueError(
                f"Cannot detect channel order from shape {rgb_frame.shape}"
            )
    else:
        raise ValueError(f"Cannot detect channel order from shape {rgb_frame.shape}")

    # Resize using torchvision
    resize_transform = T.Resize(
        (target_h, target_w), interpolation=T.InterpolationMode.BILINEAR, antialias=True
    )
    resized = resize_transform(rgb_frame)

    # Restore original channel order if needed
    if not channels_first:
        resized = resized.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

    # Convert back to numpy if needed
    if is_numpy:
        resized = resized.numpy()

    return resized


## helpers
def to_3ch(depth_map):
    """
    Prepares a depth map for VAE encoding.
    Ensures the depth map is 3-channel by repeating if needed.
    """
    if depth_map.shape[1] == 1:
        depth_map_3ch = depth_map.repeat(1, 3, 1, 1)
    else:
        depth_map_3ch = depth_map

    return depth_map_3ch


def to_neg_pos(tensor):
    """
    Converts a tensor from [0, 1] range to [-1, 1] range.

    Args:
        tensor: Input tensor in [0, 1] range

    Returns:
        Tensor in [-1, 1] range
    """
    return torch.clamp(tensor * 2.0 - 1.0, min=-1, max=1)


def to_zero_one(tensor):
    """
    Converts a tensor from [-1, 1] range to [0, 1] range.

    Args:
        tensor: Input tensor in [-1, 1] range

    Returns:
        Tensor in [0, 1] range
    """
    return torch.clamp((tensor + 1.0) / 2.0, min=0, max=1)


