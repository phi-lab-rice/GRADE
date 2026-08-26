import torch
import numpy as np
import cv2
from typing import Optional, Tuple, Union
from torchvision import transforms as T


## General Purpose Helper Functions ##
def to_zero_one(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] to [0, 1]."""
    return torch.clamp((tensor + 1.0) / 2.0, min=0, max=1)


def to_neg_pos(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [0, 1] to [-1, 1]."""
    return torch.clamp(tensor * 2.0 - 1.0, min=-1, max=1)


def to_3ch(depth_map: torch.Tensor) -> torch.Tensor:
    """Repeat depth map to 3 channels."""
    return depth_map.repeat(1, 3, 1, 1) if depth_map.shape[1] == 1 else depth_map


## Radar Handler Functions ##
def radar_collator(
    radar_amplitude: torch.Tensor,
    radar_phase: Optional[torch.Tensor] = None,
    scale_factor: float = 0.001,
) -> torch.Tensor:
    """Build the raw amplitude/phase layout consumed by GRT inference."""
    if radar_phase is None:
        return radar_amplitude

    # Kept for compatibility with the copied loader API. GRT performs its
    # fixed amplitude normalization inside translate_radar().
    _ = scale_factor
    amplitude = radar_amplitude.permute(0, 1, 3, 2, 4)
    phase = radar_phase.permute(0, 1, 3, 2, 4)
    return torch.stack([amplitude, phase], dim=-1)


## Depth Handler Functions ##
def depth_collator(
    depth: Union[torch.Tensor, np.ndarray],
    max_depth_m: float = 11.2,
    target_size: Tuple[int, int] = (128, 256),
) -> Union[torch.Tensor, np.ndarray]:
    """Clamp, normalize to [0, 1], and resize depth."""
    is_numpy = isinstance(depth, np.ndarray)
    if is_numpy:
        depth = torch.from_numpy(depth)

    depth = depth.float()
    original_shape = depth.shape

    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)

    invalid_mask = ~(torch.isfinite(depth) & (depth >= 0))
    depth[invalid_mask] = 0.0

    depth = torch.clamp(depth, min=0.0, max=max_depth_m)
    depth = depth / max_depth_m

    invalid_mask = ~torch.isfinite(depth)
    depth[invalid_mask] = 0.0

    resized = T.Resize(
        target_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True
    )(depth)

    if len(original_shape) == 2:
        resized = resized.squeeze(0)

    return resized.numpy() if is_numpy else resized


## DJI RGB Handler Functions ##
def dji_rgb_collator(
    image: torch.Tensor,
    target_size: Tuple[int, int] = (128, 256),
    calibrate: bool = True,
) -> torch.Tensor:
    """Resize (and optionally calibrate) a batch of DJI RGB images.

    Args:
        image: Batch of DJI RGB images as torch tensor (B, C, H, W) in CHW format.
        target_size: Target resolution as (height, width).
        calibrate: When False (default), simply resize to target_size without any
            undistortion or reprojection.  When True, apply fisheye undistortion,
            homography projection, and crop before resizing.

    Returns:
        Batch of processed and resized torch tensors in CHW format (B, C, H, W),
        normalised to [0, 1].
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(image)}")

    if image.dim() != 4:
        raise ValueError(
            f"Expected 4D tensor (B, C, H, W), got {image.dim()}D tensor with shape {image.shape}"
        )

    target_h, target_w = target_size

    # --- Fast path: no calibration, just resize ---
    if not calibrate:
        # Normalise to [0, 1] if not already
        img = image.float()
        if img.max() > 1.0:
            img = img / 255.0
        resized = T.Resize(
            (target_h, target_w),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )(img)
        return resized

    # --- Full calibration path ---
    # Calibration parameters (hardcoded)
    CALIB_K_DJI = np.array(
        [
            [718.48555551, 0.0, 963.36465011],
            [0.0, 720.25844189, 537.87569913],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    CALIB_D_DJI = np.array(
        [0.19022699, 0.03466753, 0.05858962, -0.07070669], dtype=np.float64
    )
    CALIB_DEFISH_SHAPE = (1920, 1080)
    CALIB_DEFISH_BALANCE = 0.2
    CALIB_H_FULL = np.array(
        [
            [0.8274446551892256, -0.0742944198979625, 80.23797348979947],
            [-0.014725864916652691, 0.8471179917075127, 28.27366063997317],
            [-5.083573451500717e-05, -6.846079418201229e-05, 1.0],
        ],
        dtype=np.float64,
    )
    CALIB_OUT_SIZE = (1918, 1105)
    CALIB_CROP = (115, 255, 1400, 760)  # top, left, right, bottom

    # Precompute defish maps
    R_DEFISH = np.eye(3)
    K_NEW_DEFISH = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        CALIB_K_DJI,
        CALIB_D_DJI,
        CALIB_DEFISH_SHAPE,
        R_DEFISH,
        balance=CALIB_DEFISH_BALANCE,
        fov_scale=1.0,
    )
    MAP1_DEFISH, MAP2_DEFISH = cv2.fisheye.initUndistortRectifyMap(
        CALIB_K_DJI,
        CALIB_D_DJI,
        R_DEFISH,
        K_NEW_DEFISH,
        CALIB_DEFISH_SHAPE,
        cv2.CV_16SC2,
    )

    batch_size = image.shape[0]

    # Handle normalized tensors [0, 1] or unnormalized
    if image.max() <= 1.0:
        img_batch = (image.permute(0, 2, 3, 1).numpy() * 255).astype(
            np.uint8
        )  # (B, H, W, C)
    else:
        img_batch = image.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # (B, H, W, C)

    # Process each image in the batch
    calibrated_images = []

    for i in range(batch_size):
        img = img_batch[i]

        # Resize to original DJI resolution (1920x1080) before calibration
        if img.shape[1] != 1920 or img.shape[0] != 1080:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        # Defish
        img = cv2.remap(img, MAP1_DEFISH, MAP2_DEFISH, interpolation=cv2.INTER_LINEAR)

        # Project
        img = cv2.warpPerspective(
            img, CALIB_H_FULL, CALIB_OUT_SIZE, flags=cv2.INTER_LINEAR
        )

        # Crop
        top, left, right, bottom = CALIB_CROP
        img = img[top:bottom, left:right]

        # Resize to target resolution
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        calibrated_images.append(img)

    # Stack and convert to torch tensor in CHW format
    img_batch = np.stack(calibrated_images, axis=0)  # (B, H, W, C)
    img_tensor = (
        torch.from_numpy(img_batch).permute(0, 3, 1, 2).float() / 255.0
    )  # (B, C, H, W)

    return img_tensor


## Fisheye RGB Handler Functions ##
def fisheye_rgb_collator(
    image: torch.Tensor,
    target_size: Tuple[int, int] = (128, 256),
) -> torch.Tensor:
    """Calibrate and resize Fisheye RGB image batch.

    Args:
        image: Batch of Fisheye RGB images as torch tensor (B, C, H, W) in CHW format
        target_size: Target resolution as (width, height)

    Returns:
        Batch of calibrated and resized torch tensors in CHW format (B, C, H, W)
    """
    # Hardcoded parameters for Black Magic Micro Studio Camera 4K G2
    # with Olympus 9mm f/8 Fisheye lens
    IMAGE_WIDTH = 1920
    IMAGE_HEIGHT = 1080
    FOCAL_LENGTH_X = 0.613260
    FOCAL_LENGTH_Y = 0.613260
    CENTER_X = 0.5
    CENTER_Y = 0.5
    K1 = -0.120000
    K2 = -0.015000

    # Precompute undistortion maps
    w, h = IMAGE_WIDTH, IMAGE_HEIGHT
    x_out, y_out = np.meshgrid(np.arange(w), np.arange(h))
    x_norm = (x_out - w * CENTER_X) / (w * FOCAL_LENGTH_X)
    y_norm = (y_out - h * CENTER_Y) / (h * FOCAL_LENGTH_Y)
    r = np.sqrt(x_norm**2 + y_norm**2)
    r_distorted = r + K1 * r**2 + K2 * r**3
    r_safe = np.where(r > 0, r, 1.0)
    scale = np.where(r > 0, r_distorted / r_safe, 1.0)
    x_norm_distorted = x_norm * scale
    y_norm_distorted = y_norm * scale
    map_x = (
        x_norm_distorted * (w * FOCAL_LENGTH_X) + w * CENTER_X
    ).astype(np.float32)
    map_y = (
        y_norm_distorted * (h * FOCAL_LENGTH_Y) + h * CENTER_Y
    ).astype(np.float32)

    # Expect batched torch tensor (B, C, H, W)
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(image)}")
    
    if image.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got {image.dim()}D tensor with shape {image.shape}")
    
    batch_size = image.shape[0]
    
    # Handle normalized tensors [0, 1] or unnormalized
    if image.max() <= 1.0:
        img_batch = (image.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)  # (B, H, W, C)
    else:
        img_batch = image.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # (B, H, W, C)

    # Process each image in the batch
    calibrated_images = []
    target_h, target_w = target_size
    
    for i in range(batch_size):
        img = img_batch[i]
        
        # Resize to original resolution (1920x1080) if needed
        if img.shape[1] != IMAGE_WIDTH or img.shape[0] != IMAGE_HEIGHT:
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Undistort fisheye
        img = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Resize to target resolution
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        calibrated_images.append(img)
    
    # Stack and convert to torch tensor in CHW format
    img_batch = np.stack(calibrated_images, axis=0)  # (B, H, W, C)
    img_tensor = torch.from_numpy(img_batch).permute(0, 3, 1, 2).float() / 255.0  # (B, C, H, W)

    return img_tensor
