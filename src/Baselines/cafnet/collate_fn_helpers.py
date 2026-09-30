import cv2
import numpy as np
import torch
from functools import lru_cache
from typing import Callable, Dict, Sequence, Tuple, Union
from torchvision import transforms as T


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ZED intrinsics at 1280x720 reference resolution.
_K_ZED_REF = np.array(
    [
        [521.581604, 0.0, 636.33398438],
        [0.0, 521.581604, 373.10964966],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_ZED_REF_W = 1280
_ZED_REF_H = 720

# DJI calibration constants.
_CALIB_K_DJI = np.array(
    [
        [718.48555551, 0.0, 963.36465011],
        [0.0, 720.25844189, 537.87569913],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_CALIB_D_DJI = np.array(
    [0.19022699, 0.03466753, 0.05858962, -0.07070669], dtype=np.float64
)
_CALIB_DEFISH_SHAPE = (1920, 1080)
_CALIB_DEFISH_BALANCE = 0.2
_CALIB_H_FULL = np.array(
    [
        [0.8274446551892256, -0.0742944198979625, 80.23797348979947],
        [-0.014725864916652691, 0.8471179917075127, 28.27366063997317],
        [-5.083573451500717e-05, -6.846079418201229e-05, 1.0],
    ],
    dtype=np.float64,
)
_CALIB_OUT_SIZE = (1918, 1105)
_CALIB_CROP = (115, 255, 1400, 760)  # top, left, right, bottom


@lru_cache(maxsize=1)
def _get_dji_defish_maps() -> Tuple[np.ndarray, np.ndarray]:
    r_defish = np.eye(3)
    k_new_defish = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        _CALIB_K_DJI,
        _CALIB_D_DJI,
        _CALIB_DEFISH_SHAPE,
        r_defish,
        balance=_CALIB_DEFISH_BALANCE,
        fov_scale=1.0,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        _CALIB_K_DJI,
        _CALIB_D_DJI,
        r_defish,
        k_new_defish,
        _CALIB_DEFISH_SHAPE,
        cv2.CV_16SC2,
    )
    return map1, map2


def resize_depth_mm(depth_mm: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_size
    if depth_mm.shape[:2] == (target_h, target_w):
        return depth_mm
    return cv2.resize(depth_mm, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


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


def dji_rgb_collator(
    image: torch.Tensor,
    target_size: Tuple[int, int] = (128, 256),
) -> torch.Tensor:
    """Rectify and resize DJI RGB image batch.

    Args:
        image: Tensor with shape (B, C, H, W).
        target_size: Target resolution as (height, width).

    Returns:
        Tensor in CHW format (B, C, H, W), float32 in [0, 1].
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(image)}")

    if image.dim() != 4:
        raise ValueError(
            f"Expected 4D tensor (B, C, H, W), got {image.dim()}D tensor with shape {image.shape}"
        )

    map1_defish, map2_defish = _get_dji_defish_maps()
    target_h, target_w = target_size

    if image.max() <= 1.0:
        img_batch = (image.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
    else:
        img_batch = image.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    calibrated_images = []
    for img in img_batch:
        if img.shape[1] != 1920 or img.shape[0] != 1080:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        img = cv2.remap(img, map1_defish, map2_defish, interpolation=cv2.INTER_LINEAR)
        img = cv2.warpPerspective(
            img, _CALIB_H_FULL, _CALIB_OUT_SIZE, flags=cv2.INTER_LINEAR
        )

        top, left, right, bottom = _CALIB_CROP
        img = img[top:bottom, left:right]
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        calibrated_images.append(img)

    out_batch = np.stack(calibrated_images, axis=0)
    out_tensor = torch.from_numpy(out_batch).permute(0, 3, 1, 2).float() / 255.0
    return out_tensor


def point_cloud_to_sparse_depth(
    points_xyz: np.ndarray,
    target_shape: Tuple[int, int],
    max_depth_m: float,
) -> np.ndarray:
    """Project xyz radar points (meters) to a sparse depth image."""
    target_h, target_w = target_shape
    sparse_depth = np.zeros((target_h, target_w), dtype=np.float32)

    if points_xyz.size == 0:
        return sparse_depth

    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return sparse_depth

    valid = np.isfinite(pts).all(axis=1)
    valid &= pts[:, 2] > 0.0
    valid &= pts[:, 2] <= float(max_depth_m)
    pts = pts[valid]
    if pts.shape[0] == 0:
        return sparse_depth

    sx = target_w / float(_ZED_REF_W)
    sy = target_h / float(_ZED_REF_H)
    fx = _K_ZED_REF[0, 0] * sx
    fy = _K_ZED_REF[1, 1] * sy
    cx = _K_ZED_REF[0, 2] * sx
    cy = _K_ZED_REF[1, 2] * sy

    z = pts[:, 2]
    u = np.rint(pts[:, 0] * fx / z + cx).astype(np.int32)
    v = np.rint(pts[:, 1] * fy / z + cy).astype(np.int32)

    in_bounds = (u >= 0) & (u < target_w) & (v >= 0) & (v < target_h)
    if not np.any(in_bounds):
        return sparse_depth

    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds].astype(np.float32)

    min_depth = np.full((target_h, target_w), np.inf, dtype=np.float32)
    np.minimum.at(min_depth, (v, u), z)
    min_depth[~np.isfinite(min_depth)] = 0.0
    return min_depth


def build_radar_gt_map(
    depth_m: np.ndarray,
    sparse_depth: np.ndarray,
    patch_size: Tuple[int, int],
    max_dist_correspondence: float,
) -> np.ndarray:
    """Build confidence GT using local depth consistency around each radar pixel."""
    h, w = depth_m.shape
    radar_gt = np.zeros((h, w), dtype=np.float32)

    ys, xs = np.where(sparse_depth > 0)
    if len(ys) == 0:
        return radar_gt

    ext_h, ext_w = int(patch_size[0]), int(patch_size[1])
    for y, x in zip(ys, xs):
        radar_depth = sparse_depth[y, x]

        delta_x1 = min(x, ext_w)
        delta_y1 = min(y, ext_h)
        delta_x2 = min(w - x, ext_w)
        delta_y2 = min(h - y, ext_h)

        x1 = x - delta_x1
        y1 = y - delta_y1
        x2 = x + delta_x2
        y2 = y + delta_y2

        distance = np.abs(depth_m[y1:y2, x1:x2] - radar_depth)
        gt_label = (distance < float(max_dist_correspondence)).astype(np.float32)
        radar_gt[y1:y2, x1:x2] = gt_label

    return radar_gt


def make_rice_collate_fn(
    input_height: int,
    input_width: int,
    radar_max_depth_m: float,
    max_dist_correspondence: float,
    patch_size: Tuple[int, int],
) -> Callable[[Sequence[Dict[str, object]]], Tuple[torch.Tensor, ...]]:
    """Create collate_fn for RiceDataset samples.

    Each dataset sample should contain:
      - sample_idx: int
      - dji_rgb: (H, W, 3) uint8
      - zed_depth_mm: (H, W) uint16
      - radar_pcd_xyz: (N, 3) float32 in meters
    """

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)

    def _collate(batch: Sequence[Dict[str, object]]) -> Tuple[torch.Tensor, ...]:
        if len(batch) == 0:
            raise ValueError("Received empty batch in collate function")

        sample_indices = []
        rgb_batch = []
        depth_batch = []
        radar_batch = []
        radar_gt_batch = []

        for sample in batch:
            sample_indices.append(int(sample["sample_idx"]))

            rgb = np.asarray(sample["dji_rgb"]).copy()
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"Expected RGB shape (H, W, 3), got {rgb.shape}")
            rgb_batch.append(torch.from_numpy(np.transpose(rgb, (2, 0, 1))))

            depth_mm = np.asarray(sample["zed_depth_mm"]).copy()
            depth_mm = resize_depth_mm(depth_mm, (input_height, input_width))
            depth_m = depth_mm.astype(np.float32) / 1000.0
            invalid = ~(np.isfinite(depth_m) & (depth_m > 0.0))
            depth_m[invalid] = 0.0
            depth_batch.append(depth_m)

            radar_points = np.asarray(sample["radar_pcd_xyz"], dtype=np.float32)
            if radar_points.ndim != 2 or radar_points.shape[1] != 3:
                radar_points = np.zeros((0, 3), dtype=np.float32)

            if radar_points.shape[0] == 0:
                center_v = float(depth_m[input_height // 2, input_width // 2])
                if not np.isfinite(center_v):
                    center_v = 0.0
                radar_points = np.array([[0.0, 0.0, center_v]], dtype=np.float32)

            sparse_depth = point_cloud_to_sparse_depth(
                radar_points,
                target_shape=(input_height, input_width),
                max_depth_m=radar_max_depth_m,
            )
            radar_gt = build_radar_gt_map(
                depth_m,
                sparse_depth,
                patch_size=patch_size,
                max_dist_correspondence=max_dist_correspondence,
            )
            radar_batch.append(sparse_depth)
            radar_gt_batch.append(radar_gt)

        rgb_tensor = torch.stack(rgb_batch, dim=0).float()
        rgb_tensor = dji_rgb_collator(rgb_tensor, target_size=(input_height, input_width))
        rgb_tensor = (rgb_tensor - mean) / std

        depth_tensor = torch.from_numpy(np.stack(depth_batch, axis=0)).float().unsqueeze(1)
        radar_tensor = torch.from_numpy(np.stack(radar_batch, axis=0)).float().unsqueeze(1)
        radar_gt_tensor = (
            torch.from_numpy(np.stack(radar_gt_batch, axis=0)).float().unsqueeze(1)
        )
        idx_tensor = torch.tensor(sample_indices, dtype=torch.long)

        return idx_tensor, rgb_tensor, depth_tensor, radar_tensor, radar_gt_tensor

    return _collate


# Fisheye RGB Handler Functions ##
def fisheye_rgb_collator(
    image: torch.Tensor,
    target_size: Tuple[int, int] = (128, 256),
) -> torch.Tensor:
    """Calibrate and resize Fisheye RGB image batch.

    Args:
        image: Batch of Fisheye RGB images as torch tensor (B, C, H, W) in CHW format
        target_size: Target resolution as (height, width)

    Returns:
        Batch of calibrated and resized torch tensors in CHW format (B, C, H, W)
    """
    IMAGE_WIDTH = 1920
    IMAGE_HEIGHT = 1080
    FOCAL_LENGTH_X = 0.613260
    FOCAL_LENGTH_Y = 0.613260
    CENTER_X = 0.5
    CENTER_Y = 0.5
    K1 = -0.120000
    K2 = -0.015000

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
    map_x = (x_norm_distorted * (w * FOCAL_LENGTH_X) + w * CENTER_X).astype(np.float32)
    map_y = (y_norm_distorted * (h * FOCAL_LENGTH_Y) + h * CENTER_Y).astype(np.float32)

    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(image)}")

    if image.dim() != 4:
        raise ValueError(
            f"Expected 4D tensor (B, C, H, W), got {image.dim()}D tensor with shape {image.shape}"
        )

    if image.max() <= 1.0:
        img_batch = (image.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    else:
        img_batch = image.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    calibrated_images = []
    target_h, target_w = target_size

    for img in img_batch:
        if img.shape[1] != IMAGE_WIDTH or img.shape[0] != IMAGE_HEIGHT:
            img = cv2.resize(
                img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR
            )

        img = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        calibrated_images.append(img)

    out_batch = np.stack(calibrated_images, axis=0)
    out_tensor = torch.from_numpy(out_batch).permute(0, 3, 1, 2).float() / 255.0
    return out_tensor
