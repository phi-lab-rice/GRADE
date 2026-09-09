import cv2
import numpy as np

# ZED intrinsics at reference resolution 1280x720 (same values as PointCloudConverter)
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


def sample_depth_as_radar(
    depth_mm: np.ndarray,
    n_samples: int = 100,
    target_shape: tuple = (300, 1280),
    max_depth_m: float = 11.2,
    seed: int | None = None,
) -> tuple:
    """
    Randomly sample points from a ground truth ZED depth map and treat them as
    radar points, mimicking the sparse depth input the model expects.

    The input depth is resized from its native resolution (e.g. 896x504) to
    target_shape using nearest-neighbor interpolation so raw mm values are
    preserved. Camera intrinsics are scaled from the 1280x720 ZED reference to
    match the target resolution.

    Args:
        depth_mm:     Ground truth depth map, shape (H, W), dtype uint16, in mm.
        n_samples:    Number of points to randomly sample (default: 100).
        target_shape: (target_H, target_W) to resize to before sampling.
                      Default (300, 1280) matches the model's required input.
        max_depth_m:  Maximum valid depth in meters — pixels beyond this are
                      treated as invalid (default: 11.2 m).
        seed:         Optional random seed for reproducibility.

    Returns:
        points (np.ndarray):       (N, 3) float32 array of [X, Y, Z] in meters,
                                   in camera coordinate frame. N <= n_samples.
        sparse_depth (np.ndarray): (target_H, target_W) float32 sparse depth map
                                   with only the N sampled pixels filled (meters),
                                   zeros elsewhere.
    """
    target_h, target_w = target_shape

    # --- 1. Resize depth map (nearest-neighbor preserves raw mm values) ---
    in_h, in_w = depth_mm.shape
    if (in_h, in_w) != (target_h, target_w):
        depth_resized = cv2.resize(
            depth_mm, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
    else:
        depth_resized = depth_mm.copy()

    # --- 2. Scale intrinsics from 1280x720 reference to target resolution ---
    sx = target_w / float(_ZED_REF_W)
    sy = target_h / float(_ZED_REF_H)
    fx = _K_ZED_REF[0, 0] * sx
    fy = _K_ZED_REF[1, 1] * sy
    cx = _K_ZED_REF[0, 2] * sx
    cy = _K_ZED_REF[1, 2] * sy

    # --- 3. Convert to float meters and find valid pixels ---
    depth_m = depth_resized.astype(np.float32) / 1000.0
    valid_mask = (depth_m > 0) & (depth_m <= max_depth_m)
    valid_v, valid_u = np.where(valid_mask)  # row (V), col (U)

    if len(valid_v) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((target_h, target_w), dtype=np.float32),
        )

    # --- 4. Randomly sample up to n_samples valid pixels ---
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(valid_v))
    indices = rng.choice(len(valid_v), size=n, replace=False)
    sampled_v = valid_v[indices]
    sampled_u = valid_u[indices]
    sampled_z = depth_m[sampled_v, sampled_u]

    # --- 5. Back-project to 3D camera coordinates (pinhole model) ---
    X = (sampled_u - cx) * sampled_z / fx
    Y = (sampled_v - cy) * sampled_z / fy
    points = np.stack([X, Y, sampled_z], axis=1).astype(np.float32)  # (N, 3)

    # --- 6. Build sparse depth map ---
    sparse_depth = np.zeros((target_h, target_w), dtype=np.float32)
    sparse_depth[sampled_v, sampled_u] = sampled_z

    return points, sparse_depth
