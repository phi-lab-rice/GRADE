import os
import logging
from typing import Iterable, Tuple

import cv2
import numpy as np
import pyzed.sl as sl
from tqdm import tqdm

log = logging.getLogger(__name__)


def extract_camera_data(
    svo_path: str,
    frame_indices: Iterable[int],
    depth_mode: str = "NEURAL_PLUS",
    confidence_threshold: int = 100,
    texture_confidence_threshold: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract selected RGB and depth frames from a recorded SVO/SVO2.

    This function ensures 1:1 pixel alignment between RGB and depth by:
    - Using the same rectified view (sl.VIEW.LEFT) for RGB
    - Using sl.MEASURE.DEPTH which matches the rectified resolution
    - Validating dimensions on the first frame to catch FOV mismatches

    Parameters:
    - svo_path: path to the .svo/.svo2 file recorded by `src/cameraRecorder.py`.
    - frame_indices: iterable of camera frame indices to keep
      (e.g. the `camera_frame_idx` column from `sync_pairs.csv`).
    - depth_mode: ZED depth mode, e.g. "NEURAL_PLUS" for the finest model.
      Note: NEURAL modes may fill in peripheral regions beyond traditional stereo overlap.
    - confidence_threshold: Depth confidence threshold (1-100). Higher values (closer to 100)
      fill in more depth holes but may add noise. Lower values filter more strictly.
      Default: 100 (least filtering, most coverage).
    - texture_confidence_threshold: Texture confidence threshold (1-100). Controls depth
      estimation in uniform/textureless areas. Lower values allow depth in less detailed
      regions. Default: 100 (least filtering).

    Returns:
    - rgb: array of shape (N, H, W, 3), dtype uint8, LEFT camera RGB frames (rectified).
    - depth_mm: array of shape (N, H, W), dtype float32, depth in millimeters (rectified).
      RGB and depth are guaranteed to have identical (H, W) dimensions for 1:1 alignment.
    """
    if not os.path.exists(svo_path):
        raise FileNotFoundError(f"SVO not found: {svo_path}")

    indices = np.asarray(list(frame_indices), dtype=np.int64)
    if indices.size == 0:
        return (
            np.empty((0, 0, 0, 3), dtype=np.uint8),
            np.empty((0, 0), dtype=np.float32),
        )

    # Ensure indices are sorted and unique.
    indices = np.unique(indices)

    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False  # Process SVO as fast as possible
    init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode.upper().replace(" ", "_"), sl.DEPTH_MODE.NEURAL)
    init.coordinate_units = sl.UNIT.MILLIMETER  # depth in millimeters

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO: {status}")
    
    log.info(f"Opened SVO with depth mode: {depth_mode}")
    
    # Validate camera calibration and resolution to ensure 1:1 RGB-Depth alignment
    cam_info = zed.get_camera_information()
    resolution = cam_info.camera_configuration.resolution
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    
    log.info(f"Camera resolution: {resolution.width}x{resolution.height}")
    log.info(f"Left camera calibration - fx={calib.fx:.2f}, fy={calib.fy:.2f}, cx={calib.cx:.2f}, cy={calib.cy:.2f}")

    # Configure runtime parameters for depth estimation
    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = confidence_threshold
    runtime_params.texture_confidence_threshold = texture_confidence_threshold
    log.info(f"Depth confidence thresholds: confidence={confidence_threshold}, texture={texture_confidence_threshold}")

    num_frames_svo = zed.get_svo_number_of_frames()
    log.info(f"SVO contains {num_frames_svo} frames")
    log.info(f"Extracting {len(indices)} frame indices (min={indices.min()}, max={indices.max()})")

    # Convert indices to a set for O(1) lookup
    indices_set = set(indices.tolist())
    
    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []

    image_left = sl.Mat()
    depth_mat = sl.Mat()

    try:
        # Loop through all frames in SVO sequentially
        # We must process sequentially because SVO cannot seek randomly
        extracted_count = 0
        first_frame_validated = False
        
        with tqdm(total=len(indices), desc="ZED camera extraction", unit="frame") as pbar:
            for frame_idx in range(num_frames_svo):
                # Grab the next frame from SVO
                grab_status = zed.grab(runtime_params)
                if grab_status != sl.ERROR_CODE.SUCCESS:
                    # End of file or error
                    log.warning(f"Grab failed at frame {frame_idx}: {grab_status}")
                    break
                
                # Always retrieve RGB and depth after grab() to keep SDK state clean
                # RGB: LEFT view (rectified)
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                bgra = image_left.get_data()  # (H, W, 4) uint8 - ZED SDK returns BGRA
                rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)  # (H, W, 3) uint8

                # Depth (millimeters) - should match RGB resolution
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                depth = depth_mat.get_data()  # (H, W) or (H, W, 1) - ZED reuses this buffer
                depth = np.asarray(depth, dtype=np.float32).squeeze().copy()  # (H, W) float32, own copy to prevent buffer reuse issues
                
                # Replace NaN values with 0 (invalid depth)
                depth = np.nan_to_num(depth, nan=0.0)

                # Validate 1:1 alignment on first frame
                if not first_frame_validated:
                    if rgb.shape[:2] != depth.shape:
                        raise ValueError(
                            f"RGB-Depth dimension mismatch! RGB: {rgb.shape[:2]}, Depth: {depth.shape}. "
                            f"This indicates FOV alignment issues."
                        )
                    log.info(f"✓ Validated 1:1 RGB-Depth alignment: {rgb.shape[:2]}")
                    first_frame_validated = True

                # Only keep this frame if it's in our desired indices
                if frame_idx in indices_set:
                    rgb_frames.append(rgb)
                    depth_frames.append(depth)
                    extracted_count += 1
                    pbar.update(1)
                # If frame_idx not in indices_set, we skip this frame (don't append)
        
        log.info(f"Finished: Extracted {extracted_count}/{len(indices)} frames from {num_frames_svo} total frames")
    finally:
        zed.close()

    if not rgb_frames:
        log.warning("No frames were extracted!")
        return (
            np.empty((0, 0, 0, 3), dtype=np.uint8),
            np.empty((0, 0), dtype=np.float32),
        )

    rgb_arr = np.stack(rgb_frames, axis=0)
    depth_arr = np.stack(depth_frames, axis=0)
    
    log.info(f"Final arrays - RGB: {rgb_arr.shape}, Depth: {depth_arr.shape}")

    return rgb_arr, depth_arr
