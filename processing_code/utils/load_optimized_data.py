"""
Load processed multimodal data from optimized format.

Supports:
- radar.npy
- dji_rgb.avi, zed_rgb.avi (FFV1, read with OpenCV)
- zed_depth.npy (uint16 millimeters)
- metadata.json
"""

import os
import json

import cv2
import numpy as np


def _load_video_rgb(path: str) -> np.ndarray:
    """Load an RGB AVI (e.g. FFV1) as (N, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    if not frames:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def load_all_modalities(processed_dir: str) -> dict:
    """
    Load all modalities from a processed sequence directory.

    Args:
        processed_dir: Path to processed/<sequence_name>/

    Returns:
        dict with keys: radar, dji_rgb, zed_rgb, zed_depth, metadata.
        Missing modalities are None. zed_depth is (N, H, W) uint16 in millimeters.
    """
    with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    result = {
        "radar": None,
        "dji_rgb": None,
        "zed_rgb": None,
        "zed_depth": None,
        "metadata": metadata,
    }

    # Radar
    radar_path = os.path.join(processed_dir, "radar.npy")
    if os.path.isfile(radar_path):
        result["radar"] = np.load(radar_path)

    # DJI RGB
    dji_path = os.path.join(processed_dir, "dji_rgb.avi")
    if os.path.isfile(dji_path):
        result["dji_rgb"] = _load_video_rgb(dji_path)

    # ZED RGB
    zed_rgb_path = os.path.join(processed_dir, "zed_rgb.avi")
    if os.path.isfile(zed_rgb_path):
        result["zed_rgb"] = _load_video_rgb(zed_rgb_path)

    # ZED depth: uint16 npy (millimeters)
    zed_depth_path = os.path.join(processed_dir, "zed_depth.npy")
    if os.path.isfile(zed_depth_path):
        result["zed_depth"] = np.load(zed_depth_path)

    return result
