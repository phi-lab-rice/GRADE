import os
import logging
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


def extract_dji_rgb(
    video_path: str,
    frame_indices: Iterable[int],
) -> np.ndarray:
    """
    Extract selected RGB frames from the prepared DJI MKV.

    Parameters:
    - video_path: path to the rectified, 1280x720, defaced DJI video.
    - frame_indices: iterable of DJI frame indices to keep
      (e.g. the `dji_frame_idx` column from `sync_triples.csv`).

    Returns:
    - rgb: array of shape (N, H, W, 3), dtype uint8, RGB frames.

    The release video is already rectified, cropped, resized, and defaced.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"DJI video not found: {video_path}")

    indices = np.asarray(list(frame_indices), dtype=np.int64)
    if indices.size == 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    # Ensure indices are sorted and unique
    indices = np.unique(indices)
    indices_set = set(indices.tolist())

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open DJI video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    log.info(f"DJI video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    log.info(f"Extracting {len(indices)} frame indices (min={indices.min()}, max={indices.max()})")

    rgb_frames = []
    frame_idx = 0
    extracted_count = 0

    with tqdm(total=len(indices), desc="DJI extraction", unit="frame") as pbar:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in indices_set:
                # OpenCV reads in BGR, convert to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frames.append(rgb)
                extracted_count += 1
                pbar.update(1)
                if extracted_count == len(indices):
                    break

            frame_idx += 1

    cap.release()
    log.info(f"Finished: Extracted {extracted_count}/{len(indices)} DJI frames from {frame_idx} total frames")

    if not rgb_frames:
        log.warning("No DJI frames were extracted!")
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    rgb_arr = np.stack(rgb_frames, axis=0)
    log.info(f"Final DJI RGB array: {rgb_arr.shape}")

    return rgb_arr
