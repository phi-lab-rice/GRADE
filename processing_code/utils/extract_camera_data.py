"""Load aligned ZED RGB and metric depth without the ZED SDK."""

from typing import Iterable, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def extract_camera_data(
    rgb_video_path: str,
    depth_h5_path: str,
    frame_indices: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected rectified-left RGB frames and aligned uint16-mm depth.

    Frame index zero is the first frame in both files. Depth zero means invalid.
    """
    indices = np.unique(np.asarray(list(frame_indices), dtype=np.int64))
    if indices.size == 0:
        return (np.empty((0, 0, 0, 3), dtype=np.uint8),
                np.empty((0, 0, 0), dtype=np.uint16),
                np.empty(4, dtype=np.float32))
    if indices[0] < 0:
        raise ValueError("ZED frame indices must be nonnegative")

    with h5py.File(depth_h5_path, "r") as f:
        if "depth_mm" not in f:
            raise ValueError(f"Missing depth_mm dataset: {depth_h5_path}")
        depth = f["depth_mm"]
        if depth.ndim != 3 or depth.dtype != np.uint16:
            raise ValueError("depth_mm must have shape (frames, height, width) and dtype uint16")
        frame_count, height, width = depth.shape
        if indices[-1] >= frame_count:
            raise ValueError(f"ZED index {indices[-1]} exceeds {frame_count} depth frames")
        intrinsics = np.asarray(f.attrs["intrinsics_fx_fy_cx_cy"], dtype=np.float32)
        if intrinsics.shape != (4,):
            raise ValueError("Invalid ZED intrinsics")

        cap = cv2.VideoCapture(rgb_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open ZED RGB video: {rgb_video_path}")
        try:
            video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if video_count > 0 and video_count != frame_count:
                raise ValueError(f"ZED RGB/depth frame count differs: {video_count} vs {frame_count}")
            if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (width, height):
                raise ValueError("ZED RGB/depth resolution differs")
            rgb_frames = []
            depth_frames = []
            wanted = set(indices.tolist())
            for frame_idx in tqdm(range(int(indices[-1]) + 1), desc="ZED extraction", unit="frame"):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"ZED RGB video ends before frame {frame_idx}")
                if frame_idx in wanted:
                    rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    depth_frames.append(depth[frame_idx])
        finally:
            cap.release()

    return np.stack(rgb_frames), np.stack(depth_frames), intrinsics
