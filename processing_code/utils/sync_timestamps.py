import numpy as np
import h5py
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Union
import os
import glob
import tyro
import pyzed.sl as sl

def setup_logging(name):
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)

log = setup_logging("Sync")

def _load_h5_dataset_1d(h5_path: str, dataset: str) -> np.ndarray:
    """
    Load a 1D dataset from an HDF5 file.
    """
    with h5py.File(h5_path, 'r') as f:
        if dataset not in f:
            raise ValueError(f"Dataset not found in HDF5 file: {dataset!r} (file: {h5_path})")
        dset = f[dataset]
        # Many recorders preallocate more slots than actual frames; honor num_frames attr if present.
        num_frames_attr = f.attrs.get("num_frames")
        if isinstance(num_frames_attr, (int, np.integer)) and 0 < num_frames_attr <= dset.shape[0]:
            arr = dset[: int(num_frames_attr)]
        else:
            arr = dset[:]

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D dataset {dataset!r}, got shape {arr.shape} (file: {h5_path})")
    # Guard against preallocated-but-unwritten zeros (should never happen with time.time()*1000).
    mask = arr > 0
    return arr[mask]


def load_camera_timestamps_h5(camera_h5_path: str) -> np.ndarray:
    """
    Load camera timestamps from HDF5 and remove the last one.
    
    This ensures the number of timestamps is always safely within the SVO frame count,
    avoiding off-by-one errors that can occur with SVO files.
    
    Expected dataset: 'timestamps_ms' (milliseconds since Unix epoch).
    Returns: timestamps in milliseconds, shape (N-1,)
    """
    timestamps = _load_h5_dataset_1d(camera_h5_path, "timestamps_ms")
    
    # Always remove the last timestamp to be safe
    if len(timestamps) > 0:
        timestamps = timestamps[:-1]
    
    return timestamps


def load_radar_timestamps_h5(radar_h5_path: str) -> np.ndarray:
    """
    Load radar timestamps from `src/radarRecorder.py` output HDF5.

    Expected dataset: 'timestamps_ms' (milliseconds since Unix epoch).
    """
    return _load_h5_dataset_1d(radar_h5_path, "timestamps_ms")


def load_dji_timestamps_h5(dji_h5_path: str) -> np.ndarray:
    """
    Load DJI timestamps from `src/djiRecorder.py` output HDF5.

    Expected dataset: 'timestamps_ms' (milliseconds since Unix epoch).
    """
    return _load_h5_dataset_1d(dji_h5_path, "timestamps_ms")


@dataclass(frozen=True)
class SyncResult:
    """
    Output of 2-way timestamp synchronization (camera + radar).

    - `pairs`: list of (camera_frame_idx, radar_frame_idx)
    - `diffs_ms`: signed difference in ms for each pair: camera_ts - radar_ts
    """

    pairs: List[Tuple[int, int]]
    diffs_ms: np.ndarray
    camera_count: int
    radar_count: int
    tolerance_ms: float


@dataclass(frozen=True)
class SyncResult3Way:
    """
    Output of 3-way timestamp synchronization (radar + ZED depth + DJI RGB).

    - `triples`: list of (radar_frame_idx, zed_frame_idx, dji_frame_idx)
    - `diffs_radar_zed_ms`: signed difference radar_ts - zed_ts
    - `diffs_radar_dji_ms`: signed difference radar_ts - dji_ts
    """

    triples: List[Tuple[int, int, int]]
    diffs_radar_zed_ms: np.ndarray
    diffs_radar_dji_ms: np.ndarray
    radar_count: int
    zed_count: int
    dji_count: int
    tolerance_ms: float


def save_sync_csv(out_csv_path: str, result: Union[SyncResult, SyncResult3Way]) -> None:
    """
    Save sync pairs/triples to CSV.
    
    For SyncResult (2-way):
        camera_frame_idx,radar_frame_idx,diff_ms
    
    For SyncResult3Way (3-way):
        radar_frame_idx,zed_frame_idx,dji_frame_idx,diff_radar_zed_ms,diff_radar_dji_ms
    """
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    
    if isinstance(result, SyncResult3Way):
        # 3-way sync
        radar_idx = np.array([t[0] for t in result.triples], dtype=np.int64)
        zed_idx = np.array([t[1] for t in result.triples], dtype=np.int64)
        dji_idx = np.array([t[2] for t in result.triples], dtype=np.int64)
        arr = np.column_stack([
            radar_idx, 
            zed_idx, 
            dji_idx,
            result.diffs_radar_zed_ms.astype(np.float64, copy=False),
            result.diffs_radar_dji_ms.astype(np.float64, copy=False)
        ])
        header = "radar_frame_idx,zed_frame_idx,dji_frame_idx,diff_radar_zed_ms,diff_radar_dji_ms"
        fmt = ["%d", "%d", "%d", "%.6f", "%.6f"]
    else:
        # 2-way sync (backward compatibility)
        cam_idx = np.array([p[0] for p in result.pairs], dtype=np.int64)
        rad_idx = np.array([p[1] for p in result.pairs], dtype=np.int64)
        arr = np.column_stack([cam_idx, rad_idx, result.diffs_ms.astype(np.float64, copy=False)])
        header = "camera_frame_idx,radar_frame_idx,diff_ms"
        fmt = ["%d", "%d", "%.6f"]
    
    np.savetxt(out_csv_path, arr, fmt=fmt, delimiter=",", header=header, comments="")


def _pick_latest_file(directory: str, pattern: str) -> str:
    """
    Pick the latest file (by lexicographic sort) matching a pattern in a directory.

    This works for our timestamped filenames like:
    - camera_timestamps_YYYYmmdd_HHMMSS.h5
    - radar_YYYYmmdd_HHMMSS.h5
    """
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} in directory: {directory}")
    matches.sort()
    return matches[-1]


def _pick_latest_file_multi_ext(directory: str, patterns: list[str]) -> str:
    """
    Pick the latest file matching any of multiple patterns.
    
    Useful for finding files with different extensions (e.g., .mkv or .mp4).
    
    Example:
        path = _pick_latest_file_multi_ext("data", ["dji_*.mkv", "dji_*.mp4"])
    """
    all_matches = []
    for pattern in patterns:
        matches = glob.glob(os.path.join(directory, pattern))
        all_matches.extend(matches)
    
    if not all_matches:
        patterns_str = " or ".join(patterns)
        raise FileNotFoundError(f"No files matching {patterns_str} in directory: {directory}")
    
    all_matches.sort()
    return all_matches[-1]


def cli(
    directory: Optional[str] = None,
    camera_timestamps_h5: Optional[str] = None,
    radar_h5: Optional[str] = None,
    svo_path: Optional[str] = None,
    out_csv: str = os.path.join("data", "sync_pairs.csv"),
    tolerance_ms: float = 50.0,
    enforce_one_to_one: bool = True,
) -> None:
    """
    CLI wrapper for timestamp synchronization.

    Usage patterns:

    1) Directory mode (auto-pick latest files):
       - Provide `directory` and omit `camera_timestamps_h5` / `radar_h5` / `svo_path`.

    2) File mode (explicit paths):
       - Provide BOTH `camera_timestamps_h5` and `radar_h5`, optionally `svo_path`.
    """
    if directory is not None:
        if camera_timestamps_h5 is not None or radar_h5 is not None or svo_path is not None:
            raise ValueError("If 'directory' is provided, do not also pass explicit file paths.")
        camera_timestamps_h5 = _pick_latest_file(directory, "camera_timestamps_*.h5")
        radar_h5 = _pick_latest_file(directory, "radar_*.h5")
        svo_path = _pick_latest_file(directory, "zed_*.svo2")
        log.info(f"Auto-picked camera timestamps: {camera_timestamps_h5}")
        log.info(f"Auto-picked radar file: {radar_h5}")
        log.info(f"Auto-picked SVO: {svo_path}")

    if camera_timestamps_h5 is None or radar_h5 is None:
        raise ValueError(
            "Provide either:\n"
            "- directory=<folder containing camera_timestamps_*.h5 and radar_*.h5>, OR\n"
            "- camera_timestamps_h5=<path> AND radar_h5=<path>."
        )

    result = synchronize_timestamps(
        camera_timestamps_h5=camera_timestamps_h5,
        radar_h5=radar_h5,
        tolerance_ms=tolerance_ms,
        enforce_one_to_one=enforce_one_to_one,
        svo_path=svo_path,
    )
    save_sync_csv(out_csv, result)
    log.info(f"Wrote sync CSV: {out_csv}")

def _nearest_unused_camera_index(
    cam_ts: np.ndarray,
    target_t: float,
    used_camera: set[int],
    start_idx: int,
) -> Optional[int]:
    """
    Find nearest unused camera index to `target_t`.

    `start_idx` is the insertion index from `np.searchsorted(cam_ts, target_t)`.
    We expand outward until we find an unused camera frame.
    """
    left = start_idx - 1
    right = start_idx
    while left >= 0 or right < len(cam_ts):
        cand_left = left if left >= 0 else None
        cand_right = right if right < len(cam_ts) else None

        if cand_left is None and cand_right is None:
            return None

        best: Optional[int] = None
        best_abs = float("inf")

        for cand in (cand_left, cand_right):
            if cand is None:
                continue
            if cand in used_camera:
                continue
            abs_diff = abs(float(cam_ts[cand] - target_t))
            if abs_diff < best_abs:
                best_abs = abs_diff
                best = int(cand)

        if best is not None:
            return best

        left -= 1
        right += 1

    return None


def synchronize_timestamps(
    camera_timestamps_h5: str,
    radar_h5: str,
    tolerance_ms: float = 50.0,
    enforce_one_to_one: bool = True,
    strategy: Literal["nearest"] = "nearest",
    svo_path: Optional[str] = None,
) -> SyncResult:
    """
    Synchronize camera and radar timestamps.

    Assumptions (true for the new recorders):
    - Both timestamp streams are in **milliseconds since Unix epoch** (from `time.time()*1000`).
    - Each stream is **monotonically increasing**.

    Parameters:
    - svo_path: Optional path to SVO2 file. If provided, validates that camera timestamps
      don't exceed the actual number of frames in the SVO.

    Returns pairs and per-pair signed diffs: Δt = t_cam - t_radar.
    """
    if strategy != "nearest":
        raise ValueError(f"Unsupported strategy: {strategy!r}")

    cam_ts = load_camera_timestamps_h5(camera_timestamps_h5)
    rad_ts = load_radar_timestamps_h5(radar_h5)
    
    log.info(f"Camera timestamps: {len(cam_ts)} (last timestamp already removed for SVO safety)")
    
    # Ensure float for arithmetic (HDF5 may store float64 already, but keep consistent)
    cam_ts = cam_ts.astype(np.float64, copy=False)
    rad_ts = rad_ts.astype(np.float64, copy=False)

    # Reverse logic: iterate radar frames and pick nearest camera frame.
    # Goal: keep (as many as possible) radar frames, since radar is the lower FPS stream.
    pairs: List[Tuple[int, int]] = []
    diffs: List[float] = []
    used_camera: set[int] = set()

    for rad_idx, r_t in enumerate(rad_ts):
        insert_idx = int(np.searchsorted(cam_ts, r_t))

        if enforce_one_to_one:
            cam_idx = _nearest_unused_camera_index(cam_ts, float(r_t), used_camera, insert_idx)
        else:
            # Nearest of the two immediate neighbors (can reuse camera frames).
            cand0 = insert_idx - 1 if insert_idx > 0 else None
            cand1 = insert_idx if insert_idx < len(cam_ts) else None
            best: Optional[int] = None
            best_abs = float("inf")
            for cand in (cand0, cand1):
                if cand is None:
                    continue
                abs_diff = abs(float(cam_ts[cand] - r_t))
                if abs_diff < best_abs:
                    best_abs = abs_diff
                    best = int(cand)
            cam_idx = best

        if cam_idx is None:
            # No camera frames at all (or no unused ones left).
            break

        signed = float(cam_ts[cam_idx] - r_t)
        abs_diff = abs(signed)

        if abs_diff > tolerance_ms:
            continue

        pairs.append((int(cam_idx), int(rad_idx)))
        diffs.append(signed)
        if enforce_one_to_one:
            used_camera.add(int(cam_idx))

    diffs_ms = np.array(diffs, dtype=np.float64)
    log.info(
        f"Synchronized {len(pairs)} pairs | "
        f"camera={len(cam_ts)} radar={len(rad_ts)} | tol={tolerance_ms}ms | "
        f"one_to_one={enforce_one_to_one}"
    )
    return SyncResult(
        pairs=pairs,
        diffs_ms=diffs_ms,
        camera_count=int(len(cam_ts)),
        radar_count=int(len(rad_ts)),
        tolerance_ms=float(tolerance_ms),
    )

if __name__ == "__main__":
    tyro.cli(cli)


def synchronize_timestamps_3way(
    radar_h5: str,
    zed_timestamps_h5: str,
    dji_timestamps_h5: str,
    tolerance_ms: float = 50.0,
    enforce_one_to_one: bool = True,
    svo_path: Optional[str] = None,
    dji_video_path: Optional[str] = None,
) -> SyncResult3Way:
    """
    Synchronize 3 timestamp streams: radar (reference), ZED depth, DJI RGB.
    
    Strategy:
    - Radar is the lowest FPS stream (typically ~10 Hz), so we use it as the reference.
    - For each radar frame, find the nearest ZED frame and nearest DJI frame.
    - Both must be within tolerance_ms of the radar timestamp.
    
    Parameters:
    - radar_h5: Path to radar HDF5 file with timestamps_ms
    - zed_timestamps_h5: Path to ZED camera timestamps HDF5
    - dji_timestamps_h5: Path to DJI timestamps HDF5
    - tolerance_ms: Maximum time difference in milliseconds
    - enforce_one_to_one: If True, each ZED/DJI frame can only be used once
    - svo_path: Optional path to SVO2 file for frame count validation
    - dji_video_path: Optional path to DJI video for frame count validation
    
    Returns:
    - SyncResult3Way with triples (radar_idx, zed_idx, dji_idx)
    """
    # Load all timestamps
    radar_ts = load_radar_timestamps_h5(radar_h5)
    zed_ts = load_camera_timestamps_h5(zed_timestamps_h5)
    dji_ts = load_dji_timestamps_h5(dji_timestamps_h5)
    
    log.info(f"ZED timestamps: {len(zed_ts)} (last timestamp already removed for SVO safety)")
    
    # Ensure float64 for arithmetic
    radar_ts = radar_ts.astype(np.float64, copy=False)
    zed_ts = zed_ts.astype(np.float64, copy=False)
    dji_ts = dji_ts.astype(np.float64, copy=False)
    
    log.info(f"Timestamp ranges:")
    log.info(f"  Radar: {len(radar_ts)} frames, {radar_ts[0]:.2f} - {radar_ts[-1]:.2f} ms")
    log.info(f"  ZED:   {len(zed_ts)} frames, {zed_ts[0]:.2f} - {zed_ts[-1]:.2f} ms")
    log.info(f"  DJI:   {len(dji_ts)} frames, {dji_ts[0]:.2f} - {dji_ts[-1]:.2f} ms")
    
    # Iterate through radar frames (lowest FPS, reference stream)
    triples = []
    diffs_radar_zed = []
    diffs_radar_dji = []
    used_zed = set()
    used_dji = set()
    
    for radar_idx, r_t in enumerate(radar_ts):
        # Find nearest ZED frame
        zed_insert_idx = int(np.searchsorted(zed_ts, r_t))
        if enforce_one_to_one:
            zed_idx = _nearest_unused_camera_index(zed_ts, float(r_t), used_zed, zed_insert_idx)
        else:
            cand0 = zed_insert_idx - 1 if zed_insert_idx > 0 else None
            cand1 = zed_insert_idx if zed_insert_idx < len(zed_ts) else None
            zed_idx = None
            best_abs = float("inf")
            for cand in (cand0, cand1):
                if cand is None:
                    continue
                abs_diff = abs(float(zed_ts[cand] - r_t))
                if abs_diff < best_abs:
                    best_abs = abs_diff
                    zed_idx = int(cand)
        
        if zed_idx is None:
            continue
        
        zed_diff = float(r_t - zed_ts[zed_idx])
        if abs(zed_diff) > tolerance_ms:
            continue
        
        # Find nearest DJI frame
        dji_insert_idx = int(np.searchsorted(dji_ts, r_t))
        if enforce_one_to_one:
            dji_idx = _nearest_unused_camera_index(dji_ts, float(r_t), used_dji, dji_insert_idx)
        else:
            cand0 = dji_insert_idx - 1 if dji_insert_idx > 0 else None
            cand1 = dji_insert_idx if dji_insert_idx < len(dji_ts) else None
            dji_idx = None
            best_abs = float("inf")
            for cand in (cand0, cand1):
                if cand is None:
                    continue
                abs_diff = abs(float(dji_ts[cand] - r_t))
                if abs_diff < best_abs:
                    best_abs = abs_diff
                    dji_idx = int(cand)
        
        if dji_idx is None:
            continue
        
        dji_diff = float(r_t - dji_ts[dji_idx])
        if abs(dji_diff) > tolerance_ms:
            continue
        
        # Both ZED and DJI within tolerance - add triple
        triples.append((int(radar_idx), int(zed_idx), int(dji_idx)))
        diffs_radar_zed.append(zed_diff)
        diffs_radar_dji.append(dji_diff)
        
        if enforce_one_to_one:
            used_zed.add(int(zed_idx))
            used_dji.add(int(dji_idx))
    
    diffs_radar_zed_ms = np.array(diffs_radar_zed, dtype=np.float64)
    diffs_radar_dji_ms = np.array(diffs_radar_dji, dtype=np.float64)
    
    log.info(
        f"Synchronized {len(triples)} triples | "
        f"radar={len(radar_ts)} zed={len(zed_ts)} dji={len(dji_ts)} | "
        f"tol={tolerance_ms}ms | one_to_one={enforce_one_to_one}"
    )
    
    return SyncResult3Way(
        triples=triples,
        diffs_radar_zed_ms=diffs_radar_zed_ms,
        diffs_radar_dji_ms=diffs_radar_dji_ms,
        radar_count=int(len(radar_ts)),
        zed_count=int(len(zed_ts)),
        dji_count=int(len(dji_ts)),
        tolerance_ms=float(tolerance_ms),
    )
