import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
import tyro
from rich.logging import RichHandler
from tqdm import tqdm

from utils.sync_timestamps import (
    synchronize_timestamps_3way,
    save_sync_csv,
    _pick_latest_file,
    SyncResult3Way,
)
from utils.extract_camera_data import extract_camera_data
from utils.extract_dji_data import extract_dji_rgb
from utils.extract_radar_data import process_single_frame
from utils.extract_max_data import extract_max30105_for_sequence
from utils.extract_pcd import extract_point_cloud_from_frame


def setup_logging(name):
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(name)-12s  %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler()]
    )
    return logging.getLogger(name)


def load_radar_frames(radar_h5_path: str, frame_indices: np.ndarray) -> np.ndarray:
    """
    Load selected radar frames from HDF5.

    Returns:
    - radar_frames: shape (N, doppler, tx, rx, range), dtype from file (typically int16)
    """
    with h5py.File(radar_h5_path, "r") as f:
        if "radar_data" not in f:
            raise ValueError(f"Dataset 'radar_data' not found in {radar_h5_path}")

        num_frames_attr = f.attrs.get("num_frames")
        if isinstance(num_frames_attr, (int, np.integer)):
            num_frames_attr = int(num_frames_attr)
        else:
            num_frames_attr = f["radar_data"].shape[0]

        # Load only the specified indices
        frames = []
        for idx in frame_indices:
            if idx < 0 or idx >= num_frames_attr:
                continue
            frames.append(f["radar_data"][int(idx)])

    if not frames:
        return np.empty((0, 0, 0, 0, 0), dtype=np.int16)

    return np.stack(frames, axis=0)


def _resize_rgb_to(
    frames: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
    log=None,
) -> np.ndarray:
    """
    Resize RGB frames (N, H, W, 3) uint8 to (N, target_h, target_w, 3).

    target_size: (width, height) for cv2 convention, e.g. (512, 288).
    """
    N, H, W, C = frames.shape
    assert C == 3 and frames.dtype == np.uint8
    w, h = target_size
    if (W, H) == (w, h):
        return frames
    out = np.empty((N, h, w, 3), dtype=np.uint8)
    if log:
        log.info(f"  Resizing RGB {frames.shape} -> (N, {h}, {w}, 3)...")
    for i in range(N):
        out[i] = cv2.resize(frames[i], (w, h), interpolation=interpolation)
    return out


def save_processed_optimized(
    output_dir: str,
    radar_cube: Optional[np.ndarray] = None,
    zed_rgb: Optional[np.ndarray] = None,
    zed_depth_mm: Optional[np.ndarray] = None,
    dji_rgb: Optional[np.ndarray] = None,
    radar_timestamps: Optional[np.ndarray] = None,
    zed_timestamps: Optional[np.ndarray] = None,
    dji_timestamps: Optional[np.ndarray] = None,
    sync_result=None,
    metadata: dict = None,
    rgb_codec: str = "mjpeg",
    radar_filename: str = "radar.npy",
    log=None,
):
    """
    Save processed multimodal data using optimized formats for fast loading.

    File structure:
    - radar.npy: Radar data (complex64)
    - dji_rgb.npy: DJI RGB (N, 504, 896, 3) uint8, resized from prepared 1280x720
    - zed_rgb.npy: ZED RGB (N, 504, 896, 3) uint8, resized from original
    - zed_depth.npy: ZED depth (uint16 millimeters)
    - metadata.json: Shapes, dtypes, timestamps, sync indices, and other metadata

    rgb_codec: Unused (kept for backward compatibility).

    Args:
        output_dir: Directory path to save files and metadata JSON
        radar_cube: (N, doppler, elevation, azimuth, range) complex64
        zed_rgb: (N, H, W, 3) uint8
        zed_depth_mm: (N, H, W) float32
        dji_rgb: (N, H, W, 3) uint8
        radar_timestamps: (N,) float64 - milliseconds
        zed_timestamps: (N,) float64 - milliseconds
        dji_timestamps: (N,) float64 - milliseconds
        sync_result: SyncResult or SyncResult3Way object
        metadata: Optional dict of additional metadata
        log: Logger instance
    """
    import json

    if log:
        log.info(f"Saving to optimized format in directory: {output_dir}")

    # Prepare metadata dictionary
    meta_dict = {}

    # Save radar data as NPY
    if radar_cube is not None:
        radar_path = os.path.join(output_dir, radar_filename)
        np.save(radar_path, radar_cube)

        meta_dict['radar'] = {
            'shape': list(radar_cube.shape),
            'dtype': str(radar_cube.dtype),
            'shape_info': '(N, doppler, elevation, azimuth, range)',
            'num_frames': int(radar_cube.shape[0]),
            'file': radar_filename
        }
        if radar_timestamps is not None:
            meta_dict['radar']['timestamps_ms'] = [float(t) for t in radar_timestamps]
        if log:
            log.info(f"  Radar: {radar_cube.shape} {radar_cube.dtype} -> {radar_filename}")

    # Save prepared 1280x720 DJI RGB at the historical 896x504 output size.
    if dji_rgb is not None:
        dji_rgb_resized = _resize_rgb_to(dji_rgb, (896, 504), log=log)
        dji_path = os.path.join(output_dir, "dji_rgb.npy")
        np.save(dji_path, dji_rgb_resized)

        meta_dict["dji_rgb"] = {
            "shape": list(dji_rgb_resized.shape),
            "dtype": str(dji_rgb_resized.dtype),
            "num_frames": int(dji_rgb_resized.shape[0]),
            "file": "dji_rgb.npy",
            "resolution": [896, 504],
            "original_resolution": [dji_rgb.shape[2], dji_rgb.shape[1]],
        }
        if dji_timestamps is not None:
            meta_dict["dji_rgb"]["timestamps_ms"] = [float(t) for t in dji_timestamps]
        if log:
            log.info(
                f"  DJI RGB: {dji_rgb.shape} -> resized {dji_rgb_resized.shape} -> dji_rgb.npy"
            )

    # Save ZED RGB as NPY: resize to 896x504
    if zed_rgb is not None:
        zed_rgb_resized = _resize_rgb_to(zed_rgb, (896, 504), log=log)
        zed_rgb_path = os.path.join(output_dir, "zed_rgb.npy")
        np.save(zed_rgb_path, zed_rgb_resized)

        meta_dict["zed_rgb"] = {
            "shape": list(zed_rgb_resized.shape),
            "dtype": str(zed_rgb_resized.dtype),
            "num_frames": int(zed_rgb_resized.shape[0]),
            "file": "zed_rgb.npy",
            "resolution": [896, 504],
            "original_resolution": [zed_rgb.shape[2], zed_rgb.shape[1]],
        }
        if zed_timestamps is not None:
            meta_dict["zed_rgb"]["timestamps_ms"] = [float(t) for t in zed_timestamps]
        if log:
            log.info(f"  ZED RGB: {zed_rgb.shape} {zed_rgb.dtype} -> zed_rgb.npy")

    # Save ZED depth as uint16 npy
    if zed_depth_mm is not None:
        zed_depth_path = os.path.join(output_dir, 'zed_depth.npy')
        depth_uint16 = np.clip(zed_depth_mm, 0, 65535).astype(np.uint16)
        np.save(zed_depth_path, depth_uint16)

        meta_dict['zed_depth'] = {
            'shape': list(zed_depth_mm.shape),
            'dtype': 'uint16',
            'original_dtype': str(zed_depth_mm.dtype),
            'num_frames': int(zed_depth_mm.shape[0]),
            'depth_units': 'millimeters',
            'depth_mode': 'NEURAL_PLUS',
            'file': 'zed_depth.npy'
        }
        if zed_timestamps is not None:
            meta_dict['zed_depth']['timestamps_ms'] = [float(t) for t in zed_timestamps]
        if log:
            log.info(f"  ZED Depth: {zed_depth_mm.shape} -> uint16 zed_depth.npy")

    # Add synchronization info to metadata
    if sync_result is not None:
        sync_meta = {
            'tolerance_ms': float(sync_result.tolerance_ms)
        }

        if hasattr(sync_result, 'triples'):  # 3-way sync
            radar_idx = [int(t[0]) for t in sync_result.triples]
            zed_idx = [int(t[1]) for t in sync_result.triples]
            dji_idx = [int(t[2]) for t in sync_result.triples]
            sync_meta['sync_type'] = '3-way'
            sync_meta['radar_indices'] = radar_idx
            sync_meta['zed_indices'] = zed_idx
            sync_meta['dji_indices'] = dji_idx
            sync_meta['diffs_radar_zed_ms'] = [float(d) for d in sync_result.diffs_radar_zed_ms]
            sync_meta['diffs_radar_dji_ms'] = [float(d) for d in sync_result.diffs_radar_dji_ms]
        else:  # 2-way sync
            camera_idx = [int(p[0]) for p in sync_result.pairs]
            radar_idx = [int(p[1]) for p in sync_result.pairs]
            sync_meta['sync_type'] = '2-way'
            sync_meta['camera_indices'] = camera_idx
            sync_meta['radar_indices'] = radar_idx
            sync_meta['diffs_ms'] = [float(d) for d in sync_result.diffs_ms]

        meta_dict['sync'] = sync_meta

    # Add optional metadata
    if metadata:
        meta_dict['metadata'] = metadata

    # Save metadata to JSON
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(meta_dict, f, indent=2)

    if log:
        log.info(f"Metadata saved to: metadata.json")
        log.info(f"Optimized files saved successfully in: {output_dir}")


def process_sequence(
    sequence_dir: str,
    no_radar: bool,
    no_camera: bool,
    no_dji: bool,
    rgb_codec: str = "mjpeg",
    no_doppler: bool = False,
    pcd: bool = True,
    base_output_dir: str = "processed",
    log=None,
):
    """
    Process a single sequence directory.

    Args:
        sequence_dir: Path to sequence directory containing sensor data files
        no_radar: Skip radar processing
        no_camera: Skip ZED camera (RGB + depth) processing
        no_dji: Skip DJI RGB processing
        log: Logger instance
    """
    # Fixed parameters
    tolerance_ms = 50.0
    enforce_one_to_one = True

    log.info("="*80)
    log.info(f"Processing sequence: {sequence_dir}")
    log.info("="*80)

    # Validate options
    if no_radar and no_camera and no_dji and not pcd:
        log.error("Cannot skip all modalities. At least one must be processed.")
        return False, 0, "all modalities skipped"

    # Log processing mode
    processing_modes = []
    if not no_radar:
        processing_modes.append("radar")
    if not no_camera:
        processing_modes.append("ZED RGB+depth")
    if not no_dji:
        processing_modes.append("DJI RGB")
    if pcd:
        processing_modes.append("canonical 3-D radar point clouds")
    log.info(f"Processing modes: {', '.join(processing_modes)}")
    log.info(f"Synchronization: 3-way (radar + ZED + DJI), tolerance={tolerance_ms}ms")

    # Auto-detect files
    log.info(f"Scanning directory: {sequence_dir}")
    radar_h5 = _pick_latest_file(sequence_dir, "radar_*.h5")
    zed_timestamps_h5 = _pick_latest_file(sequence_dir, "camera_timestamps_*.h5")
    zed_video_path = _pick_latest_file(sequence_dir, "zed_video_anonymized.mkv")
    zed_depth_h5 = _pick_latest_file(sequence_dir, "zed_depth.h5")
    dji_timestamps_h5 = _pick_latest_file(sequence_dir, "dji_timestamps_*.h5")
    dji_video_path = _pick_latest_file(sequence_dir, "dji_video_anonymized.mkv")

    log.info(f"Radar H5: {radar_h5}")
    log.info(f"ZED timestamps: {zed_timestamps_h5}")
    log.info(f"ZED RGB video: {zed_video_path}")
    log.info(f"ZED depth H5: {zed_depth_h5}")
    log.info(f"DJI timestamps: {dji_timestamps_h5}")
    log.info(f"DJI video: {dji_video_path}")

    # Create output directory
    sequence_basename = os.path.basename(os.path.normpath(sequence_dir))
    output_dir = os.path.join(base_output_dir, sequence_basename)
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Output directory: {output_dir}")

    with h5py.File(zed_depth_h5, "r") as depth_file:
        zed_frame_count = depth_file["depth_mm"].shape[0]
    dji_capture = cv2.VideoCapture(dji_video_path)
    try:
        dji_frame_count = int(dji_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        dji_capture.release()
    if dji_frame_count <= 0:
        raise ValueError(f"Could not read DJI frame count: {dji_video_path}")

    # Step 1: Synchronize timestamps (3-way)
    log.info("Synchronizing timestamps (3-way: radar + ZED + DJI)...")
    sync_result = None
    try:
        sync_result = synchronize_timestamps_3way(
            radar_h5=radar_h5,
            zed_timestamps_h5=zed_timestamps_h5,
            dji_timestamps_h5=dji_timestamps_h5,
            tolerance_ms=tolerance_ms,
            enforce_one_to_one=enforce_one_to_one,
            zed_frame_count=zed_frame_count,
            dji_frame_count=dji_frame_count,
        )
    except (OSError, ValueError) as e:
        err = f"Timestamp file corrupt or truncated: {e}"
        log.error(err)
        return False, 0, err
    log.info(f"Synchronized {len(sync_result.triples)} triples")

    if len(sync_result.triples) == 0:
        log.warning("No synchronized triples found. Skipping this sequence.")
        return False, 0, "no synchronized triples"

    # Special handling for Razor-1 sequence: filter out frames with ZED index > 10000
    sequence_name = os.path.basename(os.path.normpath(sequence_dir))
    if sequence_name == "Razor-1":
        zed_indices = np.array([t[1] for t in sync_result.triples], dtype=np.int64)
        original_count = len(zed_indices)
        valid_mask = zed_indices <= 10000

        filtered_count = int(valid_mask.sum())
        if filtered_count < original_count:
            log.warning(f"Razor-1 sequence: Filtered out {original_count - filtered_count} frames with ZED index > 10000")
            log.info(f"Remaining frames: {filtered_count}")

            # Create new SyncResult3Way with filtered data
            filtered_triples = [t for t, valid in zip(sync_result.triples, valid_mask) if valid]
            filtered_diffs_radar_zed = np.array(sync_result.diffs_radar_zed_ms)[valid_mask]
            filtered_diffs_radar_dji = np.array(sync_result.diffs_radar_dji_ms)[valid_mask]

            sync_result = SyncResult3Way(
                triples=filtered_triples,
                diffs_radar_zed_ms=filtered_diffs_radar_zed,
                diffs_radar_dji_ms=filtered_diffs_radar_dji,
                radar_count=sync_result.radar_count,
                zed_count=sync_result.zed_count,
                dji_count=sync_result.dji_count,
                tolerance_ms=sync_result.tolerance_ms
            )

        if len(sync_result.triples) == 0:
            log.warning("Razor-1 sequence: No frames left after filtering. Skipping this sequence.")
            return False, 0, "Razor-1: no frames after filter"

    sync_csv_path = os.path.join(output_dir, "sync_triples.csv")
    save_sync_csv(sync_csv_path, sync_result)
    log.info(f"Saved sync triples: {sync_csv_path}")
    radar_indices = np.array([t[0] for t in sync_result.triples], dtype=np.int64)
    zed_indices = np.array([t[1] for t in sync_result.triples], dtype=np.int64)
    dji_indices = np.array([t[2] for t in sync_result.triples], dtype=np.int64)

    # Load timestamps for metadata
    log.info("Loading timestamps for metadata...")
    try:
        with h5py.File(radar_h5, 'r') as f:
            radar_timestamps = f['timestamps_ms'][:][radar_indices]
        with h5py.File(zed_timestamps_h5, 'r') as f:
            zed_timestamps = f['timestamps_ms'][:][zed_indices]
        with h5py.File(dji_timestamps_h5, 'r') as f:
            dji_timestamps = f['timestamps_ms'][:][dji_indices]
    except (OSError, ValueError) as e:
        err = f"Failed to load timestamps for metadata: {e}"
        log.error(err)
        return False, 0, err

    if any(Path(sequence_dir).glob("max30105_*.h5")):
        extract_max30105_for_sequence(
            sequence_dir, radar_indices, radar_timestamps, output_dir,
            max_tolerance_ms=tolerance_ms, enforce_one_to_one=enforce_one_to_one, log=log,
        )
    else:
        log.info("No MAX30105 file; skipping optional sensor")

    # Initialize data holders
    radar_cube_arr = None
    zed_rgb = None
    zed_depth_mm = None
    dji_rgb = None

    # Step 2: Process radar data
    if not no_radar or pcd:
        log.info(f"Loading {len(radar_indices)} radar frames...")
        radar_raw_frames = load_radar_frames(radar_h5, radar_indices)
        log.info(f"Radar raw shape: {radar_raw_frames.shape}, dtype: {radar_raw_frames.dtype}")

        radar_cubes = []
        pcd_dir = os.path.join(output_dir, "pcd")
        if pcd:
            os.makedirs(pcd_dir, exist_ok=True)
        with tqdm(total=len(radar_raw_frames), desc="Radar processing", unit="frame") as pbar:
            for frame_index, frame in enumerate(radar_raw_frames):
                if not no_radar:
                    radar_cubes.append(process_single_frame(frame, no_doppler=no_doppler))
                if pcd:
                    np.save(os.path.join(pcd_dir, f"pcd_{frame_index}.npy"), extract_point_cloud_from_frame(frame))
                pbar.update(1)
        if not no_radar:
            radar_cube_arr = np.stack(radar_cubes, axis=0)
            log.info(f"Radar cube shape: {radar_cube_arr.shape}, dtype: {radar_cube_arr.dtype}")
    else:
        log.info("Skipping radar processing (--no-radar)")

    # Step 3: Extract ZED RGB and depth
    if not no_camera:
        log.info(f"Extracting {len(zed_indices)} ZED RGB+depth frames from video and H5...")
        zed_rgb, zed_depth_mm, _intrinsics = extract_camera_data(
            rgb_video_path=zed_video_path,
            depth_h5_path=zed_depth_h5,
            frame_indices=zed_indices,
        )
        log.info(f"ZED RGB shape: {zed_rgb.shape}, dtype: {zed_rgb.dtype}")
        log.info(f"ZED depth shape: {zed_depth_mm.shape}, dtype: {zed_depth_mm.dtype}")
    else:
        log.info("Skipping ZED RGB/depth extraction (--no-camera)")

    # Step 4: Extract DJI RGB
    if not no_dji:
        log.info(f"Extracting {len(dji_indices)} DJI RGB frames from video...")
        dji_rgb = extract_dji_rgb(
            video_path=dji_video_path,
            frame_indices=dji_indices,
        )
        log.info(f"DJI RGB shape: {dji_rgb.shape}, dtype: {dji_rgb.dtype}")
    else:
        log.info("Skipping DJI RGB extraction (--no-dji)")

    # Step 5: Save output using optimized format
    log.info("Saving to optimized format...")

    metadata = {
        'sequence_dir': sequence_dir,
        'tolerance_ms': tolerance_ms,
        'enforce_one_to_one': enforce_one_to_one,
        'depth_confidence': 100,
        'depth_texture_confidence': 100,
        'zed_intrinsics_fx_fy_cx_cy': _intrinsics.tolist() if not no_camera else None,
        'no_doppler': no_doppler,
    }

    save_processed_optimized(
        output_dir=output_dir,
        radar_cube=radar_cube_arr,
        zed_rgb=zed_rgb,
        zed_depth_mm=zed_depth_mm,
        dji_rgb=dji_rgb,
        radar_timestamps=radar_timestamps,
        zed_timestamps=zed_timestamps,
        dji_timestamps=dji_timestamps,
        sync_result=sync_result,
        metadata=metadata,
        rgb_codec=rgb_codec,
        radar_filename="radar_no_doppler.npy" if no_doppler else "radar.npy",
        log=log,
    )

    log.info("Sequence processing complete.")
    log.info(f"Output directory: {os.path.abspath(output_dir)}")

    # Return success status and number of synced frames
    num_synced_frames = len(sync_result.triples)
    return True, num_synced_frames, ""


def main(
    dataset: str,
    sequences: Optional[list[str]] = None,
    no_radar: bool = False,
    no_camera: bool = False,
    no_dji: bool = False,
    list_unprocessed: bool = False,
    rgb_codec: str = "mjpeg",
    no_doppler: bool = False,
    pcd: bool = True,
    output_dir: str = "processed",
):
    """
    Process multimodal sensor data from a dataset containing multiple sequences.

    Pipeline for each sequence:
    1. Auto-detect files in sequence directory:
       - radar_*.h5 (AWR1843 MIMO radar)
       - camera_timestamps_*.h5 (ZED camera timestamps)
       - zed_video_anonymized.mkv (rectified-left ZED RGB, lossless HEVC)
       - zed_depth.h5 (aligned uint16-mm depth for every ZED frame)
       - dji_timestamps_*.h5 (DJI action camera timestamps)
       - dji_video_anonymized.mkv (lossless HEVC)

    2. Synchronize timestamps (3-way: radar + ZED + DJI)
       - tolerance_ms: 50ms (fixed)
       - enforce_one_to_one: True (fixed)

    3. Extract and process synchronized frames:
       - Radar: Process AWR1843 raw data to Range-Doppler-Azimuth-Elevation cube
       - ZED: Extract RGB (LEFT camera) + depth maps (millimeters, confidence=100)
       - DJI: Extract RGB frames

    Output saved to: <output_dir>/<sequence_name>/
    - radar.npy: Full Range-Doppler-Angle radar data (complex64)
    - radar_no_doppler.npy: First-chirp Range-Angle data (--no-doppler)
    - dji_rgb.npy: DJI RGB (N, 504, 896, 3) uint8, resized from prepared 1280x720
    - zed_rgb.npy: ZED RGB (N, 504, 896, 3) uint8, resized from original
    - zed_depth.npy: ZED depth (uint16 millimeters)
    - max30105.npy: MAX30105 samples_uint32 aligned to radar timestamps
    - metadata.json: Shapes, dtypes, timestamps, sync indices
    - sync_triples.csv: Frame index mapping (radar, ZED, DJI)
    - pcd/pcd_<index>.npy: Canonical 3-D radar points (default)

    rgb_codec: Unused (kept for backward compatibility).

    Args:
        dataset: Path to dataset directory containing sequence subdirectories (required)
        sequences: List of specific sequence names to process. If not provided, processes all
                  sequences found in the dataset directory. If provided, processes only the
                  specified sequences.
        no_radar: Skip radar processing
        no_camera: Skip ZED camera (RGB + depth) processing
        no_dji: Skip DJI RGB processing
        no_doppler: Keep only chirp 0 and skip the Doppler FFT.
        pcd: Save a canonical 3-D point cloud for every synchronized radar frame (default: True).
        list_unprocessed: If True, only list dataset sequences not yet in output_dir and exit
        rgb_codec: "ffv1" or "mjpeg" for RGB video encoding (mjpeg = faster loading)
        output_dir: Base directory where processed sequences are written

    Examples:
        # Process all sequences in Data directory
        python processor.py --dataset Data

        # Process specific sequences only
        python processor.py --dataset Data --sequences seq1 seq2 seq3

        # Process without radar
        python processor.py --dataset Data --no-radar

        # Process only camera data
        python processor.py --dataset Data --no-radar --no-dji

        # Skip point clouds if only array outputs are needed
        python processor.py --dataset Data --no-pcd
    """
    log = setup_logging("Processor")

    log.info("=" * 80)
    log.info("MobiCom Multimodal Dataset Processor")
    log.info("=" * 80)
    log.info(f"Dataset path: {dataset}")
    log.info("Output format: Optimized (NPY for radar and RGB, ZED depth uint16)")
    log.info(f"Base output directory: {output_dir}")

    # Validate dataset path
    if not os.path.isdir(dataset):
        log.error(f"Dataset directory not found: {dataset}")
        return

    # Validate options
    if no_radar and no_camera and no_dji and not pcd and not list_unprocessed:
        log.error("Cannot skip all modalities. At least one must be processed.")
        return

    # Get list of sequences to process
    if sequences is None or len(sequences) == 0:
        # Process all subdirectories in dataset
        sequences = [d for d in os.listdir(dataset)
                    if os.path.isdir(os.path.join(dataset, d))]
        sequences.sort()
        log.info(f"Found {len(sequences)} sequences in dataset: {sequences}")
    else:
        log.info(f"Processing {len(sequences)} specified sequences: {sequences}")

    if list_unprocessed:
        processed_dir = output_dir
        done = set()
        if os.path.isdir(processed_dir):
            for name in os.listdir(processed_dir):
                meta = os.path.join(processed_dir, name, "metadata.json")
                if os.path.isfile(meta):
                    done.add(name)
        unprocessed = [s for s in sequences if s not in done]
        log.info(f"Sequences not yet processed ({len(unprocessed)}): {unprocessed}")
        return

    # Process each sequence
    success_count = 0
    total_synced_frames = 0
    failed_sequences = []

    for i, seq_name in enumerate(sequences, 1):
        seq_path = os.path.join(dataset, seq_name)

        if not os.path.isdir(seq_path):
            log.warning(f"Sequence directory not found, skipping: {seq_path}")
            failed_sequences.append((seq_name, "directory not found", 0))
            continue

        log.info(f"\n[{i}/{len(sequences)}] Processing sequence: {seq_name}")

        success, num_frames, reason = process_sequence(
            sequence_dir=seq_path,
            no_radar=no_radar,
            no_camera=no_camera,
            no_dji=no_dji,
            rgb_codec=rgb_codec,
            no_doppler=no_doppler,
            pcd=pcd,
            base_output_dir=output_dir,
            log=log,
        )

        if success:
            success_count += 1
            total_synced_frames += num_frames
            log.info(f"✓ Successfully processed: {seq_name} ({num_frames} synced frames)")
        else:
            failed_sequences.append((seq_name, reason or "processing failed", num_frames))
            log.warning(f"✗ Failed to process: {seq_name}")

    # Summary
    log.info("\n" + "="*80)
    log.info("Processing Summary")
    log.info("="*80)
    log.info(f"Total sequences: {len(sequences)}")
    log.info(f"Successfully processed: {success_count}")
    log.info(f"Failed: {len(failed_sequences)}")
    log.info(f"Total synced frames saved: {total_synced_frames}")

    if failed_sequences:
        log.warning("\nFailed sequences:")
        for seq_name, reason, _ in failed_sequences:
            log.warning(f"  - {seq_name}: {reason}")

    log.info(f"\nOutput directory: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    tyro.cli(main)
