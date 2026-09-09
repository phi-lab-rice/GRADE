import os
import logging
import subprocess
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
    _pick_latest_file_multi_ext,
    SyncResult3Way,
)
from utils.extract_camera_data import extract_camera_data
from utils.extract_dji_data import extract_dji_rgb
from utils.extract_radar_data import process_single_frame


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


def save_rgb_video(
    video_path: str,
    frames: np.ndarray,
    fps: float = 30.0,
    codec: str = "mjpeg",
    log=None,
):
    """
    Save RGB frames as a video file via ffmpeg.

    Args:
        video_path: Output video file path (.avi)
        frames: (N, H, W, 3) uint8 RGB frames
        fps: Frames per second
        codec: "ffv1" (lossless, slower to decode) or "mjpeg" (faster to decode in dataloaders)
        log: Logger instance
    """
    N, H, W, C = frames.shape
    assert C == 3, "Expected RGB frames with 3 channels"
    assert frames.dtype == np.uint8, "Expected uint8 dtype"
    assert codec in ("ffv1", "mjpeg"), f"codec must be 'ffv1' or 'mjpeg', got {codec!r}"

    if log:
        log.info(f"  Encoding {N} RGB frames with {codec.upper()} (via ffmpeg)...")

    base = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
    ]
    if codec == "ffv1":
        cmd = base + [
            "-c:v", "ffv1",
            "-level", "3",
            "-threads", "0",
            "-slices", "24",
            "-slicecrc", "1",
            "-coder", "1",
            "-context", "1",
            video_path,
        ]
    else:
        # mjpeg: each frame is independent (intra-only), so decoding/seek is fast in dataloaders
        cmd = base + [
            "-c:v", "mjpeg",
            "-q:v", "3",  # high quality (2-5; lower = better)
            "-threads", "0",
            video_path,
        ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for i in range(N):
        process.stdin.write(frames[i].tobytes())
    process.stdin.close()
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg encoding failed: {error_msg}")

    if log:
        log.info(f"  Saved {N} frames to video: {video_path}")


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
    log=None,
):
    """
    Save processed multimodal data using optimized formats for fast loading.

    File structure:
    - radar.npy: Radar data (complex64)
    - dji_rgb.npy: DJI RGB (N, 288, 512, 3) uint8, resized from 1920x1080
    - zed_rgb.avi: ZED RGB video (uint8; codec ffv1 or mjpeg)
    - zed_depth.npy: ZED depth (uint16 millimeters)
    - metadata.json: Shapes, dtypes, timestamps, sync indices, and other metadata

    rgb_codec: used only for zed_rgb.avi ("ffv1" or "mjpeg").
    
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
        radar_path = os.path.join(output_dir, 'radar.npy')
        np.save(radar_path, radar_cube)
        
        meta_dict['radar'] = {
            'shape': list(radar_cube.shape),
            'dtype': str(radar_cube.dtype),
            'shape_info': '(N, doppler, elevation, azimuth, range)',
            'num_frames': int(radar_cube.shape[0]),
            'file': 'radar.npy'
        }
        if radar_timestamps is not None:
            meta_dict['radar']['timestamps_ms'] = [float(t) for t in radar_timestamps]
        if log:
            log.info(f"  Radar: {radar_cube.shape} {radar_cube.dtype} -> radar.npy")
    
    # Save DJI RGB: resize 1920x1080 -> 512x288, then save as NPY for fast loading
    if dji_rgb is not None:
        dji_rgb_resized = _resize_rgb_to(dji_rgb, (512, 288), log=log)
        dji_path = os.path.join(output_dir, "dji_rgb.npy")
        np.save(dji_path, dji_rgb_resized)

        meta_dict["dji_rgb"] = {
            "shape": list(dji_rgb_resized.shape),
            "dtype": str(dji_rgb_resized.dtype),
            "num_frames": int(dji_rgb_resized.shape[0]),
            "file": "dji_rgb.npy",
            "resolution": [512, 288],
            "original_resolution": [dji_rgb.shape[2], dji_rgb.shape[1]],
        }
        if dji_timestamps is not None:
            meta_dict["dji_rgb"]["timestamps_ms"] = [float(t) for t in dji_timestamps]
        if log:
            log.info(
                f"  DJI RGB: {dji_rgb.shape} -> resized {dji_rgb_resized.shape} -> dji_rgb.npy"
            )

    # Save ZED RGB as video
    if zed_rgb is not None:
        zed_rgb_path = os.path.join(output_dir, "zed_rgb.avi")
        save_rgb_video(zed_rgb_path, zed_rgb, fps=30.0, codec=rgb_codec, log=log)

        meta_dict["zed_rgb"] = {
            "shape": list(zed_rgb.shape),
            "dtype": str(zed_rgb.dtype),
            "num_frames": int(zed_rgb.shape[0]),
            "file": "zed_rgb.avi",
            "codec": rgb_codec.upper(),
            "fps": 30.0,
        }
        if zed_timestamps is not None:
            meta_dict["zed_rgb"]["timestamps_ms"] = [float(t) for t in zed_timestamps]
        if log:
            log.info(f"  ZED RGB: {zed_rgb.shape} {zed_rgb.dtype} ({rgb_codec.upper()} codec)")
    
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
    depth_confidence = 100
    depth_texture_confidence = 100
    
    log.info("="*80)
    log.info(f"Processing sequence: {sequence_dir}")
    log.info("="*80)
    
    # Validate options
    if no_radar and no_camera and no_dji:
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
    log.info(f"Processing modes: {', '.join(processing_modes)}")
    log.info(f"Synchronization: 3-way (radar + ZED + DJI), tolerance={tolerance_ms}ms")
    
    # Auto-detect files
    log.info(f"Scanning directory: {sequence_dir}")
    radar_h5 = _pick_latest_file(sequence_dir, "radar_*.h5")
    zed_timestamps_h5 = _pick_latest_file(sequence_dir, "camera_timestamps_*.h5")
    svo_path = _pick_latest_file(sequence_dir, "zed_*.svo2")
    dji_timestamps_h5 = _pick_latest_file(sequence_dir, "dji_timestamps_*.h5")
    dji_video_path = _pick_latest_file_multi_ext(sequence_dir, ["dji_*.mkv", "dji_*.mp4"])
    
    log.info(f"Radar H5: {radar_h5}")
    log.info(f"ZED timestamps: {zed_timestamps_h5}")
    log.info(f"ZED SVO2: {svo_path}")
    log.info(f"DJI timestamps: {dji_timestamps_h5}")
    log.info(f"DJI video: {dji_video_path}")
    
    # Create output directory
    sequence_basename = os.path.basename(os.path.normpath(sequence_dir))
    output_dir = os.path.join("processed", sequence_basename)
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Output directory: {output_dir}")
    
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
            svo_path=svo_path,
            dji_video_path=dji_video_path,
        )
    except (OSError, ValueError) as e:
        err = f"Timestamp file corrupt or truncated: {e}"
        log.error(err)
        return False, 0, err
    sync_csv_path = os.path.join(output_dir, "sync_triples.csv")
    save_sync_csv(sync_csv_path, sync_result)
    log.info(f"Saved sync triples: {sync_csv_path}")
    log.info(f"Synchronized {len(sync_result.triples)} triples")
    
    if len(sync_result.triples) == 0:
        log.warning("No synchronized triples found. Skipping this sequence.")
        return False, 0, "no synchronized triples"
    
    # Extract indices
    radar_indices = np.array([t[0] for t in sync_result.triples], dtype=np.int64)
    zed_indices = np.array([t[1] for t in sync_result.triples], dtype=np.int64)
    dji_indices = np.array([t[2] for t in sync_result.triples], dtype=np.int64)
    
    # Special handling for Razor-1 sequence: filter out frames with ZED index > 10000
    sequence_name = os.path.basename(os.path.normpath(sequence_dir))
    if sequence_name == "Razor-1":
        original_count = len(zed_indices)
        valid_mask = zed_indices <= 10000
        
        radar_indices = radar_indices[valid_mask]
        zed_indices = zed_indices[valid_mask]
        dji_indices = dji_indices[valid_mask]
        
        filtered_count = len(zed_indices)
        if filtered_count < original_count:
            log.warning(f"Razor-1 sequence: Filtered out {original_count - filtered_count} frames with ZED index > 10000")
            log.info(f"Remaining frames: {filtered_count}")
            
            # Create new SyncResult3Way with filtered data
            filtered_triples = [(int(r), int(z), int(d)) for r, z, d in zip(radar_indices, zed_indices, dji_indices)]
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
        
        if len(zed_indices) == 0:
            log.warning("Razor-1 sequence: No frames left after filtering. Skipping this sequence.")
            return False, 0, "Razor-1: no frames after filter"

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
    
    # Initialize data holders
    radar_cube_arr = None
    zed_rgb = None
    zed_depth_mm = None
    dji_rgb = None
    
    # Step 2: Process radar data
    if not no_radar:
        log.info(f"Loading {len(radar_indices)} radar frames...")
        radar_raw_frames = load_radar_frames(radar_h5, radar_indices)
        log.info(f"Radar raw shape: {radar_raw_frames.shape}, dtype: {radar_raw_frames.dtype}")
        
        log.info("Processing radar frames (IIQQ → IQ → MIMO → Range-Doppler-Azimuth-Elevation FFT)...")
        radar_cubes = []
        with tqdm(total=len(radar_raw_frames), desc="Radar processing", unit="frame") as pbar:
            for frame in radar_raw_frames:
                cube = process_single_frame(frame)
                radar_cubes.append(cube)
                pbar.update(1)
        
        radar_cube_arr = np.stack(radar_cubes, axis=0)
        log.info(f"Radar cube shape: {radar_cube_arr.shape}, dtype: {radar_cube_arr.dtype}")
    else:
        log.info("Skipping radar processing (--no-radar)")
    
    # Step 3: Extract ZED RGB and depth
    if not no_camera:
        log.info(f"Extracting {len(zed_indices)} ZED RGB+depth frames from SVO...")
        zed_rgb, zed_depth_mm = extract_camera_data(
            svo_path=svo_path,
            frame_indices=zed_indices,
            depth_mode="NEURAL_PLUS",
            confidence_threshold=depth_confidence,
            texture_confidence_threshold=depth_texture_confidence,
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
        'depth_confidence': depth_confidence,
        'depth_texture_confidence': depth_texture_confidence,
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
):
    """
    Process multimodal sensor data from a dataset containing multiple sequences.
    
    Pipeline for each sequence:
    1. Auto-detect files in sequence directory:
       - radar_*.h5 (AWR1843 MIMO radar)
       - camera_timestamps_*.h5 (ZED camera timestamps)
       - zed_*.svo2 (ZED video with depth)
       - dji_timestamps_*.h5 (DJI action camera timestamps)
       - dji_*.mkv or dji_*.mp4 (DJI video)
       
    2. Synchronize timestamps (3-way: radar + ZED + DJI)
       - tolerance_ms: 50ms (fixed)
       - enforce_one_to_one: True (fixed)
       
    3. Extract and process synchronized frames:
       - Radar: Process AWR1843 raw data to Range-Doppler-Azimuth-Elevation cube
       - ZED: Extract RGB (LEFT camera) + depth maps (millimeters, confidence=100)
       - DJI: Extract RGB frames
    
    Output saved to: processed/<sequence_name>/
    - radar.npy: Radar data (complex64, fast loading)
    - dji_rgb.avi: DJI RGB video (uint8; codec: ffv1 or mjpeg)
    - zed_rgb.avi: ZED RGB video (uint8; codec: ffv1 or mjpeg)
    - zed_depth.npy: ZED depth (uint16 millimeters)
    - metadata.json: Shapes, dtypes, timestamps, sync indices
    - sync_triples.csv: Frame index mapping (radar, ZED, DJI)

    rgb_codec: "ffv1" (lossless, slower to decode) or "mjpeg" (faster in dataloaders).

    Args:
        dataset: Path to dataset directory containing sequence subdirectories (required)
        sequences: List of specific sequence names to process. If not provided, processes all 
                  sequences found in the dataset directory. If provided, processes only the 
                  specified sequences.
        no_radar: Skip radar processing
        no_camera: Skip ZED camera (RGB + depth) processing
        no_dji: Skip DJI RGB processing
        list_unprocessed: If True, only list dataset sequences not yet in processed/ and exit
        rgb_codec: "ffv1" or "mjpeg" for RGB video encoding (mjpeg = faster loading)

    Examples:
        # Process all sequences in Data directory
        python processor.py --dataset Data
        
        # Process specific sequences only
        python processor.py --dataset Data --sequences seq1 seq2 seq3
        
        # Process without radar
        python processor.py --dataset Data --no-radar
        
        # Process only camera data
        python processor.py --dataset Data --no-radar --no-dji

        # Use MJPEG for faster video loading in PyTorch dataloaders
        python processor.py --dataset Data --rgb-codec mjpeg
    """
    log = setup_logging("Processor")

    log.info("=" * 80)
    log.info("MobiCom Multimodal Dataset Processor")
    log.info("=" * 80)
    log.info(f"Dataset path: {dataset}")
    log.info(f"Output format: Optimized (NPY for radar, {rgb_codec.upper()} videos for RGB)")
    
    # Validate dataset path
    if not os.path.isdir(dataset):
        log.error(f"Dataset directory not found: {dataset}")
        return
    
    # Validate options
    if no_radar and no_camera and no_dji and not list_unprocessed:
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
        processed_dir = "processed"
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
    
    log.info(f"\nOutput directory: {os.path.abspath('processed')}")


if __name__ == "__main__":
    tyro.cli(main)
