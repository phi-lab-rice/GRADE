import os
import json
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


def setup_logging(name):
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(name)-12s  %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler()],
    )
    return logging.getLogger(name)


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
    save_depth: bool = True,
    log=None,
):
    """
    Save processed multimodal data using optimized formats for fast loading.

    File structure:
    - dji_rgb.npy: DJI RGB (N, 504, 896, 3) uint8, resized from 1920x1080
    - zed_rgb.npy: ZED RGB (N, 504, 896, 3) uint8, resized from original
    - zed_depth.npy: ZED depth (uint16 millimeters)
    - metadata.json: Shapes, dtypes, timestamps, sync indices, and other metadata

    Args:
        output_dir: Directory path to save files and metadata JSON
        radar_cube: (N, doppler, elevation, azimuth, range) complex64 - NOT SAVED
        zed_rgb: (N, H, W, 3) uint8
        zed_depth_mm: (N, H, W) float32
        dji_rgb: (N, H, W, 3) uint8
        radar_timestamps: (N,) float64 - milliseconds
        zed_timestamps: (N,) float64 - milliseconds
        dji_timestamps: (N,) float64 - milliseconds
        sync_result: SyncResult or SyncResult3Way object
        metadata: Optional dict of additional metadata
        rgb_codec: Unused (kept for backward compatibility)
        save_depth: If False, do not save zed_depth.npy (extraction still runs)
        log: Logger instance
    """

    if log:
        log.info(f"Saving to optimized format in directory: {output_dir}")

    # Prepare metadata dictionary
    meta_dict = {}

    # Skip radar data (not saved)
    if radar_cube is not None and log:
        log.info(f"  Radar: Skipping (not saved)")

    # Save DJI RGB: resize 1920x1080 -> 896x504, then save as NPY for fast loading
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
            log.info(
                f"  ZED RGB: {zed_rgb.shape} -> resized {zed_rgb_resized.shape} -> zed_rgb.npy"
            )

    # Save ZED depth as uint16 npy (optional; extraction still runs when not saving)
    if zed_depth_mm is not None and save_depth:
        zed_depth_path = os.path.join(output_dir, "zed_depth.npy")
        depth_uint16 = np.clip(zed_depth_mm, 0, 65535).astype(np.uint16)
        np.save(zed_depth_path, depth_uint16)

        meta_dict["zed_depth"] = {
            "shape": list(zed_depth_mm.shape),
            "dtype": "uint16",
            "original_dtype": str(zed_depth_mm.dtype),
            "num_frames": int(zed_depth_mm.shape[0]),
            "depth_units": "millimeters",
            "depth_mode": "NEURAL_PLUS",
            "file": "zed_depth.npy",
        }
        if zed_timestamps is not None:
            meta_dict["zed_depth"]["timestamps_ms"] = [float(t) for t in zed_timestamps]
        if log:
            log.info(f"  ZED Depth: {zed_depth_mm.shape} -> uint16 zed_depth.npy")
    elif zed_depth_mm is not None and not save_depth and log:
        log.info(f"  ZED Depth: extracted {zed_depth_mm.shape} (not saved, --no-save-depth)")

    # Add synchronization info to metadata
    if sync_result is not None:
        sync_meta = {"tolerance_ms": float(sync_result.tolerance_ms)}

        if hasattr(sync_result, "triples"):  # 3-way sync
            radar_idx = [int(t[0]) for t in sync_result.triples]
            zed_idx = [int(t[1]) for t in sync_result.triples]
            dji_idx = [int(t[2]) for t in sync_result.triples]
            sync_meta["sync_type"] = "3-way"
            sync_meta["radar_indices"] = radar_idx
            sync_meta["zed_indices"] = zed_idx
            sync_meta["dji_indices"] = dji_idx
            sync_meta["diffs_radar_zed_ms"] = [
                float(d) for d in sync_result.diffs_radar_zed_ms
            ]
            sync_meta["diffs_radar_dji_ms"] = [
                float(d) for d in sync_result.diffs_radar_dji_ms
            ]
        else:  # 2-way sync
            camera_idx = [int(p[0]) for p in sync_result.pairs]
            radar_idx = [int(p[1]) for p in sync_result.pairs]
            sync_meta["sync_type"] = "2-way"
            sync_meta["camera_indices"] = camera_idx
            sync_meta["radar_indices"] = radar_idx
            sync_meta["diffs_ms"] = [float(d) for d in sync_result.diffs_ms]

        meta_dict["sync"] = sync_meta

    # Add optional metadata
    if metadata:
        meta_dict["metadata"] = metadata

    # Save metadata to JSON
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(meta_dict, f, indent=2)

    if log:
        log.info(f"Metadata saved to: metadata.json")
        log.info(f"Optimized files saved successfully in: {output_dir}")


def process_sequence(
    sequence_dir: str,
    no_radar: bool,
    no_camera: bool,
    no_dji: bool,
    save_depth: bool = True,
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

    log.info("=" * 80)
    log.info(f"Processing sequence: {sequence_dir}")
    log.info("=" * 80)

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
    dji_video_path = _pick_latest_file_multi_ext(
        sequence_dir, ["dji_*.mkv", "dji_*.mp4"]
    )

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
            log.warning(
                f"Razor-1 sequence: Filtered out {original_count - filtered_count} frames with ZED index > 10000"
            )
            log.info(f"Remaining frames: {filtered_count}")

            # Create new SyncResult3Way with filtered data
            filtered_triples = [
                (int(r), int(z), int(d))
                for r, z, d in zip(radar_indices, zed_indices, dji_indices)
            ]
            filtered_diffs_radar_zed = np.array(sync_result.diffs_radar_zed_ms)[
                valid_mask
            ]
            filtered_diffs_radar_dji = np.array(sync_result.diffs_radar_dji_ms)[
                valid_mask
            ]

            sync_result = SyncResult3Way(
                triples=filtered_triples,
                diffs_radar_zed_ms=filtered_diffs_radar_zed,
                diffs_radar_dji_ms=filtered_diffs_radar_dji,
                radar_count=sync_result.radar_count,
                zed_count=sync_result.zed_count,
                dji_count=sync_result.dji_count,
                tolerance_ms=sync_result.tolerance_ms,
            )

        if len(zed_indices) == 0:
            log.warning(
                "Razor-1 sequence: No frames left after filtering. Skipping this sequence."
            )
            return False, 0, "Razor-1: no frames after filter"

    # Load timestamps for metadata
    log.info("Loading timestamps for metadata...")
    try:
        with h5py.File(radar_h5, "r") as f:
            radar_timestamps = f["timestamps_ms"][:][radar_indices]
        with h5py.File(zed_timestamps_h5, "r") as f:
            zed_timestamps = f["timestamps_ms"][:][zed_indices]
        with h5py.File(dji_timestamps_h5, "r") as f:
            dji_timestamps = f["timestamps_ms"][:][dji_indices]
    except (OSError, ValueError) as e:
        err = f"Failed to load timestamps for metadata: {e}"
        log.error(err)
        return False, 0, err

    # Initialize data holders
    radar_cube_arr = None
    zed_rgb = None
    zed_depth_mm = None
    dji_rgb = None

    # Step 2: Skip radar data processing (we only need sync indices)
    log.info("Skipping radar data processing (only using for sync)")

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
        "sequence_dir": sequence_dir,
        "tolerance_ms": tolerance_ms,
        "enforce_one_to_one": enforce_one_to_one,
        "depth_confidence": depth_confidence,
        "depth_texture_confidence": depth_texture_confidence,
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
        save_depth=save_depth,
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
    no_save_depth: bool = False,
    list_unprocessed: bool = False,
    rgb_codec: str = "mjpeg",
    split_json: str = "split.json",
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
       - ZED: Extract RGB (LEFT camera, resized to 896x504) + depth maps (millimeters, confidence=100)
       - DJI: Extract RGB frames (resized to 896x504)
       - Note: Radar timestamps used only for sync, radar data is NOT saved

    Output saved to: processed/<sequence_name>/
    - dji_rgb.npy: DJI RGB (N, 504, 896, 3) uint8
    - zed_rgb.npy: ZED RGB (N, 504, 896, 3) uint8
    - zed_depth.npy: ZED depth (uint16 millimeters)
    - metadata.json: Shapes, dtypes, timestamps, sync indices
    - sync_triples.csv: Frame index mapping (radar, ZED, DJI)

    rgb_codec: "ffv1" (lossless, slower to decode) or "mjpeg" (faster in dataloaders).

    Args:
        dataset: Path to dataset directory containing sequence subdirectories (required)
        sequences: List of sequence names to process. If not provided, uses sequences from
                  split_json. If "all", processes every subdirectory in the dataset folder
                  (ignores split_json). Otherwise filters the given list by split_json if it exists.
        no_radar: Skip radar processing (radar timestamps still used for sync)
        no_camera: Skip ZED camera (RGB + depth) processing
        no_dji: Skip DJI RGB processing
        no_save_depth: Do not save zed_depth.npy (ZED depth is still extracted, only saving is disabled)
        list_unprocessed: If True, only list dataset sequences not yet in processed/ and exit
        rgb_codec: Unused (kept for backward compatibility)
        split_json: Path to JSON file containing sequences to process (default: "split.json")
    Examples:
        # Process all sequences from split.json
        python processor_rgb.py --dataset s:\MobiCom-Dataset

        # Process specific sequence
        python processor_rgb.py --dataset s:\MobiCom-Dataset --sequences Dell-1

        # Process every sequence in the dataset folder (ignore split.json)
        python processor_rgb.py --dataset s:\MobiCom-Dataset --sequences all

        # Process without saving ZED depth (depth still extracted, not written to disk)
        python processor_rgb.py --dataset s:\MobiCom-Dataset --no-save-depth

        # Use custom split file
        python processor_rgb.py --dataset s:\MobiCom-Dataset --split-json custom_split.json
    """
    log = setup_logging("Processor")

    log.info("=" * 80)
    log.info("MobiCom Multimodal Dataset Processor")
    log.info("=" * 80)
    log.info(f"Dataset path: {dataset}")
    log.info(f"Output format: NPY for RGB (896x504), ZED depth uint16")
    if no_save_depth:
        log.info("ZED depth: will be extracted but NOT saved (--no-save-depth)")

    # Validate dataset path
    if not os.path.isdir(dataset):
        log.error(f"Dataset directory not found: {dataset}")
        return

    # Validate options
    if no_radar and no_camera and no_dji and not list_unprocessed:
        log.error("Cannot skip all modalities. At least one must be processed.")
        return

    # Load sequences from split.json if it exists
    split_sequences = None
    if os.path.isfile(split_json):
        with open(split_json, "r") as f:
            split_data = json.load(f)
            # Combine all sequences from all splits
            split_sequences = []
            for split_name, seq_list in split_data.items():
                split_sequences.extend(seq_list)
        log.info(f"Loaded {len(split_sequences)} sequences from {split_json}")

    # Get list of sequences to process
    if sequences is None or len(sequences) == 0:
        if split_sequences is not None:
            # Use sequences from split.json
            sequences = split_sequences
            log.info(
                f"Processing {len(sequences)} sequences from split.json: {sequences}"
            )
        else:
            # Process all subdirectories in dataset
            sequences = [
                d
                for d in os.listdir(dataset)
                if os.path.isdir(os.path.join(dataset, d))
            ]
            sequences.sort()
            log.info(f"Found {len(sequences)} sequences in dataset: {sequences}")
    elif len(sequences) == 1 and sequences[0].lower() == "all":
        # --sequences all: process every subdirectory in the dataset folder (ignore split.json)
        sequences = [
            d
            for d in os.listdir(dataset)
            if os.path.isdir(os.path.join(dataset, d))
        ]
        sequences.sort()
        log.info(f"--sequences all: processing every sequence in dataset ({len(sequences)}): {sequences}")
    else:
        # Filter specified sequences by split.json if available
        if split_sequences is not None:
            sequences = [s for s in sequences if s in split_sequences]
            log.info(
                f"Filtered to {len(sequences)} sequences from split.json: {sequences}"
            )
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
            save_depth=not no_save_depth,
            rgb_codec=rgb_codec,
            log=log,
        )

        if success:
            success_count += 1
            total_synced_frames += num_frames
            log.info(
                f"✓ Successfully processed: {seq_name} ({num_frames} synced frames)"
            )
        else:
            failed_sequences.append(
                (seq_name, reason or "processing failed", num_frames)
            )
            log.warning(f"✗ Failed to process: {seq_name}")

    # Summary
    log.info("\n" + "=" * 80)
    log.info("Processing Summary")
    log.info("=" * 80)
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
