import os
import logging
from pathlib import Path
from typing import Optional, Tuple

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
from utils.extract_radar_data import iiqq_to_iq, mimo, fft_w_shift


# ---------------------------------------------------------------------------
# Radar point-cloud constants
# ---------------------------------------------------------------------------
RANGE_FFT_SIZE = 256
AZIMUTH_FFT_SIZE = 32
RANGE_RES_M = 0.04375  # metres per range bin
AZ_FOV_DEG = 90.0  # half-FOV in degrees (determined by arcsin physical mapping)

# CFAR parameters (1D CA-CFAR along range)
CFAR_THRESHOLD_DB = 8.0
CFAR_TRAIN_RANGE = 5  # training cells per side
CFAR_GUARD_RANGE = 5  # guard cells per side


def setup_logging(name):
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(name)-12s  %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler()],
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


# ---------------------------------------------------------------------------
# 2-D CA-CFAR
# ---------------------------------------------------------------------------
def cfar_1d(
    power_db: np.ndarray,
    train_range: int = CFAR_TRAIN_RANGE,
    guard_range: int = CFAR_GUARD_RANGE,
    threshold_db: float = CFAR_THRESHOLD_DB,
) -> np.ndarray:
    """
    1-D Cell-Averaging CFAR along the range axis (last axis).

    Args:
        power_db: (n_azimuth, n_range) power map in dB
        train_range: training cells per side
        guard_range: guard cells per side
        threshold_db: detection threshold above local noise estimate (dB)

    Returns:
        detections: boolean mask (n_azimuth, n_range)
    """
    n_az, n_range = power_db.shape
    detections = np.zeros_like(power_db, dtype=bool)

    hw = train_range + guard_range
    n_training = 2 * train_range

    for ir in range(hw, n_range - hw):
        # Noise estimate: average of training cells on left and right
        left_train = power_db[:, ir - hw : ir - guard_range]
        right_train = power_db[:, ir + guard_range + 1 : ir + hw + 1]
        noise_sum = np.sum(left_train, axis=1) + np.sum(right_train, axis=1)
        noise_level = noise_sum / n_training

        detections[:, ir] = power_db[:, ir] > (noise_level + threshold_db)

    return detections


# ---------------------------------------------------------------------------
# Per-frame point-cloud extraction
# ---------------------------------------------------------------------------
def extract_point_cloud_from_frame(iiqq_frame: np.ndarray) -> np.ndarray:
    """
    Extract a point cloud from a single raw IIQQ radar frame via
    range-azimuth CFAR on the bottom elevation row.

    Coordinate system:
        X → right  (azimuth)
        Y → down   (elevation, always 0)
        Z → forward (range / depth)

    Args:
        iiqq_frame: (doppler, tx, rx, range) raw IIQQ data

    Returns:
        pcd: (N_pts, 3) float32  [x, y, z]
             Empty (0, 3) if no detections.
    """
    # 1. IIQQ → IQ
    iq = iiqq_to_iq(iiqq_frame)  # (doppler, tx, rx, range/2)

    # 2. MIMO — full virtual array (doppler, 2, 8, range)
    mimo_data = mimo(iq)  # (64, 2, 8, range)

    # 3. Keep only bottom elevation row (index=1) → (64, 1, 8, range)
    mimo_bottom = mimo_data[:, 1:2, :, :]

    # 4. Range FFT — 256 bins, no fftshift
    # Apply Hanning window across the range dimension
    rng_win = np.hanning(mimo_bottom.shape[-1])
    range_fft = np.fft.fft(mimo_bottom * rng_win, n=RANGE_FFT_SIZE, axis=-1)
    # shape: (64, 1, 8, 256)

    # 5. Azimuth FFT — 256 bins, with fftshift (axis=2 is azimuth/rx)
    # Apply Hanning window across the azimuth dimension (8 antennas)
    az_win = np.hanning(range_fft.shape[2])
    # Reshape az_win for broadcasting: (1, 1, 8, 1)
    az_win = az_win.reshape(1, 1, -1, 1)
    az_fft = np.fft.fftshift(
        np.fft.fft(range_fft * az_win, n=AZIMUTH_FFT_SIZE, axis=2), axes=2
    )
    # shape: (64, 1, 256, 256)  →  [doppler, 1, azimuth, range]

    # Squeeze elevation dim → (64, 64, 256)  [doppler, azimuth, range]
    ra_cube = az_fft[:, 0, :, :]

    # 6. Mapping FFT index to physical angle (azimuth)
    # k: shifted indices ranging from -N/2 to N/2-1
    k = np.arange(-AZIMUTH_FFT_SIZE // 2, AZIMUTH_FFT_SIZE // 2)
    # Physical angle theta = arcsin( (lambda * omega) / (2 * pi * d) )
    # Since index 0 is the rightmost antenna (+x), a rightward angle (+theta)
    # gives a negative spatial frequency.
    # If d = lambda / 2, then theta = arcsin( -2k / N_FFT )
    az_angles_rad = np.arcsin(np.clip(-2.0 * k / AZIMUTH_FFT_SIZE, -1.0, 1.0))

    # 7. Use only the first chirp (index 0) from the 64 doppler repetitions
    ra_slice = ra_cube[0]  # (64 azimuth, 256 range)
    power_db = 10.0 * np.log10(np.abs(ra_slice) + 1e-12)

    # 8. 1-D CA-CFAR (along range)
    det_mask = cfar_1d(power_db)
    az_det, rng_det = np.nonzero(det_mask)

    if len(az_det) == 0:
        return np.empty((0, 3), dtype=np.float32)

    # 9. Convert to cartesian
    range_m = rng_det.astype(np.float32) * RANGE_RES_M
    az_rad = az_angles_rad[az_det].astype(np.float32)

    x = range_m * np.sin(az_rad)  # right
    z = range_m * np.cos(az_rad)  # forward

    pcd = np.stack([x, np.zeros_like(x), z], axis=-1)  # y = 0
    return pcd


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_processed_optimized(
    output_dir: str,
    num_pcd_frames: int = 0,
    radar_timestamps: Optional[np.ndarray] = None,
    zed_timestamps: Optional[np.ndarray] = None,
    dji_timestamps: Optional[np.ndarray] = None,
    sync_result=None,
    metadata: dict = None,
    log=None,
):
    """
    Save metadata for the processed point-cloud sequence.

    Point clouds themselves are saved individually as pcd/pcd_{i}.npy.
    This function writes metadata.json with sync info and timestamps.

    Args:
        output_dir: Directory path to save metadata JSON
        num_pcd_frames: Number of point-cloud frames saved
        radar_timestamps: (N,) float64 - milliseconds
        zed_timestamps: (N,) float64 - milliseconds
        dji_timestamps: (N,) float64 - milliseconds
        sync_result: SyncResult3Way object
        metadata: Optional dict of additional metadata
        log: Logger instance
    """
    import json

    if log:
        log.info(f"Saving metadata in directory: {output_dir}")

    meta_dict = {}

    # Point-cloud info
    meta_dict["pcd"] = {
        "num_frames": num_pcd_frames,
        "directory": "pcd",
        "file_pattern": "pcd_{frame_index}.npy",
        "shape_info": "(N_points, 3)  columns: [x_right, y_down=0, z_forward]",
        "coordinate_system": {
            "x": "right (azimuth)",
            "y": "down (elevation, always 0)",
            "z": "forward (range/depth)",
        },
        "range_resolution_m": RANGE_RES_M,
        "azimuth_fov_deg": [-AZ_FOV_DEG, AZ_FOV_DEG],
        "range_fft_size": RANGE_FFT_SIZE,
        "azimuth_fft_size": AZIMUTH_FFT_SIZE,
        "cfar": {
            "threshold_db": CFAR_THRESHOLD_DB,
            "train_range": CFAR_TRAIN_RANGE,
            "guard_range": CFAR_GUARD_RANGE,
        },
    }
    if radar_timestamps is not None:
        meta_dict["pcd"]["timestamps_ms"] = [float(t) for t in radar_timestamps]

    # Sync info
    if sync_result is not None:
        sync_meta = {"tolerance_ms": float(sync_result.tolerance_ms)}

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

        meta_dict["sync"] = sync_meta

    # Timestamp arrays
    if zed_timestamps is not None:
        meta_dict["zed_timestamps_ms"] = [float(t) for t in zed_timestamps]
    if dji_timestamps is not None:
        meta_dict["dji_timestamps_ms"] = [float(t) for t in dji_timestamps]

    # Additional metadata
    if metadata:
        meta_dict["metadata"] = metadata

    # metadata_path = os.path.join(output_dir, "metadata.json")
    # with open(metadata_path, "w") as f:
    #     json.dump(meta_dict, f, indent=2)

    if log:
        log.info(f"Metadata saved to: metadata.json")
        log.info(f"Files saved successfully in: {output_dir}")


# ---------------------------------------------------------------------------
# Sequence processing
# ---------------------------------------------------------------------------
def process_sequence(
    sequence_dir: str,
    no_radar: bool,
    log=None,
):
    """
    Process a single sequence directory.

    Args:
        sequence_dir: Path to sequence directory containing sensor data files
        no_radar: Skip radar processing
        log: Logger instance
    """
    # Fixed parameters
    tolerance_ms = 50.0
    enforce_one_to_one = True

    log.info("=" * 80)
    log.info(f"Processing sequence: {sequence_dir}")
    log.info("=" * 80)

    # Log processing mode
    log.info("Processing modes: radar (point cloud extraction)")
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
    log.info(f"DJI timestamps: {dji_timestamps_h5}")

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

    # Step 2: Extract point clouds from radar data
    num_pcd_frames = 0
    if not no_radar:
        log.info(f"Loading {len(radar_indices)} radar frames...")
        radar_raw_frames = load_radar_frames(radar_h5, radar_indices)
        log.info(
            f"Radar raw shape: {radar_raw_frames.shape}, dtype: {radar_raw_frames.dtype}"
        )

        # Create pcd subdirectory
        pcd_dir = os.path.join(output_dir, "pcd")
        os.makedirs(pcd_dir, exist_ok=True)

        log.info(
            "Extracting point clouds (IIQQ → IQ → MIMO bottom row → Range-Azimuth FFT → CFAR)..."
        )
        total_points = 0
        with tqdm(
            total=len(radar_raw_frames), desc="PCD extraction", unit="frame"
        ) as pbar:
            for i, frame in enumerate(radar_raw_frames):
                pcd = extract_point_cloud_from_frame(frame)
                pcd_path = os.path.join(pcd_dir, f"pcd_{i}.npy")
                np.save(pcd_path, pcd)
                total_points += pcd.shape[0]
                pbar.update(1)

        num_pcd_frames = len(radar_raw_frames)
        log.info(
            f"Saved {num_pcd_frames} point clouds to {pcd_dir} "
            f"(avg {total_points / max(num_pcd_frames, 1):.0f} pts/frame)"
        )
    else:
        log.info("Skipping radar processing (--no-radar)")

    # Step 3: Save metadata
    log.info("Saving metadata...")

    metadata = {
        "sequence_dir": sequence_dir,
        "tolerance_ms": tolerance_ms,
        "enforce_one_to_one": enforce_one_to_one,
    }

    save_processed_optimized(
        output_dir=output_dir,
        num_pcd_frames=num_pcd_frames,
        radar_timestamps=radar_timestamps,
        zed_timestamps=zed_timestamps,
        dji_timestamps=dji_timestamps,
        sync_result=sync_result,
        metadata=metadata,
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
    list_unprocessed: bool = False,
    debug: bool = False,
):
    """
    Extract radar point clouds from range-azimuth plane for each sequence.

    Pipeline for each sequence:
    1. Auto-detect files in sequence directory:
       - radar_*.h5 (AWR1843 MIMO radar)
       - camera_timestamps_*.h5 (ZED camera timestamps)
       - dji_timestamps_*.h5 (DJI action camera timestamps)

    2. Synchronize timestamps (3-way: radar + ZED + DJI)
       - tolerance_ms: 50ms (fixed)
       - enforce_one_to_one: True (fixed)

    3. For each synced radar frame:
       - IIQQ → IQ → MIMO (bottom elevation row only)
       - Range FFT (256 bins) → Azimuth FFT (64 bins)
       - 2D CFAR detection per chirp (64 chirps)
       - Convert detections to cartesian point cloud (x=right, y=0, z=forward)

    Output saved to: processed/<sequence_name>/
    - pcd/pcd_{i}.npy: Per-frame point clouds (N_pts, 3) float32
    - metadata.json: Processing params, timestamps, sync indices
    - sync_triples.csv: Frame index mapping (radar, ZED, DJI)

    Args:
        dataset: Path to dataset directory containing sequence subdirectories (required)
        sequences: List of specific sequence names to process. If not provided, processes all
                  sequences found in the dataset directory.
        no_radar: Skip radar processing (only save sync info and timestamps)
        list_unprocessed: If True, only list dataset sequences not yet in processed/ and exit
        debug: If True, pick one random synced frame from Razor-1 and print PCD shape + first 5 points

    Examples:
        # Process all sequences in Data directory
        python processor_pcd.py --dataset Data

        # Process specific sequences only
        python processor_pcd.py --dataset Data --sequences seq1 seq2 seq3

        # Only sync timestamps, skip radar processing
        python processor_pcd.py --dataset Data --no-radar

        # Debug: test one random frame from Razor-1
        python processor_pcd.py --dataset Data --debug
    """
    log = setup_logging("Processor")

    log.info("=" * 80)
    log.info("MobiCom Radar Point-Cloud Extractor")
    log.info("=" * 80)
    log.info(f"Dataset path: {dataset}")
    log.info("Output format: per-frame point clouds (pcd/pcd_i.npy)")

    # Validate dataset path
    if not os.path.isdir(dataset):
        log.error(f"Dataset directory not found: {dataset}")
        return

    # --debug: pick one random frame from Razor-1 and test the pipeline
    if debug:
        log.info("DEBUG MODE: Testing one random frame from Razor-1")
        razor_dir = os.path.join(dataset, "Razor-1")
        if not os.path.isdir(razor_dir):
            log.error(f"Razor-1 directory not found: {razor_dir}")
            return

        radar_h5 = _pick_latest_file(razor_dir, "radar_*.h5")
        log.info(f"Radar H5: {radar_h5}")

        with h5py.File(radar_h5, "r") as f:
            num_frames = f.attrs.get("num_frames")
            if not isinstance(num_frames, (int, np.integer)):
                num_frames = f["radar_data"].shape[0]
            num_frames = int(num_frames)

            rand_idx = np.random.randint(0, num_frames)
            log.info(f"Random frame index: {rand_idx} / {num_frames}")
            raw_frame = f["radar_data"][rand_idx]

        log.info(f"Raw frame shape: {raw_frame.shape}, dtype: {raw_frame.dtype}")
        pcd = extract_point_cloud_from_frame(raw_frame)
        log.info(f"Point cloud shape: {pcd.shape}")
        log.info(f"First 5 points:\n{pcd[:5]}")
        return

    # Get list of sequences to process
    if sequences is None or len(sequences) == 0:
        # Process all subdirectories in dataset
        sequences = [
            d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))
        ]
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
