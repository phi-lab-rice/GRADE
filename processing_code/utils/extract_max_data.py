import os
import logging
from typing import Optional, List, Dict

import h5py
import numpy as np

from utils.sync_timestamps import _pick_latest_file


def _load_max30105_h5(max_h5_path: str) -> Dict[str, np.ndarray]:
    """
    Load MAX30105 timestamps and samples from HDF5.

    Expected datasets:
      - 'timestamps_ms' (milliseconds since Unix epoch)
      - 'samples_uint32' with the same first dimension as timestamps
    """
    with h5py.File(max_h5_path, "r") as f:
        if "timestamps_ms" not in f:
            raise ValueError(
                f"'timestamps_ms' dataset not found in MAX30105 HDF5 file: {max_h5_path}"
            )

        ts = np.asarray(f["timestamps_ms"][:], dtype=np.float64)
        if ts.ndim != 1:
            raise ValueError(
                f"Expected 1D 'timestamps_ms' in {max_h5_path}, got shape {ts.shape}"
            )

        data: Dict[str, np.ndarray] = {"timestamps_ms": ts}

        # Main MAX30105 sample dataset (typically shape (N, 3) uint32)
        if "samples_uint32" not in f:
            raise ValueError(
                f"'samples_uint32' dataset not found in MAX30105 HDF5 file: {max_h5_path}"
            )
        samples = np.asarray(f["samples_uint32"][:])
        if samples.shape[0] != ts.shape[0]:
            raise ValueError(
                f"'samples_uint32' length {samples.shape[0]} does not match "
                f"timestamps length {ts.shape[0]} in {max_h5_path}"
            )
        data["samples_uint32"] = samples

    return data


def _align_max_to_radar(
    radar_indices: np.ndarray,
    radar_ts: np.ndarray,
    max_data: Dict[str, np.ndarray],
    max_tolerance_ms: float,
    enforce_one_to_one: bool,
) -> Dict[str, np.ndarray]:
    """
    For each synchronized radar timestamp, find the nearest MAX30105 sample.

    Returns a dict with per-channel arrays for each present channel (e.g. 'samples_uint32'),
    aligned to the subset of synchronized radar timestamps that could be matched within
    `max_tolerance_ms`.
    """
    max_ts = max_data["timestamps_ms"].astype(np.float64, copy=False)

    matched_channels: Dict[str, List[np.ndarray]] = {
        ch: [] for ch in max_data.keys() if ch != "timestamps_ms"
    }

    used_max = set() if enforce_one_to_one else None

    for radar_idx, r_t in zip(radar_indices, radar_ts):
        insert_idx = int(np.searchsorted(max_ts, r_t))

        # Nearest neighbor search (optionally one-to-one)
        cand0 = insert_idx - 1 if insert_idx > 0 else None
        cand1 = insert_idx if insert_idx < len(max_ts) else None

        best_idx: Optional[int] = None
        best_abs = float("inf")
        for cand in (cand0, cand1):
            if cand is None:
                continue
            if enforce_one_to_one and cand in used_max:
                continue
            abs_diff = abs(float(max_ts[cand] - r_t))
            if abs_diff < best_abs:
                best_abs = abs_diff
                best_idx = int(cand)

        if best_idx is None:
            continue

        diff = float(r_t - max_ts[best_idx])
        if abs(diff) > max_tolerance_ms:
            continue

        for ch, lst in matched_channels.items():
            lst.append(max_data[ch][best_idx])

        if enforce_one_to_one:
            used_max.add(best_idx)

    out: Dict[str, np.ndarray] = {}
    for ch, lst in matched_channels.items():
        out[ch] = np.asarray(lst)

    return out


def extract_max30105_for_sequence(
    sequence_dir: str,
    radar_indices: np.ndarray,
    radar_timestamps: np.ndarray,
    output_dir: str,
    max_tolerance_ms: float = 50.0,
    enforce_one_to_one: bool = True,
    log: Optional[logging.Logger] = None,
) -> int:
    """
    Extract MAX30105 samples aligned to synchronized radar timestamps for a sequence
    and save them as max30105.npy in the given output directory.
    """
    seq_name = os.path.basename(os.path.normpath(sequence_dir))

    max_h5 = _pick_latest_file(sequence_dir, "max30105_*.h5")
    if log:
        log.info(f"MAX30105 H5: {max_h5}")

    max_data = _load_max30105_h5(max_h5)

    aligned = _align_max_to_radar(
        radar_indices=radar_indices,
        radar_ts=radar_timestamps,
        max_data=max_data,
        max_tolerance_ms=max_tolerance_ms,
        enforce_one_to_one=enforce_one_to_one,
    )

    any_channel = next((arr for arr in aligned.values()), None)
    num_matched = int(any_channel.shape[0]) if any_channel is not None else 0
    if log:
        log.info(
            f"Matched {num_matched} MAX30105 samples to synchronized radar frames."
        )

    samples = aligned.get("samples_uint32")
    if samples is None:
        if log:
            log.warning(
                "No 'samples_uint32' channel found in aligned MAX30105 data; nothing saved."
            )
        return num_matched

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "max30105.npy")
    np.save(out_path, samples)

    if log:
        log.info(f"Saved extracted MAX30105 readings to: {out_path}")

    return num_matched
