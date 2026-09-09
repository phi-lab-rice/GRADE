"""
3-way sync (radar + ZED + DJI) using a two-step strategy:

  Step 1: Align ZED to radar (same as 2-way sync).
          → Get (radar_idx, zed_idx) pairs and aligned ZED timestamps.

  Step 2: For each aligned ZED frame, find the closest DJI frame to the ZED
          timestamp (not to radar). Emit (radar_idx, zed_idx, dji_idx) when
          within tolerance.

This can produce different (often better) triples than matching both ZED and
DJI directly to radar, since DJI is matched to ZED time.

Example (Data/Dell-1):
  python -m utils.sync_3way_zed_then_dji --directory Data/Dell-1
  python -m utils.sync_3way_zed_then_dji --directory Data/Dell-1 --out-csv processed/Dell-1/sync_triples.csv
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional, Tuple

# When run as script from utils/, project root may not be on path; add it so "utils" resolves.
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import numpy as np
import tyro
import matplotlib.pyplot as plt

from utils.sync_timestamps import (
    SyncResult3Way,
    _nearest_unused_camera_index,
    load_camera_timestamps_h5,
    load_dji_timestamps_h5,
    load_radar_timestamps_h5,
    save_sync_csv,
    synchronize_timestamps,
    _pick_latest_file,
    _pick_latest_file_multi_ext,
)
from utils.extract_camera_data import extract_camera_data
from utils.extract_dji_data import extract_dji_rgb

log = logging.getLogger(__name__)


def synchronize_timestamps_3way_zed_then_dji(
    radar_h5: str,
    zed_timestamps_h5: str,
    dji_timestamps_h5: str,
    tolerance_radar_zed_ms: float = 50.0,
    tolerance_zed_dji_ms: float = 50.0,
    enforce_one_to_one_radar_zed: bool = True,
    enforce_one_to_one_dji: bool = True,
) -> SyncResult3Way:
    """
    Three-way sync: first align ZED to radar, then find closest DJI to each ZED.

    Step 1: Radar–ZED sync (same as 2-way). Produces (radar_idx, zed_idx) pairs
            with |radar_ts - zed_ts| <= tolerance_radar_zed_ms.

    Step 2: For each (radar_idx, zed_idx), use zed_ts[zed_idx] as reference.
            Find nearest DJI frame to that ZED time. Keep triple only if
            |zed_ts - dji_ts| <= tolerance_zed_dji_ms.

    Returns SyncResult3Way compatible with save_sync_csv (same CSV columns).
    """
    radar_ts = load_radar_timestamps_h5(radar_h5)
    zed_ts = load_camera_timestamps_h5(zed_timestamps_h5)
    dji_ts = load_dji_timestamps_h5(dji_timestamps_h5)

    radar_ts = radar_ts.astype(np.float64, copy=False)
    zed_ts = zed_ts.astype(np.float64, copy=False)
    dji_ts = dji_ts.astype(np.float64, copy=False)

    log.info(f"ZED timestamps: {len(zed_ts)} (last removed for SVO safety)")
    log.info(
        f"Ranges: radar {len(radar_ts)} [{radar_ts[0]:.2f} - {radar_ts[-1]:.2f} ms] | "
        f"ZED {len(zed_ts)} | DJI {len(dji_ts)}"
    )

    # Step 1: Radar–ZED alignment (2-way sync)
    sync_radar_zed = synchronize_timestamps(
        camera_timestamps_h5=zed_timestamps_h5,
        radar_h5=radar_h5,
        tolerance_ms=tolerance_radar_zed_ms,
        enforce_one_to_one=enforce_one_to_one_radar_zed,
        strategy="nearest",
    )
    # pairs are (camera_frame_idx, radar_frame_idx) = (zed_idx, radar_idx)
    pairs_zed_radar: List[Tuple[int, int]] = sync_radar_zed.pairs
    log.info(f"Step 1 (radar–ZED): {len(pairs_zed_radar)} pairs")

    # Step 2: For each (zed_idx, radar_idx), find closest DJI to zed_ts[zed_idx]
    triples: List[Tuple[int, int, int]] = []
    diffs_radar_zed: List[float] = []
    diffs_radar_dji: List[float] = []
    used_dji: set[int] = set()

    for zed_idx, radar_idx in pairs_zed_radar:
        z_t = float(zed_ts[zed_idx])
        r_t = float(radar_ts[radar_idx])

        insert_idx = int(np.searchsorted(dji_ts, z_t))
        if enforce_one_to_one_dji:
            dji_idx = _nearest_unused_camera_index(dji_ts, z_t, used_dji, insert_idx)
        else:
            cand0 = insert_idx - 1 if insert_idx > 0 else None
            cand1 = insert_idx if insert_idx < len(dji_ts) else None
            dji_idx = None
            best_abs = float("inf")
            for cand in (cand0, cand1):
                if cand is None:
                    continue
                abs_diff = abs(float(dji_ts[cand] - z_t))
                if abs_diff < best_abs:
                    best_abs = abs_diff
                    dji_idx = int(cand)
        if dji_idx is None:
            continue

        zed_dji_diff = abs(float(dji_ts[dji_idx] - z_t))
        if zed_dji_diff > tolerance_zed_dji_ms:
            continue

        triples.append((int(radar_idx), int(zed_idx), int(dji_idx)))
        diffs_radar_zed.append(float(r_t - z_t))
        diffs_radar_dji.append(float(r_t - dji_ts[dji_idx]))
        if enforce_one_to_one_dji:
            used_dji.add(int(dji_idx))

    diffs_radar_zed_ms = np.array(diffs_radar_zed, dtype=np.float64)
    diffs_radar_dji_ms = np.array(diffs_radar_dji, dtype=np.float64)

    log.info(
        f"Step 2 (ZED–DJI): {len(triples)} triples | "
        f"tol_radar_zed={tolerance_radar_zed_ms}ms tol_zed_dji={tolerance_zed_dji_ms}ms"
    )
    return SyncResult3Way(
        triples=triples,
        diffs_radar_zed_ms=diffs_radar_zed_ms,
        diffs_radar_dji_ms=diffs_radar_dji_ms,
        radar_count=int(len(radar_ts)),
        zed_count=int(len(zed_ts)),
        dji_count=int(len(dji_ts)),
        tolerance_ms=float(tolerance_radar_zed_ms),
    )


def _visualize_synced_rgb_from_raw(
    result: SyncResult3Way,
    svo_path: str,
    dji_video_path: str,
    num_pairs: int = 4,
) -> None:
    """
    Extract ZED and DJI RGB for the first num_pairs synced triples from raw SVO and
    DJI video, then show side-by-side (ZED | DJI) in a grid.
    """
    if len(result.triples) == 0:
        log.warning("No triples to visualize.")
        return
    K = min(num_pairs, len(result.triples))
    zed_indices = [result.triples[i][1] for i in range(K)]
    dji_indices = [result.triples[i][2] for i in range(K)]
    unique_zed = sorted(set(zed_indices))
    unique_dji = sorted(set(dji_indices))

    log.info(f"Extracting {len(unique_zed)} ZED and {len(unique_dji)} DJI frames for {K} pairs...")
    zed_rgb_all, _ = extract_camera_data(svo_path, unique_zed)
    dji_rgb_all = extract_dji_rgb(dji_video_path, unique_dji)

    zed_idx_to_pos = {idx: p for p, idx in enumerate(unique_zed)}
    dji_idx_to_pos = {idx: p for p, idx in enumerate(unique_dji)}

    nrows = (K + 1) // 2
    ncols = min(2, K)
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(5 * ncols * 2, 5 * nrows))
    axes = np.atleast_2d(axes)
    if axes.shape != (nrows, ncols * 2):
        axes = np.array(axes).reshape(nrows, ncols * 2)
    used = set()
    for i in range(K):
        row, col2 = i // 2, (i % 2) * 2
        zed_pos = zed_idx_to_pos[result.triples[i][1]]
        dji_pos = dji_idx_to_pos[result.triples[i][2]]
        zed_frame = np.asarray(zed_rgb_all[zed_pos])
        dji_frame = np.asarray(dji_rgb_all[dji_pos])
        axes[row, col2].imshow(zed_frame)
        axes[row, col2].set_title(f"ZED (triple {i}, zed_idx={result.triples[i][1]})")
        axes[row, col2].axis("off")
        axes[row, col2 + 1].imshow(dji_frame)
        axes[row, col2 + 1].set_title(f"DJI (triple {i}, dji_idx={result.triples[i][2]})")
        axes[row, col2 + 1].axis("off")
        used.add((row, col2))
        used.add((row, col2 + 1))
    for row in range(nrows):
        for col in range(ncols * 2):
            if (row, col) not in used:
                axes[row, col].set_visible(False)
    plt.suptitle("Synced ZED vs DJI RGB (from raw data)")
    plt.tight_layout()
    plt.show()


def cli(
    directory: Optional[str] = "Data/Dell-1",
    radar_h5: Optional[str] = None,
    zed_timestamps_h5: Optional[str] = None,
    dji_timestamps_h5: Optional[str] = None,
    out_csv: Optional[str] = None,
    tolerance_radar_zed_ms: float = 50.0,
    tolerance_zed_dji_ms: float = 50.0,
    enforce_one_to_one: bool = True,
    visualize: bool = False,
    visualize_num: int = 4,
) -> None:
    """
    Run 3-way sync (ZED-then-DJI strategy). Uses Data/Dell-1 as example when no args given.

    If directory is given, auto-pick latest radar_*.h5, camera_timestamps_*.h5,
    dji_timestamps_*.h5 inside it. Otherwise pass explicit radar_h5, zed_timestamps_h5, dji_timestamps_h5.

    If visualize=True and directory is set, extract ZED/DJI RGB from raw SVO and DJI video
    for the first visualize_num synced pairs and show side-by-side (requires directory with
    zed_*.svo2 and dji_*.mkv or dji_*.mp4).
    """
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    if directory is not None and (radar_h5 is not None or zed_timestamps_h5 is not None or dji_timestamps_h5 is not None):
        raise ValueError("Use either directory= or explicit radar/zed/dji paths, not both.")

    if directory is not None:
        # Resolve relative paths against project root so it works from any cwd (e.g. utils/)
        if not os.path.isabs(directory):
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            directory = os.path.normpath(os.path.join(_root, directory))
        radar_h5 = _pick_latest_file(directory, "radar_*.h5")
        zed_timestamps_h5 = _pick_latest_file(directory, "camera_timestamps_*.h5")
        dji_timestamps_h5 = _pick_latest_file(directory, "dji_timestamps_*.h5")
        log.info(f"Directory: {directory}")
        log.info(f"Radar: {radar_h5}")
        log.info(f"ZED timestamps: {zed_timestamps_h5}")
        log.info(f"DJI timestamps: {dji_timestamps_h5}")
        if out_csv is None:
            basename = os.path.basename(os.path.normpath(directory))
            out_csv = os.path.join("processed", basename, "sync_triples.csv")
    if out_csv is None:
        out_csv = "processed/Dell-1/sync_triples.csv"

    if radar_h5 is None or zed_timestamps_h5 is None or dji_timestamps_h5 is None:
        raise ValueError("Provide directory= or all of radar_h5, zed_timestamps_h5, dji_timestamps_h5.")

    result = synchronize_timestamps_3way_zed_then_dji(
        radar_h5=radar_h5,
        zed_timestamps_h5=zed_timestamps_h5,
        dji_timestamps_h5=dji_timestamps_h5,
        tolerance_radar_zed_ms=tolerance_radar_zed_ms,
        tolerance_zed_dji_ms=tolerance_zed_dji_ms,
        enforce_one_to_one_radar_zed=enforce_one_to_one,
        enforce_one_to_one_dji=enforce_one_to_one,
    )
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    save_sync_csv(out_csv, result)
    log.info(f"Wrote {out_csv}")

    if visualize:
        if directory is None:
            log.warning("--visualize requires --directory (raw data folder with zed_*.svo2 and dji_*.mkv/mp4). Skipping.")
        else:
            svo_path = _pick_latest_file(directory, "zed_*.svo2")
            dji_video_path = _pick_latest_file_multi_ext(directory, ["dji_*.mkv", "dji_*.mp4"])
            log.info(f"Visualize from raw: SVO={svo_path}, DJI={dji_video_path}")
            _visualize_synced_rgb_from_raw(result, svo_path, dji_video_path, num_pairs=visualize_num)


if __name__ == "__main__":
    tyro.cli(cli)
