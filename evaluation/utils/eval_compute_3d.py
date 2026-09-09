"""
3D evaluation computation: Chamfer Distance and Modified Hausdorff Distance.

For a selected model, computes per-frame CD and MHD by lifting depth maps to
point clouds, then writes CSVs to:
  {output_dir}/3d_csv_{model_name}/{seq_name}.csv

Metrics per frame (predicted depth vs ZED ground truth):
  - CD  (Chamfer Distance)        : mean(d(p→q)²) + mean(d(q→p)²)
  - MHD (Modified Hausdorff Dist) : max(mean(d(p→q)), mean(d(q→p)))

Also writes MAX30105 sensor readings (R, IR, G) per frame.

Ground truth data : <artifact_root>/evaluation_dataset/Smoke-Eval/{seq_name}/
Predictions       : <artifact_root>/evaluation/outputs/inference/{model_dir}/...
Models            : da3, cafnet, cafnet_no_smoke, radarcam-depth,
                    grt, grt_refine_freeze, grt_no_doppler, grt_image,
                    ours_full, ours_diffusion, ours_radar,
                    ours_radar_no_doppler, ours_radar_no_grad,
                    ours_full_no_3d, grt_refine_retrain

Every model is scored on a common 288x512 grid by default (--target-resolution).
Ground truth and predictions are both resampled onto that grid before they are
back-projected, so models whose native output is coarser than 288x512 -- GRT
and its variants, ours_radar and its variants -- are lifted to point clouds at
the same density as the rest.  Passing --target-resolution native restores the
older behaviour, where each model is scored on its own prediction grid; those
two protocols do NOT produce the same CD/MHD for coarse-output models.

Results default to <script_dir>/simple_eval_results_3d/ and can be redirected
with --output_dir.

Usage:
  python eval_compute_3d.py                       # every model
  python eval_compute_3d.py --model ours_full
  python eval_compute_3d.py --model da3 cafnet grt_image grt_no_doppler
  python eval_compute_3d.py --workers 4           # multiprocessing pool
  python eval_compute_3d.py --target-resolution native   # legacy per-model grid
  python eval_compute_3d.py --table               # existing CSVs → summary table
"""

import csv
import os
from multiprocessing import Pool
from pathlib import Path

# Total CPU budget for this script, in cores. The pool size is capped at
# MAX_WORKERS and each worker's native thread pool is capped at
# MAX_WORKERS // workers, so workers * threads never exceeds MAX_WORKERS.
#
# OpenMP-style thread counts are read by NumPy / OpenCV / Open3D when they are
# first imported, so they have to be set before those imports rather than at
# call time. main() re-exports EVAL3D_THREADS before spawning the pool; each
# worker re-imports this module and picks the new value up here. Set
# EVAL3D_THREADS yourself to override the budget entirely.
DEFAULT_WORKERS = 1
MAX_WORKERS = 8

_THREADS = os.environ.get("EVAL3D_THREADS", str(MAX_WORKERS))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = _THREADS

# The following libraries capture OpenMP settings during import, so their late
# import is intentional and is suppressed for standard lint configurations.
# pylint: disable=wrong-import-position
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from tqdm import tqdm  # noqa: E402

from point_cloud_converter import PointCloudConverter  # noqa: E402


MODELS    = [
    # baselines
    "da3",
    "cafnet",
    "cafnet_no_smoke",
    "radarcam-depth",
    # GRT family
    "grt",
    "grt_refine_freeze",
    "grt_no_doppler",
    "grt_image",
    "grt_refine_retrain",
    # ours
    "ours_full",
    "ours_full_no_3d",
    "ours_diffusion",
    "ours_radar",
    "ours_radar_no_doppler",
    "ours_radar_no_grad",
]
MAX_DEPTH = 11.2   # metres

# Common scoring grid, in (height, width). Kept in sync with the constant of the
# same name in eval_compute_simple.py: the 2D and 3D CSVs are joined per frame
# downstream, so they must describe the same evaluation protocol.
# --target-resolution native sets this to None, restoring the per-model grid.
DEFAULT_TARGET_RESOLUTION = (288, 512)

# Default output resides under evaluation/metric_results/. The unified runner
# overrides it so fresh 2D and 3D files always share one chosen output root.
OUTPUT_DIR_NAME = "simple_eval_results_3d"

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_ROOT = ARTIFACT_ROOT / "evaluation_dataset" / "Smoke-Eval"
DEFAULT_RESULTS_ROOT = ARTIFACT_ROOT / "evaluation" / "outputs" / "inference"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "metric_results" / OUTPUT_DIR_NAME

# Prediction file layout per model, relative to results_root.
# "{seq}" is substituted with the lowercased sequence name. Prediction
# folders/files are deliberately lowercase even though Smoke-Eval source
# sequence names retain their original casing.
PRED_PATH = {
    "da3":                    "da3/{seq}_pred.npy",
    "cafnet":                 "cafnet/{seq}_pred.npy",
    "cafnet_no_smoke":        "cafnet_no_smoke/{seq}_pred.npy",
    "radarcam-depth":         "radarcam-depth/{seq}_pred.npy",
    "grt":                    "grt/{seq}_pred.npy",
    "grt_refine_freeze":      "grt_refine_freeze/{seq}_pred.npy",
    "grt_no_doppler":         "grt_no_doppler/{seq}_pred.npy",
    "grt_image":              "grt_image/{seq}_pred.npy",
    "grt_refine_retrain":     "grt_refine_retrain/{seq}_pred.npy",
    "ours_full":              "ours_full/{seq}_pred.npy",
    "ours_full_no_3d":        "ours_full_no_3d/{seq}_pred.npy",
    "ours_diffusion":         "ours_diffusion/{seq}_pred.npy",
    "ours_radar":             "ours_radar/{seq}_pred.npy",
    "ours_radar_no_doppler":  "ours_radar_no_doppler/{seq}_pred.npy",
    "ours_radar_no_grad":     "ours_radar_no_grad/{seq}_pred.npy",
}

# Raw prediction units per model (multiply raw output by this to get metres).
# Verified against the observed value range of each model's arrays, except the
# two pending models, whose scales are assumed from their family and must be
# re-checked once their inference lands.
PRED_SCALE = {
    "da3":                    1.0,   # already in metres
    "cafnet":                 1.0,   # already in metres
    "cafnet_no_smoke":        1.0,   # already in metres
    "radarcam-depth":         1.0,   # already in metres
    "grt":                    11.2,  # normalised [0,1] → ×11.2
    "grt_refine_freeze":      11.2,  # normalised [0,1] → ×11.2
    "grt_no_doppler":         11.2,  # normalised [0,1] → ×11.2
    "grt_image":              11.2,  # normalised [0,1] -> x11.2 (verified: min 0.06, max 1.00)
    "grt_refine_retrain":     11.2,  # normalised [0,1] → ×11.2
    "ours_full":              11.2,  # normalised [0,1] → ×11.2
    "ours_full_no_3d":        11.2,  # ASSUMED, pending inference
    "ours_diffusion":         11.2,  # normalised [0,1] → ×11.2
    "ours_radar":             11.2,  # normalised [0,1] → ×11.2
    "ours_radar_no_doppler":  11.2,  # normalised [0,1] → ×11.2
    "ours_radar_no_grad":     11.2,  # normalised [0,1] → ×11.2
}

# Models whose inference skipped frames (e.g. no valid radar point cloud).
# The file lists the kept frames as "{seq}_{frame:06d}", one per line, in
# prediction order: prediction k corresponds to ground-truth frame index[k].
# Models absent from this mapping are assumed to be frame-for-frame with the
# ground truth.
PRED_INDEX_FILE = {
    "radarcam-depth": "radarcam-depth/test_indices.txt",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_zed_depth(seq_dir):
    """Open ZED depth as a lazy memmap of raw uint16 millimetres, shape (N, H, W).

    The array is deliberately *not* converted here: casting the whole memmap to
    float32 would materialise the entire sequence in RAM (several GB for the
    longer sequences). Per-frame conversion happens in gt_frame_metres().
    """
    path = seq_dir / "zed_depth.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return np.load(path, mmap_mode="r")


def gt_frame_metres(raw_frame):
    """Raw uint16 millimetre depth frame → float32 metres, clamped [0, MAX_DEPTH]."""
    depth = np.asarray(raw_frame).astype(np.float32) / 1000.0
    return np.clip(depth, 0, MAX_DEPTH)


def load_max30105(seq_dir):
    """Load MAX30105 sensor data (N, 3) = [R, IR, G]. Returns None if missing."""
    path = seq_dir / "max30105.npy"
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


_pred_index_cache = {}


def _read_index_file(path):
    """Parse a "{seq}_{frame:06d}" index file into {seq_name: int64 array}."""
    key = str(path)
    if key not in _pred_index_cache:
        per_seq = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                seq, frame = line.rsplit("_", 1)
                per_seq.setdefault(seq, []).append(int(frame))
        _pred_index_cache[key] = {
            s: np.asarray(v, dtype=np.int64) for s, v in per_seq.items()
        }
    return _pred_index_cache[key]


def load_pred_frame_index(model, seq_name, results_root):
    """Ground-truth frame index for each prediction, or None if frame-for-frame.

    Models listed in PRED_INDEX_FILE skipped frames during inference, so their
    prediction k is not ground-truth frame k. Returning the explicit index lets
    the caller gather the matching ground-truth frames and drop the rest.
    """
    rel = PRED_INDEX_FILE.get(model)
    if rel is None:
        return None

    path = results_root / rel
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frame index for {model}: {path}. It is required because "
            f"{model} skipped frames during inference."
        )

    index = _read_index_file(path).get(seq_name)
    if index is None or len(index) == 0:
        raise FileNotFoundError(f"No frames listed for {seq_name} in {path}")
    return index


def warn_on_frame_mismatch(model, seq_name, n_gt, n_pred):
    """Warn when a model's frame count differs from the ground truth's.

    Only reached for models with no PRED_INDEX_FILE entry, where predictions are
    paired with ground truth by array index and the longer of the two truncated.
    That is only correct when the two run frame-for-frame: if the model dropped
    frames part-way through a sequence, every later pair is silently mismatched
    and the reported metrics are meaningless. The correspondence cannot be
    recovered from the arrays alone, so this is surfaced loudly rather than
    papered over — supply a PRED_INDEX_FILE entry to fix it properly.
    """
    if n_gt == n_pred:
        return
    print(f"  WARNING: {model}/{seq_name} has {n_pred} predicted frames vs "
          f"{n_gt} ground-truth frames ({n_gt - n_pred:+d}).")
    print(f"           Pairing is by index and the tail is dropped; if the "
          f"missing frames are not all at the end, these metrics are invalid.")


def load_model_predictions(model, seq_name, results_root):
    """
    Load predictions for the given model, using the PRED_PATH layout.

    Raw predictions are float32 in [0, 1], except da3, cafnet,
    cafnet_no_smoke, and radarcam-depth, which are already in metres.
    Returns array of shape (N, H, W) or (N, 1, H, W).
    """
    if model not in PRED_PATH:
        raise ValueError(f"Unknown model: {model}. Choose from {MODELS}")

    path = results_root / PRED_PATH[model].format(seq=seq_name.lower())

    if not path.exists():
        raise FileNotFoundError(f"Missing predictions for {model}/{seq_name}: {path}")
    return np.load(path, mmap_mode="r")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resize_depth(img, target_h, target_w):
    """Resize a single (H, W) depth map to target dimensions.

    Nearest-neighbour is used so depth values are never blended across
    discontinuities, which would invent surfaces that neither input contains.
    """
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def parse_target_resolution(text):
    """Parse a --target-resolution value into (height, width) or None.

    ``"native"`` (case-insensitive) yields None, meaning "score each model on
    its own prediction grid". Anything else must be ``HxW``, e.g. ``288x512``.
    """
    if text is None:
        return None
    cleaned = str(text).strip().lower()
    if cleaned == "native":
        return None
    for separator in ("x", "*", ","):
        if separator in cleaned:
            height, _, width = cleaned.partition(separator)
            break
    else:
        raise ValueError(
            f"--target-resolution must be 'native' or HxW (e.g. 288x512); got {text!r}"
        )
    try:
        height, width = int(height.strip()), int(width.strip())
    except ValueError:
        raise ValueError(
            f"--target-resolution must be 'native' or HxW (e.g. 288x512); got {text!r}"
        ) from None
    if height < 2 or width < 2:
        raise ValueError(f"--target-resolution must be at least 2x2; got {text!r}")
    return height, width


def describe_resolution(target_resolution):
    """Human-readable label for the scoring grid, for log headers."""
    if target_resolution is None:
        return "native prediction grid (only ground truth resampled)"
    height, width = target_resolution
    return f"fixed {height}x{width} grid (ground truth and predictions resampled)"


def scoring_resolution(preds, target_resolution, model=None, seq_name=None):
    """The (height, width) this sequence is scored on.

    Returns ``target_resolution`` when one is configured, otherwise the array's
    native resolution.
    """
    if target_resolution is not None:
        return target_resolution
    return prediction_resolution(preds, model, seq_name)


def prediction_resolution(preds, model=None, seq_name=None):
    """Return the native (height, width) of a prediction array.

    Supported layouts are (N, H, W) and (N, 1, H, W). The optional model and
    sequence names are used only to make malformed-array errors actionable.
    """
    context = "/".join(x for x in (model, seq_name) if x) or "predictions"
    if preds.ndim == 3:
        target_h, target_w = preds.shape[1:]
    elif preds.ndim == 4 and preds.shape[1] == 1:
        target_h, target_w = preds.shape[2:]
    else:
        raise ValueError(
            f"{context}: expected predictions shaped (N,H,W) or (N,1,H,W), "
            f"got {tuple(preds.shape)}"
        )
    if target_h < 1 or target_w < 1:
        raise ValueError(
            f"{context}: prediction resolution must be positive, "
            f"got {target_h}x{target_w}"
        )
    return int(target_h), int(target_w)


# ---------------------------------------------------------------------------
# 3-D metric computation
# ---------------------------------------------------------------------------

def compute_cd_mhd(pcd1, pcd2):
    """
    Compute Chamfer Distance and Modified Hausdorff Distance in one pass,
    sharing the same nearest-neighbour distance arrays.

    CD  = mean(d(p→q)²) + mean(d(q→p)²)
    MHD = max(mean(d(p→q)), mean(d(q→p)))

    Returns (cd, mhd) as floats.
    """
    d_pq = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    d_qp = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    cd  = float(np.mean(d_pq ** 2) + np.mean(d_qp ** 2))
    mhd = float(max(np.mean(d_pq), np.mean(d_qp)))
    return cd, mhd


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

# Per-worker PointCloudConverter, initialised once per process by the Pool.
_worker_pcd_converter: "PointCloudConverter" = None


def _init_worker():
    """Pool initializer: create one PointCloudConverter per worker process.

    OpenCV keeps its own thread pool independent of OMP_NUM_THREADS; workers
    only do point-cloud work, so it is pinned to one thread here to stay inside
    the process-wide core budget.
    """
    global _worker_pcd_converter
    cv2.setNumThreads(1)
    _worker_pcd_converter = PointCloudConverter()


def _compute_frame_metrics(args):
    """
    Module-level worker: back-projects depth arrays to point clouds and
    computes CD / MHD.

    args = (frame_idx, gt_m, pred_m, r_val, ir_val, g_val)
      gt_m / pred_m : float32 (H, W) depth in *metres*
    """
    frame_idx, gt_m, pred_m, r_val, ir_val, g_val = args

    gt_pcd = _worker_pcd_converter.depth_to_point_cloud(
        gt_m, depth_scale=1.0, max_depth_meters=MAX_DEPTH
    )
    pred_pcd = _worker_pcd_converter.depth_to_point_cloud(
        pred_m, depth_scale=1.0, max_depth_meters=MAX_DEPTH
    )

    if len(gt_pcd.points) > 0 and len(pred_pcd.points) > 0:
        cd_val, mhd_val = compute_cd_mhd(pred_pcd, gt_pcd)
    else:
        cd_val = mhd_val = float("nan")

    return {
        "frame": frame_idx,
        "R":     r_val,
        "IR":    ir_val,
        "G":     g_val,
        "CD":    cd_val,
        "MHD":   mhd_val,
    }


# ---------------------------------------------------------------------------
# Per-sequence computation
# ---------------------------------------------------------------------------

def compute_sequence_metrics(model, seq_name, seq_dir, results_root,
                              num_workers=1,
                              target_resolution=DEFAULT_TARGET_RESOLUTION):
    """
    Compute per-frame CD and MHD for a single sequence.

    GT depth  : uint16 → float32 → /1000 m → clamp [0, MAX_DEPTH]
    Prediction: raw units → * pred_scale → metres → clamp [0, MAX_DEPTH]

    ``target_resolution`` is the common (height, width) grid every model is
    scored on; both ground truth and predictions are resampled onto it before
    being back-projected via PointCloudConverter's pinhole model, which scales
    the ZED intrinsics to match that grid. Pass None to fall back to the legacy
    behaviour, where the grid is the model's own prediction resolution and only
    the ground truth is resampled.

    When num_workers > 1 a multiprocessing Pool is used so that Open3D's
    KD-tree distance queries run on multiple CPU cores in parallel.

    Models listed in PRED_INDEX_FILE skipped frames during inference. For those,
    prediction k is scored against ground-truth frame gt_indices[k] and the
    skipped frames are simply absent from the output, so Frame_Index always
    refers to the ground-truth frame and stays joinable across models.

    Returns a list of dicts: frame, R, IR, G, CD, MHD  (sorted by frame).
    """
    gt_depth = load_zed_depth(seq_dir)   # (N, H, W) uint16 mm, lazy
    preds    = load_model_predictions(model, seq_name, results_root)
    max30105 = load_max30105(seq_dir)    # (N, 3) or None
    gt_index = load_pred_frame_index(model, seq_name, results_root)  # or None

    if gt_index is not None:
        # Prediction k ↔ ground-truth frame gt_index[k].
        num_frames = min(len(preds), len(gt_index))
        gt_indices = gt_index[:num_frames]
        if gt_indices.max(initial=-1) >= len(gt_depth):
            raise ValueError(
                f"{model}/{seq_name}: frame index {int(gt_indices.max())} is out of "
                f"range for {len(gt_depth)} ground-truth frames"
            )
    else:
        num_frames = min(len(gt_depth), len(preds))
        if max30105 is not None:
            num_frames = min(num_frames, len(max30105))
        gt_indices = np.arange(num_frames, dtype=np.int64)
        warn_on_frame_mismatch(model, seq_name, len(gt_depth), len(preds))

    pred_scale = PRED_SCALE[model]   # metres-per-unit for this model

    native_h, native_w = prediction_resolution(preds, model, seq_name)
    target_h, target_w = scoring_resolution(preds, target_resolution, model, seq_name)

    def frame_args():
        """Yield per-frame worker arguments lazily.

        Materialising every frame up front would hold two float32 depth maps
        per frame in memory at once (tens of GB at native resolution), so the
        frames are prepared on demand as the pool consumes them.
        """
        for k in range(num_frames):
            i = int(gt_indices[k])          # ground-truth frame number
            gt_m = resize_depth(gt_frame_metres(gt_depth[i]), target_h, target_w)

            pred_raw = np.asarray(preds[k])
            if pred_raw.ndim == 3 and pred_raw.shape[0] == 1:
                pred_raw = pred_raw[0]
            if pred_raw.shape != (native_h, native_w):
                raise ValueError(
                    f"{model}/{seq_name}: inconsistent prediction frame shape "
                    f"{pred_raw.shape}; expected {(native_h, native_w)}"
                )
            pred_m = np.clip(
                pred_raw.astype(np.float32) * pred_scale, 0, MAX_DEPTH
            )
            # No-op when the model already predicts at the scoring resolution.
            pred_m = resize_depth(pred_m, target_h, target_w)

            if max30105 is not None and i < len(max30105):
                r_val  = float(max30105[i][0])
                ir_val = float(max30105[i][1])
                g_val  = float(max30105[i][2])
            else:
                r_val = ir_val = g_val = float("nan")

            yield (i, gt_m, pred_m, r_val, ir_val, g_val)

    desc = f"{seq_name} [{model}]"
    if num_workers > 1:
        with Pool(
            processes=num_workers,
            initializer=_init_worker,
        ) as pool:
            rows = list(
                tqdm(pool.imap(_compute_frame_metrics, frame_args()),
                     total=num_frames, desc=desc)
            )
    else:
        _init_worker()   # initialise the global converter in the main process
        rows = [
            _compute_frame_metrics(a)
            for a in tqdm(frame_args(), total=num_frames, desc=desc)
        ]

    rows.sort(key=lambda r: r["frame"])
    return rows


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def save_sequence_csv(model, seq_name, rows, output_dir):
    """Write per-frame metrics to {output_dir}/3d_csv_{model}/{seq_name}.csv."""
    out_dir  = output_dir / f"3d_csv_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{seq_name}.csv"

    fieldnames = ["Sequence", "Frame_Index", "R", "IR", "G", "CD", "MHD"]

    def fmt(v):
        return f"{v:.6f}" if not (isinstance(v, float) and np.isnan(v)) else "nan"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Sequence":    seq_name,
                "Frame_Index": row["frame"],
                "R":           fmt(row["R"]),
                "IR":          fmt(row["IR"]),
                "G":           fmt(row["G"]),
                "CD":          fmt(row["CD"]),
                "MHD":         fmt(row["MHD"]),
            })

    print(f"  Saved: {csv_path}  ({len(rows)} frames)")


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def load_model_dfs(models, output_dir):
    """Load and concatenate the per-sequence CSVs for each model.

    Returns {model: DataFrame of all frames}, skipping models with no CSVs.
    """
    import pandas as pd

    model_dfs = {}
    for model in models:
        csv_dir = output_dir / f"3d_csv_{model}"
        csv_files = sorted(
            path for path in csv_dir.glob("*.csv") if not path.name.startswith("._")
        )
        if not csv_files:
            print(f"  WARNING: No CSVs for {model} in {csv_dir}, skipping.")
            continue
        model_dfs[model] = pd.concat(
            [pd.read_csv(p) for p in csv_files], ignore_index=True
        )
    return model_dfs


def create_sequence_summary_table(model_dfs, output_dir):
    """Build a per-sequence average-metrics table.

    Structure: rows = (Sequence × Model), columns = Sequence, Model, IR, CD, MHD.

    IR is the mean MAX30105 infrared reading over the sequence, reported as a
    raw value so smoke level can be judged directly from the table.

    Output: {output_dir}/sequence_avg_metrics_3d.csv
    """
    import pandas as pd

    rows = []
    for model, df in model_dfs.items():
        for seq, grp in df.groupby("Sequence"):
            rows.append({
                "Sequence": seq,
                "Model":    model,
                "IR":       float(grp["IR"].mean()) if "IR" in grp.columns else float("nan"),
                "CD":       float(grp["CD"].mean()),
                "CD_median":float(grp["CD"].median()),
                "MHD":      float(grp["MHD"].mean()),
                "MHD_median":float(grp["MHD"].median()),
            })

    tdf = pd.DataFrame(rows)[["Sequence", "Model", "IR", "CD", "CD_median", "MHD", "MHD_median"]]
    tdf.sort_values(["Sequence", "Model"], inplace=True, ignore_index=True)

    out_path = output_dir / "sequence_avg_metrics_3d.csv"
    tdf.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n--- Per-sequence average 3D metrics ---")
    print(tdf.to_string(index=False))
    return tdf


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def evaluate_all_sequences(model, data_root, results_root, output_dir,
                           sequences=None, num_workers=1,
                           target_resolution=DEFAULT_TARGET_RESOLUTION):
    """Evaluate sequences found in data_root for the given model.

    Predictions are read from results_root per PRED_PATH. Both they and the
    ground truth are resampled onto ``target_resolution``; pass None to score
    on each model's native prediction grid instead.
    CSVs and summary tables are written to output_dir/.
    num_workers controls the multiprocessing Pool size (1 = no pool).
    """
    all_sequences = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    if sequences is not None:
        missing = [s for s in sequences if s not in all_sequences]
        if missing:
            print(f"WARNING: sequences not found in {data_root}: {missing}")
        sequences = [s for s in sequences if s in all_sequences]
    else:
        sequences = all_sequences
    print(f"Evaluating {len(sequences)} sequence(s) in {data_root}  "
          f"[workers={num_workers}]")

    for seq_name in sequences:
        seq_dir = data_root / seq_name
        print(f"\n{'='*70}")
        print(f"  {seq_name}  [{model}]")
        print(f"{'='*70}")

        try:
            rows = compute_sequence_metrics(
                model, seq_name, seq_dir, results_root,
                num_workers=num_workers,
                target_resolution=target_resolution,
            )
        except FileNotFoundError as exc:
            print(f"  SKIP - {exc}")
            continue

        valid_cd  = [r["CD"]  for r in rows if not np.isnan(r["CD"])]
        valid_mhd = [r["MHD"] for r in rows if not np.isnan(r["MHD"])]
        if valid_cd:
            print(f"  CD:  {np.mean(valid_cd):.4f}")
        if valid_mhd:
            print(f"  MHD: {np.mean(valid_mhd):.4f}")

        save_sequence_csv(model, seq_name, rows, output_dir)

    print("\n" + "=" * 70)
    print("ALL SEQUENCES PROCESSED")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Argument parsing & entry point
# ---------------------------------------------------------------------------

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute 3D depth evaluation metrics (CD, MHD) for one or more models."
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        choices=MODELS,
        metavar="MODEL",
        help=(f"One or more models to evaluate. Omit to run every model. "
              f"Choices: {MODELS}"),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=f"Ground truth data root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=f"Prediction results root (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Per-sequence 3D CSV output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Skip computation; load existing CSVs and write the per-sequence summary table.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        nargs="+",
        default=None,
        metavar="SEQ",
        help="One or more sequence names to evaluate (default: all sequences in data_root)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=(f"Number of parallel worker processes for per-frame CD/MHD computation "
              f"(default: {DEFAULT_WORKERS}, i.e. no multiprocessing; hard cap: {MAX_WORKERS}). "
              "Each worker holds its own copy of a frame's depth maps and point "
              "clouds, so raising this trades memory for speed."),
    )
    default_resolution = "x".join(str(v) for v in DEFAULT_TARGET_RESOLUTION)
    parser.add_argument(
        "--target-resolution",
        type=str,
        default=default_resolution,
        metavar="HxW",
        help=(
            "Common grid every model is scored on, as HxW "
            f"(default: {default_resolution}). Ground truth and predictions are "
            "both resampled onto it before back-projection. Pass 'native' to "
            "score each model on its own prediction grid instead; that changes "
            "the numbers for models whose output is coarser than the default. "
            "Keep this in sync with eval_compute_simple.py."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_resolution = parse_target_resolution(args.target_resolution)
    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    results_root = Path(args.results_root) if args.results_root else DEFAULT_RESULTS_ROOT
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    models = args.model if args.model else list(MODELS)

    workers = max(1, min(args.workers, MAX_WORKERS))
    if args.workers > MAX_WORKERS:
        print(f"NOTE: --workers {args.workers} exceeds the cap of {MAX_WORKERS}; using {workers}.")

    # Split the core budget across the pool so workers * threads <= MAX_WORKERS.
    # Exported before the pool spawns; each worker re-imports this module and
    # applies it to its own native thread pools.
    threads_per_worker = max(1, MAX_WORKERS // workers)
    os.environ["EVAL3D_THREADS"] = str(threads_per_worker)
    # With a pool the parent only feeds frames; without one it does the work.
    cv2.setNumThreads(threads_per_worker if workers == 1 else 1)

    print("=" * 70)
    print("3D DEPTH EVALUATION  (CD + MHD)")
    print("=" * 70)
    print(f"Models:       {', '.join(models)}")
    print(f"Data root:    {data_root}")
    print(f"Results root: {results_root}")
    print(f"Output dir:   {output_dir}")
    print(f"Workers:      {workers} x {threads_per_worker} thread(s) "
          f"= {workers * threads_per_worker} core(s), cap {MAX_WORKERS}")
    print(f"Eval resolution: {describe_resolution(target_resolution)}")
    print("=" * 70)

    if args.table:
        model_dfs = load_model_dfs(models, output_dir)
        if model_dfs:
            create_sequence_summary_table(model_dfs, output_dir)
        else:
            print("  No model data found; cannot create summary table.")
        return

    for model in models:
        print(f"\n{'#'*70}")
        print(f"  MODEL: {model}")
        print(f"{'#'*70}")
        evaluate_all_sequences(
            model, data_root, results_root, output_dir,
            sequences=args.sequence, num_workers=workers,
            target_resolution=target_resolution,
        )
        print(f"\nCSVs saved to: {output_dir / f'3d_csv_{model}'}/")

    model_dfs = load_model_dfs(models, output_dir)
    if model_dfs:
        create_sequence_summary_table(model_dfs, output_dir)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
