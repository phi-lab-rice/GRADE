"""
Evaluation computation: depth prediction metrics per model.

For a selected model, computes per-frame metrics for each sequence and writes CSVs to:
  Inference_eval/csv_{model_name}/{seq_name}.csv

Metrics per frame (predicted depth vs ZED ground truth):
  - MAE (L1)
  - AbsRel
  - SSIM
  - LPIPS
  - GradientError

Also writes, per frame:
  - MAX30105 sensor readings (R, IR, G)
  - RMS_contrast: ground-truth RMS contrast of zed_rgb.npy (model independent)

Every model is scored on a common 288x512 grid by default (--target-resolution).
Ground truth and predictions are both resampled onto that grid, so models whose
native output is coarser than 288x512 -- GRT and its variants, ours_radar and
its variants -- are compared on the same footing as the rest.  Passing
--target-resolution native restores the older behaviour, where each model is
scored on its own prediction grid and only the ground truth is resampled; those
two protocols do NOT produce the same numbers for coarse-output models.

All computation runs on the GPU in batches of BATCH_SIZE frames — unit
conversion, resizing and every metric — so there is no CPU worker pool and no
--workers flag. Lower BATCH_SIZE if the GPU runs out of memory.

Ground truth data: <artifact_root>/evaluation_dataset/Smoke-Eval/{seq_name}/
Predictions      : <artifact_root>/evaluation/outputs/inference/{model_dir}/...
Models           : da3, cafnet, cafnet_no_smoke, radarcam-depth,
                   grt, grt_refine_freeze, grt_no_doppler, grt_image,
                   ours_full, ours_diffusion, ours_radar,
                   ours_radar_no_doppler, ours_radar_no_grad,
                   ours_full_no_3d, grt_refine_retrain

Results default to <script_dir>/simple_eval_results/ and can be redirected with
--output_dir.

Usage:
  python eval_compute_simple.py                       # every model
  python eval_compute_simple.py --model ours_full
  python eval_compute_simple.py --model da3 cafnet cafnet_no_smoke grt_image
  python eval_compute_simple.py --model grt grt_no_doppler
  python eval_compute_simple.py --target-resolution native   # legacy per-model grid
  python eval_compute_simple.py --target-resolution 480x640
  python eval_compute_simple.py --table               # existing CSVs → summary table
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm


MODELS = [
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
MAX_DEPTH = 11.2  # metres

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

BATCH_SIZE = 32  # frames per GPU batch

# Common scoring grid, in (height, width). Every model is resampled onto this
# grid so that coarse-output models (GRT and its variants, ours_radar and its
# variants) are not scored on an easier, lower-resolution raster than the rest.
# --target-resolution native sets this to None, restoring the per-model grid.
DEFAULT_TARGET_RESOLUTION = (288, 512)

# Default output resides under evaluation/metric_results/. The unified runner
# overrides it so fresh 2D and 3D files always share one chosen output root.
OUTPUT_DIR_NAME = "simple_eval_results"

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_ROOT = ARTIFACT_ROOT / "evaluation_dataset" / "Smoke-Eval"
DEFAULT_RESULTS_ROOT = ARTIFACT_ROOT / "evaluation" / "outputs" / "inference"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "metric_results" / OUTPUT_DIR_NAME


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_zed_depth(seq_dir):
    """Open ZED depth as a lazy memmap of raw uint16 millimetres, shape (N, H, W).

    The array is deliberately *not* converted here: casting the whole memmap to
    float32 would materialise the entire sequence in RAM (several GB for the
    longer sequences). Unit conversion and clamping happen per batch on the GPU
    in gt_batch_to_gpu().
    """
    path = seq_dir / "zed_depth.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return np.load(path, mmap_mode="r")


def load_max30105(seq_dir):
    """
    Load MAX30105 sensor data.
    Shape: (N, 3), columns: [R, IR, G].
    Returns None if file is missing.
    """
    path = seq_dir / "max30105.npy"
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


def load_zed_rgb(seq_dir):
    """Load ZED RGB frames (N, H, W, 3) uint8. Returns None if the file is missing."""
    path = seq_dir / "zed_rgb.npy"
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


def rms_contrast_gpu(rgb_batch, device):
    """Per-frame RMS contrast of a (B, H, W, 3) uint8 batch, computed on GPU.

    Converts to luma, normalises to [0, 1] and returns the per-frame standard
    deviation. Higher values mean more visible texture/edges; smoke reduces
    contrast. unbiased=False keeps this the population standard deviation, as
    NumPy's ndarray.std() computes.
    """
    t = torch.from_numpy(np.ascontiguousarray(rgb_batch)).to(device)
    t = t.float().div_(255.0)
    gray = 0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2]
    return gray.flatten(1).std(dim=1, unbiased=False)   # (B,)


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
        raise FileNotFoundError(
            f"No frames listed for {seq_name} in {path}"
        )
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

    Returns a numpy array of shape (N, H, W) or (N, 1, H, W).
    Raw predictions are float32 in [0, 1], except da3, cafnet,
    cafnet_no_smoke, and radarcam-depth, which are already in metres.
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

def resize_depth_gpu(t, target_h, target_w):
    """Resize a (B, 1, H, W) depth tensor to (target_h, target_w) on-device.

    Nearest-neighbour is used so depth values are never blended across
    discontinuities, which would invent surfaces that neither input contains.
    """
    if t.shape[-2] == target_h and t.shape[-1] == target_w:
        return t
    return F.interpolate(t, size=(target_h, target_w), mode="nearest")


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
    if target_h < 2 or target_w < 2:
        raise ValueError(
            f"{context}: prediction resolution must be at least 2x2, "
            f"got {target_h}x{target_w}"
        )
    return int(target_h), int(target_w)


def gt_batch_to_gpu(raw_batch, device):
    """Raw uint16 millimetre depth (B, H, W) → (B, 1, H, W) metres on GPU.

    The uint16 → float cast has to happen host-side because torch has no
    practical uint16 dtype; everything after it runs on the GPU.
    """
    t = torch.from_numpy(np.ascontiguousarray(raw_batch, dtype=np.float32))
    t = t.to(device).unsqueeze(1)
    return (t / 1000.0).clamp_(0, MAX_DEPTH)


def pred_batch_to_gpu(raw_batch, pred_scale, device):
    """Raw prediction batch (B, H, W) → (B, 1, H, W) metres on GPU."""
    # Always copy: mmap slices are read-only, and .to(cpu) would otherwise keep
    # a non-writable NumPy backing array that clamp_ must not modify in place.
    t = torch.from_numpy(np.array(raw_batch, dtype=np.float32, copy=True, order="C"))
    t = t.to(device).unsqueeze(1)
    return (t * pred_scale).clamp_(0, MAX_DEPTH)


def to_three_channel(t):
    """Expand a (B, 1, H, W) tensor to (B, 3, H, W) for SSIM / LPIPS."""
    return t.expand(-1, 3, -1, -1)


def depth_gradient(x):
    """Horizontal and vertical finite differences of a [B, 1, H, W] tensor.

    Mirrors PerceptualLoss._gradient from the training code.
    """
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def gradient_matching_error(pred_norm, gt_norm):
    """Per-frame depth-gradient L1 error, matching the training `grad` loss term.

    Equivalent to `L1Loss(pred_dx, gt_dx) + L1Loss(pred_dy, gt_dy)` in
    PerceptualLoss, but reduced per frame instead of over the whole batch.

    Args:
        pred_norm: (B, 1, H, W) tensor in [0, 1]
        gt_norm:   (B, 1, H, W) tensor in [0, 1]

    Returns:
        (B,) tensor of per-frame gradient errors.
    """
    pred_dx, pred_dy = depth_gradient(pred_norm)
    gt_dx, gt_dy = depth_gradient(gt_norm)
    loss_dx = (pred_dx - gt_dx).abs().mean(dim=(1, 2, 3))
    loss_dy = (pred_dy - gt_dy).abs().mean(dim=(1, 2, 3))
    return loss_dx + loss_dy


def compute_sequence_metrics(model, seq_name, seq_dir, results_root,
                              lpips_model, ssim_model, device,
                              pred_scale_override=None,
                              target_resolution=DEFAULT_TARGET_RESOLUTION):
    """
    Compute per-frame metrics for a single sequence with a single model.

    GT depth  : raw uint16 millimetres → /1000 m → clamp [0, MAX_DEPTH]
    Predictions: raw units → * pred_scale → metres → clamp [0, MAX_DEPTH]

    All arithmetic — unit conversion, resizing, and every metric — runs on the
    GPU. The host only slices the memmaps and casts to float32 for transfer,
    so nothing scales with CPU core count.

    ``target_resolution`` is the common (height, width) grid every model is
    scored on; both ground truth and predictions are resampled onto it. Pass
    None to fall back to the legacy behaviour, where the grid is the model's
    own prediction resolution and only the ground truth is resampled.

      - MAE    : mean(|pred_m - gt_m|)
      - AbsRel : mean(|pred_m - gt_m| / gt_m) over gt_m > 0
      - SSIM   : pred / MAX_DEPTH  vs  gt / MAX_DEPTH
      - LPIPS  : pred / MAX_DEPTH  vs  gt / MAX_DEPTH
      - GradientError : L1 of dx/dy finite differences on [0,1] depth, matching
                        the `grad` term of the training PerceptualLoss

    RMS_contrast is read from zed_rgb.npy and is model independent; it is
    recorded alongside the metrics so each CSV is self-contained.

    Models listed in PRED_INDEX_FILE skipped frames during inference. For those,
    prediction k is scored against ground-truth frame gt_indices[k] and the
    skipped frames are simply absent from the output, so Frame_Index always
    refers to the ground-truth frame and stays joinable across models.

    Returns a list of dicts (one per frame) containing:
      frame, R, IR, G, RMS_contrast, MAE, AbsRel, SSIM, LPIPS, GradientError
    """
    gt_depth = load_zed_depth(seq_dir)            # (N, H, W) uint16 mm, lazy
    preds    = load_model_predictions(model, seq_name, results_root)
    max30105 = load_max30105(seq_dir)             # (N, 3) or None
    zed_rgb  = load_zed_rgb(seq_dir)              # (N, H, W, 3) or None
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

    pred_scale = (
        pred_scale_override
        if pred_scale_override is not None
        else PRED_SCALE[model]
    )  # metres-per-unit for this model

    native_h, native_w = prediction_resolution(preds, model, seq_name)
    target_h, target_w = scoring_resolution(preds, target_resolution, model, seq_name)

    rows = []
    for batch_start in tqdm(range(0, num_frames, BATCH_SIZE), desc=f"{seq_name} [{model}]"):
        batch_end = min(batch_start + BATCH_SIZE, num_frames)
        gt_sel    = gt_indices[batch_start:batch_end]   # GT frame numbers

        # --- GT → GPU, metres, then onto the scoring grid: (B, 1, H, W) ---
        gt_m = gt_batch_to_gpu(gt_depth[gt_sel], device)
        gt_m = resize_depth_gpu(gt_m, target_h, target_w)

        # --- predictions → GPU, metres, then onto the scoring grid ---
        pred_raw = np.asarray(preds[batch_start:batch_end])
        if pred_raw.ndim == 4 and pred_raw.shape[1] == 1:
            pred_raw = pred_raw[:, 0]          # (B, 1, H, W) → (B, H, W)
        pred_m = pred_batch_to_gpu(pred_raw, pred_scale, device)
        if pred_m.shape[-2:] != (native_h, native_w):
            raise ValueError(
                f"{model}/{seq_name}: inconsistent prediction batch shape "
                f"{tuple(pred_m.shape[-2:])}; expected {(native_h, native_w)}"
            )
        # No-op when the model already predicts at the scoring resolution.
        pred_m = resize_depth_gpu(pred_m, target_h, target_w)

        with torch.no_grad():
            # --- L1 in metres, per frame ---
            abs_err   = (pred_m - gt_m).abs()
            mae_batch = abs_err.mean(dim=(1, 2, 3))                      # (B,)

            # --- AbsRel: mean(|pred - gt| / gt) over valid (gt > 0) pixels ---
            rel = torch.where(gt_m > 0, abs_err / gt_m.clamp(min=1e-6),
                              torch.zeros_like(abs_err))
            absrel_batch = rel.mean(dim=(1, 2, 3))                       # (B,)

            # --- normalised [0,1] tensors, expanded to 3 channels for SSIM/LPIPS ---
            gt_n   = gt_m / MAX_DEPTH
            pred_n = pred_m / MAX_DEPTH
            ssim_batch  = ssim_model(to_three_channel(pred_n), to_three_channel(gt_n))
            lpips_batch = lpips_model(to_three_channel(pred_n), to_three_channel(gt_n))
            # gradient error on single-channel [0,1] depth, as in the training loss
            grad_batch  = gradient_matching_error(pred_n, gt_n)          # (B,)

            # RMS contrast is a ground-truth quantity, so it follows the GT index
            if zed_rgb is not None and int(gt_sel.max()) < len(zed_rgb):
                rms_batch = rms_contrast_gpu(zed_rgb[gt_sel], device)
            else:
                rms_batch = None

        # single host sync per batch
        mae_vals    = mae_batch.cpu().numpy()
        absrel_vals = absrel_batch.cpu().numpy()
        ssim_vals   = np.atleast_1d(ssim_batch.cpu().numpy())
        lpips_vals  = np.atleast_1d(lpips_batch.cpu().numpy())
        grad_vals   = np.atleast_1d(grad_batch.cpu().numpy())
        rms_vals    = rms_batch.cpu().numpy() if rms_batch is not None else None

        for j, i in enumerate(int(g) for g in gt_sel):
            if max30105 is not None and i < len(max30105):
                r_val, ir_val, g_val = float(max30105[i][0]), float(max30105[i][1]), float(max30105[i][2])
            else:
                r_val = ir_val = g_val = float("nan")

            rms_val = (
                float(rms_vals[j])
                if rms_vals is not None and j < len(rms_vals)
                else float("nan")
            )

            rows.append({
                "frame":  i,
                "R":      r_val,
                "IR":     ir_val,
                "G":      g_val,
                "RMS_contrast": rms_val,
                "MAE":    float(mae_vals[j]),
                "AbsRel": float(absrel_vals[j]),
                "SSIM":   float(ssim_vals[j]),
                "LPIPS":  float(lpips_vals[j]),
                "GradientError": float(grad_vals[j]),
            })

    return rows


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def save_sequence_csv(model, seq_name, rows, output_dir):
    """Write per-frame metrics to {output_dir}/csv_{model}/{seq_name}.csv."""
    out_dir = output_dir / f"csv_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{seq_name}.csv"

    fieldnames = [
        "Sequence", "Frame_Index", "R", "IR", "G", "RMS_contrast",
        "MAE", "AbsRel", "SSIM", "LPIPS", "GradientError",
    ]

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
                "RMS_contrast": fmt(row["RMS_contrast"]),
                "MAE":         fmt(row["MAE"]),
                "AbsRel":      fmt(row["AbsRel"]),
                "SSIM":        fmt(row["SSIM"]),
                "LPIPS":       fmt(row["LPIPS"]),
                "GradientError": fmt(row["GradientError"]),
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
        csv_dir = output_dir / f"csv_{model}"
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

    Structure: rows = (Sequence × Model), columns = Sequence, Model, IR,
    RMS_contrast, MAE, AbsRel, 1-SSIM, LPIPS, GradientError.

    IR is the mean MAX30105 infrared reading over the sequence and RMS_contrast
    the mean ZED-RGB contrast; both are raw ground-truth values, reported so the
    smoke level can be judged directly from the table.

    Output: {output_dir}/sequence_avg_metrics.csv
    """
    import pandas as pd

    rows = []
    for model, df in model_dfs.items():
        for seq, grp in df.groupby("Sequence"):
            rows.append({
                "Sequence": seq,
                "Model":    model,
                "IR":       float(grp["IR"].mean()) if "IR" in grp.columns else float("nan"),
                "RMS_contrast": (
                    float(grp["RMS_contrast"].mean())
                    if "RMS_contrast" in grp.columns
                    else float("nan")
                ),
                "MAE":      float(grp["MAE"].mean()),
                "AbsRel":   float(grp["AbsRel"].mean()),
                "1-SSIM":   float((1 - grp["SSIM"]).mean()),
                "LPIPS":    float(grp["LPIPS"].mean()),
                "GradientError": (
                    float(grp["GradientError"].mean())
                    if "GradientError" in grp.columns
                    else float("nan")
                ),
            })

    tdf = pd.DataFrame(rows)[
        ["Sequence", "Model", "IR", "RMS_contrast",
         "MAE", "AbsRel", "1-SSIM", "LPIPS", "GradientError"]
    ]
    tdf.sort_values(["Sequence", "Model"], inplace=True, ignore_index=True)

    out_path = output_dir / "sequence_avg_metrics.csv"
    tdf.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n--- Per-sequence average metrics ---")
    print(tdf.to_string(index=False))
    return tdf


# ---------------------------------------------------------------------------

def evaluate_all_sequences(model, data_root, results_root, output_dir, sequences=None,
                           pred_scale_override=None,
                           target_resolution=DEFAULT_TARGET_RESOLUTION):
    """Evaluate sequences found in data_root for the given model.

    Predictions are loaded from results_root per PRED_PATH. Both they and the
    ground truth are resampled onto ``target_resolution``; pass None to score
    on each model's native prediction grid instead.
    CSVs and summary tables are written to output_dir/.
    If sequences is provided, only those sequence names are evaluated.
    """
    all_sequences = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    if sequences is not None:
        missing = [s for s in sequences if s not in all_sequences]
        if missing:
            print(f"WARNING: sequences not found in {data_root}: {missing}")
        sequences = [s for s in sequences if s in all_sequences]
    else:
        sequences = all_sequences
    print(f"Evaluating {len(sequences)} sequence(s) in {data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lpips_model = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="none"
    ).to(device)
    lpips_model.eval()

    ssim_model = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)

    for seq_name in sequences:
        seq_dir = data_root / seq_name
        print(f"\n{'='*70}")
        print(f"  {seq_name}  [{model}]")
        print(f"{'='*70}")

        try:
            rows = compute_sequence_metrics(
                model, seq_name, seq_dir, results_root,
                lpips_model, ssim_model, device,
                pred_scale_override=pred_scale_override,
                target_resolution=target_resolution,
            )
        except FileNotFoundError as exc:
            print(f"  SKIP - {exc}")
            continue

        print(f"  MAE:    {np.mean([r['MAE']    for r in rows]):.4f}")
        print(f"  AbsRel: {np.mean([r['AbsRel'] for r in rows]):.4f}")
        print(f"  SSIM:   {np.mean([r['SSIM']   for r in rows]):.4f}")
        print(f"  LPIPS:  {np.mean([r['LPIPS']  for r in rows]):.4f}")
        print(f"  GradientError: {np.mean([r['GradientError'] for r in rows]):.4f}")
        print(f"  RMS_contrast:  {np.nanmean([r['RMS_contrast'] for r in rows]):.4f}")

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
        description="Compute depth evaluation metrics for one or more models."
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
        help=f"Per-sequence 2D CSV output directory (default: {DEFAULT_OUTPUT_DIR})",
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
        "--pred-scale",
        type=float,
        default=None,
        help="Override the model's raw-prediction-to-metres scale for this run.",
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
            "both resampled onto it. Pass 'native' to score each model on its "
            "own prediction grid instead; that changes the numbers for models "
            "whose output is coarser than the default."
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

    print("=" * 70)
    print("DEPTH EVALUATION")
    print("=" * 70)
    print(f"Models:       {', '.join(models)}")
    print(f"Data root:    {data_root}")
    print(f"Results root: {results_root}")
    print(f"Output dir:   {output_dir}")
    print(f"Batch size:   {BATCH_SIZE} frames/GPU batch")
    if args.pred_scale is not None:
        print(f"Prediction scale override: {args.pred_scale}")
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
            sequences=args.sequence,
            pred_scale_override=args.pred_scale,
            target_resolution=target_resolution,
        )
        print(f"\nCSVs saved to: {output_dir / f'csv_{model}'}/")

    model_dfs = load_model_dfs(models, output_dir)
    if model_dfs:
        create_sequence_summary_table(model_dfs, output_dir)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
