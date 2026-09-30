"""Precompute the pixel-level absolute-error distribution against ground-truth
depth, as a fine-grained histogram that coarser bin widths are derived from.

The unit of analysis is the *pixel*, not the frame. Every valid ground-truth
pixel in the test set is assigned to a 20 cm ground-truth-depth bin by its own
depth value, and its absolute error is recorded there. Frame identity plays no
part: a frame contributes as many samples to a bin as it has pixels in that
depth range, and a frame contributing three pixels is not upweighted to parity
with one contributing five thousand. This is the discretised form of the
underlying scatter of per-pixel error against per-pixel ground-truth depth.

Storage. The population is ~3.8e9 pixels per model, far too large to keep as a
list of errors (~7.6 GB as int16 millimetres) merely to take a quantile from
it. Each 20 cm bin therefore stores a histogram of its errors in 1 mm buckets.
Ground truth is recorded in millimetres and both prediction and ground truth
are clamped to MAX_DEPTH, so errors are integers bounded by 11200 mm and the
histogram is a *lossless* stand-in for the sample multiset -- it discards only
which pixel produced which error. Cost: 56 x 11201 int64, ~5 MB per model.

Recombination. Histograms add. A coarser bin is the element-wise sum of its
constituent 20 cm rows, and quantiles read off that sum are exactly the
quantiles of the pooled pixel population, not an approximation of them. Stored
summary statistics could never be merged that way -- there is no way to
combine two medians. robustness_range_summarize.py does the collapse.

Two error histograms are stored per depth bin: absolute error in 1 mm buckets,
and AbsRel (|pred-gt|/gt) in 0.001 buckets. AbsRel is *not* redundant with
MAE/bin_centre once bins are merged: inside a merged 0-2 m bin ground truth
spans an order of magnitude, so a fixed absolute error is a wildly different
relative error at either end, and only the per-pixel ratio captures that.

Outputs:
  data/robustness/range_hist_20cm_{variant}.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

# This import follows the explicit package-root setup above.
from utils import eval_compute_simple as E  # noqa: E402


DEFAULT_DATA_ROOT = EVALUATION_DIR.parent / "evaluation_dataset" / "Smoke-Eval"
DEFAULT_RESULTS_ROOT = EVALUATION_DIR / "outputs" / "inference"
# This optional, expensive precomputation always writes newly derived data.
# It must never target evaluation/reference_results/.
DATA_DIR = EVALUATION_DIR / "metric_results" / "radar_robustness"

# Only the three series the range figure draws. The per-frame tables for the
# other baselines are left untouched in data/robustness/; nothing here reads
# or overwrites them.
MODELS = [
    ("ours_radar", "Ours_radar"),
    ("ours_diffusion", "Ours_diffusion"),
    ("ours_full", "GRADE"),
]

MAX_DEPTH_MM = int(round(E.MAX_DEPTH * 1000.0))  # 11200
BATCH_SIZE = 16

# Base ground-truth depth resolution. Any reported bin width must be a
# multiple of this and is built by summing consecutive base bins.
BASE_BIN_MM = 200  # 20 cm
N_GT_BINS = MAX_DEPTH_MM // BASE_BIN_MM  # 56, covering (0, 11.2] m exactly

# Error histogram resolution. Both prediction and ground truth are clamped to
# MAX_DEPTH, so |pred - gt| cannot exceed it; bucket k counts errors rounding
# to k millimetres, and the last bucket is a real value, not a catch-all.
N_ERR_BUCKETS = MAX_DEPTH_MM + 1  # 0..11200 mm inclusive

# AbsRel histogram. Unlike the absolute error, |pred-gt|/gt has no natural
# bound: a 0.1 m error is AbsRel 0.05 at 2 m but 10.0 at 1 cm, so the near-zero
# ground-truth tail can push it arbitrarily high. Buckets are 0.001 wide up to
# ABSREL_MAX, and the final bucket collects everything above it. Quantiles that
# land in that bucket are reported as NaN rather than as ABSREL_MAX, so a
# saturated statistic can never be mistaken for a measured one.
ABSREL_BUCKET = 0.001
ABSREL_MAX = 10.0
N_ABSREL_BUCKETS = int(round(ABSREL_MAX / ABSREL_BUCKET)) + 1  # 10001


def sequence_names(data_root: Path, selected: list[str] | None) -> list[str]:
    sequences = sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and (path / "zed_depth.npy").is_file()
    )
    if selected:
        requested = set(selected)
        sequences = [name for name in sequences if name in requested]
    return sequences


def compute_model_histogram(
    model_dir: str,
    label: str,
    data_root: Path,
    results_root: Path,
    sequences: list[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Accumulate the (gt_bin, error_bucket) counts over every valid pixel.

    Counts are kept per sequence rather than only in aggregate. Pixels within a
    sequence are heavily correlated -- a billion pixels come from ~26k frames of
    a few corridors -- so a bootstrap that resamples pixels would report
    absurdly tight intervals. Keeping the sequence axis allows a cluster
    bootstrap over sequences, which is the honest unit of independence here.
    Cost is one histogram per sequence, ~5 MB each.
    """
    # Held flat so a single bincount per batch does all the accumulation. The
    # alternative -- one boolean mask per bin -- is 56 full passes over every
    # batch instead of one.
    histogram = torch.zeros(
        N_GT_BINS * N_ERR_BUCKETS, dtype=torch.int64, device=device
    )
    # Exact running error total per bin, kept alongside the counts so the
    # reported mean does not inherit the 1 mm bucket rounding.
    error_sum_mm = torch.zeros(N_GT_BINS, dtype=torch.float64, device=device)
    absrel_histogram = torch.zeros(
        N_GT_BINS * N_ABSREL_BUCKETS, dtype=torch.int64, device=device
    )
    absrel_sum = torch.zeros(N_GT_BINS, dtype=torch.float64, device=device)
    per_sequence: list[np.ndarray] = []
    per_sequence_absrel: list[np.ndarray] = []
    kept_sequences: list[str] = []
    scale = E.PRED_SCALE[model_dir]

    for sequence in sequences:
        sequence_start = histogram.clone()
        absrel_start = absrel_histogram.clone()
        sequence_dir = data_root / sequence
        try:
            gt_raw = E.load_zed_depth(sequence_dir)
            predictions = E.load_model_predictions(model_dir, sequence, results_root)
            gt_index = E.load_pred_frame_index(model_dir, sequence, results_root)
        except FileNotFoundError as exc:
            print(f"SKIP {model_dir}/{sequence}: {exc}")
            continue

        # Same common scoring grid as eval_compute_simple. Without it a model's
        # pixel count -- and so its weight in the pooled population -- would
        # depend on its native output resolution rather than on its accuracy.
        (target_h, target_w), _ = E.eval_resolution(
            preds=predictions, model=model_dir, seq_name=sequence
        )
        if gt_index is not None:
            frame_count = min(len(predictions), len(gt_index))
            indices = gt_index[:frame_count]
        else:
            frame_count = min(len(gt_raw), len(predictions))
            indices = np.arange(frame_count, dtype=np.int64)

        for start in range(0, frame_count, BATCH_SIZE):
            end = min(start + BATCH_SIZE, frame_count)
            selected = indices[start:end]

            raw = np.ascontiguousarray(gt_raw[selected], dtype=np.float32)
            gt_mm = torch.from_numpy(raw).to(device).unsqueeze(1)
            gt_mm = E.resize_depth_gpu(gt_mm, target_h, target_w)
            valid = (gt_mm > 0) & (gt_mm <= MAX_DEPTH_MM)

            prediction_raw = np.asarray(predictions[start:end])
            if prediction_raw.ndim == 4 and prediction_raw.shape[1] == 1:
                prediction_raw = prediction_raw[:, 0]
            prediction_m = E.pred_batch_to_gpu(prediction_raw, scale, device)
            prediction_m = E.resize_depth_gpu(prediction_m, target_h, target_w)
            error_mm = (prediction_m * 1000.0 - gt_mm).abs()

            # ceil(gt / width) - 1 gives half-open (low, high] bins, matching
            # the convention the earlier tables used: a pixel at exactly
            # 200 mm belongs to bin 0, (0, 20] cm, not to bin 1.
            gt_bin = torch.ceil(gt_mm / BASE_BIN_MM).long() - 1
            gt_bin.clamp_(0, N_GT_BINS - 1)
            error_bucket = error_mm.round().long().clamp_(0, N_ERR_BUCKETS - 1)

            # gt_mm is strictly positive wherever valid is set, so this ratio
            # never divides by zero.
            absrel = error_mm / gt_mm.clamp(min=1.0)
            absrel_bucket = (absrel / ABSREL_BUCKET).round().long()
            absrel_bucket.clamp_(0, N_ABSREL_BUCKETS - 1)

            flat_bin = gt_bin[valid]
            histogram += torch.bincount(
                flat_bin * N_ERR_BUCKETS + error_bucket[valid],
                minlength=N_GT_BINS * N_ERR_BUCKETS,
            )
            error_sum_mm.scatter_add_(
                0, flat_bin, error_mm[valid].to(torch.float64)
            )
            absrel_histogram += torch.bincount(
                flat_bin * N_ABSREL_BUCKETS + absrel_bucket[valid],
                minlength=N_GT_BINS * N_ABSREL_BUCKETS,
            )
            absrel_sum.scatter_add_(0, flat_bin, absrel[valid].to(torch.float64))

        per_sequence.append(
            (histogram - sequence_start)
            .reshape(N_GT_BINS, N_ERR_BUCKETS).cpu().numpy()
        )
        per_sequence_absrel.append(
            (absrel_histogram - absrel_start)
            .reshape(N_GT_BINS, N_ABSREL_BUCKETS).cpu().numpy()
        )
        kept_sequences.append(sequence)
        print(f"{label:12s} {sequence} done")

    counts = histogram.reshape(N_GT_BINS, N_ERR_BUCKETS).cpu().numpy()
    absrel_counts = absrel_histogram.reshape(
        N_GT_BINS, N_ABSREL_BUCKETS
    ).cpu().numpy()
    return {
        "counts": counts,
        "counts_by_sequence": np.stack(per_sequence) if per_sequence
        else np.zeros((0, N_GT_BINS, N_ERR_BUCKETS), dtype=np.int64),
        "absrel_counts": absrel_counts,
        "absrel_counts_by_sequence": np.stack(per_sequence_absrel)
        if per_sequence_absrel
        else np.zeros((0, N_GT_BINS, N_ABSREL_BUCKETS), dtype=np.int64),
        "sequences": np.array(kept_sequences),
        "error_sum_mm": error_sum_mm.cpu().numpy(),
        "absrel_sum": absrel_sum.cpu().numpy(),
        "pixel_count": counts.sum(axis=1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute the pixel-level error histogram against "
                    "ground-truth depth, in 20 cm depth bins."
    )
    parser.add_argument(
        "--model", nargs="+", choices=[name for name, _ in MODELS],
        default=[name for name, _ in MODELS],
        help="Restrict computation to these models (default: all three).",
    )
    parser.add_argument("--sequence", nargs="+")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_models = [item for item in MODELS if item[0] in args.model]
    sequences = sequence_names(args.data_root, args.sequence)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device}; sequences: {len(sequences)}; "
        f"models: {[m for m, _ in selected_models]}; "
        f"{N_GT_BINS} x {BASE_BIN_MM}mm depth bins"
    )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    edges_mm = np.arange(N_GT_BINS + 1, dtype=np.int64) * BASE_BIN_MM

    for model_dir, label in selected_models:
        result = compute_model_histogram(
            model_dir, label, args.data_root, args.results_root,
            sequences, device,
        )
        # One file per model, so recomputing one never disturbs the other.
        output = args.data_dir / f"range_hist_20cm_{model_dir}.npz"
        np.savez_compressed(
            output,
            counts=result["counts"],
            counts_by_sequence=result["counts_by_sequence"],
            absrel_counts=result["absrel_counts"],
            absrel_counts_by_sequence=result["absrel_counts_by_sequence"],
            sequences=result["sequences"],
            error_sum_mm=result["error_sum_mm"],
            absrel_sum=result["absrel_sum"],
            pixel_count=result["pixel_count"],
            edges_mm=edges_mm,
            base_bin_mm=np.int64(BASE_BIN_MM),
            max_depth_mm=np.int64(MAX_DEPTH_MM),
            absrel_bucket=np.float64(ABSREL_BUCKET),
            absrel_max=np.float64(ABSREL_MAX),
            model=np.str_(label),
            variant=np.str_(model_dir),
        )
        total = int(result["pixel_count"].sum())
        print(f"Saved {output}  ({total:,} valid pixels)")


if __name__ == "__main__":
    main()
