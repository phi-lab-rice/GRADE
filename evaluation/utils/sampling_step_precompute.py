"""Precompute the DDIM sampling-step ablation, pooling clear and smoke.

The published table reported the clear and heavy-smoke sequences separately,
each summarised by a per-frame median. Pooling those two rows into one cannot
be done from the summary file: the median of a union is not the average of the
two medians. This script therefore goes back to the per-frame CSVs in the
initial-submission evaluation tree (``eval_code``), concatenates the two
sequences frame by frame, and takes the median over the pooled set.

The 2D metrics (MAE/SSIM/LPIPS) and the 3D metrics (CD/MHD) live in separate
per-frame files, so they are joined on (Sequence, Frame_Index) with a
one-to-one validation before pooling; a silent many-to-one join here would
quietly corrupt every number in the table.

Output is written to ``data/sampling_step_ablation_pooled.csv`` so that
``table_7.py`` stays runnable without the eval_code tree mounted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = EVALUATION_DIR.parent

# The initial-submission evaluation tree sits alongside the camera-ready
# folder: <...>/GRADE_rebuttal/{GRADE_MobiCom_2026_Camera_Ready,eval_code}
DEFAULT_EVAL_CODE = ARTIFACT_ROOT.parent / "eval_code"
RESULTS_SUBDIR = Path("analysis") / "new_eval_results"

# Only two sequences were ever run for this ablation: one clear, one heavy
# smoke. They are pooled into a single row per step.
SEQUENCES = ["Smoke-dell-1-0", "Smoke-dell-1-3"]

# step=2 is dropped: it sits deep in the unconverged regime and the row carried
# no argument the step=1 row does not already make.
STEPS = [1, 5, 8, 10, 50]

METRICS_2D = ["MAE", "SSIM", "LPIPS"]
METRICS_3D = ["CD", "MHD"]

OUTPUT_NAME = "sampling_step_ablation_pooled.csv"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "metric_results" / "sampling_step"


def load_pooled(results_root: Path, step: int) -> pd.DataFrame:
    """Per-frame 2D+3D metrics for both sequences at one sampling step."""
    frames = []
    for sequence in SEQUENCES:
        path_2d = results_root / "simple_eval_results" / f"csv_step_{step}" / f"{sequence}.csv"
        path_3d = (
            results_root
            / "simple_eval_results_3d"
            / f"3d_csv_step_{step}"
            / f"{sequence}.csv"
        )
        for path in (path_2d, path_3d):
            if not path.is_file():
                raise FileNotFoundError(f"missing per-frame file: {path}")
        two_d = pd.read_csv(path_2d)
        three_d = pd.read_csv(path_3d)
        merged = two_d.merge(
            three_d,
            on=["Sequence", "Frame_Index"],
            validate="one_to_one",
        )
        if len(merged) != len(two_d) or len(merged) != len(three_d):
            raise ValueError(
                f"2D/3D frame mismatch for {sequence} at step {step}: "
                f"2D={len(two_d)} 3D={len(three_d)} joined={len(merged)}"
            )
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def build(results_root: Path) -> pd.DataFrame:
    rows = []
    for step in STEPS:
        pooled = load_pooled(results_root, step)
        row = {"Sampling step": step, "Number of frames": len(pooled)}
        for metric in METRICS_2D + METRICS_3D:
            row[metric] = float(pooled[metric].median())
        rows.append(row)
    table = pd.DataFrame(rows)
    counts = table["Number of frames"].unique()
    if len(counts) != 1:
        raise ValueError(f"frame count differs across steps: {sorted(counts)}")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-code",
        type=Path,
        default=DEFAULT_EVAL_CODE,
        help=f"Initial-submission evaluation tree (default: {DEFAULT_EVAL_CODE}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Fresh destination; reference_results/pre_eval_results is read-only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.eval_code / RESULTS_SUBDIR
    if not results_root.is_dir():
        raise SystemExit(
            f"eval_code results not found at {results_root}.\n"
            "Pass --eval-code with the path to the initial-submission tree."
        )
    table = build(results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / OUTPUT_NAME
    table.to_csv(destination, index=False)
    print(f"pooled {int(table['Number of frames'].iloc[0])} frames "
          f"({' + '.join(SEQUENCES)}) per step")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
