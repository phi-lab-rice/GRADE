#!/usr/bin/env python3
"""Compute per-model metrics or the final radar-robustness inputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from model_registry import EVALUATION_VARIANTS, INFERENCE_MODELS, resolve_metric_variant


EVALUATION_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = EVALUATION_DIR.parent
DEFAULT_DATA_ROOT = ARTIFACT_ROOT / "evaluation_dataset" / "Smoke-Eval"
DEFAULT_RESULTS_ROOT = EVALUATION_DIR / "outputs" / "inference"
DEFAULT_OUTPUT_ROOT = EVALUATION_DIR / "metric_results"
RADAR_ROBUSTNESS_VARIANTS = ("ours_radar", "ours_diffusion", "ours_full")


def parse_args() -> argparse.Namespace:
    choices = sorted({*INFERENCE_MODELS, *EVALUATION_VARIANTS})
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--model", choices=choices,
        help="Run the 2D evaluator, 3D evaluator, and merged-CSV builder for one model.",
    )
    mode.add_argument(
        "--radar-robustness", action="store_true",
        help=(
            "Build the final Figure 13 sparsity summary and range histograms after "
            "the three GRADE variants have completed ordinary metric evaluation."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help="Root containing the new prediction variant, or inference_results/ for archives.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Destination for fresh raw 2D/3D CSVs and merged CSVs (default: evaluation/metric_results/).",
    )
    parser.add_argument("--sequence", nargs="+", default=None, metavar="SEQ")
    parser.add_argument("--target-resolution", default="288x512", metavar="HxW")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pred-scale", type=float, default=None)
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def radar_robustness_commands(args: argparse.Namespace) -> list[list[str]]:
    """Commands that derive the fresh Figure 13 inputs without touching golden data."""

    output_root = args.output_root.resolve()
    robustness_dir = output_root / "radar_robustness"
    sparse_command = [
        sys.executable,
        str(EVALUATION_DIR / "metric_results" / "robustness_sparsity.py"),
        "--pre-compute",
        "--paper-only",
        "--bin-cuts", "9", "13",
        "--data-root", str(args.data_root.resolve()),
        "--raw-root", str(output_root),
        "--data-dir", str(robustness_dir),
    ]
    range_command = [
        sys.executable,
        str(EVALUATION_DIR / "metric_results" / "robustness_range_precompute.py"),
        "--model", *RADAR_ROBUSTNESS_VARIANTS,
        "--data-root", str(args.data_root.resolve()),
        "--results-root", str(args.results_root.resolve()),
        "--data-dir", str(robustness_dir),
    ]
    if args.sequence:
        sparse_command.extend(("--sequence", *args.sequence))
        range_command.extend(("--sequence", *args.sequence))
    return [sparse_command, range_command]


def validate_radar_robustness_inputs(output_root: Path) -> None:
    """Fail early when the prerequisite ordinary metric runs are incomplete."""

    missing = []
    for variant in RADAR_ROBUSTNESS_VARIANTS:
        for relative in (
            Path("simple_eval_results") / f"csv_{variant}",
            Path("simple_eval_results_3d") / f"3d_csv_{variant}",
        ):
            path = output_root / relative
            if not path.is_dir() or not any(path.glob("*.csv")):
                missing.append(path)
    if missing:
        detail = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Radar robustness needs completed ordinary metrics for "
            f"{', '.join(RADAR_ROBUSTNESS_VARIANTS)}. Missing:\n  {detail}\n"
            "Run evaluation/run_metrics.py --model <variant> for each variant first."
        )


def run_radar_robustness(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    commands = radar_robustness_commands(args)
    print("Mode:         radar robustness (Figure 13)")
    print(f"Data root:    {args.data_root.resolve()}")
    print(f"Results root: {args.results_root.resolve()}")
    print(f"Raw metrics:  {output_root}")
    print(f"Output dir:   {output_root / 'radar_robustness'}")
    for command in commands:
        print("Command:      " + subprocess.list2cmdline(command))
    if args.dry_run:
        return
    validate_radar_robustness_inputs(output_root)
    for command in commands:
        subprocess.run(command, cwd=ARTIFACT_ROOT, check=True)
    print(f"\nRadar robustness inputs: {output_root / 'radar_robustness'}")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    results_root = args.results_root.resolve()
    output_root = args.output_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Smoke-Eval root not found: {data_root}")
    if not results_root.is_dir():
        raise FileNotFoundError(f"Prediction root not found: {results_root}")

    if args.radar_robustness:
        run_radar_robustness(args)
        return

    assert args.model is not None
    variant = resolve_metric_variant(args.model)

    command = [
        sys.executable,
        str(EVALUATION_DIR / "metric_results" / "run_evaluation.py"),
        "--model", variant,
        "--data-root", str(data_root),
        "--results-root", str(results_root),
        "--raw-root", str(output_root),
        "--merged-output-dir", str(output_root / "merged_csv"),
        "--target-resolution", args.target_resolution,
        "--workers", str(args.workers),
    ]
    if args.sequence:
        command.extend(("--sequence", *args.sequence))
    if args.pred_scale is not None:
        command.extend(("--pred-scale", str(args.pred_scale)))
    if args.table:
        command.append("--table")

    print(f"Model:        {args.model}")
    print(f"Evaluator:    {variant}")
    print(f"Data root:    {data_root}")
    print(f"Results root: {results_root}")
    print(f"Output root:  {output_root}")
    print("Command:      " + subprocess.list2cmdline(command))
    if args.dry_run:
        return
    subprocess.run(command, cwd=ARTIFACT_ROOT, check=True)
    print(f"\nMerged CSV: {output_root / 'merged_csv' / f'{variant}.csv'}")


if __name__ == "__main__":
    main()
