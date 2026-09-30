"""Run 2D metrics, then 3D metrics, then merge their per-frame CSVs.

The wrapper uses the same model, sequence, data, prediction, and resolution
arguments for both evaluators. A failed stage stops the pipeline, so merged
tables are never rebuilt from a partially completed evaluation.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = EVALUATION_DIR.parent
DEFAULT_DATA_ROOT = ARTIFACT_ROOT / "evaluation_dataset" / "Smoke-Eval"
DEFAULT_RESULTS_ROOT = EVALUATION_DIR / "outputs" / "inference"
DEFAULT_RAW_ROOT = EVALUATION_DIR / "metric_results"
DEFAULT_MERGED_DIR = DEFAULT_RAW_ROOT / "merged_csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 2D and 3D evaluation sequentially, then merge their CSV outputs."
    )
    parser.add_argument("--model", nargs="+", default=None, metavar="MODEL")
    parser.add_argument("--sequence", nargs="+", default=None, metavar="SEQ")
    parser.add_argument(
        "--data-root", "--data_root", dest="data_root", type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--results-root", "--results_root", dest="results_root", type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--raw-root", type=Path, default=DEFAULT_RAW_ROOT,
        help="Root that will contain simple_eval_results/ and simple_eval_results_3d/.",
    )
    parser.add_argument(
        "--merged-output-dir", type=Path, default=DEFAULT_MERGED_DIR,
        help="Destination for one merged CSV per model.",
    )
    parser.add_argument("--target-resolution", default="288x512", metavar="HxW")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pred-scale", type=float, default=None)
    parser.add_argument(
        "--table", action="store_true",
        help="Rebuild evaluator summary tables from existing per-sequence CSVs before merging.",
    )
    return parser.parse_args()


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=SCRIPT_DIR, env=env, check=True)


def common_evaluator_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--data_root", str(args.data_root),
        "--results_root", str(args.results_root),
        "--target-resolution", args.target_resolution,
    ]
    if args.model:
        command.extend(["--model", *args.model])
    if args.sequence:
        command.extend(["--sequence", *args.sequence])
    if args.table:
        command.append("--table")
    return command


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_2d = raw_root / "simple_eval_results"
    output_3d = raw_root / "simple_eval_results_3d"

    common = common_evaluator_args(args)
    simple_command = [
        sys.executable,
        str(SCRIPT_DIR / "eval_compute_simple.py"),
        *common,
        "--output_dir", str(output_2d),
    ]
    if args.pred_scale is not None:
        simple_command.extend(["--pred-scale", str(args.pred_scale)])
    run_checked(simple_command)

    run_checked([
        sys.executable,
        str(SCRIPT_DIR / "eval_compute_3d.py"),
        *common,
        "--output_dir", str(output_3d),
        "--workers", str(args.workers),
    ])

    merge_env = os.environ.copy()
    existing_pythonpath = merge_env.get("PYTHONPATH")
    merge_env["PYTHONPATH"] = (
        f"{EVALUATION_DIR}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath else str(EVALUATION_DIR)
    )
    merge_command = [
        sys.executable,
        "-m", "utils.build_merged",
        "--raw-root", str(raw_root),
        "--output-dir", str(args.merged_output_dir.resolve()),
    ]
    if args.model:
        merge_command.extend(["--variants", *args.model])
    run_checked(merge_command, env=merge_env)


if __name__ == "__main__":
    main()
