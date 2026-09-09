#!/usr/bin/env python3
"""Run deterministic local-inference smoke checks against archived predictions.

The script stages a small subset of every evaluation sequence, runs each
released model once through ``run_inference.py``, and compares the new arrays
with the corresponding source-frame slices from ``inference_results/``.  The
archived predictions and packaged datasets are always read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from model_registry import INFERENCE_MODELS


EVALUATION_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = EVALUATION_DIR.parent
DEFAULT_DATA_ROOT = ARTIFACT_ROOT / "evaluation_dataset" / "Smoke-Eval"
DEFAULT_RADARCAM_ROOT = (
    ARTIFACT_ROOT / "evaluation_dataset" / "Smoke-Eval-RadarCam-Depth"
)
DEFAULT_REFERENCE_ROOT = ARTIFACT_ROOT / "inference_results"
DEFAULT_OUTPUT_ROOT = EVALUATION_DIR / "outputs" / "smoke_tests" / "inference"
STANDARD_ARRAYS = (
    "dji_rgb.npy",
    "zed_depth.npy",
    "zed_rgb.npy",
    "radar.npy",
    "radar_no_doppler.npy",
    "max30105.npy",
)
RADARCAM_MODEL = "radarcam-depth"
DEFAULT_MODELS = tuple(name for name in sorted(INFERENCE_MODELS) if name != "grade")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=DEFAULT_MODELS, default=list(DEFAULT_MODELS),
        help="Distinct released models to check (default: all 15).",
    )
    parser.add_argument("--frames-per-sequence", type=int, default=5)
    parser.add_argument(
        "--frame-selection-seed", type=int, default=20260826,
        help="Stable seed used solely to select source frames.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--radarcam-root", type=Path, default=DEFAULT_RADARCAM_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--keep-staged-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sequence_seed(seed: int, sequence: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sequence}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def choose_frames(length: int, frames_per_sequence: int, seed: int, sequence: str) -> list[int]:
    if frames_per_sequence < 1:
        raise ValueError("--frames-per-sequence must be positive.")
    if length < frames_per_sequence:
        raise ValueError(
            f"{sequence} has only {length} frames; cannot choose {frames_per_sequence}."
        )
    return sorted(
        int(value)
        for value in np.random.default_rng(sequence_seed(seed, sequence)).choice(
            length, size=frames_per_sequence, replace=False
        )
    )


def copy_array_subset(source: Path, destination: Path, frame_indices: list[int]) -> None:
    array = np.load(source, mmap_mode="r")
    if array.ndim < 1 or array.shape[0] <= max(frame_indices):
        raise ValueError(f"Cannot select {frame_indices[-1]} from {source} with shape {array.shape}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, np.asarray(array[frame_indices]))


def stage_standard_data(
    source_root: Path, destination_root: Path, frames_per_sequence: int, seed: int
) -> dict[str, list[int]]:
    selections: dict[str, list[int]] = {}
    for source_sequence in sorted(path for path in source_root.iterdir() if path.is_dir()):
        required = source_sequence / "dji_rgb.npy"
        if not required.is_file():
            raise FileNotFoundError(f"Missing required image array: {required}")
        frame_indices = choose_frames(
            int(np.load(required, mmap_mode="r").shape[0]),
            frames_per_sequence,
            seed,
            source_sequence.name,
        )
        selections[source_sequence.name] = frame_indices
        staged_sequence = destination_root / source_sequence.name
        for name in STANDARD_ARRAYS:
            source = source_sequence / name
            if source.is_file():
                copy_array_subset(source, staged_sequence / name, frame_indices)
        source_pcd = source_sequence / "pcd"
        if source_pcd.is_dir():
            (staged_sequence / "pcd").mkdir(parents=True, exist_ok=True)
            for staged_index, source_index in enumerate(frame_indices):
                source = source_pcd / f"pcd_{source_index}.npy"
                if source.is_file():
                    destination = staged_sequence / "pcd" / f"pcd_{staged_index}.npy"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    if not selections:
        raise ValueError(f"No sequence directories found in {source_root}")
    return selections


def load_radarcam_manifest(source_root: Path) -> dict[str, list[str]]:
    manifest = source_root / "data" / "full.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"RadarCam-Depth manifest not found: {manifest}")
    grouped: dict[str, list[str]] = {}
    for name in manifest.read_text(encoding="utf-8").splitlines():
        name = name.strip()
        if not name:
            continue
        sequence, _, frame = name.rpartition("_")
        if not sequence or not frame.isdigit():
            raise ValueError(f"Invalid frame name in {manifest}: {name!r}")
        grouped.setdefault(sequence, []).append(name)
    return {sequence: sorted(names, key=lambda name: int(name.rsplit("_", 1)[1])) for sequence, names in grouped.items()}


def stage_radarcam_data(
    source_root: Path, destination_root: Path, frames_per_sequence: int, seed: int
) -> dict[str, list[int]]:
    selections: dict[str, list[int]] = {}
    grouped = load_radarcam_manifest(source_root)
    image_root = source_root / "data" / "image"
    radar_root = source_root / "data" / "radar"
    mono_root = source_root / "result" / "global_aligned_mono" / "dpt_hybrid_ls"
    staged_names: list[str] = []
    for sequence, names in sorted(grouped.items()):
        selected_positions = choose_frames(len(names), frames_per_sequence, seed, sequence)
        selections[sequence] = selected_positions
        for position in selected_positions:
            name = names[position]
            for source, destination in (
                (image_root / f"{name}.png", destination_root / "data" / "image" / f"{name}.png"),
                (radar_root / f"{name}.npy", destination_root / "data" / "radar" / f"{name}.npy"),
                (mono_root / f"{name}.png", destination_root / "result" / "global_aligned_mono" / "dpt_hybrid_ls" / f"{name}.png"),
            ):
                if not source.is_file():
                    raise FileNotFoundError(f"Missing prepared RadarCam-Depth input: {source}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            staged_names.append(name)
    (destination_root / "data").mkdir(parents=True, exist_ok=True)
    (destination_root / "data" / "full.txt").write_text(
        "\n".join(staged_names) + "\n", encoding="utf-8"
    )
    return selections


def compare_model(
    model: str,
    prediction_root: Path,
    reference_root: Path,
    selections: dict[str, list[int]],
) -> dict[str, Any]:
    spec = INFERENCE_MODELS[model]
    prediction_dir = prediction_root / spec.prediction_directory
    reference_dir = reference_root / spec.prediction_directory
    sequence_results: dict[str, dict[str, Any]] = {}
    total_abs_error = 0.0
    total_values = 0
    for sequence, frame_indices in sorted(selections.items()):
        filename = f"{sequence.lower()}_pred.npy"
        prediction_path = prediction_dir / filename
        reference_path = reference_dir / filename
        if not prediction_path.is_file() or not reference_path.is_file():
            raise FileNotFoundError(
                f"Missing prediction/reference pair for {model}/{sequence}: "
                f"{prediction_path}, {reference_path}"
            )
        prediction = np.load(prediction_path, mmap_mode="r")
        reference = np.load(reference_path, mmap_mode="r")
        if prediction.shape[0] != len(frame_indices):
            raise ValueError(
                f"{model}/{sequence}: got {prediction.shape[0]} new frames; "
                f"expected {len(frame_indices)}."
            )
        if reference.shape[0] <= max(frame_indices):
            raise ValueError(f"{model}/{sequence}: reference lacks selected frame indices.")
        selected_reference = np.asarray(reference[frame_indices])
        if prediction.shape != selected_reference.shape:
            raise ValueError(
                f"{model}/{sequence}: shape mismatch new={prediction.shape}, "
                f"reference={selected_reference.shape}."
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"{model}/{sequence}: local prediction contains NaN or Inf.")
        if not np.isfinite(selected_reference).all():
            raise ValueError(f"{model}/{sequence}: reference prediction contains NaN or Inf.")
        difference = np.asarray(prediction, dtype=np.float32) - selected_reference.astype(np.float32)
        absolute = np.abs(difference)
        sequence_results[sequence] = {
            "source_frame_indices": frame_indices,
            "shape": list(prediction.shape),
            "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "max_abs_error": float(absolute.max()),
        }
        total_abs_error += float(absolute.sum(dtype=np.float64))
        total_values += int(absolute.size)
    sequence_maes = [entry["mae"] for entry in sequence_results.values()]
    return {
        "model": model,
        "metric_variant": spec.metric_variant,
        "sequences": len(sequence_results),
        "frames_per_sequence": len(next(iter(selections.values()))),
        "overall_mae": total_abs_error / total_values,
        "mean_sequence_mae": float(np.mean(sequence_maes)),
        "max_sequence_mae": float(np.max(sequence_maes)),
        "sequence_results": sequence_results,
    }


def run_model(model: str, data_root: Path, output_root: Path, gpu: int, log_path: Path) -> None:
    command = [
        sys.executable,
        str(EVALUATION_DIR / "run_inference.py"),
        "--model", model,
        "--data-root", str(data_root),
        "--output-root", str(output_root),
        "--gpuid", str(gpu),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + subprocess.list2cmdline(command) + "\n\n")
        subprocess.run(
            command,
            cwd=ARTIFACT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def compact_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "model", "metric_variant", "sequences", "frames_per_sequence",
            "overall_mae", "mean_sequence_mae", "max_sequence_mae",
        )
    }


def main() -> None:
    args = parse_args()
    models = list(dict.fromkeys(args.models))
    if args.frames_per_sequence < 1:
        raise ValueError("--frames-per-sequence must be positive.")
    for path, label in (
        (args.data_root, "Smoke-Eval"),
        (args.radarcam_root, "Smoke-Eval-RadarCam-Depth"),
        (args.reference_root, "archived inference results"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} not found: {path}")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite prior smoke-test output: {output_root}")
    standard_stage = output_root / "staged" / "smoke_eval"
    radarcam_stage = output_root / "staged" / "radarcam_depth"
    predictions = output_root / "predictions"
    logs = output_root / "logs"
    if args.dry_run:
        print(f"Would run {len(models)} models into: {output_root}")
        print(f"Standard dataset: {args.data_root.resolve()}")
        print(f"RadarCam-Depth dataset: {args.radarcam_root.resolve()}")
        print(f"Frames per sequence: {args.frames_per_sequence}")
        return
    standard_selections = stage_standard_data(
        args.data_root.resolve(), standard_stage, args.frames_per_sequence, args.frame_selection_seed
    )
    radarcam_selections = stage_radarcam_data(
        args.radarcam_root.resolve(), radarcam_stage, args.frames_per_sequence, args.frame_selection_seed
    )
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frame_selection_seed": args.frame_selection_seed,
        "ddim_sampling_seed": 42,
        "frames_per_sequence": args.frames_per_sequence,
        "models": {},
    }
    for model in models:
        model_stage = radarcam_stage if model == RADARCAM_MODEL else standard_stage
        try:
            print(f"\n[{model}] starting", flush=True)
            run_model(model, model_stage, predictions, args.gpu, logs / f"{model}.log")
            result = compare_model(
                model,
                predictions,
                args.reference_root.resolve(),
                radarcam_selections if model == RADARCAM_MODEL else standard_selections,
            )
            report["models"][model] = {"status": "passed", **result}
            print(
                f"[{model}] MAE={result['overall_mae']:.8f}; "
                f"max per-sequence MAE={result['max_sequence_mae']:.8f}",
                flush=True,
            )
        except Exception as exc:  # Keep later models testable after one failure.
            report["models"][model] = {"status": "failed", "error": repr(exc)}
            print(f"[{model}] FAILED: {exc}", flush=True)
    report["summary"] = [
        compact_row(value)
        for value in report["models"].values()
        if value.get("status") == "passed"
    ]
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport: {report_path}")
    if not args.keep_staged_data:
        shutil.rmtree(output_root / "staged")
        print("Removed staged inputs; predictions, logs, and report remain.")
    if any(value.get("status") != "passed" for value in report["models"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
