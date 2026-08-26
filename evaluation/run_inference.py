#!/usr/bin/env python3
"""Run one released GRADE-family model on Smoke-Eval without touching archives.

The command creates a small, run-specific YAML configuration under the chosen
output root. It points the released entry point at the requested Smoke-Eval
directory and writes predictions outside ``inference_results/``, which is the
immutable cluster-inference reference used for paper-result reproduction.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from model_registry import (
    EVALUATION_VARIANTS,
    INFERENCE_MODELS,
    InferenceModel,
    artifact_path,
    canonical_model_name,
)


EVALUATION_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = EVALUATION_DIR.parent
DEFAULT_SMOKE_EVAL_ROOT = ARTIFACT_ROOT / "evaluation_dataset" / "Smoke-Eval"
DEFAULT_RADARCAM_DEPTH_ROOT = (
    ARTIFACT_ROOT / "evaluation_dataset" / "Smoke-Eval-RadarCam-Depth"
)
DEFAULT_OUTPUT_ROOT = EVALUATION_DIR / "outputs" / "inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True,
        choices=sorted({*INFERENCE_MODELS, *EVALUATION_VARIANTS}),
        help="Canonical model identifier; historical aliases are accepted for compatibility.",
    )
    parser.add_argument(
        "--data-root", type=Path,
        help=(
            "Evaluation-data root. Defaults to evaluation_dataset/Smoke-Eval, "
            "or Smoke-Eval-RadarCam-Depth for radarcam-depth."
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Prediction root. The evaluator-compatible directory is created "
            "under this root; the archived inference_results/ tree is never "
            "used as a default output."
        ),
    )
    parser.add_argument(
        "--gpuid", nargs="+", type=int, default=[0], metavar="GPU",
        help="Physical GPU IDs exposed to Accelerate (default: 0).",
    )
    parser.add_argument(
        "--launcher", choices=("accelerate", "wrapper"), default="accelerate",
        help=(
            "accelerate (default) runs the model backend once with `accelerate launch`; "
            "wrapper preserves the legacy model-entry-point launcher."
        ),
    )
    parser.add_argument(
        "--backend-args", nargs=argparse.REMAINDER, default=[],
        help="Arguments after this flag are forwarded unchanged to the model backend.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_path(mapping: dict[str, Any], keys: tuple[str, ...], value: str) -> None:
    current = mapping
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def make_checkpoint_paths_absolute(config: dict[str, Any], config_source: Path) -> None:
    """Keep released relative checkpoint paths valid after copying the YAML."""

    sections: list[dict[str, Any]] = [config]
    for section in ("pretrained", "inference"):
        values = config.get(section)
        if isinstance(values, dict):
            sections.append(values)
    for values in sections:
        for key, value in list(values.items()):
            is_checkpoint = "checkpoint" in key or key in {
                "radar_model", "unet", "controlnet", "grt_model"
            }
            if not is_checkpoint or not isinstance(value, str):
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                values[key] = str((config_source.parent / candidate).resolve())


def build_runtime_config(
    spec: InferenceModel,
    data_root: Path,
    run_root: Path,
) -> tuple[Path, dict[str, Any]]:
    source = artifact_path(ARTIFACT_ROOT, spec.config)
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {source}")
    make_checkpoint_paths_absolute(config, source)
    if spec.data_key:
        set_path(config, spec.data_key, str(data_root.resolve()))
    if spec.output_key:
        set_path(config, spec.output_key, str(run_root.resolve()))
    return source, config


def accelerate_launch_command(
    backend: Path,
    gpu_ids: list[int],
    backend_args: list[str],
    mixed_precision: str = "fp16",
) -> list[str]:
    """Build a direct, single-level ``accelerate launch`` command.

    The public model wrappers already launch Accelerate themselves.  Calling a
    wrapper from another launcher would create nested distributed jobs, so this
    path invokes the released model backend directly instead.
    """

    scripts_dir = Path(sys.executable).resolve().parent / "Scripts"
    candidate_names = ("accelerate.exe", "accelerate") if os.name == "nt" else ("accelerate",)
    accelerator = next((scripts_dir / name for name in candidate_names if (scripts_dir / name).is_file()), None)
    launcher = [str(accelerator), "launch"] if accelerator else [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
    ]
    command = [
        *launcher,
        "--num_processes", str(len(gpu_ids)),
        "--num_machines", "1",
        "--mixed_precision", mixed_precision,
        "--dynamo_backend", "no",
    ]
    if len(gpu_ids) > 1:
        command.append("--multi_gpu")
    command.extend((str(backend), *backend_args))
    return command


def render_backend_args(
    spec: InferenceModel,
    config_path: Path,
    data_root: Path,
    run_root: Path,
) -> list[str]:
    """Expand registry placeholders for a direct backend invocation."""

    values = {
        "config": str(config_path),
        "data_root": str(data_root.resolve()),
        "run_root": str(run_root.resolve()),
        "artifact_root": str(ARTIFACT_ROOT.resolve()),
    }
    return [argument.format(**values) for argument in spec.backend_args]


def main() -> None:
    args = parse_args()
    model_name = canonical_model_name(args.model)
    if model_name not in INFERENCE_MODELS:
        raise ValueError(f"No released inference backend for model {args.model!r}")
    spec = INFERENCE_MODELS[model_name]
    default_data_root = (
        DEFAULT_RADARCAM_DEPTH_ROOT
        if model_name == "radarcam-depth"
        else DEFAULT_SMOKE_EVAL_ROOT
    )
    data_root = (args.data_root or default_data_root).resolve()
    output_root = args.output_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Smoke-Eval root not found: {data_root}")
    if any(gpu < 0 for gpu in args.gpuid) or len(set(args.gpuid)) != len(args.gpuid):
        raise ValueError("--gpuid values must be distinct non-negative integers")

    destination = output_root / spec.prediction_directory
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing predictions: {destination}. "
            "Choose a new --output-root or move the previous result first."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_root / ".runs" / f"{model_name}-{stamp}"
    config_source, config = build_runtime_config(spec, data_root, run_root)
    config_path = run_root / f"{model_name}.yaml"
    if args.launcher == "accelerate":
        backend = artifact_path(ARTIFACT_ROOT, spec.accelerate_backend)
        backend_args = render_backend_args(spec, config_path, data_root, run_root)
        mixed_precision = "fp16"
        if model_name in {"cafnet", "cafnet_no_smoke"}:
            mixed_precision = str(config.get("mixed_precision", "no"))
        elif model_name == "radarcam-depth":
            mixed_precision = str(config.get("runtime", {}).get("mixed_precision", "bf16"))
        command = accelerate_launch_command(
            backend, args.gpuid, [*backend_args, *args.backend_args], mixed_precision
        )
    else:
        entrypoint = artifact_path(ARTIFACT_ROOT, spec.entrypoint)
        command = [
            sys.executable,
            str(entrypoint),
            "--gpuid",
            *(str(gpu) for gpu in args.gpuid),
            "--config",
            str(config_path),
            *args.backend_args,
        ]

    print(f"Model:       {model_name}")
    print(f"Evaluator:   {spec.metric_variant}")
    print(f"Smoke-Eval:  {data_root}")
    print(f"Output root: {output_root}")
    print(f"Launcher:    {args.launcher}")
    print("Command:     " + subprocess.list2cmdline(command))
    if args.dry_run:
        print("\nDry run: runtime YAML was not written.")
        print(yaml.safe_dump(config, sort_keys=False))
        return

    run_root.mkdir(parents=True, exist_ok=False)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    environment = os.environ.copy()
    # Released checkpoints and the small diffusion runtime assets are bundled
    # with this artifact.  Keep a review run from silently consulting or
    # downloading a changing Hugging Face cache.
    environment["HF_HUB_OFFLINE"] = "1"
    environment["PYTHONUTF8"] = "1"
    if args.launcher == "accelerate":
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpuid))
    subprocess.run(command, cwd=ARTIFACT_ROOT, env=environment, check=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if spec.produced_directory == ".":
        # The no-Doppler backend writes directly into the run root. Keep its
        # generated YAML under .runs/ and move only the standardized arrays.
        prediction_paths = sorted(run_root.glob("*_pred.npy"))
        if not prediction_paths:
            raise FileNotFoundError(
                f"Inference completed but did not create any *_pred.npy files: {run_root}"
            )
        destination.mkdir()
        for prediction_path in prediction_paths:
            shutil.move(str(prediction_path), str(destination / prediction_path.name))
    else:
        produced = run_root / spec.produced_directory
        if not produced.is_dir():
            raise FileNotFoundError(
                f"Inference completed but did not create the expected prediction directory: {produced}"
            )
        shutil.move(str(produced), str(destination))

    # Stage backends create empty sibling output folders. They are not part of
    # the public output contract; retain only the run-specific YAML in .runs/.
    for child in run_root.iterdir():
        if child.is_dir():
            try:
                child.rmdir()
            except OSError:
                pass
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "metric_variant": spec.metric_variant,
        "prediction_directory": spec.prediction_directory,
        "data_root": str(data_root),
        "prediction_dir": str(destination),
        "runtime_config": str(config_path),
        "source_config": str(config_source),
        "gpus": args.gpuid,
        "launcher": args.launcher,
        "command": command,
    }
    manifest_path = output_root / f"{spec.prediction_directory}.run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nPredictions: {destination}")
    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
