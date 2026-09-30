"""Shared launcher for the model-specific inference entry points."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


def _has_option(arguments: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in arguments)


def _merge_defaults(arguments: Sequence[str], defaults: Iterable[str]) -> list[str]:
    merged = list(arguments)
    pairs = list(defaults)
    if len(pairs) % 2:
        raise ValueError("Default inference arguments must be option/value pairs")
    for index in range(0, len(pairs), 2):
        option, value = pairs[index], pairs[index + 1]
        if not _has_option(merged, option):
            merged.extend((option, value))
    return merged


def build_command(
    backend: Path,
    gpu_ids: Sequence[int],
    arguments: Sequence[str],
) -> list[str]:
    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("--gpuid requires one or more non-negative GPU IDs")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpuid values must be unique")
    if not backend.is_file():
        raise FileNotFoundError(f"Inference backend not found: {backend}")

    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(len(gpu_ids)),
        "--num_machines",
        "1",
        "--mixed_precision",
        "fp16",
        "--dynamo_backend",
        "no",
    ]
    if len(gpu_ids) > 1:
        command.append("--multi_gpu")
    command.extend((str(backend), *arguments))
    return command


def launch(backend: Path, defaults: Iterable[str] = ()) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch this model with Hugging Face Accelerate FP16. "
            "Unrecognized arguments are forwarded to the model backend."
        )
    )
    parser.add_argument(
        "--gpuid",
        nargs="+",
        type=int,
        default=[0],
        help="Physical GPU IDs. Default: 0; example: --gpuid 0 2",
    )
    runtime, backend_args = parser.parse_known_args()
    backend = backend.resolve()
    backend_args = _merge_defaults(backend_args, defaults)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, runtime.gpuid))
    command = build_command(backend, runtime.gpuid, backend_args)
    completed = subprocess.run(
        command,
        cwd=backend.parent,
        env=environment,
        check=False,
    )
    raise SystemExit(completed.returncode)
