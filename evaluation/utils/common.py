from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from model_registry import canonical_model_name


ROOT = Path(__file__).resolve().parent
EVALUATION_DIR = ROOT.parent
REFERENCE_RESULTS_DIR = EVALUATION_DIR / "reference_results"
# Fixed paper inputs are deliberately separate from fresh evaluator output. A
# reproduction run can override only MERGED_DIR to consume newly computed CSVs;
# it never needs to mutate the golden source material.
REFERENCE_DATA_DIR = REFERENCE_RESULTS_DIR / "pre_eval_results"
DATA_DIR = Path(os.environ.get("GRADE_REFERENCE_DATA_DIR", str(REFERENCE_DATA_DIR)))
REFERENCE_MERGED_DIR = REFERENCE_DATA_DIR / "csv"
GENERATED_MERGED_DIR = EVALUATION_DIR / "metric_results" / "merged_csv"
MERGED_DIR = Path(os.environ.get("GRADE_MERGED_DIR", str(REFERENCE_MERGED_DIR)))
OUTPUT_DIR = EVALUATION_DIR / "reproduced_results"

METRICS = ["MAE", "SSIM", "LPIPS", "CD", "MHD"]
GRADIENT_ERROR = "GradientError"
ERROR_METRICS = {"MAE", "LPIPS", "CD", "MHD", GRADIENT_ERROR}

# GradientError sits around 2.5e-3, so 3 decimals collapses the variants together.
DEFAULT_PRECISION = 3
METRIC_PRECISION = {GRADIENT_ERROR: 4}


def load_merged(name: str) -> pd.DataFrame:
    path = MERGED_DIR / f"{canonical_model_name(name)}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation CSV: {path}")
    frame = pd.read_csv(path)
    required = {"sequences", "frame_idx", "R", "IR", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def load_aligned_merged(names: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Load variants on their shared (sequence, frame) evaluation support.

    RadarCam-Depth omits frames without a valid radar point cloud.  Figures
    comparing empirical distributions must therefore use the intersection of
    frame keys rather than silently giving each method a different sample set.
    """
    names = list(names)
    if not names:
        return {}

    frames = {name: load_merged(name) for name in names}
    keys: pd.MultiIndex | None = None
    for name, frame in frames.items():
        if frame.duplicated(["sequences", "frame_idx"]).any():
            raise ValueError(f"{name} has duplicate (sequences, frame_idx) rows")
        current = pd.MultiIndex.from_frame(frame[["sequences", "frame_idx"]])
        keys = current if keys is None else keys.intersection(current, sort=False)

    if keys is None:
        raise RuntimeError("No frame keys were available to align merged results.")
    key_frame = keys.to_frame(index=False)
    key_frame.columns = ["sequences", "frame_idx"]
    aligned = {
        name: key_frame.merge(
            frame,
            on=["sequences", "frame_idx"],
            how="left",
            validate="one_to_one",
        )
        for name, frame in frames.items()
    }
    return aligned


def clear_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["sequences"].astype(str).str.endswith("-0")


def overall_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Paper split: clear sequences and smoke frames with IR >= 2000."""
    is_clear = clear_mask(frame)
    return {
        "Clear": frame.loc[is_clear].copy(),
        "Smoke": frame.loc[(~is_clear) & (frame["IR"] >= 2000)].copy(),
    }


def add_smoke_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Paper density split using the MAX30105 infrared reading.

    Light smoke has IR < 2000, medium smoke has 2000 <= IR < 4000, and heavy
    smoke has IR >= 4000.
    """
    frame = frame.copy()
    is_clear = clear_mask(frame)
    readings = frame["IR"].to_numpy(dtype=float)
    smoke_class = np.select(
        [readings < 2000, (readings >= 2000) & (readings < 4000), readings >= 4000],
        ["light", "medium", "heavy"],
        default="unclassified",
    )
    frame["Smoke class"] = np.where(
        is_clear,
        "clear",
        smoke_class,
    )
    return frame


def metric_medians(frame: pd.DataFrame, metrics: list[str] | None = None) -> dict[str, float]:
    return {metric: float(frame[metric].median()) for metric in metrics or METRICS}


def format_metrics(values: dict[str, float], metrics: list[str] | None = None) -> list[str]:
    return [
        f"{values[metric]:.{METRIC_PRECISION.get(metric, DEFAULT_PRECISION)}f}"
        for metric in metrics or METRICS
    ]


def print_metric_table(
    title: str,
    groups: dict[str, list[tuple[str, list[str]]]],
    label_header: str = "Method",
    metrics: list[str] | None = None,
    headers: list[str] | None = None,
    highlight: bool = False,
) -> None:
    """Print grouped metric rows as an aligned plain-text table.

    When ``highlight`` is true, ``[1]``/``[2]`` marks the best and second-best
    displayed value in each metric column. Ties share a rank.
    """
    metrics = metrics or METRICS
    headers = headers or metrics
    if len(headers) != len(metrics):
        raise ValueError("headers and metrics must have the same length")
    ranks_by_group = {
        group: compute_ranks(dict(rows), metrics) if highlight else {}
        for group, rows in groups.items()
    }

    def render(group: str, label: str, values: list[str]) -> list[str]:
        ranks = ranks_by_group[group].get(label, [0] * len(values))
        suffix = {1: "[1]", 2: "[2]"}
        return [f"{value}{suffix.get(rank, '')}" for value, rank in zip(values, ranks)]

    labels = [label for rows in groups.values() for label, _ in rows]
    width = max(len(label_header), *(len(label) for label in labels))
    rendered = {
        group: [(label, render(group, label, values)) for label, values in rows]
        for group, rows in groups.items()
    }
    columns = [
        max(
            7,
            len(header),
            *(
                len(values[index])
                for rows in rendered.values()
                for _, values in rows
            ),
        )
        for index, header in enumerate(headers)
    ]
    header = f"{label_header:<{width}}  " + "  ".join(
        f"{name:>{size}}" for name, size in zip(headers, columns)
    )
    print(f"\n{title}")
    print("=" * len(header))
    for group, rows in rendered.items():
        print(f"\n{group}")
        print("-" * len(header))
        print(header)
        for label, values in rows:
            cells = "  ".join(f"{value:>{size}}" for value, size in zip(values, columns))
            print(f"{label:<{width}}  {cells}")
    if highlight:
        print("\n[1] best; [2] second-best (ties share rank at displayed precision)")
    print()


def compute_ranks(
    rows: dict[str, list[str]], metrics: list[str] | None = None
) -> dict[str, list[int]]:
    """Rank 1 = best, 2 = second best, per metric column, at displayed precision.

    Ties share a rank, so two methods printing the same rounded value are both
    highlighted rather than one arbitrarily winning.
    """
    metrics = metrics or METRICS
    ranks = {label: [0] * len(metrics) for label in rows}
    for index, metric in enumerate(metrics):
        column = {label: float(values[index]) for label, values in rows.items()}
        ordered = sorted(set(column.values()), reverse=metric not in ERROR_METRICS)
        for label, value in column.items():
            position = ordered.index(value)
            if position < 2:
                ranks[label][index] = position + 1
    return ranks


def tex_cell(value: str, rank: int = 0) -> str:
    if rank == 1:
        return rf"\cellcolor{{blue!18}}\textbf{{{value}}}"
    if rank == 2:
        return rf"\cellcolor{{blue!8}}{value}"
    return value


def tex_row(label: str, values: Iterable[str], ranks: Iterable[int] | None = None) -> str:
    values = list(values)
    ranks = list(ranks) if ranks is not None else [0] * len(values)
    cells = [tex_cell(value, rank) for value, rank in zip(values, ranks)]
    return f"{label} & " + " & ".join(cells) + r" \\"


def write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"Saved {path}")


def normalize_canvas(path: Path, target_width: int, target_height: int) -> None:
    """Center-crop/pad tight-bbox output to the camera-ready raster dimensions."""
    with Image.open(path) as source:
        source = source.convert("RGBA")
        left = max(0, (source.width - target_width) // 2)
        top = max(0, (source.height - target_height) // 2)
        cropped = source.crop(
            (
                left,
                top,
                min(source.width, left + target_width),
                min(source.height, top + target_height),
            )
        )
        canvas = Image.new("RGBA", (target_width, target_height), "white")
        canvas.paste(
            cropped,
            ((target_width - cropped.width) // 2, (target_height - cropped.height) // 2),
        )
        canvas.save(path)
