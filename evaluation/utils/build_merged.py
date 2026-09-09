"""Regenerate merged_csv/{variant}.csv from raw per-sequence eval CSVs.

Replaces the analysis project's `merge_table.py`. Two differences from that
script: variants keep the folder names used under `simple_eval_results*`
(no paper-name aliasing), and RMS_contrast is read from the raw 2D CSV
instead of a separate `csv_rms` folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.common import GENERATED_MERGED_DIR, GRADIENT_ERROR, METRICS


RAW_ROOT = Path(__file__).resolve().parent
SIMPLE_2D = "simple_eval_results"
SIMPLE_3D = "simple_eval_results_3d"
PREFIX_2D = "csv_"
PREFIX_3D = "3d_csv_"

OUT_COLUMNS = ["sequences", "frame_idx", "R", "IR", "RMS_contrast", *METRICS, GRADIENT_ERROR]


def discover_variants(raw_root: Path) -> list[str]:
    """Variant folders present under both the 2D and 3D raw result trees."""
    dir_2d = raw_root / SIMPLE_2D
    dir_3d = raw_root / SIMPLE_3D
    if not dir_2d.is_dir() or not dir_3d.is_dir():
        raise FileNotFoundError(f"Expected {SIMPLE_2D}/ and {SIMPLE_3D}/ under {raw_root}")
    variants = []
    for path in sorted(dir_2d.iterdir()):
        if not path.is_dir() or not path.name.startswith(PREFIX_2D):
            continue
        variant = path.name[len(PREFIX_2D) :]
        if (dir_3d / f"{PREFIX_3D}{variant}").is_dir():
            variants.append(variant)
    return variants


def merge_variant(raw_root: Path, variant: str) -> pd.DataFrame:
    """Join per-frame 2D metrics with their 3D counterparts for one variant."""
    dir_2d = raw_root / SIMPLE_2D / f"{PREFIX_2D}{variant}"
    dir_3d = raw_root / SIMPLE_3D / f"{PREFIX_3D}{variant}"
    blocks = []
    for path_2d in sorted(
        path for path in dir_2d.glob("*.csv") if not path.name.startswith("._")
    ):
        path_3d = dir_3d / path_2d.name
        if not path_3d.is_file():
            raise FileNotFoundError(f"No 3D counterpart for {variant}/{path_2d.name}")
        frame_2d = pd.read_csv(path_2d).sort_values("Frame_Index")
        frame_3d = pd.read_csv(path_3d).sort_values("Frame_Index")
        required = {
            "Sequence",
            "Frame_Index",
            "R",
            "IR",
            "RMS_contrast",
            "MAE",
            "SSIM",
            "LPIPS",
            GRADIENT_ERROR,
        }
        missing = required.difference(frame_2d.columns)
        if missing:
            raise ValueError(f"{path_2d} is missing columns: {sorted(missing)}")
        merged = frame_2d.merge(
            frame_3d[["Sequence", "Frame_Index", "CD", "MHD"]],
            on=["Sequence", "Frame_Index"],
            how="inner",
        )
        blocks.append(
            pd.DataFrame(
                {
                    "sequences": merged["Sequence"],
                    "frame_idx": merged["Frame_Index"],
                    "R": merged["R"],
                    "IR": merged["IR"],
                    "RMS_contrast": merged["RMS_contrast"],
                    "MAE": merged["MAE"],
                    "SSIM": merged["SSIM"],
                    "LPIPS": merged["LPIPS"],
                    "CD": merged["CD"],
                    "MHD": merged["MHD"],
                    GRADIENT_ERROR: merged[GRADIENT_ERROR],
                }
            )
        )
    if not blocks:
        raise FileNotFoundError(f"No sequence CSVs found for variant {variant}")
    return pd.concat(blocks, ignore_index=True)[OUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild merged per-variant evaluation CSVs.")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=GENERATED_MERGED_DIR)
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()

    variants = args.variants or discover_variants(args.raw_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        table = merge_variant(args.raw_root, variant)
        path = args.output_dir / f"{variant}.csv"
        table.to_csv(path, index=False)
        print(f"Saved {len(table):>6} rows, {table['sequences'].nunique():>2} sequences -> {path.name}")


if __name__ == "__main__":
    main()
