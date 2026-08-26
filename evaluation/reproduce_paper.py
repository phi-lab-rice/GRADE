#!/usr/bin/env python3
"""Reproduce all quantitative evaluation tables and figures from merged CSVs.

This is the single entry point for the artifact's quantitative evaluation.  It
contains the prior modular implementation directly, exposes
one function per numbered paper artifact, and keeps reference results strictly
read-only. Figure 10 is intentionally excluded because it is qualitative-only.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


EVALUATION_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = EVALUATION_DIR.parent
METRIC_DIR = EVALUATION_DIR / "metric_results"
REFERENCE_ROOT = EVALUATION_DIR / "reference_results"
DEFAULT_OUTPUT_ROOT = EVALUATION_DIR / "reproduced_results"
DEFAULT_INPUT_MERGED_DIR = METRIC_DIR / "merged_csv"


@dataclass(frozen=True)
class ReproductionContext:
    """Resolved locations for one quantitative reproduction run."""

    input_merged_dir: Path
    reference_root: Path
    output_root: Path

    @property
    def table_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def figure_dir(self) -> Path:
        return self.output_root / "figures"


def prepare_reproduction(
    input_merged_dir: Path = DEFAULT_INPUT_MERGED_DIR,
    reference_root: Path = REFERENCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> ReproductionContext:
    """Select fresh metric CSVs and read-only reference support data.

    The selected merged CSV directory is the only source for newly reproduced
    quantitative results. ``reference_root`` supplies immutable support data
    and the side-by-side comparison assets; it is never an output destination.
    """

    context = ReproductionContext(
        input_merged_dir=input_merged_dir.resolve(),
        reference_root=reference_root.resolve(),
        output_root=output_root.resolve(),
    )
    os.environ["GRADE_MERGED_DIR"] = str(context.input_merged_dir)
    os.environ["GRADE_REFERENCE_DATA_DIR"] = str(context.reference_root / "pre_eval_results")
    required_radar_inputs = (
        "radar_sparsity_cuts_9_13.csv",
        "range_hist_20cm_ours_radar.npz",
        "range_hist_20cm_ours_diffusion.npz",
        "range_hist_20cm_ours_full.npz",
    )
    fresh_radar_dir = context.input_merged_dir.parent / "radar_robustness"
    reference_radar_dir = context.reference_root / "pre_eval_results" / "radar_robustness"
    radar_data_dir = (
        fresh_radar_dir
        if all((fresh_radar_dir / name).is_file() for name in required_radar_inputs)
        else reference_radar_dir
    )
    os.environ["GRADE_RADAR_ROBUSTNESS_DIR"] = str(radar_data_dir)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(context.output_root / ".mplconfig")
    common = sys.modules.get("metric_results.common")
    if common is not None:
        importlib.reload(common)
    return context


@contextlib.contextmanager
def _temporary_argv(arguments: list[str]):
    previous = sys.argv
    sys.argv = arguments
    try:
        yield
    finally:
        sys.argv = previous


def _invoke(module_name: str, arguments: list[str]) -> Any:
    """Run an embedded former script with an isolated command-line argument list."""

    namespace = _EMBEDDED_BUILDERS[module_name]()
    with _temporary_argv([f"{module_name}.py", *arguments]):
        return namespace["main"]()


def _write_table(context: ReproductionContext, module_name: str, output_name: str) -> Path:
    """Run one table producer and save its textual report beside reproduced figures."""

    context.table_dir.mkdir(parents=True, exist_ok=True)
    output = context.table_dir / output_name
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _invoke(module_name, [])
    text = buffer.getvalue()
    output.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Saved {output}")
    return output


def table2(context: ReproductionContext) -> Path:
    """Reproduce Table 2: aggregate clear and smoke performance."""
    return _write_table(context, "table_2", "table2.txt")


def table3(context: ReproductionContext) -> Path:
    """Reproduce Table 3: light, medium, and heavy smoke performance."""
    return _write_table(context, "table_3", "table3.txt")


def table4(context: ReproductionContext) -> Path:
    """Reproduce Table 4: refinement-stage ablation."""
    return _write_table(context, "table_4", "table4.txt")


def table5(context: ReproductionContext) -> Path:
    """Reproduce Table 5: GRADE component ablations."""
    return _write_table(context, "table_5", "table5.txt")


def table6(context: ReproductionContext) -> Path:
    """Reproduce Table 6: robustness comparison."""
    return _write_table(context, "table_6", "table6.txt")


def table7(context: ReproductionContext) -> Path:
    """Reproduce Table 7: sampling-step analysis from fixed support data."""
    return _write_table(context, "table_7", "table7.txt")


def sampling_step_analysis(context: ReproductionContext) -> Path:
    """Alias for the fixed-data sampling-step analysis reported in Table 7."""
    return table7(context)


def additional_ablation_tables(context: ReproductionContext) -> list[Path]:
    """Write the supplementary merged, Doppler, and freeze-ablation tables."""
    return [
        _write_table(context, "table_ablation_merged", "table_ablation_merged.txt"),
        _write_table(context, "table_new_doppler", "table_new_doppler.txt"),
        _write_table(context, "table_new_freeze", "table_new_freeze.txt"),
    ]


def figure9(context: ReproductionContext) -> None:
    """Reproduce Figure 9 CDF panels from fresh merged metric CSVs."""
    _invoke("figure_9", ["--mode", "separate", "--output-dir", str(context.figure_dir / "cdf_plots")])


def figure11(context: ReproductionContext) -> None:
    """Reproduce Figure 11 smoke-density box plots."""
    output = context.figure_dir / "smoke_trend"
    _invoke("figure_11", ["--output-dir", str(output), "--composite", str(context.figure_dir / "smoke_trend_composite.png")])


def figure12(context: ReproductionContext) -> None:
    """Reproduce Figure 12 paired visual-guidance degradation panels."""
    _invoke("figure_12_pair", ["--output-dir", str(context.figure_dir / "revision")])


def figure13(context: ReproductionContext) -> None:
    """Reproduce Figure 13 radar-sparsity and depth-range robustness panels."""
    radar_robustness_evaluation(context)
    range_robustness_evaluation(context)


def figure14(context: ReproductionContext) -> None:
    """Reproduce Figure 14 scene-complexity analysis."""
    _invoke("figure_12", ["--output-dir", str(context.figure_dir / "scene_complexity")])


def scene_complexity_evaluation(context: ReproductionContext) -> None:
    """Run the scene-complexity evaluation used by Figure 14."""
    figure14(context)


def visual_guidance_degradation(context: ReproductionContext) -> None:
    """Produce the supplementary rolling-MAE visual-guidance degradation plots."""
    _invoke("figure_new_degradation", ["--output-dir", str(context.figure_dir / "revision"), "--metrics", "MAE"])


def radar_robustness_evaluation(context: ReproductionContext) -> None:
    """Produce radar point-sparsity robustness plots from immutable support data."""
    output = context.figure_dir / "revision"
    _invoke(
        "robustness_sparsity",
        [
            "--bin-cuts", "9", "13", "--plot-model", "ours_radar", "ours_diffusion", "ours_full",
            "--figure-style", "paper", "--paper-only", "--output-dir", str(output),
        ],
    )
    source = output / "radar_sparsity_mae_cuts_9_13.png"
    target = output / "radar_robustness_sparsity.png"
    if not source.is_file():
        raise FileNotFoundError(f"Expected figure was not created: {source}")
    shutil.copy2(source, target)


def range_robustness_evaluation(context: ReproductionContext) -> None:
    """Produce depth-range robustness plots from stored range histograms."""
    _invoke(
        "robustness_long_range",
        [
            "--bin-width", "2", "--metric", "absrel", "--figure-style", "paper",
            "--output-dir", str(context.figure_dir / "revision"),
            "--derived-output-dir", str(context.output_root / "analysis" / "range"),
        ],
    )


def normalize_reproduced_paper_figures(context: ReproductionContext) -> None:
    """Mirror regenerated quantitative panels into the indexed paper layout."""

    panels = [
        (9, 1, 1, "cdf_plots/cdf_new_eval_legend.png"),
        *((9, 2, column, f"cdf_plots/cdf_new_eval_non_smoke_{metric}.png")
          for column, metric in enumerate(("mae", "ssim", "cd", "mhd"), start=1)),
        *((9, 3, column, f"cdf_plots/cdf_new_eval_smoke_{metric}.png")
          for column, metric in enumerate(("mae", "ssim", "cd", "mhd"), start=1)),
        (11, 1, 1, "smoke_trend/smoke_class_legend.png"),
        *((11, 2, column, f"smoke_trend/smoke_class_new_eval_{metric}.png")
          for column, metric in enumerate(("mae", "lpips", "cd", "mhd"), start=1)),
        (12, 1, 1, "revision/graceful_degradation_3.png"),
        (13, 1, 1, "revision/radar_robustness_legend.png"),
        (13, 2, 1, "revision/radar_robustness_sparsity.png"),
        (13, 2, 2, "revision/range_2m_absrel.png"),
        (14, 1, 1, "scene_complexity/scene_complexity_legend.png"),
        (14, 2, 1, "scene_complexity/scene_complexity_lpips.png"),
        (14, 2, 2, "scene_complexity/scene_complexity_cd.png"),
    ]
    for figure, row, column, relative in panels:
        source = context.figure_dir / relative
        destination = (
            context.output_root / "paper_figures" / f"figure{figure}"
            / f"figure{figure}_{row}_{column}{source.suffix}"
        )
        if not source.is_file():
            raise FileNotFoundError(f"Expected regenerated paper panel was not created: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_comparison_report(context: ReproductionContext) -> None:
    """Write one HTML page showing golden and newly reproduced artifacts side by side."""
    _invoke(
        "comparison_report",
        ["--reference-root", str(context.reference_root), "--reproduced-root", str(context.output_root)],
    )


def compare_paper_figure_assets(
    context: ReproductionContext, paper_figure_root: Path
) -> dict[str, int]:
    """Compare regenerated camera-ready figures with an external paper-asset tree.

    Byte-identical PNGs count as matches.  When only PNG metadata differs,
    identical RGB pixels also count as a match.  This incorporates the former
    optional shell comparison utility without assuming an Overleaf checkout is
    bundled in the artifact package.
    """

    from PIL import Image, ImageChops

    paper_figure_root = paper_figure_root.resolve()
    generated_root = context.figure_dir
    if not paper_figure_root.is_dir():
        raise FileNotFoundError(f"Paper figure directory not found: {paper_figure_root}")
    expected = [
        *(Path("cdf_plots") / f"cdf_new_eval_{setting}_{metric}.png"
          for setting in ("non_smoke", "smoke") for metric in ("mae", "ssim", "cd", "mhd")),
        *(Path("smoke_trend") / f"smoke_class_new_eval_{metric}.png"
          for metric in ("mae", "lpips", "cd", "mhd")),
        Path("smoke_trend/smoke_class_legend.png"),
        *(Path("scene_complexity") / name for name in (
            "scene_complexity_lpips.png", "scene_complexity_cd.png", "scene_complexity_legend.png",
        )),
        Path("revision/graceful_degradation_3.png"),
        Path("revision/radar_robustness_legend.png"),
        Path("revision/radar_robustness_sparsity.png"),
        Path("revision/range_2m_absrel.png"),
    ]
    summary = {"matched": 0, "pixel_matched": 0, "differed": 0, "missing": 0}
    for relative in expected:
        generated = generated_root / relative
        paper = paper_figure_root / relative
        if not generated.is_file() or not paper.is_file():
            summary["missing"] += 1
            print(f"MISSING: {relative}")
            continue
        if generated.read_bytes() == paper.read_bytes():
            summary["matched"] += 1
            print(f"MATCH: {relative}")
            continue
        with Image.open(generated) as generated_image, Image.open(paper) as paper_image:
            generated_rgb = generated_image.convert("RGB")
            paper_rgb = paper_image.convert("RGB")
            pixels_equal = (
                generated_rgb.size == paper_rgb.size
                and ImageChops.difference(generated_rgb, paper_rgb).getbbox() is None
            )
        if pixels_equal:
            summary["pixel_matched"] += 1
            print(f"PIXEL MATCH (PNG metadata differs): {relative}")
        else:
            summary["differed"] += 1
            print(f"DIFFERS: {relative}")
    print(
        "Figure comparison: "
        f"{summary['matched']} byte matches, {summary['pixel_matched']} pixel matches, "
        f"{summary['differed']} differences, {summary['missing']} missing."
    )
    return summary


def refresh_merged_csvs(context: ReproductionContext) -> None:
    """Merge fresh 2D and 3D evaluator outputs before paper reproduction."""
    module = importlib.import_module("metric_results.build_merged")
    with _temporary_argv(
        [
            "build_merged.py", "--raw-root", str(METRIC_DIR),
            "--output-dir", str(context.input_merged_dir),
        ]
    ):
        module.main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tables-only", action="store_true")
    mode.add_argument("--figures-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-merged-dir", type=Path, default=DEFAULT_INPUT_MERGED_DIR)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--refresh-merged", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = prepare_reproduction(args.input_merged_dir, args.reference_root, args.output_root)
    include_tables = not args.figures_only
    include_figures = not args.tables_only
    if args.dry_run:
        print(f"Would read fresh CSVs from: {context.input_merged_dir}")
        print(f"Would read golden support data from: {context.reference_root}")
        print(f"Would write reproduced artifacts to: {context.output_root}")
        print("Would run Tables 2-7 and Figures 9, 11-14; Figure 10 remains qualitative-only.")
        return
    if not context.reference_root.is_dir():
        raise FileNotFoundError(f"Golden reference results not found: {context.reference_root}")
    context.output_root.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    if args.refresh_merged:
        refresh_merged_csvs(context)
    if not context.input_merged_dir.is_dir():
        raise FileNotFoundError(
            f"Fresh merged CSVs not found: {context.input_merged_dir}. Run evaluation/run_metrics.py "
            "for each required model, or pass --refresh-merged after raw metric CSVs exist."
        )
    if include_tables:
        for producer in (table2, table3, table4, table5, table6, table7):
            producer(context)
    if include_figures:
        for producer in (
            figure9, figure11, figure12, figure13, figure14,
            visual_guidance_degradation,
        ):
            producer(context)
        normalize_reproduced_paper_figures(context)
    write_comparison_report(context)
    print("\nReproduction complete. Figure 10 is intentionally excluded (qualitative-only).")
    print(f"Reproduced outputs: {context.output_root}")
    print(f"Side-by-side report: {context.output_root / 'comparison.html'}")



# ---- Embedded former module: table_2 ----
def _build_table_2_module() -> dict[str, Any]:
    """Table 2: overall performance in clear and smoke scenarios."""


    from metric_results.common import (
        format_metrics,
        load_merged,
        metric_medians,
        overall_split,
        print_metric_table,
    )


    METHODS = [
        ("DA3", "da3"),
        ("CaFNet (No-Smoke)", "cafnet_no_smoke"),
        ("CaFNet", "cafnet"),
        ("GRT", "grt"),
        ("GRT+Image", "grt_image"),
        ("RadarCam-Depth", "radarcam-depth"),
        ("GRADE", "ours_full"),
    ]

    SETTINGS = ["Clear", "Smoke"]


    def compute() -> dict[str, list[tuple[str, list[str]]]]:
        groups: dict[str, list[tuple[str, list[str]]]] = {
            f"{setting} Scenario": [] for setting in SETTINGS
        }
        for label, variant in METHODS:
            split = overall_split(load_merged(variant))
            for setting in SETTINGS:
                groups[f"{setting} Scenario"].append((label, format_metrics(metric_medians(split[setting]))))
        return groups


    def main() -> None:
        print_metric_table("Table 2: overall performance (medians)", compute(), highlight=True)
    return locals()


# ---- Embedded former module: table_3 ----
def _build_table_3_module() -> dict[str, Any]:
    """Table 3: performance by strict MAX30105 IR smoke-density ranges."""


    from metric_results.common import (
        add_smoke_class,
        format_metrics,
        load_merged,
        metric_medians,
        print_metric_table,
    )


    METHODS = [
        ("DA3", "da3"),
        ("CaFNet (No-Smoke)", "cafnet_no_smoke"),
        ("CaFNet", "cafnet"),
        ("GRT", "grt"),
        ("GRT+Image", "grt_image"),
        ("RadarCam-Depth", "radarcam-depth"),
        ("GRADE", "ours_full"),
    ]

    CLASSES = [("Light smoke", "light"), ("Medium smoke", "medium"), ("Heavy smoke", "heavy")]


    def compute() -> dict[str, list[tuple[str, list[str]]]]:
        groups: dict[str, list[tuple[str, list[str]]]] = {label: [] for label, _ in CLASSES}
        for label, variant in METHODS:
            frame = add_smoke_class(load_merged(variant))
            for class_label, smoke_class in CLASSES:
                group = frame.loc[frame["Smoke class"] == smoke_class]
                groups[class_label].append((label, format_metrics(metric_medians(group))))
        return groups


    def main() -> None:
        print_metric_table("Table 3: smoke density breakdown (medians)", compute(), highlight=True)
    return locals()


# ---- Embedded former module: table_4 ----
def _build_table_4_module() -> dict[str, Any]:
    """Table 4: module ablation pooled over clear and smoke evaluation frames."""


    import pandas as pd

    from metric_results.common import (
        GRADIENT_ERROR,
        METRICS,
        format_metrics,
        load_merged,
        metric_medians,
        overall_split,
        print_metric_table,
    )


    METHODS = [
        ("Ours: radar stage", "ours_radar"),
        ("Ours: diffusion stage", "ours_diffusion"),
        ("GRADE (full)", "ours_full"),
        ("GRT Stage 1 + frozen refinement", "grt_refine_freeze"),
        ("GRT Stage 1 + retrained refinement", "grt_refine_retrain"),
    ]

    TABLE_METRICS = [*METRICS, GRADIENT_ERROR]
    TABLE_HEADERS = [*METRICS, "DGE"]


    def compute() -> dict[str, list[tuple[str, list[str]]]]:
        """One row per module configuration; every metric is one pooled median."""
        groups: dict[str, list[tuple[str, list[str]]]] = {"Overall": []}
        for label, variant in METHODS:
            split = overall_split(load_merged(variant))
            pooled = pd.concat([split["Clear"], split["Smoke"]], ignore_index=True)
            values = format_metrics(metric_medians(pooled, TABLE_METRICS), TABLE_METRICS)
            groups["Overall"].append((label, values))
        return groups


    def main() -> None:
        print_metric_table(
            "Table 4: module ablation (pooled clear + smoke medians)",
            compute(),
            label_header="Variant",
            metrics=TABLE_METRICS,
            headers=TABLE_HEADERS,
            highlight=False,
        )
    return locals()


# ---- Embedded former module: table_5 ----
def _build_table_5_module() -> dict[str, Any]:
    """Table 5: gradient loss ablation pooled over clear and smoke frames."""


    import pandas as pd

    from metric_results.common import (
        GRADIENT_ERROR,
        METRICS,
        format_metrics,
        load_merged,
        metric_medians,
        overall_split,
        print_metric_table,
    )


    METHODS = [
        ("W/O L_grad", "ours_radar_no_grad"),
        ("W/ L_grad", "ours_radar"),
    ]

    TABLE_METRICS = [*METRICS, GRADIENT_ERROR]
    TABLE_HEADERS = [*METRICS, "DGE"]


    def compute() -> dict[str, list[tuple[str, list[str]]]]:
        groups: dict[str, list[tuple[str, list[str]]]] = {"Overall": []}
        for label, variant in METHODS:
            split = overall_split(load_merged(variant))
            pooled = pd.concat([split["Clear"], split["Smoke"]], ignore_index=True)
            medians = metric_medians(pooled, TABLE_METRICS)
            groups["Overall"].append((label, format_metrics(medians, TABLE_METRICS)))
        return groups


    def main() -> None:
        groups = compute()
        print_metric_table(
            "Table 5: gradient loss ablation (pooled clear + smoke medians)",
            groups,
            label_header="Variant",
            metrics=TABLE_METRICS,
            headers=TABLE_HEADERS,
            highlight=True,
        )
        raw_splits = {
            label: overall_split(load_merged(variant))
            for label, variant in METHODS
        }
        without_frame = pd.concat(
            [raw_splits["W/O L_grad"]["Clear"], raw_splits["W/O L_grad"]["Smoke"]],
            ignore_index=True,
        )
        with_frame = pd.concat(
            [raw_splits["W/ L_grad"]["Clear"], raw_splits["W/ L_grad"]["Smoke"]],
            ignore_index=True,
        )
        without = metric_medians(without_frame, [GRADIENT_ERROR])[GRADIENT_ERROR]
        with_grad = metric_medians(with_frame, [GRADIENT_ERROR])[GRADIENT_ERROR]
        reduction = (without - with_grad) / without * 100.0
        print(
            "Overall: gradient supervision reduces median DGE from "
            f"{without:.4f} to {with_grad:.4f} ({reduction:.1f}% reduction)."
        )
        print()
    return locals()


# ---- Embedded former module: table_6 ----
def _build_table_6_module() -> dict[str, Any]:
    """Table 6: 3D reconstruction loss ablation pooled over clear and smoke frames."""


    import pandas as pd

    from metric_results.common import (
        format_metrics,
        load_merged,
        metric_medians,
        overall_split,
        print_metric_table,
    )


    METHODS = [
        ("W/O L_3D", "ours_full_no_3d"),
        ("W/ L_3D", "ours_full"),
    ]

    def compute() -> dict[str, list[tuple[str, list[str]]]]:
        groups: dict[str, list[tuple[str, list[str]]]] = {"Overall": []}
        for label, variant in METHODS:
            split = overall_split(load_merged(variant))
            pooled = pd.concat([split["Clear"], split["Smoke"]], ignore_index=True)
            groups["Overall"].append((label, format_metrics(metric_medians(pooled))))
        return groups


    def main() -> None:
        print_metric_table(
            "Table 6: 3D reconstruction loss ablation (pooled clear + smoke medians)",
            compute(),
            label_header="Variant",
            highlight=True,
        )
    return locals()


# ---- Embedded former module: table_7 ----
def _build_table_7_module() -> dict[str, Any]:
    """Table 7 (paper Table 6): DDIM sampling step ablation, clear+smoke pooled.

    Reads ``pre_eval_results/sampling_step/sampling_step_ablation_pooled.csv``, which
    ``sampling_step_precompute.py`` builds by pooling the clear and heavy-smoke
    sequences frame by frame and taking the median over the union.

    step=2 is omitted; see sampling_step_precompute.STEPS.
    """


    import pandas as pd

    from metric_results.common import DATA_DIR, METRICS, print_metric_table


    STEPS = [1, 5, 8, 10, 50]
    GROUP_LABEL = "Clear + Heavy Smoke (pooled)"

    SOURCE = "sampling_step_ablation_pooled.csv"


    def compute() -> dict[str, list[tuple[str, list[str]]]]:
        path = DATA_DIR / "sampling_step" / SOURCE
        if not path.is_file():
            raise SystemExit(
                f"{path} not found. Run sampling_step_precompute.py first."
            )
        frame = pd.read_csv(path)
        required = {"Sampling step", *METRICS}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows: list[tuple[str, list[str]]] = []
        for step in STEPS:
            row = frame.loc[frame["Sampling step"] == step]
            if len(row) != 1:
                raise ValueError(f"Expected one row for step {step}; found {len(row)}")
            values = [f"{float(row.iloc[0][metric]):.3f}" for metric in METRICS]
            rows.append((str(step), values))
        return {GROUP_LABEL: rows}


    def main() -> None:
        frame = pd.read_csv(DATA_DIR / "sampling_step" / SOURCE)
        n_frames = int(frame["Number of frames"].iloc[0])
        print_metric_table(
            f"Table 7: DDIM sampling step ablation (pooled, n={n_frames} frames)",
            compute(),
            label_header="Steps",
            highlight=True,
        )
    return locals()


# ---- Embedded former module: table_ablation_merged ----
def _build_table_ablation_merged_module() -> dict[str, Any]:
    """Merged loss/input ablation table (paper Table 5).

    All rows pool clear frames with smoke-sequence frames whose MAX30105 IR
    reading is at least 2000, matching the paper's overall smoke evaluation.

    Replaces the three separate ablation tables that previously ran as
    ``table_5.py`` (gradient loss), ``table_new_doppler.py`` (Doppler input) and
    ``table_6.py`` (3D reconstruction loss). Only MAE and CD are reported; SSIM,
    LPIPS, MHD and DGE are dropped.

    Two things worth knowing when reading the output:

    * The blocks do not share a base model. The gradient-loss and Doppler blocks
      ablate ``ours_radar``; the 3D-loss block ablates ``ours_full``. Numbers are
      therefore comparable within a block, not across blocks.
    * ``ours_radar`` is simultaneously the "W/ L_grad" row of the gradient block
      and the "Ours\\_radar" row of the Doppler block, so those rows carry
      identical numbers by construction rather than by coincidence.

    Emits both a plain-text table and, with --latex, the tabular body.
    """


    import argparse
    import pandas as pd

    from metric_results.common import load_merged, metric_medians, overall_split


    METRICS = ["MAE", "CD"]

    # (block title, [(row label, merged-CSV variant), ...])
    BLOCKS = [
        (
            r"Gradient loss $\mathcal{L}_{\text{grad}}$ (base: Ours\_radar)",
            [
                ("W/O $\\mathcal{L}_{\\text{grad}}$", "ours_radar_no_grad"),
                ("W/ $\\mathcal{L}_{\\text{grad}}$", "ours_radar"),
            ],
        ),
        (
            r"Doppler input (base: Ours\_radar, GRT)",
            [
                ("Ours\\_radar", "ours_radar"),
                ("Ours\\_radar w/o Doppler", "ours_radar_no_doppler"),
                ("GRT", "grt"),
                ("GRT w/o Doppler", "grt_no_doppler"),
            ],
        ),
        (
            r"3D reconstruction loss $\mathcal{L}_{\text{3D}}$ (base: Ours\_full)",
            [
                ("W/O $\\mathcal{L}_{\\text{3D}}$", "ours_full_no_3d"),
                ("W/ $\\mathcal{L}_{\\text{3D}}$", "ours_full"),
            ],
        ),
    ]

    PLAIN_LABELS = {
        "W/O $\\mathcal{L}_{\\text{grad}}$": "W/O L_grad",
        "W/ $\\mathcal{L}_{\\text{grad}}$": "W/ L_grad",
        "W/O $\\mathcal{L}_{\\text{3D}}$": "W/O L_3D",
        "W/ $\\mathcal{L}_{\\text{3D}}$": "W/ L_3D",
        "Ours\\_radar": "Ours_radar",
        "Ours\\_radar w/o Doppler": "Ours_radar w/o Doppler",
        "GRT w/o Doppler": "GRT w/o Doppler",
    }

    _CACHE: dict[str, dict[str, float]] = {}


    def medians(variant: str) -> dict[str, float]:
        """Pooled clear plus IR>=2000 smoke medians for one variant."""
        if variant not in _CACHE:
            split = overall_split(load_merged(variant))
            pooled = pd.concat([split["Clear"], split["Smoke"]], ignore_index=True)
            _CACHE[variant] = metric_medians(pooled, METRICS)
        return _CACHE[variant]


    def collect() -> list[tuple[str, list[tuple[str, list[float]]]]]:
        """[(block title, [(row label, [pooled MAE, pooled CD])])]."""
        blocks = []
        for title, rows in BLOCKS:
            collected = []
            for label, variant in rows:
                stats = medians(variant)
                values = [stats[metric] for metric in METRICS]
                collected.append((label, values))
            blocks.append((title, collected))
        return blocks


    def best_indices(values: list[list[float]]) -> list[int]:
        """Row index of the best (lowest) value per column; both metrics are errors."""
        return [min(range(len(values)), key=lambda r: values[r][c]) for c in range(2)]


    def print_plain(blocks) -> None:
        header = f"{'Variant':<26}{'MAE':>9}{'CD':>9}"
        print("\nTable 5: pooled loss and input ablations (17,302-frame medians)")
        print("=" * len(header))
        print(header)
        for title, rows in blocks:
            plain_title = (
                title.replace("$\\mathcal{L}_{\\text{grad}}$", "L_grad")
                .replace("$\\mathcal{L}_{\\text{3D}}$", "L_3D")
                .replace("\\_", "_")
            )
            print("-" * len(header))
            print(plain_title)
            marks = best_indices([v for _, v in rows])
            for index, (label, values) in enumerate(rows):
                cells = "".join(
                    f"{value:>8.3f}{'*' if marks[col] == index else ' '}"
                    for col, value in enumerate(values)
                )
                print(f"{PLAIN_LABELS.get(label, label):<26}{cells}")
        print("-" * len(header))
        print("* best within block. Blocks use different base models; compare within a block.")


    def print_latex(blocks) -> None:
        print(r"\begin{tabular}{lcc}")
        print(r"\toprule")
        print(r"\textbf{Variant} & \textbf{MAE}$\downarrow$ & \textbf{CD}$\downarrow$ \\")
        print(r"\midrule")
        for block_index, (title, rows) in enumerate(blocks):
            if block_index:
                print(r"\midrule")
            print(rf"\multicolumn{{3}}{{l}}{{\textit{{{title}}}}} \\")
            marks = best_indices([v for _, v in rows])
            for index, (label, values) in enumerate(rows):
                cells = []
                for col, value in enumerate(values):
                    text = f"{value:.3f}"
                    if marks[col] == index:
                        text = rf"\cellcolor{{blue!18}}\textbf{{{text}}}"
                    cells.append(text)
                print(f"{label} & " + " & ".join(cells) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--latex", action="store_true", help="Emit the LaTeX tabular body.")
        args = parser.parse_args()
        blocks = collect()
        if args.latex:
            print_latex(blocks)
        else:
            print_plain(blocks)
    return locals()


# ---- Embedded former module: table_new_doppler ----
def _build_table_new_doppler_module() -> dict[str, Any]:
    """Doppler ablation pooled over clear and smoke evaluation frames."""


    import pandas as pd

    from metric_results.common import (
        GRADIENT_ERROR,
        format_metrics,
        load_merged,
        metric_medians,
        overall_split,
        print_metric_table,
    )


    # (display label, with-Doppler variant, without-Doppler variant)
    PAIRS = [
        ("Ours_radar", "ours_radar", "ours_radar_no_doppler"),
        ("GRT", "grt", "grt_no_doppler"),
    ]

    COLUMNS = ["MAE", GRADIENT_ERROR, "CD", "MHD"]
    HEADERS = ["MAE", "DGE", "CD", "MHD"]


    def compute() -> dict[str, list[tuple[str, list[str]]]]:
        """One row per variant; every metric is one pooled median."""
        groups: dict[str, list[tuple[str, list[str]]]] = {"Overall": []}
        for label, kept, removed in PAIRS:
            kept_split = overall_split(load_merged(kept))
            removed_split = overall_split(load_merged(removed))
            for row_label, split in ((label, kept_split), (f"{label} (w/o Doppler)", removed_split)):
                pooled = pd.concat([split["Clear"], split["Smoke"]], ignore_index=True)
                values = format_metrics(metric_medians(pooled, COLUMNS), COLUMNS)
                groups["Overall"].append((row_label, values))
        return groups


    def main() -> None:
        print_metric_table(
            "Doppler ablation (pooled clear + smoke medians; DGE = depth gradient error)",
            compute(),
            label_header="Variant",
            metrics=COLUMNS,
            headers=HEADERS,
        )
    return locals()


# ---- Embedded former module: table_new_freeze ----
def _build_table_new_freeze_module() -> dict[str, Any]:
    """GRADE vs. GRT-GRADE with frozen and retrained refinement stages.

    Reports median metrics in the paper's clear and smoke splits, followed by the
    relative performance drop from ``ours_full`` to each GRT-GRADE variant. A
    positive drop always means worse performance: an increase for error metrics and
    a decrease for SSIM.
    """


    from metric_results.common import (
        ERROR_METRICS,
        GRADIENT_ERROR,
        METRICS,
        format_metrics,
        load_merged,
        metric_medians,
        overall_split,
        print_metric_table,
    )


    OURS = ("GRADE (ours_full)", "ours_full")
    GRT_REFINE_VARIANTS = [
        ("GRT-GRADE (frozen)", "grt_refine_freeze"),
        ("GRT-GRADE (retrained)", "grt_refine_retrain"),
    ]

    SETTINGS = ["Clear", "Smoke"]
    TABLE_METRICS = [*METRICS, GRADIENT_ERROR]
    TABLE_HEADERS = [*METRICS, "DGE"]


    def performance_drop(
        ours: dict[str, float], comparison: dict[str, float]
    ) -> list[str]:
        """Return direction-aware percentage drops relative to ``ours_full``."""
        cells = []
        for metric in TABLE_METRICS:
            baseline = ours[metric]
            if baseline == 0:
                cells.append("n/a")
                continue
            if metric in ERROR_METRICS:
                drop = (comparison[metric] - baseline) / baseline * 100.0
            else:
                drop = (baseline - comparison[metric]) / baseline * 100.0
            cells.append(f"{drop:+.1f}%")
        return cells


    def compute() -> dict[str, list[tuple[str, list[str]]]]:
        ours_split = overall_split(load_merged(OURS[1]))
        comparison_splits = [
            (label, overall_split(load_merged(variant)))
            for label, variant in GRT_REFINE_VARIANTS
        ]
        groups: dict[str, list[tuple[str, list[str]]]] = {}

        for setting in SETTINGS:
            ours_values = metric_medians(ours_split[setting], TABLE_METRICS)
            rows = [(OURS[0], format_metrics(ours_values, TABLE_METRICS))]
            for label, split in comparison_splits:
                comparison_values = metric_medians(split[setting], TABLE_METRICS)
                rows.extend(
                    [
                        (label, format_metrics(comparison_values, TABLE_METRICS)),
                        (
                            f"  drop vs. {OURS[0]}",
                            performance_drop(ours_values, comparison_values),
                        ),
                    ]
                )
            groups[f"{setting} Scenario"] = rows
        return groups


    def main() -> None:
        print_metric_table(
            "GRADE vs. GRT-GRADE variants (medians; positive drop = worse)",
            compute(),
            label_header="Variant",
            metrics=TABLE_METRICS,
            headers=TABLE_HEADERS,
        )
    return locals()


# ---- Embedded former module: figure_9 ----
def _build_figure_9_module() -> dict[str, Any]:

    import argparse
    from functools import lru_cache
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from metric_results.common import OUTPUT_DIR, clear_mask, load_aligned_merged, normalize_canvas


    METHODS = ["GRADE", "GRT", "CaFNet", "DA3", "GRT+Image", "RadarCam-Depth"]
    METHOD_FILES = {
        "GRADE": "ours_full",
        "GRT": "grt",
        "CaFNet": "cafnet",
        "DA3": "da3",
        "GRT+Image": "grt_image",
        "RadarCam-Depth": "radarcam-depth",
    }
    METRICS = ["MAE", "SSIM", "LPIPS", "CD", "MHD"]

    FONT_SIZE = 40
    TICK_SIZE = 36
    LEGEND_SIZE = 26
    LINE_WIDTH = 6
    FIGSIZE = (10, 5)  # 2:1 width:height
    GRID_FIGSIZE = (20, 7.5)
    AXIS_LINEWIDTH = 2

    # MATLAB "gem" default color order, slots 1-6.
    METHOD_COLORS = {
        "GRADE": "#0072BD",
        "GRT": "#D95319",
        "CaFNet": "#EDB120",
        "DA3": "#7E2F8E",
        "GRT+Image": "#77AC30",
        "RadarCam-Depth": "#4DBEEE",
    }
    METHOD_LINESTYLES = {
        "GRADE": "-",
        "GRT": "--",
        "CaFNet": ":",
        "DA3": "-.",
        "GRT+Image": (0, (5, 1)),
        "RadarCam-Depth": (0, (3, 1, 1, 1, 1, 1)),
    }
    X_LABELS = {
        "MAE": "MAE (m)",
        "SSIM": "1 - SSIM",
        "LPIPS": "LPIPS",
        "CD": "CD (m$^2$)",
        "MHD": "MHD (m)",
    }
    VISIBILITY_TITLES = {"non-smoke": "Clear", "smoke": "Smoke"}


    @lru_cache(maxsize=1)
    def aligned_frames():
        """All methods restricted to the exact frame intersection."""
        return load_aligned_merged(METHOD_FILES.values())


    def load_metric(method: str, metric: str, visibility: str) -> np.ndarray:
        frame = aligned_frames()[METHOD_FILES[method]]
        is_clear = clear_mask(frame)
        if visibility == "non-smoke":
            frame = frame.loc[is_clear]
        else:
            frame = frame.loc[(~is_clear) & (frame["IR"] >= 2000)]
        values = frame[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return 1.0 - values if metric == "SSIM" else values


    def cdf_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.sort(values)
        y = np.arange(1, x.size + 1, dtype=float) / x.size
        return x, y


    def axis_limits(metric: str, visibility: str, arrays: list[np.ndarray]) -> tuple[str, tuple[float, float]]:
        pooled = np.concatenate(arrays)
        pooled = pooled[np.isfinite(pooled)]
        if metric == "MAE":
            upper = float(np.quantile(pooled, 0.99))
            return "linear", (0.0, upper * 1.03)
        if metric == "SSIM":
            upper = float(np.quantile(pooled, 0.995))
            return "linear", (0.0, upper * 1.05)
        if metric == "LPIPS":
            # LPIPS is already bounded near [0, 1]; clip the tail like SSIM so the
            # six curves stay separated rather than compressed against the axis.
            upper = float(np.quantile(pooled, 0.995))
            return "linear", (0.0, min(1.0, upper * 1.05))
        if metric == "CD":
            positive = pooled[pooled > 0]
            lower = float(np.quantile(positive, 0.005))
            upper = float(np.quantile(positive, 0.99))
            return "log", (lower * 0.9, upper * 1.08)
        if metric == "MHD":
            quantile = 0.99 if visibility == "smoke" else 0.995
            upper = float(np.quantile(pooled, quantile))
            return "linear", (0.0, upper * 1.05)
        raise ValueError(f"Unsupported metric: {metric}")


    @lru_cache(maxsize=None)
    def shared_axis_limits(metric: str) -> tuple[str, tuple[float, float]]:
        """One x-range per metric, taken from the smoke row.

        Both rows of the grid then share a column axis, so a reader can compare
        the clear and smoke CDFs for a metric directly instead of mentally
        rescaling. Smoke drives the range because it always has the heavier tail;
        deriving it from clear would clip the smoke curves.
        """
        arrays = [load_metric(method, metric, "smoke") for method in METHODS]
        return axis_limits(metric, "smoke", arrays)


    def plot_panel(metric: str, visibility: str, output: Path) -> None:
        arrays = [load_metric(method, metric, visibility) for method in METHODS]
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for method, values in zip(METHODS, arrays):
            x, y = cdf_curve(values)
            ax.plot(
                x,
                y,
                color=METHOD_COLORS[method],
                linewidth=LINE_WIDTH,
                linestyle=METHOD_LINESTYLES[method],
                markeredgewidth=0,
                label=method,
            )

        scale, limits = axis_limits(metric, visibility, arrays)
        if scale == "log":
            ax.set_xscale("log")
        ax.set_xlim(*limits)
        ax.set_xlabel(X_LABELS[metric], fontsize=FONT_SIZE)
        ax.set_ylabel("CDF", fontsize=FONT_SIZE)
        if metric == "MAE":
            ax.annotate(
                VISIBILITY_TITLES[visibility],
                xy=(-0.32, 0.5),
                xycoords="axes fraction",
                rotation=90,
                ha="center",
                va="center",
                fontsize=FONT_SIZE,
                fontweight="bold",
            )
        ax.tick_params(axis="both", labelsize=TICK_SIZE, width=AXIS_LINEWIDTH, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(AXIS_LINEWIDTH)
        # No in-panel legend: the LaTeX grid carries one shared legend image above
        # all panels (FIGURES/scene_complexity/scene_complexity_legend.png), which
        # already uses the same six colors and linestyles as these curves. Same
        # convention as Figures 11 and 14.
        ax.grid(False)
        plt.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        # Fixed 2:1 canvas, uniform across all 8 panels so the LaTeX grid lines
        # up cell to cell regardless of tick-label width (CD's log-scale labels
        # are the widest, MAE panels also carry the rotated Clear/Smoke label).
        target_width = 1980
        target_height = 990
        normalize_canvas(output, target_width, target_height)
        print(f"Saved {output}")


    def plot_grid(output: Path) -> None:
        """Write Figure 9 as a full-width 2 x 4 CDF grid.

        Rows encode visibility (clear, smoke) and columns retain the Table 2
        metric order.  The single shared legend avoids repeating the same six
        method styles in every panel.
        """
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        fig, axes = plt.subplots(2, len(METRICS), figsize=GRID_FIGSIZE)
        handles = []

        for row, visibility in enumerate(("non-smoke", "smoke")):
            for col, metric in enumerate(METRICS):
                ax = axes[row, col]
                arrays = [load_metric(method, metric, visibility) for method in METHODS]
                for method, values in zip(METHODS, arrays):
                    x, y = cdf_curve(values)
                    (line,) = ax.plot(
                        x,
                        y,
                        color=METHOD_COLORS[method],
                        linewidth=3.0,
                        linestyle=METHOD_LINESTYLES[method],
                        markeredgewidth=0,
                        label=method,
                    )
                    if row == 0 and col == 0:
                        handles.append(line)

                scale, limits = axis_limits(metric, visibility, arrays)
                if scale == "log":
                    ax.set_xscale("log")
                ax.set_xlim(*limits)
                ax.set_ylim(0.0, 1.0)
                if row == 1:
                    ax.set_xlabel(X_LABELS[metric], fontsize=22)
                if col == 0:
                    ax.set_ylabel("CDF", fontsize=22)
                    ax.annotate(
                        VISIBILITY_TITLES[visibility],
                        xy=(-0.34, 0.5),
                        xycoords="axes fraction",
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=25,
                        fontweight="bold",
                    )
                # Keep every CDF panel at the requested 16:9 aspect ratio.
                ax.set_box_aspect(9 / 16)
                ax.tick_params(axis="both", labelsize=18, width=AXIS_LINEWIDTH, length=5)
                for spine in ax.spines.values():
                    spine.set_linewidth(AXIS_LINEWIDTH)
                ax.grid(False)

        fig.legend(
            handles,
            METHODS,
            loc="upper center",
            ncol=len(METHODS),
            fontsize=26,
            frameon=False,
            handlelength=1.8,
            handletextpad=0.35,
            columnspacing=0.9,
            bbox_to_anchor=(0.5, 1.03),
        )
        fig.subplots_adjust(
            left=0.055,
            right=0.995,
            top=0.86,
            bottom=0.12,
            wspace=0.15,
            hspace=0.28,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output}")


    # --- Split layout: 8 separate panel PNGs (2 rows x 4 cols) + 1 shared legend
    # PNG, assembled into the same full-width grid via a LaTeX tabular instead of
    # a single combined image. A fixed subplots_adjust rect and fixed canvas size
    # (no tight bbox) keep every panel's axis box at the identical pixel position
    # regardless of tick-label width (CD's log-scale labels are the widest), so
    # the grid still lines up cell to cell -- same technique as Figure 11.
    SEPARATE_FIGSIZE = (7.2, 3.6)  # 2:1 width:height  # 2:1 width:height
    SEPARATE_FONT_SIZE = 42
    SEPARATE_TICK_SIZE = 36
    SEPARATE_LEGEND_SIZE = 30
    SEPARATE_LINE_WIDTH = 4.5
    # top is 0.93, not 0.97: the "1.0" y tick label is centred on the top spine, so
    # roughly half its height sits above the axes box. At 0.97 that overflowed the
    # fixed canvas and the label plus the top spine were clipped.
    PANEL_RECT = {"left": 0.2687, "right": 0.9750, "bottom": 0.3316, "top": 0.9381}


    def plot_separate_panel(metric: str, visibility: str, output: Path) -> None:
        arrays = [load_metric(method, metric, visibility) for method in METHODS]
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        fig, ax = plt.subplots(figsize=SEPARATE_FIGSIZE)
        for method, values in zip(METHODS, arrays):
            x, y = cdf_curve(values)
            ax.plot(
                x, y, color=METHOD_COLORS[method], linewidth=SEPARATE_LINE_WIDTH,
                linestyle=METHOD_LINESTYLES[method], markeredgewidth=0, label=method,
            )

        scale, limits = axis_limits(metric, visibility, arrays)
        if scale == "log":
            ax.set_xscale("log")
        ax.set_xlim(*limits)
        ax.set_ylim(0.0, 1.0)
        # Every panel is labelled on both axes: the grid is wide enough that a
        # reader scanning the clear row shouldn't have to track down to the smoke
        # row to find out which metric a column is.
        ax.set_xlabel(X_LABELS[metric], fontsize=SEPARATE_FONT_SIZE)
        ax.set_ylabel("CDF", fontsize=SEPARATE_FONT_SIZE)
        if metric == "MAE":
            # Bake the row label into the leftmost (MAE) panel, to the left of
            # the "CDF" ylabel, so the LaTeX grid doesn't need a separate
            # rotated "Clear"/"Smoke" text column.
            ax.annotate(
                VISIBILITY_TITLES[visibility],
                xy=(-0.32, 0.5),
                xycoords="axes fraction",
                rotation=90,
                ha="center",
                va="center",
                fontsize=SEPARATE_FONT_SIZE,
                fontweight="bold",
            )
        ax.tick_params(axis="both", labelsize=SEPARATE_TICK_SIZE, width=AXIS_LINEWIDTH, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(AXIS_LINEWIDTH)
        ax.grid(False)

        # Shared rect (not tight_layout/bbox_inches="tight") so all eight panels
        # keep the same canvas size and axis-box position; see PANEL_RECT.
        fig.subplots_adjust(**PANEL_RECT)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220)
        plt.close(fig)
        print(f"Saved {output}")


    def plot_separate_legend(output: Path) -> None:
        """Write the shared unboxed legend as its own image, placed above the grid."""
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        handles = [
            plt.Line2D(
                [0], [0], color=METHOD_COLORS[method], linewidth=SEPARATE_LINE_WIDTH,
                linestyle=METHOD_LINESTYLES[method],
            )
            for method in METHODS
        ]
        fig = plt.figure(figsize=(20, 1.0))
        fig.legend(
            handles, METHODS, loc="center", ncol=len(METHODS),
            fontsize=SEPARATE_LEGEND_SIZE, frameon=False,
            handlelength=1.8, handletextpad=0.35, columnspacing=0.9,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved {output}")


    def plot_separate(output_dir: Path) -> None:
        for visibility in ("non-smoke", "smoke"):
            tag = "non_smoke" if visibility == "non-smoke" else "smoke"
            for metric in METRICS:
                plot_separate_panel(
                    metric, visibility, output_dir / f"cdf_new_eval_{tag}_{metric.lower()}.png"
                )
        plot_separate_legend(output_dir / "cdf_new_eval_legend.png")


    def main() -> None:
        parser = argparse.ArgumentParser(description="Reproduce the full-width camera-ready Figure 9 CDF grid.")
        parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "figure_9")
        parser.add_argument(
            "--mode", choices=["grid", "separate"], default="separate",
            help="'grid' writes one combined image; 'separate' writes 8 panel PNGs + 1 legend PNG.",
        )
        args = parser.parse_args()
        common_count = len(next(iter(aligned_frames().values())))
        print(f"Figure 9 common evaluation support: {common_count:,} frames")
        if args.mode == "grid":
            plot_grid(args.output_dir / "cdf_new_eval_grid.png")
        else:
            plot_separate(args.output_dir)
    return locals()


# ---- Embedded former module: figure_11 ----
def _build_figure_11_module() -> dict[str, Any]:
    """Figure 11: grouped box plots across clear/light/medium/heavy smoke.

    This restores the original plotting style from
    ``/Users/binzhao/Desktop/analysis/smoke_class_vs_metric.py`` while retaining
    the six methods used by the revised evaluation.
    """


    import argparse
    from functools import lru_cache
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    from metric_results.common import OUTPUT_DIR, add_smoke_class, load_aligned_merged


    METHODS = ["GRADE", "GRT", "CaFNet", "DA3", "GRT_Image", "RadarCam-Depth"]
    METHOD_FILES = {
        "GRADE": "ours_full",
        "GRT": "grt",
        "CaFNet": "cafnet",
        "DA3": "da3",
        "GRT_Image": "grt_image",
        "RadarCam-Depth": "radarcam-depth",
    }
    METRICS = ["MAE", "LPIPS", "CD", "MHD"]
    CLASS_ORDER = ["clear", "light", "medium", "heavy"]

    METHOD_COLORS = {
        "GRADE": "#0072BD",
        "GRT": "#D95319",
        "CaFNet": "#EDB120",
        "DA3": "#7E2F8E",
        "GRT_Image": "#77AC30",
        "RadarCam-Depth": "#4DBEEE",
    }
    DISPLAY_NAMES = {
        "GRADE": "GRADE",
        "GRT": "GRT",
        "CaFNet": "CaFNet",
        "DA3": "DA3",
        "GRT_Image": "GRT+Image",
        "RadarCam-Depth": "RadarCam-Depth",
    }
    Y_LABELS = {"MAE": "MAE (m)", "LPIPS": "LPIPS", "CD": "CD (m$^2$)", "MHD": "MHD (m)"}

    FONT_SIZE = 62
    TICK_SIZE = 56
    LEGEND_SIZE = 40
    FIGSIZE = (13.4, 6.5)
    BOX_WIDTH = 0.12
    BOX_ALPHA = 0.6
    BOX_LINEWIDTH = 1.5

    # Fixed axis rect (figure fraction) shared by every standalone panel so the
    # plot box lands at the identical pixel position regardless of how wide that
    # metric's tick labels are (e.g. CD's log-scale "$10^{-2}$" labels versus
    # MAE's plain integers). Using a shared rect instead of per-panel
    # tight_layout/bbox_inches="tight" is what makes the four panels align when
    # tiled 2x2 in the LaTeX table: same left margin fits the widest label set
    # (CD), same canvas size (no post-hoc cropping) keeps every axis edge at the
    # same pixel offset.
    PANEL_RECT = {"left": 0.1765, "right": 0.9950, "bottom": 0.1315, "top": 0.9501}

    # Per-metric left margin. One shared rect leaves the panels with very different
    # amounts of white to the left of the ylabel, because the y tick labels differ
    # hugely in width ("0.00/0.25/0.50/0.75" for LPIPS versus "0/2/4" for MHD).
    # The figure is tiled horizontally, so that white reads as uneven spacing
    # between panels. Tuning left per metric makes every panel's ink start the same
    # distance from its edge, which makes the gaps uniform; the inter-panel gap is
    # then set once in LaTeX via \tabcolsep.
    PANEL_LEFT = {"MAE": 0.1022, "LPIPS": 0.1731, "CD": 0.1766, "MHD": 0.1022}


    def panel_rect(metric: str) -> dict:
        return {**PANEL_RECT, "left": PANEL_LEFT.get(metric, PANEL_RECT["left"])}


    @lru_cache(maxsize=1)
    def aligned_frames():
        return load_aligned_merged(METHOD_FILES.values())


    def distributions(metric: str) -> dict[str, dict[str, np.ndarray]]:
        result: dict[str, dict[str, np.ndarray]] = {}
        for method in METHODS:
            frame = add_smoke_class(aligned_frames()[METHOD_FILES[method]])
            result[method] = {}
            for smoke_class in CLASS_ORDER:
                values = frame.loc[frame["Smoke class"] == smoke_class, metric].to_numpy(dtype=float)
                result[method][smoke_class] = values[np.isfinite(values)]
        return result


    def boxplot_whisker_bounds(values: np.ndarray) -> tuple[float, float] | None:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            return float(values.min()), float(values.max())
        inliers = values[
            (values >= q1 - 1.5 * iqr)
            & (values <= q3 + 1.5 * iqr)
        ]
        if inliers.size == 0:
            return float(q1), float(q3)
        return float(inliers.min()), float(inliers.max())


    def set_axis(ax: plt.Axes, metric: str, data) -> None:
        bounds = [
            boxplot_whisker_bounds(data[method][smoke_class])
            for method in METHODS
            for smoke_class in CLASS_ORDER
        ]
        bounds = [item for item in bounds if item is not None]
        lows = [item[0] for item in bounds]
        highs = [item[1] for item in bounds]
        if metric == "CD":
            positive = [value for value in lows if value > 0]
            ax.set_yscale("log")
            ax.set_ylim(min(positive) / 2.0, 95)
            return

        lower = min(lows)
        upper = max(highs)
        span = max(upper - lower, 1e-3)
        pad = span * 0.05
        top = upper + pad * 0.45
        if metric == "LPIPS":
            top = 0.75
        ax.set_ylim(max(0.0, lower - pad * 0.35), top)


    def draw_panel(ax: plt.Axes, metric: str, add_legend: bool) -> tuple[list, list[str]]:
        data = distributions(metric)
        x = np.arange(len(CLASS_ORDER), dtype=float)
        offsets = np.linspace(
            -BOX_WIDTH * (len(METHODS) - 1) / 2.0,
            BOX_WIDTH * (len(METHODS) - 1) / 2.0,
            len(METHODS),
        )
        handles = []
        labels = []

        for offset, method in zip(offsets, METHODS):
            color = METHOD_COLORS[method]
            box = ax.boxplot(
                [data[method][smoke_class] for smoke_class in CLASS_ORDER],
                positions=x + offset,
                widths=BOX_WIDTH,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": color, "linewidth": 2.0},
                whiskerprops={"linewidth": BOX_LINEWIDTH, "color": color},
                capprops={"linewidth": BOX_LINEWIDTH, "color": color},
                boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": color},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(BOX_ALPHA)
            handles.append(box["boxes"][0])
            labels.append(DISPLAY_NAMES[method])

        ax.set_xticks(x)
        ax.set_xticklabels([name.capitalize() for name in CLASS_ORDER], fontsize=TICK_SIZE)
        ax.set_xlim(-0.62, len(CLASS_ORDER) - 0.38)
        ax.set_ylabel(Y_LABELS[metric], fontsize=FONT_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.grid(False)
        set_axis(ax, metric, data)
        if add_legend:
            ax.legend(
                handles,
                labels,
                loc="upper left",
                fontsize=LEGEND_SIZE,
                frameon=True,
                ncol=2,
                handletextpad=0.35,
                columnspacing=0.8,
                labelspacing=0.25,
                borderpad=0.4,
            )
        return handles, labels


    def plot_panel(metric: str, output: Path) -> None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        # No inline legend: the 2x2 LaTeX grid reuses the shared legend image
        # from Figure 14 (scene_complexity_legend.png), which already covers the
        # same six methods in the same colors, instead of repeating a legend on
        # every panel.
        draw_panel(ax, metric, add_legend=False)
        # Shared rect (not tight_layout / bbox_inches="tight") so all four panels
        # keep the exact same canvas size and axis-box position; see PANEL_RECT.
        fig.subplots_adjust(**panel_rect(metric))
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220)
        plt.close(fig)
        print(f"Saved {output}")


    def plot_legend(output: Path) -> None:
        """Write a shared box-plot legend using solid colored rectangles."""
        handles = [
            Patch(
                facecolor=METHOD_COLORS[method],
                edgecolor=METHOD_COLORS[method],
                label=DISPLAY_NAMES[method],
            )
            for method in METHODS
        ]
        fig = plt.figure(figsize=(8, 0.7))
        fig.legend(
            handles=handles,
            loc="center",
            ncol=6,
            fontsize=16,
            frameon=False,
            borderpad=0.25,
            labelspacing=0.0,
            handlelength=1.2,
            handletextpad=0.2,
            columnspacing=0.25,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved {output}")


    def plot_composite(output: Path) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
        handles = None
        labels = None
        for ax, metric, panel in zip(axes.flat, METRICS, ("(a)", "(b)", "(c)", "(d)")):
            handles, labels = draw_panel(ax, metric, add_legend=False)
            ax.set_title(f"{panel} {Y_LABELS[metric]}", fontsize=FONT_SIZE, fontweight="bold")
        assert handles is not None and labels is not None
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            fontsize=LEGEND_SIZE,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            borderpad=0.25,
            labelspacing=0.1,
            handlelength=1.4,
            handletextpad=0.25,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output}")


    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Median and middle-50% performance across smoke classes."
        )
        parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "figure_11")
        parser.add_argument("--composite", type=Path, default=OUTPUT_DIR / "figure_11.png")
        args = parser.parse_args()

        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        common_count = len(next(iter(aligned_frames().values())))
        print(f"Figure 11 common evaluation support: {common_count:,} frames")
        for metric in METRICS:
            # Lower-cased for case-sensitive filesystems; see figure_9.py.
            plot_panel(metric, args.output_dir / f"smoke_class_new_eval_{metric.lower()}.png")
        plot_legend(args.output_dir / "smoke_class_legend.png")
        plot_composite(args.composite)
    return locals()


# ---- Embedded former module: figure_12_pair ----
def _build_figure_12_pair_module() -> dict[str, Any]:
    """Figure 12: paired per-frame MAE gap from visual guidance.

    The figure reports Ours_full - Ours_radar, i.e. what the ControlNet
    visual-guidance branch adds on top of the radar-only stage.

    Both panels use the same rolling ±WINDOW_HALF_WIDTH IR window and the same
    support threshold.  Unlike the earlier single-panel version, low-support
    windows are dropped outright rather than drawn as a gray dashed tail: the tail
    was never interpreted in the text and it stretched the x-axis far past the
    region that carries evidence.

    Deltas are plotted in millimetres.  The underlying medians are metres, but the
    differences sit in the 1e-3 range, where metre labels are unreadable.

    Output is a single PNG/PDF sized for one ACM column.
    """


    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    from metric_results.common import OUTPUT_DIR, load_merged


    # (numerator variant, subtrahend variant, "helps" label, "hurts" label)
    PANEL = ("ours_full", "ours_radar", "GRADE better", "Radar-only better")

    # Legend wordings to choose between. One PNG is written per pair, numbered
    # graceful_degradation_1.png, _2.png, ... in this order.
    LABEL_VARIANTS = [
        ("GRADE better", "Radar-only better"),
        ("Image helps", "Image hurts"),
        ("Visual guidance", "Visual degradation"),
        ("RGB helps", "RGB hurts"),
        ("Camera helps", "Camera hurts"),
        ("Guidance gains", "Guidance costs"),
        ("RGB gain", "RGB degradation"),
    ]

    METRIC_COLUMN = "MAE"
    METRIC_NAME = "MAE"

    WINDOW_HALF_WIDTH = 250.0
    WINDOW_STEP = 120.0
    MIN_SUPPORT = 25
    IR_THRESHOLDS = [(2000.0, "IR 2000"), (4000.0, "IR 4000")]
    REGION_LABELS = ("Light", "Medium", "Heavy")

    # MATLAB "gem" default color order: slot 1 (blue) for GRADE/full-model,
    # consistent with METHOD_COLORS["GRADE"] in figure_9.py. Radar-only was
    # previously slot 2 (#D95319), which duplicates the GRT curve color
    # elsewhere in the paper; it now uses slot 7 (#A2142F) instead.
    BETTER_COLOR = "#0072BD"
    WORSE_COLOR = "#A2142F"
    DELTA_COLOR = "#333333"

    # Sized to the finalized Table 5-7 row width/height in evaluation_new.tex:
    # the figure column there is 0.255\textwidth wide and the shared row box is
    # 0.95in tall, an aspect ratio of about 1.88:1. This keeps the source width
    # large enough that FONT_SIZE etc. downscale to about the same effective
    # on-page size as the original (8, 6) draft did at its old (wider) column.
    # 4:2.5 aspect. Figure 12 is now its own \columnwidth float rather than one
    # cell of a three-panel row, so it renders about twice as wide as before; every
    # type size is halved to keep the on-page text at the same effective size.
    FIGSIZE = (9, 4)   # 4 : 2.5
    FONT_SIZE = 32
    TICK_SIZE = 28
    LEGEND_SIZE = 24
    REGION_LABEL_SIZE = FONT_SIZE * 0.9  # a touch smaller than the xlabel
    LINE_WIDTH = 6
    MARKER_SIZE = 7
    AXIS_LINEWIDTH = 2


    def finite_median(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else float("nan")


    def paired_frame(numerator: str, subtrahend: str):
        """One-to-one frame join so every delta is genuinely paired."""
        merged = load_merged(numerator).merge(
            load_merged(subtrahend),
            on=["sequences", "frame_idx"],
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        merged = merged.rename(columns={"IR_a": "IR"})
        return merged.loc[np.isfinite(merged["IR"])].copy()


    def rolling_delta(merged, metric_column: str, half_width: float, step: float):
        """Rolling window centres, paired-frame counts, and median deltas."""
        ir = merged["IR"].to_numpy(dtype=float)
        a = merged[f"{metric_column}_a"].to_numpy(dtype=float)
        b = merged[f"{metric_column}_b"].to_numpy(dtype=float)
        finite = np.isfinite(ir) & np.isfinite(a) & np.isfinite(b)
        ir, delta = ir[finite], (a - b)[finite]

        start = np.ceil(float(ir.min()) / step) * step
        stop = np.floor(float(ir.max()) / step) * step
        centres, counts, deltas = [], [], []
        for centre in np.arange(start, stop + step * 0.5, step):
            mask = np.abs(ir - centre) <= half_width
            count = int(mask.sum())
            if count == 0:
                continue
            centres.append(float(centre))
            counts.append(count)
            deltas.append(finite_median(delta[mask]))
        return np.asarray(centres), np.asarray(counts), np.asarray(deltas)


    def symmetric_limit(deltas: np.ndarray, divisions: int = 2) -> tuple[float, float]:
        """Symmetric y limit and tick step covering the data, on a round number.

        Zero has to sit at the exact vertical centre so the two signed regions are
        visually comparable, which means the limit is set by ``max|delta|`` rather
        than by the data range.  The step is taken from a 1-2-5-style ladder so the
        resulting ticks are readable at column width, and ``divisions`` per side
        keeps the label count low on a small panel.
        """
        magnitude = float(np.max(np.abs(deltas))) if deltas.size else 1.0
        if not np.isfinite(magnitude) or magnitude <= 0:
            return 1.0, 0.5
        raw_step = magnitude / divisions
        decade = 10.0 ** np.floor(np.log10(raw_step))
        step = next(
            mult * decade
            for mult in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)
            if mult * decade >= raw_step - 1e-9
        )
        limit = step * divisions
        while limit < magnitude - 1e-9:  # guards the ladder's largest rung
            limit += step
        return limit, step


    def style_axis(ax: plt.Axes) -> None:
        ax.grid(True, alpha=0.35, linewidth=0.8, color="#b0b0b0")
        ax.tick_params(axis="both", labelsize=TICK_SIZE, width=AXIS_LINEWIDTH, length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(AXIS_LINEWIDTH)
        # Few, short x ticks: the axis spans several thousand IR counts and long
        # labels collide once the figure is scaled to one column.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v/1000:g}k" if v >= 1000 else f"{v:g}")
        )


    def mark_thresholds(ax: plt.Axes, x_lo: float, x_hi: float) -> None:
        """Divide the axis into the three density regimes and name them.

        The regime boundaries carry the meaning here, not the raw IR values, so the
        thresholds are drawn as dividers and each span is labelled by name at the
        top of the panel.
        """
        for threshold, _ in IR_THRESHOLDS:
            ax.axvline(threshold, color="#666666", linestyle=":", linewidth=AXIS_LINEWIDTH, zorder=1)

        edges = [x_lo] + [t for t, _ in IR_THRESHOLDS] + [x_hi]
        for (lo, hi), name in zip(zip(edges[:-1], edges[1:]), REGION_LABELS):
            ax.text(
                0.5 * (lo + hi),
                0.94,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=REGION_LABEL_SIZE,
                fontweight="bold",
                color="#444444",
            )


    def plot_panel(ax, centres, deltas, helps_label, hurts_label, x_lo, x_hi):
        """Signed-area delta curve. Negative delta = the augmented model is better."""
        helps = deltas < 0
        hurts = ~helps

        ax.fill_between(
            centres, deltas, 0, where=hurts, color=WORSE_COLOR,
            interpolate=True, label=hurts_label,
        )
        ax.fill_between(
            centres, deltas, 0, where=helps, color=BETTER_COLOR,
            interpolate=True, label=helps_label,
        )
        ax.plot(
            centres, deltas, color=DELTA_COLOR, linewidth=LINE_WIDTH,
            marker="o", markersize=MARKER_SIZE, zorder=3,
        )
        ax.axhline(0, color="red", linestyle="--", linewidth=AXIS_LINEWIDTH, zorder=2)

        # Zero is pinned to the vertical centre so the "helps" and "hurts" areas are
        # read on the same scale; each panel is scaled from its own data.
        limit, step = symmetric_limit(deltas)
        ax.set_ylim(-limit * 1.18, limit * 1.18)
        ax.set_yticks(np.arange(-limit, limit + step * 0.5, step))

        # The curve stays close to zero at the high end of the x-range and the
        # y-axis is pinned symmetric about zero, so the bottom-right corner of
        # the panel is empty and doesn't overlap the delta curve or either fill.
        ax.legend(
            loc="lower right",
            fontsize=LEGEND_SIZE, frameon=True, fancybox=False,
            framealpha=1.0, edgecolor="black", borderpad=0.3, labelspacing=0.2,
            handlelength=1.4, handletextpad=0.4,
        )
        mark_thresholds(ax, x_lo, x_hi)
        style_axis(ax)


    def plot_legend(output: Path, helps_label: str, hurts_label: str) -> None:
        """Write the shared legend as its own image, placed above the panel.

        Same convention as Figure 14 (scene_complexity_legend.png): a single
        unboxed horizontal strip, so the panel keeps its full plotting area. The
        font is set to FONT_SIZE so the legend reads at the same size as the axis
        labels once both are scaled to their LaTeX widths.
        """
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=BETTER_COLOR, edgecolor=BETTER_COLOR, label=helps_label),
                   Patch(facecolor=WORSE_COLOR, edgecolor=WORSE_COLOR, label=hurts_label)]
        fig = plt.figure(figsize=(8.68, 0.62))
        fig.legend(handles=handles, loc="center", ncol=2, fontsize=TICK_SIZE,
                   frameon=False, borderpad=0.25, labelspacing=0.0,
                   handlelength=1.2, handletextpad=0.2, columnspacing=0.25)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        # No extra vertical padding: apparent size on the page is
        # font_em_px / canvas_height_px, so padding only shrinks the text. A tight
        # bbox keeps this legend as large as its column width permits.
        print(f"Saved {output}")


    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
        parser.add_argument("--window-half-width", type=float, default=WINDOW_HALF_WIDTH)
        parser.add_argument("--step", type=float, default=WINDOW_STEP)
        parser.add_argument("--min-support", type=int, default=MIN_SUPPORT)
        parser.add_argument("--quiet", action="store_true")
        args = parser.parse_args()

        plt.rcParams.update(
            {
                "font.family": "STIXGeneral",
                "mathtext.fontset": "stix",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

        numerator, subtrahend, _, _ = PANEL
        merged = paired_frame(numerator, subtrahend)
        centres, counts, deltas = rolling_delta(
            merged, METRIC_COLUMN, args.window_half_width, args.step
        )
        keep = counts >= args.min_support
        centres, counts, deltas = centres[keep], counts[keep], deltas[keep]
        x_lo = 0.0
        x_hi = centres.max() + 150
        deltas_mm = deltas * 1000.0

        if not args.quiet:
            better = int((deltas < 0).sum())
            print(f"\n{numerator} - {subtrahend}: {len(merged):,} paired frames")
            print(
                f"  supported windows {len(centres)}  "
                f"(dropped {int((~keep).sum())} below N={args.min_support})"
            )
            print(f"  IR span {centres.min():.0f}-{centres.max():.0f}")
            print(f"  GRADE better in {better}/{len(centres)} windows")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        # One figure per legend wording, so the wording can be picked by eye.
        for n, (helps_label, hurts_label) in enumerate(LABEL_VARIANTS, start=1):
            fig, ax = plt.subplots(figsize=FIGSIZE)
            plot_panel(ax, centres, deltas_mm, helps_label, hurts_label, x_lo, x_hi)
            ax.set_xlim(x_lo, x_hi)
            ax.set_xlabel("Smoke Density", fontsize=FONT_SIZE)
            ax.set_ylabel(rf"$\Delta${METRIC_NAME} (mm)", fontsize=FONT_SIZE)
            fig.tight_layout(pad=0.5)
            out = args.output_dir / f"graceful_degradation_{n}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out}   [{helps_label} / {hurts_label}]")
    return locals()


# ---- Embedded former module: figure_12 ----
def _build_figure_12_module() -> dict[str, Any]:
    """Figure 14: scene-complexity performance over the complete test set.

    Clear and smoke sequences are intentionally mixed before computing the shared
    RMS-contrast terciles. The compact two-column figure retains LPIPS and CD,
    with one shared full-width legend above both panels.
    """


    import argparse
    from functools import lru_cache
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

    from metric_results.common import OUTPUT_DIR, load_aligned_merged, normalize_canvas


    # Fields: (label, variant, colour, marker, linestyle).
    METHODS = [
        ("GRADE", "ours_full", "#0072BD", "o", "-"),
        ("GRT", "grt", "#D95319", "s", "--"),
        ("CaFNet", "cafnet", "#EDB120", "^", ":"),
        ("DA3", "da3", "#7E2F8E", "v", "-."),
        ("GRT_Image", "grt_image", "#77AC30", "D", (0, (5, 1))),
        ("RadarCam-Depth", "radarcam-depth", "#4DBEEE", "P", (0, (3, 1, 1, 1, 1, 1))),
    ]
    LEGEND_LABELS = {
        # Shortened for the legend strip only; prose and captions use full names.
        "GRT_Image": "GRT+Img",
    }

    METRICS = ["LPIPS", "CD"]
    Y_LABELS = {"MAE": "MAE (m)", "LPIPS": "LPIPS", "CD": "CD (m$^2$)"}
    LOG_METRICS = {"CD"}

    CLASS_ORDER = ["simple", "medium", "complex"]
    REFERENCE = "ours_full"

    FONT_SIZE = 40
    TICK_SIZE = 34
    FIGSIZE = (8, 4.30)   # shorter panels for Figure 14
    LEGEND_FIGSIZE = (8, 0.95)
    LEGEND_SIZE = 16
    LINE_WIDTH = 6
    MARKER_SIZE = 9
    AXIS_LINEWIDTH = 2
    CONFIDENCE_Z = 1.96
    CANVAS_WIDTH = 1580
    CANVAS_HEIGHT = 856   # tracks FIGSIZE height (8 x 4.30 at the same scale)


    @lru_cache(maxsize=1)
    def aligned_frames():
        return load_aligned_merged(variant for _, variant, *_ in METHODS)


    def complexity_thresholds() -> tuple[float, float]:
        """Tercile cuts on RMS_contrast, taken once over the reference frame set."""
        values = aligned_frames()[REFERENCE]["RMS_contrast"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(np.quantile(values, 1 / 3)), float(np.quantile(values, 2 / 3))


    def classify(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
        return np.where(values < lower, "simple", np.where(values < upper, "medium", "complex"))


    def median_interval(values: np.ndarray) -> tuple[float, float, float]:
        """Median with a 95% order-statistic confidence interval."""
        values = np.sort(values[np.isfinite(values)])
        size = values.size
        if size == 0:
            return float("nan"), float("nan"), float("nan")
        median = float(np.median(values))
        rank = int(np.floor(size / 2.0 - CONFIDENCE_Z * np.sqrt(size) / 2.0))
        rank = max(rank, 0)
        return median, float(values[rank]), float(values[size - 1 - rank])


    def collect(
        metric: str, lower: float, upper: float
    ) -> dict[str, dict[str, tuple[float, float, float]]]:
        result: dict[str, dict[str, tuple[float, float, float]]] = {}
        for label, variant, _, _, _ in METHODS:
            frame = aligned_frames()[variant]
            classes = classify(frame["RMS_contrast"].to_numpy(dtype=float), lower, upper)
            result[label] = {
                name: median_interval(frame.loc[classes == name, metric].to_numpy(dtype=float))
                for name in CLASS_ORDER
            }
        return result


    def class_counts(lower: float, upper: float) -> dict[str, int]:
        frame = aligned_frames()[REFERENCE]
        classes = classify(frame["RMS_contrast"].to_numpy(dtype=float), lower, upper)
        return {name: int((classes == name).sum()) for name in CLASS_ORDER}


    def legend_handles() -> list[Line2D]:
        return [
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE + 1,
                label=LEGEND_LABELS.get(label, label),
            )
            for label, _, color, marker, linestyle in METHODS
        ]


    def plot_legend(output: Path) -> None:
        """Write the compact single-row legend for the one-column Figure 14."""
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        fig = plt.figure(figsize=LEGEND_FIGSIZE)
        legend = fig.legend(
            handles=legend_handles(),
            loc="center",
            ncol=6,
            fontsize=LEGEND_SIZE,
            frameon=False,
            borderpad=0.25,
            labelspacing=0.0,
            handlelength=1.2,
            handletextpad=0.2,
            columnspacing=0.25,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        # pad_inches matches the other two legends: extra padding inflates the
        # canvas height, which directly shrinks apparent text size on the page.
        fig.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved {output}")


    def plot_panel(
        metric: str, lower: float, upper: float, counts: dict[str, int], output: Path
    ) -> None:
        data = collect(metric, lower, upper)
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        fig, ax = plt.subplots(figsize=FIGSIZE)
        x = np.arange(len(CLASS_ORDER), dtype=float)

        for label, _, color, marker, linestyle in METHODS:
            medians = np.array([data[label][name][0] for name in CLASS_ORDER])
            lows = np.array([data[label][name][1] for name in CLASS_ORDER])
            highs = np.array([data[label][name][2] for name in CLASS_ORDER])
            ax.errorbar(
                x,
                medians,
                yerr=np.vstack([medians - lows, highs - medians]),
                color=color,
                marker=marker,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                linestyle=linestyle,
                capsize=4,
                capthick=AXIS_LINEWIDTH,
                elinewidth=AXIS_LINEWIDTH,
            )

        if metric in LOG_METRICS:
            # The range spans under a decade, so label the 1/2/5 steps rather than
            # leaving a single decade tick.
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks(x)
        ax.set_xticklabels([name.capitalize() for name in CLASS_ORDER], fontsize=TICK_SIZE)
        ax.set_xlim(-0.5, len(CLASS_ORDER) - 0.5)
        ax.set_ylabel(Y_LABELS[metric], fontsize=FONT_SIZE)
        # No x-axis label: the tick labels (Simple/Medium/Complex) already name
        # the strata, and the caption states what they are.
        ax.tick_params(axis="both", labelsize=TICK_SIZE, width=AXIS_LINEWIDTH, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(AXIS_LINEWIDTH)
        ax.grid(True, alpha=0.35, linewidth=1.0, color="#b0b0b0")
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        normalize_canvas(output, CANVAS_WIDTH, CANVAS_HEIGHT)
        print(f"Saved {output}")


    def summary(lower: float, upper: float) -> None:
        print(f"\nRMS_contrast terciles (shared): simple < {lower:.4f} <= medium < {upper:.4f} <= complex")
        width = max(len(label) for label, _, _, _, _ in METHODS)
        counts = class_counts(lower, upper)
        print("\nAll clear and smoke test sequences combined")
        print("Frame counts: " + ", ".join(f"{name}={counts[name]:,}" for name in CLASS_ORDER))
        for metric in METRICS:
            data = collect(metric, lower, upper)
            print(f"\n{metric} median [95% CI]")
            print(f"{'Method':<{width}}  " + "  ".join(f"{name:>24}" for name in CLASS_ORDER))
            for label, _, _, _, _ in METHODS:
                cells = [
                    f"{data[label][name][0]:.3f} [{data[label][name][1]:.3f}, {data[label][name][2]:.3f}]".rjust(24)
                    for name in CLASS_ORDER
                ]
                print(f"{label:<{width}}  " + "  ".join(cells))
        print()


    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Scene complexity across all clear and smoke test sequences."
        )
        parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "figure_12")
        parser.add_argument("--quiet", action="store_true", help="Skip the printed summary.")
        args = parser.parse_args()

        lower, upper = complexity_thresholds()
        common_count = len(next(iter(aligned_frames().values())))
        counts = class_counts(lower, upper)
        print(f"Figure 14 common evaluation support: {common_count:,} frames")
        for metric in METRICS:
            plot_panel(
                metric,
                lower,
                upper,
                counts,
                # Lower-cased for case-sensitive filesystems; see figure_9.py.
                args.output_dir / f"scene_complexity_{metric.lower()}.png",
            )
        plot_legend(args.output_dir / "scene_complexity_legend.png")
        if not args.quiet:
            summary(lower, upper)
    return locals()


# ---- Embedded former module: figure_new_degradation ----
def _build_figure_new_degradation_module() -> dict[str, Any]:
    """Graceful degradation of visual guidance as MAX30105 IR increases.

    Emits the paired per-frame difference ``Ours_full - Ours_radar`` as a rolling
    median evaluated at fixed IR steps. It also emits a smoothed absolute MAE panel
    for the two models, truncated after the last sufficiently supported window.
    Positive deltas hurt for error metrics and help for SSIM. The figures mark the
    requested IR thresholds at 2000 and 4000.

    In the delta figure, the high-IR tail is retained rather than merged or hidden.
    Windows with fewer than ``MIN_COLOR_COUNT`` paired frames are shown as a gray
    dashed curve and excluded only from the blue/orange interpretation regions.

    The script emits a vector PDF and a PNG preview for each requested metric.
    """


    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from metric_results.common import GRADIENT_ERROR, OUTPUT_DIR, load_merged, normalize_canvas


    FULL = "ours_full"
    RADAR = "ours_radar"
    WINDOW_HALF_WIDTH = 250.0
    WINDOW_STEP = 120.0
    MIN_COLOR_COUNT = 25
    IR_THRESHOLDS = [(2000.0, "IR 2000"), (4000.0, "IR 4000")]

    # Display name -> merged-CSV column, axis label, filename slug, lower-is-better.
    METRIC_SPECS = {
        "MAE": ("MAE", "MAE (m)", "mae", True),
        "SSIM": ("SSIM", "SSIM", "ssim", False),
        "LPIPS": ("LPIPS", "LPIPS", "lpips", True),
        "CD": ("CD", "CD (m)", "cd", True),
        "MHD": ("MHD", "MHD (m)", "mhd", True),
        "DGE": (GRADIENT_ERROR, "DGE", "dge", True),
    }

    FULL_COLOR = "#0072BD"
    RADAR_COLOR = "#D95319"
    DELTA_COLOR = "#333333"
    WORSE_COLOR = "#D95319"
    BETTER_COLOR = "#0072BD"
    LOW_SUPPORT_COLOR = "#8A8A8A"

    FONT_SIZE = 40
    TICK_SIZE = 36
    LEGEND_SIZE = 26
    LINE_WIDTH = 6
    MARKER_SIZE = 6
    FIGSIZE = (8, 6)
    AXIS_LINEWIDTH = 2
    X_LABEL_SIZE = 22
    CANVAS_WIDTH = 1580
    CANVAS_HEIGHT = 1180


    def finite_median(values: np.ndarray) -> float:
        """Median after dropping non-finite values."""
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else float("nan")


    def paired_frame():
        """One-to-one frame join so every metric delta is genuinely paired."""
        full = load_merged(FULL)
        radar = load_merged(RADAR)
        merged = full.merge(
            radar,
            on=["sequences", "frame_idx"],
            suffixes=("_full", "_radar"),
            validate="one_to_one",
        )
        merged = merged.rename(columns={"IR_full": "IR"})
        return merged.loc[np.isfinite(merged["IR"])].copy()


    def rolling_stats(
        merged,
        metric_column: str,
        window_half_width: float = WINDOW_HALF_WIDTH,
        step: float = WINDOW_STEP,
    ):
        """IR centre/count, both metric medians, and paired median delta."""
        ir = merged["IR"].to_numpy(dtype=float)
        full = merged[f"{metric_column}_full"].to_numpy(dtype=float)
        radar = merged[f"{metric_column}_radar"].to_numpy(dtype=float)
        finite = np.isfinite(ir) & np.isfinite(full) & np.isfinite(radar)
        ir, full, radar = ir[finite], full[finite], radar[finite]
        delta = full - radar

        start = np.ceil(float(ir.min()) / step) * step
        stop = np.floor(float(ir.max()) / step) * step
        evaluation_points = np.arange(start, stop + step * 0.5, step)
        centres, counts, full_medians, radar_medians, delta_medians = [], [], [], [], []
        for centre in evaluation_points:
            mask = np.abs(ir - centre) <= window_half_width
            count = int(mask.sum())
            if count == 0:
                continue
            centres.append(float(centre))
            counts.append(count)
            full_medians.append(finite_median(full[mask]))
            radar_medians.append(finite_median(radar[mask]))
            delta_medians.append(finite_median(delta[mask]))
        return (
            np.asarray(centres),
            np.asarray(counts),
            np.asarray(full_medians),
            np.asarray(radar_medians),
            np.asarray(delta_medians),
        )


    def style_axis(ax: plt.Axes) -> None:
        ax.grid(True, alpha=0.35, linewidth=1.0, color="#b0b0b0")
        ax.tick_params(
            axis="both",
            labelsize=TICK_SIZE,
            width=AXIS_LINEWIDTH,
            length=6,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(AXIS_LINEWIDTH)


    def mark_thresholds(ax: plt.Axes) -> None:
        for threshold, label in IR_THRESHOLDS:
            ax.axvline(
                threshold,
                color="#666666",
                linestyle=":",
                linewidth=AXIS_LINEWIDTH,
                zorder=1,
            )
            ax.text(
                threshold,
                0.03,
                label,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=TICK_SIZE - 6,
                color="#555555",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
            )


    def plot_curves(
        ax: plt.Axes,
        centres: np.ndarray,
        counts: np.ndarray,
        full_medians: np.ndarray,
        radar_medians: np.ndarray,
        metric_name: str,
        axis_label: str,
        min_color_count: int,
    ) -> None:
        """Plot rolling absolute curves over the supported IR range only."""
        supported = counts >= min_color_count
        if not np.any(supported):
            raise ValueError(
                f"No rolling window has the required support N>={min_color_count}"
            )
        last_supported = int(np.flatnonzero(supported)[-1]) + 1
        centres = centres[:last_supported]
        supported = supported[:last_supported]
        full_medians = np.ma.masked_where(~supported, full_medians[:last_supported])
        radar_medians = np.ma.masked_where(~supported, radar_medians[:last_supported])

        ax.plot(
            centres,
            radar_medians,
            color=RADAR_COLOR,
            linestyle="--",
            linewidth=LINE_WIDTH,
            marker="s",
            markersize=MARKER_SIZE,
            label="Ours_radar",
        )
        ax.plot(
            centres,
            full_medians,
            color=FULL_COLOR,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label="Ours_full",
        )
        ax.set_ylabel(axis_label, fontsize=FONT_SIZE)
        ax.set_title(f"Absolute {metric_name}", fontsize=FONT_SIZE, fontweight="bold")
        ax.legend(
            fontsize=LEGEND_SIZE,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            borderpad=0.25,
            labelspacing=0.1,
            handlelength=1.4,
            handletextpad=0.25,
        )
        mark_thresholds(ax)
        style_axis(ax)


    def save_absolute_figure(
        output_base: Path,
        merged,
        metric_name: str,
        metric_column: str,
        axis_label: str,
        window_half_width: float,
        step: float,
        min_color_count: int,
    ) -> None:
        """Save smoothed full/radar curves after removing the low-support tail."""
        centres, counts, full_medians, radar_medians, _ = rolling_stats(
            merged, metric_column, window_half_width, step
        )
        supported = counts >= min_color_count
        last_supported = int(np.flatnonzero(supported)[-1])

        fig, ax = plt.subplots(figsize=FIGSIZE)
        plot_curves(
            ax,
            centres,
            counts,
            full_medians,
            radar_medians,
            metric_name,
            axis_label,
            min_color_count,
        )
        ax.set_xlim(float(centres[0]) - 100, float(centres[last_supported]) + 100)
        ax.set_xlabel(
            r"MAX30105 IR Readings, $\propto$ Smoke Density",
            fontsize=X_LABEL_SIZE,
        )
        fig.tight_layout()
        output_base.parent.mkdir(parents=True, exist_ok=True)
        pdf_output = output_base.with_suffix(".pdf")
        png_output = output_base.with_suffix(".png")
        fig.savefig(pdf_output, bbox_inches="tight")
        fig.savefig(png_output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        normalize_canvas(png_output, CANVAS_WIDTH, CANVAS_HEIGHT)
        print(f"Saved {pdf_output}")
        print(f"Saved {png_output}")


    def plot_delta(
        ax: plt.Axes,
        centres: np.ndarray,
        counts: np.ndarray,
        delta_medians: np.ndarray,
        metric_name: str,
        lower_is_better: bool,
        min_color_count: int,
    ) -> None:
        supported = counts >= min_color_count
        hurts = delta_medians >= 0 if lower_is_better else delta_medians < 0
        helps = ~hurts

        # Draw the complete curve first so low-support estimates remain visible.
        ax.plot(
            centres,
            delta_medians,
            color=LOW_SUPPORT_COLOR,
            linestyle="--",
            linewidth=LINE_WIDTH - 1,
            marker="o",
            markersize=MARKER_SIZE,
            label=f"Low support ($N<{min_color_count}$)",
            zorder=2,
        )

        ax.fill_between(
            centres,
            delta_medians,
            0,
            where=supported & hurts,
            color=WORSE_COLOR,
            alpha=0.35,
            interpolate=True,
            label="Radar-only better",
        )
        ax.fill_between(
            centres,
            delta_medians,
            0,
            where=supported & helps,
            color=BETTER_COLOR,
            alpha=0.30,
            interpolate=True,
            label="Full model better",
        )
        ax.plot(
            centres,
            np.ma.masked_where(~supported, delta_medians),
            color=DELTA_COLOR,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            zorder=3,
        )
        ax.axhline(0, color="red", linestyle="--", linewidth=AXIS_LINEWIDTH, zorder=2)
        ax.set_ylabel(
            rf"$\Delta${metric_name}",
            fontsize=FONT_SIZE,
        )
        ax.set_title(
            f"Full - Radar-only {metric_name}",
            fontsize=FONT_SIZE - 4,
            fontweight="bold",
        )
        ax.legend(
            loc="best",
            fontsize=LEGEND_SIZE - 6,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            borderpad=0.25,
            labelspacing=0.1,
            handlelength=1.4,
            handletextpad=0.25,
        )
        mark_thresholds(ax)
        style_axis(ax)


    def save_figure(
        output_base: Path,
        merged,
        metric_name: str,
        metric_column: str,
        axis_label: str,
        lower_is_better: bool,
        window_half_width: float,
        step: float,
        min_color_count: int,
    ) -> None:
        centres, counts, _full_medians, _radar_medians, delta_medians = rolling_stats(
            merged, metric_column, window_half_width, step
        )
        ir_lo = float(merged["IR"].min())
        ir_hi = float(merged["IR"].max())

        # The absolute Ours_full vs Ours_radar panel was dropped from the paper;
        # only the paired-difference panel is emitted. plot_curves() is retained
        # for ad-hoc inspection.
        fig, ax = plt.subplots(figsize=FIGSIZE)
        plot_delta(
            ax,
            centres,
            counts,
            delta_medians,
            metric_name,
            lower_is_better,
            min_color_count,
        )
        ax.set_xlim(ir_lo - 100, ir_hi + 100)
        ax.set_xlabel(
            r"MAX30105 IR Readings, $\propto$ Smoke Density",
            fontsize=X_LABEL_SIZE,
        )
        fig.tight_layout()
        output_base.parent.mkdir(parents=True, exist_ok=True)
        pdf_output = output_base.with_suffix(".pdf")
        png_output = output_base.with_suffix(".png")
        fig.savefig(pdf_output, bbox_inches="tight")
        fig.savefig(png_output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        normalize_canvas(png_output, CANVAS_WIDTH, CANVAS_HEIGHT)
        print(f"Saved {pdf_output}")
        print(f"Saved {png_output}")


    def summarise(
        merged,
        metric_name: str,
        metric_column: str,
        window_half_width: float,
        step: float,
        min_color_count: int,
    ) -> None:
        centres, counts, _, _, delta_medians = rolling_stats(
            merged, metric_column, window_half_width, step
        )
        print(
            f"\n{metric_name}: {len(centres)} rolling windows, "
            f"{len(merged):,} paired frames"
        )
        print("IR centre  N frames  median delta  display")
        for centre, count, delta in zip(centres, counts, delta_medians):
            display = "color" if count >= min_color_count else "gray"
            print(f"{centre:9.0f}  {count:8,d}  {delta:+12.5f}  {display}")


    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Rolling-median graceful-degradation figures for evaluation metrics."
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=OUTPUT_DIR,
        )
        parser.add_argument(
            "--metrics",
            nargs="+",
            choices=list(METRIC_SPECS),
            default=list(METRIC_SPECS),
            help="Metrics to plot (default: all).",
        )
        parser.add_argument(
            "--window-half-width",
            type=float,
            default=WINDOW_HALF_WIDTH,
            help="Half-width of each rolling IR window (default: 250).",
        )
        parser.add_argument(
            "--step",
            type=float,
            default=WINDOW_STEP,
            help="IR spacing between rolling-window evaluations (default: 100).",
        )
        parser.add_argument(
            "--min-color-count",
            type=int,
            default=MIN_COLOR_COUNT,
            help="Minimum paired-frame support for colored interpretation (default: 25).",
        )
        parser.add_argument("--quiet", action="store_true")
        args = parser.parse_args()

        merged = paired_frame()
        plt.rcParams.update(
            {
                "font.family": "STIXGeneral",
                "mathtext.fontset": "stix",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        for metric_name in args.metrics:
            metric_column, axis_label, slug, lower_is_better = METRIC_SPECS[metric_name]
            output_base = args.output_dir / f"figure_new_degradation_{slug}_difference"
            save_figure(
                output_base,
                merged,
                metric_name,
                metric_column,
                axis_label,
                lower_is_better,
                args.window_half_width,
                args.step,
                args.min_color_count,
            )
            if metric_name == "MAE":
                save_absolute_figure(
                    args.output_dir / "figure_new_degradation_mae_absolute",
                    merged,
                    metric_name,
                    metric_column,
                    axis_label,
                    args.window_half_width,
                    args.step,
                    args.min_color_count,
                )
            if not args.quiet:
                summarise(
                    merged,
                    metric_name,
                    metric_column,
                    args.window_half_width,
                    args.step,
                    args.min_color_count,
                )
        if not args.quiet:
            print()
    return locals()


# ---- Embedded former module: robustness_sparsity ----
def _build_robustness_sparsity_module() -> dict[str, Any]:
    """Radar point-count distribution and configurable sparsity robustness.

    Outputs:
      data/robustness/radar_point_counts.npz
      data/robustness/radar_point_count_distribution.csv
      data/robustness/radar_sparsity_{N}_bins.csv
      data/robustness/grade_mae_by_point_count.csv
      outputs/robustness/radar_point_count_cdf.png
      outputs/robustness/radar_sparsity_{primary}_{secondary}_{tag}.png
      outputs/robustness/grade_mae_vs_point_count.png

    The CDF uses every frame in the test set. The performance table and plot divide
    the observed point-count range into ``--num-bins`` equal-width bins and report
    per-frame medians with Q1--Q3 intervals for the six paper baselines. The
    binned performance plot runs from Dense to Coarse, while the CDF and exact
    GRADE curve use numeric radar-point counts. Pass ``--pre-compute`` to rebuild
    the CSV tables; otherwise the script only loads the saved tables and draws the
    figures.

    The separate GRADE curve is not binned: it reports MAE at every exact observed
    integer radar-point count.
    """


    import argparse
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import MaxNLocator


    SCRIPT_DIR = EVALUATION_DIR
    EVALUATION_ROOT = EVALUATION_DIR
    METRIC_RESULTS_DIR = EVALUATION_ROOT / "metric_results"
    REFERENCE_RESULTS_DIR = EVALUATION_ROOT / "reference_results"
    DEFAULT_DATA_ROOT = EVALUATION_ROOT.parent / "evaluation_dataset" / "Smoke-Eval"
    RAW_2D_ROOT = METRIC_RESULTS_DIR / "simple_eval_results"
    RAW_3D_ROOT = METRIC_RESULTS_DIR / "simple_eval_results_3d"
    REFERENCE_DATA_DIR = Path(
        os.environ.get(
            "GRADE_RADAR_ROBUSTNESS_DIR",
            str(REFERENCE_RESULTS_DIR / "pre_eval_results" / "radar_robustness"),
        )
    )
    DATA_DIR = REFERENCE_DATA_DIR
    OUTPUT_DIR = EVALUATION_ROOT / "reproduced_results" / "figures" / "revision"
    LEGACY_CACHE = REFERENCE_DATA_DIR / "radar_point_counts.npz"

    MODELS = [
        ("grt", "GRT"),
        ("cafnet", "CaFNet"),
        ("radarcam-depth", "RadarCam-Depth"),
        ("grt_image", "GRT+Image"),
        ("ours_radar", "Ours_radar"),
        ("ours_diffusion", "Ours_diffusion"),
        ("ours_full", "GRADE"),
    ]
    # Figure default: RadarCam-Depth and GRT+Image are still computed and cached
    # under --pre-compute (so the data isn't lost), just not drawn unless asked for
    # via --plot-model. binned_model_summary() always covers all of MODELS.
    DEFAULT_PLOT_MODELS = ["ours_radar", "ours_diffusion", "ours_full"]
    METRICS = ["MAE", "AbsRel", "SSIM", "LPIPS", "DGE", "CD", "MHD"]
    STYLES = {
        "GRT": {"color": "#D95319", "marker": "s", "linestyle": "--"},
        "CaFNet": {"color": "#EDB120", "marker": "^", "linestyle": ":"},
        "RadarCam-Depth": {
            "color": "#4DBEEE",
            "marker": "P",
            "linestyle": (0, (3, 1, 1, 1, 1, 1)),
        },
        "GRT+Image": {"color": "#77AC30", "marker": "D", "linestyle": (0, (5, 1))},
        "Ours_radar": {"color": "#D95319", "marker": "s", "linestyle": "--"},
        # Match CaFNet's legend style for the diffusion-only stage.
        "Ours_diffusion": {"color": "#EDB120", "marker": "^", "linestyle": ":"},
        "GRADE": {"color": "#0072BD", "marker": "o", "linestyle": "-"},
    }

    FONT_SIZE = 22
    TICK_SIZE = 19
    LEGEND_SIZE = 17
    LINE_WIDTH = 2.6
    MARKER_SIZE = 9

    # Presets for plot_binned_performance only (the CDF and exact-count figures are
    # supporting material and keep the standalone scale). "paper" matches the panel
    # geometry and type sizes in figure_9.py so the output sits at 0.49\linewidth
    # next to the rest of Section 5's panels.
    FIGURE_STYLES = {
        "paper": {
            "figsize": (8.0, 5.5),
            "font_size": 40,
            "tick_size": 36,
            "xtick_size": 30,
            "legend_size": 26,
            "line_width": 4.2,
            "marker_size": 14,
            "bin_range_labels": True,
        },
        "standalone": {
            "figsize": (7.4, 5.4),
            "font_size": 22,
            "tick_size": 19,
            "xtick_size": 19,
            "legend_size": 17,
            "line_width": 2.6,
            "marker_size": 9,
            "bin_range_labels": False,
        },
    }
    DEFAULT_FIGURE_STYLE = "paper"

    # Figure 13 (radar-robustness) pairs this dual-metric sparsity panel with the
    # single-metric range panel from robustness_long_range.py side by side at
    # 0.49\linewidth each. Both panels share these exact type sizes so axis
    # labels, ticks, and the legend read at the same scale across the pair; kept
    # in sync by hand with PAIR_* in robustness_long_range.py.
    # Figure 13 uses ONE type size for every text element: legend, axis labels and
    # both sets of tick labels. Kept in sync by hand with robustness_long_range.py.
    PAIR_FONT_SIZE = 32.2   # matches Figure 11 label size (7.58pt) on the page
    PAIR_TICK_SIZE = PAIR_FONT_SIZE
    PAIR_LEGEND_SIZE = PAIR_FONT_SIZE
    PAIR_LINE_WIDTH = 5.5
    # Error-bar weight, shared by both Figure 13 panels: thicker whiskers and
    # wider, thicker Q1/Q3 caps so the spread reads at print size.
    PAIR_ELINEWIDTH = 3.5
    PAIR_CAPSIZE = 9
    PAIR_CAPTHICK = 3.5
    PAIR_MARKER_SIZE = 11

    # Figure 13's two panels must present an IDENTICAL axis box even though their
    # canvases differ (this panel carries a twin y-axis, the range panel does not).
    # Both scripts therefore fix the axes rectangle in absolute inches and derive
    # the canvas from it by adding per-panel margins; the LaTeX \includegraphics
    # widths are then set proportional to the canvas widths so the axis boxes land
    # at the same size on the page.
    PAIR_AXES_W_IN = 5.2
    PAIR_AXES_H_IN = 2.6      # 5.2 : 2.6 = 2:1 axis box
    PAIR_MARGINS = {"left": 1.55, "right": 0.22, "bottom": 1.18, "top": 0.14}


    def pair_figure_and_rect(margins: dict) -> tuple[tuple[float, float], list[float]]:
        """Canvas size and axes rect [l,b,w,h] giving a fixed axis box in inches."""
        w = PAIR_AXES_W_IN + margins["left"] + margins["right"]
        h = PAIR_AXES_H_IN + margins["bottom"] + margins["top"]
        rect = [margins["left"] / w, margins["bottom"] / h,
                PAIR_AXES_W_IN / w, PAIR_AXES_H_IN / h]
        return (w, h), rect

    # Beyond this many bins the two-line "name + point range" tick labels collide,
    # so the axis falls back to labelling only the dense and sparse extremes.
    MAX_LABELLED_BINS = 3


    def sequence_names(data_root: Path) -> list[str]:
        return sorted(
            path.name
            for path in data_root.iterdir()
            if path.is_dir() and (path / "zed_depth.npy").is_file()
        )


    def load_cached_counts(path: Path) -> dict[str, np.ndarray]:
        with np.load(path) as cache:
            return {name: cache[name] for name in cache.files}


    def radar_point_counts(data_root: Path, cache_path: Path) -> dict[str, np.ndarray]:
        """Return one radar-detection count per test frame, cached locally."""
        if cache_path.is_file():
            return load_cached_counts(cache_path)

        if LEGACY_CACHE.is_file():
            counts = load_cached_counts(LEGACY_CACHE)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **counts)
            print(f"Migrated point-count cache -> {cache_path}")
            return counts

        counts: dict[str, np.ndarray] = {}
        for sequence in sequence_names(data_root):
            sequence_dir = data_root / sequence
            pcd_dir = sequence_dir / "pcd"
            if not pcd_dir.is_dir():
                print(f"WARNING: no pcd/ for {sequence}; skipping")
                continue
            frame_count = len(np.load(sequence_dir / "zed_depth.npy", mmap_mode="r"))
            values = np.zeros(frame_count, dtype=np.int32)
            for frame_index in range(frame_count):
                path = pcd_dir / f"pcd_{frame_index}.npy"
                values[frame_index] = len(np.load(path)) if path.is_file() else 0
            counts[sequence] = values
            print(
                f"{sequence}: n={frame_count:,}, mean={values.mean():.2f}, "
                f"range=[{values.min()}, {values.max()}]"
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **counts)
        print(f"Saved {cache_path}")
        return counts


    def point_count_frame(counts: dict[str, np.ndarray]) -> pd.DataFrame:
        blocks = [
            pd.DataFrame(
                {
                    "Sequence": sequence,
                    "Frame_Index": np.arange(len(values), dtype=np.int64),
                    "Radar_Points": values,
                }
            )
            for sequence, values in counts.items()
        ]
        if not blocks:
            raise ValueError("No radar point counts were found")
        return pd.concat(blocks, ignore_index=True)


    def save_distribution(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
        frequency = (
            frame.groupby("Radar_Points")
            .size()
            .rename("Frame_Count")
            .reset_index()
            .sort_values("Radar_Points")
        )
        frequency["Fraction"] = frequency["Frame_Count"] / len(frame)
        frequency["CDF"] = frequency["Fraction"].cumsum()
        output.parent.mkdir(parents=True, exist_ok=True)
        frequency.to_csv(output, index=False)
        print(f"Saved {output}")
        return frequency


    def plot_cdf(frequency: pd.DataFrame, output: Path) -> None:
        """Plot the empirical CDF against the numeric radar point count."""
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        x = frequency["Radar_Points"].to_numpy(dtype=float)
        y = frequency["CDF"].to_numpy(dtype=float)
        ax.step(x, y, where="post", color="#0072BD", linewidth=LINE_WIDTH)
        ax.fill_between(x, 0, y, step="post", color="#0072BD", alpha=0.12)

        ax.set_xlim(float(x.min()), float(x.max()))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Number of radar points per frame", fontsize=FONT_SIZE)
        ax.set_ylabel(r"Empirical CDF, $P(K \leq k)$", fontsize=FONT_SIZE)
        ax.set_title("Radar point-count distribution", fontsize=FONT_SIZE + 1, fontweight="bold")
        ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.5, length=5)
        ax.grid(True, alpha=0.3, linewidth=0.9, color="#b0b0b0")
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output}")


    def assign_bins_at(
        frame: pd.DataFrame, cuts: list[int]
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Bin by explicit upper-inclusive point-count cuts, e.g. [6, 12].

        Equal-width bins over the observed range leave the top stratum nearly
        empty, because the point-count distribution has a long thin tail: the
        densest equal-width third holds a few hundred frames out of 26k. Explicit
        cuts let the strata be chosen for frame balance instead, so each carries
        enough frames for its median to mean something.
        """
        cuts = sorted(int(c) for c in cuts)
        minimum = float(frame["Radar_Points"].min())
        maximum = float(frame["Radar_Points"].max())
        edges = np.array([minimum] + [c + 1.0 for c in cuts] + [maximum + 1.0])
        result = frame.copy()
        result["Bin"] = np.clip(
            np.digitize(result["Radar_Points"], edges[1:-1]), 0, len(cuts)
        )
        return result, edges


    def assign_bins(frame: pd.DataFrame, num_bins: int) -> tuple[pd.DataFrame, np.ndarray]:
        if num_bins < 2:
            raise ValueError("--num-bins must be at least 2")
        unique_counts = int(frame["Radar_Points"].nunique())
        if num_bins > unique_counts:
            raise ValueError(
                f"--num-bins cannot exceed the {unique_counts} observed point-count values"
            )
        minimum = float(frame["Radar_Points"].min())
        maximum = float(frame["Radar_Points"].max())
        edges = np.linspace(minimum, maximum + 1.0, num_bins + 1)
        result = frame.copy()
        result["Bin"] = np.clip(
            np.digitize(result["Radar_Points"], edges[1:-1]),
            0,
            num_bins - 1,
        )
        return result, edges


    def distribution_fields(values: pd.Series, metric: str) -> dict[str, float]:
        array = values.to_numpy(dtype=float)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return {
                f"{metric}_mean": float("nan"),
                f"{metric}_q1": float("nan"),
                f"{metric}_median": float("nan"),
                f"{metric}_q3": float("nan"),
            }
        q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
        return {
            f"{metric}_mean": float(np.mean(array)),
            f"{metric}_q1": float(q1),
            f"{metric}_median": float(median),
            f"{metric}_q3": float(q3),
        }


    def load_model_frame(model_dir: str) -> pd.DataFrame:
        paths_2d = sorted(
            path
            for path in (RAW_2D_ROOT / f"csv_{model_dir}").glob("*.csv")
            if not path.name.startswith("._")
        )
        if not paths_2d:
            raise FileNotFoundError(f"No 2D CSVs for {model_dir}")
        frame = pd.concat([pd.read_csv(path) for path in paths_2d], ignore_index=True)

        paths_3d = sorted(
            path
            for path in (RAW_3D_ROOT / f"3d_csv_{model_dir}").glob("*.csv")
            if not path.name.startswith("._")
        )
        if paths_3d:
            frame_3d = pd.concat([pd.read_csv(path) for path in paths_3d], ignore_index=True)
            frame = frame.merge(
                frame_3d[["Sequence", "Frame_Index", "CD", "MHD"]],
                on=["Sequence", "Frame_Index"],
                how="left",
                validate="one_to_one",
            )
        return frame.rename(columns={"GradientError": "DGE"})


    def binned_model_summary(
        models: list[tuple[str, str]],
        binned_counts: pd.DataFrame,
        edges: np.ndarray,
    ) -> pd.DataFrame:
        rows = []
        num_bins = len(edges) - 1
        for model_dir, label in models:
            frame = load_model_frame(model_dir).merge(
                binned_counts,
                on=["Sequence", "Frame_Index"],
                how="inner",
                validate="one_to_one",
            )
            for bin_id in range(num_bins):
                subset = frame.loc[frame["Bin"] == bin_id]
                if subset.empty:
                    continue
                row = {
                    "Model": label,
                    "Variant": model_dir,
                    "Bin": bin_id,
                    "Bin_Low": float(edges[bin_id]),
                    "Bin_High": float(edges[bin_id + 1]),
                    "Point_Min": int(subset["Radar_Points"].min()),
                    "Point_Max": int(subset["Radar_Points"].max()),
                    "Point_Median": float(subset["Radar_Points"].median()),
                    "Frame_N": len(subset),
                }
                for metric in METRICS:
                    if metric in subset:
                        row.update(distribution_fields(subset[metric], metric))
                rows.append(row)
        return pd.DataFrame(rows)


    def grade_mae_by_point_count(point_frame: pd.DataFrame) -> pd.DataFrame:
        """Summarize GRADE MAE separately at every exact observed point count."""
        grade = load_model_frame("ours_full")[
            ["Sequence", "Frame_Index", "MAE"]
        ].merge(
            point_frame,
            on=["Sequence", "Frame_Index"],
            how="inner",
            validate="one_to_one",
        )
        rows = []
        for point_count, subset in grade.groupby("Radar_Points", sort=True):
            row = {
                "Model": "GRADE",
                "Variant": "ours_full",
                "Radar_Points": int(point_count),
                "Frame_N": len(subset),
            }
            row.update(distribution_fields(subset["MAE"], "MAE"))
            rows.append(row)
        return pd.DataFrame(rows)


    def plot_grade_mae_by_point_count(summary: pd.DataFrame, output: Path) -> None:
        """Plot exact-count GRADE MAE medians with middle-50% error bars."""
        required = {"Radar_Points", "MAE_q1", "MAE_median", "MAE_q3"}
        missing = required.difference(summary.columns)
        if missing:
            raise ValueError(f"Missing {sorted(missing)} from exact-count GRADE table")

        summary = summary.sort_values("Radar_Points")
        x = summary["Radar_Points"].to_numpy(dtype=float)
        median = summary["MAE_median"].to_numpy(dtype=float)
        q1 = summary["MAE_q1"].to_numpy(dtype=float)
        q3 = summary["MAE_q3"].to_numpy(dtype=float)

        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        style = STYLES["GRADE"]
        ax.errorbar(
            x,
            median,
            yerr=np.vstack([median - q1, q3 - median]),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=LINE_WIDTH,
            markersize=6.5,
            capsize=2.5,
            capthick=1.2,
            elinewidth=1.2,
            label="GRADE",
        )
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.set_xlabel("Number of radar points per frame", fontsize=FONT_SIZE)
        ax.set_ylabel("MAE (m)", fontsize=FONT_SIZE)
        ax.set_title(
            "GRADE accuracy versus radar point count",
            fontsize=FONT_SIZE + 1,
            fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.5, length=5)
        ax.grid(True, alpha=0.3, linewidth=0.9, color="#b0b0b0")
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.legend(
            fontsize=LEGEND_SIZE,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
        )
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output}")


    def bin_tick_labels(summary: pd.DataFrame, dense_to_coarse: list[int]) -> list[str]:
        """Two-line 'name + observed point range' labels, densest bin first.

        The point ranges come from the summary rather than from the nominal bin
        edges, so a label always describes radar-point counts that actually occur
        in that bin. They are aggregated across models because a model that
        dropped a frame can otherwise report a narrower range than its peers for
        the same bin (RadarCam-Depth, for instance, has no 0-point frames).
        """
        extent = summary.groupby("Bin").agg(
            low=("Point_Min", "min"), high=("Point_Max", "max")
        )
        num_bins = len(dense_to_coarse)
        labels = []
        for position, bin_id in enumerate(dense_to_coarse):
            low = int(extent.loc[bin_id, "low"])
            high = int(extent.loc[bin_id, "high"])
            if position == 0:
                name, span = "Dense", rf"$k \geq {low}$"
            elif position == num_bins - 1:
                name, span = "Sparse", rf"$k \leq {high}$"
            else:
                name, span = "Medium", rf"${low} \leq k \leq {high}$"
            labels.append(f"{name}\n({span})")
        return labels


    def plot_binned_performance(
        summary: pd.DataFrame,
        metric: str,
        num_bins: int,
        output: Path,
        plot_models: list[tuple[str, str]] = MODELS,
        figure_style: str = DEFAULT_FIGURE_STYLE,
    ) -> None:
        median_column = f"{metric}_median"
        q1_column = f"{metric}_q1"
        q3_column = f"{metric}_q3"
        missing = {median_column, q1_column, q3_column}.difference(summary.columns)
        if missing:
            raise ValueError(f"Missing {sorted(missing)} from sparsity summary")

        style_preset = FIGURE_STYLES[figure_style]
        pair = figure_style == "paper"
        FONT_SIZE = PAIR_FONT_SIZE if pair else style_preset["font_size"]
        TICK_SIZE = FONT_SIZE if pair else style_preset["tick_size"]
        LEGEND_SIZE = FONT_SIZE if pair else style_preset["legend_size"]
        LINE_WIDTH = PAIR_LINE_WIDTH if pair else style_preset["line_width"]
        MARKER_SIZE = PAIR_MARKER_SIZE if pair else style_preset["marker_size"]
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        if pair:
            figsize, axes_rect = pair_figure_and_rect(PAIR_MARGINS)
        else:
            figsize, axes_rect = style_preset["figsize"], None
        fig, ax = plt.subplots(figsize=figsize)
        if axes_rect is not None:
            ax.set_position(axes_rect)
        dense_to_coarse = list(reversed(range(num_bins)))
        x = np.arange(num_bins, dtype=float)
        offsets = np.linspace(-0.06, 0.06, len(plot_models))

        for offset, (_, label) in zip(offsets, plot_models):
            model = summary.loc[summary["Model"] == label].set_index("Bin")
            medians = np.asarray([model.loc[bin_id, median_column] for bin_id in dense_to_coarse])
            q1 = np.asarray([model.loc[bin_id, q1_column] for bin_id in dense_to_coarse])
            q3 = np.asarray([model.loc[bin_id, q3_column] for bin_id in dense_to_coarse])
            style = STYLES[label]
            ax.errorbar(
                x + offset,
                medians,
                yerr=np.vstack([medians - q1, q3 - medians]),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                capsize=PAIR_CAPSIZE if pair else 4,
                capthick=PAIR_CAPTHICK if pair else 1.5,
                elinewidth=PAIR_ELINEWIDTH if pair else 1.5,
                label=label,
            )

        ax.set_xlim(-0.35, num_bins - 0.65)
        if style_preset["bin_range_labels"] and num_bins <= MAX_LABELLED_BINS:
            ax.set_xticks(x)
            ax.set_xticklabels(
                bin_tick_labels(summary, dense_to_coarse),
                fontsize=FONT_SIZE,
            )
            # The tick labels already carry the point-count ranges, so an axis
            # label repeating "radar point-count" is redundant at paper scale.
            if not pair:
                ax.set_xlabel("Radar points per frame", fontsize=FONT_SIZE)
        else:
            ax.set_xticks([0, num_bins - 1])
            ax.set_xticklabels(["Dense", "Sparse"], fontsize=FONT_SIZE)
            ax.set_xlabel(f"Radar point-count bins (N={num_bins})", fontsize=FONT_SIZE)
        ax.set_ylabel(
            f"{metric} (m)" if metric in {"MAE", "CD", "MHD"} else metric,
            fontsize=FONT_SIZE,
        )
        if pair:
            # Fixed y ticks so the two Figure 13 panels read on comparable, round
            # scales; the data (q3 max 0.567) sits inside 0.6.
            ax.set_ylim(0.0, 0.62)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6])
        if not pair:
            ax.set_title("Robustness to radar sparsity", fontsize=FONT_SIZE + 1, fontweight="bold")
        ax.tick_params(axis="y", labelsize=TICK_SIZE, width=1.5, length=5)
        ax.grid(True, alpha=0.3, linewidth=0.9, color="#b0b0b0")
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        if not pair:
            # In the Figure 13 pair, one shared legend image sits above both panels.
            ax.legend(
                fontsize=LEGEND_SIZE, frameon=True, fancybox=False,
                framealpha=1.0, edgecolor="black",
            )
            fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        # No bbox_inches="tight" in pair mode: it would undo the fixed axis box.
        if pair:
            fig.savefig(output, dpi=220)
        else:
            fig.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output}")



    def plot_pair_legend(output: Path, plot_models: list[tuple[str, str]]) -> None:
        """Shared legend for both Figure 13 panels, written as its own image.

        Covers every series drawn across the pair: solid markers for the primary
        metric (also what the range panel's bars encode) and open markers with a
        dotted line for the secondary metric.
        """
        from matplotlib.lines import Line2D
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        # GRADE first, followed by the two intermediate stages, all drawn in the
        # primary-metric (MAE) style. LPIPS is no longer plotted.
        order = {"GRADE": 0, "Ours_radar": 1, "Ours_diffusion": 2}
        handles, labels = [], []
        for _, label in sorted(plot_models, key=lambda kv: order.get(kv[1], 9)):
            style = STYLES[label]
            handles.append(Line2D([0], [0], color=style["color"], marker=style["marker"],
                                  markerfacecolor=style["color"], markeredgewidth=1.8,
                                  linestyle=style["linestyle"],
                                  linewidth=PAIR_FONT_SIZE * 0.375,
                                  markersize=PAIR_FONT_SIZE * 0.5625))
            labels.append(label)
        # One row: Figures 12-14 all render their legend strip at the same fixed
        # height in LaTeX, so a second row would halve this legend's text size
        # relative to the others.
        # Spacing and handle proportions copied from Figure 14's legend
        # (figure_12.py plot_legend): short line samples, tight text padding.
        fig = plt.figure(figsize=(13.0, 0.62))
        fig.legend(handles, labels, loc="center", ncol=len(handles), frameon=False,
                   fontsize=PAIR_FONT_SIZE, borderpad=0.25, labelspacing=0.0,
                   handlelength=1.2, handletextpad=0.2, columnspacing=0.25)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved {output}")


    def plot_binned_dual_metric(
        summary: pd.DataFrame,
        primary_metric: str,
        secondary_metric: str,
        output: Path,
        plot_models: list[tuple[str, str]] = MODELS,
        figure_style: str = DEFAULT_FIGURE_STYLE,
    ) -> None:
        """Plot two metrics against the same point-count strata on twin axes."""
        required = {
            f"{metric}_{stat}"
            for metric in (primary_metric, secondary_metric)
            for stat in ("q1", "median", "q3")
        }
        missing = required.difference(summary.columns)
        if missing:
            raise ValueError(f"Missing {sorted(missing)} from sparsity summary")

        style_preset = FIGURE_STYLES[figure_style]
        # Twin axes and a four-entry legend need a slightly smaller paper preset
        # than the single-metric panel to remain readable at 0.49\linewidth; see
        # PAIR_* above for the sizes shared with the range panel.
        if figure_style == "paper":
            font_size = PAIR_FONT_SIZE
            tick_size = PAIR_TICK_SIZE
            legend_size = PAIR_LEGEND_SIZE
            line_width = PAIR_LINE_WIDTH
            marker_size = PAIR_MARKER_SIZE
        else:
            font_size = style_preset["font_size"]
            tick_size = style_preset["tick_size"]
            legend_size = style_preset["legend_size"]
            line_width = style_preset["line_width"]
            marker_size = style_preset["marker_size"]
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        if figure_style == "paper":
            figsize, axes_rect = pair_figure_and_rect(PAIR_MARGINS)
        else:
            figsize, axes_rect = style_preset["figsize"], None
        fig, ax_primary = plt.subplots(figsize=figsize)
        if axes_rect is not None:
            ax_primary.set_position(axes_rect)
        ax_secondary = ax_primary.twinx()
        if axes_rect is not None:
            ax_secondary.set_position(axes_rect)
        bin_ids = sorted(int(value) for value in summary["Bin"].unique())
        dense_to_coarse = list(reversed(bin_ids))
        x = np.arange(len(bin_ids), dtype=float)
        offsets = np.linspace(-0.055, 0.055, len(plot_models))
        handles = []
        labels = []

        for offset, (_, label) in zip(offsets, plot_models):
            model = summary.loc[summary["Model"] == label].set_index("Bin")
            style = STYLES[label]
            for axis, metric, linestyle, open_marker in (
                (ax_primary, primary_metric, style["linestyle"], False),
                (ax_secondary, secondary_metric, ":", True),
            ):
                median = np.asarray(
                    [model.loc[bin_id, f"{metric}_median"] for bin_id in dense_to_coarse]
                )
                q1 = np.asarray(
                    [model.loc[bin_id, f"{metric}_q1"] for bin_id in dense_to_coarse]
                )
                q3 = np.asarray(
                    [model.loc[bin_id, f"{metric}_q3"] for bin_id in dense_to_coarse]
                )
                artist = axis.errorbar(
                    x + offset,
                    median,
                    yerr=np.vstack([median - q1, q3 - median]),
                    color=style["color"],
                    marker=style["marker"],
                    markerfacecolor="white" if open_marker else style["color"],
                    markeredgewidth=1.8,
                    linestyle=linestyle,
                    linewidth=line_width,
                    markersize=marker_size,
                    capsize=4,
                    capthick=1.5,
                    elinewidth=1.5,
                    label=f"{label} {metric}",
                )
                handles.append(artist)
                labels.append(f"{label} {metric}")

        ax_primary.set_xlim(-0.35, len(bin_ids) - 0.65)
        ax_primary.set_xticks(x)
        ax_primary.set_xticklabels(
            bin_tick_labels(summary, dense_to_coarse),
            fontsize=font_size,
        )
        ax_primary.set_ylabel(
            f"{primary_metric} (m)" if primary_metric in {"MAE", "CD", "MHD"} else primary_metric,
            fontsize=font_size,
        )
        ax_secondary.set_ylabel(
            f"{secondary_metric} (m)" if secondary_metric in {"MAE", "CD", "MHD"} else secondary_metric,
            fontsize=font_size,
        )
        ax_primary.tick_params(axis="y", labelsize=tick_size, width=1.5, length=5)
        ax_secondary.tick_params(axis="y", labelsize=tick_size, width=1.5, length=5)
        ax_primary.grid(True, alpha=0.3, linewidth=0.9, color="#b0b0b0")
        for axis in (ax_primary, ax_secondary):
            for spine in axis.spines.values():
                spine.set_linewidth(1.5)
        # No in-panel legend: Figure 13 carries one shared legend image above both
        # panels (see plot_pair_legend), so neither panel loses plotting area.
        if figure_style != "paper":
            fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        # No bbox_inches="tight": cropping would undo the fixed axis-box geometry.
        fig.savefig(output, dpi=220)
        plt.close(fig)
        print(f"Saved {output}")


    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Radar point-count CDF and configurable sparsity robustness."
        )
        parser.add_argument("--num-bins", type=int, default=3)
        parser.add_argument(
            "--bin-cuts", type=int, nargs="+", default=None, metavar="K",
            help="Upper-inclusive point-count cuts, e.g. --bin-cuts 6 12 gives "
                 "k<=6, 7-12, k>12. Overrides --num-bins equal-width binning.",
        )
        parser.add_argument("--metric", choices=METRICS, default="MAE")
        parser.add_argument(
            "--secondary-metric",
            choices=METRICS,
            default=None,
            help="Optional second metric drawn on a right-hand y-axis.",
        )
        parser.add_argument(
            "--plot-model", nargs="+", choices=[name for name, _ in MODELS],
            default=DEFAULT_PLOT_MODELS,
            help=f"Models to draw in the binned-performance figure (default: "
                 f"{' '.join(DEFAULT_PLOT_MODELS)}). The CDF and exact-count GRADE "
                 "figures are unaffected.",
        )
        parser.add_argument(
            "--pre-compute",
            action="store_true",
            help="Recompute and overwrite the saved distribution and sparsity CSVs.",
        )
        parser.add_argument(
            "--paper-only",
            action="store_true",
            help=(
                "Draw only the final Figure 13 sparsity panel and shared legend. "
                "This requires only the selected sparsity summary, not the "
                "exploratory CDF or exact-count support files."
            ),
        )
        parser.add_argument(
            "--figure-style",
            choices=sorted(FIGURE_STYLES),
            default=DEFAULT_FIGURE_STYLE,
            help=(
                "Applies to the binned-performance figure. 'paper' matches the "
                "panel size and type scale of the other Section 5 figures and "
                "labels every bin with its observed point-count range; "
                "'standalone' renders a wider canvas with smaller type and "
                f"labels only the extremes. Default: {DEFAULT_FIGURE_STYLE}."
            ),
        )
        parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
        parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
        parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
        return parser.parse_args()


    def load_precomputed_csv(path: Path, description: str) -> pd.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing precomputed {description}: {path}\n"
                "Run this command again with --pre-compute to create it."
            )
        print(f"Loaded {path}")
        return pd.read_csv(path)


    def main() -> None:
        args = parse_args()
        distribution_path = args.data_dir / "radar_point_count_distribution.csv"
        tag = ("cuts_" + "_".join(str(c) for c in args.bin_cuts)) if args.bin_cuts else f"{args.num_bins}_bins"
        summary_path = args.data_dir / f"radar_sparsity_{tag}.csv"
        grade_exact_path = args.data_dir / "grade_mae_by_point_count.csv"

        if args.pre_compute:
            if args.data_dir.resolve() == REFERENCE_DATA_DIR.resolve():
                raise ValueError(
                    "Refusing to overwrite golden robustness inputs. Pass --data-dir "
                    "evaluation/metric_results/robustness_generated to precompute fresh data."
                )
            counts = radar_point_counts(
                args.data_root,
                args.data_dir / "radar_point_counts.npz",
            )
            point_frame = point_count_frame(counts)
            distribution = save_distribution(point_frame, distribution_path)
            if args.bin_cuts:
                binned_counts, edges = assign_bins_at(point_frame, args.bin_cuts)
            else:
                binned_counts, edges = assign_bins(point_frame, args.num_bins)
            summary = binned_model_summary(MODELS, binned_counts, edges)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(summary_path, index=False)
            print(f"Saved {summary_path}")
            grade_exact = grade_mae_by_point_count(point_frame)
            grade_exact.to_csv(grade_exact_path, index=False)
            print(f"Saved {grade_exact_path}")
        else:
            summary = load_precomputed_csv(
                summary_path,
                f"radar sparsity table ({tag})",
            )
            if not args.paper_only:
                distribution = load_precomputed_csv(
                    distribution_path,
                    "radar point-count distribution",
                )
                grade_exact = load_precomputed_csv(
                    grade_exact_path,
                    "exact-count GRADE MAE table",
                )

        if not args.paper_only:
            plot_cdf(
                distribution,
                args.output_dir / "radar_point_count_cdf.png",
            )
        plot_models = [item for item in MODELS if item[0] in args.plot_model]
        missing = {label for _, label in plot_models}.difference(summary["Model"].unique())
        if missing:
            raise ValueError(
                f"{summary_path} does not contain {sorted(missing)}. "
                "Run this command again with --pre-compute to add them."
            )
        if args.secondary_metric:
            plot_binned_dual_metric(
                summary,
                args.metric,
                args.secondary_metric,
                args.output_dir
                / f"radar_sparsity_{args.metric.lower()}_{args.secondary_metric.lower()}_{tag}.png",
                plot_models=plot_models,
                figure_style=args.figure_style,
            )
            plot_pair_legend(args.output_dir / "radar_robustness_legend.png", plot_models)
        else:
            plot_binned_performance(
                summary,
                args.metric,
                len(summary["Bin"].unique()),
                args.output_dir / f"radar_sparsity_{args.metric.lower()}_{tag}.png",
                plot_models=plot_models,
                figure_style=args.figure_style,
            )
            plot_pair_legend(args.output_dir / "radar_robustness_legend.png", plot_models)
        if not args.paper_only:
            plot_grade_mae_by_point_count(
                grade_exact,
                args.output_dir / "grade_mae_vs_point_count.png",
            )
    return locals()


# ---- Embedded former module: robustness_long_range ----
def _build_robustness_long_range_module() -> dict[str, Any]:
    """Depth accuracy versus range: statistics and figures from stored histograms.

    Everything downstream of the GPU pass lives here. robustness_range_precompute.py
    writes one histogram archive per model; this script reads those, collapses the
    20 cm base bins to whatever width is asked for, derives the statistics, and
    draws the figure. Nothing here touches Smoke-Eval or inference_results, and the
    whole thing runs in seconds, so bin width and metric are free to change.

    The unit of analysis is the *pixel*, not the frame. Every valid ground-truth
    pixel is placed in a depth bin by its own depth value and contributes one error
    sample, so a frame counts in proportion to how many pixels it actually has in a
    bin rather than being averaged to a single value first.

    Merging is exact. Histograms are additive, so summing consecutive 20 cm rows
    gives the histogram of the union of their samples, and quantiles read off its
    cumulative counts are the quantiles that pooling raw pixels would give. Stored
    summary statistics could not be merged this way -- two medians cannot be
    combined into one.

    What it produces:
      --plot curve      median with IQR and/or bootstrap CI versus range (default)
      --plot density    P(error | depth) heatmap, the full conditional distribution
      --compare         paired cluster bootstrap between the two models
      a CSV of every statistic, alongside whichever figure was drawn

    Outputs:
      outputs/robustness/range_{width}m_{metric}.png
      outputs/robustness/range_density_{width}m_{metric}.png
      data/robustness/range_{width}m_pixel.csv
      data/robustness/range_{width}m_paired_{metric}.csv
    """


    import argparse
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LogNorm


    SCRIPT_DIR = EVALUATION_DIR
    EVALUATION_ROOT = EVALUATION_DIR
    REFERENCE_RESULTS_DIR = EVALUATION_ROOT / "reference_results"
    DATA_DIR = Path(
        os.environ.get(
            "GRADE_RADAR_ROBUSTNESS_DIR",
            str(REFERENCE_RESULTS_DIR / "pre_eval_results" / "radar_robustness"),
        )
    )
    DERIVED_OUTPUT_DIR = EVALUATION_ROOT / "metric_results" / "robustness_derived"
    OUTPUT_DIR = EVALUATION_ROOT / "reproduced_results" / "figures" / "revision"

    # Kept in sync by hand with the MODELS list in robustness_range_precompute.py.
    MODELS = [
        ("ours_radar", "Ours_radar"),
        ("ours_diffusion", "Ours_diffusion"),
        ("ours_full", "GRADE"),
    ]
    STYLES = {
        # Matches RADAR_COLOR in figure_new_degradation.py so the radar-only stage
        # keeps one colour across the evaluation section.
        "Ours_radar": {"color": "#D95319", "marker": "s", "linestyle": "--"},
        # Match CaFNet's legend style for the diffusion-only stage.
        "Ours_diffusion": {"color": "#EDB120", "marker": "^", "linestyle": ":"},
        "GRADE": {"color": "#0072BD", "marker": "o", "linestyle": "-"},
    }

    BASE_BIN_M = 0.2

    # Both histograms use a 1e-3 bucket width -- millimetres for the absolute
    # error, 0.001 for the ratio -- so one scale serves both. AbsRel is unbounded
    # and its final bucket is an overflow bin, so a quantile landing there is
    # reported as NaN rather than as the cap.
    METRICS = {
        "mae": {
            "prefix": "MAE",
            "ylabel": "MAE (m)",
            "counts": "counts",
            "by_sequence": "counts_by_sequence",
            "total": "error_sum_mm",
            "total_scale": 1e-3,
            "saturating": False,
            "density_row": 0.1,
            "density_max": 8.0,
            "density_label": "Absolute error (m)",
        },
        "absrel": {
            "prefix": "AbsRel",
            "ylabel": "AbsRel",
            "counts": "absrel_counts",
            "by_sequence": "absrel_counts_by_sequence",
            "total": "absrel_sum",
            "total_scale": 1.0,
            "saturating": True,
            "density_row": 0.02,
            "density_max": 1.5,
            "density_label": "AbsRel",
        },
    }
    BUCKET_SCALE = 1e-3
    QUANTILES = {"q1": 0.25, "median": 0.50, "q3": 0.75, "p90": 0.90}

    # What the vertical extent of each point means:
    #   iqr  - Q1..Q3 of the pixel population; how spread the errors are.
    #   ci   - bootstrap 95% CI on the median; how well the median is pinned down.
    # They answer different questions and differ by an order of magnitude, so
    # drawing both keeps a wide spread from being misread as an imprecise estimate.
    # Left unset by default and resolved from whether a bootstrap was actually
    # requested: "iqr" on a plain run, "both" once CI columns exist. The bootstrap
    # is opt-in, so the ordinary paper figure never mentions it.
    BAND_CHOICES = ("iqr", "ci", "both")

    # Two rendering presets. "paper" matches the panel geometry and type sizes in
    # figure_9.py, so the output drops into the two-column layout at
    # 0.49\linewidth looking like the rest of Section 5's panels. "standalone" is
    # the larger-canvas, smaller-type version that reads better on its own screen.
    # Figure 13 pairs this panel with the sparsity panel. Both fix the SAME axis
    # box in inches and differ only in margins, so that when their LaTeX widths are
    # set proportional to their canvas widths the two axis boxes render identically.
    # One type size is used for every text element, matching PAIR_FONT_SIZE there.
    PAIR_FONT_SIZE = 32.2   # matches Figure 11 label size (7.58pt) on the page
    PAIR_LINE_WIDTH = 5.5
    # Error-bar weight, shared by both Figure 13 panels: thicker whiskers and
    # wider, thicker Q1/Q3 caps so the spread reads at print size.
    PAIR_ELINEWIDTH = 3.5
    PAIR_CAPSIZE = 9
    PAIR_CAPTHICK = 3.5
    PAIR_MARKER_SIZE = 11
    PAIR_AXES_W_IN = 5.2
    PAIR_AXES_H_IN = 2.6      # 5.2 : 2.6 = 2:1 axis box
    PAIR_MARGINS = {"left": 1.55, "right": 0.22, "bottom": 1.18, "top": 0.14}


    def pair_figure_and_rect(margins: dict) -> tuple[tuple[float, float], list[float]]:
        w = PAIR_AXES_W_IN + margins["left"] + margins["right"]
        h = PAIR_AXES_H_IN + margins["bottom"] + margins["top"]
        rect = [margins["left"] / w, margins["bottom"] / h,
                PAIR_AXES_W_IN / w, PAIR_AXES_H_IN / h]
        return (w, h), rect


    FIGURE_STYLES = {
        "paper": {
            "figsize": (8.0, 6.0),
            "font_size": 40,
            "tick_size": 36,
            "legend_size": 26,
            "line_width": 6,
            "marker_size": 18,
            "title": "Accuracy versus range",
        },
        "standalone": {
            "figsize": (8.2, 5.4),
            "font_size": 22,
            "tick_size": 18,
            "legend_size": 17,
            "line_width": 2.6,
            "marker_size": 8,
            "title": "Depth accuracy versus range",
        },
    }
    DEFAULT_FIGURE_STYLE = "paper"

    # Quantiles overlaid on the density heatmap.
    OVERLAYS = [
        (0.90, "P90", (0, (1, 1.2)), 0.85),
        (0.75, "Q3", (0, (5, 2)), 0.85),
        (0.50, "Median", "-", 1.0),
        (0.25, "Q1", (0, (5, 2)), 0.85),
    ]


    # ---------------------------------------------------------------------------
    # Loading and statistics
    # ---------------------------------------------------------------------------

    def load_histograms(variant: str, data_dir: Path) -> dict[str, np.ndarray]:
        path = data_dir / f"range_hist_20cm_{variant}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing histogram: {path}\n"
                f"Run: python robustness_range_precompute.py --model {variant}"
            )
        archive = np.load(path, allow_pickle=False)
        return {key: archive[key] for key in archive.files} | {"__path__": path}


    def require(store: dict, key: str, variant: str) -> np.ndarray:
        if key not in store:
            raise KeyError(
                f"{store['__path__'].name} has no '{key}'. Rerun "
                f"robustness_range_precompute.py --model {variant} to add it."
            )
        return store[key]


    def group_rows(counts: np.ndarray, bins_per_group: int) -> np.ndarray:
        """Merge consecutive 20 cm depth bins; histograms add, so this is exact."""
        starts = np.arange(0, counts.shape[0], bins_per_group)
        return np.add.reduceat(counts, starts, axis=0)


    def quantiles_from_histogram(
        histogram: np.ndarray,
        probabilities: dict[str, float],
        saturating_last_bucket: bool = False,
    ) -> dict[str, float]:
        """Inverse-CDF quantiles: smallest value whose CDF reaches p.

        With a population this size the choice between this and an interpolating
        definition is far below the reported precision. A quantile landing in an
        overflow bucket is not measured, only bounded, so it returns NaN -- the cap
        would read as a real value.
        """
        total = int(histogram.sum())
        if total == 0:
            return {name: float("nan") for name in probabilities}
        cumulative = np.cumsum(histogram)
        overflow_index = len(histogram) - 1
        result = {}
        for name, p in probabilities.items():
            index = int(np.searchsorted(cumulative, p * total, side="left"))
            result[name] = (
                float("nan")
                if saturating_last_bucket and index >= overflow_index
                else index * BUCKET_SCALE
            )
        return result


    def resample_weights(
        n_sequences: int, replicates: int, seed: int
    ) -> np.ndarray:
        """Multiplicity of each sequence in each bootstrap replicate."""
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n_sequences, size=(replicates, n_sequences))
        weights = np.zeros((replicates, n_sequences), dtype=np.int64)
        np.add.at(weights, (np.arange(replicates)[:, None], draws), 1)
        return weights


    def medians_of_replicates(merged: np.ndarray) -> np.ndarray:
        """Median of each row of a (replicates, buckets) count block."""
        totals = merged.sum(axis=1)
        cumulative = np.cumsum(merged, axis=1)
        return (cumulative < (0.5 * totals)[:, None]).sum(axis=1) * BUCKET_SCALE


    def bootstrap_median_ci(
        per_sequence: np.ndarray, replicates: int, seed: int = 0
    ) -> tuple[float, float]:
        """Cluster-bootstrap CI for the median, resampling *sequences*.

        Resampling pixels would be wrong: pixels within a sequence are massively
        correlated, so a pixel bootstrap would report an interval near zero width
        on a billion samples and imply a precision the data does not support. The
        sequence is the unit that varies independently, so that is what is
        resampled -- the interval then reflects whether a bin's behaviour holds
        across scenes or is driven by one or two corridors.
        """
        n_sequences = per_sequence.shape[0]
        if n_sequences < 2 or per_sequence.sum() == 0:
            return float("nan"), float("nan")
        merged = resample_weights(n_sequences, replicates, seed) @ per_sequence
        low, high = np.quantile(medians_of_replicates(merged), [0.025, 0.975])
        return float(low), float(high)


    def summarise(
        stores: list[tuple[str, str, dict]],
        bins_per_group: int,
        replicates: int = 0,
    ) -> pd.DataFrame:
        """One row per (model, merged bin), with every available metric."""
        rows: list[dict[str, object]] = []
        for variant, label, store in stores:
            edges_mm = store["edges_mm"]
            pixel_count = store["pixel_count"]
            n_base_bins = len(pixel_count)
            grand_total = int(pixel_count.sum())

            for group_id, start in enumerate(range(0, n_base_bins, bins_per_group)):
                stop = min(start + bins_per_group, n_base_bins)
                count = int(pixel_count[start:stop].sum())
                row = {
                    "Model": label,
                    "Variant": variant,
                    "Bin": group_id,
                    "Range_Low_m": float(edges_mm[start]) / 1000.0,
                    # The final group can be narrower than the rest: 11.2 m is not
                    # a multiple of every width. The true covered edge is reported
                    # rather than the nominal one, so a 2 m table's last row reads
                    # 10.0-11.2 instead of claiming a 10-12 m span it only
                    # partially covers.
                    "Range_High_m": float(edges_mm[stop]) / 1000.0,
                    "Range_Centre_m": float(edges_mm[start] + edges_mm[stop]) / 2000.0,
                    "Pixel_N": count,
                    "Pixel_Frac": count / grand_total if grand_total else float("nan"),
                }

                for spec in METRICS.values():
                    if spec["counts"] not in store:
                        continue
                    prefix = spec["prefix"]
                    totals = store[spec["total"]]
                    # Exact float64 running sum from the precompute, so the mean
                    # does not inherit the bucket rounding the quantiles carry.
                    row[f"{prefix}_mean"] = (
                        float(totals[start:stop].sum()) * spec["total_scale"] / count
                        if count else float("nan")
                    )
                    merged = store[spec["counts"]][start:stop].sum(axis=0)
                    row.update({
                        f"{prefix}_{name}": value
                        for name, value in quantiles_from_histogram(
                            merged, QUANTILES, spec["saturating"]
                        ).items()
                    })
                    if replicates and spec["by_sequence"] in store:
                        low, high = bootstrap_median_ci(
                            store[spec["by_sequence"]][:, start:stop].sum(axis=1),
                            replicates,
                        )
                        row[f"{prefix}_median_lo"] = low
                        row[f"{prefix}_median_hi"] = high
                rows.append(row)

        frame = pd.DataFrame(rows).sort_values(["Model", "Bin"]).reset_index(drop=True)
        return frame


    def paired_compare(
        stores: list[tuple[str, str, dict]],
        metric: str,
        bins_per_group: int,
        replicates: int,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Paired cluster bootstrap of the difference between two models.

        Comparing the models' marginal confidence intervals and asking whether they
        overlap is the wrong test, and a conservative one: both are scored on
        exactly the same pixels of the same sequences, so scene-to-scene variation
        moves them together. A hard corridor inflates both medians at once and
        cancels out of the difference, but widens each margin independently. Here
        one resample is drawn per replicate and applied to *both* models, so the
        bootstrap distribution is over the paired difference.
        """
        (base_variant, base_label, base_store), (other_variant, other_label, other_store) = stores
        spec = METRICS[metric]
        a_counts = require(base_store, spec["by_sequence"], base_variant)
        b_counts = require(other_store, spec["by_sequence"], other_variant)
        if not np.array_equal(base_store["sequences"], other_store["sequences"]):
            # Pairing is by position, so a mismatch would silently compare one
            # model's corridor against another's.
            raise ValueError(
                f"Sequence lists differ between {base_variant} and {other_variant}; "
                "recompute both so the pairing is valid."
            )

        n_sequences, n_base_bins = a_counts.shape[0], a_counts.shape[1]
        weights = resample_weights(n_sequences, replicates, seed)

        rows = []
        for group_id, start in enumerate(range(0, n_base_bins, bins_per_group)):
            stop = min(start + bins_per_group, n_base_bins)
            delta = (
                medians_of_replicates(weights @ b_counts[:, start:stop].sum(axis=1))
                - medians_of_replicates(weights @ a_counts[:, start:stop].sum(axis=1))
            )
            low, high = np.quantile(delta, [0.025, 0.975])
            rows.append({
                "Base": base_label,
                "Other": other_label,
                "Metric": metric,
                "Bin": group_id,
                "Range_Low_m": start * BASE_BIN_M,
                "Range_High_m": min(stop * BASE_BIN_M, 11.2),
                "Median_Delta": float(np.median(delta)),
                "Median_Delta_lo": float(low),
                "Median_Delta_hi": float(high),
                # Share of replicates where the second model wins. Near 1.0 or 0.0
                # is the bootstrap analogue of a small p-value.
                "Win_Frac": float((delta < 0).mean()),
            })
        return pd.DataFrame(rows)


    # ---------------------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------------------

    def plot_curve(
        summary: pd.DataFrame,
        output: Path,
        metric: str,
        band: str,
        figure_style: str,
    ) -> None:
        prefix = METRICS[metric]["prefix"]
        has_ci = {f"{prefix}_median_lo", f"{prefix}_median_hi"} <= set(summary.columns)
        if band in ("ci", "both") and not has_ci:
            # Only reachable when the band was asked for explicitly -- the default
            # resolves to "iqr" whenever no bootstrap ran, so a plain run is silent.
            print(f"  no {prefix} CI columns; falling back to IQR (add --bootstrap N).")
            band = "iqr"

        style_preset = FIGURE_STYLES[figure_style]
        pair = figure_style == "paper"
        FONT_SIZE = PAIR_FONT_SIZE if pair else style_preset["font_size"]
        TICK_SIZE = FONT_SIZE if pair else style_preset["tick_size"]
        LINE_WIDTH = style_preset["line_width"]
        MARKER_SIZE = style_preset["marker_size"]
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        if pair:
            figsize, axes_rect = pair_figure_and_rect(PAIR_MARGINS)
        else:
            figsize, axes_rect = style_preset["figsize"], None
        fig, ax = plt.subplots(figsize=figsize)
        if axes_rect is not None:
            ax.set_position(axes_rect)
        available = [item for item in MODELS if item[1] in set(summary["Model"])]
        offsets = np.linspace(-0.06, 0.06, max(len(available), 1))

        # No low-support fading. The earlier per-frame tables faded far bins because
        # a median over per-frame means was noise once a frame held only a handful
        # of pixels in that bin. Pooling pixels removes that failure mode: even the
        # last bin is estimated from millions of samples.
        for offset, (_, label) in zip(offsets, available):
            model = summary.loc[summary["Model"] == label].sort_values("Bin")
            x = model["Range_Centre_m"].to_numpy(dtype=float) + offset
            median = model[f"{prefix}_median"].to_numpy(dtype=float)
            style = STYLES[label]

            # Spread as a translucent fill rather than whiskers: it is the wider of
            # the two extents by far, and drawn as caps it would swamp the CI.
            if band in ("iqr", "both") and not pair:
                ax.fill_between(
                    x,
                    model[f"{prefix}_q1"].to_numpy(dtype=float),
                    model[f"{prefix}_q3"].to_numpy(dtype=float),
                    color=style["color"],
                    alpha=0.03 if band == "both" else 0.1,
                    linewidth=0,
                )

            error = None
            if band in ("ci", "both"):
                error = np.vstack([
                    median - model[f"{prefix}_median_lo"].to_numpy(dtype=float),
                    model[f"{prefix}_median_hi"].to_numpy(dtype=float) - median,
                ])

            if pair:
                # Match the sparsity panel exactly: capped Q1-Q3 whiskers rather
                # than a translucent band, same widths and cap geometry.
                error = np.vstack([
                    median - model[f"{prefix}_q1"].to_numpy(dtype=float),
                    model[f"{prefix}_q3"].to_numpy(dtype=float) - median,
                ])
                ax.errorbar(
                    x, median, yerr=error,
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], linewidth=PAIR_LINE_WIDTH,
                    markersize=PAIR_MARKER_SIZE, capsize=PAIR_CAPSIZE,
                    capthick=PAIR_CAPTHICK, elinewidth=PAIR_ELINEWIDTH,
                    label=label,
                )
            else:
                ax.errorbar(
                    x, median, yerr=error,
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE, capsize=5.0, capthick=1.8, elinewidth=1.8,
                    label=label,
                )

        # At paper type sizes the bin width does not fit on the axis without
        # overrunning the panel, so it moves to the LaTeX caption instead.
        bin_width = float((summary["Range_High_m"] - summary["Range_Low_m"]).iloc[0])
        ax.set_xlabel(
            "Ground-truth depth (m)" if figure_style == "paper"
            else f"Ground-truth depth ({bin_width:g}-meter bins)",
            fontsize=FONT_SIZE,
        )
        ax.set_ylabel(METRICS[metric]["ylabel"], fontsize=FONT_SIZE)
        if pair:
            # Matches the sparsity panel's convention: round ticks from zero. Leave
            # room above the 0.4 tick for Ours_diffusion's far-range Q3 (0.428).
            ax.set_ylim(0.0, 0.46)
            ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        if not pair:
            ax.set_title(style_preset["title"], fontsize=FONT_SIZE + 1, fontweight="bold")
        ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.5, length=5)
        ax.grid(True, alpha=0.3, linewidth=0.9, color="#b0b0b0")
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        if not pair:
            # Figure 13 carries one shared legend image above both panels.
            ax.legend(
                fontsize=style_preset["legend_size"], frameon=True, fancybox=False,
                framealpha=1.0, edgecolor="black",
            )
            fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        if pair:
            fig.savefig(output, dpi=220)
        else:
            fig.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output}")


    def rebin_error_axis(
        counts: np.ndarray, row_buckets: int, max_bucket: int
    ) -> np.ndarray:
        """Coarsen the error axis, folding everything above max_bucket into a tail.

        The tail row keeps each column summing to its full pixel count, so the
        column normalisation stays a true conditional density.
        """
        edges = np.arange(0, max_bucket, row_buckets)
        coarse = np.add.reduceat(counts[:, :max_bucket], edges, axis=1)
        overflow = counts[:, max_bucket:].sum(axis=1, keepdims=True)
        return np.concatenate([coarse, overflow], axis=1)


    def plot_density(
        data: list[tuple[str, np.ndarray]],
        metric: str,
        bin_width_m: float,
        output: Path,
    ) -> None:
        """Heatmap of P(error | depth), one panel per model.

        The median-and-IQR view collapses each depth bin to three numbers, which
        hides the fact that the error distribution changes *shape* with range, not
        just location. Each column here is normalised to sum to 1, so colour is the
        conditional density and columns are comparable even though their pixel
        counts differ by three orders of magnitude.

        Density (a magnitude) uses a single-hue sequential ramp; model identity is
        carried by the overlaid line colour and the panel title, so the two
        encodings never compete. Colour is log-scaled because the density spans
        several decades.
        """
        spec = METRICS[metric]
        row_buckets = int(round(spec["density_row"] / BUCKET_SCALE))
        max_bucket = int(round(spec["density_max"] / BUCKET_SCALE))

        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
        fig, axes = plt.subplots(
            1, len(data), figsize=(7.2 * len(data), 5.6), sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes)

        mesh = None
        for ax, (label, counts) in zip(axes, data):
            rows = rebin_error_axis(counts, row_buckets, max_bucket).astype(float)
            totals = rows.sum(axis=1, keepdims=True)
            density = np.divide(rows, totals, out=np.zeros_like(rows), where=totals > 0)

            depth_edges = np.arange(counts.shape[0] + 1) * bin_width_m
            error_edges = (
                np.arange(density.shape[1] + 1) * row_buckets * BUCKET_SCALE
            )
            mesh = ax.pcolormesh(
                depth_edges, error_edges, density.T,
                cmap="Greys", norm=LogNorm(vmin=1e-5, vmax=1.0), shading="flat",
            )

            centres = (depth_edges[:-1] + depth_edges[1:]) / 2.0
            for probability, name, linestyle, width_scale in OVERLAYS:
                curve = np.array([
                    quantiles_from_histogram(
                        counts[column], {"v": probability}, spec["saturating"]
                    )["v"]
                    for column in range(counts.shape[0])
                ])
                ax.plot(
                    centres, curve, color=STYLES[label]["color"],
                    linestyle=linestyle, linewidth=2.4 * width_scale,
                    label=name if ax is axes[0] else None, solid_capstyle="round",
                )

            ax.set_title(label, fontsize=19, fontweight="bold",
                         color=STYLES[label]["color"])
            ax.set_xlabel("Ground-truth depth (m)", fontsize=17)
            ax.tick_params(axis="both", labelsize=14)
            ax.grid(True, alpha=0.18, linewidth=0.7, color="#ffffff")
            ax.set_axisbelow(False)

        axes[0].set_ylabel(spec["density_label"], fontsize=17)
        # One legend, on the left panel only: the linestyles mean the same thing in
        # both, and repeating it would just cover data.
        legend = axes[0].legend(
            fontsize=13, frameon=True, framealpha=0.92, edgecolor="#444444",
            loc="upper left",
        )
        for text in legend.get_texts():
            text.set_color("#222222")

        bar = fig.colorbar(mesh, ax=axes.tolist(), pad=0.02, aspect=30)
        bar.set_label(f"P(error | depth)  per {bin_width_m:g} m column", fontsize=15)
        bar.ax.tick_params(labelsize=13)

        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output}")


    # ---------------------------------------------------------------------------
    # CLI
    # ---------------------------------------------------------------------------

    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Statistics and figures for depth accuracy versus range, "
                        "derived from the stored 20 cm error histograms."
        )
        parser.add_argument(
            "--bin-width", type=float, default=2.0,
            help=f"Reported bin width in metres, a multiple of {BASE_BIN_M} "
                 f"(default: 2.0). Recombination is exact, so this is free to "
                 f"change without rerunning the precompute.",
        )
        parser.add_argument(
            "--metric", choices=sorted(METRICS), default="mae",
            help="Which per-pixel error to plot and compare (default: mae). The "
                 "CSV always carries every metric present in the histograms.",
        )
        parser.add_argument(
            "--plot", choices=("curve", "density", "none"), default="curve",
            help="'curve' draws median versus range; 'density' draws the full "
                 "P(error | depth) heatmap; 'none' writes the CSV only.",
        )
        parser.add_argument(
            "--band", choices=BAND_CHOICES, default=None,
            help="Vertical extent on the curve: 'iqr' shades Q1-Q3 (spread of the "
                 "pixel population), 'ci' caps the bootstrap 95%% CI on the median "
                 "(how precisely it is pinned down), 'both' draws them together. "
                 "Default: 'iqr', or 'both' when --bootstrap was given.",
        )
        parser.add_argument(
            "--bootstrap", type=int, default=0, metavar="N",
            help="Opt-in. Cluster-bootstrap replicates for a 95%% CI on the "
                 "median, resampling sequences (not pixels). 0 disables (default), "
                 "so neither the figure nor the CSV mentions it unless asked. "
                 "1000 is plenty.",
        )
        parser.add_argument(
            "--compare", action="store_true",
            help="Opt-in. Runs the paired cluster bootstrap between the two models "
                 "and writes range_{width}m_paired_{metric}.csv. Far more powerful "
                 "than checking whether the marginal intervals overlap.",
        )
        parser.add_argument(
            "--compare-replicates", type=int, default=2000,
            help="Replicates for --compare (default: 2000).",
        )
        parser.add_argument(
            "--model", nargs="+", choices=[name for name, _ in MODELS],
            default=[name for name, _ in MODELS],
            help="Models to load and draw (default: all three).",
        )
        parser.add_argument(
            "--figure-style", choices=sorted(FIGURE_STYLES),
            default=DEFAULT_FIGURE_STYLE,
            help="'paper' matches the panel size and type scale of the other "
                 "Section 5 figures; 'standalone' renders a wider canvas with "
                 f"smaller type. Default: {DEFAULT_FIGURE_STYLE}.",
        )
        parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
        parser.add_argument(
            "--derived-output-dir", type=Path, default=DERIVED_OUTPUT_DIR,
            help="Destination for newly derived range-summary CSVs.",
        )
        parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
        return parser.parse_args()


    def main() -> None:
        args = parse_args()
        ratio = args.bin_width / BASE_BIN_M
        bins_per_group = int(round(ratio))
        if bins_per_group < 1 or abs(ratio - bins_per_group) > 1e-6:
            raise ValueError(
                f"--bin-width must be a positive multiple of {BASE_BIN_M} m, "
                f"got {args.bin_width}"
            )

        stores = [
            (variant, label, load_histograms(variant, args.data_dir))
            for variant, label in MODELS
            if variant in args.model
        ]
        width_label = f"{args.bin_width:g}m"
        spec = METRICS[args.metric]
        print(
            f"Merging {bins_per_group} x {BASE_BIN_M:g} m bins -> "
            f"{args.bin_width:g} m bins; metric: {args.metric}"
        )

        summary = summarise(stores, bins_per_group, args.bootstrap)
        prefix = spec["prefix"]
        if f"{prefix}_median" not in summary.columns:
            raise ValueError(
                f"The histograms carry no {prefix} data. Rerun "
                f"robustness_range_precompute.py to add it."
            )
        for _, row in summary.iterrows():
            interval = (
                f"  [{row[f'{prefix}_median_lo']:.3f}, {row[f'{prefix}_median_hi']:.3f}]"
                if f"{prefix}_median_lo" in summary.columns else ""
            )
            print(
                f"  {row['Model']:12s} {row['Range_Low_m']:5.1f}-"
                f"{row['Range_High_m']:5.1f} m  N={row['Pixel_N']:>13,} "
                f"({row['Pixel_Frac']:6.2%})  {prefix} med={row[f'{prefix}_median']:.3f}"
                f"{interval}  mean={row[f'{prefix}_mean']:.3f}"
            )

        args.derived_output_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = args.derived_output_dir / f"range_{width_label}_pixel.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"Saved {summary_csv}")

        if args.compare:
            if len(stores) != 2:
                raise ValueError("--compare needs exactly two models loaded.")
            paired = paired_compare(
                stores, args.metric, bins_per_group, args.compare_replicates
            )
            other = paired["Other"].iloc[0]
            print(
                f"\nPaired bootstrap [{args.metric}]: {other} - "
                f"{paired['Base'].iloc[0]}, {args.compare_replicates} replicates "
                f"over {len(stores[0][2]['sequences'])} sequences"
            )
            print(f"{'range (m)':>12}  {'median diff (95% CI)':>30}  {'P(' + other + ' better)':>22}")
            for _, row in paired.iterrows():
                interval = (
                    f"{row['Median_Delta']:+.3f} "
                    f"[{row['Median_Delta_lo']:+.3f}, {row['Median_Delta_hi']:+.3f}]"
                )
                star = "*" if row["Median_Delta_hi"] < 0 or row["Median_Delta_lo"] > 0 else " "
                print(
                    f"{row['Range_Low_m']:5.1f}-{row['Range_High_m']:5.1f}  "
                    f"{interval:>30}{star}  {row['Win_Frac']:>21.1%}"
                )
            print("* interval excludes zero")
            paired_csv = args.derived_output_dir / f"range_{width_label}_paired_{args.metric}.csv"
            paired.to_csv(paired_csv, index=False)
            print(f"Saved {paired_csv}")

        if args.plot == "curve":
            band = args.band or ("both" if args.bootstrap else "iqr")
            plot_curve(
                summary,
                args.output_dir / f"range_{width_label}_{args.metric}.png",
                args.metric, band, args.figure_style,
            )
        elif args.plot == "density":
            data = [
                (label, group_rows(require(store, spec["counts"], variant), bins_per_group))
                for variant, label, store in stores
            ]
            plot_density(
                data, args.metric, args.bin_width,
                args.output_dir / f"range_density_{width_label}_{args.metric}.png",
            )
    return locals()


# ---- Embedded former module: comparison_report ----
def _build_comparison_report_module() -> dict[str, Any]:
    #!/usr/bin/env python3
    """Build a side-by-side HTML comparison of reproduced and golden artifacts."""


    import argparse
    import html
    import os
    from pathlib import Path
    from urllib.parse import quote


    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--reference-root", type=Path, required=True)
        parser.add_argument("--reproduced-root", type=Path, required=True)
        parser.add_argument("--output", type=Path, default=None)
        return parser.parse_args()


    def files_below(root: Path, suffixes: set[str]) -> set[Path]:
        if not root.is_dir():
            return set()
        return {
            path.relative_to(root)
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        }


    def href(path: Path) -> str:
        return quote(path.as_posix(), safe="/._-")


    def table_card(title: str, reference: Path, reproduced: Path) -> str:
        def text_or_missing(path: Path) -> str:
            if not path.is_file():
                return "<p class=\"missing\">Missing</p>"
            return f"<pre>{html.escape(path.read_text(encoding='utf-8', errors='replace'))}</pre>"

        return "\n".join(
            [
                "<section class=\"comparison\">",
                f"<h3>{html.escape(title)}</h3>",
                "<div class=\"pair\">",
                f"<article><h4>Golden reference</h4>{text_or_missing(reference)}</article>",
                f"<article><h4>Newly reproduced</h4>{text_or_missing(reproduced)}</article>",
                "</div></section>",
            ]
        )


    def figure_card(title: str, reference: Path, reproduced: Path, report_root: Path) -> str:
        def image_or_missing(path: Path) -> str:
            if not path.is_file():
                return "<p class=\"missing\">Missing</p>"
            try:
                relative = Path(os.path.relpath(path, report_root))
                location = href(relative)
            except ValueError:
                # A reviewer may place new outputs on a different Windows volume
                # than the immutable golden assets; relpath cannot cross volumes.
                location = html.escape(path.as_uri(), quote=True)
            return f"<a href=\"{location}\"><img src=\"{location}\" alt=\"{html.escape(title)}\"></a>"

        return "\n".join(
            [
                "<section class=\"comparison\">",
                f"<h3>{html.escape(title)}</h3>",
                "<div class=\"pair\">",
                f"<article><h4>Golden reference</h4>{image_or_missing(reference)}</article>",
                f"<article><h4>Newly reproduced</h4>{image_or_missing(reproduced)}</article>",
                "</div></section>",
            ]
        )


    def main() -> None:
        args = parse_args()
        reference_root = args.reference_root.resolve()
        reproduced_root = args.reproduced_root.resolve()
        output = (args.output or reproduced_root / "comparison.html").resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        reference_tables = reference_root / "paper_tables"
        reproduced_tables = reproduced_root / "tables"
        reference_figures = reference_root / "paper_figures"
        reproduced_figures = reproduced_root / "paper_figures"
        table_names = sorted(files_below(reference_tables, {".txt"}) | files_below(reproduced_tables, {".txt"}))
        figure_names = sorted(
            files_below(reference_figures, {".png", ".jpg", ".jpeg", ".svg"})
            & files_below(reproduced_figures, {".png", ".jpg", ".jpeg", ".svg"})
        )

        table_sections = "\n".join(
            table_card(name.as_posix(), reference_tables / name, reproduced_tables / name)
            for name in table_names
        ) or "<p>No table artifacts were found.</p>"
        figure_sections = "\n".join(
            figure_card(name.as_posix(), reference_figures / name, reproduced_figures / name, output.parent)
            for name in figure_names
        ) or "<p>No figure artifacts were found.</p>"

        document = f"""<!doctype html>
    <html lang=\"en\"><head><meta charset=\"utf-8\"><title>GRADE artifact comparison</title>
    <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #17202a; background: #fbfcfc; }}
    h1 {{ margin-bottom: .25rem; }} .note {{ color: #566573; }}
    .comparison {{ margin: 2rem 0; }} .pair {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }}
    article {{ background: white; border: 1px solid #d5d8dc; border-radius: .5rem; padding: 1rem; overflow: auto; }}
    h4 {{ margin-top: 0; }} pre {{ margin: 0; white-space: pre; font-family: ui-monospace, monospace; font-size: .8rem; }}
    img {{ max-width: 100%; height: auto; display: block; }} .missing {{ color: #b03a2e; font-weight: 600; }}
    @media (max-width: 900px) {{ .pair {{ grid-template-columns: 1fr; }} }}
    </style></head><body>
    <h1>GRADE evaluation: golden vs. reproduced</h1>
    <p class=\"note\">Golden inputs: {html.escape(str(reference_root))}<br>New outputs: {html.escape(str(reproduced_root))}</p>
    <h2>Tables</h2>{table_sections}
    <h2>Figures</h2>{figure_sections}
    </body></html>"""
        output.write_text(document, encoding="utf-8")
        print(f"Saved comparison report: {output}")
    return locals()


_EMBEDDED_BUILDERS: dict[str, Callable[[], dict[str, Any]]] = {
    "table_2": _build_table_2_module,
    "table_3": _build_table_3_module,
    "table_4": _build_table_4_module,
    "table_5": _build_table_5_module,
    "table_6": _build_table_6_module,
    "table_7": _build_table_7_module,
    "table_ablation_merged": _build_table_ablation_merged_module,
    "table_new_doppler": _build_table_new_doppler_module,
    "table_new_freeze": _build_table_new_freeze_module,
    "figure_9": _build_figure_9_module,
    "figure_11": _build_figure_11_module,
    "figure_12_pair": _build_figure_12_pair_module,
    "figure_12": _build_figure_12_module,
    "figure_new_degradation": _build_figure_new_degradation_module,
    "robustness_sparsity": _build_robustness_sparsity_module,
    "robustness_long_range": _build_robustness_long_range_module,
    "comparison_report": _build_comparison_report_module,
}


if __name__ == "__main__":
    main()
