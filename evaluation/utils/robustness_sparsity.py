"""Precompute the final radar point-sparsity summary for Figure 13.

Ported from the camera-ready evaluation script in
``eval_final_win/evaluation/robustness_sparsity.py``. This artifact version
uses canonical model names and writes fresh derived data under
``evaluation/metric_results/``; it never writes the golden reference inputs.

Outputs:
  radar_robustness/radar_sparsity_cuts_9_13.csv

``--pre-compute --paper-only --bin-cuts 9 13`` is the mode used by
``evaluation/run_metrics.py --radar-robustness``. It creates just the final
camera-ready point-sparsity table for Ours_radar, Ours_diffusion, and GRADE.
The optional non-paper plotting and cache paths remain available for local
analysis but are never written to ``reference_results/``.

The performance table divides each model's per-frame results into point-count
strata and reports medians with Q1--Q3 intervals. Explicit ``--bin-cuts`` use
the camera-ready sparse/medium/dense boundaries; equal-width ``--num-bins``
remains available for local exploration.

The separate GRADE curve is not binned: it reports MAE at every exact observed
integer radar-point count.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = EVALUATION_DIR.parent
DEFAULT_DATA_ROOT = ARTIFACT_ROOT / "evaluation_dataset" / "Smoke-Eval"
DEFAULT_RAW_ROOT = EVALUATION_DIR / "metric_results"
DATA_DIR = DEFAULT_RAW_ROOT / "radar_robustness"
OUTPUT_DIR = DATA_DIR / "figures"

MODELS = [
    ("ours_radar", "Ours_radar"),
    ("ours_diffusion", "Ours_diffusion"),
    ("ours_full", "GRADE"),
]
DEFAULT_PLOT_MODELS = [name for name, _ in MODELS]
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
    # Matches RADAR_COLOR in figure_new_degradation.py so the radar-only stage
    # keeps one colour across the evaluation section.
    "Ours_radar": {"color": "#D95319", "marker": "s", "linestyle": "--"},
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
        "figsize": (8.0, 6.0),
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

# Beyond this many bins the two-line "name + point range" tick labels collide,
# so the axis falls back to labelling only the dense and sparse extremes.
MAX_LABELLED_BINS = 3


def sequence_names(data_root: Path, selected: list[str] | None = None) -> list[str]:
    sequences = sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and (path / "zed_depth.npy").is_file()
    )
    if selected:
        requested = set(selected)
        sequences = [name for name in sequences if name in requested]
    return sequences


def load_cached_counts(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as cache:
        return {name: cache[name] for name in cache.files}


def radar_point_counts(
    data_root: Path,
    cache_path: Path | None,
    selected: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return one radar-detection count per test frame, cached locally."""
    if cache_path is not None and cache_path.is_file():
        return load_cached_counts(cache_path)

    counts: dict[str, np.ndarray] = {}
    for sequence in sequence_names(data_root, selected):
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

    if cache_path is not None:
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


def assign_bins_at(
    frame: pd.DataFrame, cuts: list[int]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create upper-inclusive strata, e.g. ``k<=9, 10<=k<=13, k>=14``."""
    cuts = sorted(int(cut) for cut in cuts)
    if not cuts:
        raise ValueError("--bin-cuts requires at least one cut")
    minimum = float(frame["Radar_Points"].min())
    maximum = float(frame["Radar_Points"].max())
    edges = np.array([minimum] + [cut + 1.0 for cut in cuts] + [maximum + 1.0])
    result = frame.copy()
    result["Bin"] = np.clip(
        np.digitize(result["Radar_Points"], edges[1:-1]), 0, len(cuts)
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


def load_model_frame(model_dir: str, raw_root: Path) -> pd.DataFrame:
    paths_2d = sorted(
        path for path in (raw_root / "simple_eval_results" / f"csv_{model_dir}").glob("*.csv")
        if not path.name.startswith("._")
    )
    if not paths_2d:
        raise FileNotFoundError(f"No 2D CSVs for {model_dir}")
    frame = pd.concat([pd.read_csv(path) for path in paths_2d], ignore_index=True)

    paths_3d = sorted(
        path for path in (raw_root / "simple_eval_results_3d" / f"3d_csv_{model_dir}").glob("*.csv")
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
    raw_root: Path,
) -> pd.DataFrame:
    rows = []
    num_bins = len(edges) - 1
    for model_dir, label in models:
        frame = load_model_frame(model_dir, raw_root).merge(
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


def grade_mae_by_point_count(point_frame: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    """Summarize GRADE MAE separately at every exact observed point count."""
    grade = load_model_frame("ours_full", raw_root)[
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
    FONT_SIZE = style_preset["font_size"]
    TICK_SIZE = style_preset["tick_size"]
    LEGEND_SIZE = style_preset["legend_size"]
    LINE_WIDTH = style_preset["line_width"]
    MARKER_SIZE = style_preset["marker_size"]
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
    fig, ax = plt.subplots(figsize=style_preset["figsize"])
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
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            label=label,
        )

    ax.set_xlim(-0.35, num_bins - 0.65)
    if style_preset["bin_range_labels"] and num_bins <= MAX_LABELLED_BINS:
        ax.set_xticks(x)
        ax.set_xticklabels(
            bin_tick_labels(summary, dense_to_coarse),
            fontsize=style_preset["xtick_size"],
        )
        # The tick labels already carry the point-count ranges, so an axis
        # label repeating "radar point-count" is redundant at paper scale.
        ax.set_xlabel("Radar points per frame", fontsize=FONT_SIZE)
    else:
        ax.set_xticks([0, num_bins - 1])
        ax.set_xticklabels(["Dense", "Sparse"], fontsize=style_preset["xtick_size"])
        ax.set_xlabel(f"Radar point-count bins (N={num_bins})", fontsize=FONT_SIZE)
    ax.set_ylabel(
        f"{metric} (m)" if metric in {"MAE", "CD", "MHD"} else metric,
        fontsize=FONT_SIZE,
    )
    ax.set_title("Robustness to radar sparsity", fontsize=FONT_SIZE + 1, fontweight="bold")
    ax.tick_params(axis="y", labelsize=TICK_SIZE, width=1.5, length=5)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute radar point-sparsity statistics for Figure 13."
    )
    parser.add_argument("--num-bins", type=int, default=3)
    parser.add_argument(
        "--bin-cuts", type=int, nargs="+", default=None, metavar="K",
        help="Upper-inclusive point-count cuts, e.g. --bin-cuts 9 13.",
    )
    parser.add_argument("--metric", choices=METRICS, default="MAE")
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
            "Create only the final Figure 13 sparsity CSV, without exploratory "
            "caches, CDFs, or exact-count summaries."
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
    parser.add_argument(
        "--raw-root", type=Path, default=DEFAULT_RAW_ROOT,
        help="Root containing simple_eval_results/ and simple_eval_results_3d/.",
    )
    parser.add_argument("--sequence", nargs="+", default=None, metavar="SEQ")
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
    tag = (
        "cuts_" + "_".join(str(cut) for cut in args.bin_cuts)
        if args.bin_cuts else f"{args.num_bins}_bins"
    )
    summary_path = args.data_dir / f"radar_sparsity_{tag}.csv"
    grade_exact_path = args.data_dir / "grade_mae_by_point_count.csv"
    raw_root = args.raw_root.resolve()

    if args.pre_compute:
        counts = radar_point_counts(
            args.data_root,
            None if args.paper_only else args.data_dir / "radar_point_counts.npz",
            args.sequence,
        )
        point_frame = point_count_frame(counts)
        if args.bin_cuts:
            binned_counts, edges = assign_bins_at(point_frame, args.bin_cuts)
        else:
            binned_counts, edges = assign_bins(point_frame, args.num_bins)
        summary = binned_model_summary(MODELS, binned_counts, edges, raw_root)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Saved {summary_path}")
        if not args.paper_only:
            distribution = save_distribution(point_frame, distribution_path)
            grade_exact = grade_mae_by_point_count(point_frame, raw_root)
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

    if args.paper_only:
        return

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
    plot_binned_performance(
        summary,
        args.metric,
        len(summary["Bin"].unique()),
        args.output_dir
        / f"radar_sparsity_{args.metric.lower()}_{tag}.png",
        plot_models=plot_models,
        figure_style=args.figure_style,
    )
    plot_grade_mae_by_point_count(
        grade_exact,
        args.output_dir / "grade_mae_vs_point_count.png",
    )


if __name__ == "__main__":
    main()
