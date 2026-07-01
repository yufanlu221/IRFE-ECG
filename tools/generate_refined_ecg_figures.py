"""Generate publication-quality robustness and memory figures.

The output is designed for an IEEE/BIBM single column. PDFs are vector graphics
with embedded TrueType fonts; PNGs are 600-dpi review previews. Figures report
means and sample standard deviations over the three fixed seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter


SEEDS = (42, 43, 44)
RUN_TAG = "paper"
TOP2_MODE = "domain_mlp_top2"

ORDER_RUNS = (
    ("Current", f"feature_router_primary_seed{{seed}}_mem1_{RUN_TAG}"),
    ("Reverse", f"feature_router_reverse_order_seed{{seed}}_mem1_{RUN_TAG}"),
    ("Fixed random", f"feature_router_random_order_seed{{seed}}_mem1_{RUN_TAG}"),
)

MEMORY_RUNS = (
    ("1%", f"feature_router_memory_seed{{seed}}_mem0p01_{RUN_TAG}"),
    ("5%", f"feature_router_memory_seed{{seed}}_mem0p05_{RUN_TAG}"),
    ("10%", f"feature_router_memory_seed{{seed}}_mem0p1_{RUN_TAG}"),
    ("100%", f"feature_router_primary_seed{{seed}}_mem1_{RUN_TAG}"),
)

POOLED_RUN = f"feature_pooled_linear_seed{{seed}}_{RUN_TAG}"

DOMAIN_GAPS = (
    ("CPSC", 0.7412, 0.7241, 0.0171),
    ("PTB-XL", 0.7509, 0.7510, -0.0001),
    ("Georgia", 0.7499, 0.7346, 0.0153),
    ("Chapman", 0.9241, 0.9032, 0.0209),
)

# Muted, color-blind-friendly categorical palette for domain comparisons.
DOMAIN_COLORS = ("#4C78A8", "#D98E04", "#2A9D8F", "#C85A54")

# Color-blind-safe, restrained palette.  Shapes and line styles remain distinct
# when the PDF is printed in grayscale.
NAVY = "#1F4E79"
TEAL = "#2A9D8F"
AMBER = "#E69F00"
GRAPHITE = "#4A5560"
MID_GREY = "#7B8790"
LIGHT_GREY = "#D9E0E4"
VERY_LIGHT_GREY = "#F2F5F6"


@dataclass(frozen=True)
class OrderRecord:
    order: str
    seed: int
    oracle: float
    top2: float

    @property
    def gap(self) -> float:
        return self.oracle - self.top2


@dataclass(frozen=True)
class MemoryRecord:
    budget: str
    seed: int
    retained_features: int
    memory_mib: float
    macro_f1: float
    route_accuracy: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Directory containing the fixed 2026-06-28 run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "figures",
    )
    parser.add_argument("--run-tag", default="paper")
    return parser.parse_args()


def _configure_run_tag(run_tag: str) -> None:
    global RUN_TAG, ORDER_RUNS, MEMORY_RUNS, POOLED_RUN
    RUN_TAG = run_tag
    ORDER_RUNS = (
        ("Current", f"feature_router_primary_seed{{seed}}_mem1_{run_tag}"),
        ("Reverse", f"feature_router_reverse_order_seed{{seed}}_mem1_{run_tag}"),
        ("Fixed random", f"feature_router_random_order_seed{{seed}}_mem1_{run_tag}"),
    )
    MEMORY_RUNS = (
        ("1%", f"feature_router_memory_seed{{seed}}_mem0p01_{run_tag}"),
        ("5%", f"feature_router_memory_seed{{seed}}_mem0p05_{run_tag}"),
        ("10%", f"feature_router_memory_seed{{seed}}_mem0p1_{run_tag}"),
        ("100%", f"feature_router_primary_seed{{seed}}_mem1_{run_tag}"),
    )
    POOLED_RUN = f"feature_pooled_linear_seed{{seed}}_{run_tag}"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def _available_font() -> str:
    names = {font.name for font in mpl.font_manager.fontManager.ttflist}
    for candidate in ("Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral"):
        if candidate in names:
            return candidate
    return "DejaVu Serif"


def set_publication_style() -> str:
    font = _available_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font],
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.05,
            "lines.markersize": 4.3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "stix",
        }
    )
    return font


def collect_order_records(outputs_root: Path) -> List[OrderRecord]:
    records: List[OrderRecord] = []
    for order, pattern in ORDER_RUNS:
        for seed in SEEDS:
            summary = _load_json(outputs_root / pattern.format(seed=seed) / "metrics_summary.json")
            records.append(
                OrderRecord(
                    order=order,
                    seed=seed,
                    oracle=float(summary["settings"]["oracle"]["mean_f1"]),
                    top2=float(summary["settings"][TOP2_MODE]["mean_f1"]),
                )
            )
    return records


def collect_memory_records(outputs_root: Path) -> tuple[List[MemoryRecord], List[float]]:
    records: List[MemoryRecord] = []
    for budget, pattern in MEMORY_RUNS:
        for seed in SEEDS:
            summary = _load_json(outputs_root / pattern.format(seed=seed) / "metrics_summary.json")
            top2 = summary["settings"][TOP2_MODE]
            final_domain = summary["experiment_config"]["domain_order"][-1]
            memory_counts = summary["router_metadata"][final_domain]["memory_counts"]
            retained = int(sum(int(value) for value in memory_counts.values()))
            route_values = list(top2["final_route_accuracy_by_domain"].values())
            records.append(
                MemoryRecord(
                    budget=budget,
                    seed=seed,
                    retained_features=retained,
                    memory_mib=retained * 1024 * 4 / (1024**2),
                    macro_f1=float(top2["mean_f1"]),
                    route_accuracy=float(np.mean(route_values)),
                )
            )

    pooled = []
    for seed in SEEDS:
        summary = _load_json(
            outputs_root / POOLED_RUN.format(seed=seed) / "metrics_summary.json"
        )
        pooled.append(float(summary["mean_f1"]))
    return records, pooled


def _clean_axis(axis: mpl.axes.Axes, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color=LIGHT_GREY, linewidth=0.55, linestyle="-")
    axis.set_axisbelow(True)


def _save_figure(fig: mpl.figure.Figure, output_base: Path, title: str) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": title,
        "Author": "IRFE-ECG",
        "Subject": "Publication figure generated from fixed three-seed results",
        "Creator": "Matplotlib / generate_refined_ecg_figures.py",
    }
    fig.savefig(
        output_base.with_suffix(".pdf"),
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
        metadata=metadata,
    )
    fig.savefig(
        output_base.with_suffix(".png"),
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )


def generate_order_dumbbell(
    records: Sequence[OrderRecord],
    output_dir: Path,
) -> None:
    """Plot only the paired oracle-minus-top2 gap for each domain order."""

    fig, axis = plt.subplots(figsize=(3.35, 2.0), constrained_layout=True)
    order_names = [item[0] for item in ORDER_RUNS]
    positions = np.arange(len(order_names), dtype=float)
    means = []
    stds = []
    for order in order_names:
        gaps = [row.gap for row in records if row.order == order]
        mean, std = _mean_std(gaps)
        means.append(mean)
        stds.append(std)

    means_array = np.asarray(means)
    stds_array = np.asarray(stds)
    axis.errorbar(
        positions,
        means_array,
        yerr=stds_array,
        fmt="o",
        linestyle="none",
        markersize=4.4,
        markerfacecolor="white",
        markeredgecolor=GRAPHITE,
        markeredgewidth=0.95,
        ecolor=GRAPHITE,
        elinewidth=0.8,
        capsize=2.4,
        capthick=0.75,
        zorder=3,
    )

    for x, mean, std in zip(positions, means_array, stds_array):
        label_y = min(mean + std + 0.00015, 0.01572)
        axis.text(
            x,
            label_y,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=GRAPHITE,
        )

    axis.set_xticks(positions, order_names)
    axis.set_xlim(-0.35, len(order_names) - 0.65)
    axis.set_ylim(0.008, 0.016)
    axis.set_yticks([0.008, 0.010, 0.012, 0.014, 0.016])
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axis.set_xlabel("Domain order", fontsize=8.0)
    axis.set_ylabel("Macro-F1 gap (oracle - top-2)", fontsize=8.0)
    _clean_axis(axis, grid_axis="y")

    _save_figure(
        fig,
        output_dir / "fig_oracle_gap_by_order_pretty",
        "Source-aware oracle and autonomous top-2 performance across domain orders",
    )
    plt.close(fig)


def _group_memory(records: Sequence[MemoryRecord]) -> Dict[str, List[MemoryRecord]]:
    return {
        budget: sorted(
            (row for row in records if row.budget == budget),
            key=lambda row: row.seed,
        )
        for budget, _ in MEMORY_RUNS
    }


def generate_memory_tradeoff(
    records: Sequence[MemoryRecord],
    pooled_values: Sequence[float],
    output_dir: Path,
) -> None:
    groups = _group_memory(records)
    budgets = [item[0] for item in MEMORY_RUNS]
    x_values = np.asarray(
        [np.mean([row.memory_mib for row in groups[budget]]) for budget in budgets],
        dtype=float,
    )
    f1_means = np.asarray(
        [_mean_std(row.macro_f1 for row in groups[budget])[0] for budget in budgets]
    )
    f1_stds = np.asarray(
        [_mean_std(row.macro_f1 for row in groups[budget])[1] for budget in budgets]
    )
    route_means = np.asarray(
        [_mean_std(row.route_accuracy for row in groups[budget])[0] for budget in budgets]
    )
    route_stds = np.asarray(
        [_mean_std(row.route_accuracy for row in groups[budget])[1] for budget in budgets]
    )
    fig = plt.figure(figsize=(3.35, 3.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.65, 1.0), hspace=0.08)
    f1_axis = fig.add_subplot(grid[0])
    route_axis = fig.add_subplot(grid[1], sharex=f1_axis)

    f1_axis.axhline(
        0.7551,
        color=MID_GREY,
        linewidth=0.75,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    f1_axis.text(
        x_values.min() * 1.03,
        0.7551 + 0.00025,
        "pooled head = 0.7551",
        color=MID_GREY,
        fontsize=5.2,
        alpha=0.82,
        va="bottom",
        ha="left",
        zorder=3,
    )

    f1_axis.errorbar(
        x_values,
        f1_means,
        yerr=f1_stds,
        color=NAVY,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=NAVY,
        markeredgewidth=1.1,
        markersize=4.8,
        linewidth=1.05,
        elinewidth=0.8,
        capsize=2.1,
        capthick=0.75,
        zorder=4,
    )
    route_axis.errorbar(
        x_values,
        route_means,
        yerr=route_stds,
        color=TEAL,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=TEAL,
        markeredgewidth=1.05,
        markersize=4.4,
        linewidth=1.05,
        elinewidth=0.8,
        capsize=2.1,
        capthick=0.75,
        zorder=4,
    )

    # Highlight the practical knee subtly, without a box or guide line.
    knee = budgets.index("10%")
    for axis, value in ((f1_axis, f1_means[knee]), (route_axis, route_means[knee])):
        axis.scatter(
            [x_values[knee]],
            [value],
            s=34,
            facecolors="none",
            edgecolors=AMBER,
            linewidths=0.75,
            zorder=5,
        )

    full_delta = f1_means[knee] - f1_means[-1]
    f1_axis.annotate(
        rf"10% knee, $\Delta_{{full}}={full_delta:+.4f}$",
        xy=(x_values[knee], f1_means[knee]),
        xytext=(-12, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=5.6,
        color=GRAPHITE,
        arrowprops={
            "arrowstyle": "->",
            "color": MID_GREY,
            "linewidth": 0.45,
            "mutation_scale": 5.0,
            "shrinkA": 1.5,
            "shrinkB": 2.0,
        },
        zorder=6,
    )

    # Label only the full-memory endpoints.
    f1_axis.annotate(
        f"{f1_means[-1]:.4f}",
        (x_values[-1], f1_means[-1]),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=NAVY,
    )
    route_axis.annotate(
        f"{route_means[-1]:.3f}",
        (x_values[-1], route_means[-1]),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=TEAL,
    )

    for axis in (f1_axis, route_axis):
        axis.set_xscale("log")
        axis.set_xlim(x_values.min() / 1.30, x_values.max() * 1.28)
        _clean_axis(axis, grid_axis="y")
        axis.tick_params(axis="both", which="major", labelsize=6.2)

    f1_axis.set_ylabel("Macro-F1", fontsize=7.0)
    route_axis.set_ylabel("Route accuracy", fontsize=7.0)
    f1_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    f1_axis.set_ylim(0.750, 0.782)
    f1_axis.yaxis.set_major_locator(FixedLocator([0.750, 0.760, 0.770, 0.780]))
    f1_axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    route_axis.set_ylim(0.62, 0.83)
    route_axis.yaxis.set_major_locator(FixedLocator([0.65, 0.70, 0.75, 0.80]))
    route_axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    route_axis.xaxis.set_major_locator(FixedLocator(x_values.tolist()))
    route_axis.set_xticklabels(
        [f"{budget}\n{memory:.1f} MiB" for budget, memory in zip(budgets, x_values)]
    )
    for label, alignment in zip(
        route_axis.get_xticklabels(), ("center", "right", "left", "center")
    ):
        label.set_horizontalalignment(alignment)
    route_axis.xaxis.set_minor_locator(FixedLocator([]))
    route_axis.set_xlabel("Retained train-feature budget (log scale)", fontsize=7.0)

    f1_axis.text(
        0.02,
        0.96,
        "(a)",
        transform=f1_axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color=NAVY,
    )
    route_axis.text(
        0.02,
        0.95,
        "(b)",
        transform=route_axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color=TEAL,
    )

    _save_figure(
        fig,
        output_dir / "fig_feature_memory_tradeoff_pretty",
        "Feature-memory trade-off for autonomous MLP top-2 routing",
    )
    plt.close(fig)


def generate_domain_gap_bars(output_dir: Path) -> None:
    """Plot the supplied domain-wise oracle-minus-top2 Macro-F1 gaps."""

    domains = [row[0] for row in DOMAIN_GAPS]
    gaps = np.asarray([row[3] for row in DOMAIN_GAPS], dtype=float)
    positions = np.arange(len(domains), dtype=float)

    fig, axis = plt.subplots(figsize=(3.35, 2.0), constrained_layout=True)
    bars = axis.bar(
        positions,
        gaps,
        width=0.52,
        color=DOMAIN_COLORS,
        edgecolor=DOMAIN_COLORS,
        linewidth=0.45,
        zorder=3,
    )
    axis.axhline(0.0, color=GRAPHITE, linewidth=0.7, zorder=4)

    for bar, gap in zip(bars, gaps):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(float(gap), 0.0) + 0.00055,
            f"{gap:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=GRAPHITE,
            zorder=5,
        )

    _clean_axis(axis, grid_axis="y")
    axis.set_ylim(-0.003, 0.0245)
    axis.set_xticks(positions, domains)
    axis.set_xlabel("Domain", fontsize=8.0)
    axis.set_ylabel("Oracle - top-2 Macro-F1", fontsize=8.0)
    axis.tick_params(axis="both", which="major", labelsize=7.0)
    axis.yaxis.set_major_locator(FixedLocator([0.000, 0.005, 0.010, 0.015, 0.020]))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    _save_figure(
        fig,
        output_dir / "fig_domainwise_oracle_gap_pretty",
        "Domain-wise Macro-F1 gap between oracle and autonomous MLP top-2 routing",
    )
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_source_data(
    order_records: Sequence[OrderRecord],
    memory_records: Sequence[MemoryRecord],
    pooled_values: Sequence[float],
    output_dir: Path,
) -> None:
    _write_csv(
        output_dir / "order_robustness_source.csv",
        [
            {
                "order": row.order,
                "seed": row.seed,
                "oracle_macro_f1": row.oracle,
                "top2_macro_f1": row.top2,
                "oracle_minus_top2": row.gap,
            }
            for row in order_records
        ],
    )
    _write_csv(
        output_dir / "feature_memory_source.csv",
        [
            {
                "budget": row.budget,
                "seed": row.seed,
                "retained_features": row.retained_features,
                "feature_mib": row.memory_mib,
                "macro_f1": row.macro_f1,
                "route_accuracy": row.route_accuracy,
            }
            for row in memory_records
        ],
    )
    _write_csv(
        output_dir / "pooled_baseline_source.csv",
        [{"seed": seed, "macro_f1": value} for seed, value in zip(SEEDS, pooled_values)],
    )
    _write_csv(
        output_dir / "domainwise_oracle_gap_source.csv",
        [
            {
                "domain": domain,
                "oracle_macro_f1": oracle,
                "top2_macro_f1": top2,
                "oracle_minus_top2": gap,
            }
            for domain, oracle, top2, gap in DOMAIN_GAPS
        ],
    )


def write_latex_snippets(output_dir: Path) -> None:
    snippet = r"""% Refined single-column figures. PDFs contain embedded TrueType fonts.
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig_oracle_gap_by_order_pretty.pdf}
  \caption{Within-order Macro-F1 gap between source-aware oracle selection and
  autonomous MLP top-2 routing. Points and error bars denote the mean and sample
  standard deviation of paired oracle-minus-top-2 gaps over seeds 42--44.}
  \label{fig:order_gap}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig_feature_memory_tradeoff_pretty.pdf}
  \caption{Feature-memory trade-off for autonomous MLP top-2 routing. Points and
  error bars denote the mean and sample standard deviation over seeds 42--44.
  The dashed line marks the pooled single-head baseline. The 10\% operating point retains 14.0 MiB of
  train features and remains within 0.0063 Macro-F1 of full memory.}
  \label{fig:feature_memory}
\end{figure}
"""
    (output_dir / "latex_snippets.tex").write_text(snippet, encoding="utf-8")


def main() -> None:
    args = parse_args()
    _configure_run_tag(args.run_tag)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font = set_publication_style()
    order_records = collect_order_records(args.outputs_root)
    memory_records, pooled_values = collect_memory_records(args.outputs_root)
    generate_order_dumbbell(order_records, output_dir)
    generate_memory_tradeoff(memory_records, pooled_values, output_dir)
    generate_domain_gap_bars(output_dir)
    export_source_data(order_records, memory_records, pooled_values, output_dir)
    write_latex_snippets(output_dir)

    manifest = {
        "outputs_root": str(args.outputs_root.resolve()),
        "output_dir": str(output_dir),
        "seeds": list(SEEDS),
        "font": font,
        "pdf_fonttype": int(mpl.rcParams["pdf.fonttype"]),
        "figures": [
            "fig_oracle_gap_by_order_pretty.pdf",
            "fig_feature_memory_tradeoff_pretty.pdf",
            "fig_domainwise_oracle_gap_pretty.pdf",
        ],
        "notes": (
            "Vector PDFs; points are means and error bars are sample standard "
            "deviations across seeds. Figure 3 uses paired gap SDs."
        ),
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
