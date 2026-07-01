"""Generate paper figures and tables from the fixed 2026-06-28 router runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DOMAINS = ("cpsc", "ptbxl", "georgia", "chapman")
DOMAIN_LABELS = ("CPSC", "PTB-XL", "Georgia", "Chapman")
SEEDS = (42, 43, 44)
RUN_TAG = "paper"

ROUTER_MODES = (
    ("centroid_cosine", "Centroid"),
    ("shrinkage_lda", "Shrinkage LDA"),
    ("domain_mlp", "Domain MLP"),
)

ORDER_RUNS = (
    ("Current", "feature_router_primary_seed{seed}_mem1_" + RUN_TAG),
    ("Reverse", "feature_router_reverse_order_seed{seed}_mem1_" + RUN_TAG),
    ("Fixed random", "feature_router_random_order_seed{seed}_mem1_" + RUN_TAG),
)

MEMORY_RUNS = (
    ("1%", "feature_router_memory_seed{seed}_mem0p01_" + RUN_TAG),
    ("5%", "feature_router_memory_seed{seed}_mem0p05_" + RUN_TAG),
    ("10%", "feature_router_memory_seed{seed}_mem0p1_" + RUN_TAG),
    ("100%", "feature_router_primary_seed{seed}_mem1_" + RUN_TAG),
)


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _primary_dir(outputs_root: Path, seed: int) -> Path:
    return outputs_root / f"feature_router_primary_seed{seed}_mem1_{RUN_TAG}"


def _configure_run_tag(run_tag: str) -> None:
    global RUN_TAG, ORDER_RUNS, MEMORY_RUNS
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


def _set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _route_confusion(outputs_root: Path, mode: str):
    matrices = []
    for seed in SEEDS:
        archive = np.load(
            _primary_dir(outputs_root, seed) / "continual_results.npz",
            allow_pickle=False,
        )
        matrix = np.zeros((len(DOMAINS), len(DOMAINS)), dtype=np.float64)
        for row, domain in enumerate(DOMAINS):
            routes = archive[f"{mode}__{domain}__routes"]
            matrix[row] = np.bincount(routes, minlength=len(DOMAINS))[: len(DOMAINS)]
            matrix[row] /= matrix[row].sum()
        matrices.append(matrix)
    return np.mean(matrices, axis=0), np.std(matrices, axis=0, ddof=1)


def generate_route_confusion(outputs_root: Path, output_dir: Path):
    _set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), constrained_layout=True)
    exported = []
    image = None
    for axis, (mode, title) in zip(axes, ROUTER_MODES):
        mean_matrix, std_matrix = _route_confusion(outputs_root, mode)
        image = axis.imshow(mean_matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        axis.set_title(title)
        axis.set_xticks(range(len(DOMAIN_LABELS)), DOMAIN_LABELS, rotation=35, ha="right")
        axis.set_yticks(range(len(DOMAIN_LABELS)), DOMAIN_LABELS)
        axis.set_xlabel("Selected expert")
        if axis is axes[0]:
            axis.set_ylabel("True source")
        else:
            axis.tick_params(labelleft=False)
        for row in range(len(DOMAINS)):
            for col in range(len(DOMAINS)):
                value = mean_matrix[row, col]
                color = "white" if value >= 0.55 else "black"
                axis.text(
                    col,
                    row,
                    f"{100 * value:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=6.5,
                )
                exported.append(
                    {
                        "router": mode,
                        "true_source": DOMAINS[row],
                        "selected_expert": DOMAINS[col],
                        "mean_rate": value,
                        "sample_std": std_matrix[row, col],
                    }
                )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.024, pad=0.015)
    colorbar.set_label("Routing rate")
    fig.savefig(output_dir / "route_confusion_centroid_lda_mlp.png", bbox_inches="tight")
    fig.savefig(output_dir / "route_confusion_centroid_lda_mlp.pdf", bbox_inches="tight")
    plt.close(fig)
    _write_csv(output_dir / "route_confusion_centroid_lda_mlp.csv", exported)


def generate_oracle_gap(outputs_root: Path, output_dir: Path):
    _set_plot_style()
    rows = []
    summary_rows = []
    means = []
    stds = []
    for label, directory_pattern in ORDER_RUNS:
        gaps = []
        for seed in SEEDS:
            run_dir = outputs_root / directory_pattern.format(seed=seed)
            summary = _load_json(run_dir / "metrics_summary.json")
            oracle_macro_f1 = summary["settings"]["oracle"]["mean_f1"]
            top2_macro_f1 = summary["settings"]["domain_mlp_top2"]["mean_f1"]
            gap = oracle_macro_f1 - top2_macro_f1
            gaps.append(gap)
            rows.append(
                {
                    "order": label,
                    "seed": seed,
                    "oracle_macro_f1": oracle_macro_f1,
                    "top2_macro_f1": top2_macro_f1,
                    "oracle_minus_top2": gap,
                }
            )

        # Error bars describe variation of the paired, per-seed gaps.  They are
        # not the SD of either Macro-F1 series and are not propagated estimates.
        gap_values = np.asarray(gaps, dtype=np.float64)
        mean_gap = float(gap_values.mean())
        sample_std = float(gap_values.std(ddof=1))
        means.append(mean_gap)
        stds.append(sample_std)
        summary_rows.append(
            {
                "order": label,
                "mean_oracle_minus_top2": mean_gap,
                "sample_std": sample_std,
                "n_seeds": len(gap_values),
            }
        )

    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    positions = np.arange(len(means))
    bars = axis.bar(
        positions,
        means,
        yerr=stds,
        capsize=3,
        width=0.62,
        color=("#2F6690", "#5B8E7D", "#C17C4A"),
        edgecolor="black",
        linewidth=0.6,
    )
    axis.set_xticks(positions, [item[0] for item in ORDER_RUNS])
    axis.set_ylabel("Macro-F1 gap (oracle - top-2)")
    axis.set_ylim(0.0, max(np.asarray(means) + np.asarray(stds)) * 1.35)
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.5)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.00045,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.savefig(output_dir / "oracle_gap_by_order.png", bbox_inches="tight")
    fig.savefig(output_dir / "oracle_gap_by_order.pdf", bbox_inches="tight")
    plt.close(fig)
    _write_csv(output_dir / "oracle_gap_by_order.csv", rows)
    _write_csv(output_dir / "oracle_gap_by_order_summary.csv", summary_rows)


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1))


def generate_memory_table(outputs_root: Path, output_dir: Path):
    rows = []
    for budget, directory_pattern in MEMORY_RUNS:
        counts = []
        f1_values = []
        route_values = []
        for seed in SEEDS:
            run_dir = outputs_root / directory_pattern.format(seed=seed)
            summary = _load_json(run_dir / "metrics_summary.json")
            top2 = summary["settings"]["domain_mlp_top2"]
            f1_values.append(top2["mean_f1"])
            route_values.append(
                np.mean(list(top2["final_route_accuracy_by_domain"].values()))
            )
            final_domain = summary["experiment_config"]["domain_order"][-1]
            counts.append(
                sum(summary["router_metadata"][final_domain]["memory_counts"].values())
            )
        f1_mean, f1_std = _mean_std(f1_values)
        route_mean, route_std = _mean_std(route_values)
        count_mean = float(np.mean(counts))
        rows.append(
            {
                "budget": budget,
                "retained_features": count_mean,
                "feature_mib": count_mean * 1024 * 4 / (1024**2),
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "route_accuracy_mean": route_mean,
                "route_accuracy_std": route_std,
            }
        )

    markdown = [
        "# Feature-Memory Ablation",
        "",
        "No raw ECG waveforms are replayed, but frozen 1024-dimensional float32 "
        "features are retained for router updates.",
        "",
        "| Retained fraction | Features | Feature memory (MiB) | Macro-F1 | Route accuracy |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['budget']} | {row['retained_features']:.0f} | "
            f"{row['feature_mib']:.2f} | "
            f"{row['macro_f1_mean']:.4f} $\\pm$ {row['macro_f1_std']:.4f} | "
            f"{row['route_accuracy_mean']:.4f} $\\pm$ {row['route_accuracy_std']:.4f} |"
        )
    (output_dir / "feature_memory_ablation.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Feature-memory ablation for MLP top-2 routing. No raw ECG "
        r"waveforms are replayed; the reported memory stores frozen "
        r"1024-dimensional float32 features for router updates.}",
        r"\label{tab:feature_memory}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}rrrr@{}}",
        r"\toprule",
        r"Retained & Features & Memory (MiB) & Macro-F1 / Route acc. \\",
        r"\midrule",
    ]
    for row in rows:
        latex.append(
            f"{row['budget']} & {row['retained_features']:.0f} & "
            f"{row['feature_mib']:.2f} & "
            f"${row['macro_f1_mean']:.4f}\\pm{row['macro_f1_std']:.4f}$ / "
            f"${row['route_accuracy_mean']:.4f}\\pm{row['route_accuracy_std']:.4f}$ \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (output_dir / "feature_memory_ablation.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "feature_memory_ablation.csv", rows)


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-tag", default="paper")
    args = parser.parse_args()
    _configure_run_tag(args.run_tag)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_route_confusion(args.outputs_root, args.output_dir)
    generate_oracle_gap(args.outputs_root, args.output_dir)
    generate_memory_table(args.outputs_root, args.output_dir)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
