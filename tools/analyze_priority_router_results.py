"""Aggregate the priority router queue and run paired bootstrap analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DOMAINS = ("cpsc", "ptbxl", "georgia", "chapman")
HARD_MODE = "domain_mlp"
TOP2_MODE = "domain_mlp_top2"


def _macro_f1_from_counts(tn, fp, fn, tp):
    positive_den = 2 * tp + fp + fn
    negative_den = 2 * tn + fp + fn
    f1_positive = np.divide(
        2 * tp,
        positive_den,
        out=np.zeros_like(tp, dtype=np.float64),
        where=positive_den != 0,
    )
    f1_negative = np.divide(
        2 * tn,
        negative_den,
        out=np.zeros_like(tn, dtype=np.float64),
        where=negative_den != 0,
    )
    return 0.5 * (f1_positive + f1_negative)


def _f1_from_category_counts(category_counts, labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tn = category_counts[:, (labels == 0) & (predictions == 0)].sum(axis=1)
    fp = category_counts[:, (labels == 0) & (predictions == 1)].sum(axis=1)
    fn = category_counts[:, (labels == 1) & (predictions == 0)].sum(axis=1)
    tp = category_counts[:, (labels == 1) & (predictions == 1)].sum(axis=1)
    return _macro_f1_from_counts(tn, fp, fn, tp)


def _load_runs(outputs_root: Path, stamp: str):
    run_dirs = sorted(outputs_root.glob(f"feature_router_primary_seed*_mem1_{stamp}"))
    if len(run_dirs) != 3:
        raise RuntimeError(f"Expected three primary runs, found {len(run_dirs)}")

    runs = []
    for run_dir in run_dirs:
        with (run_dir / "metrics_summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        archive = np.load(run_dir / "continual_results.npz", allow_pickle=False)
        runs.append((run_dir, summary, archive))
    return runs


def _validate_runs(runs):
    seeds = [run[1]["experiment_config"]["seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"Duplicate seeds in primary runs: {seeds}")

    for domain in DOMAINS:
        reference = runs[0][2][f"{HARD_MODE}__{domain}__labels"]
        for _, _, archive in runs:
            hard_labels = archive[f"{HARD_MODE}__{domain}__labels"]
            top2_labels = archive[f"{TOP2_MODE}__{domain}__labels"]
            if not np.array_equal(reference, hard_labels):
                raise RuntimeError(f"Test labels differ across seeds for {domain}")
            if not np.array_equal(hard_labels, top2_labels):
                raise RuntimeError(f"Hard/top-2 labels differ for {domain}")
    return seeds


def paired_bootstrap(runs, repetitions: int, random_seed: int):
    rng = np.random.default_rng(random_seed)
    hard_scores = []
    top2_scores = []

    for domain in DOMAINS:
        labels = runs[0][2][f"{HARD_MODE}__{domain}__labels"].astype(np.int64)
        predictions = []
        for mode in (HARD_MODE, TOP2_MODE):
            for _, _, archive in runs:
                predictions.append(archive[f"{mode}__{domain}__preds"].astype(np.int64))

        # Encode each sample's label and all six paired predictions. Sampling
        # the joint categories preserves method and seed correlations.
        codes = labels.copy()
        for bit, preds in enumerate(predictions, start=1):
            codes |= preds << bit
        observed_counts = np.bincount(codes, minlength=128)
        sampled_counts = rng.multinomial(
            labels.size,
            observed_counts / observed_counts.sum(),
            size=repetitions,
        )
        categories = np.arange(128, dtype=np.int64)
        category_labels = categories & 1
        domain_hard = []
        domain_top2 = []
        for bit in range(1, 4):
            category_preds = (categories >> bit) & 1
            domain_hard.append(
                _f1_from_category_counts(
                    sampled_counts, category_labels, category_preds
                )
            )
        for bit in range(4, 7):
            category_preds = (categories >> bit) & 1
            domain_top2.append(
                _f1_from_category_counts(
                    sampled_counts, category_labels, category_preds
                )
            )
        hard_scores.extend(domain_hard)
        top2_scores.extend(domain_top2)

    hard_mean = np.mean(np.stack(hard_scores), axis=0)
    top2_mean = np.mean(np.stack(top2_scores), axis=0)
    differences = top2_mean - hard_mean
    lower, upper = np.quantile(differences, [0.025, 0.975])
    p_two_sided = min(1.0, 2.0 * min(
        np.mean(differences <= 0), np.mean(differences >= 0)
    ))
    return {
        "repetitions": repetitions,
        "random_seed": random_seed,
        "bootstrap_mean_difference": float(differences.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "two_sided_p": float(p_two_sided),
    }


def _observed_f1_summary(runs):
    result = {}
    for mode in (HARD_MODE, TOP2_MODE):
        values = [run[1]["settings"][mode]["mean_f1"] for run in runs]
        result[mode] = {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)),
            "values": [float(value) for value in values],
        }
    result["observed_difference"] = (
        result[TOP2_MODE]["mean"] - result[HARD_MODE]["mean"]
    )
    return result


def _classification_confusions(runs):
    rows = []
    for domain in DOMAINS:
        seed_counts = []
        for _, summary, archive in runs:
            labels = archive[f"{TOP2_MODE}__{domain}__labels"]
            preds = archive[f"{TOP2_MODE}__{domain}__preds"]
            seed_counts.append([
                int(np.sum((labels == 0) & (preds == 0))),
                int(np.sum((labels == 0) & (preds == 1))),
                int(np.sum((labels == 1) & (preds == 0))),
                int(np.sum((labels == 1) & (preds == 1))),
            ])
        counts = np.asarray(seed_counts, dtype=np.float64)
        rows.append({
            "domain": domain,
            "mean_tn": counts[:, 0].mean(),
            "mean_fp": counts[:, 1].mean(),
            "mean_fn": counts[:, 2].mean(),
            "mean_tp": counts[:, 3].mean(),
        })
    return rows


def _route_confusions(runs):
    rows = []
    domain_order = runs[0][1]["experiment_config"]["domain_order"]
    for true_domain in DOMAINS:
        counts = np.zeros((len(runs), len(domain_order)), dtype=np.float64)
        for run_index, (_, _, archive) in enumerate(runs):
            routes = archive[f"{TOP2_MODE}__{true_domain}__routes"]
            counts[run_index] = np.bincount(
                routes, minlength=len(domain_order)
            )[: len(domain_order)]
        mean_counts = counts.mean(axis=0)
        rates = mean_counts / mean_counts.sum()
        row = {"true_domain": true_domain}
        for route_index, routed_domain in enumerate(domain_order):
            row[f"pred_{routed_domain}_count"] = mean_counts[route_index]
            row[f"pred_{routed_domain}_rate"] = rates[route_index]
        rows.append(row)
    return rows


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--stamp", default="20260628_112610")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260628)
    args = parser.parse_args()

    runs = _load_runs(args.outputs_root, args.stamp)
    seeds = _validate_runs(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "seeds": seeds,
        "source_runs": [str(run[0]) for run in runs],
        "observed": _observed_f1_summary(runs),
        "paired_bootstrap_top2_minus_hard_mlp": paired_bootstrap(
            runs, args.bootstrap_repetitions, args.bootstrap_seed
        ),
    }
    with (args.output_dir / "paired_bootstrap.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    _write_csv(
        args.output_dir / "top2_classification_confusion.csv",
        _classification_confusions(runs),
    )
    _write_csv(
        args.output_dir / "top2_route_confusion.csv",
        _route_confusions(runs),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
