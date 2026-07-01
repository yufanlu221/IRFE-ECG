"""Feature-cache sklearn baselines for ECGFounder head-bank experiments.

This evaluates whether the frozen ECGFounder final features are already strong
enough for classical linear classifiers. It uses the same grouped train/val
split, validation-only threshold calibration, and final test metrics as the
PyTorch head-bank baselines.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from evaluate.clinical_metrics import binary_clinical_metrics
from trainer.continual_cl import (
    DATA_DIR,
    DEVICE,
    DOMAIN_NAMES,
    OUTPUT_DIR,
    SEED,
    VAL_RATIO,
    calibrate_threshold,
    compute_forgetting,
    get_cinc_train_val_test_loaders,
    set_global_seed,
)
from trainer.shared_baselines import build_shared_model
from scripts.run_config import env_float, env_int, env_str


FEATURE_BATCH_SIZE = env_int("ECG_FEATURE_BATCH_SIZE", 256)
FEATURE_CACHE_DIR = Path(
    env_str(
        "ECG_FEATURE_CACHE_DIR",
        str(Path(OUTPUT_DIR).parent / "feature_cache"),
    )
)
SKLEARN_METHODS = [
    method.strip().lower()
    for method in env_str(
        "ECG_SKLEARN_METHODS",
        "logreg_balanced,logreg,linear_svm_balanced,ridge_balanced",
    ).split(",")
    if method.strip()
]
LOGREG_C_GRID = [
    float(value)
    for value in env_str("ECG_LOGREG_C_GRID", "0.01,0.1,1,10").split(",")
    if value.strip()
]
SVM_C_GRID = [
    float(value)
    for value in env_str("ECG_SVM_C_GRID", "0.01,0.1,1").split(",")
    if value.strip()
]
RIDGE_ALPHA_GRID = [
    float(value)
    for value in env_str("ECG_RIDGE_ALPHA_GRID", "0.1,1,10,100").split(",")
    if value.strip()
]
MAX_ITER = env_int("ECG_SKLEARN_MAX_ITER", 5000)

METRIC_NAMES = [
    "acc",
    "balanced_acc",
    "f1",
    "f1_negative",
    "f1_positive",
    "auc",
    "auprc",
    "sensitivity",
    "specificity",
    "precision",
    "npv",
    "threshold",
]


def _output_dir() -> Path:
    if os.environ.get("ECG_OUTPUT_DIR"):
        return Path(OUTPUT_DIR)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(OUTPUT_DIR) / f"feature_sklearn_seed{SEED}_{stamp}"


@torch.no_grad()
def _extract_loader_features(model: torch.nn.Module, loader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        output = model(x)
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError("build_shared_model(return_features=True) did not return features.")
        feats.append(output[1].detach().cpu())
        labels.append(y.detach().cpu())
    return torch.cat(feats).numpy().astype(np.float32), torch.cat(labels).numpy().astype(int)


def _feature_cache_path(domain: str, split: str) -> Path:
    safe_val = str(VAL_RATIO).replace(".", "p")
    return FEATURE_CACHE_DIR / f"ecgfounder_final_seed{SEED}_val{safe_val}_{domain}_{split}.npz"


def load_or_extract_features() -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, Any]],
]:
    train_loaders, val_loaders, test_loaders, split_infos, _ = get_cinc_train_val_test_loaders(
        data_dir=DATA_DIR,
        domains=DOMAIN_NAMES,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model = None
    features: Dict[str, Dict[str, np.ndarray]] = {}
    labels: Dict[str, Dict[str, np.ndarray]] = {}

    for domain in DOMAIN_NAMES:
        features[domain] = {}
        labels[domain] = {}
        loaders = {
            "train": train_loaders[domain],
            "val": val_loaders[domain],
            "test": test_loaders[domain],
        }
        for split, loader in loaders.items():
            if loader is None:
                continue
            cache_path = _feature_cache_path(domain, split)
            if cache_path.exists():
                cached = np.load(cache_path)
                features[domain][split] = cached["features"].astype(np.float32)
                labels[domain][split] = cached["labels"].astype(int)
                print(
                    f"  [Cache] loaded {domain}/{split}: "
                    f"{features[domain][split].shape} from {cache_path}"
                )
                continue
            if model is None:
                model = build_shared_model(return_features=True)
                model.eval()
            print(f"  [Extract] {domain}/{split} n={len(loader.dataset)}")
            x_feat, y = _extract_loader_features(model, loader)
            np.savez_compressed(cache_path, features=x_feat, labels=y)
            features[domain][split] = x_feat
            labels[domain][split] = y
            print(f"  [Cache] saved {cache_path}")

    return features, labels, split_infos


def _fit_candidates(method: str, x_train: np.ndarray, y_train: np.ndarray):
    method = method.lower()
    candidates = []
    if method == "logreg":
        for c_value in LOGREG_C_GRID:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, max_iter=MAX_ITER, solver="lbfgs"),
            )
            candidates.append((f"logreg_C{c_value:g}", clf))
        return candidates
    if method == "logreg_balanced":
        for c_value in LOGREG_C_GRID:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    max_iter=MAX_ITER,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            )
            candidates.append((f"logreg_balanced_C{c_value:g}", clf))
        return candidates
    if method == "linear_svm_balanced":
        for c_value in SVM_C_GRID:
            base = LinearSVC(
                C=c_value,
                class_weight="balanced",
                max_iter=MAX_ITER,
                dual="auto",
            )
            clf = make_pipeline(
                StandardScaler(),
                CalibratedClassifierCV(base, cv=3, method="sigmoid"),
            )
            candidates.append((f"linear_svm_balanced_C{c_value:g}", clf))
        return candidates
    if method == "ridge_balanced":
        for alpha in RIDGE_ALPHA_GRID:
            clf = make_pipeline(
                StandardScaler(),
                RidgeClassifier(alpha=alpha, class_weight="balanced"),
            )
            candidates.append((f"ridge_balanced_alpha{alpha:g}", clf))
        return candidates
    raise ValueError(
        "Unknown ECG_SKLEARN_METHODS entry. Use logreg, logreg_balanced, "
        "linear_svm_balanced, ridge_balanced."
    )


def _scores_to_positive_signal(clf, x: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x)[:, 1]
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(x).astype(float)
        # Map only for threshold-search compatibility; ordering is preserved for AUC.
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))
    raise RuntimeError("Classifier exposes neither predict_proba nor decision_function.")


def evaluate_domain_method(
    method: str,
    domain: str,
    features: Dict[str, Dict[str, np.ndarray]],
    labels: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    x_train, y_train = features[domain]["train"], labels[domain]["train"]
    x_val, y_val = features[domain]["val"], labels[domain]["val"]
    x_test, y_test = features[domain]["test"], labels[domain]["test"]

    best = None
    tried = []
    for name, clf in _fit_candidates(method, x_train, y_train):
        clf.fit(x_train, y_train)
        val_scores = _scores_to_positive_signal(clf, x_val)
        threshold, val_metrics = calibrate_threshold(y_val, val_scores, metric_name="f1")
        tried.append(
            {
                "candidate": name,
                "threshold": float(threshold),
                "val_f1": float(val_metrics["f1"]),
                "val_auc": float(val_metrics["auc"]),
            }
        )
        score = float(val_metrics["f1"])
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "candidate": name,
                "clf": clf,
                "threshold": threshold,
                "val_metrics": val_metrics,
            }
    if best is None:
        raise RuntimeError(f"No fitted candidate for {method}/{domain}.")

    test_scores = _scores_to_positive_signal(best["clf"], x_test)
    test_metrics = binary_clinical_metrics(y_test, test_scores, threshold=best["threshold"])
    details = {
        "method": method,
        "domain": domain,
        "best_candidate": best["candidate"],
        "best_threshold": float(best["threshold"]),
        "best_val_metrics": best["val_metrics"],
        "tried": tried,
    }
    return test_metrics, details


def run_feature_sklearn_baselines() -> Dict[str, Any]:
    set_global_seed(SEED)
    print(f"  Device: {DEVICE}")
    print(f"  Data: {DATA_DIR}")
    print(f"  Seed: {SEED}")
    print(f"  Feature cache: {FEATURE_CACHE_DIR}")
    print(f"  Methods: {', '.join(SKLEARN_METHODS)}")
    features, labels, split_infos = load_or_extract_features()

    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Any] = {
        "seed": SEED,
        "data_dir": DATA_DIR,
        "feature_cache_dir": str(FEATURE_CACHE_DIR),
        "val_ratio": VAL_RATIO,
        "domains": DOMAIN_NAMES,
        "methods": SKLEARN_METHODS,
        "split_info": split_infos,
        "results": {},
    }

    best_method = None
    best_mean = -float("inf")
    for method in SKLEARN_METHODS:
        print(f"\n{'=' * 64}")
        print(f"  Feature sklearn method: {method}")
        print(f"{'=' * 64}")
        final_metrics: Dict[str, Dict[str, float]] = {}
        details: Dict[str, Dict[str, Any]] = {}
        for domain in DOMAIN_NAMES:
            metrics, domain_details = evaluate_domain_method(method, domain, features, labels)
            final_metrics[domain] = metrics
            details[domain] = domain_details
            print(
                f"  {domain:<8s} {domain_details['best_candidate']:<28s} "
                f"thr={metrics['threshold']:.3f} F1={metrics['f1']:.6f} "
                f"AUC={metrics['auc']:.6f} Sen={metrics['sensitivity']:.3f} "
                f"Spe={metrics['specificity']:.3f}"
            )

        final_f1 = np.asarray([final_metrics[domain]["f1"] for domain in DOMAIN_NAMES], dtype=float)
        r_matrix = np.full((len(DOMAIN_NAMES), len(DOMAIN_NAMES)), np.nan, dtype=float)
        for idx, value in enumerate(final_f1):
            r_matrix[idx:, idx] = value
        forgetting = compute_forgetting(r_matrix, DOMAIN_NAMES)
        mean_f1 = float(final_f1.mean())
        all_results["results"][method] = {
            "mean_f1": mean_f1,
            "mean_bwt": float(forgetting["mean_bwt"]),
            "final_metrics_by_domain": final_metrics,
            "details_by_domain": details,
        }
        print(
            f"  [Summary] method={method} mean_f1={mean_f1:.6f} "
            f"mean_bwt={float(forgetting['mean_bwt']):.6f}"
        )
        if mean_f1 > best_mean:
            best_mean = mean_f1
            best_method = method

    all_results["best_method"] = best_method
    all_results["best_mean_f1"] = best_mean

    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    rows = []
    for method, result in all_results["results"].items():
        for domain in DOMAIN_NAMES:
            row = {"method": method, "domain": domain}
            row.update(result["final_metrics_by_domain"][domain])
            rows.append(row)
    metrics_path = output_dir / "final_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "domain", *METRIC_NAMES])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})

    f1_path = output_dir / "final_f1.csv"
    with f1_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "domain", "final_f1"])
        for method, result in all_results["results"].items():
            for domain in DOMAIN_NAMES:
                writer.writerow([
                    method,
                    domain,
                    result["final_metrics_by_domain"][domain]["f1"],
                ])

    print(f"\n  [Saved] {summary_path}")
    print(f"  [Saved] {metrics_path}")
    print(f"  [Saved] {f1_path}")
    print(f"  [Done] best_method={best_method} best_mean_f1={best_mean:.6f}")
    return all_results


def main() -> None:
    run_feature_sklearn_baselines()


if __name__ == "__main__":
    main()
