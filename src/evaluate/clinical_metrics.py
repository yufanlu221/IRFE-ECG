"""Lightweight clinical metrics for binary ECG classification."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")


def _safe_auprc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(average_precision_score(labels, probs))
    except ValueError:
        return float("nan")


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _as_positive_probs(probs) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        return probs
    if probs.ndim == 2 and probs.shape[1] == 2:
        return probs[:, 1]
    raise ValueError(
        "probs must be 1-D positive-class probabilities or a 2-D array with "
        "shape (N, 2)."
    )


def binary_clinical_metrics(
    labels,
    probs,
    preds: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Return paper-ready binary metrics.

    Positive class is label 1. Confusion matrix fields are always computed with
    labels=[0, 1], so missing-class batches/tests stay well-defined.
    """
    labels = np.asarray(labels).astype(int)
    probs = _as_positive_probs(probs)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"labels/probs length mismatch: {labels.shape[0]} vs {probs.shape[0]}"
        )
    if preds is None:
        preds = (probs >= threshold).astype(int)
    else:
        preds = np.asarray(preds).astype(int)
        if preds.ndim == 2 and preds.shape[1] == 2:
            preds = preds.argmax(axis=1)
        if preds.ndim != 1:
            raise ValueError(f"preds must be 1-D or (N, 2), got shape {preds.shape}")
        if preds.shape[0] != labels.shape[0]:
            raise ValueError(
                f"labels/preds length mismatch: {labels.shape[0]} vs {preds.shape[0]}"
            )

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)
    f1_positive = _safe_div(2.0 * precision * recall, precision + recall)
    f1_negative = _safe_div(2.0 * npv * specificity, npv + specificity)

    return {
        "acc": float(accuracy_score(labels, preds)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_negative": f1_negative,
        "f1_positive": f1_positive,
        "auc": _safe_auc(labels, probs),
        "auprc": _safe_auprc(labels, probs),
        "sensitivity": recall,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support_negative": int((labels == 0).sum()),
        "support_positive": int((labels == 1).sum()),
    }


def format_metric_line(metrics: Dict[str, float], clinical: bool = False) -> str:
    """Compact console formatter for training logs."""
    auc = metrics.get("auc", float("nan"))
    auc_s = "nan" if np.isnan(auc) else f"{auc:.3f}"
    line = (
        f"Acc: {metrics['acc']:.3f} | "
        f"F1: {metrics['f1']:.3f} | "
        f"AUC: {auc_s}"
    )
    if clinical:
        line += (
            f" | Sen: {metrics['sensitivity']:.3f} | "
            f"Spe: {metrics['specificity']:.3f}"
        )
    return line
