"""Train domain-aware heads from cached ECGFounder feature `.npz` files.

This avoids re-running ECGFounder for every head/loss ablation. It is the
preferred shared-server path after `process.py` has produced feature caches.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from evaluate.clinical_metrics import binary_clinical_metrics, format_metric_line
from scripts.run_config import env_bool, env_float, env_int, env_str
from trainer.continual_cl import (
    CLASS_WEIGHT_MODE,
    DATA_DIR,
    DEVICE,
    DOMAIN_NAMES,
    LABEL_SMOOTHING,
    LOSS_MODE,
    OUTPUT_DIR,
    SEED,
    SELECT_BY,
    TRAIN_KWARGS,
    VAL_RATIO,
    _metric_score,
    calibrate_threshold,
    class_counts_from_labels,
    compute_class_weights_from_labels,
    compute_forgetting,
    set_global_seed,
    training_loss,
)


FEATURE_CACHE_DIR = Path(
    env_str(
        "ECG_FEATURE_CACHE_DIR",
        str(Path(OUTPUT_DIR).parent / "feature_cache"),
    )
)
FEATURE_HEAD_METHOD = env_str("ECG_FEATURE_HEAD_METHOD", "linear").lower()
FEATURE_BATCH_SIZE = env_int("ECG_FEATURE_HEAD_BATCH_SIZE", TRAIN_KWARGS["batch_size"])
FEATURE_MLP_HIDDEN_DIM = env_int("ECG_FEATURE_MLP_HIDDEN_DIM", 256)
FEATURE_MLP_DROPOUT = env_float("ECG_FEATURE_MLP_DROPOUT", 0.1)
FEATURE_ADAPTER_DIM = env_int("ECG_FEATURE_ADAPTER_DIM", 128)
FEATURE_ADAPTER_DROPOUT = env_float("ECG_FEATURE_ADAPTER_DROPOUT", 0.1)
FEATURE_ADAPTER_SCALE = env_float("ECG_FEATURE_ADAPTER_SCALE", 1.0)
SELECT_CALIBRATED_EPOCH = env_bool(
    "ECG_SELECT_CALIBRATED_EPOCH",
    env_bool("ECG_SELECT_EPOCH_AFTER_CALIBRATION", False),
)
CPU_THREADS = env_int("CPU_THREADS", env_int("ECG_CPU_THREADS", 2))

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


class ResidualFeatureHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 2,
        adapter_dim: int = FEATURE_ADAPTER_DIM,
        dropout: float = FEATURE_ADAPTER_DROPOUT,
        scale: float = FEATURE_ADAPTER_SCALE,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, feature_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x + self.scale * self.adapter(x))


def _feature_cache_path(domain: str, split: str) -> Path:
    safe_val = str(VAL_RATIO).replace(".", "p")
    return FEATURE_CACHE_DIR / f"ecgfounder_final_seed{SEED}_val{safe_val}_{domain}_{split}.npz"


def _load_feature_npz(domain: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    path = _feature_cache_path(domain, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature cache: {path}. Run `python process.py` first."
        )
    data = np.load(path)
    return data["features"].astype(np.float32), data["labels"].astype(np.int64)


def load_feature_cache() -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
]:
    features: Dict[str, Dict[str, np.ndarray]] = {}
    labels: Dict[str, Dict[str, np.ndarray]] = {}
    for domain in DOMAIN_NAMES:
        features[domain] = {}
        labels[domain] = {}
        for split in ("train", "val", "test"):
            features[domain][split], labels[domain][split] = _load_feature_npz(domain, split)
            print(
                f"  [Cache] {domain}/{split}: "
                f"{features[domain][split].shape}"
            )
    return features, labels


def make_head(feature_dim: int, method: str) -> nn.Module:
    if method in {"linear", "domain_linear_heads"}:
        return nn.Linear(feature_dim, 2)
    if method in {"layernorm", "layernorm_linear", "domain_layernorm_heads", "domain_ln_heads"}:
        return nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 2))
    if method in {"mlp", "domain_mlp_heads"}:
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, FEATURE_MLP_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(FEATURE_MLP_DROPOUT),
            nn.Linear(FEATURE_MLP_HIDDEN_DIM, 2),
        )
    if method in {"residual", "residual_adapter", "domain_residual_heads"}:
        return ResidualFeatureHead(feature_dim=feature_dim)
    raise ValueError(
        "ECG_FEATURE_HEAD_METHOD must be linear, layernorm, mlp, or residual."
    )


def _output_dir() -> Path:
    if os.environ.get("ECG_OUTPUT_DIR"):
        return Path(OUTPUT_DIR)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(OUTPUT_DIR) / f"feature_head_bank_{FEATURE_HEAD_METHOD}_seed{SEED}_{stamp}"


def train_one_epoch_features(
    head: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor | None,
    class_counts: torch.Tensor | None,
    generator: torch.Generator,
) -> float:
    head.train()
    order = torch.randperm(y_train.shape[0], generator=generator)
    total_loss = 0.0
    steps = 0
    for start in range(0, y_train.shape[0], FEATURE_BATCH_SIZE):
        idx = order[start : start + FEATURE_BATCH_SIZE].to(x_train.device)
        logits = head(x_train[idx])
        loss = training_loss(
            logits,
            y_train[idx],
            class_weights=class_weights,
            class_counts=class_counts,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), TRAIN_KWARGS["grad_clip"])
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def collect_probs_features(
    head: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    head.eval()
    x = torch.from_numpy(features).to(DEVICE)
    logits = []
    for start in range(0, x.shape[0], FEATURE_BATCH_SIZE * 4):
        logits.append(head(x[start : start + FEATURE_BATCH_SIZE * 4]).detach().cpu())
    probs = torch.softmax(torch.cat(logits, dim=0), dim=-1)[:, 1].numpy()
    return labels.astype(int), probs


def run_feature_head_bank() -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))
    torch.set_num_threads(max(1, CPU_THREADS))
    torch.set_num_interop_threads(1)
    set_global_seed(SEED)

    print(f"  Device: {DEVICE}")
    print(f"  Data: {DATA_DIR}")
    print(f"  Feature cache: {FEATURE_CACHE_DIR}")
    print(f"  Feature head method: {FEATURE_HEAD_METHOD}")
    print(f"  Seed: {SEED}")
    print(
        f"  Hyperparams: epochs={TRAIN_KWARGS['epochs']}, "
        f"lr={TRAIN_KWARGS['lr']}, batch={FEATURE_BATCH_SIZE}, "
        f"cpu_threads={CPU_THREADS}"
    )

    features, labels = load_feature_cache()
    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    t = len(DOMAIN_NAMES)
    r_matrix = np.full((t, t), np.nan, dtype=float)
    metric_matrices = {name: np.full((t, t), np.nan, dtype=float) for name in METRIC_NAMES}
    domain_thresholds: Dict[str, float] = {}
    heads: Dict[str, nn.Module] = {}
    metadata: Dict[str, Any] = {
        "config": {
            "feature_head_method": FEATURE_HEAD_METHOD,
            "feature_cache_dir": str(FEATURE_CACHE_DIR),
            "seed": SEED,
            "val_ratio": VAL_RATIO,
            "loss_mode": LOSS_MODE,
            "class_weight_mode": CLASS_WEIGHT_MODE,
            "label_smoothing": LABEL_SMOOTHING,
            "select_by": SELECT_BY,
            "select_epoch_after_threshold_calibration": SELECT_CALIBRATED_EPOCH,
            "train_kwargs": TRAIN_KWARGS,
            "feature_batch_size": FEATURE_BATCH_SIZE,
            "cpu_threads": CPU_THREADS,
            "feature_mlp_hidden_dim": FEATURE_MLP_HIDDEN_DIM,
            "feature_mlp_dropout": FEATURE_MLP_DROPOUT,
            "feature_adapter_dim": FEATURE_ADAPTER_DIM,
            "feature_adapter_dropout": FEATURE_ADAPTER_DROPOUT,
            "feature_adapter_scale": FEATURE_ADAPTER_SCALE,
        },
        "best_epoch_by_domain": {},
        "best_threshold_by_domain": {},
        "best_val_metrics_by_domain": {},
        "class_counts_by_domain": {},
        "class_weights_by_domain": {},
    }

    feature_dim = int(features[DOMAIN_NAMES[0]]["train"].shape[1])
    for i, domain in enumerate(DOMAIN_NAMES):
        print(f"\n{'=' * 60}")
        print(f"  Feature head-bank | Task {i + 1}/{t}: {domain}")
        print(f"{'=' * 60}")
        head = make_head(feature_dim, FEATURE_HEAD_METHOD).to(DEVICE)
        heads[domain] = head
        params = [p for p in head.parameters() if p.requires_grad]
        print(f"  [Params] trainable={sum(p.numel() for p in params):,}")

        x_train = torch.from_numpy(features[domain]["train"]).to(DEVICE)
        y_train = torch.from_numpy(labels[domain]["train"]).long().to(DEVICE)
        train_labels_np = labels[domain]["train"]
        class_counts = class_counts_from_labels(train_labels_np)
        class_weights = None
        if LOSS_MODE in {"weighted_ce", "focal"} and CLASS_WEIGHT_MODE == "domain":
            class_weights = compute_class_weights_from_labels(train_labels_np)
        metadata["class_counts_by_domain"][domain] = [
            float(value) for value in class_counts.tolist()
        ]
        metadata["class_weights_by_domain"][domain] = (
            [float(value) for value in class_weights.tolist()]
            if class_weights is not None
            else None
        )
        class_counts = class_counts.to(DEVICE)
        class_weights = class_weights.to(DEVICE) if class_weights is not None else None

        optimizer = torch.optim.AdamW(
            params,
            lr=TRAIN_KWARGS["lr"],
            weight_decay=TRAIN_KWARGS["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TRAIN_KWARGS["epochs"],
            eta_min=TRAIN_KWARGS["lr"] * 0.01,
        )
        generator = torch.Generator()
        generator.manual_seed(SEED + i * 1009)
        best_score = -float("inf")
        best_epoch = 0
        best_state = None
        best_val_metrics = None
        best_val_calibrated_metrics = None

        for ep in range(1, TRAIN_KWARGS["epochs"] + 1):
            loss = train_one_epoch_features(
                head,
                x_train,
                y_train,
                optimizer,
                class_weights,
                class_counts,
                generator,
            )
            scheduler.step()
            val_labels, val_probs = collect_probs_features(
                head,
                features[domain]["val"],
                labels[domain]["val"],
            )
            val_metrics = binary_clinical_metrics(val_labels, val_probs, threshold=0.5)
            score_metrics = val_metrics
            val_calibrated_metrics = None
            if SELECT_CALIBRATED_EPOCH:
                _, val_calibrated_metrics = calibrate_threshold(
                    val_labels,
                    val_probs,
                    metric_name="f1",
                )
                score_metrics = val_calibrated_metrics
            score = _metric_score(score_metrics, SELECT_BY)
            if np.isnan(score):
                score = -float("inf")
            if score > best_score:
                best_score = score
                best_epoch = ep
                best_state = copy.deepcopy(head.state_dict())
                best_val_metrics = dict(val_metrics)
                best_val_calibrated_metrics = (
                    dict(val_calibrated_metrics)
                    if val_calibrated_metrics is not None
                    else None
                )
            if ep % 5 == 0 or ep == TRAIN_KWARGS["epochs"]:
                print(
                    f"  Epoch {ep:3d} | Loss: {loss:.4f} | "
                    f"Val {format_metric_line(val_metrics, clinical=True)}"
                    + (
                        f" | CalF1: {val_calibrated_metrics['f1']:.3f}"
                        if val_calibrated_metrics is not None
                        else ""
                    )
                )

        if best_state is not None:
            head.load_state_dict(best_state)
        val_labels, val_probs = collect_probs_features(
            head,
            features[domain]["val"],
            labels[domain]["val"],
        )
        threshold, calibrated_val_metrics = calibrate_threshold(
            val_labels,
            val_probs,
            metric_name="f1",
        )
        domain_thresholds[domain] = threshold
        metadata["best_epoch_by_domain"][domain] = int(best_epoch)
        metadata["best_threshold_by_domain"][domain] = float(threshold)
        metadata["best_val_metrics_by_domain"][domain] = {
            "selected_epoch_metrics_at_0_5": best_val_metrics,
            "selected_epoch_calibrated_metrics": best_val_calibrated_metrics,
            "calibrated_metrics": calibrated_val_metrics,
        }
        print(
            f"  [Best] epoch={best_epoch} | threshold={threshold:.3f} | "
            f"val_f1={calibrated_val_metrics.get('f1', float('nan')):.3f}"
        )

        print(f"\n  Evaluation after {i + 1}/{t} domains:")
        for j in range(i + 1):
            eval_domain = DOMAIN_NAMES[j]
            eval_head = heads[eval_domain]
            eval_labels, eval_probs = collect_probs_features(
                eval_head,
                features[eval_domain]["test"],
                labels[eval_domain]["test"],
            )
            metrics = binary_clinical_metrics(
                eval_labels,
                eval_probs,
                threshold=domain_thresholds.get(eval_domain, 0.5),
            )
            r_matrix[i][j] = metrics["f1"]
            for name, matrix in metric_matrices.items():
                if name in metrics:
                    matrix[i][j] = metrics[name]
            print(f"    {eval_domain:<12s}  {format_metric_line(metrics, clinical=True)}")

    forgetting = compute_forgetting(r_matrix, DOMAIN_NAMES)
    summary = {
        "experiment_config": metadata["config"],
        "mean_f1": float(np.nanmean(r_matrix[-1])),
        "mean_bwt": float(forgetting["mean_bwt"]),
        "final_f1_by_domain": {
            domain: float(r_matrix[-1, idx]) for idx, domain in enumerate(DOMAIN_NAMES)
        },
        "metadata": metadata,
    }
    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    metrics_path = output_dir / "final_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", *METRIC_NAMES])
        writer.writeheader()
        for idx, domain in enumerate(DOMAIN_NAMES):
            row = {"domain": domain}
            for name in METRIC_NAMES:
                value = metric_matrices[name][-1, idx]
                row[name] = "" if np.isnan(value) else float(value)
            writer.writerow(row)

    f1_path = output_dir / "final_f1.csv"
    with f1_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "final_f1"])
        for idx, domain in enumerate(DOMAIN_NAMES):
            writer.writerow([domain, float(r_matrix[-1, idx])])

    npz_path = output_dir / "continual_results.npz"
    np.savez_compressed(
        npz_path,
        r_matrix=r_matrix,
        **{f"{name}_matrix": matrix for name, matrix in metric_matrices.items()},
    )
    print(f"  [Saved] {summary_path}")
    print(f"  [Saved] {metrics_path}")
    print(f"  [Saved] {f1_path}")
    print(f"  [Saved] {npz_path}")
    print(
        f"  [Done] mean_f1={summary['mean_f1']:.6f} "
        f"mean_bwt={summary['mean_bwt']:.6f}"
    )
    return summary


def main() -> None:
    run_feature_head_bank()


if __name__ == "__main__":
    main()
