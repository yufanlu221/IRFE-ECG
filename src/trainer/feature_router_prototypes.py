"""Prototype routers on cached ECGFounder features.

This script keeps the source-aware head-bank training protocol, then replaces
oracle domain IDs at test time with simple feature-space routing rules. It is
intended for low-resource router-collapse diagnosis on shared servers.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from evaluate.clinical_metrics import binary_clinical_metrics, format_metric_line
from scripts.run_config import env_float, env_int, env_str
from trainer.continual_cl import (
    CLASS_WEIGHT_MODE,
    DATA_DIR,
    DEVICE,
    DOMAIN_NAMES,
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
)
from trainer.feature_head_bank import (
    FEATURE_CACHE_DIR,
    FEATURE_HEAD_METHOD,
    METRIC_NAMES,
    collect_probs_features,
    load_feature_cache,
    make_head,
    train_one_epoch_features,
)


CPU_THREADS = env_int("CPU_THREADS", env_int("ECG_CPU_THREADS", 1))
DOMAIN_ORDER = [
    domain.strip().lower()
    for domain in env_str("ECG_DOMAIN_ORDER", ",".join(DOMAIN_NAMES)).split(",")
    if domain.strip()
]
if sorted(DOMAIN_ORDER) != sorted(DOMAIN_NAMES):
    raise ValueError(
        "ECG_DOMAIN_ORDER must contain each configured domain exactly once: "
        f"{DOMAIN_NAMES}"
    )
DOMAIN_NAMES = DOMAIN_ORDER
ROUTER_MODES = [
    mode.strip().lower()
    for mode in env_str(
        "ECG_FEATURE_ROUTER_MODES",
        "centroid_cosine,class_conditional_cosine,diag_gaussian",
    ).split(",")
    if mode.strip()
]
ROUTER_EPS = env_float("ECG_FEATURE_ROUTER_EPS", 1e-4)
ROUTER_CLASS_AGG = env_str("ECG_FEATURE_ROUTER_CLASS_AGG", "max").lower()
ROUTER_SHRINKAGE = env_float("ECG_FEATURE_ROUTER_SHRINKAGE", 0.10)
ROUTER_USE_PRIORS = env_int("ECG_FEATURE_ROUTER_USE_PRIORS", 1) != 0
ROUTER_DOMAIN_EPOCHS = env_int("ECG_FEATURE_ROUTER_DOMAIN_EPOCHS", 50)
ROUTER_DOMAIN_LR = env_float("ECG_FEATURE_ROUTER_DOMAIN_LR", 1e-3)
ROUTER_DOMAIN_HIDDEN_DIM = env_int("ECG_FEATURE_ROUTER_DOMAIN_HIDDEN_DIM", 128)
ROUTER_DOMAIN_DROPOUT = env_float("ECG_FEATURE_ROUTER_DOMAIN_DROPOUT", 0.10)
ROUTER_DOMAIN_BATCH_SIZE = env_int("ECG_FEATURE_ROUTER_DOMAIN_BATCH_SIZE", 512)
ROUTER_DOMAIN_BALANCED = env_int("ECG_FEATURE_ROUTER_DOMAIN_BALANCED", 0) != 0
ROUTER_DOMAIN_SELECT = env_str("ECG_FEATURE_ROUTER_DOMAIN_SELECT", "accuracy").lower()
ROUTER_PREPROCESS = env_str("ECG_FEATURE_ROUTER_PREPROCESS", "none").lower()
ROUTER_TOPK = env_int("ECG_FEATURE_ROUTER_TOPK", 2)
ROUTER_SOFTMAX_TEMPERATURE = env_float("ECG_FEATURE_ROUTER_SOFTMAX_TEMPERATURE", 1.0)
ROUTER_MEMORY_FRACTION = env_float("ECG_FEATURE_ROUTER_MEMORY_FRACTION", 1.0)
ROUTER_KNN_K = env_int("ECG_FEATURE_ROUTER_KNN_K", 5)
ROUTER_KNN_QUERY_BATCH = env_int("ECG_FEATURE_ROUTER_KNN_QUERY_BATCH", 256)


def _base_router_mode(mode: str) -> str:
    if mode.endswith("_probavg_top2"):
        return mode[: -len("_probavg_top2")]
    if mode.endswith("_top2"):
        return mode[: -len("_top2")]
    return mode


def _is_topk_router_mode(mode: str) -> bool:
    return mode.endswith("_top2")


def _is_probability_average_mode(mode: str) -> bool:
    return mode.endswith("_probavg_top2")


def _output_dir() -> Path:
    if os.environ.get("ECG_OUTPUT_DIR"):
        return Path(OUTPUT_DIR)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes = "-".join(ROUTER_MODES)
    return Path(OUTPUT_DIR) / f"feature_router_{FEATURE_HEAD_METHOD}_{modes}_seed{SEED}_{stamp}"


def _normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, eps)


def _fit_router_preprocess(
    features: Dict[str, Dict[str, np.ndarray]],
    seen_domains: List[str],
    memory_indices: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    if ROUTER_PREPROCESS in {"", "none", "raw"}:
        return {"mode": "none"}
    x = np.concatenate(
        [
            features[domain]["train"][memory_indices[domain]].astype(np.float32)
            for domain in seen_domains
        ],
        axis=0,
    )
    if ROUTER_PREPROCESS in {"zscore", "standardize", "diag_whiten"}:
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32) + ROUTER_EPS
        return {"mode": "zscore", "mean": mean, "std": std}
    raise ValueError("ECG_FEATURE_ROUTER_PREPROCESS must be none or zscore.")


def _apply_router_preprocess(x: np.ndarray, preprocess: Dict[str, Any]) -> np.ndarray:
    mode = preprocess.get("mode", "none")
    x = x.astype(np.float32)
    if mode == "none":
        return x
    if mode == "zscore":
        return ((x - preprocess["mean"][None, :]) / preprocess["std"][None, :]).astype(
            np.float32
        )
    raise ValueError(f"Unsupported router preprocess mode: {mode}")


class DomainRouterMLP(nn.Module):
    def __init__(self, feature_dim: int, num_domains: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, ROUTER_DOMAIN_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(ROUTER_DOMAIN_DROPOUT),
            nn.Linear(ROUTER_DOMAIN_HIDDEN_DIM, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DomainRouterLinear(nn.Module):
    def __init__(self, feature_dim: int, num_domains: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_domains)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _router_memory_indices(domain: str, sample_count: int) -> np.ndarray:
    fraction = min(1.0, max(float(ROUTER_MEMORY_FRACTION), 0.0))
    if fraction >= 1.0:
        return np.arange(sample_count, dtype=np.int64)
    keep = max(1, int(round(sample_count * fraction)))
    domain_index = DOMAIN_NAMES.index(domain)
    rng = np.random.default_rng(SEED + 104729 * (domain_index + 1))
    return np.sort(rng.choice(sample_count, size=keep, replace=False)).astype(np.int64)


def _train_domain_head(
    domain: str,
    domain_index: int,
    feature_dim: int,
    features: Dict[str, Dict[str, np.ndarray]],
    labels: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[nn.Module, float, Dict[str, Any]]:
    head = make_head(feature_dim, FEATURE_HEAD_METHOD).to(DEVICE)
    params = [p for p in head.parameters() if p.requires_grad]
    print(f"  [Params] {domain} trainable={sum(p.numel() for p in params):,}")

    x_train = torch.from_numpy(features[domain]["train"]).to(DEVICE)
    y_train = torch.from_numpy(labels[domain]["train"]).long().to(DEVICE)
    train_labels_np = labels[domain]["train"]
    class_counts = class_counts_from_labels(train_labels_np)
    class_weights = None
    if LOSS_MODE in {"weighted_ce", "focal"} and CLASS_WEIGHT_MODE == "domain":
        class_weights = compute_class_weights_from_labels(train_labels_np)
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
    generator.manual_seed(SEED + domain_index * 1009)

    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    best_val_metrics = None
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
        score = _metric_score(val_metrics, SELECT_BY)
        if np.isnan(score):
            score = -float("inf")
        if score > best_score:
            best_score = score
            best_epoch = ep
            best_state = copy.deepcopy(head.state_dict())
            best_val_metrics = dict(val_metrics)
        if ep % 5 == 0 or ep == TRAIN_KWARGS["epochs"]:
            print(
                f"  Epoch {ep:3d} | Loss: {loss:.4f} | "
                f"Val {format_metric_line(val_metrics, clinical=True)}"
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
    info = {
        "best_epoch": int(best_epoch),
        "threshold": float(threshold),
        "best_val_metrics_at_0_5": best_val_metrics,
        "calibrated_val_metrics": calibrated_val_metrics,
    }
    print(
        f"  [Best] {domain} epoch={best_epoch} | threshold={threshold:.3f} | "
        f"val_f1={calibrated_val_metrics.get('f1', float('nan')):.3f}"
    )
    return head, float(threshold), info


def _fit_domain_classifier(
    features: Dict[str, Dict[str, np.ndarray]],
    seen_domains: List[str],
    feature_dim: int,
    task_index: int,
    preprocess: Dict[str, Any],
    memory_indices: Dict[str, np.ndarray],
    model_kind: str,
) -> Tuple[nn.Module | None, Dict[str, Any]]:
    if len(seen_domains) <= 1:
        return None, {"best_epoch": 0, "best_val_route_accuracy": 1.0}

    x_train = np.concatenate(
        [
            _apply_router_preprocess(features[domain]["train"], preprocess)
            [memory_indices[domain]]
            for domain in seen_domains
        ],
        axis=0,
    )
    y_train = np.concatenate(
        [
            np.full(memory_indices[domain].shape[0], idx, dtype=np.int64)
            for idx, domain in enumerate(seen_domains)
        ],
        axis=0,
    )
    x_val = np.concatenate(
        [
            _apply_router_preprocess(features[domain]["val"], preprocess)
            for domain in seen_domains
        ],
        axis=0,
    )
    y_val = np.concatenate(
        [
            np.full(features[domain]["val"].shape[0], idx, dtype=np.int64)
            for idx, domain in enumerate(seen_domains)
        ],
        axis=0,
    )

    if model_kind == "linear":
        model = DomainRouterLinear(feature_dim, len(seen_domains)).to(DEVICE)
    elif model_kind == "mlp":
        model = DomainRouterMLP(feature_dim, len(seen_domains)).to(DEVICE)
    else:
        raise ValueError(f"Unsupported domain classifier kind: {model_kind}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ROUTER_DOMAIN_LR,
        weight_decay=TRAIN_KWARGS["weight_decay"],
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + 7919 + task_index * 1009)

    x_train_t = torch.from_numpy(x_train).to(DEVICE)
    y_train_t = torch.from_numpy(y_train).long().to(DEVICE)
    x_val_t = torch.from_numpy(x_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).long().to(DEVICE)
    domain_loss_weights = None
    if ROUTER_DOMAIN_BALANCED:
        counts = np.bincount(y_train, minlength=len(seen_domains)).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / max(float(weights.mean()), 1e-12)
        domain_loss_weights = torch.from_numpy(weights.astype(np.float32)).to(DEVICE)

    best_state = None
    best_epoch = 0
    best_val_score = -float("inf")
    best_val_acc = -float("inf")
    best_val_macro_acc = -float("inf")
    for epoch in range(1, ROUTER_DOMAIN_EPOCHS + 1):
        model.train()
        perm = torch.randperm(x_train_t.shape[0], generator=generator).to(DEVICE)
        for start in range(0, x_train_t.shape[0], ROUTER_DOMAIN_BATCH_SIZE):
            idx = perm[start : start + ROUTER_DOMAIN_BATCH_SIZE]
            logits = model(x_train_t[idx])
            loss = torch.nn.functional.cross_entropy(
                logits,
                y_train_t[idx],
                weight=domain_loss_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x_val_t).argmax(dim=1)
            val_acc = float((pred == y_val_t).float().mean().item())
            pred_np = pred.detach().cpu().numpy()
        val_macro_acc = float(
            np.mean(
                [
                    (pred_np[y_val == idx] == idx).mean()
                    for idx in range(len(seen_domains))
                    if np.any(y_val == idx)
                ]
            )
        )
        val_score = val_macro_acc if ROUTER_DOMAIN_SELECT in {"macro_acc", "balanced_acc"} else val_acc
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_acc = val_acc
            best_val_macro_acc = val_macro_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "best_val_route_accuracy": float(best_val_acc),
        "best_val_route_macro_accuracy": float(best_val_macro_acc),
        "best_val_route_selection_score": float(best_val_score),
        "epochs": int(ROUTER_DOMAIN_EPOCHS),
        "lr": float(ROUTER_DOMAIN_LR),
        "hidden_dim": int(ROUTER_DOMAIN_HIDDEN_DIM),
        "dropout": float(ROUTER_DOMAIN_DROPOUT),
        "balanced_loss": bool(ROUTER_DOMAIN_BALANCED),
        "select_by": ROUTER_DOMAIN_SELECT,
        "model_kind": model_kind,
        "memory_fraction": float(ROUTER_MEMORY_FRACTION),
    }


def _pooled_shrinkage_inverse(
    router_train: Dict[str, np.ndarray],
    seen_domains: List[str],
    means: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    centered_chunks = []
    for domain in seen_domains:
        x = router_train[domain]
        centered_chunks.append(x - means[domain][None, :])
    centered = np.concatenate(centered_chunks, axis=0)
    denom = max(1, centered.shape[0] - len(seen_domains))
    cov = (centered.T @ centered) / float(denom)
    diag_cov = np.diag(np.diag(cov))
    shrunk = (1.0 - ROUTER_SHRINKAGE) * cov + ROUTER_SHRINKAGE * diag_cov
    shrunk = shrunk + ROUTER_EPS * np.eye(shrunk.shape[0], dtype=np.float32)
    inv_cov = np.linalg.pinv(shrunk).astype(np.float32)
    return inv_cov, {
        "shrinkage": float(ROUTER_SHRINKAGE),
        "pooled_samples": int(centered.shape[0]),
        "pooled_dim": int(centered.shape[1]),
    }


def _fit_router_stats(
    features: Dict[str, Dict[str, np.ndarray]],
    labels: Dict[str, Dict[str, np.ndarray]],
    seen_domains: List[str],
    feature_dim: int,
    task_index: int,
) -> Dict[str, Any]:
    memory_indices = {
        domain: _router_memory_indices(domain, features[domain]["train"].shape[0])
        for domain in seen_domains
    }
    preprocess = _fit_router_preprocess(features, seen_domains, memory_indices)
    router_train = {
        domain: _apply_router_preprocess(
            features[domain]["train"][memory_indices[domain]], preprocess
        )
        for domain in seen_domains
    }
    stats: Dict[str, Any] = {
        "_preprocess": preprocess,
        "_memory_counts": {
            domain: int(memory_indices[domain].shape[0]) for domain in seen_domains
        },
    }
    means: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for domain in seen_domains:
        x = router_train[domain]
        y = labels[domain]["train"][memory_indices[domain]].astype(np.int64)
        centroid = x.mean(axis=0).astype(np.float32)
        means[domain] = centroid
        counts[domain] = int(x.shape[0])
        var = x.var(axis=0).astype(np.float32) + ROUTER_EPS
        class_centroids = []
        for cls in (0, 1):
            cls_x = x[y == cls]
            if cls_x.shape[0] == 0:
                class_centroids.append(centroid)
            else:
                class_centroids.append(cls_x.mean(axis=0).astype(np.float32))
        stats[domain] = {
            "centroid": centroid,
            "centroid_norm": _normalize(centroid[None, :])[0],
            "var": var,
            "class_centroids": np.stack(class_centroids, axis=0),
            "class_centroids_norm": _normalize(np.stack(class_centroids, axis=0)),
        }
    if any(_base_router_mode(mode) == "shrinkage_lda" for mode in ROUTER_MODES):
        inv_cov, lda_info = _pooled_shrinkage_inverse(router_train, seen_domains, means)
        total_count = max(1, sum(counts.values()))
        for domain in seen_domains:
            stats[domain]["lda_mean"] = means[domain]
            stats[domain]["lda_prior"] = counts[domain] / float(total_count)
        stats["_lda_inv_cov"] = inv_cov
        stats["_lda_info"] = lda_info
    if any(_base_router_mode(mode) == "domain_mlp" for mode in ROUTER_MODES):
        model, info = _fit_domain_classifier(
            features,
            seen_domains,
            feature_dim,
            task_index,
            preprocess,
            memory_indices,
            "mlp",
        )
        stats["_domain_mlp"] = model
        stats["_domain_mlp_info"] = info
    if any(_base_router_mode(mode) == "domain_linear" for mode in ROUTER_MODES):
        model, info = _fit_domain_classifier(
            features,
            seen_domains,
            feature_dim,
            task_index,
            preprocess,
            memory_indices,
            "linear",
        )
        stats["_domain_linear"] = model
        stats["_domain_linear_info"] = info
    if any(_base_router_mode(mode) == "knn" for mode in ROUTER_MODES):
        stats["_knn_train"] = np.concatenate(
            [router_train[domain] for domain in seen_domains], axis=0
        ).astype(np.float32)
        stats["_knn_domain_labels"] = np.concatenate(
            [
                np.full(router_train[domain].shape[0], idx, dtype=np.int64)
                for idx, domain in enumerate(seen_domains)
            ],
            axis=0,
        )
    return stats


def _knn_router_scores(
    x: np.ndarray,
    train_x: np.ndarray,
    train_domains: np.ndarray,
    num_domains: int,
) -> np.ndarray:
    if train_x.shape[0] == 0:
        return np.zeros((x.shape[0], num_domains), dtype=np.float32)
    k = max(1, min(int(ROUTER_KNN_K), train_x.shape[0]))
    train_t = torch.from_numpy(_normalize(train_x).astype(np.float32)).to(DEVICE)
    train_domains_t = torch.from_numpy(train_domains.astype(np.int64)).to(DEVICE)
    output = []
    with torch.no_grad():
        for start in range(0, x.shape[0], ROUTER_KNN_QUERY_BATCH):
            query_t = torch.from_numpy(
                _normalize(x[start : start + ROUTER_KNN_QUERY_BATCH]).astype(np.float32)
            ).to(DEVICE)
            similarities = query_t @ train_t.T
            top_values, top_indices = torch.topk(similarities, k=k, dim=1)
            top_domains = train_domains_t[top_indices]
            scores = torch.zeros(
                (query_t.shape[0], num_domains), device=DEVICE, dtype=torch.float32
            )
            scores.scatter_add_(1, top_domains, torch.ones_like(top_values))
            scores.scatter_add_(1, top_domains, 1e-4 * top_values)
            output.append(scores.cpu().numpy())
    return np.concatenate(output, axis=0)


def _router_scores(
    x: np.ndarray,
    seen_domains: List[str],
    stats: Dict[str, Any],
    mode: str,
) -> np.ndarray:
    mode = _base_router_mode(mode)
    x = _apply_router_preprocess(x, stats.get("_preprocess", {"mode": "none"}))
    if mode == "centroid_cosine":
        x_norm = _normalize(x)
        proto = np.stack([stats[d]["centroid_norm"] for d in seen_domains], axis=0)
        return x_norm @ proto.T
    if mode == "class_conditional_cosine":
        x_norm = _normalize(x)
        scores = []
        for domain in seen_domains:
            class_scores = x_norm @ stats[domain]["class_centroids_norm"].T
            if ROUTER_CLASS_AGG == "mean":
                scores.append(class_scores.mean(axis=1))
            else:
                scores.append(class_scores.max(axis=1))
        return np.stack(scores, axis=1)
    if mode == "diag_gaussian":
        scores = []
        for domain in seen_domains:
            mean = stats[domain]["centroid"]
            var = stats[domain]["var"]
            nll = ((x - mean) ** 2 / var + np.log(var)).mean(axis=1)
            scores.append(-0.5 * nll)
        return np.stack(scores, axis=1)
    if mode == "shrinkage_lda":
        inv_cov = stats["_lda_inv_cov"]
        scores = []
        for domain in seen_domains:
            mean = stats[domain]["lda_mean"]
            linear = x @ inv_cov @ mean
            offset = -0.5 * float(mean @ inv_cov @ mean)
            prior = np.log(max(stats[domain]["lda_prior"], 1e-12)) if ROUTER_USE_PRIORS else 0.0
            scores.append(linear + offset + prior)
        return np.stack(scores, axis=1)
    if mode == "domain_mlp":
        model = stats.get("_domain_mlp")
        if model is None:
            return np.zeros((x.shape[0], len(seen_domains)), dtype=np.float32)
        model.eval()
        chunks = []
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32)).to(DEVICE)
            for start in range(0, x_t.shape[0], 4096):
                chunks.append(model(x_t[start : start + 4096]).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)
    if mode == "domain_linear":
        model = stats.get("_domain_linear")
        if model is None:
            return np.zeros((x.shape[0], len(seen_domains)), dtype=np.float32)
        model.eval()
        chunks = []
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32)).to(DEVICE)
            for start in range(0, x_t.shape[0], 4096):
                chunks.append(model(x_t[start : start + 4096]).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)
    if mode == "knn":
        return _knn_router_scores(
            x,
            stats["_knn_train"],
            stats["_knn_domain_labels"],
            len(seen_domains),
        )
    raise ValueError(
        "Unsupported router mode. Use centroid_cosine, "
        "class_conditional_cosine, diag_gaussian, shrinkage_lda, domain_linear, "
        "domain_mlp, or knn."
    )


@torch.no_grad()
def _positive_probs(head: nn.Module, x_np: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    head.eval()
    x = torch.from_numpy(x_np.astype(np.float32)).to(DEVICE)
    chunks = []
    for start in range(0, x.shape[0], batch_size):
        logits = head(x[start : start + batch_size]).detach().cpu()
        chunks.append(torch.softmax(logits, dim=-1)[:, 1])
    return torch.cat(chunks, dim=0).numpy()


def _routed_probs_and_preds(
    x: np.ndarray,
    seen_domains: List[str],
    route_domains: np.ndarray,
    heads: Dict[str, nn.Module],
    thresholds: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.zeros(x.shape[0], dtype=np.float32)
    preds = np.zeros(x.shape[0], dtype=np.int64)
    for seen_idx, domain in enumerate(seen_domains):
        sample_idx = np.flatnonzero(route_domains == seen_idx)
        if sample_idx.size == 0:
            continue
        domain_probs = _positive_probs(heads[domain], x[sample_idx])
        probs[sample_idx] = domain_probs
        preds[sample_idx] = (domain_probs >= thresholds.get(domain, 0.5)).astype(np.int64)
    return probs, preds


def _head_confidence_scores(
    x: np.ndarray,
    seen_domains: List[str],
    heads: Dict[str, nn.Module],
    thresholds: Dict[str, float],
) -> np.ndarray:
    """Score each domain head by distance from its validation-calibrated threshold."""

    scores = []
    for domain in seen_domains:
        probs = _positive_probs(heads[domain], x)
        scores.append(np.abs(probs - thresholds.get(domain, 0.5)))
    return np.stack(scores, axis=1).astype(np.float32)


def _softmax_np(x: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    z = x / temp
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.maximum(exp_z.sum(axis=1, keepdims=True), 1e-12)


def _routed_topk_probs_and_preds(
    x: np.ndarray,
    seen_domains: List[str],
    scores: np.ndarray,
    heads: Dict[str, nn.Module],
    thresholds: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    topk = max(1, min(int(ROUTER_TOPK), len(seen_domains)))
    top_idx = np.argsort(scores, axis=1)[:, -topk:]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    weights = _softmax_np(top_scores, ROUTER_SOFTMAX_TEMPERATURE).astype(np.float32)

    probs = np.zeros(x.shape[0], dtype=np.float32)
    margins = np.zeros(x.shape[0], dtype=np.float32)
    for rank in range(topk):
        for seen_idx, domain in enumerate(seen_domains):
            sample_idx = np.flatnonzero(top_idx[:, rank] == seen_idx)
            if sample_idx.size == 0:
                continue
            domain_probs = _positive_probs(heads[domain], x[sample_idx])
            sample_weights = weights[sample_idx, rank]
            probs[sample_idx] += sample_weights * domain_probs
            margins[sample_idx] += sample_weights * (
                domain_probs - thresholds.get(domain, 0.5)
            )
    preds = (margins >= 0.0).astype(np.int64)
    return probs, preds


def _routed_topk_probability_average(
    x: np.ndarray,
    seen_domains: List[str],
    scores: np.ndarray,
    heads: Dict[str, nn.Module],
) -> Tuple[np.ndarray, np.ndarray]:
    topk = max(1, min(int(ROUTER_TOPK), len(seen_domains)))
    top_idx = np.argsort(scores, axis=1)[:, -topk:]
    probs = np.zeros(x.shape[0], dtype=np.float32)
    for rank in range(topk):
        for seen_idx, domain in enumerate(seen_domains):
            sample_idx = np.flatnonzero(top_idx[:, rank] == seen_idx)
            if sample_idx.size == 0:
                continue
            probs[sample_idx] += _positive_probs(heads[domain], x[sample_idx]) / topk
    return probs, (probs >= 0.5).astype(np.int64)


def _write_final_metrics(
    path: Path,
    metric_matrices_by_setting: Dict[str, Dict[str, np.ndarray]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "domain", *METRIC_NAMES])
        writer.writeheader()
        for setting, metric_matrices in metric_matrices_by_setting.items():
            for idx, domain in enumerate(DOMAIN_NAMES):
                row = {"setting": setting, "domain": domain}
                for name in METRIC_NAMES:
                    value = metric_matrices[name][-1, idx]
                    row[name] = "" if np.isnan(value) else float(value)
                writer.writerow(row)


def _write_final_f1(path: Path, r_matrices: Dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "domain", "final_f1"])
        for setting, matrix in r_matrices.items():
            for idx, domain in enumerate(DOMAIN_NAMES):
                writer.writerow([setting, domain, float(matrix[-1, idx])])


def _write_route_accuracy(path: Path, route_matrices: Dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["router", "after_task", "domain", "route_accuracy"])
        for router, matrix in route_matrices.items():
            for i, after_domain in enumerate(DOMAIN_NAMES):
                for j, domain in enumerate(DOMAIN_NAMES):
                    value = matrix[i, j]
                    if not np.isnan(value):
                        writer.writerow([router, after_domain, domain, float(value)])


def run_feature_router_prototypes() -> Dict[str, Any]:
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
    print(f"  Router modes: {', '.join(ROUTER_MODES)}")
    print(f"  Seed: {SEED}")
    print(
        f"  Hyperparams: epochs={TRAIN_KWARGS['epochs']}, "
        f"lr={TRAIN_KWARGS['lr']}, batch={TRAIN_KWARGS['batch_size']}, "
        f"cpu_threads={CPU_THREADS}"
    )

    features, labels = load_feature_cache()
    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    t = len(DOMAIN_NAMES)
    feature_dim = int(features[DOMAIN_NAMES[0]]["train"].shape[1])
    heads: Dict[str, nn.Module] = {}
    thresholds: Dict[str, float] = {}
    head_metadata: Dict[str, Any] = {}
    router_metadata: Dict[str, Any] = {}

    settings = ["oracle", *ROUTER_MODES]
    r_matrices = {
        setting: np.full((t, t), np.nan, dtype=float) for setting in settings
    }
    metric_matrices_by_setting = {
        setting: {
            name: np.full((t, t), np.nan, dtype=float) for name in METRIC_NAMES
        }
        for setting in settings
    }
    route_matrices = {
        mode: np.full((t, t), np.nan, dtype=float) for mode in ROUTER_MODES
    }
    final_prediction_records: Dict[str, np.ndarray] = {}

    for i, domain in enumerate(DOMAIN_NAMES):
        print(f"\n{'=' * 60}")
        print(f"  Router prototype | Task {i + 1}/{t}: {domain}")
        print(f"{'=' * 60}")
        head, threshold, info = _train_domain_head(
            domain,
            i,
            feature_dim,
            features,
            labels,
        )
        heads[domain] = head
        thresholds[domain] = threshold
        head_metadata[domain] = info

        seen_domains = DOMAIN_NAMES[: i + 1]
        router_stats = _fit_router_stats(
            features,
            labels,
            seen_domains,
            feature_dim,
            i,
        )
        router_metadata[domain] = {
            "seen_domains": list(seen_domains),
            "lda_info": router_stats.get("_lda_info"),
            "domain_mlp_info": router_stats.get("_domain_mlp_info"),
            "domain_linear_info": router_stats.get("_domain_linear_info"),
            "memory_counts": router_stats.get("_memory_counts"),
            "preprocess": router_stats.get("_preprocess", {"mode": "none"}).get("mode"),
        }
        print(f"\n  Evaluation after {i + 1}/{t} domains:")
        for j, eval_domain in enumerate(seen_domains):
            x_test = features[eval_domain]["test"]
            y_test = labels[eval_domain]["test"].astype(int)

            oracle_probs = _positive_probs(heads[eval_domain], x_test)
            oracle_metrics = binary_clinical_metrics(
                y_test,
                oracle_probs,
                threshold=thresholds.get(eval_domain, 0.5),
            )
            r_matrices["oracle"][i, j] = oracle_metrics["f1"]
            for name in METRIC_NAMES:
                if name in oracle_metrics:
                    metric_matrices_by_setting["oracle"][name][i, j] = oracle_metrics[name]
            if i == t - 1:
                prefix = f"oracle__{eval_domain}"
                final_prediction_records[f"{prefix}__labels"] = y_test.astype(np.int64)
                final_prediction_records[f"{prefix}__probs"] = oracle_probs.astype(np.float32)
                final_prediction_records[f"{prefix}__preds"] = (
                    oracle_probs >= thresholds.get(eval_domain, 0.5)
                ).astype(np.int64)
            print(
                f"    oracle/{eval_domain:<8s} "
                f"{format_metric_line(oracle_metrics, clinical=True)}"
            )

            true_seen_idx = seen_domains.index(eval_domain)
            for mode in ROUTER_MODES:
                if _base_router_mode(mode) == "head_confidence":
                    scores = _head_confidence_scores(
                        x_test,
                        seen_domains,
                        heads,
                        thresholds,
                    )
                else:
                    scores = _router_scores(x_test, seen_domains, router_stats, mode)
                route_domains = scores.argmax(axis=1)
                route_acc = float((route_domains == true_seen_idx).mean())
                if _is_topk_router_mode(mode):
                    if _is_probability_average_mode(mode):
                        probs, preds = _routed_topk_probability_average(
                            x_test,
                            seen_domains,
                            scores,
                            heads,
                        )
                    else:
                        probs, preds = _routed_topk_probs_and_preds(
                            x_test,
                            seen_domains,
                            scores,
                            heads,
                            thresholds,
                        )
                else:
                    probs, preds = _routed_probs_and_preds(
                        x_test,
                        seen_domains,
                        route_domains,
                        heads,
                        thresholds,
                    )
                metrics = binary_clinical_metrics(
                    y_test,
                    probs,
                    preds=preds,
                    threshold=0.5,
                )
                r_matrices[mode][i, j] = metrics["f1"]
                route_matrices[mode][i, j] = route_acc
                for name in METRIC_NAMES:
                    if name in metrics:
                        metric_matrices_by_setting[mode][name][i, j] = metrics[name]
                if i == t - 1:
                    prefix = f"{mode}__{eval_domain}"
                    final_prediction_records[f"{prefix}__labels"] = y_test.astype(np.int64)
                    final_prediction_records[f"{prefix}__probs"] = probs.astype(np.float32)
                    final_prediction_records[f"{prefix}__preds"] = preds.astype(np.int64)
                    final_prediction_records[f"{prefix}__routes"] = route_domains.astype(np.int64)
                print(
                    f"    {mode}/{eval_domain:<8s} "
                    f"{format_metric_line(metrics, clinical=True)} | "
                    f"RouteAcc: {route_acc:.3f}"
                )

    forgetting = {
        setting: compute_forgetting(matrix, DOMAIN_NAMES)
        for setting, matrix in r_matrices.items()
    }
    summary = {
        "experiment_config": {
            "seed": SEED,
            "val_ratio": VAL_RATIO,
            "feature_cache_dir": str(FEATURE_CACHE_DIR),
            "feature_head_method": FEATURE_HEAD_METHOD,
            "router_modes": ROUTER_MODES,
            "router_class_agg": ROUTER_CLASS_AGG,
            "router_shrinkage": ROUTER_SHRINKAGE,
            "router_use_priors": ROUTER_USE_PRIORS,
            "router_domain_epochs": ROUTER_DOMAIN_EPOCHS,
            "router_domain_lr": ROUTER_DOMAIN_LR,
            "router_domain_balanced": ROUTER_DOMAIN_BALANCED,
            "router_domain_select": ROUTER_DOMAIN_SELECT,
            "router_preprocess": ROUTER_PREPROCESS,
            "router_topk": ROUTER_TOPK,
            "router_softmax_temperature": ROUTER_SOFTMAX_TEMPERATURE,
            "router_memory_fraction": ROUTER_MEMORY_FRACTION,
            "router_knn_k": ROUTER_KNN_K,
            "domain_order": DOMAIN_NAMES,
            "router_stats_protocol": "incremental_seen_train_only",
            "loss_mode": LOSS_MODE,
            "class_weight_mode": CLASS_WEIGHT_MODE,
            "select_by": SELECT_BY,
            "train_kwargs": TRAIN_KWARGS,
            "cpu_threads": CPU_THREADS,
        },
        "settings": {},
        "head_metadata": head_metadata,
        "router_metadata": router_metadata,
    }
    for setting, matrix in r_matrices.items():
        route_final = None
        if setting in route_matrices:
            route_final = {
                domain: float(route_matrices[setting][-1, idx])
                for idx, domain in enumerate(DOMAIN_NAMES)
            }
        summary["settings"][setting] = {
            "mean_f1": float(np.nanmean(matrix[-1])),
            "mean_bwt": float(forgetting[setting]["mean_bwt"]),
            "final_f1_by_domain": {
                domain: float(matrix[-1, idx])
                for idx, domain in enumerate(DOMAIN_NAMES)
            },
            "final_route_accuracy_by_domain": route_final,
        }

    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    metrics_path = output_dir / "final_metrics.csv"
    _write_final_metrics(metrics_path, metric_matrices_by_setting)

    f1_path = output_dir / "final_f1.csv"
    _write_final_f1(f1_path, r_matrices)

    route_path = output_dir / "route_accuracy.csv"
    _write_route_accuracy(route_path, route_matrices)

    npz_path = output_dir / "continual_results.npz"
    np.savez_compressed(
        npz_path,
        **{f"{setting}_r_matrix": matrix for setting, matrix in r_matrices.items()},
        **{
            f"{setting}_{name}_matrix": matrix
            for setting, metric_matrices in metric_matrices_by_setting.items()
            for name, matrix in metric_matrices.items()
        },
        **{f"{mode}_route_accuracy": matrix for mode, matrix in route_matrices.items()},
        **final_prediction_records,
    )

    print(f"  [Saved] {summary_path}")
    print(f"  [Saved] {metrics_path}")
    print(f"  [Saved] {f1_path}")
    print(f"  [Saved] {route_path}")
    print(f"  [Saved] {npz_path}")
    for setting in settings:
        setting_summary = summary["settings"][setting]
        print(
            f"  [Done] {setting}: "
            f"mean_f1={setting_summary['mean_f1']:.6f} "
            f"mean_bwt={setting_summary['mean_bwt']:.6f}"
        )
    return summary


def main() -> None:
    run_feature_router_prototypes()


if __name__ == "__main__":
    main()
