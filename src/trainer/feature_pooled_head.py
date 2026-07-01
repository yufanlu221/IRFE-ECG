"""Offline pooled frozen-feature baseline without domain routing."""

from __future__ import annotations

import copy
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from evaluate.clinical_metrics import binary_clinical_metrics
from scripts.run_config import env_int
from trainer.continual_cl import (
    CLASS_WEIGHT_MODE,
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


CPU_THREADS = env_int("CPU_THREADS", env_int("ECG_CPU_THREADS", 2))


def _output_dir() -> Path:
    if os.environ.get("ECG_OUTPUT_DIR"):
        return Path(OUTPUT_DIR)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(OUTPUT_DIR) / f"feature_pooled_{FEATURE_HEAD_METHOD}_seed{SEED}_{stamp}"


def run_feature_pooled_head() -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
    torch.set_num_threads(max(1, CPU_THREADS))
    torch.set_num_interop_threads(1)
    set_global_seed(SEED)

    features, labels = load_feature_cache()
    feature_dim = int(features[DOMAIN_NAMES[0]]["train"].shape[1])
    x_train = np.concatenate([features[d]["train"] for d in DOMAIN_NAMES], axis=0)
    y_train = np.concatenate([labels[d]["train"] for d in DOMAIN_NAMES], axis=0).astype(np.int64)
    x_val = np.concatenate([features[d]["val"] for d in DOMAIN_NAMES], axis=0)
    y_val = np.concatenate([labels[d]["val"] for d in DOMAIN_NAMES], axis=0).astype(np.int64)

    model = make_head(feature_dim, FEATURE_HEAD_METHOD).to(DEVICE)
    x_train_t = torch.from_numpy(x_train.astype(np.float32)).to(DEVICE)
    y_train_t = torch.from_numpy(y_train).long().to(DEVICE)
    class_counts = class_counts_from_labels(y_train).to(DEVICE)
    class_weights = None
    if LOSS_MODE in {"weighted_ce", "focal"} and CLASS_WEIGHT_MODE == "domain":
        class_weights = compute_class_weights_from_labels(y_train).to(DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=TRAIN_KWARGS["lr"],
        weight_decay=TRAIN_KWARGS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAIN_KWARGS["epochs"],
        eta_min=TRAIN_KWARGS["lr"] * 0.01,
    )
    generator = torch.Generator().manual_seed(SEED + 4241)
    best_state = None
    best_epoch = 0
    best_score = -float("inf")
    for epoch in range(1, TRAIN_KWARGS["epochs"] + 1):
        loss = train_one_epoch_features(
            model,
            x_train_t,
            y_train_t,
            optimizer,
            class_weights,
            class_counts,
            generator,
        )
        scheduler.step()
        val_labels, val_probs = collect_probs_features(model, x_val, y_val)
        val_metrics = binary_clinical_metrics(val_labels, val_probs, threshold=0.5)
        score = _metric_score(val_metrics, SELECT_BY)
        if not np.isnan(score) and score > best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch == TRAIN_KWARGS["epochs"]:
            print(
                f"Epoch {epoch:3d} | loss={loss:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | val_auc={val_metrics['auc']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    val_labels, val_probs = collect_probs_features(model, x_val, y_val)
    threshold, calibrated_val = calibrate_threshold(val_labels, val_probs, metric_name="f1")

    rows = []
    prediction_records: Dict[str, np.ndarray] = {}
    for domain in DOMAIN_NAMES:
        y_test, probs = collect_probs_features(
            model,
            features[domain]["test"],
            labels[domain]["test"],
        )
        metrics = binary_clinical_metrics(y_test, probs, threshold=threshold)
        rows.append({"domain": domain, **{name: metrics.get(name) for name in METRIC_NAMES}})
        prediction_records[f"{domain}__labels"] = y_test.astype(np.int64)
        prediction_records[f"{domain}__probs"] = probs.astype(np.float32)
        prediction_records[f"{domain}__preds"] = (probs >= threshold).astype(np.int64)

    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "final_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment_config": {
            "method": "pooled_single_head",
            "seed": SEED,
            "val_ratio": VAL_RATIO,
            "domains": DOMAIN_NAMES,
            "feature_cache_dir": str(FEATURE_CACHE_DIR),
            "feature_head_method": FEATURE_HEAD_METHOD,
            "loss_mode": LOSS_MODE,
            "class_weight_mode": CLASS_WEIGHT_MODE,
            "train_kwargs": TRAIN_KWARGS,
            "selection": "validation_macro_f1_at_0.5_then_validation_threshold_calibration",
        },
        "best_epoch": int(best_epoch),
        "threshold": float(threshold),
        "calibrated_validation_metrics": calibrated_val,
        "mean_f1": float(np.mean([float(row["f1"]) for row in rows])),
        "final_metrics_by_domain": {
            row["domain"]: {name: row.get(name) for name in METRIC_NAMES} for row in rows
        },
    }
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(output_dir / "final_predictions.npz", **prediction_records)
    print(f"[Done] pooled_single_head mean_f1={summary['mean_f1']:.6f}")
    print(f"[Saved] {output_dir}")
    return summary


def main() -> None:
    run_feature_pooled_head()


if __name__ == "__main__":
    main()
