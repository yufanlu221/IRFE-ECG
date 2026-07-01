"""Joint offline upper-bound baseline for the ECG domain sequence.

This script trains frozen-backbone domain heads when all domains are available
up front. It is not a continual-learning method; it is a reference upper bound
for the domain-aware/oracle setting. Validation is used for model selection and
threshold calibration, and held-out test data is used only once for reporting.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from evaluate.clinical_metrics import format_metric_line
from trainer.continual_cl import (
    BACKBONE_KWARGS,
    CHECKPOINT_CANDIDATES,
    CLASS_WEIGHT_MODE,
    DATA_DIR,
    DEVICE,
    DOMAIN_NAMES,
    DRY_RUN,
    EVAL_USE_MERGE,
    LABEL_SMOOTHING,
    LOSS_MODE,
    NUM_WORKERS,
    OUTPUT_DIR,
    SELECT_BY,
    SEED,
    TRAIN_KWARGS,
    VAL_RATIO,
    _metric_score,
    calibrate_threshold,
    class_counts_from_labels,
    compute_class_weights_from_labels,
    get_cinc_train_val_test_loaders,
    set_global_seed,
)
from trainer.shared_baselines import (
    DOMAIN_FEATURE_ADAPTER_DIM,
    DOMAIN_FEATURE_ADAPTER_DROPOUT,
    DOMAIN_FEATURE_ADAPTER_SCALE,
    DOMAIN_MLP_DROPOUT,
    DOMAIN_MLP_HIDDEN_DIM,
    METRIC_NAMES,
    DomainLinearHeadsModel,
    _json_ready_shared,
    build_shared_model,
    collect_labels_probs_shared,
    evaluate_shared,
    restore_model_state,
    snapshot_model_state,
    train_one_epoch_shared,
)
from scripts.run_config import env_bool, env_str


JOINT_HEAD_KIND = env_str("ECG_JOINT_HEAD_KIND", "linear").lower()
SELECT_CALIBRATED_EPOCH = env_bool(
    "ECG_SELECT_CALIBRATED_EPOCH",
    env_bool("ECG_SELECT_EPOCH_AFTER_CALIBRATION", False),
)


def _output_dir() -> Path:
    if os.environ.get("ECG_OUTPUT_DIR"):
        return Path(OUTPUT_DIR)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(OUTPUT_DIR) / f"joint_offline_{JOINT_HEAD_KIND}_seed{SEED}_{stamp}"


def _head_kind() -> str:
    aliases = {
        "linear": "linear",
        "layernorm": "layernorm_linear",
        "ln": "layernorm_linear",
        "layernorm_linear": "layernorm_linear",
        "mlp": "mlp",
        "residual": "residual_adapter",
        "residual_adapter": "residual_adapter",
    }
    if JOINT_HEAD_KIND not in aliases:
        raise ValueError(
            "ECG_JOINT_HEAD_KIND must be one of linear, layernorm, mlp, residual."
        )
    return aliases[JOINT_HEAD_KIND]


def _mean_metric(final_metrics: Dict[str, Dict[str, float]], name: str) -> float:
    values = [metrics.get(name, float("nan")) for metrics in final_metrics.values()]
    return float(np.nanmean(values)) if values else float("nan")


def _write_csv(output_dir: Path, final_metrics: Dict[str, Dict[str, float]]) -> None:
    path = output_dir / "final_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("domain," + ",".join(METRIC_NAMES) + "\n")
        for domain in DOMAIN_NAMES:
            metrics = final_metrics[domain]
            cells = [
                "" if np.isnan(float(metrics.get(name, float("nan")))) else f"{metrics[name]:.6f}"
                for name in METRIC_NAMES
            ]
            f.write(domain + "," + ",".join(cells) + "\n")


def run_joint_offline() -> Dict[str, Any]:
    domains = DOMAIN_NAMES
    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loaders, val_loaders, test_loaders, split_infos, train_labels = (
        get_cinc_train_val_test_loaders(
            data_dir=DATA_DIR,
            domains=domains,
            seed=SEED,
            val_ratio=VAL_RATIO,
        )
    )

    model = DomainLinearHeadsModel(
        backbone=build_shared_model(return_features=True),
        domain_names=domains,
        feature_dim=int(BACKBONE_KWARGS["filter_list"][-1]),
        num_classes=int(BACKBONE_KWARGS["n_classes"]),
        head_kind=_head_kind(),
        hidden_dim=DOMAIN_MLP_HIDDEN_DIM,
        dropout=DOMAIN_MLP_DROPOUT,
        adapter_dim=DOMAIN_FEATURE_ADAPTER_DIM,
        adapter_dropout=DOMAIN_FEATURE_ADAPTER_DROPOUT,
        adapter_scale=DOMAIN_FEATURE_ADAPTER_SCALE,
    ).to(DEVICE)

    metadata: Dict[str, Any] = {
        "config": {
            "baseline_method": "joint_offline",
            "baseline_family": "offline_domain_aware_upper_bound",
            "joint_head_kind": JOINT_HEAD_KIND,
            "seed": SEED,
            "val_ratio": VAL_RATIO,
            "loss_mode": LOSS_MODE,
            "class_weight_mode": CLASS_WEIGHT_MODE,
            "label_smoothing": LABEL_SMOOTHING,
            "select_by": SELECT_BY,
            "epoch_selection_threshold": 0.5,
            "select_epoch_after_threshold_calibration": SELECT_CALIBRATED_EPOCH,
            "num_workers": NUM_WORKERS,
            "threshold_calibration_stage": "after_best_epoch_restore_on_validation",
            "eval_use_merge": EVAL_USE_MERGE,
            "train_kwargs": TRAIN_KWARGS,
            "note": (
                "All domain train/val splits are available up front. Test splits "
                "are held out until the final evaluation. This is an upper bound, "
                "not a continual-learning method."
            ),
        },
        "output_dir": str(output_dir),
        "split_info": split_infos or {},
        "best_epoch_by_domain": {},
        "best_threshold_by_domain": {},
        "best_val_metrics_by_domain": {},
        "class_counts_by_domain": {},
        "class_weights_by_domain": {},
        "checkpoint_candidates": [str(path) for path in CHECKPOINT_CANDIDATES],
    }

    for domain in domains:
        print(f"\n{'=' * 60}")
        print(f"  Joint offline | domain head: {domain}")
        print(f"{'=' * 60}")
        model.set_domain(domain)
        train_loader = train_loaders[domain]
        val_loader = val_loaders[domain]
        train_subset_labels = train_labels.get(domain)

        class_counts = None
        class_weights = None
        if train_subset_labels is not None:
            class_counts = class_counts_from_labels(train_subset_labels)
            if LOSS_MODE in {"weighted_ce", "focal"} and CLASS_WEIGHT_MODE == "domain":
                class_weights = compute_class_weights_from_labels(train_subset_labels)
            metadata["class_counts_by_domain"][domain] = [
                float(value) for value in class_counts.tolist()
            ]
            metadata["class_weights_by_domain"][domain] = (
                [float(value) for value in class_weights.tolist()]
                if class_weights is not None
                else None
            )

        optimizer = torch.optim.AdamW(
            model.get_trainable_params(),
            lr=TRAIN_KWARGS["lr"],
            weight_decay=TRAIN_KWARGS["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TRAIN_KWARGS["epochs"],
            eta_min=TRAIN_KWARGS["lr"] * 0.01,
        )
        best_score = -float("inf")
        best_epoch = 0
        best_state = None
        best_val_metrics = None
        best_val_calibrated_metrics = None

        print(f"  Train n={len(train_loader.dataset)} | Val n={len(val_loader.dataset)}")
        for ep in range(1, TRAIN_KWARGS["epochs"] + 1):
            loss = train_one_epoch_shared(
                model,
                train_loader,
                optimizer,
                method="joint_offline",
                class_weights=class_weights,
                class_counts=class_counts,
                domain=domain,
            )
            scheduler.step()

            val_labels, val_probs = collect_labels_probs_shared(
                model,
                val_loader,
                domain=domain,
            )
            val_threshold_metrics = None
            val_metrics = None
            if ep == 1 or ep % 1 == 0 or ep == TRAIN_KWARGS["epochs"]:
                from evaluate.clinical_metrics import binary_clinical_metrics

                val_metrics = binary_clinical_metrics(
                    labels=val_labels,
                    probs=val_probs,
                    threshold=0.5,
                )
                score_metrics = val_metrics
                if SELECT_CALIBRATED_EPOCH:
                    _, val_threshold_metrics = calibrate_threshold(
                        val_labels,
                        val_probs,
                        metric_name="f1",
                    )
                    score_metrics = val_threshold_metrics
                score = _metric_score(score_metrics, SELECT_BY)
                if np.isnan(score):
                    score = -float("inf")
                if score > best_score:
                    best_score = score
                    best_epoch = ep
                    best_state = snapshot_model_state(model)
                    best_val_metrics = dict(val_metrics)
                    best_val_calibrated_metrics = (
                        dict(val_threshold_metrics)
                        if val_threshold_metrics is not None
                        else None
                    )

            if ep % 5 == 0 or ep == TRAIN_KWARGS["epochs"]:
                suffix = ""
                if val_threshold_metrics is not None:
                    suffix = f" | CalF1: {val_threshold_metrics['f1']:.3f}"
                print(
                    f"  Epoch {ep:3d} | Loss: {loss:.4f} | "
                    f"Val {format_metric_line(val_metrics, clinical=True)}{suffix}"
                )

        if best_state is not None:
            restore_model_state(model, best_state)
        else:
            best_epoch = TRAIN_KWARGS["epochs"]
            best_val_metrics = {}

        val_labels, val_probs = collect_labels_probs_shared(model, val_loader, domain=domain)
        best_threshold, calibrated_val_metrics = calibrate_threshold(
            val_labels,
            val_probs,
            metric_name="f1",
        )
        metadata["best_epoch_by_domain"][domain] = int(best_epoch)
        metadata["best_threshold_by_domain"][domain] = float(best_threshold)
        metadata["best_val_metrics_by_domain"][domain] = {
            "selected_epoch_metrics_at_0_5": best_val_metrics,
            "selected_epoch_calibrated_metrics": best_val_calibrated_metrics,
            "calibrated_metrics": calibrated_val_metrics,
        }
        print(
            f"  [Best] epoch={best_epoch} | threshold={best_threshold:.3f} | "
            f"val_f1={calibrated_val_metrics.get('f1', float('nan')):.3f}"
        )

    print("\nFinal held-out test evaluation:")
    final_metrics: Dict[str, Dict[str, float]] = {}
    for domain in domains:
        metrics = evaluate_shared(
            model,
            test_loaders[domain],
            domain=domain,
            threshold=metadata["best_threshold_by_domain"][domain],
        )
        final_metrics[domain] = metrics
        print(f"  {domain:<12s} {format_metric_line(metrics, clinical=True)}")

    summary = {
        "domains": domains,
        "device": str(DEVICE),
        "data_dir": str(DATA_DIR),
        "checkpoint_candidates": [str(path) for path in CHECKPOINT_CANDIDATES],
        "train_kwargs": TRAIN_KWARGS,
        "experiment_config": metadata["config"],
        "split_info": metadata["split_info"],
        "best_epoch_by_domain": metadata["best_epoch_by_domain"],
        "best_threshold_by_domain": metadata["best_threshold_by_domain"],
        "best_val_metrics_by_domain": metadata["best_val_metrics_by_domain"],
        "class_counts_by_domain": metadata["class_counts_by_domain"],
        "class_weights_by_domain": metadata["class_weights_by_domain"],
        "final_f1_by_domain": {
            domain: float(final_metrics[domain]["f1"]) for domain in domains
        },
        "final_metrics_by_domain": final_metrics,
        "mean_f1": _mean_metric(final_metrics, "f1"),
        "mean_auc": _mean_metric(final_metrics, "auc"),
        "mean_auprc": _mean_metric(final_metrics, "auprc"),
        "mean_balanced_acc": _mean_metric(final_metrics, "balanced_acc"),
        "mean_bwt": None,
        "mean_forgetting": None,
        "bwt_note": "Not applicable: joint offline is not a sequential CL run.",
    }

    _write_csv(output_dir, final_metrics)
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_ready_shared(summary), f, indent=2)
    with (output_dir / "baseline_config.json").open("w", encoding="utf-8") as f:
        json.dump(_json_ready_shared(metadata["config"]), f, indent=2)
    print(f"  [Done] summary: {output_dir / 'metrics_summary.json'}")
    return summary


def main() -> None:
    set_global_seed(SEED)
    print(f"  Device: {DEVICE}")
    print(f"  Baseline method: joint_offline")
    print(f"  Domains: {' -> '.join(DOMAIN_NAMES)}")
    print(f"  Seed: {SEED}")
    print(f"  Head kind: {JOINT_HEAD_KIND}")
    print(
        f"  Hyperparams: epochs={TRAIN_KWARGS['epochs']}, "
        f"lr={TRAIN_KWARGS['lr']}, batch={TRAIN_KWARGS['batch_size']}"
    )
    if DRY_RUN or DEVICE.type != "cuda":
        reason = "dry-run mode" if DRY_RUN else "no CUDA device or forced CPU"
        print(f"\n  [Skip] {reason}; skipping joint offline training.")
        return
    run_joint_offline()


if __name__ == "__main__":
    main()
