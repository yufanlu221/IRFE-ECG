"""Run the appendix expert/loss ablation on frozen ECGFounder features.

The script deliberately has no backbone or router training path.  Model/epoch
selection, threshold calibration, and validation-selected bank construction use
validation labels only.  Test labels are read only after a candidate expert has
been fixed, and are used solely to report held-out metrics.

The paper's main linear + Balanced Softmax expert bank can be reused from the
matched primary router runs.  This preserves the exact published source-aware
oracle reference while the two missing controls are trained independently from
the same immutable, seed-specific feature caches.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.clinical_metrics import binary_clinical_metrics  # noqa: E402


DOMAINS = ("cpsc", "ptbxl", "georgia", "chapman")
DOMAIN_LABELS = {
    "cpsc": "CPSC",
    "ptbxl": "PTB-XL",
    "georgia": "Georgia",
    "chapman": "Chapman",
}
VARIANTS = (
    "linear_balanced_softmax",
    "linear_weighted_ce",
    "residual_adapter_balanced_softmax",
)
VARIANT_LABELS = {
    "linear_balanced_softmax": "Linear + Balanced Softmax",
    "linear_weighted_ce": "Linear + weighted CE",
    "residual_adapter_balanced_softmax": "Residual adapter + Balanced Softmax",
    "validation_selected_bank": "Validation-selected bank",
}
METRIC_NAMES = (
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
)
EXPECTED_MAIN_MEAN = 0.791529
EXPECTED_MAIN_STD = 0.003608
THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: Tuple[int, ...] = (42, 43, 44)
    feature_dim: int = 1024
    val_ratio: float = 0.15
    epochs: int = 30
    learning_rate: float = 5e-4
    batch_size: int = 32
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    adapter_dim: int = 128
    adapter_dropout: float = 0.1
    adapter_scale: float = 1.0
    max_class_weight: float = 10.0


class RunLog:
    """Small tee logger used for one variant/seed training run."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")

    def write(self, message: str = "") -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class ResidualFeatureExpert(nn.Module):
    """Lightweight residual adapter followed by a binary linear expert."""

    def __init__(
        self,
        feature_dim: int,
        adapter_dim: int,
        dropout: float,
        scale: float,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, feature_dim),
        )
        # Start from a pure linear expert and let validation decide whether the
        # residual path is useful.
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        adapted = features + self.scale * self.adapter(features)
        return self.classifier(adapted)


def parse_args() -> argparse.Namespace:
    default_cache = Path(os.environ.get("ECG_FEATURE_CACHE_DIR", "feature_cache"))
    default_main = Path(os.environ.get("ECG_MAIN_RESULTS_ROOT", "outputs"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache-dir", type=Path, default=default_cache)
    parser.add_argument("--main-results-root", type=Path, default=default_main)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "expert_loss_ablation",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument(
        "--force-train-main",
        action="store_true",
        help="Train the main bank instead of reusing the verified primary-run artifacts.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Do not save best checkpoints for newly trained controls.",
    )
    parser.add_argument("--main-mean-tolerance", type=float, default=1e-3)
    parser.add_argument("--main-std-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cache_path(cache_dir: Path, seed: int, domain: str, split: str) -> Path:
    return cache_dir / f"ecgfounder_final_seed{seed}_val0p15_{domain}_{split}.npz"


def load_seed_cache(
    cache_dir: Path,
    seed: int,
    expected_dim: int,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    cache: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for domain in DOMAINS:
        cache[domain] = {}
        for split in ("train", "val", "test"):
            path = cache_path(cache_dir, seed, domain, split)
            if not path.is_file():
                raise FileNotFoundError(f"Missing frozen feature cache: {path}")
            with np.load(path) as data:
                features = data["features"].astype(np.float32, copy=False)
                labels = data["labels"].astype(np.int64, copy=False)
            if features.ndim != 2 or features.shape[1] != expected_dim:
                raise ValueError(
                    f"Unexpected feature shape in {path}: {features.shape}; "
                    f"expected (*, {expected_dim})"
                )
            if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
                raise ValueError(f"Feature/label mismatch in {path}")
            unique_labels = set(np.unique(labels).tolist())
            if not unique_labels.issubset({0, 1}):
                raise ValueError(f"Non-binary class mapping in {path}: {unique_labels}")
            cache[domain][split] = (features, labels)
    return cache


def cache_manifest(cache_dir: Path, seeds: Sequence[int]) -> List[Dict[str, Any]]:
    manifest = []
    for seed in seeds:
        for domain in DOMAINS:
            for split in ("train", "val", "test"):
                path = cache_path(cache_dir, seed, domain, split)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing frozen feature cache: {path}")
                stat = path.stat()
                manifest.append(
                    {
                        "seed": seed,
                        "domain": domain,
                        "split": split,
                        "path": str(path.resolve()),
                        "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
    return manifest


def class_counts_and_weights(
    labels: np.ndarray,
    max_class_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute n/(2*n_class) weights from the current train subset only."""

    counts = np.bincount(labels.astype(np.int64), minlength=2).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0)
    weights = safe_counts.sum() / (2.0 * safe_counts)
    return counts, np.minimum(weights, float(max_class_weight)).astype(np.float32)


def balanced_softmax_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_counts: torch.Tensor,
) -> torch.Tensor:
    """Balanced Softmax used by ``trainer.continual_cl.training_loss``."""

    adjusted_logits = logits + class_counts.to(logits.device).clamp_min(1.0).log()
    return F.cross_entropy(adjusted_logits, labels)


def weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(logits, labels, weight=class_weights.to(logits.device))


def calibrate_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Select a threshold on validation Macro-F1 only, matching the main grid."""

    best_threshold = 0.5
    best_metrics = binary_clinical_metrics(labels, probs, threshold=0.5)
    best_score = float(best_metrics["f1"])
    for threshold in THRESHOLD_GRID:
        metrics = binary_clinical_metrics(labels, probs, threshold=float(threshold))
        score = float(metrics["f1"])
        if score > best_score or (
            np.isclose(score, best_score)
            and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


def make_expert(variant: str, config: ExperimentConfig) -> nn.Module:
    if variant in {"linear_balanced_softmax", "linear_weighted_ce"}:
        return nn.Linear(config.feature_dim, 2)
    if variant == "residual_adapter_balanced_softmax":
        return ResidualFeatureExpert(
            feature_dim=config.feature_dim,
            adapter_dim=config.adapter_dim,
            dropout=config.adapter_dropout,
            scale=config.adapter_scale,
        )
    raise ValueError(f"Unknown expert variant: {variant}")


@torch.no_grad()
def collect_probs(
    expert: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    expert.eval()
    chunks: List[torch.Tensor] = []
    for start in range(0, features.shape[0], batch_size * 4):
        x = torch.from_numpy(features[start : start + batch_size * 4]).to(device)
        logits = expert(x).detach().cpu()
        chunks.append(torch.softmax(logits, dim=-1)[:, 1])
    return torch.cat(chunks, dim=0).numpy()


def train_one_domain(
    variant: str,
    seed: int,
    domain: str,
    domain_index: int,
    data: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    config: ExperimentConfig,
    device: torch.device,
    log: RunLog,
    checkpoint_path: Path | None,
) -> Dict[str, Any]:
    train_features, train_labels = data["train"]
    val_features, val_labels = data["val"]
    test_features, test_labels = data["test"]
    counts_np, weights_np = class_counts_and_weights(
        train_labels,
        max_class_weight=config.max_class_weight,
    )

    expert = make_expert(variant, config).to(device)
    parameter_count = sum(param.numel() for param in expert.parameters() if param.requires_grad)
    optimizer = torch.optim.AdamW(
        expert.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.01,
    )
    x_train = torch.from_numpy(train_features).to(device)
    y_train = torch.from_numpy(train_labels).long().to(device)
    class_counts = torch.from_numpy(counts_np).to(device)
    class_weights = torch.from_numpy(weights_np).to(device)
    generator = torch.Generator()
    generator.manual_seed(seed + domain_index * 1009)

    log.write(
        f"domain={domain} n_train={len(train_labels)} n_val={len(val_labels)} "
        f"n_test={len(test_labels)} params={parameter_count:,}"
    )
    log.write(
        f"train_class_counts={counts_np.astype(int).tolist()} "
        f"train_class_weights={weights_np.tolist()}"
    )

    best_score = -math.inf
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    best_val_metrics_at_0_5: Dict[str, float] | None = None
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        expert.train()
        order = torch.randperm(y_train.shape[0], generator=generator)
        total_loss = 0.0
        steps = 0
        for start in range(0, y_train.shape[0], config.batch_size):
            idx = order[start : start + config.batch_size].to(device)
            logits = expert(x_train[idx])
            if variant == "linear_weighted_ce":
                loss = weighted_ce_loss(logits, y_train[idx], class_weights)
            else:
                loss = balanced_softmax_loss(logits, y_train[idx], class_counts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        scheduler.step()

        # Epoch selection follows the matched main expert protocol: validation
        # Macro-F1 at 0.5, then calibration after restoring the selected epoch.
        val_probs = collect_probs(expert, val_features, device, config.batch_size)
        val_metrics = binary_clinical_metrics(val_labels, val_probs, threshold=0.5)
        score = float(val_metrics["f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(expert.state_dict())
            best_val_metrics_at_0_5 = dict(val_metrics)
        if epoch % 5 == 0 or epoch == config.epochs:
            log.write(
                f"domain={domain} epoch={epoch:02d}/{config.epochs} "
                f"loss={total_loss / max(steps, 1):.6f} "
                f"val_macro_f1_at_0.5={score:.6f}"
            )

    if best_state is None or best_val_metrics_at_0_5 is None:
        raise RuntimeError(f"No validation checkpoint selected for {variant}/{seed}/{domain}")
    expert.load_state_dict(best_state)
    val_probs = collect_probs(expert, val_features, device, config.batch_size)
    threshold, calibrated_val_metrics = calibrate_threshold(val_labels, val_probs)

    # The held-out test split is evaluated only after epoch and threshold are
    # fixed from validation data.
    test_probs = collect_probs(expert, test_features, device, config.batch_size)
    test_metrics = binary_clinical_metrics(test_labels, test_probs, threshold=threshold)
    elapsed = time.perf_counter() - started
    log.write(
        f"domain={domain} best_epoch={best_epoch} threshold={threshold:.3f} "
        f"calibrated_val_macro_f1={calibrated_val_metrics['f1']:.6f} "
        f"test_macro_f1={test_metrics['f1']:.6f} elapsed_sec={elapsed:.1f}"
    )

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_state = {name: value.detach().cpu() for name, value in expert.state_dict().items()}
        torch.save(
            {
                "variant": variant,
                "seed": seed,
                "domain": domain,
                "state_dict": cpu_state,
                "best_epoch": best_epoch,
                "threshold": threshold,
                "parameter_count": parameter_count,
                "train_class_counts": counts_np.tolist(),
                "train_class_weights": weights_np.tolist(),
                "validation_metrics": calibrated_val_metrics,
            },
            checkpoint_path,
        )

    record: Dict[str, Any] = {
        "variant": variant,
        "selection": "source_aware_oracle",
        "seed": seed,
        "domain": domain,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "val_macro_f1_at_0_5": float(best_val_metrics_at_0_5["f1"]),
        "val_macro_f1": float(calibrated_val_metrics["f1"]),
        "train_count_0": int(counts_np[0]),
        "train_count_1": int(counts_np[1]),
        "class_weight_0": float(weights_np[0]) if variant == "linear_weighted_ce" else None,
        "class_weight_1": float(weights_np[1]) if variant == "linear_weighted_ce" else None,
        "source": "trained_by_expert_loss_ablation",
    }
    for metric in METRIC_NAMES:
        record[metric] = float(test_metrics[metric])
    return record


def train_variant(
    variant: str,
    seed: int,
    seed_cache: Mapping[str, Mapping[str, Tuple[np.ndarray, np.ndarray]]],
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
    save_checkpoints: bool,
) -> List[Dict[str, Any]]:
    log = RunLog(output_dir / "train_logs" / f"{variant}_seed{seed}.log")
    try:
        set_global_seed(seed)
        log.write(
            f"variant={variant} seed={seed} device={device} optimizer=AdamW "
            f"epochs={config.epochs} lr={config.learning_rate} batch={config.batch_size}"
        )
        records = []
        for domain_index, domain in enumerate(DOMAINS):
            checkpoint = None
            if save_checkpoints:
                checkpoint = (
                    output_dir
                    / "checkpoints"
                    / variant
                    / f"seed{seed}"
                    / f"{domain}.pt"
                )
            records.append(
                train_one_domain(
                    variant=variant,
                    seed=seed,
                    domain=domain,
                    domain_index=domain_index,
                    data=seed_cache[domain],
                    config=config,
                    device=device,
                    log=log,
                    checkpoint_path=checkpoint,
                )
            )
        raw_path = output_dir / "raw_runs" / f"{variant}_seed{seed}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return records
    finally:
        log.close()


def find_main_summary(main_results_root: Path, seed: int) -> Path:
    pattern = f"feature_router_primary_seed{seed}_mem1_*/metrics_summary.json"
    candidates = list(main_results_root.glob(pattern))
    if not candidates:
        candidates = list(main_results_root.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No matched primary main result for seed {seed} below {main_results_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _require_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0.0):
        raise RuntimeError(
            f"Main-result preflight failed: {label}={actual!r}, expected "
            f"{expected!r} +/- {tolerance}. Stop before training controls and "
            "check split/cache/seed/threshold/class mapping."
        )


def load_main_records(
    main_results_root: Path,
    cache_by_seed: Mapping[int, Mapping[str, Mapping[str, Tuple[np.ndarray, np.ndarray]]]],
    config: ExperimentConfig,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for seed in config.seeds:
        summary_path = find_main_summary(main_results_root, seed)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        experiment_config = summary.get("experiment_config", {})
        train_config = experiment_config.get("train_kwargs", {})
        required = {
            "seed": seed,
            "val_ratio": config.val_ratio,
            "feature_head_method": "linear",
            "loss_mode": "balanced_softmax",
        }
        for key, expected in required.items():
            actual = experiment_config.get(key)
            if actual != expected:
                raise RuntimeError(
                    f"Main artifact {summary_path} has {key}={actual!r}, "
                    f"expected {expected!r}"
                )
        for key, expected in {
            "epochs": config.epochs,
            "lr": config.learning_rate,
            "batch_size": config.batch_size,
            "weight_decay": config.weight_decay,
        }.items():
            actual = train_config.get(key)
            if not np.isclose(float(actual), float(expected), atol=1e-12, rtol=0.0):
                raise RuntimeError(
                    f"Main artifact {summary_path} has train_kwargs.{key}={actual!r}, "
                    f"expected {expected!r}"
                )
        if tuple(experiment_config.get("domain_order", [])) != DOMAINS:
            raise RuntimeError(f"Main artifact domain order mismatch: {summary_path}")

        metrics_path = summary_path.parent / "final_metrics.csv"
        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            metric_rows = {
                row["domain"]: row
                for row in csv.DictReader(handle)
                if row.get("setting") == "oracle"
            }
        head_metadata = summary.get("head_metadata", {})
        if set(metric_rows) != set(DOMAINS) or set(head_metadata) != set(DOMAINS):
            raise RuntimeError(f"Incomplete oracle/head metadata in {summary_path.parent}")

        for domain in DOMAINS:
            train_labels = cache_by_seed[seed][domain]["train"][1]
            counts, _ = class_counts_and_weights(train_labels, config.max_class_weight)
            metadata = head_metadata[domain]
            metric_row = metric_rows[domain]
            record: Dict[str, Any] = {
                "variant": "linear_balanced_softmax",
                "selection": "source_aware_oracle",
                "seed": seed,
                "domain": domain,
                "parameter_count": config.feature_dim * 2 + 2,
                "best_epoch": int(metadata["best_epoch"]),
                "threshold": float(metadata["threshold"]),
                "val_macro_f1_at_0_5": float(metadata["best_val_metrics_at_0_5"]["f1"]),
                "val_macro_f1": float(metadata["calibrated_val_metrics"]["f1"]),
                "train_count_0": int(counts[0]),
                "train_count_1": int(counts[1]),
                "class_weight_0": None,
                "class_weight_1": None,
                "source": str(summary_path.parent.resolve()),
            }
            for metric in METRIC_NAMES:
                record[metric] = float(metric_row[metric])
            records.append(record)
    return records


def per_seed_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    variants = list(VARIANTS) + (["validation_selected_bank"] if any(
        row["variant"] == "validation_selected_bank" for row in records
    ) else [])
    seeds = sorted({int(row["seed"]) for row in records})
    for variant in variants:
        for seed in seeds:
            subset = [
                row for row in records
                if row["variant"] == variant and int(row["seed"]) == seed
            ]
            if not subset:
                continue
            by_domain = {str(row["domain"]): float(row["f1"]) for row in subset}
            rows.append(
                {
                    "variant": variant,
                    "selection": "source_aware_oracle",
                    "seed": seed,
                    "macro_f1": float(np.mean(list(by_domain.values()))),
                    **{f"{domain}_macro_f1": by_domain[domain] for domain in DOMAINS},
                }
            )
    return rows


def select_validation_bank(
    candidate_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Choose only by calibrated validation Macro-F1; never by test metrics."""

    selected: List[Dict[str, Any]] = []
    seeds = sorted({int(row["seed"]) for row in candidate_records})
    variant_priority = {variant: -index for index, variant in enumerate(VARIANTS)}
    for seed in seeds:
        for domain in DOMAINS:
            candidates = [
                row for row in candidate_records
                if int(row["seed"]) == seed and row["domain"] == domain
            ]
            if {row["variant"] for row in candidates} != set(VARIANTS):
                raise RuntimeError(f"Incomplete validation candidates for seed={seed}, {domain}")
            ranked = sorted(
                candidates,
                key=lambda row: (
                    float(row["val_macro_f1"]),
                    variant_priority[str(row["variant"])],
                ),
                reverse=True,
            )
            winner = ranked[0]
            margin = float(winner["val_macro_f1"]) - float(ranked[1]["val_macro_f1"])
            result = dict(winner)
            result["variant"] = "validation_selected_bank"
            result["selected_variant"] = winner["variant"]
            result["selection_margin"] = margin
            result["candidate_validation_macro_f1"] = json.dumps(
                {row["variant"]: float(row["val_macro_f1"]) for row in ranked},
                sort_keys=True,
            )
            selected.append(result)
    return selected


def sample_std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0


def build_summary(
    seed_rows: Sequence[Mapping[str, Any]],
    selected_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    main_values = [
        float(row["macro_f1"])
        for row in seed_rows
        if row["variant"] == "linear_balanced_softmax"
    ]
    main_mean = float(np.mean(main_values))
    selected_variation = 0
    for domain in DOMAINS:
        choices = {
            row["selected_variant"] for row in selected_records if row["domain"] == domain
        }
        selected_variation += max(0, len(choices) - 1)
    selection_margins = [float(row["selection_margin"]) for row in selected_records]

    rows = []
    for variant in (*VARIANTS, "validation_selected_bank"):
        variant_rows = [row for row in seed_rows if row["variant"] == variant]
        seed_values = [float(row["macro_f1"]) for row in variant_rows]
        mean_value = float(np.mean(seed_values))
        delta = mean_value - main_mean
        if variant == "linear_balanced_softmax":
            comment = "main expert bank"
        elif variant == "linear_weighted_ce":
            comment = "no stable gain" if delta <= 0.001 else "small gain"
        elif variant == "residual_adapter_balanced_softmax":
            comment = "higher capacity; no stable gain" if delta <= 0.001 else "higher capacity; small gain"
        else:
            sensitivity = (
                f"selection changed across seeds ({selected_variation} extra choices; "
                f"median val margin {np.median(selection_margins):.4f})"
            )
            prefix = "small gain but validation-sensitive" if delta > 0 else "no gain; validation-sensitive"
            comment = f"{prefix}; {sensitivity}"
        rows.append(
            {
                "variant": variant,
                "selection": "source_aware_oracle",
                "macro_f1_mean": mean_value,
                "macro_f1_std": sample_std(seed_values),
                "delta_vs_main": delta,
                **{
                    f"{domain}_macro_f1": float(
                        np.mean([float(row[f"{domain}_macro_f1"]) for row in variant_rows])
                    )
                    for domain in DOMAINS
                },
                "comment": comment,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Expert / loss variant | Macro-F1 | Δ vs. main | CPSC | PTB-XL | Georgia | Chapman | Comment |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        delta = "--" if row["variant"] == "linear_balanced_softmax" else f"{row['delta_vs_main']:+.4f}"
        lines.append(
            "| {label} | {mean:.4f} ± {std:.4f} | {delta} | {cpsc:.4f} | "
            "{ptbxl:.4f} | {georgia:.4f} | {chapman:.4f} | {comment} |".format(
                label=VARIANT_LABELS[str(row["variant"])],
                mean=float(row["macro_f1_mean"]),
                std=float(row["macro_f1_std"]),
                delta=delta,
                cpsc=float(row["cpsc_macro_f1"]),
                ptbxl=float(row["ptbxl_macro_f1"]),
                georgia=float(row["georgia_macro_f1"]),
                chapman=float(row["chapman_macro_f1"]),
                comment=row["comment"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(seeds=tuple(args.seeds))
    if config.seeds != (42, 43, 44):
        print(f"[warning] non-paper seed set requested: {config.seeds}", flush=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    torch.set_num_threads(max(1, args.cpu_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    manifest = cache_manifest(args.feature_cache_dir, config.seeds)
    run_config = {
        "experiment": "expert_loss_ablation_appendix",
        "created_at": datetime.now().astimezone().isoformat(),
        "repo_root": str(REPO_ROOT),
        "feature_cache_dir": str(args.feature_cache_dir.resolve()),
        "main_results_root": str(args.main_results_root.resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "cpu_threads": args.cpu_threads,
        "domains": list(DOMAINS),
        "config": asdict(config),
        "optimizer": "AdamW",
        "epoch_selection": "validation_macro_f1_at_threshold_0.5",
        "threshold_calibration": "validation_only_after_best_epoch_restore",
        "threshold_grid": {"start": 0.05, "stop": 0.95, "steps": 181},
        "test_usage": "held_out_reporting_only_after_model_and_threshold_are_fixed",
        "validation_bank_selection": "maximum calibrated validation Macro-F1 per seed/domain",
        "validation_selection_candidates": list(VARIANTS),
        "checkpoint_policy": (
            "save_new_controls; reused_main_artifacts_do_not_contain_head_checkpoints"
            if not args.no_checkpoints
            else "disabled"
        ),
        "feature_cache_manifest": manifest,
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print(f"[preflight] device={device} cache={args.feature_cache_dir}", flush=True)
    cache_by_seed = {
        seed: load_seed_cache(args.feature_cache_dir, seed, config.feature_dim)
        for seed in config.seeds
    }

    if args.force_train_main:
        main_records: List[Dict[str, Any]] = []
        for seed in config.seeds:
            main_records.extend(
                train_variant(
                    "linear_balanced_softmax",
                    seed,
                    cache_by_seed[seed],
                    config,
                    device,
                    output_dir,
                    save_checkpoints=not args.no_checkpoints,
                )
            )
    else:
        print("[preflight] reusing matched primary oracle artifacts for the main bank", flush=True)
        main_records = load_main_records(
            args.main_results_root,
            cache_by_seed,
            config,
        )

    main_seed_rows = per_seed_rows(main_records)
    main_values = [float(row["macro_f1"]) for row in main_seed_rows]
    main_mean = float(np.mean(main_values))
    main_std = sample_std(main_values)
    _require_close(main_mean, EXPECTED_MAIN_MEAN, args.main_mean_tolerance, "mean")
    _require_close(main_std, EXPECTED_MAIN_STD, args.main_std_tolerance, "sample std")
    print(
        f"[preflight] main source-aware oracle verified: {main_mean:.6f} +/- {main_std:.6f}",
        flush=True,
    )

    candidate_records = list(main_records)
    for variant in ("linear_weighted_ce", "residual_adapter_balanced_softmax"):
        for seed in config.seeds:
            candidate_records.extend(
                train_variant(
                    variant,
                    seed,
                    cache_by_seed[seed],
                    config,
                    device,
                    output_dir,
                    save_checkpoints=not args.no_checkpoints,
                )
            )

    selected_records = select_validation_bank(candidate_records)
    all_records = candidate_records + selected_records
    seed_rows = per_seed_rows(all_records)
    summary_rows = build_summary(seed_rows, selected_records)

    per_domain_fields = [
        "variant", "selection", "seed", "domain", "parameter_count",
        "best_epoch", "threshold", "val_macro_f1_at_0_5", "val_macro_f1",
        "train_count_0", "train_count_1", "class_weight_0", "class_weight_1",
        *[name for name in METRIC_NAMES if name != "threshold"], "source",
    ]
    write_csv(output_dir / "per_domain_results.csv", candidate_records, per_domain_fields)
    write_csv(
        output_dir / "per_seed_results.csv",
        seed_rows,
        [
            "variant", "selection", "seed", "macro_f1",
            *[f"{domain}_macro_f1" for domain in DOMAINS],
        ],
    )
    write_csv(
        output_dir / "selected_bank.csv",
        selected_records,
        [
            "seed", "domain", "selected_variant", "val_macro_f1",
            "selection_margin", "candidate_validation_macro_f1", "f1",
            "threshold", "best_epoch", "parameter_count",
            *[name for name in METRIC_NAMES if name not in {"f1", "threshold"}],
            "source",
        ],
    )
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "variant", "selection", "macro_f1_mean", "macro_f1_std",
            "delta_vs_main", *[f"{domain}_macro_f1" for domain in DOMAINS],
            "comment",
        ],
    )
    table = markdown_table(summary_rows)
    (output_dir / "appendix_table.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table, flush=True)
    print(f"\n[saved] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
