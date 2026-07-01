"""Shared-backbone baselines for the clean ECG continual-learning protocol.

The adapter method uses domain-specific experts. These baselines intentionally
use one shared ECGFounder/Net1D classifier across all domains so that the paper
can report how much the source-aware adapter isolation helps under the same
clean split, validation selection, threshold calibration, and clinical metrics.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate.clinical_metrics import binary_clinical_metrics, format_metric_line
from models.backbone.net1d import Net1D
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
    compute_forgetting,
    get_cinc_train_val_test_loaders,
    resolve_checkpoint_path,
    save_results,
    set_global_seed,
    training_loss,
)
from scripts.run_config import env_bool, env_float, env_int, env_str


BASELINE_METHOD = env_str("ECG_BASELINE_METHOD", "linear_probe").lower()
EWC_LAMBDA = env_float("ECG_EWC_LAMBDA", 10.0)
EWC_MAX_BATCHES = env_int("ECG_EWC_MAX_BATCHES", 50)
LWF_LAMBDA = env_float("ECG_LWF_LAMBDA", 1.0)
LWF_TEMPERATURE = env_float("ECG_LWF_TEMPERATURE", 2.0)
REPLAY_PER_DOMAIN = env_int("ECG_REPLAY_PER_DOMAIN", 256)
SI_LAMBDA = env_float("ECG_SI_LAMBDA", 1.0)
SI_XI = env_float("ECG_SI_XI", 1e-3)
VALIDATE_EVERY = max(1, env_int("ECG_VALIDATE_EVERY", env_int("ECG_VAL_EVERY", 1)))
SAVE_PARTIAL = env_bool("ECG_SAVE_PARTIAL", True)
DOMAIN_LINEAR_HEAD_METHODS = {"domain_linear_heads", "domain_aware_linear_heads"}
DOMAIN_LAYERNORM_HEAD_METHODS = {
    "domain_layernorm_heads",
    "domain_ln_heads",
    "domain_aware_layernorm_heads",
}
DOMAIN_MLP_HEAD_METHODS = {"domain_mlp_heads", "domain_aware_mlp_heads"}
DOMAIN_RESIDUAL_HEAD_METHODS = {
    "domain_residual_heads",
    "domain_feature_adapter_heads",
    "domain_aware_residual_heads",
}
DOMAIN_HEAD_METHODS = (
    DOMAIN_LINEAR_HEAD_METHODS
    | DOMAIN_LAYERNORM_HEAD_METHODS
    | DOMAIN_MLP_HEAD_METHODS
    | DOMAIN_RESIDUAL_HEAD_METHODS
)
DOMAIN_MLP_HIDDEN_DIM = env_int("ECG_DOMAIN_MLP_HIDDEN_DIM", 256)
DOMAIN_MLP_DROPOUT = env_float("ECG_DOMAIN_MLP_DROPOUT", 0.1)
DOMAIN_FEATURE_ADAPTER_DIM = env_int("ECG_DOMAIN_FEATURE_ADAPTER_DIM", 128)
DOMAIN_FEATURE_ADAPTER_DROPOUT = env_float("ECG_DOMAIN_FEATURE_ADAPTER_DROPOUT", 0.1)
DOMAIN_FEATURE_ADAPTER_SCALE = env_float("ECG_DOMAIN_FEATURE_ADAPTER_SCALE", 1.0)
SELECT_CALIBRATED_EPOCH = env_bool(
    "ECG_SELECT_CALIBRATED_EPOCH",
    env_bool("ECG_SELECT_EPOCH_AFTER_CALIBRATION", False),
)
GDUMB_BUDGET = env_int("ECG_GDUMB_BUDGET", REPLAY_PER_DOMAIN * len(DOMAIN_NAMES))
BASELINE_METHODS = {
    "linear_probe",
    "full_finetune",
    "ewc",
    "lwf",
    "small_replay",
    "icarl_ncm",
    "gdumb",
    "si",
} | DOMAIN_HEAD_METHODS
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


def _unwrap_logits(output: torch.Tensor | Tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def build_shared_model(return_features: bool = False) -> Net1D:
    """Build ECGFounder/Net1D with the same backbone config as the adapter run."""
    model_kwargs = dict(BACKBONE_KWARGS)
    model_kwargs["return_features"] = return_features
    model = Net1D(**model_kwargs)
    checkpoint = resolve_checkpoint_path()

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw_sd = ckpt.get("state_dict", ckpt)
        model_sd = model.state_dict()
        filtered = {}
        for key, value in raw_sd.items():
            if key in model_sd and model_sd[key].shape == value.shape:
                filtered[key] = value
                continue
            if key.startswith("module."):
                stripped = key[len("module.") :]
                if stripped in model_sd and model_sd[stripped].shape == value.shape:
                    filtered[stripped] = value
        model.load_state_dict(filtered, strict=False)
        print(f"  [Weights] loaded {len(filtered)}/{len(model_sd)} layers from {checkpoint}")
        if len(filtered) == 0:
            raise RuntimeError(f"Checkpoint is readable but no parameters matched: {checkpoint}")
    else:
        print("  [Weights] no pretrained checkpoint found; using random init.")

    return model.to(DEVICE)


class ResidualFeatureAdapterHead(nn.Module):
    """Small per-domain residual adapter on frozen ECGFounder final features."""

    def __init__(
        self,
        adapter: nn.Module,
        classifier: nn.Module,
        adapter_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.classifier = classifier
        self.adapter_scale = adapter_scale

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        adapted = features + self.adapter_scale * self.adapter(features)
        return self.classifier(adapted)


class DomainLinearHeadsModel(nn.Module):
    """Frozen ECGFounder final features with one classifier expert per domain."""

    def __init__(
        self,
        backbone: Net1D,
        domain_names: Iterable[str],
        feature_dim: int,
        num_classes: int = 2,
        head_kind: str = "linear",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        adapter_dim: int = 128,
        adapter_dropout: float = 0.1,
        adapter_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.return_features = True
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.domain_names = list(domain_names)
        if not self.domain_names:
            raise ValueError("DomainLinearHeadsModel requires at least one domain.")
        self.head_kind = head_kind.lower()
        self.heads = nn.ModuleDict({
            domain: self._make_head(
                feature_dim=feature_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
                adapter_dim=adapter_dim,
                adapter_dropout=adapter_dropout,
                adapter_scale=adapter_scale,
            )
            for domain in self.domain_names
        })
        self.current_domain = self.domain_names[0]
        self.set_domain(self.current_domain)

    def _make_head(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        adapter_dim: int,
        adapter_dropout: float,
        adapter_scale: float,
    ) -> nn.Module:
        if self.head_kind == "linear":
            return nn.Linear(feature_dim, num_classes)
        if self.head_kind == "layernorm_linear":
            return nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, num_classes),
            )
        if self.head_kind == "mlp":
            return nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        if self.head_kind == "residual_adapter":
            adapter = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, adapter_dim),
                nn.GELU(),
                nn.Dropout(adapter_dropout),
                nn.Linear(adapter_dim, feature_dim),
            )
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
            return ResidualFeatureAdapterHead(
                adapter=adapter,
                classifier=nn.Linear(feature_dim, num_classes),
                adapter_scale=adapter_scale,
            )
        raise ValueError(f"Unknown domain head kind: {self.head_kind}")

    def train(self, mode: bool = True) -> "DomainLinearHeadsModel":
        super().train(mode)
        self.backbone.eval()
        return self

    def set_domain(self, domain: str) -> None:
        if domain not in self.heads:
            raise ValueError(f"Unknown domain head: {domain}")
        self.current_domain = domain
        for head_domain, head in self.heads.items():
            requires_grad = head_domain == domain
            for param in head.parameters():
                param.requires_grad = requires_grad

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = [param for param in self.heads[self.current_domain].parameters() if param.requires_grad]
        if not params:
            raise RuntimeError(f"No trainable parameters for head {self.current_domain}.")
        return params

    def forward(self, x: torch.Tensor, domain: Optional[str] = None) -> torch.Tensor:
        active_domain = domain or self.current_domain
        if active_domain not in self.heads:
            raise ValueError(f"Unknown domain head: {active_domain}")
        with torch.no_grad():
            output = self.backbone(x)
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError("DomainLinearHeadsModel requires backbone return_features=True.")
            features = output[1]
        return self.heads[active_domain](features)


def configure_trainable_params(model: torch.nn.Module, method: str) -> List[torch.nn.Parameter]:
    """Select trainable parameters for a baseline method."""
    if method in {"full_finetune", "ewc", "lwf", "small_replay", "icarl_ncm", "gdumb", "si"}:
        for param in model.parameters():
            param.requires_grad = True
    elif method == "linear_probe":
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            first = name.split(".", 1)[0].lower()
            if first in {"dense", "fc", "classifier", "head"}:
                param.requires_grad = True
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError(f"No trainable parameters selected for {method}.")
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in params)
    print(f"  [Params] method={method} trainable={trainable:,}/{total:,}")
    return params


def snapshot_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def restore_model_state(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(copy.deepcopy(state), strict=True)


def named_trainable_parameters(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _move_tensor_dict_to_device(
    tensors: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device, non_blocking=device.type == "cuda")
        for name, tensor in tensors.items()
    }


def move_ewc_term_to_device(
    term: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device = DEVICE,
) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        "means": _move_tensor_dict_to_device(term["means"], device),
        "fishers": _move_tensor_dict_to_device(term["fishers"], device),
    }


def _model_train_mode(model: torch.nn.Module, method: str) -> None:
    if method == "linear_probe":
        model.eval()
    else:
        model.train()


def ewc_penalty(
    model: torch.nn.Module,
    ewc_terms: List[Dict[str, Dict[str, torch.Tensor]]],
) -> torch.Tensor:
    if not ewc_terms:
        return torch.zeros((), device=DEVICE)
    penalty = torch.zeros((), device=DEVICE)
    params = dict(named_trainable_parameters(model))
    for term in ewc_terms:
        means = term["means"]
        fishers = term["fishers"]
        for name, param in params.items():
            if name not in means:
                continue
            mean = means[name]
            fisher = fishers[name]
            if mean.device != param.device:
                mean = mean.to(param.device, non_blocking=param.device.type == "cuda")
                means[name] = mean
            if fisher.device != param.device:
                fisher = fisher.to(param.device, non_blocking=param.device.type == "cuda")
                fishers[name] = fisher
            penalty = penalty + (fisher * (param - mean).pow(2)).sum()
    return penalty


def lwf_distillation_loss(
    logits: torch.Tensor,
    old_logits: torch.Tensor,
    temperature: float = LWF_TEMPERATURE,
) -> torch.Tensor:
    """KL distillation term used by Learning without Forgetting."""
    t = max(float(temperature), 1e-6)
    log_probs = torch.log_softmax(logits / t, dim=-1)
    old_probs = torch.softmax(old_logits / t, dim=-1)
    return torch.nn.functional.kl_div(
        log_probs,
        old_probs,
        reduction="batchmean",
    ) * (t * t)


def _tensor_dataset_xy(dataset: torch.utils.data.Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(dataset, TensorDataset) or len(dataset.tensors) < 2:
        raise TypeError("Replay baseline expects TensorDataset train splits.")
    x, y = dataset.tensors[:2]
    return x, y


def _balanced_memory_indices(
    y: torch.Tensor,
    max_samples: int,
    seed: int,
) -> torch.Tensor:
    if max_samples <= 0 or y.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(seed)
    classes = [cls for cls in torch.unique(y).tolist()]
    per_class = max(1, max_samples // max(len(classes), 1))
    selected: List[torch.Tensor] = []
    for cls in classes:
        cls_idx = torch.nonzero(y == int(cls), as_tuple=False).flatten()
        if cls_idx.numel() == 0:
            continue
        order = torch.randperm(cls_idx.numel(), generator=generator)
        selected.append(cls_idx[order[: min(per_class, cls_idx.numel())]])
    if not selected:
        return torch.empty(0, dtype=torch.long)
    idx = torch.cat(selected)
    if idx.numel() < max_samples:
        remaining = torch.ones(y.numel(), dtype=torch.bool)
        remaining[idx] = False
        rest = torch.nonzero(remaining, as_tuple=False).flatten()
        if rest.numel() > 0:
            order = torch.randperm(rest.numel(), generator=generator)
            idx = torch.cat([idx, rest[order[: max_samples - idx.numel()]]])
    return idx[:max_samples]


def _make_replay_loader(
    current_loader: DataLoader,
    replay_memory: List[Tuple[torch.Tensor, torch.Tensor]],
    seed: int,
    domain_index: int,
) -> Tuple[DataLoader, torch.Tensor]:
    x_current, y_current = _tensor_dataset_xy(current_loader.dataset)
    xs = [x_current]
    ys = [y_current]
    for x_mem, y_mem in replay_memory:
        xs.append(x_mem)
        ys.append(y_mem)
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    generator = torch.Generator()
    generator.manual_seed(seed + domain_index * 1009 + 97)
    loader_kwargs = {
        "dataset": TensorDataset(x_all, y_all),
        "batch_size": current_loader.batch_size,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
        "generator": generator,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(**loader_kwargs)
    return loader, y_all


def _store_replay_memory(
    train_loader: DataLoader,
    seed: int,
    domain_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = _tensor_dataset_xy(train_loader.dataset)
    idx = _balanced_memory_indices(
        y,
        max_samples=REPLAY_PER_DOMAIN,
        seed=seed + domain_index * 1009 + 7919,
    )
    return x[idx].detach().cpu().clone(), y[idx].detach().cpu().clone()


def _gdumb_update_buffer(
    buffer_x: Optional[torch.Tensor],
    buffer_y: Optional[torch.Tensor],
    new_x: torch.Tensor,
    new_y: torch.Tensor,
    budget: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy class-balanced memory update (GDumb, Prabhu et al. 2020)."""
    if buffer_x is None:
        x_all, y_all = new_x, new_y
    else:
        x_all = torch.cat([buffer_x, new_x], dim=0)
        y_all = torch.cat([buffer_y, new_y], dim=0)
    idx = _balanced_memory_indices(y_all, max_samples=budget, seed=seed)
    return x_all[idx].detach().cpu().clone(), y_all[idx].detach().cpu().clone()


@torch.no_grad()
def extract_features_shared(
    model: torch.nn.Module,
    x: torch.Tensor,
    batch_size: int = 256,
) -> torch.Tensor:
    """Run the backbone in feature-returning mode over a raw tensor in batches."""
    model.eval()
    feats: List[torch.Tensor] = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(DEVICE)
        output = model(xb)
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError(
                "icarl_ncm requires a model built with return_features=True."
            )
        feats.append(output[1].detach().cpu())
    return torch.cat(feats, dim=0)


def compute_ncm_class_means(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """iCaRL-style nearest-class-mean prototypes computed from exemplar memory."""
    features = extract_features_shared(model, x)
    means = []
    for cls in range(num_classes):
        mask = y == cls
        if mask.sum() == 0:
            means.append(torch.zeros(features.shape[1]))
        else:
            means.append(features[mask].mean(dim=0))
    return torch.stack(means, dim=0)


@torch.no_grad()
def collect_labels_probs_ncm(
    model: torch.nn.Module,
    loader: DataLoader,
    class_means: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify via negative squared distance to the nearest class mean."""
    model.eval()
    means = class_means.to(DEVICE)
    all_labels, all_probs = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        output = model(x)
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError(
                "icarl_ncm requires a model built with return_features=True."
            )
        features = output[1]
        dists = torch.stack(
            [((features - means[cls]) ** 2).sum(dim=1) for cls in range(means.shape[0])],
            dim=1,
        )
        probs = torch.softmax(-dists, dim=-1)[:, 1]
        all_labels.append(y)
        all_probs.append(probs.detach().cpu())
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    return labels, probs


def train_one_epoch_shared(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    method: str,
    class_weights: Optional[torch.Tensor] = None,
    class_counts: Optional[torch.Tensor] = None,
    ewc_terms: Optional[List[Dict[str, Dict[str, torch.Tensor]]]] = None,
    old_model: Optional[torch.nn.Module] = None,
    domain: Optional[str] = None,
    si_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> float:
    _model_train_mode(model, method)
    total_loss = 0.0
    for batch_idx, (x, y) in enumerate(loader):
        if DRY_RUN and batch_idx >= TRAIN_KWARGS["dry_run_batches"]:
            break
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        prev_params = None
        if method == "si" and si_state is not None:
            prev_params = {
                name: param.detach().clone()
                for name, param in named_trainable_parameters(model)
            }

        optimizer.zero_grad(set_to_none=True)
        logits = _unwrap_logits(model(x, domain=domain) if domain is not None else model(x))
        loss = training_loss(
            logits,
            y,
            class_weights=class_weights,
            class_counts=class_counts,
        )
        if method == "ewc" and ewc_terms:
            loss = loss + 0.5 * EWC_LAMBDA * ewc_penalty(model, ewc_terms)
        if method == "si" and ewc_terms:
            # SI's penalty has the same quadratic form as EWC; ewc_terms here
            # carries the online-accumulated importance (omega) instead of Fisher.
            loss = loss + 0.5 * SI_LAMBDA * ewc_penalty(model, ewc_terms)
        if method == "lwf" and old_model is not None:
            with torch.no_grad():
                old_logits = _unwrap_logits(old_model(x))
            loss = loss + LWF_LAMBDA * lwf_distillation_loss(logits, old_logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            TRAIN_KWARGS["grad_clip"],
        )

        step_grads = None
        if method == "si" and si_state is not None:
            step_grads = {
                name: (
                    param.grad.detach().clone()
                    if param.grad is not None
                    else torch.zeros_like(param)
                )
                for name, param in named_trainable_parameters(model)
            }

        optimizer.step()
        total_loss += float(loss.item())

        if method == "si" and si_state is not None and prev_params is not None:
            small_omega = si_state["small_omega"]
            for name, param in named_trainable_parameters(model):
                delta = param.detach() - prev_params[name]
                small_omega[name] = small_omega.get(
                    name, torch.zeros_like(param)
                ) + (-step_grads[name] * delta)

    denom = min(len(loader), TRAIN_KWARGS["dry_run_batches"]) if DRY_RUN else len(loader)
    return total_loss / max(denom, 1)


def finalize_si_importance(
    model: torch.nn.Module,
    si_running_state: Dict[str, Dict[str, torch.Tensor]],
    si_terms: List[Dict[str, Dict[str, torch.Tensor]]],
    small_omega_override: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Finalize SI importance at the model state carried into the next task."""
    theta_before = si_running_state["theta_before_task"]
    small_omega = (
        small_omega_override
        if small_omega_override is not None
        else si_running_state["small_omega"]
    )
    omega_update: Dict[str, torch.Tensor] = {}
    theta_after: Dict[str, torch.Tensor] = {}
    for name, param in named_trainable_parameters(model):
        delta_total = param.detach() - theta_before[name]
        omega_update[name] = (
            small_omega[name] / (delta_total.pow(2) + SI_XI)
        ).clamp(min=0.0)
        theta_after[name] = param.detach().clone()

    if si_terms:
        prev_omega = si_terms[0]["fishers"]
        cumulative_omega = {
            name: prev_omega.get(name, torch.zeros_like(value)).to(value.device)
            + value
            for name, value in omega_update.items()
        }
    else:
        cumulative_omega = omega_update

    si_terms.clear()
    si_terms.append(
        {
            "means": {
                name: tensor.detach().cpu().clone()
                for name, tensor in theta_after.items()
            },
            "fishers": {
                name: tensor.detach().cpu().clone()
                for name, tensor in cumulative_omega.items()
            },
        }
    )


@torch.no_grad()
def collect_labels_probs_shared(
    model: torch.nn.Module,
    loader: DataLoader,
    domain: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = _unwrap_logits(model(x, domain=domain) if domain is not None else model(x))
        all_logits.append(logits.detach().cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
    return labels, probs


def evaluate_shared(
    model: torch.nn.Module,
    loader: DataLoader,
    domain: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    labels, probs = collect_labels_probs_shared(model, loader, domain=domain)
    return binary_clinical_metrics(labels=labels, probs=probs, threshold=threshold)


def compute_ewc_term(
    model: torch.nn.Module,
    loader: DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    class_counts: Optional[torch.Tensor] = None,
    max_batches: int = EWC_MAX_BATCHES,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Estimate diagonal Fisher on the current train subset."""
    model.eval()
    params = named_trainable_parameters(model)
    fisher = {name: torch.zeros_like(param.detach(), device="cpu") for name, param in params}
    batches = 0

    for x, y in loader:
        if max_batches > 0 and batches >= max_batches:
            break
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        model.zero_grad(set_to_none=True)
        logits = _unwrap_logits(model(x))
        loss = training_loss(
            logits,
            y,
            class_weights=class_weights,
            class_counts=class_counts,
        )
        loss.backward()
        for name, param in params:
            if param.grad is not None:
                fisher[name] += param.grad.detach().cpu().pow(2)
        batches += 1

    denom = max(batches, 1)
    fisher = {name: value / denom for name, value in fisher.items()}
    means = {name: param.detach().cpu().clone() for name, param in params}
    model.zero_grad(set_to_none=True)
    return {"means": means, "fishers": fisher}


def _output_dir_for_method(method: str) -> Path:
    if os.environ.get("ECG_OUTPUT_DIR"):
        return Path(OUTPUT_DIR)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(OUTPUT_DIR) / f"baseline_{method}_seed{SEED}_{stamp}"


def _json_ready_shared(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready_shared(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready_shared(item) for item in value]
    return value


def _ewc_terms_to_cpu(
    ewc_terms: List[Dict[str, Dict[str, torch.Tensor]]],
) -> List[Dict[str, Dict[str, torch.Tensor]]]:
    return [
        {
            "means": {
                name: tensor.detach().cpu()
                for name, tensor in term["means"].items()
            },
            "fishers": {
                name: tensor.detach().cpu()
                for name, tensor in term["fishers"].items()
            },
        }
        for term in ewc_terms
    ]


def _save_partial_state(
    output_dir: Path,
    method: str,
    task_index: int,
    domains: List[str],
    model: torch.nn.Module,
    r_matrix: np.ndarray,
    metric_matrices: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    domain_thresholds: Dict[str, float],
    ewc_terms: List[Dict[str, Dict[str, torch.Tensor]]],
    replay_memory: List[Tuple[torch.Tensor, torch.Tensor]],
    si_terms: Optional[List[Dict[str, Dict[str, torch.Tensor]]]] = None,
    gdumb_buffer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ncm_class_means: Optional[torch.Tensor] = None,
) -> None:
    if not SAVE_PARTIAL:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    completed_domains = domains[: task_index + 1]
    checkpoint = {
        "method": method,
        "seed": SEED,
        "task_index": int(task_index),
        "completed_domains": completed_domains,
        "domain_thresholds": dict(domain_thresholds),
        "model_state": snapshot_model_state(model),
        "r_matrix": r_matrix,
        "metric_matrices": metric_matrices,
        "metadata": metadata,
        "train_kwargs": TRAIN_KWARGS,
    }
    if method == "ewc":
        checkpoint["ewc_terms"] = _ewc_terms_to_cpu(ewc_terms)
    if method in {"small_replay", "icarl_ncm"}:
        checkpoint["replay_memory"] = [
            (x.detach().cpu(), y.detach().cpu())
            for x, y in replay_memory
        ]
    if method == "icarl_ncm" and ncm_class_means is not None:
        checkpoint["ncm_class_means"] = ncm_class_means.detach().cpu()
    if method == "gdumb" and gdumb_buffer is not None and gdumb_buffer[0] is not None:
        checkpoint["gdumb_buffer"] = (
            gdumb_buffer[0].detach().cpu(),
            gdumb_buffer[1].detach().cpu(),
        )
    if method == "si" and si_terms:
        checkpoint["si_terms"] = _ewc_terms_to_cpu(si_terms)

    checkpoint_path = output_dir / "partial_checkpoint_latest.pt"
    torch.save(checkpoint, checkpoint_path)
    status_path = output_dir / f"partial_after_task_{task_index + 1}_{domains[task_index]}.json"
    status = {
        "method": method,
        "seed": SEED,
        "task_index": int(task_index),
        "completed_domains": completed_domains,
        "checkpoint": str(checkpoint_path),
        "r_matrix": r_matrix,
        "final_completed_f1": {
            domain: float(r_matrix[task_index][idx])
            for idx, domain in enumerate(completed_domains)
            if not np.isnan(r_matrix[task_index][idx])
        },
        "best_epoch_by_domain": metadata.get("best_epoch_by_domain", {}),
        "best_threshold_by_domain": metadata.get("best_threshold_by_domain", {}),
    }
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready_shared(status), f, indent=2)
    print(f"  [Partial] saved recoverable state: {checkpoint_path}")


def run_shared_baseline(
    method: str,
    return_metrics: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    if method not in BASELINE_METHODS:
        raise ValueError(f"ECG_BASELINE_METHOD must be one of {sorted(BASELINE_METHODS)}")
    domains = DOMAIN_NAMES
    output_dir = _output_dir_for_method(method)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_domain_head_method = method in DOMAIN_HEAD_METHODS
    is_domain_layernorm_heads = method in DOMAIN_LAYERNORM_HEAD_METHODS
    is_domain_mlp_heads = method in DOMAIN_MLP_HEAD_METHODS
    is_domain_residual_heads = method in DOMAIN_RESIDUAL_HEAD_METHODS
    domain_head_kind = None
    baseline_family = "shared_sequential"
    if is_domain_head_method:
        if is_domain_residual_heads:
            domain_head_kind = "residual_adapter"
            baseline_family = "domain_aware_residual_feature_adapter_probe"
        elif is_domain_layernorm_heads:
            domain_head_kind = "layernorm_linear"
            baseline_family = "domain_aware_layernorm_linear_probe"
        elif is_domain_mlp_heads:
            domain_head_kind = "mlp"
            baseline_family = "domain_aware_multi_head_mlp_probe"
        else:
            domain_head_kind = "linear"
            baseline_family = "domain_aware_multi_head_linear_probe"
        model = DomainLinearHeadsModel(
            backbone=build_shared_model(return_features=True),
            domain_names=domains,
            feature_dim=int(BACKBONE_KWARGS["filter_list"][-1]),
            num_classes=int(BACKBONE_KWARGS["n_classes"]),
            head_kind=domain_head_kind,
            hidden_dim=DOMAIN_MLP_HIDDEN_DIM,
            dropout=DOMAIN_MLP_DROPOUT,
            adapter_dim=DOMAIN_FEATURE_ADAPTER_DIM,
            adapter_dropout=DOMAIN_FEATURE_ADAPTER_DROPOUT,
            adapter_scale=DOMAIN_FEATURE_ADAPTER_SCALE,
        ).to(DEVICE)
        trainable_params: List[torch.nn.Parameter] = []
        total = sum(param.numel() for param in model.parameters())
        per_head = sum(param.numel() for param in model.heads[domains[0]].parameters())
        print(
            f"  [Params] method={method} trainable_per_domain={per_head:,}/{total:,}; "
            "backbone frozen"
        )
    else:
        model = build_shared_model(return_features=(method == "icarl_ncm"))
        trainable_params = configure_trainable_params(model, method)

    train_loaders, val_loaders, test_loaders, split_infos, train_labels = (
        get_cinc_train_val_test_loaders(
            data_dir=DATA_DIR,
            domains=domains,
            seed=SEED,
            val_ratio=VAL_RATIO,
        )
    )

    t = len(domains)
    r_matrix = np.full((t, t), np.nan)
    metric_matrices = {name: np.full((t, t), np.nan) for name in METRIC_NAMES}
    domain_thresholds: Dict[str, float] = {}
    ewc_terms: List[Dict[str, Dict[str, torch.Tensor]]] = []
    si_terms: List[Dict[str, Dict[str, torch.Tensor]]] = []
    replay_memory: List[Tuple[torch.Tensor, torch.Tensor]] = []
    gdumb_buffer_x: Optional[torch.Tensor] = None
    gdumb_buffer_y: Optional[torch.Tensor] = None
    ncm_class_means: Optional[torch.Tensor] = None
    old_model: Optional[torch.nn.Module] = None
    metadata: Dict[str, Any] = {
        "config": {
            "baseline_method": method,
            "baseline_family": baseline_family,
            "domain_head_kind": domain_head_kind,
            "domain_mlp_hidden_dim": DOMAIN_MLP_HIDDEN_DIM if is_domain_mlp_heads else None,
            "domain_mlp_dropout": DOMAIN_MLP_DROPOUT if is_domain_mlp_heads else None,
            "domain_feature_adapter_dim": (
                DOMAIN_FEATURE_ADAPTER_DIM if is_domain_residual_heads else None
            ),
            "domain_feature_adapter_dropout": (
                DOMAIN_FEATURE_ADAPTER_DROPOUT if is_domain_residual_heads else None
            ),
            "domain_feature_adapter_scale": (
                DOMAIN_FEATURE_ADAPTER_SCALE if is_domain_residual_heads else None
            ),
            "seed": SEED,
            "val_ratio": VAL_RATIO,
            "loss_mode": LOSS_MODE,
            "class_weight_mode": CLASS_WEIGHT_MODE,
            "label_smoothing": LABEL_SMOOTHING,
            "select_by": SELECT_BY,
            "epoch_selection_threshold": 0.5,
            "select_epoch_after_threshold_calibration": SELECT_CALIBRATED_EPOCH,
            "validate_every": VALIDATE_EVERY,
            "num_workers": NUM_WORKERS,
            "save_partial": SAVE_PARTIAL,
            "threshold_calibration_stage": "after_best_epoch_restore_on_validation",
            "eval_use_merge": EVAL_USE_MERGE,
            "train_kwargs": TRAIN_KWARGS,
            "ewc_lambda": EWC_LAMBDA if method == "ewc" else None,
            "ewc_max_batches": EWC_MAX_BATCHES if method == "ewc" else None,
            "lwf_lambda": LWF_LAMBDA if method == "lwf" else None,
            "lwf_temperature": LWF_TEMPERATURE if method == "lwf" else None,
            "replay_per_domain": (
                REPLAY_PER_DOMAIN if method in {"small_replay", "icarl_ncm"} else None
            ),
            "si_lambda": SI_LAMBDA if method == "si" else None,
            "si_xi": SI_XI if method == "si" else None,
            "gdumb_budget": GDUMB_BUDGET if method == "gdumb" else None,
        },
        "output_dir": str(output_dir),
        "split_info": split_infos or {},
        "best_epoch_by_domain": {},
        "best_threshold_by_domain": {},
        "best_val_metrics_by_domain": {},
        "class_weights_by_domain": {},
        "class_counts_by_domain": {},
        "checkpoint_candidates": [str(path) for path in CHECKPOINT_CANDIDATES],
    }

    for i, domain in enumerate(domains):
        domain_for_forward = domain if is_domain_head_method else None
        if is_domain_head_method:
            model.set_domain(domain)
            trainable_params = model.get_trainable_params()
        elif method == "gdumb":
            # GDumb trains a fresh network from scratch on the buffer for every
            # task; discard the previous task's weights entirely.
            model = build_shared_model()
            trainable_params = configure_trainable_params(model, method)
        print(f"\n{'=' * 60}")
        print(f"  Baseline {method} | Task {i + 1}/{t}: {domain}")
        print(f"{'=' * 60}")
        train_loader = train_loaders[domain]
        effective_train_loader = train_loader
        val_loader = val_loaders.get(domain)
        train_subset_labels = train_labels.get(domain)
        if method in {"small_replay", "icarl_ncm"} and replay_memory:
            effective_train_loader, train_subset_labels = _make_replay_loader(
                train_loader,
                replay_memory,
                seed=SEED,
                domain_index=i,
            )
            print(
                f"  [Replay] current={len(train_loader.dataset)} "
                f"memory={sum(len(y_mem) for _, y_mem in replay_memory)} "
                f"effective={len(effective_train_loader.dataset)}"
            )
        if method == "gdumb":
            x_new, y_new = _tensor_dataset_xy(train_loader.dataset)
            gdumb_buffer_x, gdumb_buffer_y = _gdumb_update_buffer(
                gdumb_buffer_x,
                gdumb_buffer_y,
                x_new,
                y_new,
                budget=GDUMB_BUDGET,
                seed=SEED + i * 1009 + 31,
            )
            loader_kwargs = {
                "dataset": TensorDataset(gdumb_buffer_x, gdumb_buffer_y),
                "batch_size": train_loader.batch_size,
                "shuffle": True,
                "num_workers": NUM_WORKERS,
                "pin_memory": DEVICE.type == "cuda",
            }
            if NUM_WORKERS > 0:
                loader_kwargs["persistent_workers"] = True
            effective_train_loader = DataLoader(**loader_kwargs)
            train_subset_labels = gdumb_buffer_y
            print(
                f"  [GDumb] buffer={len(gdumb_buffer_y)} (budget={GDUMB_BUDGET}); "
                "training fresh network on buffer only"
            )
        class_weights = None
        class_counts = None
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
            trainable_params,
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
        best_si_small_omega: Optional[Dict[str, torch.Tensor]] = None
        si_running_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        if method == "si":
            si_running_state = {
                "theta_before_task": {
                    name: param.detach().clone()
                    for name, param in named_trainable_parameters(model)
                },
                "small_omega": {
                    name: torch.zeros_like(param)
                    for name, param in named_trainable_parameters(model)
                },
            }
        print(
            f"  Train n={len(effective_train_loader.dataset)} | "
            f"Val n={len(val_loader.dataset) if val_loader is not None else 0}"
        )

        for ep in range(1, TRAIN_KWARGS["epochs"] + 1):
            loss = train_one_epoch_shared(
                model,
                effective_train_loader,
                optimizer,
                method,
                class_weights=class_weights,
                class_counts=class_counts,
                ewc_terms=si_terms if method == "si" else ewc_terms,
                old_model=old_model,
                domain=domain_for_forward,
                si_state=si_running_state,
            )
            scheduler.step()

            val_metrics = None
            val_calibrated_metrics = None
            should_validate = (
                val_loader is not None
                and (
                    ep == 1
                    or ep % VALIDATE_EVERY == 0
                    or ep == TRAIN_KWARGS["epochs"]
                )
            )
            if should_validate:
                val_labels, val_probs = collect_labels_probs_shared(
                    model,
                    val_loader,
                    domain=domain_for_forward,
                )
                val_metrics = binary_clinical_metrics(
                    labels=val_labels,
                    probs=val_probs,
                    threshold=0.5,
                )
                score_metrics = val_metrics
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
                    best_state = snapshot_model_state(model)
                    if method == "si" and si_running_state is not None:
                        best_si_small_omega = {
                            name: tensor.detach().clone()
                            for name, tensor in si_running_state["small_omega"].items()
                        }
                    best_val_metrics = dict(val_metrics)
                    best_val_calibrated_metrics = (
                        dict(val_calibrated_metrics)
                        if val_calibrated_metrics is not None
                        else None
                    )

            if ep % 5 == 0 or ep == TRAIN_KWARGS["epochs"]:
                if val_metrics is None:
                    status = "disabled" if val_loader is None else f"skipped (every {VALIDATE_EVERY})"
                    print(f"  Epoch {ep:3d} | Loss: {loss:.4f} | Val: {status}")
                else:
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
            restore_model_state(model, best_state)
        else:
            best_epoch = TRAIN_KWARGS["epochs"]
            best_val_metrics = {}

        if method == "si" and si_running_state is not None:
            # Finalize this task's online path-integral importance (Zenke et al.
            # 2017) at the same validation-selected epoch as the model state used
            # for the next task. This keeps the importance path, reference means,
            # and restored parameters aligned.
            finalize_si_importance(
                model,
                si_running_state,
                si_terms,
                small_omega_override=best_si_small_omega,
            )
            print(f"  [SI] updated cumulative importance after domain {domain}")

        if val_loader is not None:
            val_labels, val_probs = collect_labels_probs_shared(
                model,
                val_loader,
                domain=domain_for_forward,
            )
            best_threshold, calibrated_val_metrics = calibrate_threshold(
                val_labels,
                val_probs,
                metric_name="f1",
            )
        else:
            best_threshold = 0.5
            calibrated_val_metrics = {}

        domain_thresholds[domain] = best_threshold
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

        if method == "ewc":
            ewc_term = compute_ewc_term(
                model,
                train_loader,
                class_weights=class_weights,
                class_counts=class_counts,
            )
            ewc_terms.append(move_ewc_term_to_device(ewc_term))
            print(f"  [EWC] stored Fisher term {len(ewc_terms)}")

        if method in {"small_replay", "icarl_ncm"}:
            replay_memory.append(_store_replay_memory(train_loader, SEED, i))
            print(
                f"  [Replay] stored memory domains={len(replay_memory)} "
                f"samples={sum(len(y_mem) for _, y_mem in replay_memory)}"
            )

        if method == "icarl_ncm":
            x_mem_all = torch.cat([x_mem for x_mem, _ in replay_memory], dim=0)
            y_mem_all = torch.cat([y_mem for _, y_mem in replay_memory], dim=0)
            ncm_class_means = compute_ncm_class_means(model, x_mem_all, y_mem_all)
            if val_loader is not None:
                ncm_val_labels, ncm_val_probs = collect_labels_probs_ncm(
                    model, val_loader, ncm_class_means
                )
                ncm_threshold, ncm_calibrated_metrics = calibrate_threshold(
                    ncm_val_labels, ncm_val_probs, metric_name="f1"
                )
                domain_thresholds[domain] = ncm_threshold
                metadata["best_threshold_by_domain"][domain] = float(ncm_threshold)
                metadata["best_val_metrics_by_domain"][domain][
                    "ncm_calibrated_metrics"
                ] = ncm_calibrated_metrics
            print(f"  [iCaRL-NCM] recalibrated threshold via NCM features for {domain}")

        print(f"\n  Evaluation after {i + 1}/{t} domains:")
        for j in range(i + 1):
            eval_domain = domains[j]
            threshold = domain_thresholds.get(eval_domain, 0.5)
            if method == "icarl_ncm":
                ncm_labels, ncm_probs = collect_labels_probs_ncm(
                    model, test_loaders[eval_domain], ncm_class_means
                )
                metrics = binary_clinical_metrics(
                    labels=ncm_labels, probs=ncm_probs, threshold=threshold
                )
            else:
                metrics = evaluate_shared(
                    model,
                    test_loaders[eval_domain],
                    domain=eval_domain if is_domain_head_method else None,
                    threshold=threshold,
                )
            r_matrix[i][j] = metrics["f1"]
            for name, matrix in metric_matrices.items():
                if name in metrics:
                    matrix[i][j] = metrics[name]
            print(
                f"    {eval_domain:<12s}  "
                f"{format_metric_line(metrics, clinical=True)}"
            )

        if method == "lwf":
            old_model = copy.deepcopy(model).to(DEVICE)
            old_model.eval()
            for param in old_model.parameters():
                param.requires_grad = False
            print("  [LwF] updated frozen teacher for next task")

        _save_partial_state(
            output_dir=output_dir,
            method=method,
            task_index=i,
            domains=domains,
            model=model,
            r_matrix=r_matrix,
            metric_matrices=metric_matrices,
            metadata=metadata,
            domain_thresholds=domain_thresholds,
            ewc_terms=ewc_terms,
            replay_memory=replay_memory,
            si_terms=si_terms,
            gdumb_buffer=(gdumb_buffer_x, gdumb_buffer_y),
            ncm_class_means=ncm_class_means,
        )

    return r_matrix, metric_matrices, metadata


def main() -> None:
    method = BASELINE_METHOD
    set_global_seed(SEED)
    print(f"  Device: {DEVICE}")
    print(f"  Baseline method: {method}")
    print(f"  Domains: {' -> '.join(DOMAIN_NAMES)}")
    print(f"  Seed: {SEED}")
    print(
        f"  Hyperparams: epochs={TRAIN_KWARGS['epochs']}, "
        f"lr={TRAIN_KWARGS['lr']}, batch={TRAIN_KWARGS['batch_size']}"
    )
    if DRY_RUN or DEVICE.type != "cuda":
        reason = "dry-run mode" if DRY_RUN else "no CUDA device or forced CPU"
        print(f"\n  [Skip] {reason}; skipping baseline training.")
        return

    r_matrix, metric_matrices, metadata = run_shared_baseline(method)
    forgetting = compute_forgetting(r_matrix, DOMAIN_NAMES)
    output_dir = (
        Path(metadata["output_dir"])
        if "output_dir" in metadata
        else _output_dir_for_method(method)
    )
    paths = save_results(
        r_matrix,
        forgetting,
        DOMAIN_NAMES,
        output_dir,
        metric_matrices=metric_matrices,
        run_metadata=metadata,
    )
    config_path = Path(output_dir) / "baseline_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(metadata["config"], f, indent=2)
    print(f"  [Saved] baseline config: {config_path}")
    print(f"  [Done] summary: {paths['summary_json']}")


if __name__ == "__main__":
    main()
