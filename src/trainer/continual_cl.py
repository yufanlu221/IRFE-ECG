"""
持续学习训练循环骨架
====================
按数据域顺序训练，每次只激活当前域对应的 Adapter（参数隔离），
训练完成后评估所有已见域，记录遗忘矩阵。

训练顺序: Task1_CPSC → Task2_PTBXL → Task3_Georgia → Task4_Chapman
"""

import csv
import copy
import json
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.adapter.domain_adapter import AdapterCLModel
from models.backbone.net1d import Net1D
from data.loaders.cinc_dataset import (
    DOMAIN_TO_FILE,
    _validate_label_metadata,
)
from evaluate.clinical_metrics import binary_clinical_metrics, format_metric_line
from scripts.run_config import (
    checkpoint_candidates,
    default_data_dir,
    default_output_dir,
    env_bool,
    env_float,
    env_int,
    env_str,
)


# ══════════════════════════════════════════════════════════════════
#  配置（后续迁移到 configs/*.yaml）
# ══════════════════════════════════════════════════════════════════

FORCE_CPU = os.environ.get("ECG_FORCE_CPU", "0") == "1"
DRY_RUN = os.environ.get("ECG_CL_DRY_RUN", "0") == "1"
STRICT_LABEL_METADATA = os.environ.get("ECG_STRICT_LABEL_METADATA", "1") == "1"
ALLOW_LEGACY_TRAINER = env_bool("ECG_ALLOW_LEGACY_TRAINER", False)
SEED = env_int("ECG_SEED", 42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")

# .pt 文件存放目录（GPU 服务器路径）
DATA_DIR = str(default_data_dir())
OUTPUT_DIR = default_output_dir()

DOMAIN_NAMES = ["cpsc", "ptbxl", "georgia", "chapman"]

CHECKPOINT_CANDIDATES = checkpoint_candidates()

BACKBONE_KWARGS = dict(
    in_channels=1,
    base_filters=64,
    ratio=1,
    filter_list=[64, 160, 160, 400, 400, 1024, 1024],
    m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
    kernel_size=16,
    stride=2,
    groups_width=16,
    n_classes=2,
    use_bn=True,
    use_do=True,
    verbose=False,
)

ADAPTER_KWARGS = dict(
    embed_dim=env_int("ECG_ADAPTER_EMBED_DIM", 160),
    bottleneck=env_int("ECG_ADAPTER_BOTTLENECK", 32),
    num_classes=2,
    ema_beta=env_float("ECG_ADAPTER_EMA_BETA", 0.9),
    fast_alpha=env_float("ECG_ADAPTER_FAST_ALPHA", 0.3),
    hook_stage=env_int("ECG_ADAPTER_HOOK_STAGE", 2),
)

TRAIN_KWARGS = dict(
    seed=SEED,
    epochs=env_int("ECG_EPOCHS", 30),
    lr=env_float("ECG_LR", 1e-3),
    batch_size=env_int("ECG_BATCH_SIZE", 64),
    weight_decay=env_float("ECG_WEIGHT_DECAY", 1e-4),
    grad_clip=env_float("ECG_GRAD_CLIP", 1.0),
    ema_update=env_str("ECG_EMA_UPDATE", "epoch"),
    dry_run_batches=env_int("ECG_DRY_RUN_BATCHES", 2),
)
NUM_WORKERS = max(0, env_int("ECG_NUM_WORKERS", 0))
PREFETCH_FACTOR = max(1, env_int("ECG_PREFETCH_FACTOR", 2))

VAL_RATIO = env_float("ECG_VAL_RATIO", 0.15)
LOSS_MODE = env_str("ECG_LOSS", "weighted_ce").lower()
CLASS_WEIGHT_MODE = env_str("ECG_CLASS_WEIGHT_MODE", "domain").lower()
CALIBRATE_THRESHOLD = env_bool("ECG_CALIBRATE_THRESHOLD", True)
SELECT_BY = env_str("ECG_SELECT_BY", "val_macro_f1").lower()
FOCAL_GAMMA = env_float("ECG_FOCAL_GAMMA", 2.0)
MAX_CLASS_WEIGHT = env_float("ECG_MAX_CLASS_WEIGHT", 10.0)
LOGIT_ADJUST_TAU = env_float("ECG_LOGIT_ADJUST_TAU", 1.0)
LABEL_SMOOTHING = env_float("ECG_LABEL_SMOOTHING", 0.0)
_DEFAULT_EVAL_USE_MERGE = env_bool("ECG_EVAL_USE_MERGE", True)
EVAL_USE_MERGE = env_bool("ECG_USE_MERGE", _DEFAULT_EVAL_USE_MERGE)
THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)
ROUTER_MODE = env_str("ECG_ROUTER_MODE", "prototype").lower()
ROUTER_DISTANCE = env_str("ECG_ROUTER_DISTANCE", "cosine").lower()
ROUTER_PROTOTYPE = env_str("ECG_ROUTER_PROTOTYPE", "domain").lower()
ROUTER_EPS = env_float("ECG_ROUTER_EPS", 1e-4)
ROUTER_SWEEP = env_str("ECG_ROUTER_SWEEP", "").lower()
ROUTER_SUBSPACE_DIM = env_int("ECG_ROUTER_SUBSPACE_DIM", 32)
ROUTER_TOPK = env_int("ECG_ROUTER_TOPK", 2)
ROUTER_TOPK_TEMPERATURE = env_float("ECG_ROUTER_TOPK_TEMPERATURE", 0.5)
ROUTER_FEATURE_MEMORY_PER_DOMAIN = env_int("ECG_ROUTER_FEATURE_MEMORY_PER_DOMAIN", 2048)
ROUTER_LEARNED_EPOCHS = env_int("ECG_ROUTER_LEARNED_EPOCHS", 80)
ROUTER_LEARNED_LR = env_float("ECG_ROUTER_LEARNED_LR", 1e-2)
ROUTER_LEARNED_WEIGHT_DECAY = env_float("ECG_ROUTER_LEARNED_WEIGHT_DECAY", 1e-4)
ROUTER_UNCERTAIN_MARGIN = env_float("ECG_ROUTER_UNCERTAIN_MARGIN", 0.15)
ROUTER_DISTANCES = {"cosine", "euclidean", "mahalanobis", "residual", "linear"}
ROUTER_PROTOTYPES = {
    "domain",
    "domain_uncertain_top2",
    "domain_top2",
    "class_conditional",
    "subspace",
    "learned_linear",
    "learned_linear_uncertain_top2",
}
ROUTER_CONFIDENCE_VARIANTS = {"confidence_margin"}
ROUTER_LEARNED_VARIANTS = {"learned_linear", "learned_linear_uncertain_top2"}
ROUTER_UNCERTAIN_TOP2_VARIANTS = {
    "domain_uncertain_top2",
    "learned_linear_uncertain_top2",
}
ROUTER_VARIANTS = ROUTER_PROTOTYPES | ROUTER_CONFIDENCE_VARIANTS


def _router_variant_name(prototype: str, distance: str) -> str:
    if prototype in ROUTER_CONFIDENCE_VARIANTS:
        return prototype
    if prototype == "learned_linear_uncertain_top2":
        return prototype
    if prototype in ROUTER_LEARNED_VARIANTS:
        return prototype
    if prototype == "domain_uncertain_top2":
        return f"domain_{distance}_uncertain_top2"
    if prototype == "domain_top2":
        return f"domain_{distance}_top2"
    return f"{prototype}_{distance}"


def router_variant_specs(
    sweep: str = ROUTER_SWEEP,
    primary_prototype: str = ROUTER_PROTOTYPE,
    primary_distance: str = ROUTER_DISTANCE,
) -> List[Tuple[str, str, str]]:
    """Return (name, prototype, distance) router variants for a single run."""
    primary_prototype = primary_prototype.lower()
    primary_distance = primary_distance.lower()
    specs = [(primary_prototype, primary_distance)]
    sweep = (sweep or "").strip().lower()
    if sweep in {"", "0", "false", "off", "none"}:
        tokens: List[str] = []
    elif sweep in {"1", "true", "yes", "on", "default"}:
        tokens = [
            "domain:cosine",
            "domain:euclidean",
            "domain:mahalanobis",
            "class_conditional:cosine",
            "class_conditional:mahalanobis",
            "confidence_margin",
        ]
    elif sweep in {"extended", "new", "all"}:
        tokens = [
            "domain:cosine",
            "domain:euclidean",
            "domain:mahalanobis",
            "class_conditional:cosine",
            "class_conditional:mahalanobis",
            "confidence_margin",
            "domain_top2:cosine",
            "subspace:residual",
            "learned_linear",
            "learned_linear_uncertain_top2",
            "domain_uncertain_top2:cosine",
        ]
    else:
        tokens = [token.strip() for token in sweep.split(",") if token.strip()]

    for token in tokens:
        if token in {"confidence", "confidence_margin"}:
            specs.append(("confidence_margin", "margin"))
            continue
        if token in {"learned", "learned_linear"}:
            specs.append(("learned_linear", "linear"))
            continue
        if token in {"learned_uncertain_top2", "learned_linear_uncertain_top2"}:
            specs.append(("learned_linear_uncertain_top2", "linear"))
            continue
        if token in {"uncertain_top2", "domain_uncertain_top2"}:
            specs.append(("domain_uncertain_top2", primary_distance))
            continue
        if token in {"top2", "domain_top2"}:
            specs.append(("domain_top2", primary_distance))
            continue
        if token == "subspace":
            specs.append(("subspace", "residual"))
            continue
        parts = [part.strip() for part in token.replace("/", ":").split(":") if part.strip()]
        prototype = primary_prototype
        distance = primary_distance
        for part in parts:
            if part in ROUTER_PROTOTYPES:
                prototype = part
            elif part in {"confidence", "confidence_margin"}:
                prototype = "confidence_margin"
                distance = "margin"
            elif part in {"learned", "learned_linear"}:
                prototype = "learned_linear"
                distance = "linear"
            elif part in {"learned_uncertain_top2", "learned_linear_uncertain_top2"}:
                prototype = "learned_linear_uncertain_top2"
                distance = "linear"
            elif part in {"uncertain_top2", "domain_uncertain_top2"}:
                prototype = "domain_uncertain_top2"
            elif part in {"top2", "domain_top2"}:
                prototype = "domain_top2"
            elif part == "subspace":
                prototype = "subspace"
                distance = "residual"
            elif part in ROUTER_DISTANCES:
                distance = part
            else:
                raise ValueError(f"Unknown router sweep token {token!r}.")
        specs.append((prototype, distance))

    seen = set()
    unique_specs = []
    for prototype, distance in specs:
        if prototype not in ROUTER_VARIANTS:
            raise ValueError(
                "ECG_ROUTER_PROTOTYPE must be domain, class_conditional, "
                "domain_uncertain_top2, domain_top2, subspace, learned_linear, "
                "learned_linear_uncertain_top2, or confidence_margin."
            )
        if prototype in ROUTER_CONFIDENCE_VARIANTS:
            distance = "margin"
        elif prototype in ROUTER_LEARNED_VARIANTS:
            distance = "linear"
        elif prototype == "subspace":
            distance = "residual"
        elif distance not in ROUTER_DISTANCES:
            raise ValueError(
                "ECG_ROUTER_DISTANCE must be cosine, euclidean, mahalanobis, "
                "residual, or linear."
            )
        name = _router_variant_name(prototype, distance)
        if name in seen:
            continue
        seen.add(name)
        unique_specs.append((name, prototype, distance))
    return unique_specs


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible paper experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ══════════════════════════════════════════════════════════════════
#  模型构建
# ══════════════════════════════════════════════════════════════════

def _torch_load_dict(pt_path: Path) -> Dict[str, Any]:
    with pt_path.open("rb") as f:
        data = torch.load(f, map_location="cpu", weights_only=False)
    _validate_label_metadata(data, pt_path, strict=STRICT_LABEL_METADATA)
    return data


def _metadata_array(data: Dict[str, Any], key: str, n: int) -> Optional[np.ndarray]:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if len(value) != n:
        warnings.warn(
            f"Ignoring metadata field {key!r}: length {len(value)} != {n}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return value.astype(str)


def _best_group_array(data: Dict[str, Any], n: int) -> Tuple[np.ndarray, str]:
    for key in ("group_id", "waveform_hash"):
        groups = _metadata_array(data, key, n)
        if groups is not None:
            return groups, key
    warnings.warn(
        "No group_id/waveform_hash found; validation split falls back to "
        "index-level groups. Paper experiments should use grouped metadata.",
        RuntimeWarning,
        stacklevel=2,
    )
    return np.asarray([f"idx_{idx}" for idx in range(n)], dtype=str), "index"


def make_val_split(
    labels: np.ndarray | torch.Tensor,
    groups: Optional[np.ndarray] = None,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
    group_key: str = "group_id",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create a deterministic stratified group-aware train/val split."""
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels).astype(int)
    n = int(labels.shape[0])
    if groups is None:
        groups = np.asarray([f"idx_{idx}" for idx in range(n)], dtype=str)
        group_key = "index"
    else:
        groups = np.asarray(groups).astype(str)
    if groups.shape[0] != n:
        raise ValueError(f"groups length {groups.shape[0]} does not match labels {n}")

    if val_ratio <= 0 or n == 0:
        train_idx = np.arange(n, dtype=int)
        val_idx = np.asarray([], dtype=int)
        return train_idx, val_idx, {
            "group_key": group_key,
            "val_ratio": float(val_ratio),
            "seed": int(seed),
            "train_n": int(len(train_idx)),
            "val_n": 0,
            "train_groups": int(len(set(groups.tolist()))),
            "val_groups": 0,
            "group_overlap_count": 0,
            "mixed_group_count": 0,
            "fallback": group_key == "index",
        }

    rng = np.random.default_rng(seed)
    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        group_to_indices[str(group)].append(idx)

    groups_by_label: Dict[int, List[str]] = defaultdict(list)
    mixed_group_count = 0
    for group, indices in group_to_indices.items():
        group_labels = labels[indices]
        counts = np.bincount(group_labels, minlength=2)
        if np.count_nonzero(counts) > 1:
            mixed_group_count += 1
        groups_by_label[int(counts.argmax())].append(group)

    val_groups: set[str] = set()
    for _, label_groups in sorted(groups_by_label.items()):
        label_groups = list(label_groups)
        rng.shuffle(label_groups)
        if len(label_groups) < 2:
            continue
        n_val_groups = max(1, int(round(len(label_groups) * val_ratio)))
        n_val_groups = min(n_val_groups, len(label_groups) - 1)
        val_groups.update(label_groups[:n_val_groups])

    if not val_groups and len(group_to_indices) > 1:
        all_groups = list(group_to_indices.keys())
        rng.shuffle(all_groups)
        val_groups.add(all_groups[0])

    val_mask = np.asarray([str(group) in val_groups for group in groups], dtype=bool)
    train_idx = np.flatnonzero(~val_mask).astype(int)
    val_idx = np.flatnonzero(val_mask).astype(int)
    train_groups = set(groups[train_idx].tolist())
    heldout_groups = set(groups[val_idx].tolist())
    overlap = train_groups.intersection(heldout_groups)
    if overlap:
        raise RuntimeError(f"train/val group overlap detected: {sorted(overlap)[:5]}")

    info = {
        "group_key": group_key,
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "train_groups": int(len(train_groups)),
        "val_groups": int(len(heldout_groups)),
        "group_overlap_count": 0,
        "mixed_group_count": int(mixed_group_count),
        "fallback": group_key == "index",
        "train_support": {
            str(cls): int((labels[train_idx] == cls).sum()) for cls in (0, 1)
        },
        "val_support": {
            str(cls): int((labels[val_idx] == cls).sum()) for cls in (0, 1)
        },
    }
    return train_idx, val_idx, info


def _loader_for_dataset(
    dataset: TensorDataset,
    shuffle: bool,
    seed: int,
    domain_index: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + domain_index)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": TRAIN_KWARGS["batch_size"],
        "shuffle": shuffle,
        "generator": generator if shuffle else None,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(**loader_kwargs)


def get_cinc_train_val_test_loaders(
    data_dir: str,
    domains: List[str],
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> Tuple[
    Dict[str, DataLoader],
    Dict[str, Optional[DataLoader]],
    Dict[str, DataLoader],
    Dict[str, Dict[str, Any]],
    Dict[str, torch.Tensor],
]:
    """Load train/test .pt files and split train into train/val without touching test."""
    data_path = Path(data_dir)
    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, Optional[DataLoader]] = {}
    test_loaders: Dict[str, DataLoader] = {}
    split_infos: Dict[str, Dict[str, Any]] = {}
    train_labels: Dict[str, torch.Tensor] = {}

    for domain_index, domain in enumerate(domains):
        if domain not in DOMAIN_TO_FILE:
            raise ValueError(f"Unknown domain {domain!r}; choices: {list(DOMAIN_TO_FILE)}")
        prefix = DOMAIN_TO_FILE[domain]
        train_pt = data_path / f"{prefix}_train.pt"
        test_pt = data_path / f"{prefix}_test.pt"
        if not train_pt.exists():
            raise FileNotFoundError(f"Missing train data: {train_pt}")
        if not test_pt.exists():
            raise FileNotFoundError(f"Missing test data: {test_pt}")

        train_data = _torch_load_dict(train_pt)
        x = train_data["x"].float()
        y = train_data["y"].long()
        groups, group_key = _best_group_array(train_data, len(y))
        train_idx, val_idx, info = make_val_split(
            y,
            groups=groups,
            val_ratio=val_ratio,
            seed=seed + domain_index,
            group_key=group_key,
        )

        train_ds = TensorDataset(x[train_idx], y[train_idx])
        train_loaders[domain] = _loader_for_dataset(
            train_ds, shuffle=True, seed=seed, domain_index=domain_index
        )
        train_labels[domain] = y[train_idx].clone()
        if len(val_idx) > 0:
            val_ds = TensorDataset(x[val_idx], y[val_idx])
            val_loaders[domain] = _loader_for_dataset(
                val_ds, shuffle=False, seed=seed, domain_index=domain_index
            )
        else:
            val_loaders[domain] = None

        test_data = _torch_load_dict(test_pt)
        test_ds = TensorDataset(test_data["x"].float(), test_data["y"].long())
        test_loaders[domain] = _loader_for_dataset(
            test_ds, shuffle=False, seed=seed, domain_index=domain_index
        )
        split_infos[domain] = info

    return train_loaders, val_loaders, test_loaders, split_infos, train_labels


def resolve_checkpoint_path(
    checkpoint_path: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a pretrained ECGFounder checkpoint and fail loudly if requested."""
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"预训练权重不存在: {path}")
        return path

    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return None

def build_model(
    checkpoint_path: Optional[str] = None,
) -> AdapterCLModel:
    """
    构建持续学习模型。

    1. 实例化 Net1D backbone
    2. 可选加载预训练权重（map_location='cpu'）
    3. 冻结 backbone，挂载多域 Adapter + 快慢流分类头
    """
    backbone = Net1D(**BACKBONE_KWARGS)
    checkpoint = resolve_checkpoint_path(checkpoint_path)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw_sd = ckpt.get("state_dict", ckpt)
        model_sd = backbone.state_dict()

        filtered = {}
        for k, v in raw_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                stripped = ".".join(k.split(".")[1:])
                if stripped in model_sd and model_sd[stripped].shape == v.shape:
                    filtered[stripped] = v

        backbone.load_state_dict(filtered, strict=False)
        print(f"  [权重] 从 {checkpoint} 加载 {len(filtered)}/{len(model_sd)} 层预训练参数")
        if len(filtered) == 0:
            raise RuntimeError(f"权重文件可读但没有匹配参数: {checkpoint}")
    else:
        print("  [权重] 未找到预训练权重，将使用随机初始化 backbone（仅限结构调试）")

    model = AdapterCLModel(
        backbone=backbone,
        domain_names=DOMAIN_NAMES,
        **ADAPTER_KWARGS,
    )
    model = model.to(DEVICE)
    return model


# ══════════════════════════════════════════════════════════════════
#  单 epoch 训练
# ══════════════════════════════════════════════════════════════════

def make_batch_class_weights(
    labels: torch.Tensor,
    num_classes: int = 2,
    max_weight: float = MAX_CLASS_WEIGHT,
) -> torch.Tensor:
    """Inverse-frequency class weights with zero-count protection."""
    counts = torch.bincount(labels.detach().cpu(), minlength=num_classes).float()
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.clamp_max(max_weight).to(labels.device)


def compute_class_weights_from_labels(
    labels: torch.Tensor | np.ndarray,
    num_classes: int = 2,
    max_weight: float = MAX_CLASS_WEIGHT,
) -> torch.Tensor:
    """Compute inverse-frequency weights from the train subset only."""
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    counts = torch.bincount(labels.detach().cpu().long(), minlength=num_classes).float()
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.clamp_max(max_weight).float()


def class_counts_from_labels(
    labels: torch.Tensor | np.ndarray,
    num_classes: int = 2,
) -> torch.Tensor:
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    return torch.bincount(labels.detach().cpu().long(), minlength=num_classes).float()


def training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    class_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Loss switch used for ablations; defaults to train-subset weighted CE."""
    if LOSS_MODE == "plain":
        return F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)

    if LOSS_MODE == "balanced_softmax":
        counts = class_counts
        if counts is None:
            counts = torch.ones(logits.shape[-1])
        adjusted_logits = logits + counts.to(logits.device).clamp_min(1.0).log()
        return F.cross_entropy(adjusted_logits, labels, label_smoothing=LABEL_SMOOTHING)

    if LOSS_MODE in {"logit_adjusted_ce", "logit_adjust"}:
        counts = class_counts
        if counts is None:
            counts = torch.ones(logits.shape[-1])
        counts = counts.to(logits.device).clamp_min(1.0)
        priors = counts / counts.sum()
        adjusted_logits = logits + LOGIT_ADJUST_TAU * priors.clamp_min(1e-12).log()
        return F.cross_entropy(adjusted_logits, labels, label_smoothing=LABEL_SMOOTHING)

    if LOSS_MODE == "focal":
        weight = class_weights.to(logits.device) if class_weights is not None else None
        ce = F.cross_entropy(
            logits,
            labels,
            weight=weight,
            reduction="none",
            label_smoothing=LABEL_SMOOTHING,
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** FOCAL_GAMMA * ce).mean()

    if LOSS_MODE != "weighted_ce":
        raise ValueError(
            "ECG_LOSS must be one of weighted_ce, balanced_softmax, "
            "logit_adjusted_ce, focal, plain."
        )

    if CLASS_WEIGHT_MODE == "batch":
        weight = make_batch_class_weights(labels)
    elif CLASS_WEIGHT_MODE == "domain":
        weight = class_weights.to(logits.device) if class_weights is not None else None
    elif CLASS_WEIGHT_MODE in {"none", "off"}:
        weight = None
    else:
        raise ValueError("ECG_CLASS_WEIGHT_MODE must be domain, batch, or none.")
    return F.cross_entropy(logits, labels, weight=weight, label_smoothing=LABEL_SMOOTHING)


def snapshot_domain_state(model: AdapterCLModel, domain: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """Copy only the active domain adapter/head state for best-epoch restore."""
    return {
        "adapter": copy.deepcopy(model.adapters[domain].state_dict()),
        "head": copy.deepcopy(model.heads[domain].state_dict()),
    }


def restore_domain_state(
    model: AdapterCLModel,
    domain: str,
    state: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    model.adapters[domain].load_state_dict(state["adapter"])
    model.heads[domain].load_state_dict(state["head"])


def _metric_score(metrics: Dict[str, float], select_by: str = SELECT_BY) -> float:
    if select_by in {"val_macro_f1", "macro_f1", "f1"}:
        return float(metrics["f1"])
    if select_by in {"val_auc", "auc"}:
        return float(metrics["auc"])
    if select_by in {"val_balanced_acc", "balanced_acc"}:
        return float(metrics["balanced_acc"])
    raise ValueError("ECG_SELECT_BY must be val_macro_f1, val_auc, or val_balanced_acc.")


def calibrate_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    metric_name: str = "f1",
    thresholds: np.ndarray = THRESHOLD_GRID,
) -> Tuple[float, Dict[str, float]]:
    """Choose a threshold on validation data only."""
    if not CALIBRATE_THRESHOLD:
        metrics = binary_clinical_metrics(labels=labels, probs=probs, threshold=0.5)
        return 0.5, metrics

    best_threshold = 0.5
    best_metrics = binary_clinical_metrics(labels=labels, probs=probs, threshold=0.5)
    best_score = _metric_score(best_metrics, metric_name)
    for threshold in thresholds:
        metrics = binary_clinical_metrics(labels=labels, probs=probs, threshold=float(threshold))
        score = _metric_score(metrics, metric_name)
        if score > best_score or (
            np.isclose(score, best_score) and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def train_one_epoch(
    model: AdapterCLModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    domain: str,
    class_weights: Optional[torch.Tensor] = None,
    class_counts: Optional[torch.Tensor] = None,
) -> float:
    """
    Train one epoch.

    - The frozen backbone stays in eval mode, including BN/Dropout.
    - Only the active domain adapter and fast-stream head receive gradients.
    - EMA timing is controlled by TRAIN_KWARGS["ema_update"]:
      "batch" updates slow weights in this loop after each optimizer step;
      "epoch" updates slow weights in run_continual_learning after each epoch.
    """
    model.train()
    model.backbone.eval()  # BN 不更新统计量
    assert not model.backbone.training
    total_loss = 0.0
    for step, (x, y) in enumerate(loader):
        if DRY_RUN and step >= TRAIN_KWARGS["dry_run_batches"]:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        logits = model(x, domain=domain, use_merge=False)
        loss = training_loss(logits, y, class_weights=class_weights, class_counts=class_counts)
        loss.backward()

        # 梯度裁剪（只对可训练参数）
        trainable = model.get_trainable_params()
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, TRAIN_KWARGS["grad_clip"])

        optimizer.step()
        if TRAIN_KWARGS["ema_update"] == "batch":
            model.update_slow(domain)

        total_loss += loss.item()

    denom = min(len(loader), TRAIN_KWARGS["dry_run_batches"]) if DRY_RUN else len(loader)
    return total_loss / max(denom, 1)


# ══════════════════════════════════════════════════════════════════
#  评估
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_labels_probs(
    model: AdapterCLModel,
    loader: DataLoader,
    domain: str,
    use_merge: bool = EVAL_USE_MERGE,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x, domain=domain, use_merge=use_merge)
        all_logits.append(logits.cpu())
        all_labels.append(y)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
    return labels, probs


@torch.no_grad()
def evaluate(
    model: AdapterCLModel,
    loader: DataLoader,
    domain: str,
    threshold: float = 0.5,
    use_merge: bool = EVAL_USE_MERGE,
) -> Dict[str, float]:
    """
    评估指定域，返回含临床指标的结果字典。
    推理时使用快慢流融合（use_merge=True）。
    """
    labels, probs = collect_labels_probs(model, loader, domain, use_merge=use_merge)
    return binary_clinical_metrics(labels=labels, probs=probs, threshold=threshold)


@torch.no_grad()
def compute_domain_prototype(
    model: AdapterCLModel,
    loader: DataLoader,
    distance: str = ROUTER_DISTANCE,
    prototype: str = ROUTER_PROTOTYPE,
) -> Dict[str, Any]:
    """Fit a privacy-preserving feature prototype from a domain train subset."""
    distance = distance.lower()
    prototype = prototype.lower()
    prototype_for_stats = (
        "domain"
        if prototype in {"domain_top2", "domain_uncertain_top2", "subspace"}
        else prototype
    )
    if distance not in ROUTER_DISTANCES:
        raise ValueError(
            "ECG_ROUTER_DISTANCE must be cosine, euclidean, mahalanobis, residual, or linear."
        )
    if prototype_for_stats not in {"domain", "class_conditional"}:
        raise ValueError(
            "Feature prototypes can be built for domain, domain_top2, subspace, "
            "domain_uncertain_top2, or class_conditional routers."
        )

    model.eval()
    feature_sum = None
    feature_sq_sum = None
    feature_chunks = []
    count = 0
    class_stats: Dict[int, Dict[str, Any]] = {}
    for x, y in loader:
        x = x.to(DEVICE)
        features = model.extract_features(x)
        if distance == "cosine" and prototype != "subspace":
            features = F.normalize(features, dim=1)
        if prototype == "subspace":
            feature_chunks.append(features.detach().cpu())
        if feature_sum is None:
            feature_sum = features.sum(dim=0)
            feature_sq_sum = (features * features).sum(dim=0)
        else:
            feature_sum += features.sum(dim=0)
            feature_sq_sum += (features * features).sum(dim=0)
        count += int(features.shape[0])

        if prototype_for_stats == "class_conditional":
            labels = y.to(features.device).long()
            for cls_tensor in labels.unique(sorted=True):
                cls = int(cls_tensor.item())
                mask = labels == cls
                cls_features = features[mask]
                if cls_features.numel() == 0:
                    continue
                stats = class_stats.setdefault(
                    cls,
                    {"sum": torch.zeros_like(cls_features[0]), "sq_sum": torch.zeros_like(cls_features[0]), "count": 0},
                )
                stats["sum"] += cls_features.sum(dim=0)
                stats["sq_sum"] += (cls_features * cls_features).sum(dim=0)
                stats["count"] += int(cls_features.shape[0])

    if feature_sum is None or count == 0:
        raise RuntimeError("Cannot build prototype from an empty loader.")
    centroid = feature_sum / count
    if distance == "cosine":
        centroid = F.normalize(centroid, dim=0)
    variance = (feature_sq_sum / count) - (feature_sum / count).pow(2)
    result = {
        "centroid": centroid.detach().cpu(),
        "variance": variance.clamp_min(ROUTER_EPS).detach().cpu(),
        "count": int(count),
        "distance": distance,
        "prototype": prototype,
        "feature_var_mean": float(variance.clamp_min(0).mean().detach().cpu()),
    }
    if prototype == "subspace":
        feature_matrix = torch.cat(feature_chunks, dim=0).float()
        feature_mean = feature_matrix.mean(dim=0)
        centered = feature_matrix - feature_mean
        denom = max(int(centered.shape[0]) - 1, 1)
        cov = centered.t().matmul(centered) / denom
        eigvals, eigvecs = torch.linalg.eigh(cov)
        k = max(1, min(int(ROUTER_SUBSPACE_DIM), eigvecs.shape[1]))
        top_indices = torch.argsort(eigvals, descending=True)[:k]
        basis = eigvecs[:, top_indices].contiguous()
        result["mean"] = feature_mean.detach().cpu()
        result["basis"] = basis.detach().cpu()
        result["basis_dim"] = int(k)
        result["explained_variance"] = eigvals[top_indices].clamp_min(0).detach().cpu()
    if prototype_for_stats == "class_conditional":
        class_centroids = {}
        class_variances = {}
        class_counts = {}
        for cls, stats in sorted(class_stats.items()):
            cls_count = int(stats["count"])
            cls_centroid = stats["sum"] / cls_count
            if distance == "cosine":
                cls_centroid = F.normalize(cls_centroid, dim=0)
            cls_variance = (stats["sq_sum"] / cls_count) - (stats["sum"] / cls_count).pow(2)
            class_centroids[str(cls)] = cls_centroid.detach().cpu()
            class_variances[str(cls)] = cls_variance.clamp_min(ROUTER_EPS).detach().cpu()
            class_counts[str(cls)] = cls_count
        result["class_centroids"] = class_centroids
        result["class_variances"] = class_variances
        result["class_counts"] = class_counts
    return result


def _prototype_scores(
    features: torch.Tensor,
    centroids: torch.Tensor,
    variances: Optional[torch.Tensor],
    distance: str,
) -> torch.Tensor:
    if distance == "cosine":
        features_for_route = F.normalize(features, dim=1)
        centroids = F.normalize(centroids, dim=1)
        return features_for_route @ centroids.t()
    if distance == "euclidean":
        return -torch.cdist(features, centroids)
    if distance == "mahalanobis":
        if variances is None:
            variances = torch.ones_like(centroids)
        diff = features.unsqueeze(1) - centroids.unsqueeze(0)
        return -((diff * diff) / variances.clamp_min(ROUTER_EPS).unsqueeze(0)).mean(dim=-1)
    raise ValueError("ECG_ROUTER_DISTANCE must be cosine, euclidean, or mahalanobis.")


def _subspace_scores(
    features: torch.Tensor,
    prototypes: Dict[str, Dict[str, Any]],
    candidate_domains: List[str],
) -> torch.Tensor:
    domain_scores = []
    for domain in candidate_domains:
        proto = prototypes[domain]
        mean = proto["mean"].to(features.device)
        basis = proto["basis"].to(features.device)
        centered = features - mean
        projected = centered.matmul(basis).matmul(basis.t())
        residual = centered - projected
        residual_energy = residual.pow(2).mean(dim=1)
        total_energy = centered.pow(2).mean(dim=1).clamp_min(ROUTER_EPS)
        domain_scores.append(-(residual_energy / total_energy))
    return torch.stack(domain_scores, dim=1)


@torch.no_grad()
def collect_feature_memory(
    model: AdapterCLModel,
    loader: DataLoader,
    max_samples: int = ROUTER_FEATURE_MEMORY_PER_DOMAIN,
) -> torch.Tensor:
    """Store a small frozen-feature memory for learned domain routing."""
    model.eval()
    chunks = []
    seen = 0
    for x, _ in loader:
        x = x.to(DEVICE)
        features = model.extract_features(x).detach().cpu()
        if max_samples > 0:
            remaining = max_samples - seen
            if remaining <= 0:
                break
            features = features[:remaining]
        chunks.append(features)
        seen += int(features.shape[0])
        if max_samples > 0 and seen >= max_samples:
            break
    if not chunks:
        raise RuntimeError("Cannot collect learned-router feature memory from an empty loader.")
    return torch.cat(chunks, dim=0).float()


def fit_learned_linear_router(
    feature_memory: Dict[str, torch.Tensor],
    candidate_domains: List[str],
) -> Dict[str, Any]:
    """Train a compact linear router on stored frozen-feature memories."""
    if not candidate_domains:
        raise ValueError("candidate_domains must not be empty.")
    features = []
    labels = []
    for idx, domain in enumerate(candidate_domains):
        if domain not in feature_memory:
            raise ValueError(f"Missing feature memory for learned router domain {domain!r}.")
        domain_features = feature_memory[domain].float()
        features.append(domain_features)
        labels.append(torch.full((domain_features.shape[0],), idx, dtype=torch.long))
    x = F.normalize(torch.cat(features, dim=0), dim=1).to(DEVICE)
    y = torch.cat(labels, dim=0).to(DEVICE)
    router = torch.nn.Linear(x.shape[1], len(candidate_domains)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=ROUTER_LEARNED_LR,
        weight_decay=ROUTER_LEARNED_WEIGHT_DECAY,
    )
    class_counts = torch.bincount(y.detach().cpu(), minlength=len(candidate_domains)).float()
    class_weights = (class_counts.sum() / (len(candidate_domains) * class_counts.clamp_min(1.0))).to(DEVICE)
    router.train()
    for _ in range(max(1, ROUTER_LEARNED_EPOCHS)):
        optimizer.zero_grad()
        logits = router(x)
        loss = F.cross_entropy(logits, y, weight=class_weights)
        loss.backward()
        optimizer.step()
    router.eval()
    with torch.no_grad():
        train_acc = float((router(x).argmax(dim=1) == y).float().mean().detach().cpu())
    return {
        "domains": list(candidate_domains),
        "weight": router.weight.detach().cpu(),
        "bias": router.bias.detach().cpu(),
        "count": int(x.shape[0]),
        "counts_by_domain": {
            domain: int(feature_memory[domain].shape[0]) for domain in candidate_domains
        },
        "train_acc": train_acc,
        "prototype": "learned_linear",
        "distance": "linear",
    }


def route_features_by_prototype(
    features: torch.Tensor,
    prototypes: Dict[str, Dict[str, Any]],
    candidate_domains: List[str],
    distance: str = ROUTER_DISTANCE,
    prototype: str = ROUTER_PROTOTYPE,
) -> Tuple[List[str], torch.Tensor]:
    """Route each sample to the closest seen-domain prototype."""
    distance = distance.lower()
    prototype = prototype.lower()
    if not candidate_domains:
        raise ValueError("candidate_domains must not be empty.")
    if prototype in {"learned_linear", "learned_linear_uncertain_top2"}:
        router = prototypes.get("__router__")
        if router is None:
            raise ValueError("Missing learned router state under prototypes['__router__'].")
        router_domains = list(router["domains"])
        missing = [domain for domain in candidate_domains if domain not in router_domains]
        if missing:
            raise ValueError(f"Learned router does not contain domains: {missing}")
        all_scores = F.linear(
            F.normalize(features, dim=1),
            router["weight"].to(features.device),
            router["bias"].to(features.device),
        )
        indices = torch.tensor(
            [router_domains.index(domain) for domain in candidate_domains],
            device=features.device,
            dtype=torch.long,
        )
        scores = all_scores.index_select(dim=1, index=indices)
        chosen = scores.argmax(dim=1)
        routed_domains = [candidate_domains[int(idx)] for idx in chosen.detach().cpu()]
        return routed_domains, scores.detach().cpu()

    if distance not in ROUTER_DISTANCES:
        raise ValueError(
            "ECG_ROUTER_DISTANCE must be cosine, euclidean, mahalanobis, residual, or linear."
        )
    if prototype not in ROUTER_PROTOTYPES:
        raise ValueError(
            "ECG_ROUTER_PROTOTYPE must be domain, domain_top2, class_conditional, "
            "domain_uncertain_top2, subspace, learned_linear, or "
            "learned_linear_uncertain_top2."
        )
    missing = [domain for domain in candidate_domains if domain not in prototypes]
    if missing:
        raise ValueError(f"Missing prototypes for domains: {missing}")

    if prototype in {"domain", "domain_top2", "domain_uncertain_top2"}:
        centroids = torch.stack(
            [prototypes[domain]["centroid"] for domain in candidate_domains], dim=0
        ).to(features.device)
        variances = None
        if distance == "mahalanobis":
            variances = torch.stack(
                [
                    prototypes[domain].get("variance", torch.ones_like(prototypes[domain]["centroid"]))
                    for domain in candidate_domains
                ],
                dim=0,
            ).to(features.device)
        scores = _prototype_scores(features, centroids, variances, distance)
    elif prototype == "subspace":
        scores = _subspace_scores(features, prototypes, candidate_domains)
    else:
        domain_scores = []
        for domain in candidate_domains:
            domain_proto = prototypes[domain]
            class_centroids = domain_proto.get("class_centroids")
            if not class_centroids:
                raise ValueError(
                    f"Missing class_conditional prototypes for domain {domain!r}."
                )
            labels = sorted(class_centroids.keys())
            centroids = torch.stack([class_centroids[label] for label in labels], dim=0).to(
                features.device
            )
            variances = None
            if distance == "mahalanobis":
                class_variances = domain_proto.get("class_variances", {})
                variances = torch.stack(
                    [
                        class_variances.get(label, torch.ones_like(class_centroids[label]))
                        for label in labels
                    ],
                    dim=0,
                ).to(features.device)
            class_scores = _prototype_scores(features, centroids, variances, distance)
            domain_scores.append(class_scores.max(dim=1).values)
        scores = torch.stack(domain_scores, dim=1)

    chosen = scores.argmax(dim=1)
    routed_domains = [candidate_domains[int(idx)] for idx in chosen.detach().cpu()]
    return routed_domains, scores.detach().cpu()


def route_features_by_confidence_margin(
    model: AdapterCLModel,
    features: torch.Tensor,
    candidate_domains: List[str],
    use_merge: bool = EVAL_USE_MERGE,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Route to the expert whose head gives the largest class-confidence margin."""
    if not candidate_domains:
        raise ValueError("candidate_domains must not be empty.")

    logits_by_domain = []
    scores_by_domain = []
    for domain in candidate_domains:
        logits = model.forward_features(features, domain=domain, use_merge=use_merge)
        probs = torch.softmax(logits, dim=-1)
        if probs.shape[1] < 2:
            raise ValueError("confidence_margin routing requires at least two classes.")
        top2 = probs.topk(k=2, dim=1).values
        scores_by_domain.append(top2[:, 0] - top2[:, 1])
        logits_by_domain.append(logits)

    scores = torch.stack(scores_by_domain, dim=1)
    chosen = scores.argmax(dim=1)
    routed_domains = [candidate_domains[int(idx)] for idx in chosen.detach().cpu()]
    stacked_logits = torch.stack(logits_by_domain, dim=1)
    selected_logits = stacked_logits[
        torch.arange(features.shape[0], device=features.device), chosen
    ]
    return routed_domains, scores.detach().cpu(), selected_logits


def summarize_router_routes(
    routes: List[str],
    candidate_domains: List[str],
) -> Dict[str, Any]:
    """Summarize route destinations for one true-domain evaluation row."""
    total = len(routes)
    counts = {domain: 0 for domain in candidate_domains}
    for domain in routes:
        counts[domain] = counts.get(domain, 0) + 1
    distribution = {
        domain: (float(count) / total if total else float("nan"))
        for domain, count in counts.items()
    }
    return {
        "total": int(total),
        "counts": {domain: int(count) for domain, count in counts.items()},
        "distribution": distribution,
    }


@torch.no_grad()
def collect_labels_probs_routed(
    model: AdapterCLModel,
    loader: DataLoader,
    eval_domain: str,
    candidate_domains: List[str],
    prototypes: Dict[str, Dict[str, Any]],
    thresholds: Dict[str, float],
    use_merge: bool = EVAL_USE_MERGE,
    distance: str = ROUTER_DISTANCE,
    prototype: str = ROUTER_PROTOTYPE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Evaluate without oracle domain IDs by routing each sample to an expert."""
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    all_routes: List[str] = []
    all_scores = []

    for x, y in loader:
        x = x.to(DEVICE)
        features = model.extract_features(x)
        if prototype in ROUTER_CONFIDENCE_VARIANTS:
            routed_domains, route_scores, logits = route_features_by_confidence_margin(
                model,
                features,
                candidate_domains=candidate_domains,
                use_merge=use_merge,
            )
        else:
            routed_domains, route_scores = route_features_by_prototype(
                features,
                prototypes=prototypes,
                candidate_domains=candidate_domains,
                distance=distance,
                prototype=prototype,
            )
            if prototype == "domain_top2" or prototype in ROUTER_UNCERTAIN_TOP2_VARIANTS:
                score_tensor = route_scores.to(features.device)
                topk = max(1, min(int(ROUTER_TOPK), len(candidate_domains)))
                top_scores, top_indices = score_tensor.topk(k=topk, dim=1)
                weights = torch.softmax(top_scores / max(ROUTER_TOPK_TEMPERATURE, ROUTER_EPS), dim=1)
                logits = torch.zeros(
                    (features.shape[0], 2),
                    device=features.device,
                    dtype=features.dtype,
                )
                hard_logits = None
                uncertain_mask = torch.ones(features.shape[0], device=features.device, dtype=torch.bool)
                if prototype in ROUTER_UNCERTAIN_TOP2_VARIANTS and topk > 1:
                    margins = top_scores[:, 0] - top_scores[:, 1]
                    uncertain_mask = margins <= ROUTER_UNCERTAIN_MARGIN
                    hard_logits = torch.empty_like(logits)
                    for domain in sorted(set(routed_domains)):
                        indices = [
                            idx for idx, routed_domain in enumerate(routed_domains)
                            if routed_domain == domain
                        ]
                        index_tensor = torch.tensor(indices, device=features.device, dtype=torch.long)
                        hard_logits[index_tensor] = model.forward_features(
                            features[index_tensor],
                            domain=domain,
                            use_merge=use_merge,
                        )
                    logits[:] = hard_logits
                for rank in range(topk):
                    rank_indices = top_indices[:, rank]
                    rank_weights = weights[:, rank]
                    for domain_idx, domain in enumerate(candidate_domains):
                        mask = (rank_indices == domain_idx) & uncertain_mask
                        if not mask.any():
                            continue
                        domain_logits = model.forward_features(
                            features[mask],
                            domain=domain,
                            use_merge=use_merge,
                        )
                        if prototype in ROUTER_UNCERTAIN_TOP2_VARIANTS:
                            if rank == 0:
                                logits[mask] = 0.0
                            logits[mask] += domain_logits * rank_weights[mask].unsqueeze(1)
                        else:
                            logits[mask] += domain_logits * rank_weights[mask].unsqueeze(1)
            else:
                logits = torch.empty(
                    (features.shape[0], 2),
                    device=features.device,
                    dtype=features.dtype,
                )
                for domain in sorted(set(routed_domains)):
                    indices = [
                        idx for idx, routed_domain in enumerate(routed_domains)
                        if routed_domain == domain
                    ]
                    index_tensor = torch.tensor(indices, device=features.device, dtype=torch.long)
                    logits[index_tensor] = model.forward_features(
                        features[index_tensor],
                        domain=domain,
                        use_merge=use_merge,
                    )

        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        sample_thresholds = np.asarray(
            [thresholds.get(domain, 0.5) for domain in routed_domains],
            dtype=float,
        )
        preds = (probs >= sample_thresholds).astype(int)
        all_labels.append(y.numpy())
        all_probs.append(probs)
        all_preds.append(preds)
        all_routes.extend(routed_domains)
        all_scores.append(route_scores.numpy())

    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    scores = np.concatenate(all_scores, axis=0)
    return labels, probs, preds, all_routes, scores


def evaluate_routed(
    model: AdapterCLModel,
    loader: DataLoader,
    eval_domain: str,
    candidate_domains: List[str],
    prototypes: Dict[str, Dict[str, Any]],
    thresholds: Dict[str, float],
    use_merge: bool = EVAL_USE_MERGE,
    distance: str = ROUTER_DISTANCE,
    prototype: str = ROUTER_PROTOTYPE,
) -> Dict[str, Any]:
    labels, probs, preds, routes, scores = collect_labels_probs_routed(
        model,
        loader,
        eval_domain=eval_domain,
        candidate_domains=candidate_domains,
        prototypes=prototypes,
        thresholds=thresholds,
        use_merge=use_merge,
        distance=distance,
        prototype=prototype,
    )
    metrics = binary_clinical_metrics(
        labels=labels,
        probs=probs,
        preds=preds,
        threshold=float("nan"),
    )
    metrics["router_acc"] = float(np.mean(np.asarray(routes) == eval_domain))
    metrics["router_margin"] = float(
        np.mean(
            np.partition(scores, kth=-2, axis=1)[:, -1]
            - np.partition(scores, kth=-2, axis=1)[:, -2]
        )
    ) if scores.shape[1] > 1 else float("nan")
    if prototype in ROUTER_UNCERTAIN_TOP2_VARIANTS and scores.shape[1] > 1:
        top2 = np.partition(scores, kth=-2, axis=1)[:, -2:]
        margins = top2[:, 1] - top2[:, 0]
        metrics["router_uncertain_fraction"] = float(
            np.mean(margins <= ROUTER_UNCERTAIN_MARGIN)
        )
    else:
        metrics["router_uncertain_fraction"] = float("nan")
    route_summary = summarize_router_routes(routes, candidate_domains)
    metrics["router_total"] = route_summary["total"]
    metrics["router_counts"] = route_summary["counts"]
    metrics["router_distribution"] = route_summary["distribution"]
    return metrics


# ══════════════════════════════════════════════════════════════════
#  持续学习主循环
# ══════════════════════════════════════════════════════════════════

def _run_continual_learning_legacy(
    model: AdapterCLModel,
    train_loaders: Dict[str, DataLoader],
    test_loaders: Dict[str, DataLoader],
    domain_order: Optional[List[str]] = None,
    return_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    按域顺序持续学习，返回遗忘矩阵 R。

    R[i][j] = 训练完第 i 个域后，在第 j 个域测试集上的 Macro-F1

    每个域的训练流程：
      1. set_domain(domain)  → 激活对应 Adapter，冻结其余
      2. 构建 optimizer（仅当前域 Adapter + 快流分类头）
      3. train_one_epoch × N epochs
      4. 评估所有已见域 → 填入 R 矩阵
    """
    if not ALLOW_LEGACY_TRAINER:
        raise RuntimeError(
            "_run_continual_learning_legacy evaluates the test set during the "
            "training loop and is kept only to reproduce old pilot runs. Use "
            "run_continual_learning() for formal experiments, or set "
            "ECG_ALLOW_LEGACY_TRAINER=1 explicitly for legacy reproduction."
        )
    domains = domain_order or DOMAIN_NAMES
    if TRAIN_KWARGS["ema_update"] not in {"batch", "epoch"}:
        raise ValueError("TRAIN_KWARGS['ema_update'] must be 'batch' or 'epoch'.")
    T = len(domains)
    R = np.full((T, T), np.nan)
    metric_names = [
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
    ]
    metric_matrices = {name: np.full((T, T), np.nan) for name in metric_names}

    for i, domain in enumerate(domains):
        print(f"\n{'='*60}")
        print(f"  Task {i+1}/{T}: {domain}")
        print(f"{'='*60}")

        # ── 切换域 ──────────────────────────────────────────────
        model.set_domain(domain)

        # ── 构建优化器（仅可训练参数）───────────────────────────
        trainable = model.get_trainable_params()
        optimizer = torch.optim.AdamW(
            trainable,
            lr=TRAIN_KWARGS["lr"],
            weight_decay=TRAIN_KWARGS["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TRAIN_KWARGS["epochs"],
            eta_min=TRAIN_KWARGS["lr"] * 0.01,
        )

        # ── 训练 ────────────────────────────────────────────────
        loader = train_loaders[domain]
        for ep in range(1, TRAIN_KWARGS["epochs"] + 1):
            loss = train_one_epoch(model, loader, optimizer, domain)
            if TRAIN_KWARGS["ema_update"] == "epoch":
                model.update_slow(domain)
            scheduler.step()

            if ep % 5 == 0 or ep == TRAIN_KWARGS["epochs"]:
                metrics = evaluate(model, test_loaders[domain], domain)
                print(
                    f"  Epoch {ep:3d} | Loss: {loss:.4f} | "
                    f"{format_metric_line(metrics, clinical=True)}"
                )

        # ── 评估所有已见域 ──────────────────────────────────────
        print(f"\n  全任务评估（已见 {i+1}/{T} 个域）:")
        for j in range(i + 1):
            m = evaluate(model, test_loaders[domains[j]], domains[j])
            R[i][j] = m["f1"]
            for name, matrix in metric_matrices.items():
                if name in m:
                    matrix[i][j] = m[name]
            print(f"    {domains[j]:<12s}  {format_metric_line(m, clinical=True)}")

    return (R, metric_matrices) if return_metrics else R


# ══════════════════════════════════════════════════════════════════
#  遗忘分析
# ══════════════════════════════════════════════════════════════════

def run_continual_learning(
    model: AdapterCLModel,
    train_loaders: Dict[str, DataLoader],
    test_loaders: Dict[str, DataLoader],
    domain_order: Optional[List[str]] = None,
    val_loaders: Optional[Dict[str, Optional[DataLoader]]] = None,
    split_infos: Optional[Dict[str, Dict[str, Any]]] = None,
    train_labels_by_domain: Optional[Dict[str, torch.Tensor]] = None,
    return_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """Continual training with train-only weights, val-only model selection, and test-only reporting."""
    domains = domain_order or DOMAIN_NAMES
    if TRAIN_KWARGS["ema_update"] not in {"batch", "epoch"}:
        raise ValueError("TRAIN_KWARGS['ema_update'] must be 'batch' or 'epoch'.")
    if ROUTER_MODE not in {"prototype", "oracle"}:
        raise ValueError("ECG_ROUTER_MODE must be prototype or oracle.")
    router_specs = router_variant_specs() if ROUTER_MODE == "prototype" else []
    primary_router_name = (
        _router_variant_name(ROUTER_PROTOTYPE, ROUTER_DISTANCE)
        if ROUTER_MODE == "prototype"
        else ""
    )

    t = len(domains)
    r_matrix = np.full((t, t), np.nan)
    metric_names = [
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
        "router_acc",
        "router_margin",
        "router_uncertain_fraction",
    ]
    metric_matrices = {name: np.full((t, t), np.nan) for name in metric_names}
    oracle_r_matrix = np.full((t, t), np.nan)
    routed_r_matrix = np.full((t, t), np.nan)
    oracle_metric_matrices = {
        name: np.full((t, t), np.nan) for name in metric_names
    }
    routed_metric_matrices = {
        name: np.full((t, t), np.nan) for name in metric_names
    }
    domain_thresholds: Dict[str, float] = {}
    router_prototypes: Dict[str, Dict[str, Any]] = {}
    router_feature_memory: Dict[str, torch.Tensor] = {}
    router_prototypes_by_variant: Dict[str, Dict[str, Dict[str, Any]]] = {
        name: {} for name, _, _ in router_specs
    }
    router_sweep_metric_matrices = {
        name: {metric: np.full((t, t), np.nan) for metric in metric_names}
        for name, _, _ in router_specs
    }
    metadata: Dict[str, Any] = {
        "config": {
            "val_ratio": VAL_RATIO,
            "loss_mode": LOSS_MODE,
            "class_weight_mode": CLASS_WEIGHT_MODE,
            "label_smoothing": LABEL_SMOOTHING,
            "calibrate_threshold": CALIBRATE_THRESHOLD,
            "select_by": SELECT_BY,
            "epoch_selection_threshold": 0.5,
            "threshold_calibration_stage": "after_best_epoch_restore_on_validation",
            "eval_use_merge": EVAL_USE_MERGE,
            "router_mode": ROUTER_MODE,
            "router_distance": ROUTER_DISTANCE,
            "router_prototype": ROUTER_PROTOTYPE,
            "router_sweep": [
                {"name": name, "prototype": prototype, "distance": distance}
                for name, prototype, distance in router_specs
            ],
            "router_feature_memory_per_domain": ROUTER_FEATURE_MEMORY_PER_DOMAIN,
            "router_learned_epochs": ROUTER_LEARNED_EPOCHS,
            "router_learned_lr": ROUTER_LEARNED_LR,
            "router_subspace_dim": ROUTER_SUBSPACE_DIM,
            "router_topk": ROUTER_TOPK,
            "router_topk_temperature": ROUTER_TOPK_TEMPERATURE,
            "router_uncertain_margin": ROUTER_UNCERTAIN_MARGIN,
            "adapter_kwargs": ADAPTER_KWARGS,
            "primary_router_name": primary_router_name,
            "primary_eval": "routed" if ROUTER_MODE == "prototype" else "oracle",
            "ema_update": TRAIN_KWARGS["ema_update"],
            "seed": SEED,
        },
        "split_info": split_infos or {},
        "best_epoch_by_domain": {},
        "best_threshold_by_domain": {},
        "best_val_metrics_by_domain": {},
        "class_weights_by_domain": {},
        "class_counts_by_domain": {},
        "router_prototypes": {},
        "router_sweep_prototypes": {},
        "router_sweep_route_counts": {name: {} for name, _, _ in router_specs},
        "router_sweep_route_distribution": {name: {} for name, _, _ in router_specs},
    }

    for i, domain in enumerate(domains):
        print(f"\n{'='*60}")
        print(f"  Task {i+1}/{t}: {domain}")
        print(f"{'='*60}")

        model.set_domain(domain)
        trainable = model.get_trainable_params()
        optimizer = torch.optim.AdamW(
            trainable,
            lr=TRAIN_KWARGS["lr"],
            weight_decay=TRAIN_KWARGS["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TRAIN_KWARGS["epochs"],
            eta_min=TRAIN_KWARGS["lr"] * 0.01,
        )

        train_labels = None
        if train_labels_by_domain is not None:
            train_labels = train_labels_by_domain.get(domain)
        class_weights = None
        class_counts = None
        if train_labels is not None:
            class_counts = class_counts_from_labels(train_labels)
            if LOSS_MODE in {"weighted_ce", "focal"} and CLASS_WEIGHT_MODE == "domain":
                class_weights = compute_class_weights_from_labels(train_labels)
            metadata["class_counts_by_domain"][domain] = [
                float(value) for value in class_counts.tolist()
            ]
            metadata["class_weights_by_domain"][domain] = (
                [float(value) for value in class_weights.tolist()]
                if class_weights is not None
                else None
            )

        train_loader = train_loaders[domain]
        val_loader = val_loaders.get(domain) if val_loaders is not None else None
        best_score = -float("inf")
        best_epoch = 0
        best_state = None
        best_val_metrics = None

        print(
            f"  Train n={len(train_loader.dataset)} | "
            f"Val n={len(val_loader.dataset) if val_loader is not None else 0} | "
            f"loss={LOSS_MODE}/{CLASS_WEIGHT_MODE} | use_merge={int(EVAL_USE_MERGE)}"
        )

        for ep in range(1, TRAIN_KWARGS["epochs"] + 1):
            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                domain,
                class_weights=class_weights,
                class_counts=class_counts,
            )
            if TRAIN_KWARGS["ema_update"] == "epoch":
                model.update_slow(domain)
            scheduler.step()

            val_metrics = None
            if val_loader is not None:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    domain,
                    threshold=0.5,
                    use_merge=EVAL_USE_MERGE,
                )
                score = _metric_score(val_metrics, SELECT_BY)
                if np.isnan(score):
                    score = -float("inf")
                if score > best_score:
                    best_score = score
                    best_epoch = ep
                    best_state = snapshot_domain_state(model, domain)
                    best_val_metrics = dict(val_metrics)

            if ep % 5 == 0 or ep == TRAIN_KWARGS["epochs"]:
                if val_metrics is not None:
                    print(
                        f"  Epoch {ep:3d} | Loss: {loss:.4f} | "
                        f"Val {format_metric_line(val_metrics, clinical=True)}"
                    )
                else:
                    print(f"  Epoch {ep:3d} | Loss: {loss:.4f} | Val: disabled")

        if best_state is not None:
            restore_domain_state(model, domain, best_state)
        else:
            best_epoch = TRAIN_KWARGS["epochs"]
            best_val_metrics = {}

        if val_loader is not None:
            val_labels, val_probs = collect_labels_probs(
                model, val_loader, domain, use_merge=EVAL_USE_MERGE
            )
            best_threshold, calibrated_val_metrics = calibrate_threshold(
                val_labels, val_probs, metric_name="f1"
            )
        else:
            best_threshold = 0.5
            calibrated_val_metrics = {}

        domain_thresholds[domain] = best_threshold
        metadata["best_epoch_by_domain"][domain] = int(best_epoch)
        metadata["best_threshold_by_domain"][domain] = float(best_threshold)
        metadata["best_val_metrics_by_domain"][domain] = {
            "selected_epoch_metrics_at_0_5": best_val_metrics,
            "calibrated_metrics": calibrated_val_metrics,
        }
        if ROUTER_MODE == "prototype":
            needs_learned_router = any(
                variant_prototype in ROUTER_LEARNED_VARIANTS
                for _, variant_prototype, _ in router_specs
            )
            if needs_learned_router:
                router_feature_memory[domain] = collect_feature_memory(
                    model,
                    train_loader,
                    max_samples=ROUTER_FEATURE_MEMORY_PER_DOMAIN,
                )
            for variant_name, variant_prototype, variant_distance in router_specs:
                if variant_prototype in ROUTER_CONFIDENCE_VARIANTS:
                    metadata["router_sweep_prototypes"].setdefault(variant_name, {})[
                        domain
                    ] = {
                        "prototype": variant_prototype,
                        "distance": variant_distance,
                        "count": 0,
                        "note": "confidence router uses expert logits; no feature prototype is stored.",
                    }
                    continue
                if variant_prototype in ROUTER_LEARNED_VARIANTS:
                    learned_router = fit_learned_linear_router(
                        router_feature_memory,
                        domains[: i + 1],
                    )
                    router_prototypes_by_variant[variant_name]["__router__"] = learned_router
                    metadata["router_sweep_prototypes"].setdefault(variant_name, {})[
                        domains[i]
                    ] = {
                        "prototype": variant_prototype,
                        "distance": variant_distance,
                        "count": learned_router["count"],
                        "counts_by_domain": learned_router["counts_by_domain"],
                        "train_acc": learned_router["train_acc"],
                    }
                    if variant_name == primary_router_name:
                        router_prototypes["__router__"] = learned_router
                        metadata["router_prototypes"]["__router__"] = {
                            "prototype": variant_prototype,
                            "distance": variant_distance,
                            "count": learned_router["count"],
                            "counts_by_domain": learned_router["counts_by_domain"],
                            "train_acc": learned_router["train_acc"],
                        }
                    continue
                proto = compute_domain_prototype(
                    model,
                    train_loader,
                    distance=variant_distance,
                    prototype=variant_prototype,
                )
                router_prototypes_by_variant[variant_name][domain] = proto
                proto_summary = {
                    "count": proto["count"],
                    "distance": proto["distance"],
                    "prototype": proto["prototype"],
                    "feature_var_mean": proto["feature_var_mean"],
                    "class_counts": proto.get("class_counts"),
                    "basis_dim": proto.get("basis_dim"),
                }
                metadata["router_sweep_prototypes"].setdefault(variant_name, {})[
                    domain
                ] = proto_summary
                if variant_name == primary_router_name:
                    router_prototypes[domain] = proto
                    metadata["router_prototypes"][domain] = proto_summary
        print(
            f"  [Best] epoch={best_epoch} | threshold={best_threshold:.3f} | "
            f"val_f1={calibrated_val_metrics.get('f1', float('nan')):.3f}"
        )

        print(f"\n  Evaluation after {i+1}/{t} domains:")
        seen_domains = domains[: i + 1]
        for j in range(i + 1):
            eval_domain = domains[j]
            threshold = domain_thresholds.get(eval_domain, 0.5)
            oracle_metrics = evaluate(
                model,
                test_loaders[eval_domain],
                eval_domain,
                threshold=threshold,
                use_merge=EVAL_USE_MERGE,
            )
            oracle_r_matrix[i][j] = oracle_metrics["f1"]
            for name, matrix in oracle_metric_matrices.items():
                if name in oracle_metrics:
                    matrix[i][j] = oracle_metrics[name]

            routed_metrics = None
            if ROUTER_MODE == "prototype":
                for variant_name, variant_prototype, variant_distance in router_specs:
                    variant_metrics = evaluate_routed(
                        model,
                        test_loaders[eval_domain],
                        eval_domain=eval_domain,
                        candidate_domains=seen_domains,
                        prototypes=router_prototypes_by_variant[variant_name],
                        thresholds=domain_thresholds,
                        use_merge=EVAL_USE_MERGE,
                        distance=variant_distance,
                        prototype=variant_prototype,
                    )
                    after_key = domains[i]
                    metadata["router_sweep_route_counts"].setdefault(
                        variant_name, {}
                    ).setdefault(after_key, {})[eval_domain] = variant_metrics.get(
                        "router_counts", {}
                    )
                    metadata["router_sweep_route_distribution"].setdefault(
                        variant_name, {}
                    ).setdefault(after_key, {})[eval_domain] = variant_metrics.get(
                        "router_distribution", {}
                    )
                    for name, matrix in router_sweep_metric_matrices[
                        variant_name
                    ].items():
                        if name in variant_metrics:
                            matrix[i][j] = variant_metrics[name]
                    if variant_name == primary_router_name:
                        routed_metrics = variant_metrics
                if routed_metrics is None:
                    raise RuntimeError("Primary router variant was not evaluated.")
                routed_r_matrix[i][j] = routed_metrics["f1"]
                for name, matrix in routed_metric_matrices.items():
                    if name in routed_metrics:
                        matrix[i][j] = routed_metrics[name]
                primary_metrics = routed_metrics
            else:
                routed_r_matrix[i][j] = np.nan
                primary_metrics = oracle_metrics

            r_matrix[i][j] = primary_metrics["f1"]
            for name, matrix in metric_matrices.items():
                if name in primary_metrics:
                    matrix[i][j] = primary_metrics[name]

            if routed_metrics is None:
                print(
                    f"    {eval_domain:<12s}  "
                    f"Oracle {format_metric_line(oracle_metrics, clinical=True)}"
                )
            else:
                print(
                    f"    {eval_domain:<12s}  "
                    f"Routed {format_metric_line(routed_metrics, clinical=True)} | "
                    f"RouterAcc: {routed_metrics['router_acc']:.3f} | "
                    f"OracleF1: {oracle_metrics['f1']:.3f}"
                )

    metadata["oracle_forgetting_matrix"] = oracle_r_matrix
    metadata["routed_forgetting_matrix"] = routed_r_matrix
    metadata["oracle_metric_matrices"] = oracle_metric_matrices
    metadata["routed_metric_matrices"] = routed_metric_matrices
    metadata["router_sweep_metric_matrices"] = router_sweep_metric_matrices
    return (r_matrix, metric_matrices, metadata) if return_metrics else r_matrix


def compute_forgetting(R: np.ndarray, domain_names: List[str]) -> Dict:
    """
    从遗忘矩阵 R 计算遗忘指标。

    返回:
      - diag_f1:     每个域刚训练完时的 F1
      - final_f1:    最终（全部训完）每个域的 F1
      - bwt:         Backward transfer, final - initial（负值 = 遗忘）
      - forgetting:  遗忘量，等于 -BWT（正值 = 遗忘）
      - mean_f1:     最终平均 F1
      - mean_bwt:    前 T-1 个域的平均 BWT；最后一个域无后续训练，默认排除
    """
    T = R.shape[0]
    diag = np.array([R[i][i] for i in range(T)])
    final = R[T - 1, :T]
    bwt = np.array([R[T - 1, i] - R[i][i] for i in range(T - 1)])
    forgetting = -bwt

    print(f"\n{'='*60}")
    print(f"  遗忘矩阵 R (训练完域i后在域j的F1)")
    print(f"{'='*60}")
    print(f"{'':>12}", end="")
    for j in range(T):
        print(f"{domain_names[j]:>10}", end="")
    print()
    for i in range(T):
        print(f"{'After ' + domain_names[i]:>12}", end="")
        for j in range(T):
            if j <= i:
                print(f"{R[i][j]:>10.3f}", end="")
            else:
                print(f"{' ---':>10}", end="")
        print()

    print(f"\n  最终平均 F1: {final.mean():.3f}")
    print(
        f"  平均 BWT（前 {len(bwt)}/{T} 个域，final - initial，负值=遗忘）: "
        f"{bwt.mean():+.3f}"
    )
    print(
        f"  平均遗忘量（前 {len(forgetting)}/{T} 个域，-BWT，正值=遗忘）: "
        f"{forgetting.mean():+.3f}"
    )

    return {
        "forgetting_matrix": R,
        "diag_f1": diag,
        "final_f1": final,
        "bwt": bwt,
        "forgetting": forgetting,
        "bwt_num_tasks": len(bwt),
        "mean_f1": float(final.mean()),
        "mean_bwt": float(bwt.mean()),
        "mean_forgetting": float(forgetting.mean()),
    }


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def save_results(
    R: np.ndarray,
    forgetting: Dict,
    domain_names: List[str],
    output_dir: Path | str,
    metric_matrices: Optional[Dict[str, np.ndarray]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Save continual-learning metrics in machine-readable formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_csv = output_dir / "forgetting_matrix.csv"
    with matrix_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["after_domain", *domain_names])
        for i, domain in enumerate(domain_names):
            row = [f"after_{domain}"]
            for j in range(len(domain_names)):
                row.append("" if np.isnan(R[i][j]) else f"{R[i][j]:.6f}")
            writer.writerow(row)

    final_metrics_csv = output_dir / "final_f1.csv"
    with final_metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "initial_f1", "final_f1", "bwt", "forgetting"])
        for i, domain in enumerate(domain_names):
            bwt = forgetting["bwt"][i] if i < len(forgetting["bwt"]) else np.nan
            fg = forgetting["forgetting"][i] if i < len(forgetting["forgetting"]) else np.nan
            writer.writerow(
                [
                    domain,
                    f"{forgetting['diag_f1'][i]:.6f}",
                    f"{forgetting['final_f1'][i]:.6f}",
                    "" if np.isnan(bwt) else f"{bwt:.6f}",
                    "" if np.isnan(fg) else f"{fg:.6f}",
                ]
            )

    full_metrics_csv = None
    if metric_matrices:
        full_metrics_csv = output_dir / "final_metrics.csv"
        metric_names = list(metric_matrices.keys())
        with full_metrics_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["domain", *metric_names])
            for i, domain in enumerate(domain_names):
                writer.writerow(
                    [
                        domain,
                        *[
                            ""
                            if np.isnan(metric_matrices[name][-1][i])
                            else f"{metric_matrices[name][-1][i]:.6f}"
                            for name in metric_names
                        ],
                    ]
                )

    npz_path = output_dir / "continual_results.npz"
    npz_payload = {
        "forgetting_matrix": R,
        "domain_names": np.array(domain_names),
        "diag_f1": forgetting["diag_f1"],
        "final_f1": forgetting["final_f1"],
        "bwt": forgetting["bwt"],
        "forgetting": forgetting["forgetting"],
    }
    if metric_matrices:
        npz_payload.update(
            {f"metric_{name}": matrix for name, matrix in metric_matrices.items()}
        )
    if run_metadata:
        thresholds = run_metadata.get("best_threshold_by_domain", {})
        best_epochs = run_metadata.get("best_epoch_by_domain", {})
        npz_payload["best_thresholds"] = np.array(
            [thresholds.get(domain, np.nan) for domain in domain_names],
            dtype=float,
        )
        npz_payload["best_epochs"] = np.array(
            [best_epochs.get(domain, -1) for domain in domain_names],
            dtype=int,
        )
        if "oracle_forgetting_matrix" in run_metadata:
            npz_payload["oracle_forgetting_matrix"] = run_metadata[
                "oracle_forgetting_matrix"
            ]
        if "routed_forgetting_matrix" in run_metadata:
            npz_payload["routed_forgetting_matrix"] = run_metadata[
                "routed_forgetting_matrix"
            ]
        for family in ("oracle_metric_matrices", "routed_metric_matrices"):
            for name, matrix in run_metadata.get(family, {}).items():
                npz_payload[f"{family}_{name}"] = matrix
        for variant_name, matrices in run_metadata.get(
            "router_sweep_metric_matrices", {}
        ).items():
            for name, matrix in matrices.items():
                npz_payload[f"router_sweep_{variant_name}_{name}"] = matrix
    np.savez(npz_path, **npz_payload)

    final_metrics_by_domain = None
    if metric_matrices:
        final_metrics_by_domain = {
            domain: {
                name: float(matrix[-1][i])
                for name, matrix in metric_matrices.items()
                if not np.isnan(matrix[-1][i])
            }
            for i, domain in enumerate(domain_names)
        }

    summary = {
        "domains": domain_names,
        "device": str(DEVICE),
        "data_dir": DATA_DIR,
        "checkpoint_candidates": [str(path) for path in CHECKPOINT_CANDIDATES],
        "train_kwargs": TRAIN_KWARGS,
        "final_f1_by_domain": {
            domain: float(forgetting["final_f1"][i])
            for i, domain in enumerate(domain_names)
        },
        "initial_f1_by_domain": {
            domain: float(forgetting["diag_f1"][i])
            for i, domain in enumerate(domain_names)
        },
        "bwt_by_domain": {
            domain_names[i]: float(forgetting["bwt"][i])
            for i in range(len(forgetting["bwt"]))
        },
        "forgetting_by_domain": {
            domain_names[i]: float(forgetting["forgetting"][i])
            for i in range(len(forgetting["forgetting"]))
        },
        "mean_f1": forgetting["mean_f1"],
        "mean_bwt": forgetting["mean_bwt"],
        "mean_forgetting": forgetting["mean_forgetting"],
        "bwt_num_tasks": forgetting["bwt_num_tasks"],
        "bwt_note": (
            "BWT averages the first T-1 domains only; the final domain has no "
            "subsequent task on which forgetting can be measured."
        ),
    }
    if final_metrics_by_domain is not None:
        summary["final_metrics_by_domain"] = final_metrics_by_domain
    if run_metadata:
        oracle_final_metrics_by_domain = None
        routed_final_metrics_by_domain = None
        router_sweep_summary = {}
        router_sweep_final_metrics_by_variant = {}
        router_sweep_final_route_counts_by_variant = {}
        router_sweep_final_route_distribution_by_variant = {}
        oracle_metric_matrices = run_metadata.get("oracle_metric_matrices", {})
        routed_metric_matrices = run_metadata.get("routed_metric_matrices", {})
        router_sweep_metric_matrices = run_metadata.get(
            "router_sweep_metric_matrices", {}
        )
        router_sweep_route_counts = run_metadata.get(
            "router_sweep_route_counts", {}
        )
        router_sweep_route_distribution = run_metadata.get(
            "router_sweep_route_distribution", {}
        )
        final_after_key = domain_names[-1] if domain_names else ""
        if oracle_metric_matrices:
            oracle_final_metrics_by_domain = {
                domain: {
                    name: float(matrix[-1][i])
                    for name, matrix in oracle_metric_matrices.items()
                    if not np.isnan(matrix[-1][i])
                }
                for i, domain in enumerate(domain_names)
            }
        if routed_metric_matrices:
            routed_final_metrics_by_domain = {
                domain: {
                    name: float(matrix[-1][i])
                    for name, matrix in routed_metric_matrices.items()
                    if not np.isnan(matrix[-1][i])
                }
                for i, domain in enumerate(domain_names)
            }
        for variant_name, matrices in router_sweep_metric_matrices.items():
            f1_matrix = matrices.get("f1")
            if f1_matrix is not None:
                final = f1_matrix[-1, : len(domain_names)]
                diag = np.array([f1_matrix[i][i] for i in range(len(domain_names))])
                bwt = np.array(
                    [
                        f1_matrix[len(domain_names) - 1, i] - f1_matrix[i][i]
                        for i in range(len(domain_names) - 1)
                    ]
                )
                router_sweep_summary[variant_name] = {
                    "mean_f1": float(np.nanmean(final)),
                    "mean_bwt": float(np.nanmean(bwt)),
                    "mean_forgetting": float(np.nanmean(-bwt)),
                    "final_f1_by_domain": {
                        domain: float(final[i]) for i, domain in enumerate(domain_names)
                    },
                    "initial_f1_by_domain": {
                        domain: float(diag[i]) for i, domain in enumerate(domain_names)
                    },
                }
            router_sweep_final_metrics_by_variant[variant_name] = {
                domain: {
                    name: float(matrix[-1][i])
                    for name, matrix in matrices.items()
                    if not np.isnan(matrix[-1][i])
                }
                for i, domain in enumerate(domain_names)
            }
            router_sweep_final_route_counts_by_variant[variant_name] = (
                router_sweep_route_counts.get(variant_name, {}).get(final_after_key, {})
            )
            router_sweep_final_route_distribution_by_variant[variant_name] = (
                router_sweep_route_distribution.get(variant_name, {}).get(
                    final_after_key, {}
                )
            )
        primary_router_name = run_metadata.get("config", {}).get("primary_router_name", "")
        summary.update(
            {
                "experiment_config": run_metadata.get("config", {}),
                "split_info": run_metadata.get("split_info", {}),
                "best_epoch_by_domain": run_metadata.get("best_epoch_by_domain", {}),
                "best_threshold_by_domain": run_metadata.get(
                    "best_threshold_by_domain", {}
                ),
                "best_val_metrics_by_domain": run_metadata.get(
                    "best_val_metrics_by_domain", {}
                ),
                "class_weights_by_domain": run_metadata.get(
                    "class_weights_by_domain", {}
                ),
                "class_counts_by_domain": run_metadata.get(
                    "class_counts_by_domain", {}
                ),
                "router_prototypes": run_metadata.get("router_prototypes", {}),
                "router_sweep_prototypes": run_metadata.get(
                    "router_sweep_prototypes", {}
                ),
                "oracle_forgetting_matrix": run_metadata.get(
                    "oracle_forgetting_matrix"
                ),
                "routed_forgetting_matrix": run_metadata.get(
                    "routed_forgetting_matrix"
                ),
                "oracle_final_metrics_by_domain": oracle_final_metrics_by_domain,
                "routed_final_metrics_by_domain": routed_final_metrics_by_domain,
                "router_sweep_summary": router_sweep_summary,
                "router_sweep_final_metrics_by_variant": (
                    router_sweep_final_metrics_by_variant
                ),
                "router_sweep_route_counts": router_sweep_route_counts,
                "router_sweep_route_distribution": router_sweep_route_distribution,
                "router_sweep_final_route_counts_by_variant": (
                    router_sweep_final_route_counts_by_variant
                ),
                "router_sweep_final_route_distribution_by_variant": (
                    router_sweep_final_route_distribution_by_variant
                ),
                "routed_final_route_counts_by_domain": (
                    router_sweep_final_route_counts_by_variant.get(
                        primary_router_name, {}
                    )
                ),
                "routed_final_route_distribution_by_domain": (
                    router_sweep_final_route_distribution_by_variant.get(
                        primary_router_name, {}
                    )
                ),
                "router_sweep_f1_matrices": {
                    variant_name: matrices["f1"]
                    for variant_name, matrices in router_sweep_metric_matrices.items()
                    if "f1" in matrices
                },
            }
        )
    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({k: _json_ready(v) for k, v in summary.items()}, f, indent=2)

    print(f"\n  [保存] 遗忘矩阵: {matrix_csv}")
    print(f"  [保存] 最终 F1/BWT: {final_metrics_csv}")
    if full_metrics_csv is not None:
        print(f"  [保存] 最终完整指标: {full_metrics_csv}")
    print(f"  [保存] 结构化结果: {npz_path}")
    print(f"  [保存] 汇总 JSON: {summary_path}")
    paths = {
        "matrix_csv": matrix_csv,
        "final_metrics_csv": final_metrics_csv,
        "npz": npz_path,
        "summary_json": summary_path,
    }
    if full_metrics_csv is not None:
        paths["full_metrics_csv"] = full_metrics_csv
    return paths


# ══════════════════════════════════════════════════════════════════
#  入口（GPU 可用时执行，当前 CPU 模式跳过训练）
# ══════════════════════════════════════════════════════════════════

def main():
    set_global_seed(SEED)
    print(f"  设备: {DEVICE}")
    print(f"  域顺序: {' → '.join(DOMAIN_NAMES)}")
    print(f"  Seed: {SEED}")
    print(f"  超参: epochs={TRAIN_KWARGS['epochs']}, lr={TRAIN_KWARGS['lr']}, "
          f"batch={TRAIN_KWARGS['batch_size']}")

    if DRY_RUN or DEVICE.type != "cuda":
        reason = "dry-run 模式" if DRY_RUN else "当前无 GPU 或已强制 CPU"
        print(f"\n  [Skip] {reason}，跳过训练。")
        print("    模型结构验证通过，等待 GPU 实例启动后执行。")
        print("    可用 ECG_CL_DRY_RUN=1 做入口检查，ECG_FORCE_CPU=1 强制无卡模式。")
        print(f"    启动命令: python -m trainer.continual_cl")
        return

    # ── GPU 可用时执行以下流程 ──────────────────────────────────
    model = build_model()

    train_loaders, val_loaders, test_loaders, split_infos, train_labels = (
        get_cinc_train_val_test_loaders(
            data_dir=DATA_DIR,
            domains=DOMAIN_NAMES,
            seed=SEED,
            val_ratio=VAL_RATIO,
        )
    )

    R, metric_matrices, run_metadata = run_continual_learning(
        model,
        train_loaders,
        test_loaders,
        val_loaders=val_loaders,
        split_infos=split_infos,
        train_labels_by_domain=train_labels,
        return_metrics=True,
    )
    forgetting = compute_forgetting(R, DOMAIN_NAMES)
    save_results(
        R,
        forgetting,
        DOMAIN_NAMES,
        OUTPUT_DIR,
        metric_matrices,
        run_metadata=run_metadata,
    )

    model.remove_hook()


if __name__ == "__main__":
    main()
