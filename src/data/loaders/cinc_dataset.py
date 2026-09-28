"""
CinC2021 continual learning dataset loader
=========================================
Load ECG signals and labels for each domain from preprocessed .pt files,
providing domain-indexed DataLoaders for continual learning training loops.

Data format (.pt files):
  {"x": (N, 1, L) float32 tensor  — single-lead ECG signals
   "y": (N,)    int64   tensor  — labels (0=Normal, 1=Abnormal)}

Note:
  Current caches use binary Normal/Abnormal labels: 0 only when the Dx set
  contains solely the normal sinus rhythm SNOMED code 426783006; any
  non-normal diagnosis code results in label 1.

Usage:
  from data.loaders.cinc_dataset import get_cinc_dataloaders

  train_loaders, test_loaders = get_cinc_dataloaders(
      data_dir="data/processed",
      domains=["cpsc", "ptbxl", "georgia", "chapman"],
      batch_size=64,
  )
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import TensorDataset, DataLoader


# ══════════════════════════════════════════════════════════════════
#  Domain configuration
# ══════════════════════════════════════════════════════════════════

# Continual learning task order (domain identifiers aligned with continual_cl.py)
DOMAIN_NAMES: List[str] = [
    "cpsc", "ptbxl", "georgia", "chapman", "ptb", "ningbo"
]

# Domain identifier → .pt filename prefix
DOMAIN_TO_FILE: Dict[str, str] = {
    "cpsc":    "Task1_CPSC",
    "ptbxl":   "Task2_PTBXL",
    "georgia": "Task3_Georgia",
    "chapman": "Task4_Chapman",
    "ptb":     "Task5_PTB",
    "ningbo":  "Task6_Ningbo",
}

# Current CinC2021 caches are Normal/Abnormal:
#   0 = pure normal rhythm records, 1 = any non-normal diagnosis code.
LABEL_TASK = "normal_abnormal"
NEGATIVE_LABEL = "Normal"
POSITIVE_LABEL = "Abnormal"
LABEL_DESCRIPTION = "binary ECG abnormality screening: 0=Normal, 1=Abnormal"
STRICT_LABEL_METADATA_DEFAULT = (
    os.environ.get("ECG_STRICT_LABEL_METADATA", "1") == "1"
)


# ══════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════

def _validate_label_metadata(data: dict, pt_path: Path, strict: bool) -> None:
    label_task = data.get("label_task")
    label_description = data.get("label_description")

    if label_task is not None and label_task != LABEL_TASK:
        raise ValueError(
            f"{pt_path} label_task={label_task!r}, expected {LABEL_TASK!r}."
        )
    if label_description is not None and label_description != LABEL_DESCRIPTION:
        raise ValueError(
            f"{pt_path} label_description={label_description!r}, "
            f"expected {LABEL_DESCRIPTION!r}."
        )
    if label_task is None and label_description is None:
        if strict:
            raise ValueError(
                f"{pt_path} has no label metadata. Regenerate .pt files with "
                f"label_task={LABEL_TASK!r} before paper experiments."
            )
        warnings.warn(
            f"{pt_path} has no label metadata; assuming {LABEL_DESCRIPTION}.",
            RuntimeWarning,
            stacklevel=2,
        )


def _load_single_pt(
    pt_path: str,
    strict_label_metadata: bool = STRICT_LABEL_METADATA_DEFAULT,
) -> TensorDataset:
    """Load a single .pt file as a TensorDataset."""
    path = Path(pt_path)
    with path.open("rb") as f:
        data = torch.load(f, map_location="cpu", weights_only=False)
    _validate_label_metadata(data, path, strict=strict_label_metadata)
    x = data["x"].float()
    y = data["y"].long()
    return TensorDataset(x, y)


# ══════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════

def get_cinc_dataloaders(
    data_dir: str,
    domains: Optional[List[str]] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    strict_label_metadata: bool = STRICT_LABEL_METADATA_DEFAULT,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """
    Create training and test DataLoaders for each domain.

    Args:
        data_dir: Directory containing the .pt files.
        domains: Domains to load; defaults to all six entries in DOMAIN_NAMES.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers (set to 0 on Windows).
        pin_memory: Whether to pin memory (recommended for GPU training).

    Returns:
        (train_loaders, test_loaders)
          - train_loaders[domain] → DataLoader (shuffle=True)
          - test_loaders[domain]  → DataLoader (shuffle=False)

    Raises:
        FileNotFoundError: If a required .pt file is missing.
    """
    data_path = Path(data_dir)
    domains = domains if domains is not None else DOMAIN_NAMES

    train_loaders: Dict[str, DataLoader] = {}
    test_loaders: Dict[str, DataLoader] = {}

    for domain_index, domain in enumerate(domains):
        if domain not in DOMAIN_TO_FILE:
            raise ValueError(f"Unknown domain: {domain}, available domains: {list(DOMAIN_TO_FILE)}")
        file_prefix = DOMAIN_TO_FILE[domain]

        train_pt = data_path / f"{file_prefix}_train.pt"
        test_pt  = data_path / f"{file_prefix}_test.pt"

        if not train_pt.exists():
            raise FileNotFoundError(f"Missing training data: {train_pt}")
        if not test_pt.exists():
            raise FileNotFoundError(f"Missing test data: {test_pt}")

        train_ds = _load_single_pt(str(train_pt), strict_label_metadata)
        test_ds  = _load_single_pt(str(test_pt), strict_label_metadata)

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed + domain_index)

        train_loaders[domain] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loaders[domain] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loaders, test_loaders


def get_single_loader(
    data_dir: str,
    domain: str,
    mode: str = "train",
    batch_size: int = 64,
    strict_label_metadata: bool = STRICT_LABEL_METADATA_DEFAULT,
    seed: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for a single domain to simplify domain-specific debugging.

    Args:
        data_dir: Directory containing the .pt files.
        domain: Domain identifier, such as "cpsc".
        mode: "train" or "test".
        batch_size: Batch size.
        **kwargs: Additional arguments passed to DataLoader.

    Returns:
        A DataLoader instance.
    """
    data_path = Path(data_dir)
    if domain not in DOMAIN_TO_FILE:
        raise ValueError(f"Unknown domain: {domain}, available domains: {list(DOMAIN_TO_FILE)}")
    file_prefix = DOMAIN_TO_FILE[domain]
    pt_path = data_path / f"{file_prefix}_{mode}.pt"

    if not pt_path.exists():
        raise FileNotFoundError(f"Missing data file: {pt_path}")

    ds = _load_single_pt(str(pt_path), strict_label_metadata)
    shuffle = (mode == "train")
    num_workers = kwargs.pop("num_workers", 0)
    generator = kwargs.pop("generator", None)
    if generator is None and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=generator, **kwargs)
