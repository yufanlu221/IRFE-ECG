"""
CinC2021 持续学习数据集加载器
=============================
从预处理好的 .pt 文件中加载各数据域的 ECG 信号与标签，
为持续学习训练循环提供按域索引的 DataLoader。

数据格式（.pt 文件）:
  {"x": (N, 1, L) float32 tensor  — 单导联 ECG 信号
   "y": (N,)    int64   tensor  — 标签 (0=Normal, 1=Abnormal)}

注意:
  当前缓存按 Normal/Abnormal 二分类读取：只有 Dx 集合等于正常窦性心律
  SNOMED code 426783006 时为 0，其余任意非正常诊断代码为 1。

用法:
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
#  域配置
# ══════════════════════════════════════════════════════════════════

# 持续学习任务顺序（域标识符，与 continual_cl.py 对齐）
DOMAIN_NAMES: List[str] = [
    "cpsc", "ptbxl", "georgia", "chapman", "ptb", "ningbo"
]

# 域标识符 → .pt 文件名前缀
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
#  内部工具
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
    """加载单个 .pt 文件为 TensorDataset。"""
    path = Path(pt_path)
    with path.open("rb") as f:
        data = torch.load(f, map_location="cpu", weights_only=False)
    _validate_label_metadata(data, path, strict=strict_label_metadata)
    x = data["x"].float()
    y = data["y"].long()
    return TensorDataset(x, y)


# ══════════════════════════════════════════════════════════════════
#  公开 API
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
    为每个域创建 train / test DataLoader。

    Args:
        data_dir:   存放 .pt 文件的目录路径
        domains:    要加载的域列表，默认 DOMAIN_NAMES 全部 6 个
        batch_size: batch 大小
        num_workers:DataLoader worker 数（Windows 下须设为 0）
        pin_memory: 是否 pin_memory（GPU 训练建议开启）

    Returns:
        (train_loaders, test_loaders)
          - train_loaders[domain] → DataLoader (shuffle=True)
          - test_loaders[domain]  → DataLoader (shuffle=False)

    Raises:
        FileNotFoundError: .pt 文件缺失时抛出
    """
    data_path = Path(data_dir)
    domains = domains if domains is not None else DOMAIN_NAMES

    train_loaders: Dict[str, DataLoader] = {}
    test_loaders: Dict[str, DataLoader] = {}

    for domain_index, domain in enumerate(domains):
        if domain not in DOMAIN_TO_FILE:
            raise ValueError(f"未知域: {domain}，可选: {list(DOMAIN_TO_FILE)}")
        file_prefix = DOMAIN_TO_FILE[domain]

        train_pt = data_path / f"{file_prefix}_train.pt"
        test_pt  = data_path / f"{file_prefix}_test.pt"

        if not train_pt.exists():
            raise FileNotFoundError(f"训练数据缺失: {train_pt}")
        if not test_pt.exists():
            raise FileNotFoundError(f"测试数据缺失: {test_pt}")

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
    加载单个域的单个 DataLoader，便于单域调试。

    Args:
        data_dir:   .pt 文件目录
        domain:     域标识符，如 "cpsc"
        mode:       "train" 或 "test"
        batch_size: batch 大小
        **kwargs:   透传给 DataLoader 的额外参数

    Returns:
        DataLoader 实例
    """
    data_path = Path(data_dir)
    if domain not in DOMAIN_TO_FILE:
        raise ValueError(f"未知域: {domain}，可选: {list(DOMAIN_TO_FILE)}")
    file_prefix = DOMAIN_TO_FILE[domain]
    pt_path = data_path / f"{file_prefix}_{mode}.pt"

    if not pt_path.exists():
        raise FileNotFoundError(f"数据文件缺失: {pt_path}")

    ds = _load_single_pt(str(pt_path), strict_label_metadata)
    shuffle = (mode == "train")
    num_workers = kwargs.pop("num_workers", 0)
    generator = kwargs.pop("generator", None)
    if generator is None and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=generator, **kwargs)
