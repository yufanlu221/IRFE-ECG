"""Patch label metadata into existing CinC2021 ``.pt`` caches.

The current server caches were generated before label metadata existed.  This
script adds the Normal/Abnormal task description without touching signal or
label tensors.

Dry-run first:

    python -m scripts.patch_cinc_metadata \
        --data-dir data/processed

Apply in place:

    python -m scripts.patch_cinc_metadata \
        --data-dir data/processed \
        --write
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from data.loaders.cinc_dataset import (
    DOMAIN_TO_FILE,
    LABEL_DESCRIPTION,
    LABEL_TASK,
)


def torch_load(path: Path) -> Dict:
    with path.open("rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)


def validate_binary_labels(data: Dict, path: Path) -> None:
    if "y" not in data:
        raise ValueError(f"{path} has no 'y' label tensor.")

    y = data["y"].long().view(-1)
    bad = y[(y != 0) & (y != 1)]
    if bad.numel():
        unique_bad = sorted({int(v) for v in bad[:20].tolist()})
        raise ValueError(f"{path} contains labels outside {{0,1}}: {unique_bad}")


def patch_file(path: Path, write: bool, force: bool) -> Dict:
    if not path.exists():
        return {"path": str(path), "status": "missing"}

    data = torch_load(path)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is {type(data).__name__}, expected dict.")

    validate_binary_labels(data, path)

    old_task = data.get("label_task")
    old_description = data.get("label_description")
    matches = old_task == LABEL_TASK and old_description == LABEL_DESCRIPTION
    missing = old_task is None and old_description is None

    if matches:
        status = "ok"
    elif missing:
        status = "would_patch"
    elif force:
        status = "would_overwrite"
    else:
        status = "metadata_mismatch"

    if write and status in {"would_patch", "would_overwrite"}:
        data["label_task"] = LABEL_TASK
        data["label_description"] = LABEL_DESCRIPTION
        torch.save(data, path)
        status = "patched" if status == "would_patch" else "overwritten"

    y = data["y"].long().view(-1)
    return {
        "path": str(path),
        "status": status,
        "n": int(y.numel()),
        "label_0": int((y == 0).sum().item()),
        "label_1": int((y == 1).sum().item()),
        "old_label_task": old_task,
        "old_label_description": old_description,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--domains", nargs="+", default=["cpsc", "ptbxl", "georgia", "chapman"])
    parser.add_argument("--write", action="store_true", help="Actually rewrite .pt files in place.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite non-matching existing metadata. Use only after auditing labels.",
    )
    args = parser.parse_args()

    print("CinC metadata patcher")
    print(f"data_dir: {args.data_dir}")
    print(f"target label_task: {LABEL_TASK}")
    print(f"target label_description: {LABEL_DESCRIPTION}")
    print(f"mode: {'WRITE' if args.write else 'DRY-RUN'}")
    print()

    rows: List[Dict] = []
    for domain in args.domains:
        if domain not in DOMAIN_TO_FILE:
            raise ValueError(f"Unknown domain '{domain}'. Choices: {list(DOMAIN_TO_FILE)}")
        prefix = DOMAIN_TO_FILE[domain]
        for split in ("train", "test"):
            rows.append(patch_file(args.data_dir / f"{prefix}_{split}.pt", args.write, args.force))

    for row in rows:
        print(
            f"{Path(row['path']).name:28s} {row['status']:18s} "
            f"n={row.get('n', 0):6d} label0={row.get('label_0', 0):6d} "
            f"label1={row.get('label_1', 0):6d}"
        )

    bad = [row for row in rows if row["status"] == "metadata_mismatch"]
    if bad:
        print("\nMetadata mismatch found. Rerun with --force only if the audit confirms the task.")
        raise SystemExit(1)

    if not args.write:
        print("\nDry-run only. Add --write to patch files in place.")


if __name__ == "__main__":
    main()
