"""Rebuild CinC processed .pt splits with exact waveform de-duplication.

The existing caches may contain identical waveforms in both train and test.
This script merges the old split, groups identical ECG tensors by hash, and
assigns each hash group wholly to either train or test. It cannot prove
patient-level separation, but it removes exact waveform leakage and records
split metadata for paper experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


DOMAIN_TO_FILE = {
    "cpsc": "Task1_CPSC",
    "ptbxl": "Task2_PTBXL",
    "georgia": "Task3_Georgia",
    "chapman": "Task4_Chapman",
}
DEFAULT_DOMAINS = ["cpsc", "ptbxl", "georgia", "chapman"]
LABEL_TASK = "normal_abnormal"
LABEL_DESCRIPTION = "binary ECG abnormality screening: 0=Normal, 1=Abnormal"


@dataclass
class HashGroup:
    digest: str
    indices: list[int]
    label: int

    @property
    def size(self) -> int:
        return len(self.indices)


def row_digest(row: torch.Tensor) -> str:
    arr = row.detach().cpu().contiguous().numpy()
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def load_domain(data_dir: Path, prefix: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    pieces = []
    labels = []
    meta = {}
    for split in ("train", "test"):
        path = data_dir / f"{prefix}_{split}.pt"
        data = torch.load(path, map_location="cpu")
        pieces.append(data["x"].float())
        labels.append(data["y"].long())
        if not meta:
            meta = {
                "source_label_task": data.get("label_task"),
                "source_label_description": data.get("label_description"),
            }
    return torch.cat(pieces, dim=0), torch.cat(labels, dim=0), meta


def build_hash_groups(
    x: torch.Tensor,
    y: torch.Tensor,
    conflict_policy: str,
) -> tuple[list[HashGroup], list[dict]]:
    by_hash: dict[str, list[int]] = {}
    for idx, row in enumerate(x):
        by_hash.setdefault(row_digest(row), []).append(idx)

    groups = []
    conflicts = []
    for digest, indices in by_hash.items():
        labels = [int(y[idx].item()) for idx in indices]
        unique = sorted(set(labels))
        if len(unique) > 1:
            conflicts.append(
                {
                    "hash": digest,
                    "indices": indices,
                    "labels": labels,
                    "label_counts": {str(label): labels.count(label) for label in unique},
                }
            )
            if conflict_policy == "error":
                raise ValueError(f"Conflicting labels for hash {digest}: {labels}")
            if conflict_policy == "drop":
                continue
            if conflict_policy == "majority":
                label = max(unique, key=labels.count)
            else:
                raise ValueError(f"Unknown conflict policy: {conflict_policy}")
        else:
            label = unique[0]
        groups.append(HashGroup(digest=digest, indices=indices, label=label))
    return groups, conflicts


def split_groups(
    groups: Iterable[HashGroup],
    train_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label in (0, 1):
        label_groups = [group for group in groups if group.label == label]
        rng.shuffle(label_groups)
        total = sum(group.size for group in label_groups)
        target_train = round(total * train_ratio)
        train_count = 0
        for group in label_groups:
            if train_count < target_train:
                train_indices.extend(group.indices)
                train_count += group.size
            else:
                test_indices.extend(group.indices)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def label_counts(y: torch.Tensor) -> dict[str, int]:
    counts = torch.bincount(y.cpu(), minlength=2).tolist()
    return {"normal_0": int(counts[0]), "abnormal_1": int(counts[1])}


def overlap_count(x_train: torch.Tensor, x_test: torch.Tensor) -> int:
    train_hashes = {row_digest(row) for row in x_train}
    return sum(1 for row in x_test if row_digest(row) in train_hashes)


def save_split(
    output_dir: Path,
    prefix: str,
    x: torch.Tensor,
    y: torch.Tensor,
    train_indices: list[int],
    test_indices: list[int],
    metadata: dict,
) -> dict:
    train_idx = torch.tensor(train_indices, dtype=torch.long)
    test_idx = torch.tensor(test_indices, dtype=torch.long)
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    train_hashes = [row_digest(row) for row in x_train]
    test_hashes = [row_digest(row) for row in x_test]
    split_meta = {
        **metadata,
        "label_task": LABEL_TASK,
        "label_description": LABEL_DESCRIPTION,
        "waveform_hash_algo": "blake2b-128",
    }

    torch.save(
        {"x": x_train, "y": y_train, "waveform_hash": train_hashes, **split_meta},
        output_dir / f"{prefix}_train.pt",
    )
    torch.save(
        {"x": x_test, "y": y_test, "waveform_hash": test_hashes, **split_meta},
        output_dir / f"{prefix}_test.pt",
    )

    return {
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_labels": label_counts(y_train),
        "test_labels": label_counts(y_test),
        "exact_train_test_overlap": overlap_count(x_train, x_test),
    }


def rebuild_domain(
    domain: str,
    data_dir: Path,
    output_dir: Path,
    train_ratio: float,
    seed: int,
    conflict_policy: str,
) -> dict:
    prefix = DOMAIN_TO_FILE[domain]
    x, y, source_meta = load_domain(data_dir, prefix)
    groups, conflicts = build_hash_groups(x, y, conflict_policy=conflict_policy)
    train_indices, test_indices = split_groups(groups, train_ratio=train_ratio, seed=seed)

    kept = len(train_indices) + len(test_indices)
    metadata = {
        **source_meta,
        "clean_split": True,
        "split_method": "exact_waveform_hash_grouped_stratified_split",
        "split_seed": int(seed),
        "train_ratio": float(train_ratio),
        "conflict_policy": conflict_policy,
        "source_total_samples": int(len(y)),
        "kept_samples": int(kept),
        "dropped_conflict_samples": int(len(y) - kept),
        "hash_groups": int(len(groups)),
        "conflicting_hash_groups": int(len(conflicts)),
    }
    saved = save_split(output_dir, prefix, x, y, train_indices, test_indices, metadata)
    return {
        "domain": domain,
        "prefix": prefix,
        **metadata,
        **saved,
        "conflicts": conflicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conflict-policy",
        choices=["drop", "majority", "error"],
        default="drop",
        help="How to handle identical waveforms with conflicting labels.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for domain in args.domains:
        if domain not in DOMAIN_TO_FILE:
            raise ValueError(f"Unknown domain: {domain}")
        report = rebuild_domain(
            domain=domain,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            conflict_policy=args.conflict_policy,
        )
        reports.append(report)
        print(
            f"{domain}: train={report['train_samples']} test={report['test_samples']} "
            f"overlap={report['exact_train_test_overlap']} "
            f"conflict_groups={report['conflicting_hash_groups']} "
            f"dropped={report['dropped_conflict_samples']}"
        )

    summary = {
        "source_data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "domains": args.domains,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "conflict_policy": args.conflict_policy,
        "reports": reports,
    }
    manifest_path = args.output_dir / "clean_split_manifest.json"
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
