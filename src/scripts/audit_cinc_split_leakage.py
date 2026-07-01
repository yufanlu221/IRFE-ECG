"""Audit processed CinC .pt train/test splits for exact waveform leakage."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


DOMAIN_TO_FILE = {
    "cpsc": "Task1_CPSC",
    "ptbxl": "Task2_PTBXL",
    "georgia": "Task3_Georgia",
    "chapman": "Task4_Chapman",
}
DEFAULT_DOMAINS = ["cpsc", "ptbxl", "georgia", "chapman"]


def row_digest(row: torch.Tensor) -> str:
    arr = row.detach().cpu().contiguous().numpy()
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def load_split(data_dir: Path, prefix: str, split: str) -> dict:
    path = data_dir / f"{prefix}_{split}.pt"
    data = torch.load(path, map_location="cpu")
    x = data["x"].float()
    y = data["y"].long()
    counts = torch.bincount(y, minlength=2).tolist()
    return {
        "path": path,
        "x": x,
        "y": y,
        "hashes": {row_digest(row) for row in x},
        "counts": counts,
        "label_task": data.get("label_task"),
        "label_description": data.get("label_description"),
        "clean_split": data.get("clean_split"),
    }


def audit_domain(data_dir: Path, domain: str) -> dict:
    prefix = DOMAIN_TO_FILE[domain]
    train = load_split(data_dir, prefix, "train")
    test = load_split(data_dir, prefix, "test")
    overlap = train["hashes"] & test["hashes"]
    return {
        "domain": domain,
        "prefix": prefix,
        "train_samples": int(train["x"].shape[0]),
        "test_samples": int(test["x"].shape[0]),
        "train_labels": {"normal_0": train["counts"][0], "abnormal_1": train["counts"][1]},
        "test_labels": {"normal_0": test["counts"][0], "abnormal_1": test["counts"][1]},
        "train_positive_rate": train["counts"][1] / max(sum(train["counts"]), 1),
        "test_positive_rate": test["counts"][1] / max(sum(test["counts"]), 1),
        "exact_train_test_overlap": len(overlap),
        "label_task": train["label_task"],
        "label_description": train["label_description"],
        "clean_split": train["clean_split"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--fail-on-overlap", action="store_true")
    args = parser.parse_args()

    total_overlap = 0
    for domain in args.domains:
        report = audit_domain(args.data_dir, domain)
        total_overlap += report["exact_train_test_overlap"]
        print(
            f"{domain}: train={report['train_samples']} "
            f"test={report['test_samples']} "
            f"train_pos={report['train_positive_rate']:.3f} "
            f"test_pos={report['test_positive_rate']:.3f} "
            f"overlap={report['exact_train_test_overlap']} "
            f"clean={report['clean_split']}"
        )

    print(f"total_exact_train_test_overlap={total_overlap}")
    if args.fail_on_overlap and total_overlap:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
