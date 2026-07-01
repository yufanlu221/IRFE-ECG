"""Audit CinC2021 binary label construction.

This script checks both raw CinC2021 header files and cached ``.pt`` tensors.
It is designed to answer one paper-critical question: are the experiments using
AF/non-AF labels or Normal/Abnormal labels?

Example on the server:

    python -m scripts.audit_cinc_labels \
        --raw-root data/raw \
        --pt-root data/processed \
        --domains cpsc ptbxl georgia chapman
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from data.loaders.cinc_dataset import (
    DOMAIN_TO_FILE,
    LABEL_DESCRIPTION,
    LABEL_TASK,
)

AF_CODE = "164889003"
NORMAL_CODE = "426783006"


def parse_dx_codes(hea_path: Path) -> List[str]:
    """Parse SNOMED-CT Dx codes from a CinC ``.hea`` file."""
    try:
        with hea_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("#dx") or line.lower().startswith("# dx"):
                    _, _, value = line.partition(":")
                    return [code.strip() for code in value.split(",") if code.strip()]
    except OSError:
        return []
    return []


def classify_codes(codes: List[str]) -> Dict[str, bool]:
    """Return AF/non-AF and Normal/Abnormal interpretations for one record."""
    code_set = set(codes)
    has_af = AF_CODE in code_set
    has_normal = NORMAL_CODE in code_set
    has_non_normal = bool(code_set - {NORMAL_CODE})
    normal_only = bool(code_set) and has_normal and not has_non_normal

    return {
        "af_positive": has_af,
        "non_af": not has_af,
        "normal_only": normal_only,
        "abnormal_by_non_normal_code": has_non_normal,
        "normal_with_other_codes": has_normal and has_non_normal,
        "missing_dx": not bool(code_set),
    }


def find_raw_domain_dir(raw_root: Path, domain: str) -> Path | None:
    """Find raw domain directory for either task folder names or domain names."""
    prefix = DOMAIN_TO_FILE[domain]
    candidates = [
        raw_root / prefix,
        raw_root / domain,
        raw_root / domain.upper(),
        raw_root / prefix.replace("_", "-"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = [
        path for path in raw_root.iterdir()
        if path.is_dir() and (
            path.name.lower() == domain.lower()
            or path.name.lower().startswith(prefix.lower())
        )
    ] if raw_root.exists() else []
    return matches[0] if matches else None


def audit_raw_domain(raw_root: Path, domain: str) -> Dict:
    raw_dir = find_raw_domain_dir(raw_root, domain)
    if raw_dir is None:
        return {
            "domain": domain,
            "raw_dir": None,
            "raw_found": False,
            "raw_total": 0,
        }

    hea_files = sorted(raw_dir.rglob("*.hea"))
    code_counter: Counter[str] = Counter()
    stats: Counter[str] = Counter()

    for hea_path in hea_files:
        codes = parse_dx_codes(hea_path)
        code_counter.update(codes)
        flags = classify_codes(codes)
        for key, value in flags.items():
            stats[key] += int(value)

    total = len(hea_files)
    return {
        "domain": domain,
        "raw_dir": str(raw_dir),
        "raw_found": True,
        "raw_total": total,
        "raw_af_positive": stats["af_positive"],
        "raw_non_af": stats["non_af"],
        "raw_af_ratio": stats["af_positive"] / total if total else None,
        "raw_normal_only": stats["normal_only"],
        "raw_abnormal_by_non_normal_code": stats["abnormal_by_non_normal_code"],
        "raw_normal_with_other_codes": stats["normal_with_other_codes"],
        "raw_missing_dx": stats["missing_dx"],
        "raw_top_codes": code_counter.most_common(10),
    }


def torch_load(path: Path) -> Dict:
    with path.open("rb") as f:
        return torch.load(f, map_location="cpu")


def audit_pt_file(path: Path) -> Dict:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "total": 0,
        }

    data = torch_load(path)
    y = data["y"].long().cpu()
    total = int(y.numel())
    positives = int((y == 1).sum().item())
    negatives = int((y == 0).sum().item())
    other = total - positives - negatives

    label_task = data.get("label_task")
    label_description = data.get("label_description")

    return {
        "path": str(path),
        "exists": True,
        "total": total,
        "negative_0": negatives,
        "positive_1": positives,
        "other_labels": other,
        "positive_ratio": positives / total if total else None,
        "label_task": label_task,
        "label_description": label_description,
        "metadata_ok": (
            label_task == LABEL_TASK and label_description == LABEL_DESCRIPTION
        ),
        "missing_metadata": label_task is None and label_description is None,
    }


def audit_pt_domain(pt_root: Path, domain: str) -> Dict:
    prefix = DOMAIN_TO_FILE[domain]
    train = audit_pt_file(pt_root / f"{prefix}_train.pt")
    test = audit_pt_file(pt_root / f"{prefix}_test.pt")
    total = train["total"] + test["total"]
    positives = train.get("positive_1", 0) + test.get("positive_1", 0)
    negatives = train.get("negative_0", 0) + test.get("negative_0", 0)

    return {
        "domain": domain,
        "pt_train": train,
        "pt_test": test,
        "pt_total": total,
        "pt_negative_0": negatives,
        "pt_positive_1": positives,
        "pt_positive_ratio": positives / total if total else None,
        "pt_metadata_ok": train.get("metadata_ok", False) and test.get("metadata_ok", False),
        "pt_missing_metadata": train.get("missing_metadata", False) or test.get("missing_metadata", False),
    }


def add_label_match_inference(row: Dict) -> None:
    """Infer whether cached 1-labels look closer to AF or abnormal labels."""
    if not row.get("raw_found") or not row.get("pt_total"):
        row["pt_label_match_guess"] = "unknown"
        return

    positive_1 = row["pt_positive_1"]
    negative_0 = row["pt_negative_0"]
    raw_af = row["raw_af_positive"]
    raw_abnormal = row["raw_abnormal_by_non_normal_code"]
    raw_normal = row["raw_normal_only"]

    row["pt_positive_1_minus_raw_af"] = positive_1 - raw_af
    row["pt_positive_1_minus_raw_abnormal"] = positive_1 - raw_abnormal
    row["pt_negative_0_minus_raw_normal"] = negative_0 - raw_normal

    af_error = abs(positive_1 - raw_af)
    abnormal_error = abs(positive_1 - raw_abnormal) + abs(negative_0 - raw_normal)
    row["pt_label_match_guess"] = (
        "normal_abnormal" if abnormal_error < af_error else "af_non_af"
    )


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def write_csv(rows: List[Dict], path: Path) -> None:
    fields = [
        "domain",
        "raw_total",
        "raw_af_positive",
        "raw_af_ratio",
        "raw_normal_only",
        "raw_abnormal_by_non_normal_code",
        "raw_normal_with_other_codes",
        "raw_missing_dx",
        "pt_total",
        "pt_positive_1",
        "pt_positive_ratio",
        "pt_label_match_guess",
        "pt_positive_1_minus_raw_af",
        "pt_positive_1_minus_raw_abnormal",
        "pt_negative_0_minus_raw_normal",
        "pt_metadata_ok",
        "pt_missing_metadata",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--pt-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--domains", nargs="+", default=["cpsc", "ptbxl", "georgia", "chapman"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/label_audit"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    detail: Dict[str, Dict] = {
        "label_task": LABEL_TASK,
        "label_description": LABEL_DESCRIPTION,
        "af_code": AF_CODE,
        "normal_code": NORMAL_CODE,
        "raw_root": str(args.raw_root),
        "pt_root": str(args.pt_root),
        "domains": {},
    }

    print("\nCinC2021 Label Audit")
    print("=" * 80)
    print(f"Label task: {LABEL_TASK}")
    print(f"Label description: {LABEL_DESCRIPTION}")
    print(f"AF code: {AF_CODE} | Normal code: {NORMAL_CODE}")
    print("=" * 80)

    for domain in args.domains:
        if domain not in DOMAIN_TO_FILE:
            raise ValueError(f"Unknown domain '{domain}'. Choices: {list(DOMAIN_TO_FILE)}")

        raw = audit_raw_domain(args.raw_root, domain)
        pt = audit_pt_domain(args.pt_root, domain)
        row = {**raw, **pt}
        add_label_match_inference(row)
        rows.append(row)
        detail["domains"][domain] = row

        print(f"\n[{domain}]")
        if raw.get("raw_found"):
            print(
                f"  raw: total={raw['raw_total']}  "
                f"AF={raw['raw_af_positive']} ({pct(raw['raw_af_ratio'])})  "
                f"normal_only={raw['raw_normal_only']}  "
                f"normal+other={raw['raw_normal_with_other_codes']}  "
                f"missing_dx={raw['raw_missing_dx']}"
            )
            print(f"  raw top codes: {raw['raw_top_codes'][:5]}")
        else:
            print("  raw: not found")

        print(
            f"  pt : total={pt['pt_total']}  "
            f"positive_1={pt['pt_positive_1']} ({pct(pt['pt_positive_ratio'])})  "
            f"metadata_ok={pt['pt_metadata_ok']}  "
            f"missing_metadata={pt['pt_missing_metadata']}"
        )
        print(
            f"       label guess: {row['pt_label_match_guess']}  "
            f"pos1-rawAF={row.get('pt_positive_1_minus_raw_af')}  "
            f"pos1-rawAbnormal={row.get('pt_positive_1_minus_raw_abnormal')}  "
            f"neg0-rawNormal={row.get('pt_negative_0_minus_raw_normal')}"
        )
        for split in ("pt_train", "pt_test"):
            item = pt[split]
            print(
                f"       {split[3:]}: exists={item['exists']} total={item['total']} "
                f"pos={item.get('positive_1', 0)} neg={item.get('negative_0', 0)} "
                f"label_task={item.get('label_task')!r}"
            )

    csv_path = args.out_dir / "cinc_label_audit_summary.csv"
    json_path = args.out_dir / "cinc_label_audit_detail.json"
    write_csv(rows, csv_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
