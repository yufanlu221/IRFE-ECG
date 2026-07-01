"""Build paper-ready CinC2021 Normal/Abnormal .pt files from raw WFDB data.

This script preserves record metadata that the old cached tensors lacked. It is
intended for BIBM experiments where the split protocol must be auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io
import scipy.signal
import torch


NORMAL_CODE = "426783006"
LABEL_TASK = "normal_abnormal"
LABEL_DESCRIPTION = "binary ECG abnormality screening: 0=Normal, 1=Abnormal"

DOMAIN_CONFIG = {
    "cpsc": ("cpsc_2018", "Task1_CPSC"),
    "ptbxl": ("ptb-xl", "Task2_PTBXL"),
    "georgia": ("georgia", "Task3_Georgia"),
    "chapman": ("chapman_shaoxing", "Task4_Chapman"),
    "ptb": ("ptb", "Task5_PTB"),
    "ningbo": ("ningbo", "Task6_Ningbo"),
    "cpsc_extra": ("cpsc_2018_extra", "Task7_CPSCExtra"),
}
DEFAULT_DOMAINS = ["cpsc", "ptbxl", "georgia", "chapman"]


@dataclass
class HeaderMeta:
    record_id: str
    num_leads: int
    fs: int
    n_samples: int
    lead_names: list[str]
    age: str | None
    sex: str | None
    dx_codes: list[str]
    patient_id: str | None


@dataclass
class Record:
    x: np.ndarray
    y: int
    record_id: str
    source_database: str
    source_path: str
    dx_codes: list[str]
    age: str | None
    sex: str | None
    fs: int
    n_samples: int
    group_id: str
    group_kind: str
    waveform_hash: str


def parse_header(path: Path) -> HeaderMeta:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"empty header: {path}")

    first = lines[0].split()
    record_id = first[0]
    num_leads = int(first[1])
    fs = int(float(first[2]))
    n_samples = int(first[3])

    lead_names: list[str] = []
    for line in lines[1 : 1 + num_leads]:
        parts = line.split()
        if parts:
            lead_names.append(parts[-1])

    fields: dict[str, str] = {}
    for line in lines[1 + num_leads :]:
        if not line.startswith("#"):
            continue
        key, _, value = line[1:].partition(":")
        fields[key.strip().lower()] = value.strip()

    dx = fields.get("dx", "")
    patient_id = (
        fields.get("patientid")
        or fields.get("patient id")
        or fields.get("pid")
        or fields.get("subjectid")
        or fields.get("subject id")
    )
    return HeaderMeta(
        record_id=record_id,
        num_leads=num_leads,
        fs=fs,
        n_samples=n_samples,
        lead_names=lead_names,
        age=fields.get("age"),
        sex=fields.get("sex"),
        dx_codes=[code.strip() for code in dx.split(",") if code.strip()],
        patient_id=patient_id if patient_id and patient_id.lower() != "unknown" else None,
    )


def label_from_dx(dx_codes: list[str]) -> int:
    if not dx_codes:
        raise ValueError("missing Dx codes; cannot assign Normal/Abnormal label")
    non_normal = set(dx_codes) - {NORMAL_CODE}
    return 1 if non_normal else 0


def load_signal(mat_path: Path, expected_leads: int) -> np.ndarray:
    data = scipy.io.loadmat(mat_path)
    keys = [key for key in data if not key.startswith("__")]
    if not keys:
        raise ValueError(f"no signal array in {mat_path}")
    arr = np.asarray(data[keys[0]], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D signal in {mat_path}, got {arr.shape}")
    if arr.shape[0] != expected_leads and arr.shape[1] == expected_leads:
        arr = arr.T
    return arr


def choose_lead(signal: np.ndarray, meta: HeaderMeta, lead: str) -> np.ndarray:
    if lead in meta.lead_names:
        idx = meta.lead_names.index(lead)
    elif lead.upper() in [name.upper() for name in meta.lead_names]:
        idx = [name.upper() for name in meta.lead_names].index(lead.upper())
    else:
        idx = 0
    return signal[idx].astype(np.float32, copy=False)


def crop_pad_resample(
    lead_signal: np.ndarray,
    source_fs: int,
    duration: float,
    target_fs: int,
    normalize: str,
) -> np.ndarray:
    source_len = int(round(source_fs * duration))
    target_len = int(round(target_fs * duration))

    signal = np.nan_to_num(lead_signal, nan=0.0, posinf=0.0, neginf=0.0)
    if signal.shape[0] >= source_len:
        signal = signal[:source_len]
    else:
        signal = np.pad(signal, (0, source_len - signal.shape[0]), mode="constant")

    if source_len != target_len:
        signal = scipy.signal.resample(signal, target_len).astype(np.float32)
    else:
        signal = signal.astype(np.float32, copy=False)

    if normalize == "zscore":
        mean = float(signal.mean())
        std = float(signal.std())
        if std > 1e-6:
            signal = (signal - mean) / std
    elif normalize != "none":
        raise ValueError(f"unknown normalize mode: {normalize}")
    return signal[None, :]


def waveform_hash(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(x.astype(np.float32, copy=False))
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def group_key(meta: HeaderMeta, group_by: str, waveform_hash_value: str) -> tuple[str, str]:
    if group_by == "record":
        return meta.record_id, "record"
    if group_by == "patient":
        if not meta.patient_id:
            return meta.record_id, "record_fallback"
        return meta.patient_id, "patient"
    if group_by == "auto":
        if meta.patient_id:
            return meta.patient_id, "patient"
        return waveform_hash_value, "waveform_hash_fallback"
    raise ValueError(f"unknown group_by: {group_by}")


def scan_domain(
    raw_root: Path,
    domain: str,
    lead: str,
    duration: float,
    target_fs: int,
    group_by: str,
    normalize: str,
    max_records: int | None,
) -> list[Record]:
    raw_folder, _ = DOMAIN_CONFIG[domain]
    domain_root = raw_root / "training" / raw_folder
    if not domain_root.exists():
        domain_root = raw_root / raw_folder
    if not domain_root.exists():
        raise FileNotFoundError(f"raw domain directory not found: {raw_folder}")

    headers = sorted(domain_root.rglob("*.hea"))
    if max_records is not None:
        headers = headers[:max_records]

    records: list[Record] = []
    for idx, hea_path in enumerate(headers, start=1):
        mat_path = hea_path.with_suffix(".mat")
        if not mat_path.exists():
            continue
        try:
            meta = parse_header(hea_path)
            raw_signal = load_signal(mat_path, expected_leads=meta.num_leads)
            selected = choose_lead(raw_signal, meta, lead=lead)
            x = crop_pad_resample(
                selected,
                source_fs=meta.fs,
                duration=duration,
                target_fs=target_fs,
                normalize=normalize,
            )
            x_hash = waveform_hash(x)
            gid, gkind = group_key(meta, group_by=group_by, waveform_hash_value=x_hash)
            records.append(
                Record(
                    x=x,
                    y=label_from_dx(meta.dx_codes),
                    record_id=meta.record_id,
                    source_database=domain,
                    source_path=str(hea_path),
                    dx_codes=meta.dx_codes,
                    age=meta.age,
                    sex=meta.sex,
                    fs=meta.fs,
                    n_samples=meta.n_samples,
                    group_id=gid,
                    group_kind=gkind,
                    waveform_hash=x_hash,
                )
            )
        except Exception as exc:
            print(f"[skip] {hea_path}: {exc}", flush=True)
        if idx % 1000 == 0:
            print(f"[scan] {domain}: {idx}/{len(headers)} headers", flush=True)
    return records


def split_records(
    records: list[Record],
    train_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        groups.setdefault(record.group_id, []).append(idx)

    group_items: list[tuple[str, int, int]] = []
    for gid, indices in groups.items():
        labels = [records[idx].y for idx in indices]
        majority_label = Counter(labels).most_common(1)[0][0]
        group_items.append((gid, majority_label, len(indices)))

    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in (0, 1):
        label_groups = [item for item in group_items if item[1] == label]
        rng.shuffle(label_groups)
        total = sum(size for _, _, size in label_groups)
        target_train = round(total * train_ratio)
        seen = 0
        for gid, _, size in label_groups:
            if seen < target_train:
                train_indices.extend(groups[gid])
                seen += size
            else:
                test_indices.extend(groups[gid])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def take(records: list[Record], indices: Iterable[int]) -> list[Record]:
    return [records[idx] for idx in indices]


def tensorize(records: list[Record]) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.stack([record.x for record in records]).astype(np.float32)
    y = np.asarray([record.y for record in records], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def metadata(records: list[Record]) -> dict:
    return {
        "record_id": [record.record_id for record in records],
        "source_database": [record.source_database for record in records],
        "source_path": [record.source_path for record in records],
        "dx_codes": [record.dx_codes for record in records],
        "age": [record.age for record in records],
        "sex": [record.sex for record in records],
        "fs": [record.fs for record in records],
        "n_samples": [record.n_samples for record in records],
        "group_id": [record.group_id for record in records],
        "group_kind": [record.group_kind for record in records],
        "waveform_hash": [record.waveform_hash for record in records],
    }


def label_counts(records: list[Record]) -> dict[str, int]:
    counts = Counter(record.y for record in records)
    return {"normal_0": int(counts[0]), "abnormal_1": int(counts[1])}


def exact_overlap(train_records: list[Record], test_records: list[Record]) -> int:
    train_hashes = {record.waveform_hash for record in train_records}
    return sum(1 for record in test_records if record.waveform_hash in train_hashes)


def save_domain(
    output_dir: Path,
    domain: str,
    records: list[Record],
    train_indices: list[int],
    test_indices: list[int],
    args: argparse.Namespace,
) -> dict:
    _, prefix = DOMAIN_CONFIG[domain]
    train_records = take(records, train_indices)
    test_records = take(records, test_indices)

    shared_meta = {
        "label_task": LABEL_TASK,
        "label_description": LABEL_DESCRIPTION,
        "source": "PhysioNet Challenge 2021 v1.0.3",
        "normal_code": NORMAL_CODE,
        "lead": args.lead,
        "duration_seconds": args.duration,
        "target_fs": args.target_fs,
        "target_length": int(round(args.duration * args.target_fs)),
        "normalize": args.normalize,
        "split_method": f"{args.group_by}_grouped_stratified_split",
        "split_seed": args.seed,
        "train_ratio": args.train_ratio,
    }

    for split, split_records in (("train", train_records), ("test", test_records)):
        x, y = tensorize(split_records)
        payload = {
            "x": x,
            "y": y,
            **metadata(split_records),
            **shared_meta,
        }
        torch.save(payload, output_dir / f"{prefix}_{split}.pt")

    group_kinds = Counter(record.group_kind for record in records)
    report = {
        "domain": domain,
        "prefix": prefix,
        "records": len(records),
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "train_labels": label_counts(train_records),
        "test_labels": label_counts(test_records),
        "group_kinds": dict(group_kinds),
        "unique_groups": len({record.group_id for record in records}),
        "unique_waveforms": len({record.waveform_hash for record in records}),
        "duplicate_waveform_records": len(records) - len({record.waveform_hash for record in records}),
        "exact_train_test_overlap": exact_overlap(train_records, test_records),
    }
    print(
        f"{domain}: train={report['train_samples']} test={report['test_samples']} "
        f"groups={report['unique_groups']} kinds={report['group_kinds']} "
        f"overlap={report['exact_train_test_overlap']}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--lead", default="I")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--target-fs", type=int, default=500)
    parser.add_argument("--normalize", choices=["none", "zscore"], default="none")
    parser.add_argument("--group-by", choices=["auto", "patient", "record"], default="auto")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records-per-domain", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for domain in args.domains:
        if domain not in DOMAIN_CONFIG:
            raise ValueError(f"unknown domain {domain}; options={list(DOMAIN_CONFIG)}")
        records = scan_domain(
            raw_root=args.raw_root,
            domain=domain,
            lead=args.lead,
            duration=args.duration,
            target_fs=args.target_fs,
            group_by=args.group_by,
            normalize=args.normalize,
            max_records=args.max_records_per_domain,
        )
        train_indices, test_indices = split_records(
            records,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        reports.append(save_domain(args.output_dir, domain, records, train_indices, test_indices, args))

    manifest = {
        "raw_root": str(args.raw_root),
        "output_dir": str(args.output_dir),
        "domains": args.domains,
        "label_task": LABEL_TASK,
        "label_description": LABEL_DESCRIPTION,
        "normal_code": NORMAL_CODE,
        "lead": args.lead,
        "duration_seconds": args.duration,
        "target_fs": args.target_fs,
        "normalize": args.normalize,
        "group_by": args.group_by,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "reports": reports,
    }
    manifest_path = args.output_dir / "raw_build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
