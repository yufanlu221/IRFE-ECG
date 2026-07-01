"""Offline ECGFounder feature extraction for head-bank experiments.

This is intentionally separated from head training on the shared server:

1. Run this script once to cache frozen ECGFounder train/val/test features as
   `.npz` files.
2. Run head-only experiments from those `.npz` files without repeatedly
   forwarding ECG waveforms through ECGFounder.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import torch

from scripts.run_config import env_int
from trainer.feature_sklearn_baselines import (
    FEATURE_CACHE_DIR,
    load_or_extract_features,
)


CPU_THREADS = env_int("CPU_THREADS", env_int("ECG_CPU_THREADS", 2))


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))
    torch.set_num_threads(max(1, CPU_THREADS))
    torch.set_num_interop_threads(1)

    features, labels, split_infos = load_or_extract_features()
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_cache_dir": str(FEATURE_CACHE_DIR),
        "cpu_threads": CPU_THREADS,
        "domains": {
            domain: {
                split: {
                    "features_shape": list(features[domain][split].shape),
                    "labels_shape": list(labels[domain][split].shape),
                }
                for split in sorted(features[domain])
            }
            for domain in sorted(features)
        },
        "split_info": split_infos,
    }
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(FEATURE_CACHE_DIR) / "feature_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[process.py] feature cache ready: {FEATURE_CACHE_DIR}")
    print(f"[process.py] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
