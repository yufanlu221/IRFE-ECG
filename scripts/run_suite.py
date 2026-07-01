"""Run a paper experiment suite from a versioned JSON configuration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_SEEDS = (42, 43, 44)
PATH_ENV_KEYS = {
    "ECG_DATA_DIR",
    "ECG_RAW_DATA_DIR",
    "ECG_CKPT_PATH",
    "ECG_FEATURE_CACHE_DIR",
    "ECG_MAIN_RESULTS_ROOT",
}


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    seeds = tuple(int(seed) for seed in config.get("seeds", PAPER_SEEDS))
    if seeds != PAPER_SEEDS:
        raise ValueError(f"Paper suites require seeds {PAPER_SEEDS}, received {seeds}.")
    config["seeds"] = seeds
    return config


def _format(value: Any, **fields: Any) -> str:
    return str(value).format(**fields)


def _base_environment(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    for key, value in config.get("environment", {}).items():
        rendered = str(value)
        if key in PATH_ENV_KEYS:
            path = Path(rendered)
            if not path.is_absolute():
                rendered = str((REPO_ROOT / path).resolve())
        env[key] = rendered
    return env


def _validate_inputs(env: dict[str, str]) -> None:
    data_dir = Path(env["ECG_DATA_DIR"])
    checkpoint = Path(env["ECG_CKPT_PATH"])
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ECGFounder checkpoint not found: {checkpoint}")


def _run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("$", subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _prepare_feature_cache(
    config: dict[str, Any], base_env: dict[str, str], dry_run: bool
) -> None:
    if not config.get("prepare_feature_cache", False):
        return
    cache_dir = Path(base_env["ECG_FEATURE_CACHE_DIR"])
    for seed in config["seeds"]:
        probe = cache_dir / f"ecgfounder_final_seed{seed}_val0p15_chapman_test.npz"
        if probe.is_file():
            print(f"[skip] feature cache seed={seed}: {probe}")
            continue
        env = dict(base_env)
        env["ECG_SEED"] = str(seed)
        env["ECG_OUTPUT_DIR"] = str((REPO_ROOT / "outputs" / f"cache_seed{seed}").resolve())
        _run([sys.executable, str(REPO_ROOT / "process.py")], env, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tag", default="paper")
    parser.add_argument("--job", action="append", dest="selected_jobs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    base_env = _base_environment(config)
    if not args.dry_run:
        _validate_inputs(base_env)
    _prepare_feature_cache(config, base_env, args.dry_run)

    for job in config.get("jobs", []):
        name = str(job["name"])
        if args.selected_jobs and name not in args.selected_jobs:
            continue
        seeds = (None,) if job.get("once", False) else config["seeds"]
        for seed in seeds:
            fields = {"seed": seed, "tag": args.tag, "name": name}
            output_name = _format(job["output"], **fields)
            output_dir = (REPO_ROOT / "outputs" / output_name).resolve()
            summary = output_dir / "metrics_summary.json"
            if summary.is_file() and not args.force:
                print(f"[skip] {name} seed={seed}: {summary}")
                continue

            env = dict(base_env)
            env["ECG_OUTPUT_DIR"] = str(output_dir)
            if seed is not None:
                env["ECG_SEED"] = str(seed)
            for key, value in job.get("environment", {}).items():
                env[key] = _format(value, **fields)
            command = [sys.executable, "-m", str(job["module"])]
            command.extend(_format(value, **fields) for value in job.get("args", []))
            print(f"[run] {name} seed={seed} output={output_dir}", flush=True)
            _run(command, env, args.dry_run)


if __name__ == "__main__":
    main()
