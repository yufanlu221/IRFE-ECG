"""Summarize completed baseline run directories.

The training scripts write one ``metrics_summary.json`` per completed run. This
utility scans one or more output roots and produces a compact CSV/Markdown
table for paper bookkeeping, without depending on screenshots or terminal
scrollback.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DOMAINS = ["cpsc", "ptbxl", "georgia", "chapman"]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.6f}"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_method(summary: Dict[str, Any], run_dir: Path) -> str:
    config = summary.get("experiment_config") or summary.get("config") or {}
    method = config.get("baseline_method") or config.get("method")
    if method:
        return str(method)
    router_mode = config.get("router_mode")
    if router_mode:
        return f"adapter_{router_mode}"
    name = run_dir.name
    if name.startswith("baseline_"):
        parts = name.split("_seed", 1)[0].split("baseline_", 1)[1]
        return parts
    if "_seed" in name:
        return name.split("_seed", 1)[0]
    for suffix in ("_full", "_smoke"):
        if suffix in name:
            return name.split(suffix, 1)[0]
    return ""


def _infer_seed(summary: Dict[str, Any], run_dir: Path) -> str:
    config = summary.get("experiment_config") or summary.get("config") or {}
    if "seed" in config:
        return str(config["seed"])
    marker = "_seed"
    if marker in run_dir.name:
        tail = run_dir.name.split(marker, 1)[1]
        digits = []
        for ch in tail:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            return "".join(digits)
    return ""


def summarize_run(summary_path: Path) -> Dict[str, str]:
    summary = _load_json(summary_path)
    run_dir = summary_path.parent
    final_f1 = summary.get("final_f1_by_domain") or {}
    final_metrics = summary.get("final_metrics_by_domain") or {}

    row: Dict[str, str] = {
        "method": _infer_method(summary, run_dir),
        "seed": _infer_seed(summary, run_dir),
        "mean_f1": _fmt(summary.get("mean_f1")),
        "mean_bwt": _fmt(summary.get("mean_bwt")),
        "mean_forgetting": _fmt(summary.get("mean_forgetting")),
        "run_dir": str(run_dir),
    }
    for domain in DOMAINS:
        row[f"{domain}_f1"] = _fmt(final_f1.get(domain))
        metrics = final_metrics.get(domain) or {}
        row[f"{domain}_auprc"] = _fmt(metrics.get("auprc"))
        row[f"{domain}_bal_acc"] = _fmt(metrics.get("balanced_acc"))
    return row


def iter_summary_paths(roots: Iterable[Path]) -> Iterable[Path]:
    seen = set()
    for root in roots:
        if root.is_file() and root.name == "metrics_summary.json":
            paths = [root]
        elif root.is_dir():
            paths = sorted(root.rglob("metrics_summary.json"))
        else:
            continue
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]], path: Path) -> None:
    columns = ["method", "seed", "mean_f1", "mean_bwt"] + [
        f"{domain}_f1" for domain in DOMAINS
    ]
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Output roots or individual metrics_summary.json files.",
    )
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    rows = [summarize_run(path) for path in iter_summary_paths(args.roots)]
    rows.sort(key=lambda r: (r["method"], r["seed"], r["run_dir"]))

    if args.out_csv:
        write_csv(rows, args.out_csv)
    if args.out_md:
        write_markdown(rows, args.out_md)

    for row in rows:
        print(
            f"{row['method']:<18s} seed={row['seed']:<3s} "
            f"mean_f1={row['mean_f1']:<8s} mean_bwt={row['mean_bwt']:<9s} "
            f"dir={row['run_dir']}"
        )


if __name__ == "__main__":
    main()
