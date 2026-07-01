r"""Download selected PhysioNet/CinC Challenge 2021 raw folders.

The official project page recommends wget recursion, but wget is not always
available on Windows. This script mirrors only the requested training folders,
keeps a JSON inventory, and resumes partial downloads when the server supports
HTTP Range requests.

Examples
--------
Inventory only:

    python -m scripts.download_physionet_challenge2021 \
        --dest data/raw \
        --domains cpsc_2018 ptb-xl georgia chapman_shaoxing \
        --inventory-only

Download the four current paper domains:

    python -m scripts.download_physionet_challenge2021 \
        --dest data/raw \
        --domains cpsc_2018 ptb-xl georgia chapman_shaoxing
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


BASE_URL = "https://physionet.org/files/challenge-2021/1.0.3/"
TRAINING_URL = urllib.parse.urljoin(BASE_URL, "training/")
DEFAULT_DOMAINS = ["cpsc_2018", "ptb-xl", "georgia", "chapman_shaoxing"]
ROOT_FILES = ["LICENSE.txt", "RECORDS", "SHA256SUMS.txt"]
USER_AGENT = "ecg-cl-downloader/1.0"


@dataclass
class RemoteFile:
    url: str
    rel_path: str
    size: int | None = None


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr = dict(attrs)
        href = attr.get("href")
        if href:
            self.links.append(href)


def request(url: str, *, method: str = "GET", headers: dict[str, str] | None = None):
    req = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
    )
    return urllib.request.urlopen(req, timeout=60)


def read_text(url: str) -> str:
    with request(url) as resp:
        return resp.read().decode("utf-8", errors="replace")


def read_lines(url: str) -> list[str]:
    return [line.strip() for line in read_text(url).splitlines() if line.strip()]


def read_links(url: str) -> list[str]:
    html = read_text(url)
    parser = LinkParser()
    parser.feed(html)
    return parser.links


def normalize_child_url(base_url: str, href: str) -> str | None:
    if href.startswith("?") or href.startswith("#"):
        return None
    child = urllib.parse.urljoin(base_url, href)
    parsed_base = urllib.parse.urlparse(base_url)
    parsed_child = urllib.parse.urlparse(child)
    if parsed_child.netloc != parsed_base.netloc:
        return None
    if not parsed_child.path.startswith(parsed_base.path.rstrip("/") + "/"):
        return None
    if child.rstrip("/") == base_url.rstrip("/"):
        return None
    return child


def remote_size(url: str) -> int | None:
    try:
        with request(url, method="HEAD") as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except (OSError, urllib.error.HTTPError):
        return None


def collect_files(url: str, rel_root: str = "", fetch_sizes: bool = False) -> list[RemoteFile]:
    files: list[RemoteFile] = []
    for href in read_links(url):
        child = normalize_child_url(url, href)
        if child is None:
            continue
        name = urllib.parse.unquote(Path(urllib.parse.urlparse(child).path).name)
        if name in {"", ".", ".."}:
            continue
        if child.endswith("/"):
            files.extend(collect_files(child, f"{rel_root}{name}/", fetch_sizes=fetch_sizes))
        else:
            files.append(
                RemoteFile(
                    url=child,
                    rel_path=f"{rel_root}{name}",
                    size=remote_size(child) if fetch_sizes else None,
                )
            )
    return files


def iter_requested_files(
    domains: Iterable[str],
    include_root_files: bool,
    fetch_sizes: bool,
    index_mode: str,
    group_prefixes: Iterable[str] | None = None,
) -> list[RemoteFile]:
    files: list[RemoteFile] = []
    if include_root_files:
        for name in ROOT_FILES:
            url = urllib.parse.urljoin(BASE_URL, name)
            files.append(
                RemoteFile(
                    url=url,
                    rel_path=name,
                    size=remote_size(url) if fetch_sizes else None,
                )
            )

    if index_mode == "records":
        top_records_url = urllib.parse.urljoin(BASE_URL, "RECORDS")
        group_dirs = read_lines(top_records_url)
        requested = {domain.strip("/") for domain in domains}
        requested_groups = None
        if group_prefixes:
            requested_groups = {
                prefix.strip().strip("/") + "/" for prefix in group_prefixes
            }
        selected_groups = [
            rel for rel in group_dirs
            if any(rel.startswith(f"training/{domain}/") for domain in requested)
        ]
        if requested_groups is not None:
            selected_groups = [
                rel for rel in selected_groups
                if rel.strip().strip("/") + "/" in requested_groups
            ]
        for group_rel in selected_groups:
            group_records_rel = f"{group_rel.rstrip('/')}/RECORDS"
            group_records_url = urllib.parse.urljoin(BASE_URL, group_records_rel)
            files.append(
                RemoteFile(
                    url=group_records_url,
                    rel_path=group_records_rel,
                    size=remote_size(group_records_url) if fetch_sizes else None,
                )
            )
            stems = read_lines(group_records_url)
            for stem in stems:
                for suffix in (".hea", ".mat"):
                    rel_path = f"{group_rel.rstrip('/')}/{stem}{suffix}"
                    url = urllib.parse.urljoin(BASE_URL, rel_path)
                    files.append(
                        RemoteFile(
                            url=url,
                            rel_path=rel_path,
                            size=remote_size(url) if fetch_sizes else None,
                        )
                    )
        by_domain = {domain: 0 for domain in requested}
        for file in files:
            for domain in requested:
                if file.rel_path.startswith(f"training/{domain}/"):
                    by_domain[domain] += 1
        for domain, count in sorted(by_domain.items()):
            print(f"[inventory] {domain}: {count} files")
        return files

    if index_mode != "html":
        raise ValueError(f"unknown index mode: {index_mode}")

    if group_prefixes:
        for group in group_prefixes:
            group_rel = group.strip().strip("/") + "/"
            group_url = urllib.parse.urljoin(BASE_URL, group_rel)
            group_files = collect_files(
                group_url,
                group_rel,
                fetch_sizes=fetch_sizes,
            )
            files.extend(group_files)
            print(f"[inventory] {group_rel}: {len(group_files)} files")
        return files

    for domain in domains:
        domain_url = urllib.parse.urljoin(TRAINING_URL, f"{domain.strip('/')}/")
        domain_files = collect_files(
            domain_url,
            f"training/{domain.strip('/')}/",
            fetch_sizes=fetch_sizes,
        )
        files.extend(domain_files)
        print(f"[inventory] {domain}: {len(domain_files)} files")
    return files


def format_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def write_inventory(files: list[RemoteFile], path: Path) -> None:
    total_known = sum(file.size or 0 for file in files)
    unknown = sum(1 for file in files if file.size is None)
    groups = sorted(
        {
            "/".join(file.rel_path.split("/")[:3]) + "/"
            for file in files
            if file.rel_path.startswith("training/") and len(file.rel_path.split("/")) >= 3
        }
    )
    payload = {
        "base_url": BASE_URL,
        "file_count": len(files),
        "known_total_bytes": total_known,
        "known_total_human": format_size(total_known),
        "unknown_size_files": unknown,
        "groups": groups,
        "files": [asdict(file) for file in files],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[inventory] wrote {path} | files={len(files)} "
        f"known_total={format_size(total_known)} unknown={unknown}"
    )


def write_curl_config(files: list[RemoteFile], path: Path, dest_root: Path) -> None:
    lines = [
        "# Generated by scripts.download_physionet_challenge2021.",
        "# Run with curl -L --fail --retry 5 --retry-all-errors --create-dirs",
        "# plus --parallel/--parallel-max as desired.",
    ]
    for remote in files:
        target = (dest_root / remote.rel_path).as_posix()
        lines.append(f'url = "{remote.url}"')
        lines.append(f'output = "{target}"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[curl] wrote {path} | urls={len(files)}")


def download_file(
    remote: RemoteFile,
    dest_root: Path,
    retries: int,
    verify_size: bool,
) -> str:
    target = dest_root / remote.rel_path
    partial = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = remote.size
    if verify_size and expected_size is None:
        expected_size = remote_size(remote.url)

    if target.exists() and expected_size is not None and target.stat().st_size == expected_size:
        return "skip"
    if target.exists() and expected_size is None:
        return "skip"

    start = partial.stat().st_size if partial.exists() else 0
    mode = "ab" if start else "wb"
    headers = {"Range": f"bytes={start}-"} if start else {}

    for attempt in range(1, retries + 1):
        try:
            with request(remote.url, headers=headers) as resp:
                if start and resp.status != 206:
                    start = 0
                    mode = "wb"
                with partial.open(mode) as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise IOError(
                    f"incomplete download: {partial.stat().st_size} != {expected_size}"
                )
            partial.replace(target)
            return "downloaded"
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            if attempt >= retries:
                raise
            wait = min(30, 2 * attempt)
            print(f"[retry] {remote.rel_path}: {exc} | sleeping {wait}s")
            time.sleep(wait)
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--group-prefixes", nargs="*", default=None)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--curl-config", type=Path, default=None)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--include-root-files", action="store_true", default=True)
    parser.add_argument("--no-root-files", dest="include_root_files", action="store_false")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fetch-sizes", action="store_true")
    parser.add_argument(
        "--verify-size-on-download",
        action="store_true",
        help="Issue HEAD requests during download and verify byte counts.",
    )
    parser.add_argument("--index-mode", choices=["records", "html"], default="records")
    args = parser.parse_args()

    files = iter_requested_files(
        args.domains,
        include_root_files=args.include_root_files,
        fetch_sizes=args.fetch_sizes,
        index_mode=args.index_mode,
        group_prefixes=args.group_prefixes,
    )
    if args.max_files is not None:
        files = files[: args.max_files]

    inventory_path = args.inventory or args.dest / "download_inventory.json"
    write_inventory(files, inventory_path)
    if args.curl_config is not None:
        write_curl_config(files, args.curl_config, args.dest)
    if args.inventory_only:
        return

    counts = {"skip": 0, "downloaded": 0, "failed": 0}

    def run_one(remote: RemoteFile) -> tuple[str, str]:
        status = download_file(
            remote,
            args.dest,
            retries=args.retries,
            verify_size=args.verify_size_on_download,
        )
        return status, remote.rel_path

    if args.workers <= 1:
        for idx, remote in enumerate(files, start=1):
            if args.verbose:
                print(
                    f"[{idx}/{len(files)}] {remote.rel_path} "
                    f"({format_size(remote.size)})"
                )
            status, rel_path = run_one(remote)
            counts[status] = counts.get(status, 0) + 1
            if status == "failed" or idx % args.progress_every == 0 or idx == len(files):
                print(f"[progress] {idx}/{len(files)} {counts} last={rel_path}")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, remote): remote for remote in files}
            for future in as_completed(futures):
                done += 1
                remote = futures[future]
                try:
                    status, rel_path = future.result()
                except Exception as exc:
                    status = "failed"
                    rel_path = remote.rel_path
                    print(f"[failed] {rel_path}: {exc}")
                counts[status] = counts.get(status, 0) + 1
                if (
                    args.verbose
                    or status == "failed"
                    or done % args.progress_every == 0
                    or done == len(files)
                ):
                    print(f"[progress] {done}/{len(files)} {counts} last={rel_path}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
