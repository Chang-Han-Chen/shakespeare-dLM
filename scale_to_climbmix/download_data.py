"""Download the selected shuffled raw-text ClimbMix parquet shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import requests

from config import (
    DATASET_REPO,
    DATASET_REVISION,
    RAW_DIR,
    TRAIN_SHARDS,
    VALIDATION_SHARDS,
)


API_URL = f"https://huggingface.co/api/datasets/{DATASET_REPO}/tree/{DATASET_REVISION}"
RESOLVE_URL = (
    f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/"
    f"{DATASET_REVISION}"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_entries() -> dict[str, dict]:
    response = requests.get(
        API_URL,
        params={"recursive": "false", "expand": "false", "limit": 100},
        timeout=60,
    )
    response.raise_for_status()
    return {
        entry["path"]: entry
        for entry in response.json()
        if entry["type"] == "file"
    }


def download(entry: dict, destination: Path) -> None:
    expected_size = int(entry["size"])
    expected_sha = entry.get("lfs", {}).get("oid")
    if destination.exists() and destination.stat().st_size == expected_size:
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    current = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={current}-"} if current else {}
    with requests.get(
        f"{RESOLVE_URL}/{entry['path']}?download=true",
        headers=headers,
        stream=True,
        timeout=(30, 300),
    ) as response:
        if current and response.status_code == 200:
            current = 0
        response.raise_for_status()
        mode = "ab" if current and response.status_code == 206 else "wb"
        with temporary.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 << 20):
                if chunk:
                    handle.write(chunk)
    if temporary.stat().st_size != expected_size:
        raise IOError(
            f"Size mismatch for {entry['path']}: "
            f"{temporary.stat().st_size} != {expected_size}"
        )
    if expected_sha and sha256(temporary) != expected_sha:
        raise IOError(f"SHA256 mismatch for {entry['path']}")
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries = remote_entries()
    selected = sorted(set(VALIDATION_SHARDS + TRAIN_SHARDS))
    manifest_entries = []
    for position, index in enumerate(selected, start=1):
        name = f"shard_{index:05d}.parquet"
        if name not in entries:
            raise FileNotFoundError(f"{name} not present in {DATASET_REPO}")
        destination = RAW_DIR / name
        print(
            f"[{position}/{len(selected)}] {name} "
            f"{entries[name]['size'] / 1e6:.1f} MB",
            flush=True,
        )
        download(entries[name], destination)
        expected_sha = entries[name].get("lfs", {}).get("oid")
        if args.verify_existing and expected_sha and sha256(destination) != expected_sha:
            raise IOError(f"SHA256 mismatch for existing {name}")
        manifest_entries.append(
            {
                "index": index,
                "path": name,
                "bytes": int(entries[name]["size"]),
                "sha256": expected_sha,
                "split": "validation" if index in VALIDATION_SHARDS else "train",
            }
        )

    payload = {
        "dataset": "NVIDIA Nemotron-ClimbMix",
        "source_repository": DATASET_REPO,
        "source_revision": DATASET_REVISION,
        "source_note": (
            "Community raw-text shuffle derived from NVIDIA's GPT-2-tokenized "
            "ClimbMix release."
        ),
        "entries": manifest_entries,
    }
    path = RAW_DIR / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    print(f"downloaded {sum(row['bytes'] for row in manifest_entries) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
