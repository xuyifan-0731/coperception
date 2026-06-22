#!/usr/bin/env python3
"""Inspect the local V2X-Sim-det zip without extracting it."""

from __future__ import annotations

import argparse
import csv
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("V2X-Sim-det-001.zip"))
    parser.add_argument("--out", type=Path, default=Path("results/revision_plan/v2xsim_zip_inventory.csv"))
    args = parser.parse_args()

    split_agent_counts: Counter[tuple[str, str]] = Counter()
    split_counts: Counter[str] = Counter()
    uncompressed_bytes = 0
    compressed_bytes = 0
    file_count = 0
    tiny_files = 0
    scenes: dict[tuple[str, str], set[str]] = defaultdict(set)

    with zipfile.ZipFile(args.zip) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            file_count += 1
            uncompressed_bytes += info.file_size
            compressed_bytes += info.compress_size
            if info.file_size <= 256:
                tiny_files += 1
            parts = Path(info.filename).parts
            if len(parts) >= 4 and parts[0] == "V2X-Sim-det":
                split, agent, sample = parts[1], parts[2], parts[3]
                split_counts[split] += 1
                split_agent_counts[(split, agent)] += 1
                scenes[(split, agent)].add(sample.split("_")[0])

    rows = []
    for (split, agent), samples in sorted(split_agent_counts.items()):
        rows.append(
            {
                "split": split,
                "agent": agent,
                "sample_files": samples,
                "scene_count": len(scenes[(split, agent)]),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "agent", "sample_files", "scene_count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"zip={args.zip}")
    print(f"files={file_count}")
    print(f"tiny_files={tiny_files}")
    print(f"compressed_gib={compressed_bytes / 1024**3:.3f}")
    print(f"uncompressed_gib={uncompressed_bytes / 1024**3:.3f}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()

