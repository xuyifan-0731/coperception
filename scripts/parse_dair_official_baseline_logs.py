#!/usr/bin/env python
"""Parse OpenDAIRV2X official baseline logs into a compact CSV table."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


AP_RE = re.compile(
    r"car\s+(?P<kind>3d|bev)\s+IoU threshold\s+(?P<thr>\d+\.\d+),\s+Average Precision\s+=\s+(?P<ap>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
COMM_RE = re.compile(r"Average Communication Cost\s+=\s+(?P<comm>[-+]?\d+(?:\.\d+)?)\s+Bytes", re.IGNORECASE)


def parse_log(path: Path) -> dict[str, str | float]:
    row: dict[str, str | float] = {"run": path.stem, "log": str(path)}
    text = path.read_text(errors="replace")
    for match in AP_RE.finditer(text):
        key = f"car_{match.group('kind').lower()}_ap_{match.group('thr').replace('.', '_')}"
        row[key] = float(match.group("ap"))
    comm = COMM_RE.search(text)
    if comm:
        row["avg_comm_bytes"] = float(comm.group("comm"))
    row["status"] = "complete" if "avg_comm_bytes" in row else "incomplete"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("results/dair_v2x/official_baselines/logs"))
    parser.add_argument("--out", type=Path, default=Path("results/dair_v2x/official_baselines/summary.csv"))
    args = parser.parse_args()

    logs = sorted(args.log_dir.glob("*.log"))
    rows = [parse_log(path) for path in logs]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run",
        "status",
        "car_3d_ap_0_30",
        "car_3d_ap_0_50",
        "car_3d_ap_0_70",
        "car_bev_ap_0_30",
        "car_bev_ap_0_50",
        "car_bev_ap_0_70",
        "avg_comm_bytes",
        "log",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    complete = sum(1 for row in rows if row["status"] == "complete")
    print(f"parsed {len(rows)} logs; complete={complete}; wrote {args.out}")


if __name__ == "__main__":
    main()
