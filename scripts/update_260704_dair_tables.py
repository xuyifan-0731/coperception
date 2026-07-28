#!/usr/bin/env python3
"""Fill DAIR TCLF rows in 260704结果.md from strict CSV artifacts.

The strict DAIR CSVs may contain ``SyncNet`` rows produced by the local
diagnostic branch, but that branch reuses the DAIR TCLF temporal-offset path
and is not a real DAIR SyncNet checkpoint. By default this script refuses to
write those pseudo-SyncNet rows back into the paper table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC = ROOT / "260704结果.md"
DEFAULT_LATE = ROOT / "results/260615_fill/artifacts/dair_late_fusion_strict_tclf_syncnet_20260704.csv"
DEFAULT_DET = ROOT / "results/260615_fill/artifacts/dair_detector_strict_tclf_syncnet_20260704.csv"
COMMS = [0, 1, 2, 3, 4, 5]
COMPS = [0, 2, 4]


def load_rows(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            method = row["method"]
            comm = int(row["comm_delay_frames"])
            comp = int(row["comp_delay_frames"])
            rows[(method, comp, comm)] = row
    return rows


def replacement(
    rows: dict[tuple[str, int, int], dict[str, str]],
    category: str,
    method: str,
    comp: int,
    metric: str,
) -> str | None:
    vals: list[float] = []
    for comm in COMMS:
        row = rows.get((method, comp, comm))
        if row is None or not row.get(metric):
            return None
        vals.append(float(row[metric]))
    avg = sum(vals) / len(vals)
    numbers = [f"{avg:.2f}", *[f"{value:.2f}" for value in vals]]
    return f"| {category} | {method} | {comp} | {' | '.join(numbers)} |"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--late-csv", type=Path, default=DEFAULT_LATE)
    parser.add_argument("--detector-csv", type=Path, default=DEFAULT_DET)
    parser.add_argument(
        "--allow-pseudo-syncnet",
        action="store_true",
        help="Also fill pseudo SyncNet rows from diagnostic CSVs. Do not use for final DAIR tables.",
    )
    args = parser.parse_args()

    late_rows = load_rows(args.late_csv)
    detector_rows = load_rows(args.detector_csv)
    metric = ""
    changed = 0
    output: list[str] = []
    for line in args.doc.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#### 2.1.1") or stripped.startswith("##### 2.1.3.1"):
            metric = "BEV AP@0.5"
        elif stripped.startswith("#### 2.1.2") or stripped.startswith("##### 2.1.3.2"):
            metric = "BEV AP@0.7"
        elif stripped.startswith("### "):
            metric = ""

        new_line = None
        if metric:
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            if len(parts) == 10:
                category, method, comp_text = parts[0], parts[1], parts[2]
                if comp_text in {"0", "2", "4"}:
                    comp = int(comp_text)
                    if method == "Late fusion + TCLF":
                        new_line = replacement(late_rows, category, method, comp, metric)
                    elif args.allow_pseudo_syncnet and method == "Late fusion + SyncNet":
                        new_line = replacement(late_rows, category, method, comp, metric)
                    elif args.allow_pseudo_syncnet and method == "Detector + SyncNet":
                        new_line = replacement(detector_rows, category, method, comp, metric)

        if new_line is not None:
            output.append(new_line)
            changed += int(new_line != line)
        else:
            output.append(line)

    args.doc.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(f"updated {changed} rows in {args.doc}")


if __name__ == "__main__":
    main()
