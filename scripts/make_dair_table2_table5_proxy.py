#!/usr/bin/env python3
"""Build DAIR proxy tables using the Table2/Table5 latency grid.

This does not produce V2VNet/DiscoNet detection AP. It reshapes the existing
DAIR label-derived BEV delay-grid CSVs into the same communication/computation
latency layout used by the original paper tables.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = ("pred_iou", "dynamic_iou", "copy_last_iou", "delta_iou")
METHODS = (
    ("no_comp", "No compensation (V2VNet-style delayed baseline)"),
    ("dual", "Dual compensation (V2VNet+DLPCM-style proxy)"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float) -> str:
    return f"{value:.4f}"


def build_summary(input_dir: Path) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for mode, label in METHODS:
        rows = read_rows(input_dir / f"delay_grid_{mode}.csv")
        lookup = {
            (int(row["comm_delay_frames"]), int(row["comp_delay_frames"])): row
            for row in rows
        }
        for comp in (0, 2, 4):
            out: dict[str, object] = {
                "method": label,
                "mode": mode,
                "comp_delay_frames": comp,
            }
            for metric in METRICS:
                vals = [float(lookup[(comm, comp)][metric]) for comm in range(6)]
                out[f"{metric}_avg"] = sum(vals) / len(vals)
                for comm, value in enumerate(vals):
                    out[f"{metric}_comm{comm}"] = value
            summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "mode", "comp_delay_frames"]
    for metric in METRICS:
        fields += [f"{metric}_avg"] + [f"{metric}_comm{i}" for i in range(6)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], source: Path) -> None:
    lines = [
        "# DAIR Table2/Table5 proxy summary",
        "",
        f"Source: `{source}`",
        "",
        "This table uses the original paper latency grid, but reports DAIR label-derived BEV IoU, not detection AP.",
        "",
    ]
    for metric, title in (("pred_iou", "IoU"), ("dynamic_iou", "Dynamic IoU")):
        lines += [
            f"## {title}",
            "",
            "| Method | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            vals = [float(row[f"{metric}_comm{i}"]) for i in range(6)]
            lines.append(
                "| {method} | {comp} | {avg} | {vals} |".format(
                    method=row["method"],
                    comp=row["comp_delay_frames"],
                    avg=fmt(float(row[f"{metric}_avg"])),
                    vals=" | ".join(fmt(v) for v in vals),
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/dair_v2x/our_method/full_T10_n10_ckpt15000"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/dair_v2x/table2_table5_dair_proxy"),
    )
    args = parser.parse_args()

    rows = build_summary(args.input_dir)
    write_csv(args.out_dir / "dair_table2_table5_proxy_summary.csv", rows)
    write_markdown(args.out_dir / "dair_table2_table5_proxy_summary.md", rows, args.input_dir)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
