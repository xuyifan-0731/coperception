#!/usr/bin/env python3
"""Collect lightweight experiment result tables from local training artifacts.

This script intentionally reads from ignored local folders such as ``model1`` and
writes compact CSV summaries under ``results/summary``. The generated CSV files
are suitable for version control; raw checkpoints, predictions, and datasets are
not.
"""

from __future__ import annotations

import argparse
import ast
import csv
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any


METRIC_KEYS = {
    "IoU": ("val/IoU", "IoU"),
    "Baseline_IoU": ("val/Baseline_IoU", "Baseline_IoU"),
    "Delta_IoU": ("val/Delta_IoU", "Delta_IoU"),
    "Dynamic_IoU": ("val/Dynamic_IoU", "Dynamic_IoU"),
    "Precision": ("val/Precision", "Precision"),
    "Recall": ("val/Recall", "Recall"),
    "F1": ("val/F1", "F1"),
    "Accuracy": ("val/Accuracy", "Accuracy"),
}


def parse_dict_from_line(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = ast.literal_eval(line[start : end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in row:
            return row[key]
    return None


def experiment_group(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parent = rel.parent
    if str(parent) == ".":
        return "root"
    if parent.name.startswith("checkpoints"):
        group = str(parent.parent)
        return "root" if group == "." else group
    return str(parent)


def run_name(path: Path) -> str:
    return path.parent.name


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def read_wandb_log(path: Path, root: Path) -> list[dict[str, Any]]:
    rows: OrderedDict[int, dict[str, Any]] = OrderedDict()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "IoU" not in line or "step" not in line:
                continue
            raw = parse_dict_from_line(line)
            if not raw or "step" not in raw:
                continue
            try:
                step = int(raw["step"])
            except Exception:
                continue
            row: dict[str, Any] = {
                "experiment_group": experiment_group(path, root),
                "run": run_name(path),
                "source": portable_path(path),
                "epoch": raw.get("epoch", ""),
                "step": step,
            }
            has_metric = False
            for out_key, aliases in METRIC_KEYS.items():
                value = first_present(raw, aliases)
                if value is None:
                    row[out_key] = ""
                    continue
                try:
                    row[out_key] = float(value)
                    has_metric = True
                except Exception:
                    row[out_key] = ""
            if has_metric:
                rows[step] = row
    return [rows[k] for k in sorted(rows)]


def summarize_series(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    valid_iou = [r for r in rows if r.get("IoU") != ""]
    best = max(valid_iou, key=lambda r: float(r["IoU"])) if valid_iou else rows[-1]
    final = rows[-1]
    return {
        "experiment_group": first["experiment_group"],
        "run": first["run"],
        "source": first["source"],
        "points": len(rows),
        "first_step": first["step"],
        "last_step": final["step"],
        "best_step": best["step"],
        "best_IoU": best.get("IoU", ""),
        "best_Delta_IoU": best.get("Delta_IoU", ""),
        "best_Dynamic_IoU": best.get("Dynamic_IoU", ""),
        "final_IoU": final.get("IoU", ""),
        "final_Delta_IoU": final.get("Delta_IoU", ""),
        "final_Dynamic_IoU": final.get("Dynamic_IoU", ""),
    }


def read_eval_metrics(path: Path, root: Path) -> list[dict[str, Any]]:
    rel = path.relative_to(root)
    result_name = rel.parts[-3] if len(rel.parts) >= 3 else rel.parent.name
    agent = rel.parts[-2] if len(rel.parts) >= 2 else ""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                iou = float(raw["iou"])
                baseline_iou = float(raw["baseline_iou"])
                interval = int(raw["interval"])
            except Exception:
                continue
            rows.append(
                {
                    "result": result_name,
                    "agent": agent,
                    "source": portable_path(path),
                    "scene": raw.get("scene", ""),
                    "interval": interval,
                    "last_input_frame": raw.get("last_input_frame", ""),
                    "target_frame": raw.get("target_frame", ""),
                    "iou": iou,
                    "baseline_iou": baseline_iou,
                    "delta_iou": iou - baseline_iou,
                }
            )
    return rows


def average(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(r[key]) for r in rows]
    return sum(values) / len(values) if values else float("nan")


def summarize_eval(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_interval: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    by_agent: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_interval.setdefault((row["result"], row["agent"], int(row["interval"])), []).append(row)
        by_agent.setdefault((row["result"], row["agent"]), []).append(row)

    interval_rows = []
    for (result, agent, interval), group in sorted(by_interval.items()):
        interval_rows.append(
            {
                "result": result,
                "agent": agent,
                "interval": interval,
                "samples": len(group),
                "mean_iou": average(group, "iou"),
                "mean_baseline_iou": average(group, "baseline_iou"),
                "mean_delta_iou": average(group, "delta_iou"),
            }
        )

    overall_rows = []
    for (result, agent), group in sorted(by_agent.items()):
        overall_rows.append(
            {
                "result": result,
                "agent": agent,
                "samples": len(group),
                "mean_iou": average(group, "iou"),
                "mean_baseline_iou": average(group, "baseline_iou"),
                "mean_delta_iou": average(group, "delta_iou"),
            }
        )
    return interval_rows, overall_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def copy_raw_metrics(metric_paths: list[Path], root: Path, out_dir: Path) -> None:
    target_root = out_dir / "raw_metrics"
    if target_root.exists():
        shutil.rmtree(target_root)
    for path in metric_paths:
        target = target_root / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("model1"))
    parser.add_argument("--out", type=Path, default=Path("results/summary"))
    parser.add_argument("--copy-raw-metrics", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out.resolve()
    log_paths = sorted(root.rglob("wandb_logs.txt"))
    metric_paths = sorted(root.rglob("metrics.csv"))

    timeseries: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for log_path in log_paths:
        rows = read_wandb_log(log_path, root)
        if not rows:
            continue
        timeseries.extend(rows)
        summaries.append(summarize_series(rows))

    metric_rows: list[dict[str, Any]] = []
    for metric_path in metric_paths:
        metric_rows.extend(read_eval_metrics(metric_path, root))
    interval_rows, overall_rows = summarize_eval(metric_rows)

    write_csv(
        out_dir / "training_timeseries.csv",
        timeseries,
        [
            "experiment_group",
            "run",
            "source",
            "epoch",
            "step",
            "IoU",
            "Baseline_IoU",
            "Delta_IoU",
            "Dynamic_IoU",
            "Precision",
            "Recall",
            "F1",
            "Accuracy",
        ],
    )
    write_csv(
        out_dir / "training_summary.csv",
        summaries,
        [
            "experiment_group",
            "run",
            "source",
            "points",
            "first_step",
            "last_step",
            "best_step",
            "best_IoU",
            "best_Delta_IoU",
            "best_Dynamic_IoU",
            "final_IoU",
            "final_Delta_IoU",
            "final_Dynamic_IoU",
        ],
    )
    write_csv(
        out_dir / "eval_by_interval.csv",
        interval_rows,
        ["result", "agent", "interval", "samples", "mean_iou", "mean_baseline_iou", "mean_delta_iou"],
    )
    write_csv(
        out_dir / "eval_overall.csv",
        overall_rows,
        ["result", "agent", "samples", "mean_iou", "mean_baseline_iou", "mean_delta_iou"],
    )

    if args.copy_raw_metrics:
        copy_raw_metrics(metric_paths, root, out_dir)

    print(f"Parsed {len(log_paths)} log files and {len(metric_paths)} metrics files.")
    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
