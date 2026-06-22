#!/usr/bin/env python3
"""Report the current local revision pipeline status.

The script reads only local artifacts under the current workspace. It is safe to
run while training is active because it only inspects logs, checkpoint files, and
process metadata.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MetricPoint:
    run: str
    epoch: int | str
    step: int
    iou: float | str
    delta_iou: float | str
    dynamic_iou: float | str
    source: str


def portable(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


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


def read_points(run_root: Path) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for log_path in sorted(run_root.rglob("wandb_logs.txt")):
        run = log_path.parent.parent.name if log_path.parent.name == "checkpoints" else log_path.parent.name
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                raw = parse_dict_from_line(line)
                if not raw or "step" not in raw:
                    continue
                points.append(
                    MetricPoint(
                        run=run,
                        epoch=raw.get("epoch", ""),
                        step=int(raw["step"]),
                        iou=raw.get("IoU", raw.get("val/IoU", "")),
                        delta_iou=raw.get("Delta_IoU", raw.get("val/Delta_IoU", "")),
                        dynamic_iou=raw.get("Dynamic_IoU", raw.get("val/Dynamic_IoU", "")),
                        source=portable(log_path),
                    )
                )
    return points


def latest_by_run(points: list[MetricPoint]) -> list[MetricPoint]:
    latest: dict[str, MetricPoint] = {}
    for point in points:
        if point.run not in latest or point.step > latest[point.run].step:
            latest[point.run] = point
    return [latest[key] for key in sorted(latest)]


def process_status(pid_file: Path) -> str:
    if not pid_file.exists():
        return "pid_file_missing"
    pid_text = pid_file.read_text(encoding="utf-8").strip()
    if not pid_text:
        return "pid_file_empty"
    try:
        os.kill(int(pid_text), 0)
    except ProcessLookupError:
        return f"stopped(pid={pid_text})"
    except PermissionError:
        return f"unknown_permission(pid={pid_text})"
    return f"running(pid={pid_text})"


def current_process_lines() -> list[str]:
    cmd = ["ps", "-o", "pid,ppid,stat,etime,pcpu,pmem,cmd", "-u", os.environ.get("USER", "")]
    try:
        out = subprocess.check_output(cmd, text=True, errors="ignore")
    except Exception:
        return []
    keep = []
    needles = (
        "run_revision_full_pipeline",
        "recover_vspm_sensitivity",
        "run_vspm_sensitivity",
        "run_mst_ablation",
        "train_bev",
    )
    for line in out.splitlines():
        if any(needle in line for needle in needles):
            keep.append(line)
    return keep


def checkpoint_rows(run_root: Path) -> list[dict[str, str | int]]:
    rows = []
    for ckpt_dir in sorted(run_root.rglob("checkpoints")):
        def ckpt_step(path: Path) -> int:
            match = re.search(r"ckpt_(\d+)\.pth$", path.name)
            return int(match.group(1)) if match else -1

        ckpts = sorted(ckpt_dir.glob("ckpt_*.pth"), key=ckpt_step)
        rows.append(
            {
                "run": ckpt_dir.parent.name,
                "checkpoint_count": len(ckpts),
                "latest_checkpoint": ckpts[-1].name if ckpts else "",
                "source": portable(ckpt_dir),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("results/revision_plan"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-md", type=Path, default=Path("results/revision_plan/pipeline_status.md"))
    args = parser.parse_args()

    run_roots = [
        args.runs_root / "revision_vspm_sensitivity",
        args.runs_root / "revision_mst_ablation",
    ]
    points: list[MetricPoint] = []
    checkpoints: list[dict[str, str | int]] = []
    for root in run_roots:
        if root.exists():
            points.extend(read_points(root))
            checkpoints.extend(checkpoint_rows(root))

    latest_points = latest_by_run(points)
    metric_rows = [
        {
            "run": p.run,
            "epoch": p.epoch,
            "step": p.step,
            "IoU": p.iou,
            "Delta_IoU": p.delta_iou,
            "Dynamic_IoU": p.dynamic_iou,
            "source": p.source,
        }
        for p in latest_points
    ]
    write_csv(
        args.result_root / "pipeline_latest_metrics.csv",
        metric_rows,
        ["run", "epoch", "step", "IoU", "Delta_IoU", "Dynamic_IoU", "source"],
    )
    write_csv(
        args.result_root / "pipeline_checkpoints.csv",
        checkpoints,
        ["run", "checkpoint_count", "latest_checkpoint", "source"],
    )

    lines = [
        "# Pipeline Status",
        "",
        f"- Full pipeline: {process_status(args.result_root / 'full_pipeline.pid')}",
        f"- VSPM recovery: {process_status(args.result_root / 'recover_vspm_sensitivity.pid')}",
        f"- MST ablation: {process_status(args.result_root / 'run_mst_ablation.pid')}",
        f"- Metric log points parsed: {len(points)}",
        f"- Runs with metrics: {len(latest_points)}",
        "",
        "## Active Processes",
        "",
    ]
    proc_lines = current_process_lines()
    if proc_lines:
        lines.extend([f"```text", *proc_lines, "```", ""])
    else:
        lines.extend(["No active revision training process found.", ""])
    lines.extend(["## Latest Metrics", ""])
    if metric_rows:
        lines.append("| run | epoch | step | IoU | Delta IoU | Dynamic IoU |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in metric_rows:
            lines.append(
                f"| {row['run']} | {row['epoch']} | {row['step']} | "
                f"{row['IoU']} | {row['Delta_IoU']} | {row['Dynamic_IoU']} |"
            )
    else:
        lines.append("No validation metrics have been written yet.")
    lines.extend(["", "## Checkpoints", ""])
    if checkpoints:
        lines.append("| run | count | latest |")
        lines.append("|---|---:|---|")
        for row in checkpoints:
            lines.append(f"| {row['run']} | {row['checkpoint_count']} | {row['latest_checkpoint']} |")
    else:
        lines.append("No checkpoints found.")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.result_root / 'pipeline_latest_metrics.csv'}")
    print(f"wrote={args.result_root / 'pipeline_checkpoints.csv'}")


if __name__ == "__main__":
    main()
