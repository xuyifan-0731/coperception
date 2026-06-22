#!/usr/bin/env python3
"""Continuously fill 260615 result tables from completed experiment artifacts.

This script is intentionally conservative:

* It never converts DAIR BEV-IoU proxy numbers into detector AP.
* It fills values only when a matching completed artifact exists.
* It writes a machine-readable missing-cell manifest for every remaining "-".
* In --loop mode it keeps running until the markdown file has no remaining
  actionable blank result cells. Structural "-" cells, such as non-Comm0
  columns in no-delay rows, are tracked but do not block loop completion.

Additional experiment outputs can be supplied via:

  results/260615_fill/incoming_results.csv

with columns:

  table_id,category,method,comp,column,value

Example:

  2.1.3.2,不同时延 + 我们方法补偿,Detector + Ours,0,Comm0,51.23
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC = REPO_ROOT / "260615结果.md"
STATE_DIR = REPO_ROOT / "results" / "260615_fill"
INCOMING_CSV = STATE_DIR / "incoming_results.csv"
MISSING_CSV = STATE_DIR / "missing_cells.csv"
STATUS_JSON = STATE_DIR / "status.json"
LOG_DIR = STATE_DIR / "logs"
ARTIFACT_DIR = STATE_DIR / "artifacts"
TASK_INTERFACES_JSON = STATE_DIR / "task_interfaces.json"
TASK_STATUS_JSON = STATE_DIR / "task_status.json"


@dataclass(frozen=True)
class FillTarget:
    table_id: str
    category: str
    doc_method: str
    source_method: str


@dataclass(frozen=True)
class ResultInterface:
    task_id: str
    description: str
    dataset: str
    comparison: str
    artifact: Path
    targets: tuple[FillTarget, ...]
    metrics: dict[str, str]
    requirements: tuple[str, ...]


RESULT_INTERFACES: tuple[ResultInterface, ...] = (
    ResultInterface(
        task_id="v2xsim_v2vnet_syncnet",
        description="V2X-Sim 车车协同：V2VNet baseline 下补齐 SyncNet 补偿 detector AP。",
        dataset="V2X-Sim / 车车协同",
        comparison="V2VNet + SyncNet vs V2VNet / Ours",
        artifact=ARTIFACT_DIR / "v2xsim_v2vnet_syncnet_ap.csv",
        targets=(
            FillTarget("1.1.1", "不同时延 + SyncNet 补偿", "V2VNet + SyncNet", "V2VNet + SyncNet"),
            FillTarget("1.1.2", "不同时延 + SyncNet 补偿", "V2VNet + SyncNet", "V2VNet + SyncNet"),
        ),
        metrics={"1.1.1": "AP@0.5", "1.1.2": "AP@0.7"},
        requirements=(
            "需要可运行的 V2X-Sim detector AP pipeline。",
            "需要定义并实现 V2VNet + SyncNet 补偿方式。",
            "输出 comm_delay_frames=0..5、comp_delay_frames=0/2/4 的 AP@0.5/AP@0.7。",
        ),
    ),
    ResultInterface(
        task_id="v2xsim_data_compensation",
        description="V2X-Sim 车车协同：DATA baseline 下补齐 Ours 和 SyncNet 补偿 detector AP。",
        dataset="V2X-Sim 2.0 / 车车协同",
        comparison="DATA + Ours / DATA + SyncNet vs DATA no compensation",
        artifact=ARTIFACT_DIR / "v2xsim_data_compensation_ap.csv",
        targets=(
            FillTarget("1.2.1", "不同时延 + 我们方法补偿", "DATA + Ours", "DATA + Ours"),
            FillTarget("1.2.2", "不同时延 + 我们方法补偿", "DATA + Ours", "DATA + Ours"),
            FillTarget("1.2.1", "不同时延 + SyncNet 补偿", "DATA + SyncNet", "DATA + SyncNet"),
            FillTarget("1.2.2", "不同时延 + SyncNet 补偿", "DATA + SyncNet", "DATA + SyncNet"),
        ),
        metrics={"1.2.1": "AP@0.5", "1.2.2": "AP@0.7"},
        requirements=(
            "需要在 external/DATA/OpenCOOD pipeline 中实现 Ours 补偿接口。",
            "需要在 DATA pipeline 中实现 SyncNet 或等价同步补偿接口。",
            "输出 comm_delay_frames=0..5、comp_delay_frames=0/2/4 的 AP@0.5/AP@0.7。",
        ),
    ),
    ResultInterface(
        task_id="dair_tclf_comp_latency",
        description="DAIR-V2X 车路协同：官方 TCLF / no-comp baseline 增加计算时延维度 detector AP。",
        dataset="DAIR-V2X / 车路协同",
        comparison="Late fusion + TCLF / Late fusion w/o comp. under comm+comp latency",
        artifact=ARTIFACT_DIR / "dair_tclf_comp_latency_ap.csv",
        targets=(
            FillTarget("2.1.1", "不同时延无补偿", "Late fusion w/o comp.", "Late fusion w/o comp."),
            FillTarget("2.1.2", "不同时延无补偿", "Late fusion w/o comp.", "Late fusion w/o comp."),
            FillTarget("2.1.1", "不同时延 + TCLF 补偿", "Late fusion + TCLF", "Late fusion + TCLF"),
            FillTarget("2.1.2", "不同时延 + TCLF 补偿", "Late fusion + TCLF", "Late fusion + TCLF"),
        ),
        metrics={"2.1.1": "BEV AP@0.5", "2.1.2": "BEV AP@0.7"},
        requirements=(
            "需要扩展 OpenDAIRV2X official eval，使其支持 comp_delay_frames=2/4。",
            "需要明确计算时延模拟方式：输出延迟时 GT 对齐当前帧还是结果帧。",
            "输出 method、comm_delay_frames、comp_delay_frames、BEV AP@0.5、BEV AP@0.7。",
        ),
    ),
    ResultInterface(
        task_id="dair_late_fusion_ours_syncnet",
        description="DAIR-V2X 车路协同：2.1 TCLF baseline 表中补齐 Late fusion + Ours / SyncNet detector AP。",
        dataset="DAIR-V2X / 车路协同",
        comparison="Late fusion + Ours / Late fusion + SyncNet under comm+comp latency",
        artifact=ARTIFACT_DIR / "dair_late_fusion_ours_syncnet_ap.csv",
        targets=(
            FillTarget("2.1.1", "不同时延 + 我们方法补偿", "Late fusion + Ours", "Late fusion + Ours"),
            FillTarget("2.1.2", "不同时延 + 我们方法补偿", "Late fusion + Ours", "Late fusion + Ours"),
            FillTarget("2.1.1", "不同时延 + SyncNet 补偿", "Late fusion + SyncNet", "Late fusion + SyncNet"),
            FillTarget("2.1.2", "不同时延 + SyncNet 补偿", "Late fusion + SyncNet", "Late fusion + SyncNet"),
        ),
        metrics={"2.1.1": "BEV AP@0.5", "2.1.2": "BEV AP@0.7"},
        requirements=(
            "需要在 DAIR late-fusion detector eval 中接入 Ours 双时延补偿。",
            "需要在同一 DAIR late-fusion detector eval 中接入 SyncNet/同步补偿。",
            "需要覆盖 comm_delay_frames=0..5、comp_delay_frames=0/2/4。",
            "输出 method、comm_delay_frames、comp_delay_frames、BEV AP@0.5、BEV AP@0.7。",
        ),
    ),
    ResultInterface(
        task_id="dair_detector_ours_syncnet",
        description="DAIR-V2X 车路协同：Detector AP 口径下补齐 Ours / SyncNet 补偿。",
        dataset="DAIR-V2X / 车路协同",
        comparison="Detector + Ours / Detector + SyncNet / Detector w/o compensation",
        artifact=ARTIFACT_DIR / "dair_detector_ours_syncnet_ap.csv",
        targets=(
            FillTarget("2.1.3.1", "无延时", "Ours detector AP", "Ours detector AP"),
            FillTarget("2.1.3.2", "无延时", "Ours detector AP", "Ours detector AP"),
            FillTarget("2.1.3.1", "不同时延无补偿", "Detector w/o compensation", "Detector w/o compensation"),
            FillTarget("2.1.3.2", "不同时延无补偿", "Detector w/o compensation", "Detector w/o compensation"),
            FillTarget("2.1.3.1", "不同时延 + 我们方法补偿", "Detector + Ours", "Detector + Ours"),
            FillTarget("2.1.3.2", "不同时延 + 我们方法补偿", "Detector + Ours", "Detector + Ours"),
            FillTarget("2.1.3.1", "不同时延 + SyncNet 补偿", "Detector + SyncNet", "Detector + SyncNet"),
            FillTarget("2.1.3.2", "不同时延 + SyncNet 补偿", "Detector + SyncNet", "Detector + SyncNet"),
        ),
        metrics={"2.1.3.1": "BEV AP@0.5", "2.1.3.2": "BEV AP@0.7"},
        requirements=(
            "需要将 Ours dual-latency compensation 接入 DAIR detector pipeline。",
            "需要将 SyncNet/同步补偿接入同一 DAIR detector pipeline。",
            "不能使用 DAIR BEV-IoU 代理结果代替 detector AP。",
            "输出 method、comm_delay_frames、comp_delay_frames、BEV AP@0.5、BEV AP@0.7。",
        ),
    ),
    ResultInterface(
        task_id="dair_lrcp_latency",
        description="DAIR-V2X 车路协同：LRCP baseline latency grid detector AP。",
        dataset="DAIR-V2X / 车路协同",
        comparison="LRCP / LRCP + Ours / LRCP + SyncNet",
        artifact=ARTIFACT_DIR / "dair_lrcp_latency_ap.csv",
        targets=(
            FillTarget("2.2.1", "无延时", "LRCP", "LRCP"),
            FillTarget("2.2.2", "无延时", "LRCP", "LRCP"),
            FillTarget("2.2.1", "不同时延无补偿", "LRCP w/o compensation", "LRCP w/o compensation"),
            FillTarget("2.2.2", "不同时延无补偿", "LRCP w/o compensation", "LRCP w/o compensation"),
            FillTarget("2.2.1", "不同时延 + 我们方法补偿", "LRCP + Ours", "LRCP + Ours"),
            FillTarget("2.2.2", "不同时延 + 我们方法补偿", "LRCP + Ours", "LRCP + Ours"),
            FillTarget("2.2.1", "不同时延 + SyncNet 补偿", "LRCP + SyncNet", "LRCP + SyncNet"),
            FillTarget("2.2.2", "不同时延 + SyncNet 补偿", "LRCP + SyncNet", "LRCP + SyncNet"),
        ),
        metrics={"2.2.1": "BEV AP@0.5", "2.2.2": "BEV AP@0.7"},
        requirements=(
            "需要 DAIR-V2X 的 LRCP yaml/config。",
            "需要 LRCP checkpoint。",
            "需要 DAIR flow labels / delay-pair files。",
            "需要确认 external/LRCP/opencood/tools/inference_w_delay.py 支持 DAIR。",
            "输出 method、comm_delay_frames、comp_delay_frames、BEV AP@0.5、BEV AP@0.7。",
        ),
    ),
)


@dataclass
class Table:
    table_id: str
    title: str
    header_line: int
    start_line: int
    end_line: int
    header: list[str]
    separator: str
    rows: list[list[str]]


def split_row(line: str) -> list[str]:
    line = line.strip()
    if not (line.startswith("|") and line.endswith("|")):
        raise ValueError(f"not a markdown table row: {line}")
    return [cell.strip() for cell in line.strip("|").split("|")]


def format_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def table_id_from_heading(heading: str) -> str:
    parts = heading.lstrip("#").strip().split(maxsplit=1)
    if parts and parts[0][0].isdigit():
        return parts[0]
    return "unknown"


def parse_tables(lines: list[str]) -> list[Table]:
    tables: list[Table] = []
    current_heading = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            current_heading = line
            i += 1
            continue
        if (
            line.startswith("|")
            and i + 1 < len(lines)
            and lines[i + 1].startswith("|---")
        ):
            header_line = i
            header = split_row(lines[i])
            separator = lines[i + 1]
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            tables.append(
                Table(
                    table_id=table_id_from_heading(current_heading),
                    title=current_heading.lstrip("#").strip(),
                    header_line=header_line,
                    start_line=i,
                    end_line=j,
                    header=header,
                    separator=separator,
                    rows=rows,
                )
            )
            i = j
            continue
        i += 1
    return tables


def rewrite_tables(lines: list[str], tables: list[Table], changed_ids: set[int]) -> list[str]:
    if not changed_ids:
        return lines
    out = list(lines)
    for index in sorted(changed_ids, reverse=True):
        table = tables[index]
        block = [format_row(table.header), table.separator]
        block.extend(format_row(row) for row in table.rows)
        out[table.start_line : table.end_line] = block
    return out


def col_index(table: Table, name: str) -> int | None:
    try:
        return table.header.index(name)
    except ValueError:
        return None


def row_matches(table: Table, row: list[str], *, category: str, method: str, comp: str) -> bool:
    category_col = col_index(table, "类别")
    metric_col = col_index(table, "指标")
    method_col = col_index(table, "方法/补偿")
    comp_col = col_index(table, "Comp")

    if category_col is not None and row[category_col] != category:
        return False
    if metric_col is not None and row[metric_col] != category:
        return False
    if method_col is not None and row[method_col] != method:
        return False
    if comp_col is not None and row[comp_col] != str(comp):
        return False
    return True


def set_cell(
    tables: list[Table],
    *,
    table_id: str,
    category: str,
    method: str,
    comp: str | int,
    column: str,
    value: str,
) -> bool:
    changed = False
    for table in tables:
        if table.table_id != table_id:
            continue
        value_col = col_index(table, column)
        if value_col is None:
            continue
        for row in table.rows:
            if row_matches(table, row, category=category, method=method, comp=str(comp)):
                if row[value_col] != value:
                    row[value_col] = value
                    changed = True
    return changed


def fmt_percent(value: str | float) -> str:
    return f"{float(value):.2f}"


def fmt_iou(value: str | float) -> str:
    return f"{float(value):.4f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def artifact_schema() -> list[str]:
    return [
        "method",
        "comm_delay_frames",
        "comp_delay_frames",
        "AP@0.5",
        "AP@0.7",
        "BEV AP@0.5",
        "BEV AP@0.7",
        "source",
        "notes",
    ]


def task_template(interface: ResultInterface) -> dict[str, object]:
    return {
        "task_id": interface.task_id,
        "enabled": False,
        "description": interface.description,
        "dataset": interface.dataset,
        "comparison": interface.comparison,
        "artifact": str(interface.artifact),
        "expected_csv_columns": artifact_schema(),
        "command": [],
        "cwd": str(REPO_ROOT),
        "requirements": list(interface.requirements),
        "notes": (
            "Set enabled=true and command=[...] after the corresponding detector "
            "pipeline/checkpoints/configs are available. The command must write the "
            "artifact CSV above. The loop will parse and fill 260615结果.md."
        ),
    }


def ensure_task_interfaces_template() -> list[dict[str, object]]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not TASK_INTERFACES_JSON.exists():
        payload = {
            "schema_version": 1,
            "updated_by": "scripts/fill_260615_results.py",
            "tasks": [task_template(interface) for interface in RESULT_INTERFACES],
        }
        TASK_INTERFACES_JSON.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    payload = json.loads(TASK_INTERFACES_JSON.read_text(encoding="utf-8"))
    existing = {task.get("task_id"): task for task in payload.get("tasks", [])}
    changed = False
    for interface in RESULT_INTERFACES:
        if interface.task_id not in existing:
            payload.setdefault("tasks", []).append(task_template(interface))
            changed = True
    if changed:
        TASK_INTERFACES_JSON.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload.get("tasks", [])


def write_artifact_templates() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for interface in RESULT_INTERFACES:
        template = interface.artifact.with_suffix(".template.csv")
        if template.exists():
            continue
        with template.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=artifact_schema(), lineterminator="\n")
            writer.writeheader()
            for target in interface.targets:
                for comp in (0, 2, 4):
                    for comm in range(6):
                        if target.category == "无延时" and (comp != 0 or comm != 0):
                            continue
                        row = {
                            "method": target.source_method,
                            "comm_delay_frames": comm,
                            "comp_delay_frames": comp,
                            "source": interface.task_id,
                            "notes": "fill metric columns after experiment completes",
                        }
                        writer.writerow(row)


def metric_value(row: dict[str, str], metric: str) -> str | None:
    candidates = [metric]
    compact = metric.replace(" ", "")
    if compact != metric:
        candidates.append(compact)
    lower_map = {key.lower(): key for key in row.keys()}
    for key in candidates:
        if key in row and row[key].strip():
            return row[key].strip()
        lower_key = key.lower()
        if lower_key in lower_map and row[lower_map[lower_key]].strip():
            return row[lower_map[lower_key]].strip()
    return None


def apply_result_interface(tables: list[Table], interface: ResultInterface) -> set[int]:
    changed: set[int] = set()
    rows = read_csv(interface.artifact)
    if not rows:
        return changed

    for target in interface.targets:
        metric = interface.metrics.get(target.table_id)
        if not metric:
            continue
        target_rows = [
            row
            for row in rows
            if row.get("method", "").strip() == target.source_method
            and row.get("comm_delay_frames", "").strip() != ""
            and row.get("comp_delay_frames", "").strip() != ""
        ]
        by_comp: dict[int, dict[int, str]] = {}
        for row in target_rows:
            value = metric_value(row, metric)
            if value is None:
                continue
            try:
                comm = int(row["comm_delay_frames"])
                comp = int(row["comp_delay_frames"])
                formatted = fmt_percent(value)
            except ValueError:
                continue
            by_comp.setdefault(comp, {})[comm] = formatted

        for comp, comm_values in by_comp.items():
            for table_idx, table in enumerate(tables):
                if table.table_id != target.table_id:
                    continue
                before = [list(r) for r in table.rows]
                for comm, formatted in sorted(comm_values.items()):
                    set_cell(
                        [table],
                        table_id=target.table_id,
                        category=target.category,
                        method=target.doc_method,
                        comp=comp,
                        column=f"Comm{comm}",
                        value=formatted,
                    )
                if len(comm_values) == 6:
                    avg = sum(float(v) for v in comm_values.values()) / len(comm_values)
                    set_cell(
                        [table],
                        table_id=target.table_id,
                        category=target.category,
                        method=target.doc_method,
                        comp=comp,
                        column="AVG",
                        value=fmt_percent(avg),
                    )
                if target.category == "无延时" and comp == 0 and 0 in comm_values:
                    set_cell(
                        [table],
                        table_id=target.table_id,
                        category=target.category,
                        method=target.doc_method,
                        comp=0,
                        column="Comm0",
                        value=comm_values[0],
                    )
                if table.rows != before:
                    changed.add(table_idx)
    return changed


def apply_registered_result_interfaces(tables: list[Table]) -> set[int]:
    changed: set[int] = set()
    for interface in RESULT_INTERFACES:
        changed.update(apply_result_interface(tables, interface))
    return changed


def apply_data_v2xsim(tables: list[Table]) -> set[int]:
    changed: set[int] = set()
    path = REPO_ROOT / "runs" / "data_v2xsim_pipeline" / "results" / "data_v2xsim_latency_ap.csv"
    rows = read_csv(path)
    if not rows:
        return changed

    by_comp: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_comp.setdefault(int(row["comp_delay_frames"]), []).append(row)

    for metric, table_id in (("AP@0.5", "1.2.1"), ("AP@0.7", "1.2.2")):
        for comp, comp_rows in by_comp.items():
            lookup = {int(row["comm_delay_frames"]): row for row in comp_rows}
            vals = [float(lookup[comm][metric]) * 100.0 for comm in range(6) if comm in lookup]
            if len(vals) == 6:
                for table_idx, table in enumerate(tables):
                    before = [list(r) for r in table.rows]
                    set_cell(
                        [table],
                        table_id=table_id,
                        category="不同时延无补偿",
                        method="DATA",
                        comp=comp,
                        column="AVG",
                        value=fmt_percent(sum(vals) / len(vals)),
                    )
                    for comm, value in enumerate(vals):
                        set_cell(
                            [table],
                            table_id=table_id,
                            category="不同时延无补偿",
                            method="DATA",
                            comp=comp,
                            column=f"Comm{comm}",
                            value=fmt_percent(value),
                        )
                    if table.rows != before:
                        changed.add(table_idx)
        if rows:
            no_delay = next(
                (
                    row
                    for row in rows
                    if row["comm_delay_frames"] == "0" and row["comp_delay_frames"] == "0"
                ),
                None,
            )
            if no_delay is not None:
                for table_idx, table in enumerate(tables):
                    before = [list(r) for r in table.rows]
                    set_cell(
                        [table],
                        table_id=table_id,
                        category="无延时",
                        method="DATA",
                        comp=0,
                        column="Comm0",
                        value=fmt_percent(float(no_delay[metric]) * 100.0),
                    )
                    if table.rows != before:
                        changed.add(table_idx)
    return changed


def official_run_key(prefix: str, comm: int) -> str:
    return f"{prefix}_k{comm}"


def apply_dair_official(tables: list[Table]) -> set[int]:
    changed: set[int] = set()
    path = REPO_ROOT / "results" / "dair_v2x" / "tables" / "official_baselines_table.csv"
    rows = read_csv(path)
    if not rows:
        return changed
    lookup = {row["run"]: row for row in rows}

    specs = [
        ("2.1.1", "car_bev_ap_0_50"),
        ("2.1.2", "car_bev_ap_0_70"),
    ]
    for table_id, metric in specs:
        for table_idx, table in enumerate(tables):
            if table.table_id != table_id:
                continue
            before = [list(r) for r in table.rows]
            tclf_vals = []
            for comm in range(6):
                row = lookup.get(official_run_key("late_fusion_tclf", comm))
                if row and row.get(metric):
                    value = fmt_percent(row[metric])
                    tclf_vals.append(float(row[metric]))
                    set_cell(
                        [table],
                        table_id=table_id,
                        category="不同时延 + TCLF 补偿",
                        method="Late fusion + TCLF",
                        comp=0,
                        column=f"Comm{comm}",
                        value=value,
                    )
                    if comm == 0:
                        set_cell(
                            [table],
                            table_id=table_id,
                            category="无延时",
                            method="Late fusion + TCLF",
                            comp=0,
                            column="Comm0",
                            value=value,
                        )
            if len(tclf_vals) == 6:
                set_cell(
                    [table],
                    table_id=table_id,
                    category="不同时延 + TCLF 补偿",
                    method="Late fusion + TCLF",
                    comp=0,
                    column="AVG",
                    value=fmt_percent(sum(tclf_vals) / len(tclf_vals)),
                )

            no_comp_vals = []
            for comm in range(1, 6):
                row = lookup.get(official_run_key("late_fusion_no_comp", comm))
                if row and row.get(metric):
                    no_comp_vals.append(float(row[metric]))
                    set_cell(
                        [table],
                        table_id=table_id,
                        category="不同时延无补偿",
                        method="Late fusion w/o comp.",
                        comp=0,
                        column=f"Comm{comm}",
                        value=fmt_percent(row[metric]),
                    )
            if no_comp_vals:
                set_cell(
                    [table],
                    table_id=table_id,
                    category="不同时延无补偿",
                    method="Late fusion w/o comp.",
                    comp=0,
                    column="AVG",
                    value=fmt_percent(sum(no_comp_vals) / len(no_comp_vals)),
                )
            if table.rows != before:
                changed.add(table_idx)
    return changed


def apply_dair_proxy_iou(tables: list[Table]) -> set[int]:
    changed: set[int] = set()
    path = (
        REPO_ROOT
        / "results"
        / "dair_v2x"
        / "table2_table5_dair_proxy"
        / "dair_table2_table5_proxy_summary.csv"
    )
    rows = read_csv(path)
    if not rows:
        return changed
    label = {
        "No compensation (V2VNet-style delayed baseline)": "No compensation",
        "Dual compensation (V2VNet+DLPCM-style proxy)": "Dual compensation",
    }
    for row in rows:
        method = label.get(row["method"], row["method"])
        comp = row["comp_delay_frames"]
        for table_idx, table in enumerate(tables):
            if table.table_id != "2.1.3.3":
                continue
            before = [list(r) for r in table.rows]
            set_cell(
                [table],
                table_id="2.1.3.3",
                category="BEV IoU",
                method=method,
                comp=comp,
                column="AVG",
                value=fmt_iou(row["pred_iou_avg"]),
            )
            for comm in range(6):
                set_cell(
                    [table],
                    table_id="2.1.3.3",
                    category="BEV IoU",
                    method=method,
                    comp=comp,
                    column=f"Comm{comm}",
                    value=fmt_iou(row[f"pred_iou_comm{comm}"]),
                )
            if table.rows != before:
                changed.add(table_idx)
    return changed


def apply_incoming_results(tables: list[Table]) -> set[int]:
    changed: set[int] = set()
    rows = read_csv(INCOMING_CSV)
    for incoming in rows:
        table_id = incoming.get("table_id", "").strip()
        category = incoming.get("category", "").strip()
        method = incoming.get("method", "").strip()
        comp = incoming.get("comp", "").strip()
        column = incoming.get("column", "").strip()
        value = incoming.get("value", "").strip()
        if not all([table_id, category, method, comp, column, value]):
            continue
        for table_idx, table in enumerate(tables):
            before = [list(r) for r in table.rows]
            set_cell(
                [table],
                table_id=table_id,
                category=category,
                method=method,
                comp=comp,
                column=column,
                value=value,
            )
            if table.rows != before:
                changed.add(table_idx)
    return changed


def run_command(cmd: list[str], name: str, cwd: Path = REPO_ROOT) -> dict[str, object]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    return {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "log": str(log_path),
        "seconds": round(time.time() - started, 2),
    }


def run_known_refresh_jobs() -> list[dict[str, object]]:
    jobs = [
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_data_v2xsim_pipeline.py"),
            "--stage",
            "summarize",
        ],
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "make_dair_table2_table5_proxy.py"),
        ],
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "parse_dair_official_baseline_logs.py"),
            "--log-dir",
            str(REPO_ROOT / "results" / "dair_v2x" / "official_baselines" / "logs"),
            "--out",
            str(REPO_ROOT / "results" / "dair_v2x" / "official_baselines" / "summary_partial.csv"),
        ],
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "make_dair_revision_tables.py"),
        ],
    ]
    results = []
    for cmd in jobs:
        results.append(run_command(cmd, Path(cmd[1]).stem))
    return results


def load_task_status() -> dict[str, object]:
    if TASK_STATUS_JSON.exists():
        return json.loads(TASK_STATUS_JSON.read_text(encoding="utf-8"))
    return {"tasks": {}}


def save_task_status(status: dict[str, object]) -> None:
    TASK_STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    TASK_STATUS_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def artifact_has_rows(path: Path) -> bool:
    rows = read_csv(path)
    return bool(rows)


def interface_by_id() -> dict[str, ResultInterface]:
    return {interface.task_id: interface for interface in RESULT_INTERFACES}


def run_experiment_interfaces(*, force: bool = False) -> list[dict[str, object]]:
    tasks = ensure_task_interfaces_template()
    interfaces = interface_by_id()
    task_status = load_task_status()
    task_status.setdefault("tasks", {})
    results: list[dict[str, object]] = []

    for task in tasks:
        task_id = str(task.get("task_id", ""))
        interface = interfaces.get(task_id)
        if interface is None:
            continue
        enabled = bool(task.get("enabled", False))
        command = task.get("command", [])
        if not enabled:
            task_status["tasks"][task_id] = {
                "state": "disabled",
                "artifact": str(interface.artifact),
                "reason": "enabled=false in task_interfaces.json",
            }
            continue
        if not isinstance(command, list) or not command:
            task_status["tasks"][task_id] = {
                "state": "blocked_no_command",
                "artifact": str(interface.artifact),
                "reason": "enabled=true but command is empty",
            }
            continue
        if artifact_has_rows(interface.artifact) and not force:
            task_status["tasks"][task_id] = {
                "state": "artifact_ready",
                "artifact": str(interface.artifact),
                "reason": "artifact CSV already has rows; parser will fill markdown",
            }
            continue
        cwd = Path(str(task.get("cwd") or REPO_ROOT))
        result = run_command([str(part) for part in command], task_id, cwd=cwd)
        results.append(result)
        task_status["tasks"][task_id] = {
            "state": "completed" if result["returncode"] == 0 else "failed",
            "artifact": str(interface.artifact),
            "returncode": result["returncode"],
            "log": result["log"],
            "time_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
    save_task_status(task_status)
    return results


def classify_missing(table: Table, row: list[str], col_name: str) -> tuple[str, str, str]:
    method_col = col_index(table, "方法/补偿")
    category_col = col_index(table, "类别")
    comp_col = col_index(table, "Comp")
    method = row[method_col] if method_col is not None else ""
    category = row[category_col] if category_col is not None else ""
    comp = row[comp_col] if comp_col is not None else ""

    if category == "无延时" and col_name not in {"Comm0"}:
        return "structural_blank", "无延时行只有 Comm0 有实验含义，其他通信时延列不适用。", ""
    if category == "无延时" and col_name == "AVG":
        return "structural_blank", "无延时单点不需要 AVG；如论文要求可手工填同 Comm0。", ""

    interface = interface_for_missing(table.table_id, category, method)
    if interface is not None:
        if artifact_has_rows(interface.artifact):
            return "waiting_parse", "结果 artifact 已存在，等待下一轮解析回填。", str(interface.artifact)
        return (
            "waiting_interface",
            "脚本接口已实现；需要在 task_interfaces.json 中启用命令，或生成对应 artifact CSV。",
            str(interface.artifact),
        )

    if table.table_id.startswith("1.1") and "SyncNet" in method:
        return "blocked", "当前没有 V2X-Sim 上 V2VNet + SyncNet 的检测 AP pipeline 或结果 artifact。", ""
    if table.table_id.startswith("1.2") and ("Ours" in method or "SyncNet" in method):
        return "blocked", "当前只有 DATA 无补偿 latency grid；DATA + Ours / DATA + SyncNet 尚未实现。", ""
    if table.table_id.startswith("2.1") and table.table_id not in {"2.1.3.3"}:
        if comp in {"2", "4"}:
            return "blocked", "OpenDAIRV2X 官方 baseline 只支持消息/通信时延 k，当前没有计算时延 Comp=2/4 的 detector AP。", ""
        if "Ours" in method or "SyncNet" in method or "Detector" in method:
            return "blocked", "当前没有 DAIR detector AP 口径下的 Ours / SyncNet 补偿 pipeline 或结果 artifact。", ""
        if "w/o comp" in method and col_name == "Comm0":
            return "structural_blank", "无补偿 late fusion 在 k=0 与补偿版等价，官方未单独输出 no-comp k0。", ""
    if table.table_id.startswith("2.2"):
        return "blocked", "LRCP 本地缺少 DAIR yaml/checkpoint/flow 文件，尚无法产生 DAIR-V2X detector AP latency grid。", ""

    return "unknown", "未识别的空白项；可用 incoming_results.csv 按 table_id/category/method/comp/column/value 手工注入。", ""


def interface_for_missing(table_id: str, category: str, method: str) -> ResultInterface | None:
    for interface in RESULT_INTERFACES:
        for target in interface.targets:
            if target.table_id == table_id and target.category == category and target.doc_method == method:
                return interface
    return None


def missing_cells(tables: list[Table]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for table in tables:
        for row_idx, row in enumerate(table.rows):
            for col_idx, value in enumerate(row):
                if value != "-":
                    continue
                col_name = table.header[col_idx]
                status, reason, expected = classify_missing(table, row, col_name)
                method = row[col_index(table, "方法/补偿")] if col_index(table, "方法/补偿") is not None else ""
                category_col = col_index(table, "类别")
                metric_col = col_index(table, "指标")
                category = ""
                if category_col is not None:
                    category = row[category_col]
                elif metric_col is not None:
                    category = row[metric_col]
                comp = row[col_index(table, "Comp")] if col_index(table, "Comp") is not None else ""
                rows.append(
                    {
                        "table_id": table.table_id,
                        "title": table.title,
                        "row": str(row_idx + 1),
                        "category": category,
                        "method": method,
                        "comp": comp,
                        "column": col_name,
                        "status": status,
                        "reason": reason,
                        "expected_artifact": expected,
                    }
                )
    return rows


def write_csv_rows(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "table_id",
            "title",
            "row",
            "category",
            "method",
            "comp",
            "column",
            "status",
            "reason",
            "expected_artifact",
        ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ensure_incoming_template() -> None:
    if INCOMING_CSV.exists():
        return
    INCOMING_CSV.parent.mkdir(parents=True, exist_ok=True)
    with INCOMING_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["table_id", "category", "method", "comp", "column", "value"],
            lineterminator="\n",
        )
        writer.writeheader()


def one_pass(doc_path: Path, *, run_known: bool, run_experiments: bool, force_experiments: bool) -> dict[str, object]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_incoming_template()
    ensure_task_interfaces_template()
    write_artifact_templates()
    job_results = run_known_refresh_jobs() if run_known else []
    experiment_results = run_experiment_interfaces(force=force_experiments) if run_experiments else []

    lines = doc_path.read_text(encoding="utf-8").splitlines()
    tables = parse_tables(lines)
    changed_ids: set[int] = set()
    for fn in (
        apply_data_v2xsim,
        apply_dair_official,
        apply_dair_proxy_iou,
        apply_registered_result_interfaces,
        apply_incoming_results,
    ):
        changed_ids.update(fn(tables))
    if changed_ids:
        lines = rewrite_tables(lines, tables, changed_ids)
        doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        lines = doc_path.read_text(encoding="utf-8").splitlines()
        tables = parse_tables(lines)

    missing = missing_cells(tables)
    write_csv_rows(MISSING_CSV, missing)
    counts: dict[str, int] = {}
    for row in missing:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    actionable_missing = [row for row in missing if row["status"] != "structural_blank"]
    actionable_counts: dict[str, int] = {}
    for row in actionable_missing:
        actionable_counts[row["status"]] = actionable_counts.get(row["status"], 0) + 1
    status = {
        "time_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "doc": str(doc_path),
        "changed_tables": len(changed_ids),
        "remaining_blank_cells": len(missing),
        "remaining_actionable_blank_cells": len(actionable_missing),
        "remaining_by_status": counts,
        "remaining_actionable_by_status": actionable_counts,
        "missing_csv": str(MISSING_CSV),
        "incoming_csv": str(INCOMING_CSV),
        "task_interfaces_json": str(TASK_INTERFACES_JSON),
        "task_status_json": str(TASK_STATUS_JSON),
        "job_results": job_results,
        "experiment_results": experiment_results,
    }
    STATUS_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--once", action="store_true", help="Run one scan/fill pass and exit.")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep running until no actionable experiment-result '-' cells remain.",
    )
    parser.add_argument("--sleep", type=int, default=600, help="Sleep seconds between loop passes.")
    parser.add_argument("--run-known", action="store_true", help="Run safe existing summarizer/parser jobs before filling.")
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run enabled experiment commands from task_interfaces.json before filling.",
    )
    parser.add_argument(
        "--force-experiments",
        action="store_true",
        help="Run enabled experiment commands even if their artifact CSV already has rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.once and not args.loop:
        args.once = True
    while True:
        status = one_pass(
            args.doc,
            run_known=args.run_known,
            run_experiments=args.run_experiments,
            force_experiments=args.force_experiments,
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
        if args.once:
            return
        if status["remaining_actionable_blank_cells"] == 0:
            return
        time.sleep(max(args.sleep, 1))


if __name__ == "__main__":
    main()
