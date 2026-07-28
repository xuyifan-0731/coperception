#!/usr/bin/env python3
"""Insert Comp=1/3/5 results into 260704 result tables."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "260704结果.md"


def fmt(x: float | str) -> str:
    if isinstance(x, str):
        if x in {"", "-"}:
            return "-"
        x = float(x)
    return f"{x:.2f}"


def avg(vals: list[str]) -> str:
    if any(v == "-" for v in vals):
        return "-"
    return fmt(sum(float(v) for v in vals) / len(vals))


def read_rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_grid_csv(path: str, method_col: str, method_name: str, comp_col: str, comm_col: str, val_col: str, scale: float = 1.0) -> dict[int, dict[int, str]]:
    out: dict[int, dict[int, str]] = {}
    for row in read_rows(path):
        if method_col and row.get(method_col) != method_name:
            continue
        c = int(row[comp_col])
        m = int(row[comm_col])
        v = row[val_col]
        if v == "":
            continue
        out.setdefault(c, {})[m] = fmt(float(v) * scale)
    return out


def parse_table(lines: list[str], marker: str) -> tuple[int, int, list[list[str]]]:
    start = next(i for i, line in enumerate(lines) if marker in line)
    i = start + 1
    while i < len(lines) and not lines[i].startswith("| "):
        i += 1
    table_start = i
    rows: list[list[str]] = []
    i += 2
    while i < len(lines) and lines[i].startswith("|"):
        rows.append([cell.strip() for cell in lines[i].strip().strip("|").split("|")])
        i += 1
    return table_start, i, rows


def old_values(rows: list[list[str]], method: str) -> dict[int, dict[int, str]]:
    out: dict[int, dict[int, str]] = {}
    for cells in rows:
        if len(cells) < 10 or cells[1] != method:
            continue
        comp = int(cells[2])
        out[comp] = {i: cells[4 + i] for i in range(6)}
    return out


def merge_values(*dicts: dict[int, dict[int, str]]) -> dict[int, dict[int, str]]:
    out: dict[int, dict[int, str]] = {}
    for d in dicts:
        for comp, vals in d.items():
            out.setdefault(comp, {}).update(vals)
    return out


def ensure_comp_rows(values: dict[int, dict[int, str]], comps: list[int]) -> dict[int, dict[int, str]]:
    out = {comp: dict(comm_vals) for comp, comm_vals in values.items()}
    for comp in comps:
        out.setdefault(comp, {i: "-" for i in range(6)})
    return out


def table_row(category: str, method: str, comp: int, vals_by_comm: dict[int, str]) -> str:
    vals = [vals_by_comm.get(i, "-") for i in range(6)]
    return f"| {category} | {method} | {comp} | {avg(vals)} | " + " | ".join(vals) + " |"


def replace_standard_table(lines: list[str], marker: str, specs: list[tuple[str, str, dict[int, dict[int, str]]]]) -> list[str]:
    start, end, rows = parse_table(lines, marker)
    header = lines[start:start + 2]
    no_delay = [line for line in lines[start + 2:end] if line.startswith("| 无延时 ")]
    new_rows: list[str] = []
    new_rows.extend(no_delay)
    for category, method, values in specs:
        for comp in range(6):
            if comp in values:
                new_rows.append(table_row(category, method, comp, values[comp]))
    return lines[:start] + header + new_rows + lines[end:]


def replace_paper_table(lines: list[str], marker: str, methods: list[tuple[str, str]]) -> list[str]:
    start, end, rows = parse_table(lines, marker)
    header = lines[start:start + 2]
    no_delay = [line for line in lines[start + 2:end] if line.startswith("| 无延时 ")]
    old_by_method = {method: old_values(rows, method) for _, method in methods}
    new_rows = list(no_delay)
    for category, method in methods:
        vals = old_by_method[method]
        for comp in range(6):
            row_vals = vals.get(comp, {i: "-" for i in range(6)})
            new_rows.append(table_row(category, method, comp, row_vals))
    return lines[:start] + header + new_rows + lines[end:]


def traf_values(path: str, ap_col: str) -> dict[int, dict[int, str]]:
    rows = read_rows(path)
    if not rows:
        return {}
    comp = int(int(rows[0]["delay_ego_ms"]) / 100)
    out = {comp: {}}
    for idx, row in enumerate(rows):
        out[comp][idx] = fmt(row[ap_col])
    return out


def replace_traf_table(lines: list[str], marker: str, ap_col: str) -> list[str]:
    start, end, rows = parse_table(lines, marker)
    header = lines[start:start + 2]
    method_specs = [
        ("无时延补偿", "TraF-Align", [
            "results/dair_v2x_seq/trafalign_baseline_comp1_spconv2121.csv",
            "results/dair_v2x_seq/trafalign_baseline_comp3_spconv2121.csv",
            "results/dair_v2x_seq/trafalign_baseline_comp5_spconv2121.csv",
        ]),
        ("不同时延 + 我们方法补偿", "TraF-Align + Ours feature T10-N10", [
            "results/dair_v2x_seq/trafalign_ours_t10n10_comp1_spconv2121_comp135.csv",
            "results/dair_v2x_seq/trafalign_ours_t10n10_comp3_spconv2121_comp135.csv",
            "results/dair_v2x_seq/trafalign_ours_t10n10_comp5_spconv2121_comp135.csv",
        ]),
        ("不同时延 + 联合训练补偿", "TraF-Align + Ours joint delay-comp only", [
            "results/dair_v2x_seq/trafalign_joint_delay_comp_only_comp1_spconv2121.csv",
            "results/dair_v2x_seq/trafalign_joint_delay_comp_only_comp3_spconv2121.csv",
            "results/dair_v2x_seq/trafalign_joint_delay_comp_only_comp5_spconv2121.csv",
        ]),
        ("不同时延 + 联合训练补偿改进", "TraF-Align + Ours joint delay-comp adaptive scale", [
            "results/dair_v2x_seq/trafalign_joint_delay_comp_only_adaptive2to4_comp1_spconv2121.csv",
            "results/dair_v2x_seq/trafalign_joint_delay_comp_only_adaptive2to4_comp3_spconv2121.csv",
            "results/dair_v2x_seq/trafalign_joint_delay_comp_only_adaptive2to4_comp5_spconv2121.csv",
        ]),
        ("不同时延 + 联合训练补偿推荐", "TraF-Align + Ours adaptive scale + Comp0 bypass", [
            "results/dair_v2x_seq/trafalign_joint_delay_comp_only_adaptive2to4_comp1_spconv2121.csv",
            "results/dair_v2x_seq/trafalign_joint_delay_comp_only_adaptive2to4_comp3_spconv2121.csv",
            "results/dair_v2x_seq/trafalign_joint_delay_comp_only_adaptive2to4_comp5_spconv2121.csv",
        ]),
    ]
    old: dict[str, dict[int, dict[int, str]]] = {}
    for cells in rows:
        if len(cells) < 10:
            continue
        old.setdefault(cells[1], {})[int(cells[2])] = {i: cells[4 + i] for i in range(6)}
    new_rows: list[str] = []
    for category, method, files in method_specs:
        vals = old.get(method, {})
        for path in files:
            vals = merge_values(vals, traf_values(path, ap_col))
        for comp in range(6):
            if comp in vals:
                new_rows.append(f"| {category} | {method} | {comp} | {avg([vals[comp].get(i, '-') for i in range(6)])} | " + " | ".join(vals[comp].get(i, "-") for i in range(6)) + " |")
    for cells in rows:
        if len(cells) >= 3 and "Comp2/4 推荐" in cells[0]:
            new_rows.append("| " + " | ".join(cells) + " |")
    return lines[:start] + header + new_rows + lines[end:]


def main() -> None:
    lines = MD.read_text(encoding="utf-8").splitlines()

    paper_methods_v2v = [
        ("不同时延无补偿", "V2VNet"),
        ("不同时延 + 我们方法补偿", "V2VNet + Ours"),
        ("不同时延 + SyncNet 补偿", "V2VNet + SyncNet"),
    ]
    paper_methods_disco = [
        ("不同时延无补偿", "DiscoNet"),
        ("不同时延 + 我们方法补偿", "DiscoNet + Ours"),
        ("不同时延 + SyncNet 补偿", "SyncNet w/ Comp."),
    ]
    for marker in ["#### 1.1.1 AP@0.5", "#### 1.1.2 AP@0.7"]:
        lines = replace_paper_table(lines, marker, paper_methods_v2v)
    for marker in ["#### 1.3.1 AP@0.5", "#### 1.3.2 AP@0.7"]:
        lines = replace_paper_table(lines, marker, paper_methods_disco)

    for marker, ap in [("#### 1.2.1 AP@0.5", "AP@0.5"), ("#### 1.2.2 AP@0.7", "AP@0.7")]:
        _, _, rows = parse_table(lines, marker)
        lines = replace_standard_table(lines, marker, [
            ("不同时延无补偿", "DATA", merge_values(old_values(rows, "DATA"), load_grid_csv("runs/data_v2xsim_pipeline_comp135/results/data_v2xsim_latency_ap.csv", "method", "DATA", "comp_delay_frames", "comm_delay_frames", ap, 100.0))),
            ("不同时延 + 我们方法补偿", "DATA + Ours", merge_values(old_values(rows, "DATA + Ours"), load_grid_csv("results/260615_fill/artifacts/v2xsim_data_compensation_ap_comp135_20260716.csv", "method", "DATA + Ours", "comp_delay_frames", "comm_delay_frames", ap))),
            ("不同时延 + 联合训练补偿", "DATA + Ours joint IFAM residual", merge_values(old_values(rows, "DATA + Ours joint IFAM residual"), load_grid_csv("results/260615_fill/artifacts/v2xsim_data_ifam_joint_delay_comp_scale01_comp135_20260716.csv", "method", "DATA + Ours joint IFAM residual", "comp_delay_frames", "comm_delay_frames", ap))),
            ("不同时延 + SyncNet 补偿", "DATA + SyncNet", merge_values(old_values(rows, "DATA + SyncNet"), load_grid_csv("results/260615_fill/artifacts/v2xsim_data_compensation_ap_comp135_20260716.csv", "method", "DATA + SyncNet", "comp_delay_frames", "comm_delay_frames", ap))),
        ])

    for marker, ap in [
        ("#### 2.1.1 BEV AP@0.5", "BEV AP@0.5"),
        ("#### 2.1.2 BEV AP@0.7", "BEV AP@0.7"),
        ("##### 2.1.3.1 BEV AP@0.5", "BEV AP@0.5"),
        ("##### 2.1.3.2 BEV AP@0.7", "BEV AP@0.7"),
    ]:
        _, _, rows = parse_table(lines, marker)
        ours_method = "Detector + Ours" if "2.1.3" in marker else "Late fusion + Ours"
        base_method = "Detector" if "2.1.3" in marker else "Late fusion"
        tclf_method = "Detector + TCLF" if "2.1.3" in marker else "Late fusion + TCLF"
        lines = replace_standard_table(lines, marker, [
            ("不同时延无补偿", base_method, merge_values(load_grid_csv("results/260615_fill/artifacts/dair_late_fusion_no_comp_strict_20260704_rechecked.csv", "", "", "comp_delay_frames", "comm_delay_frames", ap), load_grid_csv("results/260615_fill/artifacts/dair_late_fusion_no_comp_strict_comp135_20260716.csv", "", "", "comp_delay_frames", "comm_delay_frames", ap))),
            ("不同时延 + 我们方法补偿", ours_method, merge_values(load_grid_csv("results/260615_fill/artifacts/dair_late_fusion_ours_history2_rf_trainonly_strict_20260714.csv", "", "", "comp_delay_frames", "comm_delay_frames", ap), load_grid_csv("results/260615_fill/artifacts/dair_late_fusion_ours_history2_rf_trainonly_strict_comp135_20260716.csv", "", "", "comp_delay_frames", "comm_delay_frames", ap))),
            ("不同时延 + 对比方法补偿", tclf_method, merge_values(load_grid_csv("results/260615_fill/artifacts/dair_late_fusion_tclf_receiver_strict_20260704_rechecked.csv", "", "", "comp_delay_frames", "comm_delay_frames", ap), load_grid_csv("results/260615_fill/artifacts/dair_late_fusion_tclf_receiver_strict_comp135_20260716.csv", "", "", "comp_delay_frames", "comm_delay_frames", ap))),
        ])

    for marker, ap in [("#### 2.2.1 AP@0.5", "ap50"), ("#### 2.2.2 AP@0.7", "ap70")]:
        _, _, rows = parse_table(lines, marker)
        lines = replace_standard_table(lines, marker, [
            ("不同时延无补偿", "DATA", merge_values(load_grid_csv("results/260615_fill/artifacts/dair_data_strict_dual_delay_none_ours_20260706.csv", "method", "DATA w/o compensation", "comp_delay", "comm_delay", ap), load_grid_csv("results/260615_fill/artifacts/dair_data_strict_dual_delay_none_ours_comp135_20260716.csv", "method", "DATA w/o compensation", "comp_delay", "comm_delay", ap))),
            ("不同时延 + DATA fork 输入级补偿", "DATA + pose-warp", merge_values(load_grid_csv("results/260615_fill/artifacts/dair_data_strict_dual_delay_none_ours_20260706.csv", "method", "DATA + Ours", "comp_delay", "comm_delay", ap), load_grid_csv("results/260615_fill/artifacts/dair_data_strict_dual_delay_none_ours_comp135_20260716.csv", "method", "DATA + Ours", "comp_delay", "comm_delay", ap))),
            ("不同时延 + 我们方法特征级补偿", "DATA + Ours IFAM-feature T10-N10", merge_values(load_grid_csv("results/260615_fill/artifacts/dair_data_ifam_feature_t10n10_ours_20260707.csv", "method", "DATA + Ours IFAM-feature T10-N10", "comp_delay", "comm_delay", ap, 100.0), load_grid_csv("results/260615_fill/artifacts/dair_data_ifam_feature_t10n10_ours_comp135_20260716.csv", "method", "DATA + Ours IFAM-feature T10-N10", "comp_delay", "comm_delay", ap, 100.0))),
            ("不同时延 + 联合训练补偿", "DATA + Ours IFAM joint delay-comp", merge_values(load_grid_csv("results/260615_fill/artifacts/dair_data_ifam_joint_delay_comp_20260713.csv", "method", "DATA + Ours IFAM joint delay-comp", "comp_delay", "comm_delay", ap, 100.0), load_grid_csv("results/260615_fill/artifacts/dair_data_ifam_joint_delay_comp_comp135_20260716.csv", "method", "DATA + Ours IFAM joint delay-comp", "comp_delay", "comm_delay", ap, 100.0))),
        ])

    for marker, ap in [("#### 2.3.1 BEV AP@0.5", "BEV AP@0.5"), ("#### 2.3.2 BEV AP@0.7", "BEV AP@0.7")]:
        _, _, rows = parse_table(lines, marker)
        part_vals = {}
        for part in (ROOT / "results/260615_fill/artifacts/dair_lrcp_strict_dual_delay_parts_comp135_20260716").glob("*.csv"):
            part_vals = merge_values(part_vals, load_grid_csv(str(part.relative_to(ROOT)), "", "", "comp_delay_frames", "comm_delay_frames", ap))
        lines = replace_standard_table(lines, marker, [
            ("不同时延无补偿", "LRCP w/o compensation", ensure_comp_rows(merge_values(old_values(rows, "LRCP w/o compensation"), part_vals), [1, 3, 5])),
            ("不同时延 + 我们方法补偿", "LRCP + Ours", ensure_comp_rows(old_values(rows, "LRCP + Ours"), [1, 3, 5])),
            ("不同时延 + SyncNet 补偿", "LRCP + SyncNet", ensure_comp_rows(old_values(rows, "LRCP + SyncNet"), [1, 3, 5])),
        ])

    lines = replace_traf_table(lines, "#### 2.4.1 AP@0.5", "AP@0.5")
    lines = replace_traf_table(lines, "#### 2.4.2 AP@0.7", "AP@0.7")

    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
