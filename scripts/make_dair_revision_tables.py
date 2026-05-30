#!/usr/bin/env python3
"""Build DAIR-V2X revision table drafts from completed local artifacts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def fmt(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def latex_escape(text: object) -> str:
    return str(text).replace("_", "\\_")


def latex_table(caption: str, label: str, headers: list[str], rows: list[list[str]], colspec: str) -> str:
    if not rows:
        return ""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def official_table(result_root: Path, out_dir: Path) -> str:
    rows = read_csv(result_root / "official_baselines" / "summary_partial.csv")
    fields = [
        "run",
        "status",
        "car_3d_ap_0_50",
        "car_3d_ap_0_70",
        "car_bev_ap_0_50",
        "car_bev_ap_0_70",
        "avg_comm_bytes",
    ]
    out_rows: list[dict[str, object]] = []
    latex_rows: list[list[str]] = []
    for row in rows:
        out = {
            "run": row["run"],
            "status": row["status"],
            "car_3d_ap_0_50": fmt(row.get("car_3d_ap_0_50", ""), 2),
            "car_3d_ap_0_70": fmt(row.get("car_3d_ap_0_70", ""), 2),
            "car_bev_ap_0_50": fmt(row.get("car_bev_ap_0_50", ""), 2),
            "car_bev_ap_0_70": fmt(row.get("car_bev_ap_0_70", ""), 2),
            "avg_comm_bytes": fmt(row.get("avg_comm_bytes", ""), 2),
        }
        out_rows.append(out)
        if row["status"] == "complete":
            latex_rows.append(
                [
                    latex_escape(out["run"]),
                    out["car_3d_ap_0_50"],
                    out["car_3d_ap_0_70"],
                    out["car_bev_ap_0_50"],
                    out["car_bev_ap_0_70"],
                    out["avg_comm_bytes"],
                ]
            )
    write_csv(out_dir / "official_baselines_table.csv", out_rows, fields)
    return latex_table(
        "DAIR-V2X official detection baselines completed so far.",
        "tab:dair-official-baselines",
        ["Run", "3D AP@0.5", "3D AP@0.7", "BEV AP@0.5", "BEV AP@0.7", "Comm. B"],
        latex_rows,
        "lrrrrr",
    )


def delay_table(result_root: Path, out_dir: Path) -> str:
    configs = [
        ("T10_n10_ckpt15000", result_root / "our_method" / "full_T10_n10_ckpt15000"),
        ("T10_n5_ckpt16000", result_root / "our_method" / "final_T10_n5_ckpt16000"),
    ]
    fields = [
        "setting",
        "mode",
        "cases",
        "mean_pred_iou",
        "mean_copy_last_iou",
        "mean_delta_iou",
        "mean_dynamic_iou",
        "worst_pred_iou",
        "best_pred_iou",
    ]
    out_rows: list[dict[str, object]] = []
    latex_rows: list[list[str]] = []
    for setting, root in configs:
        for path in sorted(root.glob("delay_grid_*.csv")):
            rows = read_csv(path)
            if not rows:
                continue
            mode = path.stem.replace("delay_grid_", "")
            pred = [as_float(r["pred_iou"]) for r in rows]
            base = [as_float(r["copy_last_iou"]) for r in rows]
            delta = [as_float(r["delta_iou"]) for r in rows]
            dyn = [as_float(r["dynamic_iou"]) for r in rows]
            out = {
                "setting": setting,
                "mode": mode,
                "cases": len(rows),
                "mean_pred_iou": fmt(sum(pred) / len(pred), 4),
                "mean_copy_last_iou": fmt(sum(base) / len(base), 4),
                "mean_delta_iou": fmt(sum(delta) / len(delta), 4),
                "mean_dynamic_iou": fmt(sum(dyn) / len(dyn), 4),
                "worst_pred_iou": fmt(min(pred), 4),
                "best_pred_iou": fmt(max(pred), 4),
            }
            out_rows.append(out)
            latex_rows.append(
                [
                    latex_escape(setting),
                    latex_escape(mode),
                    str(out["cases"]),
                    out["mean_pred_iou"],
                    out["mean_copy_last_iou"],
                    out["mean_delta_iou"],
                    out["mean_dynamic_iou"],
                ]
            )
    write_csv(out_dir / "delay_system_ablation_table.csv", out_rows, fields)
    return latex_table(
        "DAIR-V2X delay and system ablation summary.",
        "tab:dair-delay-system",
        ["Setting", "Mode", "Cases", "IoU", "Copy", "$\\Delta$IoU", "Dyn. IoU"],
        latex_rows,
        "llrrrrr",
    )


def training_tables(summary_root: Path, out_dir: Path) -> str:
    rows = read_csv(summary_root / "training_summary.csv")
    selected = []
    for row in rows:
        group = row["experiment_group"]
        if group.startswith("mst_ablation") or group.startswith("vspm_sensitivity") or group.startswith("our_method"):
            selected.append(row)
    fields = [
        "experiment_group",
        "best_step",
        "best_IoU",
        "best_Delta_IoU",
        "best_Dynamic_IoU",
        "final_IoU",
    ]
    out_rows: list[dict[str, object]] = []
    latex_rows: list[list[str]] = []
    for row in selected:
        out = {
            "experiment_group": row["experiment_group"],
            "best_step": row["best_step"],
            "best_IoU": fmt(row["best_IoU"], 4),
            "best_Delta_IoU": fmt(row["best_Delta_IoU"], 4),
            "best_Dynamic_IoU": fmt(row["best_Dynamic_IoU"], 4),
            "final_IoU": fmt(row["final_IoU"], 4),
        }
        out_rows.append(out)
        latex_rows.append(
            [
                latex_escape(out["experiment_group"]),
                str(out["best_step"]),
                out["best_IoU"],
                out["best_Delta_IoU"],
                out["best_Dynamic_IoU"],
                out["final_IoU"],
            ]
        )
    write_csv(out_dir / "training_summary_table.csv", out_rows, fields)
    return latex_table(
        "DAIR-V2X VSPM training and ablation summary.",
        "tab:dair-vspm-training",
        ["Run", "Best step", "Best IoU", "$\\Delta$IoU", "Dyn. IoU", "Final IoU"],
        latex_rows,
        "lrrrrr",
    )


def robustness_table(result_root: Path, out_dir: Path) -> str:
    configs = [
        ("T10_n5_ckpt16000", result_root / "robustness_T10_n5_ckpt16000.csv"),
        ("T10_n10_ckpt15000", result_root / "robustness_T10_n10_ckpt15000.csv"),
    ]
    fields = [
        "setting",
        "condition",
        "samples",
        "steps",
        "mean_iou",
        "mean_baseline_iou",
        "mean_delta_iou",
        "mean_dynamic_iou",
    ]
    out_rows: list[dict[str, object]] = []
    latex_rows: list[list[str]] = []
    for setting, path in configs:
        for row in read_csv(path):
            out = {
                "setting": setting,
                "condition": row["condition"],
                "samples": row["samples"],
                "steps": row["steps"],
                "mean_iou": fmt(row["mean_iou"], 4),
                "mean_baseline_iou": fmt(row["mean_baseline_iou"], 4),
                "mean_delta_iou": fmt(row["mean_delta_iou"], 4),
                "mean_dynamic_iou": fmt(row["mean_dynamic_iou"], 4),
            }
            out_rows.append(out)
            if row["condition"] in {"clean", "packet_loss_0.3", "packet_loss_0.5", "dropout_0.2", "false_positive_0.005"}:
                latex_rows.append(
                    [
                        latex_escape(setting),
                        latex_escape(row["condition"]),
                        out["mean_iou"],
                        out["mean_baseline_iou"],
                        out["mean_delta_iou"],
                        out["mean_dynamic_iou"],
                    ]
                )
    write_csv(out_dir / "robustness_table.csv", out_rows, fields)
    return latex_table(
        "DAIR-V2X VSPM robustness summary.",
        "tab:dair-vspm-robustness",
        ["Setting", "Condition", "IoU", "Copy", "$\\Delta$IoU", "Dyn. IoU"],
        latex_rows,
        "llrrrr",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("results/dair_v2x"))
    parser.add_argument("--summary-root", type=Path, default=Path("results/dair_v2x/summary"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/dair_v2x/tables"))
    args = parser.parse_args()

    parts = [
        official_table(args.result_root, args.out_dir),
        delay_table(args.result_root, args.out_dir),
        training_tables(args.summary_root, args.out_dir),
        robustness_table(args.result_root, args.out_dir),
    ]
    tex = "\n".join(part for part in parts if part)
    tex_path = args.out_dir / "dair_revision_tables.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding="utf-8")
    print(f"wrote={tex_path}")


if __name__ == "__main__":
    main()
