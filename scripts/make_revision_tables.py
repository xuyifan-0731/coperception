#!/usr/bin/env python3
"""Build paper-facing CSV/LaTeX table drafts from local revision artifacts."""

from __future__ import annotations

import argparse
import csv
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


def fmt_float(value: str, digits: int = 3) -> str:
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def latex_table(
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
    colspec: str,
) -> str:
    body = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    body.extend(" & ".join(row) + " \\\\" for row in rows)
    body.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(body)


def make_profile_table(result_root: Path, out_dir: Path) -> str:
    rows = read_csv(result_root / "vspm_profile.csv")
    out_rows = []
    latex_rows = []
    for row in rows:
        out = {
            "model": row["model"],
            "params": row.get("params", ""),
            "state_mib_per_agent": fmt_float(row.get("state_mib_per_agent", ""), 3),
            "mean_ms": fmt_float(row.get("mean_ms", ""), 3),
            "min_ms": fmt_float(row.get("min_ms", ""), 3),
            "max_ms": fmt_float(row.get("max_ms", ""), 3),
        }
        out_rows.append(out)
        latex_rows.append(
            [
                out["model"].replace("_", "\\_"),
                out["params"],
                out["state_mib_per_agent"],
                out["mean_ms"],
            ]
        )
    fields = ["model", "params", "state_mib_per_agent", "mean_ms", "min_ms", "max_ms"]
    write_csv(out_dir / "vspm_profile_table.csv", out_rows, fields)
    return latex_table(
        "VSPM state size and synthetic rollout latency.",
        "tab:vspm-profile",
        ["Variant", "Params", "State MiB/agent", "Mean ms"],
        latex_rows,
        "lrrr",
    )


def make_scalability_table(result_root: Path, out_dir: Path) -> str:
    rows = read_csv(result_root / "scalability_profile.csv")
    out_rows = []
    latex_rows = []
    for row in rows:
        out = {
            "num_agents": row["num_agents"],
            "fp32_hc_state_mib": fmt_float(row["fp32_hc_state_mib"], 2),
            "gru_ds64_fp16_state_mib": fmt_float(row["gru_ds64_fp16_state_mib"], 2),
            "cache_capacity": row["cache_capacity"],
            "evictions": row["evictions"],
            "hit_rate": fmt_float(row["hit_rate"], 2),
        }
        out_rows.append(out)
        latex_rows.append(
            [
                out["num_agents"],
                out["fp32_hc_state_mib"],
                out["gru_ds64_fp16_state_mib"],
                out["evictions"],
                out["hit_rate"],
            ]
        )
    fields = [
        "num_agents",
        "fp32_hc_state_mib",
        "gru_ds64_fp16_state_mib",
        "cache_capacity",
        "evictions",
        "hit_rate",
    ]
    write_csv(out_dir / "scalability_table.csv", out_rows, fields)
    return latex_table(
        "Receiver cache state memory under increasing agent count.",
        "tab:cache-scalability",
        ["Agents", "ConvLSTM MiB", "MST MiB", "Evictions", "Hit rate"],
        latex_rows,
        "rrrrr",
    )


def make_training_table(result_root: Path, out_dir: Path) -> str:
    rows = read_csv(result_root / "pipeline_latest_metrics.csv")
    out_rows = []
    latex_rows = []
    for row in rows:
        out = {
            "run": row["run"],
            "epoch": row["epoch"],
            "step": row["step"],
            "IoU": fmt_float(row["IoU"], 4),
            "Delta_IoU": fmt_float(row["Delta_IoU"], 4),
            "Dynamic_IoU": fmt_float(row["Dynamic_IoU"], 4),
        }
        out_rows.append(out)
        latex_rows.append(
            [
                out["run"].replace("_", "\\_"),
                out["epoch"],
                out["step"],
                out["IoU"],
                out["Delta_IoU"],
                out["Dynamic_IoU"],
            ]
        )
    fields = ["run", "epoch", "step", "IoU", "Delta_IoU", "Dynamic_IoU"]
    write_csv(out_dir / "active_training_table.csv", out_rows, fields)
    if not latex_rows:
        return ""
    return latex_table(
        "Current validation progress for running VSPM sensitivity experiments.",
        "tab:vspm-progress",
        ["Run", "Epoch", "Step", "IoU", "$\\Delta$IoU", "Dynamic IoU"],
        latex_rows,
        "lrrrrr",
    )


def make_robustness_table(result_root: Path, out_dir: Path) -> str:
    source = result_root / "vspm_robustness_T10_n5_gpu.csv"
    if not source.exists():
        source = result_root / "vspm_robustness_T10_n5.csv"
    rows = read_csv(source)
    out_rows = []
    latex_rows = []
    for row in rows:
        out = {
            "condition": row["condition"],
            "packet_loss": fmt_float(row["packet_loss"], 3),
            "dropout_rate": fmt_float(row["dropout_rate"], 3),
            "false_positive_rate": fmt_float(row["false_positive_rate"], 4),
            "samples": row["samples"],
            "mean_iou": fmt_float(row["mean_iou"], 4),
            "mean_baseline_iou": fmt_float(row["mean_baseline_iou"], 4),
            "mean_delta_iou": fmt_float(row["mean_delta_iou"], 4),
            "mean_dynamic_iou": fmt_float(row["mean_dynamic_iou"], 4),
        }
        out_rows.append(out)
        latex_rows.append(
            [
                out["condition"].replace("_", "\\_"),
                out["samples"],
                out["mean_iou"],
                out["mean_baseline_iou"],
                out["mean_delta_iou"],
                out["mean_dynamic_iou"],
            ]
        )
    fields = [
        "condition",
        "packet_loss",
        "dropout_rate",
        "false_positive_rate",
        "samples",
        "mean_iou",
        "mean_baseline_iou",
        "mean_delta_iou",
        "mean_dynamic_iou",
    ]
    write_csv(out_dir / "robustness_table.csv", out_rows, fields)
    if not latex_rows:
        return ""
    return latex_table(
        "VSPM robustness under local packet loss and BEV perturbations.",
        "tab:vspm-robustness",
        ["Condition", "Samples", "IoU", "Baseline IoU", "$\\Delta$IoU", "Dynamic IoU"],
        latex_rows,
        "lrrrrr",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("results/revision_plan"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision_tables"))
    args = parser.parse_args()

    parts = [
        make_profile_table(args.result_root, args.out_dir),
        make_scalability_table(args.result_root, args.out_dir),
        make_training_table(args.result_root, args.out_dir),
        make_robustness_table(args.result_root, args.out_dir),
    ]
    latex_path = args.out_dir / "revision_tables.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text("\n".join(part for part in parts if part), encoding="utf-8")
    print(f"wrote={latex_path}")


if __name__ == "__main__":
    main()
