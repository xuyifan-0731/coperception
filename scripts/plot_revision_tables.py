#!/usr/bin/env python3
"""Plot revision-preparation figures from local CSV summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_training(summary_root: Path, out_dir: Path) -> None:
    rows = read_csv(summary_root / "training_timeseries.csv")
    if not rows:
        return
    by_group: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        group = row["experiment_group"]
        if group.startswith("smoke"):
            continue
        by_group.setdefault(group, []).append(row)
    if not by_group:
        return
    plt.figure(figsize=(6.0, 3.6))
    for group, group_rows in sorted(by_group.items()):
        group_rows = sorted(group_rows, key=lambda r: int(r["step"]))
        steps = [int(r["step"]) for r in group_rows]
        ious = [float(r["IoU"]) for r in group_rows]
        plt.plot(steps, ious, marker="o", linewidth=1.8, label=group.split("/")[-1])
    plt.xlabel("Training step")
    plt.ylabel("Validation IoU")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    savefig(out_dir / "vspm_training_iou.pdf")


def plot_profile(result_root: Path, out_dir: Path) -> None:
    rows = read_csv(result_root / "vspm_profile.csv")
    if not rows:
        return
    names = [r["model"] for r in rows]
    state = [float(r["state_mib_per_agent"]) for r in rows]
    latency = [float(r["mean_ms"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    x = list(range(len(names)))
    ax1.bar([v - 0.18 for v in x], state, width=0.36, label="State MiB/agent")
    ax1.set_ylabel("State MiB/agent")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot([v + 0.18 for v in x], latency, color="#b23a48", marker="o", label="Mean ms")
    ax2.set_ylabel("Mean rollout latency (ms)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")
    savefig(out_dir / "vspm_profile_tradeoff.pdf")


def plot_scalability(result_root: Path, out_dir: Path) -> None:
    rows = read_csv(result_root / "scalability_profile.csv")
    if not rows:
        return
    agents = [int(r["num_agents"]) for r in rows]
    conv = [float(r["fp32_hc_state_mib"]) for r in rows]
    mst = [float(r["gru_ds64_fp16_state_mib"]) for r in rows]
    plt.figure(figsize=(5.8, 3.5))
    plt.plot(agents, conv, marker="o", linewidth=1.8, label="ConvLSTM h+c FP32")
    plt.plot(agents, mst, marker="s", linewidth=1.8, label="MST DS64 FP16")
    plt.xlabel("Number of remote agents")
    plt.ylabel("Cached state memory (MiB)")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    savefig(out_dir / "cache_scalability.pdf")


def plot_robustness(result_root: Path, out_dir: Path) -> None:
    source = result_root / "vspm_robustness_T10_n5_gpu.csv"
    if not source.exists():
        source = result_root / "vspm_robustness_T10_n5.csv"
    rows = read_csv(source)
    if not rows:
        return
    labels = [r["condition"].replace("packet_loss_", "pkt ").replace("dropout_", "drop ").replace("false_positive_", "fp ") for r in rows]
    ious = [float(r["mean_iou"]) for r in rows]
    deltas = [float(r["mean_delta_iou"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.2, 3.8))
    x = list(range(len(rows)))
    ax1.bar([v - 0.18 for v in x], ious, width=0.36, label="Mean IoU")
    ax1.set_ylabel("Mean IoU")
    ax1.set_ylim(max(0.0, min(ious) - 0.03), min(1.0, max(ious) + 0.03))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot([v + 0.18 for v in x], deltas, color="#b23a48", marker="o", linewidth=1.8, label="Delta IoU")
    ax2.set_ylabel("Delta IoU over last-frame baseline")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")
    savefig(out_dir / "vspm_robustness.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("results/revision_plan"))
    parser.add_argument("--summary-root", type=Path, default=Path("results/summary_revision"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision_figures"))
    args = parser.parse_args()

    plot_training(args.summary_root, args.out_dir)
    plot_profile(args.result_root, args.out_dir)
    plot_scalability(args.result_root, args.out_dir)
    plot_robustness(args.result_root, args.out_dir)
    print(f"wrote figures under {args.out_dir}")


if __name__ == "__main__":
    main()
