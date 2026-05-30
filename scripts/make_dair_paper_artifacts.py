#!/usr/bin/env python3
"""Generate DAIR-V2X paper tables and figures for the DLPCM revision."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DAIR = ROOT / "results" / "dair_v2x"
REVISION = ROOT / "results" / "revision_plan"
PAPER = ROOT / "DLPCM"
TABLE_DIR = PAPER / "table"
PIC_DIR = PAPER / "pics"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def fmt(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def esc(text: object) -> str:
    return str(text).replace("_", "\\_")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def table_block(
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
    colspec: str,
    star: bool = False,
    small: bool = False,
) -> str:
    env = "table*" if star else "table"
    lines = [
        f"\\begin{{{env}}}[t]",
        "\\centering",
    ]
    if small:
        lines.append("\\scriptsize")
    lines.extend(
        [
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{colspec}}}",
            "\\toprule",
            " & ".join(headers) + " \\\\",
            "\\midrule",
        ]
    )
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""])
    return "\n".join(lines)


def generate_official_table() -> None:
    rows = {row["run"]: row for row in read_csv(DAIR / "tables" / "official_baselines_table.csv")}
    order = [
        ("veh_only_k0", "Vehicle only", "0"),
        ("inf_only_k0", "Infrastructure only", "0"),
        ("late_fusion_tclf_k0", "Late fusion + TCLF", "0"),
        ("late_fusion_tclf_k3", "Late fusion + TCLF", "3"),
        ("late_fusion_tclf_k5", "Late fusion + TCLF", "5"),
        ("late_fusion_no_comp_k1", "Late fusion w/o comp.", "1"),
        ("late_fusion_no_comp_k3", "Late fusion w/o comp.", "3"),
        ("late_fusion_no_comp_k5", "Late fusion w/o comp.", "5"),
        ("early_fusion_k0", "Early fusion", "0"),
        ("early_fusion_k2", "Early fusion", "2"),
    ]
    body = []
    for run, name, delay in order:
        row = rows[run]
        body.append(
            [
                name,
                delay,
                fmt(row["car_3d_ap_0_50"]),
                fmt(row["car_3d_ap_0_70"]),
                fmt(row["car_bev_ap_0_50"]),
                fmt(row["car_bev_ap_0_70"]),
                fmt(row["avg_comm_bytes"]),
            ]
        )
    write_text(
        TABLE_DIR / "dair_official_baselines.tex",
        table_block(
            "Representative DAIR-V2X-C official detection baselines. Delay is measured in frames.",
            "tab:dair-official-baselines",
            ["Method", "$k$", "3D@0.5", "3D@0.7", "BEV@0.5", "BEV@0.7", "Comm. B"],
            body,
            "lrrrrrr",
            star=True,
            small=True,
        ),
    )


def generate_delay_table() -> None:
    rows = read_csv(DAIR / "tables" / "delay_system_ablation_table.csv")
    mode_order = {"no_comp": 0, "comm_only": 1, "comp_only": 2, "dual": 3, "oracle": 4}
    rows = sorted(rows, key=lambda r: (r["setting"], mode_order.get(r["mode"], 99)))
    body = []
    for row in rows:
        setting = row["setting"].replace("_ckpt", " ckpt ")
        body.append(
            [
                esc(setting),
                esc(row["mode"]),
                row["cases"],
                fmt(row["mean_pred_iou"], 4),
                fmt(row["mean_copy_last_iou"], 4),
                fmt(row["mean_delta_iou"], 4),
                fmt(row["mean_dynamic_iou"], 4),
            ]
        )
    write_text(
        TABLE_DIR / "dair_delay_system.tex",
        table_block(
            "DAIR-V2X label-derived BEV delay and system ablation.",
            "tab:dair-delay-system",
            ["Setting", "Mode", "Cases", "IoU", "Copy", "$\\Delta$IoU", "Dyn. IoU"],
            body,
            "llrrrrr",
            star=True,
            small=True,
        ),
    )


def generate_mst_table() -> None:
    train_rows = {row["experiment_group"]: row for row in read_csv(DAIR / "tables" / "training_summary_table.csv")}
    profile_rows = {row["model"]: row for row in read_csv(REVISION / "vspm_profile.csv")}
    mapping = [
        ("mst_ablation/baseline", "ConvLSTM state", "convlstm"),
        ("mst_ablation/gru_fp16", "GRU FP16", "gru_fp16"),
        ("mst_ablation/ds64_fp16", "DS64 FP16", "ds64_fp16"),
        ("mst_ablation/bottleneck12", "Bottleneck12", "bottleneck12"),
    ]
    baseline_state = as_float(profile_rows["convlstm"]["state_mib_per_agent"])
    body = []
    for group, name, profile_name in mapping:
        train = train_rows[group]
        profile = profile_rows[profile_name]
        state = as_float(profile["state_mib_per_agent"])
        body.append(
            [
                name,
                fmt(state, 3),
                fmt(baseline_state / state, 1) + "$\\times$",
                fmt(profile["mean_ms"], 2),
                fmt(train["best_IoU"], 4),
                fmt(train["best_Delta_IoU"], 4),
                fmt(train["best_Dynamic_IoU"], 4),
            ]
        )
    write_text(
        TABLE_DIR / "dair_mst_tradeoff.tex",
        table_block(
            "MST communication-state trade-off on DAIR-V2X label-derived BEV.",
            "tab:dair-mst-tradeoff",
            ["Variant", "MiB/agent", "Reduction", "ms", "IoU", "$\\Delta$IoU", "Dyn. IoU"],
            body,
            "lrrrrrr",
            small=True,
        ),
    )


def generate_sensitivity_table() -> None:
    rows = {row["experiment_group"]: row for row in read_csv(DAIR / "tables" / "training_summary_table.csv")}
    order = [
        "vspm_sensitivity/T5_n5",
        "our_method_T10_n5",
        "vspm_sensitivity/T20_n5",
        "vspm_sensitivity/T30_n5",
        "vspm_sensitivity/T10_n3",
        "our_method_T10_n10",
        "vspm_sensitivity/T10_n15",
    ]
    body = []
    for group in order:
        row = rows[group]
        name = group.split("/")[-1]
        body.append(
            [
                esc(name),
                row["best_step"],
                fmt(row["best_IoU"], 4),
                fmt(row["best_Delta_IoU"], 4),
                fmt(row["best_Dynamic_IoU"], 4),
            ]
        )
    write_text(
        TABLE_DIR / "dair_sensitivity.tex",
        table_block(
            "History length and rollout-horizon sensitivity on DAIR-V2X label-derived BEV.",
            "tab:dair-sensitivity",
            ["Setting", "Best step", "IoU", "$\\Delta$IoU", "Dyn. IoU"],
            body,
            "lrrrr",
            small=True,
        ),
    )


def generate_robustness_table() -> None:
    rows = read_csv(DAIR / "tables" / "robustness_table.csv")
    keep = {"clean", "packet_loss_0.3", "packet_loss_0.5", "dropout_0.2", "false_positive_0.005"}
    body = []
    for row in rows:
        if row["condition"] not in keep:
            continue
        body.append(
            [
                esc(row["setting"].replace("_ckpt", " ckpt ")),
                esc(row["condition"]),
                fmt(row["mean_iou"], 4),
                fmt(row["mean_baseline_iou"], 4),
                fmt(row["mean_delta_iou"], 4),
                fmt(row["mean_dynamic_iou"], 4),
            ]
        )
    write_text(
        TABLE_DIR / "dair_robustness.tex",
        table_block(
            "Robustness of DAIR-V2X VSPM under packet loss and BEV perturbations.",
            "tab:dair-robustness",
            ["Setting", "Condition", "IoU", "Copy", "$\\Delta$IoU", "Dyn. IoU"],
            body,
            "llrrrr",
            star=True,
            small=True,
        ),
    )


def setup_plot(figsize: tuple[float, float]) -> None:
    plt.figure(figsize=figsize)
    plt.rcParams.update({"font.size": 9})


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_official_delay() -> None:
    rows = read_csv(DAIR / "tables" / "official_baselines_table.csv")
    by_run = {row["run"]: row for row in rows}

    def series(prefix: str, ks: list[int], metric: str) -> list[float]:
        return [as_float(by_run[f"{prefix}_k{k}"][metric]) for k in ks]

    setup_plot((5.2, 3.3))
    for prefix, label, ks, marker in [
        ("late_fusion_tclf", "Late fusion + TCLF", [0, 1, 2, 3, 4, 5], "o"),
        ("late_fusion_no_comp", "Late fusion w/o comp.", [1, 2, 3, 4, 5], "s"),
        ("early_fusion", "Early fusion", [0, 1, 2], "^"),
    ]:
        plt.plot(
            ks,
            series(prefix, ks, "car_bev_ap_0_70"),
            marker=marker,
            linewidth=1.8,
            label=label,
        )
    plt.xlabel("Delay k (frames)")
    plt.ylabel("BEV AP@0.7")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    savefig(PIC_DIR / "dair_official_delay.pdf")


def plot_delay_heatmap() -> None:
    rows = read_csv(DAIR / "our_method" / "full_T10_n10_ckpt15000" / "delay_grid_dual.csv")
    comms = sorted({int(r["comm_delay_frames"]) for r in rows})
    comps = sorted({int(r["comp_delay_frames"]) for r in rows})
    value = {(int(r["comm_delay_frames"]), int(r["comp_delay_frames"])): as_float(r["pred_iou"]) for r in rows}
    grid = [[value[(c, p)] for c in comms] for p in comps]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap="YlGnBu", vmin=0.2, vmax=1.0)
    ax.set_xticks(range(len(comms)))
    ax.set_xticklabels(comms)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels(comps)
    ax.set_xlabel("Communication delay (frames)")
    ax.set_ylabel("Computation delay (frames)")
    for y, _ in enumerate(comps):
        for x, _ in enumerate(comms):
            ax.text(x, y, fmt(grid[y][x], 2), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="IoU")
    savefig(PIC_DIR / "dair_delay_heatmap.pdf")


def plot_mst_pareto() -> None:
    train_rows = {row["experiment_group"]: row for row in read_csv(DAIR / "tables" / "training_summary_table.csv")}
    profile_rows = {row["model"]: row for row in read_csv(REVISION / "vspm_profile.csv")}
    points = [
        ("ConvLSTM", "mst_ablation/baseline", "convlstm"),
        ("GRU FP16", "mst_ablation/gru_fp16", "gru_fp16"),
        ("DS64 FP16", "mst_ablation/ds64_fp16", "ds64_fp16"),
        ("Bottleneck12", "mst_ablation/bottleneck12", "bottleneck12"),
    ]
    setup_plot((5.2, 3.3))
    for label, group, profile_name in points:
        x = as_float(profile_rows[profile_name]["state_mib_per_agent"])
        y = as_float(train_rows[group]["best_Dynamic_IoU"])
        plt.scatter([x], [y], s=45)
        plt.annotate(label, (x, y), xytext=(5, 3), textcoords="offset points", fontsize=8)
    plt.xscale("log", base=2)
    plt.xlabel("State size per agent (MiB, log scale)")
    plt.ylabel("Dynamic IoU")
    plt.grid(True, alpha=0.25, which="both")
    savefig(PIC_DIR / "dair_mst_pareto.pdf")


def plot_robustness() -> None:
    rows = read_csv(DAIR / "tables" / "robustness_table.csv")
    packet = [0.0, 0.1, 0.2, 0.3, 0.5]
    setup_plot((5.2, 3.3))
    for setting, marker in [("T10_n5_ckpt16000", "o"), ("T10_n10_ckpt15000", "s")]:
        lookup = {}
        for row in rows:
            if row["setting"] != setting:
                continue
            if row["condition"] == "clean":
                lookup[0.0] = as_float(row["mean_iou"])
            elif row["condition"].startswith("packet_loss_"):
                lookup[float(row["condition"].split("_")[-1])] = as_float(row["mean_iou"])
        plt.plot(packet, [lookup[p] for p in packet], marker=marker, linewidth=1.8, label=setting.replace("_ckpt", " ckpt "))
    plt.xlabel("Packet loss rate")
    plt.ylabel("Mean IoU")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    savefig(PIC_DIR / "dair_packet_loss.pdf")


def main() -> None:
    generate_official_table()
    generate_delay_table()
    generate_mst_table()
    generate_sensitivity_table()
    generate_robustness_table()
    plot_official_delay()
    plot_delay_heatmap()
    plot_mst_pareto()
    plot_robustness()
    print(f"wrote tables under {TABLE_DIR}")
    print(f"wrote figures under {PIC_DIR}")


if __name__ == "__main__":
    main()
