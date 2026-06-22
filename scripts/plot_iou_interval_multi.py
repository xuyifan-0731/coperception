#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_and_group(csv_path: Path) -> pd.DataFrame:
    """读取CSV并按 interval 聚合，返回包含 iou（均值）以及可选 baseline_iou（均值）的表。"""
    df = pd.read_csv(csv_path)
    required = {"interval", "iou"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少必要列: {', '.join(sorted(missing))}")

    # 按 interval 取均值
    g = df.groupby("interval", as_index=False).agg(iou=("iou", "mean"))
    if "baseline_iou" in df.columns:
        g_baseline = df.groupby("interval", as_index=False).agg(baseline_iou=("baseline_iou", "mean"))
        g = g.merge(g_baseline, on="interval", how="left")
    return g

def plot_pair(ax, file1: Path, file2: Path, title: str):
    """绘制单个子图：file1 IOU、file2 IOU、baseline IOU"""
    g1 = load_and_group(file1).rename(columns={"iou": "iou_file1"})
    g2 = load_and_group(file2).rename(columns={"iou": "iou_file2"})
    merged = g1.merge(g2[["interval", "iou_file2"]], on="interval", how="outer").sort_values("interval")

    # 绘制曲线
    if "iou_file1" in merged.columns:
        ax.plot(merged["interval"], merged["iou_file1"], marker="o", label="w/o specific training data")
    if "iou_file2" in merged.columns:
        ax.plot(merged["interval"], merged["iou_file2"], marker="o", label="w/ specific training data")
    if "baseline_iou" in merged.columns:
        ax.plot(merged["interval"], merged["baseline_iou"], marker="o", linestyle="--", label="baseline")

    ax.set_title(title)
    ax.set_xlabel("interval")
    ax.grid(True, alpha=0.3)

def parse_args():
    p = argparse.ArgumentParser(description="Plot 5 pairs IoU vs interval (baseline from each file1).")
    # 允许传入5组file1/file2
    for i in range(5):
        p.add_argument(f"--file1_{i+1}", type=str, required=True, help=f"第{i+1}组 file1（提供baseline_iou）")
        p.add_argument(f"--file2_{i+1}", type=str, required=True, help=f"第{i+1}组 file2")
    p.add_argument("-o", "--out", type=Path,
                   default=Path("./iou_vs_interval_5agents.pdf"),
                   help="输出图像路径")
    p.add_argument("--show", action="store_true", help="绘制后弹窗显示（服务器环境可不加）")
    return p.parse_args()

def main():
    args = parse_args()

    # 读取5组路径
    pairs = []
    for i in range(5):
        f1 = Path(getattr(args, f"file1_{i+1}"))
        f2 = Path(getattr(args, f"file2_{i+1}"))
        pairs.append((f1, f2))

    # 创建1行5列子图
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2), sharey=True)
    agent_titles = [f"Agent{i}" for i in range(1, 6)]

    # 绘制每组数据
    for idx, (ax, (f1, f2), title) in enumerate(zip(axes, pairs, agent_titles)):
        plot_pair(ax, f1, f2, title)
        if idx == 0:
            ax.set_ylabel("IoU")  # 只保留第一个子图的y轴标题
        else:
            ax.set_ylabel("")

    # 合并图例
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    # 在图的上方放置统一图例
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))

    # 调整布局：进一步压缩横向空白
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, wspace=0.04)

    # 保存图片
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()
    print(f"Saved figure to: {args.out}")

if __name__ == "__main__":
    main()
