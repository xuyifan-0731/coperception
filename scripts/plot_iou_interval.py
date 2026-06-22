
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

def plot_iou_vs_interval(file1: Path, file2: Path, out_path: Path, show: bool = False):
    g1 = load_and_group(file1).rename(columns={"iou": "iou_file1"})
    g2 = load_and_group(file2).rename(columns={"iou": "iou_file2"})

    # 合并（取并集，缺失值会断线可视化）
    merged = g1.merge(g2[["interval", "iou_file2"]], on="interval", how="outer").sort_values("interval")

    # 画图
    plt.figure(figsize=(8, 5))
    if "iou_file1" in merged.columns:
        plt.plot(merged["interval"], merged["iou_file1"], marker="o", label="File1 IoU")
    if "iou_file2" in merged.columns:
        plt.plot(merged["interval"], merged["iou_file2"], marker="o", label="File2 IoU")
    if "baseline_iou" in merged.columns:
        plt.plot(merged["interval"], merged["baseline_iou"], marker="o", linestyle="--", label="Baseline IoU (from File1)")

    plt.xlabel("interval")
    plt.ylabel("IoU")
    plt.title("IoU vs interval (two files + baseline from file1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot IoU vs interval for two CSVs (baseline from file1).")
    parser.add_argument("--file1", required=True, help="第一个CSV（提供baseline_iou）")
    parser.add_argument("--file2", required=True, help="第二个CSV")
    parser.add_argument("-o", "--out", type=Path, default=Path("iou_vs_interval.png"), help="输出图像路径")
    parser.add_argument("--show", action="store_true", help="绘制后弹窗显示（服务器环境可不加）")
    args = parser.parse_args()

    plot_iou_vs_interval(args.file1, args.file2, args.out, show=args.show)
    print(f"Saved figure to: {args.out}")

if __name__ == "__main__":
    main()
