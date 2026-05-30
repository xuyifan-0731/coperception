#!/usr/bin/env python3
"""Evaluate a VSPM checkpoint under local BEV robustness perturbations.

This is a lightweight VSPM-level robustness experiment for revision evidence.
It evaluates occupancy forecasting IoU under packet loss and BEV degradation.
The script defaults to CPU so it can run while the main GPU training pipeline is
active.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
from pathlib import Path

import numpy as np

from v2x_forecasting.robustness import degrade_bev, packet_keep_mask


def sparse2bev(idx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    bev = np.zeros(shape, np.uint8)
    y, x = idx[:, 0], idx[:, 1]
    m = (y >= 0) & (y < shape[0]) & (x >= 0) & (x < shape[1])
    bev[y[m], x[m]] = 1
    return bev


def scene_samples(root: Path, t_history: int, n_span: int, limit: int = 0) -> list[tuple[int, list[int], list[int]]]:
    by_scene: dict[int, list[int]] = {}
    for path in sorted(root.glob("*_*")):
        if not path.is_dir():
            continue
        try:
            scene, frame = map(int, path.name.split("_"))
        except ValueError:
            continue
        data_path = path / "0.npy"
        if not data_path.exists():
            continue
        data = np.load(data_path, allow_pickle=True).item()
        if data:
            by_scene.setdefault(scene, []).append(frame)

    samples: list[tuple[int, list[int], list[int]]] = []
    for scene, frames in sorted(by_scene.items()):
        frames.sort()
        count = max(0, len(frames) - t_history - (n_span - 1) + 1)
        for start in range(count):
            hist = frames[start : start + t_history]
            target = frames[start + t_history : start + t_history + (n_span - 1)]
            samples.append((scene, hist, target))
            if limit and len(samples) >= limit:
                return samples
    return samples


def load_bev(root: Path, scene: int, frame: int) -> np.ndarray:
    data = np.load(root / f"{scene}_{frame}" / "0.npy", allow_pickle=True).item()
    return sparse2bev(data["voxel_indices_0"], (256, 256))


def apply_packet_loss_to_history(history: list[np.ndarray], loss_rate: float, seed: int, fill: str) -> list[np.ndarray]:
    if loss_rate <= 0:
        return history
    mask = packet_keep_mask(len(history), loss_rate, seed)
    out: list[np.ndarray] = []
    last = np.zeros_like(history[0])
    for bev, keep in zip(history, mask):
        if keep:
            last = bev
            out.append(bev)
        elif fill == "hold":
            out.append(last.copy())
        elif fill == "zero":
            out.append(np.zeros_like(bev))
        else:
            raise ValueError(f"unknown packet-loss fill policy: {fill}")
    return out


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    p = pred.astype(bool)
    t = target.astype(bool)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float(inter / union) if union > 0 else 1.0


def dynamic_iou(pred: np.ndarray, target: np.ndarray, baseline: np.ndarray) -> float:
    mask = baseline != target
    p = np.logical_and(pred.astype(bool), mask)
    t = np.logical_and(target.astype(bool), mask)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float(inter / union) if union > 0 else 1.0


def run_condition(args: argparse.Namespace, condition: dict[str, float | str], model: object, torch: object) -> dict[str, object]:
    device = torch.device(args.device)
    samples = scene_samples(args.data_root, args.T, args.n, args.max_samples)
    steps = args.n - 1

    ious = [[] for _ in range(steps)]
    baseline_ious = [[] for _ in range(steps)]
    dynamic_ious = [[] for _ in range(steps)]

    dropout = float(condition["dropout_rate"])
    false_positive = float(condition["false_positive_rate"])
    packet_loss = float(condition["packet_loss"])

    model.eval()
    with torch.no_grad():
        for sample_id, (scene, hist_ids, target_ids) in enumerate(samples):
            history = [load_bev(args.data_root, scene, frame) for frame in hist_ids]
            targets = [load_bev(args.data_root, scene, frame) for frame in target_ids]

            history = apply_packet_loss_to_history(
                history,
                packet_loss,
                args.seed + sample_id,
                args.loss_fill,
            )
            if dropout > 0 or false_positive > 0:
                history = [
                    degrade_bev(
                        bev,
                        dropout_rate=dropout,
                        false_positive_rate=false_positive,
                        seed=args.seed + sample_id * 1000 + idx,
                    )
                    for idx, bev in enumerate(history)
                ]

            x_np = np.stack(history).astype(np.float32)[None, :, None]
            y_np = np.stack(targets).astype(np.float32)
            x = torch.from_numpy(x_np).to(device)

            def step_input_fn(_t: int, prev_pred: object) -> object:
                return prev_pred.detach()

            occ_seq, _ = model(x, steps, step_input_fn, use_change_head=False)
            pred = (occ_seq.detach().cpu().numpy()[0, :, 0] > 0.5).astype(np.uint8)
            baseline = history[-1]
            for step in range(steps):
                ious[step].append(iou(pred[step], y_np[step]))
                baseline_ious[step].append(iou(baseline, y_np[step]))
                dynamic_ious[step].append(dynamic_iou(pred[step], y_np[step], baseline))

    flat_iou = [v for row in ious for v in row]
    flat_base = [v for row in baseline_ious for v in row]
    flat_dyn = [v for row in dynamic_ious for v in row]
    return {
        **condition,
        "samples": len(samples),
        "steps": steps,
        "mean_iou": float(np.mean(flat_iou)) if flat_iou else float("nan"),
        "mean_baseline_iou": float(np.mean(flat_base)) if flat_base else float("nan"),
        "mean_delta_iou": float(np.mean(flat_iou) - np.mean(flat_base)) if flat_iou else float("nan"),
        "mean_dynamic_iou": float(np.mean(flat_dyn)) if flat_dyn else float("nan"),
    }


def parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/revision_plan/vspm_robustness_eval.csv"))
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=0, help="Limit PyTorch CPU worker threads; 0 keeps PyTorch default.")
    parser.add_argument("--packet-loss", default="0,0.1,0.2,0.3,0.5")
    parser.add_argument("--dropout", default="0,0.05,0.1,0.2")
    parser.add_argument("--false-positive", default="0,0.001,0.005")
    parser.add_argument("--loss-fill", choices=["hold", "zero"], default="hold")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import torch
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(4, args.torch_threads)))

    train_bev = importlib.import_module("v2x_forecasting.train_bev")
    model = train_bev.BEVForecastNet(use_change_head=False).to(torch.device(args.device))
    checkpoint = torch.load(args.ckpt, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint["model"])

    conditions: list[dict[str, float | str]] = [{"condition": "clean", "packet_loss": 0.0, "dropout_rate": 0.0, "false_positive_rate": 0.0}]
    for value in parse_float_list(args.packet_loss):
        if value:
            conditions.append({"condition": f"packet_loss_{value}", "packet_loss": value, "dropout_rate": 0.0, "false_positive_rate": 0.0})
    for value in parse_float_list(args.dropout):
        if value:
            conditions.append({"condition": f"dropout_{value}", "packet_loss": 0.0, "dropout_rate": value, "false_positive_rate": 0.0})
    for value in parse_float_list(args.false_positive):
        if value:
            conditions.append({"condition": f"false_positive_{value}", "packet_loss": 0.0, "dropout_rate": 0.0, "false_positive_rate": value})

    rows = [run_condition(args, condition, model, torch) for condition in conditions]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "condition",
        "packet_loss",
        "dropout_rate",
        "false_positive_rate",
        "samples",
        "steps",
        "mean_iou",
        "mean_baseline_iou",
        "mean_delta_iou",
        "mean_dynamic_iou",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
