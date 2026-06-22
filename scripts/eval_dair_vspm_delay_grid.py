#!/usr/bin/env python
"""Evaluate a VSPM checkpoint over communication/computation delay grids.

This script evaluates the label-derived BEV forecasting checkpoint used by the
DLPCM/VSPM experiments. In the default ``dual`` mode, it maps a dual-latency
case to the remote rollout horizon defined in
``src.v2x_forecasting.delay_simulator``:

    remote horizon = communication delay + computation delay

The output CSV is intended for DAIR-V2X tables analogous to the V2X-Sim
delay-grid results. Detection AP still requires the OpenDAIRV2X pipeline.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from v2x_forecasting import delay_simulator
from v2x_forecasting import train_bev


def parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError("delays must be non-negative")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/dair_vspm_delay_grid.csv"))
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--n", type=int, default=6, help="Model span; supports horizons 1..n-1")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--comm-delays", type=parse_int_list, default=parse_int_list("0,1,2,3,4,5"))
    p.add_argument("--comp-delays", type=parse_int_list, default=parse_int_list("0,1,2,3,4,5"))
    p.add_argument(
        "--latency-grid-csv",
        type=Path,
        default=None,
        help="Optional CSV with comm_delay_frames and comp_delay_frames columns. Overrides --comm-delays/--comp-delays.",
    )
    p.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Skip latency cases whose total horizon exceeds n-1 instead of raising an error.",
    )
    p.add_argument("--frame-interval", type=float, default=0.2)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--use-change-head", action="store_true")
    p.add_argument(
        "--mode",
        choices=["no_comp", "comm_only", "comp_only", "dual", "oracle"],
        default="dual",
        help=(
            "System ablation mode. Prediction horizon differs by mode, while "
            "all non-oracle modes are compared against the final result-time target."
        ),
    )
    return p.parse_args()


def load_latency_cases(args: argparse.Namespace) -> list[delay_simulator.LatencyCase]:
    if args.latency_grid_csv is None:
        return delay_simulator.make_latency_grid(args.comm_delays, args.comp_delays, args.frame_interval)

    cases: list[delay_simulator.LatencyCase] = []
    with args.latency_grid_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"comm_delay_frames", "comp_delay_frames"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{args.latency_grid_csv} missing columns: {sorted(missing)}")
        for row in reader:
            comm = int(row["comm_delay_frames"])
            comp = int(row["comp_delay_frames"])
            cases.append(
                delay_simulator.LatencyCase(
                    comm_delay=comm,
                    comp_delay=comp,
                    frame_interval_s=args.frame_interval,
                )
            )
    if not cases:
        raise ValueError(f"{args.latency_grid_csv} contains no latency cases")
    return cases


def iou_bool(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.detach().cpu().numpy().astype(bool)
    t = target.detach().cpu().numpy().astype(bool)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float(inter / union) if union > 0 else 1.0


def dynamic_iou(pred: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor) -> float:
    mask = baseline != target
    p = (pred * mask).detach().cpu().numpy().astype(bool)
    t = (target * mask).detach().cpu().numpy().astype(bool)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float(inter / union) if union > 0 else 1.0


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def prediction_horizon(case: delay_simulator.LatencyCase, mode: str) -> int:
    if mode == "no_comp":
        return 0
    if mode == "comm_only":
        return case.comm_delay
    if mode == "comp_only":
        return case.comp_delay
    if mode in {"dual", "oracle"}:
        return case.remote_rollout_frames
    raise ValueError(f"unsupported mode: {mode}")


def select_horizon_frame(last: torch.Tensor, future: torch.Tensor, horizon: int) -> torch.Tensor:
    if horizon == 0:
        return last
    return future[horizon - 1]


@torch.no_grad()
def evaluate_grid(args: argparse.Namespace) -> list[dict[str, float | int]]:
    device = train_bev.resolve_device(args.device)
    train_bev.DEVICE = device

    cases = load_latency_cases(args)
    if args.skip_unsupported:
        cases = [case for case in cases if case.remote_rollout_frames <= args.n - 1]
        if not cases:
            raise ValueError(f"no latency cases are supported by --n {args.n}")

    max_horizon = max(case.remote_rollout_frames for case in cases)
    if args.n - 1 < max_horizon:
        raise ValueError(f"--n {args.n} supports horizons up to {args.n - 1}, but grid needs {max_horizon}")

    dataset = train_bev.SeqDataset(str(args.data_root), args.T, args.n, aug=False, H=256, W=256)
    if args.max_samples:
        dataset.samples = dataset.samples[: args.max_samples]
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    model = train_bev.BEVForecastNet(use_change_head=args.use_change_head).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    buckets = {
        (case.comm_delay, case.comp_delay): {
            "pred_iou": [],
            "copy_last_iou": [],
            "delta_iou": [],
            "dynamic_iou": [],
        }
        for case in cases
    }

    for X, Y, *_ in loader:
        X = X.to(device)
        Y = Y.to(device)
        steps = args.n - 1

        def step_input_fn(_t: int, prev_pred: torch.Tensor) -> torch.Tensor:
            return prev_pred.detach()

        occ_seq, _ = model(X, steps, step_input_fn, use_change_head=False)
        pred_bin = (occ_seq > 0.5).float()
        last = X[:, -1, 0]

        for case in cases:
            key = (case.comm_delay, case.comp_delay)
            pred_horizon = prediction_horizon(case, args.mode)
            target_horizon = case.remote_rollout_frames
            B = X.size(0)
            for b in range(B):
                baseline = last[b]
                target = select_horizon_frame(last[b], Y[b, :, 0], target_horizon)
                if args.mode == "oracle":
                    pred = target
                elif pred_horizon == 0:
                    pred = baseline
                else:
                    pred = pred_bin[b, pred_horizon - 1, 0]
                pred_iou = iou_bool(pred, target)
                base_iou = iou_bool(baseline, target)
                buckets[key]["pred_iou"].append(pred_iou)
                buckets[key]["copy_last_iou"].append(base_iou)
                buckets[key]["delta_iou"].append(pred_iou - base_iou)
                buckets[key]["dynamic_iou"].append(dynamic_iou(pred, target, baseline))

    rows = []
    for case in cases:
        key = (case.comm_delay, case.comp_delay)
        vals = buckets[key]
        rows.append(
            {
                "comm_delay_frames": case.comm_delay,
                "comp_delay_frames": case.comp_delay,
                "comm_delay_s": case.comm_delay_s,
                "comp_delay_s": case.comp_delay_s,
                "remote_rollout_frames": case.remote_rollout_frames,
                "local_rollout_frames": case.local_rollout_frames,
                "mode": args.mode,
                "prediction_horizon_frames": prediction_horizon(case, args.mode),
                "target_horizon_frames": case.remote_rollout_frames,
                "samples": len(vals["pred_iou"]),
                "pred_iou": finite_mean(vals["pred_iou"]),
                "copy_last_iou": finite_mean(vals["copy_last_iou"]),
                "delta_iou": finite_mean(vals["delta_iou"]),
                "dynamic_iou": finite_mean(vals["dynamic_iou"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = evaluate_grid(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        fieldnames = [
            "comm_delay_frames",
            "comp_delay_frames",
            "comm_delay_s",
            "comp_delay_s",
            "remote_rollout_frames",
            "local_rollout_frames",
            "mode",
            "prediction_horizon_frames",
            "target_horizon_frames",
            "samples",
            "pred_iou",
            "copy_last_iou",
            "delta_iou",
            "dynamic_iou",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} delay cases to {args.out}")


if __name__ == "__main__":
    main()
