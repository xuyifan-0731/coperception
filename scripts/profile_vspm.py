#!/usr/bin/env python3
"""Profile local VSPM variants on synthetic BEV tensors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

from v2x_forecasting.profiling import TensorStateShape, benchmark_callable, count_parameters


def compressed_args(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        enable_gru_fp16=mode in {"gru_fp16", "ds64_fp16", "bottleneck12"},
        enable_ds64_fp16=mode in {"ds64_fp16", "bottleneck12"},
        enable_bottleneck_1x1=mode == "bottleneck12",
        bottleneck_dim=12,
    )


def state_shape_for(name: str, height: int, width: int) -> TensorStateShape:
    return TensorStateShape(
        channels=32,
        height=64 if name in {"ds64_fp16", "bottleneck12"} else height,
        width=64 if name in {"ds64_fp16", "bottleneck12"} else width,
        tensors_per_agent=1 if name in {"gru_fp16", "ds64_fp16", "bottleneck12"} else 2,
        bytes_per_value=2 if name in {"gru_fp16", "ds64_fp16", "bottleneck12"} else 4,
    )


def make_model(name: str):
    from v2x_forecasting.train_bev import BEVForecastNet as ConvLSTMForecastNet
    from v2x_forecasting.train_bev_compressed import BEVForecastNet as CompressedForecastNet

    if name == "convlstm":
        return ConvLSTMForecastNet(use_change_head=False)
    if name in {"compressed_lstm", "gru_fp16", "ds64_fp16", "bottleneck12"}:
        mode = "none" if name == "compressed_lstm" else name
        return CompressedForecastNet(compressed_args(mode), use_change_head=False)
    raise ValueError(f"unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/revision_plan/vspm_profile.csv"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only write state-memory estimates; does not import torch or run models.",
    )
    args = parser.parse_args()

    model_names = ["convlstm", "compressed_lstm", "gru_fp16", "ds64_fp16", "bottleneck12"]

    if args.metadata_only:
        rows = []
        for name in model_names:
            state_shape = state_shape_for(name, args.H, args.W)
            rows.append(
                {
                    "model": name,
                    "device": "metadata_only",
                    "batch": args.batch,
                    "T": args.T,
                    "steps": args.steps,
                    "params": "",
                    "state_mib_per_agent": state_shape.mib_per_agent(),
                    "mean_ms": "",
                    "min_ms": "",
                    "max_ms": "",
                }
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote={args.out}")
        return

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "torch is required for latency profiling. Re-run with --metadata-only "
            "or install torch in the active environment."
        ) from exc

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    x = torch.rand(args.batch, args.T, 1, args.H, args.W, device=device)

    rows = []
    for name in model_names:
        model = make_model(name).to(device).eval()

        def step_input_fn(_t: int, prev: torch.Tensor) -> torch.Tensor:
            return prev.detach()

        def run() -> None:
            with torch.no_grad():
                model(x, args.steps, step_input_fn, use_change_head=False)

        stats = benchmark_callable(run, warmup=args.warmup, repeat=args.repeat)
        params = count_parameters(model)
        state_shape = state_shape_for(name, args.H, args.W)
        rows.append(
            {
                "model": name,
                "device": str(device),
                "batch": args.batch,
                "T": args.T,
                "steps": args.steps,
                "params": params,
                "state_mib_per_agent": state_shape.mib_per_agent(),
                **stats,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
