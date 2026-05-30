#!/usr/bin/env python3
"""Synthetic N-agent cache/state scalability profile for local DLPCM prep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from v2x_forecasting.cache import VSPMCache
from v2x_forecasting.profiling import TensorStateShape, estimate_cache_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/revision_plan/scalability_profile.csv"))
    parser.add_argument("--max-cache", type=int, default=8)
    parser.add_argument("--ttl-frames", type=int, default=10)
    parser.add_argument("--state-height", type=int, default=256)
    parser.add_argument("--state-width", type=int, default=256)
    parser.add_argument("--channels", type=int, default=32)
    args = parser.parse_args()

    shape_fp32_hc = TensorStateShape(args.channels, args.state_height, args.state_width, 2, 4)
    shape_gru_ds64_fp16 = TensorStateShape(args.channels, 64, 64, 1, 2)

    rows = []
    for n in [1, 2, 3, 4, 5, 8, 16]:
        cache = VSPMCache(capacity=args.max_cache, ttl_frames=args.ttl_frames)
        for frame in range(n):
            cache.put(f"veh{frame}", {"state": frame}, frame)
        for frame in range(n):
            cache.get(f"veh{frame}", current_frame=n)
        rows.append(
            {
                "num_agents": n,
                "cache_capacity": args.max_cache,
                "ttl_frames": args.ttl_frames,
                "fp32_hc_state_mib": estimate_cache_bytes(n, shape_fp32_hc) / 1024**2,
                "gru_ds64_fp16_state_mib": estimate_cache_bytes(n, shape_gru_ds64_fp16) / 1024**2,
                **cache.stats(),
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

