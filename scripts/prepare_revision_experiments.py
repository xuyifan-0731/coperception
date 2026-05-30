#!/usr/bin/env python3
"""Create CSV manifests for DLPCM revision experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from v2x_forecasting.delay_simulator import make_latency_grid


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision_plan"))
    parser.add_argument("--frame-interval", type=float, default=0.2)
    args = parser.parse_args()

    latency_rows = []
    for case in make_latency_grid(range(0, 6), [0, 1, 2, 3, 4], args.frame_interval):
        latency_rows.append(
            {
                "comm_delay_frames": case.comm_delay,
                "comp_delay_frames": case.comp_delay,
                "remote_rollout_frames": case.remote_rollout_frames,
                "local_rollout_frames": case.local_rollout_frames,
                "comm_delay_s": case.comm_delay_s,
                "comp_delay_s": case.comp_delay_s,
            }
        )
    write_csv(
        args.out_dir / "latency_grid.csv",
        latency_rows,
        [
            "comm_delay_frames",
            "comp_delay_frames",
            "remote_rollout_frames",
            "local_rollout_frames",
            "comm_delay_s",
            "comp_delay_s",
        ],
    )

    robustness_rows = []
    for loss in [0, 0.1, 0.2, 0.3, 0.5]:
        robustness_rows.append({"type": "packet_loss", "value": loss, "unit": "rate"})
    for jitter in [0, 1, 2, 3]:
        robustness_rows.append({"type": "delay_jitter", "value": jitter, "unit": "frames"})
    for trans in [0, 0.2, 0.5, 1.0, 2.0]:
        robustness_rows.append({"type": "pose_translation_noise", "value": trans, "unit": "m"})
    for rot in [0, 0.5, 1.0, 2.0]:
        robustness_rows.append({"type": "pose_rotation_noise", "value": rot, "unit": "deg"})
    write_csv(args.out_dir / "robustness_grid.csv", robustness_rows, ["type", "value", "unit"])

    scalability_rows = [
        {
            "num_agents": n,
            "source": "real" if n <= 5 else "synthetic",
            "measure_runtime": True,
            "measure_memory": True,
            "measure_cache": True,
        }
        for n in [1, 2, 3, 4, 5, 8, 16]
    ]
    write_csv(
        args.out_dir / "scalability_grid.csv",
        scalability_rows,
        ["num_agents", "source", "measure_runtime", "measure_memory", "measure_cache"],
    )

    print(f"Wrote revision experiment manifests to {args.out_dir}")


if __name__ == "__main__":
    main()

