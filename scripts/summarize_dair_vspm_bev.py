#!/usr/bin/env python
"""Summarize DAIR label-derived VSPM BEV sequence directories."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("datasets/DAIR-V2X/vspm_bev"))
    p.add_argument("--out", type=Path, default=Path("results/dair_vspm_bev_summary.csv"))
    return p.parse_args()


def scene_frame(path: Path) -> tuple[int, int] | None:
    try:
        scene, frame = path.parent.name.split("_", 1)
        return int(scene), int(frame)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    rows = []
    grouped: dict[tuple[str, str, int], list[tuple[int, int]]] = defaultdict(list)

    for npy in sorted(args.root.glob("*/*/*/0.npy")):
        parsed = scene_frame(npy)
        if parsed is None:
            continue
        scene, frame = parsed
        split = npy.parts[-4]
        agent = npy.parts[-3]
        data = np.load(npy, allow_pickle=True).item()
        occupied = len(data.get("voxel_indices_0", []))
        grouped[(split, agent, scene)].append((frame, occupied))

    for (split, agent, scene), samples in sorted(grouped.items()):
        frames = sorted(frame for frame, _ in samples)
        occupied = np.array([occ for _, occ in samples], dtype=np.float64)
        gaps = np.diff(frames)
        rows.append(
            {
                "split": split,
                "agent": agent,
                "scene": scene,
                "frames": len(frames),
                "first_frame_idx": frames[0],
                "last_frame_idx": frames[-1],
                "max_contiguous_gap": int(gaps.max()) if len(gaps) else 0,
                "mean_occupied_cells": float(occupied.mean()) if len(occupied) else 0.0,
                "min_occupied_cells": int(occupied.min()) if len(occupied) else 0,
                "max_occupied_cells": int(occupied.max()) if len(occupied) else 0,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        fieldnames = [
            "split",
            "agent",
            "scene",
            "frames",
            "first_frame_idx",
            "last_frame_idx",
            "max_contiguous_gap",
            "mean_occupied_cells",
            "min_occupied_cells",
            "max_occupied_cells",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_frames = sum(int(r["frames"]) for r in rows)
    print(f"wrote {len(rows)} scene rows, {total_frames} frames to {args.out}")


if __name__ == "__main__":
    main()
