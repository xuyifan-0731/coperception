#!/usr/bin/env python
"""Convert DAIR-V2X labels into VSPM-style sparse BEV occupancy sequences.

The local VSPM scripts expect directories like scene_frame/0.npy containing a
voxel_indices_0 array. This adapter rasterizes DAIR labels into that format so
VSPM-level forecasting can be trained on DAIR sequences once the dataset exists.
It does not replace OpenDAIRV2X AP evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


VEHICLE_CLASSES = {"Bus", "Car", "Truck", "Van"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dair-root", required=True, help="Path to cooperative-vehicle-infrastructure")
    p.add_argument("--split-path", default="external/DAIR-V2X/data/split_datas/cooperative-split-data.json")
    p.add_argument("--output-root", default="datasets/DAIR-V2X/vspm_bev")
    p.add_argument("--sensor-view", choices=["vehicle", "infrastructure"], default="vehicle")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--x-range", nargs=2, type=float, default=(0.0, 100.0))
    p.add_argument("--y-range", nargs=2, type=float, default=(-39.68, 39.68))
    p.add_argument("--classes", nargs="+", default=sorted(VEHICLE_CLASSES))
    p.add_argument("--max-frames", type=int, default=0, help="For smoke tests; 0 converts all frames.")
    return p.parse_args()


def frame_id_from_path(path: str) -> str:
    return Path(path).stem


def load_split_ids(split_path: Path, sensor_view: str) -> dict[str, set[str]]:
    obj = json.loads(split_path.read_text())
    key = "vehicle_split" if sensor_view == "vehicle" else "infrastructure_split"
    return {name: set(values) for name, values in obj[key].items() if name in {"train", "val", "test"}}


def label_path_from_info(dair_root: Path, sensor_view: str, info: dict) -> Path:
    if sensor_view == "vehicle":
        rel = info.get("label_lidar_path")
        if rel:
            return dair_root / "vehicle-side" / rel
        fid = frame_id_from_path(info["pointcloud_path"])
        return dair_root / "vehicle-side" / "label" / "lidar" / f"{fid}.json"

    rel = info.get("label_lidar_path") or info.get("label_virtuallidar_path")
    if rel:
        return dair_root / "infrastructure-side" / rel
    fid = frame_id_from_path(info["pointcloud_path"])
    return dair_root / "infrastructure-side" / "label" / "virtuallidar" / f"{fid}.json"


def corners_xy(label: dict) -> np.ndarray | None:
    dims = label.get("3d_dimensions") or {}
    loc = label.get("3d_location") or {}
    try:
        length = float(dims["l"])
        width = float(dims["w"])
        x = float(loc["x"])
        y = float(loc["y"])
        rot = float(label.get("rotation", 0.0))
    except (KeyError, TypeError, ValueError):
        return None

    local = np.array(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        dtype=np.float32,
    )
    c, s = math.cos(rot), math.sin(rot)
    rot_mat = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot_mat.T + np.array([x, y], dtype=np.float32)


def xy_to_rc(
    points_xy: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    h: int,
    w: int,
) -> np.ndarray:
    x0, x1 = x_range
    y0, y1 = y_range
    rows = (points_xy[:, 0] - x0) / (x1 - x0) * h
    cols = (points_xy[:, 1] - y0) / (y1 - y0) * w
    return np.stack([rows, cols], axis=1)


def point_in_poly(row: float, col: float, poly_rc: np.ndarray) -> bool:
    inside = False
    j = len(poly_rc) - 1
    for i in range(len(poly_rc)):
        ri, ci = poly_rc[i]
        rj, cj = poly_rc[j]
        crosses = (ci > col) != (cj > col)
        if crosses:
            r_at_col = (rj - ri) * (col - ci) / (cj - ci + 1e-12) + ri
            if row < r_at_col:
                inside = not inside
        j = i
    return inside


def rasterize(polys_rc: Iterable[np.ndarray], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for poly in polys_rc:
        r_min = max(0, int(math.floor(float(poly[:, 0].min()))))
        r_max = min(h - 1, int(math.ceil(float(poly[:, 0].max()))))
        c_min = max(0, int(math.floor(float(poly[:, 1].min()))))
        c_max = min(w - 1, int(math.ceil(float(poly[:, 1].max()))))
        if r_min > r_max or c_min > c_max:
            continue
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if point_in_poly(r + 0.5, c + 0.5, poly):
                    mask[r, c] = True
    return np.argwhere(mask).astype(np.int64)


def main() -> None:
    args = parse_args()
    dair_root = Path(args.dair_root)
    split_path = Path(args.split_path)
    output_root = Path(args.output_root)
    side = "vehicle-side" if args.sensor_view == "vehicle" else "infrastructure-side"
    classes = set(args.classes)
    split_ids = load_split_ids(split_path, args.sensor_view)

    info_path = dair_root / side / "data_info.json"
    infos = json.loads(info_path.read_text())
    if args.max_frames:
        infos = infos[: args.max_frames]

    scene_maps: dict[str, dict[str, int]] = {name: {} for name in split_ids}
    scene_next: dict[str, int] = {name: 0 for name in split_ids}
    counts = {name: 0 for name in split_ids}
    manifest_rows = []

    for info in infos:
        frame_id = frame_id_from_path(info.get("pointcloud_path") or info.get("image_path") or "")
        target_split = next((name for name, ids in split_ids.items() if frame_id in ids), None)
        if target_split is None:
            continue

        batch_id = str(info.get("batch_id", "0"))
        if batch_id not in scene_maps[target_split]:
            scene_maps[target_split][batch_id] = scene_next[target_split]
            scene_next[target_split] += 1
        scene_id = scene_maps[target_split][batch_id]
        frame_idx = counts[target_split]
        counts[target_split] += 1

        label_path = label_path_from_info(dair_root, args.sensor_view, info)
        labels = json.loads(label_path.read_text()) if label_path.exists() else []
        polys = []
        for label in labels:
            if label.get("type") not in classes:
                continue
            poly_xy = corners_xy(label)
            if poly_xy is None:
                continue
            poly_rc = xy_to_rc(poly_xy, tuple(args.x_range), tuple(args.y_range), args.height, args.width)
            polys.append(poly_rc)

        indices = rasterize(polys, args.height, args.width)
        sample_dir = output_root / target_split / f"agent_{args.sensor_view}" / f"{scene_id}_{frame_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(sample_dir / "0.npy", {"voxel_indices_0": indices})
        manifest_rows.append([target_split, args.sensor_view, batch_id, scene_id, frame_idx, frame_id, len(indices)])

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"manifest_{args.sensor_view}.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "sensor_view", "batch_id", "scene_id", "frame_idx", "frame_id", "occupied_cells"])
        writer.writerows(manifest_rows)

    print(f"wrote {len(manifest_rows)} frames to {output_root}")
    for split_name in sorted(counts):
        print(f"{split_name}: {counts[split_name]} frames")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
