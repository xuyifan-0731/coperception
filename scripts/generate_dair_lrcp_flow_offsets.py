#!/usr/bin/env python3
"""Generate DAIR-V2X LRCP BEV flow offsets from local labels.

LRCP's DAIR loader expects ``offset_maps_fix_flip/offset_<veh_id>.npy`` with
shape ``[delay, bev_h, bev_w, 2]``.  The official DAIR flow archive is not
available locally, so this script follows LRCP's V2X-Sim flow-label logic:

* project current and delayed world boxes into the current vehicle lidar frame;
* match objects across frames by class and BEV center distance;
* fill each current BEV box with the rigid 2D offset from current to delayed box.

The resulting offsets are detector-training labels, not placeholders.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAIR_ROOT = ROOT / "datasets" / "DAIR-V2X" / "cooperative-vehicle-infrastructure"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dair-root", type=Path, default=DEFAULT_DAIR_ROOT)
    parser.add_argument("--processed-info", type=Path, default=None)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--bev-h", type=int, default=80)
    parser.add_argument("--bev-w", type=int, default=80)
    parser.add_argument("--pc-range", type=float, nargs=6, default=[-32, -32, -3, 32, 32, 2])
    parser.add_argument("--match-dist", type=float, default=6.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit on number of cooperative frames.")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def veh_id(row: dict) -> str:
    return Path(row["vehicle_pointcloud_path"]).stem


def inf_id_from_path(path: str) -> str:
    return Path(path).stem


def normalize_type(name: str) -> str:
    name = str(name).lower()
    if name in {"car", "van", "truck", "bus"}:
        return "vehicle"
    if name in {"pedestrian", "cyclist", "motorcyclist", "tricyclist"}:
        return name
    return name


def make_tfm(rotation: Iterable[Iterable[float]], translation: Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3, 1)[:, 0]
    return matrix


def vehicle_lidar_to_world(dair_root: Path, vid: str) -> np.ndarray:
    lidar_to_novatel = read_json(dair_root / "vehicle-side" / "calib" / "lidar_to_novatel" / f"{vid}.json")
    novatel_to_world = read_json(dair_root / "vehicle-side" / "calib" / "novatel_to_world" / f"{vid}.json")

    rot_a = np.asarray(lidar_to_novatel["transform"]["rotation"], dtype=np.float64).reshape(3, 3)
    trans_a = np.asarray(lidar_to_novatel["transform"]["translation"], dtype=np.float64).reshape(3, 1)
    rot_b = np.asarray(novatel_to_world["rotation"], dtype=np.float64).reshape(3, 3)
    trans_b = np.asarray(novatel_to_world["translation"], dtype=np.float64).reshape(3, 1)

    rot = rot_b @ rot_a
    trans = rot_b @ trans_a + trans_b
    return make_tfm(rot, trans)


def world_boxes_in_current_vehicle(dair_root: Path, row: dict, world_to_current_vehicle: np.ndarray) -> list[dict]:
    labels = read_json(dair_root / row["cooperative_label_path"])
    out = []
    for idx, obj in enumerate(labels):
        if "world_8_points" not in obj:
            continue
        corners_world = np.asarray(obj["world_8_points"], dtype=np.float64)
        corners_world_h = np.pad(corners_world, ((0, 0), (0, 1)), constant_values=1.0)
        corners_lidar = (world_to_current_vehicle @ corners_world_h.T).T[:, :3]
        bottom = corners_lidar[:4, :2]
        center = bottom.mean(axis=0)
        out.append(
            {
                "id": idx,
                "type": normalize_type(obj.get("type", "")),
                "bottom_xy": bottom,
                "center": center,
            }
        )
    return out


def box_to_bev(box_xy: np.ndarray, pc_range: list[float], bev_h: int, bev_w: int) -> np.ndarray:
    out = box_xy.copy()
    out[:, 0] = (out[:, 0] - pc_range[0]) * (bev_w / (pc_range[3] - pc_range[0]))
    out[:, 1] = (out[:, 1] - pc_range[1]) * (bev_h / (pc_range[4] - pc_range[1]))
    return out


def center_yaw(box: np.ndarray) -> np.ndarray:
    center = box.mean(axis=0)
    diff = box[1] - box[2]
    yaw = math.atan2(float(diff[1]), float(diff[0]))
    return np.array([center[0], center[1], yaw], dtype=np.float64)


def pose2d(pose: np.ndarray) -> np.ndarray:
    x, y, yaw = pose
    return np.array(
        [
            [math.cos(yaw), -math.sin(yaw), x],
            [math.sin(yaw), math.cos(yaw), y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def transform_current_to_past(current_box: np.ndarray, past_box: np.ndarray) -> np.ndarray:
    return pose2d(center_yaw(past_box)) @ np.linalg.inv(pose2d(center_yaw(current_box)))


def match_boxes(current: list[dict], past: list[dict], max_dist: float) -> list[tuple[dict, dict]]:
    candidates = []
    for cur in current:
        for old in past:
            if cur["type"] != old["type"]:
                continue
            dist = float(np.linalg.norm(cur["center"] - old["center"]))
            if dist <= max_dist:
                candidates.append((dist, cur, old))
    candidates.sort(key=lambda x: x[0])

    used_cur: set[int] = set()
    used_old: set[int] = set()
    matches = []
    for _, cur, old in candidates:
        if cur["id"] in used_cur or old["id"] in used_old:
            continue
        used_cur.add(cur["id"])
        used_old.add(old["id"])
        matches.append((cur, old))
    return matches


def fill_offset(offset_map: np.ndarray, current_bev: np.ndarray, past_bev: np.ndarray) -> None:
    pts = np.round(current_bev).astype(np.int32)
    mask = np.zeros(offset_map.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return

    points = np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float64)
    tfm = transform_current_to_past(current_bev, past_bev)
    transformed = (tfm @ points.T).T[:, :2]
    valid = (
        (transformed[:, 0] >= 0)
        & (transformed[:, 0] <= offset_map.shape[1] - 1)
        & (transformed[:, 1] >= 0)
        & (transformed[:, 1] <= offset_map.shape[0] - 1)
    )
    if not np.any(valid):
        return
    src = points[valid, :2]
    dst = transformed[valid]
    offset_map[src[:, 1].astype(np.int64), src[:, 0].astype(np.int64), :] = (dst - src).astype(np.float32)


def generate_one(
    dair_root: Path,
    row: dict,
    by_inf_id: dict[str, dict],
    max_history: int,
    pc_range: list[float],
    bev_h: int,
    bev_w: int,
    match_dist: float,
) -> np.ndarray:
    current_vid = veh_id(row)
    world_to_current_vehicle = np.linalg.inv(vehicle_lidar_to_world(dair_root, current_vid))
    current_boxes = world_boxes_in_current_vehicle(dair_root, row, world_to_current_vehicle)
    current_bev = {
        box["id"]: box_to_bev(box["bottom_xy"], pc_range, bev_h, bev_w)
        for box in current_boxes
    }

    offsets = np.zeros((max_history, bev_h, bev_w, 2), dtype=np.float32)
    for delay in range(1, max_history + 1):
        prev = row.get(f"previous_inf_{delay}")
        if not prev:
            continue
        past_row = by_inf_id.get(inf_id_from_path(prev[0]))
        if past_row is None:
            continue
        past_boxes = world_boxes_in_current_vehicle(dair_root, past_row, world_to_current_vehicle)
        past_bev = {
            box["id"]: box_to_bev(box["bottom_xy"], pc_range, bev_h, bev_w)
            for box in past_boxes
        }
        for cur, old in match_boxes(current_boxes, past_boxes, match_dist):
            fill_offset(offsets[delay - 1], current_bev[cur["id"]], past_bev[old["id"]])
    return offsets


def main() -> None:
    args = parse_args()
    dair_root = args.dair_root.resolve()
    processed_info = args.processed_info or dair_root / "cooperative" / "data_info_processed_updated.json"
    rows = read_json(processed_info)
    by_inf_id = {inf_id_from_path(row["infrastructure_pointcloud_path"]): row for row in rows}

    out_dir = dair_root / "offset_maps_fix_flip"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = rows[: args.limit] if args.limit else rows
    written = 0
    skipped = 0
    nonzero = 0
    for row in tqdm(selected, desc="dair_lrcp_offsets"):
        path = out_dir / f"offset_{veh_id(row)}.npy"
        if path.exists() and not args.force:
            skipped += 1
            continue
        arr = generate_one(
            dair_root=dair_root,
            row=row,
            by_inf_id=by_inf_id,
            max_history=args.max_history,
            pc_range=args.pc_range,
            bev_h=args.bev_h,
            bev_w=args.bev_w,
            match_dist=args.match_dist,
        )
        np.save(path, arr)
        written += 1
        if np.count_nonzero(arr):
            nonzero += 1
    print(f"written={written} skipped={skipped} nonzero_files={nonzero} out_dir={out_dir}")


if __name__ == "__main__":
    main()
