#!/usr/bin/env python3
"""Prepare a DAIR-V2X-C compatibility view for TraF-Align.

TraF-Align's public DAIR config targets DAIR-V2X-Seq.  The local DAIR-V2X-C
copy lacks sequence fields and per-object track ids, so this script creates a
temporary compatibility view without modifying the source dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import yaml
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "datasets" / "DAIR-V2X" / "cooperative-vehicle-infrastructure"
DEFAULT_OUT = ROOT / "results" / "trafalign_dair_v2x_c_compat"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False)


def frame_id(path: str) -> str:
    return Path(path).stem


def normalize_type(value: str) -> str:
    mapping = {
        "car": "Car",
        "truck": "Truck",
        "van": "Van",
        "bus": "Bus",
        "pedestrian": "Pedestrian",
        "cyclist": "Cyclist",
        "tricyclist": "Tricyclist",
        "motorcyclist": "Motorcyclist",
        "barrowlist": "Barrowlist",
        "trafficcone": "Trafficcone",
    }
    return mapping.get(str(value).lower(), str(value))


def add_track_ids(labels: list[dict]) -> list[dict]:
    out = []
    for i, item in enumerate(labels):
        copied = dict(item)
        if "rotation" not in copied and "world_8_points" in copied:
            pts = copied["world_8_points"]
            if len(pts) >= 3:
                dx = float(pts[2][0]) - float(pts[1][0])
                dy = float(pts[2][1]) - float(pts[1][1])
                copied["rotation"] = math.atan2(dy, dx)
        if not all(k in copied for k in ("rotation", "3d_location", "3d_dimensions", "type")):
            continue
        copied["type"] = normalize_type(copied.get("type", "Car"))
        copied["rotation"] = float(copied["rotation"])
        copied["truncated_state"] = int(float(copied.get("truncated_state", 0)))
        copied["occluded_state"] = int(float(copied.get("occluded_state", 0)))
        for key in ("x", "y", "z"):
            copied["3d_location"][key] = float(copied["3d_location"][key])
        for key in ("h", "w", "l"):
            copied["3d_dimensions"][key] = float(copied["3d_dimensions"][key])
        copied.setdefault("track_id", i)
        copied["track_id"] = int(copied["track_id"])
        out.append(copied)
    return out


def vehicle_lidar_to_world(src_root: Path, veh_id: str, veh_infos: dict[str, dict]) -> np.ndarray | None:
    info = veh_infos.get(veh_id)
    if not info:
        return None
    lidar_to_novatel_path = src_root / "vehicle-side" / info["calib_lidar_to_novatel_path"]
    novatel_to_world_path = src_root / "vehicle-side" / info["calib_novatel_to_world_path"]
    if not lidar_to_novatel_path.exists() or not novatel_to_world_path.exists():
        return None
    lidar_to_novatel = read_json(lidar_to_novatel_path)["transform"]
    novatel_to_world = read_json(novatel_to_world_path)
    t1 = np.eye(4, dtype=np.float64)
    t1[:3, :3] = np.asarray(lidar_to_novatel["rotation"], dtype=np.float64)
    t1[:3, 3] = np.asarray(lidar_to_novatel["translation"], dtype=np.float64).reshape(3)
    t2 = np.eye(4, dtype=np.float64)
    t2[:3, :3] = np.asarray(novatel_to_world["rotation"], dtype=np.float64)
    t2[:3, 3] = np.asarray(novatel_to_world["translation"], dtype=np.float64).reshape(3)
    return t2 @ t1


def transform_world_labels_to_vehicle_lidar(labels: list[dict], world_to_lidar: np.ndarray) -> list[dict]:
    out = []
    for item in labels:
        copied = dict(item)
        if "world_8_points" in copied and len(copied["world_8_points"]) >= 8:
            pts = np.asarray(copied["world_8_points"], dtype=np.float64)
            homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
            lidar_pts = (world_to_lidar @ homo.T).T[:, :3]
            center = lidar_pts.mean(axis=0)
            copied["3d_location"] = {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}
            dx = float(lidar_pts[2, 0] - lidar_pts[1, 0])
            dy = float(lidar_pts[2, 1] - lidar_pts[1, 1])
            copied["rotation"] = math.atan2(dy, dx)
        elif "3d_location" in copied:
            loc = copied["3d_location"]
            point = np.asarray([float(loc["x"]), float(loc["y"]), float(loc["z"]), 1.0], dtype=np.float64)
            mapped = world_to_lidar @ point
            copied["3d_location"] = {"x": float(mapped[0]), "y": float(mapped[1]), "z": float(mapped[2])}
        out.append(copied)
    return out


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def info_by_frame(items: list[dict], agent: str) -> dict[str, dict]:
    out = {}
    for item in items:
        key = frame_id(item["pointcloud_path"])
        out[key] = dict(item)
    return out


def add_neighbor_ids(target: set[str], center: str, max_back: int = 20) -> None:
    width = len(center)
    try:
        base = int(center)
    except ValueError:
        target.add(center)
        return
    for delay in range(max_back + 1):
        value = base - delay
        if value >= 0:
            target.add(str(value).zfill(width))


def build(args: argparse.Namespace) -> None:
    src = args.src.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    cooperative_info_path = src / "cooperative" / "data_info_processed_updated.json"
    if not cooperative_info_path.exists():
        cooperative_info_path = src / "cooperative" / "data_info.json"
    coop_infos = read_json(cooperative_info_path)
    veh_infos = info_by_frame(read_json(src / "vehicle-side" / "data_info.json"), "vehicle")
    inf_infos = info_by_frame(read_json(src / "infrastructure-side" / "data_info.json"), "infrastructure")

    veh_to_seq: dict[str, str] = {}
    inf_to_seq: dict[str, str] = {}
    used_veh: set[str] = set()
    used_inf: set[str] = set()
    coop_out = []
    all_seqs: set[str] = set()

    for item in coop_infos:
        veh = frame_id(item["vehicle_pointcloud_path"])
        inf = frame_id(item["infrastructure_pointcloud_path"])
        veh_info = veh_infos.get(veh, {})
        seq = str(veh_info.get("batch_id", "0")).zfill(4)
        all_seqs.add(seq)
        veh_to_seq[veh] = seq
        inf_to_seq[inf] = seq
        add_neighbor_ids(used_veh, veh, args.history + args.max_delay)
        add_neighbor_ids(used_inf, inf, args.history + args.max_delay)

        row = dict(item)
        row["vehicle_sequence"] = seq
        row["infrastructure_sequence"] = seq
        row["vehicle_frame"] = veh
        row["infrastructure_frame"] = inf
        coop_out.append(row)

    for veh in list(used_veh):
        if veh not in veh_infos:
            continue
        seq = str(veh_infos[veh].get("batch_id", "0")).zfill(4)
        veh_to_seq.setdefault(veh, seq)
    for inf in list(used_inf):
        inf_to_seq.setdefault(inf, "0000")

    def filtered_side_infos(source: dict[str, dict], used: set[str], seq_map: dict[str, str]) -> list[dict]:
        rows = []
        for fid in sorted(used):
            if fid not in source:
                continue
            row = dict(source[fid])
            row["sequence_id"] = seq_map.get(fid, str(row.get("batch_id", "0")).zfill(4))
            rows.append(row)
        return rows

    write_json(out / "cooperative" / "data_info.json", coop_out)
    write_json(out / "vehicle-side" / "data_info.json", filtered_side_infos(veh_infos, used_veh, veh_to_seq))
    write_json(out / "infrastructure-side" / "data_info.json", filtered_side_infos(inf_infos, used_inf, inf_to_seq))

    for side in ("vehicle-side", "infrastructure-side"):
        for name in ("velodyne", "image", "calib"):
            safe_symlink(src / side / name, out / side / name)

    # TraF-Align expects cooperative/label/ keyed by vehicle frame id.
    coop_label_ids = set(used_veh)
    for fid in sorted(coop_label_ids):
        src_label = src / "cooperative" / "label_world" / f"{fid}.json"
        if src_label.exists():
            lidar_to_world = vehicle_lidar_to_world(src, fid, veh_infos)
            if lidar_to_world is None:
                continue
            world_to_lidar = np.linalg.inv(lidar_to_world)
            labels = transform_world_labels_to_vehicle_lidar(read_json(src_label), world_to_lidar)
            write_json(out / "cooperative" / "label" / f"{fid}.json", add_track_ids(labels))

    for fid in sorted(used_veh):
        src_label = src / "vehicle-side" / "label" / "lidar" / f"{fid}.json"
        if src_label.exists():
            write_json(out / "vehicle-side" / "label" / "lidar" / f"{fid}.json", add_track_ids(read_json(src_label)))

    for fid in sorted(used_inf):
        src_label = src / "infrastructure-side" / "label" / "virtuallidar" / f"{fid}.json"
        if src_label.exists():
            write_json(
                out / "infrastructure-side" / "label" / "virtuallidar" / f"{fid}.json",
                add_track_ids(read_json(src_label)),
            )

    test_ids = set(read_json(src / "test.json")) if (src / "test.json").exists() else set()
    val_ids = set(read_json(src / "val.json")) if (src / "val.json").exists() else set()
    train_ids = set(read_json(src / "train.json")) if (src / "train.json").exists() else set()
    split = {"train": set(), "val": set(), "test": set(), "debug": set()}
    for item in coop_out:
        seq = item["vehicle_sequence"]
        veh = item["vehicle_frame"]
        if veh in test_ids:
            split["test"].add(seq)
        if veh in val_ids:
            split["val"].add(seq)
        if veh in train_ids:
            split["train"].add(seq)
    if not split["test"]:
        split["test"] = set(all_seqs)
    split["debug"] = set(sorted(split["test"])[:1])
    split_out = {k: sorted(v) for k, v in split.items()}
    with (out / "trafalign_split.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(split_out, handle, sort_keys=False)

    meta = {
        "source": str(src),
        "output": str(out),
        "cooperative_frames": len(coop_out),
        "vehicle_infos": len(filtered_side_infos(veh_infos, used_veh, veh_to_seq)),
        "infrastructure_infos": len(filtered_side_infos(inf_infos, used_inf, inf_to_seq)),
        "sequence_counts": {k: len(v) for k, v in split.items()},
        "note": "Compatibility view for TraF-Align on DAIR-V2X-C; track_id is pseudo-generated.",
    }
    write_json(out / "metadata.json", meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--max-delay", type=int, default=10)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
