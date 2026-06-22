#!/usr/bin/env python3
"""Generate DATA/OpenCOOD V2X-Sim 2.0 info pkl files.

The DATA V2X-Sim loader expects pkl samples with:
agent_num, lidar_path_i, lidar_pose_i, labels_i.gt_boxes_global,
labels_i.gt_object_ids.  This script builds those files from the official
NuScenes-style V2X-Sim 2.0 release.
"""

import argparse
import json
import os
import pickle
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "external" / "DATA"
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from nuscenes.nuscenes import NuScenes  # noqa: E402
from opencood.utils.box_utils import (  # noqa: E402
    corner_to_center,
    create_bbx,
    get_points_in_rotated_box_3d,
    mask_boxes_outside_range_numpy,
)


TRAIN_SPLIT = [
    82, 25, 95, 0, 2, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
    21, 22, 23, 24, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 64, 66, 67, 69, 70, 71, 72, 73, 74, 75, 77, 80, 81,
    83, 85, 86, 87, 88, 89, 90, 93, 94, 98, 99,
]
VAL_SPLIT = [1, 3, 4, 63, 65, 68, 76, 78, 79, 84]
TEST_SPLIT = [5, 8, 19, 27, 28, 29, 91, 92, 96, 97]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default=str(REPO_ROOT / "datasets" / "V2X-Sim-2.0"))
    parser.add_argument("--version", default="auto")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "datasets" / "v2xsim2_info"))
    parser.add_argument("--n-track-frame", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--lidar-range", default="-90,-90,-3,90,90,2")
    parser.add_argument("--backend", choices=["fast-json", "nuscenes"], default="fast-json")
    return parser.parse_args()


def resolve_version(dataroot, version):
    root = Path(dataroot)
    if version != "auto":
        return version
    for candidate in ("v2xsim-det", "v1.0-mini", "v2.0"):
        if (root / candidate / "scene.json").exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find NuScenes metadata under {root}/v2.0 or {root}/v1.0-mini")


def build_hash_map(nusc):
    instance_tokens = sorted({
        nusc.get("sample_annotation", ann)["instance_token"]
        for sample in nusc.sample
        for ann in sample["anns"]
    })
    return {token: idx for idx, token in enumerate(instance_tokens)}


def world_box_visible_from_lidar(box, lidar_np, t_world_lidar, lidar_range):
    x, y, z, dx, dy, dz, w, a, b, c = box
    quat = Quaternion([w, a, b, c])
    t_world_object = quat.transformation_matrix
    t_world_object[:3, 3] = [x, y, z]
    object_to_lidar = np.linalg.solve(t_world_lidar, t_world_object)

    x_corners = dx / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = dy / 2 * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    z_corners = dz / 2 * np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    bbx = np.vstack((x_corners, y_corners, z_corners))
    bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]
    bbx_lidar = np.dot(object_to_lidar, bbx).T[:, :3]

    center = corner_to_center(np.expand_dims(bbx_lidar, 0), order="hwl")
    if mask_boxes_outside_range_numpy(center, lidar_range, "hwl").shape[0] == 0:
        return False

    enlarge = 0.2
    enlarge_extent = [dx / 2 + enlarge, dy / 2 + enlarge, dz / 2 + enlarge]
    enlarge_bbx = np.r_[create_bbx(enlarge_extent).T,
                         [np.ones(create_bbx(enlarge_extent).shape[0])]]
    enlarge_bbx_lidar = np.dot(object_to_lidar, enlarge_bbx).T[:, :3]
    return get_points_in_rotated_box_3d(lidar_np[:, :3], enlarge_bbx_lidar).shape[0] > 0


def sample_agent_num(sample):
    ids = []
    for key in sample["data"].keys():
        if key.startswith("LIDAR_TOP_id_"):
            ids.append(int(key.split("_")[-1]))
    if not ids:
        raise RuntimeError(f"No LIDAR_TOP_id_* entry in sample {sample['token']}")
    return max(ids)


def fill_split_info(nusc, scene_ids, hash_map, n_track_frame, lidar_range):
    split_infos = []
    for scene_id in tqdm(scene_ids, desc="scenes"):
        scene = nusc.scene[scene_id]
        sample_token = scene["first_sample_token"]
        frame_id = 0
        while sample_token:
            sample = nusc.get("sample", sample_token)
            agent_num = sample_agent_num(sample)
            info = {
                "token": sample["token"],
                "sample_token": sample_token,
                "timestamp": sample["timestamp"],
                "scene_id": scene_id,
                "scene_token": scene["token"],
                "frame_id": frame_id,
                "agent_num": agent_num,
            }

            boxes = nusc.get_boxes(sample["data"]["LIDAR_TOP_id_1"])
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)[:, [1, 0, 2]]
            rots = np.array([b.orientation.elements for b in boxes]).reshape(-1, 4)
            names = np.array([b.name for b in boxes])
            tokens = np.array([b.token for b in boxes])
            object_ids = np.array([
                hash_map[nusc.get("sample_annotation", token)["instance_token"]]
                for token in tokens
            ])
            gt_boxes = np.concatenate([locs, dims, rots], axis=1)
            vehicle_mask = np.array([
                name.startswith("vehicle") and dim[1] > 1.5
                for name, dim in zip(names, dims)
            ], dtype=bool)
            names = names[vehicle_mask]
            tokens = tokens[vehicle_mask]
            gt_boxes = gt_boxes[vehicle_mask]
            object_ids = object_ids[vehicle_mask]

            for cav_id in range(1, agent_num + 1):
                lidar_sample_data = nusc.get(
                    "sample_data", sample["data"][f"LIDAR_TOP_id_{cav_id}"])
                info[f"lidar_path_{cav_id}"] = nusc.get_sample_data_path(
                    lidar_sample_data["token"])

                ego_pose_record = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
                q_world_ego = Quaternion(ego_pose_record["rotation"])
                t_world_ego = q_world_ego.transformation_matrix
                t_world_ego[:3, 3] = ego_pose_record["translation"]

                cs_record = nusc.get(
                    "calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
                q_ego_lidar = Quaternion(cs_record["rotation"])
                t_ego_lidar = q_ego_lidar.transformation_matrix
                t_ego_lidar[:3, 3] = cs_record["translation"]
                t_world_lidar = np.dot(t_world_ego, t_ego_lidar)
                info[f"lidar_pose_{cav_id}"] = t_world_lidar

                scan = np.fromfile(info[f"lidar_path_{cav_id}"], dtype="float32")
                lidar_np = scan.reshape((-1, 5))[:, :4]
                box_mask = np.array([
                    world_box_visible_from_lidar(box, lidar_np, t_world_lidar, lidar_range)
                    for box in gt_boxes
                ], dtype=bool)

                info[f"labels_{cav_id}"] = OrderedDict()
                info[f"labels_{cav_id}"]["gt_names"] = names[box_mask]
                info[f"labels_{cav_id}"]["gt_boxes_token"] = tokens[box_mask]
                info[f"labels_{cav_id}"]["gt_boxes_global"] = gt_boxes[box_mask]
                info[f"labels_{cav_id}"]["gt_object_ids"] = object_ids[box_mask]

            tmp_sample = sample
            info["prev_samples"] = {i: None for i in range(1, n_track_frame + 1)}
            for i in range(1, n_track_frame + 1):
                prev_token = tmp_sample["prev"]
                if not prev_token:
                    break
                info["prev_samples"][i] = prev_token
                tmp_sample = nusc.get("sample", prev_token)

            split_infos.append(info)
            sample_token = sample["next"]
            frame_id += 1

    return split_infos


def save_split(name, infos, output_dir):
    path = Path(output_dir) / f"v2xsim_infos_{name}.pkl"
    with path.open("wb") as f:
        pickle.dump(infos, f)
    return path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_tables_fast(dataroot, version):
    table_dir = Path(dataroot) / version
    needed = [
        "scene", "sample", "sample_data", "ego_pose", "calibrated_sensor",
        "sensor", "sample_annotation", "instance", "category",
    ]
    tables = {}
    for name in needed:
        path = table_dir / f"{name}.json"
        print(f"loading {path}", flush=True)
        tables[name] = load_json(path)
        print(f"  {name}: {len(tables[name])}", flush=True)
    return tables


def build_fast_indices(tables):
    samples = {item["token"]: item for item in tables["sample"]}
    scenes = tables["scene"]
    ego_pose = {item["token"]: item for item in tables["ego_pose"]}
    calibrated = {item["token"]: item for item in tables["calibrated_sensor"]}
    sensor = {item["token"]: item for item in tables["sensor"]}
    category = {item["token"]: item["name"] for item in tables["category"]}
    instance_name = {
        item["token"]: category[tables_category_token]
        for item in tables["instance"]
        for tables_category_token in [item["category_token"]]
    }
    instance_tokens = sorted(item["token"] for item in tables["instance"])
    hash_map = {token: idx for idx, token in enumerate(instance_tokens)}

    cal_to_channel = {}
    for token, cal in calibrated.items():
        cal_to_channel[token] = sensor[cal["sensor_token"]]["channel"]

    sample_data_by_sample = defaultdict(dict)
    for sd in tables["sample_data"]:
        channel = cal_to_channel[sd["calibrated_sensor_token"]]
        sample_data_by_sample[sd["sample_token"]][channel] = sd

    ann_by_sample = defaultdict(list)
    for ann in tables["sample_annotation"]:
        ann_by_sample[ann["sample_token"]].append(ann)

    return {
        "samples": samples,
        "scenes": scenes,
        "ego_pose": ego_pose,
        "calibrated": calibrated,
        "sample_data_by_sample": sample_data_by_sample,
        "ann_by_sample": ann_by_sample,
        "instance_name": instance_name,
        "hash_map": hash_map,
    }


def lidar_channels(sample_data_map):
    return sorted(
        [key for key in sample_data_map.keys() if key.startswith("LIDAR_TOP_id_")],
        key=lambda key: int(key.split("_")[-1]),
    )


def pose_from_sample_data(sd, indices):
    ego_pose_record = indices["ego_pose"][sd["ego_pose_token"]]
    q_world_ego = Quaternion(ego_pose_record["rotation"])
    t_world_ego = q_world_ego.transformation_matrix
    t_world_ego[:3, 3] = ego_pose_record["translation"]

    cs_record = indices["calibrated"][sd["calibrated_sensor_token"]]
    q_ego_lidar = Quaternion(cs_record["rotation"])
    t_ego_lidar = q_ego_lidar.transformation_matrix
    t_ego_lidar[:3, 3] = cs_record["translation"]
    return np.dot(t_world_ego, t_ego_lidar)


def in_lidar_range(box, t_world_lidar, lidar_range):
    x, y, z, _, _, _, w, a, b, c = box
    quat = Quaternion([w, a, b, c])
    t_world_object = quat.transformation_matrix
    t_world_object[:3, 3] = [x, y, z]
    object_to_lidar = np.linalg.solve(t_world_lidar, t_world_object)
    cx, cy, cz = object_to_lidar[:3, 3]
    return (
        lidar_range[0] <= cx <= lidar_range[3]
        and lidar_range[1] <= cy <= lidar_range[4]
        and lidar_range[2] <= cz <= lidar_range[5]
    )


def ann_hits_agent(ann, cav_id):
    hits = ann.get("num_lidar_pts", [])
    if isinstance(hits, int):
        return hits > 0
    if cav_id < len(hits):
        return hits[cav_id] > 0
    if cav_id - 1 < len(hits):
        return hits[cav_id - 1] > 0
    return sum(hits) > 0


def build_sample_labels(sample_token, cav_id, t_world_lidar, indices, lidar_range):
    names, tokens, object_ids, boxes = [], [], [], []
    for ann in indices["ann_by_sample"].get(sample_token, []):
        name = indices["instance_name"].get(ann["instance_token"], "")
        size = ann["size"]
        box = np.array([
            ann["translation"][0], ann["translation"][1], ann["translation"][2],
            size[1], size[0], size[2],
            ann["rotation"][0], ann["rotation"][1], ann["rotation"][2], ann["rotation"][3],
        ], dtype=np.float32)
        if not (name.startswith("vehicle") and box[4] > 1.5):
            continue
        if not ann_hits_agent(ann, cav_id):
            continue
        if not in_lidar_range(box, t_world_lidar, lidar_range):
            continue
        names.append(name)
        tokens.append(ann["token"])
        object_ids.append(indices["hash_map"][ann["instance_token"]])
        boxes.append(box)

    return OrderedDict({
        "gt_names": np.array(names),
        "gt_boxes_token": np.array(tokens),
        "gt_boxes_global": np.array(boxes, dtype=np.float32).reshape(-1, 10),
        "gt_object_ids": np.array(object_ids, dtype=np.int64),
    })


def fill_split_info_fast(dataroot, scene_ids, indices, n_track_frame, lidar_range):
    split_infos = []
    scenes = indices["scenes"]
    samples = indices["samples"]
    sample_data_by_sample = indices["sample_data_by_sample"]

    for scene_id in tqdm(scene_ids, desc="scenes"):
        scene = scenes[scene_id]
        sample_token = scene["first_sample_token"]
        frame_id = 0
        while sample_token:
            sample = samples[sample_token]
            sd_map = sample_data_by_sample[sample_token]
            channels = lidar_channels(sd_map)
            agent_ids = [int(channel.split("_")[-1]) for channel in channels]
            agent_num = max(agent_ids)
            info = {
                "token": sample["token"],
                "sample_token": sample_token,
                "timestamp": sample["timestamp"],
                "scene_id": scene_id,
                "scene_token": scene["token"],
                "frame_id": frame_id,
                "agent_num": agent_num,
            }

            for cav_id in range(1, agent_num + 1):
                channel = f"LIDAR_TOP_id_{cav_id}"
                if channel not in sd_map:
                    continue
                sd = sd_map[channel]
                lidar_path = Path(dataroot) / sd["filename"]
                info[f"lidar_path_{cav_id}"] = str(lidar_path)
                t_world_lidar = pose_from_sample_data(sd, indices)
                info[f"lidar_pose_{cav_id}"] = t_world_lidar
                info[f"labels_{cav_id}"] = build_sample_labels(
                    sample_token, cav_id, t_world_lidar, indices, lidar_range)

            tmp_sample = sample
            info["prev_samples"] = {i: None for i in range(1, n_track_frame + 1)}
            for i in range(1, n_track_frame + 1):
                prev_token = tmp_sample["prev"]
                if not prev_token:
                    break
                info["prev_samples"][i] = prev_token
                tmp_sample = samples[prev_token]

            split_infos.append(info)
            sample_token = sample["next"]
            frame_id += 1
    return split_infos


def create_pkl_fast(args, version, lidar_range):
    tables = load_tables_fast(args.dataroot, version)
    indices = build_fast_indices(tables)
    splits = {
        "train": TRAIN_SPLIT,
        "val": VAL_SPLIT,
        "test": TEST_SPLIT,
    }
    summary = {"version": version, "dataroot": os.path.abspath(args.dataroot), "backend": "fast-json", "splits": {}}
    for split_name, scene_ids in splits.items():
        infos = fill_split_info_fast(
            args.dataroot, scene_ids, indices, args.n_track_frame, lidar_range)
        path = save_split(split_name, infos, args.output_dir)
        summary["splits"][split_name] = {
            "scene_count": len(scene_ids),
            "sample_count": len(infos),
            "path": str(path),
        }
        print(f"{split_name}: {len(infos)} samples -> {path}", flush=True)
    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / f"v2xsim_infos_{s}.pkl" for s in ("train", "val", "test")]
    if args.skip_existing and all(path.exists() for path in expected):
        print("Existing pkl files found; skip generation.")
        return

    version = resolve_version(args.dataroot, args.version)
    lidar_range = [float(x) for x in args.lidar_range.split(",")]
    if args.backend == "fast-json":
        summary = create_pkl_fast(args, version, lidar_range)
    else:
        nusc = NuScenes(version=version, dataroot=args.dataroot, verbose=True)
        if len(nusc.scene) != 100:
            raise RuntimeError(f"Expected 100 scenes for V2X-Sim 2.0, got {len(nusc.scene)}")

        hash_map = build_hash_map(nusc)
        splits = {
            "train": TRAIN_SPLIT,
            "val": VAL_SPLIT,
            "test": TEST_SPLIT,
        }
        summary = {"version": version, "dataroot": os.path.abspath(args.dataroot), "backend": "nuscenes", "splits": {}}
        for split_name, scene_ids in splits.items():
            infos = fill_split_info(
                nusc, scene_ids, hash_map, args.n_track_frame, lidar_range)
            path = save_split(split_name, infos, output_dir)
            summary["splits"][split_name] = {
                "scene_count": len(scene_ids),
                "sample_count": len(infos),
                "path": str(path),
            }
            print(f"{split_name}: {len(infos)} samples -> {path}")

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
