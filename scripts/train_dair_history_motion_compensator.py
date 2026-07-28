#!/usr/bin/env python3
"""Train a detector-side DAIR motion compensator with per-agent history."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from train_dair_motion_compensator import (
    ROOT,
    box_features,
    box_info,
    fit_normalizer,
    frame_id,
    id_to_str,
    label_boxes,
    load_infos,
    ridge_fit,
    predict,
    split_coop_infos,
    train_mlp,
    transform_boxes,
)
from v2x_utils.transformation_utils import Coord_transformation


RAW_PRED_CACHE: dict[tuple[str, str], dict | None] = {}
TRANSFORM_CACHE: dict[tuple[str, str, str, str, str], Coord_transformation] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dair-root", type=Path, default=ROOT / "datasets/DAIR-V2X/cooperative-vehicle-infrastructure")
    parser.add_argument(
        "--split-data-path",
        type=Path,
        default=ROOT / "external/DAIR-V2X/data/split_datas/cooperative-split-data.json",
    )
    parser.add_argument(
        "--cache-roots",
        default=",".join(
            str(ROOT / "external" / "DAIR-V2X-isolated" / f"gpu{gpu}" / "cache" / "vic-late-lidar")
            for gpu in [0, 1, 2, 3, 4, 6]
        ),
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results/260615_fill/artifacts/dair_ours_history_motion_compensator.npz")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--history-frames", type=int, default=2)
    parser.add_argument("--history-stride", choices=["horizon", "consecutive"], default="horizon")
    parser.add_argument("--max-match-dist", type=float, default=8.0)
    parser.add_argument("--history-match-dist", type=float, default=8.0)
    parser.add_argument("--extended-range", nargs="+", type=float, default=[0, -39.68, -3, 100, 39.68, 1])
    parser.add_argument("--max-train-frames", type=int, default=0)
    parser.add_argument("--max-val-frames", type=int, default=0)
    parser.add_argument("--model-kind", choices=["ridge", "mlp", "rf"], default="mlp")
    parser.add_argument("--alphas", default="0,1e-6,1e-4,1e-3,1e-2,1e-1,1")
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=18)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=4)
    parser.add_argument("--rf-max-samples", type=float, default=0.85)
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=20260705)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def history_feature_dim(history_frames: int) -> int:
    return 21 + 5 * max(int(history_frames) - 1, 0)


def raw_pred(cache_roots: list[Path], side: str, idx: str) -> dict | None:
    cache_key = (side, idx)
    if cache_key in RAW_PRED_CACHE:
        return RAW_PRED_CACHE[cache_key]
    for cache_root in cache_roots:
        path = cache_root / side / "lidar_raw" / f"{idx}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                RAW_PRED_CACHE[cache_key] = pickle.load(handle)
                return RAW_PRED_CACHE[cache_key]
    RAW_PRED_CACHE[cache_key] = None
    return RAW_PRED_CACHE[cache_key]


def cached_transform(from_coord: str, to_coord: str, root: Path, infra_name: str, veh_name: str) -> Coord_transformation:
    key = (from_coord, to_coord, str(root), infra_name, veh_name)
    if key not in TRANSFORM_CACHE:
        TRANSFORM_CACHE[key] = Coord_transformation(from_coord, to_coord, str(root), infra_name, veh_name)
    return TRANSFORM_CACHE[key]


def valid_detector_boxes(pred: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(pred["boxes_3d"], dtype=np.float64)
    labels = np.asarray(pred["labels_3d"], dtype=np.float64)
    scores = np.asarray(pred["scores_3d"], dtype=np.float64)
    if len(boxes) == 0:
        return boxes, labels, scores
    mask = (scores > 0.0) & (labels >= 0.0) & (np.linalg.norm(boxes.reshape(len(boxes), -1), axis=1) > 1e-6)
    if np.any(mask):
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
    else:
        return np.zeros((0, 8, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return boxes, labels, scores


def range_filter_boxes(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    extended_range: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, labels, scores
    lo = np.asarray(extended_range[:3], dtype=np.float64)
    hi = np.asarray(extended_range[3:], dtype=np.float64)
    in_range = np.any(np.all((boxes >= lo) & (boxes <= hi), axis=2), axis=1)
    return boxes[in_range], labels[in_range], scores[in_range]


def vehicle_to_target(boxes: np.ndarray, root: Path, source_veh: dict, target_veh: dict) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    source_name = frame_id(source_veh["image_path"])
    target_name = frame_id(target_veh["image_path"])
    if source_name == target_name:
        return boxes.astype(np.float64)
    to_world = cached_transform("Vehicle_lidar", "World", root, "", source_name)
    to_target = cached_transform("World", "Vehicle_lidar", root, "", target_name)
    return to_target(to_world(boxes)).astype(np.float64)


def infrastructure_to_target(boxes: np.ndarray, root: Path, source_inf: dict, target_veh: dict) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    trans = cached_transform(
        "Infrastructure_lidar",
        "Vehicle_lidar",
        root,
        frame_id(source_inf["image_path"]),
        frame_id(target_veh["image_path"]),
    )
    return trans(boxes).astype(np.float64)


def detector_history_features(
    cur_boxes: np.ndarray,
    cur_labels: np.ndarray,
    cur_scores: np.ndarray,
    prev_boxes_list: list[np.ndarray],
    prev_labels_list: list[np.ndarray],
    prev_time_diffs: list[float],
    horizon: int,
    target_time_diff: float,
    side: str,
    history_match_dist: float,
    history_frames: int,
) -> np.ndarray:
    base = box_features(cur_boxes, horizon)
    n = len(cur_boxes)
    history_slots = max(int(history_frames) - 1, 0)
    hist_delta = np.zeros((n, history_slots * 2), dtype=np.float64)
    temporal_delta = np.zeros((n, history_slots * 2), dtype=np.float64)
    has_prev = np.zeros((n, history_slots), dtype=np.float64)
    if n and history_slots:
        cur_center, _ = box_info(cur_boxes)
        for slot in range(history_slots):
            if slot >= len(prev_boxes_list) or slot >= len(prev_time_diffs):
                continue
            prev_boxes = prev_boxes_list[slot]
            prev_labels = prev_labels_list[slot]
            prev_time_diff = float(prev_time_diffs[slot])
            if len(prev_boxes) == 0 or prev_time_diff <= 0:
                continue
            prev_center, _ = box_info(prev_boxes)
            dist = np.linalg.norm(prev_center[:, None, :2] - cur_center[None, :, :2], axis=2)
            dist[np.asarray(prev_labels)[:, None] != np.asarray(cur_labels)[None, :]] = 1e6
            rows, cols = linear_sum_assignment(dist)
            keep = dist[rows, cols] <= history_match_dist
            if not np.any(keep):
                continue
            rows = rows[keep]
            cols = cols[keep]
            delta = cur_center[cols, :2] - prev_center[rows, :2]
            hist_delta[cols, slot * 2 : slot * 2 + 2] = delta
            temporal_delta[cols, slot * 2 : slot * 2 + 2] = delta * (target_time_diff / max(prev_time_diff, 1e-6))
            has_prev[cols, slot] = 1.0
    score = np.asarray(cur_scores, dtype=np.float64)[:, None]
    label = np.asarray(cur_labels, dtype=np.float64)[:, None]
    side_flag = np.full((n, 1), 1.0 if side == "veh" else 0.0, dtype=np.float64)
    return np.concatenate([base, hist_delta, temporal_delta, has_prev, score, label, side_flag], axis=1)


def side_examples(
    args: argparse.Namespace,
    cache_roots: list[Path],
    root: Path,
    cur_inf: dict,
    cur_veh: dict,
    cur_gt_boxes: np.ndarray,
    horizon: int,
    side: str,
    inf_map: dict[str, dict],
    veh_map: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, int]:
    feat_dim = history_feature_dim(args.history_frames)
    if side == "inf":
        cur_id = int(frame_id(cur_inf["pointcloud_path"]))
        source_key = "infrastructure-side/velodyne/" + id_to_str(cur_id - horizon) + ".pcd"
        source_info = inf_map.get(source_key)
        if source_info is None or source_info.get("batch_id") != cur_inf.get("batch_id"):
            return np.zeros((0, feat_dim)), np.zeros((0, 2)), 0
        source_pred = raw_pred(cache_roots, "inf", frame_id(source_info["pointcloud_path"]))
        if source_pred is None:
            return np.zeros((0, feat_dim)), np.zeros((0, 2)), 1
        source_boxes, source_labels, source_scores = valid_detector_boxes(source_pred)
        source_boxes = infrastructure_to_target(source_boxes, root, source_info, cur_veh)
        source_boxes, source_labels, source_scores = range_filter_boxes(
            source_boxes, source_labels, source_scores, args.extended_range
        )
        prev_boxes_list: list[np.ndarray] = []
        prev_labels_list: list[np.ndarray] = []
        prev_time_diffs: list[float] = []
        missing_cache = 0
        for slot in range(1, max(int(args.history_frames), 1)):
            step = slot if args.history_stride == "consecutive" else slot * horizon
            prev_key = "infrastructure-side/velodyne/" + id_to_str(cur_id - horizon - step) + ".pcd"
            prev_info = inf_map.get(prev_key)
            if prev_info is None or prev_info.get("batch_id") != source_info.get("batch_id"):
                prev_boxes_list.append(np.zeros((0, 8, 3), dtype=np.float64))
                prev_labels_list.append(np.zeros((0,), dtype=np.float64))
                prev_time_diffs.append(0.0)
                continue
            prev_pred = raw_pred(cache_roots, "inf", frame_id(prev_info["pointcloud_path"]))
            if prev_pred is None:
                missing_cache += 1
                prev_boxes_list.append(np.zeros((0, 8, 3), dtype=np.float64))
                prev_labels_list.append(np.zeros((0,), dtype=np.float64))
                prev_time_diffs.append(0.0)
                continue
            prev_boxes, prev_labels, prev_scores = valid_detector_boxes(prev_pred)
            prev_boxes = infrastructure_to_target(prev_boxes, root, prev_info, cur_veh)
            prev_boxes, prev_labels, _ = range_filter_boxes(prev_boxes, prev_labels, prev_scores, args.extended_range)
            prev_boxes_list.append(prev_boxes)
            prev_labels_list.append(prev_labels)
            prev_time_diffs.append((int(source_info["pointcloud_timestamp"]) - int(prev_info["pointcloud_timestamp"])) / 1000.0)
        target_time_diff = (int(cur_inf["pointcloud_timestamp"]) - int(source_info["pointcloud_timestamp"])) / 1000.0
    else:
        cur_id = int(frame_id(cur_veh["pointcloud_path"]))
        source_key = "vehicle-side/velodyne/" + id_to_str(cur_id - horizon) + ".pcd"
        source_info = veh_map.get(source_key)
        if source_info is None or source_info.get("batch_id") != cur_veh.get("batch_id"):
            return np.zeros((0, feat_dim)), np.zeros((0, 2)), 0
        source_pred = raw_pred(cache_roots, "veh", frame_id(source_info["pointcloud_path"]))
        if source_pred is None:
            return np.zeros((0, feat_dim)), np.zeros((0, 2)), 1
        source_boxes, source_labels, source_scores = valid_detector_boxes(source_pred)
        source_boxes = vehicle_to_target(source_boxes, root, source_info, cur_veh)
        source_boxes, source_labels, source_scores = range_filter_boxes(
            source_boxes, source_labels, source_scores, args.extended_range
        )
        prev_boxes_list = []
        prev_labels_list = []
        prev_time_diffs = []
        missing_cache = 0
        for slot in range(1, max(int(args.history_frames), 1)):
            step = slot if args.history_stride == "consecutive" else slot * horizon
            prev_key = "vehicle-side/velodyne/" + id_to_str(cur_id - horizon - step) + ".pcd"
            prev_info = veh_map.get(prev_key)
            if prev_info is None or prev_info.get("batch_id") != source_info.get("batch_id"):
                prev_boxes_list.append(np.zeros((0, 8, 3), dtype=np.float64))
                prev_labels_list.append(np.zeros((0,), dtype=np.float64))
                prev_time_diffs.append(0.0)
                continue
            prev_pred = raw_pred(cache_roots, "veh", frame_id(prev_info["pointcloud_path"]))
            if prev_pred is None:
                missing_cache += 1
                prev_boxes_list.append(np.zeros((0, 8, 3), dtype=np.float64))
                prev_labels_list.append(np.zeros((0,), dtype=np.float64))
                prev_time_diffs.append(0.0)
                continue
            prev_boxes, prev_labels, prev_scores = valid_detector_boxes(prev_pred)
            prev_boxes = vehicle_to_target(prev_boxes, root, prev_info, cur_veh)
            prev_boxes, prev_labels, _ = range_filter_boxes(prev_boxes, prev_labels, prev_scores, args.extended_range)
            prev_boxes_list.append(prev_boxes)
            prev_labels_list.append(prev_labels)
            prev_time_diffs.append((int(source_info["pointcloud_timestamp"]) - int(prev_info["pointcloud_timestamp"])) / 1000.0)
        target_time_diff = (int(cur_veh["pointcloud_timestamp"]) - int(source_info["pointcloud_timestamp"])) / 1000.0

    if len(source_boxes) == 0 or len(cur_gt_boxes) == 0:
        return np.zeros((0, feat_dim)), np.zeros((0, 2)), missing_cache
    source_center, _ = box_info(source_boxes)
    gt_center, _ = box_info(cur_gt_boxes)
    dist = np.linalg.norm(source_center[:, None, :2] - gt_center[None, :, :2], axis=2)
    rows, cols = linear_sum_assignment(dist)
    keep = dist[rows, cols] <= args.max_match_dist
    if not np.any(keep):
        return np.zeros((0, feat_dim)), np.zeros((0, 2)), missing_cache
    rows = rows[keep]
    cols = cols[keep]
    feats = detector_history_features(
        source_boxes,
        source_labels,
        source_scores,
        prev_boxes_list,
        prev_labels_list,
        prev_time_diffs,
        horizon,
        target_time_diff,
        side,
        args.history_match_dist,
        args.history_frames,
    )
    return feats[rows], gt_center[cols, :2] - source_center[rows, :2], missing_cache


def parse_split_spec(split_spec: str) -> list[str]:
    splits = [item.strip() for item in re.split(r"[,+]", split_spec) if item.strip()]
    if not splits:
        raise ValueError(f"empty split spec: {split_spec!r}")
    return splits


def collect_split(args: argparse.Namespace, split: str, horizons: list[int], max_frames: int) -> tuple[np.ndarray, np.ndarray, dict[str, int], int]:
    cache_roots = [Path(item) for item in args.cache_roots.split(",") if item.strip()]
    inf_map, veh_map, coop_infos_all = load_infos(args.dair_root)
    split_names = parse_split_spec(split)
    coop_infos = []
    seen_keys: set[tuple[str, str]] = set()
    for split_name in split_names:
        for info in split_coop_infos(coop_infos_all, args.split_data_path, split_name):
            key = (info["infrastructure_pointcloud_path"], info["vehicle_pointcloud_path"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            coop_infos.append(info)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    counts: dict[str, int] = {}
    missing_cache = 0
    frames_seen = 0
    for horizon in horizons:
        start_len = sum(len(y) for y in ys)
        for info in coop_infos:
            if max_frames and frames_seen >= max_frames:
                break
            frames_seen += 1
            cur_inf = inf_map.get(info["infrastructure_pointcloud_path"])
            cur_veh = veh_map.get(info["vehicle_pointcloud_path"])
            if cur_inf is None or cur_veh is None:
                continue
            cur_gt = label_boxes(args.dair_root / info["cooperative_label_path"], prefer_world=True)
            cur_gt = transform_boxes(
                cur_gt,
                args.dair_root,
                "World",
                "Vehicle_lidar",
                frame_id(cur_inf["image_path"]),
                frame_id(cur_veh["image_path"]),
            )
            for side in ["inf", "veh"]:
                x_side, y_side, missing = side_examples(
                    args,
                    cache_roots,
                    args.dair_root,
                    cur_inf,
                    cur_veh,
                    cur_gt,
                    horizon,
                    side,
                    inf_map,
                    veh_map,
                )
                missing_cache += missing
                if len(y_side):
                    xs.append(x_side)
                    ys.append(y_side)
        counts[str(horizon)] = int(sum(len(y) for y in ys) - start_len)
    if not xs:
        raise RuntimeError(f"no detector-side examples for split={split}; missing_cache_events={missing_cache}")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), counts, missing_cache


def main() -> None:
    args = parse_args()
    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    x_train, y_train, train_counts, train_missing = collect_split(args, args.train_split, horizons, args.max_train_frames)
    x_val, y_val, val_counts, val_missing = collect_split(args, args.val_split, horizons, args.max_val_frames)
    mean, std = fit_normalizer(x_train)
    xt = (x_train - mean) / std
    xv = (x_val - mean) / std
    estimator = None
    if args.model_kind == "ridge":
        best = None
        evals = []
        for alpha in alphas:
            weights = ridge_fit(xt, y_train, alpha)
            pred = predict(xv, weights)
            rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
            mae = float(np.mean(np.abs(pred - y_val)))
            evals.append({"alpha": alpha, "val_rmse": rmse, "val_mae": mae})
            if best is None or rmse < best[0]:
                best = (rmse, mae, alpha, weights)
        assert best is not None
        rmse, mae, alpha, weights = best
        arrays = {"weights": weights}
        model_metadata = {"model_kind": "ridge", "selected_alpha": alpha, "val_rmse": rmse, "val_mae": mae, "evals": evals}
    elif args.model_kind == "rf":
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise RuntimeError("Random Forest training requires scikit-learn in the active environment") from exc

        estimator = RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth if args.rf_max_depth > 0 else None,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_samples=args.rf_max_samples if 0.0 < args.rf_max_samples < 1.0 else None,
            bootstrap=True,
            n_jobs=args.rf_n_jobs,
            random_state=args.random_state,
            verbose=1,
        )
        estimator.fit(xt, y_train)
        pred = estimator.predict(xv)
        rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
        mae = float(np.mean(np.abs(pred - y_val)))
        arrays = {}
        model_metadata = {
            "model_kind": "rf",
            "val_rmse": rmse,
            "val_mae": mae,
            "rf_n_estimators": int(args.rf_n_estimators),
            "rf_max_depth": int(args.rf_max_depth),
            "rf_min_samples_leaf": int(args.rf_min_samples_leaf),
            "rf_max_samples": float(args.rf_max_samples),
            "random_state": int(args.random_state),
        }
    else:
        arrays, model_metadata = train_mlp(xt, y_train, xv, y_val, args)
        rmse = float(model_metadata["val_rmse"])
        mae = float(model_metadata["val_mae"])
    metadata = {
        "model": "dair_ours_history_motion_compensator",
        "history_aware": True,
        "detector_side": True,
        "feature_dim": int(mean.shape[0]),
        "history_frames": int(args.history_frames),
        "history_stride": args.history_stride,
        "horizons": horizons,
        "train_split": args.train_split,
        "train_splits": parse_split_spec(args.train_split),
        "val_split": args.val_split,
        "val_splits": parse_split_spec(args.val_split),
        "train_examples": int(len(y_train)),
        "val_examples": int(len(y_val)),
        "train_counts_by_horizon": train_counts,
        "val_counts_by_horizon": val_counts,
        "train_missing_cache_events": int(train_missing),
        "val_missing_cache_events": int(val_missing),
        "max_match_dist": args.max_match_dist,
        "history_match_dist": args.history_match_dist,
        **model_metadata,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.model_kind == "rf":
        with args.out.open("wb") as handle:
            pickle.dump({"mean": mean, "std": std, "metadata": metadata, "estimator": estimator}, handle)
    else:
        np.savez(args.out, mean=mean, std=std, metadata=json.dumps(metadata, sort_keys=True), **arrays)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
