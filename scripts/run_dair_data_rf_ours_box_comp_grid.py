#!/usr/bin/env python3
"""Evaluate DATA single-side detections with optional DAIR RF compensation.

DATA is an intermediate-fusion detector and its checkpoint does not expose
single-side prediction heads.  This script reuses DATA's encoder/backbone/head
on each CAV independently, applies the trained detector-side RF motion
compensator to the resulting boxes, fuses boxes with the same late-fusion
utility logic, and evaluates against current-frame DAIR GT.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import multiprocessing as mp
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "external" / "DATA"
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "data_dair_official_eval"
DEFAULT_RF = REPO_ROOT / "checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours.pkl"
DEFAULT_OUT = REPO_ROOT / "results/260615_fill/artifacts/dair_data_strict_dual_delay_rf_ours_20260706.csv"
DEFAULT_CACHE = REPO_ROOT / "results/260615_fill/cache/dair_data_rf_ours_single_boxes"

if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.hypes_yaml import yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.utils import eval_utils  # noqa: E402
from opencood.utils.common_utils import update_dict  # noqa: E402
from opencood.utils.transformation_utils import get_relative_transformation  # noqa: E402


FRAME_SECONDS = 0.2
_EVAL_CONTEXT = {}


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def build_hypes(model_dir: Path, comm_delay: int, comp_delay: int, range_xy: str):
    class Opt:
        pass

    opt = Opt()
    opt.model_dir = str(model_dir)
    hypes = yaml_utils.load_yaml(None, opt)
    hypes["validate_dir"] = hypes["test_dir"]
    hypes["latency_eval"] = {
        "enabled": True,
        "comm_delay": int(comm_delay),
        "comp_delay": int(comp_delay),
        "compensation_method": "none",
    }

    x_range, y_range = [float(x) for x in range_xy.split(",")]
    new_cav_range = [
        -x_range,
        -y_range,
        hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
        x_range,
        y_range,
        hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
    ]
    hypes = update_dict(
        hypes,
        {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range,
        },
    )
    parser_func = getattr(importlib.import_module("opencood.hypes_yaml.yaml_utils"), hypes["yaml_parser"])
    return parser_func(hypes)


def empty_boxes() -> np.ndarray:
    return np.zeros((0, 8, 3), dtype=np.float32)


def empty_scores() -> np.ndarray:
    return np.zeros((0,), dtype=np.float32)


@dataclass
class SidePrediction:
    valid: bool
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


class BBoxList:
    def __init__(self, boxes, label, score):
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.size == 0:
            boxes = np.zeros((0, 8, 3), dtype=np.float64)
        self.boxes = boxes
        self.num_boxes = boxes.shape[0]
        self.num_dims = boxes.shape[2]
        self.label = np.asarray(label if label is not None else np.zeros((self.num_boxes,)), dtype=np.int64)
        self.confidence = np.asarray(score if score is not None else np.zeros((self.num_boxes,)), dtype=np.float64)
        self.center, self.size = self._box_info(self.boxes)

    @staticmethod
    def _box_info(boxes):
        if boxes.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
        center = np.mean(boxes, axis=1)
        size = np.zeros((boxes.shape[0], 3), dtype=np.float64)
        size[:, 0] = (
            np.linalg.norm(boxes[:, 2, :] - boxes[:, 1, :], axis=1)
            + np.linalg.norm(boxes[:, 6, :] - boxes[:, 5, :], axis=1)
        ) / 2
        size[:, 1] = (
            np.linalg.norm(boxes[:, 4, :] - boxes[:, 0, :], axis=1)
            + np.linalg.norm(boxes[:, 6, :] - boxes[:, 2, :], axis=1)
        ) / 2
        size[:, 2] = (
            boxes[:, 1, :]
            + boxes[:, 2, :]
            + boxes[:, 5, :]
            + boxes[:, 6, :]
            - boxes[:, 0, :]
            - boxes[:, 3, :]
            - boxes[:, 4, :]
            - boxes[:, 7, :]
        )[:, 2] / 4
        size = np.clip(size, 1e-3, None)
        return center, size

    def clone(self) -> "BBoxList":
        return BBoxList(self.boxes.copy(), self.label.copy(), self.confidence.copy())

    def move_center(self, offset):
        if self.num_boxes == 0:
            return
        delta_xy = np.asarray(offset, dtype=np.float64)
        delta = np.insert(delta_xy, 2, values=np.zeros(self.num_boxes), axis=1)
        self.boxes += delta[:, None, :]
        self.center[:, :2] += delta_xy


class TrainedMotionCompensator:
    def __init__(self, model_path: Path):
        with open(model_path, "rb") as handle:
            payload = pickle.load(handle)
        self.metadata = dict(payload.get("metadata", {}))
        self.model_kind = self.metadata.get("model_kind", "rf")
        self.mean = np.asarray(payload["mean"], dtype=np.float64)
        self.std = np.asarray(payload["std"], dtype=np.float64)
        self.estimator = payload["estimator"]
        if hasattr(self.estimator, "n_jobs"):
            self.estimator.n_jobs = 1
        if hasattr(self.estimator, "verbose"):
            self.estimator.verbose = 0
        self.history_frames = int(self.metadata.get("history_frames", 2))

    @staticmethod
    def _geometry(boxes):
        if len(boxes) == 0:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )
        boxes = boxes.astype(np.float64)
        center = np.mean(boxes, axis=1)
        size = np.zeros((boxes.shape[0], 3), dtype=np.float64)
        yaw = np.zeros((boxes.shape[0],), dtype=np.float64)
        size[:, 2] = np.max(boxes[:, :, 2], axis=1) - np.min(boxes[:, :, 2], axis=1)
        for i, box in enumerate(boxes):
            xy = box[:, :2] - center[i, :2]
            cov = xy.T @ xy
            vals, vecs = np.linalg.eigh(cov)
            main_axis = vecs[:, int(np.argmax(vals))]
            side_axis = np.array([-main_axis[1], main_axis[0]], dtype=np.float64)
            length = float((xy @ main_axis).max() - (xy @ main_axis).min())
            width = float((xy @ side_axis).max() - (xy @ side_axis).min())
            if width > length:
                length, width = width, length
            size[i, 0] = length
            size[i, 1] = width
            yaw[i] = np.arctan2(float(main_axis[1]), float(main_axis[0]))
        return center, np.clip(size, 1e-3, None), yaw

    @staticmethod
    def _valid(frame: BBoxList):
        if frame.num_boxes == 0:
            return np.zeros((0,), dtype=bool)
        mask = np.ones((frame.num_boxes,), dtype=bool)
        mask &= frame.confidence > 0.0
        mask &= frame.label >= 0
        mask &= np.linalg.norm(frame.boxes.reshape(frame.num_boxes, -1), axis=1) > 1e-6
        return mask

    @classmethod
    def _features(cls, frame: BBoxList, horizon_frames: int):
        n = frame.num_boxes
        if n == 0:
            return np.zeros((0, 18), dtype=np.float64)
        center, size, yaw = cls._geometry(frame.boxes)
        h = np.full((n, 1), float(horizon_frames), dtype=np.float64)
        base = np.concatenate(
            [
                center[:, :2],
                center[:, 2:3],
                size,
                np.sin(2.0 * yaw)[:, None],
                np.cos(2.0 * yaw)[:, None],
                h,
            ],
            axis=1,
        )
        return np.concatenate(
            [base, base[:, [0, 1]] * h, h * h, base[:, :2] ** 2, base[:, 3:5] ** 2, 1.0 / size[:, :2]],
            axis=1,
        )

    @classmethod
    def _history_features(
        cls,
        frame: BBoxList,
        prev_frames: list[BBoxList | None],
        prev_time_diffs: list[float],
        target_time_diff: float,
        horizon_frames: int,
        prefix: str,
        history_frames: int,
    ):
        base = cls._features(frame, horizon_frames)
        n = frame.num_boxes
        if n == 0:
            return np.zeros((0, 21 + 5 * max(int(history_frames) - 1, 0)), dtype=np.float64)

        slots = max(int(history_frames) - 1, 0)
        hist_delta = np.zeros((n, slots * 2), dtype=np.float64)
        temporal_delta = np.zeros((n, slots * 2), dtype=np.float64)
        has_prev = np.zeros((n, slots), dtype=np.float64)
        cur_valid = cls._valid(frame)
        cur_idx = np.where(cur_valid)[0]
        cur_centers = frame.center[cur_idx, :2] if len(cur_idx) else np.zeros((0, 2), dtype=np.float64)
        cur_labels = frame.label[cur_idx] if len(cur_idx) else np.zeros((0,), dtype=np.int64)
        for slot in range(slots):
            if slot >= len(prev_frames) or slot >= len(prev_time_diffs) or len(cur_idx) == 0:
                continue
            prev = prev_frames[slot]
            prev_dt = prev_time_diffs[slot]
            if prev is None or prev.num_boxes == 0 or prev_dt is None or float(prev_dt) <= 0:
                continue
            prev_idx = np.where(cls._valid(prev))[0]
            if len(prev_idx) == 0:
                continue
            dist = np.linalg.norm(prev.center[prev_idx, None, :2] - cur_centers[None, :, :], axis=2)
            dist[prev.label[prev_idx, None] != cur_labels[None, :]] = 1e6
            rows, cols = [], []
            used_prev, used_cur = set(), set()
            for flat in np.argsort(dist, axis=None):
                row, col = np.unravel_index(int(flat), dist.shape)
                if dist[row, col] > 8.0:
                    break
                if row in used_prev or col in used_cur:
                    continue
                used_prev.add(row)
                used_cur.add(col)
                rows.append(row)
                cols.append(col)
            if not rows:
                continue
            dst = cur_idx[np.asarray(cols, dtype=np.int64)]
            src = prev_idx[np.asarray(rows, dtype=np.int64)]
            delta = frame.center[dst, :2] - prev.center[src, :2]
            hist_delta[dst, slot * 2 : slot * 2 + 2] = delta
            temporal_delta[dst, slot * 2 : slot * 2 + 2] = delta * (float(target_time_diff) / max(float(prev_dt), 1e-6))
            has_prev[dst, slot] = 1.0

        score = frame.confidence[:, None] if frame.num_boxes else np.zeros((0, 1), dtype=np.float64)
        label = frame.label[:, None].astype(np.float64) if frame.num_boxes else np.zeros((0, 1), dtype=np.float64)
        side = np.full((n, 1), 1.0 if prefix == "veh" else 0.0, dtype=np.float64)
        return np.concatenate([base, hist_delta, temporal_delta, has_prev, score, label, side], axis=1)

    def compensate_history(self, frame: BBoxList, prev_frames, prev_time_diffs, target_time_diff, horizon_frames, prefix):
        if frame.num_boxes == 0 or int(horizon_frames) <= 0:
            return np.zeros((frame.num_boxes, 2), dtype=np.float64)
        valid = self._valid(frame)
        if not np.any(valid):
            return np.zeros((frame.num_boxes, 2), dtype=np.float64)
        x_all = self._history_features(
            frame,
            prev_frames or [],
            prev_time_diffs or [],
            target_time_diff,
            int(horizon_frames),
            prefix,
            self.history_frames,
        )
        x = (x_all[valid] - self.mean) / self.std
        pred = self.estimator.predict(x)
        offset = np.zeros((frame.num_boxes, 2), dtype=np.float64)
        offset[valid] = pred
        return offset


def match_frames(frame1: BBoxList, frame2: BBoxList):
    cost = np.zeros((frame1.num_boxes, frame2.num_boxes), dtype=np.float64)
    for i in range(frame1.num_boxes):
        for j in range(frame2.num_boxes):
            diff = np.abs(frame1.center[i] - frame2.center[j]) / np.clip(frame1.size[i], 1e-3, None)
            if diff[0] > 1 or diff[1] > 1 or diff[2] > 1 or frame1.label[i] != frame2.label[j]:
                cost[i, j] = 1e6
            else:
                cost[i, j] = np.linalg.norm(frame1.center[i] - frame2.center[j])
    ind1, ind2 = linear_sum_assignment(cost)
    accepted = [k for k in range(len(ind1)) if cost[ind1[k], ind2[k]] < 1e5]
    return ind1[accepted], ind2[accepted]


def fuse_boxes(inf_frame: BBoxList, veh_frame: BBoxList):
    ind_inf, ind_veh = match_frames(inf_frame, veh_frame)
    matched_boxes = []
    matched_labels = []
    matched_scores = []
    for i_inf, i_veh in zip(ind_inf, ind_veh):
        matched_boxes.append(veh_frame.boxes[i_veh])
        matched_labels.append(veh_frame.label[i_veh])
        matched_scores.append(veh_frame.confidence[i_veh])

    used_inf = set(ind_inf.tolist())
    used_veh = set(ind_veh.tolist())
    for i in range(veh_frame.num_boxes):
        if i not in used_veh and veh_frame.label[i] != -1:
            matched_boxes.append(veh_frame.boxes[i])
            matched_labels.append(veh_frame.label[i])
            matched_scores.append(veh_frame.confidence[i])
    for i in range(inf_frame.num_boxes):
        if i not in used_inf and inf_frame.label[i] != -1:
            matched_boxes.append(inf_frame.boxes[i])
            matched_labels.append(inf_frame.label[i])
            matched_scores.append(inf_frame.confidence[i] * 0.4)

    if not matched_boxes:
        return empty_boxes(), empty_scores()
    return np.stack(matched_boxes).astype(np.float32), np.asarray(matched_scores, dtype=np.float32)


def to_device(batch, device):
    return train_utils.to_device(batch, device)


def split_single_content(ego, side_id: int):
    cav_ids = list(ego["cav_id_list"])
    if side_id not in cav_ids:
        return None
    side = cav_ids.index(side_id)
    out = {}
    for key, value in ego.items():
        if key == "inputs_m1":
            continue
        out[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)

    coords_all = ego["inputs_m1"]["voxel_coords"]
    mask = coords_all[:, 0].long() == side
    if not torch.any(mask):
        return None
    out["inputs_m1"] = {
        "voxel_features": ego["inputs_m1"]["voxel_features"][mask].clone(),
        "voxel_num_points": ego["inputs_m1"]["voxel_num_points"][mask].clone(),
        "voxel_coords": coords_all[mask].clone(),
    }
    out["inputs_m1"]["voxel_coords"][:, 0] = 0
    out["record_len"] = torch.tensor([1], dtype=ego["record_len"].dtype, device=ego["record_len"].device)
    out["pairwise_t_matrix"] = torch.eye(
        4,
        dtype=ego["pairwise_t_matrix"].dtype,
        device=ego["pairwise_t_matrix"].device,
    ).view(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
    rel = torch.from_numpy(get_relative_transformation(ego["lidar_pose"].detach().cpu().numpy())).to(
        device=ego["lidar_pose"].device,
        dtype=torch.float32,
    )
    out["transformation_matrix"] = rel[side]
    out["transformation_matrix_clean"] = rel[side]
    out["lidar_pose"] = ego["lidar_pose"][side : side + 1].clone()
    out["lidar_pose_clean"] = ego["lidar_pose_clean"][side : side + 1].clone()
    out["cav_id_list"] = [side_id]
    out["agent_modality_list"] = [ego["agent_modality_list"][side]]
    return out


def cache_path(cache_dir: Path, side: str, delay: int) -> Path:
    return cache_dir / f"{side}_delay{delay}.pkl"


def build_side_cache(args, model, device, side: str, delay: int):
    side_id = 0 if side == "veh" else 1
    path = cache_path(args.cache_dir, side, delay)
    if path.exists() and not args.force_cache:
        with path.open("rb") as handle:
            return pickle.load(handle)
    if model is None:
        raise FileNotFoundError(f"Missing cache {path}; rerun without cache-only mode or use --force-cache.")

    if side == "veh":
        hypes = build_hypes(args.model_dir, comm_delay=0, comp_delay=delay, range_xy=args.range)
    else:
        hypes = build_hypes(args.model_dir, comm_delay=delay, comp_delay=0, range_xy=args.range)
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    cache: list[SidePrediction] = []
    for idx, batch in enumerate(loader):
        if args.limit is not None and idx >= args.limit:
            break
        if batch is None:
            cache.append(SidePrediction(False, empty_boxes(), empty_scores(), np.zeros((0,), dtype=np.int64)))
            continue
        batch = to_device(batch, device)
        single = split_single_content(batch["ego"], side_id)
        if single is None:
            cache.append(SidePrediction(True, empty_boxes(), empty_scores(), np.zeros((0,), dtype=np.int64)))
            continue
        with torch.no_grad():
            output = model(single)
            boxes, scores = dataset.post_processor.post_process({"ego": single}, {"ego": output})
        if boxes is None or scores is None:
            cache.append(SidePrediction(True, empty_boxes(), empty_scores(), np.zeros((0,), dtype=np.int64)))
        else:
            boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
            scores_np = scores.detach().cpu().numpy().astype(np.float32)
            labels_np = np.zeros((boxes_np.shape[0],), dtype=np.int64)
            cache.append(SidePrediction(True, boxes_np, scores_np, labels_np))
        if (idx + 1) % args.progress_every == 0:
            print(f"cached {side} delay={delay}: {idx + 1}/{len(dataset)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(cache, handle)
    return cache


def build_gt_cache(args, device):
    path = args.cache_dir / "gt_current.pkl"
    if path.exists() and not args.force_cache:
        with path.open("rb") as handle:
            return pickle.load(handle)
    hypes = build_hypes(args.model_dir, comm_delay=0, comp_delay=0, range_xy=args.range)
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    gt_cache = []
    for idx, batch in enumerate(loader):
        if args.limit is not None and idx >= args.limit:
            break
        if batch is None:
            gt_cache.append(None)
            continue
        batch = to_device(batch, device)
        gt = dataset.post_processor.generate_gt_bbx(batch)
        gt_cache.append(gt.detach().cpu().numpy().astype(np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(gt_cache, handle)
    return gt_cache


def caches_available(args, comps: list[int], comms: list[int]) -> bool:
    if args.force_cache:
        return False
    max_delay = max(comp + comm for comp in comps for comm in comms) + 1
    required = [args.cache_dir / "gt_current.pkl"]
    required.extend(cache_path(args.cache_dir, "veh", delay) for delay in range(max(comps) + 2))
    required.extend(cache_path(args.cache_dir, "inf", delay) for delay in range(max_delay + 1))
    return all(path.exists() for path in required)


def frame_from_prediction(pred: SidePrediction) -> BBoxList:
    return BBoxList(pred.boxes, pred.labels, pred.scores)


def compensate_frame(compensator, frame: BBoxList, prev_pred: SidePrediction | None, delay: int, prefix: str):
    out = frame.clone()
    prev_frames = []
    prev_dts = []
    if prev_pred is not None and prev_pred.valid:
        prev_frames.append(frame_from_prediction(prev_pred))
        prev_dts.append(FRAME_SECONDS)
    offset = compensator.compensate_history(
        out,
        prev_frames,
        prev_dts,
        float(delay) * FRAME_SECONDS,
        int(delay),
        prefix,
    )
    out.move_center(offset)
    return out


def evaluate_cell(compensator, gt_cache, veh_cache, inf_cache, comp: int, comm: int, mode: str):
    veh_delay = comp
    inf_delay = comp + comm
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    skipped = 0
    total = min(len(gt_cache), len(veh_cache[veh_delay]), len(inf_cache[inf_delay]))
    for idx in range(total):
        gt_np = gt_cache[idx]
        veh_pred = veh_cache[veh_delay][idx]
        inf_pred = inf_cache[inf_delay][idx]
        if gt_np is None or not veh_pred.valid or not inf_pred.valid:
            skipped += 1
            continue
        veh_frame = frame_from_prediction(veh_pred)
        inf_frame = frame_from_prediction(inf_pred)
        if mode == "ours":
            veh_prev = veh_cache.get(veh_delay + 1, [None] * total)[idx] if veh_delay > 0 else None
            inf_prev = inf_cache.get(inf_delay + 1, [None] * total)[idx] if inf_delay > 0 else None
            veh_frame = compensate_frame(compensator, veh_frame, veh_prev, veh_delay, "veh")
            inf_frame = compensate_frame(compensator, inf_frame, inf_prev, inf_delay, "inf")
        elif mode != "none":
            raise ValueError(f"Unknown mode={mode}")
        fused_boxes, fused_scores = fuse_boxes(inf_frame, veh_frame)
        det_boxes = torch.from_numpy(fused_boxes)
        det_scores = torch.from_numpy(fused_scores)
        gt_boxes = torch.from_numpy(gt_np)
        for thresh in (0.3, 0.5, 0.7):
            eval_utils.caluclate_tp_fp(det_boxes, det_scores, gt_boxes, result_stat, thresh)

    ap30 = eval_utils.calculate_ap(result_stat, 0.3)[0] * 100.0
    ap50 = eval_utils.calculate_ap(result_stat, 0.5)[0] * 100.0
    ap70 = eval_utils.calculate_ap(result_stat, 0.7)[0] * 100.0
    return ap30, ap50, ap70, skipped, total


def init_eval_worker(compensator, gt_cache, veh_cache, inf_cache, rf_model: str):
    _EVAL_CONTEXT.clear()
    _EVAL_CONTEXT.update(
        {
            "compensator": compensator,
            "gt_cache": gt_cache,
            "veh_cache": veh_cache,
            "inf_cache": inf_cache,
            "rf_model": rf_model,
        }
    )


def eval_task(task):
    mode, comp, comm, method_name = task
    ap30, ap50, ap70, skipped, total = evaluate_cell(
        _EVAL_CONTEXT["compensator"],
        _EVAL_CONTEXT["gt_cache"],
        _EVAL_CONTEXT["veh_cache"],
        _EVAL_CONTEXT["inf_cache"],
        comp,
        comm,
        mode,
    )
    return {
        "dataset": "DAIR-V2X-C",
        "method": method_name,
        "comp_delay": comp,
        "comm_delay": comm,
        "ap30": f"{ap30:.4f}",
        "ap50": f"{ap50:.4f}",
        "ap70": f"{ap70:.4f}",
        "skipped": skipped,
        "total": total,
        "rf_model": _EVAL_CONTEXT["rf_model"] if mode == "ours" else "",
    }


def load_model(args, device):
    hypes = build_hypes(args.model_dir, comm_delay=0, comp_delay=0, range_xy=args.range)
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(args.model_dir), model)
    model.to(device)
    model.eval()
    return model


def write_rows(path: Path, rows: list[dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--rf-model", type=Path, default=DEFAULT_RF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--range", default="102.4,51.2")
    parser.add_argument("--comps", default="0,2,4")
    parser.add_argument("--comms", default="0,1,2,3,4,5")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-jobs", type=int, default=1, help="Parallel AP evaluation workers after caches are ready.")
    parser.add_argument(
        "--mode",
        choices=["none", "ours", "both"],
        default="ours",
        help="Evaluate box-level DATA late fusion without compensation, with RF compensation, or both.",
    )
    args = parser.parse_args()

    for attr in ("model_dir", "rf_model", "out", "cache_dir"):
        value = getattr(args, attr)
        if not value.is_absolute():
            setattr(args, attr, (REPO_ROOT / value).resolve())

    comps = [0] if args.smoke else parse_ints(args.comps)
    comms = [0] if args.smoke else parse_ints(args.comms)
    if args.smoke and args.limit is None:
        args.limit = 64

    os.chdir(DATA_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None if caches_available(args, comps, comms) else load_model(args, device)
    compensator = TrainedMotionCompensator(args.rf_model)

    gt_cache = build_gt_cache(args, device)
    max_delay = max(comp + comm for comp in comps for comm in comms) + 1
    veh_cache = {}
    inf_cache = {}
    for delay in range(max(comps) + 2):
        veh_cache[delay] = build_side_cache(args, model, device, "veh", delay)
    for delay in range(max_delay + 1):
        inf_cache[delay] = build_side_cache(args, model, device, "inf", delay)

    modes = ["none", "ours"] if args.mode == "both" else [args.mode]
    method_names = {
        "none": "DATA box late-fusion w/o compensation",
        "ours": "DATA box late-fusion + Ours RF 2f",
    }
    tasks = []
    for mode in modes:
        for comp in comps:
            for comm in comms:
                tasks.append((mode, comp, comm, method_names[mode]))

    if args.eval_jobs > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=args.eval_jobs,
            initializer=init_eval_worker,
            initargs=(compensator, gt_cache, veh_cache, inf_cache, str(args.rf_model)),
        ) as pool:
            rows = []
            for row in pool.imap(eval_task, tasks):
                print(row, flush=True)
                rows.append(row)
    else:
        init_eval_worker(compensator, gt_cache, veh_cache, inf_cache, str(args.rf_model))
        rows = []
        for task in tasks:
            row = eval_task(task)
            print(row, flush=True)
            rows.append(row)
    write_rows(args.out, rows)


if __name__ == "__main__":
    main()
