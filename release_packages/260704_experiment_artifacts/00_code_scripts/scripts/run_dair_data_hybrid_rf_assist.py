#!/usr/bin/env python3
"""Use the detector-side RF predictor to assist DATA intermediate outputs.

This is a post-detection diagnostic.  It keeps DATA intermediate predictions as
the primary output and combines them with DATA single-side boxes corrected by
the DAIR detector-side RF 2f compensator.
"""

from __future__ import annotations

import argparse
import csv
import importlib
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
SCRIPTS_ROOT = REPO_ROOT / "scripts"
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "data_dair_official_eval"
DEFAULT_RF = REPO_ROOT / "checkpoints" / "curated" / "ours_models" / "dair_v2x_late_fusion_rf2_ours.pkl"
DEFAULT_RF_CACHE = REPO_ROOT / "results" / "260615_fill" / "cache" / "dair_data_rf_ours_20260706"
DEFAULT_DATA_CACHE = REPO_ROOT / "results" / "260615_fill" / "cache" / "dair_data_intermediate_boxes_20260707"
DEFAULT_OUT = REPO_ROOT / "results" / "260615_fill" / "artifacts" / "dair_data_intermediate_rf_assist_20260707.csv"

if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.hypes_yaml import yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.utils import common_utils, eval_utils  # noqa: E402
from opencood.utils import box_utils  # noqa: E402
from opencood.utils.common_utils import update_dict  # noqa: E402

from run_dair_data_rf_ours_box_comp_grid import (  # noqa: E402
    BBoxList,
    FRAME_SECONDS,
    TrainedMotionCompensator,
    cache_path,
    compensate_frame,
    empty_boxes,
    empty_scores,
    frame_from_prediction,
    fuse_boxes,
)


@dataclass
class SidePrediction:
    valid: bool
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


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
        {"cav_lidar_range": new_cav_range, "lidar_range": new_cav_range, "gt_range": new_cav_range},
    )
    parser_func = getattr(importlib.import_module("opencood.hypes_yaml.yaml_utils"), hypes["yaml_parser"])
    return parser_func(hypes)


def load_model(args, device):
    hypes = build_hypes(args.model_dir, comm_delay=0, comp_delay=0, range_xy=args.range)
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(args.model_dir), model)
    model.to(device)
    model.eval()
    return model


def data_cache_path(cache_dir: Path, comp: int, comm: int) -> Path:
    return cache_dir / f"data_intermediate_comp{comp}_comm{comm}.pkl"


def build_data_cache(args, model, device, comp: int, comm: int):
    path = data_cache_path(args.data_cache_dir, comp, comm)
    if path.exists() and not args.force_cache:
        with path.open("rb") as handle:
            return pickle.load(handle)

    hypes = build_hypes(args.model_dir, comm_delay=comm, comp_delay=comp, range_xy=args.range)
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
        batch = train_utils.to_device(batch, device)
        with torch.no_grad():
            output = model(batch["ego"])
            boxes, scores, _ = dataset.post_process(batch, {"ego": output})
        if boxes is None or scores is None:
            cache.append(SidePrediction(True, empty_boxes(), empty_scores(), np.zeros((0,), dtype=np.int64)))
        else:
            boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
            scores_np = scores.detach().cpu().numpy().astype(np.float32)
            cache.append(SidePrediction(True, boxes_np, scores_np, np.zeros((boxes_np.shape[0],), dtype=np.int64)))
        if (idx + 1) % args.progress_every == 0:
            print(f"cached DATA intermediate comp={comp} comm={comm}: {idx + 1}/{len(dataset)}", flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(cache, handle)
    print(f"saved DATA intermediate cache {path}", flush=True)
    return cache


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_rf_caches(args, comps: list[int], comms: list[int]):
    gt_cache = load_pickle(args.rf_cache_dir / "gt_current.pkl")
    max_delay = max(comp + comm for comp in comps for comm in comms) + 1
    veh_cache = {delay: load_pickle(cache_path(args.rf_cache_dir, "veh", delay)) for delay in range(max(comps) + 2)}
    inf_cache = {delay: load_pickle(cache_path(args.rf_cache_dir, "inf", delay)) for delay in range(max_delay + 1)}
    return gt_cache, veh_cache, inf_cache


def rf_assist_frame(compensator, veh_cache, inf_cache, idx: int, comp: int, comm: int):
    veh_delay = comp
    inf_delay = comp + comm
    veh_pred = veh_cache[veh_delay][idx]
    inf_pred = inf_cache[inf_delay][idx]
    if not veh_pred.valid or not inf_pred.valid:
        return BBoxList(empty_boxes(), np.zeros((0,), dtype=np.int64), empty_scores())
    veh_frame = frame_from_prediction(veh_pred)
    inf_frame = frame_from_prediction(inf_pred)
    veh_prev = veh_cache.get(veh_delay + 1, [None])[idx] if veh_delay > 0 else None
    inf_prev = inf_cache.get(inf_delay + 1, [None])[idx] if inf_delay > 0 else None
    veh_frame = compensate_frame(compensator, veh_frame, veh_prev, veh_delay, "veh")
    inf_frame = compensate_frame(compensator, inf_frame, inf_prev, inf_delay, "inf")
    boxes, scores = fuse_boxes(inf_frame, veh_frame)
    return BBoxList(boxes, np.zeros((boxes.shape[0],), dtype=np.int64), scores)


def empty_result():
    return {0.3: {"tp": [], "fp": [], "gt": 0, "score": []}, 0.5: {"tp": [], "fp": [], "gt": 0, "score": []}, 0.7: {"tp": [], "fp": [], "gt": 0, "score": []}}


def polygons_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    poly_a = list(common_utils.convert_format(boxes_a))
    poly_b = list(common_utils.convert_format(boxes_b))
    out = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    for i, poly in enumerate(poly_a):
        out[i] = common_utils.compute_iou(poly, poly_b)
    return out


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, thresh: float):
    if len(boxes) == 0:
        return boxes.astype(np.float32), scores.astype(np.float32)
    t_boxes = torch.from_numpy(boxes.astype(np.float32))
    t_scores = torch.from_numpy(scores.astype(np.float32))
    keep = box_utils.nms_rotated(t_boxes, t_scores, thresh)
    return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)


def combine_frames(data_frame: BBoxList, rf_frame: BBoxList, policy: str, rf_score_scale: float, iou_thresh: float, nms_thresh: float):
    data_boxes = data_frame.boxes.astype(np.float32)
    data_scores = data_frame.confidence.astype(np.float32)
    rf_boxes = rf_frame.boxes.astype(np.float32)
    rf_scores = (rf_frame.confidence * rf_score_scale).astype(np.float32)

    if policy == "data_only":
        return data_boxes, data_scores
    if len(rf_boxes) == 0:
        return data_boxes, data_scores

    if policy == "concat_nms":
        boxes = np.concatenate([data_boxes, rf_boxes], axis=0) if len(data_boxes) else rf_boxes
        scores = np.concatenate([data_scores, rf_scores], axis=0) if len(data_scores) else rf_scores
        return nms_boxes(boxes, scores, nms_thresh)

    if policy == "add_unmatched":
        if len(data_boxes):
            iou = polygons_iou(rf_boxes, data_boxes)
            keep_rf = np.max(iou, axis=1) < iou_thresh
            rf_boxes = rf_boxes[keep_rf]
            rf_scores = rf_scores[keep_rf]
        boxes = np.concatenate([data_boxes, rf_boxes], axis=0) if len(rf_boxes) else data_boxes
        scores = np.concatenate([data_scores, rf_scores], axis=0) if len(rf_scores) else data_scores
        return nms_boxes(boxes, scores, nms_thresh)

    if policy == "replace_matched":
        if len(data_boxes) == 0:
            return nms_boxes(rf_boxes, rf_scores, nms_thresh)
        if len(rf_boxes) == 0:
            return data_boxes, data_scores
        iou = polygons_iou(data_boxes, rf_boxes)
        rows, cols = linear_sum_assignment(-iou)
        used_data, used_rf = set(), set()
        out_boxes, out_scores = [], []
        for r, c in zip(rows, cols):
            if iou[r, c] >= iou_thresh:
                out_boxes.append(rf_boxes[c])
                out_scores.append(max(float(data_scores[r]), float(rf_scores[c])))
                used_data.add(int(r))
                used_rf.add(int(c))
        for i in range(len(data_boxes)):
            if i not in used_data:
                out_boxes.append(data_boxes[i])
                out_scores.append(data_scores[i])
        for i in range(len(rf_boxes)):
            if i not in used_rf:
                out_boxes.append(rf_boxes[i])
                out_scores.append(rf_scores[i])
        return nms_boxes(np.asarray(out_boxes, dtype=np.float32), np.asarray(out_scores, dtype=np.float32), nms_thresh)

    raise ValueError(f"unknown policy={policy}")


def evaluate_policy(args, data_cache, gt_cache, veh_cache, inf_cache, compensator, comp: int, comm: int, policy: str, scale: float):
    stat = empty_result()
    skipped = 0
    total = min(len(data_cache), len(gt_cache), len(veh_cache[comp]), len(inf_cache[comp + comm]))
    for idx in range(total):
        gt_np = gt_cache[idx]
        data_pred = data_cache[idx]
        if gt_np is None or not data_pred.valid:
            skipped += 1
            continue
        data_frame = frame_from_prediction(data_pred)
        rf_frame = rf_assist_frame(compensator, veh_cache, inf_cache, idx, comp, comm)
        boxes, scores = combine_frames(data_frame, rf_frame, policy, scale, args.iou_thresh, args.nms_thresh)
        det_boxes = torch.from_numpy(boxes)
        det_scores = torch.from_numpy(scores)
        gt_boxes = torch.from_numpy(gt_np)
        for thresh in (0.3, 0.5, 0.7):
            eval_utils.caluclate_tp_fp(det_boxes, det_scores, gt_boxes, stat, thresh)
    return {
        "ap30": eval_utils.calculate_ap(stat, 0.3)[0] * 100.0,
        "ap50": eval_utils.calculate_ap(stat, 0.5)[0] * 100.0,
        "ap70": eval_utils.calculate_ap(stat, 0.7)[0] * 100.0,
        "skipped": skipped,
        "total": total,
    }


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
    parser.add_argument("--rf-cache-dir", type=Path, default=DEFAULT_RF_CACHE)
    parser.add_argument("--data-cache-dir", type=Path, default=DEFAULT_DATA_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--range", default="102.4,51.2")
    parser.add_argument("--comps", default="0,2,4")
    parser.add_argument("--comms", default="0,1,2,3,4,5")
    parser.add_argument("--policies", default="concat_nms,add_unmatched,replace_matched")
    parser.add_argument("--rf-score-scales", default="0.3,0.5,0.7,1.0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--nms-thresh", type=float, default=0.05)
    parser.add_argument("--iou-thresh", type=float, default=0.05)
    args = parser.parse_args()

    for attr in ("model_dir", "rf_model", "rf_cache_dir", "data_cache_dir", "out"):
        value = getattr(args, attr)
        if not value.is_absolute():
            setattr(args, attr, (REPO_ROOT / value).resolve())

    comps = parse_ints(args.comps)
    comms = parse_ints(args.comms)
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    scales = [float(item) for item in args.rf_score_scales.split(",") if item.strip()]

    os.chdir(DATA_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None if args.eval_only else load_model(args, device)

    data_caches = {}
    for comp in comps:
        for comm in comms:
            path = data_cache_path(args.data_cache_dir, comp, comm)
            if args.eval_only:
                data_caches[(comp, comm)] = load_pickle(path)
            else:
                data_caches[(comp, comm)] = build_data_cache(args, model, device, comp, comm)
    if args.cache_only:
        return

    compensator = TrainedMotionCompensator(args.rf_model)
    gt_cache, veh_cache, inf_cache = load_rf_caches(args, comps, comms)

    rows = []
    for policy in policies:
        for scale in scales:
            for comp in comps:
                for comm in comms:
                    metrics = evaluate_policy(
                        args,
                        data_caches[(comp, comm)],
                        gt_cache,
                        veh_cache,
                        inf_cache,
                        compensator,
                        comp,
                        comm,
                        policy,
                        scale,
                    )
                    row = {
                        "dataset": "DAIR-V2X-C",
                        "method": f"DATA intermediate + RF assist ({policy},scale={scale:g})",
                        "policy": policy,
                        "rf_score_scale": scale,
                        "comp_delay": comp,
                        "comm_delay": comm,
                        "ap30": f"{metrics['ap30']:.4f}",
                        "ap50": f"{metrics['ap50']:.4f}",
                        "ap70": f"{metrics['ap70']:.4f}",
                        "skipped": metrics["skipped"],
                        "total": metrics["total"],
                    }
                    print(row, flush=True)
                    rows.append(row)
    write_rows(args.out, rows)


if __name__ == "__main__":
    main()
