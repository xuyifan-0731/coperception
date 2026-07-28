#!/usr/bin/env python3
"""Train/evaluate a DATA-level T10-N10 feature compensator on DAIR-V2X.

The DATA detector fuses BEV backbone features at IFAM input.  This script keeps
the DATA checkpoint frozen and learns a lightweight temporal residual predictor
on the same 384-channel BEV feature tensor:

    pred(t+h) = feat(t) + sum_i w[h, i] * (feat(t-i) - feat(t-i-1))

The predictor consumes ten historical features and supports horizons 1..10.
Evaluation replaces delayed CAV features before IFAM and then runs the original
DATA fusion/head/post-processing path.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "external" / "DATA"
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "data_dair_official_eval"
DEFAULT_DAIR_ROOT = REPO_ROOT / "datasets" / "DAIR-V2X" / "cooperative-vehicle-infrastructure"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "260615_fill" / "cache" / "dair_data_feature_t10n10"
DEFAULT_PREDICTOR = REPO_ROOT / "checkpoints" / "curated" / "ours_models" / "dair_data_ifam_feature_t10n10_ours.npz"
DEFAULT_OUT = REPO_ROOT / "results" / "260615_fill" / "artifacts" / "dair_data_ifam_feature_t10n10_ours_20260707.csv"

if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.hypes_yaml import yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.utils import eval_utils  # noqa: E402
from opencood.utils.common_utils import update_dict  # noqa: E402
from opencood.utils.transformation_utils import normalize_pairwise_tfm  # noqa: E402


SIDES = ("veh", "inf")
SIDE_TO_INDEX = {side: i for i, side in enumerate(SIDES)}


@dataclass(frozen=True)
class GridCase:
    comp: int
    comm: int


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def empty_stat() -> dict:
    return {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }


def ap_from_stat(stat: dict, iou: float) -> float:
    if stat[iou]["gt"] == 0 or len(stat[iou]["score"]) == 0:
        return 0.0
    return float(eval_utils.calculate_ap(stat, iou)[0])


def read_json(path: Path):
    with path.open("r") as handle:
        return json.load(handle)


def frame_id(path: str) -> str:
    return Path(path).stem


def id_to_str(value: int | str, width: int = 6) -> str:
    return str(int(value)).zfill(width)


def load_infos(dair_root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    info_path = dair_root / "cooperative" / "data_info_processed_updated.json"
    if not info_path.exists():
        info_path = dair_root / "cooperative" / "data_info.json"
    infos = read_json(info_path)
    by_veh: dict[str, dict] = {}
    by_inf: dict[str, dict] = {}
    for item in infos:
        veh = frame_id(item["vehicle_pointcloud_path"])
        inf = frame_id(item["infrastructure_pointcloud_path"])
        by_veh[veh] = item
        by_inf[inf] = item
    return by_veh, by_inf


def get_veh_id_at_delay(info: dict, delay: int) -> str | None:
    if delay == 0:
        return frame_id(info["vehicle_pointcloud_path"])
    key = f"previous_veh_{delay}"
    if key in info and info[key] is not None:
        prev = info[key][0] if isinstance(info[key], list) else info[key]
        return frame_id(prev)
    cur = frame_id(info["vehicle_pointcloud_path"])
    try:
        return id_to_str(int(cur) - int(delay), len(cur))
    except ValueError:
        return None


def get_inf_id_at_delay(info: dict, delay: int) -> str | None:
    if delay == 0:
        return frame_id(info["infrastructure_pointcloud_path"])
    key = f"previous_inf_{delay}"
    prev = info.get(key)
    if prev is None:
        return None
    prev = prev[0] if isinstance(prev, list) else prev
    return frame_id(prev)


def get_side_id_at_delay(info: dict, side: str, delay: int) -> str | None:
    if side == "veh":
        return get_veh_id_at_delay(info, delay)
    if side == "inf":
        return get_inf_id_at_delay(info, delay)
    raise ValueError(f"unknown side: {side}")


def feature_path(cache_dir: Path, side: str, frame: str) -> Path:
    return cache_dir / side / f"{frame}.npy"


def has_feature(cache_dir: Path, side: str, frame: str | None) -> bool:
    return bool(frame) and feature_path(cache_dir, side, frame).exists()


def load_feature(cache_dir: Path, side: str, frame: str, device: torch.device | None = None) -> torch.Tensor:
    arr = np.load(feature_path(cache_dir, side, frame), mmap_mode="r")
    tensor = torch.from_numpy(np.asarray(arr, dtype=np.float16)).to(torch.float32)
    if device is not None:
        tensor = tensor.to(device, non_blocking=True)
    return tensor


def load_feature_np(cache_dir: Path, side: str, frame: str):
    return np.load(feature_path(cache_dir, side, frame), mmap_mode="r")


def build_hypes(model_dir: Path, dair_root: Path, split_json: Path, range_xy: str, cache_mode: bool = False):
    class Opt:
        pass

    opt = Opt()
    opt.model_dir = str(model_dir)
    hypes = yaml_utils.load_yaml(None, opt)
    hypes["data_dir"] = str(dair_root)
    hypes["root_dir"] = str(split_json)
    hypes["validate_dir"] = str(split_json)
    hypes["test_dir"] = str(split_json)
    if cache_mode:
        hypes["comm_range"] = 1e9
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


def load_data_model(model_dir: Path, hypes: dict, device: torch.device):
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(model_dir), model)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_ifam_feature(model, ego_dict: dict) -> torch.Tensor:
    m = model.modality_name
    feat = getattr(model, f"encoder_{m}")(ego_dict, m)
    feat = getattr(model, f"backbone_{m}")({"spatial_features": feat})["spatial_features_2d"]
    feat = getattr(model, f"aligner_{m}")(feat)
    if getattr(model, "compress", False):
        feat = model.compressor(feat)
    return feat


@torch.no_grad()
def forward_from_ifam_feature(model, feat: torch.Tensor, ego_dict: dict) -> dict:
    affine = normalize_pairwise_tfm(
        ego_dict["pairwise_t_matrix"], model.H, model.W, model.fake_voxel_size
    )
    fused, domain_set, occ_maps = model.ifam.forward_collab(feat, ego_dict["record_len"], affine)
    fused = model.se_block(fused)
    if model.shrink_flag:
        fused = model.shrink_conv(fused)
    return {
        "pyramid": "collab",
        "cls_preds": model.cls_head(fused),
        "reg_preds": model.reg_head(fused),
        "dir_preds": model.dir_head(fused),
        "occ_single_list": occ_maps,
        "domain": domain_set,
    }


def make_all_split(dair_root: Path, out_path: Path) -> Path:
    by_veh, _ = load_infos(dair_root)
    ids = sorted(by_veh.keys(), key=lambda x: int(x) if x.isdigit() else x)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(ids, handle)
    return out_path


def cache_features(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "veh").mkdir(exist_ok=True)
    (cache_dir / "inf").mkdir(exist_ok=True)
    split_json = make_all_split(args.dair_root, cache_dir / "all_frames.json")
    by_veh, _ = load_infos(args.dair_root)

    hypes = build_hypes(args.model_dir, args.dair_root, split_json, args.range, cache_mode=True)
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_data_model(args.model_dir, hypes, device)

    written = 0
    skipped = 0
    started = time.time()
    for i, batch in enumerate(loader):
        if args.max_items and i >= args.max_items:
            break
        if args.num_shards > 1 and (i % args.num_shards) != args.shard_index:
            continue
        if batch is None:
            continue
        sample_idx = int(batch["ego"]["sample_idx"])
        veh_id = dataset.split_info[sample_idx]
        info = by_veh[veh_id]
        inf_id = frame_id(info["infrastructure_pointcloud_path"])
        wanted = {
            0: feature_path(cache_dir, "veh", veh_id),
            1: feature_path(cache_dir, "inf", inf_id),
        }
        if all(path.exists() for path in wanted.values()) and not args.overwrite:
            skipped += 1
            if (i + 1) % args.log_interval == 0:
                print(f"cache progress {i+1}/{len(dataset)} skipped={skipped} written={written}", flush=True)
            continue
        batch = train_utils.to_device(batch, device)
        feat = extract_ifam_feature(model, batch["ego"]).detach().cpu().numpy().astype(np.float16)
        cav_ids = [int(x) for x in batch["ego"]["cav_id_list"]]
        for row, cav_id in enumerate(cav_ids):
            out_path = wanted.get(cav_id)
            if out_path is None:
                continue
            if out_path.exists() and not args.overwrite:
                continue
            np.save(out_path, feat[row])
            written += 1
        if (i + 1) % args.log_interval == 0:
            elapsed = time.time() - started
            print(
                f"cache progress {i+1}/{len(dataset)} skipped={skipped} written={written} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    meta = {
        "feature": "DATA IFAM input spatial_features_2d after aligner",
        "dtype": "float16",
        "shape": [384, 128, 256],
        "model_dir": str(args.model_dir),
        "dair_root": str(args.dair_root),
        "range": args.range,
        "written": written,
        "skipped": skipped,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (cache_dir / "metadata.json").open("w") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def valid_train_sequence(cache_dir: Path, info: dict, side: str, horizon: int, history: int) -> bool:
    target_id = get_side_id_at_delay(info, side, 0)
    if not has_feature(cache_dir, side, target_id):
        return False
    src_id = get_side_id_at_delay(info, side, horizon)
    return has_feature(cache_dir, side, src_id)


def sample_flat_indices(shape: tuple[int, ...], count: int, rng: np.random.Generator) -> np.ndarray:
    total = int(np.prod(shape))
    if count >= total:
        return np.arange(total)
    return rng.integers(0, total, size=count, endpoint=False)


def fit_horizon_weights(
    cache_dir: Path,
    infos: list[dict],
    side: str,
    horizon: int,
    history: int,
    samples_per_sequence: int,
    max_sequences: int,
    ridge: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    candidates = [
        info for info in infos if valid_train_sequence(cache_dir, info, side, horizon, history)
    ]
    rng.shuffle(candidates)
    if max_sequences > 0:
        candidates = candidates[:max_sequences]
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for info in candidates:
        target_id = get_side_id_at_delay(info, side, 0)
        target = load_feature_np(cache_dir, side, target_id)
        ids = [get_side_id_at_delay(info, side, delay) for delay in range(horizon, horizon + history)]
        feats = []
        last_valid = ids[0]
        for item in ids:
            if not has_feature(cache_dir, side, item):
                item = last_valid
            else:
                last_valid = item
            feats.append(load_feature_np(cache_dir, side, item))
        indices = sample_flat_indices(target.shape, samples_per_sequence, rng)
        y = np.asarray(target.reshape(-1)[indices], dtype=np.float32)
        x0 = np.asarray(feats[0].reshape(-1)[indices], dtype=np.float32)
        diffs = []
        for i in range(history - 1):
            a = np.asarray(feats[i].reshape(-1)[indices], dtype=np.float32)
            b = np.asarray(feats[i + 1].reshape(-1)[indices], dtype=np.float32)
            diffs.append(a - b)
        x_rows.append(np.stack(diffs, axis=1))
        y_rows.append(y - x0)
    if not x_rows:
        return np.zeros((history - 1,), dtype=np.float32), {"sequences": 0, "samples": 0, "mse": None}
    X = np.concatenate(x_rows, axis=0).astype(np.float64)
    y = np.concatenate(y_rows, axis=0).astype(np.float64)
    xtx = X.T @ X
    xty = X.T @ y
    xtx += np.eye(xtx.shape[0], dtype=np.float64) * ridge
    weights = np.linalg.solve(xtx, xty).astype(np.float32)
    pred_resid = X @ weights.astype(np.float64)
    mse = float(np.mean((pred_resid - y) ** 2))
    return weights, {"sequences": len(candidates), "samples": int(X.shape[0]), "mse": mse}


def train_predictor(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    train_ids = read_json(args.dair_root / "train.json")
    by_veh, _ = load_infos(args.dair_root)
    infos = [by_veh[item] for item in train_ids if item in by_veh]
    weights = np.zeros((len(SIDES), args.max_horizon + 1, args.history - 1), dtype=np.float32)
    stats: dict[str, dict] = {}
    for side in SIDES:
        for horizon in range(1, args.max_horizon + 1):
            w, stat = fit_horizon_weights(
                args.cache_dir,
                infos,
                side,
                horizon,
                args.history,
                args.samples_per_sequence,
                args.max_sequences_per_fit,
                args.ridge,
                rng,
            )
            weights[SIDE_TO_INDEX[side], horizon, :] = w
            stats[f"{side}_h{horizon}"] = stat
            print(f"fit side={side} horizon={horizon} stat={stat} weights={w.tolist()}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        weights=weights,
        sides=np.asarray(SIDES),
        history=np.asarray(args.history),
        max_horizon=np.asarray(args.max_horizon),
        stats=json.dumps(stats, ensure_ascii=False),
        metadata=json.dumps(
            {
                "name": "DAIR-V2X DATA IFAM-feature T10-N10 ours",
                "feature_level": "DATA IFAM input BEV feature, 384x128x256",
                "formula": "feat(t+h)=feat(t)+sum_i w[h,i]*(feat(t-i)-feat(t-i-1))",
                "history_frames": args.history,
                "future_horizons": args.max_horizon,
                "model_dir": str(args.model_dir),
                "cache_dir": str(args.cache_dir),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
        ),
    )
    print(f"saved predictor to {args.out}", flush=True)


def load_predictor(path: Path) -> tuple[np.ndarray, int, int]:
    payload = np.load(path, allow_pickle=True)
    weights = np.asarray(payload["weights"], dtype=np.float32)
    history = int(np.asarray(payload["history"]).item())
    max_horizon = int(np.asarray(payload["max_horizon"]).item())
    return weights, history, max_horizon


def sequence_ids(info: dict, side: str, horizon: int, history: int) -> list[str | None]:
    return [get_side_id_at_delay(info, side, delay) for delay in range(horizon, horizon + history)]


def predict_feature(
    cache_dir: Path,
    info: dict,
    side: str,
    horizon: int,
    weights: np.ndarray,
    history: int,
    max_horizon: int,
    device: torch.device,
    loaded: dict[tuple[str, str], torch.Tensor],
) -> torch.Tensor | None:
    if horizon <= 0:
        src_id = get_side_id_at_delay(info, side, 0)
        if not has_feature(cache_dir, side, src_id):
            return None
        key = (side, src_id)
        if key not in loaded:
            loaded[key] = load_feature(cache_dir, side, src_id, device)
        return loaded[key]
    src_id = get_side_id_at_delay(info, side, horizon)
    if not has_feature(cache_dir, side, src_id):
        return None
    ids = sequence_ids(info, side, horizon, history)
    first_valid = src_id
    feats: list[torch.Tensor] = []
    for item in ids:
        if not has_feature(cache_dir, side, item):
            item = first_valid
        key = (side, item)
        if key not in loaded:
            loaded[key] = load_feature(cache_dir, side, item, device)
        feats.append(loaded[key])
    h = min(int(horizon), int(max_horizon))
    w = torch.as_tensor(weights[SIDE_TO_INDEX[side], h], device=device, dtype=torch.float32)
    pred = feats[0].clone()
    for i in range(min(history - 1, len(feats) - 1)):
        pred = pred + w[i] * (feats[i] - feats[i + 1])
    return pred


def current_case_feature(
    cache_dir: Path,
    info: dict,
    cav_ids: list[int],
    comp: int,
    comm: int,
    weights: np.ndarray,
    history: int,
    max_horizon: int,
    device: torch.device,
    loaded: dict[tuple[str, str], torch.Tensor],
) -> torch.Tensor | None:
    rows: list[torch.Tensor] = []
    for cav_id in cav_ids:
        if cav_id == 0:
            feat = predict_feature(
                cache_dir, info, "veh", comp, weights, history, max_horizon, device, loaded
            )
        elif cav_id == 1:
            feat = predict_feature(
                cache_dir, info, "inf", comp + comm, weights, history, max_horizon, device, loaded
            )
        else:
            return None
        if feat is None:
            return None
        rows.append(feat)
    return torch.stack(rows, dim=0)


def eval_grid(args: argparse.Namespace) -> None:
    weights, history, max_horizon = load_predictor(args.predictor)
    val_split = args.dair_root / "val.json"
    hypes = build_hypes(args.model_dir, args.dair_root, val_split, args.range, cache_mode=False)
    dataset = build_dataset(hypes, visualize=False, train=False)
    eval_dataset = dataset
    if args.num_shards > 1:
        shard_indices = list(range(args.shard_index, len(dataset), args.num_shards))
        eval_dataset = Subset(dataset, shard_indices)
    loader = DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    by_veh, _ = load_infos(args.dair_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_data_model(args.model_dir, hypes, device)
    cases = [GridCase(comp, comm) for comp in parse_ints(args.comp_delays) for comm in parse_ints(args.comm_delays)]
    stats = {case: empty_stat() for case in cases}
    counts = {case: {"used": 0, "skipped": 0} for case in cases}
    started = time.time()

    for i, batch in enumerate(loader):
        if args.max_items and i >= args.max_items:
            break
        if batch is None:
            for case in cases:
                counts[case]["skipped"] += 1
            continue
        sample_idx = int(batch["ego"]["sample_idx"])
        veh_id = dataset.split_info[sample_idx]
        info = by_veh.get(veh_id)
        if info is None:
            for case in cases:
                counts[case]["skipped"] += 1
            continue
        cav_ids = [int(x) for x in batch["ego"]["cav_id_list"]]
        batch = train_utils.to_device(batch, device)
        loaded: dict[tuple[str, str], torch.Tensor] = {}
        with torch.no_grad():
            for case in cases:
                feat = current_case_feature(
                    args.cache_dir,
                    info,
                    cav_ids,
                    case.comp,
                    case.comm,
                    weights,
                    history,
                    max_horizon,
                    device,
                    loaded,
                )
                if feat is None:
                    counts[case]["skipped"] += 1
                    continue
                output = forward_from_ifam_feature(model, feat, batch["ego"])
                pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch, {"ego": output})
                for iou in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, stats[case], iou)
                counts[case]["used"] += 1
        if (i + 1) % args.log_interval == 0:
            elapsed = time.time() - started
            print(
                f"eval shard {args.shard_index}/{args.num_shards} progress "
                f"{i+1}/{len(eval_dataset)} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        with args.stats_out.open("wb") as handle:
            pickle.dump(
                {
                    "stats": stats,
                    "counts": counts,
                    "cases": cases,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                },
                handle,
            )
        print(f"saved raw stats to {args.stats_out}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "baseline",
                "method",
                "comp_delay",
                "comm_delay",
                "ap30",
                "ap50",
                "ap70",
                "used_frames",
                "skipped_frames",
                "notes",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "dataset": "DAIR-V2X-C",
                    "baseline": "DATA intermediate",
                    "method": "DATA + Ours IFAM-feature T10-N10",
                    "comp_delay": case.comp,
                    "comm_delay": case.comm,
                    "ap30": ap_from_stat(stats[case], 0.3),
                    "ap50": ap_from_stat(stats[case], 0.5),
                    "ap70": ap_from_stat(stats[case], 0.7),
                    "used_frames": counts[case]["used"],
                    "skipped_frames": counts[case]["skipped"],
                    "notes": f"predictor={args.predictor}; feature_cache={args.cache_dir}",
                }
            )
    print(f"saved grid results to {args.out}", flush=True)


def merge_stats_dict(dst: dict, src: dict) -> None:
    for iou in (0.3, 0.5, 0.7):
        dst[iou]["tp"].extend(src[iou]["tp"])
        dst[iou]["fp"].extend(src[iou]["fp"])
        dst[iou]["score"].extend(src[iou]["score"])
        dst[iou]["gt"] += src[iou]["gt"]


def merge_grid(args: argparse.Namespace) -> None:
    files: list[Path] = []
    for pattern in args.stats_files.split(","):
        files.extend(sorted(Path().glob(pattern.strip())))
    if not files:
        raise FileNotFoundError(f"no stats files matched: {args.stats_files}")
    merged_stats: dict[GridCase, dict] = {}
    merged_counts: dict[GridCase, dict] = {}
    for path in files:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        for case, stat in payload["stats"].items():
            merged_stats.setdefault(case, empty_stat())
            merge_stats_dict(merged_stats[case], stat)
        for case, count in payload["counts"].items():
            merged_counts.setdefault(case, {"used": 0, "skipped": 0})
            merged_counts[case]["used"] += int(count["used"])
            merged_counts[case]["skipped"] += int(count["skipped"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "baseline",
                "method",
                "comp_delay",
                "comm_delay",
                "ap30",
                "ap50",
                "ap70",
                "used_frames",
                "skipped_frames",
                "notes",
            ],
        )
        writer.writeheader()
        for case in sorted(merged_stats, key=lambda x: (x.comp, x.comm)):
            writer.writerow(
                {
                    "dataset": "DAIR-V2X-C",
                    "baseline": "DATA intermediate",
                    "method": "DATA + Ours IFAM-feature T10-N10",
                    "comp_delay": case.comp,
                    "comm_delay": case.comm,
                    "ap30": ap_from_stat(merged_stats[case], 0.3),
                    "ap50": ap_from_stat(merged_stats[case], 0.5),
                    "ap70": ap_from_stat(merged_stats[case], 0.7),
                    "used_frames": merged_counts[case]["used"],
                    "skipped_frames": merged_counts[case]["skipped"],
                    "notes": f"merged_stats={len(files)} shards",
                }
            )
    print(f"merged {len(files)} stats files into {args.out}", flush=True)


def fmt_pct(value: str | float) -> str:
    return f"{float(value) * 100.0:.2f}"


def table_rows_from_csv(csv_path: Path, ap_key: str) -> list[str]:
    rows = list(csv.DictReader(csv_path.open()))
    rows.sort(key=lambda r: (int(r["comp_delay"]), int(r["comm_delay"])))
    by_comp: dict[int, dict[int, str]] = {}
    for row in rows:
        by_comp.setdefault(int(row["comp_delay"]), {})[int(row["comm_delay"])] = fmt_pct(row[ap_key])
    output = []
    for comp in sorted(by_comp):
        vals = [by_comp[comp].get(comm, "-") for comm in range(0, 6)]
        output.append(
            "| 不同时延 + 我们方法特征级补偿 | DATA + Ours IFAM-feature T10-N10 | "
            f"{comp} | " + " | ".join(vals) + " |"
        )
    return output


def update_md(args: argparse.Namespace) -> None:
    md_path = args.md
    text = md_path.read_text()
    ap50_rows = "\n".join(table_rows_from_csv(args.out, "ap50"))
    ap70_rows = "\n".join(table_rows_from_csv(args.out, "ap70"))
    marker50 = "| 不同时延 + DATA fork 输入级补偿 | DATA + pose-warp | 4 | 13.73 | 13.71 | 13.78 | 13.90 | 13.86 | 13.62 | 13.50 |"
    marker70 = "| 不同时延 + DATA fork 输入级补偿 | DATA + pose-warp | 4 | 6.74 | 6.62 | 6.68 | 6.85 | 6.79 | 6.77 | 6.75 |"
    if marker50 in text and "DATA + Ours IFAM-feature T10-N10 | 0 |" not in text:
        text = text.replace(marker50, marker50 + "\n" + ap50_rows)
    if marker70 in text and "DATA + Ours IFAM-feature T10-N10 | 0 |" not in text:
        text = text.replace(marker70, marker70 + "\n" + ap70_rows)
    old = (
        "分析：当前本地 DATA strict detector AP@0.7 的 18 格均值为无补偿 `16.37`、"
        "DATA fork 输入级补偿 `15.41`；AP@0.5 的 18 格均值为无补偿 `28.07`、"
        "DATA fork 输入级补偿 `26.78`。"
    )
    rows = list(csv.DictReader(args.out.open()))
    mean50 = np.mean([float(r["ap50"]) * 100.0 for r in rows])
    mean70 = np.mean([float(r["ap70"]) * 100.0 for r in rows])
    new = (
        old
        + f"新增的 `DATA + Ours IFAM-feature T10-N10` 是按 DATA 原始 intermediate fusion 级别实现的特征级预测补偿，"
        + f"18 格均值为 AP@0.5 `{mean50:.2f}`、AP@0.7 `{mean70:.2f}`。"
    )
    if old in text and "IFAM-feature T10-N10` 是按 DATA 原始 intermediate fusion" not in text:
        text = text.replace(old, new)
    if str(args.out) not in text:
        src_marker = "| DAIR-V2X / DATA no-comp 与 DATA fork 输入级补偿 strict 结果 | `results/260615_fill/artifacts/dair_data_strict_dual_delay_none_ours_20260706.csv` |"
        addition = f"\n| DAIR-V2X / DATA + Ours IFAM-feature T10-N10 结果 | `{args.out}` |"
        if src_marker in text:
            text = text.replace(src_marker, src_marker + addition)
    md_path.write_text(text)
    print(f"updated {md_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
        p.add_argument("--dair-root", type=Path, default=DEFAULT_DAIR_ROOT)
        p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
        p.add_argument("--range", default="102.4,51.2")
        p.add_argument("--device", default="cuda")
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--log-interval", type=int, default=100)
        p.add_argument("--max-items", type=int, default=0)

    p_cache = sub.add_parser("cache-features")
    common(p_cache)
    p_cache.add_argument("--overwrite", action="store_true")
    p_cache.add_argument("--num-shards", type=int, default=1)
    p_cache.add_argument("--shard-index", type=int, default=0)
    p_cache.set_defaults(func=cache_features)

    p_train = sub.add_parser("train-predictor")
    common(p_train)
    p_train.add_argument("--out", type=Path, default=DEFAULT_PREDICTOR)
    p_train.add_argument("--history", type=int, default=10)
    p_train.add_argument("--max-horizon", type=int, default=10)
    p_train.add_argument("--samples-per-sequence", type=int, default=1024)
    p_train.add_argument("--max-sequences-per-fit", type=int, default=1200)
    p_train.add_argument("--ridge", type=float, default=1e-3)
    p_train.add_argument("--seed", type=int, default=20260707)
    p_train.set_defaults(func=train_predictor)

    p_eval = sub.add_parser("eval-grid")
    common(p_eval)
    p_eval.add_argument("--predictor", type=Path, default=DEFAULT_PREDICTOR)
    p_eval.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_eval.add_argument("--comp-delays", default="0,2,4")
    p_eval.add_argument("--comm-delays", default="0,1,2,3,4,5")
    p_eval.add_argument("--num-shards", type=int, default=1)
    p_eval.add_argument("--shard-index", type=int, default=0)
    p_eval.add_argument("--stats-out", type=Path, default=None)
    p_eval.set_defaults(func=eval_grid)

    p_merge = sub.add_parser("merge-grid")
    p_merge.add_argument("--stats-files", required=True)
    p_merge.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_merge.set_defaults(func=merge_grid)

    p_md = sub.add_parser("update-md")
    p_md.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_md.add_argument("--md", type=Path, default=REPO_ROOT / "260704结果.md")
    p_md.set_defaults(func=update_md)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
