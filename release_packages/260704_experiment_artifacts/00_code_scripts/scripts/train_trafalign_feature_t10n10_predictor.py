#!/usr/bin/env python3
"""Cache TraF-Align features and train a T10-N10 residual predictor.

This trains in TraF-Align's own feature space: the cached tensor is the dense
backbone output immediately after ``model.backbone`` and before
``model.fusion_net``.  The predictor is intentionally lightweight and mirrors
the DATA IFAM feature diagnostic: for each horizon h, learn scalar residual
weights that extrapolate from ten historical feature maps.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TRAF_ROOT = Path("/tmp/TraF-Align_partial")
DEFAULT_CKPT = TRAF_ROOT / "checkpoints" / "dair_v2x_seq_Trafalign_2024_12_28_04_42_36"
DEFAULT_DATA = ROOT / "datasets" / "DAIR-V2X-Seq" / "V2X-Seq-SPD"
DEFAULT_CACHE = ROOT / "results" / "dair_v2x_seq" / "trafalign_feature_cache"
DEFAULT_OUT = ROOT / "checkpoints" / "curated" / "ours_models" / "dair_v2x_seq_trafalign_feature_t10n10_ours.npz"


class SafeCollate:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, batch):
        try:
            return self.dataset.collate_batch(batch)
        except (KeyError, ValueError, RuntimeError) as exc:
            return {"__skip__": True, "__skip_reason__": repr(exc)}


def import_trafalign():
    if str(TRAF_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAF_ROOT))
    from datasets import build_dataset
    from hypes_yaml import yaml_utils
    from utils import train_utils

    return build_dataset, yaml_utils, train_utils


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def frame_id(path: str | Path) -> str:
    return Path(path).stem


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_data_root(data_root: Path) -> None:
    required = [
        data_root / "cooperative" / "data_info.json",
        data_root / "vehicle-side" / "data_info.json",
        data_root / "infrastructure-side" / "data_info.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("DAIR-V2X-Seq/SPD is not prepared. Missing: " + "; ".join(missing))


def load_converted_checkpoint(model, ckpt_dir: Path, epoch: int):
    checkpoint = torch.load(ckpt_dir / f"net_epoch{epoch}.pth", map_location="cpu")
    state = model.state_dict()
    converted = {}
    transformed = 0
    for key, value in checkpoint.items():
        if key in state and tuple(value.shape) != tuple(state[key].shape):
            if value.ndim == 4 and tuple(value.permute(3, 0, 1, 2).shape) == tuple(state[key].shape):
                value = value.permute(3, 0, 1, 2).contiguous()
                transformed += 1
        converted[key] = value
    missing, unexpected = model.load_state_dict(converted, strict=False)
    return {"transformed": transformed, "missing": len(missing), "unexpected": len(unexpected)}


def prepare_hypes(args: argparse.Namespace, frame_his: int, cav_frame_his: int):
    _, yaml_utils, _ = import_trafalign()
    src = args.config if args.config else args.ckpt_dir / "config.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["root_dir"] = str(args.data_root).rstrip("/") + "/"
    cfg["split_dir"] = str(args.split_dir)
    cfg["dataset"]["infer_range"] = True
    cfg["dataset"]["augment"] = False
    cfg["dataset"]["frame_his"] = frame_his
    cfg["dataset"]["cav_frame_his"] = cav_frame_his
    cfg["wild_setting"]["agent_i_delay"] = 0
    cfg["wild_setting"]["agent_i_delay_train_aug"] = [0, 0]
    cfg["train_params"]["val_batch_size"] = args.batch_size
    cfg["train_params"]["train_batch_size"] = args.batch_size
    cfg["voxelization"]["grid_size"] = np.asarray(cfg["voxelization"]["grid_size"])
    cfg = yaml_utils.check_pillar_params(cfg)
    return cfg


def side_frame_paths(batch: dict) -> tuple[str, str]:
    paths = [p for p in batch.get("lidar_path", []) if p]
    veh = next((p for p in paths if "/vehicle-side/" in p or "vehicle-side" in p), None)
    inf = next((p for p in paths if "/infrastructure-side/" in p or "infrastructure-side" in p), None)
    if veh is None or inf is None:
        raise RuntimeError(f"cannot infer vehicle/infrastructure paths from lidar_path={paths[:4]}")
    return veh, inf


@torch.no_grad()
def cache_features(args: argparse.Namespace) -> None:
    check_data_root(args.data_root)
    build_dataset, _, train_utils = import_trafalign()
    cfg = prepare_hypes(args, frame_his=1, cav_frame_his=1)
    dataset = build_dataset(cfg, set=args.split)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=SafeCollate(dataset),
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    device = torch.device(args.device)
    model = train_utils.create_model(cfg).float().to(device).eval()
    load_info = load_converted_checkpoint(model, args.ckpt_dir, args.epoch)

    for side in ("veh", "inf"):
        (args.cache_dir / side).mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    for batch_idx, batch in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        if batch is None or batch.get("__skip__"):
            skipped += 1
            continue
        veh_path, inf_path = side_frame_paths(batch)
        targets = {
            "veh": args.cache_dir / "veh" / f"{frame_id(veh_path)}.npy",
            "inf": args.cache_dir / "inf" / f"{frame_id(inf_path)}.npy",
        }
        if all(path.exists() for path in targets.values()) and not args.force:
            skipped += 1
            continue
        batch = train_utils.to_device(batch, device)
        x = model.reader(batch["processed_lidar"])
        x = model.backbone(x, batch).detach().cpu().numpy().astype(np.float16)
        record_len = int(batch["record_len"][0].item())
        if record_len < 2 or x.shape[0] < 2:
            skipped += 1
            continue
        np.save(targets["veh"], x[0])
        np.save(targets["inf"], x[1])
        saved += 2
        if args.log_interval and saved % args.log_interval == 0:
            print(f"cached_features={saved} skipped_batches={skipped}", flush=True)

    meta = {
        "updated_at": utc_now(),
        "data_root": str(args.data_root),
        "split": args.split,
        "feature": "TraF-Align backbone output before fusion_net",
        "checkpoint": str(args.ckpt_dir),
        "load_info": load_info,
        "saved_features": saved,
        "skipped_batches": skipped,
    }
    (args.cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"cached {saved} feature tensors to {args.cache_dir}", flush=True)


def side_infos(data_root: Path, side: str) -> dict[str, dict]:
    if side == "veh":
        items = load_json(data_root / "vehicle-side" / "data_info.json")
    else:
        items = load_json(data_root / "infrastructure-side" / "data_info.json")
    out = {}
    for item in items:
        fid = str(item.get("frame_id") or frame_id(item.get("pointcloud_path", ""))).zfill(6)
        seq = str(item.get("sequence_id") or item.get("vehicle_sequence") or item.get("infrastructure_sequence") or "")
        out[fid] = {"frame_id": fid, "sequence_id": seq}
    return out


def build_prev_maps(data_root: Path) -> dict[str, dict[str, list[str]]]:
    maps: dict[str, dict[str, list[str]]] = {}
    for side in ("veh", "inf"):
        by_seq: dict[str, list[str]] = defaultdict(list)
        for fid, item in side_infos(data_root, side).items():
            by_seq[item["sequence_id"]].append(fid)
        prev = {}
        for _, frames in by_seq.items():
            frames = sorted(frames, key=lambda x: int(x) if x.isdigit() else x)
            for idx, fid in enumerate(frames):
                prev[fid] = list(reversed(frames[max(0, idx - 40):idx]))
        maps[side] = prev
    return maps


def feature_path(cache_dir: Path, side: str, fid: str) -> Path:
    return cache_dir / side / f"{fid}.npy"


def has_feature(cache_dir: Path, side: str, fid: str | None) -> bool:
    return bool(fid) and feature_path(cache_dir, side, fid).exists()


def load_feature_np(cache_dir: Path, side: str, fid: str):
    return np.load(feature_path(cache_dir, side, fid), mmap_mode="r")


def solve_horizon(
    args: argparse.Namespace,
    side: str,
    horizon: int,
    targets: list[str],
    prev_map: dict[str, list[str]],
) -> tuple[np.ndarray, dict]:
    rng = random.Random(args.seed + horizon + (0 if side == "veh" else 100))
    targets = list(targets)
    rng.shuffle(targets)
    rows_x = []
    rows_y = []
    used = 0
    sampled_rows = 0
    for target in targets:
        prev = prev_map.get(target, [])
        needed = horizon + args.history
        if len(prev) < needed:
            continue
        ids = prev[horizon - 1 : horizon - 1 + args.history]
        if not all(has_feature(args.cache_dir, side, fid) for fid in [target] + ids):
            continue
        target_feat = np.asarray(load_feature_np(args.cache_dir, side, target), dtype=np.float32)
        hist = [np.asarray(load_feature_np(args.cache_dir, side, fid), dtype=np.float32) for fid in ids]
        diffs = [hist[i] - hist[i + 1] for i in range(args.history - 1)]
        y = target_feat - hist[0]
        flat_count = y.size
        sample = min(args.samples_per_sequence, flat_count)
        idx = rng.sample(range(flat_count), sample)
        x_sample = np.stack([d.reshape(-1)[idx] for d in diffs], axis=1)
        y_sample = y.reshape(-1)[idx]
        rows_x.append(x_sample)
        rows_y.append(y_sample)
        used += 1
        sampled_rows += sample
        if sampled_rows >= args.max_samples_per_horizon:
            break
    if not rows_x:
        return np.zeros((args.history - 1,), dtype=np.float32), {"sequences": 0, "samples": 0, "mse": None}
    x_mat = np.concatenate(rows_x, axis=0)
    y_vec = np.concatenate(rows_y, axis=0)
    if x_mat.shape[0] > args.max_samples_per_horizon:
        keep = np.random.default_rng(args.seed + horizon).choice(x_mat.shape[0], args.max_samples_per_horizon, replace=False)
        x_mat = x_mat[keep]
        y_vec = y_vec[keep]
    weights, *_ = np.linalg.lstsq(x_mat.astype(np.float64), y_vec.astype(np.float64), rcond=args.rcond)
    pred = x_mat @ weights
    mse = float(np.mean((pred - y_vec) ** 2))
    return weights.astype(np.float32), {"sequences": used, "samples": int(x_mat.shape[0]), "mse": mse}


def train_predictor(args: argparse.Namespace) -> None:
    check_data_root(args.data_root)
    prev_maps = build_prev_maps(args.data_root)
    split_data = yaml.safe_load(args.split_dir.read_text(encoding="utf-8"))
    allowed_sequences = {str(seq) for seq in split_data.get(args.split, [])}
    veh_infos = side_infos(args.data_root, "veh")
    inf_infos = side_infos(args.data_root, "inf")
    coop = load_json(args.data_root / "cooperative" / "data_info.json")
    side_targets = {"veh": [], "inf": []}
    for item in coop:
        veh = str(item.get("vehicle_frame") or item.get("vehicle_frame_id") or "").zfill(6)
        inf = str(item.get("infrastructure_frame") or item.get("infrastructure_frame_id") or "").zfill(6)
        if allowed_sequences and veh_infos.get(veh, {}).get("sequence_id") not in allowed_sequences:
            continue
        if allowed_sequences and inf_infos.get(inf, {}).get("sequence_id") not in allowed_sequences:
            continue
        if veh and has_feature(args.cache_dir, "veh", veh):
            side_targets["veh"].append(veh)
        if inf and has_feature(args.cache_dir, "inf", inf):
            side_targets["inf"].append(inf)

    weights = np.zeros((2, args.max_horizon + 1, args.history - 1), dtype=np.float32)
    stats = {}
    for side_idx, side in enumerate(("veh", "inf")):
        stats[side] = {}
        for horizon in range(1, args.max_horizon + 1):
            print(f"training side={side} horizon={horizon}", flush=True)
            w, stat = solve_horizon(args, side, horizon, side_targets[side], prev_maps[side])
            weights[side_idx, horizon, :] = w
            stats[side][str(horizon)] = stat

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        weights=weights,
        history=np.asarray(args.history),
        max_horizon=np.asarray(args.max_horizon),
        side_order=np.asarray(["veh", "inf"]),
        metadata=np.asarray(
            json.dumps(
                {
                    "name": "DAIR-V2X-Seq TraF-Align feature-space T10-N10 ours",
                    "feature_level": "TraF-Align backbone dense feature before fusion_net",
                    "trained_at": utc_now(),
                    "data_root": str(args.data_root),
                    "cache_dir": str(args.cache_dir),
                    "split": args.split,
                    "split_dir": str(args.split_dir),
                    "stats": stats,
                },
                ensure_ascii=False,
            )
        ),
    )
    print(f"wrote predictor to {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=TRAF_ROOT / "datasets" / "Basedataset" / "V2XSeq_dataset_split_official.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--max-horizon", type=int, default=10)
    parser.add_argument("--samples-per-sequence", type=int, default=4096)
    parser.add_argument("--max-samples-per-horizon", type=int, default=2_000_000)
    parser.add_argument("--rcond", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-interval", type=int, default=500)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache-features")
    sub.add_parser("train")
    args = parser.parse_args()

    if args.cmd == "cache-features":
        cache_features(args)
    elif args.cmd == "train":
        train_predictor(args)


if __name__ == "__main__":
    main()
