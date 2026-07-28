#!/usr/bin/env python3
"""Evaluate the official TraF-Align checkpoint on DAIR-V2X-Seq/SPD."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
TRAF_ROOT = Path("/tmp/TraF-Align_partial")
DEFAULT_CKPT = TRAF_ROOT / "checkpoints" / "dair_v2x_seq_Trafalign_2024_12_28_04_42_36"
DEFAULT_DATA = ROOT / "datasets" / "DAIR-V2X-Seq" / "V2X-Seq-SPD"
DEFAULT_OUT = ROOT / "results" / "dair_v2x_seq" / "trafalign_delay_eval.csv"


def import_trafalign():
    if str(TRAF_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAF_ROOT))
    from datasets import build_dataset
    from hypes_yaml import yaml_utils
    from utils import box_utils, eval_utils, train_utils

    return build_dataset, yaml_utils, box_utils, eval_utils, train_utils


def init_result_stat(eval_cls, ious):
    result_stat = {}
    for i, cls in enumerate(eval_cls):
        result_stat[cls] = {}
        for iou in ious[i]:
            result_stat[cls][float(iou)] = {"tp": [], "fp": [], "gt": 0, "score": [], "frame": [], "iou": []}
    return result_stat


def mask_box_out_of_range(dets, params):
    mask_range = params["dataset"]["eval_range"]
    boxes = dets["box3d_lidar"]
    if boxes.shape[0] > 0:
        mask = (
            (boxes[:, 0] >= mask_range[0])
            & (boxes[:, 0] <= mask_range[3])
            & (boxes[:, 1] >= mask_range[1])
            & (boxes[:, 1] <= mask_range[4])
        )
        for key, value in dets.items():
            dets[key] = value[mask]
    return dets


def map_gt_cls(hypes, gt_box_i):
    cls_map = torch.as_tensor(hypes["dataset"]["cls_map"], device=gt_box_i.device)
    gt_box_i[:, 7] = cls_map[gt_box_i[:, 7].long()]
    return gt_box_i


def load_converted_checkpoint(model, ckpt_dir: Path, epoch: int):
    model_file = ckpt_dir / f"net_epoch{epoch}.pth"
    checkpoint = torch.load(model_file, map_location="cpu")
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


def frame_id(path: str | Path) -> str:
    return Path(path).stem.zfill(6)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def side_infos(data_root: Path, side: str) -> dict[str, dict]:
    rel = "vehicle-side/data_info.json" if side == "veh" else "infrastructure-side/data_info.json"
    out = {}
    for item in load_json(data_root / rel):
        fid = str(item.get("frame_id") or frame_id(item.get("pointcloud_path", ""))).zfill(6)
        out[fid] = {
            "frame_id": fid,
            "sequence_id": str(item.get("sequence_id") or ""),
        }
    return out


def build_prev_maps(data_root: Path) -> dict[str, dict[str, list[str]]]:
    prev_maps = {}
    for side in ("veh", "inf"):
        by_seq: dict[str, list[str]] = {}
        for fid, item in side_infos(data_root, side).items():
            by_seq.setdefault(item["sequence_id"], []).append(fid)
        side_prev = {}
        for frames in by_seq.values():
            frames = sorted(frames, key=lambda x: int(x) if x.isdigit() else x)
            for idx, fid in enumerate(frames):
                side_prev[fid] = list(reversed(frames[max(0, idx - 80):idx]))
        prev_maps[side] = side_prev
    return prev_maps


def build_coop_map(data_root: Path) -> dict[str, str]:
    mapping = {}
    for item in load_json(data_root / "cooperative" / "data_info.json"):
        veh = str(item["vehicle_frame"]).zfill(6)
        inf = str(item["infrastructure_frame"]).zfill(6)
        mapping[veh] = inf
    return mapping


class FeaturePredictor:
    def __init__(self, predictor: Path, cache_dir: Path, data_root: Path, device: torch.device, max_cache: int = 512):
        payload = np.load(predictor, allow_pickle=True)
        self.weights = payload["weights"].astype(np.float32)
        self.history = int(np.asarray(payload["history"]).item())
        self.max_horizon = int(np.asarray(payload["max_horizon"]).item())
        self.side_order = [str(x) for x in payload["side_order"].tolist()]
        self.side_index = {side: i for i, side in enumerate(self.side_order)}
        self.cache_dir = cache_dir
        self.device = device
        self.prev_maps = build_prev_maps(data_root)
        self.coop_map = build_coop_map(data_root)
        self.tensor_cache: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()
        self.max_cache = int(max_cache)
        self.miss_count = 0
        self.hit_count = 0
        self.predict_count = 0

    def feature_path(self, side: str, fid: str) -> Path:
        return self.cache_dir / side / f"{fid}.npy"

    def has_feature(self, side: str, fid: str | None) -> bool:
        return bool(fid) and self.feature_path(side, fid).exists()

    def load_feature(self, side: str, fid: str) -> torch.Tensor | None:
        key = (side, fid)
        if key in self.tensor_cache:
            self.tensor_cache.move_to_end(key)
            self.hit_count += 1
            return self.tensor_cache[key]
        path = self.feature_path(side, fid)
        if not path.exists():
            self.miss_count += 1
            return None
        arr = np.load(path, mmap_mode="r")
        tensor = torch.from_numpy(np.asarray(arr, dtype=np.float16)).to(self.device, non_blocking=True).float()
        self.tensor_cache[key] = tensor
        self.tensor_cache.move_to_end(key)
        while len(self.tensor_cache) > self.max_cache:
            self.tensor_cache.popitem(last=False)
        return tensor

    def target_slots(self, side: str, current_fid: str, count: int) -> list[str | None]:
        prev = self.prev_maps[side].get(current_fid, [])
        slots = [current_fid]
        for idx in range(count - 1):
            slots.append(prev[idx] if idx < len(prev) else current_fid)
        return slots

    def predict(self, side: str, target_fid: str | None, horizon: int) -> torch.Tensor | None:
        if not target_fid:
            self.miss_count += 1
            return None
        if horizon <= 0:
            return None
        if horizon > self.max_horizon:
            self.miss_count += 1
            return None
        prev = self.prev_maps[side].get(target_fid, [])
        ids = prev[horizon - 1 : horizon - 1 + self.history]
        if len(ids) < self.history:
            self.miss_count += 1
            return None
        feats = [self.load_feature(side, fid) for fid in ids]
        if any(feat is None for feat in feats):
            self.miss_count += 1
            return None
        side_idx = self.side_index[side]
        w = torch.as_tensor(self.weights[side_idx, horizon], device=self.device, dtype=torch.float32)
        pred = feats[0].clone()
        for idx in range(min(self.history - 1, len(feats) - 1)):
            pred = pred + w[idx] * (feats[idx] - feats[idx + 1])
        self.predict_count += 1
        return pred

    def replace_backbone_features(self, x: torch.Tensor, batch_data: dict, cfg: dict) -> torch.Tensor:
        ego_frames = int(cfg["dataset"]["frame_his"])
        cav_frames = int(cfg["dataset"]["cav_frame_his"])
        expected = ego_frames + cav_frames
        record_len = batch_data["record_len"].detach().cpu().numpy().astype(int).tolist()
        time_delays = batch_data["time_delays"]
        token_dicts = batch_data["token_dicts"]
        x_new = x.clone()
        offset = 0
        for batch_idx, rec in enumerate(record_len):
            if rec < expected:
                offset += rec
                continue
            veh_cur = str(token_dicts[batch_idx]["frame_id"]).zfill(6)
            inf_cur = self.coop_map.get(veh_cur)
            delays = time_delays[batch_idx]
            veh_delay = int(round(float(delays[0])))
            inf_delay = int(round(float(delays[1]))) if len(delays) > 1 else 0
            for slot, target in enumerate(self.target_slots("veh", veh_cur, ego_frames)):
                feat = self.predict("veh", target, veh_delay)
                if feat is not None and tuple(feat.shape) == tuple(x_new[offset + slot].shape):
                    x_new[offset + slot] = feat
            cav_base = offset + ego_frames
            for slot, target in enumerate(self.target_slots("inf", inf_cur, cav_frames)):
                feat = self.predict("inf", target, inf_delay)
                if feat is not None and tuple(feat.shape) == tuple(x_new[cav_base + slot].shape):
                    x_new[cav_base + slot] = feat
            offset += rec
        batch_data["time_delays"] = [[0.0 for _ in delays] for delays in time_delays]
        return x_new


def model_forward_with_predictor(model, batch_data: dict, feature_predictor: FeaturePredictor, cfg: dict):
    lidar = batch_data["processed_lidar"]
    x = model.reader(lidar)
    x = model.backbone(x, batch_data)
    x = feature_predictor.replace_backbone_features(x, batch_data, cfg)
    x, x_traj, offset_list = model.fusion_net(x, batch_data, lidar)

    if model.shrink_flag:
        x = model.shrink_conv(x)
    x = model.aspp(x)

    if model.headtype == "anchor":
        batch_dict = model.head(x)
        batch_dict = model.processor(batch_data, batch_dict)
        batch_dict.update({"x_traj": x_traj, "x_offset": offset_list})
        dets = model.processor.post_processing(batch_dict)
        return batch_dict, dets

    x = model.head(x)
    dets = model.map2det(x)
    return [x, x_traj, offset_list], dets


def prepare_config(args: argparse.Namespace, delay_ms: int) -> Path:
    src = args.config if args.config else args.ckpt_dir / "config.yaml"
    text = src.read_text(encoding="utf-8")
    try:
        cfg = yaml.safe_load(text)
    except yaml.constructor.ConstructorError:
        # The official TraF-Align checkpoint config may contain numpy YAML tags.
        # It is a local trusted config, so fall back only for this read path.
        cfg = yaml.unsafe_load(text)
    cfg["root_dir"] = str(args.data_root).rstrip("/") + "/"
    cfg["split_dir"] = str(args.split_dir)
    cfg["dataset"]["infer_range"] = True
    cfg["train_params"]["val_batch_size"] = args.batch_size
    cfg["wild_setting"]["agent_i_delay"] = delay_ms
    cfg["delay_ego"] = args.delay_ego_ms
    if args.frame_his is not None:
        cfg["dataset"]["frame_his"] = args.frame_his
    if args.cav_frame_his is not None:
        cfg["dataset"]["cav_frame_his"] = args.cav_frame_his

    tmp_dir = Path(tempfile.mkdtemp(prefix="trafalign_seq_eval_"))
    dst = tmp_dir / "config.yaml"
    dst.write_text(yaml.safe_dump(to_yaml_safe(cfg), sort_keys=False), encoding="utf-8")
    return dst


def to_yaml_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {to_yaml_safe(k): to_yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_yaml_safe(v) for v in value]
    return value


def evaluate_one(args: argparse.Namespace, delay_ms: int) -> dict[str, str]:
    build_dataset, yaml_utils, box_utils, eval_utils, train_utils = import_trafalign()
    cfg_path = prepare_config(args, delay_ms)
    try:
        cfg = yaml_utils.load_yaml(str(cfg_path), argparse.Namespace(model_dir=None))
        cfg["voxelization"]["grid_size"] = np.asarray(cfg["voxelization"]["grid_size"])
        cfg = yaml_utils.check_pillar_params(cfg)

        dataset = build_dataset(cfg, set=args.split)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_batch,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
        )

        device = torch.device(args.device)
        model = train_utils.create_model(cfg).float().to(device).eval()
        load_info = load_converted_checkpoint(model, args.ckpt_dir, args.epoch)
        feature_predictor = None
        if args.ours_predictor:
            feature_predictor = FeaturePredictor(
                args.ours_predictor,
                args.feature_cache,
                args.data_root,
                device,
                max_cache=args.feature_cache_tensors,
            )

        eval_cls = cfg["dataset"].get("eval_cls", ["vehicle"])
        ious = np.asarray(cfg["dataset"].get("eval_iou_threshold", [[0.5, 0.7]]), dtype=float)
        result_stat = init_result_stat(eval_cls, ious)
        processed = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                if args.max_batches > 0 and batch_idx >= args.max_batches:
                    break
                batch_data = train_utils.to_device(batch_data, device)
                if feature_predictor is None:
                    _, dets = model(batch_data)
                else:
                    _, dets = model_forward_with_predictor(model, batch_data, feature_predictor, cfg)
                gt_box = batch_data["label_dict"]
                for j, dets_j in enumerate(dets):
                    dets_j = mask_box_out_of_range(dets_j, cfg)
                    if gt_box.shape[0] == 0:
                        continue
                    gt_box_j = gt_box[gt_box[:, 0] == j, 1:]
                    if gt_box_j.shape[0] == 0:
                        continue
                    gt_box_j = map_gt_cls(cfg, gt_box_j)
                    det_box42 = box_utils.boxes_to_corners2d(dets_j["box3d_lidar"], order="hwl")[:, :, :2]
                    gt_box42 = box_utils.boxes_to_corners2d(gt_box_j[:, :7], order="hwl")[:, :, :2]
                    result_stat = eval_utils.caluclate_tp_fp(
                        det_box42,
                        dets_j["scores"],
                        dets_j["label_preds"],
                        gt_box42,
                        gt_box_j[:, 7],
                        result_stat,
                        ious,
                        eval_cls,
                    )
                processed += int(batch_data["record_len"].shape[0])
                if args.log_interval and processed % args.log_interval == 0:
                    print(f"delay={delay_ms} processed={processed}", flush=True)

        method_name = args.method_name
        if not method_name:
            method_name = "TraF-Align + Ours feature T10-N10" if feature_predictor else "TraF-Align"
        row = {
            "dataset": "DAIR-V2X-Seq/SPD",
            "method": method_name,
            "split": args.split,
            "delay_ms": str(delay_ms),
            "delay_ego_ms": str(args.delay_ego_ms),
            "samples": str(processed),
            "AP@0.5": "",
            "AP@0.7": "",
            "notes": f"checkpoint={args.ckpt_dir}; epoch={args.epoch}; load_info={load_info}"
            + (
                f"; predictor={args.ours_predictor}; feature_cache={args.feature_cache}; "
                f"predicted={feature_predictor.predict_count}; cache_hit={feature_predictor.hit_count}; cache_miss={feature_predictor.miss_count}"
                if feature_predictor
                else ""
            ),
        }
        for iou in ious[0]:
            ap, _, _ = eval_utils.calculate_ap(result_stat[eval_cls[0]], float(iou), args.global_sort)
            row[f"AP@{iou:.1f}"] = f"{ap * 100:.2f}"
        return row
    finally:
        shutil.rmtree(cfg_path.parent, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=TRAF_ROOT / "datasets" / "Basedataset" / "V2XSeq_dataset_split_official.yaml")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--delay-ms-list", default="0,100,200,300,400,500")
    parser.add_argument("--delay-ego-ms", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--frame-his", type=int, default=None)
    parser.add_argument("--cav-frame-his", type=int, default=None)
    parser.add_argument("--global-sort", action="store_true")
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--ours-predictor", type=Path, default=None)
    parser.add_argument("--method-name", default="")
    parser.add_argument("--feature-cache", type=Path, default=ROOT / "results" / "dair_v2x_seq" / "trafalign_feature_cache")
    parser.add_argument("--feature-cache-tensors", type=int, default=512)
    args = parser.parse_args()

    required = [
        args.data_root / "cooperative" / "data_info.json",
        args.data_root / "vehicle-side" / "data_info.json",
        args.data_root / "infrastructure-side" / "data_info.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("DAIR-V2X-Seq/SPD is not prepared. Missing: " + "; ".join(missing))

    rows = []
    for delay in [int(x) for x in args.delay_ms_list.split(",") if x.strip()]:
        print(f"evaluating TraF-Align delay={delay}ms split={args.split}", flush=True)
        rows.append(evaluate_one(args, delay))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"updated {args.out} with {len(rows)} row(s)", flush=True)

    print(f"wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
