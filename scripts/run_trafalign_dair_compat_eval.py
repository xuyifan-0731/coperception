#!/usr/bin/env python3
"""Evaluate TraF-Align on the local DAIR-V2X-C compatibility view.

This is a diagnostic adapter.  TraF-Align's released DAIR model targets
DAIR-V2X-Seq; the compatibility view is produced by
scripts/prepare_trafalign_dair_compat.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TRAF_ROOT = Path("/tmp/TraF-Align_partial")
DEFAULT_CONFIG = ROOT / "results" / "trafalign_dair_v2x_c_compat" / "config_compat_debug.yaml"
DEFAULT_CKPT = TRAF_ROOT / "checkpoints" / "dair_v2x_seq_Trafalign_2024_12_28_04_42_36"
DEFAULT_OUT = ROOT / "results" / "260615_fill" / "artifacts" / "dair_trafalign_compat_debug_20260708.csv"


def import_trafalign():
    if str(TRAF_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAF_ROOT))
    from datasets import build_dataset
    from utils import box_utils, eval_utils, train_utils

    return build_dataset, box_utils, eval_utils, train_utils


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


def load_converted_checkpoint(model, ckpt_dir: Path):
    ckpt = torch.load(ckpt_dir / "net_epoch60.pth", map_location="cpu")
    state = model.state_dict()
    converted = {}
    transformed = 0
    for key, value in ckpt.items():
        if key in state and tuple(value.shape) != tuple(state[key].shape):
            if value.ndim == 4 and tuple(value.permute(3, 0, 1, 2).shape) == tuple(state[key].shape):
                value = value.permute(3, 0, 1, 2).contiguous()
                transformed += 1
        converted[key] = value
    missing, unexpected = model.load_state_dict(converted, strict=False)
    return {"transformed": transformed, "missing": len(missing), "unexpected": len(unexpected)}


def evaluate(args: argparse.Namespace) -> dict[str, str]:
    build_dataset, box_utils, eval_utils, train_utils = import_trafalign()
    cfg = yaml.safe_load(args.config.open("r", encoding="utf-8"))
    cfg["voxelization"]["grid_size"] = np.asarray(cfg["voxelization"]["grid_size"])
    cfg["train_params"]["val_batch_size"] = args.batch_size
    cfg["dataset"]["infer_range"] = True
    cfg["wild_setting"]["agent_i_delay"] = args.delay_ms
    cfg["delay_ego"] = args.delay_ego_ms

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
    load_info = load_converted_checkpoint(model, args.ckpt_dir)

    eval_cls = cfg["dataset"].get("eval_cls", ["vehicle"])
    ious = np.asarray(cfg["dataset"].get("eval_iou_threshold", [[0.5, 0.7]]), dtype=float)
    result_stat = init_result_stat(eval_cls, ious)

    processed = 0
    empty_gt = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batch_data = train_utils.to_device(batch_data, device)
            _, dets = model(batch_data)
            gt_box = batch_data["label_dict"]
            for j, dets_j in enumerate(dets):
                dets_j = mask_box_out_of_range(dets_j, cfg)
                if gt_box.shape[0] > 0:
                    gt_box_j = gt_box[gt_box[:, 0] == j, 1:]
                    if gt_box_j.shape[0] == 0:
                        empty_gt += 1
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
                else:
                    empty_gt += 1
            processed += int(batch_data["record_len"].shape[0])
            if processed % args.log_interval == 0:
                print(f"processed={processed}", flush=True)

    row = {
        "dataset": "DAIR-V2X-C compat-for-TraFAlign",
        "method": "TraF-Align pretrained",
        "split": args.split,
        "delay_ms": str(args.delay_ms),
        "delay_ego_ms": str(args.delay_ego_ms),
        "samples": str(processed),
        "empty_gt_batches": str(empty_gt),
        "AP@0.5": "",
        "AP@0.7": "",
        "notes": (
            "diagnostic adapter; source model is official DAIR-V2X-Seq checkpoint; "
            f"load_info={load_info}; max_batches={args.max_batches}"
        ),
    }
    for iou in ious[0]:
        ap, _, _ = eval_utils.calculate_ap(result_stat[eval_cls[0]], float(iou), args.global_sort)
        row[f"AP@{iou:.1f}"] = f"{ap * 100:.2f}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", default="debug")
    parser.add_argument("--delay-ms-list", default="0,400")
    parser.add_argument("--delay-ego-ms", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--global-sort", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    rows = []
    for delay in [int(x) for x in args.delay_ms_list.split(",") if x.strip()]:
        args.delay_ms = delay
        print(f"evaluating delay_ms={delay} split={args.split}", flush=True)
        rows.append(evaluate(args))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
