#!/usr/bin/env python3
"""Joint train/evaluate a DATA IFAM-level gated delay compensator on DAIR-V2X."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "external" / "DATA"
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "data_dair_official_eval"
DEFAULT_DAIR_ROOT = REPO_ROOT / "datasets" / "DAIR-V2X" / "cooperative-vehicle-infrastructure"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "260615_fill" / "cache" / "dair_data_feature_t10n10"
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "curated" / "ours_models" / "dair_data_ifam_joint_delay_comp.pth"
DEFAULT_OUT = REPO_ROOT / "results" / "260615_fill" / "artifacts" / "dair_data_ifam_joint_delay_comp_20260713.csv"

if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.hypes_yaml import yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.utils import eval_utils  # noqa: E402
from opencood.utils.common_utils import update_dict  # noqa: E402
from opencood.utils.transformation_utils import normalize_pairwise_tfm  # noqa: E402

from run_dair_data_feature_t10n10_comp import (  # noqa: E402
    GridCase,
    ap_from_stat,
    empty_stat,
    feature_path,
    frame_id,
    get_inf_id_at_delay,
    get_veh_id_at_delay,
    load_infos,
    merge_stats_dict,
    parse_ints,
)


class GatedIFAMDelayCompensator(nn.Module):
    def __init__(self, channels: int = 384, hidden: int = 64, max_delay: float = 8.0, residual_scale: float = 0.5, gate_init: float = -2.0):
        super().__init__()
        self.max_delay = float(max_delay)
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, feat: torch.Tensor, delays: torch.Tensor) -> torch.Tensor:
        delays = delays.to(device=feat.device, dtype=feat.dtype).view(-1, 1, 1, 1)
        active = (delays > 0).to(dtype=feat.dtype)
        if active.sum().item() == 0:
            return feat
        delay_norm = torch.clamp(delays / max(self.max_delay, 1.0), 0.0, 1.0)
        delay_map = delay_norm.expand(-1, 1, feat.shape[-2], feat.shape[-1])
        residual = self.net(torch.cat([feat, delay_map], dim=1))
        gate = torch.sigmoid(self.gate_logit) * self.residual_scale
        return feat + residual * active * gate

    def gate_value(self) -> float:
        return float((torch.sigmoid(self.gate_logit) * self.residual_scale).detach().cpu())


def read_json(path: Path):
    with path.open("r") as handle:
        return json.load(handle)


def build_hypes(model_dir: Path, dair_root: Path, split_json: Path, range_xy: str):
    class Opt:
        pass

    opt = Opt()
    opt.model_dir = str(model_dir)
    hypes = yaml_utils.load_yaml(None, opt)
    hypes["data_dir"] = str(dair_root)
    hypes["root_dir"] = str(split_json)
    hypes["validate_dir"] = str(split_json)
    hypes["test_dir"] = str(split_json)
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
    parser_func = getattr(__import__("opencood.hypes_yaml.yaml_utils", fromlist=[hypes["yaml_parser"]]), hypes["yaml_parser"])
    return parser_func(hypes)


def load_data_model(model_dir: Path, hypes: dict, device: torch.device):
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(model_dir), model)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def forward_from_ifam_feature_grad(model, feat: torch.Tensor, ego_dict: dict) -> dict:
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


def load_feature(cache_dir: Path, side: str, frame: str, device: torch.device) -> torch.Tensor:
    arr = np.load(feature_path(cache_dir, side, frame), mmap_mode="r")
    return torch.from_numpy(np.asarray(arr, dtype=np.float16)).to(device=device, dtype=torch.float32)


def case_feature(cache_dir: Path, info: dict, comp: int, comm: int, device: torch.device):
    veh = get_veh_id_at_delay(info, comp)
    inf = get_inf_id_at_delay(info, comp + comm)
    if veh is None or inf is None:
        return None, None
    veh_path = feature_path(cache_dir, "veh", veh)
    inf_path = feature_path(cache_dir, "inf", inf)
    if not veh_path.exists() or not inf_path.exists():
        return None, None
    feat = torch.stack(
        [load_feature(cache_dir, "veh", veh, device), load_feature(cache_dir, "inf", inf, device)],
        dim=0,
    )
    delays = torch.tensor([comp, comp + comm], device=device, dtype=torch.float32)
    return feat, delays


def make_loader(args, split: str, train: bool):
    split_json = args.dair_root / f"{split}.json"
    hypes = build_hypes(args.model_dir, args.dair_root, split_json, args.range)
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=train,
        pin_memory=False,
        drop_last=False,
    )
    return hypes, dataset, loader


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hypes, dataset, loader = make_loader(args, args.train_split, train=True)
    model = load_data_model(args.model_dir, hypes, device)
    criterion = train_utils.create_loss(hypes)
    comp = GatedIFAMDelayCompensator(
        hidden=args.hidden_channels,
        max_delay=args.max_total_delay,
        residual_scale=args.residual_scale,
        gate_init=args.gate_init,
    ).to(device)
    comp.train()
    optimizer = torch.optim.AdamW(comp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    by_veh, _ = load_infos(args.dair_root)
    comp_choices = parse_ints(args.comp_delays)
    comm_choices = parse_ints(args.comm_delays)
    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.ckpt.with_suffix(".train_metrics.csv")
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "gate", "comp", "comm", "veh_id"])
        writer.writeheader()

    step = 0
    while step < args.max_steps:
        for batch in loader:
            if batch is None:
                continue
            sample_idx = int(batch["ego"]["sample_idx"])
            veh_id = dataset.split_info[sample_idx]
            info = by_veh.get(veh_id)
            if info is None:
                continue
            comp_delay = random.choice(comp_choices)
            comm_delay = random.choice(comm_choices)
            if comp_delay == 0 and comm_delay == 0:
                continue
            feat, delays = case_feature(args.cache_dir, info, comp_delay, comm_delay, device)
            if feat is None:
                continue
            batch = train_utils.to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = forward_from_ifam_feature_grad(model, comp(feat, delays), batch["ego"])
            loss = criterion(output, batch["ego"]["label_dict"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(comp.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            if step == 1 or step % args.log_interval == 0:
                gate = comp.gate_value()
                print(
                    f"step={step} loss={float(loss.detach().cpu()):.4f} "
                    f"gate={gate:.6f} comp={comp_delay} comm={comm_delay}",
                    flush=True,
                )
                with metrics_path.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["step", "loss", "gate", "comp", "comm", "veh_id"])
                    writer.writerow(
                        {
                            "step": step,
                            "loss": f"{float(loss.detach().cpu()):.6f}",
                            "gate": f"{gate:.8f}",
                            "comp": comp_delay,
                            "comm": comm_delay,
                            "veh_id": veh_id,
                        }
                    )
            if step >= args.max_steps:
                break
    torch.save(
        {
            "state_dict": comp.state_dict(),
            "hidden_channels": args.hidden_channels,
            "max_total_delay": args.max_total_delay,
            "residual_scale": args.residual_scale,
            "gate_init": args.gate_init,
            "steps": step,
            "metrics": str(metrics_path),
        },
        args.ckpt,
    )
    print(f"saved {args.ckpt}", flush=True)


def load_compensator(path: Path, device: torch.device) -> GatedIFAMDelayCompensator:
    payload = torch.load(path, map_location="cpu")
    comp = GatedIFAMDelayCompensator(
        hidden=int(payload.get("hidden_channels", 64)),
        max_delay=float(payload.get("max_total_delay", 8)),
        residual_scale=float(payload.get("residual_scale", 0.5)),
        gate_init=float(payload.get("gate_init", -2.0)),
    )
    comp.load_state_dict(payload["state_dict"])
    comp.to(device).eval()
    return comp


def eval_grid(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hypes, dataset, loader = make_loader(args, args.eval_split, train=False)
    model = load_data_model(args.model_dir, hypes, device)
    comp_module = load_compensator(args.ckpt, device)
    by_veh, _ = load_infos(args.dair_root)
    cases = [GridCase(comp, comm) for comp in parse_ints(args.comp_delays) for comm in parse_ints(args.comm_delays)]
    stats = {case: empty_stat() for case in cases}
    counts = {case: {"used": 0, "skipped": 0} for case in cases}
    started = time.time()
    with torch.no_grad():
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
            batch = train_utils.to_device(batch, device)
            for case in cases:
                feat, delays = case_feature(args.cache_dir, info, case.comp, case.comm, device)
                if feat is None:
                    counts[case]["skipped"] += 1
                    continue
                output = forward_from_ifam_feature_grad(model, comp_module(feat, delays), batch["ego"])
                pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch, {"ego": output})
                for iou in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, stats[case], iou)
                counts[case]["used"] += 1
            if (i + 1) % args.log_interval == 0:
                print(f"eval progress {i+1}/{len(loader)} elapsed={time.time()-started:.1f}s", flush=True)

    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        with args.stats_out.open("wb") as handle:
            pickle.dump({"stats": stats, "counts": counts, "cases": cases}, handle)
    write_csv(args.out, stats, counts, f"joint_ckpt={args.ckpt}; gate={comp_module.gate_value():.6f}")


def merge_grid(args: argparse.Namespace) -> None:
    files: list[Path] = []
    for pattern in args.stats_files.split(","):
        files.extend(sorted(Path().glob(pattern.strip())))
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
    write_csv(args.out, merged_stats, merged_counts, f"merged_stats={len(files)}")


def write_csv(path: Path, stats: dict[GridCase, dict], counts: dict[GridCase, dict], notes: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
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
        for case in sorted(stats, key=lambda x: (x.comp, x.comm)):
            writer.writerow(
                {
                    "dataset": "DAIR-V2X-C",
                    "baseline": "DATA intermediate",
                    "method": "DATA + Ours IFAM joint delay-comp",
                    "comp_delay": case.comp,
                    "comm_delay": case.comm,
                    "ap30": ap_from_stat(stats[case], 0.3),
                    "ap50": ap_from_stat(stats[case], 0.5),
                    "ap70": ap_from_stat(stats[case], 0.7),
                    "used_frames": counts[case]["used"],
                    "skipped_frames": counts[case]["skipped"],
                    "notes": notes,
                }
            )
    print(f"wrote {path}", flush=True)


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
        p.add_argument("--comp-delays", default="0,2,4")
        p.add_argument("--comm-delays", default="0,1,2,3,4,5")
        p.add_argument("--max-items", type=int, default=0)

    p_train = sub.add_parser("train")
    common(p_train)
    p_train.add_argument("--train-split", default="train")
    p_train.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p_train.add_argument("--max-steps", type=int, default=1500)
    p_train.add_argument("--lr", type=float, default=5e-4)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--grad-clip", type=float, default=5.0)
    p_train.add_argument("--hidden-channels", type=int, default=64)
    p_train.add_argument("--max-total-delay", type=float, default=9.0)
    p_train.add_argument("--residual-scale", type=float, default=0.5)
    p_train.add_argument("--gate-init", type=float, default=-2.0)
    p_train.add_argument("--seed", type=int, default=2026)
    p_train.set_defaults(func=train)

    p_eval = sub.add_parser("eval")
    common(p_eval)
    p_eval.add_argument("--eval-split", default="val")
    p_eval.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p_eval.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_eval.add_argument("--stats-out", type=Path, default=None)
    p_eval.set_defaults(func=eval_grid)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--stats-files", required=True)
    p_merge.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_merge.set_defaults(func=merge_grid)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
