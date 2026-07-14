#!/usr/bin/env python3
"""Joint train/evaluate DATA + feature residual compensation on V2X-Sim.

This keeps the trained DATA detector frozen and trains only a small gated
residual module at the IFAM input feature level.  The delayed input is generated
through DATA's existing V2X-Sim latency dataset path.  By default the delayed
point cloud is first pose-warped to the target frame, matching the existing
`DATA + Ours` V2X-Sim baseline, and the learned residual then corrects the
feature before IFAM fusion.
"""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_RUN_STATUS = REPO_ROOT / "runs" / "data_v2xsim_pipeline" / "status.json"
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "curated" / "ours_models" / "v2xsim_data_ifam_joint_delay_comp.pth"
DEFAULT_OUT = REPO_ROOT / "results" / "260615_fill" / "artifacts" / "v2xsim_data_ifam_joint_delay_comp_20260713.csv"

if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.hypes_yaml import yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.utils import eval_utils  # noqa: E402
from opencood.utils.common_utils import update_dict  # noqa: E402
from opencood.utils.transformation_utils import normalize_pairwise_tfm  # noqa: E402


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
    return float(eval_utils.calculate_ap(stat, iou)[0]) * 100.0


class GatedIFAMDelayCompensator(nn.Module):
    def __init__(
        self,
        channels: int = 384,
        hidden: int = 64,
        max_delay: float = 9.0,
        residual_scale: float = 0.5,
        gate_init: float = -2.0,
    ):
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


def model_dir_from_status(path: Path) -> Path:
    with path.open("r") as handle:
        data = json.load(handle)
    return Path(data["artifacts"]["model_dir"])


def build_hypes(model_dir: Path, info_pkl: Path, range_xy: str, latency_method: str | None = None) -> dict:
    class Opt:
        pass

    opt = Opt()
    opt.model_dir = str(model_dir)
    hypes = yaml_utils.load_yaml(None, opt)
    hypes["root_dir"] = str(info_pkl)
    hypes["validate_dir"] = str(info_pkl)
    hypes["test_dir"] = str(info_pkl)
    if latency_method is not None:
        hypes["latency_eval"] = {
            "enabled": True,
            "comm_delay": 0,
            "comp_delay": 0,
            "compensation_method": latency_method,
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
    parser_func = getattr(__import__("opencood.hypes_yaml.yaml_utils", fromlist=[hypes["yaml_parser"]]), hypes["yaml_parser"])
    return parser_func(hypes)


def build_latency_dataset(hypes: dict):
    dataset = build_dataset(hypes, visualize=False, train=False)
    dataset.latency_eval = True
    return dataset


def set_latency(dataset, comp: int, comm: int, method: str) -> None:
    dataset.latency_eval = True
    dataset.comp_delay = int(comp)
    dataset.comm_delay = int(comm)
    dataset.compensation_method = method


def collate_one(dataset, index: int):
    item = dataset[index]
    if item is None:
        return None
    return dataset.collate_batch_test([item])


def load_data_model(model_dir: Path, hypes: dict, device: torch.device):
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(model_dir), model)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def extract_ifam_feature(model, data_dict: dict) -> torch.Tensor:
    m = model.modality_name
    feat = getattr(model, f"encoder_{m}")(data_dict, m)
    feat = getattr(model, f"backbone_{m}")({"spatial_features": feat})["spatial_features_2d"]
    feat = getattr(model, f"aligner_{m}")(feat)
    if model.compress:
        feat = model.compressor(feat)
    return feat


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


def delay_tensor(batch: dict, comp: int, comm: int, device: torch.device) -> torch.Tensor:
    record_len = batch["ego"]["record_len"].detach().cpu().numpy().astype(int).tolist()
    values: list[float] = []
    for cav_num in record_len:
        if cav_num <= 0:
            continue
        values.append(float(comp))
        values.extend([float(comp + comm)] * (cav_num - 1))
    return torch.tensor(values, device=device, dtype=torch.float32)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dir = args.model_dir or model_dir_from_status(args.status)
    train_info = args.info_dir / "v2xsim_infos_train.pkl"
    hypes = build_hypes(model_dir, train_info, args.range, latency_method=args.input_compensation)
    dataset = build_latency_dataset(hypes)
    model = load_data_model(model_dir, hypes, device)
    criterion = train_utils.create_loss(hypes)
    comp_module = GatedIFAMDelayCompensator(
        hidden=args.hidden_channels,
        max_delay=args.max_total_delay,
        residual_scale=args.residual_scale,
        gate_init=args.gate_init,
    ).to(device)
    comp_module.train()
    optimizer = torch.optim.AdamW(comp_module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    comp_choices = parse_ints(args.comp_delays)
    comm_choices = parse_ints(args.comm_delays)
    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.ckpt.with_suffix(".train_metrics.csv")
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "gate", "comp", "comm", "sample_idx", "cav_num"])
        writer.writeheader()

    indices = list(range(len(dataset)))
    step = 0
    while step < args.max_steps:
        random.shuffle(indices)
        for idx in indices:
            comp_delay = random.choice(comp_choices)
            comm_delay = random.choice(comm_choices)
            if comp_delay == 0 and comm_delay == 0:
                continue
            set_latency(dataset, comp_delay, comm_delay, args.input_compensation)
            batch = collate_one(dataset, idx)
            if batch is None:
                continue
            batch = train_utils.to_device(batch, device)
            delays = delay_tensor(batch, comp_delay, comm_delay, device)
            if delays.numel() == 0 or float(delays.max().detach().cpu()) <= 0.0:
                continue
            optimizer.zero_grad(set_to_none=True)
            feat = extract_ifam_feature(model, batch["ego"])
            output = forward_from_ifam_feature(model, comp_module(feat, delays), batch["ego"])
            loss = criterion(output, batch["ego"]["label_dict"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(comp_module.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            if step == 1 or step % args.log_interval == 0:
                gate = comp_module.gate_value()
                cav_num = int(batch["ego"]["record_len"].sum().detach().cpu())
                print(
                    f"step={step} loss={float(loss.detach().cpu()):.4f} "
                    f"gate={gate:.6f} comp={comp_delay} comm={comm_delay} cav={cav_num}",
                    flush=True,
                )
                with metrics_path.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["step", "loss", "gate", "comp", "comm", "sample_idx", "cav_num"])
                    writer.writerow(
                        {
                            "step": step,
                            "loss": f"{float(loss.detach().cpu()):.6f}",
                            "gate": f"{gate:.8f}",
                            "comp": comp_delay,
                            "comm": comm_delay,
                            "sample_idx": idx,
                            "cav_num": cav_num,
                        }
                    )
            if step >= args.max_steps:
                break

    torch.save(
        {
            "state_dict": comp_module.state_dict(),
            "hidden_channels": args.hidden_channels,
            "max_total_delay": args.max_total_delay,
            "residual_scale": args.residual_scale,
            "gate_init": args.gate_init,
            "input_compensation": args.input_compensation,
            "steps": step,
            "metrics": str(metrics_path),
        },
        args.ckpt,
    )
    print(f"saved {args.ckpt}", flush=True)


def load_compensator(path: Path, device: torch.device) -> tuple[GatedIFAMDelayCompensator, dict]:
    payload = torch.load(path, map_location="cpu")
    comp_module = GatedIFAMDelayCompensator(
        hidden=int(payload.get("hidden_channels", 64)),
        max_delay=float(payload.get("max_total_delay", 9.0)),
        residual_scale=float(payload.get("residual_scale", 0.5)),
        gate_init=float(payload.get("gate_init", -2.0)),
    )
    comp_module.load_state_dict(payload["state_dict"])
    comp_module.to(device).eval()
    return comp_module, payload


def eval_grid(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dir = args.model_dir or model_dir_from_status(args.status)
    test_info = args.info_dir / "v2xsim_infos_test.pkl"
    comp_module, payload = load_compensator(args.ckpt, device)
    input_comp = args.input_compensation or payload.get("input_compensation", "ours")
    hypes = build_hypes(model_dir, test_info, args.range, latency_method=input_comp)
    dataset = build_latency_dataset(hypes)
    model = load_data_model(model_dir, hypes, device)
    cases = [GridCase(comp, comm) for comp in parse_ints(args.comp_delays) for comm in parse_ints(args.comm_delays)]
    stats = {case: empty_stat() for case in cases}
    counts = {case: {"used": 0, "skipped": 0} for case in cases}
    started = time.time()
    with torch.no_grad():
        for case in cases:
            set_latency(dataset, case.comp, case.comm, input_comp)
            loader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=args.num_workers,
                collate_fn=dataset.collate_batch_test,
                shuffle=False,
                pin_memory=False,
                drop_last=False,
            )
            for idx, batch in enumerate(loader):
                if args.max_items and idx >= args.max_items:
                    break
                if batch is None:
                    counts[case]["skipped"] += 1
                    continue
                batch = train_utils.to_device(batch, device)
                feat = extract_ifam_feature(model, batch["ego"])
                delays = delay_tensor(batch, case.comp, case.comm, device)
                output = forward_from_ifam_feature(model, comp_module(feat, delays), batch["ego"])
                pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch, {"ego": output})
                for iou in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, stats[case], iou)
                counts[case]["used"] += 1
                if args.log_interval and counts[case]["used"] % args.log_interval == 0:
                    elapsed = time.time() - started
                    print(
                        f"eval comp={case.comp} comm={case.comm} "
                        f"used={counts[case]['used']}/{len(dataset)} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
    write_csv(args.out, stats, counts, args.ckpt, comp_module.gate_value(), input_comp)


def write_csv(path: Path, stats: dict[GridCase, dict], counts: dict[GridCase, dict], ckpt: Path, gate: float, input_comp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "comm_delay_frames",
                "comp_delay_frames",
                "AP@0.5",
                "AP@0.7",
                "BEV AP@0.5",
                "BEV AP@0.7",
                "source",
                "notes",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for case in sorted(stats, key=lambda item: (item.comp, item.comm)):
            writer.writerow(
                {
                    "method": "DATA + Ours joint IFAM residual",
                    "comm_delay_frames": case.comm,
                    "comp_delay_frames": case.comp,
                    "AP@0.5": f"{ap_from_stat(stats[case], 0.5):.2f}",
                    "AP@0.7": f"{ap_from_stat(stats[case], 0.7):.2f}",
                    "BEV AP@0.5": "",
                    "BEV AP@0.7": "",
                    "source": "data_v2xsim_joint_detector_compensation",
                    "notes": (
                        f"input_compensation={input_comp}; joint_ckpt={ckpt}; "
                        f"gate={gate:.6f}; used={counts[case]['used']}; skipped={counts[case]['skipped']}"
                    ),
                }
            )
    print(f"wrote {path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--status", type=Path, default=DEFAULT_RUN_STATUS)
        p.add_argument("--model-dir", type=Path, default=None)
        p.add_argument("--info-dir", type=Path, default=REPO_ROOT / "datasets" / "v2xsim2_info")
        p.add_argument("--range", default="32,32")
        p.add_argument("--device", default="cuda")
        p.add_argument("--input-compensation", choices=["none", "ours", "syncnet"], default="ours")
        p.add_argument("--comp-delays", default="0,2,4")
        p.add_argument("--comm-delays", default="0,1,2,3,4,5")
        p.add_argument("--log-interval", type=int, default=100)
        p.add_argument("--max-items", type=int, default=0)
        p.add_argument("--num-workers", type=int, default=4)

    p_train = sub.add_parser("train")
    common(p_train)
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
    p_eval.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p_eval.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_eval.set_defaults(func=eval_grid)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
