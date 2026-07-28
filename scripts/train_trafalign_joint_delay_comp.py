#!/usr/bin/env python3
"""Jointly finetune TraF-Align with a gated residual delay-compensation module."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
TRAF_ROOT = Path("/tmp/TraF-Align_partial")
DEFAULT_CONFIG = TRAF_ROOT / "hypes_yaml" / "dair_v2x_seq" / "dair_v2x_seq_Trafalign.yaml"
DEFAULT_CKPT = TRAF_ROOT / "checkpoints" / "dair_v2x_seq_Trafalign_2024_12_28_04_42_36"
DEFAULT_DATA = ROOT / "datasets" / "DAIR-V2X-Seq" / "V2X-Seq-SPD"
DEFAULT_OUT = ROOT / "checkpoints" / "curated" / "ours_models" / "trafalign_joint_delay_comp"


def import_trafalign():
    if str(TRAF_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAF_ROOT))
    from datasets import build_dataset
    from hypes_yaml import yaml_utils
    from utils import train_utils

    return build_dataset, yaml_utils, train_utils


def load_converted_checkpoint(model: nn.Module, ckpt_dir: Path, epoch: int) -> dict[str, int]:
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


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta: float = 1.0 / 9.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        target = torch.where(torch.isnan(target), pred, target)
        diff = pred - target
        abs_diff = diff.abs()
        if self.beta < 1e-5:
            loss = abs_diff
        else:
            loss = torch.where(abs_diff < self.beta, 0.5 * abs_diff**2 / self.beta, abs_diff - 0.5 * self.beta)
        return loss * weights.unsqueeze(-1)


class AnchorDetectionLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        weights = cfg["model"]["head"]["loss_config"]["loss_weights"]
        self.cls_weight = float(weights.get("cls_weight", 1.0))
        self.loc_weight = float(weights.get("loc_weight", 2.0))
        self.alpha = 0.25
        self.gamma = 2.0
        self.reg_loss = WeightedSmoothL1Loss()
        self.last = {}

    @staticmethod
    def sigmoid_ce_with_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.clamp(logits, min=0) - logits * target + torch.log1p(torch.exp(-torch.abs(logits)))

    @staticmethod
    def add_sin_difference(boxes1: torch.Tensor, boxes2: torch.Tensor, dim: int = 6):
        rad_pred = torch.sin(boxes1[..., dim : dim + 1]) * torch.cos(boxes2[..., dim : dim + 1])
        rad_tgt = torch.cos(boxes1[..., dim : dim + 1]) * torch.sin(boxes2[..., dim : dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred, boxes1[..., dim + 1 :]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tgt, boxes2[..., dim + 1 :]], dim=-1)
        return boxes1, boxes2

    def focal_loss(self, logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        alpha_weight = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        pt = target * (1.0 - pred) + (1.0 - target) * pred
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)
        loss = focal_weight * self.sigmoid_ce_with_logits(logits, target)
        return loss * weights.unsqueeze(-1)

    def forward(self, batch_dict: dict) -> torch.Tensor:
        cls_preds = batch_dict["cls_preds"]
        box_preds = batch_dict["box_preds"]
        labels = batch_dict["box_cls_labels"].long()
        reg_targets = batch_dict["box_reg_targets"]

        batch_size = cls_preds.shape[0]
        cls_preds = cls_preds.view(batch_size, -1, 1)
        box_preds = box_preds.view(batch_size, -1, 7)

        cared = labels >= 0
        positives = labels > 0
        negatives = labels == 0
        cls_weights = (positives.float() + negatives.float()) * cared.float()
        reg_weights = positives.float()
        pos_normalizer = positives.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        cls_weights = cls_weights / pos_normalizer
        reg_weights = reg_weights / pos_normalizer

        cls_targets = positives.float().unsqueeze(-1)
        cls_loss = self.focal_loss(cls_preds, cls_targets, cls_weights).sum() / batch_size

        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, reg_targets)
        loc_loss = self.reg_loss(box_preds_sin, reg_targets_sin, reg_weights).sum() / batch_size

        total = self.cls_weight * cls_loss + self.loc_weight * loc_loss
        self.last = {
            "total": float(total.detach().cpu()),
            "cls": float(cls_loss.detach().cpu()),
            "loc": float(loc_loss.detach().cpu()),
            "pos": float(positives.sum().detach().cpu()),
        }
        return total


def prepare_config(args: argparse.Namespace) -> Path:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["name"] = args.name
    cfg["root_dir"] = str(args.data_root).rstrip("/") + "/"
    cfg["split_dir"] = str(args.split_dir)
    cfg["train_params"]["train_batch_size"] = args.batch_size
    cfg["train_params"]["val_batch_size"] = args.batch_size
    cfg["train_params"]["epoches"] = args.epochs
    cfg["wild_setting"]["async"] = True
    cfg["wild_setting"]["async_ego"] = True
    cfg["wild_setting"]["agent_i_delay"] = 0
    cfg["wild_setting"]["ego_delay_train_aug"] = [0, args.max_ego_delay_ms]
    cfg["wild_setting"]["agent_i_delay_train_aug"] = [0, args.max_agent_delay_ms]
    if args.ego_delay_train_choices:
        cfg["wild_setting"]["ego_delay_train_choices"] = [
            int(x) for x in args.ego_delay_train_choices.split(",") if x.strip()
        ]
    if args.agent_delay_train_choices:
        cfg["wild_setting"]["agent_i_delay_train_choices"] = [
            int(x) for x in args.agent_delay_train_choices.split(",") if x.strip()
        ]
    cfg["model"]["delay_compensation"] = {
        "enabled": True,
        "hidden_channels": args.hidden_channels,
        "max_delay_frames": args.max_agent_delay_ms // 100,
        "residual_scale": args.residual_scale,
        "gate_init": args.gate_init,
    }
    tmp_dir = Path(tempfile.mkdtemp(prefix="trafalign_joint_train_"))
    cfg_path = tmp_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def freeze_modules(model: nn.Module, freeze_reader_backbone: bool) -> None:
    if not freeze_reader_backbone:
        return
    for module_name in ("reader", "backbone"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = False


def train_only_compensator(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    comp = getattr(model.fusion_net, "delay_compensator", None)
    if comp is None:
        raise RuntimeError("delay_compensator is not enabled in the model config")
    for param in comp.parameters():
        param.requires_grad = True


def train_matching_parameters(model: nn.Module, patterns: str) -> list[str]:
    tokens = [token.strip() for token in patterns.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--trainable-module-patterns was provided but no non-empty pattern was found")
    for param in model.parameters():
        param.requires_grad = False
    matched = []
    for name, param in model.named_parameters():
        if any(token in name for token in tokens):
            param.requires_grad = True
            matched.append(name)
    if not matched:
        raise RuntimeError(f"No trainable parameters matched patterns: {tokens}")
    return matched


def freeze_batch_norm(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            child.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split-dir", type=Path, default=TRAF_ROOT / "datasets" / "Basedataset" / "V2XSeq_dataset_split_official.yaml")
    parser.add_argument("--pretrained-dir", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--pretrained-epoch", type=int, default=60)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--name", default="dair_v2x_seq_Trafalign_joint_delay_comp")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--comp-lr",
        type=float,
        default=0.0,
        help="Optional learning rate for fusion_net.delay_compensator params; other trainable params use --lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-ego-delay-ms", type=int, default=400)
    parser.add_argument("--max-agent-delay-ms", type=int, default=900)
    parser.add_argument(
        "--ego-delay-train-choices",
        default="",
        help="Comma-separated ego delay choices in ms for targeted training, e.g. 200,400.",
    )
    parser.add_argument(
        "--agent-delay-train-choices",
        default="",
        help="Comma-separated non-ego total delay choices in ms for targeted training.",
    )
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--residual-scale", type=float, default=0.2)
    parser.add_argument("--gate-init", type=float, default=-4.0)
    parser.add_argument("--freeze-reader-backbone", action="store_true", default=True)
    parser.add_argument("--train-only-compensator", action="store_true")
    parser.add_argument(
        "--trainable-module-patterns",
        default="",
        help=(
            "Comma-separated parameter-name substrings to unfreeze, e.g. "
            "'fusion_net.delay_compensator,head' or "
            "'fusion_net.delay_compensator,fusion_net.fusion_net,fusion_net.mapping,head'. "
            "If set, this overrides --train-only-compensator and --freeze-reader-backbone."
        ),
    )
    parser.add_argument("--freeze-bn", action="store_true", default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()

    build_dataset, yaml_utils, train_utils = import_trafalign()
    cfg_path = prepare_config(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, args.out_dir / "config.yaml")

    cfg = yaml_utils.load_yaml(str(cfg_path), argparse.Namespace(model_dir=None))
    cfg["voxelization"]["grid_size"] = np.asarray(cfg["voxelization"]["grid_size"])
    cfg = yaml_utils.check_pillar_params(cfg)

    dataset = build_dataset(cfg, set="train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    device = torch.device(args.device)
    model = train_utils.create_model(cfg).float().to(device)
    load_info = load_converted_checkpoint(model, args.pretrained_dir, args.pretrained_epoch)
    matched_trainable = None
    if args.trainable_module_patterns:
        matched_trainable = train_matching_parameters(model, args.trainable_module_patterns)
    elif args.train_only_compensator:
        train_only_compensator(model)
    else:
        freeze_modules(model, args.freeze_reader_backbone)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.comp_lr > 0:
        comp_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "fusion_net.delay_compensator" in name:
                comp_params.append(param)
            else:
                other_params.append(param)
        param_groups = []
        if comp_params:
            param_groups.append({"params": comp_params, "lr": args.comp_lr, "weight_decay": args.weight_decay})
        if other_params:
            param_groups.append({"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay})
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    criterion = AnchorDetectionLoss(cfg)

    metrics_path = args.out_dir / "train_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "epoch", "loss", "cls_loss", "loc_loss", "positives", "gate"])
        writer.writeheader()

    print(f"loaded pretrained: {load_info}", flush=True)
    print(f"trainable_params={sum(p.numel() for p in trainable)}", flush=True)
    if args.comp_lr > 0:
        print(f"optimizer_lrs=delay_compensator:{args.comp_lr}, other_trainable:{args.lr}", flush=True)
    if matched_trainable is not None:
        print(
            "trainable_patterns="
            f"{args.trainable_module_patterns}; matched_parameter_tensors={len(matched_trainable)}",
            flush=True,
        )
        for name in matched_trainable[:40]:
            print(f"  trainable: {name}", flush=True)
        if len(matched_trainable) > 40:
            print(f"  ... {len(matched_trainable) - 40} more trainable tensors", flush=True)

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch_data in loader:
            model.train()
            if args.freeze_bn:
                freeze_batch_norm(model)
            batch_data = train_utils.to_device(batch_data, device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_data)
            loss = criterion(preds)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            global_step += 1

            gate = None
            comp = getattr(model.fusion_net, "delay_compensator", None)
            if comp is not None:
                gate = float((torch.sigmoid(comp.gate_logit) * comp.residual_scale).detach().cpu())

            if global_step % args.log_interval == 0 or global_step == 1:
                last = criterion.last
                print(
                    f"step={global_step} epoch={epoch} loss={last['total']:.4f} "
                    f"cls={last['cls']:.4f} loc={last['loc']:.4f} pos={last['pos']:.0f} gate={gate:.6f}",
                    flush=True,
                )
                with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["step", "epoch", "loss", "cls_loss", "loc_loss", "positives", "gate"])
                    writer.writerow(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "loss": f"{last['total']:.6f}",
                            "cls_loss": f"{last['cls']:.6f}",
                            "loc_loss": f"{last['loc']:.6f}",
                            "positives": f"{last['pos']:.0f}",
                            "gate": f"{gate:.8f}",
                        }
                    )

            if args.save_every > 0 and global_step % args.save_every == 0:
                torch.save(model.state_dict(), args.out_dir / f"net_epoch{global_step}.pth")

            if args.max_steps > 0 and global_step >= args.max_steps:
                torch.save(model.state_dict(), args.out_dir / "net_epoch1.pth")
                shutil.rmtree(cfg_path.parent, ignore_errors=True)
                print(f"saved {args.out_dir / 'net_epoch1.pth'}", flush=True)
                return

    torch.save(model.state_dict(), args.out_dir / "net_epoch1.pth")
    shutil.rmtree(cfg_path.parent, ignore_errors=True)
    print(f"saved {args.out_dir / 'net_epoch1.pth'}", flush=True)


if __name__ == "__main__":
    main()
