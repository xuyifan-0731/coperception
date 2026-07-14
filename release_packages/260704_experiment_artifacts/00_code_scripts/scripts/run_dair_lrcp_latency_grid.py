#!/usr/bin/env python3
"""Evaluate DAIR-V2X LRCP latency grids and write the 260615 artifact CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
LRCP_ROOT = ROOT / "external" / "LRCP"
if str(LRCP_ROOT) not in sys.path:
    sys.path.insert(0, str(LRCP_ROOT))

from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.hypes_yaml import yaml_utils  # noqa: E402
from opencood.tools import inference_utils, train_utils  # noqa: E402
from opencood.utils import eval_utils  # noqa: E402


SCHEMA = [
    "method",
    "comm_delay_frames",
    "comp_delay_frames",
    "AP@0.5",
    "AP@0.7",
    "BEV AP@0.5",
    "BEV AP@0.7",
    "source",
    "notes",
]


METHODS = {
    "none": "LRCP w/o compensation",
    "ours": "LRCP + Ours",
    "syncnet": "LRCP + SyncNet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Stage2 LRCP model directory for LRCP + Ours.")
    parser.add_argument("--stage1-model-dir", type=Path, default=None, help="Stage1 model directory for GT-flow sync baseline.")
    parser.add_argument("--out", type=Path, default=ROOT / "results/260615_fill/artifacts/dair_lrcp_latency_ap.csv")
    parser.add_argument("--methods", default="none,ours")
    parser.add_argument("--delays", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--comps", default="0,2,4")
    parser.add_argument("--comms", default="0,1,2,3,4,5")
    parser.add_argument(
        "--strict-dual-delay",
        action="store_true",
        help="Use ego=T-Comp and infrastructure=T-(Comm+Comp), with labels at T.",
    )
    parser.add_argument(
        "--allow-pseudo-syncnet",
        action="store_true",
        help="Allow the old LRCP stage1 GT-flow reference rows to be emitted as LRCP + SyncNet. This is not a real DAIR SyncNet checkpoint.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit on validation samples.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_nested(mapping: dict, keys: tuple[str, ...], value) -> None:
    cur = mapping
    for key in keys[:-1]:
        cur = cur[key]
    cur[keys[-1]] = value


def configure_hypes(base_hypes: dict, mode: str, delay: int, ego_delay: int = 0, strict_dual_delay: bool = False) -> dict:
    hypes = deepcopy(base_hypes)
    hypes["time_delay"] = delay
    hypes["strict_dual_delay"] = strict_dual_delay
    hypes["ego_time_delay"] = ego_delay
    hypes["align_delayed_ego_to_current"] = True
    hypes["validate_dir"] = hypes["test_dir"]
    if hypes["train_params"]["batch_size"] != 1:
        hypes["train_params"]["batch_size"] = 1

    if mode == "none":
        set_nested(hypes, ("model", "args", "calibrate"), False)
        set_nested(hypes, ("model", "args", "base_bev_backbone", "defor_encoder", "calibrate"), False)
        set_nested(hypes, ("model", "args", "base_bev_backbone", "defor_encoder_single", "calibrate"), False)
    elif mode == "ours":
        for path in (
            ("model", "args", "train_stage"),
            ("model", "args", "base_bev_backbone", "defor_encoder", "train_stage"),
        ):
            set_nested(hypes, path, "stage2")
        hypes["train_stage"] = "stage2"
        hypes.setdefault("loss", {}).setdefault("args", {})["train_stage"] = "stage2"
    elif mode == "syncnet":
        # LRCP has no DAIR SyncNet checkpoint. This mode uses the same delayed
        # detector with LRCP's generated DAIR flow supervision at inference as a
        # deterministic temporal synchronization reference.
        for path in (
            ("model", "args", "train_stage"),
            ("model", "args", "base_bev_backbone", "defor_encoder", "train_stage"),
        ):
            set_nested(hypes, path, "stage1")
        hypes["train_stage"] = "stage1"
        hypes.setdefault("loss", {}).setdefault("args", {})["train_stage"] = "stage1"
    else:
        raise ValueError(f"unknown mode: {mode}")
    return hypes


def load_model(model_dir: Path, hypes: dict, device: torch.device):
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(model_dir), model)
    model.to(device)
    model.eval()
    return model


def evaluate_setting(
    model_dir: Path,
    base_hypes: dict,
    mode: str,
    inf_delay: int,
    ego_delay: int,
    args: argparse.Namespace,
) -> tuple[float, float]:
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    hypes = configure_hypes(base_hypes, mode, inf_delay, ego_delay, args.strict_dual_delay)
    model = load_model(model_dir, hypes, device)

    dataset = build_dataset(hypes, visualize=False, train=False)
    if args.limit:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        collate_fn = dataset.dataset.collate_batch_test
    else:
        collate_fn = dataset.collate_batch_test

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }

    for i, batch_data in enumerate(loader):
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            infer_result = inference_utils.inference_intermediate_fusion(batch_data, model, dataset.dataset if args.limit else dataset)
            pred_box_tensor = infer_result["pred_box_tensor"]
            gt_box_tensor = infer_result["gt_box_tensor"]
            pred_score = infer_result["pred_score"]
            for thresh in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, thresh)
        if i % 100 == 0:
            print(f"{mode} inf_delay={inf_delay} ego_delay={ego_delay} sample={i}/{len(loader)}", flush=True)

    ap30 = safe_ap(result_stat, 0.3)
    ap50 = safe_ap(result_stat, 0.5)
    ap70 = safe_ap(result_stat, 0.7)
    print(
        f"{mode} inf_delay={inf_delay} ego_delay={ego_delay}: "
        f"AP30={ap30:.4f} AP50={ap50:.4f} AP70={ap70:.4f}",
        flush=True,
    )
    return ap50 * 100.0, ap70 * 100.0


def evaluate_delay(model_dir: Path, base_hypes: dict, mode: str, delay: int, args: argparse.Namespace) -> tuple[float, float]:
    return evaluate_setting(model_dir, base_hypes, mode, delay, 0, args)


def safe_ap(result_stat: dict, thresh: float) -> float:
    if result_stat[thresh]["gt"] == 0:
        return 0.0
    return eval_utils.calculate_ap(result_stat, thresh)[0]


def project_rows(results: dict[tuple[str, int], tuple[float, float]], comps: list[int], comms: list[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if ("ours", 0) in results:
        ap50, ap70 = results[("ours", 0)]
        rows.append(
            {
                "method": "LRCP",
                "comm_delay_frames": "0",
                "comp_delay_frames": "0",
                "AP@0.5": "",
                "AP@0.7": "",
                "BEV AP@0.5": f"{ap50:.4f}",
                "BEV AP@0.7": f"{ap70:.4f}",
                "source": "dair_lrcp_latency_eval",
                "notes": "no-delay LRCP from stage2 model at k=0",
            }
        )

    for mode, method in METHODS.items():
        for comp in comps:
            for comm in comms:
                k = comp + comm
                if (mode, k) not in results:
                    continue
                ap50, ap70 = results[(mode, k)]
                note = f"k=comm+comp={k}"
                if mode == "syncnet":
                    note += "; LRCP DAIR SyncNet-style GT-flow synchronization reference"
                if mode == "ours" and k > 5:
                    note += "; stage2 trained with max delay 5, evaluated by temporal extrapolation"
                rows.append(
                    {
                        "method": method,
                        "comm_delay_frames": str(comm),
                        "comp_delay_frames": str(comp),
                        "AP@0.5": "",
                        "AP@0.7": "",
                        "BEV AP@0.5": f"{ap50:.4f}",
                        "BEV AP@0.7": f"{ap70:.4f}",
                        "source": "dair_lrcp_latency_eval",
                        "notes": note,
                    }
                )
    return rows


def project_strict_rows(
    results: dict[tuple[str, int, int], tuple[float, float]],
    comps: list[int],
    comms: list[int],
    methods: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    no_delay_key = ("ours", 0, 0) if ("ours", 0, 0) in results else ("none", 0, 0)
    if no_delay_key in results:
        ap50, ap70 = results[no_delay_key]
        rows.append(
            {
                "method": "LRCP",
                "comm_delay_frames": "0",
                "comp_delay_frames": "0",
                "AP@0.5": "",
                "AP@0.7": "",
                "BEV AP@0.5": f"{ap50:.4f}",
                "BEV AP@0.7": f"{ap70:.4f}",
                "source": "dair_lrcp_strict_dual_delay_eval",
                "notes": "no-delay LRCP at sender=T receiver=T",
            }
        )

    for mode in methods:
        method = METHODS[mode]
        for comp in comps:
            for comm in comms:
                key = (mode, comp, comm)
                if key not in results:
                    continue
                ap50, ap70 = results[key]
                inf_delay = comp + comm
                note = (
                    f"strict-ish LRCP two-agent adapter: receiver=T-{comp} "
                    f"ego-motion aligned to T, sender=T-{inf_delay}; labels=T"
                )
                if mode == "ours":
                    note += "; LRCP learned compensation applies to the non-ego infrastructure agent only"
                    if inf_delay > 5:
                        note += "; stage2 trained with max delay 5, evaluated by temporal extrapolation"
                if mode == "syncnet":
                    note += "; pseudo SyncNet is LRCP stage1 GT-flow reference, not a DAIR SyncNet checkpoint"
                rows.append(
                    {
                        "method": method,
                        "comm_delay_frames": str(comm),
                        "comp_delay_frames": str(comp),
                        "AP@0.5": "",
                        "AP@0.7": "",
                        "BEV AP@0.5": f"{ap50:.4f}",
                        "BEV AP@0.7": f"{ap70:.4f}",
                        "source": "dair_lrcp_strict_dual_delay_eval",
                        "notes": note,
                    }
                )
    return rows


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    np.random.seed(303)
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    delays = [int(item) for item in args.delays.split(",") if item.strip()]
    comps = [int(item) for item in args.comps.split(",") if item.strip()]
    comms = [int(item) for item in args.comms.split(",") if item.strip()]
    if "syncnet" in methods and not args.allow_pseudo_syncnet:
        raise ValueError(
            "LRCP has no real DAIR SyncNet checkpoint in this workspace. "
            "Remove syncnet from --methods, or pass --allow-pseudo-syncnet only for diagnostic GT-flow reference rows."
        )

    stage2_hypes = yaml_utils.load_yaml(str(args.model_dir / "config.yaml"))
    stage1_dir = args.stage1_model_dir or args.model_dir
    stage1_hypes = yaml_utils.load_yaml(str(stage1_dir / "config.yaml"))

    if args.strict_dual_delay:
        strict_results: dict[tuple[str, int, int], tuple[float, float]] = {}
        for method in methods:
            if method not in METHODS:
                raise ValueError(f"unknown method {method}; valid={sorted(METHODS)}")
            source_dir = stage1_dir if method == "syncnet" else args.model_dir
            source_hypes = stage1_hypes if method == "syncnet" else stage2_hypes
            for comp in comps:
                for comm in comms:
                    strict_results[(method, comp, comm)] = evaluate_setting(
                        source_dir,
                        source_hypes,
                        method,
                        inf_delay=comm + comp,
                        ego_delay=comp,
                        args=args,
                    )
        rows = project_strict_rows(strict_results, comps, comms, methods)
    else:
        results: dict[tuple[str, int], tuple[float, float]] = {}
        for method in methods:
            if method not in METHODS:
                raise ValueError(f"unknown method {method}; valid={sorted(METHODS)}")
            source_dir = stage1_dir if method == "syncnet" else args.model_dir
            source_hypes = stage1_hypes if method == "syncnet" else stage2_hypes
            for delay in delays:
                results[(method, delay)] = evaluate_delay(source_dir, source_hypes, method, delay, args)
        rows = project_rows(results, comps, comms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
