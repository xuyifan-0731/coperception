#!/usr/bin/env python3
"""Run and parse DAIR late-fusion detector-side compensation grids."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time
from pathlib import Path

from parse_dair_official_baseline_logs import parse_log


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "results" / "dair_v2x" / "official_baselines" / "logs"
ARTIFACT_DIR = ROOT / "results" / "260615_fill" / "artifacts"
DEFAULT_OURS_MODEL = ARTIFACT_DIR / "dair_ours_motion_compensator.npz"
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


def complete_log(path: Path) -> bool:
    if not path.exists():
        return False
    row = parse_log(path)
    return row.get("status") == "complete" and bool(row.get("car_bev_ap_0_50")) and bool(row.get("car_bev_ap_0_70"))


def isolated_root(gpu: int) -> Path:
    return ROOT / "external" / "DAIR-V2X-isolated" / f"gpu{gpu}"


def sync_late_fusion_code(gpus: list[int]) -> None:
    rel_paths = [
        Path("v2x/models/detection_models/mmdet3d_anymodel_anymodality_late.py"),
        Path("v2x/models/model_utils/late_fusion_utils.py"),
    ]
    for rel in rel_paths:
        src = isolated_root(0) / rel
        for gpu in gpus:
            dst = isolated_root(gpu) / rel
            if not dst.exists():
                raise FileNotFoundError(dst)
            if dst == src or os.path.samefile(src, dst):
                continue
            shutil.copy2(src, dst)


def run_one(
    method: str,
    k: int,
    gpu: int,
    log_dir: Path,
    range_start: int,
    range_end: int,
    ours_model: Path,
    ours_use_residual: bool,
    ours_temporal_source: str,
    ours_model_blend: float,
    ours_disagreement_threshold: float,
    ours_max_offset: float,
    ours_temporal_max_k: int,
    ours_fallback_model: bool,
    allow_pseudo_syncnet: bool,
) -> subprocess.Popen:
    root = isolated_root(gpu)
    if not root.exists():
        raise FileNotFoundError(root)
    log = log_dir / f"late_fusion_{method}_k{k}.log"
    if method == "syncnet" and not allow_pseudo_syncnet:
        raise ValueError(
            "DAIR SyncNet is not connected to a real DAIR checkpoint in this workspace; "
            "the local syncnet branch reuses TCLF temporal offsets. "
            "Pass --allow-pseudo-syncnet only for diagnostic reproduction."
        )
    comp_arg = f"--time-compensation-method {method}"
    if method == "ours":
        if not ours_model.exists():
            raise FileNotFoundError(ours_model)
        comp_arg += f" --ours-compensation-model {ours_model}"
        comp_arg += f" --ours-temporal-source {ours_temporal_source}"
        comp_arg += f" --ours-model-blend {ours_model_blend}"
        comp_arg += f" --ours-disagreement-threshold {ours_disagreement_threshold}"
        comp_arg += f" --ours-max-offset {ours_max_offset}"
        comp_arg += f" --ours-temporal-max-k {ours_temporal_max_k}"
        if ours_fallback_model:
            comp_arg += " --ours-fallback-model"
        if ours_use_residual:
            comp_arg += " --ours-use-residual"
    cmd = f"""
set -eo pipefail
source /raid/xuyifan/miniconda3/etc/profile.d/conda.sh
conda activate dair-v2x-openmmlab
rm -rf {root}/cache
mkdir -p {root}/cache/vic-late-lidar
(
  cd {root}/v2x
  export WANDB_MODE=offline
  export PYTHONUNBUFFERED=1
  echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] late_fusion_{method}_k{k}: isolated_root={root} gpu={gpu}"
  bash scripts/eval_lidar_late_fusion_pointpillars.sh {gpu} late_fusion {k} {range_start} {range_end} "{comp_arg}"
) > {log} 2>&1
"""
    return subprocess.Popen(["bash", "-lc", cmd], cwd=str(ROOT))


def run_missing(args: argparse.Namespace, missing: list[tuple[str, int]]) -> None:
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    if args.sync_code:
        sync_late_fusion_code(gpus)
    active: dict[subprocess.Popen, tuple[str, int, int, float]] = {}
    free_gpus = gpus[:]
    pending = list(missing)
    failed: subprocess.CalledProcessError | None = None
    while pending or active:
        while pending and free_gpus and len(active) < args.jobs:
            method, k = pending.pop(0)
            gpu = free_gpus.pop(0)
            proc = run_one(
                method,
                k,
                gpu,
                args.log_dir,
                args.range_start,
                args.range_end,
                args.ours_model,
                args.ours_use_residual,
                args.ours_temporal_source,
                args.ours_model_blend,
                args.ours_disagreement_threshold,
                args.ours_max_offset,
                args.ours_temporal_max_k,
                args.ours_fallback_model,
                args.allow_pseudo_syncnet,
            )
            active[proc] = (method, k, gpu, time.time())
            print(f"started late_fusion_{method}_k{k} on gpu{gpu} pid={proc.pid}", flush=True)
        time.sleep(30)
        done = [proc for proc in active if proc.poll() is not None]
        if not done:
            for proc, (method, k, gpu, started) in active.items():
                elapsed = int(time.time() - started)
                print(f"running late_fusion_{method}_k{k} on gpu{gpu} elapsed={elapsed}s", flush=True)
            continue
        for proc in done:
            method, k, gpu, started = active.pop(proc)
            free_gpus.append(gpu)
            rc = proc.wait()
            elapsed = int(time.time() - started)
            print(f"finished late_fusion_{method}_k{k} on gpu{gpu} rc={rc} elapsed={elapsed}s", flush=True)
            if rc != 0:
                failed = subprocess.CalledProcessError(rc, f"late_fusion_{method}_k{k}")
                pending.clear()
                for other in active:
                    other.terminate()
                break
        if failed is not None and not active:
            raise failed


def rows_from_logs(log_dir: Path, methods: list[str], comps: list[int], comms: list[int]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    late_rows: list[dict[str, str]] = []
    detector_rows: list[dict[str, str]] = []
    method_names = {
        "ours": ("Late fusion + Ours", "Detector + Ours"),
        "syncnet": ("Late fusion + SyncNet", "Detector + SyncNet"),
    }
    for method in methods:
        late_name, detector_name = method_names[method]
        parsed: dict[int, dict[str, object]] = {}
        for k in range(max(comps) + max(comms) + 1):
            log = log_dir / f"late_fusion_{method}_k{k}.log"
            if complete_log(log):
                parsed[k] = parse_log(log)
        for comp in comps:
            for comm in comms:
                k = comp + comm
                row = parsed.get(k)
                if not row:
                    continue
                base = {
                    "comm_delay_frames": str(comm),
                    "comp_delay_frames": str(comp),
                    "AP@0.5": "",
                    "AP@0.7": "",
                    "BEV AP@0.5": f"{float(row['car_bev_ap_0_50']):.2f}",
                    "BEV AP@0.7": f"{float(row['car_bev_ap_0_70']):.2f}",
                    "source": "dair_official_late_fusion_detector_comp",
                    "notes": f"k={k}; log={row['log']}",
                }
                late_rows.append({"method": late_name, **base})
                detector_rows.append({"method": detector_name, **base})
                if method == "ours" and comp == 0 and comm == 0:
                    detector_rows.append({"method": "Ours detector AP", **base})
    return late_rows, detector_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="ours,syncnet")
    parser.add_argument("--k-list", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--comps", default="0,2,4")
    parser.add_argument("--comms", default="0,1,2,3,4,5")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--gpus", default="0,1,2,3,4,6")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--range-start", type=int, default=0)
    parser.add_argument("--range-end", type=int, default=100)
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument("--sync-code", action="store_true")
    parser.add_argument(
        "--allow-pseudo-syncnet",
        action="store_true",
        help="Allow the diagnostic DAIR syncnet branch, which reuses TCLF temporal offsets.",
    )
    parser.add_argument("--ours-model", type=Path, default=DEFAULT_OURS_MODEL)
    parser.add_argument(
        "--ours-temporal-source",
        choices=["model", "temporal", "blend", "safe_hybrid"],
        default="model",
    )
    parser.add_argument("--ours-model-blend", type=float, default=0.10)
    parser.add_argument("--ours-disagreement-threshold", type=float, default=2.0)
    parser.add_argument("--ours-max-offset", type=float, default=8.0)
    parser.add_argument("--ours-temporal-max-k", type=int, default=-1)
    parser.add_argument("--ours-fallback-model", action="store_true")
    parser.add_argument("--ours-use-residual", action="store_true")
    parser.add_argument("--late-out", type=Path, default=ARTIFACT_DIR / "dair_late_fusion_ours_syncnet_ap.csv")
    parser.add_argument("--detector-out", type=Path, default=ARTIFACT_DIR / "dair_detector_ours_syncnet_ap.csv")
    args = parser.parse_args()

    args.ours_model = args.ours_model.resolve()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if "syncnet" in methods and not args.allow_pseudo_syncnet:
        raise ValueError(
            "DAIR SyncNet rows would be pseudo results: the local branch reuses TCLF temporal offsets. "
            "Remove syncnet from --methods, or pass --allow-pseudo-syncnet for diagnostics only."
        )
    k_list = [int(item) for item in args.k_list.split(",") if item.strip()]
    comps = [int(item) for item in args.comps.split(",") if item.strip()]
    comms = [int(item) for item in args.comms.split(",") if item.strip()]
    args.log_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        (method, k)
        for method in methods
        for k in k_list
        if not complete_log(args.log_dir / f"late_fusion_{method}_k{k}.log")
    ]
    print(f"missing logs: {len(missing)}")
    for method, k in missing:
        print(f"  late_fusion_{method}_k{k}")
    if args.run_missing and missing:
        run_missing(args, missing)

    late_rows, detector_rows = rows_from_logs(args.log_dir, methods, comps, comms)
    write_csv(args.late_out, late_rows)
    write_csv(args.detector_out, detector_rows)


if __name__ == "__main__":
    main()
