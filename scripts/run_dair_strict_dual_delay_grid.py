#!/usr/bin/env python3
"""Run DAIR late-fusion AP under strict communication/computation delays."""

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


def isolated_root(gpu: int) -> Path:
    return ROOT / "external" / "DAIR-V2X-isolated" / f"gpu{gpu}"


def complete_log(path: Path) -> bool:
    if not path.exists():
        return False
    row = parse_log(path)
    return (
        row.get("status") == "complete"
        and row.get("car_bev_ap_0_50") is not None
        and row.get("car_bev_ap_0_70") is not None
    )


def sync_code(gpus: list[int]) -> None:
    rel_paths = [
        Path("v2x/config.py"),
        Path("v2x/scripts/eval_lidar_late_fusion_pointpillars.sh"),
        Path("v2x/dataset/dair_v2x_for_detection.py"),
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


def log_name(method: str, comm: int, comp: int) -> str:
    return f"late_fusion_{method}_comm{comm}_comp{comp}.log"


def run_one(
    method: str,
    comm: int,
    comp: int,
    gpu: int,
    args: argparse.Namespace,
) -> subprocess.Popen:
    root = isolated_root(gpu)
    log = args.log_dir / log_name(method, comm, comp)
    total_delay = comm + comp
    if method == "no_comp":
        comp_arg = "--no-comp"
    elif method == "tclf":
        comp_arg = f"--time-compensation-method {method} --temporal-compensation-target receiver"
    elif method == "syncnet":
        if not args.allow_pseudo_syncnet:
            raise ValueError(
                "DAIR SyncNet is not connected to a real DAIR checkpoint in this workspace; "
                "the local syncnet branch reuses TCLF temporal offsets. "
                "Pass --allow-pseudo-syncnet only for diagnostic reproduction."
            )
        comp_arg = "--time-compensation-method syncnet"
    elif method == "ours":
        if not args.ours_model.exists():
            raise FileNotFoundError(args.ours_model)
        comp_arg = (
            "--time-compensation-method ours"
            f" --ours-compensation-model {args.ours_model}"
            f" --ours-temporal-source {args.ours_temporal_source}"
            f" --ours-model-blend {args.ours_model_blend}"
            f" --ours-disagreement-threshold {args.ours_disagreement_threshold}"
            f" --ours-max-offset {args.ours_max_offset}"
            f" --ours-temporal-max-k {args.ours_temporal_max_k}"
            f" --ours-history-frames {args.ours_history_frames}"
            f" --ours-history-stride {args.ours_history_stride}"
        )
        if args.ours_fallback_model:
            comp_arg += " --ours-fallback-model"
        if args.ours_use_residual:
            comp_arg += " --ours-use-residual"
    else:
        raise ValueError(method)

    comp_arg += f" --strict-dual-delay --comm-delay-frames {comm} --comp-delay-frames {comp}"
    cmd = f"""
set -eo pipefail
source /raid/xuyifan/miniconda3/etc/profile.d/conda.sh
conda activate dair-v2x-openmmlab
mkdir -p {root}/cache/vic-late-lidar
(
	  cd {root}/v2x
	  export WANDB_MODE=offline
	  export PYTHONUNBUFFERED=1
	  export DAIR_KEEP_CACHE=1
	  echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] strict late_fusion_{method}_comm{comm}_comp{comp}: isolated_root={root} gpu={gpu}"
  bash scripts/eval_lidar_late_fusion_pointpillars.sh {gpu} late_fusion {total_delay} {args.range_start} {args.range_end} "{comp_arg}"
) > {log} 2>&1
"""
    return subprocess.Popen(["bash", "-lc", cmd], cwd=str(ROOT))


def run_missing(args: argparse.Namespace, missing: list[tuple[str, int, int]]) -> None:
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    if args.sync_code:
        sync_code(gpus)
    active: dict[subprocess.Popen, tuple[str, int, int, int, float]] = {}
    free_gpus = gpus[:]
    pending = list(missing)
    failed: subprocess.CalledProcessError | None = None
    while pending or active:
        while pending and free_gpus and len(active) < args.jobs:
            method, comm, comp = pending.pop(0)
            gpu = free_gpus.pop(0)
            proc = run_one(method, comm, comp, gpu, args)
            active[proc] = (method, comm, comp, gpu, time.time())
            print(f"started {method} comm={comm} comp={comp} on gpu{gpu} pid={proc.pid}", flush=True)
        time.sleep(args.poll_seconds)
        done = [proc for proc in active if proc.poll() is not None]
        if not done:
            for proc, (method, comm, comp, gpu, started) in active.items():
                print(
                    f"running {method} comm={comm} comp={comp} on gpu{gpu} elapsed={int(time.time() - started)}s",
                    flush=True,
                )
            continue
        for proc in done:
            method, comm, comp, gpu, started = active.pop(proc)
            free_gpus.append(gpu)
            rc = proc.wait()
            print(
                f"finished {method} comm={comm} comp={comp} on gpu{gpu} rc={rc} elapsed={int(time.time() - started)}s",
                flush=True,
            )
            if rc != 0:
                failed = subprocess.CalledProcessError(rc, f"{method}_comm{comm}_comp{comp}")
                pending.clear()
                for other in active:
                    other.terminate()
                break
        if failed is not None and not active:
            raise failed


def rows_from_logs(log_dir: Path, methods: list[str], comms: list[int], comps: list[int]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    names = {
        "no_comp": ("Late fusion w/o comp.", "Detector w/o compensation"),
        "tclf": ("Late fusion + TCLF", "Detector + TCLF"),
        "syncnet": ("Late fusion + SyncNet", "Detector + SyncNet"),
        "ours": ("Late fusion + Ours", "Detector + Ours"),
    }
    late_rows: list[dict[str, str]] = []
    detector_rows: list[dict[str, str]] = []
    for method in methods:
        late_name, detector_name = names[method]
        for comp in comps:
            for comm in comms:
                log = log_dir / log_name(method, comm, comp)
                if not complete_log(log):
                    continue
                row = parse_log(log)
                base = {
                    "comm_delay_frames": str(comm),
                    "comp_delay_frames": str(comp),
                    "AP@0.5": "",
                    "AP@0.7": "",
                    "BEV AP@0.5": f"{float(row['car_bev_ap_0_50']):.2f}",
                    "BEV AP@0.7": f"{float(row['car_bev_ap_0_70']):.2f}",
                    "source": "dair_strict_dual_delay_detector",
                    "notes": f"strict sender=T-(comm+comp), receiver=T-comp; log={row['log']}",
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
    parser.add_argument("--methods", default="no_comp,ours")
    parser.add_argument("--comms", default="0,1,2,3,4,5")
    parser.add_argument("--comps", default="0,2,4")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "results" / "dair_v2x" / "official_baselines" / "logs_strict_dual_delay_20260703")
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
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--ours-model", type=Path, default=DEFAULT_OURS_MODEL)
    parser.add_argument(
        "--ours-temporal-source",
        choices=["model", "temporal", "blend", "safe_hybrid", "history_model", "history_safe_hybrid"],
        default="safe_hybrid",
    )
    parser.add_argument("--ours-model-blend", type=float, default=0.10)
    parser.add_argument("--ours-disagreement-threshold", type=float, default=2.0)
    parser.add_argument("--ours-max-offset", type=float, default=8.0)
    parser.add_argument("--ours-temporal-max-k", type=int, default=7)
    parser.add_argument("--ours-history-frames", type=int, default=2)
    parser.add_argument("--ours-history-stride", choices=["horizon", "consecutive"], default="horizon")
    parser.add_argument("--ours-fallback-model", action="store_true")
    parser.add_argument("--ours-use-residual", action="store_true")
    parser.add_argument("--late-out", type=Path, default=ARTIFACT_DIR / "dair_late_fusion_ours_syncnet_ap.csv")
    parser.add_argument("--detector-out", type=Path, default=ARTIFACT_DIR / "dair_detector_ours_syncnet_ap.csv")
    args = parser.parse_args()

    args.ours_model = args.ours_model.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if "syncnet" in methods and not args.allow_pseudo_syncnet:
        raise ValueError(
            "DAIR SyncNet rows would be pseudo results: the local branch reuses TCLF temporal offsets. "
            "Remove syncnet from --methods, or pass --allow-pseudo-syncnet for diagnostics only."
        )
    comms = [int(item) for item in args.comms.split(",") if item.strip()]
    comps = [int(item) for item in args.comps.split(",") if item.strip()]
    missing = [
        (method, comm, comp)
        for method in methods
        for comp in comps
        for comm in comms
        if not complete_log(args.log_dir / log_name(method, comm, comp))
    ]
    print(f"missing logs: {len(missing)}")
    for method, comm, comp in missing:
        print(f"  {method} comm={comm} comp={comp}")
    if args.run_missing and missing:
        run_missing(args, missing)
    late_rows, detector_rows = rows_from_logs(args.log_dir, methods, comms, comps)
    write_csv(args.late_out, late_rows)
    write_csv(args.detector_out, detector_rows)


if __name__ == "__main__":
    main()
