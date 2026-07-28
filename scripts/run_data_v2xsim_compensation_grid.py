#!/usr/bin/env python3
"""Run and parse DATA compensation grids on V2X-Sim 2.0."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "external" / "DATA"
STATE_PATH = ROOT / "runs" / "data_v2xsim_pipeline" / "status.json"
ARTIFACT = ROOT / "results" / "260615_fill" / "artifacts" / "v2xsim_data_compensation_ap.csv"
LOG_DIR = ROOT / "results" / "260615_fill" / "logs" / "data_compensation"
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


def parse_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def model_dir_from_status() -> Path:
    data = yaml.safe_load(STATE_PATH.read_text())
    return Path(data["artifacts"]["model_dir"])


def eval_yaml(model_dir: Path, method: str, comm: int, comp: int) -> Path | None:
    patterns = [
        f"eval_intermediate_{method}_comm{comm}_comp{comp}_32_32_epoch*.yaml",
        f"eval_intermediate_{method}_comm{comm}_comp{comp}_32.0_32.0_epoch*.yaml",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(model_dir.glob(pattern))
    complete = []
    for path in matches:
        if parse_eval(path) is not None:
            complete.append(path)
    if not complete:
        return None
    return sorted(complete, key=lambda p: p.stat().st_mtime)[-1]


def parse_eval(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    try:
        text = path.read_text()
        ap50 = re.search(r"^ap_50:\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)
        ap70 = re.search(r"^ap_70:\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)
        if not ap50 or not ap70:
            return None
        return {
            "AP@0.5": float(ap50.group(1)) * 100.0,
            "AP@0.7": float(ap70.group(1)) * 100.0,
        }
    except Exception:
        return None


def run_one(model_dir: Path, method: str, comm: int, comp: int, gpu: int, conda_env: str) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"data_{method}_comm{comm}_comp{comp}.log"
    cmd = f"""
set -eo pipefail
source /raid/xuyifan/miniconda3/etc/profile.d/conda.sh
conda activate {conda_env}
cd {DATA_ROOT}
export PYTHONPATH={DATA_ROOT}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES={gpu}
export PYTHONUNBUFFERED=1
python opencood/tools/inference.py \\
  --model_dir {model_dir} \\
  --fusion_method intermediate \\
  --range 32,32 \\
  --save_vis_interval 1000000 \\
  --comm_delay {comm} \\
  --comp_delay {comp} \\
  --compensation_method {method}
"""
    with log.open("w") as handle:
        handle.write(cmd)
        handle.write("\n")
    return subprocess.Popen(["bash", "-lc", f"({cmd}) >> {log} 2>&1"], cwd=str(ROOT))


def run_missing(args: argparse.Namespace, missing: list[tuple[str, int, int]], model_dir: Path) -> None:
    gpus = parse_list(args.gpus)
    free_gpus = gpus[:]
    active: dict[subprocess.Popen, tuple[str, int, int, int, float]] = {}
    pending = list(missing)
    while pending or active:
        while pending and free_gpus and len(active) < args.jobs:
            method, comm, comp = pending.pop(0)
            gpu = free_gpus.pop(0)
            proc = run_one(model_dir, method, comm, comp, gpu, args.conda_env)
            active[proc] = (method, comm, comp, gpu, time.time())
            print(f"started DATA {method} comm{comm} comp{comp} on gpu{gpu} pid={proc.pid}", flush=True)
        time.sleep(args.poll_seconds)
        done = [proc for proc in active if proc.poll() is not None]
        if not done:
            for _, (method, comm, comp, gpu, started) in active.items():
                elapsed = int(time.time() - started)
                print(f"running DATA {method} comm{comm} comp{comp} on gpu{gpu} elapsed={elapsed}s", flush=True)
            continue
        for proc in done:
            method, comm, comp, gpu, started = active.pop(proc)
            free_gpus.append(gpu)
            rc = proc.wait()
            elapsed = int(time.time() - started)
            print(f"finished DATA {method} comm{comm} comp{comp} on gpu{gpu} rc={rc} elapsed={elapsed}s", flush=True)
            if rc != 0:
                raise subprocess.CalledProcessError(rc, f"DATA {method} comm{comm} comp{comp}")


def rows_from_yaml(model_dir: Path, methods: list[str], comms: list[int], comps: list[int]) -> list[dict[str, str]]:
    names = {"ours": "DATA + Ours", "syncnet": "DATA + SyncNet"}
    rows = []
    for method in methods:
        for comp in comps:
            for comm in comms:
                path = eval_yaml(model_dir, method, comm, comp)
                if path is None:
                    continue
                metrics = parse_eval(path)
                if metrics is None:
                    continue
                rows.append({
                    "method": names[method],
                    "comm_delay_frames": str(comm),
                    "comp_delay_frames": str(comp),
                    "AP@0.5": f"{metrics['AP@0.5']:.2f}",
                    "AP@0.7": f"{metrics['AP@0.7']:.2f}",
                    "BEV AP@0.5": "",
                    "BEV AP@0.7": "",
                    "source": "data_v2xsim_detector_compensation",
                    "notes": f"compensation_method={method}; eval_file={path}",
                })
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="ours,syncnet")
    parser.add_argument("--comms", default="0,1,2,3,4,5")
    parser.add_argument("--comps", default="0,2,4")
    parser.add_argument("--gpus", default="0,1,2,3,4,6")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--conda-env", default="dair-v2x-openmmlab")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument("--force", action="store_true", help="Run requested grid even if eval yaml already exists.")
    parser.add_argument("--out", type=Path, default=ARTIFACT)
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    comms = parse_list(args.comms)
    comps = parse_list(args.comps)
    model_dir = model_dir_from_status()
    missing = [
        (method, comm, comp)
        for method in methods
        for comp in comps
        for comm in comms
        if args.force or eval_yaml(model_dir, method, comm, comp) is None
    ]
    print(f"missing eval yaml: {len(missing)}")
    for method, comm, comp in missing:
        print(f"  DATA {method} comm{comm} comp{comp}")
    if args.run_missing and missing:
        run_missing(args, missing, model_dir)
    write_rows(args.out, rows_from_yaml(model_dir, methods, comms, comps))


if __name__ == "__main__":
    main()
