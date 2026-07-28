#!/usr/bin/env python3
"""Run DAIR LRCP strict dual-delay settings in parallel subprocesses."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    ROOT
    / "external/LRCP/opencood/logs/dairv2x_lrcp_point_pillar_lidar_deformable_attn_stage2_2026_07_01_19_32_46"
)
DEFAULT_PART_DIR = ROOT / "results/260615_fill/artifacts/dair_lrcp_strict_dual_delay_parts"
DEFAULT_OUT = ROOT / "results/260615_fill/artifacts/dair_lrcp_strict_dual_delay_ap.csv"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--methods", default="none,ours")
    parser.add_argument("--comps", default="0,2,4")
    parser.add_argument("--comms", default="0,1,2,3,4,5")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--part-dir", type=Path, default=DEFAULT_PART_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def part_path(part_dir: Path, method: str, comp: int, comm: int) -> Path:
    return part_dir / f"{method}_comp{comp}_comm{comm}.csv"


def part_complete(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return bool(rows)


def launch(args: argparse.Namespace, method: str, comp: int, comm: int, gpu: str) -> subprocess.Popen:
    out = part_path(args.part_dir, method, comp, comm)
    log = out.with_suffix(".log")
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_dair_lrcp_latency_grid.py"),
        "--model-dir",
        str(args.model_dir),
        "--methods",
        method,
        "--strict-dual-delay",
        "--comps",
        str(comp),
        "--comms",
        str(comm),
        "--num-workers",
        str(args.num_workers),
        "--device",
        "cuda",
        "--out",
        str(out),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_handle = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    proc._lrcp_log_handle = log_handle  # type: ignore[attr-defined]
    return proc


def merge(part_dir: Path, out: Path, methods: list[str], comps: list[int], comms: list[int]) -> None:
    rows_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for method in methods:
        for comp in comps:
            for comm in comms:
                path = part_path(part_dir, method, comp, comm)
                if not path.exists():
                    continue
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        key = (row["method"], row["comp_delay_frames"], row["comm_delay_frames"])
                        rows_by_key[key] = row

    def sort_key(item: tuple[tuple[str, str, str], dict[str, str]]) -> tuple[int, int, int]:
        row = item[1]
        order = {"LRCP": 0, "LRCP w/o compensation": 1, "LRCP + Ours": 2}
        return (
            order.get(row["method"], 99),
            int(row["comp_delay_frames"]),
            int(row["comm_delay_frames"]),
        )

    rows = [row for _, row in sorted(rows_by_key.items(), key=sort_key)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"merged {len(rows)} rows to {out}")


def main() -> None:
    args = parse_args()
    args.part_dir.mkdir(parents=True, exist_ok=True)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    comps = [int(item) for item in args.comps.split(",") if item.strip()]
    comms = [int(item) for item in args.comms.split(",") if item.strip()]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]

    pending = [
        (method, comp, comm)
        for method in methods
        for comp in comps
        for comm in comms
        if args.force or not part_complete(part_path(args.part_dir, method, comp, comm))
    ]
    print(f"pending={len(pending)}")
    active: dict[subprocess.Popen, tuple[str, int, int, str, float]] = {}
    free_gpus = gpus[:]

    while pending or active:
        while pending and free_gpus and len(active) < args.jobs:
            method, comp, comm = pending.pop(0)
            gpu = free_gpus.pop(0)
            proc = launch(args, method, comp, comm, gpu)
            active[proc] = (method, comp, comm, gpu, time.time())
            print(f"started {method} comp={comp} comm={comm} gpu={gpu} pid={proc.pid}", flush=True)
        time.sleep(args.poll_seconds)
        for proc in list(active):
            if proc.poll() is None:
                continue
            method, comp, comm, gpu, started = active.pop(proc)
            free_gpus.append(gpu)
            handle = getattr(proc, "_lrcp_log_handle", None)
            if handle is not None:
                handle.close()
            elapsed = int(time.time() - started)
            print(f"finished {method} comp={comp} comm={comm} gpu={gpu} rc={proc.returncode} elapsed={elapsed}s", flush=True)
            if proc.returncode != 0:
                for other in active:
                    other.terminate()
                raise subprocess.CalledProcessError(proc.returncode, f"{method}_comp{comp}_comm{comm}")
        if active:
            status = []
            for _, (method, comp, comm, gpu, started) in active.items():
                status.append(f"{method}:c{comp}/m{comm}@gpu{gpu}:{int(time.time() - started)}s")
            print("running " + ", ".join(status), flush=True)

    merge(args.part_dir, args.out, methods, comps, comms)


if __name__ == "__main__":
    main()
