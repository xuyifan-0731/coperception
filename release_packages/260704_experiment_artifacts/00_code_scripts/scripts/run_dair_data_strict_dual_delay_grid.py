#!/usr/bin/env python3
"""Run DATA on DAIR-V2X-C under the strict sender/receiver delay protocol."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "external" / "DATA"
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "data_dair_official_eval"
DEFAULT_OUT = REPO_ROOT / "results" / "260615_fill" / "artifacts" / "dair_data_strict_dual_delay_ap.csv"
METHOD_NAMES = {
    "none": "DATA w/o compensation",
    "ours": "DATA + Ours",
}


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def conda_cmd(env_name: str, cmd: list[str]) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", env_name] + cmd


def latest_eval_yaml(model_dir: Path, started_at: float) -> Path:
    candidates = [
        p for p in model_dir.glob("eval_intermediate*.yaml")
        if p.stat().st_mtime >= started_at - 1.0
    ]
    if not candidates:
        raise FileNotFoundError(f"No eval_intermediate*.yaml written in {model_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_ap(path: Path) -> tuple[float, float, float]:
    text = path.read_text()

    def find_ap(*names: str) -> float:
        for name in names:
            match = re.search(rf"^{re.escape(name)}:\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)
            if match:
                return float(match.group(1)) * 100.0
        raise KeyError(f"AP key not found in {path}: {names}")

    return (
        find_ap("ap30", "ap_30"),
        find_ap("ap50", "ap_50"),
        find_ap("ap70", "ap_70"),
    )


def eval_yaml(model_dir: Path, method: str, comp: int, comm: int) -> Path | None:
    note = f"_data_strict_comp{comp}_comm{comm}"
    if method != "none":
        note += f"_{method}_comm{comm}_comp{comp}"
    patterns = [
        f"eval_intermediate{note}_102.4_51.2_epoch*.yaml",
        f"eval_intermediate{note}_102.4_51.2_epoch*.yml",
    ]
    if method == "none" and (comp or comm):
        patterns.extend([
            f"eval_intermediate{note}_comm{comm}_comp{comp}_102.4_51.2_epoch*.yaml",
            f"eval_intermediate{note}_comm{comm}_comp{comp}_102.4_51.2_epoch*.yml",
        ])
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(model_dir.glob(pattern))
    complete = []
    for path in matches:
        try:
            load_ap(path)
        except Exception:
            continue
        complete.append(path)
    if not complete:
        return None
    return sorted(complete, key=lambda p: p.stat().st_mtime)[-1]


def row_from_yaml(method: str, comp: int, comm: int, eval_path: Path, log_path: Path) -> dict[str, object]:
    ap30, ap50, ap70 = load_ap(eval_path)
    return {
        "dataset": "DAIR-V2X-C",
        "method": METHOD_NAMES[method],
        "comp_delay": comp,
        "comm_delay": comm,
        "ap30": f"{ap30:.4f}",
        "ap50": f"{ap50:.4f}",
        "ap70": f"{ap70:.4f}",
        "eval_yaml": str(eval_path),
        "log": str(log_path),
    }


def command_for_case(args: argparse.Namespace, method: str, comp: int, comm: int) -> list[str]:
    note = f"_data_strict_comp{comp}_comm{comm}"
    cmd = [
        "python",
        "opencood/tools/inference.py",
        "--model_dir",
        str(args.model_dir),
        "--fusion_method",
        "intermediate",
        "--range",
        args.range,
        "--comm_delay",
        str(comm),
        "--comp_delay",
        str(comp),
        "--compensation_method",
        method,
        "--note",
        note,
    ]
    return conda_cmd(args.conda_env, cmd)


def log_path_for_case(log_dir: Path, method: str, comp: int, comm: int) -> Path:
    return log_dir / f"data_{method}_comp{comp}_comm{comm}.log"


def start_case(args: argparse.Namespace, method: str, comp: int, comm: int, gpu: int, log_dir: Path) -> subprocess.Popen:
    full_cmd = command_for_case(args, method, comp, comm)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DATA_ROOT}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    log_path = log_path_for_case(log_dir, method, comp, comm)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(" ".join(full_cmd))
        log.write("\n")
    log = log_path.open("a")
    return subprocess.Popen(
        full_cmd,
        cwd=DATA_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )


def run_case(args: argparse.Namespace, method: str, comp: int, comm: int, log_dir: Path) -> dict[str, object]:
    full_cmd = command_for_case(args, method, comp, comm)
    log_path = log_path_for_case(log_dir, method, comp, comm)
    if args.dry_run:
        print(" ".join(full_cmd))
        return {
            "dataset": "DAIR-V2X-C",
            "method": METHOD_NAMES[method],
            "comp_delay": comp,
            "comm_delay": comm,
            "ap30": "",
            "ap50": "",
            "ap70": "",
            "eval_yaml": "",
            "log": str(log_path),
        }

    proc = start_case(args, method, comp, comm, args.gpus[0], log_dir)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"DATA case failed method={method} comp={comp} comm={comm}; see {log_path}")

    path = eval_yaml(args.model_dir, method, comp, comm)
    if path is None:
        raise FileNotFoundError(f"No eval yaml for method={method} comp={comp} comm={comm}")
    return row_from_yaml(method, comp, comm, path, log_path)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rows_from_yaml(args: argparse.Namespace, methods: list[str], comps: list[int], comms: list[int]) -> list[dict[str, object]]:
    rows = []
    for method in methods:
        for comp in comps:
            for comm in comms:
                path = eval_yaml(args.model_dir, method, comp, comm)
                if path is None:
                    continue
                rows.append(row_from_yaml(method, comp, comm, path, log_path_for_case(args.log_dir, method, comp, comm)))
    return rows


def run_missing_parallel(args: argparse.Namespace, missing: list[tuple[str, int, int]]) -> None:
    free_gpus = list(args.gpus)
    active: dict[subprocess.Popen, tuple[str, int, int, int, float]] = {}
    pending = list(missing)
    while pending or active:
        while pending and free_gpus and len(active) < args.jobs:
            method, comp, comm = pending.pop(0)
            gpu = free_gpus.pop(0)
            proc = start_case(args, method, comp, comm, gpu, args.log_dir)
            active[proc] = (method, comp, comm, gpu, time.time())
            print(f"started DATA {method} comp{comp} comm{comm} on gpu{gpu} pid={proc.pid}", flush=True)
        time.sleep(args.poll_seconds)
        done = [proc for proc in active if proc.poll() is not None]
        if not done:
            for method, comp, comm, gpu, started in active.values():
                elapsed = int(time.time() - started)
                print(f"running DATA {method} comp{comp} comm{comm} on gpu{gpu} elapsed={elapsed}s", flush=True)
            continue
        for proc in done:
            method, comp, comm, gpu, started = active.pop(proc)
            free_gpus.append(gpu)
            rc = proc.wait()
            elapsed = int(time.time() - started)
            print(f"finished DATA {method} comp{comp} comm{comm} on gpu{gpu} rc={rc} elapsed={elapsed}s", flush=True)
            if rc != 0:
                raise RuntimeError(
                    f"DATA case failed method={method} comp={comp} comm={comm}; "
                    f"see {log_path_for_case(args.log_dir, method, comp, comm)}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "260615_fill" / "logs" / "data_dair_strict")
    parser.add_argument("--conda-env", default="dair-v2x-openmmlab")
    parser.add_argument("--range", default="102.4,51.2")
    parser.add_argument("--comps", default="0,2,4")
    parser.add_argument("--comms", default="0,1,2,3,4,5")
    parser.add_argument("--methods", default="none,ours")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run only comp=0,comm=0.")
    args = parser.parse_args()
    args.gpus = parse_ints(args.gpus)

    if args.smoke:
        comps, comms = [0], [0]
    else:
        comps, comms = parse_ints(args.comps), parse_ints(args.comms)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    for method in methods:
        if method not in METHOD_NAMES:
            raise ValueError(f"Unknown method={method}; expected one of {sorted(METHOD_NAMES)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        (method, comp, comm)
        for method in methods
        for comp in comps
        for comm in comms
        if args.force or eval_yaml(args.model_dir, method, comp, comm) is None
    ]
    print(f"missing eval yaml: {len(missing)}")
    for method, comp, comm in missing:
        print(f"  DATA {method} comp{comp} comm{comm}")

    if args.dry_run:
        for method, comp, comm in missing:
            run_case(args, method, comp, comm, args.log_dir)
        return

    if args.run_missing and missing:
        if args.jobs > 1:
            run_missing_parallel(args, missing)
        else:
            for method, comp, comm in missing:
                run_case(args, method, comp, comm, args.log_dir)
                rows = rows_from_yaml(args, methods, comps, comms)
                if rows:
                    write_rows(args.out, rows)

    rows = rows_from_yaml(args, methods, comps, comms)
    if rows:
        write_rows(args.out, rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
