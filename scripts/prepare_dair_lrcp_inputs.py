#!/usr/bin/env python3
"""Prepare DAIR-V2X files expected by the LRCP OpenCOOD fork.

The LRCP DAIR dataset loader expects a lightly preprocessed DAIR layout:

* cooperative/data_info_processed_updated.json with previous_inf_* fields.
* split JSON files containing vehicle frame ids as a list.
* cooperative/label_world_backup and vehicle-side/label/lidar_backup.
* offset_maps_fix_flip/offset_<veh_id>.npy files.

The public LRCP repository points to a separate DAIR flow-file download, but the
local workspace does not include it. For stage1/smoke execution we create zero
offset maps with the same shape the loader consumes. Stage2 can overwrite these
files with real flow maps before training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAIR_ROOT = ROOT / "datasets" / "DAIR-V2X" / "cooperative-vehicle-infrastructure"
DEFAULT_SPLIT_JSON = ROOT / "external" / "DAIR-V2X" / "data" / "split_datas" / "cooperative-split-data.json"
DEFAULT_OUT_SPLITS = ROOT / "results" / "260615_fill" / "dair_lrcp_splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dair-root", type=Path, default=DEFAULT_DAIR_ROOT)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--out-split-dir", type=Path, default=DEFAULT_OUT_SPLITS)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--bev-h", type=int, default=80)
    parser.add_argument("--bev-w", type=int, default=80)
    parser.add_argument("--copy-labels", action="store_true", help="Copy labels instead of symlinking backup dirs.")
    parser.add_argument("--force-offsets", action="store_true", help="Overwrite existing offset npy files.")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_backup_dir(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        os.symlink(src, dst, target_is_directory=True)
        return
    import shutil

    shutil.copytree(src, dst)


def veh_id(info: dict) -> str:
    return Path(info["vehicle_pointcloud_path"]).stem


def inf_path(info: dict) -> str:
    return info["infrastructure_pointcloud_path"]


def build_processed_info(dair_root: Path, max_history: int) -> list[dict]:
    info_path = dair_root / "cooperative" / "data_info.json"
    rows = read_json(info_path)
    rows = sorted(rows, key=veh_id)
    out: list[dict] = []
    for idx, row in enumerate(rows):
        item = dict(row)
        for delay in range(1, max_history + 1):
            prev_idx = idx - delay
            if prev_idx < 0:
                item[f"previous_inf_{delay}"] = None
                continue
            item[f"previous_inf_{delay}"] = [inf_path(rows[prev_idx])]
        out.append(item)
    return out


def write_split_lists(processed_info: list[dict], split_json: Path, out_dir: Path) -> dict[str, int]:
    available = {veh_id(row) for row in processed_info}
    split = read_json(split_json)
    cooperative_split = split.get("cooperative_split", {})
    counts: dict[str, int] = {}
    for name in ("train", "val", "test"):
        values = [str(item).zfill(6) for item in cooperative_split.get(name, [])]
        values = [item for item in values if item in available]
        if not values and name == "val":
            # Some DAIR split files use validation-like ids not present after
            # zero padding. Fall back to the tail of available frames.
            values = sorted(available)[-max(1, len(available) // 10) :]
        write_json(out_dir / f"{name}.json", values)
        counts[name] = len(values)
    return counts


def write_zero_offsets(dair_root: Path, processed_info: list[dict], max_history: int, bev_h: int, bev_w: int, force: bool) -> int:
    out_dir = dair_root / "offset_maps_fix_flip"
    out_dir.mkdir(parents=True, exist_ok=True)
    zero = np.zeros((max_history, bev_h, bev_w, 2), dtype=np.float32)
    written = 0
    for row in processed_info:
        path = out_dir / f"offset_{veh_id(row)}.npy"
        if path.exists() and not force:
            continue
        np.save(path, zero)
        written += 1
    return written


def main() -> None:
    args = parse_args()
    dair_root = args.dair_root.resolve()
    ensure_backup_dir(
        dair_root / "cooperative" / "label_world",
        dair_root / "cooperative" / "label_world_backup",
        copy=args.copy_labels,
    )
    ensure_backup_dir(
        dair_root / "vehicle-side" / "label" / "lidar",
        dair_root / "vehicle-side" / "label" / "lidar_backup",
        copy=args.copy_labels,
    )
    processed = build_processed_info(dair_root, args.max_history)
    processed_path = dair_root / "cooperative" / "data_info_processed_updated.json"
    write_json(processed_path, processed)
    split_counts = write_split_lists(processed, args.split_json, args.out_split_dir)
    offset_count = write_zero_offsets(
        dair_root,
        processed,
        args.max_history,
        args.bev_h,
        args.bev_w,
        args.force_offsets,
    )
    print(f"processed_info={processed_path} rows={len(processed)}")
    print(f"split_dir={args.out_split_dir} counts={split_counts}")
    print(f"zero_offsets_written={offset_count}")


if __name__ == "__main__":
    main()
