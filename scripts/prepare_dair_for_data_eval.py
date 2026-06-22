#!/usr/bin/env python3
"""Prepare local DAIR-V2X-C tree for the DATA/OpenCOOD-style loader.

The DATA repository expects short directory names such as
``vehicle-side/image`` and split files named ``train.json``/``val.json``.
The local DAIR download keeps the original long directory names and stores
split definitions in the official DAIR-V2X repository. This script creates
only symlinks and split JSON files; it does not copy or modify point clouds.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def link_or_keep(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    rel_target = os.path.relpath(target, link_path.parent)
    link_path.symlink_to(rel_target)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dair-root",
        default="datasets/DAIR-V2X/cooperative-vehicle-infrastructure",
        help="Local DAIR-V2X-C cooperative-vehicle-infrastructure directory.",
    )
    parser.add_argument(
        "--split-json",
        default="external/DAIR-V2X/data/split_datas/cooperative-split-data.json",
        help="Official DAIR cooperative split JSON.",
    )
    args = parser.parse_args()

    root = Path(args.dair_root).resolve()
    split_json = Path(args.split_json).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if not split_json.exists():
        raise FileNotFoundError(split_json)

    link_or_keep(
        root / "vehicle-side" / "image",
        root / "vehicle-side" / "cooperative-vehicle-infrastructure-vehicle-side-image",
    )
    link_or_keep(
        root / "vehicle-side" / "velodyne",
        root / "vehicle-side" / "cooperative-vehicle-infrastructure-vehicle-side-velodyne",
    )
    link_or_keep(
        root / "infrastructure-side" / "image",
        root / "infrastructure-side" / "cooperative-vehicle-infrastructure-infrastructure-side-image",
    )
    link_or_keep(
        root / "infrastructure-side" / "velodyne",
        root / "infrastructure-side" / "cooperative-vehicle-infrastructure-infrastructure-side-velodyne",
    )

    # DATA's loader asks for complemented labels. If the local dataset does not
    # include those folders, reuse the available labels so the loader can run.
    link_or_keep(root / "cooperative" / "label_world_complemented", root / "cooperative" / "label_world")
    link_or_keep(root / "vehicle-side" / "label" / "lidar_complemented", root / "vehicle-side" / "label" / "lidar")

    split = json.loads(split_json.read_text())
    co_data = json.loads((root / "cooperative" / "data_info.json").read_text())
    available_vehicle_ids = {
        Path(item["vehicle_image_path"]).stem
        for item in co_data
        if "vehicle_image_path" in item
    }

    vehicle_split = split["vehicle_split"]
    outputs = {
        "train.json": [x for x in vehicle_split["train"] if x in available_vehicle_ids],
        "val.json": [x for x in vehicle_split["val"] if x in available_vehicle_ids],
        "test.json": [x for x in vehicle_split.get("test", []) if x in available_vehicle_ids],
        "test_A.json": [x for x in vehicle_split.get("test_A", []) if x in available_vehicle_ids],
    }
    for name, payload in outputs.items():
        write_json(root / name, payload)
        print(f"{root / name}: {len(payload)} frames")


if __name__ == "__main__":
    main()
