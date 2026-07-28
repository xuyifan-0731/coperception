#!/usr/bin/env python3
"""Smoke-test LRCP DAIR dataset construction and one-item collation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LRCP_ROOT = ROOT / "external" / "LRCP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml",
        default=str(LRCP_ROOT / "opencood" / "hypes_yaml" / "dairv2x" / "pointpillar_deformable_attn_stage1.yaml"),
    )
    parser.add_argument("--eval", action="store_true", help="Build the validation/test side of the dataset.")
    parser.add_argument("--time-delay", type=int, default=None)
    parser.add_argument("--max-scan", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(LRCP_ROOT))

    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils

    hypes = yaml_utils.load_yaml(args.yaml)
    if args.time_delay is not None:
        hypes["time_delay"] = args.time_delay
    dataset = build_dataset(hypes, visualize=False, train=not args.eval)
    print(
        "dataset",
        hypes["fusion"]["dataset"],
        "len",
        len(dataset),
        "time_delay",
        hypes.get("time_delay"),
        "max_cav",
        getattr(dataset, "max_cav", None),
    )

    item = None
    idx_used = None
    for idx in range(min(args.max_scan, len(dataset))):
        item = dataset[idx]
        if item is not None:
            idx_used = idx
            break
    if item is None:
        raise RuntimeError(f"No non-empty item found in first {args.max_scan} samples")

    batch = dataset.collate_batch_train([item])
    print("first_item", idx_used)
    print("batch_keys", sorted(batch.keys()))
    print("ego_keys", sorted(batch["ego"].keys()))
    print("object_mask_sum", int(batch["ego"]["object_bbx_mask"].sum()))


if __name__ == "__main__":
    main()
