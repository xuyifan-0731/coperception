#!/usr/bin/env python3
"""Smoke-test one LRCP DAIR forward/loss/backward step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
LRCP_ROOT = ROOT / "external" / "LRCP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml",
        default=str(LRCP_ROOT / "opencood" / "hypes_yaml" / "dairv2x" / "pointpillar_deformable_attn_stage1.yaml"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(LRCP_ROOT))

    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    hypes = yaml_utils.load_yaml(args.yaml)
    hypes["train_params"]["batch_size"] = args.batch_size
    dataset = build_dataset(hypes, visualize=False, train=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=dataset.collate_batch_train,
        shuffle=False,
        drop_last=False,
    )
    model = train_utils.create_model(hypes)
    criterion = train_utils.create_loss(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for batch_data in loader:
        if batch_data is None or batch_data["ego"]["object_bbx_mask"].sum() == 0:
            continue
        batch_data = train_utils.to_device(batch_data, device)
        batch_data["ego"]["epoch"] = 0
        output = model(batch_data["ego"])
        loss = criterion(output, batch_data["ego"]["label_dict"])
        loss.backward()
        print("device", device, "loss", float(loss.detach().cpu()))
        return
    raise RuntimeError("No usable batch found")


if __name__ == "__main__":
    main()
