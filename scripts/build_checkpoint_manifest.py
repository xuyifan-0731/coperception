#!/usr/bin/env python3
"""Build a manifest for local checkpoints referenced by result summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_checkpoint(log_source: Path, step: str) -> Path | None:
    ckpt_dir = log_source.parent
    candidates = [
        ckpt_dir / f"ckpt_{step}.pth",
        ckpt_dir / f"ckpt_{step}_h.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/summary/training_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/summary/checkpoint_manifest.csv"))
    args = parser.parse_args()

    rows = []
    with args.summary.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for label, step_key in (("best", "best_step"), ("final", "last_step")):
                step = row.get(step_key, "")
                if not step:
                    continue
                ckpt = find_checkpoint(Path(row["source"]), step)
                rows.append(
                    {
                        "experiment_group": row["experiment_group"],
                        "run": row["run"],
                        "kind": label,
                        "step": step,
                        "checkpoint": str(ckpt) if ckpt else "",
                        "exists": bool(ckpt),
                        "bytes": ckpt.stat().st_size if ckpt else "",
                        "sha256": sha256(ckpt) if ckpt else "",
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_group",
                "run",
                "kind",
                "step",
                "checkpoint",
                "exists",
                "bytes",
                "sha256",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} checkpoint references to {args.out}")


if __name__ == "__main__":
    main()
