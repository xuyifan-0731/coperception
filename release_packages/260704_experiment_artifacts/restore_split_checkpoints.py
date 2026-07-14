#!/usr/bin/env python3
"""Restore checkpoint files split for GitHub upload.

Run from this package root:
    python restore_split_checkpoints.py
"""
from pathlib import Path
import csv, hashlib, sys

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'CHECKPOINT_SPLIT_MANIFEST.csv'

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    if not MANIFEST.exists():
        print(f'Missing {MANIFEST}', file=sys.stderr)
        return 1
    rows = list(csv.DictReader(MANIFEST.open(newline='', encoding='utf-8')))
    for row in rows:
        out = ROOT / row['original_package_path']
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('wb') as w:
            for rel in row['part_paths'].split('|'):
                part = ROOT / rel
                if not part.exists():
                    print(f'Missing split part: {rel}', file=sys.stderr)
                    return 1
                with part.open('rb') as r:
                    for block in iter(lambda: r.read(1024 * 1024), b''):
                        w.write(block)
        size = out.stat().st_size
        digest = sha256(out)
        if size != int(row['original_size_bytes']) or digest != row['original_sha256']:
            print(f'Checksum mismatch for {out}', file=sys.stderr)
            return 1
        print(f'Restored {row["original_package_path"]}')
    print(f'Restored {len(rows)} checkpoint files.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
