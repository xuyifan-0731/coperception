#!/usr/bin/env python3
"""List or download the official V2X-Seq-SPD files from the public Drive folder.

The DAIR-V2X README points the full DAIR-V2X/V2X-Seq release to a shared
Google Drive folder.  This script first enumerates that folder with gdown's
``skip_download`` mode, then filters for SPD files so we do not accidentally
download the whole public dataset bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "datasets" / "DAIR-V2X-Seq"
DEFAULT_STATUS = ROOT / "results" / "dair_v2x_seq" / "download_status.json"
DEFAULT_FOLDER_URL = "https://drive.google.com/drive/folders/1gnrw5llXAIxuB9sEKKCm6xTaJ5HQAw2e?usp=sharing"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def import_gdown():
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "gdown is required. In this workspace it is available in the "
            "`Android-Lab` conda env: `conda activate Android-Lab`."
        ) from exc
    return gdown


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": utc_now(), **payload}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def infer_proxy(args: argparse.Namespace) -> str | None:
    if args.proxy:
        return args.proxy
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def list_drive_files(args: argparse.Namespace):
    gdown = import_gdown()
    return gdown.download_folder(
        url=args.folder_url,
        output=str(args.out_dir),
        quiet=args.quiet,
        proxy=infer_proxy(args),
        use_cookies=not args.no_cookies,
        verify=not args.no_check_certificate,
        skip_download=True,
    )


def select_spd_files(files, pattern: str, include_example: bool):
    regex = re.compile(pattern, re.IGNORECASE)
    selected = []
    for item in files:
        rel = getattr(item, "path", "")
        if not regex.search(rel):
            continue
        if not include_example and re.search(r"example", rel, re.IGNORECASE):
            continue
        selected.append(item)
    return selected


def download_selected(args: argparse.Namespace, files) -> list[str]:
    gdown = import_gdown()
    downloaded: list[str] = []
    root = args.out_dir
    root.mkdir(parents=True, exist_ok=True)
    for item in files:
        rel = Path(item.path)
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {rel} -> {target}", flush=True)
        out = gdown.download(
            url="https://drive.google.com/uc?id=" + item.id,
            output=str(target),
            quiet=args.quiet,
            proxy=infer_proxy(args),
            use_cookies=not args.no_cookies,
            verify=not args.no_check_certificate,
            resume=True,
        )
        downloaded.append(str(out))
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-url", default=DEFAULT_FOLDER_URL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--match", default=r"V2X-Seq-SPD|SPD")
    parser.add_argument("--include-example", action="store_true")
    parser.add_argument("--download", action="store_true", help="Download selected files after listing.")
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--no-cookies", action="store_true")
    parser.add_argument("--no-check-certificate", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        files = list_drive_files(args)
        selected = select_spd_files(files, args.match, args.include_example)
        payload = {
            "folder_url": args.folder_url,
            "out_dir": str(args.out_dir),
            "proxy": infer_proxy(args),
            "listed_files": [{"id": f.id, "path": f.path, "local_path": f.local_path} for f in files],
            "selected_files": [{"id": f.id, "path": f.path, "local_path": f.local_path} for f in selected],
            "downloaded": [],
            "error": None,
        }
        if args.download:
            if not selected:
                raise RuntimeError(
                    f"No files matched {args.match!r}. Use --include-example only for the small example set."
                )
            payload["downloaded"] = download_selected(args, selected)
        write_status(args.status, payload)
        print(f"selected {len(selected)} of {len(files)} listed files; status={args.status}", flush=True)
        if selected:
            for item in selected:
                print(f"  {item.path}  id={item.id}", flush=True)
    except Exception as exc:
        write_status(
            args.status,
            {
                "folder_url": args.folder_url,
                "out_dir": str(args.out_dir),
                "proxy": infer_proxy(args),
                "listed_files": [],
                "selected_files": [],
                "downloaded": [],
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        print(f"download/list failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
