#!/usr/bin/env python3
"""Download DAIR-V2X-C files from Baidu AI Studio with resume support.

AI Studio requires either a logged-in browser cookie or an access token for the
download-link API. The actual large file URL returned by that API is downloaded
with HTTP Range requests into a fixed ``.aistudio.part`` file and then atomically
renamed to the target file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


DATASET_ID = 179509
DETAIL_URL = f"https://aistudio.baidu.com/aistudio/datasetdetail/{DATASET_ID}"
DETAIL_API = "https://aistudio.baidu.com/studio/dataset/detail"
DOWNLOAD_API = "https://aistudio.baidu.com/llm/files/datasets/{dataset_id}/file/{file_id}/download"


@dataclass(frozen=True)
class DairFile:
    file_id: int
    aistudio_name: str
    target_name: str
    expected_size: int
    required_for: str


FILES = [
    DairFile(
        1162206,
        "infrastructure.zip",
        "cooperative-vehicle-infrastructure.zip",
        245642216,
        "labels_calib_splits",
    ),
    DairFile(
        1162254,
        "infrastructure-vehicle-side-image.zip",
        "cooperative-vehicle-infrastructure-vehicle-side-image.zip",
        2714689094,
        "camera_baselines",
    ),
    DairFile(
        1162282,
        "infrastructure-infrastructure-side-image.zip",
        "cooperative-vehicle-infrastructure-infrastructure-side-image-deprecated.zip",
        3915131337,
        "camera_baselines_legacy",
    ),
    DairFile(
        1162356,
        "infrastructure-infrastructure-side-velodyne.zip",
        "cooperative-vehicle-infrastructure-infrastructure-side-velodyne.zip",
        8874327777,
        "lidar_baselines",
    ),
    DairFile(
        1162390,
        "infrastructure-vehicle-side-velodyne.zip",
        "cooperative-vehicle-infrastructure-vehicle-side-velodyne.zip",
        12814964729,
        "lidar_baselines",
    ),
]


def fmt_bytes(num: int | float) -> str:
    num = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024 or unit == "TiB":
            return f"{num:.1f}{unit}" if unit != "B" else f"{int(num)}B"
        num /= 1024
    return f"{num:.1f}TiB"


def read_cookie_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    if "\t" not in text and "=" in text and "\n" not in text:
        return text

    pairs: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            fields = line.split("\t")
            if len(fields) >= 7:
                pairs.append(f"{fields[5]}={fields[6]}")
        elif "=" in line and ";" not in line:
            pairs.append(line)
    return "; ".join(pairs)


def build_session(args: argparse.Namespace) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": args.user_agent,
            "Accept": "application/json, text/plain, */*",
            "Referer": DETAIL_URL,
            "Origin": "https://aistudio.baidu.com",
            "X-Requested-With": "XMLHttpRequest",
            "Cache-Control": "no-cache",
        }
    )

    cookie = args.cookie or os.environ.get("AISTUDIO_COOKIE", "")
    if args.cookie_file:
        cookie = read_cookie_file(Path(args.cookie_file))
    if cookie:
        session.headers["Cookie"] = cookie

    token = args.x_studio_token or os.environ.get("AISTUDIO_X_STUDIO_TOKEN", "")
    if token:
        session.headers["x-studio-token"] = token

    access_token = args.access_token or os.environ.get("AISTUDIO_ACCESS_TOKEN", "")
    if args.access_token_file:
        access_token = Path(args.access_token_file).read_text(encoding="utf-8").strip()
    if access_token:
        session.headers["Authorization"] = f"Bearer {access_token}"
        session.headers["X-AIStudio-Token"] = access_token
    return session


def hydrate_x_studio_token(session: requests.Session) -> None:
    if session.headers.get("x-studio-token"):
        return
    try:
        response = session.get(DETAIL_URL, timeout=30)
    except requests.RequestException:
        return
    match = re.search(r"bdToken:\s*['\"]([^'\"]+)['\"]", response.text)
    if match:
        session.headers["x-studio-token"] = match.group(1)


def validate_detail(session: requests.Session) -> None:
    response = session.post(DETAIL_API, data={"datasetId": str(DATASET_ID)}, timeout=30)
    response.raise_for_status()
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"detail API returned non-JSON response: {response.text[:200]}") from exc
    if body.get("errorCode") != 0:
        raise RuntimeError(f"detail API failed: {body}")
    remote = {
        item["fileId"]: (item["fileOriginName"], item["fileSize"])
        for item in body.get("result", {}).get("fileList", [])
    }
    for item in FILES:
        actual = remote.get(item.file_id)
        if not actual:
            raise RuntimeError(f"AI Studio file id missing from detail response: {item.file_id}")
        if actual[0] != item.aistudio_name or int(actual[1]) != item.expected_size:
            raise RuntimeError(
                "AI Studio file metadata changed for "
                f"{item.file_id}: expected {(item.aistudio_name, item.expected_size)}, got {actual}"
            )


def get_file_url(session: requests.Session, item: DairFile) -> str:
    url = DOWNLOAD_API.format(dataset_id=DATASET_ID, file_id=item.file_id)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"download API returned non-JSON response for {item.aistudio_name}") from exc
    if body.get("errorCode") != 0:
        raise RuntimeError(f"download API failed for {item.aistudio_name}: {body}")
    result = body.get("result")
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("fileUrl", "downloadUrl", "url"):
            value = result.get(key)
            if value:
                return str(value)
    raise RuntimeError(f"download API did not return a file URL for {item.aistudio_name}: {body}")


def download_with_resume(url: str, target: Path, expected_size: int, progress_seconds: int) -> None:
    part = target.with_name(target.name + ".aistudio.part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        if existing > 0 and response.status_code == 200:
            print(
                f"  server ignored Range for {target.name}; restarting from 0 "
                f"in {part}",
                flush=True,
            )
            existing = 0
            mode = "wb"
        elif response.status_code not in (200, 206):
            raise RuntimeError(f"file download HTTP {response.status_code}: {response.text[:200]}")

        downloaded = existing
        started = time.time()
        last_print = 0.0
        part.parent.mkdir(parents=True, exist_ok=True)
        with part.open(mode + ("" if "b" in mode else "b")) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= progress_seconds:
                    elapsed = max(now - started, 1e-6)
                    rate = (downloaded - existing) / elapsed
                    pct = downloaded / expected_size * 100 if expected_size else 0.0
                    print(
                        f"  progress {target.name}: {fmt_bytes(downloaded)}/"
                        f"{fmt_bytes(expected_size)} ({pct:.2f}%), {fmt_bytes(rate)}/s",
                        flush=True,
                    )
                    last_print = now

    actual = part.stat().st_size
    if actual != expected_size:
        raise RuntimeError(
            f"incomplete download for {target.name}: {actual} bytes, expected {expected_size}; "
            f"kept partial at {part}"
        )
    part.replace(target)
    print(f"  complete {target} ({fmt_bytes(expected_size)})", flush=True)


def selected_files(names: Iterable[str]) -> list[DairFile]:
    wanted = set(names)
    if not wanted:
        return FILES
    items = [item for item in FILES if item.target_name in wanted or item.aistudio_name in wanted]
    missing = wanted.difference({item.target_name for item in items}, {item.aistudio_name for item in items})
    if missing:
        raise SystemExit(f"unknown file selection: {', '.join(sorted(missing))}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="/raid/xuyifan/v2x_code_ckpt/datasets/DAIR-V2X/raw",
        help="directory for downloaded DAIR-V2X raw archives",
    )
    parser.add_argument("--cookie", default="", help="raw Cookie header copied from a logged-in AI Studio browser")
    parser.add_argument("--cookie-file", default="", help="file containing a raw Cookie header or Netscape cookies")
    parser.add_argument("--x-studio-token", default="", help="AI Studio x-studio-token/bdToken if available")
    parser.add_argument("--access-token", default="", help="AI Studio access token if available")
    parser.add_argument("--access-token-file", default="", help="file containing an AI Studio access token")
    parser.add_argument("--only", action="append", default=[], help="download only this target or AI Studio filename")
    parser.add_argument("--progress-seconds", type=int, default=30, help="seconds between progress lines")
    parser.add_argument("--skip-detail-check", action="store_true", help="skip AI Studio metadata validation")
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    session = build_session(args)
    hydrate_x_studio_token(session)

    if not args.skip_detail_check:
        validate_detail(session)

    for item in selected_files(args.only):
        target = out_dir / item.target_name
        if target.exists() and target.stat().st_size == item.expected_size:
            print(f"skip existing {target} ({fmt_bytes(item.expected_size)})", flush=True)
            continue
        if target.exists() and target.stat().st_size != item.expected_size:
            raise RuntimeError(
                f"existing file has unexpected size: {target} has {target.stat().st_size}, "
                f"expected {item.expected_size}"
            )
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] "
            f"downloading {item.aistudio_name} -> {target.name} ({item.required_for})",
            flush=True,
        )
        file_url = get_file_url(session, item)
        download_with_resume(file_url, target, item.expected_size, args.progress_seconds)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; partial .aistudio.part files are kept for resume", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
