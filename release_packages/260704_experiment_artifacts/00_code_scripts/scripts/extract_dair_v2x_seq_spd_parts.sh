#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PARTS_ROOT="${PARTS_ROOT:-${ROOT}/datasets/DAIR-V2X-Seq/hf_mirror/SPD/google_drive_parts}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/datasets/DAIR-V2X-Seq}"
DATA_ROOT="${DATA_ROOT:-${OUT_ROOT}/V2X-Seq-SPD}"
TMP_ROOT="${TMP_ROOT:-${OUT_ROOT}/.extract_tmp}"
MARKER_DIR="${MARKER_DIR:-${OUT_ROOT}/.extract_markers}"

mkdir -p "${DATA_ROOT}" "${TMP_ROOT}" "${MARKER_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"
}

run_once() {
  local marker="$1"
  shift
  if [[ -e "${MARKER_DIR}/${marker}" ]]; then
    log "skip ${marker}"
    return 0
  fi
  log "start ${marker}"
  "$@"
  touch "${MARKER_DIR}/${marker}"
  log "done ${marker}"
}

extract_metadata() {
  local archive="$1"
  unzip -q -n "${archive}" -d "${OUT_ROOT}"
}

extract_flat() {
  local archive="$1"
  local dest="$2"
  mkdir -p "${dest}"
  unzip -q -n -j "${archive}" -d "${dest}"
}

extract_split_flat() {
  local archive="$1"
  local dest="$2"
  local name
  name="$(basename "${archive}" .zip)"
  local unsplit="${TMP_ROOT}/${name}.unsplit.zip"
  rm -f "${unsplit}"
  zip -q -s 0 "${archive}" --out "${unsplit}"
  mkdir -p "${dest}"
  unzip -q -n -j "${unsplit}" -d "${dest}"
  rm -f "${unsplit}"
}

merge_data_info() {
  python - "${PARTS_ROOT}" "${DATA_ROOT}" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

parts = Path(sys.argv[1])
out = Path(sys.argv[2])
archives = [
    ("train_val", parts / "train_val" / "V2X-Seq-SPD.zip"),
    ("test", parts / "test" / "V2X-Seq-SPD.zip"),
]
rels = [
    "vehicle-side/data_info.json",
    "infrastructure-side/data_info.json",
    "cooperative/data_info.json",
]

for rel in rels:
    merged = []
    for split, archive in archives:
        with zipfile.ZipFile(archive) as zf:
            payload = json.loads(zf.read("V2X-Seq-SPD/" + rel))
        split_path = out / rel.replace("data_info.json", f"data_info_{split}.json")
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        merged.extend(payload)
    target = out / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"merged {rel}: {len(merged)} records")
PY
}

validate_layout() {
  python - "${DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = [
    root / "cooperative" / "data_info.json",
    root / "vehicle-side" / "data_info.json",
    root / "infrastructure-side" / "data_info.json",
]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise SystemExit("missing required files: " + "; ".join(missing))

checks = {
    "vehicle-side/image/*.jpg": 17308,
    "vehicle-side/velodyne/*.pcd": 17308,
    "infrastructure-side/image/*.jpg": 16137,
    "infrastructure-side/velodyne/*.pcd": 16137,
}
for pattern, expected in checks.items():
    count = len(list(root.glob(pattern)))
    print(f"{pattern}: {count}")
    if count != expected:
        raise SystemExit(f"unexpected count for {pattern}: {count} != {expected}")

for rel, expected in [
    ("vehicle-side/data_info.json", 17308),
    ("infrastructure-side/data_info.json", 16137),
    ("cooperative/data_info.json", 15371),
]:
    data = json.loads((root / rel).read_text(encoding="utf-8"))
    print(f"{rel}: {len(data)}")
    if len(data) != expected:
        raise SystemExit(f"unexpected data_info count for {rel}: {len(data)} != {expected}")
PY
}

log "extracting SPD parts from ${PARTS_ROOT}"
log "output root ${DATA_ROOT}"

run_once metadata_train_val extract_metadata "${PARTS_ROOT}/train_val/V2X-Seq-SPD.zip"
run_once metadata_test extract_metadata "${PARTS_ROOT}/test/V2X-Seq-SPD.zip"

run_once train_val_infrastructure_image extract_flat \
  "${PARTS_ROOT}/train_val/V2X-Seq-SPD-infrastructure-side-image.zip" \
  "${DATA_ROOT}/infrastructure-side/image"
run_once test_infrastructure_image extract_flat \
  "${PARTS_ROOT}/test/V2X-Seq-SPD-infrastructure-side-image.zip" \
  "${DATA_ROOT}/infrastructure-side/image"

run_once train_val_vehicle_image extract_flat \
  "${PARTS_ROOT}/train_val/V2X-Seq-SPD-vehicle-side-image.zip" \
  "${DATA_ROOT}/vehicle-side/image"
run_once test_vehicle_image extract_flat \
  "${PARTS_ROOT}/test/V2X-Seq-SPD-vehicle-side-image.zip" \
  "${DATA_ROOT}/vehicle-side/image"

run_once train_val_infrastructure_velodyne extract_split_flat \
  "${PARTS_ROOT}/train_val/V2X-Seq-SPD-infrastructure-side-velodyne.zip" \
  "${DATA_ROOT}/infrastructure-side/velodyne"
run_once test_infrastructure_velodyne extract_flat \
  "${PARTS_ROOT}/test/V2X-Seq-SPD-infrastructure-side-velodyne.zip" \
  "${DATA_ROOT}/infrastructure-side/velodyne"

run_once train_val_vehicle_velodyne extract_split_flat \
  "${PARTS_ROOT}/train_val/V2X-Seq-SPD-vehicle-side-velodyne.zip" \
  "${DATA_ROOT}/vehicle-side/velodyne"
run_once test_vehicle_velodyne extract_flat \
  "${PARTS_ROOT}/test/V2X-Seq-SPD-vehicle-side-velodyne.zip" \
  "${DATA_ROOT}/vehicle-side/velodyne"

run_once data_info_merged merge_data_info
run_once layout_validated validate_layout

log "SPD extraction complete"
