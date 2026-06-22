#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/datasets/DAIR-V2X}"
RAW_DIR="${RAW_DIR:-${DATASET_DIR}/raw}"
OPENDAIR_ROOT="${OPENDAIR_ROOT:-${ROOT}/external/DAIR-V2X}"

mkdir -p "${RAW_DIR}"
mkdir -p "${OPENDAIR_ROOT}/data/DAIR-V2X"

extract_if_present() {
  local archive="$1"
  local out_dir="$2"
  local expected_rel="${3:-}"
  if [[ ! -s "${archive}" ]]; then
    return 0
  fi
  if [[ -n "${expected_rel}" && -e "${out_dir}/${expected_rel}" ]]; then
    echo "skip extraction; already present: ${out_dir}/${expected_rel}"
    return 0
  fi
  mkdir -p "${out_dir}"
  echo "extracting ${archive} -> ${out_dir}"
  if command -v 7z >/dev/null 2>&1; then
    7z x -y "-o${out_dir}" "${archive}"
  else
    unzip -o "${archive}" -d "${out_dir}"
  fi
}

link_if_needed() {
  local target="$1"
  local link_path="$2"
  if [[ -e "${link_path}" ]]; then
    return 0
  fi
  if [[ -e "${target}" ]]; then
    ln -sfn "${target}" "${link_path}"
    echo "linked ${link_path} -> ${target}"
  fi
}

normalize_aistudio_layout() {
  local data_root="$1"
  link_if_needed \
    "${data_root}/vehicle-side/cooperative-vehicle-infrastructure-vehicle-side-image" \
    "${data_root}/vehicle-side/image"
  link_if_needed \
    "${data_root}/vehicle-side/cooperative-vehicle-infrastructure-vehicle-side-velodyne" \
    "${data_root}/vehicle-side/velodyne"
  link_if_needed \
    "${data_root}/infrastructure-side/cooperative-vehicle-infrastructure-infrastructure-side-image" \
    "${data_root}/infrastructure-side/image"
  link_if_needed \
    "${data_root}/infrastructure-side/cooperative-vehicle-infrastructure-infrastructure-side-velodyne" \
    "${data_root}/infrastructure-side/velodyne"
}

if [[ ! -d "${DATASET_DIR}/cooperative-vehicle-infrastructure" ]]; then
  extract_if_present "${RAW_DIR}/cooperative-vehicle-infrastructure.zip" "${DATASET_DIR}"
fi

if [[ -d "${DATASET_DIR}/cooperative-vehicle-infrastructure" ]]; then
  extract_if_present "${RAW_DIR}/cooperative-vehicle-infrastructure-vehicle-side-image.zip" "${DATASET_DIR}/cooperative-vehicle-infrastructure/vehicle-side" "cooperative-vehicle-infrastructure-vehicle-side-image"
  extract_if_present "${RAW_DIR}/cooperative-vehicle-infrastructure-infrastructure-side-image-deprecated.zip" "${DATASET_DIR}/cooperative-vehicle-infrastructure/infrastructure-side" "cooperative-vehicle-infrastructure-infrastructure-side-image"
  extract_if_present "${RAW_DIR}/cooperative-vehicle-infrastructure-vehicle-side-velodyne.zip" "${DATASET_DIR}/cooperative-vehicle-infrastructure/vehicle-side" "cooperative-vehicle-infrastructure-vehicle-side-velodyne"
  extract_if_present "${RAW_DIR}/cooperative-vehicle-infrastructure-infrastructure-side-velodyne.zip" "${DATASET_DIR}/cooperative-vehicle-infrastructure/infrastructure-side" "cooperative-vehicle-infrastructure-infrastructure-side-velodyne"
  normalize_aistudio_layout "${DATASET_DIR}/cooperative-vehicle-infrastructure"
fi

DATA_ROOT="${DATA_ROOT:-${DATASET_DIR}/cooperative-vehicle-infrastructure}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "missing DAIR-V2X-C root: ${DATA_ROOT}" >&2
  echo "download archives into ${RAW_DIR}, then re-run this script" >&2
  exit 2
fi

required=(
  "infrastructure-side"
  "vehicle-side"
  "cooperative"
  "vehicle-side/data_info.json"
  "infrastructure-side/data_info.json"
  "cooperative/data_info.json"
)

for rel in "${required[@]}"; do
  if [[ ! -e "${DATA_ROOT}/${rel}" ]]; then
    echo "missing required DAIR path: ${DATA_ROOT}/${rel}" >&2
    exit 3
  fi
done

ln -sfn "${DATA_ROOT}" "${OPENDAIR_ROOT}/data/DAIR-V2X/cooperative-vehicle-infrastructure"

echo "DAIR-V2X-C root: ${DATA_ROOT}"
echo "OpenDAIRV2X symlink: ${OPENDAIR_ROOT}/data/DAIR-V2X/cooperative-vehicle-infrastructure"
