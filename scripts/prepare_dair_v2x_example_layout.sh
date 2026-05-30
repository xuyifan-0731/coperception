#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/datasets/DAIR-V2X}"
RAW_DIR="${RAW_DIR:-${DATASET_DIR}/raw}"
OPENDAIR_ROOT="${OPENDAIR_ROOT:-${ROOT}/external/DAIR-V2X}"
EXAMPLE_ZIP="${EXAMPLE_ZIP:-${RAW_DIR}/DAIR-V2X-C-Example.zip}"
EXAMPLE_ROOT="${EXAMPLE_ROOT:-${DATASET_DIR}/example-cooperative-vehicle-infrastructure}"

if [[ ! -s "${EXAMPLE_ZIP}" ]]; then
  echo "missing example zip: ${EXAMPLE_ZIP}" >&2
  exit 2
fi

if [[ ! -d "${EXAMPLE_ROOT}" ]]; then
  echo "extracting ${EXAMPLE_ZIP} -> ${DATASET_DIR}"
  unzip -o "${EXAMPLE_ZIP}" -d "${DATASET_DIR}"
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
  if [[ ! -e "${EXAMPLE_ROOT}/${rel}" ]]; then
    echo "missing required example DAIR path: ${EXAMPLE_ROOT}/${rel}" >&2
    exit 3
  fi
done

mkdir -p "${OPENDAIR_ROOT}/data/DAIR-V2X"
ln -sfn "${EXAMPLE_ROOT}" "${OPENDAIR_ROOT}/data/DAIR-V2X/example-cooperative-vehicle-infrastructure"

echo "DAIR-V2X-C example root: ${EXAMPLE_ROOT}"
echo "OpenDAIRV2X example symlink: ${OPENDAIR_ROOT}/data/DAIR-V2X/example-cooperative-vehicle-infrastructure"
