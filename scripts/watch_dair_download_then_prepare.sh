#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
RAW_DIR="${RAW_DIR:-${ROOT}/datasets/DAIR-V2X/raw}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
RUN_OFFICIAL_BASELINES="${RUN_OFFICIAL_BASELINES:-0}"
ENV_NAME="${ENV_NAME:-dair-v2x-openmmlab}"
DEVICE_ID="${DEVICE_ID:-0}"
K_LIST="${K_LIST:-0 1 2}"

LOG_DIR="${ROOT}/results/dair_v2x/post_download"
mkdir -p "${LOG_DIR}"

required_files=(
  "cooperative-vehicle-infrastructure.zip"
  "cooperative-vehicle-infrastructure-vehicle-side-image.zip"
  "cooperative-vehicle-infrastructure-infrastructure-side-image-deprecated.zip"
  "cooperative-vehicle-infrastructure-infrastructure-side-velodyne.zip"
  "cooperative-vehicle-infrastructure-vehicle-side-velodyne.zip"
)

timestamp() {
  date -u '+%Y-%m-%d %H:%M:%S UTC'
}

downloads_complete() {
  local rel path
  for rel in "${required_files[@]}"; do
    path="${RAW_DIR}/${rel}"
    if [[ ! -s "${path}" ]]; then
      return 1
    fi
    if [[ -e "${path}.aistudio.part" ]]; then
      return 1
    fi
  done
  return 0
}

echo "[$(timestamp)] waiting for DAIR-V2X AI Studio files in ${RAW_DIR}"
while ! downloads_complete; do
  echo "[$(timestamp)] not complete yet; sleeping ${SLEEP_SECONDS}s"
  sleep "${SLEEP_SECONDS}"
done

echo "[$(timestamp)] download files complete; running layout preparation"
layout_log="${LOG_DIR}/prepare_layout_$(date -u +%Y%m%d_%H%M%S).log"
(
  cd "${ROOT}"
  bash scripts/prepare_dair_v2x_layout.sh
) 2>&1 | tee "${layout_log}"

echo "[$(timestamp)] running prerequisite check"
check_log="${LOG_DIR}/check_prereqs_$(date -u +%Y%m%d_%H%M%S).log"
(
  cd "${ROOT}"
  bash scripts/check_dair_v2x_prereqs.sh
) 2>&1 | tee "${check_log}"

if [[ "${RUN_OFFICIAL_BASELINES}" == "1" ]]; then
  echo "[$(timestamp)] running official DAIR baselines"
  baseline_log="${LOG_DIR}/official_baselines_$(date -u +%Y%m%d_%H%M%S).log"
  (
    cd "${ROOT}"
    ENV_NAME="${ENV_NAME}" DEVICE_ID="${DEVICE_ID}" K_LIST="${K_LIST}" STRICT_CHECKPOINTS=0 \
      bash scripts/run_dair_v2x_official_baselines.sh
  ) 2>&1 | tee "${baseline_log}"
else
  echo "[$(timestamp)] RUN_OFFICIAL_BASELINES=0; stopping after layout and prereq check"
fi
