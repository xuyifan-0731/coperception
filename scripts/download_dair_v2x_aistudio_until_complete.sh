#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
OUT_DIR="${OUT_DIR:-${ROOT}/datasets/DAIR-V2X/raw}"
LOG_DIR="${LOG_DIR:-${ROOT}/results}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-600}"
RUN_PREPARE="${RUN_PREPARE:-0}"
USE_LOCAL_PROXY="${USE_LOCAL_PROXY:-1}"
PROXY_URL="${PROXY_URL:-http://localhost:15002}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

if [[ "${USE_LOCAL_PROXY}" == "1" ]]; then
  export HTTP_PROXY="${HTTP_PROXY:-${PROXY_URL}}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${PROXY_URL}}"
  export http_proxy="${http_proxy:-${HTTP_PROXY}}"
  export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
fi

attempt=1
while true; do
  log="${LOG_DIR}/dair_v2x_aistudio_download_attempt_${attempt}_$(date -u '+%Y%m%d_%H%M%S').log"
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] AI Studio download attempt ${attempt}; log=${log}"
  set +e
  python scripts/download_dair_v2x_aistudio.py --out-dir "${OUT_DIR}" 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  if (( rc == 0 )); then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] AI Studio download completed"
    break
  fi
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] attempt ${attempt} failed with rc=${rc}"
  attempt=$((attempt + 1))
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] sleeping ${RETRY_SLEEP_SECONDS}s before retry"
  sleep "${RETRY_SLEEP_SECONDS}"
done

if [[ "${RUN_PREPARE}" == "1" ]]; then
  prep_log="${LOG_DIR}/dair_v2x_prepare_after_aistudio_download_$(date -u '+%Y%m%d_%H%M%S').log"
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] preparing DAIR-V2X layout; log=${prep_log}"
  bash scripts/prepare_dair_v2x_layout.sh 2>&1 | tee "${prep_log}"
fi
