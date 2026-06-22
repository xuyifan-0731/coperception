#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
MANIFEST="${MANIFEST:-${ROOT}/datasets/DAIR-V2X/dair_v2x_c_manifest.csv}"
OUT_DIR="${OUT_DIR:-${ROOT}/datasets/DAIR-V2X/raw}"
LOG_DIR="${LOG_DIR:-${ROOT}/results}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-1800}"
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

manifest_complete() {
  while IFS=, read -r _subset _part filename _file_id _required_for; do
    [[ -n "${filename}" ]] || continue
    if [[ ! -s "${OUT_DIR}/${filename}" ]]; then
      return 1
    fi
  done < <(tail -n +2 "${MANIFEST}")
  return 0
}

attempt=1
while true; do
  if manifest_complete; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] all manifest files are present"
    break
  fi

  log="${LOG_DIR}/dair_v2x_dataset_download_until_complete_attempt_${attempt}_$(date -u '+%Y%m%d_%H%M%S').log"
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] download attempt ${attempt}; log=${log}"
  set +e
  bash scripts/download_dair_v2x_c.sh 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  if (( rc == 0 )); then
    if manifest_complete; then
      echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] download completed"
      break
    fi
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] downloader exited 0, but manifest is incomplete"
  else
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] attempt ${attempt} failed with rc=${rc}"
  fi

  attempt=$((attempt + 1))
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] sleeping ${RETRY_SLEEP_SECONDS}s before retry"
  sleep "${RETRY_SLEEP_SECONDS}"
done

if [[ "${RUN_PREPARE}" == "1" ]]; then
  prep_log="${LOG_DIR}/dair_v2x_prepare_after_download_$(date -u '+%Y%m%d_%H%M%S').log"
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] preparing DAIR-V2X layout; log=${prep_log}"
  bash scripts/prepare_dair_v2x_layout.sh 2>&1 | tee "${prep_log}"
fi
