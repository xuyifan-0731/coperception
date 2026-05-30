#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
OUT_DIR="${OUT_DIR:-${ROOT}/datasets/DAIR-V2X/raw}"
LOG_DIR="${LOG_DIR:-${ROOT}/results}"
TOKEN_FILE="${TOKEN_FILE:-${LOG_DIR}/.aistudio_token}"
PIDFILE="${PIDFILE:-${LOG_DIR}/dair_v2x_aistudio_download.pid}"
KEEPALIVE_PIDFILE="${KEEPALIVE_PIDFILE:-${LOG_DIR}/dair_v2x_aistudio_keepalive.pid}"
KEEPALIVE_LOG="${KEEPALIVE_LOG:-${LOG_DIR}/dair_v2x_aistudio_keepalive.log}"
CHILD_LOG="${CHILD_LOG:-${LOG_DIR}/dair_v2x_aistudio_download_current.log}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
STALL_SECONDS="${STALL_SECONDS:-900}"
PROGRESS_SECONDS="${PROGRESS_SECONDS:-30}"
USE_LOCAL_PROXY="${USE_LOCAL_PROXY:-1}"
PROXY_URL="${PROXY_URL:-http://localhost:15002}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

log_msg() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "${KEEPALIVE_LOG}"
}

have_token() {
  [[ -s "${TOKEN_FILE}" ]]
}

is_pid_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1
}

child_pid() {
  [[ -f "${PIDFILE}" ]] && cat "${PIDFILE}" || true
}

is_download_complete() {
  python - <<'PY'
from pathlib import Path
root = Path("datasets/DAIR-V2X/raw")
required = {
    "cooperative-vehicle-infrastructure.zip": 245642216,
    "cooperative-vehicle-infrastructure-vehicle-side-image.zip": 2714689094,
    "cooperative-vehicle-infrastructure-infrastructure-side-image-deprecated.zip": 3915131337,
    "cooperative-vehicle-infrastructure-infrastructure-side-velodyne.zip": 8874327777,
    "cooperative-vehicle-infrastructure-vehicle-side-velodyne.zip": 12814964729,
}
for name, size in required.items():
    path = root / name
    if not path.exists() or path.stat().st_size != size:
        raise SystemExit(1)
PY
}

latest_partial_mtime() {
  find "${OUT_DIR}" -maxdepth 1 -type f \( -name '*.aistudio.part' -o -name '*.part' \) \
    -printf '%T@\n' 2>/dev/null | sort -nr | awk 'NR==1 {printf "%.0f", $1}'
}

latest_partial_size() {
  find "${OUT_DIR}" -maxdepth 1 -type f \( -name '*.aistudio.part' -o -name '*.part' \) \
    -printf '%s\n' 2>/dev/null | awk '{s += $1} END {print s + 0}'
}

start_child() {
  if ! have_token; then
    log_msg "missing token file: ${TOKEN_FILE}"
    return 1
  fi

  local env_args=()
  if [[ "${USE_LOCAL_PROXY}" == "1" ]]; then
    env_args+=("HTTP_PROXY=${HTTP_PROXY:-${PROXY_URL}}")
    env_args+=("HTTPS_PROXY=${HTTPS_PROXY:-${PROXY_URL}}")
    env_args+=("http_proxy=${http_proxy:-${HTTP_PROXY:-${PROXY_URL}}}")
    env_args+=("https_proxy=${https_proxy:-${HTTPS_PROXY:-${PROXY_URL}}}")
  fi

  log_msg "starting AI Studio downloader; child log=${CHILD_LOG}"
  env "${env_args[@]}" python scripts/download_dair_v2x_aistudio.py \
    --out-dir "${OUT_DIR}" \
    --access-token-file "${TOKEN_FILE}" \
    --progress-seconds "${PROGRESS_SECONDS}" \
    >> "${CHILD_LOG}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${PIDFILE}"
  log_msg "started downloader pid=${pid}"
}

stop_child() {
  local pid
  pid="$(child_pid)"
  if is_pid_running "${pid}"; then
    log_msg "stopping downloader pid=${pid}"
    kill "${pid}" || true
  fi
}

main() {
  echo "$$" > "${KEEPALIVE_PIDFILE}"
  trap 'log_msg "keepalive exiting"; stop_child' INT TERM

  local last_size current_size last_mtime now pid
  last_size="$(latest_partial_size)"
  log_msg "keepalive started; interval=${CHECK_INTERVAL_SECONDS}s stall=${STALL_SECONDS}s"

  while true; do
    if is_download_complete; then
      log_msg "all AI Studio mirror files are complete"
      stop_child
      exit 0
    fi

    pid="$(child_pid)"
    if ! is_pid_running "${pid}"; then
      log_msg "downloader is not running; restarting"
      start_child || true
      last_size="$(latest_partial_size)"
      sleep "${CHECK_INTERVAL_SECONDS}"
      continue
    fi

    current_size="$(latest_partial_size)"
    last_mtime="$(latest_partial_mtime)"
    now="$(date +%s)"
    if [[ -n "${last_mtime}" && $((now - last_mtime)) -gt "${STALL_SECONDS}" ]]; then
      log_msg "no partial-file writes for $((now - last_mtime))s; restarting downloader pid=${pid}"
      kill "${pid}" || true
      sleep 5
      start_child || true
      last_size="$(latest_partial_size)"
      sleep "${CHECK_INTERVAL_SECONDS}"
      continue
    fi

    if [[ "${current_size}" != "${last_size}" ]]; then
      log_msg "partial bytes changed: ${last_size} -> ${current_size}"
      last_size="${current_size}"
    else
      log_msg "downloader pid=${pid} alive; partial bytes=${current_size}"
    fi

    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

main "$@"
