#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INTERVAL="${INTERVAL:-600}"
STALL_LIMIT="${STALL_LIMIT:-2}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"

DOWNLOAD_SCRIPT="${DOWNLOAD_SCRIPT:-${ROOT}/scripts/download_dair_v2x_seq_hf_mirror_parts.sh}"
OUT_DIR="${OUT_DIR:-${ROOT}/datasets/DAIR-V2X-Seq/hf_mirror}"
LIST="${LIST:-${ROOT}/results/dair_v2x_seq/hf_spd_parts_files.txt}"
PIDFILE="${PIDFILE:-${ROOT}/results/dair_v2x_seq/hf_spd_parts_curl.pid}"
MONITOR_LOG="${MONITOR_LOG:-${ROOT}/results/dair_v2x_seq/hf_spd_parts_monitor.log}"

mkdir -p "$(dirname "${MONITOR_LOG}")" "${OUT_DIR}"

stamp() {
  date -u '+%Y-%m-%d %H:%M:%S UTC'
}

log() {
  printf '[%s] %s\n' "$(stamp)" "$*" | tee -a "${MONITOR_LOG}"
}

running_pid() {
  local pid=""
  if [[ -s "${PIDFILE}" ]]; then
    pid="$(cat "${PIDFILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
      printf '%s\n' "${pid}"
      return 0
    fi
  fi
  pgrep -f "bash ${DOWNLOAD_SCRIPT}" | head -1 || true
}

start_downloader() {
  local ts log_file pid
  ts="$(date -u +%Y%m%d_%H%M%S)"
  log_file="${ROOT}/results/dair_v2x_seq/hf_spd_parts_curl_${ts}.log"
  setsid env \
    -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
    bash "${DOWNLOAD_SCRIPT}" >"${log_file}" 2>&1 < /dev/null &
  pid="$!"
  echo "${pid}" > "${PIDFILE}"
  log "started downloader pid=${pid} log=${log_file}"
}

stop_downloader() {
  local pid="$1"
  if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
    log "stopping stalled downloader pid=${pid}"
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    sleep 5
    if ps -p "${pid}" >/dev/null 2>&1; then
      kill -9 -- "-${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
    fi
  fi
}

progress_state() {
  find "${OUT_DIR}/SPD/google_drive_parts" -type f \
    \( -name '*.zip' -o -name '*.z01' -o -name '*.part' \) \
    -printf '%P %s\n' 2>/dev/null | sort
}

progress_summary() {
  local total complete partial bytes free_gb
  total="$(grep -cve '^[[:space:]]*$' "${LIST}" 2>/dev/null || echo 0)"
  complete=0
  while IFS= read -r rel; do
    [[ -n "${rel}" ]] || continue
    [[ -s "${OUT_DIR}/${rel}" ]] && complete=$((complete + 1))
  done < "${LIST}"
  partial="$(find "${OUT_DIR}/SPD/google_drive_parts" -type f -name '*.part' 2>/dev/null | wc -l | tr -d ' ')"
  bytes="$(find "${OUT_DIR}/SPD/google_drive_parts" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')"
  free_gb="$(df -BG "${OUT_DIR}" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"
  printf 'complete=%s/%s partial=%s bytes=%s free_gb=%s' "${complete}" "${total}" "${partial}" "${bytes}" "${free_gb}"
}

all_complete() {
  while IFS= read -r rel; do
    [[ -n "${rel}" ]] || continue
    [[ -s "${OUT_DIR}/${rel}" ]] || return 1
  done < "${LIST}"
  return 0
}

last_state=""
stalled_checks=0
log "monitor started interval=${INTERVAL}s stall_limit=${STALL_LIMIT} min_free_gb=${MIN_FREE_GB}"

while true; do
  summary="$(progress_summary)"
  log "progress ${summary}"

  free_gb="$(printf '%s\n' "${summary}" | sed -n 's/.*free_gb=\([0-9][0-9]*\).*/\1/p')"
  if [[ -n "${free_gb}" ]] && (( free_gb < MIN_FREE_GB )); then
    log "ERROR low disk space: free_gb=${free_gb}; stopping monitor"
    exit 2
  fi

  if all_complete; then
    log "all listed SPD parts are present; monitor exiting"
    exit 0
  fi

  pid="$(running_pid)"
  if [[ -z "${pid}" ]]; then
    log "downloader is not running; restarting"
    start_downloader
    stalled_checks=0
  else
    log "downloader running pid=${pid}"
  fi

  state="$(progress_state)"
  if [[ "${state}" == "${last_state}" ]]; then
    stalled_checks=$((stalled_checks + 1))
    log "no file-size progress detected; stalled_checks=${stalled_checks}/${STALL_LIMIT}"
  else
    stalled_checks=0
    last_state="${state}"
  fi

  if (( stalled_checks >= STALL_LIMIT )); then
    pid="$(running_pid)"
    stop_downloader "${pid}"
    start_downloader
    stalled_checks=0
    last_state="$(progress_state)"
  fi

  sleep "${INTERVAL}"
done
