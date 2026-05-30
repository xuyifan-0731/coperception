#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
MANIFEST="${MANIFEST:-${ROOT}/datasets/DAIR-V2X/dair_v2x_c_manifest.csv}"
OUT_DIR="${OUT_DIR:-${ROOT}/datasets/DAIR-V2X/raw}"
LOG_DIR="${LOG_DIR:-${ROOT}/results}"
PIDFILE="${PIDFILE:-${LOG_DIR}/dair_v2x_dataset_download_until_complete_detached.pid}"
AISTUDIO_KEEPALIVE_PIDFILE="${AISTUDIO_KEEPALIVE_PIDFILE:-${LOG_DIR}/dair_v2x_aistudio_keepalive.pid}"
AISTUDIO_DOWNLOAD_PIDFILE="${AISTUDIO_DOWNLOAD_PIDFILE:-${LOG_DIR}/dair_v2x_aistudio_download.pid}"

WATCH_SECONDS=0
TAIL_LINES=30

usage() {
  cat <<'EOF'
Usage: bash scripts/monitor_dair_v2x_download.sh [--watch [SECONDS]] [--tail LINES]

Read-only DAIR-V2X download monitor. It does not stop, restart, or signal the
download process.

Options:
  --watch [SECONDS]  Refresh repeatedly. Default refresh interval is 30s.
  --tail LINES       Show this many lines from the latest logs. Default: 30.
  --once             Print one snapshot and exit. This is the default.
  -h, --help         Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --watch)
      WATCH_SECONDS="${2:-30}"
      if [[ "${WATCH_SECONDS}" =~ ^-- ]] || [[ -z "${WATCH_SECONDS}" ]]; then
        WATCH_SECONDS=30
      else
        shift
      fi
      ;;
    --tail)
      TAIL_LINES="${2:?--tail requires a line count}"
      shift
      ;;
    --once)
      WATCH_SECONDS=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

human_size() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    du -h "${path}" 2>/dev/null | awk '{print $1}'
  else
    printf '-'
  fi
}

latest_matching_log() {
  local pattern="$1"
  find "${LOG_DIR}" -maxdepth 1 -type f -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR==1 {print $2}'
}

print_process_status() {
  echo "Process"
  if [[ -f "${PIDFILE}" ]]; then
    local pid
    pid="$(cat "${PIDFILE}")"
    if ps -p "${pid}" >/dev/null 2>&1; then
      ps -p "${pid}" -o pid,ppid,stat,etime,cmd
    else
      echo "  pidfile exists but process is not running: ${PIDFILE} -> ${pid}"
    fi
  else
    echo "  no pidfile: ${PIDFILE}"
  fi

  if [[ -f "${AISTUDIO_KEEPALIVE_PIDFILE}" ]]; then
    local keepalive_pid
    keepalive_pid="$(cat "${AISTUDIO_KEEPALIVE_PIDFILE}")"
    if ps -p "${keepalive_pid}" >/dev/null 2>&1; then
      echo
      echo "AI Studio keepalive"
      ps -p "${keepalive_pid}" -o pid,ppid,stat,etime,cmd
    else
      echo "  AI Studio keepalive pidfile exists but process is not running: ${AISTUDIO_KEEPALIVE_PIDFILE} -> ${keepalive_pid}"
    fi
  fi

  if [[ -f "${AISTUDIO_DOWNLOAD_PIDFILE}" ]]; then
    local aistudio_pid
    aistudio_pid="$(cat "${AISTUDIO_DOWNLOAD_PIDFILE}")"
    if ps -p "${aistudio_pid}" >/dev/null 2>&1; then
      echo
      echo "AI Studio downloader"
      ps -p "${aistudio_pid}" -o pid,ppid,stat,etime,cmd
    else
      echo "  AI Studio downloader pidfile exists but process is not running: ${AISTUDIO_DOWNLOAD_PIDFILE} -> ${aistudio_pid}"
    fi
  fi

  local active
  active="$(ps -eo pid,ppid,stat,etime,cmd | rg 'download_dair_v2x_c|download_dair_v2x_aistudio|keepalive_dair_v2x_aistudio|gdown' | rg -v 'monitor_dair_v2x_download|rg ' || true)"
  if [[ -n "${active}" ]]; then
    echo
    echo "Active downloader/gdown processes"
    printf '%s\n' "${active}"
  fi
}

print_manifest_status() {
  echo
  echo "Manifest files"
  printf '%-10s %-8s %-70s %s\n' "status" "size" "filename" "required_for"
  printf '%-10s %-8s %-70s %s\n' "----------" "--------" "----------------------------------------------------------------------" "------------"

  local total=0
  local done=0
  local partial=0
  local missing=0

  while IFS=, read -r _subset _part filename _file_id required_for; do
    [[ -n "${filename}" ]] || continue
    total=$((total + 1))
    local target="${OUT_DIR}/${filename}"
    local status size
    if [[ -s "${target}" ]]; then
      status="complete"
      size="$(human_size "${target}")"
      done=$((done + 1))
    else
      local partial_files=()
      while IFS= read -r part_file; do
        partial_files+=("${part_file}")
      done < <(find "${OUT_DIR}" -maxdepth 1 -type f -name "${filename}*.part" -print 2>/dev/null)
      if (( ${#partial_files[@]} > 0 )); then
        status="partial"
        size="$(du -ch "${partial_files[@]}" 2>/dev/null | awk '/total$/ {print $1}')"
        partial=$((partial + 1))
      else
        status="missing"
        size="-"
        missing=$((missing + 1))
      fi
    fi
    printf '%-10s %-8s %-70s %s\n' "${status}" "${size}" "${filename}" "${required_for}"
  done < <(tail -n +2 "${MANIFEST}")

  echo
  echo "Summary: complete=${done}/${total}, partial=${partial}, missing=${missing}"
  echo "Raw dir: ${OUT_DIR} ($(human_size "${OUT_DIR}"))"
}

print_aistudio_status() {
  echo
  echo "AI Studio mirror files"
  printf '%-10s %-8s %-74s %s\n' "status" "size" "filename" "source"
  printf '%-10s %-8s %-74s %s\n' "----------" "--------" "--------------------------------------------------------------------------" "------"

  local rows=(
    "cooperative-vehicle-infrastructure.zip|infrastructure.zip"
    "cooperative-vehicle-infrastructure-vehicle-side-image.zip|infrastructure-vehicle-side-image.zip"
    "cooperative-vehicle-infrastructure-infrastructure-side-image-deprecated.zip|infrastructure-infrastructure-side-image.zip"
    "cooperative-vehicle-infrastructure-infrastructure-side-velodyne.zip|infrastructure-infrastructure-side-velodyne.zip"
    "cooperative-vehicle-infrastructure-vehicle-side-velodyne.zip|infrastructure-vehicle-side-velodyne.zip"
  )

  local complete=0
  local partial=0
  local missing=0
  local row target source status size part
  for row in "${rows[@]}"; do
    target="${row%%|*}"
    source="${row#*|}"
    if [[ -s "${OUT_DIR}/${target}" ]]; then
      status="complete"
      size="$(human_size "${OUT_DIR}/${target}")"
      complete=$((complete + 1))
    elif [[ -s "${OUT_DIR}/${target}.aistudio.part" ]]; then
      status="partial"
      size="$(human_size "${OUT_DIR}/${target}.aistudio.part")"
      partial=$((partial + 1))
    else
      part="$(find "${OUT_DIR}" -maxdepth 1 -type f -name "${target}*.part" -print -quit 2>/dev/null || true)"
      if [[ -n "${part}" ]]; then
        status="partial-gd"
        size="$(human_size "${part}")"
        partial=$((partial + 1))
      else
        status="missing"
        size="-"
        missing=$((missing + 1))
      fi
    fi
    printf '%-10s %-8s %-74s %s\n' "${status}" "${size}" "${target}" "${source}"
  done

  echo
  echo "AI Studio summary: complete=${complete}/5, partial=${partial}, missing=${missing}"
  echo "Note: AI Studio velodyne archives are single .zip files; Google Drive uses split .z01/.zip archives."
}

print_latest_logs() {
  local detached_log attempt_log aistudio_log aistudio_current keepalive_log
  detached_log="$(latest_matching_log 'dair_v2x_dataset_download_until_complete_detached_*.log')"
  attempt_log="$(latest_matching_log 'dair_v2x_dataset_download_until_complete_attempt_*.log')"
  aistudio_log="$(latest_matching_log 'dair_v2x_aistudio_download_attempt_*.log')"
  aistudio_current="${LOG_DIR}/dair_v2x_aistudio_download_current.log"
  keepalive_log="${LOG_DIR}/dair_v2x_aistudio_keepalive.log"

  echo
  echo "Latest logs"
  if [[ -n "${detached_log}" ]]; then
    echo "- detached: ${detached_log}"
    tail -n "${TAIL_LINES}" "${detached_log}" || true
  fi
  if [[ -n "${attempt_log}" && "${attempt_log}" != "${detached_log}" ]]; then
    echo
    echo "- attempt: ${attempt_log}"
    tail -n "${TAIL_LINES}" "${attempt_log}" || true
  fi
  if [[ -n "${aistudio_log}" ]]; then
    echo
    echo "- aistudio: ${aistudio_log}"
    tail -n "${TAIL_LINES}" "${aistudio_log}" || true
  fi
  if [[ -f "${aistudio_current}" ]]; then
    echo
    echo "- aistudio-current: ${aistudio_current}"
    tail -n "${TAIL_LINES}" "${aistudio_current}" || true
  fi
  if [[ -f "${keepalive_log}" ]]; then
    echo
    echo "- aistudio-keepalive: ${keepalive_log}"
    tail -n "${TAIL_LINES}" "${keepalive_log}" || true
  fi
}

print_snapshot() {
  echo "DAIR-V2X download monitor"
  echo "time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "root: ${ROOT}"
  echo
  print_process_status
  print_manifest_status
  print_aistudio_status
  print_latest_logs
}

if (( WATCH_SECONDS > 0 )); then
  while true; do
    clear || true
    print_snapshot
    echo
    echo "refreshing every ${WATCH_SECONDS}s; press Ctrl-C to stop the monitor only"
    sleep "${WATCH_SECONDS}"
  done
else
  print_snapshot
fi
