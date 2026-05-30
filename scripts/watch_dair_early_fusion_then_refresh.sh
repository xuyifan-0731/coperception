#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/dair_v2x/official_baselines}"
STATUS_CSV="${STATUS_CSV:-${ROOT}/results/dair_v2x/summary/current_status.csv}"
POLL_SECONDS="${POLL_SECONDS:-300}"

cd "${ROOT}"

pid_files=(
  "${RESULT_ROOT}/isolated_early_fusion_k3.pid"
  "${RESULT_ROOT}/isolated_early_fusion_k4.pid"
  "${RESULT_ROOT}/isolated_early_fusion_k5.pid"
)

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"
}

running() {
  local pid_file pid
  for pid_file in "${pid_files[@]}"; do
    [[ -s "${pid_file}" ]] || continue
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

all_complete_logs_present() {
  local k log_file
  for k in 3 4 5; do
    log_file="${RESULT_ROOT}/logs/early_fusion_k${k}.log"
    if ! grep -q "Average Communication Cost" "${log_file}" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

log "watching optional early_fusion_k3..k5 jobs"
while running; do
  log "still running; sleeping ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

if ! all_complete_logs_present; then
  log "early fusion extension finished but at least one log is incomplete"
  python - <<'PY'
from pathlib import Path
path = Path("results/dair_v2x/summary/current_status.csv")
text = path.read_text()
text = text.replace(
    "Optional DAIR early-fusion extension,running",
    "Optional DAIR early-fusion extension,failed_or_incomplete",
)
text = text.replace(
    "Active experiment processes,running,nvidia-smi,wait for optional early-fusion extension to finish",
    "Active experiment processes,none,nvidia-smi,inspect early-fusion logs",
)
path.write_text(text)
PY
  exit 2
fi

log "all early fusion extension logs complete; refreshing summaries"
python scripts/parse_dair_official_baseline_logs.py \
  --log-dir "${RESULT_ROOT}/logs" \
  --out "${RESULT_ROOT}/summary_partial.csv"
python scripts/make_dair_revision_tables.py
python scripts/make_dair_paper_artifacts.py

python - <<'PY'
from pathlib import Path
path = Path("results/dair_v2x/summary/current_status.csv")
text = path.read_text()
text = text.replace(
    "Optional DAIR early-fusion extension,running",
    "Optional DAIR early-fusion extension,completed",
)
text = text.replace(
    "Active experiment processes,running,nvidia-smi,wait for optional early-fusion extension to finish",
    "Active experiment processes,none,nvidia-smi,none",
)
path.write_text(text)
PY
log "refresh complete"
