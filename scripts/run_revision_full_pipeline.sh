#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-V2X-Sim-det}"
RESULT_ROOT="${RESULT_ROOT:-results/revision_plan}"
VSPM_RUN_ROOT="${VSPM_RUN_ROOT:-runs/revision_vspm_sensitivity}"
MST_RUN_ROOT="${MST_RUN_ROOT:-runs/revision_mst_ablation}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${RESULT_ROOT}" "${VSPM_RUN_ROOT}" "${MST_RUN_ROOT}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 2
fi

echo "[revision-pipeline] start $(date --iso-8601=seconds)"
echo "[revision-pipeline] DATA_ROOT=${DATA_ROOT}"
echo "[revision-pipeline] WANDB_MODE=${WANDB_MODE}"

python scripts/profile_vspm.py --device auto --repeat 5 \
  2>&1 | tee "${RESULT_ROOT}/profile_vspm.full.log"

python scripts/run_scalability_profile.py \
  2>&1 | tee "${RESULT_ROOT}/scalability_profile.full.log"

DRY_RUN=0 DATA_ROOT="${DATA_ROOT}" RUN_ROOT="${VSPM_RUN_ROOT}" \
  bash scripts/run_vspm_sensitivity.sh \
  2>&1 | tee "${RESULT_ROOT}/run_vspm_sensitivity.full.log"

DRY_RUN=0 DATA_ROOT="${DATA_ROOT}" RUN_ROOT="${MST_RUN_ROOT}" \
  bash scripts/run_mst_ablation.sh \
  2>&1 | tee "${RESULT_ROOT}/run_mst_ablation.full.log"

python scripts/collect_results.py --root runs --out results/summary_revision --copy-raw-metrics \
  2>&1 | tee "${RESULT_ROOT}/collect_results.full.log"

echo "[revision-pipeline] done $(date --iso-8601=seconds)"
