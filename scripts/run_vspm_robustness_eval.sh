#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-V2X-Sim-det/test/agent1}"
RUN_ROOT="${RUN_ROOT:-runs/revision_vspm_sensitivity}"
RUN_NAME="${RUN_NAME:-T5_n5}"
CKPT="${CKPT:-}"
OUT="${OUT:-results/revision_plan/vspm_robustness_eval.csv}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
DEVICE="${DEVICE:-cpu}"
T="${T:-5}"
N="${N:-5}"
CPU_THREADS="${CPU_THREADS:-16}"

if [[ -z "${CKPT}" ]]; then
  CKPT="$(find "${RUN_ROOT}/${RUN_NAME}/checkpoints" -maxdepth 1 -name 'ckpt_*.pth' | sort -V | tail -1)"
fi

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "No checkpoint found. Set CKPT=/path/to/ckpt_*.pth." >&2
  exit 2
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 2
fi

if [[ "${DEVICE}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS}}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREADS}}"
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:-src}"

python scripts/eval_vspm_robustness.py \
  --ckpt "${CKPT}" \
  --data-root "${DATA_ROOT}" \
  --T "${T}" \
  --n "${N}" \
  --max-samples "${MAX_SAMPLES}" \
  --device "${DEVICE}" \
  --torch-threads "${CPU_THREADS}" \
  --out "${OUT}"
