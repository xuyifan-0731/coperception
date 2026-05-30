#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-V2X-Sim-det}"
RUN_ROOT="${RUN_ROOT:-runs/revision_vspm_sensitivity}"
DRY_RUN="${DRY_RUN:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_MODE

if [[ "${DRY_RUN}" != "1" && ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  echo "Extract V2X-Sim-det-001.zip first or set DATA_ROOT=/path/to/V2X-Sim-det." >&2
  exit 2
fi

run_cmd() {
  local name="$1"
  shift
  local cmd=(python src/v2x_forecasting/train_bev.py "$@")
  echo "[${name}] ${cmd[*]}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}"
  fi
}

for t in 5 10 20 30; do
  run_cmd "T${t}_n5" \
    --data_root "${DATA_ROOT}/train/agent1" \
    --data_root_val "${DATA_ROOT}/test/agent1" \
    --T "${t}" --n 5 --batch 16 --epochs 30 \
    --use_dice \
    --teacher_forcing_start 1.0 --teacher_forcing_end 0.0 --teacher_forcing_epochs 10 \
    --ckpt_interval 1000 --eval_interval 1000 \
    --ckpt_dir "${RUN_ROOT}/T${t}_n5/checkpoints" \
    --project bev-encdec --name "revision_T${t}_n5"
done

for n in 3 5 10 15; do
  run_cmd "T10_n${n}" \
    --data_root "${DATA_ROOT}/train/agent1" \
    --data_root_val "${DATA_ROOT}/test/agent1" \
    --T 10 --n "${n}" --batch 16 --epochs 30 \
    --use_dice \
    --teacher_forcing_start 1.0 --teacher_forcing_end 0.0 --teacher_forcing_epochs 10 \
    --ckpt_interval 1000 --eval_interval 1000 \
    --ckpt_dir "${RUN_ROOT}/T10_n${n}/checkpoints" \
    --project bev-encdec --name "revision_T10_n${n}"
done
