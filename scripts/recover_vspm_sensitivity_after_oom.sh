#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-V2X-Sim-det}"
RUN_ROOT="${RUN_ROOT:-runs/revision_vspm_sensitivity}"
WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_MODE
export PYTHONDONTWRITEBYTECODE=1

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 2
fi

run_cmd() {
  local name="$1"
  local batch="$2"
  shift 2
  local ckpt_dir="${RUN_ROOT}/${name}/checkpoints"
  local done_marker="${RUN_ROOT}/${name}/.done"
  if [[ -f "${done_marker}" ]]; then
    echo "[skip:${name}] done marker exists"
    return
  fi
  if [[ -f "${ckpt_dir}/wandb_logs.txt" ]] && rg -q "'epoch': 30" "${ckpt_dir}/wandb_logs.txt"; then
    echo "[skip:${name}] completed epoch 30 according to wandb_logs.txt"
    touch "${done_marker}"
    return
  fi

  local cmd=(python src/v2x_forecasting/train_bev.py "$@")
  echo "[${name}] batch=${batch} ${cmd[*]}"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" "${cmd[@]}"
  touch "${done_marker}"
}

# T5_n5 and T10_n5 were completed before the OOM. Continue from the larger
# memory cases with smaller batch sizes.
run_cmd "T20_n5" 8 \
  --data_root "${DATA_ROOT}/train/agent1" \
  --data_root_val "${DATA_ROOT}/test/agent1" \
  --T 20 --n 5 --batch 8 --epochs 30 \
  --use_dice \
  --teacher_forcing_start 1.0 --teacher_forcing_end 0.0 --teacher_forcing_epochs 10 \
  --ckpt_interval 1000 --eval_interval 1000 \
  --ckpt_dir "${RUN_ROOT}/T20_n5/checkpoints" \
  --project bev-encdec --name "revision_T20_n5_b8"

run_cmd "T30_n5" 4 \
  --data_root "${DATA_ROOT}/train/agent1" \
  --data_root_val "${DATA_ROOT}/test/agent1" \
  --T 30 --n 5 --batch 4 --epochs 30 \
  --use_dice \
  --teacher_forcing_start 1.0 --teacher_forcing_end 0.0 --teacher_forcing_epochs 10 \
  --ckpt_interval 1000 --eval_interval 1000 \
  --ckpt_dir "${RUN_ROOT}/T30_n5/checkpoints" \
  --project bev-encdec --name "revision_T30_n5_b4"

run_cmd "T10_n3" 16 \
  --data_root "${DATA_ROOT}/train/agent1" \
  --data_root_val "${DATA_ROOT}/test/agent1" \
  --T 10 --n 3 --batch 16 --epochs 30 \
  --use_dice \
  --teacher_forcing_start 1.0 --teacher_forcing_end 0.0 --teacher_forcing_epochs 10 \
  --ckpt_interval 1000 --eval_interval 1000 \
  --ckpt_dir "${RUN_ROOT}/T10_n3/checkpoints" \
  --project bev-encdec --name "revision_T10_n3"

run_cmd "T10_n10" 8 \
  --data_root "${DATA_ROOT}/train/agent1" \
  --data_root_val "${DATA_ROOT}/test/agent1" \
  --T 10 --n 10 --batch 8 --epochs 30 \
  --use_dice \
  --teacher_forcing_start 1.0 --teacher_forcing_end 0.0 --teacher_forcing_epochs 10 \
  --ckpt_interval 1000 --eval_interval 1000 \
  --ckpt_dir "${RUN_ROOT}/T10_n10/checkpoints" \
  --project bev-encdec --name "revision_T10_n10_b8"

run_cmd "T10_n15" 4 \
  --data_root "${DATA_ROOT}/train/agent1" \
  --data_root_val "${DATA_ROOT}/test/agent1" \
  --T 10 --n 15 --batch 4 --epochs 30 \
  --use_dice \
  --teacher_forcing_start 1.0 --teacher_forcing_end 0.0 --teacher_forcing_epochs 10 \
  --ckpt_interval 1000 --eval_interval 1000 \
  --ckpt_dir "${RUN_ROOT}/T10_n15/checkpoints" \
  --project bev-encdec --name "revision_T10_n15_b4"
