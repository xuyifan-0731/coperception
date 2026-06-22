#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <held_out_agent_id: 1..5>" >&2
  exit 2
fi

HELD_OUT="$1"
DATA_ROOT="${DATA_ROOT:-/workspace/xuyifan/v2x/V2X-Sim-det}"
RUN_ROOT="${RUN_ROOT:-runs/leave_one_out}"
WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_MODE

train_roots=()
for agent in 1 2 3 4 5; do
  if [[ "${agent}" != "${HELD_OUT}" ]]; then
    train_roots+=("${DATA_ROOT}/train/agent${agent}")
  fi
done

python src/v2x_forecasting/train_bev.py \
  --data_root "${train_roots[@]}" \
  --data_root_val "${DATA_ROOT}/test/agent${HELD_OUT}" \
  --T 10 --n 5 --batch 16 --epochs 30 \
  --use_dice --use_focal --use_contrast --use_change_head \
  --teacher_forcing_start 1.0 --teacher_forcing_end 0.0 --teacher_forcing_epochs 10 \
  --ckpt_interval 1000 \
  --eval_interval 1000 \
  --ckpt_dir "${RUN_ROOT}/checkpoints_without_agent${HELD_OUT}" \
  --out_dir "${RUN_ROOT}/predictions_agent${HELD_OUT}" \
  --project bev-encdec \
  --name "leave_one_out_agent${HELD_OUT}"

