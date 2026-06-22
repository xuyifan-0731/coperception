#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/xuyifan/v2x/V2X-Sim-det}"
RUN_ROOT="${RUN_ROOT:-runs/all_agents}"
WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_MODE

python src/v2x_forecasting/train_bev.py \
  --data_root \
    "${DATA_ROOT}/train/agent1" \
    "${DATA_ROOT}/train/agent2" \
    "${DATA_ROOT}/train/agent3" \
    "${DATA_ROOT}/train/agent4" \
    "${DATA_ROOT}/train/agent5" \
  --data_root_val \
    "${DATA_ROOT}/test/agent1" \
    "${DATA_ROOT}/test/agent2" \
    "${DATA_ROOT}/test/agent3" \
    "${DATA_ROOT}/test/agent4" \
    "${DATA_ROOT}/test/agent5" \
  --T 20 --n 10 --batch 16 --epochs 30 \
  --use_dice \
  --teacher_forcing_start 1.0 --teacher_forcing_end 0.0 --teacher_forcing_epochs 10 \
  --ckpt_interval 1000 \
  --eval_interval 1000 \
  --ckpt_dir "${RUN_ROOT}/checkpoints" \
  --out_dir "${RUN_ROOT}/predictions" \
  --project bev-encdec \
  --name all_agents_T20_n10

