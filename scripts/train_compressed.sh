#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-gru_fp16}"
DATA_ROOT="${DATA_ROOT:-/workspace/xuyifan/v2x/V2X-Sim-det}"
RUN_ROOT="${RUN_ROOT:-runs/compressed/${MODE}}"
WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_MODE

extra_args=()
case "${MODE}" in
  gru_fp16)
    extra_args+=(--enable_gru_fp16)
    ;;
  ds64_fp16)
    extra_args+=(--enable_gru_fp16 --enable_ds64_fp16)
    ;;
  bottleneck12)
    extra_args+=(--enable_gru_fp16 --enable_ds64_fp16 --enable_bottleneck_1x1 --bottleneck_dim 12)
    ;;
  *)
    echo "Usage: $0 [gru_fp16|ds64_fp16|bottleneck12]" >&2
    exit 2
    ;;
esac

python src/v2x_forecasting/train_bev_compressed.py \
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
  --name "compressed_${MODE}" \
  "${extra_args[@]}"

