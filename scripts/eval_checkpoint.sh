#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <checkpoint.pth> <agent_id: 1..5> <out_dir>" >&2
  exit 2
fi

CKPT_PATH="$1"
AGENT="$2"
OUT_DIR="$3"
DATA_ROOT="${DATA_ROOT:-/workspace/xuyifan/v2x/V2X-Sim-det}"
T="${T:-10}"
N="${N:-5}"
BATCH="${BATCH:-16}"

python src/v2x_forecasting/train_bev.py \
  --data_root "${DATA_ROOT}/train/agent${AGENT}" \
  --data_root_val "${DATA_ROOT}/test/agent${AGENT}" \
  --eval_only \
  --ckpt_path "${CKPT_PATH}" \
  --out_dir "${OUT_DIR}" \
  --T "${T}" --n "${N}" --batch "${BATCH}" --epochs 30 \
  --use_dice --use_focal --use_contrast --use_change_head \
  --teacher_forcing_start 1.0 --teacher_forcing_end 0.0 --teacher_forcing_epochs 10
