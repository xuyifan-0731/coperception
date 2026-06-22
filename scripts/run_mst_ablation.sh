#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-V2X-Sim-det}"
RUN_ROOT="${RUN_ROOT:-runs/revision_mst_ablation}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-test}"
TRAIN_AGENT="${TRAIN_AGENT:-agent1}"
VAL_AGENT="${VAL_AGENT:-${TRAIN_AGENT}}"
DEVICE="${DEVICE:-auto}"
DRY_RUN="${DRY_RUN:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_MODE

if [[ "${DRY_RUN}" != "1" && ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  echo "Extract V2X-Sim-det-001.zip first or set DATA_ROOT=/path/to/V2X-Sim-det." >&2
  exit 2
fi

run_cmd() {
  local mode="$1"
  shift
  local cmd=(python src/v2x_forecasting/train_bev_compressed.py "$@")
  echo "[${mode}] ${cmd[*]}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}"
  fi
}

for mode in baseline gru_fp16 ds64_fp16 bottleneck12; do
  extra=()
  case "${mode}" in
    baseline) ;;
    gru_fp16) extra+=(--enable_gru_fp16) ;;
    ds64_fp16) extra+=(--enable_gru_fp16 --enable_ds64_fp16) ;;
    bottleneck12) extra+=(--enable_gru_fp16 --enable_ds64_fp16 --enable_bottleneck_1x1 --bottleneck_dim 12) ;;
  esac
  run_cmd "${mode}" \
    --data_root "${DATA_ROOT}/${TRAIN_SPLIT}/${TRAIN_AGENT}" \
    --data_root_val "${DATA_ROOT}/${VAL_SPLIT}/${VAL_AGENT}" \
    --T 10 --n 5 --batch 16 --epochs 30 \
    --use_dice \
    --teacher_forcing_start 1.0 --teacher_forcing_end 0.0 --teacher_forcing_epochs 10 \
    --ckpt_interval 1000 --eval_interval 1000 \
    --ckpt_dir "${RUN_ROOT}/${mode}/checkpoints" \
    --project bev-encdec --name "revision_mst_${mode}" \
    --device "${DEVICE}" \
    "${extra[@]}"
done
