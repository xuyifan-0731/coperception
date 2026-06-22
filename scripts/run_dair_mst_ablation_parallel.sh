#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
DATA_ROOT="${DATA_ROOT:-datasets/DAIR-V2X/vspm_bev}"
RUN_ROOT="${RUN_ROOT:-runs/dair_v2x_full/mst_ablation}"
LOG_ROOT="${LOG_ROOT:-results/dair_v2x/mst_ablation/logs}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
TRAIN_AGENT="${TRAIN_AGENT:-agent_vehicle}"
VAL_AGENT="${VAL_AGENT:-agent_vehicle}"
GPU_LIST="${GPU_LIST:-1 2 3 4}"
DRY_RUN="${DRY_RUN:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WAIT_AFTER_LAUNCH="${WAIT_AFTER_LAUNCH:-1}"

cd "${ROOT}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

eval "$(conda shell.bash hook)"
conda activate Android-Lab

modes=(baseline gru_fp16 ds64_fp16 bottleneck12)
read -r -a gpus <<< "${GPU_LIST}"
if (( ${#gpus[@]} < ${#modes[@]} )); then
  echo "GPU_LIST must contain at least ${#modes[@]} GPU ids" >&2
  exit 2
fi

launch_mode() {
  local mode="$1"
  local gpu="$2"
  shift 2
  local extra=("$@")
  local log="${LOG_ROOT}/${mode}_$(date -u '+%Y%m%d_%H%M%S').log"
  local pidfile="${LOG_ROOT}/${mode}.pid"
  local cmd=(
    python src/v2x_forecasting/train_bev_compressed.py
    --data_root "${DATA_ROOT}/${TRAIN_SPLIT}/${TRAIN_AGENT}"
    --data_root_val "${DATA_ROOT}/${VAL_SPLIT}/${VAL_AGENT}"
    --T 10 --n 5 --batch 16 --epochs 30
    --use_dice
    --teacher_forcing_start 1.0 --teacher_forcing_end 0.0 --teacher_forcing_epochs 10
    --ckpt_interval 1000 --eval_interval 1000
    --ckpt_dir "${RUN_ROOT}/${mode}/checkpoints"
    --project bev-encdec --name "dair_mst_${mode}"
    --device cuda
    "${extra[@]}"
  )

  echo "[${mode}] gpu=${gpu} log=${log}"
  printf '  %q' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export WANDB_MODE
    export PYTHONUNBUFFERED=1
    "${cmd[@]}"
  ) >"${log}" 2>&1 &
  echo "$!" > "${pidfile}"
}

launch_mode baseline "${gpus[0]}"
launch_mode gru_fp16 "${gpus[1]}" --enable_gru_fp16
launch_mode ds64_fp16 "${gpus[2]}" --enable_gru_fp16 --enable_ds64_fp16
launch_mode bottleneck12 "${gpus[3]}" --enable_gru_fp16 --enable_ds64_fp16 --enable_bottleneck_1x1 --bottleneck_dim 12

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "launched DAIR MST ablation jobs"
  for pidfile in "${LOG_ROOT}"/*.pid; do
    pid="$(cat "${pidfile}")"
    ps -p "${pid}" -o pid,ppid,stat,etime,cmd || true
  done
  if [[ "${WAIT_AFTER_LAUNCH}" == "1" ]]; then
    wait
  fi
fi
