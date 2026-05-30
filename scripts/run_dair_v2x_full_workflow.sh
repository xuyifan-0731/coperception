#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
ENV_NAME="${ENV_NAME:-dair-v2x-openmmlab}"
DEVICE_ID="${DEVICE_ID:-0}"
K_LIST="${K_LIST:-0 1 2}"

RUN_DOWNLOADS="${RUN_DOWNLOADS:-0}"
RUN_OFFICIAL="${RUN_OFFICIAL:-1}"
RUN_OURS="${RUN_OURS:-1}"
STRICT_CHECKPOINTS="${STRICT_CHECKPOINTS:-0}"

DAIR_ROOT="${DAIR_ROOT:-${ROOT}/datasets/DAIR-V2X/cooperative-vehicle-infrastructure}"
SPLIT_PATH="${SPLIT_PATH:-${ROOT}/external/DAIR-V2X/data/split_datas/cooperative-split-data.json}"
VSPM_OUT="${VSPM_OUT:-${ROOT}/datasets/DAIR-V2X/vspm_bev}"
OUR_ENV_NAME="${OUR_ENV_NAME:-Android-Lab}"
OUR_DEVICE="${OUR_DEVICE:-cuda}"
OUR_T="${OUR_T:-10}"
OUR_N="${OUR_N:-5}"
OUR_BATCH="${OUR_BATCH:-16}"
OUR_EPOCHS="${OUR_EPOCHS:-30}"
OUR_MAX_STEPS="${OUR_MAX_STEPS:-0}"
OUR_MAX_VAL_BATCHES="${OUR_MAX_VAL_BATCHES:-0}"
OUR_CKPT_DIR="${OUR_CKPT_DIR:-${ROOT}/runs/dair_v2x_full/our_method/checkpoints}"
OUR_RUN_NAME="${OUR_RUN_NAME:-dair_v2x_full_our_method}"
OUR_PROJECT="${OUR_PROJECT:-bev-encdec}"
USE_DICE="${USE_DICE:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"

cd "${ROOT}"
mkdir -p "${ROOT}/results/dair_v2x/full_workflow"

log_step() {
  echo
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

if [[ "${RUN_DOWNLOADS}" == "1" ]]; then
  log_step "retrying DAIR-V2X-C full data download"
  bash scripts/download_dair_v2x_c.sh

  log_step "retrying official checkpoint download"
  bash scripts/download_dair_v2x_checkpoints.sh || {
    if [[ "${STRICT_CHECKPOINTS}" == "1" ]]; then
      exit 4
    fi
    echo "continuing because STRICT_CHECKPOINTS=0"
  }
fi

log_step "preparing DAIR-V2X layout"
bash scripts/prepare_dair_v2x_layout.sh

log_step "checking DAIR-V2X prerequisites"
bash scripts/check_dair_v2x_prereqs.sh

if [[ "${RUN_OFFICIAL}" == "1" ]]; then
  log_step "running official OpenDAIRV2X point-cloud baselines"
  ENV_NAME="${ENV_NAME}" DEVICE_ID="${DEVICE_ID}" K_LIST="${K_LIST}" STRICT_CHECKPOINTS="${STRICT_CHECKPOINTS}" \
    bash scripts/run_dair_v2x_official_baselines.sh
fi

if [[ "${RUN_OURS}" == "1" ]]; then
  log_step "converting DAIR-V2X labels to VSPM BEV sequences"
  max_args=()
  if [[ "${MAX_FRAMES}" != "0" ]]; then
    max_args+=(--max-frames "${MAX_FRAMES}")
  fi
  python scripts/prepare_dair_vspm_bev.py \
    --dair-root "${DAIR_ROOT}" \
    --split-path "${SPLIT_PATH}" \
    --sensor-view vehicle \
    --output-root "${VSPM_OUT}" \
    "${max_args[@]}"

  train_args=(
    --data_root "${VSPM_OUT}/train/agent_vehicle"
    --data_root_val "${VSPM_OUT}/val/agent_vehicle"
    --T "${OUR_T}"
    --n "${OUR_N}"
    --batch "${OUR_BATCH}"
    --epochs "${OUR_EPOCHS}"
    --ckpt_interval 1000
    --eval_interval 1000
    --ckpt_dir "${OUR_CKPT_DIR}"
    --project "${OUR_PROJECT}"
    --name "${OUR_RUN_NAME}"
    --device "${OUR_DEVICE}"
  )
  if [[ "${USE_DICE}" == "1" ]]; then
    train_args+=(--use_dice)
  fi
  if [[ "${OUR_MAX_STEPS}" != "0" ]]; then
    train_args+=(--max_steps "${OUR_MAX_STEPS}")
  fi
  if [[ "${OUR_MAX_VAL_BATCHES}" != "0" ]]; then
    train_args+=(--max_val_batches "${OUR_MAX_VAL_BATCHES}")
  fi

  log_step "training our VSPM/DLPCM path on DAIR-V2X BEV sequences"
  eval "$(conda shell.bash hook)"
  conda activate "${OUR_ENV_NAME}"
  WANDB_MODE="${WANDB_MODE:-offline}" python src/v2x_forecasting/train_bev.py "${train_args[@]}"
fi

log_step "DAIR-V2X full workflow completed"
