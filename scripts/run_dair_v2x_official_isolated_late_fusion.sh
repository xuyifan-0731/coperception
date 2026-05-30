#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
BASE_OPENDAIR_ROOT="${BASE_OPENDAIR_ROOT:-${ROOT}/external/DAIR-V2X}"
ISOLATED_ROOT_BASE="${ISOLATED_ROOT_BASE:-${ROOT}/external/DAIR-V2X-isolated}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/dair_v2x/official_baselines}"
ENV_NAME="${ENV_NAME:-dair-v2x-openmmlab}"
K_LIST="${K_LIST:-3 4 5}"
GPU_LIST="${GPU_LIST:-1 4 6}"
RANGE_START="${RANGE_START:-0}"
RANGE_END="${RANGE_END:-100}"
WANDB_MODE="${WANDB_MODE:-offline}"

cd "${ROOT}"
mkdir -p "${RESULT_ROOT}/logs" "${ISOLATED_ROOT_BASE}"

set +u
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
set -u

prepare_isolated_root() {
  local gpu="$1"
  local root="${ISOLATED_ROOT_BASE}/gpu${gpu}"
  mkdir -p "${root}"
  for item in configs data docs resources tools v2x; do
    if [[ ! -e "${root}/${item}" ]]; then
      cp -al "${BASE_OPENDAIR_ROOT}/${item}" "${root}/${item}"
    fi
  done
  rm -rf "${root}/cache"
  mkdir -p "${root}/cache"
  printf '%s\n' "${root}"
}

run_k() {
  local k="$1"
  local gpu="$2"
  local isolated_root
  isolated_root="$(prepare_isolated_root "${gpu}")"

  (
    cd "${isolated_root}/v2x"
    mkdir -p ../cache/vic-late-lidar
    export WANDB_MODE
    export PYTHONUNBUFFERED=1
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] late_fusion_tclf_k${k}: isolated_root=${isolated_root} gpu=${gpu}"
    bash scripts/eval_lidar_late_fusion_pointpillars.sh "${gpu}" late_fusion "${k}" "${RANGE_START}" "${RANGE_END}" ""
  ) > "${RESULT_ROOT}/logs/late_fusion_tclf_k${k}.log" 2>&1

  rm -rf "${isolated_root}/cache"
  mkdir -p "${isolated_root}/cache/vic-late-lidar"

  (
    cd "${isolated_root}/v2x"
    export WANDB_MODE
    export PYTHONUNBUFFERED=1
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] late_fusion_no_comp_k${k}: isolated_root=${isolated_root} gpu=${gpu}"
    bash scripts/eval_lidar_late_fusion_pointpillars.sh "${gpu}" late_fusion "${k}" "${RANGE_START}" "${RANGE_END}" "--no-comp"
  ) > "${RESULT_ROOT}/logs/late_fusion_no_comp_k${k}.log" 2>&1
}

read -r -a ks <<< "${K_LIST}"
read -r -a gpus <<< "${GPU_LIST}"
if (( ${#gpus[@]} < ${#ks[@]} )); then
  echo "GPU_LIST must contain at least as many entries as K_LIST" >&2
  exit 2
fi

for idx in "${!ks[@]}"; do
  k="${ks[$idx]}"
  gpu="${gpus[$idx]}"
  run_k "${k}" "${gpu}" &
  echo "$!" > "${RESULT_ROOT}/logs/isolated_late_fusion_k${k}.pid"
done

wait
