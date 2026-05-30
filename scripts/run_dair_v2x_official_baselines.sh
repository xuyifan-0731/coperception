#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
ENV_NAME="${ENV_NAME:-dair-v2x-openmmlab}"
OPENDAIR_ROOT="${OPENDAIR_ROOT:-${ROOT}/external/DAIR-V2X}"
DEVICE_ID="${DEVICE_ID:-0}"
K_LIST="${K_LIST:-0 1 2}"
RANGE_START="${RANGE_START:-0}"
RANGE_END="${RANGE_END:-100}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/dair_v2x/official_baselines}"
STRICT_CHECKPOINTS="${STRICT_CHECKPOINTS:-0}"

DATA_LINK="${OPENDAIR_ROOT}/data/DAIR-V2X/cooperative-vehicle-infrastructure"
if [[ ! -e "${DATA_LINK}" ]]; then
  echo "missing dataset symlink: ${DATA_LINK}" >&2
  echo "run scripts/prepare_dair_v2x_layout.sh after the dataset is downloaded/extracted" >&2
  exit 2
fi

late_ckpts=(
  "${OPENDAIR_ROOT}/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_inf_pointpillars_596784ad6127866fcfb286301757c949.pth"
  "${OPENDAIR_ROOT}/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth"
)
early_ckpt="${OPENDAIR_ROOT}/configs/vic3d/early-fusion-pointcloud/pointpillars/vic3d_earlyfusion_veh_pointpillars_67fe2b82320754481ef37f176b647e43.pth"

run_late=1
for ckpt in "${late_ckpts[@]}"; do
  if [[ ! -s "${ckpt}" ]]; then
    echo "missing late-fusion checkpoint: ${ckpt}" >&2
    run_late=0
  fi
done

run_early=1
if [[ ! -s "${early_ckpt}" ]]; then
  echo "missing early-fusion checkpoint: ${early_ckpt}" >&2
  run_early=0
fi

if [[ "${STRICT_CHECKPOINTS}" == "1" && ( "${run_late}" == "0" || "${run_early}" == "0" ) ]]; then
  echo "STRICT_CHECKPOINTS=1 and at least one required checkpoint is missing" >&2
  echo "run scripts/download_dair_v2x_checkpoints.sh first" >&2
  exit 3
fi

if [[ "${run_late}" == "0" && "${run_early}" == "0" ]]; then
  echo "no runnable official point-cloud baselines because checkpoints are missing" >&2
  exit 3
fi

set +u
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
set -u

mkdir -p "${RESULT_ROOT}/logs"
cd "${OPENDAIR_ROOT}/v2x"

run_logged() {
  local name="$1"
  shift
  local log="${RESULT_ROOT}/logs/${name}.log"
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ${name}: $*" | tee "${log}"
  "$@" 2>&1 | tee -a "${log}"
}

if [[ "${run_late}" == "1" ]]; then
  run_logged "veh_only_k0" bash scripts/eval_lidar_late_fusion_pointpillars.sh "${DEVICE_ID}" veh_only 0 "${RANGE_START}" "${RANGE_END}" ""
  run_logged "inf_only_k0" bash scripts/eval_lidar_late_fusion_pointpillars.sh "${DEVICE_ID}" inf_only 0 "${RANGE_START}" "${RANGE_END}" ""
else
  echo "skipping veh_only/inf_only/late_fusion because late-fusion checkpoints are incomplete"
fi

for k in ${K_LIST}; do
  if [[ "${run_late}" == "1" ]]; then
    run_logged "late_fusion_tclf_k${k}" bash scripts/eval_lidar_late_fusion_pointpillars.sh "${DEVICE_ID}" late_fusion "${k}" "${RANGE_START}" "${RANGE_END}" ""
    if [[ "${k}" != "0" ]]; then
      run_logged "late_fusion_no_comp_k${k}" bash scripts/eval_lidar_late_fusion_pointpillars.sh "${DEVICE_ID}" late_fusion "${k}" "${RANGE_START}" "${RANGE_END}" "--no-comp"
    fi
  fi
  if [[ "${run_early}" == "1" ]]; then
    run_logged "early_fusion_k${k}" bash scripts/eval_lidar_early_fusion_pointpillars.sh "${DEVICE_ID}" early_fusion "${k}" "${RANGE_START}" "${RANGE_END}"
  else
    echo "skipping early_fusion_k${k} because the early-fusion checkpoint is missing"
  fi
done

echo "official DAIR-V2X baseline pass completed"
