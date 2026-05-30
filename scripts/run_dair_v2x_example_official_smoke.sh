#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
ENV_NAME="${ENV_NAME:-dair-v2x-openmmlab}"
OPENDAIR_ROOT="${OPENDAIR_ROOT:-${ROOT}/external/DAIR-V2X}"
DEVICE_ID="${DEVICE_ID:-0}"
K_LIST="${K_LIST:-0 2}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/dair_v2x/example_official_smoke}"
EXAMPLE_DATA="${EXAMPLE_DATA:-../data/DAIR-V2X/example-cooperative-vehicle-infrastructure}"
EXAMPLE_SPLIT="${EXAMPLE_SPLIT:-../data/split_datas/example-cooperative-split-data.json}"

set +u
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
set -u

cd "${OPENDAIR_ROOT}/v2x"
mkdir -p ../cache/tmps "${RESULT_ROOT}/logs"

INFRA_MODEL_PATH="../configs/vic3d/late-fusion-pointcloud/pointpillars"
INFRA_CONFIG_NAME="trainval_config_i.py"
INFRA_MODEL_NAME="vic3d_latefusion_inf_pointpillars_596784ad6127866fcfb286301757c949.pth"
VEHICLE_MODEL_PATH="../configs/vic3d/late-fusion-pointcloud/pointpillars"
VEHICLE_CONFIG_NAME="trainval_config_v.py"
VEHICLE_MODEL_NAME="vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth"

run_one() {
  local name="$1"
  local model="$2"
  local k="$3"
  local extra="${4:-}"
  local out="${RESULT_ROOT}/${name}"
  local log="${RESULT_ROOT}/logs/${name}.log"
  rm -rf "${out}"
  mkdir -p "${out}/result" "${out}/veh/lidar" "${out}/inf/lidar"
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ${name}" | tee "${log}"
  python eval.py \
    --input "${EXAMPLE_DATA}" \
    --output "${out}" \
    --model "${model}" \
    --dataset vic-async \
    --k "${k}" \
    --split train \
    --split-data-path "${EXAMPLE_SPLIT}" \
    --inf-config-path "${INFRA_MODEL_PATH}/${INFRA_CONFIG_NAME}" \
    --inf-model-path "${INFRA_MODEL_PATH}/${INFRA_MODEL_NAME}" \
    --veh-config-path "${VEHICLE_MODEL_PATH}/${VEHICLE_CONFIG_NAME}" \
    --veh-model-path "${VEHICLE_MODEL_PATH}/${VEHICLE_MODEL_NAME}" \
    --device "${DEVICE_ID}" \
    --pred-class car \
    --sensortype lidar \
    --extended-range 0 -39.68 -3 100 39.68 1 \
    --overwrite-cache \
    ${extra} 2>&1 | tee -a "${log}"
}

run_one "veh_only_k0" "veh_only" 0
run_one "inf_only_k0" "inf_only" 0

for k in ${K_LIST}; do
  run_one "late_fusion_tclf_k${k}" "late_fusion" "${k}"
  if [[ "${k}" != "0" ]]; then
    run_one "late_fusion_no_comp_k${k}" "late_fusion" "${k}" "--no-comp"
  fi
done

echo "DAIR-V2X-C example official baseline smoke completed"
