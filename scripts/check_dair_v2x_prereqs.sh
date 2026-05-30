#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
ENV_NAME="${ENV_NAME:-dair-v2x-openmmlab}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/datasets/DAIR-V2X}"
OPENDAIR_ROOT="${OPENDAIR_ROOT:-${ROOT}/external/DAIR-V2X}"

ok=0
warn=0
fail=0

report() {
  local level="$1"
  local message="$2"
  printf '[%s] %s\n' "${level}" "${message}"
  case "${level}" in
    OK) ok=$((ok + 1)) ;;
    WARN) warn=$((warn + 1)) ;;
    FAIL) fail=$((fail + 1)) ;;
  esac
}

check_file() {
  local label="$1"
  local path="$2"
  if [[ -s "${path}" ]]; then
    report OK "${label}: ${path}"
  else
    report FAIL "${label} missing: ${path}"
  fi
}

check_dir() {
  local label="$1"
  local path="$2"
  if [[ -d "${path}" ]]; then
    report OK "${label}: ${path}"
  else
    report FAIL "${label} missing: ${path}"
  fi
}

echo "DAIR-V2X prerequisite check"
echo "root: ${ROOT}"
echo

check_file "manifest" "${DATASET_DIR}/dair_v2x_c_manifest.csv"
check_dir "OpenDAIRV2X repo" "${OPENDAIR_ROOT}/v2x"

full_root="${DATASET_DIR}/cooperative-vehicle-infrastructure"
if [[ -d "${full_root}" ]]; then
  check_dir "full DAIR root" "${full_root}"
  for rel in \
    "infrastructure-side" \
    "vehicle-side" \
    "cooperative" \
    "vehicle-side/data_info.json" \
    "infrastructure-side/data_info.json" \
    "cooperative/data_info.json"; do
    if [[ -e "${full_root}/${rel}" ]]; then
      report OK "full DAIR path present: ${rel}"
    else
      report FAIL "full DAIR path missing: ${rel}"
    fi
  done
else
  report FAIL "full DAIR root missing: ${full_root}"
fi

example_zip="${DATASET_DIR}/raw/DAIR-V2X-C-Example.zip"
if [[ -s "${example_zip}" ]]; then
  report OK "example zip present: ${example_zip}"
elif compgen -G "${example_zip}"'*'.part >/dev/null; then
  report WARN "example zip is still downloading: ${example_zip}*.part"
else
  report WARN "example zip not present: ${example_zip}"
fi

data_link="${OPENDAIR_ROOT}/data/DAIR-V2X/cooperative-vehicle-infrastructure"
if [[ -e "${data_link}" ]]; then
  report OK "OpenDAIRV2X data link: ${data_link}"
else
  report FAIL "OpenDAIRV2X data link missing: ${data_link}"
fi

check_file "late-fusion infrastructure PointPillars checkpoint" \
  "${OPENDAIR_ROOT}/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_inf_pointpillars_596784ad6127866fcfb286301757c949.pth"
check_file "late-fusion vehicle PointPillars checkpoint" \
  "${OPENDAIR_ROOT}/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth"

early_ckpt="${OPENDAIR_ROOT}/configs/vic3d/early-fusion-pointcloud/pointpillars/vic3d_earlyfusion_veh_pointpillars_67fe2b82320754481ef37f176b647e43.pth"
if [[ -s "${early_ckpt}" ]]; then
  report OK "early-fusion PointPillars checkpoint: ${early_ckpt}"
else
  report WARN "early-fusion PointPillars checkpoint missing: ${early_ckpt}"
fi

if command -v conda >/dev/null 2>&1; then
  set +u
  eval "$(conda shell.bash hook)"
  if conda activate "${ENV_NAME}" >/dev/null 2>&1; then
    set -u
    if python - <<'PY' >/dev/null 2>&1
import torch, mmcv, mmdet, mmseg, mmdet3d
from mmdet3d.ops import Voxelization
assert torch.cuda.is_available()
PY
    then
      report OK "environment imports and CUDA smoke passed: ${ENV_NAME}"
    else
      report FAIL "environment import/CUDA smoke failed: ${ENV_NAME}"
    fi
  else
    set -u
    report FAIL "conda env missing or cannot activate: ${ENV_NAME}"
  fi
else
  report FAIL "conda command not available"
fi

echo
printf 'summary: OK=%d WARN=%d FAIL=%d\n' "${ok}" "${warn}" "${fail}"
if (( fail > 0 )); then
  exit 1
fi
