#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
ENV_NAME="${ENV_NAME:-Android-Lab}"
OPENDAIR_ROOT="${OPENDAIR_ROOT:-${ROOT}/external/DAIR-V2X}"

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

FAILED_DOWNLOADS=()

download_file() {
  local file_id="$1"
  local target="$2"
  mkdir -p "$(dirname "${target}")"
  if [[ -s "${target}" ]]; then
    echo "skip existing ${target}"
    return 0
  fi
  echo "downloading ${target}"
  if ! gdown --continue "${file_id}" -O "${target}"; then
    echo "failed: ${target}" >&2
    FAILED_DOWNLOADS+=("${target}")
  fi
}

download_file \
  "1BO5dbqmLjC3gTjvQTyfEjhIikFz2P_Om" \
  "${OPENDAIR_ROOT}/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_inf_pointpillars_596784ad6127866fcfb286301757c949.pth"

download_file \
  "1tY1sqQGGSaRoA8KDeIQPjcUZ20I82wTK" \
  "${OPENDAIR_ROOT}/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth"

download_file \
  "1mo8o1iIZ2fQHYBkpfmjyMnCqYKs3wQk7" \
  "${OPENDAIR_ROOT}/configs/vic3d/early-fusion-pointcloud/pointpillars/vic3d_earlyfusion_veh_pointpillars_67fe2b82320754481ef37f176b647e43.pth"

# Optional image baselines.
download_file \
  "1F0QSlsGQhtMd3Q66CcXgQJKZptERYhhk" \
  "${OPENDAIR_ROOT}/configs/vic3d/late-fusion-image/imvoxelnet/vic3d_latefusion_inf_imvoxelnet.pth"

download_file \
  "1dNupazp9t2D6mN8cs1ER8zuf3j9ZHNd6" \
  "${OPENDAIR_ROOT}/configs/vic3d/late-fusion-image/imvoxelnet/vic3d_latefusion_veh_imvoxelnet.pth"

# Optional FFNet checkpoint. The FFNet code itself is not included in OpenDAIRV2X.
download_file \
  "1eX2wZ7vSxq8y9lAyjHyrmBQ30qNHcFC6" \
  "${ROOT}/external/FFNet-VIC3D/ffnet_work_dir/work_dir_ffnet/epoch_40.pth"

if (( ${#FAILED_DOWNLOADS[@]} > 0 )); then
  echo "checkpoint download pass completed with failures:" >&2
  printf '  %s\n' "${FAILED_DOWNLOADS[@]}" >&2
  exit 4
fi

echo "checkpoint download pass completed"
