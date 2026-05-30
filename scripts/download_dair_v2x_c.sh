#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
MANIFEST="${MANIFEST:-${ROOT}/datasets/DAIR-V2X/dair_v2x_c_manifest.csv}"
OUT_DIR="${OUT_DIR:-${ROOT}/datasets/DAIR-V2X/raw}"
USE_FIXED_PARTIAL="${USE_FIXED_PARTIAL:-1}"
USE_LOCAL_PROXY="${USE_LOCAL_PROXY:-1}"
PROXY_URL="${PROXY_URL:-http://localhost:15002}"
mkdir -p "${OUT_DIR}"

if [[ "${USE_LOCAL_PROXY}" == "1" ]]; then
  export HTTP_PROXY="${HTTP_PROXY:-${PROXY_URL}}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${PROXY_URL}}"
  export http_proxy="${http_proxy:-${HTTP_PROXY}}"
  export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
fi

eval "$(conda shell.bash hook)"
conda activate Android-Lab

FAILED_DOWNLOADS=()

while IFS=, read -r subset part filename file_id required_for; do
  target="${OUT_DIR}/${filename}"
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] downloading ${filename} (${required_for})"
  if [[ -s "${target}" ]]; then
    echo "  skip existing ${target}"
    continue
  fi
  partial_target="${target}"
  if [[ "${USE_FIXED_PARTIAL}" == "1" ]]; then
    partial_target="${target}.part"
    shopt -s nullglob
    old_parts=("${target}"*.part)
    shopt -u nullglob
    if [[ ! -e "${partial_target}" && ${#old_parts[@]} -gt 0 ]]; then
      largest_part="${old_parts[0]}"
      for candidate in "${old_parts[@]}"; do
        if [[ "$(stat -c '%s' "${candidate}")" -gt "$(stat -c '%s' "${largest_part}")" ]]; then
          largest_part="${candidate}"
        fi
      done
      echo "  adopting partial ${largest_part} -> ${partial_target}"
      mv "${largest_part}" "${partial_target}"
    fi
  fi
  if gdown --continue "${file_id}" -O "${partial_target}"; then
    if [[ "${partial_target}" != "${target}" ]]; then
      mv "${partial_target}" "${target}"
    fi
  else
    echo "  failed ${target}" >&2
    FAILED_DOWNLOADS+=("${target}")
  fi
done < <(tail -n +2 "${MANIFEST}")

if (( ${#FAILED_DOWNLOADS[@]} > 0 )); then
  echo "DAIR-V2X download pass completed with failures:" >&2
  printf '  %s\n' "${FAILED_DOWNLOADS[@]}" >&2
  exit 4
fi
