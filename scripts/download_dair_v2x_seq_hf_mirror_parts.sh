#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LIST="${LIST:-${ROOT}/results/dair_v2x_seq/hf_spd_parts_files.txt}"
OUT_DIR="${OUT_DIR:-${ROOT}/datasets/DAIR-V2X-Seq/hf_mirror}"
ENDPOINT="${ENDPOINT:-https://hf-mirror.com}"
REPO="${REPO:-apdoa/DAIR-V2X}"
LOG_EVERY="${LOG_EVERY:-1}"

if [[ ! -s "${LIST}" ]]; then
  echo "missing file list: ${LIST}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

urlencode_path() {
  python - "$1" <<'PY'
import sys
from urllib.parse import quote
print(quote(sys.argv[1], safe="/"))
PY
}

total="$(grep -cve '^[[:space:]]*$' "${LIST}")"
idx=0
while IFS= read -r rel; do
  [[ -n "${rel}" ]] || continue
  idx=$((idx + 1))
  target="${OUT_DIR}/${rel}"
  part="${target}.part"
  mkdir -p "$(dirname "${target}")"

  if [[ -s "${target}" ]]; then
    echo "[$idx/$total] skip existing ${rel}"
    continue
  fi

  encoded="$(urlencode_path "${rel}")"
  url="${ENDPOINT}/datasets/${REPO}/resolve/main/${encoded}?download=true"
  echo "[$idx/$total] downloading ${rel}"
  curl \
    --fail \
    --location \
    --continue-at - \
    --retry 12 \
    --retry-delay 5 \
    --connect-timeout 30 \
    --speed-time 300 \
    --speed-limit 1024 \
    --output "${part}" \
    "${url}"
  mv "${part}" "${target}"
done < "${LIST}"

echo "download pass completed: ${OUT_DIR}"
