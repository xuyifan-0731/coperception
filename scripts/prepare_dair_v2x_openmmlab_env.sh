#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-dair-v2x-openmmlab}"
BASE_ENV="${BASE_ENV:-v2xverse}"

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "conda env already exists: ${ENV_NAME}"
else
  if conda env list | awk '{print $1}' | grep -qx "${BASE_ENV}"; then
    echo "cloning ${BASE_ENV} -> ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" --clone "${BASE_ENV}"
  else
    echo "creating ${ENV_NAME} with Python 3.8"
    conda create -y -n "${ENV_NAME}" python=3.8
  fi
fi

set +u
conda activate "${ENV_NAME}"
set -u
python -m pip install -U pip setuptools wheel
python -m pip install \
  "mmcv-full==1.3.18" \
  -f "https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html"
python -m pip install "terminaltables" "prettytable" "pycocotools==2.0.7"
python -m pip install --no-deps "mmdet==2.14.0" "mmsegmentation==0.14.1" "mmdet3d==0.17.1"

python - <<'PY'
mods = ["torch", "mmcv", "mmdet", "mmseg", "mmdet3d", "spconv", "pypcd"]
for mod_name in mods:
    mod = __import__(mod_name)
    print(mod_name, getattr(mod, "__version__", "ok"))
PY

echo "OpenDAIRV2X environment prepared: ${ENV_NAME}"
