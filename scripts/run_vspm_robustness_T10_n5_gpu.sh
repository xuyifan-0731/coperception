#!/usr/bin/env bash
set -euo pipefail

cd /raid/xuyifan/v2x_code_ckpt
eval "$(conda shell.bash hook)"
conda activate Android-Lab

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CKPT="${CKPT:-runs/revision_vspm_sensitivity/T10_n5/checkpoints/ckpt_13000.pth}"
export RUN_NAME="${RUN_NAME:-T10_n5}"
export DEVICE="${DEVICE:-cuda}"
export T="${T:-10}"
export N="${N:-5}"
export MAX_SAMPLES="${MAX_SAMPLES:-256}"
export OUT="${OUT:-results/revision_plan/vspm_robustness_T10_n5_gpu.csv}"

bash scripts/run_vspm_robustness_eval.sh
