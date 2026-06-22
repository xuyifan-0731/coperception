#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

copy_ckpt() {
  local src="$1"
  local dst="$2"
  install -D -m 0644 "${ROOT}/${src}" "${ROOT}/${dst}"
  echo "restored ${dst}"
}

copy_ckpt "checkpoints/curated/v2xsim/all_agents/ckpt_38000.pth" \
  "model1/checkpoints_A/ckpt_38000.pth"

copy_ckpt "checkpoints/curated/v2xsim/leave_one_out/agent1_best_ckpt_27000.pth" \
  "model1/通用模型对比实验_baseline/checkpoints_2345_for_Agent1/ckpt_27000.pth"
copy_ckpt "checkpoints/curated/v2xsim/leave_one_out/agent2_best_ckpt_33000_h.pth" \
  "model1/通用模型对比实验_baseline/checkpoints_1345_for_Agent2/ckpt_33000_h.pth"
copy_ckpt "checkpoints/curated/v2xsim/leave_one_out/agent3_best_ckpt_35000_h.pth" \
  "model1/通用模型对比实验_baseline/checkpoints_1245_for_Agent3/ckpt_35000_h.pth"
copy_ckpt "checkpoints/curated/v2xsim/leave_one_out/agent4_best_ckpt_33000_h.pth" \
  "model1/通用模型对比实验_baseline/checkpoints_1235_for_Agent4/ckpt_33000_h.pth"
copy_ckpt "checkpoints/curated/v2xsim/leave_one_out/agent5_best_ckpt_44000_h.pth" \
  "model1/通用模型对比实验_baseline/checkpoints_1234_for_Agent5/ckpt_44000_h.pth"

copy_ckpt "checkpoints/curated/v2xsim/ablation1/A_ckpt_13000.pth" \
  "model1/消融实验-1-baseline/checkpoints_1_for_Agent1_A/ckpt_13000.pth"
copy_ckpt "checkpoints/curated/v2xsim/ablation1/B_ckpt_13000.pth" \
  "model1/消融实验-1-baseline/checkpoints_1_for_Agent1_B/ckpt_13000.pth"
copy_ckpt "checkpoints/curated/v2xsim/ablation1/C_final_ckpt_13000.pth" \
  "model1/消融实验-1-baseline/checkpoints_1_for_Agent1_C/ckpt_13000.pth"
copy_ckpt "checkpoints/curated/v2xsim/ablation1/D_ckpt_13000.pth" \
  "model1/消融实验-1-baseline/checkpoints_1_for_Agent1_D/ckpt_13000.pth"

copy_ckpt "checkpoints/curated/v2xsim/ablation2/A1_ckpt_13000.pth" \
  "model1/消融实验-2-A/checkpoints_1_for_Agent1_A1/ckpt_13000.pth"
copy_ckpt "checkpoints/curated/v2xsim/ablation2/A2_ckpt_13000.pth" \
  "model1/消融实验-2-A/checkpoints_1_for_Agent1_A2/ckpt_13000.pth"
copy_ckpt "checkpoints/curated/v2xsim/ablation2/A3_nearest_ckpt_13000.pth" \
  "model1/消融实验-2-A/checkpoints_1_for_Agent1_A3/ckpt_13000.pth"
copy_ckpt "checkpoints/curated/v2xsim/ablation2/A4_ckpt_13000.pth" \
  "model1/消融实验-2-A/checkpoints_1_for_Agent1_A4/ckpt_13000.pth"
copy_ckpt "checkpoints/curated/v2xsim/ablation2/ABaseline_ckpt_13000.pth" \
  "model1/消融实验-2-A/checkpoints_1_for_Agent1_ABaseline/ckpt_13000.pth"

copy_ckpt "checkpoints/curated/dair_v2x/our_method/T10_n5_ckpt_16000.pth" \
  "runs/dair_v2x_full/our_method_T10_n5/checkpoints/ckpt_16000.pth"
copy_ckpt "checkpoints/curated/dair_v2x/our_method/T10_n10_ckpt_15000.pth" \
  "runs/dair_v2x_full/our_method_T10_n10/checkpoints/ckpt_15000.pth"

copy_ckpt "checkpoints/curated/dair_v2x/mst_ablation/baseline_ckpt_16000.pth" \
  "runs/dair_v2x_full/mst_ablation/baseline/checkpoints/ckpt_16000.pth"
copy_ckpt "checkpoints/curated/dair_v2x/mst_ablation/bottleneck12_ckpt_16000.pth" \
  "runs/dair_v2x_full/mst_ablation/bottleneck12/checkpoints/ckpt_16000.pth"
copy_ckpt "checkpoints/curated/dair_v2x/mst_ablation/ds64_fp16_ckpt_16000.pth" \
  "runs/dair_v2x_full/mst_ablation/ds64_fp16/checkpoints/ckpt_16000.pth"
copy_ckpt "checkpoints/curated/dair_v2x/mst_ablation/gru_fp16_ckpt_16000.pth" \
  "runs/dair_v2x_full/mst_ablation/gru_fp16/checkpoints/ckpt_16000.pth"

copy_ckpt "checkpoints/curated/dair_v2x/sensitivity/T5_n5_ckpt_16000.pth" \
  "runs/dair_v2x_full/vspm_sensitivity/T5_n5/checkpoints/ckpt_16000.pth"
copy_ckpt "checkpoints/curated/dair_v2x/sensitivity/T10_n3_ckpt_16000.pth" \
  "runs/dair_v2x_full/vspm_sensitivity/T10_n3/checkpoints/ckpt_16000.pth"
copy_ckpt "checkpoints/curated/dair_v2x/sensitivity/T10_n15_ckpt_29000.pth" \
  "runs/dair_v2x_full/vspm_sensitivity/T10_n15/checkpoints/ckpt_29000.pth"
copy_ckpt "checkpoints/curated/dair_v2x/sensitivity/T20_n5_ckpt_28000.pth" \
  "runs/dair_v2x_full/vspm_sensitivity/T20_n5/checkpoints/ckpt_28000.pth"
copy_ckpt "checkpoints/curated/dair_v2x/sensitivity/T30_n5_ckpt_52000.pth" \
  "runs/dair_v2x_full/vspm_sensitivity/T30_n5/checkpoints/ckpt_52000.pth"
