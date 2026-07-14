# 260704 Runbook: Dataset / Baseline / Experiment Commands

This document maps each 260704 result group to environment, code, checkpoint, training command, evaluation command, and output CSV.

It assumes the external resources in `260704_EXTERNAL_RESOURCES.md` have already been prepared.

## 0. Common Setup

Clone the repository and checkout the uploaded branch:

```bash
git clone ssh://git@ssh.github.com:443/xuyifan-0731/coperception.git
cd coperception
git checkout workspace-upload-20260622
```

Restore split checkpoints:

```bash
cd release_packages/260704_experiment_artifacts
python restore_split_checkpoints.py
cd ../..
```

Expected local external roots:

```bash
export REPO_ROOT=$PWD
export DATA_ROOT=$REPO_ROOT/external/DATA
export LRCP_ROOT=$REPO_ROOT/external/LRCP
export DAIR_CODE_ROOT=$REPO_ROOT/external/DAIR-V2X
export TRAF_ROOT=/tmp/TraF-Align_partial
```

Expected dataset roots:

```bash
export V2XSIM_ROOT=$REPO_ROOT/datasets/V2X-Sim-2.0
export DAIR_ROOT=$REPO_ROOT/datasets/DAIR-V2X/cooperative-vehicle-infrastructure
export DAIR_SEQ_ROOT=$REPO_ROOT/datasets/DAIR-V2X-Seq/V2X-Seq-SPD
```

All output CSVs below correspond to uploaded files in:

```text
release_packages/260704_experiment_artifacts/<dataset>/<baseline>/results/
```

## 1. V2X-Sim

### 1.1 V2VNet Baseline

| Item | Setting |
|---|---|
| Environment | Original DLPCM/V2X-Sim paper environment. |
| Baseline code | Not rerun in this package; paper table source is uploaded. |
| Ours checkpoint | `01_v2xsim/V2VNet/checkpoints/ours/v2xsim_ours_prediction_model.pth` |
| Result source | `01_v2xsim/V2VNet/paper_sources/main_result.tex` and `main_result_disco.tex` |
| Output in 260704 | V2X-Sim V2VNet AP@0.5/AP@0.7 tables. |

Current package supports provenance checking, not full rerun, for V2VNet rows.

### 1.2 DiscoNet Baseline

| Item | Setting |
|---|---|
| Environment | Original DLPCM/V2X-Sim paper environment. |
| Baseline code | Not rerun in this package; paper table source is uploaded. |
| Ours checkpoint | `01_v2xsim/DiscoNet/checkpoints/ours/v2xsim_ours_prediction_model.pth` |
| Result source | `01_v2xsim/DiscoNet/paper_sources/main_result.tex` and `main_result_disco.tex` |
| Output in 260704 | V2X-Sim DiscoNet AP@0.5/AP@0.7 tables. |

Current package supports provenance checking, not full rerun, for DiscoNet rows.

### 1.3 DATA Baseline

| Item | Setting |
|---|---|
| Environment | DATA/OpenCOOD environment. |
| External code | `external/DATA` at commit `5df7eb6f5659db0d6809fa3cc218aa425bc287b4` plus local patches. |
| Dataset | V2X-Sim 2.0 full dataset and info files. |
| Baseline checkpoint | `01_v2xsim/DATA/checkpoints/baseline_DATA/net_epoch_bestval_at19.pth` after restoring split parts. |
| Ours checkpoint | `01_v2xsim/DATA/checkpoints/ours/v2xsim_data_ifam_joint_delay_comp_scale01.pth` for main joint IFAM residual result. |

Run DATA no-comp / DATA + pose-warp / SyncNet strict grid:

```bash
python scripts/run_data_v2xsim_compensation_grid.py \
  --methods ours,syncnet \
  --comps 0,2,4 \
  --comms 0,1,2,3,4,5 \
  --gpus 0,1,2,3 \
  --jobs 4 \
  --run-missing \
  --out results/260615_fill/artifacts/v2xsim_data_compensation_ap.csv
```

Train V2X-Sim DATA + Ours joint IFAM residual:

```bash
python scripts/run_v2xsim_data_joint_delay_comp.py train \
  --model-dir external/DATA/opencood/logs/DATA_v2xsim_pipeline_2026_06_10_14_27_14 \
  --info-dir datasets/v2xsim2_info \
  --input-compensation ours \
  --residual-scale 0.1 \
  --ckpt checkpoints/curated/ours_models/v2xsim_data_ifam_joint_delay_comp_scale01.pth
```

Evaluate V2X-Sim DATA + Ours joint IFAM residual:

```bash
python scripts/run_v2xsim_data_joint_delay_comp.py eval \
  --model-dir external/DATA/opencood/logs/DATA_v2xsim_pipeline_2026_06_10_14_27_14 \
  --info-dir datasets/v2xsim2_info \
  --input-compensation ours \
  --ckpt checkpoints/curated/ours_models/v2xsim_data_ifam_joint_delay_comp_scale01.pth \
  --out results/260615_fill/artifacts/v2xsim_data_ifam_joint_delay_comp_scale01_20260713.csv
```

Uploaded result files:

```text
01_v2xsim/DATA/results/data_v2xsim_latency_ap.csv
01_v2xsim/DATA/results/v2xsim_data_compensation_ap.csv
01_v2xsim/DATA/results/v2xsim_data_syncnet_strict_20260704.csv
01_v2xsim/DATA/results/v2xsim_data_intermediate_fairness_audit_20260707.csv
01_v2xsim/DATA/results/v2xsim_data_ifam_joint_delay_comp_scale01_20260713.csv
```

## 2. DAIR-V2X-C

### 2.1 Official Late Fusion / TCLF Baseline

| Item | Setting |
|---|---|
| Environment | Patched OpenDAIRV2X / official detector environment. |
| External code | `external/DAIR-V2X` commit `c885c54af0c34bc515fa9ca8b5e8fda76a15462c` plus local patches and isolated GPU copies. |
| Dataset | DAIR-V2X-C cooperative vehicle-infrastructure data. |
| Main Ours checkpoint | `02_dair_v2x_c/official_late_fusion/checkpoints/ours_rf2/dair_v2x_late_fusion_rf2_ours_trainonly.pkl` after restoring split parts. |
| Diagnostic checkpoints | `dair_v2x_late_fusion_rf2_ours_trainval_diagnostic.pkl`, `dair_v2x_late_fusion_rf2_ours.pkl`. |

Train strict RF 2-frame detector-side predictor:

```bash
python scripts/train_dair_history_motion_compensator.py \
  --dair-root datasets/DAIR-V2X/cooperative-vehicle-infrastructure \
  --split-json external/DAIR-V2X/data/split_datas/cooperative-split-data.json \
  --train-split train \
  --val-split val \
  --history-frames 2 \
  --history-stride consecutive \
  --model-type rf \
  --out checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainonly.pkl
```

Run strict no-comp, Ours, and TCLF grids:

```bash
python scripts/run_dair_strict_dual_delay_grid.py \
  --methods none,ours,tclf_receiver \
  --comps 0,2,4 \
  --comms 0,1,2,3,4,5 \
  --ours-model checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainonly.pkl \
  --ours-temporal-source history_model \
  --ours-history-frames 2 \
  --ours-history-stride consecutive \
  --gpus 0,1,2,3,4,6 \
  --jobs 6 \
  --run-missing \
  --late-out results/260615_fill/artifacts/dair_late_fusion_ours_history2_rf_trainonly_strict_20260714.csv \
  --detector-out results/260615_fill/artifacts/dair_detector_ours_history2_rf_trainonly_strict_20260714.csv
```

Uploaded result files:

```text
02_dair_v2x_c/official_late_fusion/results/dair_late_fusion_no_comp_strict_20260704_rechecked.csv
02_dair_v2x_c/official_late_fusion/results/dair_late_fusion_ours_history2_rf_trainonly_strict_20260714.csv
02_dair_v2x_c/official_late_fusion/results/dair_late_fusion_tclf_receiver_strict_20260704_rechecked.csv
02_dair_v2x_c/official_late_fusion/results/dair_detector_*.csv
02_dair_v2x_c/official_late_fusion/logs/*
```

### 2.2 DATA Baseline On DAIR-V2X-C

| Item | Setting |
|---|---|
| Environment | DATA/OpenCOOD environment. |
| External code | `external/DATA` commit `5df7eb6f5659db0d6809fa3cc218aa425bc287b4` plus local patches. |
| Dataset | DAIR-V2X-C. |
| Baseline checkpoint | `02_dair_v2x_c/DATA/checkpoints/baseline_DATA/net_epoch_bestval_at21.pth` after restoring split parts. |
| Ours feature checkpoint | `02_dair_v2x_c/DATA/checkpoints/ours/dair_data_ifam_feature_t10n10_ours.npz`. |
| Ours joint checkpoint | `02_dair_v2x_c/DATA/checkpoints/ours/dair_data_ifam_joint_delay_comp.pth`. |

Run DATA strict no-comp and DATA fork pose-warp:

```bash
python scripts/run_dair_data_strict_dual_delay_grid.py \
  --model-dir runs/data_dair_official_eval \
  --methods none,ours \
  --comps 0,2,4 \
  --comms 0,1,2,3,4,5 \
  --gpus 0,1,2,3 \
  --jobs 4 \
  --run-missing \
  --out results/260615_fill/artifacts/dair_data_strict_dual_delay_none_ours_20260706.csv
```

Regenerate DAIR DATA IFAM feature cache:

```bash
python scripts/run_dair_data_feature_t10n10_comp.py cache-features \
  --model-dir runs/data_dair_official_eval \
  --dair-root datasets/DAIR-V2X/cooperative-vehicle-infrastructure \
  --cache-dir results/260615_fill/cache/dair_data_feature_t10n10 \
  --num-shards 1 \
  --shard-index 0
```

Train DAIR DATA IFAM feature T10-N10 predictor:

```bash
python scripts/run_dair_data_feature_t10n10_comp.py train-predictor \
  --cache-dir results/260615_fill/cache/dair_data_feature_t10n10 \
  --history 10 \
  --max-horizon 10 \
  --out checkpoints/curated/ours_models/dair_data_ifam_feature_t10n10_ours.npz
```

Evaluate DAIR DATA IFAM feature T10-N10 predictor:

```bash
python scripts/run_dair_data_feature_t10n10_comp.py eval-grid \
  --model-dir runs/data_dair_official_eval \
  --dair-root datasets/DAIR-V2X/cooperative-vehicle-infrastructure \
  --cache-dir results/260615_fill/cache/dair_data_feature_t10n10 \
  --predictor checkpoints/curated/ours_models/dair_data_ifam_feature_t10n10_ours.npz \
  --out results/260615_fill/artifacts/dair_data_ifam_feature_t10n10_ours_20260707.csv
```

Train DAIR DATA IFAM joint delay-comp:

```bash
python scripts/run_dair_data_joint_delay_comp.py train \
  --model-dir runs/data_dair_official_eval \
  --dair-root datasets/DAIR-V2X/cooperative-vehicle-infrastructure \
  --cache-dir results/260615_fill/cache/dair_data_feature_t10n10 \
  --train-split train \
  --max-steps 1500 \
  --residual-scale 0.5 \
  --gate-init -2.0 \
  --ckpt checkpoints/curated/ours_models/dair_data_ifam_joint_delay_comp.pth
```

Evaluate DAIR DATA IFAM joint delay-comp:

```bash
python scripts/run_dair_data_joint_delay_comp.py eval \
  --model-dir runs/data_dair_official_eval \
  --dair-root datasets/DAIR-V2X/cooperative-vehicle-infrastructure \
  --cache-dir results/260615_fill/cache/dair_data_feature_t10n10 \
  --ckpt checkpoints/curated/ours_models/dair_data_ifam_joint_delay_comp.pth \
  --out results/260615_fill/artifacts/dair_data_ifam_joint_delay_comp_20260713.csv
```

DATA box-level RF diagnostic:

```bash
python scripts/run_dair_data_rf_ours_box_comp_grid.py \
  --model-dir runs/data_dair_official_eval \
  --rf-model checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainonly.pkl \
  --mode both \
  --eval-jobs 6 \
  --out results/260615_fill/artifacts/dair_data_box_latefusion_fair_rf_trainonly_20260714.csv
```

DATA intermediate + RF assist diagnostic:

```bash
python scripts/run_dair_data_hybrid_rf_assist.py \
  --model-dir runs/data_dair_official_eval \
  --rf-model checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainonly.pkl \
  --comps 0,2 \
  --comms 0 \
  --policies concat_nms,add_unmatched,replace_matched \
  --rf-score-scales 0.3,0.5,0.7,1.0 \
  --out results/260615_fill/artifacts/dair_data_intermediate_rf_assist_trainonly_keycells_20260714.csv
```

Uploaded result files are under:

```text
02_dair_v2x_c/DATA/results/
```

### 2.3 LRCP Baseline On DAIR-V2X-C

| Item | Setting |
|---|---|
| Environment | LRCP/OpenCOOD environment with deformable attention ops and spconv 1.2.1. |
| External code | `external/LRCP` commit `1eb3793b9befd9d36dd8705e461d895a41b39d33` plus local DAIR patches. |
| Dataset | DAIR-V2X-C. |
| Baseline checkpoint | `02_dair_v2x_c/LRCP/checkpoints/baseline_LRCP_stage2/net_epoch_bestval_at1.pth` after restoring split parts. |

Prepare DAIR inputs for LRCP:

```bash
python scripts/prepare_dair_lrcp_inputs.py \
  --dair-root datasets/DAIR-V2X/cooperative-vehicle-infrastructure
```

Generate DAIR LRCP flow offsets if public flow files are not used:

```bash
python scripts/generate_dair_lrcp_flow_offsets.py \
  --dair-root datasets/DAIR-V2X/cooperative-vehicle-infrastructure
```

Run LRCP strict dual-delay grid:

```bash
python scripts/run_dair_lrcp_strict_dual_delay_parallel.py \
  --model-dir external/LRCP/opencood/logs/dairv2x_lrcp_point_pillar_lidar_deformable_attn_stage2_2026_07_01_19_32_46 \
  --methods none,ours \
  --comps 0,2,4 \
  --comms 0,1,2,3,4,5 \
  --gpus 0,1,2,3 \
  --jobs 4 \
  --out results/260615_fill/artifacts/dair_lrcp_strict_dual_delay_ap.csv
```

Uploaded result files:

```text
02_dair_v2x_c/LRCP/results/dair_lrcp_strict_dual_delay_ap.csv
02_dair_v2x_c/LRCP/results/dair_lrcp_strict_dual_delay_parts/*.csv
02_dair_v2x_c/LRCP/results/dair_lrcp_strict_dual_delay_parts/*.log
```

## 3. DAIR-V2X-Seq / TraF-Align

| Item | Setting |
|---|---|
| Environment | TraF-Align environment, but local successful reproduction used `spconv-cu113==2.1.21` and `cumm-cu113==0.2.9`. |
| External code | `/tmp/TraF-Align_partial` commit `0f6f98de60dcb86f29f4eb99ed871054b05bf280` plus local patches. |
| Dataset | DAIR-V2X-Seq / V2X-Seq-SPD. |
| Baseline checkpoint | `03_dair_v2x_seq/TraF-Align/checkpoints/baseline_official_trafalign/net_epoch60.pth` after restoring split parts. |
| Ours feature checkpoint | `03_dair_v2x_seq/TraF-Align/checkpoints/ours_feature_t10n10/dair_v2x_seq_trafalign_feature_t10n10_ours_spconv2121.npz`. |
| Ours joint checkpoints | `03_dair_v2x_seq/TraF-Align/checkpoints/ours_joint_delay_comp/*/net_epoch1.pth` after restoring split parts. |

Run official TraF-Align baseline strict dual-delay by comp group:

```bash
PYTHONPATH=/tmp/trafalign_spconv_2121:$PYTHONPATH \
python scripts/run_trafalign_dair_v2x_seq_eval.py \
  --data-root datasets/DAIR-V2X-Seq/V2X-Seq-SPD \
  --ckpt-dir /tmp/TraF-Align_partial/checkpoints/dair_v2x_seq_Trafalign_2024_12_28_04_42_36 \
  --delay-ego-ms 0 \
  --delay-ms-list 0,100,200,300,400,500 \
  --out results/dair_v2x_seq/trafalign_baseline_comp0_spconv2121.csv

PYTHONPATH=/tmp/trafalign_spconv_2121:$PYTHONPATH \
python scripts/run_trafalign_dair_v2x_seq_eval.py \
  --data-root datasets/DAIR-V2X-Seq/V2X-Seq-SPD \
  --ckpt-dir /tmp/TraF-Align_partial/checkpoints/dair_v2x_seq_Trafalign_2024_12_28_04_42_36 \
  --delay-ego-ms 200 \
  --delay-ms-list 0,100,200,300,400,500 \
  --out results/dair_v2x_seq/trafalign_baseline_comp2_spconv2121.csv

PYTHONPATH=/tmp/trafalign_spconv_2121:$PYTHONPATH \
python scripts/run_trafalign_dair_v2x_seq_eval.py \
  --data-root datasets/DAIR-V2X-Seq/V2X-Seq-SPD \
  --ckpt-dir /tmp/TraF-Align_partial/checkpoints/dair_v2x_seq_Trafalign_2024_12_28_04_42_36 \
  --delay-ego-ms 400 \
  --delay-ms-list 0,100,200,300,400,500 \
  --out results/dair_v2x_seq/trafalign_baseline_comp4_spconv2121.csv
```

Regenerate TraF-Align feature cache:

```bash
PYTHONPATH=/tmp/trafalign_spconv_2121:$PYTHONPATH \
python scripts/train_trafalign_feature_t10n10_predictor.py cache-features \
  --data-root datasets/DAIR-V2X-Seq/V2X-Seq-SPD \
  --ckpt-dir /tmp/TraF-Align_partial/checkpoints/dair_v2x_seq_Trafalign_2024_12_28_04_42_36 \
  --cache-dir results/dair_v2x_seq/trafalign_feature_cache_spconv2121 \
  --split train
```

Train TraF-Align feature T10-N10 predictor:

```bash
PYTHONPATH=/tmp/trafalign_spconv_2121:$PYTHONPATH \
python scripts/train_trafalign_feature_t10n10_predictor.py train \
  --cache-dir results/dair_v2x_seq/trafalign_feature_cache_spconv2121 \
  --history 10 \
  --max-horizon 10 \
  --out checkpoints/curated/ours_models/dair_v2x_seq_trafalign_feature_t10n10_ours_spconv2121.npz
```

Evaluate TraF-Align + Ours feature T10-N10:

```bash
PYTHONPATH=/tmp/trafalign_spconv_2121:$PYTHONPATH \
python scripts/run_trafalign_dair_v2x_seq_eval.py \
  --data-root datasets/DAIR-V2X-Seq/V2X-Seq-SPD \
  --ckpt-dir /tmp/TraF-Align_partial/checkpoints/dair_v2x_seq_Trafalign_2024_12_28_04_42_36 \
  --ours-predictor checkpoints/curated/ours_models/dair_v2x_seq_trafalign_feature_t10n10_ours_spconv2121.npz \
  --feature-cache results/dair_v2x_seq/trafalign_feature_cache_spconv2121 \
  --delay-ego-ms 200 \
  --delay-ms-list 0,100,200,300,400,500 \
  --out results/dair_v2x_seq/trafalign_ours_t10n10_comp2_spconv2121.csv
```

Train TraF-Align + Ours joint delay-comp:

```bash
PYTHONPATH=/tmp/trafalign_spconv_2121:$PYTHONPATH \
python scripts/train_trafalign_joint_delay_comp.py \
  --data-root datasets/DAIR-V2X-Seq/V2X-Seq-SPD \
  --pretrained-dir /tmp/TraF-Align_partial/checkpoints/dair_v2x_seq_Trafalign_2024_12_28_04_42_36 \
  --pretrained-epoch 60 \
  --train-only-compensator \
  --freeze-reader-backbone \
  --freeze-bn \
  --max-steps 1000 \
  --residual-scale 0.2 \
  --gate-init -4.0 \
  --out-dir checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_spconv2121
```

Evaluate TraF-Align + Ours joint delay-comp / adaptive / Comp0 bypass:

```bash
PYTHONPATH=/tmp/trafalign_spconv_2121:$PYTHONPATH \
python scripts/run_trafalign_dair_v2x_seq_eval.py \
  --data-root datasets/DAIR-V2X-Seq/V2X-Seq-SPD \
  --ckpt-dir checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_adaptive2to4_bypass_comp0_spconv2121 \
  --config checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_adaptive2to4_bypass_comp0_spconv2121/config.yaml \
  --delay-ego-ms 200 \
  --delay-ms-list 0,100,200,300,400,500 \
  --out results/dair_v2x_seq/trafalign_joint_delay_comp_only_adaptive2to4_comp2_spconv2121.csv
```

Uploaded result files:

```text
03_dair_v2x_seq/TraF-Align/results/trafalign_baseline_comp*.csv
03_dair_v2x_seq/TraF-Align/results/trafalign_ours_t10n10_comp*.csv
03_dair_v2x_seq/TraF-Align/results/trafalign_joint_delay_comp_only_*.csv
03_dair_v2x_seq/TraF-Align/results/trafalign_joint_delay_comp_only_adaptive2to4_bypass_comp0_spconv2121.csv
```

## 4. Regenerating Large Generated Data

### 4.1 DAIR DATA IFAM Feature Cache

Purpose: required by DAIR DATA IFAM T10-N10 and joint delay-comp training/eval.

```bash
python scripts/run_dair_data_feature_t10n10_comp.py cache-features \
  --model-dir runs/data_dair_official_eval \
  --dair-root datasets/DAIR-V2X/cooperative-vehicle-infrastructure \
  --cache-dir results/260615_fill/cache/dair_data_feature_t10n10
```

This cache was about 310G locally and is intentionally excluded from GitHub.

### 4.2 TraF-Align Feature Cache

Purpose: required by TraF-Align feature T10-N10 predictor.

```bash
PYTHONPATH=/tmp/trafalign_spconv_2121:$PYTHONPATH \
python scripts/train_trafalign_feature_t10n10_predictor.py cache-features \
  --data-root datasets/DAIR-V2X-Seq/V2X-Seq-SPD \
  --cache-dir results/dair_v2x_seq/trafalign_feature_cache_spconv2121 \
  --ckpt-dir /tmp/TraF-Align_partial/checkpoints/dair_v2x_seq_Trafalign_2024_12_28_04_42_36 \
  --split train
```

This cache was about 31G locally and is intentionally excluded from GitHub.

### 4.3 DAIR DATA Intermediate Box Cache

Purpose: DATA intermediate + RF assist diagnostic.

```bash
python scripts/run_dair_data_hybrid_rf_assist.py \
  --model-dir runs/data_dair_official_eval \
  --rf-model checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainonly.pkl \
  --cache-only \
  --data-cache-dir results/260615_fill/cache/dair_data_intermediate_boxes_20260707
```

### 4.4 DAIR DATA RF Box Cache

Purpose: DATA box-level RF diagnostic.

```bash
python scripts/run_dair_data_rf_ours_box_comp_grid.py \
  --model-dir runs/data_dair_official_eval \
  --rf-model checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainonly.pkl \
  --force-cache \
  --cache-dir results/260615_fill/cache/dair_data_rf_ours_20260706
```

### 4.5 LRCP DAIR Flow Offsets

Purpose: LRCP DAIR adapter when official flow files are unavailable.

```bash
python scripts/generate_dair_lrcp_flow_offsets.py \
  --dair-root datasets/DAIR-V2X/cooperative-vehicle-infrastructure
```

## 5. Result Verification

Use `MANIFEST.csv` to verify uploaded file integrity. For restored split checkpoints, use:

```bash
cd release_packages/260704_experiment_artifacts
python restore_split_checkpoints.py
```

The restore script verifies original sizes and SHA256 values from `CHECKPOINT_SPLIT_MANIFEST.csv`.

For numeric result verification, compare regenerated CSVs against the uploaded package CSVs under each dataset/baseline folder.
