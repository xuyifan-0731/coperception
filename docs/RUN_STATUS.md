# DLPCM Revision Run Status

Last updated: 2026-05-22 08:59 UTC

## Environment

- Working directory: `/raid/xuyifan/v2x_code_ckpt`
- Conda environment: `Android-Lab`
- PyTorch: `2.7.1+cu126`
- GPU detected by PyTorch: `NVIDIA GeForce RTX 4090`
- W&B mode: `offline`

## Completed Preparation

- Extracted local V2X-Sim data at `V2X-Sim-det/`.
- Verified local split structure for `train`, `val`, and `test` across `agent1` to `agent5`.
- Generated revision planning CSVs under `results/revision_plan/`.
- Added local-only utility modules for delay simulation, robustness perturbation, cache policy, and profiling.
- Added controlled smoke-run arguments to both training scripts:
  - `--max_steps`
  - `--max_val_batches`
- Pinned `requirements.txt` to `torch==2.7.1` to avoid installing CUDA 13 wheels that are incompatible with the current driver.

## Completed Checks

- `profile_vspm.py` completed on CUDA and wrote `results/revision_plan/vspm_profile.csv`.
- `run_scalability_profile.py` completed and wrote `results/revision_plan/scalability_profile.csv`.
- Baseline smoke training completed:
  - Script: `src/v2x_forecasting/train_bev.py`
  - Data: `data_smoke/V2X-Sim-det`
  - Run dir: `runs/smoke_baseline/`
- Compressed/MST smoke training completed:
  - Script: `src/v2x_forecasting/train_bev_compressed.py`
  - Data: `data_smoke/V2X-Sim-det`
  - Run dir: `runs/smoke_compressed/`
- Pipeline status reporting completed:
  - `scripts/report_pipeline_status.py`
  - `results/revision_plan/pipeline_status.md`
  - `results/revision_plan/pipeline_latest_metrics.csv`
  - `results/revision_plan/pipeline_checkpoints.csv`
- Paper table draft generation completed:
  - `scripts/make_revision_tables.py`
  - `results/revision_tables/revision_tables.tex`
- VSPM robustness evaluation script completed and smoke-tested:
  - `scripts/eval_vspm_robustness.py`
  - `scripts/run_vspm_robustness_eval.sh`
  - Smoke output: `results/revision_plan/vspm_robustness_smoke.csv`
- Full `T10_n5` GPU robustness evaluation completed:
  - `results/revision_plan/vspm_robustness_T10_n5_gpu.csv`
  - `results/revision_tables/robustness_table.csv`
  - `results/revision_figures/vspm_robustness.pdf`

## Active Full Run

The original full revision pipeline stopped after a CUDA OOM during `T20_n5`.
The recovery pipeline completed the remaining VSPM sensitivity jobs. The MST
ablation pipeline also completed all four variants.

Live working log:

- `docs/ACTIVE_WORK_LOG.md`

- Original full-pipeline PID file: `results/revision_plan/full_pipeline.pid`
- MST PID file: `results/revision_plan/run_mst_ablation.pid`
- Original launcher log: `results/revision_plan/full_pipeline.nohup.log`
- VSPM sensitivity log: `results/revision_plan/run_vspm_sensitivity.full.log`
- MST ablation log: `results/revision_plan/run_mst_ablation.full.log`
- VSPM run root: `runs/revision_vspm_sensitivity/`
- MST run root: `runs/revision_mst_ablation/`

Pipeline order and current state:

1. VSPM latency/profile refresh: completed.
2. Scalability/cache profile refresh: completed.
3. VSPM sensitivity training for `T` and rollout horizon settings: completed.
4. VSPM robustness evaluation: completed for `T10_n5`.
5. MST/system-state ablation training: completed.
6. Result collection into `results/summary_revision/`: refreshed.

Current observed stage:

- Active training job: none.
- Active revision processes: none.
- Latest MST final metrics:
  - `baseline`: IoU `0.7488`, Delta IoU `0.0545`, Dynamic IoU `0.3296`
  - `gru_fp16`: IoU `0.7462`, Delta IoU `0.0520`, Dynamic IoU `0.3075`
  - `ds64_fp16`: IoU `0.7200`, Delta IoU `0.0258`, Dynamic IoU `0.0914`
  - `bottleneck12`: IoU `0.7193`, Delta IoU `0.0250`, Dynamic IoU `0.0986`
- Current overview: `docs/CURRENT_WORKSPACE_OVERVIEW.md`

## Monitor Commands

```bash
cd /raid/xuyifan/v2x_code_ckpt
PYTHONPATH=src python scripts/report_pipeline_status.py
python scripts/make_revision_tables.py
MPLBACKEND=Agg python scripts/plot_revision_tables.py
```
