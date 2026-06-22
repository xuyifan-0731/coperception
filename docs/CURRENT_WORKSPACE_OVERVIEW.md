# DLPCM Workspace Overview

Last updated: 2026-05-22 08:59 UTC

Scope constraint: this summary only covers files under `/raid/xuyifan/v2x_code_ckpt`.

## 1. Workspace Layout

```text
/raid/xuyifan/v2x_code_ckpt/
├── DLPCM/                     # Original paper assets/reference materials supplied by user.
├── V2X-Sim-det/               # Extracted V2X-Sim dataset; active data root.
│   ├── train/agent1..agent5
│   ├── val/agent1..agent5
│   └── test/agent1..agent5
├── data_smoke/                # Small smoke-test dataset.
├── docs/                      # Revision notes, run status, TODOs, and this overview.
├── model1/                    # Existing older/local experiment artifacts.
├── results/                   # Generated CSV summaries, paper draft tables, and figures.
├── runs/                      # Training outputs/checkpoints for smoke, sensitivity, and MST runs.
├── scripts/                   # Experiment launchers, status collectors, profilers, plotting utilities.
├── src/v2x_forecasting/       # Local Python package with VSPM models/utilities.
└── wandb/                     # Offline W&B run directories.
```

Current disk footprint:

| path | approximate size | role |
|---|---:|---|
| `V2X-Sim-det/` | 73G | Main extracted dataset |
| `wandb/` | 184M | Offline W&B logs |
| `runs/` | 72M | Checkpoints and run logs |
| `results/` | 52M | Processed summaries, figures, tables |
| `DLPCM/` | 2.8M | Original paper/reference directory |
| `src/` | 92K | Source code |
| `scripts/` | 168K | Experiment scripts |
| `docs/` | 108K | Documentation |

## 2. Important Code Locations

Core model/training:

| file | purpose |
|---|---|
| `src/v2x_forecasting/train_bev.py` | Baseline BEV/VSPM forecasting training and eval script. Added smoke controls. |
| `src/v2x_forecasting/train_bev_compressed.py` | Compressed/MST variants: baseline, GRU FP16, DS64 FP16, bottleneck. Added smoke controls. |
| `src/v2x_forecasting/profiling.py` | Shared parameter/state-size/cache-memory profiling helpers. |
| `src/v2x_forecasting/cache.py` | LRU-style receiver cache simulation helpers. |
| `src/v2x_forecasting/robustness.py` | Packet loss, jitter, BEV degradation, and pose-noise utility functions. |
| `src/v2x_forecasting/delay_simulator.py` | Latency/grid simulation support utilities. |

Experiment launchers:

| file | purpose |
|---|---|
| `scripts/run_revision_full_pipeline.sh` | Sequential revision pipeline launcher. Original run OOMed at `T20_n5` batch 16. |
| `scripts/run_vspm_sensitivity.sh` | VSPM history/rollout sweep launcher. |
| `scripts/recover_vspm_sensitivity_after_oom.sh` | Recovery launcher for sensitivity jobs after OOM, using smaller batch sizes. |
| `scripts/run_mst_ablation.sh` | MST/system-state ablation launcher for `baseline`, `gru_fp16`, `ds64_fp16`, `bottleneck12`. |
| `scripts/run_vspm_robustness_eval.sh` | Robustness evaluation wrapper with CPU thread limits. |
| `scripts/run_vspm_robustness_T10_n5_gpu.sh` | Dedicated GPU robustness launcher for `T10_n5`. |

Analysis/reporting:

| file | purpose |
|---|---|
| `scripts/report_pipeline_status.py` | Parses active/stopped status, latest metrics, and checkpoint lists. |
| `scripts/collect_results.py` | Collects W&B-style logs and metric CSVs into `results/summary_revision/`. |
| `scripts/make_revision_tables.py` | Builds CSV and LaTeX table drafts, including robustness table. |
| `scripts/plot_revision_tables.py` | Builds PDF figures for training, state profile, scalability, robustness. |
| `scripts/profile_vspm.py` | Measures/estimates VSPM variant state size and synthetic rollout latency. |
| `scripts/run_scalability_profile.py` | Synthetic N-agent cache/state-memory profile. |
| `scripts/eval_vspm_robustness.py` | VSPM-level robustness evaluation under packet loss and BEV perturbations. |
| `scripts/prepare_revision_experiments.py` | Generates revision planning grids. |
| `scripts/inspect_v2xsim_zip.py` | Dataset archive inventory helper. |

Key documentation:

| file | role |
|---|---|
| `docs/ACTIVE_WORK_LOG.md` | Dynamic working log. |
| `docs/RUN_STATUS.md` | Current run status and monitor commands. |
| `docs/EXPERIMENT_PREP.md` | Experiment preparation and run commands. |
| `docs/DLPCM_REVISION_TODO.md` / `docs/REVISION_TODO.md` | Revision TODO planning. |
| `docs/RESOURCE_INVENTORY.md` | Local resource inventory. |

## 3. Completed Code Changes

- Added controlled smoke-run options:
  - `--max_steps`
  - `--max_val_batches`
  - in `src/v2x_forecasting/train_bev.py`
  - in `src/v2x_forecasting/train_bev_compressed.py`
- Added local utilities for reviewer-requested evidence:
  - robustness perturbations: `src/v2x_forecasting/robustness.py`
  - cache policy/profile helpers: `src/v2x_forecasting/cache.py`, `src/v2x_forecasting/profiling.py`
  - delay/planning utilities: `src/v2x_forecasting/delay_simulator.py`
- Added launchers for:
  - VSPM sensitivity sweeps
  - OOM recovery
  - MST ablation
  - robustness evaluation
  - full revision pipeline
- Added reporting/generation scripts:
  - status parser
  - result collector
  - revision table generator
  - revision figure generator
- Added CPU thread controls to robustness evaluation:
  - `CPU_THREADS=16` default in `scripts/run_vspm_robustness_eval.sh`
  - `--torch-threads` in `scripts/eval_vspm_robustness.py`
- Added robustness outputs into paper-facing table/figure generation:
  - `results/revision_tables/robustness_table.csv`
  - `results/revision_figures/vspm_robustness.pdf`

## 4. Completed Experiment Results

### 4.1 VSPM Sensitivity

All planned VSPM sensitivity jobs completed.

| run | epoch | final step | final IoU | final Delta IoU | final Dynamic IoU |
|---|---:|---:|---:|---:|---:|
| `T5_n5` | 30 | 13800 | 0.7505 | 0.0540 | 0.3204 |
| `T10_n5` | 30 | 13050 | 0.7513 | 0.0570 | 0.3356 |
| `T20_n5` | 30 | 23100 | 0.7496 | 0.0581 | 0.3297 |
| `T30_n5` | 30 | 40200 | 0.7518 | 0.0597 | 0.3326 |
| `T10_n3` | 30 | 13350 | 0.7678 | 0.0558 | 0.3413 |
| `T10_n10` | 30 | 24600 | 0.7243 | 0.0559 | 0.3332 |
| `T10_n15` | 30 | 46200 | 0.7075 | 0.0549 | 0.3109 |

Main artifacts:

- `runs/revision_vspm_sensitivity/*/checkpoints/`
- `results/summary_revision/training_summary.csv`
- `results/summary_revision/training_timeseries.csv`
- `results/revision_tables/active_training_table.csv`
- `results/revision_figures/vspm_training_iou.pdf`

### 4.2 MST/System-State Ablation

All four MST ablation variants completed to epoch 30.

| run | epoch | final step | final IoU | final Delta IoU | final Dynamic IoU | best IoU |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 30 | 13050 | 0.7488 | 0.0545 | 0.3296 | 0.7495 |
| `gru_fp16` | 30 | 13050 | 0.7462 | 0.0520 | 0.3075 | 0.7469 |
| `ds64_fp16` | 30 | 13050 | 0.7200 | 0.0258 | 0.0914 | 0.7200 |
| `bottleneck12` | 30 | 13050 | 0.7193 | 0.0250 | 0.0986 | 0.7213 |

Main artifacts:

- `runs/revision_mst_ablation/*/checkpoints/`
- `runs/revision_mst_ablation/*/checkpoints/wandb_logs.txt`
- `results/revision_plan/pipeline_latest_metrics.csv`
- `results/revision_plan/pipeline_checkpoints.csv`

### 4.3 VSPM Profile

Output: `results/revision_tables/vspm_profile_table.csv`

| model | params | state MiB/agent | mean rollout ms |
|---|---:|---:|---:|
| `convlstm` | 85569 | 16.000 | 3.590 |
| `compressed_lstm` | 85569 | 16.000 | 3.632 |
| `gru_fp16` | 66497 | 4.000 | 4.400 |
| `ds64_fp16` | 66497 | 0.250 | 4.209 |
| `bottleneck12` | 67309 | 0.250 | 4.314 |

Main artifacts:

- `results/revision_plan/vspm_profile.csv`
- `results/revision_tables/vspm_profile_table.csv`
- `results/revision_figures/vspm_profile_tradeoff.pdf`

### 4.4 N-Agent Cache/Scalability

Output: `results/revision_tables/scalability_table.csv`

| agents | ConvLSTM h+c MiB | GRU DS64 FP16 MiB | cache capacity | evictions | hit rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 16.00 | 0.25 | 8 | 0 | 1.00 |
| 2 | 32.00 | 0.50 | 8 | 0 | 1.00 |
| 3 | 48.00 | 0.75 | 8 | 0 | 1.00 |
| 4 | 64.00 | 1.00 | 8 | 0 | 1.00 |
| 5 | 80.00 | 1.25 | 8 | 0 | 1.00 |
| 8 | 128.00 | 2.00 | 8 | 0 | 1.00 |
| 16 | 256.00 | 4.00 | 8 | 8 | 0.50 |

Main artifacts:

- `results/revision_plan/scalability_profile.csv`
- `results/revision_tables/scalability_table.csv`
- `results/revision_figures/cache_scalability.pdf`

### 4.5 Robustness Evaluation

Completed for `T10_n5` checkpoint on 256 samples.

| condition | mean IoU | baseline IoU | Delta IoU | Dynamic IoU |
|---|---:|---:|---:|---:|
| `clean` | 0.7884 | 0.7366 | 0.0517 | 0.3858 |
| `packet_loss_0.1` | 0.7849 | 0.7356 | 0.0493 | 0.3734 |
| `packet_loss_0.2` | 0.7813 | 0.7347 | 0.0466 | 0.3588 |
| `packet_loss_0.3` | 0.7765 | 0.7334 | 0.0431 | 0.3409 |
| `packet_loss_0.5` | 0.7652 | 0.7293 | 0.0359 | 0.2871 |
| `dropout_0.05` | 0.7866 | 0.7041 | 0.0825 | 0.4677 |
| `dropout_0.1` | 0.7835 | 0.6713 | 0.1122 | 0.5227 |
| `dropout_0.2` | 0.7679 | 0.6043 | 0.1636 | 0.5715 |
| `false_positive_0.001` | 0.7870 | 0.7262 | 0.0608 | 0.3765 |
| `false_positive_0.005` | 0.7820 | 0.6871 | 0.0949 | 0.3431 |

Main artifacts:

- `results/revision_plan/vspm_robustness_T10_n5_gpu.csv`
- `results/revision_tables/robustness_table.csv`
- `results/revision_figures/vspm_robustness.pdf`

## 5. Current Running Tasks

No revision training/evaluation process is currently running.

Checked process pattern:

```bash
run_mst_ablation|train_bev_compressed|train_bev.py|eval_vspm_robustness|run_vspm_robustness
```

Status output:

- `results/revision_plan/pipeline_status.md`
- Full pipeline: stopped.
- VSPM recovery: stopped.
- MST ablation: stopped.
- Active revision processes: none.

## 6. Not Started / Still Needed

High-priority reviewer-facing work not completed in this workspace:

| task | status | reason / next action |
|---|---|---|
| Full cooperative detection/AP experiments | Not started | Current active code is VSPM/BEV occupancy forecasting, not the full detection/AP pipeline. |
| Additional real-world dataset validation, e.g. DAIR-V2X | Not started | No real-world dataset adapter/result is currently present under this workspace. |
| Competitive latency-aware baseline implementation/comparison | Not started | Current repo has no imported implementations/logs for recent delay-compensation baselines. |
| Spatial/pose noise full experiment | Partially prepared | Robustness utilities include pose-noise sampling, but the completed robustness table only covers packet loss, dropout, and false positives. |
| Loss/objective ablation | Not started | Current VSPM sensitivity sweeps vary `T` and `n`; loss ablations were planned but not launched. |
| Bottleneck dimension sweep `8/12/16/24/32` | Not started | Only `bottleneck12` has been trained. |
| Embedded hardware timing, e.g. NVIDIA Orin | Not started | No Orin/hardware benchmark data in workspace. |
| Paper text edits and response letter | Not started in code workspace | Experimental evidence is prepared; manuscript/response drafting still needs to be done separately. |

Lower-priority cleanup/follow-up:

- Update `docs/ACTIVE_WORK_LOG.md` and `docs/RUN_STATUS.md` from the old "running" wording to the current stopped/completed state.
- Add a compact MST ablation table with state-size plus performance in one CSV/LaTeX table.
- Decide which VSPM-level robustness results are paper-ready versus preliminary.
- If adding real dataset support, create a clear dataset adapter rather than modifying V2X-Sim loaders in place.

## 7. Main Generated Artifacts To Use In Revision

Tables:

- `results/revision_tables/active_training_table.csv`
- `results/revision_tables/vspm_profile_table.csv`
- `results/revision_tables/scalability_table.csv`
- `results/revision_tables/robustness_table.csv`
- `results/revision_tables/revision_tables.tex`

Figures:

- `results/revision_figures/vspm_training_iou.pdf`
- `results/revision_figures/vspm_profile_tradeoff.pdf`
- `results/revision_figures/cache_scalability.pdf`
- `results/revision_figures/vspm_robustness.pdf`

Status/summaries:

- `results/revision_plan/pipeline_status.md`
- `results/revision_plan/pipeline_latest_metrics.csv`
- `results/revision_plan/pipeline_checkpoints.csv`
- `results/summary_revision/training_summary.csv`
- `results/summary_revision/training_timeseries.csv`

