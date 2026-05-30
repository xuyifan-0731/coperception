# Local Experiment Preparation

Scope: only `/raid/xuyifan/v2x_code_ckpt`.

This document lists the next experiments that can be prepared or run using the
current repository contents only.

## 1. Environment

Use the requested conda environment with:

```bash
eval "$(conda shell.bash hook)" && conda activate Android-Lab
```

For scripts importing local modules:

```bash
PYTHONPATH=src python <script>
```

Current environment note:

- `Android-Lab` has `torch==2.7.1+cu126`, `scikit-learn`, and `wandb`.
- PyTorch detects the local RTX 4090 after pinning `torch==2.7.1`.
- `WANDB_MODE=offline` is used for revision runs.

## 2. Prepared Manifests

Generate revision experiment grids:

```bash
PYTHONPATH=src python scripts/prepare_revision_experiments.py
```

Outputs:

- `results/revision_plan/latency_grid.csv`
- `results/revision_plan/robustness_grid.csv`
- `results/revision_plan/scalability_grid.csv`

These grids define:

- communication latency `0-5` frames.
- computation latency `0-4` frames.
- packet loss, delay jitter, pose noise values.
- N-agent scalability settings `1,2,3,4,5,8,16`.

## 3. Data Preparation

Inspect the local archive:

```bash
PYTHONPATH=src python scripts/inspect_v2xsim_zip.py
```

Output:

- `results/revision_plan/v2xsim_zip_inventory.csv`

Current inventory result:

- `60001` files in the archive.
- `4.554 GiB` compressed.
- `72.273 GiB` uncompressed.
- `train` contains agents `0-5` with `8000` sample files each.
- `test` contains agents `0-5` with `1000` sample files each.

Training/evaluation scripts require extracted data:

```text
V2X-Sim-det/
  train/agent1 ... train/agent5
  test/agent1  ... test/agent5
  val/agent1   ... val/agent5
```

`V2X-Sim-det/` has been extracted under the workspace and is the active
`DATA_ROOT` for full revision runs.

## 4. Experiment Track A: VSPM Sensitivity

Purpose:

- Support method-design justification for ConvLSTM, history length, rollout
  horizon, and training objectives.

Prepared script:

```bash
DRY_RUN=1 bash scripts/run_vspm_sensitivity.sh
```

Actual run after extracting data:

```bash
DRY_RUN=0 DATA_ROOT=V2X-Sim-det bash scripts/run_vspm_sensitivity.sh
```

Sweeps:

- `T=5/10/20/30` with `n=5`.
- `n=3/5/10/15` with `T=10`.

Next additions:

- Add explicit loss ablation commands.
- Add pure ConvGRU without FP16 if needed for a clean architecture comparison.

## 5. Experiment Track B: MST Ablation

Purpose:

- Quantify model-state transmission bandwidth/latency trade-offs.

Prepared script:

```bash
DRY_RUN=1 bash scripts/run_mst_ablation.sh
```

Actual run after extracting data:

```bash
DRY_RUN=0 DATA_ROOT=V2X-Sim-det bash scripts/run_mst_ablation.sh
```

Variants:

- baseline ConvLSTM state.
- GRU + FP16.
- DS64 + FP16.
- DS64 + FP16 + bottleneck12.

Next additions:

- Add bottleneck dimension sweep `8/12/16/24/32`.
- Pair performance results with `scripts/verify_compress.py`.

## 6. Experiment Track C: Synthetic Profiling

Purpose:

- Generate early numbers for parameter count, state memory, and synthetic
  inference latency without requiring dataset extraction.

Run:

```bash
PYTHONPATH=src python scripts/profile_vspm.py --device auto --repeat 5
```

Output:

- `results/revision_plan/vspm_profile.csv`

If `torch` is unavailable in the active environment, run metadata-only mode:

```bash
PYTHONPATH=src python scripts/profile_vspm.py --metadata-only \
  --out results/revision_plan/vspm_profile_metadata.csv
```

Current metadata-only result:

- ConvLSTM h+c state at 256x256: `16.0 MiB/agent`.
- GRU FP16 state at 256x256: `4.0 MiB/agent`.
- DS64 FP16 state: `0.25 MiB/agent`.
- Bottleneck12 currently reports the same spatial/state container estimate as
  DS64 FP16 in this lightweight metadata table; detailed bottleneck byte counts
  should be reported by the MST compression ablation.

This supports:

- N-agent complexity discussion.
- MST memory comparison.
- Initial compute-overhead estimates.

## 7. Experiment Track D: N-Agent Cache Scalability

Purpose:

- Prepare cache memory and eviction discussion requested by reviewer 2.

Run:

```bash
PYTHONPATH=src python scripts/run_scalability_profile.py
```

Output:

- `results/revision_plan/scalability_profile.csv`

This is a synthetic cache/state memory profile. It does not claim detection AP.

Current synthetic result:

- FP32 ConvLSTM h+c state scales from `16 MiB` at N=1 to `256 MiB` at N=16.
- GRU DS64 FP16 state scales from `0.25 MiB` at N=1 to `4 MiB` at N=16.
- With cache capacity `8`, N=16 causes 8 LRU evictions and 0.5 hit rate in the
  simple synthetic access pattern.

## 8. Experiment Track E: Robustness Utilities

Prepared module:

- `src/v2x_forecasting/robustness.py`
- `scripts/eval_vspm_robustness.py`
- `scripts/run_vspm_robustness_eval.sh`

Capabilities:

- packet loss masks.
- delay jitter samples.
- BEV occupancy dropout/false positives.
- pose noise sampling.

Run a VSPM-level robustness table from the latest checkpoint:

```bash
DEVICE=cpu MAX_SAMPLES=256 bash scripts/run_vspm_robustness_eval.sh
```

Output:

- `results/revision_plan/vspm_robustness_eval.csv`

Current completed `T10_n5` robustness run used GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 DEVICE=cuda T=10 N=5 MAX_SAMPLES=256 \
  CKPT=runs/revision_vspm_sensitivity/T10_n5/checkpoints/ckpt_13000.pth \
  OUT=results/revision_plan/vspm_robustness_T10_n5_gpu.csv \
  bash scripts/run_vspm_robustness_eval.sh
```

Outputs:

- `results/revision_plan/vspm_robustness_T10_n5_gpu.csv`
- `results/revision_tables/robustness_table.csv`
- `results/revision_figures/vspm_robustness.pdf`

The script defaults to CPU so it can run while the GPU is occupied by full
training. It perturbs BEV histories and reports IoU, baseline IoU, delta IoU,
and Dynamic IoU.

## 9. Parallel Monitoring and Table Generation

These tools can run while training is active:

```bash
PYTHONPATH=src python scripts/report_pipeline_status.py
python scripts/make_revision_tables.py
```

Outputs:

- `results/revision_plan/pipeline_status.md`
- `results/revision_plan/pipeline_latest_metrics.csv`
- `results/revision_plan/pipeline_checkpoints.csv`
- `results/revision_tables/vspm_profile_table.csv`
- `results/revision_tables/scalability_table.csv`
- `results/revision_tables/active_training_table.csv`
- `results/revision_tables/robustness_table.csv`
- `results/revision_tables/revision_tables.tex`

## 10. Current Local Blockers

- Current repo alone does not contain the full detection/AP pipeline used by the
  paper tables.
- Current repo alone cannot add a new real-world dataset validation unless the
  dataset and adapter are added under this workspace.
- Current repo alone cannot reproduce baselines whose code/logs are absent from
  this workspace.

## 11. Recommended Next Step

Immediate, while full training is active:

1. Periodically run `scripts/report_pipeline_status.py`.
2. Regenerate `scripts/make_revision_tables.py` after each validation point.
3. Run CPU robustness evaluation from the latest stable checkpoint.
4. Draft Method complexity, cache, and robustness tables from the generated CSVs.

After each full run finishes:

1. Run `scripts/collect_results.py --root runs --out results/summary_revision --copy-raw-metrics`.
2. Regenerate paper-ready tables with `scripts/make_revision_tables.py`.
3. Run robustness evaluation on the final checkpoint for the selected model.
4. Update paper tables and captions.
