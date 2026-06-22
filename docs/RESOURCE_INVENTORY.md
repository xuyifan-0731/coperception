# Local Resource Inventory

Scope: only files and directories under `/raid/xuyifan/v2x_code_ckpt`.

## 1. Top-Level Resources

| Path | Role | Notes |
|---|---|---|
| `DLPCM/` | Original paper source | LaTeX sections, figures, tables, bibliography |
| `src/v2x_forecasting/` | Local VSPM code | BEV forecasting, compressed-state variants, new preparation utilities |
| `scripts/` | Local experiment scripts | Training/eval/plotting plus revision-prep scripts |
| `results/summary/` | Existing result summaries | CSVs parsed from local `model1/` artifacts |
| `results/figures/` | Existing analysis figures | IoU curves and ablation figures |
| `results/revision_plan/` | Generated revision manifests | Latency, robustness, scalability grids |
| `model1/` | Local checkpoints/artifacts | Existing checkpoints and logs, not a clean reproducibility root |
| `V2X-Sim-det-001.zip` | Local V2X-Sim archive | Contains `V2X-Sim-det/` layout but is not extracted |

Current workspace size is about `4.7G`.

## 2. Paper Assets

Important files in `DLPCM/`:

- `main.tex`: abstract and paper entry.
- `1_introduction.tex`: motivation and contributions.
- `2_related_work.tex`: related work and comparison table.
- `3_dual.tex`: DLPCM/VSPM/MST method and Algorithm 1.
- `4_experiment.tex`: experiments and analysis.
- `5_conclusion.tex`: conclusion.
- `table/main_result.tex`: V2VNet AP table under dual latency.
- `table/main_result_disco.tex`: DiscoNet/SyncNet AP table under dual latency.
- `pics/*.pdf`: existing figures.
- `figure1.jpg`, `figure2.png`: motivation/framework figures.

Revision planning:

- `docs/DLPCM_REVISION_TODO.md`
- `docs/REVISION_TODO.md`

## 3. Local Code Assets

Existing:

- `src/v2x_forecasting/train_bev.py`
  - ConvLSTM BEV occupancy forecasting.
  - VSPM prediction metrics: IoU, Delta IoU, Dynamic IoU, P/R/F1/Acc.
- `src/v2x_forecasting/train_bev_compressed.py`
  - GRU/FP16/DS64/bottleneck variants for model-state transmission.
- `scripts/verify_compress.py`
  - Compression ratio/NMSE inspection for hidden states.
- `scripts/collect_results.py`
  - Parses existing logs and metrics into compact CSV files.

New local-only preparation modules:

- `src/v2x_forecasting/delay_simulator.py`
- `src/v2x_forecasting/robustness.py`
- `src/v2x_forecasting/cache.py`
- `src/v2x_forecasting/profiling.py`

New local-only preparation scripts:

- `scripts/prepare_revision_experiments.py`
- `scripts/inspect_v2xsim_zip.py`
- `scripts/profile_vspm.py`
- `scripts/run_scalability_profile.py`
- `scripts/run_vspm_sensitivity.sh`
- `scripts/run_mst_ablation.sh`

## 4. Data Status

Available inside this workspace:

- `V2X-Sim-det-001.zip`

Archive inspection confirms it contains:

```text
V2X-Sim-det/
  train/
    agent*/<scene>_<frame>/0.npy
```

Current archive inventory:

- Files: `60001`.
- Compressed size: about `4.554 GiB`.
- Uncompressed size: about `72.273 GiB`.
- Tiny files (`<=256` bytes): `12800`, likely empty/invalid frame payloads that
  existing dataset code skips when `np.load(...).item()` is empty.
- Per split/agent inventory is in
  `results/revision_plan/v2xsim_zip_inventory.csv`.

Current blocker:

- `V2X-Sim-det/` is not extracted under the workspace. Full training/evaluation
  scripts expect the extracted directory.

Prepared command:

```bash
python scripts/inspect_v2xsim_zip.py
```

This writes:

- `results/revision_plan/v2xsim_zip_inventory.csv`

## 5. Existing Result Summaries

Available:

- `results/summary/training_summary.csv`
- `results/summary/training_timeseries.csv`
- `results/summary/eval_overall.csv`
- `results/summary/eval_by_interval.csv`
- `results/summary/checkpoint_manifest.csv`

Use these for:

- VSPM baseline and leave-one-agent-out summaries.
- Existing ablation summaries.
- IoU-vs-interval figures.
- Checkpoint provenance.

## 6. Local Experiments We Can Prepare Now

No full data extraction required:

- V2X-Sim archive inventory.
- Synthetic VSPM profiling.
- Synthetic N-agent state/cache memory profiling.
- Latency/robustness/scalability manifest generation.

Generated local preparation CSVs:

- `results/revision_plan/latency_grid.csv`
- `results/revision_plan/robustness_grid.csv`
- `results/revision_plan/scalability_grid.csv`
- `results/revision_plan/v2xsim_zip_inventory.csv`
- `results/revision_plan/scalability_profile.csv`
- `results/revision_plan/vspm_profile_metadata.csv`

Requires extracted `V2X-Sim-det/`:

- VSPM sensitivity training.
- MST ablation training.
- Evaluation/export of prediction metrics from checkpoints.
- Compression verification over real hidden states.

Not available from current repo alone:

- New real-world dataset validation.
- Baseline reproduction for methods whose code/logs are absent from this
  workspace.
- End-to-end AP regeneration unless the corresponding detection pipeline/logs
  are added to this workspace.
