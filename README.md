# V2X BEV Forecasting Experiments

This repository contains the curated, reproducible version of the V2X BEV
occupancy forecasting experiments from the local workspace. The original
workspace also contains raw V2X-Sim data, checkpoints, prediction dumps, and
scratch scripts; those large artifacts are intentionally excluded from Git.

## What Is Tracked

- `src/v2x_forecasting/train_bev.py`: ConvLSTM BEV forecasting training and
  evaluation script.
- `src/v2x_forecasting/train_bev_compressed.py`: compressed-link variants
  using GRU, FP16, 64x64 temporal encoding, and optional 1x1 bottleneck.
- `scripts/*.sh`: reproducible train/eval entry points with configurable
  `DATA_ROOT` and `RUN_ROOT`.
- `scripts/collect_results.py`: regenerates compact CSV summaries from local
  experiment logs.
- `results/summary/*.csv`: curated metrics extracted from the existing
  experiments.
- `results/figures/`: paper/analysis figures that are small enough for Git.
- `docs/`: dataset, experiment, and reproduction notes.

## What Is Not Tracked

The following remain local and are ignored by `.gitignore`:

- `V2X-Sim-det/` and `V2X-Sim-det-*.zip`
- `model1/` scratch experiment workspace
- `coperception-master-2D/` upstream checkout
- checkpoints (`*.pth`, `*.pt`) and prediction arrays (`*.npy`)
- runtime outputs under `runs/`, `outputs/`, `wandb/`, and `test_result*/`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For offline runs without a Weights & Biases account:

```bash
export WANDB_MODE=offline
```

## Data Layout

The training scripts expect the V2X-Sim detection data to be unpacked as:

```text
V2X-Sim-det/
  train/agent1 ... train/agent5
  test/agent1  ... test/agent5
  val/agent1   ... val/agent5
```

Override the dataset location with:

```bash
export DATA_ROOT=/path/to/V2X-Sim-det
```

See `docs/DATA.md` for details.

## Reproduce Main Runs

All-agent baseline:

```bash
bash scripts/train_all_agents.sh
```

Leave-one-agent-out generalization model:

```bash
bash scripts/train_leave_one_agent.sh 1
bash scripts/train_leave_one_agent.sh 2
bash scripts/train_leave_one_agent.sh 3
bash scripts/train_leave_one_agent.sh 4
bash scripts/train_leave_one_agent.sh 5
```

Compressed communication variants:

```bash
bash scripts/train_compressed.sh gru_fp16
bash scripts/train_compressed.sh ds64_fp16
bash scripts/train_compressed.sh bottleneck12
```

Evaluate a checkpoint and export per-sample metrics:

```bash
bash scripts/eval_checkpoint.sh runs/leave_one_out/checkpoints_without_agent1/ckpt_27000.pth 1 outputs/agent1_eval
```

## Results

Curated results are in `results/summary/`.

- `training_summary.csv`: best and final validation metrics per run.
- `training_timeseries.csv`: parsed validation curve points.
- `eval_overall.csv`: exported prediction metrics averaged by result/agent.
- `eval_by_interval.csv`: exported prediction metrics averaged by forecast
  interval.
- `raw_metrics/`: copied raw `metrics.csv` files for exported prediction
  evaluations.
- `checkpoint_manifest.csv`: local checkpoint references, sizes, and SHA256
  hashes for runs where the checkpoint file still exists.

See `docs/EXPERIMENTS.md` for a compact interpretation of the existing results.

Regenerate the summaries from the local ignored `model1/` workspace:

```bash
python scripts/collect_results.py --root model1 --out results/summary --copy-raw-metrics
python scripts/build_checkpoint_manifest.py
```
