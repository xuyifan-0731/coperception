# Reproduction Guide

## 1. Prepare Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export WANDB_MODE=offline
```

Use GPU if available. The original runs used PyTorch and CUDA; CPU execution is
possible for smoke tests but impractical for full training.

## 2. Prepare Data

Place the V2X-Sim data under `V2X-Sim-det/` or export:

```bash
export DATA_ROOT=/path/to/V2X-Sim-det
```

Confirm this path exists:

```bash
ls "${DATA_ROOT}/train/agent1"
ls "${DATA_ROOT}/test/agent1"
```

## 3. Train

All-agent model:

```bash
bash scripts/train_all_agents.sh
```

Held-out agent model:

```bash
bash scripts/train_leave_one_agent.sh 1
```

Compression variants:

```bash
bash scripts/train_compressed.sh gru_fp16
bash scripts/train_compressed.sh ds64_fp16
bash scripts/train_compressed.sh bottleneck12
```

Set a custom output root:

```bash
export RUN_ROOT=/path/to/output
```

## 4. Evaluate

```bash
bash scripts/eval_checkpoint.sh /path/to/ckpt_27000.pth 1 outputs/agent1_eval
```

This writes prediction `.npy` files and `outputs/agent1_eval/metrics.csv`.

## 5. Summarize Existing Workspace Results

If the ignored local `model1/` experiment workspace is available:

```bash
python scripts/collect_results.py --root model1 --out results/summary --copy-raw-metrics
python scripts/build_checkpoint_manifest.py
```

The command regenerates all CSV files under `results/summary/`.

## 6. Plot

The plotting scripts read exported `metrics.csv` files:

```bash
python scripts/plot_iou_interval.py \
  --file1 outputs/agent1_eval/metrics.csv \
  --file2 outputs/baseline_agent1/metrics.csv \
  --out outputs/iou_vs_interval.png
```

For the five-agent panel:

```bash
python scripts/plot_iou_interval_multi.py \
  --file1_1 outputs/a1_general/metrics.csv --file2_1 outputs/a1_specific/metrics.csv \
  --file1_2 outputs/a2_general/metrics.csv --file2_2 outputs/a2_specific/metrics.csv \
  --file1_3 outputs/a3_general/metrics.csv --file2_3 outputs/a3_specific/metrics.csv \
  --file1_4 outputs/a4_general/metrics.csv --file2_4 outputs/a4_specific/metrics.csv \
  --file1_5 outputs/a5_general/metrics.csv --file2_5 outputs/a5_specific/metrics.csv \
  --out outputs/iou_vs_interval_5agents.pdf
```
