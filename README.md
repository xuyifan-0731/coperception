# V2X BEV Forecasting Experiments

## 260704 实验结果复现入口

如果你是第一次打开这个 GitHub 仓库，建议先从这里开始，而不是直接看代码目录。

260704 实验包位置：

```text
release_packages/260704_experiment_artifacts/
```

这个实验包按下面的逻辑整理：

```text
数据集 -> baseline 方法 -> checkpoints / results / logs / paper_sources
```

也就是说，先按数据集找，再按对比方法找，最后根据需要查看结果、模型、日志或论文表格来源。

### 新人阅读顺序

1. `release_packages/260704_experiment_artifacts/00_docs/260704_ARTIFACT_INDEX.md`
   先看整体目录结构、每个文件夹的含义、哪些内容已经上传、哪些内容没有上传。

2. `release_packages/260704_experiment_artifacts/00_docs/260704_REPRODUCTION_CHECKLIST.md`
   按步骤准备复现环境，包括数据集下载、external baseline 克隆、patch 应用、split 文件放置、checkpoint 恢复。

3. `release_packages/260704_experiment_artifacts/00_docs/260704_results.md`
   查看最终实验结果。

4. `release_packages/260704_experiment_artifacts/00_docs/260704_EXTERNAL_RESOURCES.md`
   查看公开数据集地址、外部 baseline 代码地址、外部 checkpoint 状态、环境要求和仍然缺失的条件。

5. `release_packages/260704_experiment_artifacts/00_reproduction_assets/README.md`
   查看本仓库已经补充的小型复现材料，包括外部代码 patch、split 文件、环境 requirements 参考和 V2X-Sim info summary。

6. `release_packages/260704_experiment_artifacts/00_docs/260704_RUNBOOK.md`
   按数据集和 baseline 找具体训练、测试、评估命令。

7. `release_packages/260704_experiment_artifacts/MANIFEST.csv`
   校验每个上传文件的位置、大小和 SHA256。

8. `release_packages/260704_experiment_artifacts/00_docs/260704_GITHUB_SYNC_AUDIT.md`
   查看本轮 GitHub 同步核查：哪些 checkpoint、结果 CSV、脚本和配置已经补齐，哪些大型 cache/原始数据没有上传以及如何重建。

### 快速定位

| 你想找什么 | 应该看哪里 |
|---|---|
| 最终结果 | `00_docs/260704_results.md` 或各 baseline 的 `results/` |
| Word 版结果报告 | `00_docs/260704_experiment_results.docx` |
| 我们方法或 baseline 的 checkpoint | 各 baseline 下的 `checkpoints/` |
| 真实运行日志 | 各 baseline 下的 `logs/` |
| 外部代码 patch | `00_reproduction_assets/patches/` |
| DAIR / TraF-Align split 文件 | `00_reproduction_assets/split_files/` |
| 环境参考文件 | `00_reproduction_assets/env_refs/` |
| 大 checkpoint 恢复方式 | `restore_split_checkpoints.py` |
| 完整文件清单和 SHA256 | `MANIFEST.csv` |

### 数据集和大文件说明

原始数据集不上传到 GitHub，需要从公开地址下载：

| 数据集 | 用途 | 下载/说明入口 |
|---|---|---|
| V2X-Sim 2.0 | V2X-Sim / DATA、V2VNet、DiscoNet | `https://ai4ce.github.io/V2X-Sim/download.html` |
| DAIR-V2X-C | DAIR official late fusion、DATA、LRCP | `https://thudair.baai.ac.cn/index`，代码说明见 `https://github.com/AIR-THU/DAIR-V2X` |
| DAIR-V2X-Seq / V2X-Seq-SPD | TraF-Align | `https://thudair.baai.ac.cn/index`，代码说明见 `https://github.com/AIR-THU/DAIR-V2X-Seq` |

本仓库已经上传了关键结果、checkpoint、脚本快照、patch、split 文件和说明文档；没有上传原始数据集、大型 feature cache、box cache，以及 V2X-Sim info `.pkl`。这些大文件需要按文档重新下载或生成。

大 checkpoint 已经拆分成 `*.split/*.partNNN`，复现前运行：

```bash
cd release_packages/260704_experiment_artifacts
python restore_split_checkpoints.py
python materialize_runtime_paths.py --repo-root /path/to/coperception --traf-root /tmp/TraF-Align_partial --mode symlink
```

`restore_split_checkpoints.py` 只在归档包内恢复 checkpoint；`materialize_runtime_paths.py` 会把这些 checkpoint/config 软链接或复制到训练、测试脚本默认查找的历史路径。

### 260704 实验包主目录

```text
release_packages/260704_experiment_artifacts/
  00_docs/                  # 结果、说明、runbook、checklist
  00_reproduction_assets/   # patch、split、requirements、info summary
  00_code_scripts/          # 本次实验脚本快照
  00_code_version/          # 打包时 Git 状态
  01_v2xsim/                # V2X-Sim 实验
  02_dair_v2x_c/            # DAIR-V2X-C 实验
  03_dair_v2x_seq/          # DAIR-V2X-Seq / TraF-Align 实验
```

当前 260704 包适合做结果审计、checkpoint 追溯和按文档复现。它不是包含所有原始数据和大型 cache 的一键镜像。

## Original Project Overview

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
- `checkpoints/curated/`: selected small project-owned checkpoints needed for
  the main paper/revision experiments.
- `docs/`: dataset, experiment, and reproduction notes.

## What Is Not Tracked

The following remain local and are ignored by `.gitignore`:

- `V2X-Sim-det/` and `V2X-Sim-det-*.zip`
- `model1/` scratch experiment workspace
- `external/` third-party upstream checkouts such as OpenDAIRV2X, DATA,
  DiscoNet, SyncNet, LRCP, and CoPerception
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

Restore the selected checkpoints to their legacy local paths with:

```bash
bash scripts/restore_curated_checkpoints.sh
```

See `docs/RESTORE_LOCAL_ARTIFACTS.md` for restoring ignored datasets, external
repositories, and official third-party checkpoints.

## Optional Third-Party Checkouts

The core BEV forecasting code under `src/` and the V2X-Sim training scripts do
not vendor third-party repositories. Some reproduction and comparison scripts
expect local upstream checkouts under `external/`, for example:

- `external/DAIR-V2X` for OpenDAIRV2X official DAIR-V2X baselines.
- `external/DATA` for the DATA/OpenCOOD V2X-Sim 2.0 pipeline.
- `external/LRCP`, `external/DiscoNet`, and `external/SyncNet` for comparison
  experiments.

Those directories are intentionally ignored because they are independent Git
repositories plus generated data/checkpoint state. Clone or place them locally
only when running the corresponding reproduction scripts.

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
