# 当前代码与实验资源总览

更新时间：2026-05-22 11:24 UTC

范围限制：本文档只整理 `/raid/xuyifan/v2x_code_ckpt` 下的内容。

## 1. 总体结论

- 当前没有正在运行的 revision 训练或评估进程。
- 当前新增 DAIR-V2X 准备任务；完整 DAIR 数据下载仍被 Google Drive 配额阻塞。
- `dair-v2x-openmmlab` 旧 OpenMMLab 环境已装好并通过 CUDA smoke test。
- 官方 `DAIR-V2X-C-Example.zip` 已下载和解压，并完成 OpenDAIRV2X example smoke test。
- VSPM sensitivity 实验已完成。
- MST/system-state ablation 四个变体已完成。
- VSPM-level robustness 评估已完成一组 `T10_n5` 结果。
- 表格和图已经重新生成，当前结果入口在：
  - `results/revision_tables/`
  - `results/revision_figures/`
  - `results/summary_revision/`
  - `results/revision_plan/pipeline_status.md`

## 2. 文件组织结构

```text
/raid/xuyifan/v2x_code_ckpt/
├── DLPCM/                     # 原论文相关文件/图表/参考材料
├── V2X-Sim-det/               # 已解压 V2X-Sim 数据集，当前实验主数据
│   ├── train/agent1..agent5
│   ├── val/agent1..agent5
│   └── test/agent1..agent5
├── data_smoke/                # smoke test 小数据
├── datasets/DAIR-V2X/         # DAIR-V2X 下载/准备目录，目前 full 数据缺失，example 已可用于 smoke test
├── docs/                      # todo、状态、实验说明、当前总览
├── external/DAIR-V2X/         # 官方 OpenDAIRV2X 代码
├── model1/                    # 旧实验/已有本地实验产物
├── results/                   # 汇总结果、表格、图、pipeline 状态
├── runs/                      # 本轮训练 checkpoint 和 wandb_logs
├── scripts/                   # 启动、评估、统计、画图脚本
├── src/v2x_forecasting/       # 核心 Python 代码
└── wandb/                     # W&B offline 日志
```

当前主要目录大小：

| 目录 | 大小 | 说明 |
|---|---:|---|
| `V2X-Sim-det/` | 73G | 主数据集 |
| `wandb/` | 184M | offline wandb 日志 |
| `runs/` | 72M | 训练 checkpoint / 日志 |
| `results/` | 52M | 汇总表格和图 |
| `DLPCM/` | 2.8M | 原论文参考目录 |
| `src/` | 92K | 核心代码 |
| `scripts/` | 168K | 实验脚本 |
| `docs/` | 108K | 文档 |

## 3. 核心代码位置

### 3.1 模型与训练

| 文件 | 作用 |
|---|---|
| `src/v2x_forecasting/train_bev.py` | 原始/基线 BEV occupancy forecasting 训练与评估脚本。已加入 `--max_steps`、`--max_val_batches`。 |
| `src/v2x_forecasting/train_bev_compressed.py` | MST/压缩状态训练脚本，支持 `baseline`、`gru_fp16`、`ds64_fp16`、`bottleneck12`。已加入 smoke 控制参数。 |
| `src/v2x_forecasting/profiling.py` | 参数量、状态大小、cache 内存估算工具。 |
| `src/v2x_forecasting/cache.py` | 接收端 LRU cache 模拟。 |
| `src/v2x_forecasting/robustness.py` | 丢包、jitter、BEV dropout/false positive、pose noise 等扰动工具。 |
| `src/v2x_forecasting/delay_simulator.py` | 延迟/实验网格辅助工具。 |

### 3.2 实验启动脚本

| 文件 | 作用 |
|---|---|
| `scripts/run_revision_full_pipeline.sh` | 原完整顺序 pipeline；曾在 `T20_n5` batch 16 OOM。 |
| `scripts/run_vspm_sensitivity.sh` | VSPM sensitivity sweep。 |
| `scripts/recover_vspm_sensitivity_after_oom.sh` | OOM 后的小 batch 恢复脚本。 |
| `scripts/run_mst_ablation.sh` | MST 四个变体消融：`baseline`、`gru_fp16`、`ds64_fp16`、`bottleneck12`。 |
| `scripts/run_vspm_robustness_eval.sh` | robustness 评估封装脚本，已加 CPU 线程限制。 |
| `scripts/run_vspm_robustness_T10_n5_gpu.sh` | `T10_n5` GPU robustness 专用启动脚本。 |
| `scripts/download_dair_v2x_c.sh` | DAIR-V2X-C Google Drive manifest 下载脚本；当前被 quota 阻塞。 |
| `scripts/prepare_dair_v2x_layout.sh` | DAIR 解压后校验目录并建立 OpenDAIRV2X 数据软链接。 |
| `scripts/download_dair_v2x_checkpoints.sh` | 下载官方 OpenDAIRV2X baseline checkpoint。 |
| `scripts/run_dair_v2x_official_baselines.sh` | 运行官方 DAIR 点云 baseline：veh/inf/late/no-comp/TCLF/early；缺失 checkpoint 时默认跳过对应 baseline，可用 `STRICT_CHECKPOINTS=1` 严格检查。 |
| `scripts/prepare_dair_v2x_openmmlab_env.sh` | 准备 OpenDAIRV2X 旧 OpenMMLab 环境。 |
| `scripts/prepare_dair_vspm_bev.py` | 将 DAIR 标注转换成当前 VSPM 训练脚本可读的稀疏 BEV 序列。 |
| `scripts/prepare_dair_v2x_example_layout.sh` | 准备官方 DAIR example 数据布局和 OpenDAIRV2X 软链接。 |
| `scripts/run_dair_v2x_example_official_smoke.sh` | 在官方 DAIR example 上跑 veh/inf/late/no-comp/TCLF smoke test。 |
| `scripts/run_dair_v2x_full_workflow.sh` | full DAIR 数据可用后，一键执行布局、官方 baseline、DAIR-BEV 适配和我们的方法训练。 |

### 3.3 结果整理与画图

| 文件 | 作用 |
|---|---|
| `scripts/report_pipeline_status.py` | 解析进程状态、最新指标、checkpoint 列表。 |
| `scripts/collect_results.py` | 从 `wandb_logs.txt` 和 metric CSV 汇总到 `results/summary_revision/`。 |
| `scripts/make_revision_tables.py` | 生成 CSV 和 LaTeX 表格草稿。 |
| `scripts/plot_revision_tables.py` | 生成训练曲线、profile、scalability、robustness 图。 |
| `scripts/profile_vspm.py` | VSPM 参数量、状态大小、rollout latency profile。 |
| `scripts/run_scalability_profile.py` | N-agent cache/state memory profile。 |
| `scripts/eval_vspm_robustness.py` | VSPM-level robustness 评估。 |

## 4. 已完成的代码改动

- 训练脚本新增 smoke 控制：
  - `--max_steps`
  - `--max_val_batches`
- 新增或补充工具模块：
  - `src/v2x_forecasting/robustness.py`
  - `src/v2x_forecasting/cache.py`
  - `src/v2x_forecasting/profiling.py`
  - `src/v2x_forecasting/delay_simulator.py`
- 新增实验启动脚本：
  - VSPM sensitivity
  - OOM recovery
  - MST ablation
  - robustness eval
  - full revision pipeline
- 新增结果汇总与绘图脚本：
  - pipeline status
  - result collection
  - revision tables
  - revision figures
- robustness eval 已限制 CPU 线程：
  - `CPU_THREADS=16`
  - `--torch-threads`
- robustness 表格和图已纳入生成流程。
- DAIR 官方 baseline 准备：
  - `dair-v2x-openmmlab` 环境已安装 `torch 1.10.1/cu113`、`mmcv 1.3.18`、`mmdet 2.14.0`、`mmseg 0.14.1`、`mmdet3d 0.17.1`。
  - `mmdet3d` 从 `external/mmdetection3d-v0.17.1` 本地源码编译安装。
  - `torch` CUDA 基础运算和 `mmdet3d.ops.Voxelization` 导入已验证通过。
  - `pypcd`、`spconv`、`pycocotools`、`terminaltables`、`prettytable` 导入已验证通过。
  - 已下载 late-fusion PointPillars 的 infrastructure/vehicle 两个 checkpoint 和 FFNet checkpoint。
- DAIR example smoke test 已完成：
  - `veh_only_k0`
  - `inf_only_k0`
  - `late_fusion_tclf_k0`
  - `late_fusion_tclf_k2`
  - `late_fusion_no_comp_k2`
- 我们的方法的 DAIR example smoke test 已完成：
  - adapter 输出：`datasets/DAIR-V2X/vspm_bev_example/train/agent_vehicle/`
  - training log：`runs/dair_v2x_example_smoke/checkpoints/wandb_logs.txt`
  - 仅跑通 1 个 optimizer step，用于验证链路，不是正式结果。

## 5. 已完成实验与结果

### 5.1 VSPM sensitivity

| run | epoch | final step | final IoU | final Delta IoU | final Dynamic IoU |
|---|---:|---:|---:|---:|---:|
| `T5_n5` | 30 | 13800 | 0.7505 | 0.0540 | 0.3204 |
| `T10_n5` | 30 | 13050 | 0.7513 | 0.0570 | 0.3356 |
| `T20_n5` | 30 | 23100 | 0.7496 | 0.0581 | 0.3297 |
| `T30_n5` | 30 | 40200 | 0.7518 | 0.0597 | 0.3326 |
| `T10_n3` | 30 | 13350 | 0.7678 | 0.0558 | 0.3413 |
| `T10_n10` | 30 | 24600 | 0.7243 | 0.0559 | 0.3332 |
| `T10_n15` | 30 | 46200 | 0.7075 | 0.0549 | 0.3109 |

主要文件：

- `runs/revision_vspm_sensitivity/*/checkpoints/`
- `results/summary_revision/training_summary.csv`
- `results/summary_revision/training_timeseries.csv`

### 5.2 MST/system-state ablation

| run | epoch | final step | final IoU | final Delta IoU | final Dynamic IoU | best IoU |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 30 | 13050 | 0.7488 | 0.0545 | 0.3296 | 0.7495 |
| `gru_fp16` | 30 | 13050 | 0.7462 | 0.0520 | 0.3075 | 0.7469 |
| `ds64_fp16` | 30 | 13050 | 0.7200 | 0.0258 | 0.0914 | 0.7200 |
| `bottleneck12` | 30 | 13050 | 0.7193 | 0.0250 | 0.0986 | 0.7213 |

主要文件：

- `runs/revision_mst_ablation/*/checkpoints/`
- `runs/revision_mst_ablation/*/checkpoints/wandb_logs.txt`
- `results/revision_plan/pipeline_latest_metrics.csv`
- `results/revision_plan/pipeline_checkpoints.csv`

### 5.3 VSPM profile

| model | params | state MiB/agent | mean rollout ms |
|---|---:|---:|---:|
| `convlstm` | 85569 | 16.000 | 3.590 |
| `compressed_lstm` | 85569 | 16.000 | 3.632 |
| `gru_fp16` | 66497 | 4.000 | 4.400 |
| `ds64_fp16` | 66497 | 0.250 | 4.209 |
| `bottleneck12` | 67309 | 0.250 | 4.314 |

主要文件：

- `results/revision_plan/vspm_profile.csv`
- `results/revision_tables/vspm_profile_table.csv`
- `results/revision_figures/vspm_profile_tradeoff.pdf`

### 5.4 N-agent cache/scalability

| agents | ConvLSTM h+c MiB | GRU DS64 FP16 MiB | cache capacity | evictions | hit rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 16.00 | 0.25 | 8 | 0 | 1.00 |
| 2 | 32.00 | 0.50 | 8 | 0 | 1.00 |
| 3 | 48.00 | 0.75 | 8 | 0 | 1.00 |
| 4 | 64.00 | 1.00 | 8 | 0 | 1.00 |
| 5 | 80.00 | 1.25 | 8 | 0 | 1.00 |
| 8 | 128.00 | 2.00 | 8 | 0 | 1.00 |
| 16 | 256.00 | 4.00 | 8 | 8 | 0.50 |

主要文件：

- `results/revision_plan/scalability_profile.csv`
- `results/revision_tables/scalability_table.csv`
- `results/revision_figures/cache_scalability.pdf`

### 5.5 robustness

已完成 `T10_n5` checkpoint 上的 VSPM-level robustness，样本数 256。

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

主要文件：

- `results/revision_plan/vspm_robustness_T10_n5_gpu.csv`
- `results/revision_tables/robustness_table.csv`
- `results/revision_figures/vspm_robustness.pdf`

## 6. 当前正在进行中的任务

DAIR-V2X 真实数据集补充实验准备。

当前状态：

- `external/DAIR-V2X` 已存在。
- `datasets/DAIR-V2X/dair_v2x_c_manifest.csv` 已存在。
- `datasets/DAIR-V2X/raw/` 暂无 full 数据 zip。
- Full DAIR Google Drive 下载在 `2026-05-22 11:23 UTC` 仍因 quota 失败。
- 官方 `DAIR-V2X-C-Example.zip` 已下载和解压：
  - zip: `datasets/DAIR-V2X/raw/DAIR-V2X-C-Example.zip`
  - extracted root: `datasets/DAIR-V2X/example-cooperative-vehicle-infrastructure`
- 已准备数据布局、checkpoint 下载、baseline 运行和 VSPM-BEV 适配脚本。
- OpenDAIRV2X 环境已可用；example smoke test 已完成；完整训练/评估仍等待 full DAIR 数据。

DAIR example smoke 结果，46 帧，仅用于证明流程可运行，不可作为完整 benchmark：

| run | car 3D AP@0.30 | car 3D AP@0.50 | car 3D AP@0.70 | car BEV AP@0.30 | car BEV AP@0.50 | car BEV AP@0.70 | avg comm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `veh_only_k0` | 59.05 | 46.10 | 35.68 | 60.64 | 49.77 | 48.07 | 0.00 B |
| `inf_only_k0` | 19.39 | 13.45 | 4.15 | 19.39 | 14.04 | 5.48 | 247.30 B |
| `late_fusion_tclf_k0` | 61.79 | 48.49 | 37.44 | 63.41 | 52.31 | 50.17 | 475.13 B |
| `late_fusion_tclf_k2` | 60.66 | 47.05 | 36.33 | 62.16 | 51.09 | 48.89 | 463.30 B |
| `late_fusion_no_comp_k2` | 59.56 | 46.12 | 35.82 | 61.11 | 49.75 | 48.08 | 241.04 B |

最新日志：

- `results/dair_v2x_example_official_smoke_all_20260522_112110.log`
- `results/dair_v2x_dataset_download_20260522_112331.log`
- `results/dair_v2x_checkpoint_download_20260522_112331.log`
- `runs/dair_v2x_example_smoke/checkpoints/wandb_logs.txt`

可打开动态状态文档：

```bash
code /raid/xuyifan/v2x_code_ckpt/docs/DAIR_V2X_RUN_STATUS.md
```

## 7. 未开始或仍需补充的任务

| 任务 | 当前状态 | 说明 |
|---|---|---|
| 完整 cooperative detection / AP 实验 | 阻塞 | OpenDAIRV2X 环境已就绪，但 full DAIR 数据还未下载成功。 |
| DAIR-V2X 或其他真实数据集验证 | 部分准备 | DAIR 适配脚本和 official baseline runner 已准备，等待 full 数据；example smoke test 已完成。 |
| 更多 latency-aware SOTA baseline | 未开始 | 当前 workspace 没有引入近期 delay-compensation baseline 实现或日志。 |
| 空间/定位噪声完整实验 | 部分准备 | 已有 pose-noise 工具，但已完成 robustness 表只覆盖丢包、dropout、false positive。 |
| loss/objective ablation | 未开始 | 当前 sensitivity 只扫了 `T` 和 `n`。 |
| bottleneck dim sweep | 未开始 | 当前只训练了 `bottleneck12`，未扫 `8/16/24/32`。 |
| Orin/嵌入式硬件计时 | 未开始 | 当前只有本机 RTX 4090 profile。 |
| 论文正文和 response letter 改写 | 未开始 | 实验材料已整理，写作需单独推进。 |

## 8. 修回可直接引用的文件

表格：

- `results/revision_tables/active_training_table.csv`
- `results/revision_tables/vspm_profile_table.csv`
- `results/revision_tables/scalability_table.csv`
- `results/revision_tables/robustness_table.csv`
- `results/revision_tables/revision_tables.tex`

图：

- `results/revision_figures/vspm_training_iou.pdf`
- `results/revision_figures/vspm_profile_tradeoff.pdf`
- `results/revision_figures/cache_scalability.pdf`
- `results/revision_figures/vspm_robustness.pdf`

状态和汇总：

- `results/revision_plan/pipeline_status.md`
- `results/revision_plan/pipeline_latest_metrics.csv`
- `results/revision_plan/pipeline_checkpoints.csv`
- `results/summary_revision/training_summary.csv`
- `results/summary_revision/training_timeseries.csv`
