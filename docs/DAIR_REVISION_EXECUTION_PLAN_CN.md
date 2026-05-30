# DAIR-V2X 与审稿意见后续执行计划

更新时间：2026-05-27 10:05 UTC

范围：仅基于 `/raid/xuyifan/v2x_code_ckpt` 当前已有代码、checkpoint、实验结果和论文目录整理。

## 0. 本轮执行更新

本轮已按“先数据闭环，再并行推进不依赖 full raw sensor 的任务”的顺序执行：

1. full DAIR-V2X-C 再次重试下载，日志为 `results/dair_v2x_dataset_download_retry_20260527_084446.log`。结果仍被 Google Drive quota/public-link 限制阻塞，7 个图像/点云大包未完成，`cooperative-vehicle-infrastructure.zip` 继续保持已完成。
   - 已启动 detached 持续重试：PID 记录在 `results/dair_v2x_dataset_download_until_complete_detached.pid`，主日志为 `results/dair_v2x_dataset_download_until_complete_detached_20260527_084934.log`。
   - 当前重试间隔为 1800 秒；第 1 次尝试失败后已进入 sleep，等待 quota 释放后自动继续。
2. `bash scripts/check_dair_v2x_prereqs.sh` 已在 partial layout 上通过，日志为 `results/dair_v2x_prereq_check_after_partial_layout_20260527_084347.log`，结果为 `OK=15 WARN=0 FAIL=0`。
3. DAIR label-derived BEV adapter 已完成 vehicle/infrastructure 两个视角 smoke：
   - `datasets/DAIR-V2X/vspm_bev_partial/manifest_vehicle.csv`，200 train frames。
   - `datasets/DAIR-V2X/vspm_bev_partial/manifest_infrastructure.csv`，200 train frames。
4. partial DAIR BEV 训练 smoke 已跑通，日志为 `results/dair_vspm_partial_train_vehicle_smoke_conda_20260527_084456.log`。该 smoke 使用 `T=10,n=5,batch=1,max_steps=1,device=cpu`，证明当前 adapter 输出可被 `SeqDataset` 和 `train_bev.py` 消费。
5. 新增 `scripts/summarize_dair_vspm_bev.py`，并生成 `results/dair_vspm_bev_partial_summary.csv`。当前 partial 输出共 5 个 scene rows、400 frames。
6. 新增 `scripts/eval_dair_vspm_delay_grid.py`，并用已有 `runs/revision_vspm_sensitivity/T10_n5/checkpoints/ckpt_13000.pth` 在 partial DAIR BEV 上跑通 9 个 delay cases，输出 `results/dair_vspm_delay_grid_partial_smoke.csv`。
7. 新增 `docs/SYSTEM_ABLATION_PROTOCOL_CN.md`，定义审稿意见所需的 system-level ablation mode、输入输出和执行条件。
8. 已修正文稿低风险问题：
   - `DLPCM/main.tex`：`Vehivle-to-Vehicle` -> `Vehicle-to-Vehicle`。
   - `DLPCM/4_experiment.tex`：`DiscnNet` -> `DiscoNet`。
   - `DLPCM/4_experiment.tex`：`step-iou.pdf` caption 的 x-axis 从 training step 改为 prediction horizon。
9. 官方 checkpoint 下载已完成，日志为 `results/dair_v2x_checkpoint_download_20260527_084044.log`。当前已有 late-fusion PointPillars inf/veh、early-fusion PointPillars、late-fusion ImVoxelNet inf/veh、FFNet checkpoint。
10. checkpoint 完成后再次运行 prereq check，日志为 `results/dair_v2x_prereq_check_after_ckpt_download_20260527_084817.log`，结果仍为 `OK=15 WARN=0 FAIL=0`。该 check 只能证明目录、环境和 checkpoint 可识别；当前 full raw image/velodyne 文件仍缺失，不能据此启动正式 DAIR detection baseline。
11. 已完成 DAIR label-derived BEV 全量转换：
   - vehicle：15285 frames，train 9322，val 5963，manifest 为 `datasets/DAIR-V2X/vspm_bev/manifest_vehicle.csv`。
   - infrastructure：12424 frames，train 8800，val 3624，manifest 为 `datasets/DAIR-V2X/vspm_bev/manifest_infrastructure.csv`。
   - 总输出：27709 个 `0.npy`，约 465M。
12. 已生成全量 BEV 统计：`results/dair_vspm_bev_full_summary.csv`，共 160 个 scene rows。
13. 已用 full label-derived BEV 跑通训练 smoke：`results/dair_vspm_full_train_vehicle_smoke_20260527_085926.log`。
14. 已扩展 `scripts/eval_dair_vspm_delay_grid.py`，支持 `--mode no_comp|comm_only|comp_only|dual|oracle`。full label-derived BEV 上 5 个 mode 的 smoke 均已通过：
   - `results/dair_vspm_system_no_comp_full_label_smoke.csv`
   - `results/dair_vspm_system_comm_only_full_label_smoke.csv`
   - `results/dair_vspm_system_comp_only_full_label_smoke.csv`
   - `results/dair_vspm_system_dual_full_label_smoke.csv`
   - `results/dair_vspm_system_oracle_full_label_smoke.csv`
15. 已启动 DAIR label-derived BEV 主训练 C3：
   - PID：`results/dair_v2x_full_train_vehicle_T10_n5.pid`
   - 日志：`results/dair_v2x_full_train_vehicle_T10_n5_20260527_090013.log`
   - 输出目录：`runs/dair_v2x_full/our_method_T10_n5/checkpoints`
   - 当前运行在 GPU0。
16. 已参数化 `scripts/run_mst_ablation.sh`，并新增 `scripts/run_dair_mst_ablation_parallel.sh`。
17. 已启动 DAIR MST ablation 四个 variant 并行训练：
   - launch PID：`results/dair_v2x_mst_ablation_parallel_launch.pid`
   - launch 日志：`results/dair_v2x_mst_ablation_parallel_launch_20260527_090219.log`
   - per-mode PID：`results/dair_v2x/mst_ablation/logs/*.pid`
   - per-mode 日志：`results/dair_v2x/mst_ablation/logs/*_20260527_090220.log`
   - GPU 分配：baseline GPU1，gru_fp16 GPU2，ds64_fp16 GPU3，bottleneck12 GPU4。
18. 已将 DAIR-V2X-C Google Drive 下载脚本改为固定 `.part` 断点文件，并将 detached 重试窗口改为 600 秒。当前主进程 PID 仍记录在 `results/dair_v2x_dataset_download_until_complete_detached.pid`，最新日志为 `results/dair_v2x_dataset_download_until_complete_detached_20260527_095131_retry600_fixedpart.log`。
19. 已新增 AI Studio 镜像下载器：
   - `scripts/download_dair_v2x_aistudio.py`
   - `scripts/download_dair_v2x_aistudio_until_complete.sh`
   - 支持登录 cookie / `x-studio-token` / access token、固定 `.aistudio.part`、HTTP Range 断点续传和进度输出。
20. 已解析 AI Studio 数据集 `179509`。其真实下载 API 为 `/llm/files/datasets/179509/file/{fileId}/download`，文件列表包含 `infrastructure.zip`、两侧 image zip、两侧 velodyne zip、源码 zip 和 example zip。未登录调用下载 API 会返回 `need login or access token`，因此当前服务器尚不能直接从 AI Studio 拉取大包，需要提供有效 AI Studio 登录 cookie 或 access token。

## 1. 当前 DAIR-V2X 下载状态

完整 DAIR-V2X-C 下载已重新尝试：

```bash
cd /raid/xuyifan/v2x_code_ckpt
bash scripts/download_dair_v2x_c.sh
```

本次日志：

```text
results/dair_v2x_dataset_download_20260527_081942.log
results/dair_v2x_dataset_download_pass_20260527_082829.log
```

当前 raw 目录已存在：

```text
datasets/DAIR-V2X/raw/DAIR-V2X-C-Example.zip
datasets/DAIR-V2X/raw/cooperative-vehicle-infrastructure.zip
datasets/DAIR-V2X/raw/cooperative-vehicle-infrastructure-infrastructure-side-image-deprecated.zip.part
```

本次结果：

- `cooperative-vehicle-infrastructure.zip` 已下载完成，大小约 `235M`。
- `cooperative-vehicle-infrastructure-infrastructure-side-image-deprecated.zip` 曾下载到约 `169M` 的 `.part` 临时文件，但 full 文件未完成。
- 其余 7 个 full 数据文件仍被 Google Drive quota/public-link 限制阻塞，`gdown` 返回 `Too many users have viewed or downloaded this file recently` 或 `Cannot retrieve the public link of the file`。
- 本轮再次重试后仍然失败。当前 detached wrapper 每 600 秒自动重试一次；后续 quota 释放后会跳过已完成文件并继续尝试缺失文件。
- AI Studio 镜像可作为替代源，但下载 API 需要登录态。已验证无登录态时返回 `need login or access token`。

AI Studio 镜像与本地目标文件的映射如下：

| AI Studio 文件 | 本地目标文件 | 备注 |
|---|---|---|
| `infrastructure.zip` | `cooperative-vehicle-infrastructure.zip` | 已完成 |
| `infrastructure-vehicle-side-image.zip` | `cooperative-vehicle-infrastructure-vehicle-side-image.zip` | 待下载 |
| `infrastructure-infrastructure-side-image.zip` | `cooperative-vehicle-infrastructure-infrastructure-side-image-deprecated.zip` | 待下载 |
| `infrastructure-infrastructure-side-velodyne.zip` | `cooperative-vehicle-infrastructure-infrastructure-side-velodyne.zip` | AI Studio 为单个完整 zip |
| `infrastructure-vehicle-side-velodyne.zip` | `cooperative-vehicle-infrastructure-vehicle-side-velodyne.zip` | AI Studio 为单个完整 zip |

如需从 AI Studio 继续下载，先提供登录 cookie 或 token，然后运行：

```bash
cd /raid/xuyifan/v2x_code_ckpt
AISTUDIO_COOKIE='...' RETRY_SLEEP_SECONDS=600 bash scripts/download_dair_v2x_aistudio_until_complete.sh
```

监控下载状态：

```bash
cd /raid/xuyifan/v2x_code_ckpt
bash scripts/monitor_dair_v2x_download.sh --watch 30 --tail 50
```

完整数据 manifest 记录了 8 个 full 数据文件，目标目录为：

```text
datasets/DAIR-V2X/raw/
```

需要下载完成的文件：

| 文件 | 用途 |
|---|---|
| `cooperative-vehicle-infrastructure-infrastructure-side-image-deprecated.zip` | legacy camera baselines |
| `cooperative-vehicle-infrastructure-infrastructure-side-velodyne.z01` | lidar baselines |
| `cooperative-vehicle-infrastructure-infrastructure-side-velodyne.zip` | lidar baselines |
| `cooperative-vehicle-infrastructure-vehicle-side-image.zip` | camera baselines |
| `cooperative-vehicle-infrastructure-vehicle-side-velodyne.z01` | lidar baselines |
| `cooperative-vehicle-infrastructure-vehicle-side-velodyne.z02` | lidar baselines |
| `cooperative-vehicle-infrastructure-vehicle-side-velodyne.zip` | lidar baselines |
| `cooperative-vehicle-infrastructure.zip` | labels, calibration, splits |

完成判定：

1. 8 个 manifest 文件均非空存在于 `datasets/DAIR-V2X/raw/`。
2. `bash scripts/prepare_dair_v2x_layout.sh` 能成功解压/校验并创建 OpenDAIRV2X symlink。
3. `bash scripts/check_dair_v2x_prereqs.sh` 通过。
4. `external/DAIR-V2X/data/DAIR-V2X/cooperative-vehicle-infrastructure` 指向可用 full 数据目录。

## 2. 已完成工作进展

### 2.1 数据与环境

- V2X-Sim 数据已解压：
  - `V2X-Sim-det/`
  - 约 `73G`
  - 包含 `train/val/test` 和 `agent1..agent5`
- DAIR-V2X example 数据已下载、解压并完成 smoke：
  - `datasets/DAIR-V2X/example-cooperative-vehicle-infrastructure`
  - `external/DAIR-V2X/data/DAIR-V2X/example-cooperative-vehicle-infrastructure`
- full DAIR-V2X-C 数据仍在下载/重试。
- `dair-v2x-openmmlab` 环境已准备好，用于官方 OpenDAIRV2X baseline。
- `Android-Lab` 环境用于本项目 VSPM/DLPCM 训练。
- DAIR 官方 checkpoint 已补齐：
  - late-fusion PointPillars inf/veh。
  - early-fusion PointPillars。
  - late-fusion ImVoxelNet inf/veh。
  - FFNet checkpoint。

### 2.2 当前核心代码

| 文件 | 状态 |
|---|---|
| `src/v2x_forecasting/train_bev.py` | ConvLSTM VSPM 训练/评估，支持 smoke 控制和 `--device` |
| `src/v2x_forecasting/train_bev_compressed.py` | MST/压缩态训练，支持 GRU/FP16/DS64/bottleneck 和 `--device` |
| `src/v2x_forecasting/robustness.py` | packet loss、BEV dropout、false positive、pose noise 采样工具 |
| `src/v2x_forecasting/cache.py` | LRU/TTL/range cache 模拟 |
| `src/v2x_forecasting/profiling.py` | 参数量、state size、latency profile |
| `src/v2x_forecasting/delay_simulator.py` | 通信/计算延迟 frame-level helper |
| `scripts/prepare_dair_vspm_bev.py` | DAIR 标注到 VSPM BEV 序列的轻量 adapter |
| `scripts/run_dair_v2x_full_workflow.sh` | full DAIR 到官方 baseline 和我们方法的一键 workflow |

### 2.3 V2X-Sim revision 实验完成情况

VSPM sensitivity 已完成：

| run | best step | best IoU | final IoU |
|---|---:|---:|---:|
| `T5_n5` | 13800 | 0.7505 | 0.7505 |
| `T10_n5` | 13050 | 0.7513 | 0.7513 |
| `T20_n5` | 20000 | 0.7500 | 0.7496 |
| `T30_n5` | 33000 | 0.7521 | 0.7518 |
| `T10_n3` | 13000 | 0.7681 | 0.7678 |
| `T10_n10` | 23000 | 0.7248 | 0.7243 |
| `T10_n15` | 46200 | 0.7075 | 0.7075 |

MST/system-state ablation 已完成：

| run | best step | best IoU | final IoU |
|---|---:|---:|---:|
| `baseline` | 11000 | 0.7495 | 0.7488 |
| `gru_fp16` | 13000 | 0.7469 | 0.7462 |
| `ds64_fp16` | 13050 | 0.7200 | 0.7200 |
| `bottleneck12` | 13000 | 0.7213 | 0.7193 |

Checkpoint 位置：

```text
runs/revision_vspm_sensitivity/*/checkpoints/ckpt_*.pth
runs/revision_mst_ablation/*/checkpoints/ckpt_*.pth
```

结果入口：

```text
results/summary_revision/training_summary.csv
results/summary_revision/training_timeseries.csv
results/revision_tables/
results/revision_figures/
results/revision_plan/pipeline_status.md
```

### 2.4 Profile、scalability、robustness 已完成

VSPM profile：

| model | params | state MiB/agent | mean rollout ms |
|---|---:|---:|---:|
| `convlstm` | 85569 | 16.000 | 3.590 |
| `compressed_lstm` | 85569 | 16.000 | 3.632 |
| `gru_fp16` | 66497 | 4.000 | 4.400 |
| `ds64_fp16` | 66497 | 0.250 | 4.209 |
| `bottleneck12` | 67309 | 0.250 | 4.314 |

Scalability/cache profile 已完成 `N=1/2/3/4/5/8/16`。

Robustness 已完成 `T10_n5` checkpoint、256 samples：

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

## 3. 主要缺口

1. full DAIR-V2X-C 数据尚未完成下载，因此 DAIR 正式实验还未开始。
2. DAIR example smoke 只能证明 pipeline 可运行，不能作为正式 benchmark。
3. 当前 VSPM 代码主要评估 BEV occupancy IoU；完整 detection mAP 依赖 OpenDAIRV2X 官方 pipeline。
4. 当前论文 `DLPCM/4_experiment.tex` 尚未系统整合 revision 新结果。
5. 还缺少面向审稿意见的系统级 ablation、pose noise、blind/NLOS 信息增益、强 baseline 补充或不可复现说明。

## 4. 后续待完成事项与可执行小任务

### A. full DAIR-V2X-C 数据闭环

任务 A1：完成 full 数据下载。

- 输入：`datasets/DAIR-V2X/dair_v2x_c_manifest.csv`
- 命令：

```bash
cd /raid/xuyifan/v2x_code_ckpt
bash scripts/download_dair_v2x_c.sh
```

- 输出：
  - `datasets/DAIR-V2X/raw/` 下 8 个 full 数据文件。
  - 下载日志 `results/dair_v2x_dataset_download_*.log`。
- 完成标准：
  - 所有 manifest 文件非空存在。
  - 脚本 exit code 为 0。

任务 A2：解压并准备 OpenDAIRV2X 数据布局。

- 前置：A1 完成。
- 命令：

```bash
cd /raid/xuyifan/v2x_code_ckpt
bash scripts/prepare_dair_v2x_layout.sh
```

- 输出：
  - `datasets/DAIR-V2X/cooperative-vehicle-infrastructure/`
  - `external/DAIR-V2X/data/DAIR-V2X/cooperative-vehicle-infrastructure`
- 完成标准：
  - cooperative、vehicle-side、infrastructure-side 目录和 split/calib/label 文件可读。

任务 A3：检查 DAIR 官方 baseline 运行前置条件。

- 前置：A2 完成。
- 命令：

```bash
cd /raid/xuyifan/v2x_code_ckpt
bash scripts/check_dair_v2x_prereqs.sh
```

- 输出：
  - `results/dair_v2x_prereq_check_*.log`
- 完成标准：
  - OpenDAIRV2X 数据、环境、关键 checkpoint 均可被脚本识别。

### B. DAIR 官方 baseline 正式实验

任务 B1：运行可用 late-fusion point-cloud baselines。

- 前置：A3 完成。
- 命令：

```bash
cd /raid/xuyifan/v2x_code_ckpt
ENV_NAME=dair-v2x-openmmlab DEVICE_ID=0 K_LIST="0 1 2" STRICT_CHECKPOINTS=0 \
  bash scripts/run_dair_v2x_official_baselines.sh
```

- 输出：
  - `results/dair_v2x/official_baselines/logs/`
  - veh-only、inf-only、late-fusion TCLF/no-comp 的 AP 和通信量。
- 实验设置：
  - `k=0,1,2` 与现有 example smoke 保持一致。
  - 若 full 论文表需要更密集延迟，扩展到 `K_LIST="0 1 2 3 4 5"`。
- 完成标准：
  - 每个可用 baseline 均有完整 log，包含 3D AP、BEV AP、avg communication。

任务 B2：下载/补齐被 quota 阻塞的官方 checkpoints。

- 前置：网络可用。
- 当前状态：已完成，日志为 `results/dair_v2x_checkpoint_download_20260527_084044.log`。
- 命令：

```bash
cd /raid/xuyifan/v2x_code_ckpt
bash scripts/download_dair_v2x_checkpoints.sh
```

- 输出：
  - early-fusion PointPillars checkpoint。
  - infra/vehicle ImVoxelNet checkpoints。
- 完成标准：
  - `STRICT_CHECKPOINTS=1 bash scripts/run_dair_v2x_official_baselines.sh` 不再因 checkpoint 缺失退出。

任务 B3：整理 DAIR 官方 baseline 表。

- 前置：B1/B2 至少 B1 完成。
- 小任务：
  1. 写解析脚本，从 official baseline logs 中抽取 `AP@0.3/0.5/0.7` 和 `avg comm`。
  2. 输出 `results/dair_v2x/official_baselines/summary.csv`。
  3. 输出 `results/dair_v2x/official_baselines/summary.tex`。
  4. 在论文实验部分明确哪些 baseline 是 full benchmark，哪些 checkpoint 缺失导致暂不可跑。

### C. DAIR 上训练我们的方法

任务 C1：将 full DAIR 转换为 VSPM BEV 序列。

- 前置：A2 完成。
- 当前状态：已完成 full label-derived BEV 转换，尽管 raw image/velodyne full 包仍缺失。
- 命令：

```bash
cd /raid/xuyifan/v2x_code_ckpt
python scripts/prepare_dair_vspm_bev.py \
  --dair-root datasets/DAIR-V2X/cooperative-vehicle-infrastructure \
  --split-path external/DAIR-V2X/data/split_datas/cooperative-split-data.json \
  --sensor-view vehicle \
  --output-root datasets/DAIR-V2X/vspm_bev
```

- 输出：
  - `datasets/DAIR-V2X/vspm_bev/train/agent_vehicle/`
  - `datasets/DAIR-V2X/vspm_bev/val/agent_vehicle/`
- 完成标准：
  - train/val 下生成符合 `SeqDataset` 的 `<scene>_<frame>/0.npy` 序列。

任务 C2：统计 DAIR BEV adapter 输出。

- 前置：C1 完成。
- 当前状态：
  - 脚本已完成：`scripts/summarize_dair_vspm_bev.py`。
  - partial smoke 输出：`results/dair_vspm_bev_partial_summary.csv`。
  - full 输出：`results/dair_vspm_bev_full_summary.csv`。
- 小任务：
  1. 统计 train/val frame 数、scene 数、空帧比例。
  2. 检查每个 scene 是否有足够长度支持 `T=10,n=5`。
  3. 输出 `results/dair_v2x/vspm_bev_dataset_stats.csv`。
  4. 若序列不足，调整 `T/n` 或 adapter split 策略。

任务 C3：在 DAIR 上训练我们方法的主模型。

- 前置：C1/C2 完成。
- 当前状态：已启动正式训练，见 `results/dair_v2x_full_train_vehicle_T10_n5.pid` 和 `results/dair_v2x_full_train_vehicle_T10_n5_20260527_090013.log`。
- 命令：

```bash
cd /raid/xuyifan/v2x_code_ckpt
WANDB_MODE=offline conda run -n Android-Lab bash -lc '
python src/v2x_forecasting/train_bev.py \
  --data_root datasets/DAIR-V2X/vspm_bev/train/agent_vehicle \
  --data_root_val datasets/DAIR-V2X/vspm_bev/val/agent_vehicle \
  --T 10 --n 5 --batch 16 --epochs 30 \
  --use_dice \
  --teacher_forcing_start 1.0 \
  --teacher_forcing_end 0.0 \
  --teacher_forcing_epochs 10 \
  --ckpt_interval 1000 \
  --eval_interval 1000 \
  --ckpt_dir runs/dair_v2x_full/our_method_T10_n5/checkpoints \
  --project bev-encdec \
  --name dair_v2x_full_T10_n5 \
  --device cuda
'
```

- 输出：
  - `runs/dair_v2x_full/our_method_T10_n5/checkpoints/ckpt_*.pth`
  - `runs/dair_v2x_full/our_method_T10_n5/checkpoints/wandb_logs.txt`
- 完成标准：
  - 至少 30 epoch 完成。
  - 选出 best checkpoint，记录 IoU、Delta IoU、Dynamic IoU。

任务 C4：补充 DAIR 上的 T/n sensitivity。

- 前置：C3 主模型确认能稳定训练。
- 设置参考 V2X-Sim：
  - history length：`T=5,10,20,30`，固定 `n=5`
  - rollout horizon：`n=3,5,10,15`，固定 `T=10`
- 输出：
  - `runs/dair_v2x_full/vspm_sensitivity/*`
  - `results/dair_v2x/summary/training_summary.csv`
  - `results/dair_v2x/figures/dair_vspm_training_iou.pdf`
- 完成标准：
  - 形成可与 V2X-Sim sensitivity 对照的 DAIR 表。

### D. DAIR 不同通信/计算时延结果

任务 D1：实现或扩展 DAIR delay evaluation 脚本。

- 前置：C3 best checkpoint。
- 当前状态：
  - 已新增 `scripts/eval_dair_vspm_delay_grid.py`。
  - 已完成 partial smoke：`results/dair_vspm_delay_grid_partial_smoke.csv`。
  - 当前脚本实现 dual-latency mode，即 remote rollout horizon = `d_comm + d_comp`。
  - 后续仍需扩展 `--mode` 支持 `no_comp/comm_only/comp_only/dual`。
- 小任务：
  1. 输入 best checkpoint、DAIR BEV val/test root、`T/n`、`d_comm`、`d_comp`。
  2. 对每个样本构造 delayed remote 和 local result-time target。
  3. 输出每组 `(d_comm,d_comp)` 下的 IoU、Delta IoU、Dynamic IoU。
  4. 增加 `--mode` 支持系统级消融。

任务 D2：跑主时延矩阵。

- 前置：D1 完成。
- 参数参考现有 DLPCM/V2X-Sim 论文设置：
  - communication delay：`0,1,2,3,4,5` frames。
  - computation delay：主表使用 `0,2,4` frames。
  - 如需完整 heatmap，补充 `0,1,2,3,4` frames。
  - 每帧 `0.2s`。
- 输出：
  - `results/dair_v2x/our_method/delay_grid_T10_n5.csv`
  - `results/dair_v2x/our_method/delay_grid_T10_n5.tex`
  - `results/dair_v2x/figures/dair_delay_heatmap.pdf`
- 完成标准：
  - 每个 `d_comm x d_comp` 组合都有结果。

任务 D3：导出不同延迟下预测结果。

- 前置：D1/D2 完成。
- 小任务：
  1. 对 `d_comm=0..5,d_comp=0/2/4` 保存 per-sample prediction。
  2. 输出目录按 `results/dair_v2x/our_method/predictions/comm{c}_comp{k}/` 组织。
  3. 每组保存 `metrics.csv`、预测 sparse `.npy`、配置 `config.json`。
- 完成标准：
  - 可复查任意延迟设置下的样本级预测和汇总指标。

### E. DAIR 上 MST/通信带宽实验

任务 E1：训练 DAIR 上的 MST variants。

- 前置：C1/C2。
- 当前状态：
  - 已参数化 `scripts/run_mst_ablation.sh` 支持 `TRAIN_SPLIT/VAL_SPLIT/TRAIN_AGENT/VAL_AGENT/DEVICE`。
  - 已新增并启动 `scripts/run_dair_mst_ablation_parallel.sh`，四个 variant 在 GPU1-4 并行运行。
- variants：
  - `baseline`
  - `gru_fp16`
  - `ds64_fp16`
  - `bottleneck12`
- 命令模板：

```bash
cd /raid/xuyifan/v2x_code_ckpt
DATA_ROOT=datasets/DAIR-V2X/vspm_bev RUN_ROOT=runs/dair_v2x_full/mst_ablation DRY_RUN=0 \
  bash scripts/run_mst_ablation.sh
```

- 注意：
  - 当前 `run_mst_ablation.sh` 默认数据路径是 `${DATA_ROOT}/train/agent1`，DAIR 目录是 `train/agent_vehicle`，需要先扩展脚本参数或写 DAIR 专用 wrapper。
- 输出：
  - `runs/dair_v2x_full/mst_ablation/*/checkpoints/`
  - `results/dair_v2x/summary/mst_training_summary.csv`

任务 E2：计算 DAIR MST bandwidth/performance tradeoff。

- 前置：E1。
- 小任务：
  1. 对每个 variant 统计 state bytes/frame。
  2. 统计 params、state MiB/agent、latency。
  3. 汇总 best IoU、Delta IoU、Dynamic IoU。
  4. 输出 Pareto 图。
- 输出：
  - `results/dair_v2x/tables/mst_bandwidth_table.csv`
  - `results/dair_v2x/figures/mst_bandwidth_pareto.pdf`

### F. 面向审稿意见的系统级 ablation

任务 F1：定义 system-level ablation protocol。

- variants：
  - No compensation。
  - Communication-only compensation。
  - Computation-only compensation。
  - Dual-latency compensation。
  - Sender-side VSPM without MST。
  - MST only。
  - Receiver-side generic prediction。
  - Full DLPCM。
- 输出：
  - `docs/SYSTEM_ABLATION_PROTOCOL_CN.md`
  - 每个 variant 的输入、输出、延迟定义、是否使用 state transmission。
- 当前状态：已完成 `docs/SYSTEM_ABLATION_PROTOCOL_CN.md`。

任务 F2：实现 VSPM-level system ablation evaluation。

- 前置：D1。
- 小任务：
  1. 在 `eval_dair_vspm_delay_grid.py` 增加 `--mode`。
  2. mode 支持 `no_comp`, `comm_only`, `comp_only`, `dual`。
  3. 输出统一 CSV。
- 输出：
  - `results/dair_v2x/system_ablation/vspm_ablation.csv`

任务 F3：如 full detection baseline 可跑，补 detection-level ablation。

- 前置：B1/B2。
- 小任务：
  1. 在 OpenDAIRV2X pipeline 中映射 no-comp/TCLF/late-fusion variants。
  2. 对 `k=0..5` 和 `comp=0/2/4` 跑 AP。
  3. 与我们方法表格对齐。
- 输出：
  - `results/dair_v2x/system_ablation/detection_ablation.csv`

### G. 鲁棒性、pose noise、真实部署风险

任务 G1：DAIR 上 packet loss / sensor degradation robustness。

- 前置：C3 best checkpoint。
- 参数沿用 V2X-Sim：
  - packet loss：`0,0.1,0.2,0.3,0.5`
  - BEV dropout：`0,0.05,0.1,0.2`
  - false positive：`0,0.001,0.005`
- 输出：
  - `results/dair_v2x/robustness/vspm_robustness_T10_n5.csv`
  - `results/dair_v2x/figures/dair_vspm_robustness.pdf`

任务 G2：实现 pose noise robustness。

- 当前已有 `sample_pose_noise`，但没有 BEV warp/pose transform evaluation。
- 小任务：
  1. 在 DAIR adapter 中保留 vehicle/infrastructure pose 元数据。
  2. 新增 BEV warp/transform 工具。
  3. 注入 translation noise：`0,0.2,0.5,1.0,2.0 m`。
  4. 注入 rotation noise：`0,0.5,1.0,2.0 deg`。
  5. 输出 IoU/AP degradation 曲线。
- 输出：
  - `results/dair_v2x/robustness/pose_noise.csv`
  - `results/dair_v2x/figures/pose_noise.pdf`

任务 G3：写 Discussion 中的 domain gap 和 spatial calibration 限制。

- 前置：G1/G2 至少 G1。
- 输出：
  - 修改 `DLPCM/5_conclusion.tex` 或新增 Discussion 段。
  - 明确 DLPCM 解决 temporal alignment，不替代 spatial calibration。

### H. N-agent scalability 与 cache management

任务 H1：将已有 V2X-Sim scalability 结果写入论文。

- 输入：
  - `results/revision_tables/scalability_table.csv`
  - `results/revision_figures/cache_scalability.pdf`
- 输出：
  - `DLPCM/table/scalability_table.tex`
  - `DLPCM/pics/cache_scalability.pdf`
  - `DLPCM/3_dual.tex` 中 cache policy 段。

任务 H2：补 cache eviction ablation。

- variants：
  - no eviction。
  - TTL only。
  - LRU only。
  - range + TTL + LRU。
- 小任务：
  1. 扩展 `scripts/run_scalability_profile.py`。
  2. 输出 hit/miss/eviction/memory。
  3. 生成表格。
- 输出：
  - `results/revision_tables/cache_eviction_table.csv`
  - `DLPCM/table/cache_eviction_table.tex`

### I. Blind spot / NLOS 信息增益

任务 I1：定义 ego-visible 与 ego-invisible/NLOS 指标。

- 当前缺口：
  - 本 repo 没有 visibility/occlusion mask。
- 小任务：
  1. 对 V2X-Sim 或 DAIR 选定评价对象。
  2. 基于 ego LiDAR FOV 或 occupancy ray-casting 近似 invisible region。
  3. 输出 mask 生成脚本。
- 输出：
  - `scripts/build_visibility_masks.py`
  - `results/*/visibility_masks/`

任务 I2：计算 blind/NLOS 区域指标。

- 对比：
  - onboard-only。
  - delayed cooperative。
  - DLPCM。
  - oracle。
- 输出：
  - `results/revision_tables/blind_spot_table.csv`
  - qualitative case figure。

### J. 论文与 response letter 整理

任务 J1：把 revision 结果写入论文实验部分。

- 输入：
  - `results/revision_tables/*.csv`
  - `results/revision_figures/*.pdf`
  - `results/summary_revision/*.csv`
- 小任务：
  1. 新增 VSPM sensitivity 表。
  2. 新增 MST profile/bandwidth 表。
  3. 新增 robustness 表。
  4. 新增 scalability/cache 表。
  5. 替换或补充现有图。
- 输出：
  - `DLPCM/4_experiment.tex`
  - `DLPCM/table/*.tex`
  - `DLPCM/pics/*.pdf`

任务 J2：修正文稿显式问题。

- 当前状态：
  1. 已完成：`Vehivle-to-Vehicle` 改为 `Vehicle-to-Vehicle`。
  2. 已完成：`DiscnNet` 改为 `DiscoNet`。
  3. 已完成：`step-iou.pdf` 图注中 x-axis 改为 prediction horizon。
  4. 待完成：避免 abstract/conclusion 中过强 real-world deployment 声称，除非 DAIR full 结果已完成。

任务 J3：逐条整理 reviewer response。

- 输出：
  - `docs/REVIEWER_RESPONSE_DRAFT_CN.md`
- 每条 response 包含：
  1. 审稿人问题。
  2. 已完成修改/实验。
  3. 对应论文位置。
  4. 新增表/图/结果文件。
  5. 若无法完成，给出技术原因和诚实限制。

## 5. 建议执行顺序

1. A1-A3：先完成 full DAIR 数据闭环。
2. B1/B2：跑官方 DAIR baseline，确认 full benchmark 可用。
3. C1-C3：训练 DAIR 上我们方法主 checkpoint。
4. D1-D3：得到 DAIR 不同通信/计算时延结果。
5. E1-E2：补 DAIR MST 带宽/性能权衡。
6. G1：先补 DAIR robustness，pose noise 可作为下一层增强。
7. F1-F3：系统级 ablation，根据 DAIR/full detection 可用程度决定 VSPM-level 还是 detection-level。
8. H/I：cache scalability 与 blind/NLOS 信息增益。
9. J1-J3：论文和 response letter 整合。
