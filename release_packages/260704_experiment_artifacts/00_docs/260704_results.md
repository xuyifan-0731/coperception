# 260704 实验结果整理

## 0. 口径说明

本文档沿用 `260615结果.md` 的表格顺序和行文结构整理。表中 `-` 表示该方法当前没有可证明满足本轮口径的真实结果，因此不写入旧数值或伪结果。

本轮统一实验设置如下：

1. 无时延补偿的基线方法：发送端使用 `t - 通信时延 - 计算时延` 那一帧的感知数据；接收端使用 `t - 计算时延` 那一帧的感知数据；将上述两帧数据直接使用 baseline 方法融合并检测；检测结果与 `t` 时刻 GT 比对。
2. 我们的预测补偿方法：发送端将 `t - 通信时延 - 计算时延` 那一帧的感知数据通过我们的预测模型补偿到 `t`；接收端将 `t - 计算时延` 的感知信息预测到 `t`；将双方 `t` 时刻预测结果使用同一 baseline 融合并检测；检测结果与 `t` 时刻 GT 比对。
3. 对比实验方法：发送端将 `t - 通信时延 - 计算时延` 的感知信息通过对比方法预测补偿到 `t - 计算时延`；接收端使用 `t - 计算时延` 时刻的感知信息；将二者融合；融合结果与 `t` 时刻 GT 比对。

时延单位均为 frame。当前实验默认 `1 frame = 0.2 s`。V2X-Sim 表格中的 AP 统一写成百分数形式；DATA 原始 CSV 为 0 到 1 的小数，这里已乘以 100。DAIR-V2X 表格为官方 detector 的 BEV AP 或 LRCP 复现 detector BEV AP。

## 1. V2X-Sim 车车协同

### 1.1 V2VNet 作为 baseline

数据集：`V2X-Sim`。Baseline 方法：`V2VNet`。预测补偿方法包括 `Ours` 和 `SyncNet`。本节按要求直接填入 DLPCM 论文已有实验结果：`V2VNet` 与 `V2VNet + Ours` 来自 `DLPCM/table/main_result.tex`；`V2VNet + SyncNet` 使用论文中 `DLPCM/table/main_result_disco.tex` 的 `SyncNet w/ Comp.` 结果作为已有对比方法结果。

#### 1.1.1 AP@0.5

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | V2VNet | 0 | - | 64.91 | - | - | - | - | - |
| 不同时延无补偿 | V2VNet | 0 | 57.43 | 64.91 | 60.01 | 55.69 | 54.44 | 53.79 | 53.21 |
| 不同时延无补偿 | V2VNet | 2 | 54.63 | 55.45 | 54.91 | 54.73 | 54.55 | 54.23 | 53.89 |
| 不同时延无补偿 | V2VNet | 4 | 53.52 | 53.33 | 53.55 | 53.69 | 53.61 | 53.53 | 53.39 |
| 不同时延 + 我们方法补偿 | V2VNet + Ours | 0 | 64.41 | 67.45 | 65.16 | 64.06 | 63.56 | 63.19 | 63.04 |
| 不同时延 + 我们方法补偿 | V2VNet + Ours | 2 | 57.01 | 57.54 | 57.22 | 56.97 | 56.91 | 56.78 | 56.63 |
| 不同时延 + 我们方法补偿 | V2VNet + Ours | 4 | 56.07 | 56.40 | 56.21 | 56.15 | 55.96 | 55.90 | 55.80 |
| 不同时延 + SyncNet 补偿 | V2VNet + SyncNet | 0 | 55.07 | 56.62 | 53.49 | 55.73 | 55.72 | 54.90 | 52.56 |
| 不同时延 + SyncNet 补偿 | V2VNet + SyncNet | 2 | 39.94 | 41.25 | 39.73 | 41.52 | 40.53 | 39.28 | 37.35 |
| 不同时延 + SyncNet 补偿 | V2VNet + SyncNet | 4 | 38.55 | 38.65 | 38.25 | 40.34 | 39.53 | 38.21 | 36.31 |

#### 1.1.2 AP@0.7

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | V2VNet | 0 | - | 57.26 | - | - | - | - | - |
| 不同时延无补偿 | V2VNet | 0 | 47.61 | 57.26 | 47.03 | 45.46 | 45.24 | 44.78 | 44.51 |
| 不同时延无补偿 | V2VNet | 2 | 45.20 | 45.29 | 45.40 | 45.35 | 45.27 | 45.05 | 44.82 |
| 不同时延无补偿 | V2VNet | 4 | 44.45 | 44.35 | 44.65 | 44.57 | 44.47 | 44.39 | 44.28 |
| 不同时延 + 我们方法补偿 | V2VNet + Ours | 0 | 56.41 | 58.64 | 57.12 | 56.25 | 55.68 | 55.43 | 55.34 |
| 不同时延 + 我们方法补偿 | V2VNet + Ours | 2 | 50.45 | 51.02 | 50.64 | 50.42 | 50.36 | 50.22 | 50.03 |
| 不同时延 + 我们方法补偿 | V2VNet + Ours | 4 | 49.45 | 49.87 | 49.65 | 49.46 | 49.39 | 49.25 | 49.09 |
| 不同时延 + SyncNet 补偿 | V2VNet + SyncNet | 0 | 47.27 | 50.20 | 47.08 | 47.84 | 47.72 | 46.53 | 44.26 |
| 不同时延 + SyncNet 补偿 | V2VNet + SyncNet | 2 | 33.20 | 33.26 | 34.02 | 34.84 | 33.93 | 32.52 | 30.62 |
| 不同时延 + SyncNet 补偿 | V2VNet + SyncNet | 4 | 32.43 | 31.86 | 33.06 | 34.17 | 33.44 | 31.97 | 30.06 |

### 1.2 DATA 作为 baseline

数据集：`V2X-Sim 2.0`。Baseline 方法：`DATA` intermediate fusion detector。无补偿、`DATA + Ours` 和 `DATA + SyncNet` 均已检查/更新到新口径：ego 使用 `comp_delay` 旧帧，非 ego 使用 `comm_delay + comp_delay` 旧帧，标签仍为当前帧；Ours 与 SyncNet 行都将两侧输入补到当前帧 `T` 后评估。

数据来源：无补偿为 `runs/data_v2xsim_pipeline/results/data_v2xsim_latency_ap.csv`；Ours 为 `results/260615_fill/artifacts/v2xsim_data_compensation_ap.csv`；SyncNet strict 重测为 `results/260615_fill/artifacts/v2xsim_data_syncnet_strict_20260704.csv`。新增联合训练补偿结果来自 `results/260615_fill/artifacts/v2xsim_data_ifam_joint_delay_comp_scale01_20260713.csv`，模型为 `checkpoints/curated/ours_models/v2xsim_data_ifam_joint_delay_comp_scale01.pth`；该方法冻结 DATA detector，在现有输入级 pose-warp 之后、IFAM 融合之前加入 gated residual feature compensator，并采用保守 residual scale。未缩放诊断版结果保存在 `results/260615_fill/artifacts/v2xsim_data_ifam_joint_delay_comp_20260713.csv`。

#### 1.2.1 AP@0.5

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | DATA | 0 | - | 84.46 | - | - | - | - | - |
| 不同时延无补偿 | DATA | 0 | 82.75 | 84.46 | 83.47 | 82.71 | 82.19 | 81.89 | 81.78 |
| 不同时延无补偿 | DATA | 2 | 72.22 | 72.14 | 72.03 | 72.34 | 72.24 | 72.26 | 72.33 |
| 不同时延无补偿 | DATA | 4 | 68.37 | 67.28 | 67.79 | 68.59 | 68.69 | 68.94 | 68.90 |
| 不同时延 + 我们方法补偿 | DATA + Ours | 0 | 82.78 | 84.47 | 83.53 | 82.72 | 82.21 | 81.95 | 81.79 |
| 不同时延 + 我们方法补偿 | DATA + Ours | 2 | 72.36 | 72.21 | 72.28 | 72.58 | 72.44 | 72.29 | 72.35 |
| 不同时延 + 我们方法补偿 | DATA + Ours | 4 | 68.56 | 67.63 | 68.13 | 68.78 | 68.82 | 69.02 | 68.98 |
| 不同时延 + 联合训练补偿 | DATA + Ours joint IFAM residual | 0 | 82.85 | 84.47 | 83.55 | 82.82 | 82.32 | 82.07 | 81.88 |
| 不同时延 + 联合训练补偿 | DATA + Ours joint IFAM residual | 2 | 72.82 | 72.93 | 72.81 | 73.01 | 72.84 | 72.57 | 72.77 |
| 不同时延 + 联合训练补偿 | DATA + Ours joint IFAM residual | 4 | 69.21 | 68.70 | 68.94 | 69.38 | 69.28 | 69.51 | 69.46 |
| 不同时延 + SyncNet 补偿 | DATA + SyncNet | 0 | 82.78 | 84.47 | 83.51 | 82.72 | 82.20 | 81.96 | 81.81 |
| 不同时延 + SyncNet 补偿 | DATA + SyncNet | 2 | 72.37 | 72.24 | 72.28 | 72.59 | 72.43 | 72.31 | 72.35 |
| 不同时延 + SyncNet 补偿 | DATA + SyncNet | 4 | 68.56 | 67.66 | 68.16 | 68.71 | 68.80 | 69.01 | 69.00 |

#### 1.2.2 AP@0.7

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | DATA | 0 | - | 78.85 | - | - | - | - | - |
| 不同时延无补偿 | DATA | 0 | 76.28 | 78.85 | 76.67 | 76.15 | 75.40 | 75.47 | 75.16 |
| 不同时延无补偿 | DATA | 2 | 62.83 | 62.31 | 62.61 | 63.06 | 62.88 | 63.04 | 63.09 |
| 不同时延无补偿 | DATA | 4 | 58.93 | 58.08 | 58.66 | 59.21 | 59.17 | 59.28 | 59.18 |
| 不同时延 + 我们方法补偿 | DATA + Ours | 0 | 76.38 | 78.90 | 76.75 | 76.28 | 75.46 | 75.47 | 75.39 |
| 不同时延 + 我们方法补偿 | DATA + Ours | 2 | 63.26 | 62.80 | 63.20 | 63.46 | 63.34 | 63.39 | 63.36 |
| 不同时延 + 我们方法补偿 | DATA + Ours | 4 | 59.80 | 59.14 | 59.50 | 60.00 | 59.80 | 60.19 | 60.15 |
| 不同时延 + 联合训练补偿 | DATA + Ours joint IFAM residual | 0 | 76.36 | 78.89 | 76.70 | 76.21 | 75.48 | 75.60 | 75.28 |
| 不同时延 + 联合训练补偿 | DATA + Ours joint IFAM residual | 2 | 63.74 | 63.62 | 63.79 | 63.92 | 63.69 | 63.70 | 63.71 |
| 不同时延 + 联合训练补偿 | DATA + Ours joint IFAM residual | 4 | 60.32 | 59.89 | 60.28 | 60.31 | 60.21 | 60.56 | 60.65 |
| 不同时延 + SyncNet 补偿 | DATA + SyncNet | 0 | 76.37 | 78.87 | 76.68 | 76.32 | 75.51 | 75.40 | 75.46 |
| 不同时延 + SyncNet 补偿 | DATA + SyncNet | 2 | 63.25 | 62.81 | 63.11 | 63.50 | 63.37 | 63.31 | 63.38 |
| 不同时延 + SyncNet 补偿 | DATA + SyncNet | 4 | 59.84 | 59.24 | 59.51 | 60.03 | 59.87 | 60.20 | 60.17 |

分析：DATA 的无补偿、Ours、SyncNet 和 joint IFAM residual 均按新口径验证。18 个双时延格点上，AP@0.5 从无补偿 `74.45` 提升到 Ours pose-warp `74.57`，再提升到 joint IFAM residual `74.96`；AP@0.7 从无补偿 `66.01` 提升到 Ours pose-warp `66.48`，再提升到 joint IFAM residual `66.81`。极端设置 `Comp4/Comm5` 下，无补偿为 `68.90/59.18`，Ours pose-warp 为 `68.98/60.15`，joint IFAM residual 为 `69.46/60.65`。未缩放 joint 诊断版的 AP@0.5 为 `75.33`，但 AP@0.7 降到 `64.66`，说明 residual 过强会提高宽松 IoU 下的召回但损害高 IoU 定位精度；因此主表采用保守 residual scale 版本。

#### 1.2.3 V2X-Sim DATA 公平性核查

核查结论：V2X-Sim 上没有出现 DAIR-V2X DATA 那种“把 DATA intermediate fusion 与单侧 box late-fusion 外接 RF 混在一起比较”的问题。现有 V2X-Sim DATA 表格里的无补偿、`DATA + Ours` 和 `DATA + SyncNet` 都走同一个 DATA intermediate inference pipeline、同一个 checkpoint、同一个 `t` 时刻 GT。`Comm0/Comp0` 下，无补偿为 `84.4644/78.8502`，`DATA + Ours` 为 `84.47/78.90`，identity 情况基本一致。

但需要注意：这里的 `DATA + Ours` 是 DATA fork 中的输入级 pose-warp 补偿，不是 DAIR 上当前最优的 `History-aware RF 2f` detector-side box predictor。本地没有 V2X-Sim 版 RF/History-aware box predictor 模型，因此没有把 DAIR RF 跨数据集直接套到 V2X-Sim 上；那样会引入新的不公平。

数据来源：`results/260615_fill/artifacts/v2xsim_data_intermediate_fairness_audit_20260707.csv` 与新增 `results/260615_fill/artifacts/v2xsim_data_ifam_joint_delay_comp_scale01_20260713.csv`。原始公平性核查 CSV 由无补偿、Ours pose-warp、SyncNet 三个结果文件合并生成；joint IFAM residual 为本轮新增联合训练结果，不删除原始结果。

| 指标 | 方法 | 18 格 AVG | Comp0 AVG | Comp2 AVG | Comp4 AVG |
|---|---|---:|---:|---:|---:|
| AP@0.5 | DATA w/o compensation | 74.45 | 82.75 | 72.22 | 68.37 |
| AP@0.5 | DATA + Ours pose-warp | 74.57 | 82.78 | 72.36 | 68.56 |
| AP@0.5 | DATA + Ours joint IFAM residual | 74.96 | 82.85 | 72.82 | 69.21 |
| AP@0.5 | Ours - w/o compensation | +0.12 | +0.03 | +0.14 | +0.19 |
| AP@0.5 | Joint - Ours pose-warp | +0.40 | +0.07 | +0.46 | +0.65 |
| AP@0.7 | DATA w/o compensation | 66.01 | 76.28 | 62.83 | 58.93 |
| AP@0.7 | DATA + Ours pose-warp | 66.48 | 76.38 | 63.26 | 59.80 |
| AP@0.7 | DATA + Ours joint IFAM residual | 66.81 | 76.36 | 63.74 | 60.32 |
| AP@0.7 | Ours - w/o compensation | +0.46 | +0.09 | +0.43 | +0.87 |
| AP@0.7 | Joint - Ours pose-warp | +0.33 | -0.02 | +0.48 | +0.52 |

分析：V2X-Sim 上公平性核查通过，问题不在实验口径，而在方法收益本身较小。`DATA + Ours pose-warp` 在 AP@0.7 上随计算时延增大收益更明显，`Comp4` 平均提升 `+0.87`；AP@0.5 总体只提升 `+0.12`，说明 DATA intermediate detector 在 V2X-Sim 上本身已经较稳，输入级 pose-warp 的边际收益有限。加入联合训练的 IFAM residual 后，`Comp2/Comp4` 的收益进一步扩大，尤其 `Comp4` 相比 pose-warp 又提升 `+0.65/+0.52`；但 `Comp0` 的 AP@0.7 基本持平略降 `-0.02`，说明该模块主要改善存在计算时延的场景，不应放大到破坏无计算时延 identity。

### 1.3 DiscoNet 作为 baseline

数据集：`V2X-Sim`。Baseline 方法：`DiscoNet`。预测补偿方法包括 `Ours` 和 `SyncNet`。本节按要求直接填入 DLPCM 论文已有实验结果，来源为 `DLPCM/table/main_result_disco.tex`。

#### 1.3.1 AP@0.5

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | DiscoNet | 0 | - | 56.62 | - | - | - | - | - |
| 不同时延无补偿 | DiscoNet | 0 | 51.46 | 56.62 | 54.88 | 51.24 | 49.46 | 48.60 | 47.94 |
| 不同时延无补偿 | DiscoNet | 2 | 39.19 | 41.25 | 41.03 | 39.75 | 38.91 | 37.38 | 36.80 |
| 不同时延无补偿 | DiscoNet | 4 | 37.87 | 38.65 | 39.41 | 38.35 | 37.70 | 36.85 | 36.25 |
| 不同时延 + 我们方法补偿 | DiscoNet + Ours | 0 | 54.52 | 56.83 | 55.09 | 54.11 | 53.83 | 53.56 | 53.70 |
| 不同时延 + 我们方法补偿 | DiscoNet + Ours | 2 | 48.15 | 48.59 | 48.22 | 48.16 | 48.11 | 47.92 | 47.89 |
| 不同时延 + 我们方法补偿 | DiscoNet + Ours | 4 | 46.99 | 47.31 | 47.13 | 46.95 | 46.98 | 46.85 | 46.75 |
| 不同时延 + SyncNet 补偿 | SyncNet w/ Comp. | 0 | 55.07 | 56.62 | 53.49 | 55.73 | 55.72 | 54.90 | 52.56 |
| 不同时延 + SyncNet 补偿 | SyncNet w/ Comp. | 2 | 39.94 | 41.25 | 39.73 | 41.52 | 40.53 | 39.28 | 37.35 |
| 不同时延 + SyncNet 补偿 | SyncNet w/ Comp. | 4 | 38.55 | 38.65 | 38.25 | 40.34 | 39.53 | 38.21 | 36.31 |

#### 1.3.2 AP@0.7

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | DiscoNet | 0 | - | 51.70 | - | - | - | - | - |
| 不同时延无补偿 | DiscoNet | 0 | 42.84 | 51.70 | 44.53 | 41.57 | 40.32 | 39.54 | 39.34 |
| 不同时延无补偿 | DiscoNet | 2 | 31.78 | 33.26 | 32.70 | 32.05 | 31.61 | 30.72 | 30.36 |
| 不同时延无补偿 | DiscoNet | 4 | 31.35 | 31.86 | 32.29 | 31.73 | 31.38 | 30.65 | 30.21 |
| 不同时延 + 我们方法补偿 | DiscoNet + Ours | 0 | 50.52 | 52.16 | 51.24 | 50.32 | 49.86 | 49.75 | 49.82 |
| 不同时延 + 我们方法补偿 | DiscoNet + Ours | 2 | 43.65 | 44.11 | 43.70 | 43.65 | 43.57 | 43.49 | 43.39 |
| 不同时延 + 我们方法补偿 | DiscoNet + Ours | 4 | 42.54 | 42.75 | 42.77 | 42.52 | 42.46 | 42.44 | 42.31 |
| 不同时延 + SyncNet 补偿 | SyncNet w/ Comp. | 0 | 47.27 | 50.20 | 47.08 | 47.84 | 47.72 | 46.53 | 44.26 |
| 不同时延 + SyncNet 补偿 | SyncNet w/ Comp. | 2 | 33.20 | 33.26 | 34.02 | 34.84 | 33.93 | 32.52 | 30.62 |
| 不同时延 + SyncNet 补偿 | SyncNet w/ Comp. | 4 | 32.43 | 31.86 | 33.06 | 34.17 | 33.44 | 31.97 | 30.06 |

## 2. DAIR-V2X 车路协同

本节已按文件开头的三类实验设置重测并重排。所有 DAIR 主表先按无补偿基线、再按我们的预测补偿、最后按对比方法补偿写入；旧 single-delay 结果、pseudo-SyncNet 诊断结果、Safe Hybrid/MLP 诊断结果和 BEV-IoU 代理表均不作为本节最终 detector AP 结果写入。注意：2.1 official late-fusion 主表已改用严格公平 `train -> val` 的 `History-aware RF 2f` detector-side box predictor；2.2 box-level/RF assist 是诊断项，旧 RF 与 strict RF 结果均已保留用于对比；2.2 DATA IFAM 与 2.4 TraF-Align 使用各自 baseline 特征空间内重新训练的补偿模型。最新真实性审计见第 5 节。

### 2.1 TCLF / official late fusion 作为 baseline

数据集：`DAIR-V2X-C`。Baseline 方法：OpenDAIRV2X official late-fusion detector。无补偿行使用发送端 `t-(Comm+Comp)` detector 结果和接收端 `t-Comp` detector 结果直接融合，并与 `t` 时刻 cooperative GT 比对。`Late fusion + Ours` 使用当前最优 `History-aware RF 2f` 模型，对发送端和接收端延时 detector box 分别补偿到 `t` 后再融合。对比方法 `Late fusion + TCLF` 使用 DAIR official `TimeCompensator(EuclidianMatcher)`，将发送端信息补偿到 `t-Comp` 后与接收端 `t-Comp` 信息融合，再与 `t` 时刻 GT 比对。当前仓库没有真实 DAIR 版 SyncNet checkpoint，因此 SyncNet 行保留为空。

数据来源：无补偿为 `results/260615_fill/artifacts/dair_late_fusion_no_comp_strict_20260704_rechecked.csv`；严格公平 Ours RF 为 `results/260615_fill/artifacts/dair_late_fusion_ours_history2_rf_trainonly_strict_20260714.csv`，checkpoint 为 `checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainonly.pkl`；TCLF receiver-target 对比方法为 `results/260615_fill/artifacts/dair_late_fusion_tclf_receiver_strict_20260704_rechecked.csv`。对应日志位于 `results/dair_v2x/official_baselines/logs_strict_dual_delay_20260703/`、`results/dair_v2x/official_baselines/logs_strict_dual_delay_ours_history2_rf_trainonly_20260714/` 与 `results/dair_v2x/official_baselines/logs_strict_dual_delay_tclf_receiver_20260704/`。

真实性审计备注：本表 Ours RF 已完成最紧急修复：训练使用 DAIR official `train` split，评测使用 DAIR official `val` split（日志显示 1789 frames，存在有效历史帧过滤时部分格点为 1760/1741 frames），不再使用旧 `val -> val` 的 RF checkpoint。旧 `checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours.pkl` 仍保留为泄漏诊断参考，不作为严格公平主结果。

#### 2.1.1 BEV AP@0.5

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | Late fusion baseline | 0 | - | 62.00 | - | - | - | - | - |
| 不同时延无补偿 | Late fusion w/o comp. | 0 | 58.25 | 62.00 | 59.90 | 58.07 | 56.90 | 56.47 | 56.18 |
| 不同时延无补偿 | Late fusion w/o comp. | 2 | 20.88 | 21.92 | 21.16 | 20.85 | 20.60 | 20.44 | 20.30 |
| 不同时延无补偿 | Late fusion w/o comp. | 4 | 12.60 | 12.96 | 12.80 | 12.66 | 12.53 | 12.39 | 12.26 |
| 不同时延 + 我们方法补偿 | Late fusion + Ours | 0 | 59.11 | 62.00 | 59.63 | 59.20 | 58.58 | 57.94 | 57.29 |
| 不同时延 + 我们方法补偿 | Late fusion + Ours | 2 | 46.69 | 48.06 | 47.50 | 46.91 | 46.26 | 45.90 | 45.52 |
| 不同时延 + 我们方法补偿 | Late fusion + Ours | 4 | 32.71 | 33.63 | 33.13 | 32.74 | 32.48 | 32.26 | 32.00 |
| 不同时延 + 对比方法补偿 | Late fusion + TCLF | 0 | 58.89 | 62.00 | 60.18 | 59.25 | 57.83 | 57.31 | 56.78 |
| 不同时延 + 对比方法补偿 | Late fusion + TCLF | 2 | 21.02 | 21.93 | 21.36 | 21.07 | 20.79 | 20.58 | 20.38 |
| 不同时延 + 对比方法补偿 | Late fusion + TCLF | 4 | 12.65 | 12.96 | 12.83 | 12.72 | 12.58 | 12.49 | 12.34 |
| 不同时延 + SyncNet 补偿 | Late fusion + SyncNet | 0 | - | - | - | - | - | - | - |
| 不同时延 + SyncNet 补偿 | Late fusion + SyncNet | 2 | - | - | - | - | - | - | - |
| 不同时延 + SyncNet 补偿 | Late fusion + SyncNet | 4 | - | - | - | - | - | - | - |

#### 2.1.2 BEV AP@0.7

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | Late fusion baseline | 0 | - | 54.11 | - | - | - | - | - |
| 不同时延无补偿 | Late fusion w/o comp. | 0 | 49.88 | 54.11 | 50.11 | 49.26 | 48.80 | 48.63 | 48.36 |
| 不同时延无补偿 | Late fusion w/o comp. | 2 | 10.22 | 10.55 | 10.36 | 10.23 | 10.08 | 10.13 | 10.00 |
| 不同时延无补偿 | Late fusion w/o comp. | 4 | 8.03 | 8.20 | 8.07 | 8.11 | 8.01 | 7.93 | 7.87 |
| 不同时延 + 我们方法补偿 | Late fusion + Ours | 0 | 50.21 | 54.11 | 50.41 | 50.00 | 49.35 | 48.85 | 48.54 |
| 不同时延 + 我们方法补偿 | Late fusion + Ours | 2 | 25.24 | 26.02 | 25.59 | 25.24 | 24.98 | 24.87 | 24.73 |
| 不同时延 + 我们方法补偿 | Late fusion + Ours | 4 | 14.41 | 14.70 | 14.51 | 14.44 | 14.34 | 14.28 | 14.21 |
| 不同时延 + 对比方法补偿 | Late fusion + TCLF | 0 | 50.50 | 54.11 | 51.13 | 50.48 | 49.46 | 49.12 | 48.70 |
| 不同时延 + 对比方法补偿 | Late fusion + TCLF | 2 | 10.27 | 10.55 | 10.41 | 10.31 | 10.17 | 10.17 | 10.01 |
| 不同时延 + 对比方法补偿 | Late fusion + TCLF | 4 | 8.04 | 8.20 | 8.08 | 8.13 | 8.00 | 7.94 | 7.87 |
| 不同时延 + SyncNet 补偿 | Late fusion + SyncNet | 0 | - | - | - | - | - | - | - |
| 不同时延 + SyncNet 补偿 | Late fusion + SyncNet | 2 | - | - | - | - | - | - | - |
| 不同时延 + SyncNet 补偿 | Late fusion + SyncNet | 4 | - | - | - | - | - | - | - |

分析：official late-fusion strict detector AP@0.7 的 18 格均值为无补偿 `22.71`、严格公平 Ours RF `29.95`、TCLF receiver-target 对比方法 `22.94`。严格公平 Ours RF 相比无补偿提升 `+7.24`，相比 TCLF 提升 `+7.01`；AP@0.5 的 18 格均值从无补偿 `30.58` 提升到 `46.17`。这说明修复数据泄漏后，我们的方法仍然有正向收益，但收益低于旧 `val -> val` 泄漏版本。

补充诊断：为回答“把 train 和 test/validation 一起训练再评估是否更好”，本地可评测 split 为 `val`，因此额外跑了 `train+val -> val` 诊断上限版，checkpoint 为 `checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainval_diagnostic.pkl`，结果为 `results/260615_fill/artifacts/dair_late_fusion_ours_history2_rf_trainval_diag_strict_20260714.csv`。该版本 AP@0.5/AP@0.7 的 18 格均值为 `50.73/35.15`，高于严格公平版 `46.17/29.95`，接近旧 `val -> val` 泄漏版 `51.39/36.67`；但它使用了评测 split 参与训练，只能作为诊断/上限，不应作为论文公平主结果。

#### 2.1.3 Detector AP 表述

该表与 2.1.1/2.1.2 使用同一批 official late-fusion detector 结果，只是按 `Detector + ...` 命名保留论文表述。数值来源同上。

##### 2.1.3.1 BEV AP@0.5

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | Detector baseline | 0 | - | 62.00 | - | - | - | - | - |
| 不同时延无补偿 | Detector w/o compensation | 0 | 58.25 | 62.00 | 59.90 | 58.07 | 56.90 | 56.47 | 56.18 |
| 不同时延无补偿 | Detector w/o compensation | 2 | 20.88 | 21.92 | 21.16 | 20.85 | 20.60 | 20.44 | 20.30 |
| 不同时延无补偿 | Detector w/o compensation | 4 | 12.60 | 12.96 | 12.80 | 12.66 | 12.53 | 12.39 | 12.26 |
| 不同时延 + 我们方法补偿 | Detector + Ours | 0 | 59.11 | 62.00 | 59.63 | 59.20 | 58.58 | 57.94 | 57.29 |
| 不同时延 + 我们方法补偿 | Detector + Ours | 2 | 46.69 | 48.06 | 47.50 | 46.91 | 46.26 | 45.90 | 45.52 |
| 不同时延 + 我们方法补偿 | Detector + Ours | 4 | 32.71 | 33.63 | 33.13 | 32.74 | 32.48 | 32.26 | 32.00 |
| 不同时延 + 对比方法补偿 | Detector + TCLF | 0 | 58.89 | 62.00 | 60.18 | 59.25 | 57.83 | 57.31 | 56.78 |
| 不同时延 + 对比方法补偿 | Detector + TCLF | 2 | 21.02 | 21.93 | 21.36 | 21.07 | 20.79 | 20.58 | 20.38 |
| 不同时延 + 对比方法补偿 | Detector + TCLF | 4 | 12.65 | 12.96 | 12.83 | 12.72 | 12.58 | 12.49 | 12.34 |
| 不同时延 + SyncNet 补偿 | Detector + SyncNet | 0 | - | - | - | - | - | - | - |
| 不同时延 + SyncNet 补偿 | Detector + SyncNet | 2 | - | - | - | - | - | - | - |
| 不同时延 + SyncNet 补偿 | Detector + SyncNet | 4 | - | - | - | - | - | - | - |

##### 2.1.3.2 BEV AP@0.7

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | Detector baseline | 0 | - | 54.11 | - | - | - | - | - |
| 不同时延无补偿 | Detector w/o compensation | 0 | 49.88 | 54.11 | 50.11 | 49.26 | 48.80 | 48.63 | 48.36 |
| 不同时延无补偿 | Detector w/o compensation | 2 | 10.22 | 10.55 | 10.36 | 10.23 | 10.08 | 10.13 | 10.00 |
| 不同时延无补偿 | Detector w/o compensation | 4 | 8.03 | 8.20 | 8.07 | 8.11 | 8.01 | 7.93 | 7.87 |
| 不同时延 + 我们方法补偿 | Detector + Ours | 0 | 50.21 | 54.11 | 50.41 | 50.00 | 49.35 | 48.85 | 48.54 |
| 不同时延 + 我们方法补偿 | Detector + Ours | 2 | 25.24 | 26.02 | 25.59 | 25.24 | 24.98 | 24.87 | 24.73 |
| 不同时延 + 我们方法补偿 | Detector + Ours | 4 | 14.41 | 14.70 | 14.51 | 14.44 | 14.34 | 14.28 | 14.21 |
| 不同时延 + 对比方法补偿 | Detector + TCLF | 0 | 50.50 | 54.11 | 51.13 | 50.48 | 49.46 | 49.12 | 48.70 |
| 不同时延 + 对比方法补偿 | Detector + TCLF | 2 | 10.27 | 10.55 | 10.41 | 10.31 | 10.17 | 10.17 | 10.01 |
| 不同时延 + 对比方法补偿 | Detector + TCLF | 4 | 8.04 | 8.20 | 8.08 | 8.13 | 8.00 | 7.94 | 7.87 |
| 不同时延 + SyncNet 补偿 | Detector + SyncNet | 0 | - | - | - | - | - | - | - |
| 不同时延 + SyncNet 补偿 | Detector + SyncNet | 2 | - | - | - | - | - | - | - |
| 不同时延 + SyncNet 补偿 | Detector + SyncNet | 4 | - | - | - | - | - | - | - |

### 2.2 DATA 作为 baseline

数据集：`DAIR-V2X-C`。Baseline 方法：`DATA` intermediate fusion detector。官方 DATA 仓库 README/论文表格报告的 DAIR-V2X-C 结果为 `AP@0.5/AP@0.7 = 79.94/65.48`。但当前仓库可直接运行的 `runs/data_dair_official_eval` 与 `external/DATA/opencood/logs/official_checkpoints` 使用同一个 checkpoint，其本地无时延复现结果只有 `54.03/37.43`；该配置是 LiDAR-only DATA，`input_source=['lidar']`，`heter.mapping_dict` 将 `m2` 也映射到 `m1`，不是 README 中完整报告结果的可复现实验包。因此本节只能作为“当前本地 DATA checkpoint/config”的实验，不应声称复现论文主表的 `79.94/65.48`。

DATA 的实际融合级别是 IFAM 输入处的 BEV dense feature：PointPillar encoder/backbone 输出 `spatial_features_2d`，经过 aligner 后送入 `ifam.forward_collab` 做 intermediate feature fusion；它不是原始点云级融合，也不是检测框级 late fusion。无补偿行使用发送端 `t-(Comm+Comp)` 点云和接收端 `t-Comp` 点云，并与 `t` 时刻 cooperative GT 比对。`DATA + pose-warp` 是当前 DATA fork 中 `--compensation_method ours` 的输入级补偿：对 delayed LiDAR 按传感器位姿投影到目标时刻坐标系后再走 DATA intermediate fusion。它不是当前 DAIR official late-fusion 主结果中使用的 `History-aware RF 2f` detector-side box predictor。

数据来源：`results/260615_fill/artifacts/dair_data_strict_dual_delay_none_ours_20260706.csv`；DATA IFAM 特征级 T10-N10 补偿结果来自 `results/260615_fill/artifacts/dair_data_ifam_feature_t10n10_ours_20260707.csv`，模型为 `checkpoints/curated/ours_models/dair_data_ifam_feature_t10n10_ours.npz`。新增联合训练补偿结果来自 `results/260615_fill/artifacts/dair_data_ifam_joint_delay_comp_20260713.csv`，模型为 `checkpoints/curated/ours_models/dair_data_ifam_joint_delay_comp.pth`；该方法冻结原 DATA checkpoint，只训练 IFAM 输入特征上的 gated residual delay compensator，并保持 `Comp0/Comm0` identity。此前生成的 `results/260615_fill/artifacts/dair_data_strict_dual_delay_rf_ours_20260706.csv` 是诊断结果：它把 DATA 拆成单侧检测框，再用 RF 2f 做 box-level late fusion；该路径改变了 DATA 的 intermediate fusion 机制，所以不作为本节 DATA 主实验写入。其 `Comm0/Comp0` 的 `49.42` 低于 `54.03`，原因正是融合路径已变，不是数值波动。

#### 2.2.1 AP@0.5

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | DATA | 0 | - | 54.03 | - | - | - | - | - |
| 不同时延无补偿 | DATA | 0 | 48.20 | 54.03 | 51.51 | 47.85 | 46.05 | 45.12 | 44.64 |
| 不同时延无补偿 | DATA | 2 | 20.93 | 21.74 | 21.00 | 20.92 | 20.76 | 20.63 | 20.51 |
| 不同时延无补偿 | DATA | 4 | 15.09 | 14.65 | 14.86 | 15.22 | 15.22 | 15.25 | 15.34 |
| 不同时延 + DATA fork 输入级补偿 | DATA + pose-warp | 0 | 46.93 | 54.03 | 49.98 | 46.55 | 44.58 | 43.67 | 42.77 |
| 不同时延 + DATA fork 输入级补偿 | DATA + pose-warp | 2 | 19.67 | 20.76 | 20.04 | 19.77 | 19.54 | 19.11 | 18.81 |
| 不同时延 + DATA fork 输入级补偿 | DATA + pose-warp | 4 | 13.73 | 13.71 | 13.78 | 13.90 | 13.86 | 13.62 | 13.50 |
| 不同时延 + 我们方法特征级补偿 | DATA + Ours IFAM-feature T10-N10 | 0 | 48.20 | 54.03 | 48.88 | 47.32 | 46.67 | 46.32 | 45.96 |
| 不同时延 + 我们方法特征级补偿 | DATA + Ours IFAM-feature T10-N10 | 2 | 11.70 | 12.59 | 12.00 | 11.70 | 11.51 | 11.27 | 11.13 |
| 不同时延 + 我们方法特征级补偿 | DATA + Ours IFAM-feature T10-N10 | 4 | 13.75 | 14.49 | 14.21 | 13.88 | 13.55 | 13.32 | 13.05 |
| 不同时延 + 联合训练补偿 | DATA + Ours IFAM joint delay-comp | 0 | 47.83 | 54.03 | 49.51 | 47.31 | 45.94 | 45.39 | 44.82 |
| 不同时延 + 联合训练补偿 | DATA + Ours IFAM joint delay-comp | 2 | 23.79 | 26.40 | 23.99 | 23.35 | 23.15 | 23.04 | 22.83 |
| 不同时延 + 联合训练补偿 | DATA + Ours IFAM joint delay-comp | 4 | 18.75 | 19.05 | 18.77 | 18.84 | 18.87 | 18.45 | 18.51 |

#### 2.2.2 AP@0.7

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | DATA | 0 | - | 37.43 | - | - | - | - | - |
| 不同时延无补偿 | DATA | 0 | 32.13 | 37.43 | 32.82 | 31.30 | 30.77 | 30.42 | 30.05 |
| 不同时延无补偿 | DATA | 2 | 9.29 | 9.04 | 9.11 | 9.42 | 9.43 | 9.38 | 9.33 |
| 不同时延无补偿 | DATA | 4 | 7.71 | 7.44 | 7.64 | 7.75 | 7.83 | 7.86 | 7.72 |
| 不同时延 + DATA fork 输入级补偿 | DATA + pose-warp | 0 | 31.04 | 37.44 | 31.69 | 30.27 | 29.53 | 28.95 | 28.37 |
| 不同时延 + DATA fork 输入级补偿 | DATA + pose-warp | 2 | 8.46 | 8.43 | 8.51 | 8.57 | 8.49 | 8.43 | 8.29 |
| 不同时延 + DATA fork 输入级补偿 | DATA + pose-warp | 4 | 6.74 | 6.62 | 6.68 | 6.85 | 6.79 | 6.77 | 6.75 |
| 不同时延 + 我们方法特征级补偿 | DATA + Ours IFAM-feature T10-N10 | 0 | 32.77 | 37.47 | 32.48 | 32.05 | 31.83 | 31.55 | 31.25 |
| 不同时延 + 我们方法特征级补偿 | DATA + Ours IFAM-feature T10-N10 | 2 | 4.45 | 4.67 | 4.58 | 4.49 | 4.37 | 4.35 | 4.23 |
| 不同时延 + 我们方法特征级补偿 | DATA + Ours IFAM-feature T10-N10 | 4 | 6.76 | 7.14 | 6.93 | 6.83 | 6.70 | 6.60 | 6.34 |
| 不同时延 + 联合训练补偿 | DATA + Ours IFAM joint delay-comp | 0 | 31.96 | 37.47 | 32.29 | 31.08 | 30.58 | 30.29 | 30.06 |
| 不同时延 + 联合训练补偿 | DATA + Ours IFAM joint delay-comp | 2 | 10.95 | 11.18 | 10.95 | 10.89 | 10.94 | 10.97 | 10.81 |
| 不同时延 + 联合训练补偿 | DATA + Ours IFAM joint delay-comp | 4 | 9.47 | 9.47 | 9.48 | 9.55 | 9.48 | 9.49 | 9.37 |

分析：当前本地 DATA strict detector AP@0.7 的 18 格均值为无补偿 `16.37`、DATA fork 输入级补偿 `15.41`、DATA IFAM 特征级 T10-N10 补偿 `14.66`、DATA IFAM joint delay-comp `17.46`；AP@0.5 的 18 格均值为无补偿 `28.07`、DATA fork 输入级补偿 `26.78`、DATA IFAM 特征级 T10-N10 补偿 `24.55`、DATA IFAM joint delay-comp `30.12`。离线 T10-N10 的 `Comm0/Comp0` identity 正常，但存在计算时延时明显下降，尤其 `Comp2` 只有 `11.70/4.45`，低于无补偿 `20.93/9.29`。联合训练版修复了这一点：`Comp2` 提升到 `23.79/10.95`，`Comp4` 提升到 `18.75/9.47`，均高于无补偿；`Comp0/Comm0` 保持 `54.03/37.47`，identity 成立。因此，在 DATA 的 IFAM 特征级补偿上，离线线性 T10-N10 不适合，但冻结 DATA detector、只训练 IFAM 输入处 gated residual delay compensator 的联合训练方式可以形成正向提升。当前单侧 box late-fusion 外接版本仍只能作为诊断，不是正式 DATA + Ours。

#### 2.2.3 DATA box-level 公平诊断实验

为避免把 DATA intermediate fusion 与 box-level late fusion 混比，本诊断实验固定同一条 pipeline：先用 DATA checkpoint 分别生成车端/路端单侧检测框，再做 box-level late fusion；对照组直接融合 delayed box，Ours 组先用 `History-aware RF 2f` 将 delayed box 补偿到 `t` 后再融合。该实验不代表原版 DATA intermediate fusion，只用于回答“DATA 产生的检测框上，我们的 detector-side box 补偿是否有效”。

数据来源：旧 RF 诊断为 `results/260615_fill/artifacts/dair_data_box_latefusion_fair_none_vs_rf_20260706.csv`；严格公平 RF 追加结果为 `results/260615_fill/artifacts/dair_data_box_latefusion_fair_rf_trainonly_20260714.csv`，使用 checkpoint `checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainonly.pkl`。脚本：`scripts/run_dair_data_rf_ours_box_comp_grid.py --mode ours --eval-jobs 6`。

| 指标 | 方法 | 18 格 AVG | Comp0 AVG | Comp2 AVG | Comp4 AVG |
|---|---|---:|---:|---:|---:|
| AP@0.5 | DATA box late-fusion w/o compensation | 24.70 | 44.90 | 17.45 | 11.74 |
| AP@0.5 | DATA box late-fusion + Ours RF 2f | 31.77 | 44.39 | 29.34 | 21.59 |
| AP@0.5 | Ours - w/o compensation | +7.08 | -0.50 | +11.89 | +9.85 |
| AP@0.5 | DATA box late-fusion + Ours RF 2f strict train->val | 31.74 | 44.18 | 30.86 | 20.18 |
| AP@0.5 | strict Ours - w/o compensation | +7.05 | -0.72 | +13.41 | +8.44 |
| AP@0.5 | strict Ours - old Ours | -0.03 | -0.21 | +1.52 | -1.41 |
| AP@0.7 | DATA box late-fusion w/o compensation | 15.29 | 31.62 | 7.99 | 6.28 |
| AP@0.7 | DATA box late-fusion + Ours RF 2f | 15.67 | 30.31 | 9.85 | 6.85 |
| AP@0.7 | Ours - w/o compensation | +0.38 | -1.30 | +1.87 | +0.57 |
| AP@0.7 | DATA box late-fusion + Ours RF 2f strict train->val | 16.63 | 30.24 | 12.60 | 7.06 |
| AP@0.7 | strict Ours - w/o compensation | +1.34 | -1.38 | +4.61 | +0.78 |
| AP@0.7 | strict Ours - old Ours | +0.96 | -0.08 | +2.75 | +0.20 |

分析：这组诊断结果可以用来体现我们的 RF box compensation 在 DAIR-V2X DATA 检测框上的有效性，但表述必须限定为 `DATA box late-fusion`。旧 RF 与严格 RF 的 `Comm0/Comp0` 都保持 `49.42/37.41`，identity 成立。严格 RF 相比无补偿的 AP@0.5 18 格均值提升 `+7.05`，AP@0.7 提升 `+1.34`；相比旧 RF，AP@0.5 基本持平 `-0.03`，AP@0.7 反而提高 `+0.96`。差异主要来自 `Comp2`：严格 RF 的 AP@0.5/AP@0.7 比旧 RF 高 `+1.52/+2.75`；但 `Comp4` 的 AP@0.5 比旧 RF 低 `-1.41`，说明旧 RF 的泄漏收益主要体现在更长时延的宽松 IoU 召回上。该诊断仍不是 DATA intermediate fusion 主结果，只说明“DATA 单侧检测框 + box-level RF 补偿”这个后处理路径的效果。

#### 2.2.4 DATA intermediate + RF assist 后处理诊断

这组实验尝试使用 2.1 中 `Detector + Ours` 的 detector-side RF 两帧预测机制来辅助 DATA：先保留 DATA intermediate 的最终检测框，再把 `DATA 单侧检测框 + Ours RF 2f` 得到的补偿框作为辅助候选框，与 DATA 框做 `concat + rotated NMS` 后评估。该实验仍是后处理级诊断，不是 DATA 原始 IFAM feature fusion 的内部补偿。

数据来源：旧 RF 诊断为 `results/260615_fill/artifacts/dair_data_intermediate_rf_assist_keycells_20260707.csv`；严格公平 RF 追加结果为 `results/260615_fill/artifacts/dair_data_intermediate_rf_assist_trainonly_keycells_20260714.csv`。脚本：`scripts/run_dair_data_hybrid_rf_assist.py`。当前先跑关键格点 `Comp0/Comm0` 和 `Comp2/Comm0`，用于判断该思路是否值得扩展全 18 格。

| 设置 | RF 分数权重 | Comp | Comm | AP@0.5 | AP@0.7 | 相比 DATA AP@0.5 | 相比 DATA AP@0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DATA intermediate baseline | - | 0 | 0 | 54.03 | 37.43 | - | - |
| DATA + RF assist concat/NMS | 0.3 | 0 | 0 | 54.71 | 37.75 | +0.68 | +0.32 |
| DATA + RF assist concat/NMS strict train->val RF | 0.3 | 0 | 0 | 54.71 | 37.75 | +0.68 | +0.32 |
| DATA + RF assist concat/NMS | 1.0 | 0 | 0 | 49.44 | 34.21 | -4.59 | -3.22 |
| DATA + RF assist concat/NMS strict train->val RF | 1.0 | 0 | 0 | 49.44 | 34.21 | -4.59 | -3.22 |
| DATA intermediate baseline | - | 2 | 0 | 21.74 | 9.04 | - | - |
| DATA + RF assist concat/NMS | 0.3 | 2 | 0 | 21.00 | 8.44 | -0.74 | -0.60 |
| DATA + RF assist concat/NMS strict train->val RF | 0.3 | 2 | 0 | 21.01 | 8.46 | -0.73 | -0.58 |
| DATA + RF assist concat/NMS | 1.0 | 2 | 0 | 28.44 | 9.59 | +6.70 | +0.55 |
| DATA + RF assist concat/NMS strict train->val RF | 1.0 | 2 | 0 | 30.39 | 12.18 | +8.65 | +3.14 |

分析：这个办法能在计算时延场景下补回一部分 DATA 的召回，且严格 RF 比旧 RF 更明显：`Comp2/Comm0, scale=1.0` 下，旧 RF 将 DATA 从 `21.74/9.04` 提升到 `28.44/9.59`，严格 RF 进一步提升到 `30.39/12.18`。但它仍不是稳定的全局策略：同样的 RF 原始分数权重在无时延 `Comp0/Comm0` 会把 DATA 从 `54.03/37.43` 拉低到 `49.44/34.21`；降低 RF 分数权重到 `0.3` 虽然不伤无时延，`Comp2/Comm0` 又没有提升。因此该方法更适合作为“检测到存在计算时延时启用的补召回模块”，不适合作为无条件接入 DATA 的主补偿方案。若要扩展全 18 格，应先设计时延感知 gating 或 score calibration，再跑完整表。

### 2.3 LRCP 作为 baseline

数据集：`DAIR-V2X-C`。Baseline 方法：`LRCP`。旧 `results/260615_fill/artifacts/dair_lrcp_latency_ap.csv` 是 single-delay 复现实验，只设置 `k=Comm+Comp`，没有让接收端使用 `t-Comp`，因此不写入本版 strict 表格。本节使用 `scripts/run_dair_lrcp_latency_grid.py --strict-dual-delay` 生成的 strict dual-delay 复现结果。

LRCP strict adapter 让路侧发送端使用 `t-(Comm+Comp)` 历史帧，让车端接收端使用 `t-Comp` 车辆点云，并用车辆自车位姿把接收端前帧点云对齐到当前 `t` ego 坐标后送入现有 `max_cav=2` checkpoint。`LRCP + Ours` 是 LRCP 自身 stage2 learned compensation，作用于非 ego 路侧 agent；它不是 2.1 中的 RF detector-side box compensator。当前仓库没有真实 DAIR 版 LRCP + SyncNet checkpoint，因此 SyncNet 行保留为空。

LRCP 表格中的 AP 也是百分数形式。`1.49` 表示 `1.49%`，不是米、帧或 0 到 1 的小数；对应原始 AP 约为 `0.0149`。因此 AP@0.5 的 `1.x` 和 AP@0.7 的 `0.24` 都说明当前 LRCP DAIR 复现 pipeline 的绝对检测 AP 很低。

数据来源：`results/260615_fill/artifacts/dair_lrcp_strict_dual_delay_ap.csv`；分片结果位于 `results/260615_fill/artifacts/dair_lrcp_strict_dual_delay_parts/`。

#### 2.3.1 BEV AP@0.5

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | LRCP | 0 | - | 1.46 | - | - | - | - | - |
| 不同时延无补偿 | LRCP w/o compensation | 0 | 1.47 | 1.45 | 1.47 | 1.48 | 1.46 | 1.49 | 1.48 |
| 不同时延无补偿 | LRCP w/o compensation | 2 | 1.48 | 1.49 | 1.47 | 1.48 | 1.46 | 1.48 | 1.49 |
| 不同时延无补偿 | LRCP w/o compensation | 4 | 1.49 | 1.50 | 1.49 | 1.49 | 1.49 | 1.50 | 1.49 |
| 不同时延 + 我们方法补偿 | LRCP + Ours | 0 | 1.48 | 1.46 | 1.47 | 1.48 | 1.49 | 1.51 | 1.49 |
| 不同时延 + 我们方法补偿 | LRCP + Ours | 2 | 1.49 | 1.50 | 1.48 | 1.50 | 1.47 | 1.51 | 1.51 |
| 不同时延 + 我们方法补偿 | LRCP + Ours | 4 | 1.50 | 1.49 | 1.46 | 1.51 | 1.51 | 1.51 | 1.50 |
| 不同时延 + SyncNet 补偿 | LRCP + SyncNet | 0 | - | - | - | - | - | - | - |
| 不同时延 + SyncNet 补偿 | LRCP + SyncNet | 2 | - | - | - | - | - | - | - |
| 不同时延 + SyncNet 补偿 | LRCP + SyncNet | 4 | - | - | - | - | - | - | - |

#### 2.3.2 BEV AP@0.7

| 类别 | 方法/补偿 | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无延时 | LRCP | 0 | - | 0.22 | - | - | - | - | - |
| 不同时延无补偿 | LRCP w/o compensation | 0 | 0.24 | 0.22 | 0.22 | 0.24 | 0.24 | 0.25 | 0.25 |
| 不同时延无补偿 | LRCP w/o compensation | 2 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 | 0.24 | 0.25 |
| 不同时延无补偿 | LRCP w/o compensation | 4 | 0.24 | 0.23 | 0.23 | 0.23 | 0.25 | 0.26 | 0.25 |
| 不同时延 + 我们方法补偿 | LRCP + Ours | 0 | 0.22 | 0.22 | 0.22 | 0.22 | 0.21 | 0.22 | 0.21 |
| 不同时延 + 我们方法补偿 | LRCP + Ours | 2 | 0.22 | 0.23 | 0.21 | 0.22 | 0.22 | 0.24 | 0.23 |
| 不同时延 + 我们方法补偿 | LRCP + Ours | 4 | 0.23 | 0.22 | 0.22 | 0.24 | 0.23 | 0.24 | 0.25 |
| 不同时延 + SyncNet 补偿 | LRCP + SyncNet | 0 | - | - | - | - | - | - | - |
| 不同时延 + SyncNet 补偿 | LRCP + SyncNet | 2 | - | - | - | - | - | - | - |
| 不同时延 + SyncNet 补偿 | LRCP + SyncNet | 4 | - | - | - | - | - | - | - |

分析：LRCP strict dual-delay 复现的 BEV AP@0.7 18 格均值为无补偿 `0.24`、LRCP + Ours `0.22`。AP@0.5 均值为无补偿 `1.48`、LRCP + Ours `1.49`。这些数值是百分数 AP；绝对 AP 明显低于 official late-fusion detector，说明当前 LRCP DAIR 适配/复现 pipeline 本身质量很低，只适合同一 LRCP pipeline 内部对照，不应与 2.1 的 official detector AP 混写成主结论。

### 2.4 TraF-Align 作为 baseline

数据集：`DAIR-V2X-Seq/SPD`。目标方法：`TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception`。官方开源代码提供 `DAIR-V2X-Seq` 配置与 checkpoint，本地使用 checkpoint `checkpoints/dair_v2x_seq_Trafalign_2024_12_28_04_42_36/net_epoch60.pth`。注意：本节使用 DAIR-V2X-Seq 的真实时序数据，`1 frame = 100ms`，与本文档前部 DAIR-V2X-C 表格默认 `1 frame = 0.2s` 不同。

环境核查：当前默认 `spconv 2.3.6 / cumm 0.4.11` 会导致该 checkpoint 复现失败，AP 近似为 0；本节结果使用隔离环境 `PYTHONPATH=/tmp/trafalign_spconv_2121:$PYTHONPATH`，其中 `spconv-cu113==2.1.21`、`cumm-cu113==0.2.9`。在该环境下，本地官方复现为 `0ms AP@0.5/AP@0.7 = 77.02/59.04`、`400ms = 71.32/53.08`，与官方随 checkpoint 保存的 `76.94/58.95`、`71.26/53.04` 基本一致。

实验口径按本文档开头的 strict dual-delay 设置执行：`Comp c` 表示接收端 ego 使用 `t-c`，发送端 agent 使用 `t-(c+Comm)`，最终检测结果均与 `t` 时刻 GT 对比。表中 `Comp0/2/4` 分别对应 `delay_ego_ms=0/200/400`；`Comm0..5` 对应额外通信时延 `0..500ms`。

`TraF-Align + Ours feature T10-N10` 是在 TraF-Align backbone feature 空间重新训练的 T10-N10 residual predictor，训练只使用 official train split 的缓存特征；插入位置为 TraF-Align backbone 输出之后、后续 trajectory alignment 模块之前。`horizon <= 0` 时保持原 feature 不变，因此 `Comp0/Comm0` 的 identity 结果应与 baseline 一致。

新增联合训练修复版 `TraF-Align + Ours joint delay-comp only`：在 TraF-Align 的 temporal feature regroup 之后、communication/trajectory alignment 之前加入 gated residual delay-compensation module，并从官方 checkpoint 初始化。训练时只更新该新增补偿模块，冻结原 TraF-Align reader/backbone/fusion/head 权重并冻结 BN 统计；训练增强设置为 ego 计算时延 `0..400ms`、agent 总时延 `0..900ms`。这样 `delay=0` 时补偿模块被强制 bypass，保证无时延 identity。

进一步改进版 `TraF-Align + Ours joint delay-comp adaptive scale` 使用同一版 joint-only 训练权重，但在推理/部署时对 residual gate 做 delay-adaptive scale：`delay<=2 frame` 保留完整补偿强度，`delay>=4 frame` 降到 `0.1x`，中间线性过渡。原因是原 joint-only 在 `Comp2` 上收益明显，但在 `Comp4` 高 IoU 定位上存在过补偿；adaptive scale 用较强 residual 处理 2 帧中等计算时延，用保守 residual 处理 4 帧及以上大计算时延。

进一步修正的推荐版 `TraF-Align + Ours joint delay-comp adaptive scale + Comp0 bypass`：当接收端计算时延 `Comp=0` 时，直接 bypass 我们的 residual compensator，保留 TraF-Align 原始通信/轨迹对齐结果；当 `Comp>0` 时使用 adaptive scale 补偿。原因是 `Comp0` 表示接收端没有计算滞后，只有发送端通信延迟，而 TraF-Align 原本就是为异步通信/轨迹对齐设计的；在这个场景额外 residual 容易轻微过补偿。

#### 2.4.1 AP@0.5

| 设置 | 方法 | Comp | 平均 | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无时延补偿 | TraF-Align | 0 | 73.04 | 77.02 | 74.03 | 73.02 | 72.25 | 71.32 | 70.57 |
| 无时延补偿 | TraF-Align | 2 | 29.12 | 31.89 | 30.55 | 29.80 | 28.48 | 27.44 | 26.57 |
| 无时延补偿 | TraF-Align | 4 | 23.87 | 25.61 | 24.86 | 24.15 | 23.33 | 22.74 | 22.50 |
| 不同时延 + 我们方法补偿 | TraF-Align + Ours feature T10-N10 | 0 | 64.00 | 77.02 | 63.02 | 61.70 | 61.32 | 60.76 | 60.18 |
| 不同时延 + 我们方法补偿 | TraF-Align + Ours feature T10-N10 | 2 | 3.34 | 3.61 | 3.51 | 3.31 | 3.23 | 3.19 | 3.17 |
| 不同时延 + 我们方法补偿 | TraF-Align + Ours feature T10-N10 | 4 | 2.49 | 2.60 | 2.52 | 2.51 | 2.47 | 2.40 | 2.42 |
| 不同时延 + 联合训练补偿 | TraF-Align + Ours joint delay-comp only | 0 | 72.62 | 77.01 | 73.61 | 72.30 | 71.82 | 70.91 | 70.04 |
| 不同时延 + 联合训练补偿 | TraF-Align + Ours joint delay-comp only | 2 | 45.48 | 48.62 | 47.35 | 46.09 | 45.05 | 43.57 | 42.17 |
| 不同时延 + 联合训练补偿 | TraF-Align + Ours joint delay-comp only | 4 | 23.99 | 26.39 | 24.88 | 23.95 | 23.21 | 22.95 | 22.56 |
| 不同时延 + 联合训练补偿改进 | TraF-Align + Ours joint delay-comp adaptive scale | 0 | 72.80 | 77.02 | 73.62 | 72.27 | 72.07 | 71.25 | 70.56 |
| 不同时延 + 联合训练补偿改进 | TraF-Align + Ours joint delay-comp adaptive scale | 2 | 45.75 | 48.62 | 47.47 | 46.30 | 45.19 | 43.81 | 43.09 |
| 不同时延 + 联合训练补偿改进 | TraF-Align + Ours joint delay-comp adaptive scale | 4 | 23.89 | 25.65 | 24.94 | 24.11 | 23.34 | 22.78 | 22.52 |
| 不同时延 + 联合训练补偿推荐 | TraF-Align + Ours adaptive scale + Comp0 bypass | 0 | 73.04 | 77.02 | 74.03 | 73.02 | 72.25 | 71.32 | 70.57 |
| 不同时延 + 联合训练补偿推荐 | TraF-Align + Ours adaptive scale + Comp0 bypass | 2 | 45.75 | 48.62 | 47.47 | 46.30 | 45.19 | 43.81 | 43.09 |
| 不同时延 + 联合训练补偿推荐 | TraF-Align + Ours adaptive scale + Comp0 bypass | 4 | 23.89 | 25.65 | 24.94 | 24.11 | 23.34 | 22.78 | 22.52 |

#### 2.4.2 AP@0.7

| 设置 | 方法 | Comp | 平均 | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无时延补偿 | TraF-Align | 0 | 54.44 | 59.04 | 54.82 | 53.97 | 53.17 | 53.08 | 52.54 |
| 无时延补偿 | TraF-Align | 2 | 11.98 | 13.20 | 12.61 | 12.27 | 11.60 | 11.31 | 10.90 |
| 无时延补偿 | TraF-Align | 4 | 10.82 | 11.72 | 11.36 | 11.04 | 10.59 | 10.24 | 9.95 |
| 不同时延 + 我们方法补偿 | TraF-Align + Ours feature T10-N10 | 0 | 49.03 | 59.04 | 46.50 | 46.89 | 47.36 | 47.34 | 47.04 |
| 不同时延 + 我们方法补偿 | TraF-Align + Ours feature T10-N10 | 2 | 0.69 | 0.71 | 0.75 | 0.70 | 0.66 | 0.67 | 0.67 |
| 不同时延 + 我们方法补偿 | TraF-Align + Ours feature T10-N10 | 4 | 0.65 | 0.68 | 0.65 | 0.65 | 0.64 | 0.63 | 0.62 |
| 不同时延 + 联合训练补偿 | TraF-Align + Ours joint delay-comp only | 0 | 53.92 | 59.04 | 53.88 | 53.12 | 52.70 | 52.72 | 52.08 |
| 不同时延 + 联合训练补偿 | TraF-Align + Ours joint delay-comp only | 2 | 14.87 | 16.38 | 15.67 | 15.18 | 14.49 | 13.98 | 13.49 |
| 不同时延 + 联合训练补偿 | TraF-Align + Ours joint delay-comp only | 4 | 8.43 | 9.21 | 8.81 | 8.40 | 8.20 | 8.03 | 7.90 |
| 不同时延 + 联合训练补偿改进 | TraF-Align + Ours joint delay-comp adaptive scale | 0 | 54.09 | 59.03 | 53.91 | 53.14 | 52.84 | 53.03 | 52.58 |
| 不同时延 + 联合训练补偿改进 | TraF-Align + Ours joint delay-comp adaptive scale | 2 | 14.80 | 16.38 | 15.63 | 15.01 | 14.37 | 13.90 | 13.52 |
| 不同时延 + 联合训练补偿改进 | TraF-Align + Ours joint delay-comp adaptive scale | 4 | 10.78 | 11.62 | 11.29 | 10.96 | 10.55 | 10.24 | 10.00 |
| 不同时延 + 联合训练补偿推荐 | TraF-Align + Ours adaptive scale + Comp0 bypass | 0 | 54.44 | 59.04 | 54.82 | 53.97 | 53.17 | 53.08 | 52.54 |
| 不同时延 + 联合训练补偿推荐 | TraF-Align + Ours adaptive scale + Comp0 bypass | 2 | 14.80 | 16.38 | 15.63 | 15.01 | 14.37 | 13.90 | 13.52 |
| 不同时延 + 联合训练补偿推荐 | TraF-Align + Ours adaptive scale + Comp0 bypass | 4 | 10.78 | 11.62 | 11.29 | 10.96 | 10.55 | 10.24 | 10.00 |

分析：TraF-Align baseline 的 18 格均值为 AP@0.5 `42.01`、AP@0.7 `25.75`；离线 `TraF-Align + Ours feature T10-N10` 为 AP@0.5 `23.27`、AP@0.7 `16.79`，明显失败。联合训练修复版 `TraF-Align + Ours joint delay-comp only` 为 AP@0.5 `47.36`、AP@0.7 `25.74`：AP@0.5 相比 baseline 提升 `+5.35`，AP@0.7 基本持平。进一步加入 delay-adaptive scale 后，18 格均值提升到 AP@0.5 `47.48`、AP@0.7 `26.56`，相对 baseline 分别提升 `+5.47/+0.81`。再加入 `Comp0 bypass` 后，18 格均值提升到 AP@0.5 `47.56`、AP@0.7 `26.67`，且 `Comp0` 平均值恢复为 baseline 的 `73.04/54.44`。该提升主要来自计算时延 `Comp2`，其 AP@0.5 从 `29.12` 提升到 `45.75`，AP@0.7 从 `11.98` 提升到 `14.80`；同时 adaptive scale 修复了原 joint-only 在 `Comp4` 的高 IoU 下降，`Comp4` AP@0.7 从 joint-only `8.43` 回升到 `10.78`，基本接近 baseline `10.82`。无时延 `Comp0/Comm0` 保持 `77.02/59.04`，identity 成立。

结论：简单离线 feature predictor 不能用，因为它破坏了 TraF-Align 后续 `field_predictor / offset_generator / trajectory attention` 所依赖的特征分布；但把补偿模块改成可微的 gated residual module，并只训练该新增模块，可以在不伤害无时延结果的情况下显著改善中等计算时延场景。更好的联合方式是在 joint-only 训练后加入 delay-adaptive residual scale，并在 `Comp0` 时 bypass 我们的 residual：2 帧计算时延保留强补偿，4 帧及以上使用保守补偿，无计算时延交给 TraF-Align 原始轨迹对齐。另一个尝试版曾同时训练补偿模块和 TraF-Align fusion/head，`Comp0/Comm0` 降到 `74.97/55.86`，因此不作为正式结果。

## 3. 当前可写入论文的结论边界

1. DAIR-V2X official late-fusion detector 的无补偿、Ours RF 和 TCLF receiver-target 对比方法已按本轮三类设置重新校验，并已修复最紧急的数据泄漏问题。当前 2.1 主表使用 `checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainonly.pkl`，即 `train -> val` 严格公平 RF；DAIR strict detector BEV AP@0.7 的 18 格均值为 `29.95`，高于无补偿 `22.71` 和 TCLF 对比方法 `22.94`。额外诊断版 `train+val -> val` 的均值为 `35.15`，只能作为上限参考，不作为公平主结果。
2. DAIR-V2X DATA strict detector 结果已重新核查：官方 DATA README/论文表格报告为 `79.94/65.48`，但当前本地可运行 checkpoint/config 的无时延复现只有 `54.03/37.43`，不能当作论文主表复现。当前本地 DATA AP@0.7 的 18 格均值为无补偿 `16.37`、DATA fork 输入级补偿 `15.41`、离线 IFAM T10-N10 `14.66`、IFAM joint delay-comp `17.46`；AP@0.5 的 18 格均值为无补偿 `28.07`、DATA fork 输入级补偿 `26.78`、离线 IFAM T10-N10 `24.55`、IFAM joint delay-comp `30.12`。因此 DATA 主实验中可写入的正向结果是 `DATA intermediate + Ours IFAM joint delay-comp`，它冻结原 DATA detector、只训练 IFAM 输入处 gated residual compensator，并在 `Comp0/Comm0` 保持 identity。补充的 DATA box-level 公平诊断实验仍只能表述为 `DATA box late-fusion + Ours`，不是原版 DATA intermediate fusion。
3. DAIR-V2X SyncNet 当前没有真实 DAIR checkpoint/模型接入；本地 `syncnet` 或 `--allow-pseudo-syncnet` 分支会复用 TCLF temporal offset 或 LRCP stage1 GT-flow reference，因此不写入最终表格。
4. 当前最优 RF 模型是 official late-fusion detector-side box compensator；它未接入 LRCP point-cloud pipeline。LRCP + Ours 行使用的是 LRCP 自身 stage2 learned compensation，只能作为 LRCP pipeline 内部对照。
5. LRCP strict adapter 可按 `receiver=t-Comp, sender=t-(Comm+Comp), GT=t` 跑出 `LRCP w/o compensation` 与 `LRCP + Ours` 数值；边界是当前 LRCP 复现 pipeline 绝对 AP 明显低于 official late-fusion detector，不与 2.1 的 official detector AP 混写成主结论。
6. TraF-Align 已在真实 `DAIR-V2X-Seq/SPD` 上完成正式评测。使用 `spconv-cu113==2.1.21 / cumm-cu113==0.2.9` 后，官方 checkpoint 的本地复现与官方保存结果基本一致；TraF-Align baseline 的 18 格均值为 AP@0.5 `42.01`、AP@0.7 `25.75`。离线 `TraF-Align + Ours feature T10-N10` 整体下降到 AP@0.5 `23.27`、AP@0.7 `16.79`，不能作为正向结果。修复后的 joint delay-comp only 版只训练新增 gated residual compensator，保持无时延 identity，并将 18 格 AP@0.5 提升到 `47.36`；AP@0.7 为 `25.74`，整体基本持平，但 `Comp4` AP@0.7 从 baseline `10.82` 降到 `8.43`。本轮新增的 adaptive scale 版在 joint-only 权重上按 delay 调整 residual 强度，将 18 格 AP@0.5/AP@0.7 提升到 `47.48/26.56`；进一步加入 `Comp0 bypass` 后提升到 `47.56/26.67`，其中 `Comp0` 恢复为 baseline `73.04/54.44`，`Comp2` AP@0.7 为 `14.80`，`Comp4` AP@0.7 回升到 `10.78`，是当前 TraF-Align + Ours 推荐版本。
7. V2X-Sim DATA 的无补偿和 Ours 已按新口径检查并可写入：AP@0.7 的 18 格均值从 `66.01` 提升到 Ours pose-warp `66.48`。新增联合训练 IFAM residual 后，保守 scale 版本进一步提升到 AP@0.5 `74.96`、AP@0.7 `66.81`；未缩放版本 AP@0.5 更高为 `75.33`，但 AP@0.7 降到 `64.66`，因此不作为主表推荐版本。补充的 V2X-Sim DATA 公平性核查显示，该表不存在 DAIR-V2X DATA 的 intermediate/box-level 混比问题；`DATA + Ours` 和 joint IFAM residual 都是同一 DATA intermediate pipeline 下的补偿。V2X-Sim V2VNet/DiscoNet 按要求直接保留 DLPCM 论文已有结果。
8. 当前 260704 表格中实际使用的 Ours/补偿模型不再只有两版：除 `v2xsim_ours_prediction_model.pth` 与严格公平的 `dair_v2x_late_fusion_rf2_ours_trainonly.pkl` 外，还包括 V2X-Sim DATA joint IFAM residual、DAIR DATA IFAM T10-N10、DAIR DATA IFAM joint delay-comp、TraF-Align feature T10-N10、TraF-Align joint delay-comp 等模型。旧 `dair_v2x_late_fusion_rf2_ours.pkl` 仅保留为泄漏诊断参考。最终使用与未使用 checkpoint 的整理见第 5 节；删除任何未使用模型前需再次人工确认。

## 4. 结果文件索引

| 内容 | 路径 |
|---|---|
| V2X-Sim / DATA 无补偿 strict 结果 | `runs/data_v2xsim_pipeline/results/data_v2xsim_latency_ap.csv` |
| V2X-Sim / DATA + Ours strict 结果 | `results/260615_fill/artifacts/v2xsim_data_compensation_ap.csv` |
| V2X-Sim / DATA + Ours joint IFAM residual 保守 scale 结果 | `results/260615_fill/artifacts/v2xsim_data_ifam_joint_delay_comp_scale01_20260713.csv`; `checkpoints/curated/ours_models/v2xsim_data_ifam_joint_delay_comp_scale01.pth` |
| V2X-Sim / DATA + Ours joint IFAM residual 未缩放诊断结果 | `results/260615_fill/artifacts/v2xsim_data_ifam_joint_delay_comp_20260713.csv`; `checkpoints/curated/ours_models/v2xsim_data_ifam_joint_delay_comp.pth`; `checkpoints/curated/ours_models/v2xsim_data_ifam_joint_delay_comp.train_metrics.csv`; `scripts/run_v2xsim_data_joint_delay_comp.py` |
| V2X-Sim / DATA + SyncNet strict 结果 | `results/260615_fill/artifacts/v2xsim_data_syncnet_strict_20260704.csv` |
| V2X-Sim / DATA intermediate 公平性核查 | `results/260615_fill/artifacts/v2xsim_data_intermediate_fairness_audit_20260707.csv` |
| V2X-Sim / V2X-Sim 上的 ours 预测模型 | `checkpoints/curated/ours_models/v2xsim_ours_prediction_model.pth` |
| DAIR-V2X / official late-fusion no-comp strict recheck | `results/260615_fill/artifacts/dair_late_fusion_no_comp_strict_20260704_rechecked.csv` |
| DAIR-V2X / official late-fusion Ours RF strict fair | `results/260615_fill/artifacts/dair_late_fusion_ours_history2_rf_trainonly_strict_20260714.csv`; `results/260615_fill/artifacts/dair_detector_ours_history2_rf_trainonly_strict_20260714.csv`; `checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainonly.pkl` |
| DAIR-V2X / official late-fusion Ours RF train+val 诊断上限 | `results/260615_fill/artifacts/dair_late_fusion_ours_history2_rf_trainval_diag_strict_20260714.csv`; `results/260615_fill/artifacts/dair_detector_ours_history2_rf_trainval_diag_strict_20260714.csv`; `checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainval_diagnostic.pkl` |
| DAIR-V2X / official late-fusion Ours RF 旧 val->val 泄漏诊断 | `results/260615_fill/artifacts/dair_late_fusion_ours_history2_rf_valcache_strict_20260705.csv`; `results/260615_fill/artifacts/dair_detector_ours_history2_rf_valcache_strict_20260705.csv`; `checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours.pkl` |
| DAIR-V2X / official late-fusion TCLF receiver-target strict recheck | `results/260615_fill/artifacts/dair_late_fusion_tclf_receiver_strict_20260704_rechecked.csv`; `results/260615_fill/artifacts/dair_detector_tclf_receiver_strict_20260704_rechecked.csv` |
| DAIR-V2X / DATA no-comp 与 DATA fork 输入级补偿 strict 结果 | `results/260615_fill/artifacts/dair_data_strict_dual_delay_none_ours_20260706.csv` |
| DAIR-V2X / DATA IFAM 特征级 T10-N10 补偿结果 | `results/260615_fill/artifacts/dair_data_ifam_feature_t10n10_ours_20260707.csv`; `checkpoints/curated/ours_models/dair_data_ifam_feature_t10n10_ours.npz`; `results/260615_fill/cache/dair_data_feature_t10n10/` |
| DAIR-V2X / DATA IFAM joint delay-comp 联合训练结果 | `results/260615_fill/artifacts/dair_data_ifam_joint_delay_comp_20260713.csv`; `checkpoints/curated/ours_models/dair_data_ifam_joint_delay_comp.pth`; `checkpoints/curated/ours_models/dair_data_ifam_joint_delay_comp.train_metrics.csv`; `scripts/run_dair_data_joint_delay_comp.py` |
| DAIR-V2X / DATA intermediate + RF assist 后处理诊断 | 旧 RF：`results/260615_fill/artifacts/dair_data_intermediate_rf_assist_keycells_20260707.csv`; strict RF：`results/260615_fill/artifacts/dair_data_intermediate_rf_assist_trainonly_keycells_20260714.csv`; `results/260615_fill/cache/dair_data_intermediate_boxes_20260707/`; `scripts/run_dair_data_hybrid_rf_assist.py` |
| DAIR-V2X / DATA box-level 公平诊断结果 | 旧 RF：`results/260615_fill/artifacts/dair_data_box_latefusion_fair_none_vs_rf_20260706.csv`; strict RF：`results/260615_fill/artifacts/dair_data_box_latefusion_fair_rf_trainonly_20260714.csv` |
| DAIR-V2X / DATA + Ours RF 2f box-level 诊断结果，不作为 DATA 主实验 | `results/260615_fill/artifacts/dair_data_strict_dual_delay_rf_ours_20260706.csv`; `results/260615_fill/cache/dair_data_rf_ours_20260706/` |
| DAIR-V2X / strict 运行日志 | `results/dair_v2x/official_baselines/logs_strict_dual_delay_20260703/`; `results/dair_v2x/official_baselines/logs_strict_dual_delay_ours_history2_rf_trainonly_20260714/`; `results/dair_v2x/official_baselines/logs_strict_dual_delay_ours_history2_rf_trainval_diag_20260714/`; `results/dair_v2x/official_baselines/logs_strict_dual_delay_tclf_receiver_20260704/` |
| DAIR-V2X / train split detector raw cache 生成日志 | `results/dair_v2x/official_baselines/logs_train_cache_20260714/late_fusion_no_comp_train_cache_full_gpu0_resume.log` |
| DAIR-V2X / DAIR-V2X 数据集上晚期融合 RF 两帧 ours 严格公平模型 | `checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainonly.pkl` |
| DAIR-V2X / LRCP strict dual-delay 结果 | `results/260615_fill/artifacts/dair_lrcp_strict_dual_delay_ap.csv` |
| DAIR-V2X / LRCP strict dual-delay 分片 | `results/260615_fill/artifacts/dair_lrcp_strict_dual_delay_parts/` |
| 旧 DAIR-V2X / LRCP single-delay 参考，不写入本版表格 | `results/260615_fill/artifacts/dair_lrcp_latency_ap.csv` |
| DAIR-V2X / TraF-Align DAIR-V2X-C 兼容 smoke 结果，不作为正式 baseline | `results/trafalign_dair_v2x_c_compat/`; `scripts/prepare_trafalign_dair_compat.py`; `scripts/run_trafalign_dair_compat_eval.py`; `results/260615_fill/artifacts/dair_trafalign_compat_debug_20260708.csv`; `results/260615_fill/artifacts/dair_trafalign_compat_debug_score0_20260708.csv` |
| DAIR-V2X-Seq / TraF-Align 官方 checkpoint 复现核查 | `results/dair_v2x_seq/trafalign_official_check_0_400ms_spconv2121.csv` |
| DAIR-V2X-Seq / TraF-Align strict dual-delay baseline 结果 | `results/dair_v2x_seq/trafalign_baseline_comp0_spconv2121.csv`; `results/dair_v2x_seq/trafalign_baseline_comp2_spconv2121.csv`; `results/dair_v2x_seq/trafalign_baseline_comp4_spconv2121.csv` |
| DAIR-V2X-Seq / TraF-Align + Ours feature T10-N10 strict dual-delay 结果 | `results/dair_v2x_seq/trafalign_ours_t10n10_comp0_spconv2121.csv`; `results/dair_v2x_seq/trafalign_ours_t10n10_comp2_spconv2121.csv`; `results/dair_v2x_seq/trafalign_ours_t10n10_comp4_spconv2121.csv` |
| DAIR-V2X-Seq / TraF-Align feature-space T10-N10 predictor 与缓存 | `checkpoints/curated/ours_models/dair_v2x_seq_trafalign_feature_t10n10_ours_spconv2121.npz`; `results/dair_v2x_seq/trafalign_feature_cache_spconv2121/`; `scripts/train_trafalign_feature_t10n10_predictor.py` |
| DAIR-V2X-Seq / TraF-Align + Ours joint delay-comp only strict dual-delay 结果 | `results/dair_v2x_seq/trafalign_joint_delay_comp_only_comp0_spconv2121.csv`; `results/dair_v2x_seq/trafalign_joint_delay_comp_only_comp2_spconv2121.csv`; `results/dair_v2x_seq/trafalign_joint_delay_comp_only_comp4_spconv2121.csv` |
| DAIR-V2X-Seq / TraF-Align + Ours joint delay-comp adaptive scale strict dual-delay 结果 | `results/dair_v2x_seq/trafalign_joint_delay_comp_only_adaptive2to4_spconv2121.csv`; `results/dair_v2x_seq/trafalign_joint_delay_comp_only_adaptive2to4_comp0_spconv2121.csv`; `results/dair_v2x_seq/trafalign_joint_delay_comp_only_adaptive2to4_comp2_spconv2121.csv`; `results/dair_v2x_seq/trafalign_joint_delay_comp_only_adaptive2to4_comp4_spconv2121.csv` |
| DAIR-V2X-Seq / TraF-Align + Ours adaptive scale + Comp0 bypass 推荐结果 | `results/dair_v2x_seq/trafalign_joint_delay_comp_only_adaptive2to4_bypass_comp0_spconv2121.csv`; `checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_adaptive2to4_bypass_comp0_spconv2121/config.yaml`; `checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_adaptive2to4_bypass_comp0_spconv2121/net_epoch1.pth` |
| DAIR-V2X-Seq / TraF-Align + Ours joint delay-comp only 训练脚本与 checkpoint | `scripts/train_trafalign_joint_delay_comp.py`; `checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_spconv2121/net_epoch1.pth`; `checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_spconv2121/config.yaml`; `checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_spconv2121/train_metrics.csv` |
| DAIR-V2X-Seq / TraF-Align + Ours joint delay-comp adaptive scale checkpoint | `checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_adaptive2to4_spconv2121/net_epoch1.pth`; `checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_adaptive2to4_spconv2121/config.yaml` |
| DAIR-V2X-Seq / TraF-Align joint delay-comp 源码改动 | `/tmp/TraF-Align_partial/models/modules/deform/joint_delay_compensation.py`; `/tmp/TraF-Align_partial/models/modules/deform/traf_align_fusion.py`; `/tmp/TraF-Align_partial/datasets/Basedataset/v2xseq_basedataset.py` |
| DAIR-V2X-Seq / TraF-Align 正式评测脚本与下载 probe 记录 | `scripts/run_trafalign_dair_v2x_seq_eval.py`; `scripts/download_dair_v2x_seq_spd.py`; `results/dair_v2x_seq/download_status_20260708.json`; `results/dair_v2x_seq/download_status_retry_20260708.json` |
| DLPCM 论文已有 V2X-Sim 结果来源 | `DLPCM/table/main_result.tex`; `DLPCM/table/main_result_disco.tex` |

## 5. 真实性与 Checkpoint 审计（2026-07-14）

本节只做核查和标注，不删除任何模型。删除前需要用户再次确认。

### 5.1 结果真实性与公平性核查

| 范围 | 核查结论 | 训练/测试 split | 完整性 | 风险 |
|---|---|---|---|---|
| 1.1 V2X-Sim / V2VNet 与 1.3 V2X-Sim / DiscoNet | 数值来自 DLPCM 论文表格 `DLPCM/table/main_result.tex`、`DLPCM/table/main_result_disco.tex`，不是本轮重新跑出的本地测试。 | 按论文既有结果引用。 | 表格完整。 | 只能表述为“论文已有结果/引用结果”，不能说是本轮重新测试。 |
| 1.2 V2X-Sim / DATA | 无补偿、pose-warp、SyncNet、joint IFAM residual 均有本地 CSV；joint 训练使用 `v2xsim_infos_train.pkl`，评测使用 `v2xsim_infos_test.pkl`。 | train -> test。 | 主表均为 18 格完整结果。 | 通过。 |
| 2.1 DAIR-V2X-C / official late fusion | 无补偿、TCLF、严格公平 Ours RF 都有完整日志和 18 格 CSV，结果真实。新增 train cache 生成日志存在。 | Ours RF 训练使用 DAIR official `train` split，评测使用 `val` split；诊断版使用 `train+val -> val`，已明确标注。 | 18 格完整。 | 主表通过；`train+val -> val` 和旧 `val -> val` 只能作为诊断/上限。 |
| 2.2 DAIR-V2X-C / DATA intermediate | 无补偿、DATA fork pose-warp、IFAM T10-N10、IFAM joint delay-comp 都有本地 CSV。 | IFAM T10-N10/joint 训练使用 `train.json`，评测使用 `val.json`。 | 18 格完整；部分大时延格点因历史特征缺失有 skipped frames，CSV 已记录。 | DATA 官方可运行 checkpoint 复现低于论文主表，只能作为当前本地 DATA checkpoint/config 的结果。 |
| 2.2.3 / 2.2.4 DATA box-level 与 RF assist 诊断 | box-level 诊断是 18 格完整；RF assist 只跑了关键格点。 | 使用 `dair_v2x_late_fusion_rf2_ours.pkl`。 | 2.2.3 完整；2.2.4 不完整且已标为诊断。 | 同样受 RF `val` split 训练/评测重合影响，不可作为最终公平主结果。 |
| 2.3 DAIR-V2X-C / LRCP | strict dual-delay 分片日志和合并 CSV 存在，结果真实。 | LRCP adapter 评测 1783 samples，按 config 的 validation/test label split。 | 37 行：1 行 no-delay LRCP + 18 格 no-comp + 18 格 LRCP stage2 compensation。 | AP 绝对值极低，只适合同一 LRCP pipeline 内部对照。 |
| 2.4 DAIR-V2X-Seq / TraF-Align | baseline、feature T10-N10、joint-only、adaptive scale 都有真实测试 CSV；官方 checkpoint 复现核查通过。 | joint/feature predictor 训练使用 official train split；评测使用 official test split，3316 samples。 | baseline/feature/joint/adaptive 均覆盖 18 格。 | 推荐版 `adaptive scale + Comp0 bypass` 的 Comp0 行是由 baseline CSV 合成，因为 bypass 等价于原始 TraF-Align；应标为组合结果。 |

### 5.2 最终使用的 Ours / 补偿 Checkpoint

| Checkpoint | Baseline / 数据集 | 预测对象与训练方式 | 使用位置 | 审计状态 |
|---|---|---|---|---|
| `checkpoints/curated/ours_models/v2xsim_ours_prediction_model.pth` | V2X-Sim / V2VNet、DiscoNet 论文结果对应的 ours 模型 | 原 DLPCM/V2X-Sim ours 预测模型；本地文件存在，`step=38000`。 | 1.1、1.3 作为论文既有结果对应模型备注。 | 存在；本轮未重新评测。 |
| `checkpoints/curated/ours_models/v2xsim_data_ifam_joint_delay_comp_scale01.pth` | V2X-Sim / DATA | 冻结 DATA detector，在 IFAM 输入特征上训练 gated residual delay compensator；训练 train，评测 test。 | 1.2 `DATA + Ours joint IFAM residual`。 | 存在；18 格真实测试。 |
| `checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainonly.pkl` | DAIR-V2X-C / official late fusion | detector-side RF box predictor；2 帧 consecutive history；输出 box center `(dx, dy)`；训练 `train`，验证/评测 `val`。 | 2.1、2.1.3 主表。 | 存在；18 格真实测试；严格公平主模型。SHA256 `8f11c8cea2b5551dfc728db0d55b6a9cb4400f45268c33e75dd939102bca68eb`。 |
| `checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours_trainval_diagnostic.pkl` | DAIR-V2X-C / official late fusion | detector-side RF box predictor；2 帧 consecutive history；训练 `train+val`，评测 `val`。 | 2.1 补充诊断/上限。 | 存在；18 格真实测试；评测 split 参与训练，不作为公平主结果。SHA256 `488e1cf07cb2b4984c8fc44179737a0487f399c1b7b0ede765fbfe9639cb4808`。 |
| `checkpoints/curated/ours_models/dair_v2x_late_fusion_rf2_ours.pkl` | DAIR-V2X-C / official late fusion 旧诊断；DATA box diagnostics 旧行 | detector-side RF box predictor；2 帧 consecutive history；输出 box center `(dx, dy)`。 | 2.1 旧泄漏诊断；2.2.3、2.2.4 旧 RF 对照行。 | 存在；结果真实；metadata 显示 `train_split=val`、`val_split=val`，不得作为严格公平最终模型。 |
| `checkpoints/curated/ours_models/dair_data_ifam_feature_t10n10_ours.npz` | DAIR-V2X-C / DATA intermediate | DATA IFAM 输入 BEV feature 的 T10-N10 线性 residual predictor；训练 `train.json`。 | 2.2 `DATA + Ours IFAM-feature T10-N10`。 | 存在；18 格真实测试；效果失败，保留为诊断。 |
| `checkpoints/curated/ours_models/dair_data_ifam_joint_delay_comp.pth` | DAIR-V2X-C / DATA intermediate | 冻结 DATA detector，只训练 IFAM 输入特征 gated residual delay compensator；1500 steps。 | 2.2 `DATA + Ours IFAM joint delay-comp`。 | 存在；18 格真实测试；当前 DATA 主表正向版本。 |
| `checkpoints/curated/ours_models/dair_v2x_seq_trafalign_feature_t10n10_ours_spconv2121.npz` | DAIR-V2X-Seq / TraF-Align | TraF-Align backbone feature 空间 T10-N10 线性 residual predictor；训练 official train split。 | 2.4 `TraF-Align + Ours feature T10-N10`。 | 存在；18 格真实测试；效果失败，保留为诊断。 |
| `checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_spconv2121/net_epoch1.pth` | DAIR-V2X-Seq / TraF-Align | 从官方 TraF-Align checkpoint 初始化，只训练新增 gated residual delay-comp module；冻结原 detector。 | 2.4 `TraF-Align + Ours joint delay-comp only`。 | 存在；18 格真实测试。 |
| `checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_adaptive2to4_spconv2121/net_epoch1.pth` | DAIR-V2X-Seq / TraF-Align | 与 joint-only 相同权重；通过 config/推理策略加入 delay-adaptive residual scale。 | 2.4 adaptive scale。 | 存在；18 格真实测试；该 checkpoint SHA 与 joint-only 相同，区别在 config/推理策略。 |
| `checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_adaptive2to4_bypass_comp0_spconv2121/net_epoch1.pth` | DAIR-V2X-Seq / TraF-Align | 与 joint-only 相同权重；推理时 `Comp0` bypass，`Comp>0` 使用 adaptive scale。 | 2.4 推荐版。 | 存在；推荐 CSV 为 Comp0 baseline + Comp2/4 adaptive 的公平组合结果。 |

### 5.3 Baseline Checkpoint 存在性

| Baseline | Checkpoint / 配置 | 状态 |
|---|---|---|
| DAIR-V2X-C / DATA | `runs/data_dair_official_eval/net_epoch_bestval_at21.pth`; `runs/data_dair_official_eval/config.yaml` | 存在；本地 LiDAR-only DATA checkpoint。 |
| DAIR-V2X-C / LRCP stage2 | `external/LRCP/opencood/logs/dairv2x_lrcp_point_pillar_lidar_deformable_attn_stage2_2026_07_01_19_32_46/net_epoch_bestval_at1.pth` | 存在；2.3 strict dual-delay 使用的默认 stage2 目录。 |
| DAIR-V2X-Seq / TraF-Align official | `/tmp/TraF-Align_partial/checkpoints/dair_v2x_seq_Trafalign_2024_12_28_04_42_36/net_epoch60.pth` | 存在；官方 checkpoint 复现核查通过。 |
| DAIR-V2X-C / official late fusion | `external/DAIR-V2X-isolated/gpu*/v2x` 下官方 detector pipeline 与缓存 | 存在；2.1 使用日志可追溯。 |

### 5.4 当前未使用或不建议作为最终模型的 Checkpoint

| Checkpoint | 作用 | 建议 |
|---|---|---|
| `checkpoints/curated/dair_v2x/our_method/T10_n10_ckpt_15000.pth` | DAIR 早期/中期 VSPM/T10-N10 历史模型；不在当前 260704 主表使用。 | 先保留，待确认是否还服务消融或旧图表。 |
| `checkpoints/curated/dair_v2x/our_method/T10_n5_ckpt_16000.pth` | DAIR 早期/中期 VSPM/T10-N5 历史模型；不在当前 260704 主表使用。 | 先保留，待确认是否还服务消融或鲁棒性表。 |
| `checkpoints/curated/ours_models/dair_data_ifam_joint_delay_comp_smoke.pth` | DAIR DATA IFAM joint 的 smoke 训练。 | 可删除候选，需用户确认。 |
| `checkpoints/curated/ours_models/v2xsim_data_ifam_joint_delay_comp_smoke.pth` | V2X-Sim DATA IFAM joint 的 smoke 训练。 | 可删除候选，需用户确认。 |
| `checkpoints/curated/ours_models/v2xsim_data_ifam_joint_delay_comp.pth` | V2X-Sim DATA joint 未缩放诊断版；AP@0.5 高但 AP@0.7 下降。 | 诊断用；如只保留最终主表模型，可作为删除候选。 |
| `checkpoints/curated/ours_models/trafalign_joint_delay_comp_smoke/`、`trafalign_joint_delay_comp_only_smoke/` | TraF-Align smoke 训练。 | 可删除候选，需用户确认。 |
| `checkpoints/curated/ours_models/trafalign_joint_delay_comp_spconv2121/` | 曾尝试训练更多 TraF-Align 模块，非当前正式结果。 | 可删除候选，需用户确认。 |
| `checkpoints/curated/ours_models/trafalign_joint_delay_comp_only_scale01_spconv2121/`、`scale02_spconv2121/`、`adaptive_spconv2121/` | TraF-Align 推理策略/缩放实验目录；权重 SHA 与 joint-only 相同。 | 若保留推荐版和 adaptive2to4，可删除这些中间策略目录，需用户确认。 |

已完成的最重要修复：重新生成 DAIR official late-fusion `train` split detector raw cache，训练 `train -> val` 严格公平 RF，并重跑 2.1、2.1.3 的 18 格结果。本轮进一步补跑了所有仍使用旧 RF 的 DATA 诊断：2.2.3 的 DATA box-level 18 格和 2.2.4 的 RF assist 关键格点都已追加 strict RF 行；旧 RF 行保留为对照。
