# Revision TODO

本文档把两位审稿人的意见整理成可执行 TODO。目标是明确：

- 哪些内容只需要改论文文字。
- 哪些实验可直接基于当前代码补充。
- 哪些实验需要新增代码或接入完整 cooperative perception/detection 框架。
- 每个 TODO 最终要在论文中形成什么表、图或段落。

当前仓库主要覆盖 BEV occupancy forecasting、ConvLSTM/GRU 时序预测、传输态压缩和 leave-one-agent-out 泛化实验。当前仓库尚未包含完整 V2X 检测融合、mAP 评估、DAIR-V2X 数据读取、定位噪声、丢包、N 车缓存管理等模块。

## 0. 当前代码对应关系

| 内容 | 当前代码/结果 | 可直接支撑的论文内容 |
|---|---|---|
| 数据序列构造 | `src/v2x_forecasting/train_bev.py::SeqDataset` | V2X-Sim BEV 序列预测设置，历史帧 `T`、预测跨度 `n` |
| ConvLSTM 预测模块 | `src/v2x_forecasting/train_bev.py::BEVForecastNet` | VSPM/预测模块设计、rollout 机制 |
| 训练目标 | `src/v2x_forecasting/train_bev.py::train` | BCE、Dice、Focal、change head、contrast、dynamic weighting |
| 逐 horizon 评估 | `src/v2x_forecasting/train_bev.py::evaluate` | 不同预测间隔下的 IoU、Delta IoU、Dynamic IoU |
| 预测结果导出 | `src/v2x_forecasting/train_bev.py::export_predictions` | `metrics.csv`、IoU-vs-interval 图 |
| 压缩/模型状态传输 | `src/v2x_forecasting/train_bev_compressed.py` | GRU hidden state、FP16、64x64 编码、1x1 bottleneck |
| 压缩率验证 | `scripts/verify_compress.py` | 传输体积、压缩比、NMSE |
| leave-one-agent-out | `scripts/train_leave_one_agent.sh` | 跨 agent 泛化 |
| 已有结果汇总 | `results/summary/*.csv`, `docs/EXPERIMENTS.md` | 已有 IoU/Dynamic IoU 表格 |

## 1. 总体优先级

### P0: 必须补

- 引言和相关工作重写，明确核心 gap。
- 方法部分补充 ConvLSTM、state representation、rollout granularity、loss 的设计理由。
- 系统级消融：sender-side modeling、model-state transmission、dual-latency fusion。
- 更强 latency baseline 或至少充分解释 baseline 选择。
- 增加真实数据集或严肃讨论 domain gap；优先补 DAIR-V2X/V2V4Real/OPV2V 之一。
- 增加通信延迟 vs 计算延迟的区分解释和实验。

### P1: 强烈建议补

- 定位噪声鲁棒性。
- N 车扩展性、计算复杂度、内存开销。
- 丢包、jitter、sensor degradation。
- NLOS/blind spot 信息增益分析。
- 缓存管理和 eviction 策略。

### P2: 图表和呈现

- 重新绘制已有图，放大字体、图例、坐标轴。
- 每个图表在正文中明确对应一个实验 claim。
- 增加方法对比表、实验设置表、消融表、复杂度表。

## 2. 论文文字修改 TODO

### 2.1 Introduction

TODO:

- [ ] 在第一部分明确问题设置：cooperative perception 中 remote feature/message 到达 ego 端时同时受到 communication delay 和 computation delay 影响。
- [ ] 明确 prior failure modes：
  - [ ] 只补偿通信延迟的方法无法处理最终融合/检测输出滞后。
  - [ ] 只做本地预测的方法无法利用 V2X blind spot/NLOS 信息。
  - [ ] 只做 feature alignment 的方法通常不显式建模计算延迟。
- [ ] 明确 deployment assumptions：
  - [ ] 多车共享历史 BEV feature/state。
  - [ ] ego 端缓存每个 remote agent 的模型状态。
  - [ ] 延迟以 frame 为单位估计或量化。
  - [ ] remote branch 和 ego branch 可以分别预测到统一目标时间。
- [ ] 加入 reviewer 2 第 3 点的关键解释：
  - [ ] communication delay 主要使 remote data 过时，但 ego local sensor 仍是当前帧。
  - [ ] computation delay 会使最终输出时刻滞后，因此 local ego feature 也过时。
  - [ ] 这解释了为什么需要 local latency compensation，而不是只补偿通信。
- [ ] 加入 V2X information gain 的价值主张：
  - [ ] V2V/V2X 的价值不仅是更快，还包括检测 ego LiDAR 看不到的 blind spot/NLOS 目标。
  - [ ] DLPCM 的目标是在延迟存在时保留这部分 safety-critical information。

输出:

- Introduction 新增 2-4 段。
- 一个清晰 problem statement，可写成公式或文字定义。

### 2.2 Related Work

TODO:

- [ ] 扩展 latency-aware cooperative perception 文献。
- [ ] 扩展 prediction-based cooperative perception 文献。
- [ ] 增加方法对比表。

建议对比表字段:

| Method | Delay modeled at | Prediction executed at | Comp. latency considered? | Comm. latency considered? | Transmitted representation | Limitation |
|---|---|---|---|---|---|---|
| V2VNet | feature/message fusion | no explicit rollout | No | Limited | feature/message | weak under delay |
| DiscoNet | feature distillation/fusion | no explicit rollout | No | Limited | feature | no dual-latency model |
| SyncNet | temporal synchronization | receiver/fusion side | No or limited | Yes | feature | focuses on temporal misalignment |
| Extended SyncNet | temporal alignment + extension | receiver side | Partial | Yes | feature/state | needs clarify |
| New latency baseline 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| New latency baseline 2 | TBD | TBD | TBD | TBD | TBD | TBD |
| DLPCM | sender/receiver state rollout + fusion | sender-side model state and receiver rollout | Yes | Yes | model/state | proposed |

输出:

- Related Work 新增一张 method comparison table。
- 相关工作最后一段明确 novelty boundary。

### 2.3 Method

TODO:

- [ ] ConvLSTM 选择理由：
  - [ ] BEV feature 是空间网格，ConvLSTM 保留局部空间结构。
  - [ ] LSTM cell state 适合缓存运动历史，比单帧外推更稳定。
  - [ ] 可用 ConvGRU 作为轻量替代，在 ablation 中比较。
- [ ] State representation 选择理由：
  - [ ] 传输 hidden/cell state 或压缩后的 model state，而不是传输完整历史 BEV。
  - [ ] 解释 h/c 或 h-only state 的带宽/性能权衡。
- [ ] Rollout granularity 选择理由：
  - [ ] frame-level rollout 与感知系统帧率、延迟估计单位一致。
  - [ ] 逐帧 rollout 便于同时处理通信延迟和计算延迟。
- [ ] Training objective 选择理由：
  - [ ] BCE 处理 occupancy。
  - [ ] Dice 缓解稀疏前景。
  - [ ] change/focal/dynamic weighting 强调动态区域。
  - [ ] contrast 保持短期时间连续性。
- [ ] 加入复杂度公式：
  - [ ] 单个 agent rollout cost。
  - [ ] N 个 agent 并发 cost。
  - [ ] cache memory cost。

输出:

- Method 每个设计点增加 1 段解释。
- 新增 complexity analysis 小节。

### 2.4 Discussion and Limitations

TODO:

- [ ] 增加 simulation-to-real domain gap 讨论：
  - [ ] heterogeneous agents。
  - [ ] sensing noise。
  - [ ] packet loss。
  - [ ] hardware jitter。
  - [ ] positioning noise。
  - [ ] weather/sensor degradation。
- [ ] 明确 spatial calibration 不属于当前 DLPCM 主贡献，或说明可作为互补模块。
- [ ] 说明 cache eviction 策略。
- [ ] 如果无法完成真实数据集实验，必须在 Discussion 中诚实说明限制。

输出:

- Discussion 新增 2-3 段。
- Limitations 中明确 future work，避免过度声称 deployment value。

## 3. 实验 TODO

### 3.1 Existing Forecasting/Sensitivity Experiments

目的:

- 回应 reviewer 1 第 3 点：设计选择是否 principled。
- 基于当前代码可直接跑。

实验 A: ConvLSTM vs ConvGRU

- [ ] ConvLSTM baseline: `src/v2x_forecasting/train_bev.py`
- [ ] ConvGRU: `src/v2x_forecasting/train_bev_compressed.py --enable_gru_fp16`，如果要公平比较，需要新增不启用 FP16 的 pure ConvGRU flag。
- [ ] 指标：IoU、Delta IoU、Dynamic IoU、params、latency、memory。

实验 B: History length `T`

- [ ] `T=5`
- [ ] `T=10`
- [ ] `T=20`
- [ ] `T=30`
- [ ] 固定 `n`，比较预测稳定性和计算开销。

实验 C: Rollout horizon `n`

- [ ] `n=3`
- [ ] `n=5`
- [ ] `n=10`
- [ ] `n=15`
- [ ] 输出 IoU-vs-interval 曲线。

实验 D: Loss ablation

- [ ] BCE only。
- [ ] BCE + Dice。
- [ ] BCE + Dice + change head。
- [ ] BCE + Dice + change head + focal。
- [ ] Full objective + contrast。
- [ ] Full objective + dynamic weighting。

产出:

- Table: sensitivity of model design。
- Figure: IoU vs prediction interval。

### 3.2 System-Level Ablation

目的:

- 回应 reviewer 1 第 4 点：不能只消融 prediction model，还要消融系统组件。

必须做的 variant:

- [ ] No compensation: 直接使用 delayed remote feature/local output。
- [ ] Communication-only compensation: 只预测 remote branch 到当前或目标时刻。
- [ ] Computation-only compensation: 只补偿最终输出计算延迟。
- [ ] Dual-latency compensation: 同时补偿 communication 和 computation。
- [ ] Without sender-side modeling。
- [ ] Without model-state transmission，改为传输原始 BEV/feature。
- [ ] Without dual-latency fusion，改为 naive fusion。
- [ ] Full DLPCM。

指标:

- [ ] mAP 或 BEV IoU。
- [ ] Dynamic IoU。
- [ ] Bandwidth per agent/frame。
- [ ] Computation latency。
- [ ] GPU memory。
- [ ] FLOPs 或 MACs。

当前代码状态:

- 当前代码能做 BEV IoU/Dynamic IoU。
- 当前代码不能做完整 detection mAP，需要接入检测框架。
- `train_bev_compressed.py` 可支撑 model-state transmission 的带宽消融。

产出:

- Main ablation table。
- Bandwidth/performance trade-off table。
- Latency breakdown table。

### 3.3 Model-State Transmission / Bandwidth Ablation

目的:

- 量化 MST/model-state transmission 对带宽和计算量的贡献。

实验 variant:

- [ ] Raw BEV history transmission。
- [ ] Full ConvLSTM h+c state, FP32。
- [ ] Full ConvLSTM h+c state, FP16。
- [ ] ConvGRU h-only state, FP32。
- [ ] ConvGRU h-only state, FP16。
- [ ] 64x64 temporal state。
- [ ] 1x1 bottleneck dim = 8。
- [ ] 1x1 bottleneck dim = 12。
- [ ] 1x1 bottleneck dim = 16。
- [ ] 1x1 bottleneck dim = 24。
- [ ] 1x1 bottleneck dim = 32。

可用代码:

- `scripts/verify_compress.py`
- `src/v2x_forecasting/train_bev_compressed.py`
- `scripts/train_compressed.sh`

需要新增:

- [ ] 统一输出每个 variant 的 bytes/frame。
- [ ] 统一输出 encoder/decoder latency。
- [ ] 统一输出 params 和 GPU memory。

产出:

- Table: bandwidth reduction and computation saving。
- Figure: performance-bandwidth Pareto curve。

### 3.4 Communication Delay vs Computation Delay

目的:

- 回应 reviewer 2 第 3 点。
- 证明计算延迟比通信延迟更影响最终输出，因为 ego feature 也会过时。

实验设计:

- [ ] communication delay `d_comm = 0,1,2,3,5` frames。
- [ ] computation delay `d_comp = 0,1,2,3,5` frames。
- [ ] 固定其中一个，扫描另一个。
- [ ] 同时扫描二维矩阵。

对比:

- [ ] No compensation。
- [ ] Remote-only compensation。
- [ ] Local-only compensation。
- [ ] Dual compensation。

需要新增代码:

- [ ] delay simulator：按 frame index 偏移 remote input 和 final output。
- [ ] local branch outdated simulation。
- [ ] 结果聚合脚本。

产出:

- 2D heatmap: mAP/IoU under `d_comm x d_comp`。
- Figure: communication-only vs computation-only degradation。
- 正文解释为什么 local latency compensation 必要。

### 3.5 Positioning Noise Robustness

目的:

- 回应 reviewer 2 第 1 点。

实验设计:

- [ ] Translation noise: `0, 0.2, 0.5, 1.0, 2.0 m`。
- [ ] Rotation noise: `0, 0.5, 1.0, 2.0 deg`。
- [ ] Bias noise 和 Gaussian noise 分开测。
- [ ] 组合噪声：translation + rotation。

对比:

- [ ] Baseline fusion。
- [ ] DLPCM without spatial correction。
- [ ] DLPCM + optional calibration/spatial refinement，如果能实现。

当前代码状态:

- 当前仓库没有 pose transform/fusion 模块，不能直接做。

需要新增:

- [ ] 数据读取中保留 agent pose。
- [ ] BEV feature warp/transform。
- [ ] pose noise injection。
- [ ] 可选 spatial calibration/refinement 模块。

产出:

- Table/Figure: performance under positioning noise。
- Discussion: DLPCM 解决 temporal alignment，不替代 spatial calibration。

### 3.6 Real-World Dataset Validation

目的:

- 回应 reviewer 1 第 6 点和 reviewer 2 第 6 点。

优先级:

1. DAIR-V2X。
2. V2V4Real。
3. OPV2V。

TODO:

- [ ] 确定选哪个真实/更真实 benchmark。
- [ ] 写 dataset adapter，将数据转换到当前 BEV occupancy 或接入完整 detection pipeline。
- [ ] 统一 train/val/test split。
- [ ] 跑主方法。
- [ ] 跑至少 2 个 baseline。
- [ ] 报告 mAP/AP@IoU 或 BEV IoU。
- [ ] 报告 latency 和带宽。

如果无法完成真实数据集:

- [ ] 至少新增 domain gap robustness：packet loss、jitter、sensor degradation。
- [ ] 在 Discussion 中明确真实部署验证仍是限制。

产出:

- Table: cross-dataset validation。
- Discussion: sim-to-real gap。

### 3.7 Packet Loss / Jitter / Sensor Degradation

目的:

- 回应真实环境非理想通信和硬件噪声。

实验设计:

- [ ] Packet loss rate: `0%, 10%, 20%, 30%, 50%`。
- [ ] Delay jitter: uniform 或 Gaussian frame jitter。
- [ ] Sensor degradation:
  - [ ] random BEV dropout。
  - [ ] sparsity increase。
  - [ ] occupancy noise。

需要新增:

- [ ] packet loss simulator。
- [ ] jitter simulator。
- [ ] BEV degradation augmentation。

产出:

- Table: robustness under non-ideal communication。
- Figure: performance vs packet loss。

### 3.8 N-Agent Scalability and Complexity

目的:

- 回应 reviewer 2 第 2 点。

实验设计:

- [ ] `N=1`
- [ ] `N=2`
- [ ] `N=3`
- [ ] `N=4`
- [ ] `N=5`
- [ ] 如果数据不支持更多 agent，则模拟 `N=8/16` cache and rollout workload。

指标:

- [ ] Runtime latency per frame。
- [ ] FPS。
- [ ] GPU memory。
- [ ] CPU memory/cache memory。
- [ ] Number of cached VSPM states。
- [ ] mAP/IoU。
- [ ] Bandwidth。

理论分析:

- [ ] 单 agent ConvRNN rollout complexity。
- [ ] N agent total complexity。
- [ ] cache memory complexity。
- [ ] 说明可并行化的部分和瓶颈。

需要新增:

- [ ] profiling script。
- [ ] synthetic multi-agent workload。
- [ ] cache state size estimator。

产出:

- Table: scalability with N agents。
- Complexity paragraph in Method。

### 3.9 Cache Management and Eviction

目的:

- 回应 reviewer 2 第 7 点。

策略:

- [ ] LRU eviction。
- [ ] TTL eviction：超过若干 frames 未收到消息则删除。
- [ ] Range-based eviction：车辆离开通信范围则删除。
- [ ] Hybrid: range + TTL + LRU capacity。

需要新增:

- [ ] VSPM cache manager。
- [ ] cache hit/miss 统计。
- [ ] memory usage 统计。
- [ ] lookup time profiling。

实验:

- [ ] 不设 eviction。
- [ ] TTL only。
- [ ] LRU only。
- [ ] Range + TTL + LRU。

产出:

- Table: cache memory and lookup latency。
- Method/Implementation paragraph: cache policy。

### 3.10 V2X Information Gain / Blind Spot / NLOS

目的:

- 回应 reviewer 2 第 4 点。
- 证明 V2X 提供本车单 LiDAR 无法获得的信息。

实验设计:

- [ ] 将目标或 BEV cells 划分为 ego-visible 和 ego-invisible/NLOS。
- [ ] 对 blind spot/NLOS 区域单独计算 AP/Recall/IoU。
- [ ] 比较 onboard-only、delayed cooperative、DLPCM。

需要新增:

- [ ] visibility/occlusion mask。
- [ ] NLOS split metric。
- [ ] 如果用 detection，需要保留目标级 visibility label 或用 ray casting 近似。

产出:

- Table: NLOS/blind spot AP or recall。
- Qualitative visualization: blind spot object recovered by DLPCM。

### 3.11 Stronger Latency Baselines

目的:

- 回应 reviewer 1 第 7 点和 reviewer 1 第 2 点。

TODO:

- [ ] 梳理已有 baseline 是否真的包含 V2VNet、DiscoNet、SyncNet、extended SyncNet。
- [ ] 增加 2-4 个 recent latency/delay/asynchronous cooperative perception 方法。
- [ ] 如果无法复现，至少给出充分理由：
  - [ ] 代码不可用。
  - [ ] 数据格式不兼容。
  - [ ] 不支持 computation delay。
  - [ ] 只处理 communication delay。
- [ ] 对所有 baseline 统一延迟设置、通信带宽和 backbone。

候选 baseline 类型:

- [ ] temporal synchronization method。
- [ ] latency-aware fusion method。
- [ ] asynchronous feature fusion method。
- [ ] prediction-based cooperative perception method。
- [ ] communication interruption recovery method。

产出:

- Main comparison table。
- Related Work comparison table。
- Response letter 中解释 baseline 充分性。

## 4. 图表 TODO

### 4.1 必须新增/重做的表

- [ ] Table 1: Related work comparison。
- [ ] Table 2: Main comparison with stronger baselines。
- [ ] Table 3: System-level ablation。
- [ ] Table 4: Model design sensitivity。
- [ ] Table 5: Bandwidth/computation saving。
- [ ] Table 6: Real-world or cross-dataset validation。
- [ ] Table 7: N-agent scalability and cache overhead。
- [ ] Table 8: Robustness under pose noise/packet loss/jitter。

### 4.2 必须新增/重做的图

- [ ] Figure: framework diagram with communication delay and computation delay clearly marked。
- [ ] Figure: communication vs computation delay heatmap。
- [ ] Figure: IoU/mAP vs delay frames。
- [ ] Figure: performance-bandwidth Pareto curve。
- [ ] Figure: NLOS/blind spot qualitative visualization。
- [ ] Figure: N-agent scalability curve。

### 4.3 现有图表改进要求

- [ ] 字体增大到论文双栏可读。
- [ ] legend 使用明确方法名，不用 file1/file2。
- [ ] 坐标轴写清单位，例如 delay frames、bandwidth KB/frame。
- [ ] caption 写清实验结论。
- [ ] 正文每个图至少有一句明确 claim。

## 5. 代码 TODO

### 5.1 基于当前代码可直接新增

- [ ] 新增 `scripts/run_sensitivity.sh`：批量跑 `T`、`n`、loss ablation。
- [ ] 新增 `scripts/profile_forecasting.py`：统计 latency、显存、params。
- [ ] 新增 `scripts/summarize_ablation.py`：统一生成 ablation CSV。
- [ ] 改进 `scripts/plot_iou_interval.py`：支持自定义 label、字体大小、论文风格。
- [ ] 改进 `scripts/plot_iou_interval_multi.py`：替换 file1/file2 label。
- [ ] 给 `train_bev_compressed.py` 增加 pure ConvGRU without FP16 flag。
- [ ] 给 `train_bev_compressed.py` 增加 bottleneck dim sweep 脚本。

### 5.2 需要新增模块

- [ ] `src/v2x_forecasting/delay_simulator.py`
  - [ ] communication delay。
  - [ ] computation delay。
  - [ ] dual delay。
- [ ] `src/v2x_forecasting/robustness.py`
  - [ ] packet loss。
  - [ ] jitter。
  - [ ] BEV noise/degradation。
- [ ] `src/v2x_forecasting/profiling.py`
  - [ ] model params。
  - [ ] FLOPs/MACs。
  - [ ] GPU latency。
  - [ ] memory。
- [ ] `src/v2x_forecasting/cache.py`
  - [ ] VSPM cache。
  - [ ] LRU。
  - [ ] TTL。
  - [ ] range-based eviction。
- [ ] `scripts/run_system_ablation.sh`
- [ ] `scripts/run_robustness.sh`
- [ ] `scripts/run_scalability.sh`

### 5.3 需要接入外部/完整框架

- [ ] Detection mAP evaluation。
- [ ] V2VNet/DiscoNet/SyncNet baseline reproduction。
- [ ] DAIR-V2X/V2V4Real/OPV2V adapter。
- [ ] pose transform and spatial alignment。
- [ ] NLOS/blind spot visibility metric。

## 6. 回复审稿人 TODO

### Reviewer 1

R1-1 Introduction gap

- [ ] 回复：已重写 Introduction，明确 dual-latency problem、failure modes、deployment assumptions。
- [ ] 指向论文 Section I。

R1-2 Related Work

- [ ] 回复：新增 latency-aware 和 prediction-based 方法比较。
- [ ] 指向新 Table。

R1-3 Method rationale

- [ ] 回复：新增 ConvLSTM、state、rollout、loss rationale。
- [ ] 指向 Method 和 sensitivity ablation。

R1-4 System ablation

- [ ] 回复：新增 sender-side modeling、model-state transmission、dual-latency fusion ablation。
- [ ] 指向系统消融表。

R1-5 Figure/table readability

- [ ] 回复：所有图表重绘，caption 和正文 claim 补充。

R1-6 Generalization/validity

- [ ] 回复：新增真实数据集或 domain gap robustness。
- [ ] 指向 DAIR-V2X/robustness 表和 Discussion。

R1-7 Stronger baselines

- [ ] 回复：新增 recent latency-handling baseline，或解释 baseline 选择充分性。
- [ ] 指向 main comparison table。

### Reviewer 2

R2-1 Positioning noise

- [ ] 回复：新增 positioning noise 实验，或说明 spatial calibration 是互补模块并加入 limitation。

R2-2 Scalability

- [ ] 回复：新增 N-agent complexity and profiling。

R2-3 Communication vs computation delay

- [ ] 回复：新增解释和二维 delay 实验。

R2-4 V2X information gain

- [ ] 回复：新增 blind spot/NLOS 结果，强调 safety-critical information。

R2-6 Real-world validation

- [ ] 回复：新增真实数据集或非理想通信/硬件噪声实验。

R2-7 Cache management

- [ ] 回复：新增 cache eviction policy 和 memory/lookup profiling。

## 7. 建议执行顺序

第一阶段：最快能完成的 revision 支撑

- [ ] 修改 Introduction、Related Work、Method rationale。
- [ ] 整理已有 `results/summary/*.csv`，形成当前已有表格。
- [ ] 跑 sensitivity：`T`、`n`、loss、ConvLSTM/GRU。
- [ ] 跑 model-state transmission bandwidth ablation。
- [ ] 重画图。

第二阶段：系统级实验

- [ ] 实现 delay simulator。
- [ ] 跑 communication vs computation delay。
- [ ] 跑 system-level ablation。
- [ ] 加 profiling。
- [ ] 加 N-agent scalability。

第三阶段：强审稿点

- [ ] 接入真实数据集。
- [ ] 接入 stronger baselines。
- [ ] 做 positioning noise。
- [ ] 做 packet loss/jitter/sensor degradation。
- [ ] 做 blind spot/NLOS。
- [ ] 做 cache eviction。

第四阶段：写 rebuttal/response

- [ ] 每条审稿意见逐条回复。
- [ ] 每条回复包含：感谢、修改位置、实验/结果、结论。
- [ ] 避免只说 future work，除非确实无法完成实验。

## 8. 最低可接受修改包

如果时间非常紧，最低限度应完成：

- [ ] Introduction gap 重写。
- [ ] Related Work 对比表。
- [ ] Method rationale。
- [ ] System ablation 至少 4 组：no comp、comm-only、comp-only、dual。
- [ ] Bandwidth/computation table。
- [ ] Delay heatmap 或 delay curve。
- [ ] 至少一个真实/更真实数据集，或完整 non-ideal robustness 代替。
- [ ] 强 baseline 增加至少 2 个，或明确解释不可复现原因。
- [ ] Discussion 诚实讨论 positioning noise、cache、domain gap。

## 9. 当前不能直接声称的内容

在补实验前，论文中不要强声称：

- [ ] 已在真实道路环境验证。
- [ ] 已证明对定位噪声鲁棒。
- [ ] 已证明对大规模 N 车密集交通可实时运行。
- [ ] 已证明 cache 管理不会成为瓶颈。
- [ ] 已全面优于所有 recent latency-aware methods。
- [ ] 已在 mAP 上验证 detection performance，除非接入 detection pipeline 并跑完。
