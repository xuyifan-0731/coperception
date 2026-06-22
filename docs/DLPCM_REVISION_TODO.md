# DLPCM Revision TODO

本文档基于原论文目录 `/raid/xuyifan/v2x_code_ckpt/DLPCM` 和两位审稿人的意见整理。它比 `docs/REVISION_TODO.md` 更具体，直接对应论文文件、已有图表、已有实验和需要新增的实验。

## 0. 原论文文件对应

| 论文内容 | 文件 | 当前状态 | 主要修改目标 |
|---|---|---|---|
| 摘要、关键词、整体编译入口 | `DLPCM/main.tex` | 已有摘要，已强调 DLPCM/VSPM/MST | 摘要需避免过强部署声称，补充真实/非理想实验后再强化 |
| Introduction | `DLPCM/1_introduction.tex` | 已说明 CP、dual latency、VSPM/MST | 需要更清晰定义 gap、failure modes、deployment assumptions |
| Related Work | `DLPCM/2_related_work.tex` | 已有分类表，但比较维度偏粗 | 需要细粒度比较 delay modeled at、prediction location、computation latency |
| Method | `DLPCM/3_dual.tex` | 已有 VSPM、MST、DLPCM、Algorithm 1 | 需要补设计理由、复杂度、cache、pose noise/spatial calibration 说明 |
| Experiments | `DLPCM/4_experiment.tex` | 已有 V2X-Sim、VSPM、MST、V2VNet/DiscoNet/SyncNet 结果 | 需要新增系统级消融、真实/鲁棒性、N-agent、NLOS、定位噪声 |
| Conclusion | `DLPCM/5_conclusion.tex` | 已有总结和 future work | 需要加入限制和 domain gap，避免过度声称 |
| V2VNet 主表 | `DLPCM/table/main_result.tex` | 已有 AP@0.5/0.7，comm 0-5，comp 0/2/4 | 需要补 dual-latency 消融和更强 baseline |
| DiscoNet/SyncNet 主表 | `DLPCM/table/main_result_disco.tex` | 已有 SyncNet、SyncNet w/ Comp.、DiscoNet、Ours | 需要说明 baseline 选择充分性，最好新增近期方法 |
| 框架图 | `DLPCM/figure2.png` | 已有 dual-latency pipeline | 需要标注 cache、eviction、pose/spatial calibration、local/remote compensation |
| 动机图 | `DLPCM/figure1.jpg` | 已有 No CP / uncompensated / compensated | 需要在正文中更明确对应 blind spot/NLOS 信息增益 |

## 1. 总体修稿策略

### 必须优先完成

- [ ] 重写 Introduction 中的核心 gap 段落，明确 prior methods 各自失败在哪里，以及为什么必须统一处理 communication latency 和 computation latency。
- [ ] 扩充 Related Work 表格，让 novelty boundary 可追踪。
- [ ] 在 Method 中补充 VSPM、ConvLSTM、state representation、rollout granularity、training objective 的设计理由。
- [ ] 在 Experiment 中增加系统级消融，而不是只消融 VSPM 预测模块。
- [ ] 增加至少一种真实/更真实数据集，或补充充分的 domain gap/非理想条件实验。
- [ ] 增加定位噪声、N-agent scalability、cache management 的实验或明确讨论。
- [ ] 增加或解释更强 latency-aware baseline。

### 可以快速完成

- [ ] 使用已有 `DLPCM/table/main_result.tex` 和 `main_result_disco.tex` 重组一个更清晰的 latency comparison。
- [ ] 使用已有 `results/summary/*.csv` 补充 VSPM sensitivity 表。
- [ ] 用当前 `src/v2x_forecasting/train_bev_compressed.py` 和 `scripts/verify_compress.py` 补 MST 带宽/压缩率消融。
- [ ] 修改图表 caption 和正文分析，让每张图对应一个明确结论。

### 当前代码不足

- [ ] 当前 repo 的 `src/v2x_forecasting` 只覆盖 BEV occupancy forecasting，不包含完整 V2VNet/DiscoNet detection pipeline。
- [ ] 当前 repo 没有 DAIR-V2X/V2V4Real/OPV2V adapter。
- [ ] 当前 repo 没有 pose noise injection、packet loss、jitter、cache manager、NLOS split。
- [ ] 如果这些结果来自外部目录或旧实验，需要把脚本、日志、表格和生成方式整理进本仓库或补充说明。

## 2. Reviewer 1 TODO

### R1-1 Introduction: core research gap 不够清晰

对应文件:

- `DLPCM/1_introduction.tex`
- `DLPCM/main.tex` 摘要可同步微调

当前已有内容:

- 已说明 communication latency 造成历史数据滞后。
- 已说明 computation latency 造成结果更新滞后。
- 已说明 receiver-side prediction 有历史不足、泛化弱、计算重的问题。

需要修改:

- [ ] 在 Introduction 第二段后新增一个明确 problem setting：
  - [ ] 每个 remote packet 带有 sender perception time `t_per`。
  - [ ] receiver 在 `t_receive` 接收 packet。
  - [ ] 最终 fusion result 在 `t_result = t_receive + tau_comp` 输出。
  - [ ] 因此 remote branch 需要补偿 `t_result - t_per`。
  - [ ] local branch 也需要补偿 `t_result - t_receive` 或 `t_result - t_per^re`。
- [ ] 明确 failure modes：
  - [ ] Data compression/ROI 只能降低 payload，不能消除残余 latency。
  - [ ] Temporal alignment 主要处理 received remote data 与 ego 当前帧的对齐，不能处理 final output 计算滞后。
  - [ ] Receiver-side prediction 初次通信缺少历史，且 N 个 remote agents 会带来重复预测计算。
  - [ ] Scheduling/hardware acceleration 缩短计算时间，但不能让已经滞后的 perception result 回到真实物理时刻。
- [ ] 明确 deployment assumptions：
  - [ ] sender 可以维护自己的 VSPM state。
  - [ ] receiver 缓存每个 sender 的 `theta_i`。
  - [ ] latency 以 frame 为单位量化，当前实验中每帧 `0.2s`。
  - [ ] receiver 能获得 packet timestamp 和 sender pose。
  - [ ] fusion model 可为 V2VNet/DiscoNet 等通用 fusion head。
- [ ] 加一句强调 unified treatment 的必要性：
  - [ ] 只补偿 communication latency 会让 remote data 更及时，但 final output 仍滞后。
  - [ ] 只加速 computation 不能恢复 remote message 的历史状态。
  - [ ] DLPCM 的关键是把 local 和 remote 都推进到统一 `t_result`。

论文产出:

- [ ] Introduction 新增 2-3 段。
- [ ] 新增或修改 Figure 2 caption，强化 `t_per`、`t_receive`、`t_result`。

### R1-2 Related Work: 需要更细粒度比较

对应文件:

- `DLPCM/2_related_work.tex`

当前已有内容:

- 当前 Table `related work` 按类别比较 proactive/reaction/computation latency，粒度偏粗。

需要修改:

- [ ] 把现有 Related Work 表改成更细粒度表，建议字段：
  - [ ] Method/category。
  - [ ] Delay modeled at: transmission, feature alignment, fusion, final output。
  - [ ] Prediction location: sender, receiver, fusion module, none。
  - [ ] Prediction input: raw frames, features, historical states, current packet。
  - [ ] Vehicle-specific history: yes/no。
  - [ ] Computation latency explicitly modeled: yes/no。
  - [ ] State/model transmission: yes/no。
  - [ ] Main limitation。
- [ ] 增加近期 latency-aware / asynchronous / prediction-based cooperative perception references。
- [ ] Related Work 最后一段要明确：
  - [ ] 现有方法多数关注 communication latency。
  - [ ] 少数方法做 prediction，但通常在 receiver 端。
  - [ ] 很少方法显式把 computation latency 加入 output-time alignment。
  - [ ] DLPCM 的 novelty boundary 是 sender-side VSPM + MST + local/remote dual-latency rollout。

论文产出:

- [ ] 新 Table: Fine-grained comparison of latency-handling methods。
- [ ] Related Work 新增 1-2 段 discussion。

### R1-3 Method: 设计选择依据不足

对应文件:

- `DLPCM/3_dual.tex`

当前已有内容:

- `Vehicle-Specific Prediction Model` 小节已经描述 ConvLSTM 架构、state、rollout、loss。
- 但说明偏“怎么做”，不够解释“为什么这样做”。

需要修改:

- [ ] 在 `Model architecture` 后补 ConvLSTM rationale：
  - [ ] BEV occupancy 是空间网格，ConvLSTM 用卷积门控保留局部空间结构。
  - [ ] ConvLSTM 的 cell state 适合保留较长历史，和 MST 传输 state 的设定一致。
  - [ ] 相比 Transformer/MLP，ConvLSTM 更轻量，适合多 agent rollout。
- [ ] 在 `Transmissible state and rollout interface` 后补 state rationale：
  - [ ] state 比 raw history 小。
  - [ ] state 已编码历史，解决 receiver 初次通信历史不足。
  - [ ] state 允许 receiver 从任意 `t_receive` 继续 rollout 到 `t_result`。
- [ ] 在 rollout 公式后补 granularity rationale：
  - [ ] frame-level rollout 与数据采样周期 `0.2s/frame` 和 latency 估计单位一致。
  - [ ] 粗粒度会错过动态物体运动，细粒度需要插值和更高计算成本。
- [ ] 在 loss 后补 objective rationale：
  - [ ] BCE 保证 occupancy calibration。
  - [ ] Dice 缓解 occupancy 稀疏和前景不平衡。
  - [ ] dynamic weights 强调运动区域。
  - [ ] scheduled sampling 缓解 train-test mismatch。
- [ ] 增加 sensitivity analysis 实验来支撑这些选择：
  - [ ] ConvLSTM vs ConvGRU。
  - [ ] `T=5/10/20/30`。
  - [ ] `n=3/5/10/15`。
  - [ ] BCE only / +Dice / +dynamic weights / +teacher forcing。

论文产出:

- [ ] Method 新增 `Design Rationale` 段。
- [ ] Experiment 新增 `Sensitivity Analysis` 表。

### R1-4 System-level ablation 不足

对应文件:

- `DLPCM/4_experiment.tex`
- `DLPCM/table/main_result.tex`
- `DLPCM/table/main_result_disco.tex`

当前已有内容:

- 已有 VSPM teacher forcing/Dice 消融。
- 已有 V2VNet/DiscoNet baseline under latency。
- 已有 MST 通信 1.9MB -> 0.65MB 和 redundant computation 57.1%。

缺口:

- 还没有明确 isolate sender-side modeling、MST、dual-latency fusion 各自贡献。

需要新增实验:

- [ ] `No Compensation`: delayed fusion，不做任何 prediction。
- [ ] `Receiver-side Generic Prediction`: 在 receiver 用通用模型补偿 remote。
- [ ] `Sender-side VSPM w/o MST`: sender-side personalized model，但传 raw/feature 或不传 state。
- [ ] `MST only`: 传 state 降低带宽，但不做 dual-latency local compensation。
- [ ] `Comm-only Compensation`: 只把 remote branch 补偿到 `t_receive`。
- [ ] `Comp-only Compensation`: 只补偿 local/final output computation latency。
- [ ] `Dual-latency Compensation`: remote 和 local 都补偿到 `t_result`。
- [ ] `Full DLPCM`: VSPM + MST + dual-latency fusion。

指标:

- [ ] AP@0.5。
- [ ] AP@0.7。
- [ ] Communication size。
- [ ] Computation FLOPs or latency。
- [ ] Memory/cache size。

论文产出:

- [ ] 新 Table: System-level ablation。
- [ ] 正文说明每个组件贡献。

### R1-5 图表可读性

对应文件:

- `DLPCM/4_experiment.tex`
- `DLPCM/pics/*.pdf`
- `DLPCM/figure1.jpg`
- `DLPCM/figure2.png`

需要修改:

- [x] `pics/step-iou.pdf`: 当前 caption 说 x-axis 是 training step，但图名和正文像是 horizon/step，需要核对并修正。
- [ ] `pics/iou_vs_interval_5agents.pdf`: 放大字体，legend 从 generic/personalized/baseline 写清。
- [ ] `pics/main_compare.pdf`: 坐标轴标清 communication latency、computation latency、mAP@0.7。
- [x] `pics/compare_disco-07.pdf`: caption 中 `DiscnNet` 拼写改为 `DiscoNet`。
- [ ] 所有图 caption 增加一句具体结论。
- [ ] 正文每张图后至少有一段解释它支撑哪个 claim。
- [ ] 大表 `main_result.tex` 和 `main_result_disco.tex` 太宽，考虑拆成 AP@0.5/AP@0.7 两个表或用更清晰的 heatmap。

论文产出:

- [ ] 更新所有 figure caption。
- [ ] 更新实验正文分析。
- [ ] 重新生成论文风格图。

### R1-6 泛化性和真实有效性

对应文件:

- `DLPCM/4_experiment.tex`
- `DLPCM/5_conclusion.tex`

当前已有内容:

- 只在 V2X-Sim 上验证。

需要新增实验或讨论:

- [ ] 真实/更真实数据集优先级：
  - [ ] DAIR-V2X。
  - [ ] V2V4Real。
  - [ ] OPV2V。
- [ ] 如果短期无法接入真实数据，至少补非理想环境实验：
  - [ ] packet loss: 0/10/20/30/50%。
  - [ ] delay jitter。
  - [ ] sensor degradation。
  - [ ] heterogeneous agents。
  - [ ] positioning noise。
- [ ] Discussion 明确 sim-to-real domain gap：
  - [ ] V2X-Sim 基于仿真，缺少硬件抖动、真实通信丢包、传感器退化。
  - [ ] DLPCM 主要解决 temporal alignment，不直接解决 spatial calibration。

论文产出:

- [ ] 新 Table: Real-world or robustness validation。
- [ ] Conclusion/Discussion 增加 limitation。

### R1-7 latency-handling baseline 不够强

对应文件:

- `DLPCM/2_related_work.tex`
- `DLPCM/4_experiment.tex`
- `DLPCM/table/main_result_disco.tex`

当前已有 baseline:

- V2VNet。
- DiscoNet。
- SyncNet。
- SyncNet w/ Comp.
- Oracle。
- No Collaboration。

需要新增或解释:

- [ ] 增加 2-3 个近期 explicit delay/asynchronous/latency robust cooperative perception 方法。
- [ ] 如果不能复现，写清楚原因：
  - [ ] 官方代码不可用。
  - [ ] 不支持 computation latency。
  - [ ] 数据格式不兼容。
  - [ ] 只处理通信延迟而非 dual latency。
- [ ] 保证所有 baseline 使用相同 latency setting：
  - [ ] communication latency = 0-5 frames。
  - [ ] computation latency = 0/2/4 frames，或扩展为 0-4。
  - [ ] 每帧 0.2s。
- [ ] 对 `SyncNet w/ Comp.` 给出更明确实现说明，避免 reviewer 质疑它只是人为扩展。

论文产出:

- [ ] 新 baseline comparison 表。
- [ ] Related Work 和 Experiments 中解释 baseline 充分性。

## 3. Reviewer 2 TODO

### R2-1 其他智能体定位噪声

对应文件:

- `DLPCM/3_dual.tex`
- `DLPCM/4_experiment.tex`
- `DLPCM/5_conclusion.tex`

当前已有内容:

- Algorithm 1 Step 5 使用 sender pose 和 receiver pose 做坐标变换。
- 但没有说明 pose noise，也没有 robustness 实验。

需要修改:

- [ ] Method 中增加 `Spatial Alignment under Pose Uncertainty` 段。
- [ ] 明确 DLPCM 负责 temporal compensation，spatial calibration 是互补模块。
- [ ] 实验中注入 pose noise：
  - [ ] translation noise: 0 / 0.2 / 0.5 / 1.0 / 2.0 m。
  - [ ] rotation noise: 0 / 0.5 / 1.0 / 2.0 deg。
  - [ ] Gaussian noise 和 fixed bias 分开。
- [ ] 报告 AP@0.5/AP@0.7 或 BEV IoU。
- [ ] 如果不实现 calibration，讨论未来加入 spatial correction/refinement。

论文产出:

- [ ] Table/Figure: Pose noise robustness。
- [ ] Method/Discussion 中说明 DLPCM 与 spatial calibration 的关系。

### R2-2 N 车可扩展性和复杂度

对应文件:

- `DLPCM/3_dual.tex`
- `DLPCM/4_experiment.tex`

当前已有内容:

- MST 小节中有 5 车、4 receiving vehicles、57.1% redundant computation。

不足:

- 没有随 N 变化的复杂度和内存实验。

需要新增:

- [ ] Method 中补理论复杂度：
  - [ ] receiver cache memory: `O(N * |theta_i|)` for model cache + `O(N * |state_i|)` for active states。
  - [ ] rollout computation: `O(N * K * C_decoder)`。
  - [ ] fusion cost: depends on fusion head, approximately `O(N * C_fusion)`。
  - [ ] sender-side encoder offloading avoids receiver repeating `O(N * C_encoder)` for each sender history。
- [ ] 实验中测试 N：
  - [ ] N=1/2/3/4/5。
  - [ ] 如果数据集最多 5 agents，模拟 N=8/16 的 cache and rollout workload。
- [ ] 指标：
  - [ ] latency ms/frame。
  - [ ] FPS。
  - [ ] GPU memory。
  - [ ] CPU memory/cache memory。
  - [ ] AP@0.7 或 IoU。
  - [ ] bandwidth。

论文产出:

- [ ] Table: Scalability w.r.t. number of agents。
- [ ] Figure: runtime/memory vs N。

### R2-3 通信延迟与计算延迟的不同影响

对应文件:

- `DLPCM/1_introduction.tex`
- `DLPCM/4_experiment.tex`
- `DLPCM/table/main_result.tex`
- `DLPCM/table/main_result_disco.tex`

当前已有内容:

- 表中已经有 comm latency 和 comp latency。
- 正文已提到 computation latency 影响大。

需要强化:

- [ ] Introduction 加入审稿人指出的关键解释：
  - [ ] communication latency 只让 remote data 过时，ego 仍有实时 local sensors。
  - [ ] computation latency 让 final output 变晚，因此 remote 和 ego local data 都过时。
  - [ ] 所以 DLPCM 必须做 local latency compensation。
- [ ] Experiments 中新增或重画曲线：
  - [ ] 固定 comp=0，扫 comm=0-5。
  - [ ] 固定 comm=0，扫 comp=0-4。
  - [ ] 比较 AP@0.7 drop。
- [ ] 用 `main_result.tex` 的 V2VNet 和 Ours 数据算出 drop rate。

论文产出:

- [ ] Figure: Differential impact of communication vs computation latency。
- [ ] 正文专门一段解释 local compensation 的必要性。

### R2-4 V2X 信息增益 vs onboard-only

对应文件:

- `DLPCM/1_introduction.tex`
- `DLPCM/4_experiment.tex`

当前已有内容:

- Figure 1 已展示 No CP blind spot。
- 表中有 No Collaboration。

不足:

- 没有专门 blind spot/NLOS 指标。

需要新增:

- [ ] 定义 ego-visible 和 ego-invisible/NLOS 区域或目标。
- [ ] 比较：
  - [ ] No Collaboration。
  - [ ] V2VNet/DiscoNet with latency。
  - [ ] DLPCM。
  - [ ] Oracle。
- [ ] 指标：
  - [ ] NLOS AP@0.5/AP@0.7。
  - [ ] blind spot recall。
  - [ ] occluded-object recall。
- [ ] 增加 qualitative case：
  - [ ] 本车 LiDAR 不可见。
  - [ ] remote agent 可见。
  - [ ] latency baseline missed or late。
  - [ ] DLPCM timely detects。

论文产出:

- [ ] Table: Information gain in blind/NLOS regions。
- [ ] Figure: qualitative visualization。
- [ ] Introduction/Discussion 强调 safety-critical information。

### R2-6 真实环境和数据集验证

对应文件:

- `DLPCM/4_experiment.tex`
- `DLPCM/5_conclusion.tex`

需要新增:

- [ ] DAIR-V2X/V2V4Real/OPV2V 至少一个。
- [ ] 如果没有检测 pipeline，先做 BEV occupancy adapter，并诚实说明不是完整 detection。
- [ ] 评估 ConvLSTM/VSPM inference time：
  - [ ] GPU server。
  - [ ] 如果有条件，NVIDIA Orin 或相近嵌入式平台。
  - [ ] 报告 encoder、decoder rollout、fusion 各部分时间。
- [ ] 非理想条件：
  - [ ] packet loss。
  - [ ] hardware-level jitter 模拟。
  - [ ] sensor degradation。

论文产出:

- [ ] Table: Real-world validation or embedded runtime。
- [ ] Discussion: remaining deployment gap。

### R2-7 Cache management and eviction

对应文件:

- `DLPCM/3_dual.tex`
- `DLPCM/4_experiment.tex`

当前已有内容:

- Algorithm 1 提到 receiver caches `theta_i`，最后 discard outdated packets。

不足:

- 没有 cache capacity、eviction policy、memory profiling。

需要修改:

- [ ] Method 加 `VSPM Cache Management` 小节：
  - [ ] cache key: vehicle ID。
  - [ ] value: `theta_i`、last state、timestamp、pose。
  - [ ] TTL: 超过 `T_cache` 未收到消息则删除。
  - [ ] LRU: cache 超过容量时删除 least recently used。
  - [ ] range-based: 离开通信范围即删除。
- [ ] Algorithm 1 Step 7 改写：
  - [ ] discard outdated packets。
  - [ ] evict stale vehicle states by TTL/range/LRU。
- [ ] 实验：
  - [ ] no eviction。
  - [ ] TTL only。
  - [ ] LRU only。
  - [ ] TTL + LRU + range。
- [ ] 指标：
  - [ ] cache memory。
  - [ ] lookup time。
  - [ ] cache hit rate。
  - [ ] AP/IoU impact。

论文产出:

- [ ] Method cache policy paragraph。
- [ ] Table: Cache overhead and eviction。

## 4. 实验新增清单

### 4.1 VSPM sensitivity

基于当前 repo 可做。

TODO:

- [ ] ConvLSTM vs ConvGRU。
- [ ] Hidden width: 16 / 32 / 64。
- [ ] History length `T`: 5 / 10 / 20 / 30。
- [ ] Horizon `n`: 3 / 5 / 10 / 15。
- [ ] Loss ablation:
  - [ ] BCE。
  - [ ] BCE + Dice。
  - [ ] BCE + dynamic weights。
  - [ ] BCE + Dice + dynamic weights。
  - [ ] with/without scheduled sampling。
- [ ] 指标：
  - [ ] IoU。
  - [ ] Dynamic IoU。
  - [ ] Delta IoU。
  - [ ] inference latency。
  - [ ] memory。

输出:

- [ ] `DLPCM/4_experiment.tex` 新增 `Sensitivity Analysis`。
- [ ] 新表 `DLPCM/table/vspm_sensitivity.tex`。

### 4.2 System-level ablation

TODO:

- [ ] No Collaboration。
- [ ] Latency without compensation。
- [ ] Receiver-side generic prediction。
- [ ] Sender-side VSPM only。
- [ ] MST only。
- [ ] Communication-only compensation。
- [ ] Computation-only compensation。
- [ ] Full DLPCM。

输出:

- [ ] 新表 `DLPCM/table/system_ablation.tex`。

### 4.3 MST compression and computation

当前已有结果:

- Feature-level data: 1.9MB。
- Compressed model states: 0.65MB。
- Reduction: 65%。
- NMSE: 0.0054%。
- Encoder: 1.6 MFLOPs。
- Decoder: 0.5 MFLOPs。
- Redundant percentage: 57.1%。

TODO:

- [ ] 说明这些数字的生成脚本和设置。
- [ ] 增加 compression variants：
  - [ ] FP32 state。
  - [ ] FP16 state。
  - [ ] int8pc。
  - [ ] int4pc。
  - [ ] bottleneck dim 8/12/16/24/32。
- [ ] 增加性能影响：
  - [ ] AP@0.7 or IoU drop。
  - [ ] NMSE。
  - [ ] bytes/frame。

输出:

- [ ] 新表 `DLPCM/table/mst_compression_ablation.tex`。
- [ ] Pareto 图：accuracy vs transmission size。

### 4.4 Dual-latency heatmap

TODO:

- [ ] V2VNet baseline AP@0.7 heatmap。
- [ ] DiscoNet baseline AP@0.7 heatmap。
- [ ] Ours AP@0.7 heatmap。
- [ ] Difference heatmap: Ours - baseline。
- [ ] Communication-only degradation curve。
- [ ] Computation-only degradation curve。

输出:

- [ ] 新图 `DLPCM/pics/dual_latency_heatmap.pdf`。
- [ ] 新图 `DLPCM/pics/comm_vs_comp_drop.pdf`。

### 4.5 Strong baselines

TODO:

- [ ] 确认可复现 baseline 列表。
- [ ] 至少新增 2 个 recent latency/asynchrony/prediction-based CP 方法。
- [ ] 每个 baseline 记录：
  - [ ] official code。
  - [ ] dataset support。
  - [ ] whether comp latency is modeled。
  - [ ] whether prediction is sender-side or receiver-side。
  - [ ] whether state/model transmission exists。
- [ ] 如果不能加入，response letter 中必须逐个说明原因。

输出:

- [ ] 更新 `main_result_disco.tex` 或新增 `strong_baselines.tex`。

### 4.6 Real-world/domain-gap

TODO:

- [ ] Dataset adapter：
  - [ ] DAIR-V2X or V2V4Real or OPV2V。
  - [ ] 统一 BEV grid resolution。
  - [ ] 统一 latency frame interval。
- [ ] 如果只做 discussion，需要补足 non-ideal simulation：
  - [ ] packet loss。
  - [ ] jitter。
  - [ ] pose noise。
  - [ ] sensor degradation。

输出:

- [ ] 新表 `DLPCM/table/domain_gap_robustness.tex`。

### 4.7 N-agent and cache

TODO:

- [ ] N=1/2/3/4/5 real data。
- [ ] N=8/16 synthetic workload。
- [ ] Cache policy:
  - [ ] no eviction。
  - [ ] TTL。
  - [ ] LRU。
  - [ ] TTL + LRU + range。
- [ ] 指标：
  - [ ] latency。
  - [ ] memory。
  - [ ] cache hit rate。
  - [ ] lookup time。
  - [ ] AP/IoU。

输出:

- [ ] 新表 `DLPCM/table/scalability_cache.tex`。

## 5. 代码整理 TODO

### 当前 repo 可直接整理

- [ ] 把 `src/v2x_forecasting/train_bev.py` 和 `train_bev_compressed.py` 的实验设置与论文 `DLPCM/3_dual.tex` 对齐：
  - [ ] 论文写 batch size 8，脚本默认/训练脚本多处为 16 或 64，需要统一说明。
  - [ ] 论文写 Dice warmup/ramp，但代码目前是 `ep > curriculum_epochs` 后直接加入，需统一。
  - [ ] 论文训练目标只写 BCE + Dice，但代码支持 focal/change/contrast，需要确认最终采用哪个版本。
- [ ] 新增 `scripts/run_vspm_sensitivity.sh`。
- [ ] 新增 `scripts/profile_vspm.py`。
- [ ] 新增 `scripts/summarize_paper_tables.py`。
- [ ] 改进绘图脚本：
  - [ ] label 可配置。
  - [ ] 字体大小适合 IEEE 双栏。
  - [ ] 输出 PDF。

### 需要新增或找回外部代码

- [ ] V2VNet/DiscoNet/SyncNet evaluation pipeline。
- [ ] AP@0.5/AP@0.7 计算脚本。
- [ ] delay simulator。
- [ ] pose noise simulator。
- [ ] packet loss/jitter simulator。
- [ ] cache manager。
- [ ] NLOS/blind spot metric。
- [ ] DAIR-V2X/V2V4Real/OPV2V adapter。

## 6. 逐文件修改清单

### `DLPCM/main.tex`

- [x] 摘要中 `Vehivle-to-Vehicle` 拼写改为 `Vehicle-to-Vehicle`。
- [ ] 摘要中若没有真实数据集/鲁棒性实验，不要写过强 real-world deployment claim。
- [ ] 关键词保留 communication latency、computation latency、prediction、real-time performance。

### `DLPCM/1_introduction.tex`

- [ ] 新增 problem setting。
- [ ] 新增 failure modes。
- [ ] 新增 deployment assumptions。
- [ ] 加入 communication vs computation latency differential impact。
- [ ] 加入 V2X information gain/blind spot 安全价值。
- [ ] 修改 contribution：
  - [ ] Contribution 1 强调 local + remote aligned to `t_result`。
  - [ ] Contribution 2 强调 sender-side vehicle-specific history。
  - [ ] Contribution 3 强调 MST 的 bandwidth + redundant computation。
  - [ ] Contribution 4 如无真实验证，避免写 overly broad adaptability。

### `DLPCM/2_related_work.tex`

- [ ] 改 Table `related work` 为细粒度比较。
- [ ] 每类方法增加最近方法。
- [ ] 最后一段更明确 DLPCM 与 prediction-based compensation 的差异。

### `DLPCM/3_dual.tex`

- [ ] 在 VSPM 小节补 design rationale。
- [ ] 统一术语：`Vehicle-Specific Prediction Model` 不要写成 `Visual Specific Prediction Module`。
- [ ] 加 complexity analysis。
- [ ] 加 cache management。
- [ ] 加 pose uncertainty/spatial calibration 讨论。
- [ ] Algorithm 1 增加 cache eviction 或在 Step 7 明确 TTL/LRU。
- [ ] 确认 Eq. `f_theta_i(h,c)` 是否缺少 decoder input，必要时写成 `f_theta_i(x_{s+k-1}, h, c)` 并说明 free mode 输入为上一预测。

### `DLPCM/4_experiment.tex`

- [ ] Experimental setup 增加硬件、训练 batch、epoch、frame interval、latency 设置。
- [ ] 指标定义分开写：
  - [ ] VSPM: IoU/Dynamic IoU。
  - [ ] Detection: AP@0.5/AP@0.7。
  - [ ] Efficiency: MB/frame, FLOPs, latency, memory。
- [ ] VSPM ablation 加具体表，而不仅是图。
- [ ] MST 小节增加 compression variants。
- [ ] Dual-latency fusion 小节增加 system ablation。
- [ ] 新增 robustness/scalability/cache/NLOS 小节。
- [x] 修正 typo：`DiscnNet` -> `DiscoNet`。

### `DLPCM/5_conclusion.tex`

- [ ] 增加 limitations：
  - [ ] spatial calibration。
  - [ ] real-world hardware jitter。
  - [ ] packet loss。
  - [ ] cache in high-density traffic。
  - [ ] real dataset validation if not completed。
- [ ] Future work 写具体，而不是泛泛说 large-scale heterogeneous cooperation。

## 7. Response Letter TODO

### R1 response

- [ ] R1-1: 说明 Introduction 已重写，列出新增 problem setting/failure modes/deployment assumptions。
- [ ] R1-2: 说明 Related Work 新增细粒度对比表和近期 latency-aware 方法。
- [ ] R1-3: 说明 Method 新增 design rationale，并用 sensitivity analysis 支撑。
- [ ] R1-4: 说明新增 system-level ablation，量化 VSPM/MST/dual-latency fusion。
- [ ] R1-5: 说明图表重绘和 caption/正文 claim 已增强。
- [ ] R1-6: 说明新增真实数据或非理想条件鲁棒性，并讨论 domain gap。
- [ ] R1-7: 说明新增 baseline 或解释 baseline 选择充分性。

### R2 response

- [ ] R2-1: 说明新增 pose noise 实验或 spatial calibration discussion。
- [ ] R2-2: 说明新增 N-agent complexity/scalability。
- [ ] R2-3: 说明新增 communication vs computation latency 分析和图。
- [ ] R2-4: 说明新增 blind spot/NLOS information gain 分析。
- [ ] R2-6: 说明新增 DAIR-V2X/V2V4Real/robustness/embedded runtime。
- [ ] R2-7: 说明新增 VSPM cache policy、TTL/LRU/range eviction 和 profiling。

## 8. 最低提交包

如果时间不够，最低也要完成：

- [ ] Introduction problem setting + failure modes + deployment assumptions。
- [ ] Related Work 细粒度比较表。
- [ ] Method design rationale + complexity/cache discussion。
- [ ] System ablation 表。
- [ ] Communication vs computation latency differential impact 图。
- [ ] MST compression variants 表。
- [ ] Pose noise 或 packet loss/jitter 鲁棒性至少一个。
- [ ] N-agent scalability 至少 runtime/memory profiling。
- [ ] Discussion 限制真实部署、定位噪声、cache。

## 9. 当前不应再强声称

补实验前，正文中避免以下表述：

- [ ] 已充分证明真实道路部署价值。
- [ ] 已对定位噪声鲁棒。
- [ ] 已在高密度 N 车交通中实时可扩展。
- [ ] cache 不会成为瓶颈。
- [ ] 全面优于所有近期 latency-aware 方法。
- [ ] 已解决所有 spatial-temporal alignment 问题。
