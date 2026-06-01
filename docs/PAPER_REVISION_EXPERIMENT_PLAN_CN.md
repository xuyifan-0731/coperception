# 论文修正实验总计划

更新时间：2026-05-30 UTC

## 目标

本轮修正需要把论文从“方法在单一模拟数据上有效”推进到“方法在真实协同感知数据、不同系统时延、不同通信预算和异常条件下都有可解释、可复现的收益”。最终论文中应形成四条主要证据链：

1. **真实数据验证**：在 DAIR-V2X-C 上补充官方 detection baseline 和我们方法的 DAIR label-derived BEV 预测补偿结果。
2. **系统时延有效性**：在通信时延和计算时延联合变化时，我们的方法相对 copy-last/no-comp 有稳定增益。
3. **通信效率**：MST/压缩状态在显著降低通信状态量的同时，保持或提升预测质量。
4. **鲁棒性与适用边界**：在丢包、BEV dropout、false positive、不同 T/n 设置下，结果退化趋势合理，并能解释方法的边界。

## 当前已完成的实验资产

1. DAIR-V2X full raw 数据已下载、解压、检查通过。
2. 官方 OpenDAIRV2X point-cloud baselines 已完成并解析：
   - `veh_only_k0`, `inf_only_k0`
   - `late_fusion_tclf_k0..k5`
   - `late_fusion_no_comp_k1..k5`
   - `early_fusion_k0..k2`
   - 汇总：`results/dair_v2x/official_baselines/summary_partial.csv`
3. 我们方法 DAIR label-derived BEV 结果已完成：
   - `T10_n5` 主模型和短时延 system ablation。
   - `T10_n10` 完整 V2X-Sim 风格时延网格 system ablation。
   - 汇总：`results/dair_v2x/tables/delay_system_ablation_table.csv`
4. DAIR MST 消融已完成：
   - `baseline`, `gru_fp16`, `ds64_fp16`, `bottleneck12`
   - 汇总：`results/dair_v2x/tables/training_summary_table.csv`
5. DAIR sensitivity 已完成：
   - `T5_n5`, `T10_n5`, `T20_n5`, `T30_n5`
   - `T10_n3`, `T10_n10`, `T10_n15`
6. DAIR robustness 已完成：
   - `T10_n5 ckpt16000`
   - `T10_n10 ckpt15000`
   - 条件：clean、packet loss、dropout、false positive。
   - 汇总：`results/dair_v2x/tables/robustness_table.csv`
7. 表格草稿已生成：
   - `results/dair_v2x/tables/dair_revision_tables.tex`

## 2026-05-30 执行结果

已将上述实验资产整理进论文修订稿：

1. 新增生成脚本 `scripts/make_dair_paper_artifacts.py`，从现有 CSV 自动生成论文表格和图。
2. 新增论文表格：
   - `DLPCM/table/dair_official_baselines.tex`
   - `DLPCM/table/dair_delay_system.tex`
   - `DLPCM/table/dair_mst_tradeoff.tex`
   - `DLPCM/table/dair_sensitivity.tex`
   - `DLPCM/table/dair_robustness.tex`
3. 新增论文图：
   - `DLPCM/pics/dair_official_delay.pdf`
   - `DLPCM/pics/dair_delay_heatmap.pdf`
   - `DLPCM/pics/dair_mst_pareto.pdf`
   - `DLPCM/pics/dair_packet_loss.pdf`
4. 已更新论文正文：
   - `DLPCM/main.tex`
   - `DLPCM/1_introduction.tex`
   - `DLPCM/4_experiment.tex`
   - `DLPCM/5_conclusion.tex`
5. 已更新审稿回复草稿：
   - `docs/REVIEWER_RESPONSE_DRAFT_CN.md`
6. 已新增执行报告：
   - `docs/REVISION_EXECUTION_REPORT_CN.md`

## 需要纳入论文的实验

### 实验 A：官方 DAIR-V2X Detection Baselines

目的：回应“缺少真实数据集验证”和“需要和标准协同感知 baseline 对齐”。

已完成设置：

1. `veh_only_k0`：车辆单端，无通信。
2. `inf_only_k0`：路侧单端。
3. `late_fusion_tclf_k0..k5`：late fusion + time compensation。
4. `late_fusion_no_comp_k1..k5`：late fusion without compensation。
5. `early_fusion_k0..k5`：early fusion，高通信量官方参考 baseline。

需要分析的结果：

1. AP 随 k 增大的退化曲线。
2. `late_fusion_tclf` 相对 `late_fusion_no_comp` 的 AP 增益。
3. `early_fusion` 的 AP/通信量变化；只作为高通信量参考，不作为严格性能上界。
4. `veh_only` 和 `inf_only` 作为无协作/单端基线。

论文呈现方式：

1. 主文表格放 compact 结果：3D AP@0.5、3D AP@0.7、BEV AP@0.5、BEV AP@0.7、通信量。
2. 主文曲线或 appendix 放 k=0..5 的 AP-vs-delay。
3. 文字结论不要声称官方 baseline 就是我们方法；它用于证明 DAIR 数据和时延补偿问题真实存在。

验收标准：

1. 所有官方 log 均 parse 为 `complete`。
2. k 越大，AP 总体下降；time compensation 与 no-comp 的差异方向可解释。
3. 通信量排序合理：`veh_only` 最低，`early_fusion` 最高，late fusion 居中。

### 实验 B：我们方法在 DAIR 上的时延补偿

目的：回应“仅在 V2X-Sim 上验证不够”和“通信/计算双时延设置不充分”。

已完成设置：

1. `T10_n5 ckpt16000`：短时延网格，覆盖总 horizon <= 4。
2. `T10_n10 ckpt15000`：完整 latency grid，覆盖通信时延 `0..5`、计算时延 `0..4`，总 horizon <= 9。
3. system modes：
   - `no_comp`
   - `comm_only`
   - `comp_only`
   - `dual`
   - `oracle`

需要分析的结果：

1. `dual` 是否显著高于 `no_comp/copy-last`。
2. 在完整 `T10_n10` grid 上，`dual` 是否高于 `comm_only` 和 `comp_only`，证明联合时延建模必要。
3. `oracle` 与 `dual` 的差距，用于说明剩余改进空间。
4. `T10_n5` 和 `T10_n10` 的差异：
   - `T10_n5` 更适合短 horizon，IoU 高。
   - `T10_n10` 能覆盖完整时延网格，但长期预测难度更高。

论文呈现方式：

1. 主文放 system ablation 聚合表：每种 mode 的 mean IoU、copy-last IoU、Delta IoU、Dynamic IoU。
2. appendix 放完整 delay grid CSV 转 heatmap。
3. 与 V2X-Sim 中原有 DLPCM 表格使用一致的通信/计算时延定义。

验收标准：

1. `dual` 在 `T10_n5` 和 `T10_n10` 上都高于 `no_comp`。
2. `T10_n10 dual` 高于 `comm_only/comp_only`，至少在平均 IoU 和 Dynamic IoU 上成立。
3. 若长期 horizon 上绝对 IoU 下降，应明确解释为 horizon 难度，而不是方法失效。

### 实验 C：MST 通信效率和性能消融

目的：回应“MST 是否真正降低通信负载，以及是否牺牲性能”。

已完成设置：

1. `baseline`
2. `gru_fp16`
3. `ds64_fp16`
4. `bottleneck12`

需要补齐或确认的分析：

1. 每个 variant 的 best IoU、Delta IoU、Dynamic IoU。
2. state MiB/agent：
   - ConvLSTM hidden+cell FP32 baseline。
   - GRU FP16。
   - DS64 FP16。
   - bottleneck12。
3. rollout latency 或至少 metadata/profile latency。
4. bandwidth-performance Pareto 表：state size 降低倍率 vs IoU。

论文呈现方式：

1. 主文放 MST 表：Variant、state MiB/agent、compression ratio、IoU、Delta IoU、Dynamic IoU。
2. 画 Pareto 图：x 轴通信状态量，y 轴 IoU 或 Dynamic IoU。
3. 强调 MST 是“状态传输/缓存开销”的优化，而不是仅靠减少检测结果字节。

验收标准：

1. `ds64_fp16` 和 `bottleneck12` 的通信状态量显著小于 baseline。
2. 性能不低于 baseline，或者在性能略降时能用通信收益解释 trade-off。
3. 当前 DAIR 结果中 `ds64_fp16`、`bottleneck12` 优于 baseline，应作为强结果主推；但要避免过度宣称所有压缩形式都必然提升。

### 实验 D：Sensitivity 分析

目的：回应“模型设计超参数是否任意，以及方法适用边界是什么”。

已完成设置：

1. 固定 `n=5`，比较 `T=5/10/20/30`。
2. 固定 `T=10`，比较 `n=3/5/10/15`。

需要分析的结果：

1. `T` 变化：检查历史长度增加是否带来稳定收益，或收益是否趋于饱和。
2. `n` 变化：检查预测跨度变长后性能下降是否合理。
3. `T10_n3` 当前表现最好，说明短 horizon 更容易预测。
4. `T10_n15` 表现明显低，应解释为长 horizon + sparse BEV 监督难度，而不是方法核心失败。

论文呈现方式：

1. 主文或 appendix 放 sensitivity 表。
2. 主文仅报告关键结论：默认设置不是任意选择，而是在延迟覆盖范围和预测难度之间折中。
3. 不建议把 `T10_n15` 作为主结果；它用于边界分析。

验收标准：

1. 结果趋势可解释。
2. 默认设置 `T10_n5` 或完整网格设置 `T10_n10` 的选择有实验依据。
3. 不用 sensitivity 证明“越大越好”，而是证明方法在合理范围内稳定。

### 实验 E：Robustness 分析

目的：回应“真实系统中通信和感知噪声不理想”。

已完成设置：

1. Packet loss：0.1、0.2、0.3、0.5。
2. BEV dropout：0.05、0.1、0.2。
3. False positive：0.001、0.005。
4. `T10_n5` 和 `T10_n10` 两个 checkpoint。

需要分析的结果：

1. clean 到 packet loss 的单调退化。
2. dropout/false positive 下 Delta IoU 是否仍为正。
3. `T10_n10` 在长期 horizon 下绝对 IoU 更低，但相对 copy-last 仍有正收益。
4. 高丢包下性能下降的程度是否与实际系统直觉一致。

论文呈现方式：

1. 主文放简化 robustness 表：clean、packet loss 0.3/0.5、dropout 0.2、false positive 0.005。
2. 完整结果放 appendix。
3. 强调“鲁棒性不是不退化，而是退化平滑且仍优于 copy-last baseline”。

验收标准：

1. 所有扰动下 mean_delta_iou 保持为正。
2. packet loss 增大时 mean_iou 下降。
3. 长 horizon 和短 horizon 都能支撑同一结论。

### 实验 F：V2X-Sim 原实验结果复核

目的：保证修正稿中的新 DAIR 结果与原 DLPCM/V2X-Sim 结果在设置和表述上统一。

需要完成：

1. 复核 `results/revision_tables/` 中的 V2X-Sim 表格。
2. 确认 latency grid 与 DAIR 使用同一 `comm_delay_frames` / `comp_delay_frames` 定义。
3. 确认 V2X-Sim robustness 与 DAIR robustness 条件命名一致。
4. 若论文主文篇幅有限，V2X-Sim 作为原主结果，DAIR 作为真实数据补充；不要让两个数据集的指标混用。

验收标准：

1. V2X-Sim 与 DAIR 的实验符号、delay 定义一致。
2. 同一表述下不把 V2X-Sim detection AP 和 DAIR BEV IoU 直接比较。

## 需要分析的最终结果表/图

1. **Table 1：DAIR official detection baselines**
   - 来源：`results/dair_v2x/tables/official_baselines_table.csv`
   - 分析重点：official AP、通信量、时延退化。

2. **Table 2：DAIR delay/system ablation for our method**
   - 来源：`results/dair_v2x/tables/delay_system_ablation_table.csv`
   - 分析重点：`T10_n10` 上 dual > no_comp/comm_only/comp_only；`T10_n5` 用于短时延收益确认。

3. **Table 3：MST bandwidth-performance**
   - 来源：`results/dair_v2x/tables/training_summary_table.csv` + profile/scalability CSV。
   - 分析重点：通信状态量下降与 IoU/Dynamic IoU 保持。

4. **Table 4：Sensitivity**
   - 来源：`results/dair_v2x/tables/training_summary_table.csv`
   - 分析重点：T/n 的合理范围和边界。

5. **Table 5：Robustness**
   - 来源：`results/dair_v2x/tables/robustness_table.csv`
   - 分析重点：扰动下仍优于 copy-last，退化趋势合理。

6. **Figure 1：AP vs Delay on DAIR official baselines**
   - late TCLF vs no-comp，k=1..5。

7. **Figure 2：Delay heatmap for our method**
   - T10_n10 dual mode，通信/计算二维网格。

8. **Figure 3：MST Pareto**
   - state MiB/agent vs IoU/Dynamic IoU。

## 如何改进，使结果最终合理成立

### 1. 统一指标和实验叙事

改进方向：

1. 官方 DAIR 使用 detection AP；我们方法 DAIR 使用 label-derived BEV forecasting IoU。
2. 不能直接说我们的 BEV IoU 超过官方 detection AP。
3. 应写成两层验证：
   - 官方 baseline 证明真实 DAIR 场景中时延补偿有 detection 层收益。
   - 我们方法证明预测式补偿在同一 DAIR 数据上能改善 BEV 时序状态。

验收：

1. 每张表的指标单位清楚。
2. AP 表和 IoU 表分开。
3. 文中所有 cross-dataset/cross-metric 比较都只讨论趋势，不讨论绝对数值优劣。

### 2. 处理 `T10_n10` 绝对 IoU 较低的问题

现象：

1. `T10_n10` 覆盖完整 latency grid，但平均 IoU 低于 `T10_n5`。
2. 原因是长期 horizon 更难，且 DAIR label-derived BEV 较稀疏。

改进方向：

1. 主结果同时报告：
   - `T10_n5`：短时延高质量预测。
   - `T10_n10`：完整时延覆盖。
2. 主结论强调相对 copy-last 的 Delta IoU，而不是只看绝对 IoU。
3. heatmap 中标出 horizon 越大越难，避免读者误解。

可选补充实验：

1. 对 `T10_n10` 用 `T10_n5` best checkpoint 的短 horizon 子集做对照。
2. 若时间允许，训练 `T10_n10` 的更长 epoch 或尝试 weighted loss，观察长期 horizon 是否改善。

验收：

1. `T10_n10 dual` 平均 Delta IoU 为正。
2. `T10_n10 dual` 高于 `comm_only/comp_only/no_comp`。
3. 文中明确说明其用途是完整 grid 覆盖。

### 3. 处理 `T10_n15` sensitivity 边界

现象：

1. `T10_n15` 明显低于其他设置。

改进方向：

1. 不把 `T10_n15` 当作主模型。
2. 在 sensitivity 讨论中解释：
   - 长 horizon 监督更难。
   - 当前网络容量和 sparse occupancy 标签不适合过长 rollout。
   - 本文目标是补偿实际通信/计算时延，主设置覆盖所需时延即可。

可选补充：

1. 若审稿人质疑，可以补跑 `T10_n15` smaller LR 或 curriculum，但这不是主结论必要条件。

验收：

1. sensitivity 表中默认设置选择有依据。
2. 长 horizon 失败不影响主要 delay grid 结论。

### 4. 强化 MST 的“通信收益”证明

改进方向：

1. 生成最终 MST 表时必须含：
   - state MiB/agent
   - compression ratio
   - IoU
   - Dynamic IoU
   - optional latency
2. 说明 MST 不是简单降低检测框字节，而是降低时序状态传输和缓存。
3. 使用 `ds64_fp16` 或 `bottleneck12` 作为主推荐配置；`gru_fp16` 作为中间配置。

验收：

1. 至少一个 MST variant 同时满足更小 state 和更高 IoU。
2. Pareto 图中推荐点清楚。

### 5. 强化 robustness 的结论边界

改进方向：

1. 不宣称模型对噪声“不敏感”。
2. 改写为：
   - 扰动越强，IoU 合理下降。
   - 但相对 copy-last 仍保持正收益。
3. 高丢包结果用来说明 graceful degradation。

验收：

1. 所有 robustness 条件下 Delta IoU 为正。
2. packet loss 曲线单调或近似单调下降。

### 6. 结果复现和可信度检查

需要执行：

1. 所有表格生成脚本可重复运行。
2. 所有 checkpoint 路径、log 路径、CSV 路径写入 `docs/ACTIVE_WORK_LOG.md`。
3. 最终论文表格数值从 CSV 自动生成，避免手抄错误。

可选增强：

1. 对关键 DAIR 模型补 2 个 random seed：
   - `T10_n5`
   - `ds64_fp16`
2. 如果时间不足，至少在文中说明 DAIR 训练成本，并把现有单 seed 作为真实数据补充。

验收：

1. `python scripts/parse_dair_official_baseline_logs.py ...` 可复现官方表。
2. `python scripts/collect_results.py ...` 可复现训练汇总。
3. `python scripts/make_dair_revision_tables.py` 可复现最终表格。

## 后续执行顺序

### 第一优先级：结果固化

1. 重新运行官方 parser。
2. 重新运行训练 summary collector。
3. 重新运行 `make_dair_revision_tables.py`。
4. 检查 `dair_revision_tables.tex` 是否包含全部最新结果。

### 第二优先级：图表生成

1. 写/运行 DAIR official AP-vs-delay plot。
2. 写/运行 delay heatmap plot。
3. 写/运行 MST Pareto plot。
4. 写/运行 robustness plot。

### 第三优先级：论文文字更新

1. 修改实验设置：
   - DAIR 数据集说明。
   - label-derived BEV 生成方式。
   - delay grid 定义。
   - MST state-size 定义。
2. 修改实验结果：
   - 官方 DAIR baseline 表。
   - 我们方法 DAIR delay/system ablation。
   - MST 表。
   - robustness/sensitivity。
3. 修改 limitation：
   - DAIR 上我们的预测模型使用 label-derived BEV 而非直接 detection AP。
   - 长 horizon 性能下降是已观察边界。

### 第四优先级：审稿回复

1. Reviewer 1：
   - 真实数据验证。
   - 更多 baseline。
   - 时延设置。
2. Reviewer 2：
   - 系统级消融。
   - 通信效率。
   - 鲁棒性和边界。
3. 每条回复都引用具体表/图，而不是泛泛描述。

## 最终交付清单

1. `results/dair_v2x/official_baselines/summary_partial.csv`
2. `results/dair_v2x/summary/training_summary.csv`
3. `results/dair_v2x/tables/dair_revision_tables.tex`
4. DAIR official AP-vs-delay figure。
5. DAIR delay heatmap figure。
6. MST Pareto figure。
7. Robustness figure。
8. 更新后的 paper experiment section。
9. 更新后的 reviewer response。
