# 审稿回复草稿

更新时间：2026-05-30 UTC

## 总体回复

感谢审稿人对实验完整性、真实数据集验证、双时延建模、系统消融和鲁棒性分析提出的建议。我们已根据意见补充 DAIR-V2X-C 真实数据实验，并系统整理通信时延、计算时延、MST 状态传输、超参数敏感性和异常扰动下的结果。修订稿中新增的主要材料已写入 `DLPCM/4_experiment.tex`，新增表格位于 `DLPCM/table/dair_*.tex`，新增图位于 `DLPCM/pics/dair_*.pdf`。

## 对审稿人主要意见的回复

### 1. 关于真实数据集验证不足

回复要点：

- 我们新增 DAIR-V2X-C 实验，补充真实车路协同场景下的验证。
- 新实验分为两层：官方 point-cloud detection baseline 用 AP 证明真实 DAIR 场景中协同感知确实受到时延影响；我们的方法在 DAIR annotation-derived BEV sequence 上验证 VSPM、dual-latency compensation 和 MST 的有效性。
- 修订稿新增 Table `tab:dair-official-baselines` 和 Fig. `fig:dair-official-delay`。

可写入 rebuttal 的结果：

- DAIR 官方 detection baseline 中，late fusion + TCLF 在 `k=0` 时 BEV AP@0.7 为 54.11，`k=5` 时为 48.71，说明真实场景中时延会造成检测性能下降。
- Vehicle-only BEV AP@0.7 为 46.51，late fusion + TCLF 在多个 delay setting 下高于 vehicle-only，说明 V2X 协作信息在真实场景中仍有价值。

### 2. 关于通信时延和计算时延是否都需要建模

回复要点：

- 我们新增 DAIR system ablation，分别比较 `no_comp`、`comm_only`、`comp_only`、`dual` 和 `oracle`。
- `comm_only` 只补偿通信时延，`comp_only` 只补偿计算时延，`dual` 同时补偿二者。
- 修订稿新增 Table `tab:dair-delay-system` 和 Fig. `fig:dair-delay-heatmap`。

可写入 rebuttal 的结果：

- 在完整 `T10_n10` delay grid 上，`dual` 平均 IoU 为 0.4097，高于 `no_comp` 的 0.1945、`comm_only` 的 0.2960 和 `comp_only` 的 0.2784。
- 在短时延 `T10_n5` 设置上，`dual` 平均 IoU 为 0.5207，高于 `no_comp` 的 0.2857，并带来 0.2350 的 IoU 增益。
- 这些结果支持“通信时延与计算时延需要联合补偿”的结论。

### 3. 关于 MST 是否真正降低通信/缓存开销

回复要点：

- 我们新增 MST bandwidth-performance 表和 Pareto 图，报告每个状态传输变体的 state MiB/agent、状态压缩倍率、rollout latency、IoU、Delta IoU 和 Dynamic IoU。
- 修订稿新增 Table `tab:dair-mst-tradeoff` 和 Fig. `fig:dair-mst-pareto`。

可写入 rebuttal 的结果：

- 原始 ConvLSTM hidden+cell 状态为 16.0 MiB/agent。
- DS64-FP16 与 Bottleneck12 均为 0.25 MiB/agent，相当于 64x 状态压缩。
- DS64-FP16 在 DAIR 上达到 0.6066 IoU 和 0.5888 Dynamic IoU，高于 ConvLSTM-state baseline 的 0.5188 IoU 和 0.4646 Dynamic IoU。
- 结果说明 MST 不只是减少字节量，也能保留对预测补偿有用的历史状态信息。

### 4. 关于超参数选择和适用边界

回复要点：

- 我们新增 DAIR sensitivity 表，比较不同历史长度 `T` 和预测跨度 `n`。
- 修订稿新增 Table `tab:dair-sensitivity`。

可写入 rebuttal 的结果：

- 当 `n=5` 时，`T=5/10/20/30` 的 IoU 约为 0.5164 到 0.5237，说明默认历史长度不是脆弱选择。
- 当固定 `T=10` 并增大预测跨度时，`T10_n3` IoU 为 0.5954，`T10_n10` 为 0.3853，`T10_n15` 为 0.3065，体现长时延预测难度增加。
- 我们将在修订稿中明确：`T10_n5` 用于短时延高精度补偿，`T10_n10` 用于覆盖完整通信/计算时延网格。

### 5. 关于真实系统噪声和鲁棒性

回复要点：

- 我们新增 DAIR robustness 实验，覆盖 packet loss、BEV dropout 和 false positive。
- 修订稿新增 Table `tab:dair-robustness` 和 Fig. `fig:dair-packet-loss`。

可写入 rebuttal 的结果：

- `T10_n5` clean IoU 为 0.5478；packet loss 0.3 时为 0.4411，packet loss 0.5 时为 0.3476，呈现平滑退化。
- 即使在 packet loss 0.5 下，`T10_n5` 仍有 0.1351 Delta IoU；`T10_n10` 仍有 0.1203 Delta IoU。
- dropout 和 false positive 下 Delta IoU 也保持为正，说明方法不是不退化，而是在扰动下仍保持相对 copy-last-frame baseline 的收益。

## 修订稿已完成的主要改动

1. `DLPCM/main.tex`：摘要中增加 V2X-Sim 和 DAIR-V2X-C 双数据集验证表述。
2. `DLPCM/1_introduction.tex`：贡献部分增加真实 DAIR-V2X-C 验证。
3. `DLPCM/4_experiment.tex`：新增 DAIR-V2X-C real-world validation 小节，纳入 official detection baseline、delay/system ablation、MST trade-off、sensitivity 和 robustness。
4. `DLPCM/5_conclusion.tex`：结论中加入 DAIR-V2X-C 实验对真实场景有效性的支撑。
5. `scripts/make_dair_paper_artifacts.py`：新增论文表格和图生成脚本，保证结果可复现。
