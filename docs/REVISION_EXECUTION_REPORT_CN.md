# 论文修正执行报告

更新时间：2026-05-30 UTC

## 已完成

1. 生成 DAIR-V2X-C 论文表格：
   - `DLPCM/table/dair_official_baselines.tex`
   - `DLPCM/table/dair_delay_system.tex`
   - `DLPCM/table/dair_mst_tradeoff.tex`
   - `DLPCM/table/dair_sensitivity.tex`
   - `DLPCM/table/dair_robustness.tex`
2. 生成 DAIR-V2X-C 论文图：
   - `DLPCM/pics/dair_official_delay.pdf`
   - `DLPCM/pics/dair_delay_heatmap.pdf`
   - `DLPCM/pics/dair_mst_pareto.pdf`
   - `DLPCM/pics/dair_packet_loss.pdf`
3. 更新论文：
   - `DLPCM/main.tex`
   - `DLPCM/1_introduction.tex`
   - `DLPCM/4_experiment.tex`
   - `DLPCM/5_conclusion.tex`
4. 新增生成脚本：
   - `scripts/make_dair_paper_artifacts.py`
5. 更新审稿回复草稿：
   - `docs/REVIEWER_RESPONSE_DRAFT_CN.md`

## 核心结果

1. DAIR official detection baseline：
   - late fusion + TCLF 的 BEV AP@0.7 从 `k=0` 的 54.11 降到 `k=5` 的 48.71。
   - vehicle-only BEV AP@0.7 为 46.51，说明真实 DAIR 场景中协作信息有收益，同时也受到时延影响。
2. DAIR dual-latency system ablation：
   - `T10_n10` 完整网格：`dual` IoU 0.4097，高于 `no_comp` 0.1945、`comm_only` 0.2960、`comp_only` 0.2784。
   - `T10_n5` 短时延网格：`dual` IoU 0.5207，高于 `no_comp` 0.2857。
3. MST：
   - ConvLSTM state 为 16.0 MiB/agent。
   - DS64-FP16 和 Bottleneck12 为 0.25 MiB/agent，压缩倍率 64x。
   - DS64-FP16 达到 0.6066 IoU 和 0.5888 Dynamic IoU。
4. Sensitivity：
   - `T=5/10/20/30, n=5` 的 IoU 稳定在约 0.5164 到 0.5237。
   - 长 horizon 更难，`T10_n15` IoU 为 0.3065。
5. Robustness：
   - packet loss 增大时 IoU 平滑下降。
   - 所有列入表格的扰动下 Delta IoU 均保持为正。

## 检查结果

本机缺少 `latexmk`、`pdflatex`、`bibtex`、`xelatex` 和 `tectonic`，因此无法生成 PDF。已完成静态检查：

1. `main.tex` 和 `4_experiment.tex` 中所有 `\input{...}` 均存在。
2. `4_experiment.tex` 中所有 `\includegraphics{...}` 均存在。
3. 新增 DAIR 表格和主要章节的 `\begin{...}` / `\end{...}` 配对检查通过。
