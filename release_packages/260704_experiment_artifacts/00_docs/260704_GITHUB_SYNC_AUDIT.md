# 260704 GitHub Sync Audit

更新日期：2026-07-28

本文档记录当前 `260704结果.md` 中涉及的可复现材料与 GitHub 归档包之间的同步状态。归档包路径为：

```text
release_packages/260704_experiment_artifacts/
```

## 1. 本次补齐的内容

本轮已按“数据集 -> baseline -> 文件类型”的结构补齐以下材料：

| 类别 | 已同步内容 |
|---|---|
| 最终结果文档 | 当前最新版 `260704结果.md` 同步为 `00_docs/260704_results.md`。 |
| 训练/评测脚本 | `scripts/` 下当前 260704 涉及的 DAIR、DATA、LRCP、TraF-Align、V2X-Sim 训练与评测脚本同步到 `00_code_scripts/scripts/`。 |
| V2X-Sim / DATA 结果 | 补齐 `Comp=1/3/5`、joint IFAM residual、保守 scale、fairness audit 等 CSV。 |
| DAIR-V2X-C / official late fusion | 补齐 no-comp、strict RF、train+val 诊断、TCLF receiver-target 的 `Comp=1/3/5` CSV。 |
| DAIR-V2X-C / DATA | 补齐 IFAM feature T10-N10、joint delay-comp、解冻 DATA 后段、低计算时延加权诊断、box-level/RF assist 诊断等 CSV。 |
| DAIR-V2X-C / LRCP | 保留 strict dual-delay 合并结果、single-delay 参考结果和分片结果目录。 |
| DAIR-V2X-Seq / TraF-Align | 补齐 baseline、feature T10-N10、joint-only、adaptive scale、Comp0/4/5 bypass、Comp0/3/4/5 bypass、解冻范围 key-cell 诊断等 CSV。 |
| Checkpoint | 补齐当前主表和诊断表引用的 ours checkpoint、baseline checkpoint、config、train metrics；大文件使用 `.split/part` 保存并登记到 `CHECKPOINT_SPLIT_MANIFEST.csv`。 |
| 运行路径恢复 | 新增 `materialize_runtime_paths.py`，用于把归档包中的 checkpoint/config 恢复到脚本默认查找的历史路径。 |

## 2. 已上传或可恢复的关键 checkpoint

| 数据集 / Baseline | Checkpoint | 归档包位置 |
|---|---|---|
| V2X-Sim / V2VNet、DiscoNet | `v2xsim_ours_prediction_model.pth` | `01_v2xsim/*/checkpoints/ours/` |
| V2X-Sim / DATA | DATA baseline checkpoint | `01_v2xsim/DATA/checkpoints/baseline_DATA/*.split/` |
| V2X-Sim / DATA | `v2xsim_data_ifam_joint_delay_comp_scale01.pth` | `01_v2xsim/DATA/checkpoints/ours/` |
| DAIR-V2X-C / official late fusion | strict RF train-only | `02_dair_v2x_c/official_late_fusion/checkpoints/ours_rf2/*.split/` |
| DAIR-V2X-C / official late fusion | train+val diagnostic RF、旧 RF 泄漏诊断 | `02_dair_v2x_c/official_late_fusion/checkpoints/ours_rf2/*.split/` |
| DAIR-V2X-C / DATA | local DATA baseline checkpoint/config | `02_dair_v2x_c/DATA/checkpoints/baseline_DATA/` |
| DAIR-V2X-C / DATA | IFAM feature T10-N10、joint delay-comp | `02_dair_v2x_c/DATA/checkpoints/ours/` |
| DAIR-V2X-C / DATA | 解冻 DATA 后段、低计算时延加权诊断 | `02_dair_v2x_c/DATA/checkpoints/ours/*.split/` |
| DAIR-V2X-C / LRCP | LRCP stage2 checkpoint/config | `02_dair_v2x_c/LRCP/checkpoints/baseline_LRCP_stage2/` |
| DAIR-V2X-Seq / TraF-Align | official checkpoint/config | `03_dair_v2x_seq/TraF-Align/checkpoints/baseline_official_trafalign/` |
| DAIR-V2X-Seq / TraF-Align | feature T10-N10 predictor | `03_dair_v2x_seq/TraF-Align/checkpoints/ours_feature_t10n10/` |
| DAIR-V2X-Seq / TraF-Align | joint-only、adaptive2to4、Comp0 bypass | `03_dair_v2x_seq/TraF-Align/checkpoints/ours_joint_delay_comp/` |
| DAIR-V2X-Seq / TraF-Align | 不同解冻范围诊断模型 | `03_dair_v2x_seq/TraF-Align/checkpoints/diagnostic_unfreeze/` |

恢复方式：

```bash
cd release_packages/260704_experiment_artifacts
python restore_split_checkpoints.py
python materialize_runtime_paths.py --repo-root /path/to/coperception --traf-root /tmp/TraF-Align_partial --mode symlink
```

如果不希望使用软链接，可改为：

```bash
python materialize_runtime_paths.py --repo-root /path/to/coperception --traf-root /tmp/TraF-Align_partial --mode copy
```

## 3. 没有上传的大型数据和 cache

以下内容影响从零训练或完整重跑，但不适合上传到 GitHub：

| 未上传内容 | 本地规模 / 原因 | 复现方式 |
|---|---:|---|
| V2X-Sim 2.0 原始数据 | 外部公开数据集 | 按 `260704_EXTERNAL_RESOURCES.md` 下载。 |
| DAIR-V2X-C 原始数据 | 外部公开数据集 | 按 `260704_EXTERNAL_RESOURCES.md` 下载。 |
| DAIR-V2X-Seq / SPD 原始数据 | 外部公开数据集 | 按 `260704_EXTERNAL_RESOURCES.md` 下载。 |
| `results/260615_fill/cache/dair_data_feature_t10n10/` | 约 332GB | 用 `scripts/run_dair_data_feature_t10n10_comp.py` 重新生成。 |
| `results/dair_v2x_seq/trafalign_feature_cache_spconv2121/` | 约 32GB | 用 `scripts/train_trafalign_feature_t10n10_predictor.py` 重新生成。 |
| `results/260615_fill/cache/dair_data_rf_ours_20260706/` | 约 64MB | 用 DATA box/RF assist 脚本重新生成。 |
| `results/trafalign_dair_v2x_c_compat/` | 兼容 smoke 目录较大，非正式主实验 | 只上传关键 CSV 和配置；如需完整 smoke 过程，重新运行 `scripts/run_trafalign_dair_compat_eval.py`。 |
| external baseline 源码仓库完整目录 | 第三方代码，体积大且应从 upstream 获取 | 按 `260704_REPRODUCTION_CHECKLIST.md` clone 指定 commit 并应用 patch。 |

## 4. 当前仍需外部准备的条件

1. 原始数据集必须自行从公开源下载。
2. DATA、LRCP、OpenDAIRV2X、TraF-Align 代码需要按文档 checkout 指定 commit。
3. 环境不是完整 lock；已有 requirements 和关键版本说明，但严格复现还应记录本地 CUDA、PyTorch、mmcv、mmdet3d、spconv/cumm 版本。
4. 大型中间 cache 不随 GitHub 上传，训练前需要重新生成。

## 5. 核查输出

本轮生成了引用路径审计表：

```text
results/summary/260704_referenced_artifact_audit.csv
release_packages/260704_experiment_artifacts/00_docs/260704_referenced_artifact_audit.csv
```

其中记录了 `260704结果.md` 中引用路径的存在性、文件大小、是否已被 Git 跟踪或是否被 `.gitignore` 忽略。该表主要用于后续排查缺口；最终 GitHub 上传以 release package 的 `MANIFEST.csv` 和 `CHECKPOINT_SPLIT_MANIFEST.csv` 为准。
