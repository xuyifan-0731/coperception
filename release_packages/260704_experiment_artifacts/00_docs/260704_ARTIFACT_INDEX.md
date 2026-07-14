# 260704 实验包索引与复现缺口说明

本文档是给第一次接手本项目的人看的入口索引，目标是让读者能快速理解文件结构、定位结果、找到 checkpoint，并明确从零复现还缺哪些必要条件。

实验包根目录：

```text
release_packages/260704_experiment_artifacts/
```

## 1. 第一次阅读时先看哪些文件

| 目的 | 优先查看 |
|---|---|
| 看最终实验结果 | `00_docs/260704_results.md` |
| 看 Word 版整理报告 | `00_docs/260704_experiment_results.docx` |
| 查公开数据集、外部 baseline 代码、外部 checkpoint 和缺口 | `00_docs/260704_EXTERNAL_RESOURCES.md` |
| 从零准备复现环境 | `00_docs/260704_REPRODUCTION_CHECKLIST.md` |
| 按命令重新训练/测试 | `00_docs/260704_RUNBOOK.md` |
| 理解目录结构和复现缺口 | `00_docs/260704_ARTIFACT_INDEX.md` |
| 使用 patch、split、环境参考文件 | `00_reproduction_assets/README.md` |
| 查每个上传文件的位置、大小、SHA256 | `MANIFEST.csv` |
| 恢复被切分的大 checkpoint | `restore_split_checkpoints.py` 和 `CHECKPOINT_SPLIT_MANIFEST.csv` |

## 2. 当前目录结构是否合理

当前结构是合理的，采用的是：

```text
数据集 -> baseline 方法 -> 文件类型
```

也就是：

```text
release_packages/260704_experiment_artifacts/
  00_docs/             # 结果文档、复现说明、资源说明
  00_reproduction_assets/ # 小型复现材料：patch、split、requirements、info summary
  00_code_scripts/     # 本次实验用到的脚本快照
  00_code_version/     # 打包时的 Git 状态、diff、commit 信息
  01_v2xsim/           # V2X-Sim 数据集相关实验
  02_dair_v2x_c/       # DAIR-V2X-C 数据集相关实验
  03_dair_v2x_seq/     # DAIR-V2X-Seq / SPD 数据集相关实验
```

进入某个数据集后，再按 baseline 分：

```text
dataset/
  baseline/
    checkpoints/
    results/
    logs/
    paper_sources/
```

查找规则很简单：

| 想找什么 | 去哪里找 |
|---|---|
| 实验结果数字 | `results/` |
| 我们方法或 baseline 的模型参数 | `checkpoints/` |
| 真实运行日志 | `logs/` |
| 不是本次重跑、而是来自旧论文表格的结果 | `paper_sources/` |
| 本次使用的脚本 | `00_code_scripts/scripts/` |
| 外部代码本地修改 | `00_reproduction_assets/patches/` |
| DAIR/TraF-Align 划分文件 | `00_reproduction_assets/split_files/` |
| 环境 requirements 参考 | `00_reproduction_assets/env_refs/` |
| 文件校验和来源路径 | `MANIFEST.csv` |

## 3. 按数据集快速定位

### 3.1 V2X-Sim

目录：

```text
01_v2xsim/
  V2VNet/
  DATA/
  DiscoNet/
```

| Baseline | 内容 | 说明 |
|---|---|---|
| `V2VNet` | `paper_sources/`, `checkpoints/ours/` | 主要保存旧 DLPCM 表格来源和我们历史模型引用，不是完整重跑包。 |
| `DiscoNet` | `paper_sources/`, `checkpoints/ours/` | 同上，主要用于结果来源追溯。 |
| `DATA` | `checkpoints/baseline_DATA/`, `checkpoints/ours/`, `results/` | V2X-Sim 上 DATA baseline 与 DATA + ours 的主要可复现结果包。 |

重点文件：

```text
01_v2xsim/DATA/results/v2xsim_data_compensation_ap.csv
01_v2xsim/DATA/results/v2xsim_data_ifam_joint_delay_comp_scale01_20260713.csv
01_v2xsim/DATA/checkpoints/ours/v2xsim_data_ifam_joint_delay_comp_scale01.pth
```

### 3.2 DAIR-V2X-C

目录：

```text
02_dair_v2x_c/
  official_late_fusion/
  DATA/
  LRCP/
```

| Baseline | 内容 | 说明 |
|---|---|---|
| `official_late_fusion` | RF 两帧 ours checkpoint、严格 no-comp / ours / TCLF 结果、运行日志 | DAIR-V2X-C 上官方晚期融合和 detector-side ours 的主要结果。 |
| `DATA` | DATA baseline checkpoint/config、IFAM feature/joint ours checkpoint、结果 CSV | DAIR-V2X-C 上 DATA intermediate fusion 的结果。 |
| `LRCP` | LRCP stage2 checkpoint/config、strict dual-delay 结果和日志 | 本地适配 DAIR-V2X-C 后得到的 LRCP 结果。 |

重点文件：

```text
02_dair_v2x_c/official_late_fusion/results/dair_late_fusion_ours_history2_rf_trainonly_strict_20260714.csv
02_dair_v2x_c/DATA/results/dair_data_ifam_joint_delay_comp_20260713.csv
02_dair_v2x_c/LRCP/results/dair_lrcp_strict_dual_delay_ap.csv
```

### 3.3 DAIR-V2X-Seq / SPD

目录：

```text
03_dair_v2x_seq/
  TraF-Align/
```

| Baseline | 内容 | 说明 |
|---|---|---|
| `TraF-Align` | 官方 TraF-Align checkpoint/config、ours feature T10-N10 predictor、ours joint delay-comp checkpoint/config、结果 CSV | DAIR-V2X-Seq 上 TraF-Align baseline 与 TraF-Align + ours 的对比实验。 |

重点文件：

```text
03_dair_v2x_seq/TraF-Align/results/trafalign_baseline_comp0_spconv2121.csv
03_dair_v2x_seq/TraF-Align/results/trafalign_joint_delay_comp_only_adaptive2to4_bypass_comp0_spconv2121.csv
```

## 4. 大 checkpoint 的恢复方式

GitHub 普通仓库不适合直接上传很大的 `.pth/.pkl`，所以本包把超过阈值的大 checkpoint 拆成了：

```text
*.split/*.part000
*.split/*.part001
...
```

恢复方式：

```bash
cd release_packages/260704_experiment_artifacts
python restore_split_checkpoints.py
```

恢复脚本会按照 `CHECKPOINT_SPLIT_MANIFEST.csv` 合并文件，并校验原始 SHA256。

## 5. 目前已经上传了什么

已经上传的内容足够支持“结果来源追溯”和“多数 checkpoint/结果文件校验”：

| 类型 | 是否已上传 | 说明 |
|---|---:|---|
| 最终结果 CSV | 是 | 存在于各 baseline 的 `results/` 下。 |
| Word 和 Markdown 结果整理 | 是 | 存在于 `00_docs/` 下。 |
| 我们最终使用的大部分 checkpoint | 是 | 大文件用 split part 保存。 |
| baseline checkpoint/config | 部分是 | 本地使用过的 DATA、LRCP、TraF-Align 等关键 checkpoint/config 已放入 package。 |
| 运行日志 | 部分是 | DAIR official late fusion、LRCP 等有日志；部分 V2X-Sim/DATA 结果主要保留 CSV。 |
| 实验脚本快照 | 是 | 存在于 `00_code_scripts/scripts/`。 |
| Git 状态和 diff | 是 | 存在于 `00_code_version/`。 |
| 外部代码 patch | 是 | 存在于 `00_reproduction_assets/patches/`，不包含原始数据或结果目录。 |
| DAIR/TraF-Align split 文件 | 是 | 存在于 `00_reproduction_assets/split_files/`。 |
| 环境 requirements 参考 | 是 | 存在于 `00_reproduction_assets/env_refs/`。 |

## 6. 从零复现还缺什么

目前最大问题不是结果文件缺失，而是“新机器从零跑起来”还缺几个关键条件。

| 缺口 | 是否已有说明 | 是否已上传 | 影响 |
|---|---:|---:|---|
| V2X-Sim 2.0 原始数据集 | 是，公开网址已列出 | 否 | 不能从零重跑 V2X-Sim DATA / V2VNet / DiscoNet。 |
| DAIR-V2X-C 原始数据集 | 是，公开网址已列出 | 否 | 不能从零重跑 DAIR official late fusion、DATA、LRCP。 |
| DAIR-V2X-Seq / SPD 原始数据集 | 是，公开网址已列出 | 否 | 不能从零重跑 TraF-Align。 |
| 原始数据集的精确文件列表和 checksum | 不够完整 | 否 | 无法严格确认别人下载的数据版本与本地完全一致。 |
| V2X-Sim info 文件，如 `datasets/v2xsim2_info/*.pkl` | 有 summary | 否 | `.pkl` 约 226MB，不上传；DATA/OpenCOOD 脚本依赖这些 info 文件，需要本地生成。 |
| DAIR cooperative split JSON 的规范副本 | 是 | 是 | 已放入 `00_reproduction_assets/split_files/dair_v2x/`。 |
| TraF-Align 官方 split YAML 的规范副本 | 是 | 是 | 已放入 `00_reproduction_assets/split_files/trafalign/`。 |
| `external/DATA` 本地修改补丁 | 是 | 是 | 已放入 `00_reproduction_assets/patches/DATA_local_changes.patch`。 |
| `external/LRCP` 本地修改补丁 | 是 | 是 | 已放入 `00_reproduction_assets/patches/LRCP_local_changes.patch`，新增 yaml 在 `extra_files/`。 |
| `external/DAIR-V2X` 本地修改补丁 | 是 | 是 | 已放入 `00_reproduction_assets/patches/OpenDAIRV2X_local_changes.patch`。 |
| TraF-Align 本地修改补丁 | 是 | 是 | 已放入 `00_reproduction_assets/patches/TraFAlign_local_changes.patch`，新增模块在 `extra_files/`。 |
| 每个 baseline 的环境 lock 文件 | 有 requirements 参考 | 部分 | 已放入 requirements/environment 参考，但还不是严格 lock。 |
| 运行路径恢复脚本 | 否 | 否 | 包内路径是归档结构，但很多脚本仍期望历史路径，例如 `checkpoints/curated/ours_models`、`runs/data_dair_official_eval`。 |
| 大型中间 cache | 有生成命令 | 否 | 不是最终结果，但重训 DATA IFAM、TraF-Align feature predictor 时需要重新生成。 |

结论：当前包已经适合做结果审计和 checkpoint 追溯，但还不是完全自包含的一键复现包。

## 7. 最需要补充的复现材料

如果目标是让新来的人直接复现，优先补这几项：

1. 导出完整环境 lock，例如 `conda env export --from-history`、`pip freeze`、CUDA/PyTorch/spconv/cumm/mmcv/mmdet3d 版本。
2. 增加数据集校验文档：压缩包版本、解压后的目录树、关键文件数量、checksum。
3. 补充 V2X-Sim info `.pkl` 的生成脚本或更详细生成说明。
4. 增加一个 `materialize_runtime_paths.py`，把 package 中的 checkpoint/config 复制或软链接到 runbook 里的历史运行路径。
5. 增加一个 `validate_repro_env.py`，运行前检查数据集、代码、checkpoint、环境版本是否齐全。

## 8. 给新人的最短使用路径

新读者建议按这个顺序看：

1. 打开 `00_docs/260704_ARTIFACT_INDEX.md`，先理解目录结构。
2. 打开 `00_docs/260704_REPRODUCTION_CHECKLIST.md`，理解从零复现需要准备什么。
3. 打开 `00_docs/260704_results.md` 或 Word 文档，看最终结果。
4. 根据想看的数据集进入 `01_v2xsim/`、`02_dair_v2x_c/` 或 `03_dair_v2x_seq/`。
5. 根据 baseline 进入对应文件夹，例如 `DATA/`、`LRCP/`、`TraF-Align/`。
6. 只看数字就打开 `results/`；查模型就打开 `checkpoints/`；查运行证据就打开 `logs/`。
7. 如果要复现，先看 `00_docs/260704_EXTERNAL_RESOURCES.md`，再看 `00_reproduction_assets/README.md` 和 `00_docs/260704_RUNBOOK.md`。
8. 如果 checkpoint 是 split 形式，先运行 `python restore_split_checkpoints.py`。
9. 用 `MANIFEST.csv` 核对文件 SHA256，确认文件没有丢失或损坏。
