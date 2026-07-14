# 260704 新人复现 Checklist

本文档用于指导第一次接手项目的人，在不上传原始大数据的前提下，从 GitHub 仓库和公开资源把 260704 实验环境准备起来。

## 0. 先理解这件事

当前 GitHub 上传的是“结果归档 + 关键 checkpoint + 复现说明 + 小型复现材料”，不是包含所有原始数据和大型 cache 的一键运行镜像。

已经上传：

- 最终结果表、Word 报告、Markdown 结果文件。
- 关键 baseline checkpoint 和 ours checkpoint；大文件以 split part 形式保存。
- 本次实验脚本快照。
- 外部 baseline 的本地代码 patch。
- DAIR/TraF-Align 的 split 文件。
- 环境 requirements 参考。

没有上传：

- V2X-Sim、DAIR-V2X-C、DAIR-V2X-Seq/SPD 原始数据集。
- 大型特征 cache、box cache、V2X-Sim info `.pkl`。
- 完整 conda/pip lock；目前只有 requirements 参考和关键版本说明。

## 1. 建议阅读顺序

1. `00_docs/260704_ARTIFACT_INDEX.md`：先看目录结构和每类文件含义。
2. `00_docs/260704_results.md`：看最终结果。
3. `00_docs/260704_EXTERNAL_RESOURCES.md`：看数据集、外部代码、checkpoint、环境要求。
4. `00_reproduction_assets/README.md`：看补充 patch、split、requirements 这些小材料怎么用。
5. `00_docs/260704_RUNBOOK.md`：按数据集和 baseline 找训练/测试命令。
6. `MANIFEST.csv`：确认文件是否存在、大小是否正确、SHA256 是否一致。

## 2. 下载原始数据集

原始数据集不随 GitHub 上传，需要从公开源下载。

| 数据集 | 用于哪些实验 | 下载/说明入口 | 本地建议路径 |
|---|---|---|---|
| V2X-Sim 2.0 | V2X-Sim / DATA、V2VNet、DiscoNet | `https://ai4ce.github.io/V2X-Sim/download.html` | `datasets/V2X-Sim-2.0/` |
| DAIR-V2X-C | DAIR official late fusion、DATA、LRCP | `https://thudair.baai.ac.cn/index`；代码和说明：`https://github.com/AIR-THU/DAIR-V2X` | `datasets/DAIR-V2X/cooperative-vehicle-infrastructure/` |
| DAIR-V2X-Seq / V2X-Seq-SPD | TraF-Align | `https://thudair.baai.ac.cn/index`；代码和说明：`https://github.com/AIR-THU/DAIR-V2X-Seq` | `datasets/DAIR-V2X-Seq/V2X-Seq-SPD/` |

下载后建议自己生成一个本地文件列表：

```bash
find datasets/V2X-Sim-2.0 -type f | sort > local_v2xsim_files.txt
find datasets/DAIR-V2X/cooperative-vehicle-infrastructure -type f | sort > local_dair_v2x_c_files.txt
find datasets/DAIR-V2X-Seq/V2X-Seq-SPD -type f | sort > local_dair_v2x_seq_files.txt
```

如果要做严格复现，还应记录每个压缩包或关键文件的 SHA256。

## 3. 准备外部 baseline 代码

外部 baseline 代码不完整 vendoring 到本仓库，需要从 upstream 克隆，再应用本包里的 patch。

| 代码 | upstream | 本地实验用 commit | patch 位置 |
|---|---|---|---|
| DATA | `https://github.com/ChengchangTian/DATA` | `5df7eb6f5659db0d6809fa3cc218aa425bc287b4` | `00_reproduction_assets/patches/DATA_local_changes.patch` |
| LRCP | `https://github.com/JesseWong333/LRCP` | `1eb3793b9befd9d36dd8705e461d895a41b39d33` | `00_reproduction_assets/patches/LRCP_local_changes.patch` |
| OpenDAIRV2X / DAIR-V2X | `https://github.com/AIR-THU/DAIR-V2X` | `c885c54af0c34bc515fa9ca8b5e8fda76a15462c` | `00_reproduction_assets/patches/OpenDAIRV2X_local_changes.patch` |
| TraF-Align | `https://github.com/zhyingS/TraF-Align` | `0f6f98de60dcb86f29f4eb99ed871054b05bf280` | `00_reproduction_assets/patches/TraFAlign_local_changes.patch` |

示例：

```bash
git clone https://github.com/ChengchangTian/DATA external/DATA
cd external/DATA
git checkout 5df7eb6f5659db0d6809fa3cc218aa425bc287b4
git apply ../../release_packages/260704_experiment_artifacts/00_reproduction_assets/patches/DATA_local_changes.patch
```

LRCP 和 TraF-Align 还有新增文件，不只需要 `git apply`：

```bash
# LRCP 新增 DAIR 配置
cp -r release_packages/260704_experiment_artifacts/00_reproduction_assets/patches/extra_files/LRCP/opencood/hypes_yaml/dairv2x \
  external/LRCP/opencood/hypes_yaml/

# TraF-Align 新增 joint delay compensation 模块
cp release_packages/260704_experiment_artifacts/00_reproduction_assets/patches/extra_files/TraF-Align/models/modules/deform/joint_delay_compensation.py \
  /tmp/TraF-Align_partial/models/modules/deform/
```

## 4. 放置 split 文件和 info 文件

DAIR 和 TraF-Align 的 split 文件已经作为小文件上传：

```text
00_reproduction_assets/split_files/dair_v2x/
00_reproduction_assets/split_files/trafalign/
```

建议复制到对应 external 代码目录：

```bash
cp release_packages/260704_experiment_artifacts/00_reproduction_assets/split_files/dair_v2x/*.json \
  external/DAIR-V2X/data/split_datas/

cp release_packages/260704_experiment_artifacts/00_reproduction_assets/split_files/trafalign/V2XSeq_dataset_split_official.yaml \
  /tmp/TraF-Align_partial/datasets/Basedataset/
```

V2X-Sim 的 `datasets/v2xsim2_info/*.pkl` 没有上传，因为约 226MB。当前只上传了 summary：

```text
00_reproduction_assets/dataset_info/v2xsim2_info/summary.json
```

本地实验使用的 V2X-Sim info 统计是：

| split | scene_count | sample_count |
|---|---:|---:|
| train | 80 | 8000 |
| val | 10 | 1000 |
| test | 10 | 1000 |

如果重新生成 info 文件，应确保这三个数量一致。

## 5. 恢复 checkpoint

大 checkpoint 已经拆分上传。复现前必须先恢复：

```bash
cd release_packages/260704_experiment_artifacts
python restore_split_checkpoints.py
cd ../..
```

恢复后用 `CHECKPOINT_SPLIT_MANIFEST.csv` 和 `MANIFEST.csv` 核对 SHA256。

## 6. 准备环境

环境不能只看根目录 `requirements.txt`，不同 baseline 要分开建环境。

参考文件已放在：

```text
00_reproduction_assets/env_refs/DATA/requirements.txt
00_reproduction_assets/env_refs/LRCP/requirements.txt
00_reproduction_assets/env_refs/LRCP/environment.yml
00_reproduction_assets/env_refs/TraF-Align/requirements.txt
```

关键提醒：

- DATA 和 LRCP 都基于 OpenCOOD 风格代码，需要 `python setup.py develop`，并编译 `opencood/utils/setup.py` 和 `opencood/pcdet_utils/setup.py`。
- LRCP 还需要编译 `opencood/utils/ms_deform_attn_ops`。
- TraF-Align 对 `spconv/cumm` 非常敏感。本地成功复现使用的是 `spconv-cu113==2.1.21` 和 `cumm-cu113==0.2.9`，并通过 `PYTHONPATH=/tmp/trafalign_spconv_2121:$PYTHONPATH` 指定。
- OpenDAIRV2X late fusion 依赖 mmdet3d/mmcv/torch/cuda 组合，需要按原 OpenDAIRV2X 环境准备，再应用本地 patch。

## 7. 大型 cache 的处理原则

大型中间 cache 不上传，只重新生成。

| cache | 用途 | 生成命令入口 |
|---|---|---|
| DAIR DATA IFAM feature cache | DAIR DATA feature T10-N10 predictor 和 joint delay-comp | `scripts/run_dair_data_feature_t10n10_comp.py cache-features` |
| TraF-Align feature cache | TraF-Align feature T10-N10 predictor | `scripts/train_trafalign_feature_t10n10_predictor.py cache-features` |
| DAIR DATA box/intermediate cache | DATA + RF assist 诊断 | `scripts/run_dair_data_hybrid_rf_assist.py --cache-only` 或 `scripts/run_dair_data_rf_ours_box_comp_grid.py --force-cache` |
| LRCP DAIR flow offsets | LRCP DAIR 适配 | `scripts/generate_dair_lrcp_flow_offsets.py` |

## 8. 最小复现路径

如果只是验证已有结果文件：

1. 克隆本仓库并切到 `workspace-upload-20260622`。
2. 打开 `release_packages/260704_experiment_artifacts/00_docs/260704_results.md`。
3. 用 `MANIFEST.csv` 校验结果 CSV 和 checkpoint split parts。
4. 不需要下载原始数据集。

如果要重新跑测试：

1. 下载对应原始数据集。
2. 克隆 external baseline。
3. checkout 到表中 commit。
4. 应用 `00_reproduction_assets/patches/`。
5. 放置 split 文件。
6. 恢复 checkpoint。
7. 按 `00_docs/260704_RUNBOOK.md` 运行对应 baseline 的命令。

如果要重新训练 ours：

1. 先完成“重新跑测试”的所有准备。
2. 生成必要 feature/box cache。
3. 运行 runbook 中对应 train 命令。
4. 再运行 eval-grid 或 strict dual-delay grid。
5. 将输出 CSV 与 package 中 `results/` 的结果对比。

## 9. 当前仍不能完全保证的部分

即使补齐了这些说明，仍有几个地方不是严格一键复现：

- 原始数据集由外部下载，文件版本和镜像可能变化。
- 环境文件不是完整 lock，只是 requirements 和关键版本说明。
- 部分旧 V2X-Sim V2VNet/DiscoNet 结果来自历史表格，不是本 package 重新跑出来的。
- 大型 cache 重新生成可能耗时很长，并依赖 GPU/CPU/磁盘性能。

这些限制不会影响当前 package 作为结果归档和审计材料使用，但如果要面向第三方完全复现，后续还需要补完整环境 lock 和数据集 checksum。
