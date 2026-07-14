# 260704 Reproduction Assets

本目录只保存“小而关键”的复现材料，不保存原始数据集、大型中间 cache 或未经压缩的大模型副本。

## 1. 目录内容

```text
00_reproduction_assets/
  patches/          # 外部 baseline 的本地修改 diff 和新增小文件
  split_files/      # DAIR / TraF-Align 的划分文件
  dataset_info/     # 不上传大 info 文件，只保存 summary
  env_refs/         # requirements / environment 参考文件
```

## 2. patches

| 文件 | 用途 |
|---|---|
| `patches/DATA_local_changes.patch` | DATA 本地 strict-delay、DAIR/V2X-Sim loader、fusion/inference 修改。 |
| `patches/LRCP_local_changes.patch` | LRCP 本地 DAIR 适配、latency dataset、flow/meta/deformable attention 修改。 |
| `patches/OpenDAIRV2X_local_changes.patch` | OpenDAIRV2X late-fusion/TCLF/RF cache 相关修改。 |
| `patches/TraFAlign_local_changes.patch` | TraF-Align V2X-Seq dataset 和 fusion 适配修改。 |
| `patches/extra_files/LRCP/...` | LRCP 新增 DAIR hypes yaml。 |
| `patches/extra_files/TraF-Align/...` | TraF-Align 新增 joint delay compensation 模块。 |

使用方式：

```bash
cd external/DATA
git checkout 5df7eb6f5659db0d6809fa3cc218aa425bc287b4
git apply ../../release_packages/260704_experiment_artifacts/00_reproduction_assets/patches/DATA_local_changes.patch
```

其它 baseline 同理，但 LRCP 和 TraF-Align 还需要复制 `extra_files/` 中的新增文件。

## 3. split_files

```text
split_files/dair_v2x/
split_files/trafalign/
```

这些是小型划分文件，可以直接复制到 external 代码期望的位置：

```bash
cp split_files/dair_v2x/*.json external/DAIR-V2X/data/split_datas/
cp split_files/trafalign/V2XSeq_dataset_split_official.yaml /tmp/TraF-Align_partial/datasets/Basedataset/
```

## 4. dataset_info

V2X-Sim info `.pkl` 文件约 226MB，没有上传。这里只保存：

```text
dataset_info/v2xsim2_info/summary.json
```

它记录本地 info 文件对应的 split 规模：

- train: 80 scenes, 8000 samples
- val: 10 scenes, 1000 samples
- test: 10 scenes, 1000 samples

## 5. env_refs

这里保存外部 baseline 的 requirements/environment 参考：

```text
env_refs/DATA/requirements.txt
env_refs/LRCP/requirements.txt
env_refs/LRCP/environment.yml
env_refs/TraF-Align/requirements.txt
```

它们不是完整 lock 文件。严格复现时还需要记录 Python、CUDA、PyTorch、spconv、cumm、mmcv、mmdet3d 的实际版本。

## 6. 与其它文档的关系

- 新人入口：`../00_docs/260704_ARTIFACT_INDEX.md`
- 从零准备 checklist：`../00_docs/260704_REPRODUCTION_CHECKLIST.md`
- 外部资源说明：`../00_docs/260704_EXTERNAL_RESOURCES.md`
- 运行命令：`../00_docs/260704_RUNBOOK.md`
