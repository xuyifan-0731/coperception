# DAIR Revision Waiting/Now Task Split

更新时间：2026-05-27 18:26 UTC

## 需要等待 full raw 下载完成的任务

这些任务依赖 `datasets/DAIR-V2X/raw` 中 5 个 AI Studio 镜像 zip 全部完成。

1. 解压并准备 full DAIR-V2X-C layout。
   - 执行：`bash scripts/prepare_dair_v2x_layout.sh`
   - 产物：`datasets/DAIR-V2X/cooperative-vehicle-infrastructure`
   - 目的：让 OpenDAIRV2X 官方 detection pipeline 能读取 image/velodyne/full labels。

2. 运行 full DAIR 前置检查。
   - 执行：`bash scripts/check_dair_v2x_prereqs.sh`
   - 产物：`results/dair_v2x/post_download/check_prereqs_*.log`
   - 目的：确认数据、checkpoint、OpenMMLab 环境都可用。

3. 运行 DAIR 官方 detection baselines。
   - 执行：`ENV_NAME=dair-v2x-openmmlab DEVICE_ID=0 K_LIST="0 1 2" STRICT_CHECKPOINTS=0 bash scripts/run_dair_v2x_official_baselines.sh`
   - 后续扩展：`K_LIST="0 1 2 3 4 5"`
   - 产物：官方 3D/BEV AP、avg communication。

4. 解析官方 baseline 日志并生成论文表。
   - 产物：`results/dair_v2x/official_baselines/summary.csv`
   - 产物：`results/dair_v2x/official_baselines/summary.tex`

## 当前可以进行且已开始的任务

1. DAIR `T10_n5` 当前 checkpoint 的短时延系统消融。
   - checkpoint：`runs/dair_v2x_full/our_method_T10_n5/checkpoints/ckpt_11000.pth`
   - 运行模式：`no_comp/comm_only/comp_only/dual/oracle`
   - 输出目录：`results/dair_v2x/our_method/current_T10_n5_ckpt11000`
   - PID：`results/dair_v2x/our_method/current_T10_n5_ckpt11000_eval.pid`
   - 说明：`T10_n5` 只能覆盖总时延 `0..4` 帧，完整 `0..9` 帧等待 `T10_n10`。

2. DAIR `T10_n5` 主训练继续跑到 30 epoch。
   - 当前 best：见 `results/dair_v2x/summary/training_summary.csv`
   - 完成后：选择 best checkpoint，重新跑正式短时延矩阵。

3. DAIR `T10_n10` 主训练继续跑到 30 epoch。
   - 目的：覆盖 V2X-Sim 风格完整 latency grid，总时延到 9 帧。
   - 完成后：跑完整 DAIR delay grid，并生成 heatmap/table。

4. DAIR MST 四个分支继续并行训练。
   - 分支：`baseline/gru_fp16/ds64_fp16/bottleneck12`
   - 完成后：汇总 IoU、state MiB/agent、latency、带宽-性能 Pareto 表。

5. 下载完成后自动接续 layout/prereq 的 watcher 已启动。
   - 脚本：`scripts/watch_dair_download_then_prepare.sh`
   - PID：`results/dair_v2x/post_download_watcher.pid`
   - 日志：`results/dair_v2x/post_download_watcher_*.log`
   - 说明：watcher 只用 `.aistudio.part` 判断 AI Studio 当前文件是否未完成；旧 Google `.part` 残留不阻塞后续 layout。

## 当前可以进行但尚未启动的任务

1. DAIR robustness 轻量评估。
   - 建议等待 `T10_n5` 或 `T10_n10` 训练完成后用 best checkpoint 跑。
   - 原因：当前中间 checkpoint 能跑，但结果容易被后续 best checkpoint 覆盖。

2. DAIR T/n sensitivity。
   - 建议在主结果、系统消融、MST 消融完成后启动。
   - 原因：计算量大，优先级低于回应审稿意见的核心实验。

3. 论文正文表格合并。
   - 现在可维护草稿，但正式数值要等 full raw baseline、best checkpoint delay grid、MST 完成后再落稿。
