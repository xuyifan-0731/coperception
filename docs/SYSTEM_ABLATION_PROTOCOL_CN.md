# DLPCM 系统级消融实验协议

更新时间：2026-05-27 08:55 UTC

目标：把审稿意见中关于“系统贡献是否来自 dual-latency、sender-side prediction、MST 或 cache”的问题拆成可执行实验。该协议先服务于 VSPM/BEV 层面的快速验证；full detection AP 版本依赖 DAIR-V2X 图像/点云大包和 OpenDAIRV2X baseline 完整跑通。

## 1. 统一输入和输出

统一输入：

- 数据：`datasets/DAIR-V2X/vspm_bev/{val,test}/agent_vehicle` 或 V2X-Sim 对应 agent root。
- checkpoint：目标数据集上训练出的 VSPM best checkpoint。
- 延迟网格：communication delay `d_comm` 与 computation delay `d_comp`，单位为 frame；每帧 `0.2s`。
- 历史长度与预测跨度：默认 `T=10,n=5`；若延迟网格最大 `d_comm+d_comp > 4`，需要将 `n` 增大到 `max(d_comm+d_comp)+1` 并重新训练或评估支持更长 rollout 的 checkpoint。

统一输出：

- 每个 `(d_comm,d_comp,mode)` 的 `IoU`、`copy_last_IoU`、`Delta_IoU`、`Dynamic_IoU`。
- 样本数、remote rollout frame、local rollout frame。
- CSV：`results/{dataset}/system_ablation/vspm_ablation.csv`。
- 表格：`results/{dataset}/system_ablation/vspm_ablation.tex`。

## 2. 消融模式定义

| mode | 通信延迟补偿 | 计算延迟补偿 | sender-side VSPM | MST | 评价目标 |
|---|---|---|---|---|---|
| `no_comp` | 否 | 否 | 否 | 否 | 直接 copy delayed/latest BEV，作为无补偿下界 |
| `comm_only` | 是 | 否 | 是 | 否 | 只补偿通信延迟，目标 horizon 为 `d_comm` |
| `comp_only` | 否 | 是 | 是 | 否 | 只补偿计算延迟，目标 horizon 为 `d_comp` |
| `dual` | 是 | 是 | 是 | 否 | 同时补偿通信和计算延迟，目标 horizon 为 `d_comm+d_comp` |
| `dual_mst` | 是 | 是 | 是 | 是 | 使用压缩态 checkpoint，衡量带宽-性能折中 |
| `receiver_generic` | 是 | 是 | 否 | 否 | 使用 receiver-side/generic checkpoint，验证 vehicle-specific 的必要性 |
| `oracle` | 是 | 是 | 不适用 | 不适用 | 使用 result-time GT，作为上界；只用于表格参考 |

## 3. VSPM 层面执行命令

主 delay grid：

```bash
cd /raid/xuyifan/v2x_code_ckpt
eval "$(conda shell.bash hook)"
conda activate Android-Lab
PYTHONPATH=src python scripts/eval_dair_vspm_delay_grid.py \
  --ckpt runs/dair_v2x_full/our_method_T10_n5/checkpoints/ckpt_best.pth \
  --data-root datasets/DAIR-V2X/vspm_bev/val/agent_vehicle \
  --out results/dair_v2x/our_method/delay_grid_T10_n5.csv \
  --T 10 --n 6 --batch 8 \
  --comm-delays 0,1,2,3,4,5 \
  --comp-delays 0,2,4 \
  --device cuda
```

系统消融扩展：

```bash
cd /raid/xuyifan/v2x_code_ckpt
eval "$(conda shell.bash hook)"
conda activate Android-Lab
for mode in no_comp comm_only comp_only dual; do
  PYTHONPATH=src python scripts/eval_dair_vspm_delay_grid.py \
    --ckpt runs/dair_v2x_full/our_method_T10_n5/checkpoints/ckpt_best.pth \
    --data-root datasets/DAIR-V2X/vspm_bev/val/agent_vehicle \
    --out "results/dair_v2x/system_ablation/${mode}.csv" \
    --T 10 --n 6 --batch 8 \
    --comm-delays 0,1,2,3,4,5 \
    --comp-delays 0,2,4 \
    --device cuda
done
```

当前 `scripts/eval_dair_vspm_delay_grid.py` 已实现 `dual` 的 delay-grid 评价；`--mode` 扩展是下一步具体实现项。

## 4. Detection 层面执行条件

Detection AP 消融只在以下条件满足后执行：

- full DAIR-V2X-C 图像/点云 8 个 manifest 文件全部下载完成。
- `bash scripts/prepare_dair_v2x_layout.sh` 完成。
- `bash scripts/check_dair_v2x_prereqs.sh` 通过。
- OpenDAIRV2X 官方 baseline 可在 `K_LIST="0 1 2 3 4 5"` 下完成。

输出目录：

- `results/dair_v2x/system_ablation/detection_ablation.csv`
- `results/dair_v2x/system_ablation/detection_ablation.tex`
- `results/dair_v2x/system_ablation/logs/`

## 5. 完成标准

1. 每个 mode 在相同数据 split、相同 checkpoint family、相同延迟网格下评估。
2. 每个 CSV 至少包含 `samples > 0`，且覆盖所有指定 `(d_comm,d_comp)`。
3. 论文表格中明确区分 BEV forecasting IoU 与 detection AP，不混用指标。
4. response letter 中将每个审稿人问题映射到对应 mode、表格、图或限制说明。
