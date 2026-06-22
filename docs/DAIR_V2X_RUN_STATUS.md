# DAIR-V2X Run Status

Last updated: 2026-05-22 11:24 UTC

## Goal

Download DAIR-V2X and run our method plus available comparison methods.

## Current Reality

- Official OpenDAIRV2X code has been cloned to `external/DAIR-V2X`.
- Official DAIR-V2X-C example data has been downloaded/extracted and is usable
  for pipeline smoke tests.
- Original local DLPCM/VSPM code did not contain a DAIR-V2X dataset adapter;
  a lightweight VSPM-level BEV adapter now exists at `scripts/prepare_dair_vspm_bev.py`.
- Current local DLPCM/VSPM code does not contain V2VNet, DiscoNet, SyncNet, or recent cooperative-perception baseline implementations.
- Official OpenDAIRV2X supports DAIR-V2X baselines such as:
  - `veh_only`
  - `inf_only`
  - `late_fusion`
  - `early_fusion`
  - TCLF via late fusion with/without `--no-comp`
  - FFNet through a separate external repository/checkpoint
- Official OpenDAIRV2X requires `mmdetection3d==0.17.1`, which is not compatible with the current `Android-Lab` environment (`Python 3.11`, `torch==2.7.1`).
- A separate `dair-v2x-openmmlab` environment has been prepared for official DAIR baselines.
- Full DAIR-V2X-C training/evaluation is still blocked by Google Drive quota
  for the public raw data files.

## Completed Setup

- Created:
  - `external/DAIR-V2X`
  - `datasets/DAIR-V2X/`
  - `datasets/DAIR-V2X/dair_v2x_c_manifest.csv`
  - `scripts/download_dair_v2x_c.sh`
- Added:
  - `scripts/prepare_dair_v2x_layout.sh`
  - `scripts/download_dair_v2x_checkpoints.sh`
  - `scripts/run_dair_v2x_official_baselines.sh`
  - `scripts/prepare_dair_v2x_openmmlab_env.sh`
  - `scripts/prepare_dair_vspm_bev.py`
- Installed `gdown` in `Android-Lab`.
- Google Drive folder listing was readable, but the DAIR dataset zip files hit Google Drive quota.
- Retry at `2026-05-22 10:11 UTC` still failed with Google Drive quota on
  `cooperative-vehicle-infrastructure-infrastructure-side-image-deprecated.zip`.
- Retry at `2026-05-22 10:52 UTC` failed for all DAIR raw data zip/z01/z02 files with the Google Drive message
  `Too many users have viewed or downloaded this file recently`.
- Current local DAIR folders under `datasets/DAIR-V2X/DAIR-V2X (CVPR2022)/DAIR-V2X-C/Example`
  are empty placeholders only; no usable DAIR data files are present.
- Existing conda environments checked:
  - `Android-Lab`: Python 3.11.11, torch 2.7.1+cu126, CUDA available.
  - `v2xverse`: Python 3.7.12, torch 1.10.1/cu113, `spconv` and `pypcd` present, but `mmcv/mmdet/mmdet3d` absent.
- `dair-v2x-openmmlab` environment status:
  - Python 3.7.12
  - torch 1.10.1/cu113
  - mmcv-full 1.3.18
  - mmdet 2.14.0
  - mmsegmentation 0.14.1
  - CUDA 11.3 headers are installed through `cudatoolkit-dev`
  - `mmdet3d==0.17.1` compiled and installed from `external/mmdetection3d-v0.17.1`
  - smoke import passed for `torch`, `mmcv`, `mmdet`, `mmseg`, `mmdet3d`, `pypcd`, `spconv`, `pycocotools`
  - CUDA smoke test passed with a small tensor operation on GPU
- `DAIR-V2X-C-Example.zip` was downloaded from the official README example link.
  - This is only for pipeline smoke tests and cannot replace the full training/evaluation dataset.
- Extracted official example data:
  - `datasets/DAIR-V2X/example-cooperative-vehicle-infrastructure`
  - symlinked into `external/DAIR-V2X/data/DAIR-V2X/example-cooperative-vehicle-infrastructure`
- Added:
  - `scripts/prepare_dair_v2x_example_layout.sh`
  - `scripts/run_dair_v2x_example_official_smoke.sh`
  - `scripts/run_dair_v2x_full_workflow.sh`

## Checkpoint Status

Downloaded:

- `external/DAIR-V2X/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_inf_pointpillars_596784ad6127866fcfb286301757c949.pth`
- `external/DAIR-V2X/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth`
- `external/FFNet-VIC3D/ffnet_work_dir/work_dir_ffnet/epoch_40.pth`

Still blocked by Google Drive quota:

- `external/DAIR-V2X/configs/vic3d/early-fusion-pointcloud/pointpillars/vic3d_earlyfusion_veh_pointpillars_67fe2b82320754481ef37f176b647e43.pth`
- `external/DAIR-V2X/configs/vic3d/late-fusion-image/imvoxelnet/vic3d_latefusion_inf_imvoxelnet.pth`
- `external/DAIR-V2X/configs/vic3d/late-fusion-image/imvoxelnet/vic3d_latefusion_veh_imvoxelnet.pth`

Latest checkpoint retry:

- Time: `2026-05-22 11:23 UTC`
- Log: `results/dair_v2x_checkpoint_download_20260522_112331.log`
- Result: late-fusion PointPillars and FFNet checkpoints were already present;
  early-fusion PointPillars and ImVoxelNet checkpoints remain blocked by Google
  Drive quota.

## Example Smoke Results

The official OpenDAIRV2X example smoke run completed on 46 example frames.
These results verify that the environment, data layout, checkpoint loading, and
evaluation scripts work. They are not full DAIR-V2X results and should not be
reported as benchmark numbers.

Run command:

```bash
cd /raid/xuyifan/v2x_code_ckpt
ENV_NAME=dair-v2x-openmmlab DEVICE_ID=0 K_LIST='0 2' \
  bash scripts/run_dair_v2x_example_official_smoke.sh
```

Main log:

```text
results/dair_v2x_example_official_smoke_all_20260522_112110.log
```

| run | frames | car 3D AP@0.30 | car 3D AP@0.50 | car 3D AP@0.70 | car BEV AP@0.30 | car BEV AP@0.50 | car BEV AP@0.70 | avg comm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `veh_only_k0` | 46 | 59.05 | 46.10 | 35.68 | 60.64 | 49.77 | 48.07 | 0.00 B |
| `inf_only_k0` | 46 | 19.39 | 13.45 | 4.15 | 19.39 | 14.04 | 5.48 | 247.30 B |
| `late_fusion_tclf_k0` | 46 | 61.79 | 48.49 | 37.44 | 63.41 | 52.31 | 50.17 | 475.13 B |
| `late_fusion_tclf_k2` | 46 | 60.66 | 47.05 | 36.33 | 62.16 | 51.09 | 48.89 | 463.30 B |
| `late_fusion_no_comp_k2` | 46 | 59.56 | 46.12 | 35.82 | 61.11 | 49.75 | 48.08 | 241.04 B |

Per-run logs:

```text
results/dair_v2x/example_official_smoke/logs/
```

## Our Method Smoke On DAIR Example

The current DLPCM/VSPM code is trained on sparse BEV sequence folders rather
than raw DAIR camera/LiDAR files. A lightweight DAIR-to-BEV adapter has been
smoke-tested on the official example data.

Adapter command:

```bash
cd /raid/xuyifan/v2x_code_ckpt
python scripts/prepare_dair_vspm_bev.py \
  --dair-root datasets/DAIR-V2X/example-cooperative-vehicle-infrastructure \
  --split-path external/DAIR-V2X/data/split_datas/example-cooperative-split-data.json \
  --sensor-view vehicle \
  --output-root datasets/DAIR-V2X/vspm_bev_example \
  --max-frames 20
```

Adapter output:

```text
datasets/DAIR-V2X/vspm_bev_example/train/agent_vehicle/
```

Training smoke command:

```bash
cd /raid/xuyifan/v2x_code_ckpt
WANDB_MODE=offline python src/v2x_forecasting/train_bev.py \
  --data_root datasets/DAIR-V2X/vspm_bev_example/train/agent_vehicle \
  --data_root_val datasets/DAIR-V2X/vspm_bev_example/train/agent_vehicle \
  --T 10 --n 5 --batch 2 --epochs 1 \
  --max_steps 1 --max_val_batches 1 \
  --eval_interval 1 --ckpt_interval 100 \
  --ckpt_dir runs/dair_v2x_example_smoke/checkpoints \
  --project bev-encdec --name dair_v2x_example_smoke
```

Training smoke result:

| run | epoch | step | IoU | Delta IoU | Dynamic IoU |
|---|---:|---:|---:|---:|---:|
| `dair_v2x_example_smoke` | 1 | 1 | 0.0135 | -0.2742 | 0.5436 |

Log:

```text
runs/dair_v2x_example_smoke/checkpoints/wandb_logs.txt
```

This only proves the local DAIR adapter and current training script can execute.
It is not a final result because the official example split contains only a
small number of frames and does not support full training/evaluation.

## Download Manifest

DAIR-V2X-C full dataset file IDs are recorded in:

```text
datasets/DAIR-V2X/dair_v2x_c_manifest.csv
```

Download command:

```bash
cd /raid/xuyifan/v2x_code_ckpt
bash scripts/download_dair_v2x_c.sh
```

Latest full-data retry:

- Time: `2026-05-22 11:23 UTC`
- Log: `results/dair_v2x_dataset_download_20260522_112331.log`
- Result: all full raw DAIR zip/z01/z02 files remain blocked by Google Drive
  quota with the message `Too many users have viewed or downloaded this file recently`.

If Google Drive quota blocks the script, manually download the files from a
browser into:

```text
/raid/xuyifan/v2x_code_ckpt/datasets/DAIR-V2X/raw/
```

Then run:

```bash
cd /raid/xuyifan/v2x_code_ckpt
bash scripts/prepare_dair_v2x_layout.sh
```

This validates the expected `cooperative-vehicle-infrastructure/` tree and
creates the OpenDAIRV2X symlink:

```text
/raid/xuyifan/v2x_code_ckpt/external/DAIR-V2X/data/DAIR-V2X/cooperative-vehicle-infrastructure
```

## Prepared Run Commands

Prepare the old OpenMMLab environment:

```bash
cd /raid/xuyifan/v2x_code_ckpt
bash scripts/prepare_dair_v2x_openmmlab_env.sh
```

Download official OpenDAIRV2X baseline checkpoints:

```bash
cd /raid/xuyifan/v2x_code_ckpt
bash scripts/download_dair_v2x_checkpoints.sh
```

Run official point-cloud baselines after data/checkpoints/env are ready:

```bash
cd /raid/xuyifan/v2x_code_ckpt
ENV_NAME=dair-v2x-openmmlab DEVICE_ID=0 K_LIST="0 1 2" \
  bash scripts/run_dair_v2x_official_baselines.sh
```

The official baseline runner now skips baseline families whose checkpoints are
missing instead of blocking already runnable baselines. Set
`STRICT_CHECKPOINTS=1` to require all point-cloud checkpoints.

Prepare a VSPM-level DAIR BEV sequence adapter after data is ready:

```bash
cd /raid/xuyifan/v2x_code_ckpt
python scripts/prepare_dair_vspm_bev.py \
  --dair-root datasets/DAIR-V2X/cooperative-vehicle-infrastructure \
  --sensor-view vehicle \
  --output-root datasets/DAIR-V2X/vspm_bev
```

Run the full prepared DAIR workflow after the full dataset exists:

```bash
cd /raid/xuyifan/v2x_code_ckpt
ENV_NAME=dair-v2x-openmmlab OUR_ENV_NAME=Android-Lab DEVICE_ID=0 K_LIST="0 1 2" \
  bash scripts/run_dair_v2x_full_workflow.sh
```

This workflow performs layout validation, official runnable baselines,
DAIR-to-VSPM BEV conversion, and our current VSPM/DLPCM training path.
Use `RUN_DOWNLOADS=1` to retry downloads first and `STRICT_CHECKPOINTS=1` to
make missing official checkpoints fatal.

## Immediate Blockers

1. DAIR-V2X public files are hosted on Google Drive and currently hit quota limits from this server.
2. The old MMDetection3D environment is ready, but official OpenDAIRV2X full baselines still require the full DAIR data tree.
3. Early-fusion PointPillars and ImVoxelNet official checkpoints are also quota-blocked.
4. Running "our method" on DAIR-V2X at AP level is not plug-and-play: the current method code consumes V2X-Sim BEV sparse `.npy` sequences, while DAIR-V2X provides camera/LiDAR/calibration/label JSON/PCD files.

## Baseline Scope

Runnable official OpenDAIRV2X baselines prepared in this workspace:

- `veh_only`
- `inf_only`
- `late_fusion` on VIC-Sync / VIC-Async-k
- `late_fusion --no-comp` as the no time-compensation comparison
- TCLF, represented by official `late_fusion` with compensation enabled
- `early_fusion`

FFNet checkpoint is present, but full FFNet integration/evaluation still needs
the external implementation wired into the local runner and the full DAIR data.

## Required Next Work

1. Finish DAIR-V2X-C download and extraction. This is the hard blocker for full DAIR training/evaluation.
2. Download the missing official early-fusion PointPillars and ImVoxelNet checkpoints.
3. Run full OpenDAIRV2X official baselines once data exists.
4. Expand `prepare_dair_vspm_bev.py` from example smoke to full DAIR sequence preparation once data exists.
5. Implement AP-level DLPCM integration inside OpenDAIRV2X if the paper needs detector AP on DAIR, not only VSPM-level BEV forecasting.
