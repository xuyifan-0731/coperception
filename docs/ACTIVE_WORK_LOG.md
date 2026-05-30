# DLPCM Active Work Log

This is the live working log for the current revision-experiment workflow.
Update this file whenever the active plan, running process, generated artifact,
or blocker changes.

Scope constraint: only use files under `/raid/xuyifan/v2x_code_ckpt`.

## Current Status

Last updated: 2026-05-30 20:25 UTC

- DAIR-V2X-C full raw data is downloaded, laid out, and prerequisite checks passed.
- Core DAIR revision experiments are complete:
  - official baselines: `veh_only_k0`, `inf_only_k0`, `late_fusion_tclf_k0..k5`, `late_fusion_no_comp_k1..k5`, `early_fusion_k0..k2`;
  - our DAIR VSPM delay/system ablations: `T10_n5 ckpt16000` and `T10_n10 ckpt15000`;
  - DAIR MST ablation: `baseline`, `gru_fp16`, `ds64_fp16`, `bottleneck12`;
  - DAIR sensitivity: `T5_n5`, `T10_n3`, `T10_n5`, `T10_n10`, `T10_n15`, `T20_n5`, `T30_n5`;
  - DAIR robustness: `T10_n5` and `T10_n10` under clean, packet loss, dropout, and false-positive perturbations.
- Current active optional experiment: official DAIR `early_fusion_k3/k4/k5`, running detached on GPUs 0/1/2.
  - logs: `results/dair_v2x/official_baselines/logs/early_fusion_k{3,4,5}.log`
  - root PID files: `results/dair_v2x/official_baselines/isolated_early_fusion_k{3,4,5}.pid`
  - runner: `scripts/run_dair_v2x_official_isolated_early_fusion.sh`
  - watcher: `scripts/watch_dair_early_fusion_then_refresh.sh`
- Watcher behavior after `early_fusion_k3/k4/k5` complete:
  - refresh `results/dair_v2x/official_baselines/summary_partial.csv`;
  - refresh `results/dair_v2x/tables/dair_revision_tables.tex`;
  - refresh `DLPCM/table/dair_*.tex` and `DLPCM/pics/dair_*.pdf`;
  - update `results/dair_v2x/summary/current_status.csv`.
- Paper artifacts are generated from CSV by `scripts/make_dair_paper_artifacts.py`.
- Static LaTeX path check passed for `\input{...}` and `\includegraphics{...}` targets. PDF compilation is unavailable on this machine because no LaTeX engine is installed.

Completed VSPM sensitivity results:

| run | epoch | step | IoU | Delta IoU | Dynamic IoU |
|---|---:|---:|---:|---:|---:|
| `T5_n5` | 30 | 13800 | 0.7505 | 0.0540 | 0.3204 |
| `T10_n5` | 30 | 13050 | 0.7513 | 0.0570 | 0.3356 |
| `T20_n5` | 30 | 23100 | 0.7496 | 0.0581 | 0.3297 |
| `T30_n5` | 30 | 40200 | 0.7518 | 0.0597 | 0.3326 |
| `T10_n3` | 30 | 13350 | 0.7678 | 0.0558 | 0.3413 |
| `T10_n10` | 30 | 24600 | 0.7243 | 0.0559 | 0.3332 |
| `T10_n15` | 30 | 46200 | 0.7075 | 0.0549 | 0.3109 |

Latest completed checkpoints:

- `runs/revision_vspm_sensitivity/T5_n5/checkpoints/ckpt_13000.pth`
- `runs/revision_vspm_sensitivity/T10_n5/checkpoints/ckpt_13000.pth`
- `runs/revision_vspm_sensitivity/T20_n5/checkpoints/ckpt_23000.pth`
- `runs/revision_vspm_sensitivity/T30_n5/checkpoints/ckpt_40000.pth`
- `runs/revision_vspm_sensitivity/T10_n3/checkpoints/ckpt_13000.pth`
- `runs/revision_vspm_sensitivity/T10_n10/checkpoints/ckpt_24000.pth`
- `runs/revision_vspm_sensitivity/T10_n15/checkpoints/ckpt_46000.pth`

Completed MST ablation results:

| run | epoch | step | IoU | Delta IoU | Dynamic IoU | latest checkpoint |
|---|---:|---:|---:|---:|---:|---|
| `baseline` | 30 | 13050 | 0.7488 | 0.0545 | 0.3296 | `ckpt_13000.pth` |
| `gru_fp16` | 30 | 13050 | 0.7462 | 0.0520 | 0.3075 | `ckpt_13000.pth` |
| `ds64_fp16` | 30 | 13050 | 0.7200 | 0.0258 | 0.0914 | `ckpt_13000.pth` |
| `bottleneck12` | 30 | 13050 | 0.7193 | 0.0250 | 0.0986 | `ckpt_13000.pth` |

Live process snapshot:

- No active revision training/evaluation process is currently running.
- `results/revision_plan/pipeline_status.md` reports full pipeline, VSPM recovery, and MST ablation as stopped.
- Resource tuning was applied while the CPU duplicate was still running: CPU robustness child PID `3625652` was lowered to nice `+10` and restricted to CPU cores `0-31`.
- Future CPU robustness launches now default to `CPU_THREADS=16` and pass `--torch-threads` to prevent host-side oversubscription.
- A two-sample GPU robustness smoke test passed before the full run.
- Repeated `nohup` attempts for the GPU robustness run exited immediately with empty logs in this environment, so the full GPU run was executed in a controlled foreground exec session.

```bash
python scripts/eval_vspm_robustness.py \
  --ckpt runs/revision_vspm_sensitivity/T10_n5/checkpoints/ckpt_13000.pth \
  --data-root V2X-Sim-det/test/agent1 \
  --T 10 --n 5 --max-samples 256 --device cuda \
  --torch-threads 16 \
  --out results/revision_plan/vspm_robustness_T10_n5_gpu.csv
```

Completed robustness results:

| condition | mean IoU | baseline IoU | Delta IoU | Dynamic IoU |
|---|---:|---:|---:|---:|
| `clean` | 0.7884 | 0.7366 | 0.0517 | 0.3858 |
| `packet_loss_0.1` | 0.7849 | 0.7356 | 0.0493 | 0.3734 |
| `packet_loss_0.2` | 0.7813 | 0.7347 | 0.0466 | 0.3588 |
| `packet_loss_0.3` | 0.7765 | 0.7334 | 0.0431 | 0.3409 |
| `packet_loss_0.5` | 0.7652 | 0.7293 | 0.0359 | 0.2871 |
| `dropout_0.05` | 0.7866 | 0.7041 | 0.0825 | 0.4677 |
| `dropout_0.1` | 0.7835 | 0.6713 | 0.1122 | 0.5227 |
| `dropout_0.2` | 0.7679 | 0.6043 | 0.1636 | 0.5715 |
| `false_positive_0.001` | 0.7870 | 0.7262 | 0.0608 | 0.3765 |
| `false_positive_0.005` | 0.7820 | 0.6871 | 0.0949 | 0.3431 |

## Work Completed In This Session

- 2026-05-30 07:12 UTC final experiment status check:
  - No active DAIR experiment processes remain.
  - All 8 GPUs are idle.
  - Main official baseline completed:
    - `results/dair_v2x/official_baselines_full.pid` is stopped.
    - log `results/dair_v2x/official_baselines_full_20260528_124300.log` ends with `official DAIR-V2X baseline pass completed`.
  - Isolated official late-fusion `k=3,4,5` runner completed:
    - `results/dair_v2x/official_baselines/isolated_late_fusion_k3_k4_k5.pid` is stopped.
    - all `late_fusion_tclf_k{3,4,5}` and `late_fusion_no_comp_k{3,4,5}` logs parse as complete.
  - All DAIR sensitivity runs completed:
    - `T5_n5`, `T10_n3`, `T20_n5`, `T30_n5`, `T10_n15`.
  - Refreshed result summaries:
    - official baseline CSV `results/dair_v2x/official_baselines/summary_partial.csv`, 16 complete logs.
    - training CSV `results/dair_v2x/summary/training_summary.csv`.
    - table drafts under `results/dair_v2x/tables/`.

- 2026-05-29 04:04 UTC update:
  - Refreshed DAIR official baseline parser and DAIR training summaries.
  - Current main official baseline is still running:
    - PID `results/dair_v2x/official_baselines_full.pid`
    - current slow stage: `early_fusion_k1`
    - latest observed progress: about `292/1777`, `16.4%`
  - Completed since last update:
    - official `early_fusion_k0`
    - official `late_fusion_tclf_k1`
    - official `late_fusion_no_comp_k1`
    - DAIR sensitivity `T5_n5`
    - DAIR sensitivity `T10_n3`
    - final `T10_n5 ckpt_16000` short-latency system ablation
    - `T10_n10 ckpt_15000` robustness evaluation
  - Added DAIR table builder:
    - script `scripts/make_dair_revision_tables.py`
    - outputs under `results/dair_v2x/tables`
    - combined LaTeX draft `results/dair_v2x/tables/dair_revision_tables.tex`
  - Added isolated official late-fusion runner to use idle GPUs without sharing `external/DAIR-V2X/cache`:
    - script `scripts/run_dair_v2x_official_isolated_late_fusion.sh`
    - launched `k=3,4,5` on GPUs `1,4,6`
    - parent PID `results/dair_v2x/official_baselines/isolated_late_fusion_k3_k4_k5.pid`
    - child PIDs under `results/dair_v2x/official_baselines/logs/isolated_late_fusion_k*.pid`
    - logs: `results/dair_v2x/official_baselines/logs/late_fusion_tclf_k{3,4,5}.log`
    - after each TCLF run completes, the same process runs `late_fusion_no_comp_k{3,4,5}`.

- 2026-05-28 17:35 UTC progress check:
  - Official DAIR baseline is still running on GPU0:
    - PID `174438`
    - current run `early_fusion_k0`
    - latest observed progress about `634/1789` frames, approximately `35.4%`
    - parsed official summary remains complete for `veh_only_k0`, `inf_only_k0`, and `late_fusion_tclf_k0`; `early_fusion_k0` is incomplete.
  - DAIR VSPM sensitivity jobs are still alive on GPUs 1-5:
    - `T5_n5`, `T20_n5`, `T30_n5`, `T10_n3`, `T10_n15`
    - latest refreshed summary is `results/dair_v2x/summary/training_summary.csv`.
  - Started additional non-conflicting DAIR evaluations using idle GPUs:
    - GPU6: final `T10_n5` short-latency system ablation with `ckpt_16000.pth`
      - PID `results/dair_v2x/our_method/final_T10_n5_ckpt16000_eval.pid`
      - latest log pointer `results/dair_v2x/our_method/final_T10_n5_ckpt16000_eval.latest_log`
      - outputs `results/dair_v2x/our_method/final_T10_n5_ckpt16000/delay_grid_{mode}.csv`
    - GPU7: `T10_n10` robustness evaluation with `ckpt_15000.pth`
      - PID `results/dair_v2x/robustness_T10_n10_ckpt15000.pid`
      - latest log pointer `results/dair_v2x/robustness_T10_n10_ckpt15000.latest_log`
      - output `results/dair_v2x/robustness_T10_n10_ckpt15000.csv`
  - Note: new detached jobs must use `setsid ... < /dev/null &`; plain `nohup ... &` did not survive this execution environment.

- 2026-05-27 18:10 UTC update:
  - Confirmed AI Studio DAIR-V2X downloader and keepalive are still running.
  - Current AI Studio mirror status: 2/5 complete, infrastructure-side image zip in progress, velodyne zips pending.
  - Refreshed DAIR training summaries with:

```bash
python scripts/collect_results.py --root runs/dair_v2x_full --out results/dair_v2x/summary --copy-raw-metrics
```

  - Wrote current DAIR status table: `results/dair_v2x/summary/current_status.csv`.
  - Wrote reviewer-response working draft: `docs/REVIEWER_RESPONSE_DRAFT_CN.md`.
  - Started DAIR `T10_n10` main training on GPU5 so the DAIR delay-grid evaluation can cover the V2X-Sim-style total horizon up to 9 frames:

```bash
CUDA_VISIBLE_DEVICES=5 /raid/xuyifan/miniconda3/envs/Android-Lab/bin/python \
  src/v2x_forecasting/train_bev.py \
  --data_root datasets/DAIR-V2X/vspm_bev/train/agent_vehicle \
  --data_root_val datasets/DAIR-V2X/vspm_bev/val/agent_vehicle \
  --T 10 --n 10 --batch 16 --epochs 30 \
  --use_dice \
  --teacher_forcing_start 1.0 \
  --teacher_forcing_end 0.0 \
  --teacher_forcing_epochs 10 \
  --ckpt_interval 1000 \
  --eval_interval 1000 \
  --ckpt_dir runs/dair_v2x_full/our_method_T10_n10/checkpoints \
  --project bev-encdec \
  --name dair_v2x_full_T10_n10 \
  --device cuda
```

  - `T10_n10` PID: `results/dair_v2x_full_train_vehicle_T10_n10.pid`.
  - `T10_n10` log: `results/dair_v2x_full_train_vehicle_T10_n10_20260527_180946.log`.
  - Extended `scripts/eval_dair_vspm_delay_grid.py` with `--latency-grid-csv` and `--skip-unsupported`, so DAIR evaluations can directly reuse `results/revision_plan/latency_grid.csv`.
  - Verified the new latency-grid path using current DAIR `T10_n5` `ckpt_10000.pth` on a two-sample smoke run:
    - `results/dair_v2x/our_method/delay_grid_T10_n5_ckpt10000_latency_csv_smoke.csv`

- 2026-05-27 18:27 UTC update:
  - Added current/waiting split document: `docs/DAIR_REVISION_WAITING_AND_NOW_CN.md`.
  - Started current DAIR `T10_n5` short-latency system ablation using `ckpt_11000.pth` on GPU6.
    - PID: `results/dair_v2x/our_method/current_T10_n5_ckpt11000_eval.pid`
    - Log: `results/dair_v2x/our_method/current_T10_n5_ckpt11000_eval_20260527_182606.log`
    - Outputs: `results/dair_v2x/our_method/current_T10_n5_ckpt11000/delay_grid_{mode}.csv`
    - Modes: `no_comp`, `comm_only`, `comp_only`, `dual`, `oracle`
  - Added and started post-download watcher:
    - Script: `scripts/watch_dair_download_then_prepare.sh`
    - PID: `results/dair_v2x/post_download_watcher.pid`
    - Log: `results/dair_v2x/post_download_watcher_20260527_182753.log`
    - Behavior: wait for all 5 AI Studio mirror zip files, then run `prepare_dair_v2x_layout.sh` and `check_dair_v2x_prereqs.sh`; official baselines remain opt-in via `RUN_OFFICIAL_BASELINES=1`.
    - The watcher ignores stale Google `.part` files and only treats `.aistudio.part` as an active incomplete AI Studio download.

- 2026-05-28 12:43 UTC update:
  - AI Studio DAIR-V2X download completed: 5/5 mirror files present under `datasets/DAIR-V2X/raw`.
  - Post-download watcher completed layout preparation and prereq check:
    - `results/dair_v2x/post_download/prepare_layout_20260528_012753.log`
    - `results/dair_v2x/post_download/check_prereqs_20260528_013213.log`
    - prereq result: `OK=15 WARN=0 FAIL=0`
  - Fixed `scripts/run_dair_v2x_official_baselines.sh` so conda activation works under `set -u`.
  - Fixed `scripts/prepare_dair_v2x_layout.sh` for AI Studio archive layout:
    - skip already extracted large archives,
    - link nested AI Studio image/velodyne folders to the official `image/` and `velodyne/` paths expected by OpenDAIRV2X.
  - Verified representative image/point-cloud paths:
    - `vehicle-side/velodyne/004105.pcd`
    - `infrastructure-side/velodyne/000009.pcd`
    - `vehicle-side/image/000000.jpg`
    - `infrastructure-side/image/000009.jpg`
  - Restarted full DAIR official point-cloud baselines:
    - PID: `results/dair_v2x/official_baselines_full.pid`
    - Log: `results/dair_v2x/official_baselines_full_20260528_124300.log`
    - current first run: `veh_only_k0`
  - Added parser for official baseline logs:
    - `scripts/parse_dair_official_baseline_logs.py`
  - Started full DAIR `T10_n10` delay/system ablation using `ckpt_15000.pth`:
    - PID: `results/dair_v2x/our_method/full_T10_n10_ckpt15000_eval.pid`
    - Log: `results/dair_v2x/our_method/full_T10_n10_ckpt15000_eval_20260528_124019.log`
    - Output dir: `results/dair_v2x/our_method/full_T10_n10_ckpt15000`

- 2026-05-28 13:34 UTC update:
  - Increased DAIR parallel utilization after confirming only the official baseline was active.
  - Started DAIR VSPM sensitivity runs:
    - `T5_n5`, GPU1, PID `results/dair_v2x/vspm_sensitivity/logs/T5_n5.pid`, log `results/dair_v2x/vspm_sensitivity/logs/T5_n5_20260528_133341.log`
    - `T20_n5`, GPU2, PID `results/dair_v2x/vspm_sensitivity/logs/T20_n5.pid`, log `results/dair_v2x/vspm_sensitivity/logs/T20_n5_20260528_133341.log`
    - `T30_n5`, GPU3, PID `results/dair_v2x/vspm_sensitivity/logs/T30_n5.pid`, log `results/dair_v2x/vspm_sensitivity/logs/T30_n5_20260528_133341.log`
    - `T10_n3`, GPU4, PID `results/dair_v2x/vspm_sensitivity/logs/T10_n3.pid`, log `results/dair_v2x/vspm_sensitivity/logs/T10_n3_20260528_133341.log`
    - `T10_n15`, GPU5, PID `results/dair_v2x/vspm_sensitivity/logs/T10_n15.pid`, log `results/dair_v2x/vspm_sensitivity/logs/T10_n15_20260528_133341.log`
  - Started DAIR VSPM robustness evaluation:
    - checkpoint: `runs/dair_v2x_full/our_method_T10_n5/checkpoints/ckpt_16000.pth`
    - GPU6, PID `results/dair_v2x/robustness_T10_n5_ckpt16000.pid`
    - log `results/dair_v2x/robustness_T10_n5_ckpt16000_20260528_133341.log`
    - output `results/dair_v2x/robustness_T10_n5_ckpt16000.csv`
  - Left GPU7 free for OOM recovery, short follow-up evaluations, or a new task after the first batch stabilizes.

- Added AI Studio DAIR-V2X downloader after Google Drive quota failures:
  - `scripts/download_dair_v2x_aistudio.py`
  - `scripts/download_dair_v2x_aistudio_until_complete.sh`
- Verified AI Studio dataset `179509` exposes the needed file metadata and download API, but unauthenticated download-link requests return `need login or access token`.
- Updated `scripts/monitor_dair_v2x_download.sh` to show both Google Drive manifest status and AI Studio single-zip mirror status.
- Google Drive downloader now uses fixed `.part` resume names and the detached retry window is 600s.
- Re-entered DAIR-V2X task with scope restricted to `/raid/xuyifan/v2x_code_ckpt`.
- Confirmed `external/DAIR-V2X` is present at commit `c885c54af0c34bc515fa9ca8b5e8fda76a15462c`.
- Confirmed local DAIR data is not available yet; only manifest and empty placeholder folders exist.
- Retried DAIR-V2X-C Google Drive download; failed again due to public quota at `2026-05-22 10:11 UTC`.
- Checked compute resources: 8 RTX 4090 GPUs are idle and `/raid` has about 6.1T free.
- Checked conda environments:
  - `Android-Lab`: Python 3.11.11, torch 2.7.1+cu126.
  - `v2xverse`: Python 3.7.12, torch 1.10.1/cu113, `spconv`/`pypcd` available, `mmcv/mmdet/mmdet3d` missing.
- Added DAIR preparation/run scripts:
  - `scripts/prepare_dair_v2x_layout.sh`
  - `scripts/download_dair_v2x_checkpoints.sh`
  - `scripts/run_dair_v2x_official_baselines.sh`
  - `scripts/prepare_dair_v2x_openmmlab_env.sh`
  - `scripts/prepare_dair_vspm_bev.py`
- Installed CUDA 11.3 headers into `dair-v2x-openmmlab` via `cudatoolkit-dev`.
- Built and installed local source `mmdet3d==0.17.1` with:

```bash
cd /raid/xuyifan/v2x_code_ckpt/external/mmdetection3d-v0.17.1
CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST='8.6+PTX' FORCE_CUDA=1 MAX_JOBS=8 \
  python -m pip install -v --no-deps -e .
```

- Latest mmdet3d build log:
  - `results/mmdet3d_install_20260522_105357.log`
- Verified `dair-v2x-openmmlab` imports and CUDA smoke test:
  - Python 3.7.12
  - torch 1.10.1/cu113
  - mmcv 1.3.18
  - mmdet 2.14.0
  - mmseg 0.14.1
  - mmdet3d 0.17.1
  - `torch.ones(..., device='cuda').sum()` passed
  - `mmdet3d.ops.Voxelization` import passed
  - `pypcd`, `spconv`, `pycocotools`, `terminaltables`, `prettytable` imports passed
- Patched `scripts/run_dair_v2x_official_baselines.sh` so missing early-fusion/image checkpoints do not block runnable late-fusion point-cloud baselines. Use `STRICT_CHECKPOINTS=1` to enforce complete checkpoint availability.
- Started official `DAIR-V2X-C-Example.zip` download for pipeline smoke tests:
  - output target: `datasets/DAIR-V2X/raw/DAIR-V2X-C-Example.zip`
  - log: latest `results/dair_v2x_example_download_*.log`
- Retried DAIR dataset download at `2026-05-22 10:52 UTC`.
  - Result: all raw data zip/z01/z02 files are still blocked by Google Drive quota.
  - Log: `results/dair_v2x_dataset_download_20260522_105229.log`
- Retried official checkpoint download at `2026-05-22 10:52 UTC`.
  - Newly downloaded: `vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth`
  - Still blocked: early-fusion PointPillars and two ImVoxelNet checkpoints.
  - Log: `results/dair_v2x_checkpoint_download_20260522_105229.log`

- Prepared and ran official OpenDAIRV2X example smoke tests:
  - Script: `scripts/run_dair_v2x_example_official_smoke.sh`
  - Main log: `results/dair_v2x_example_official_smoke_all_20260522_112110.log`
  - Data root: `datasets/DAIR-V2X/example-cooperative-vehicle-infrastructure`
  - OpenDAIRV2X symlink: `external/DAIR-V2X/data/DAIR-V2X/example-cooperative-vehicle-infrastructure`
  - Completed runs:
    - `veh_only_k0`
    - `inf_only_k0`
    - `late_fusion_tclf_k0`
    - `late_fusion_tclf_k2`
    - `late_fusion_no_comp_k2`

Official example smoke results, 46 frames:

| run | car 3D AP@0.30 | car 3D AP@0.50 | car 3D AP@0.70 | car BEV AP@0.30 | car BEV AP@0.50 | car BEV AP@0.70 | avg comm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `veh_only_k0` | 59.05 | 46.10 | 35.68 | 60.64 | 49.77 | 48.07 | 0.00 B |
| `inf_only_k0` | 19.39 | 13.45 | 4.15 | 19.39 | 14.04 | 5.48 | 247.30 B |
| `late_fusion_tclf_k0` | 61.79 | 48.49 | 37.44 | 63.41 | 52.31 | 50.17 | 475.13 B |
| `late_fusion_tclf_k2` | 60.66 | 47.05 | 36.33 | 62.16 | 51.09 | 48.89 | 463.30 B |
| `late_fusion_no_comp_k2` | 59.56 | 46.12 | 35.82 | 61.11 | 49.75 | 48.08 | 241.04 B |

Note: these are example smoke results only, not full DAIR benchmark results.

- Smoke-tested our current VSPM training path on DAIR example data:
  - Adapter: `scripts/prepare_dair_vspm_bev.py`
  - Adapter output: `datasets/DAIR-V2X/vspm_bev_example/train/agent_vehicle/`
  - Frames written: 20
  - Training log: `runs/dair_v2x_example_smoke/checkpoints/wandb_logs.txt`
  - One-step metrics: IoU `0.0135`, Delta IoU `-0.2742`, Dynamic IoU `0.5436`
  - Note: this is only a pipeline smoke test, not a full DAIR result.

- Retried official checkpoint download at `2026-05-22 11:23 UTC`.
  - Log: `results/dair_v2x_checkpoint_download_20260522_112331.log`
  - Still blocked by Google Drive quota:
    - early-fusion PointPillars checkpoint
    - infrastructure ImVoxelNet checkpoint
    - vehicle ImVoxelNet checkpoint
- Retried full DAIR-V2X-C dataset download at `2026-05-22 11:23 UTC`.
  - Log: `results/dair_v2x_dataset_download_20260522_112331.log`
  - Result: all full raw zip/z01/z02 files remain blocked by Google Drive quota.

- Current DAIR task queue:
  - Running: none.
  - Waiting/blocked: DAIR-V2X-C dataset download because Google Drive quota blocks public files.
  - Ready to run after data: layout validation and OpenDAIRV2X symlink.
  - Ready to run after data and environment verification: official late-fusion point-cloud baselines.
  - Waiting on checkpoint quota: early-fusion point-cloud baseline and image baselines.
  - Ready to run after data: VSPM-level DAIR BEV adapter.
  - Ready to run after data: full orchestrated workflow `scripts/run_dair_v2x_full_workflow.sh`.

- Open manual doc path:
  - `/raid/xuyifan/v2x_code_ckpt/docs/DAIR_V2X_RUN_STATUS.md`
  - VS Code command: `code /raid/xuyifan/v2x_code_ckpt/docs/DAIR_V2X_RUN_STATUS.md`

- Extracted and verified local `V2X-Sim-det/` dataset.
- Installed dependencies in `Android-Lab`.
- Fixed PyTorch compatibility by pinning `torch==2.7.1`.
- Verified CUDA availability on RTX 4090.
- Added smoke controls to training scripts:
  - `--max_steps`
  - `--max_val_batches`
- Smoke-tested baseline VSPM training.
- Smoke-tested compressed/MST VSPM training.
- Started full sequential revision pipeline:
  - `scripts/run_revision_full_pipeline.sh`
- Generated VSPM profile:
  - `results/revision_plan/vspm_profile.csv`
- Generated scalability/cache profile:
  - `results/revision_plan/scalability_profile.csv`
- Added active status reporting:
  - `scripts/report_pipeline_status.py`
  - `results/revision_plan/pipeline_status.md`
  - `results/revision_plan/pipeline_latest_metrics.csv`
  - `results/revision_plan/pipeline_checkpoints.csv`
- Added table draft generation:
  - `scripts/make_revision_tables.py`
  - `results/revision_tables/revision_tables.tex`
- Added figure draft generation:
  - `scripts/plot_revision_tables.py`
  - `results/revision_figures/vspm_training_iou.pdf`
  - `results/revision_figures/vspm_profile_tradeoff.pdf`
  - `results/revision_figures/cache_scalability.pdf`
- Added VSPM robustness evaluation:
  - `scripts/eval_vspm_robustness.py`
  - `scripts/run_vspm_robustness_eval.sh`
  - Smoke output: `results/revision_plan/vspm_robustness_smoke.csv`
- Collected current intermediate summaries:
  - `results/summary_revision/training_summary.csv`
  - `results/summary_revision/training_timeseries.csv`
- Completed `T5_n5` full training.
- Completed `T10_n5` full training.
- Detected CUDA OOM at `T20_n5` with batch `16`.
- Added checkpoint numeric sorting fix in:
  - `scripts/report_pipeline_status.py`
- Added recovery script for remaining VSPM sensitivity runs with reduced batch sizes:
  - `scripts/recover_vspm_sensitivity_after_oom.sh`
- Started recovery run:
  - PID: `1606101`
  - Active run: `T20_n5`, batch `8`
- Completed VSPM recovery queue:
  - `T20_n5`, batch `8`
  - `T30_n5`, batch `4`
  - `T10_n3`, batch `16`
  - `T10_n10`, batch `8`
  - `T10_n15`, batch `4`
- Refreshed intermediate summary after VSPM sensitivity:
  - `results/summary_revision/training_summary.csv`
  - `results/summary_revision/training_timeseries.csv`
- Started MST ablation:
  - PID: `3595781`
  - Active run: `baseline`
  - Log: `results/revision_plan/run_mst_ablation.full.log`
- Refreshed MST intermediate summary:
  - `results/summary_revision/training_summary.csv`
  - `results/revision_plan/pipeline_latest_metrics.csv`
  - `results/revision_plan/pipeline_checkpoints.csv`
- Regenerated revision tables and figures while MST baseline is running:
  - `results/revision_tables/revision_tables.tex`
  - `results/revision_figures/vspm_training_iou.pdf`
  - `results/revision_figures/vspm_profile_tradeoff.pdf`
  - `results/revision_figures/cache_scalability.pdf`
- Refreshed summaries while MST baseline and CPU robustness evaluation are still running:
  - `results/summary_revision/training_summary.csv`
  - `results/revision_plan/pipeline_status.md`
  - `results/revision_plan/pipeline_latest_metrics.csv`
  - `results/revision_plan/pipeline_checkpoints.csv`
- Captured MST baseline step `3000` validation:
  - IoU `0.7358`
  - Delta IoU `0.0415`
  - Dynamic IoU `0.2611`
  - Checkpoint `runs/revision_mst_ablation/baseline/checkpoints/ckpt_3000.pth`
- Captured MST baseline step `4000` validation:
  - IoU `0.7396`
  - Delta IoU `0.0454`
  - Dynamic IoU `0.2863`
  - Checkpoint `runs/revision_mst_ablation/baseline/checkpoints/ckpt_4000.pth`
- Refreshed summaries, status files, tables, and figures after step `4000`:
  - `results/revision_plan/pipeline_status.md`
  - `results/summary_revision/training_summary.csv`
  - `results/revision_tables/revision_tables.tex`
  - `results/revision_figures/vspm_training_iou.pdf`
- Regenerated table and figure drafts again after the latest summary refresh:
  - `results/revision_tables/revision_tables.tex`
  - `results/revision_figures/vspm_training_iou.pdf`
  - `results/revision_figures/vspm_profile_tradeoff.pdf`
  - `results/revision_figures/cache_scalability.pdf`
- Added CPU-thread controls for future robustness evaluations:
  - `scripts/run_vspm_robustness_eval.sh`
  - `scripts/eval_vspm_robustness.py`
- Added a dedicated GPU robustness launcher for `T10_n5`:
  - `scripts/run_vspm_robustness_T10_n5_gpu.sh`
- Verified GPU robustness smoke test:
  - `results/revision_plan/vspm_robustness_T10_n5_gpu_smoke.csv`
- Completed full GPU robustness evaluation:
  - `results/revision_plan/vspm_robustness_T10_n5_gpu.csv`
- Stopped redundant CPU robustness evaluation after the GPU output was complete.
- Added robustness outputs to revision table and figure generation:
  - `results/revision_tables/robustness_table.csv`
  - `results/revision_figures/vspm_robustness.pdf`
  - `results/revision_tables/revision_tables.tex`
- Refreshed pipeline status and result summaries after robustness completion:
  - `results/revision_plan/pipeline_status.md`
  - `results/summary_revision/training_summary.csv`
- Synchronized higher-level status/prep docs with current completed/running work:
  - `docs/RUN_STATUS.md`
  - `docs/EXPERIMENT_PREP.md`
- Added current workspace/code/result/task overview:
  - `docs/CURRENT_WORKSPACE_OVERVIEW.md`
  - `docs/CURRENT_WORKSPACE_OVERVIEW_CN.md`
- Refreshed summaries after all MST variants completed:
  - `results/revision_plan/pipeline_status.md`
  - `results/revision_plan/pipeline_latest_metrics.csv`
  - `results/summary_revision/training_summary.csv`
  - `results/revision_tables/active_training_table.csv`
- Verified the script edits:
  - `bash -n scripts/run_vspm_robustness_eval.sh`
  - `python -m py_compile scripts/eval_vspm_robustness.py`

## Active Work Queue

### While MST Training Runs

- [x] Periodically refresh:
  - `results/revision_plan/pipeline_status.md`
  - `results/revision_plan/pipeline_latest_metrics.csv`
  - `results/revision_plan/pipeline_checkpoints.csv`
- [x] Regenerate paper table drafts after current MST validation point.
- [x] Regenerate paper figure drafts after current MST validation point.
- [x] Run CPU robustness evaluation from stable checkpoints when useful.
- [x] Keep this file updated with new outputs and decisions.

### After Current MST Run Finishes

- [ ] Confirm final checkpoint and final validation metrics.
- [ ] Collect results with:

```bash
python scripts/collect_results.py --root runs --out results/summary_revision --copy-raw-metrics
```

- [ ] Regenerate:

```bash
python scripts/make_revision_tables.py
MPLBACKEND=Agg python scripts/plot_revision_tables.py
```

- [ ] Continue monitoring the next queued VSPM sensitivity job.
- [ ] Continue monitoring next MST variant.

MST queue:

- [ ] `baseline`
- [ ] `gru_fp16`
- [ ] `ds64_fp16`
- [ ] `bottleneck12`

### After VSPM Sensitivity Finishes

- [ ] Confirm all sensitivity runs completed:
  - `T5_n5`
  - `T10_n5`
  - `T20_n5`
  - `T30_n5`
  - `T10_n3`
  - `T10_n10`
  - `T10_n15`
- [ ] Summarize best/final IoU, Delta IoU, Dynamic IoU for each run.
- [ ] Decide which sensitivity results are paper-ready and which are only preliminary.
- [ ] Monitor MST ablation stage.

### After MST Ablation Finishes

- [ ] Summarize baseline / GRU FP16 / DS64 FP16 / bottleneck12 performance.
- [ ] Combine MST performance with state-size and latency profile.
- [ ] Prepare reviewer-facing table for:
  - performance
  - bandwidth/state memory
  - computation latency

## Known Constraints

- Current workspace does not contain the full detection/AP pipeline used by the original paper tables.
- Current workspace does not contain an additional real-world dataset adapter.
- Current full pipeline is training occupancy forecasting VSPM components, not full cooperative detection AP baselines.
- CPU robustness evaluation is useful for non-ideal VSPM evidence but is not a replacement for full detection robustness experiments.

## Useful Commands

Monitor active process:

```bash
cd /raid/xuyifan/v2x_code_ckpt
ps -p "$(cat results/revision_plan/full_pipeline.pid)" -o pid,ppid,stat,etime,pcpu,pmem,cmd
```

Refresh status:

```bash
cd /raid/xuyifan/v2x_code_ckpt
PYTHONPATH=src python scripts/report_pipeline_status.py
```

Refresh tables and figures:

```bash
cd /raid/xuyifan/v2x_code_ckpt
python scripts/make_revision_tables.py
MPLBACKEND=Agg python scripts/plot_revision_tables.py
```

Run CPU robustness evaluation from latest `T5_n5` checkpoint:

```bash
cd /raid/xuyifan/v2x_code_ckpt
DEVICE=cpu MAX_SAMPLES=256 bash scripts/run_vspm_robustness_eval.sh
```

Stop full pipeline deliberately:

```bash
cd /raid/xuyifan/v2x_code_ckpt
kill "$(cat results/revision_plan/full_pipeline.pid)"
```

## Update Policy

Maintain this document as the running source of truth:

- Update `Current Status` when the active process, active run, metric, or checkpoint changes.
- Append to `Work Completed In This Session` when a new script, result, table, or figure is created.
- Move queue items forward as they are completed.
- Add blockers immediately when a command fails or an expected artifact is missing.
