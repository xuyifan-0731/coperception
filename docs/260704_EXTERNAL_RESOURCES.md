# 260704 External Resources And Missing Conditions

This document lists the external datasets, baseline code, external checkpoints, environments, and generated-data requirements for reproducing the 260704 result package.

Artifact package:

```text
release_packages/260704_experiment_artifacts/
```

Current uploaded branch:

```text
origin/workspace-upload-20260622
```

## 1. How The Uploaded Files Are Organized

The current artifact package is already organized by the requested hierarchy:

```text
release_packages/260704_experiment_artifacts/
  00_docs/                         # reports and documentation
  00_reproduction_assets/           # small reproduction assets: patches, split files, env refs
  00_code_scripts/                 # scripts used by the experiments
  00_code_version/                 # git state when the package was created
  01_v2xsim/
    V2VNet/
    DATA/
    DiscoNet/
  02_dair_v2x_c/
    official_late_fusion/
    DATA/
    LRCP/
  03_dair_v2x_seq/
    TraF-Align/
```

Each dataset/baseline folder contains the relevant subset of:

```text
checkpoints/   # baseline and ours checkpoints, with large files split if needed
results/       # final CSV results and raw per-cell CSV/logs where available
logs/          # full evaluation/training logs for DAIR official late fusion
paper_sources/ # paper table source files for results imported from DLPCM
```

Use these package-level indexes:

| File | Role |
|---|---|
| `MANIFEST.csv` | Authoritative list of uploaded files with SHA256 and source paths. |
| `MANIFEST_MISSING.csv` | Missing-file check from packaging; currently only the header, meaning no missing files. |
| `CHECKPOINT_SPLIT_MANIFEST.csv` | Original checkpoint path, size, SHA256, and split parts. |
| `MANIFEST_EXCLUDED_LARGE_CACHE.csv` | Large generated caches intentionally excluded from GitHub. |
| `restore_split_checkpoints.py` | Rebuilds split checkpoints and verifies SHA256. |
| `00_docs/260704_REPRODUCTION_CHECKLIST.md` | Newcomer checklist for preparing datasets, external code, patches, environments, and reruns. |
| `00_reproduction_assets/README.md` | Explains uploaded small reproduction assets. |

Restore split checkpoints before rerunning experiments:

```bash
cd release_packages/260704_experiment_artifacts
python restore_split_checkpoints.py
```

## 2. Public Datasets

| Dataset | Used In | Public Source | Required Content / Version | Expected Local Layout | Current GitHub Status |
|---|---|---|---|---|---|
| V2X-Sim 2.0 | V2X-Sim DATA; V2VNet/DiscoNet paper-table references | `https://ai4ce.github.io/V2X-Sim/download.html` | Select the V2X-Sim 2.0 full dataset. LRCP README describes the decompressed root as `v2xsim2/` with `lidarseg`, `maps`, `sweeps`, `v1.0-mini`. | DATA/OpenCOOD scripts use local dataset/info paths such as `datasets/v2xsim2_info/` and DATA model configs. | Dataset not uploaded. `v2xsim2_info` summary is uploaded, but `.pkl` files are not uploaded because they are large generated metadata. |
| DAIR-V2X-C | DAIR official late fusion/TCLF; DAIR DATA; DAIR LRCP | `https://thudair.baai.ac.cn/index`; code/docs at `https://github.com/AIR-THU/DAIR-V2X` | DAIR-V2X cooperative vehicle-infrastructure data. Official README also points to full dataset via Public-V2X-Datasets Google Drive. | `datasets/DAIR-V2X/cooperative-vehicle-infrastructure/` plus split file under `external/DAIR-V2X/data/split_datas/`. | Dataset not uploaded. Split JSON files are uploaded under `00_reproduction_assets/split_files/dair_v2x/`. |
| DAIR-V2X-Seq / V2X-Seq-SPD | TraF-Align baseline and ours | `https://thudair.baai.ac.cn/index`; DAIR-V2X README links V2X-Seq-SPD example and full Public-V2X-Datasets Drive; code at `https://github.com/AIR-THU/DAIR-V2X` and `https://github.com/AIR-THU/DAIR-V2X-Seq` | V2X-Seq-SPD sequential perception dataset. | `datasets/DAIR-V2X-Seq/V2X-Seq-SPD/`; TraF-Align split file under `/tmp/TraF-Align_partial/datasets/Basedataset/V2XSeq_dataset_split_official.yaml`. | Dataset not uploaded. Official split YAML is uploaded under `00_reproduction_assets/split_files/trafalign/`. |

Dataset helper scripts included in the package:

| Script | Purpose |
|---|---|
| `00_code_scripts/scripts/download_dair_v2x_seq_spd.py` | Lists/downloads DAIR-V2X-Seq SPD files from the public Drive folder. |
| `00_code_scripts/scripts/extract_dair_v2x_seq_spd_parts.sh` | Extracts downloaded DAIR-V2X-Seq parts. |
| `00_code_scripts/scripts/monitor_dair_v2x_seq_download.sh` | Monitors DAIR-V2X-Seq download status. |
| `00_code_scripts/scripts/prepare_dair_lrcp_inputs.py` | Builds the DAIR layout expected by LRCP. |
| `00_code_scripts/scripts/generate_dair_lrcp_flow_offsets.py` | Generates LRCP DAIR BEV flow-offset labels when official flow files are not present. |
| `00_code_scripts/scripts/prepare_trafalign_dair_compat.py` | Builds a DAIR-V2X-C compatibility view for TraF-Align diagnostics; not a formal main result. |

Small dataset-side reproduction files now included:

| File / Folder | Role |
|---|---|
| `00_reproduction_assets/split_files/dair_v2x/*.json` | DAIR-V2X split files used by OpenDAIRV2X and related scripts. |
| `00_reproduction_assets/split_files/trafalign/V2XSeq_dataset_split_official.yaml` | TraF-Align official V2X-Seq split file used locally. |
| `00_reproduction_assets/dataset_info/v2xsim2_info/summary.json` | Summary of the local V2X-Sim info files. The large `.pkl` files are intentionally excluded. |

## 3. Public Baseline Code And Local Patch Status

| Baseline / Codebase | Public Repository | Local Commit Used | Local Changes? | Required For | GitHub Upload Status |
|---|---|---:|---|---|---|
| DATA | `https://github.com/ChengchangTian/DATA` | `5df7eb6f5659db0d6809fa3cc218aa425bc287b4` | Yes: 5 modified files, including DAIR/V2XSim dataset loaders, fusion module, inference. | V2X-Sim DATA and DAIR-V2X-C DATA. | Patch uploaded: `00_reproduction_assets/patches/DATA_local_changes.patch`. Full external repo is not vendored. |
| LRCP | `https://github.com/JesseWong333/LRCP` | `1eb3793b9befd9d36dd8705e461d895a41b39d33` | Yes: 6 modified files and added DAIR hypes directory. | DAIR-V2X-C LRCP strict dual-delay. | Patch uploaded: `00_reproduction_assets/patches/LRCP_local_changes.patch`; new DAIR hypes yaml uploaded under `patches/extra_files/LRCP/`. Full external repo is not vendored. |
| OpenDAIRV2X / DAIR-V2X | `https://github.com/AIR-THU/DAIR-V2X` | `c885c54af0c34bc515fa9ca8b5e8fda76a15462c` | Yes: 6 modified files, including late-fusion detector/eval scripts. | DAIR official late fusion/TCLF and RF cache generation. | Patch uploaded: `00_reproduction_assets/patches/OpenDAIRV2X_local_changes.patch`. Full external repo is not vendored. |
| DAIR-V2X isolated copies | Derived from OpenDAIRV2X | no `.git` in `external/DAIR-V2X-isolated/gpu*/v2x` | Local copies used for multi-GPU runs. | DAIR official late fusion/TCLF. | Not uploaded. Need recreate from patched OpenDAIRV2X or copy local tree. |
| TraF-Align | `https://github.com/zhyingS/TraF-Align` | `0f6f98de60dcb86f29f4eb99ed871054b05bf280` | Yes: 2 modified files and new `models/modules/deform/joint_delay_compensation.py`. | DAIR-V2X-Seq TraF-Align baseline and ours joint delay-comp. | Patch uploaded: `00_reproduction_assets/patches/TraFAlign_local_changes.patch`; new module uploaded under `patches/extra_files/TraF-Align/`. Full external repo is not vendored. |
| V2VNet / DiscoNet / SyncNet | V2X-Sim/DLPCM paper comparison sources; DiscoNet public code exists at `https://github.com/ai4ce/DiscoNet` | not rerun in this package | Not applicable for current package. | V2X-Sim V2VNet/DiscoNet paper-table rows. | Only DLPCM paper table sources are uploaded; not a full rerun package for these baselines. |

Important: public upstream repos still need to be cloned separately, but the local patch bundles and small added files are now included in `00_reproduction_assets/patches/`.

## 4. External Checkpoints

| Dataset / Baseline | Checkpoint | Public Status | Current Package Status | Notes |
|---|---|---|---|---|
| V2X-Sim / DATA | DATA baseline `net_epoch_bestval_at19.pth` | DATA README says model code and pretrained weights are available, but exact local checkpoint provenance should be treated as local unless verified against a public artifact. | Uploaded as split parts under `01_v2xsim/DATA/checkpoints/baseline_DATA/`. | Restore with `restore_split_checkpoints.py`. |
| DAIR-V2X-C / DATA | DATA baseline `net_epoch_bestval_at21.pth` | DATA README reports pretrained weights available; exact local checkpoint reproduced lower than paper table, so treat as local available checkpoint/config. | Uploaded as split parts under `02_dair_v2x_c/DATA/checkpoints/baseline_DATA/`. | This is the local LiDAR-only DATA config used in 260704. |
| DAIR-V2X-C / LRCP | LRCP stage2 `net_epoch_bestval_at1.pth` | LRCP repo documents training; public DAIR flow files are linked, but this exact stage2 checkpoint is local. | Uploaded as split parts under `02_dair_v2x_c/LRCP/checkpoints/baseline_LRCP_stage2/`. | Used only as a local LRCP pipeline baseline. |
| DAIR-V2X-Seq / TraF-Align | Official `net_epoch60.pth` | TraF-Align README says pretrained V2V4Real and V2X-Seq checkpoints are provided in `checkpoints/`. | Uploaded as split parts under `03_dair_v2x_seq/TraF-Align/checkpoints/baseline_official_trafalign/`. | Official checkpoint reproduced 0ms/400ms within expected range only under compatible spconv/cumm env. |
| DAIR official late fusion/TCLF | Official detector weights/caches | OpenDAIRV2X code is public; this package uses local isolated detector pipelines and generated detector caches. | Final logs/results and Ours RF checkpoints uploaded; original external official detector code/checkpoint not fully packaged. | Exact rerun needs patched OpenDAIRV2X environment. |

## 5. Our Checkpoints And Whether They Need Uploading

| Ours / Diagnostic Model | Used In 260704? | Package Status | Regeneration Difficulty | What To Do |
|---|---:|---|---|---|
| `v2xsim_ours_prediction_model.pth` | V2X-Sim V2VNet/DiscoNet paper-row model reference | Uploaded | Historical DLPCM model; not rerun in this package. | Keep uploaded as provenance. |
| V2X-Sim DATA joint IFAM residual scale01 | Yes | Uploaded | Needs DATA pipeline and joint residual training; not just a trivial rerun. | Keep uploaded and document training command. |
| DAIR official late-fusion RF 2f train-only | Yes, main strict DAIR result | Uploaded as split parts | Needs train-split detector raw cache and strict train->val setup. | Keep uploaded and document training/eval command. |
| DAIR official late-fusion RF train+val diagnostic | Diagnostic upper-bound | Uploaded as split parts | Same mechanism; intentionally not fair as paper main result. | Keep marked as diagnostic only. |
| Old DAIR RF val->val | Diagnostic/leakage reference | Uploaded as split parts | Not fair for final result. | Keep only for audit; do not use as main. |
| DAIR DATA IFAM feature T10-N10 | Diagnostic | Uploaded as tiny `.npz` | Linear/ridge predictor is easy once 310G cache exists; cache generation is expensive. | Upload is acceptable; can also regenerate from cache. |
| DAIR DATA IFAM joint delay-comp | Yes, positive DATA result | Uploaded | Requires special joint gated residual training on DATA IFAM features. | Keep uploaded and document command. |
| TraF-Align feature T10-N10 | Diagnostic failure | Uploaded as tiny `.npz` | Easy once feature cache exists; cache generation is expensive. | Keep as diagnostic. |
| TraF-Align joint delay-comp / adaptive / bypass | Yes, recommended TraF-Align + Ours result | Uploaded as split parts/configs | Requires patched TraF-Align module, frozen original model, delay-adaptive inference policy. | Keep uploaded and document command. |
| Smoke / temporary checkpoints | No | Not included in the package | Easy to regenerate; not part of final results. | Do not upload. |

## 6. Environment Requirements

The root `requirements.txt` is not enough for 260704 detection experiments. Use separate environments per baseline.

### 6.1 DATA / OpenCOOD Environment

Public DATA repo provides `requirements.txt` with packages including `cmake`, `easydict`, `numba`, `opencv-python-headless`, `open3d`, `tensorboardX`, `shapely==2.0.0`, `einops`, `timm`, `h5py`, `pyquaternion`, `efficientnet_pytorch`, `python-lzf`, and `pypcd`.

Required local condition:

```bash
cd external/DATA
pip install -r requirements.txt
python setup.py develop
python opencood/utils/setup.py build_ext --inplace
python opencood/pcdet_utils/setup.py build_ext --inplace
```

Reference file uploaded: `00_reproduction_assets/env_refs/DATA/requirements.txt`.

Remaining gap: exact Python, PyTorch, CUDA, and spconv versions used by the local DATA reruns are not yet locked.

### 6.2 LRCP Environment

LRCP README recommends:

```bash
conda create --name LRCP python=3.8
conda activate LRCP
conda install pytorch==2.0.0 torchvision==0.15.1 cudatoolkit=11.8 -c pytorch -c conda-forge
pip install -r requirements.txt
python setup.py develop
python opencood/utils/setup.py build_ext --inplace
python opencood/pcdet_utils/setup.py build_ext --inplace
cd opencood/utils/ms_deform_attn_ops && sh ./make.sh
```

It also asks for `spconv 1.2.1`.

Reference files uploaded:

```text
00_reproduction_assets/env_refs/LRCP/requirements.txt
00_reproduction_assets/env_refs/LRCP/environment.yml
```

Remaining gap: exact local DAIR adapter dependencies and compiled extension versions are not exported as a lock file.

### 6.3 OpenDAIRV2X Official Late-Fusion Environment

OpenDAIRV2X code is public at `https://github.com/AIR-THU/DAIR-V2X`, but the official late-fusion reruns used patched local detector code and isolated GPU copies.

Required local condition:

```text
external/DAIR-V2X-isolated/gpu*/v2x
```

Gap: provide exact install commands, mmdet3d/mmcv/torch/cuda versions, and patch bundle for the local late-fusion changes.

### 6.4 TraF-Align Environment

TraF-Align README recommends Python 3.9.13, PyTorch 2.1.2, CUDA 12.1 wheels, and spconv-cu111. However, the successful 260704 reproduction used a compatible isolated spconv stack:

```bash
# local successful replay setting
PYTHONPATH=/tmp/trafalign_spconv_2121:$PYTHONPATH
spconv-cu113==2.1.21
cumm-cu113==0.2.9
```

This matters: newer `spconv 2.3.6 / cumm 0.4.11` produced near-zero AP for the official TraF-Align checkpoint in local testing.

Reference file uploaded: `00_reproduction_assets/env_refs/TraF-Align/requirements.txt`.

## 7. Large Generated Data Not Uploaded

| Generated Data | Size / Reason | Regeneration Script |
|---|---|---|
| DAIR DATA IFAM feature cache | About 310G; needed for DAIR DATA feature T10-N10 training/eval. | `scripts/run_dair_data_feature_t10n10_comp.py cache-features` |
| TraF-Align feature cache | About 31G; needed for TraF-Align feature T10-N10 predictor. | `scripts/train_trafalign_feature_t10n10_predictor.py cache-features` |
| DAIR DATA intermediate box cache | Intermediate diagnostic cache. | `scripts/run_dair_data_hybrid_rf_assist.py --cache-only` |
| DAIR DATA RF box cache | Intermediate diagnostic cache. | `scripts/run_dair_data_rf_ours_box_comp_grid.py --force-cache` |
| LRCP DAIR flow offsets | Generated from local labels when official DAIR flow archive is absent. | `scripts/generate_dair_lrcp_flow_offsets.py` |

These are not final result files and are too large for GitHub. They should be regenerated from datasets and uploaded checkpoints when a full rerun is required.

## 8. Remaining Missing Items To Reach Full Reproducibility

| Priority | Missing Item | Why It Is Needed |
|---:|---|---|
| 1 | Full environment lock files per baseline | Prevents dependency drift and spconv/cumm mismatch. Current package has requirements references, not exact lock files. |
| 2 | Dataset archive checksum manifests and post-extract file counts | Avoids silent dataset version/path mismatches. |
| 3 | V2X-Sim info `.pkl` generation details | The `.pkl` files are large and not uploaded; only summary is included. |
| 4 | Runtime path materialization script | Package paths are archival; run commands still expect historical runtime paths. |
| 5 | Cache regeneration validation | Needed for large omitted feature/box caches. |
