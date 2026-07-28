# 260704 GitHub Artifact Reproducibility Audit

Audit date: 2026-07-14

Remote branch: `origin/workspace-upload-20260622`

Uploaded commit: `660a647edaca289ab213222ceaa612dddc5e88b4`

Artifact root:

```text
release_packages/260704_experiment_artifacts/
```

## 1. What Is Already On GitHub

The current GitHub branch contains a curated 260704 artifact package. It is organized by dataset, baseline, and experiment type.

| Category | Status | Main Paths |
|---|---:|---|
| Final 260704 report | Available | `00_docs/260704_results.md`, `00_docs/260704_experiment_results.docx` |
| File manifest | Available | `MANIFEST.csv` |
| Missing-file manifest | Available, no missing files | `MANIFEST_MISSING.csv` |
| Large-cache exclusion list | Available | `MANIFEST_EXCLUDED_LARGE_CACHE.csv` |
| Checkpoint split manifest | Available | `CHECKPOINT_SPLIT_MANIFEST.csv` |
| Checkpoint restore script | Available | `restore_split_checkpoints.py` |
| Code snapshot records | Available | `00_code_version/` |
| Experiment scripts used in 260704 | Available | `00_code_scripts/scripts/` |
| Final CSV results and logs | Available | dataset/baseline result folders |
| Ours and baseline checkpoints used by final tables | Available, some split | dataset/baseline checkpoint folders |

Current package statistics:

| Item | Value |
|---|---:|
| Manifest records | 318 |
| Package files | 319 |
| Split checkpoint files | 10 original checkpoints split into parts |
| Maximum single uploaded file size | 41,943,040 bytes |
| Missing files recorded by packaging | 0 |

Large checkpoints were split into `*.split/*.partNNN` files to avoid GitHub single-file limits and avoid relying on Git LFS. Restore them with:

```bash
cd release_packages/260704_experiment_artifacts
python restore_split_checkpoints.py
```

## 2. Uploaded Content By Experiment

| Dataset | Baseline | Uploaded Contents |
|---|---|---|
| V2X-Sim | V2VNet | DLPCM paper table source and `v2xsim_ours_prediction_model.pth` |
| V2X-Sim | DATA | DATA baseline config/checkpoint, Ours joint IFAM checkpoints, result CSVs, fairness audit CSVs |
| V2X-Sim | DiscoNet | DLPCM paper table source and `v2xsim_ours_prediction_model.pth` |
| DAIR-V2X-C | Official late fusion / TCLF | No-comp, Ours RF, train+val diagnostic, old RF diagnostic, TCLF CSVs/logs, RF checkpoints |
| DAIR-V2X-C | DATA | DATA baseline checkpoint/config, IFAM T10-N10 and joint delay-comp checkpoints, strict result CSVs, box-level/RF diagnostics |
| DAIR-V2X-C | LRCP | LRCP stage2 checkpoint/config, strict dual-delay merged CSV, per-cell CSV/log shards |
| DAIR-V2X-Seq | TraF-Align | Official checkpoint/config, Ours T10-N10 feature predictor, joint delay-comp checkpoints/configs, baseline and Ours result CSVs |

## 3. What Can Be Reproduced From GitHub Alone

The current GitHub contents are enough for result provenance audit:

| Reproduction Level | Current Status | Explanation |
|---|---|---|
| Verify table numbers against final CSVs | Mostly supported | Final CSVs and `260704_results.md` are uploaded. |
| Verify which checkpoint produced which result | Mostly supported | `MANIFEST.csv`, `CHECKPOINT_SPLIT_MANIFEST.csv`, and checkpoint files are uploaded. |
| Restore uploaded checkpoints exactly | Supported | `restore_split_checkpoints.py` verifies original size and SHA256. |
| Re-run final inference from scratch | Not yet supported from GitHub alone | Requires external baseline code trees, datasets, and environment setup not included in GitHub. |
| Re-train Ours models from scratch | Not yet supported from GitHub alone | Requires datasets, external baseline code, feature caches or cache generation steps, and locked environments. |

## 4. Missing Conditions For Full Reproduction

### 4.1 External Baseline Code Trees

The uploaded package includes our wrapper/training/evaluation scripts, but not the full external codebases they import or execute.

| Needed Code | Required By | Current GitHub Status | Missing Condition |
|---|---|---|---|
| `external/DATA` OpenCOOD/DATA fork | V2X-Sim DATA, DAIR-V2X-C DATA | Not uploaded | Need exact fork commit or patch bundle. |
| `external/LRCP` | DAIR-V2X-C LRCP | Not uploaded | Need exact LRCP commit plus local DAIR adapter patches if any. |
| `external/DAIR-V2X-isolated/gpu*/v2x` | DAIR official late fusion / TCLF | Not uploaded | Need exact OpenDAIRV2X code copy or patchable commit. |
| `/tmp/TraF-Align_partial` | DAIR-V2X-Seq TraF-Align | Not uploaded | Need exact TraF-Align repo snapshot and local modifications. |
| Original V2VNet/DiscoNet/SyncNet code | V2X-Sim paper-result reproduction | Not uploaded | Current package only stores paper table sources, not rerunnable code. |

This is the largest reproducibility gap. Without these code trees, the uploaded scripts cannot be executed directly even though results and checkpoints are present.

### 4.2 Datasets

Datasets are not uploaded, which is expected because of size/licensing. They must be downloaded and placed in the expected local layout.

| Dataset | Required By | Expected Local Layout / Notes |
|---|---|---|
| V2X-Sim / V2X-Sim 2.0 | V2VNet, DATA, DiscoNet experiments | `V2X-Sim-det/` or `datasets/V2X-Sim-2.0/` depending on script/baseline |
| DAIR-V2X-C | Official late fusion, DATA, LRCP | `datasets/DAIR-V2X/cooperative-vehicle-infrastructure/` |
| DAIR-V2X-Seq / SPD | TraF-Align | `datasets/DAIR-V2X-Seq/V2X-Seq-SPD/` |
| DAIR split files | DAIR official, LRCP, RF training | `external/DAIR-V2X/data/split_datas/cooperative-split-data.json` or equivalent |

The package contains download/helper scripts for DAIR-V2X-Seq and some preparation scripts, but not a complete verified end-to-end data setup script for all baselines.

### 4.3 Python/CUDA Environments

The root `requirements.txt` only covers the early V2X BEV forecasting code. It is not sufficient for the 260704 detection baselines.

Missing environment locks:

| Environment | Required By | Known Requirement / Gap |
|---|---|---|
| DATA/OpenCOOD env | DATA on V2X-Sim and DAIR | Need exact PyTorch, CUDA, OpenCOOD dependencies, spconv version, pyyaml/scipy stack. |
| LRCP env | DAIR LRCP | Need exact LRCP/OpenCOOD dependency versions. |
| OpenDAIRV2X official env | DAIR official late fusion / TCLF | Need exact detector environment and CUDA dependencies. |
| TraF-Align env | DAIR-V2X-Seq TraF-Align | Must use compatible `spconv-cu113==2.1.21` and `cumm-cu113==0.2.9`; current notes say newer `spconv 2.3.6 / cumm 0.4.11` reproduced near-zero AP. |

Recommended fix: add one `environment.yml` or `requirements-lock.txt` per baseline environment, plus `python --version`, `torch`, `cuda`, `spconv`, `cumm`, `numpy`, `scipy`, `pyyaml`, and GPU driver notes.

### 4.4 Exact Run Commands

The package includes scripts, but it does not yet include a single ordered runbook that maps every table row to a command.

Needed runbook fields:

| Field | Why It Matters |
|---|---|
| Dataset path variables | Avoid hidden local absolute paths. |
| External repo path variables | Current scripts use paths such as `external/DATA` and `/tmp/TraF-Align_partial`. |
| Checkpoint restore commands | Split checkpoints must be restored before use. |
| Per-baseline inference commands | Needed to regenerate each CSV. |
| Training commands for Ours | Needed for from-scratch reproduction of each Ours checkpoint. |
| Expected output CSV names | Needed to compare regenerated outputs with uploaded files. |
| Approximate GPU/time requirements | Needed to plan full reruns. |

### 4.5 Intermediate Caches

The following large caches were intentionally not uploaded:

| Path | Reason |
|---|---|
| `results/260615_fill/cache/dair_data_feature_t10n10/` | About 310G; feature cache, not final result CSV/checkpoint. |
| `results/dair_v2x_seq/trafalign_feature_cache_spconv2121/` | About 31G; feature cache, regenerable. |
| `results/260615_fill/cache/dair_data_intermediate_boxes_20260707/` | Intermediate box cache for diagnostics. |
| `results/260615_fill/cache/dair_data_rf_ours_20260706/` | Intermediate RF cache for diagnostics. |

This is acceptable for GitHub upload, but full retraining/rerun needs documented cache regeneration commands and expected cache schema.

### 4.6 Local Patches And Dirty Workspace State

The artifact package stores `00_code_version/working_tree_diff.patch`, but external repositories are ignored and their local patches are not guaranteed to be captured.

Missing:

- Patch files for `external/DATA`.
- Patch files for `external/LRCP`.
- Patch files for `external/DAIR-V2X-isolated`.
- Patch files for TraF-Align.
- Exact upstream commit hashes for each external repo.

Without these, another machine may clone the upstream repositories but still fail to match the uploaded results.

## 5. Priority Fixes

| Priority | Fix | Impact |
|---:|---|---|
| 1 | Add exact external repo commit hashes and local patch bundles for DATA, LRCP, OpenDAIRV2X, and TraF-Align. | Converts scripts from local-only to portable. |
| 2 | Add per-baseline environment lock files. | Prevents dependency drift, especially spconv/cumm issues. |
| 3 | Add `260704_REPRODUCE.md` with ordered commands from data preparation to CSV regeneration. | Makes the artifact package usable by someone else. |
| 4 | Add dataset layout checks for all baselines. | Reduces silent path/split errors. |
| 5 | Add cache regeneration commands and schema notes for omitted caches. | Enables from-scratch training of feature-level methods. |
| 6 | Add a compact validation script that checks restored checkpoints, required paths, and expected output CSV hashes. | Makes reproduction audit automatic. |

## 6. Bottom Line

Current GitHub content is sufficient for a provenance package: it records the final results, exact uploaded checkpoints, scripts used, logs, and SHA256 manifests.

It is not yet sufficient for full independent reproduction from a fresh clone. The missing pieces are mainly:

1. exact external baseline code snapshots or patches,
2. datasets and split files,
3. locked Python/CUDA environments,
4. a command-level runbook for each baseline/table,
5. regeneration instructions for large omitted caches.

The next concrete step should be to add external repo commit/patch manifests and per-baseline environment files before trying to advertise the GitHub repository as fully reproducible.
