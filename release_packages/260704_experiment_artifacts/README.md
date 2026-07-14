# 260704 Experiment Artifacts

This package organizes all checkpoints and final original result files referenced by `260704结果.md` by dataset, baseline, and experiment type.

Directory layout:

- `00_docs/`: final markdown and Word report.
- `00_reproduction_assets/`: small reproduction assets: external-code patches, split files, dataset-info summaries, and environment reference files.
- `00_code_version/`: Git HEAD, branch, status, diff, and tracked-file list at packaging time.
- `00_code_scripts/`: scripts referenced by the 260704 experiments.
- `01_v2xsim/`: V2X-Sim artifacts, grouped by V2VNet, DATA, and DiscoNet.
- `02_dair_v2x_c/`: DAIR-V2X-C artifacts, grouped by official late fusion/TCLF, DATA, and LRCP.
- `03_dair_v2x_seq/`: DAIR-V2X-Seq TraF-Align artifacts.

Use `MANIFEST.csv` as the authoritative file list. It records source path, packaged path, file size, SHA256, dataset, baseline, experiment, and role.

Large intermediate caches are intentionally not included. See `MANIFEST_EXCLUDED_LARGE_CACHE.csv`; these caches are not final result data and are too large for a GitHub artifact package.

The DAIR-V2X-C TraF-Align compatibility smoke directory is also not copied in full; only its two debug CSV files are retained because it is not a formal 260704 result.

Checkpoint files larger than 45MB are split into `*.split/*.partNNN` files for plain GitHub upload without Git LFS. Run `python restore_split_checkpoints.py` from this package root to reconstruct the exact original checkpoint files; original sizes and SHA256 hashes are recorded in `CHECKPOINT_SPLIT_MANIFEST.csv`.

Additional reproduction documentation:

- `00_docs/260704_ARTIFACT_INDEX.md`: beginner-facing Chinese directory map, artifact index, and remaining reproduction gaps.
- `00_docs/260704_REPRODUCTION_CHECKLIST.md`: Chinese newcomer checklist for preparing data, external code, patches, environments, checkpoints, and reruns.
- `00_docs/260704_EXTERNAL_RESOURCES.md`: public datasets, external baseline code/checkpoints, environment requirements, and missing conditions.
- `00_docs/260704_RUNBOOK.md`: command-level runbook organized by dataset, baseline, and experiment.
- `00_reproduction_assets/README.md`: explains the packaged patch bundles, split files, dataset summaries, and environment references.
