# Curated Checkpoints

This directory contains the small, project-owned checkpoints that are most
useful for restoring the paper/revision experiments without uploading the full
local training workspaces.

## Selection Policy

- Keep checkpoints that correspond to reported V2X-Sim and DAIR-V2X summary
  tables.
- Keep best or nearest saved checkpoints for leave-one-agent, ablation, DAIR
  system, and DAIR sensitivity experiments.
- Do not vendor external official checkpoints from OpenDAIRV2X, DATA/OpenCOOD,
  FFNet, or other upstream projects.
- Do not include logs, raw predictions, datasets, or temporary process files.

## Contents

- `v2xsim/all_agents/`: all-agent V2X-Sim model.
- `v2xsim/leave_one_out/`: five held-out-agent generalization checkpoints.
- `v2xsim/ablation1/`: first V2X-Sim ablation group.
- `v2xsim/ablation2/`: second V2X-Sim ablation group.
- `dair_v2x/our_method/`: DAIR-V2X VSPM checkpoints used by the delay-grid
  summaries.
- `dair_v2x/mst_ablation/`: DAIR-V2X MST/system ablation checkpoints.
- `dair_v2x/sensitivity/`: DAIR-V2X history/horizon sensitivity checkpoints.

## Restore Legacy Paths

The repository keeps these checkpoints in a stable curated directory. To copy
them back to the ignored legacy paths expected by older scripts, run:

```bash
bash scripts/restore_curated_checkpoints.sh
```

This restores only curated checkpoints. It does not restore logs, raw datasets,
third-party repositories, prediction dumps, or wandb runs.
