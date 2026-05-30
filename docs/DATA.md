# Data Notes

## Expected Dataset

The experiments use V2X-Sim BEV occupancy data stored as sparse `.npy` files.
Each sample folder is named `<scene>_<frame>` and contains `0.npy`, where the
loaded dictionary includes `voxel_indices_0`.

Expected local layout:

```text
V2X-Sim-det/
  train/
    agent1/
    agent2/
    agent3/
    agent4/
    agent5/
  test/
    agent1/
    agent2/
    agent3/
    agent4/
    agent5/
  val/
    agent1/
    agent2/
    agent3/
    agent4/
    agent5/
```

`agent0` also exists in the local dataset, but the curated experiment scripts
use agents 1 through 5 to match the recorded runs.

## Local Dataset Size

The local workspace currently contains about 363 GB under `V2X-Sim-det/` plus a
4.9 GB archive. These files are ignored and should not be uploaded to a normal
Git remote.

## Path Configuration

All shell entry points use:

```bash
DATA_ROOT=/workspace/xuyifan/v2x/V2X-Sim-det
```

by default. Use a different dataset location with:

```bash
export DATA_ROOT=/path/to/V2X-Sim-det
```

## Generated Artifacts

Training creates checkpoints and logs under `RUN_ROOT`; evaluation creates
prediction `.npy` files and a `metrics.csv`. These are generated artifacts. Keep
them outside Git, then summarize them with:

```bash
python scripts/collect_results.py --root model1 --out results/summary --copy-raw-metrics
```

