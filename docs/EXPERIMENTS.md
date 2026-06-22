# Experiment Results

All numbers below come from the generated CSV files in `results/summary/`.
`Delta_IoU` is model IoU minus the baseline IoU from the last observed BEV
frame.

## Leave-One-Agent-Out Generalization

These runs train on four agents and validate on the held-out agent.

| Held-out agent | Run | Best step | Best IoU | Best Delta IoU | Best Dynamic IoU |
|---:|---|---:|---:|---:|---:|
| 1 | `checkpoints_2345_for_Agent1` | 27000 | 0.7491 | 0.0576 | 0.3287 |
| 2 | `checkpoints_1345_for_Agent2` | 33000 | 0.7400 | 0.0622 | 0.3455 |
| 3 | `checkpoints_1245_for_Agent3` | 35250 | 0.7553 | 0.0623 | 0.3645 |
| 4 | `checkpoints_1235_for_Agent4` | 33000 | 0.7972 | 0.0604 | 0.3930 |
| 5 | `checkpoints_1234_for_Agent5` | 44000 | 0.7433 | 0.0641 | 0.3673 |

The exported prediction metrics for `result_2345_for_Agent1-20000` average over
all available scenes and forecast intervals:

| Agent | Samples | Mean IoU | Baseline IoU | Delta IoU |
|---|---:|---:|---:|---:|
| agent1 | 6480 | 0.7199 | 0.6666 | 0.0533 |
| agent2 | 6480 | 0.7173 | 0.6558 | 0.0615 |
| agent3 | 6480 | 0.7336 | 0.6716 | 0.0620 |
| agent4 | 4536 | 0.7675 | 0.7104 | 0.0571 |
| agent5 | 1944 | 0.7116 | 0.6502 | 0.0614 |

For agent1, the exported checkpoint at step 30000 is slightly higher than step
20000: mean IoU 0.7222 vs. 0.7199.

## All-Agent Baseline

The `checkpoints_A` run trains on all five agents with `T=20`, `n=10`. Its best
logged validation point is:

| Run | Best step | Best IoU | Best Delta IoU | Best Dynamic IoU |
|---|---:|---:|---:|---:|
| `checkpoints_A` | 38000 | 0.7338 | 0.0622 | 0.3470 |

## Ablation 1

The first ablation group compares variants under `消融实验-1-baseline`.

| Run | Best step | Best IoU | Best Delta IoU | Best Dynamic IoU |
|---|---:|---:|---:|---:|
| `checkpoints_1_for_Agent1_A` | 13000 | 0.7487 | 0.0571 | 0.3385 |
| `checkpoints_1_for_Agent1_B` | 13000 | 0.7460 | 0.0544 | 0.3250 |
| `checkpoints_1_for_Agent1_C` | 12000 | 0.7448 | 0.0532 | 0.3305 |
| `checkpoints_1_for_Agent1_D` | 13000 | 0.7440 | 0.0524 | 0.3647 |
| `checkpoints_1_for_Agent1_baseline` | 13050 | 0.7480 | 0.0564 | 0.3303 |

## Ablation 2

The second ablation group compares variants under `消融实验-2-A`.

| Run | Best step | Best IoU | Best Delta IoU | Best Dynamic IoU |
|---|---:|---:|---:|---:|
| `checkpoints_1_for_Agent1_A1` | 13000 | 0.7487 | 0.0571 | 0.3400 |
| `checkpoints_1_for_Agent1_A2` | 13000 | 0.7436 | 0.0520 | 0.3529 |
| `checkpoints_1_for_Agent1_A3` | 13050 | 0.7452 | 0.0536 | 0.3333 |
| `checkpoints_1_for_Agent1_A4` | 13000 | 0.7409 | 0.0493 | 0.2844 |
| `checkpoints_1_for_Agent1_ABaseline` | 13000 | 0.7487 | 0.0571 | 0.3385 |

## Files

- `results/summary/training_summary.csv`: table source for validation summaries.
- `results/summary/training_timeseries.csv`: full parsed validation curves.
- `results/summary/eval_overall.csv`: exported prediction averages by agent.
- `results/summary/eval_by_interval.csv`: exported prediction averages by
  forecast interval.
- `results/summary/checkpoint_manifest.csv`: ignored local checkpoint paths and
  SHA256 hashes for available best/final checkpoints.
- `results/figures/`: existing figures copied from the experiment workspace.
