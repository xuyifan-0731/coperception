# Candidate 2025-2026 cooperative perception methods

Initial shortlist from web search on 2026-06-04:

1. LRCP, WACV 2025, Latency Robust Cooperative Perception using Asynchronous Feature Fusion. Directly reports V2X-Sim and DAIR-V2X under latency, with official code.
2. DATA, ICCV 2025, Domain-And-Time Alignment for High-Quality Feature Fusion in Collaborative Perception. Official code includes DAIR-V2X dataset support and targets time alignment; V2X-Sim needs adapter.
3. SparseAlign, CVPR 2025. Strong venue and code, but reports OPV2Vt/DAIR-V2Xt rather than V2X-Sim.
4. V2XPnP, ICCV 2025. Strong venue and code, but centered on its own sequential V2XPnP dataset/framework, not a drop-in V2X-Sim/DAIR latency AP baseline.
5. IPDA, AAAI 2026. Strong relevance to delay, but reports DAIR-V2X/OPV2V rather than V2X-Sim.

Selected for reproduction attempt: LRCP and DATA.

## Local reproduction status on 2026-06-04

| Method | Dataset | Current local status | Result / blocker |
|---|---|---|---|
| DATA | DAIR-V2X-C | Completed one no-delay local inference run | AP@0.5 = 54.0460, AP@0.7 = 37.4688. Output: `runs/data_dair_official_eval/eval_intermediate_local_dair_nodelay_102.4_51.2_epoch21.yaml`. |
| DATA | DAIR-V2X-C latency grid | Not completed | The public code has `time_delay` field remnants, but the released `DATA.py` / `data_fusion.py` forward path does not consume a delay value, and README does not provide the latency-eval script. |
| DATA | V2X-Sim | Not completed | V2X-Sim configs exist, but the cloned repo only includes a DAIR official checkpoint. No released V2X-Sim DATA checkpoint was found locally. |
| LRCP | V2X-Sim | Not completed | Repo has V2X-Sim configs and latency inference script, but no checkpoint. It also needs V2X-Sim 2.0 NuScenes-format info pkl and generated flow labels; the local `V2X-Sim-det` is the older per-agent `.npy` layout, not this format. |
| LRCP | DAIR-V2X-C | Not completed | README only mentions a DAIR flow-file download link. No DAIR yaml/checkpoint is included in the cloned repo. |

The detailed machine-readable status table is in `results/literature_search_250604/reproduction_status.csv`.
