# DAIR Table2/Table5 proxy summary

Source: `results/dair_v2x/our_method/full_T10_n10_ckpt15000`

This table uses the original paper latency grid, but reports DAIR label-derived BEV IoU, not detection AP.

## IoU

| Method | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No compensation (V2VNet-style delayed baseline) | 0 | 0.3675 | 1.0000 | 0.4667 | 0.2774 | 0.1931 | 0.1480 | 0.1197 |
| No compensation (V2VNet-style delayed baseline) | 2 | 0.1562 | 0.2774 | 0.1931 | 0.1480 | 0.1197 | 0.1047 | 0.0942 |
| No compensation (V2VNet-style delayed baseline) | 4 | 0.1059 | 0.1480 | 0.1197 | 0.1047 | 0.0942 | 0.0880 | 0.0807 |
| Dual compensation (V2VNet+DLPCM-style proxy) | 0 | 0.5687 | 1.0000 | 0.6539 | 0.5463 | 0.4633 | 0.4000 | 0.3486 |
| Dual compensation (V2VNet+DLPCM-style proxy) | 2 | 0.3896 | 0.5463 | 0.4633 | 0.4000 | 0.3486 | 0.3073 | 0.2719 |
| Dual compensation (V2VNet+DLPCM-style proxy) | 4 | 0.2982 | 0.4000 | 0.3486 | 0.3073 | 0.2719 | 0.2431 | 0.2181 |

## Dynamic IoU

| Method | Comp | AVG | Comm0 | Comm1 | Comm2 | Comm3 | Comm4 | Comm5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No compensation (V2VNet-style delayed baseline) | 0 | 0.1821 | 1.0000 | 0.0211 | 0.0193 | 0.0181 | 0.0174 | 0.0165 |
| No compensation (V2VNet-style delayed baseline) | 2 | 0.0170 | 0.0193 | 0.0181 | 0.0174 | 0.0165 | 0.0158 | 0.0149 |
| No compensation (V2VNet-style delayed baseline) | 4 | 0.0153 | 0.0174 | 0.0165 | 0.0158 | 0.0149 | 0.0141 | 0.0134 |
| Dual compensation (V2VNet+DLPCM-style proxy) | 0 | 0.5519 | 1.0000 | 0.4367 | 0.5115 | 0.4871 | 0.4528 | 0.4233 |
| Dual compensation (V2VNet+DLPCM-style proxy) | 2 | 0.4369 | 0.5115 | 0.4871 | 0.4528 | 0.4233 | 0.3913 | 0.3551 |
| Dual compensation (V2VNet+DLPCM-style proxy) | 4 | 0.3733 | 0.4528 | 0.4233 | 0.3913 | 0.3551 | 0.3223 | 0.2949 |
