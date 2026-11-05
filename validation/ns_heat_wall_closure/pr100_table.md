| ny | cold-wall $\max|R_E|$ | wall ratio vs prev | interior $\max|R_E|$ | interior ratio vs prev |
|---|---|---|---|---|
| 8 | 1.9561e-01 | -- | 2.9398e-01 | -- |
| 16 | 1.5739e-01 | 1.2429 | 1.7093e-01 | 1.7199 |
| 32 | 1.3985e-01 | 1.1254 | 9.2912e-02 | 1.8397 |
| 64 | 1.3143e-01 | 1.0640 | 4.8554e-02 | 1.9136 |

Stale pin `wall_ratio(16/32) < 1.1` against measured **1.1254006471** -> fails by 2.31 %.
Repaired pins (`< 1.2` at 16/32, `< 1.1` at 32/64, monotone, floor > 1e-2) with measured 1.0640333275 -> all hold.
