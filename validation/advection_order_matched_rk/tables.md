# CASE-DG-ADVECT-ORDER round 2 — matched-stage re-measurement

## A. Order, like-for-like stage count

| p | integrator | stages | 阶 | 该级数可达最大阶 | ref column | ref stages | ref 阶 | TensorDG 阶 | 差 | 判定 |
|--:|---|---:|---:|---:|---|--:|---:|---:|---:|:--:|
| P3 | rk4 | 4 | 4 | 4 | RK4DG3 | 4 | 4.001 | 4.000 | -0.001 | PASS |

## B. L2 magnitude ratio (reference dt unknown)

| p | N=20 | N=40 | N=80 | N=160 | N=320 | 均值 | round-1 均值 | 在 [0.85,1.18] |
|--:|---:|---:|---:|---:|---:|---:|---:|:--:|
| P3 | 0.060 / 0.059 / 0.060 | 0.060 | 0.060 | FAIL |

## C. 精确时间/空间误差拆分（expm 精确半离散解）

| p | N | 格式 | E_total | E_space | E_time | E_time 占比 |
|--:|--:|---|--:|--:|--:|--:|
| P1 | 8 | rk2 | 3.718e-02 | 3.688e-02 | 1.960e-03 | 5.27% |
| P1 | 16 | rk2 | 7.502e-03 | 7.464e-03 | 5.051e-04 | 6.73% |
| P2 | 8 | rk3 | 1.677e-03 | 1.676e-03 | 5.738e-06 | 0.34% |
| P2 | 16 | rk3 | 2.090e-04 | 2.089e-04 | 7.175e-07 | 0.34% |
| P3 | 8 | rk4 | 7.982e-05 | 7.982e-05 | 9.431e-09 | 0.01% |
| P3 | 16 | rk4 | 5.042e-06 | 5.042e-06 | 5.868e-10 | 0.01% |
| P4 | 8 | ssp_rk54 | 3.093e-06 | 3.093e-06 | 1.590e-09 | 0.05% |
| P4 | 16 | ssp_rk54 | 9.732e-08 | 9.732e-08 | 9.935e-11 | 0.10% |

## D. 时间阶独立实测（固定网格，步数细分 vs expm 精确解）

| p | 格式 | 级数 | 设计阶 | 实测时间阶 | 拟合点数 | 判定 |
|--:|---|---:|---:|---:|---:|:--:|
| P1 | rk2 | 2 | 2 | 2.000 | 6 | PASS |
| P2 | rk3 | 3 | 3 | 3.000 | 6 | PASS |
| P3 | rk4 | 4 | 4 | 4.000 | 4 | PASS |
| P4 | ssp_rk54 | 5 | 4 | 4.000 | 3 | PASS |

## 判定

- M1_rate: **FAIL**
- M2_stages: **FAIL**
- M3_stage_gap_rejected: **FAIL**
- M4_time_share: **PASS**
- M5_dt_order: **PASS**

**总判定：FAIL**（未过：M1_rate, M2_stages, M3_stage_gap_rejected）
