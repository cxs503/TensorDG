# Sod L1 网格阶梯（REPRO-DG-CKS1998 round 2 驱动工件）

- git `103e778f8c77` · 生成于 2026-11-12T20:42:44+0000
- degree P2 · cfl=0.12 · tvb M=1.0 · limiter_stage=final · 求积 8 点
- 墙钟 672.92 s

## 阶梯（相对 L1，8 点 Gauss 逐单元积分）

| N | steps | L1 ρ | L1 u | L1 p | L1 最差 | 单元平均相对 L2 最差 | 密度超调 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1810 | 1.290130e-02 | 2.852610e-02 | 1.127489e-02 | 2.852610e-02 | 6.002236e-02 | 1.785e-03 |
| 200 | 3646 | 7.023707e-03 | 1.432058e-02 | 5.683936e-03 | 1.432058e-02 | 4.173408e-02 | 1.075e-03 |
| 400 | 7321 | 3.848541e-03 | 7.239022e-03 | 2.874135e-03 | 7.239022e-03 | 2.367695e-02 | 6.865e-04 |
| 800 | 14669 | 2.124046e-03 | 3.685489e-03 | 1.449810e-03 | 3.685489e-03 | 1.362013e-02 | 5.248e-04 |
| 800 (unlimited) | 15066 | 7.734126e-04 | 2.503755e-03 | 7.569516e-04 | 2.503755e-03 | 9.689547e-03 | 3.536e-02 |

## 观测阶（相邻两级的 log 斜率）

| 量 | 100→200 | 200→400 | 400→800 |
|---|---:|---:|---:|
| l1_rel_density | 0.8772 | 0.8679 | 0.8575 |
| l1_rel_velocity | 0.9942 | 0.9842 | 0.9739 |
| l1_rel_pressure | 0.9882 | 0.9838 | 0.9873 |
| l1_rel_worst | 0.9942 | 0.9842 | 0.9739 |
| l2_cellavg_worst | 0.5243 | 0.8177 | 0.7977 |

## 预登记判据

| 判据 | 含义 | 结果 |
|---|---|---|
| C1_monotone | relative L1 error strictly decreases on every rung, every field | **PASS** |
| C2_order_bracket | density L1 order on the finest pair within [0.5, 2.0] (limiter half-order floor .. below the design order p+1=3) | **PASS** |
| C3_cross_check | cell-average relative L2 at overlapping meshes reproduces the round-1 E2 numbers to <=1e-6 relative | **PASS** |
| C4_gate_extension | report whether the finest rung meets the 3% relative-L2 gate (verdict recorded, gate itself unchanged) | **PASS** |
| C5_not_roundoff | finest mesh relative L1 error > 1e-10 (not a roundoff floor) | **PASS** |
| C6_limiter_active | unlimited control at the finest mesh has a strictly larger density overshoot than the limited run | **PASS** |

**总判定：PASS**

## 与 round-1 工件的交叉核对（C3）

| profile | 用例 | N | 最大相对差 |
|---|---|---:|---:|
| full | p2_n200 | 200 | 0.000e+00 |
| full | p2_n400 | 400 | 0.000e+00 |
| quick | p2_n200 | 200 | 0.000e+00 |
| quick | p2_n400 | 400 | 0.000e+00 |
