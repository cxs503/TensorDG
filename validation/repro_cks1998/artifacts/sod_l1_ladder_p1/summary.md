# Sod L1 网格阶梯（REPRO-DG-CKS1998 round 2 驱动工件）

- git `103e778f8c77` · 生成于 2026-11-12T20:39:48+0000
- degree P1 · cfl=0.12 · tvb M=1.0 · limiter_stage=final · 求积 8 点
- 墙钟 386.95 s

## 阶梯（相对 L1，8 点 Gauss 逐单元积分）

| N | steps | L1 ρ | L1 u | L1 p | L1 最差 | 单元平均相对 L2 最差 | 密度超调 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1096 | 1.252433e-02 | 2.725989e-02 | 1.105363e-02 | 2.725989e-02 | 5.783529e-02 | 3.131e-03 |
| 200 | 2208 | 6.819765e-03 | 1.371325e-02 | 5.575409e-03 | 1.371325e-02 | 4.057721e-02 | 2.578e-03 |
| 400 | 4433 | 3.747326e-03 | 6.999306e-03 | 2.833357e-03 | 6.999306e-03 | 2.346151e-02 | 1.949e-03 |
| 800 | 8883 | 2.071295e-03 | 3.589981e-03 | 1.434492e-03 | 3.589981e-03 | 1.394115e-02 | 1.535e-03 |
| 800 (unlimited) | 9263 | 9.590022e-04 | 1.902359e-03 | 6.071657e-04 | 1.902359e-03 | 9.547157e-03 | 2.862e-02 |

## 观测阶（相邻两级的 log 斜率）

| 量 | 100→200 | 200→400 | 400→800 |
|---|---:|---:|---:|
| l1_rel_density | 0.8769 | 0.8639 | 0.8553 |
| l1_rel_velocity | 0.9912 | 0.9703 | 0.9632 |
| l1_rel_pressure | 0.9874 | 0.9766 | 0.9820 |
| l1_rel_worst | 0.9912 | 0.9703 | 0.9632 |
| l2_cellavg_worst | 0.5113 | 0.7904 | 0.7509 |

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
| full | p1_n400 | 400 | 0.000e+00 |
