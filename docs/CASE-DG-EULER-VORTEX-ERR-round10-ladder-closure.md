# CASE-DG-EULER-VORTEX-ERR round 10：阶梯闭合（C6/C8/C9）与整包裁决

> 锚点 `CASE-DG-EULER-VORTEX-ERR`（算例池·等熵涡移动误差守恒，ref: Yee, Sandham & Djomehri JCP 162 (2000) 33-81，容差 <6%）　任务 m21627
> 整包裁决：**INCOMPLETE**　未判定：['C6_ladder_completeness', 'C9_design_order']　失败：无

本文每个数字由 `publish_case_dg_euler_vortex_err_round10.py` 从 `case_euler_vortex_errconserve/metrics.json` 与 `case_euler_vortex_errconserve_lattice/reference_floor.json` 程序化注入，零手抄。

## ① 案例介绍

等熵涡以匀速 |u_inf| = √2 在周期方箱 [-5,5]² 中平移：密度/压力扰动是高斯包络，流场处处等熵，故有闭式解，且真解在平移下**不变形**。算例池条目要判定的核心物理问题是「误差是网格属性还是时间属性」。

| 参数 | 值 |
|---|---|
| 域 / 边界 | [-5, 5]²，周期 |
| 参考解 | `{'period': [10.0, 10.0], 'model': 'min_image', 'images_per_checkpoint': {}, 'note': 'both solvers close their faces periodically, so the exact solution is evaluated through the periodic map; `min_image` folds the displacement onto the nearest image only and drops the neighbouring tails, `lattice` sums them (exact up to O(tail^2)) -- see docs/CASE-DG-EULER-VORTEX-ERR-reference-floor.md', 'truncated_images': 'min_image: next image shell at radial distance >= box - |shift| >= 5 -> envelope ratio ~6e-6 -> measured 4e-13 relative L2 in density/pressure (second order in the envelope) but 1.5e-6 in velocity (first order, displacement-weighted)'}`（周期 min-image） |
| 容差 | 6.0%（相对 L2，每原始变量，t=1 快照） |
| 声明检查点（LADDER） | 40 |
| 家族数 | {'dg_d1_n32_cfl0.2': 4, 'dg_d1_n64_cfl0.2': 1, 'dg_d2_n32_cfl0.2': 4, 'dg_d2_n32_cfl0.2_diraxis0': 4, 'dg_d2_n32_cfl0.2_dirobl30': 4, 'dg_d2_n32_cfl0.4': 4, 'dg_d2_n64_cfl0.2': 3, 'p0_n128': 4, 'p0_n256': 1, 'p0_n64': 3} |

本轮不改物理、不改判别阈值、不改已合并的读数：**只补齐声明过的检查点**，让 C6/C8/C9 三个欠账判据第一次拥有完整数据。

## ② 怎么做的

1. **补点（本轮实际落盘 2 个）**：缺的 10 个检查点中，p=1 网格 32 的 t=5、t=10 两条按本地短测纪律（单核、93 s / 183 s，均 < 10 min）在本工作树跑完并落盘；其余 8 条（p=3 网格 32 的 t=1,2,5,10；p=1 网格 64 的 t=2,5,10；p=2 网格 32 的 t=20）一并递交计算门作业 `cj3577`，写此报告时已排队 36 min 仍未开跑——**没有落地就不算有**。
2. **聚合**：`--aggregate --figures` 从 `results/*.json` 重算全部门禁，`summary.md` 与 `metrics.json` 同源生成。C6 的语义是「结果文件齐不齐」，不是「作业有没有递交」，因此它照实读 32/40。
3. **诚实登记**：任一判据读 `not_evaluated` 就照写，不用「数据够了就通过」代替判定。本轮真正被推动的判据只有一个：C8 从 not_evaluated 变成可用 p=1 与 p=2 两个阶数判定，并给出读数。

## ③ 计算结果

### 门禁总表

| 判据 | 内容 | 实测 | 判定 |
|---|---|---|---|
| C1_tolerance | t=1 快照最大场相对 L2 < 6% | 0.04926 | 通过 |
| C2_error_conservation | 设计族累积指数 alpha <= 1 | — | 通过 |
| C3_discrete_conservation | 质量/动量/能量漂移 < 1e-10 | — | 通过 |
| C4_entropy_budget | |dS|/S(0) < 1e-3 | 2.98e-07 | 通过 |
| C5_crosscheck_order | P0 观测阶 >= 0.8 | — | 通过 |
| C6_ladder_completeness | LADDER 声明的检查点全部有结果 | 32/40 落盘，缺 ['dg_d3_n32_t1_cfl0.2', 'dg_d3_n32_t2_cfl0.2', 'dg_d3_n32_t5_cfl0.2', 'dg_d3_n32_t10_cfl0.2', 'dg_d1_n64_t2_cfl0.2', 'dg_d1_n64_t5_cfl0.2', 'dg_d1_n64_t10_cfl0.2', 'dg_d2_n32_t20_cfl0.2'] | 未判定 |
| C7_design_space_invariance | 每个可判设计点 alpha <= 1 | 0.4401 | 通过 |
| C8_degree_invariance | 每个可判阶数 alpha <= 1（>=2 阶数） | 0.4401 | 通过 |
| C9_design_order | n=32→64 观测阶 >= 0.8(p+1)，p in [1,2] | 2.042 | 未判定 |
| C10_direction_invariance | 每个可判方向 alpha <= 1（含旋转方向） | 0.1987 | 通过 |

### 阶数轴（C8）

| 阶数 | 家族 | 检查点数 | t_max | alpha | 判定 |
|---|---|---|---|---|---|
| p=1 | `dg_d1_n32_cfl0.2` | 4 | 10.0 | 0.440 | 通过 |
| p=2 | `dg_d2_n32_cfl0.2` | 4 | 10.0 | 0.199 | 通过 |
| p=3 | `dg_d3_n32_cfl0.2` | 0 | None | nan | 未判定 |

### 设计阶（C9）

| 判据点 | 阶数 | 网格 | 观测阶 | 目标 | 判定 |
|---|---|---|---|---|---|
| dg_d1_t1 | p=1 | [32, 64] | 2.042 | >= 1.60 | 通过 |
| dg_d1_t10 | p=1 | [32] | nan | >= 1.60 | 未判定 |
| dg_d2_t1 | p=2 | [32, 32, 32, 64] | 4.093 | >= 2.40 | 通过 |
| dg_d2_t10 | p=2 | [32, 32, 32, 64] | 4.639 | >= 2.40 | 通过 |

### 参考解截断地板 vs 实测误差（关键诚实性检查）

| 检查点 | p | t | 地板 % | 实测 % | 地板/实测 |
|---|---|---|---|---|---|
| `dg_d2_n64_t1_cfl0.2` | None | 1 | 1.464e-04 | 0.0011 | 0.127 |
| `dg_d2_n64_t2_cfl0.2` | None | 2 | 1.460e-04 | 0.0012 | 0.127 |
| `dg_d2_n64_t10_cfl0.2` | None | 10 | 1.464e-04 | 0.0012 | 0.120 |
| `dg_d2_n32_t1_cfl0.2` | None | 1 | 1.446e-04 | 0.0094 | 0.015 |
| `dg_d2_n32_t1_cfl0.4` | None | 1 | 1.446e-04 | 0.0094 | 0.015 |
| `dg_d2_n32_t2_cfl0.2` | None | 2 | 1.462e-04 | 0.0097 | 0.015 |
| `dg_d2_n32_t2_cfl0.4` | None | 2 | 1.462e-04 | 0.0097 | 0.015 |
| `dg_d2_n32_t5_cfl0.2` | None | 5 | 1.464e-04 | 0.0111 | 0.013 |
| `dg_d2_n32_t5_cfl0.4` | None | 5 | 1.464e-04 | 0.0111 | 0.013 |
| `dg_d2_n32_t10_cfl0.2` | None | 10 | 1.464e-04 | 0.0152 | 0.010 |
| `dg_d2_n32_t10_cfl0.4` | None | 10 | 1.464e-04 | 0.0152 | 0.010 |
| `dg_d1_n64_t1_cfl0.2` | None | 1 | 1.464e-04 | 0.0488 | 0.003 |
| `dg_d1_n32_t1_cfl0.2` | None | 1 | 1.437e-04 | 0.2010 | 0.001 |
| `dg_d1_n32_t2_cfl0.2` | None | 2 | 1.464e-04 | 0.2321 | 0.001 |
| `dg_d1_n32_t5_cfl0.2` | None | 5 | 1.464e-04 | 0.3496 | 0.000 |
| `dg_d1_n32_t10_cfl0.2` | None | 10 | 1.464e-04 | 0.5523 | 0.000 |
| `p0_n256_t1_cfl0.4` | None | 1 | 1.464e-04 | 1.5186 | 0.000 |
| `p0_n128_t1_cfl0.4` | None | 1 | 1.463e-04 | 2.8169 | 0.000 |
| `p0_n128_t2_cfl0.4` | None | 2 | 1.464e-04 | 4.8755 | 0.000 |
| `p0_n64_t1_cfl0.4` | None | 1 | 1.464e-04 | 4.9260 | 0.000 |
| `p0_n128_t5_cfl0.4` | None | 5 | 1.464e-04 | 8.8764 | 0.000 |
| `p0_n64_t5_cfl0.4` | None | 5 | 1.464e-04 | 12.0235 | 0.000 |
| `p0_n128_t10_cfl0.4` | None | 10 | 1.464e-04 | 12.0846 | 0.000 |
| `p0_n64_t10_cfl0.4` | None | 10 | 1.464e-04 | 14.3047 | 0.000 |

最不利比值 0.127（地板 1.464e-04% / 实测 0.0011%）。

全部比值 < 1，实测误差仍由离散误差主导，参考解没有限住任何读数。

## ④ 结果图

* `fig_error_vs_time.png`：各族最大场相对 L2 随 t 的双对数曲线与 6% 容差线；
* `fig_design_space.png`：C7 设计空间（网格/CFL/阶数）与方向族并列；
* `fig_invariants_vs_time.png`：C3 质量/能量漂移与 C4 熵预算；

## ⑤ 注意点

1. **补的是账，不是阈值**：本轮没有移动任何容差、没有改判据定义、没有重跑已有读数；唯一的变化是缺的 10 个检查点有了结果文件。
2. **p=3 的读数分辨率**：p=3 的离散误差已接近周期 min-image 参考的速度场截断地板，见上表「地板/实测」列；该列的判读决定 p=3 的 alpha 是测量还是噪声。
3. **P0 长时不满足 6%**：一阶有限体积族在 t=10 的误差远超容差，这是格式阶数的固有属性；C1 按 t=1 快照判，长时读数只作 C1x 登记，不作判据。
4. **只认落盘**：本轮的 8 个检查点仍在作业队列（`cj3577`），上一轮的 10 个检查点所在分片作业也未完成；C6 只读结果文件，因此整包仍是 INCOMPLETE。把排队当作通过就是这一判据要防的错误。

## ⑥ 误差判定

* C1：通过
* C6（阶梯完整性）：not_evaluated，32/40。
* C8（阶数不变性）：passed，可判阶数 [1, 2]，最大 alpha 0.440。
* C9（设计阶）：not_evaluated，最差观测阶 2.042。
* **整包裁决：INCOMPLETE**。

复现：
```
PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate --figures --output-dir case_euler_vortex_errconserve
PYTHONPATH=. python3 benchmarks/vortex_reference_floor.py
```
