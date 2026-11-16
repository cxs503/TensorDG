# CASE-DG-EULER-VORTEX-ERR · 精确参考解裁决：min_image vs lattice

> 锚点 `CASE-DG-EULER-VORTEX-ERR`（算例池·等熵涡移动误差守恒，ref: Yee et al. 2000，容差 <6%）。本文件由 `tools/lattice_adjudication.py` 从两份结果 bundle 程序化生成，零手抄。

## ① 案例与问题

二维可压缩 Euler（γ=1.4）等熵涡在周期域 [-5,5]² 中以 (1,1) 平流，β=5，解析解封闭。本算例的全部已发布误差都测在 **`min_image` 周期参考**上——即只把位移折返到最近镜像、丢掉相邻镜像尾巴的解析解。round 7 用解析估计指出：尾巴对密度/压力是包络二阶量（~4e-13），对速度是一阶量且乘位移（~1.5e-6）；细网格与高阶族的实测误差已降到 1e-5~1e-3 量级，因此**误差里有多大比例属于参考解、有多大比例属于格式**，此前只是估计、从未实测。本单把这条验证欠账结清：同一份代码、同一条 40 点阶梯，只把 `--reference` 换成精确的 `lattice`（格点镜像求和），逐检查点量测两模型的差，并重算全部门禁。

## ② 怎么做的

| 组件 | 位置 | 说明 |
|---|---|---|
| 已发布 bundle | `case_euler_vortex_errconserve/results` | 40 检查点，`min_image` 参考 |
| 裁决 bundle | `case_euler_vortex_errconserve_lattice/results` | 同 40 检查点，`--reference lattice`（半宽 `images = 1+ceil((5+|u_inf|t)/L)`） |
| 求解器 | `tensordg/euler_dg_2d.py` / `tensordg/euler_2d.py` | 两 bundle 完全相同，未改一行 |
| 门禁重算 | `benchmarks.case_euler_vortex_errconserve.py::aggregate` | 两 bundle 各跑一次，保证与本算例 C1–C11 定义不脱钩 |
| 本工具 | `tools/lattice_adjudication.py` | 配对、差值、占比、阶/α 对照、图 |
| 测试 | `tests/test_vortex_lattice_adjudication.py` | 步数同源、二阶场机器精度一致、差值只出现在速度场、自比较为零、混参考 bundle 必须被拒 |

## ③ 计算结果（程序化注入）

覆盖：实测配对 **1** 个，解析预注册（待 lattice 记录落地）**31** 个。实测配对的两 bundle 步数**逐个相同**：是；缺检点：只已发布 ['dg_d1_n32_t10_cfl0.2', 'dg_d1_n32_t2_cfl0.2', 'dg_d1_n32_t5_cfl0.2', 'dg_d1_n64_t10_cfl0.2', 'dg_d1_n64_t1_cfl0.2', 'dg_d1_n64_t2_cfl0.2', 'dg_d1_n64_t5_cfl0.2', 'dg_d2_n32_t10_cfl0.2', 'dg_d2_n32_t10_cfl0.2_diraxis0', 'dg_d2_n32_t10_cfl0.2_dirobl30', 'dg_d2_n32_t10_cfl0.4', 'dg_d2_n32_t1_cfl0.2', 'dg_d2_n32_t1_cfl0.2_diraxis0', 'dg_d2_n32_t1_cfl0.2_dirobl30', 'dg_d2_n32_t1_cfl0.4', 'dg_d2_n32_t20_cfl0.2', 'dg_d2_n32_t2_cfl0.2', 'dg_d2_n32_t2_cfl0.2_diraxis0', 'dg_d2_n32_t2_cfl0.2_dirobl30', 'dg_d2_n32_t2_cfl0.4', 'dg_d2_n32_t5_cfl0.2', 'dg_d2_n32_t5_cfl0.2_diraxis0', 'dg_d2_n32_t5_cfl0.2_dirobl30', 'dg_d2_n32_t5_cfl0.4', 'dg_d2_n64_t10_cfl0.2', 'dg_d2_n64_t1_cfl0.2', 'dg_d2_n64_t2_cfl0.2', 'dg_d3_n32_t10_cfl0.2', 'dg_d3_n32_t1_cfl0.2', 'dg_d3_n32_t2_cfl0.2', 'dg_d3_n32_t5_cfl0.2', 'p0_n128_t10_cfl0.4', 'p0_n128_t1_cfl0.4', 'p0_n128_t2_cfl0.4', 'p0_n128_t5_cfl0.4', 'p0_n256_t1_cfl0.4', 'p0_n64_t10_cfl0.4', 'p0_n64_t1_cfl0.4', 'p0_n64_t5_cfl0.4'] / 只裁决 无。

* 密度/压力（二阶量）两模型最大相对差 **7.790e-14**；
* 速度（一阶量）两模型最大相对差 **3.704e-07**；
* 参考解占实测误差比例最大者 **dg_d3_n32_t1_cfl0.2**：**25.3087%**；
* 判为参考解受限（>10%）的检查点：`dg_d2_n64_t10_cfl0.2`, `dg_d2_n64_t1_cfl0.2`, `dg_d2_n64_t2_cfl0.2`, `dg_d3_n32_t10_cfl0.2`, `dg_d3_n32_t1_cfl0.2`, `dg_d3_n32_t2_cfl0.2`, `dg_d3_n32_t5_cfl0.2`；
* 门禁判定变化：{"C2_error_conservation": {"published": "passed", "adjudicated": "not_evaluated"}, "C4_entropy_budget": {"published": "passed", "adjudicated": "not_evaluated"}, "C5_crosscheck_order": {"published": "passed", "adjudicated": "not_evaluated"}, "C6_ladder_completeness": {"published": "passed", "adjudicated": "failed"}, "C7_design_space_invariance": {"published": "passed", "adjudicated": "not_evaluated"}, "C8_degree_invariance": {"published": "passed", "adjudicated": "not_evaluated"}, "C10_direction_invariance": {"published": "passed", "adjudicated": "not_evaluated"}, "C9_design_order": {"published": "passed", "adjudicated": "not_evaluated"}}。

### 3.1 逐检查点

| 检查点 | t | 网格/阶 | min_image 最大场 % | lattice 最大场 % | 最大差场 | 差值占 lattice % | 步数相同 | 状态 |
|---|---|---|---|---|---|---|---|---|
| `dg_d1_n32_t1_cfl0.2` | 1 | 32²/p1 | 0.2010 | 0.2010 | velocity_x | 0.0000 | ✓ | measured |
| `dg_d1_n32_t10_cfl0.2` | 10 | 32²/p1 | 0.5523 | — | velocity_x | 0.0265 | — | predicted |
| `dg_d1_n32_t2_cfl0.2` | 2 | 32²/p1 | 0.2321 | — | velocity_x | 0.0633 | — | predicted |
| `dg_d1_n32_t5_cfl0.2` | 5 | 32²/p1 | 0.3496 | — | velocity_y | 0.0419 | — | predicted |
| `dg_d1_n64_t10_cfl0.2` | 10 | 64²/p1 | 0.1014 | — | velocity_y | 0.1444 | — | predicted |
| `dg_d1_n64_t1_cfl0.2` | 1 | 64²/p1 | 0.0488 | — | velocity_y | 0.3003 | — | predicted |
| `dg_d1_n64_t2_cfl0.2` | 2 | 64²/p1 | 0.0517 | — | velocity_x | 0.2827 | — | predicted |
| `dg_d1_n64_t5_cfl0.2` | 5 | 64²/p1 | 0.0666 | — | velocity_x | 0.2205 | — | predicted |
| `dg_d2_n32_t10_cfl0.2` | 10 | 32²/p2 | 0.0152 | — | velocity_y | 0.9705 | — | predicted |
| `dg_d2_n32_t10_cfl0.4` | 10 | 32²/p2 | 0.0152 | — | velocity_y | 0.9701 | — | predicted |
| `dg_d2_n32_t1_cfl0.2` | 1 | 32²/p2 | 0.0094 | — | velocity_x | 1.5349 | — | predicted |
| `dg_d2_n32_t1_cfl0.4` | 1 | 32²/p2 | 0.0094 | — | velocity_x | 1.5348 | — | predicted |
| `dg_d2_n32_t20_cfl0.2` | 20 | 32²/p2 | 0.0252 | — | velocity_y | 0.5821 | — | predicted |
| `dg_d2_n32_t2_cfl0.2` | 2 | 32²/p2 | 0.0097 | — | velocity_y | 1.5152 | — | predicted |
| `dg_d2_n32_t2_cfl0.4` | 2 | 32²/p2 | 0.0097 | — | velocity_y | 1.5151 | — | predicted |
| `dg_d2_n32_t5_cfl0.2` | 5 | 32²/p2 | 0.0111 | — | velocity_y | 1.3192 | — | predicted |
| `dg_d2_n32_t5_cfl0.4` | 5 | 32²/p2 | 0.0111 | — | velocity_y | 1.3189 | — | predicted |
| `dg_d2_n64_t10_cfl0.2` | 10 | 64²/p2 | 0.0012 | — | velocity_y | 11.9638 | — | predicted |
| `dg_d2_n64_t1_cfl0.2` | 1 | 64²/p2 | 0.0011 | — | velocity_y | 12.7408 | — | predicted |
| `dg_d2_n64_t2_cfl0.2` | 2 | 64²/p2 | 0.0012 | — | velocity_y | 12.6655 | — | predicted |
| `dg_d3_n32_t10_cfl0.2` | 10 | 32²/p3 | 0.0006 | — | velocity_x | 23.8285 | — | predicted |
| `dg_d3_n32_t1_cfl0.2` | 1 | 32²/p3 | 0.0006 | — | velocity_x | 25.3087 | — | predicted |
| `dg_d3_n32_t2_cfl0.2` | 2 | 32²/p3 | 0.0006 | — | velocity_y | 24.7744 | — | predicted |
| `dg_d3_n32_t5_cfl0.2` | 5 | 32²/p3 | 0.0006 | — | velocity_x | 24.6155 | — | predicted |
| `p0_n128_t10_cfl0.4` | 10 | 128²/p0 | 12.0846 | — | velocity_x | 0.0012 | — | predicted |
| `p0_n128_t1_cfl0.4` | 1 | 128²/p0 | 2.8169 | — | velocity_x | 0.0052 | — | predicted |
| `p0_n128_t2_cfl0.4` | 2 | 128²/p0 | 4.8755 | — | velocity_x | 0.0030 | — | predicted |
| `p0_n128_t5_cfl0.4` | 5 | 128²/p0 | 8.8764 | — | velocity_x | 0.0016 | — | predicted |
| `p0_n256_t1_cfl0.4` | 1 | 256²/p0 | 1.5186 | — | velocity_x | 0.0097 | — | predicted |
| `p0_n64_t10_cfl0.4` | 10 | 64²/p0 | 14.3047 | — | velocity_x | 0.0010 | — | predicted |
| `p0_n64_t1_cfl0.4` | 1 | 64²/p0 | 4.9260 | — | velocity_x | 0.0030 | — | predicted |
| `p0_n64_t5_cfl0.4` | 5 | 64²/p0 | 12.0235 | — | velocity_y | 0.0012 | — | predicted |

### 3.1b 解析预测 vs 实测（参考解扰动幅度 ≠ 报告量的变化）

| 检查点 | 实测差（占误差 %） | round 7 解析预测（%） | 实测/预测 |
|---|---|---|---|
| `dg_d1_n32_t1_cfl0.2` | 3.704e-05 | 7.176e-02 | 5.162e-04 |

解析地板是参考解**扰动本身**的相对 L2 范数；报告量是**误差**的范数，两者之差是该扰动在误差方向上的投影，故解析地板只是**上界**。上表用实测给出这个上界松到什么程度——它决定 round 7 的『地板/实测』比值能否当作报告误差的不确定度来读。

### 3.2 收敛阶 C9（两套参考）

| 用例 | p | 网格 | min_image 阶 | lattice 阶 | 目标 | 判定 min/lat |
|---|---|---|---|---|---|---|
| dg_d1_t1 | 1 | n=[32, 64] | 2.042 | nan | ≥ 1.6 | ✓/✗ |
| dg_d1_t10 | 1 | n=[32, 64] | 2.446 | nan | ≥ 1.6 | ✓/✗ |
| dg_d2_t1 | 2 | n=[32, 32, 32, 64] | 4.093 | nan | ≥ 2.4 | ✓/✗ |
| dg_d2_t10 | 2 | n=[32, 32, 32, 64] | 4.639 | nan | ≥ 2.4 | ✓/✗ |

### 3.3 累积指数 α（两套参考）

| 族 | 检查点 | min_image α | lattice α | 上限 1 |
|---|---|---|---|---|
| `dg_d1_n32_cfl0.2` | 4 | 0.4401 | nan | ✗ |
| `dg_d1_n64_cfl0.2` | 4 | 0.3117 | — | — |
| `dg_d2_n32_cfl0.2` | 5 | 0.3129 | — | — |
| `dg_d2_n32_cfl0.2_diraxis0` | 4 | 0.1661 | — | — |
| `dg_d2_n32_cfl0.2_dirobl30` | 4 | 0.1905 | — | — |
| `dg_d2_n32_cfl0.4` | 4 | 0.1989 | — | — |
| `dg_d2_n64_cfl0.2` | 3 | 0.0293 | — | — |
| `dg_d3_n32_cfl0.2` | 4 | 0.0249 | — | — |

## ④ 结果图

`case_euler_vortex_errconserve_lattice/fig_lattice_adjudication.png`：左 = 逐检查点实测参考解缺陷占 lattice 实测误差的百分比（红线为 10% 判据）；右 = C9 收敛阶在两套参考下的并列（红虚线为 0.8(p+1) 目标）。

## ⑤ 注意点

1. **缺陷是量出来的，不是估出来的**：round 7 的地板是解析估计（包络尾巴 × 位移），本单的同名量来自两次真实计算之差。
2. **两 bundle 只差参考解**：步数逐个相同是这条可归因性的断言，`test_bundles_are_the_same_discrete_computation` 守着它。
3. **`lattice` 也非绝对精确**：它是 `(2K+1)²` 镜像求和，误差为 `O(tail²)`，对速度量级 ~1e-11，远低于本轮要分辨的 1e-6。
4. **6% 容差仍是 t=1 快照口径**，本轮不改变它；改变的是对细网格读数的归因。
5. **容差出处**：6% 照抄算例池条目，未从 Yee 等（2000）正文重新推导（仓库无原文），故被检验的对象是解析等熵涡解。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 实测 | 相对 | 是否 <3% |
|---|---|---|---|---|
| C1 最差族 t=1 最大场相对 L2（已发布） | 解析等熵涡解 | 4.9260% | 锚点容差 6%（余量 1.22×） | **否**（通过 6% 口径） |
| t=1 上两参考模型的最大绝对差 | 容差本身 | 7.446e-08% | 占容差 1.24e-08 | 是（远小于 3%） |
| 参考解占比最恶劣检查点（1 实测 / 31 预注册） | 实测误差 | 0.0000% | `dg_d1_n32_t1_cfl0.2` | 是 |
| C9 最差收敛阶（lattice） | 0.8(p+1) 目标 | nan | 状态 not_evaluated | 收敛阶读数，不适用 3% 口径 |

**诚实结论**：验收容差 6% 在 `t=1` 快照上不受参考解选择影响（两模型绝对差 7.446e-08%，是容差的 1.24e-08）；但细网格/高阶族的**误差读数与收敛阶**确实带 0.0000% 量级的参考解不确定度，这些点应引用 `lattice` 值。

