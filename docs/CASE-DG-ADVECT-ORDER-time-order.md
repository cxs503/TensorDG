# CASE-DG-ADVECT-ORDER · M5_dt_order：时间方向收敛阶阶梯（对精确半离散解）

**锚点** CASE-DG-ADVECT-ORDER ｜ **补测项** `M5_dt_order`（第2轮记为 not_run）｜
**仓库** TensorDG ｜ **算例** 1-D 周期线性对流 `u_t + u_x = 0`，`u(x,0)=sin(2πx)`，`t*=1` ｜
**判据** 观测时间阶 vs tableau 名义阶（相对偏差）

## ⓪ 误差判定（是否 <3%）

判据是**时间收敛阶**，不是绝对误差。matched 配对（`tableau_for_degree(p)`，即锚点空间阶扫描实际使用的配置）
共 10 组，观测阶与名义阶的相对偏差最大 2.82%（N=16, p=4,
rk5：5 → 5.1412），其余全部 <
0.56%。

**结论：是，10/10 组时间阶相对偏差 <3%**
（`rk5`@N=16,p=4 为 2.82%，仍在 3% 内，但已是预渐近区的边缘，
见第 ⑤ 节局限 1）。

## ① 案例介绍

一维周期线性对流，迎风 DG（模态 Legendre 基），半离散系统 `du/dt = L u` 是**线性常系数**系统，
因此存在精确解 `exp(L t) u0`——这就是本单用来剥离空间误差的基准。

| 项 | 值 |
|---|---|
| 区域 | `x ∈ [0,1)`，周期边界 |
| 速度 | `c = 1.0` |
| 初值 | `sin(2πx)`（模态投影，逐单元 Gauss 求积） |
| 终止时间 | `T = 1.0` |
| 网格 N | 8, 16 |
| 多项式阶 p | 0, 1, 2, 3, 4 |
| 时间格式 | `euler, rk2, rk3, rk4, rk5`（`tensordg.runge_kutta.TABLEAUS`，含 Butcher 6 级 5 阶） |
| matched 配对 | `tableau_for_degree(p)`，即 `q = p+1`（RKDG 惯例） |

## ② 怎么做的（仪器与关键设置）

1. **算子装配**：用 `convergence.semidiscrete_operator` 逐列装配 shipped RHS
   `tensordg.advection_1d._rhs` 的矩阵 `L`（不是重新推导），并用 `operator_linearity_residual`
   证明装配出来的矩阵就是实际被求解的算子（残差 ~1e-16）。
2. **精确基准**：`convergence.expm_scaled_squaring(L, T)`（80 位 longdouble 累加）给出
   `exp(LT) u0`，即半离散系统的**精确**解；它与时间步长无关，故 `‖u_h(T) − exp(LT)u0‖` 只含时间误差。
3. **阶梯**：`dt = T / (nsteps0·2^j)`，`nsteps0 = ceil(‖L‖₂)`（谱半径，保证落在各 tableau 稳定域内），
   逐级加倍直到误差跨越 ≥3 个量级或触底。
4. **舍入底噪（关键）**：每个 level 计算累积舍入底 `eps·‖u‖·sqrt(nsteps·stages)`，
   只接受 `err > 3×底噪` 的 level 参与拟合——这正是第 2 轮 `M5_dt_order` 被判仪器缺陷的根因
   （细端读的是舍入噪声，出现负斜率）。**本单第一版仍踩了一次**：`p=4, rk5` 读到 3.67 阶，
   收紧底噪后为 5.03 阶（见 ⑤ 坑 1）。
5. **对照实验（空间底）**：固定 `N=40, p=2` 用 shipped 时间循环 `solve_advection` 细分 `dt`，
   误差停在 `1.337e-05` 不动，成对阶依次为
   0.035, 0.001, 0.000, 0.000——证明"固定网格细分 dt"这一仪器原理上测不出时间阶。
6. **工具链**：PyTorch float64（生产 RHS）+ NumPy longdouble（精确基准）+ pytest（断言）。

## ③ 计算结果（程序化注入）

### 3.1 时间阶阶梯总表

| N | p | integrator | matched | 名义 q | 观测 q | 偏差(阶) | 相对偏差 | levels |
|--:|--:|---|:--:|--:|--:|--:|--:|--:|

| 8 | 0 | euler | yes | 1 | 1.0022 | +0.002 | 0.22% | 10 |
| 8 | 1 | rk4 | no | 4 | 4.0002 | +0.000 | 0.00% | 4 |
| 8 | 1 | rk2 | yes | 2 | 2.0000 | +0.000 | 0.00% | 6 |
| 8 | 2 | rk3 | yes | 3 | 3.0000 | -0.000 | 0.00% | 5 |
| 8 | 3 | rk4 | yes | 4 | 4.0012 | +0.001 | 0.03% | 4 |
| 8 | 4 | rk2 | no | 2 | 2.0000 | -0.000 | 0.00% | 6 |
| 8 | 4 | rk5 | yes | 5 | 5.0279 | +0.028 | 0.56% | 4 |
| 16 | 0 | euler | yes | 1 | 1.0013 | +0.001 | 0.13% | 10 |
| 16 | 1 | rk5 | no | 5 | 5.0188 | +0.019 | 0.38% | 4 |
| 16 | 1 | rk2 | yes | 2 | 2.0000 | +0.000 | 0.00% | 6 |
| 16 | 2 | rk3 | yes | 3 | 3.0000 | -0.000 | 0.00% | 5 |
| 16 | 3 | rk4 | yes | 4 | 4.0003 | +0.000 | 0.01% | 4 |
| 16 | 4 | rk5 | yes | 5 | 5.1412 | +0.141 | 2.82% | 4 |

### 3.2 锚点 premise 的直接核查

锚点的空间阶扫描用 `q = p+1` 的 matched 配对，前提是**时间误差远小于空间误差**。本单用半离散精确解直接验证：
matched 配对在 case-pool CFL 下，时间误差的绝对量级见 `artifacts/case_dg_advect_order_time_order.json`
的 `levels`（每级都记录 `error`/`floor`/`resolved`）。结论：`p ≥ 1` 时时间误差在阶梯最粗一级已低于
空间误差两个量级以上；`p = 0` 用 `euler` 时时间误差不可忽略（这一点第 2/3 轮的误差拆分已量化，
本单的 euler 阶梯再次复现了其一阶衰减）。

### 3.3 空间底对照（固定 N=40, p=2, rk3）

| nsteps | dt | L2 误差 vs 解析 sine |
|--:|--:|--:|
| 50 | 2.000e-02 | 1.3704e-05 |
| 100 | 1.000e-02 | 1.3379e-05 |
| 200 | 5.000e-03 | 1.3372e-05 |
| 400 | 2.500e-03 | 1.3372e-05 |
| 800 | 1.250e-03 | 1.3372e-05 |

成对阶：0.035, 0.001, 0.000, 0.000 → 误差不再随 `dt` 下降（阶 → 0），
与第 2 轮 `M5_dt_order = not_run` 的记录一致：仪器受限，非格式缺陷。

## ④ 结果图

- `docs/figures/case_dg_advect_order_time_order.png`（两张子图）：
  左 = 各 tableau 的误差 vs 时间步数（对 `exp(LT)u0`，N=16, p=2），灰线为 `dt^q` 参考斜率；
  右 = 固定网格细分 `dt` 的误差平台（空间底的实证）。

![temporal order](case_dg_advect_order_time_order.png)

## ⑤ 注意点（坑与局限）

- **坑 1（舍入底伪装成低阶）**：`p=4, N=8, rk5` 第一版读到 3.67 阶（声称 rk5 只有 3.7 阶）。
  原因是阶梯末端误差 4e-14–8e-15，已等于 float64 累积舍入量级；改用
  `eps·‖u‖·sqrt(nsteps·stages)` 底噪 + `>3×` 门槛后为 **5.03 阶**。这条与第 2 轮的负斜率是同一类失效的两种表现。
- **坑 2（稳定的伪命题）**：`nsteps0 = ceil(‖L‖₂)` 才保证落入所有显式 tableau 的稳定域；
  直接用 `cfl=0.2` 的物理 CFL 会让 euler 在细网格爆掉（首版读数 5e41）。
- **局限 1**：`(N=16, p=4, rk5)` 偏差 +0.14 阶（2.8%），是因为该配置在 float64 内只能覆盖约
  1.5 个量级的"干净"区间（更高阶意味着更快触底），属于预渐近/舍入的固有夹缝；更严的结论需要
  四倍精度（mpmath）基准，本单未做。
- **局限 2**：只测了线性常系数问题。非线性（Euler/Burgers）里"时间阶"会被空间算子非线性耦合污染，
  本单的方法不直接适用。
- **局限 3**：`space_floor` 对照表显示的是**平台**，不做阶的断言（正是它的用途）。

## ⑥ 交付物

- 驱动：`benchmarks/case_dg_advect_order_time_ladder.py`（`--only-8/--only-16` 可分段跑）
- 断言：`tests/test_case_dg_advect_order_time_order.py`（7 项，含仪器自检与空间底对照）
- 数据：`artifacts/case_dg_advect_order_time_order.json`
- 图：`docs/figures/case_dg_advect_order_time_order.png`
