# CASE-DG-EULER-VORTEX-ERR：参考解与量测仪表认证（程序化生成）

整包裁决：**INSTRUMENT-LIMITED**（5 项证书，失败 ['R6_reproduction']；突变对照 全部触发）

通过判据统一为「仪表误差 ≤ 被判量 / 10」：仪表若不比它认证的量细一个量级，该量就是仪表限而非物理。

## R1 解析参考解是否真是 Euler 方程的解

| 时刻 | 方程 | 残差绝对最大值 | 残差/最大通量散度项 |
|---|---|---|---|
| t=0 | energy | 8.882e-16 | 7.717e-16 |
| t=0 | mass | 1.665e-16 | 4.167e-16 |
| t=0 | momentum_x | 4.441e-16 | 3.020e-15 |
| t=0 | momentum_y | 4.441e-16 | 2.012e-15 |
| t=1 | energy | 1.332e-15 | 6.476e-16 |
| t=1 | mass | 1.665e-16 | 3.950e-16 |
| t=1 | momentum_x | 4.441e-16 | 6.842e-15 |
| t=1 | momentum_y | 4.441e-16 | 6.200e-16 |
| t=2.5 | energy | 1.332e-15 | 2.862e-15 |
| t=2.5 | mass | 1.110e-16 | 5.639e-16 |
| t=2.5 | momentum_x | 3.331e-16 | 8.445e-16 |
| t=2.5 | momentum_y | 3.331e-16 | 1.679e-15 |

最差相对残差 6.842e-15（限 1e-10）：通过。周期折叠的平移不变性 9.839e-16（通过）。

| 突变（合理误抄） | 残差/最大项 | 是否被检出 |
|---|---|---|
| envelope_exponent | 2.356e+00 | 是 |
| temperature_factor | 3.960e-01 | 是 |
| velocity_one_sign | 2.546e+00 | 是 |

## R2 等熵结构

解析场 max|p/rho^gamma - 1| = 0.000e+00（限 1e-14）：通过；等温突变体读数 2.637e-01（已检出）。

## R3 不变量分母（C3 的分母）

| Gauss 点数/方向 | energy | entropy | mass | momentum_x | momentum_y |
|---|---|---|---|---|---|
| 1280 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 320 | 3.30e-16 | 2.84e-16 | 4.34e-16 | 2.89e-16 | 2.89e-16 |
| 480 | 1.32e-15 | 2.84e-16 | 1.45e-15 | 1.88e-15 | 1.59e-15 |
| 640 | 1.15e-15 | 1.42e-16 | 1.45e-15 | 1.59e-15 | 1.59e-15 |
| 960 | 1.65e-15 | 1.42e-16 | 1.88e-15 | 1.88e-15 | 2.03e-15 |

门禁用的 320 点：最差相对偏差 4.340e-16，相对最小漂移 5.50e-15 的比值为 7.89e-02（要求 ≤ 1e-01）：通过。

核密度采样：张量网格最小值 0.494290680，射线最小值 0.493807324，相对差 9.79e-04。`density_min` is a minimum *over quadrature nodes*, so the tensor grid samples the core off-centre; the gate uses it only as a same-instrument drift (final vs initial), which is unaffected, but it is not the true core minimum。

## R4 相对 L2 误差范数的积分收敛性

| 状态 | 门禁求积读数 | 4× 求积读数 | 相对漂移 | 欠积分对照 | 复现存储值 |
|---|---|---|---|---|---|
| projection_p1_n32 | 0.112983% | 0.112990% | -0.0065% | -95.04% | — |
| projection_p2_n32 | 0.005492% | 0.005496% | -0.0716% | -93.58% | — |
| projection_p3_n32 | 0.000350% | 0.000350% | -0.0245% | -93.44% | — |

最差相对漂移 7.164e-04，相对容差 6% 的比值 1.194e-02（要求 ≤ 1e-01）：通过。

## R6 复现：已入库读数是本求解器现算的吗

判据：re-solve the declared checkpoint in a fresh process and compare every published scalar with results/*.json; |fresh - stored| / |stored| > 1e-09 means the bundle is stale；覆盖 4/40 个梯级检查点，每点在不同线程数 [1, 4] 下各重解一次。

| 检查点 | 作用 | 线程数 | 最差相对偏差 | 出现在 | 步数一致 |
|---|---|---|---|---|---|
| `dg_d2_n32_t1_cfl0.2` | C1/C2 门禁判定点（设计族快照） | 1 | 9.480e-09 | entropy_drift | 是 |
| `dg_d2_n32_t1_cfl0.2` | C1/C2 门禁判定点（设计族快照） | 1_repeat | 0.000e+00 | max_relative_l2 | 是 |
| `dg_d2_n32_t1_cfl0.2` | C1/C2 门禁判定点（设计族快照） | 4 | 9.480e-09 | entropy_drift | 是 |
| `dg_d2_n32_t1_cfl0.2` | C1/C2 门禁判定点（设计族快照） | 4_repeat | 0.000e+00 | max_relative_l2 | 是 |
| `dg_d2_n32_t10_cfl0.2` | 最长梯级（4684 步，检验偏差是否随步数增长） | 1 | 4.521e-11 | density_min_relative_error | 是 |
| `dg_d2_n32_t10_cfl0.2` | 最长梯级（4684 步，检验偏差是否随步数增长） | 4 | 4.521e-11 | density_min_relative_error | 是 |
| `dg_d3_n32_t1_cfl0.2` | C8 最高阶族快照 | 1 | 0.000e+00 | max_relative_l2 | 是 |
| `dg_d3_n32_t1_cfl0.2` | C8 最高阶族快照 | 1_repeat | 0.000e+00 | max_relative_l2 | 是 |
| `dg_d3_n32_t1_cfl0.2` | C8 最高阶族快照 | 4 | 0.000e+00 | max_relative_l2 | 是 |
| `dg_d3_n32_t1_cfl0.2` | C8 最高阶族快照 | 4_repeat | 0.000e+00 | max_relative_l2 | 是 |
| `p0_n128_t1_cfl0.4` | C5 第二条代码路径（P0 有限体积） | 1 | 3.365e-15 | density_min_relative_error | 是 |
| `p0_n128_t1_cfl0.4` | C5 第二条代码路径（P0 有限体积） | 1_repeat | 0.000e+00 | max_relative_l2 | 是 |
| `p0_n128_t1_cfl0.4` | C5 第二条代码路径（P0 有限体积） | 4 | 3.365e-15 | density_min_relative_error | 是 |
| `p0_n128_t1_cfl0.4` | C5 第二条代码路径（P0 有限体积） | 4_repeat | 0.000e+00 | max_relative_l2 | 是 |

最差相对偏差 9.480e-09（dg_d2_n32_t1_cfl0.2/threads=1/entropy_drift），限 1e-09，比值 9.480e+00（要求 ≤ 1）：未通过；「10 倍裕度」**不保持**。跨线程数包络 9.480e-09（dg_d2_n32_t1_cfl0.2/entropy_drift）。

| 合成陈旧读数（把扰动值放进入库槽） | R6 读到的偏差 | 是否判陈旧 |
|---|---|---|
| magnitude_1e-11 | 1.000e-11 | 否 |
| magnitude_1e-10 | 1.000e-10 | 否 |
| magnitude_1e-09 | 1.000e-09 | 是 |
| magnitude_1.001e-09 | 1.001e-09 | 是 |
| magnitude_1e-08 | 1.000e-08 | 是 |
| wrong_checkpoint | 2.668e-02 | 是 |
| wrong_family | 3.780e-01 | 是 |

翻转点：magnitude_1e-09。a one-sided gate: it can only reject staleness above the limit, so the probe that matters is the smallest flagged perturbation relative to the measured reproduction envelope。

固有确定性（同设置、第二个新进程重解）：最差 0.000e+00（None）——为 0 即「同机同设置位级可复现」，故与入库读数之间的残余偏差是固定的代码/环境偏移而非进程抖动。重复的检查点：dg_d2_n32_t1_cfl0.2, dg_d3_n32_t1_cfl0.2, p0_n128_t1_cfl0.4。

量测装置对照（继承环境 vs 钉死线程变量）：最差 9.480e-09——pinned vs inherited OMP/MKL environment: a harness that changes the answer is not measuring the bundle。
