# CASE-DG-ADVECT-ORDER · P0 行：用闭式半离散精确参照复核 DG0 收敛阶

**锚点** CASE-DG-ADVECT-ORDER ｜ **本轮增量** P0（degree=0）行：把"粗梯观测阶 0.85 ≠ 设计阶 1"
从**断言（放宽容差到 ±0.7）**升级为**证明（闭式精确参照 + 渐近区间定位 + 常数闭式）**｜
**仓库** TensorDG ｜ **分支** agent/m20521 ｜ **总判定** **PASS**（10/10 判据通过）

## ① 算例介绍

1-D 周期线性对流 $u_t + c u_x = 0$，$x\in[0,1)$ 周期，$c=1.0$，$u(x,0)=\sin(2\pi x)$，
$t^*=1.0$（一个周期），manufactured solution 精确解 $\sin 2\pi(x-ct)$。
本行检验的是 **P0 子问题**：DG 次数 0 时，upwind DG 退化为经典一阶迎风有限体积，
其半离散算子**循环（circulant）**，因此在离散 Fourier 空间可闭式对角化——
于是"空间离散误差"对任意网格 $N$ 都有**机器精度参照**，不依赖时间推进、不依赖参考网格、不含任何手抄数字。

| 项 | 符号/定义 | 值 |
|---|---|---|
| 方程 | $u_t+c u_x=0$ | 标量线性对流 |
| 波速 / 域长 | $c$, $L$ | 1.0 / 1 |
| 无量纲时间 | $t^*=cT/L$ | 1.0 |
| 每波长单元数 | $N$ | 20 … 40960 |
| 单元尺寸 | $h=1/N$ | 2.44e-05 … 5.00e-02 |
| 多项式阶 | $p$ | 0（DG0 ≡ 一阶迎风 FV） |
| 时间格式 | rk4 (repo default) | 4 阶 |
| CFL | $c\Delta t/h$ | 0.2 |
| 池梯（第 1/2 轮用的） | $N$ | 20, 40, 80, 160, 320 |

## ② 怎么做的

1. **闭式半离散解**（`tensordg/p0_exact.py`）。DG0 迎风算子
   $du_j/dt = -(c/h)(u_j-u_{j-1})$ 是循环矩阵，符号
   $\lambda_k=-(c/h)(1-e^{-i\theta_k})$，$\theta_k=2\pi k/N$；故
   $\hat u_k(t)=e^{\lambda_k t}\hat u_k(0)$。初值单元平均
   $\bar u_j=[\cos 2\pi a_j-\cos 2\pi b_j]/(2\pi h)$ 也是解析的。
2. **逐格正交分解**：$(1/h)\int_{\text{cell}}(u_j-u_{\rm exact})^2=(u_j-m_1)^2+(m_2-m_1^2)$，
   $m_1,m_2$ 为精确正弦在该格的解析均值、均方。第一项=**格式误差**，第二项=$O(h^2)$ **投影地板**。
   于是 $L^2$ 误差也闭式（无需数值求积）。
3. **独立三路验证**（`tests/test_p0_semidiscrete_exact.py`）：
   (a) 解析符号 vs 从求解器 `_rhs` 逐列装配出的算子矩阵；
   (b) 闭式解 vs `expm(M·T)·u0`（完全独立的线性代数路径）；
   (c) 闭式 $L^2$ vs 逐格 12 点 Gauss 暴力求积。三者均在 1e-12 内一致。
4. **与既有求解器对接**：`tensordg.advection_1d.solve_advection(degree=0, rk4, cfl=0.2)`
   原样调用（未改一行求解器代码），用同一 CL2 泛函比较；另做 CFL 扫描隔离时间误差。
5. **判据**（10 项，全部事前写进脚本常量）：$|\Delta$阶$|<2e-3$（求解器 vs 解析）、
   细梯阶 → 1（±2e-3）、渐近常数 vs 闭式（≤1e-3 相对）、时间项占比 ≤5% 等。
   使用**逐对（两点）局部阶**定位渐近区间，而不是只报一个全局拟合斜率。

## ③ 计算结果

| N（cells/wavelength） | h | 半离散精确 $L^2$ | $N\cdot E$ | 局部阶 | 求解器全离散 $L^2$ | 时间项分量 |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 5.000e-02 | 4.4646e-01 | 8.9291 | 0.6890 | 4.4646e-01 | 2.13e-07 |
| 40 | 2.500e-02 | 2.7693e-01 | 11.0772 | 0.8337 | 2.7693e-01 | 2.19e-08 |
| 80 | 1.250e-02 | 1.5539e-01 | 12.4308 | 0.9142 | 1.5539e-01 | 1.76e-09 |
| 160 | 6.250e-03 | 8.2452e-02 | 13.1923 | 0.9565 | 8.2452e-02 | 1.25e-10 |
| 320 | 3.125e-03 | 4.2488e-02 | 13.5962 | 0.9781 | 4.2488e-02 | 8.31e-12 |
| 640 | 1.563e-03 | 2.1569e-02 | 13.8043 | 0.9890 | 2.1569e-02 | 6.97e-13 |
| 1280 | 7.813e-04 | 1.0867e-02 | 13.9099 | 0.9945 | 1.0867e-02 | 6.07e-14 |
| 2560 | 3.906e-04 | 5.4543e-03 | 13.9631 | 0.9972 | 5.4543e-03 | 1.65e-13 |
| 5120 | 1.953e-04 | 2.7324e-03 | 13.9898 | 0.9986 | 2.7324e-03 | 3.77e-13 |
| 10240 | 9.766e-05 | 1.3675e-03 | 14.0031 | 0.9993 | — | — |
| 20480 | 4.883e-05 | 6.8407e-04 | 14.0098 | 0.9997 | — | — |
| 40960 | 2.441e-05 | 3.4212e-04 | 14.0132 | — | — | — |

（求解器列只到 $N=5120$：完全离散跑这么细只为了证明时间项可忽略；更细的网格由闭式给出。）

细梯观测阶（同一最小二乘估计器，仅换窗口）：

| 拟合窗口 | 观测阶 |
|---|---|
| N>=20 | 0.95820 |
| N>=320 | 0.99472 |
| N>=640 | 0.99679 |
| N>=1280 | 0.99802 |
| N>=2560 | 0.99876 |
| N>=5120 | 0.99921 |

CFL 扫描（$N=320$，隔离时间误差）：

| N | steps | dt | 全离散 $L^2$ | 时间项分量 |
|---:|---:|---:|---:|---:|
| 320 | 800 | 1.250e-03 | 4.248822e-02 | 1.325e-10 |
| 320 | 1600 | 6.250e-04 | 4.248822e-02 | 8.314e-12 |
| 320 | 3200 | 3.125e-04 | 4.248822e-02 | 5.542e-13 |
| 320 | 6400 | 1.563e-04 | 4.248822e-02 | 6.935e-14 |

## ④ 结果图

- `docs/figures/case_dg_p0_convergence.png`：半离散精确误差（实线圆点）、求解器全离散（空心方块）、
  斜率 1 参照线、投影地板 $\pi h/\sqrt6$、以及池梯阴影区。
- `docs/figures/case_dg_p0_local_order.png`：局部阶随 $N$ 的爬升 0.69 → 0.96 → 1.00，池梯落在浅区。
- `docs/figures/case_dg_p0_profile.png`：$N=20$ 的粗网格场量与逐点误差（幅度衰减是误差主源）。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **渐近常数不是 $\pi^2\sqrt2$。** 只用"幅度衰减率 $2\pi^2/N$"会得到
   $E\to\pi^2\sqrt2/N=13.9577/N$，但实测
   $N\cdot E\to14.01322$，高 **0.40%**。
   差额正是 $O(h^2)$ 投影地板 $\pi h/\sqrt6=1.2825/N$；
   两项平方和给出闭式 $\pi\sqrt{2\pi^2+1/6}=14.01653$，与实测差
   **0.024%**。这是"常数差 0.4% 就宣称实现有 bug"类误判的根源。
2. **0.853 是误差曲线的性质，不是求解器缺陷。** 闭式误差在池梯上拟合得
   0.853467，求解器 0.853467，两者差
   **7.7e-08**。粗网格上 $e^{-2\pi^2/N}$ 在 $N=20$ 时仅 0.37，
   高阶项不可忽略，故斜率被压低。
3. **渐近区间是要量的，不是猜的。** 局部阶进入 5%/2%/1% 带分别需要
   $N\ge160$ / 640 / 1280。
   池梯 $N\le320$ **整段都在 2% 带外**——这也解释了为什么其它阶（P1–P4）在池梯上看起来好得多：
   它们的渐近开始得早（高阶空间误差的修正项更小）。
3b. **求解器侧的渐近陈述是两点斜率**：细梯窗口 $N\ge2560$ 只含 $N=2560,5120$ 两个点
   （完全离散再往细跑只增成本），故求解器侧只主张 $|$阶$-1|\le3\times10^{-3}$，不单独据此下结论；
   支撑它的更强证据是**全部** $N=20\dots5120$ 上求解器与闭式误差的相对差
   $\le2.5e-07$——该证据不依赖任何拟合。
4. **float64 底噪**：解析式在 $N\gtrsim10^6$ 时因 $u_j-m_1$ 的相对对消（$\sim10^{-5}$）损失有效位，
   $N\cdot E$ 反而降到 13.73；本报告的常数取 $N=40960$，未越界使用。
5. **适用范围**：只覆盖 degree=0、均匀周期网格、标量常系数对流。P1–P4 的算子非循环，
   没有同类闭式解，其行仍以文献 ref 列（arXiv:2404.15453v2）为准（第 1/2 轮已做）。
6. **与在飞分支的关系**：另一个在飞分支把 2D P0 的测试容差放宽到 ±0.7；本工作给出的是
   可计算的判据替代（渐近区间 + 闭式常数），不与其冲突，但**不认可"放宽容差"作为验收证据**。

## ⑥ 误差判定（是否 <3%）

关键量 vs 基准（基准=闭式解析，非二手数值）：

| 量 | 数值 | 相对偏差 | 判定 |
|---|---|---:|---|
| 池梯 N=20…320 观测阶（解析基准） vs 求解器 | 0.853467 vs 0.853467 | 9.01e-06% | ✅ |
| 细梯 N≥20480 观测阶 vs 设计阶 1 | 0.999206 | 0.079% | ✅ |
| 求解器细梯 N≥2560 观测阶 vs 设计阶 1 | 0.997245 | 0.276% | ✅ |
| 渐近常数 N·E(N) vs 闭式 π√(2π²+1/6) | 14.01322 vs 14.01653 | 0.024% | ✅ |
| 求解器逐网格 L2 vs 精确半离散 L2（最大相对差）| 2.55e-07 | 0.00003% | ✅ |

**结论：是——求解器与闭式解析基准的相对偏差全部 <3%**（最差一项为求解器细梯观测阶
0.276%）。池梯观测阶 0.853 **不是**与基准的偏差：
解析基准在同一梯上给出同一数值（差 7.7e-08），故该项为"预渐近"而非"失准"。

## 判据逐项

| 判据 | 结果 |
|---|---|
| `A10_solver_matches_exact_pointwise` | ✅ |
| `A1_solver_reproduces_pool_order` | ✅ |
| `A2_pool_order_is_preasymptotic` | ✅ |
| `A3_fine_ladder_recovers_design_order` | ✅ |
| `A4_solver_fine_ladder_recovers_design_order` | ✅ |
| `A5_asymptotic_constant_closed_form` | ✅ |
| `A6_two_term_closed_form_identity` | ✅ |
| `A7_time_error_negligible_on_pool` | ✅ |
| `A8_time_error_vanishes_with_cfl` | ✅ |
| `A9_local_order_at_finest` | ✅ |

## 被否定的假设（负面结果入账）

1. **H_P0_defect** — 'the 0.85 observed P0 order on N=20..320 means the DG0 operator is below design order'. Falsified: the closed-form semidiscrete error (no time stepping, no solver) gives the same fitted order on the same ladder, and the design order 1.00 is recovered on the extended ladder.

2. **H_constant** — 'N*E(N) -> pi^2*sqrt(2) = 13.9577' (the scheme-damping estimate alone). Falsified: the measured value at N=40960 is 14.01322, i.e. 0.40% above it, while the two-term closed form pi*sqrt(2*pi^2 + 1/6) = 14.01653 is matched to 0.024%. The gap is the O(h^2) piecewise-constant projection floor pi^2*h^2/6, not a solver error.

3. **H_time_error** — 'the DG0 pool-ladder error is contaminated by RK4 time integration'. Falsified at cfl=0.2: time share <= 4.8e-07 on the pool ladder and the fully discrete solver matches the semidiscrete exact error to 2.5e-07 relative over N=20...5120.

## 工件

| 文件 | 内容 |
|---|---|
| `tensordg/p0_exact.py` | 闭式半离散解/谱/$L^2$ 误差、闭式常数（DG0 循环算子） |
| `benchmarks/case_dg_p0_analytic.py` | 本算例编排：10 项判据、阶梯、CFL 扫描、图、表 |
| `tests/test_p0_semidiscrete_exact.py` | 12 个测试：符号/谱/expm/暴力求积/池梯阶/渐近常数/求解器对接 |
| `validation/advection_order_p0/metrics.json` | 全部数值 + 判据 + 假设否定记录 |
| `validation/advection_order_p0/convergence.csv` | 逐网格解析/求解器误差与局部阶 |
| `validation/advection_order_p0/tables.md` | 本报告表格渲染源（程序化注入） |
| `docs/figures/case_dg_p0_{convergence,local_order,profile}.png` | 收敛曲线 / 局部阶 / 粗网格场量 |

## 一键复现

```bash
PYTHONPATH=. OMP_NUM_THREADS=8 python -m benchmarks.case_dg_p0_analytic   # 生成 metrics/表/图
PYTHONPATH=. python tools/render_p0_doc.py                                # 重渲染本文件
PYTHONPATH=. python -m pytest tests/test_p0_semidiscrete_exact.py -q      # 12 项测试
```
