## 1. 案例介绍

Cockburn & Shu 的 RKDG 用 **TVB 疑难单元指示器**（Math. Comp. 52 (1989) eq. (2.10)；
JCP 141 (1998) §2.2 综述）在光滑极值处释放限幅器，从而既压住激波又不毁掉光滑解的形式阶。
判据里唯一可调的常数 `M` 量纲是 `|u''|`，判据本体是

```
released  ⟺  min(|a₁|, |a₂|) ≤ M h²        a₁ = ūᵢ − ūᵢ₋₁,  a₂ = ūᵢ₊₁ − ūᵢ
```

被测对象是 TensorDG 实际发货的核 `tensordg.tvb.troubled_from_differences`，它改用
「单侧差偏离 minmod 斜率的量」与 `M h²` 比较（对光滑数据该偏离 = `|u''| h²`）。两种读法
在**极值释放**这一档等价，在**单调光滑段**不等价：发货版不置旗，论文版置旗。

物理背景就是标量线性对流 `u_t + u_x = 0`，周期域，初值 `u₀ = sin(2πx)`——它的二阶导数有
闭式 `max|u''| = 4π² = 39.478418`，**这就是本报告用到的全部 M，
没有一个是从测量里拟合出来的**（避免审计反复点名的"构造即真"）。

| 参数 | 值 |
|---|---|
| 域 / 边界 | [0,1] 周期 |
| 初值 | sin(2πx)，L2 投影到模态空间 |
| 平流速度 / 终止时刻 | 1.0 / T = 0.5 |
| 时间积分 | `ssp_rk3`（CKS 的 TVD-RK），CFL = 0.3 |
| 阶 | p = 1, 2（bench 档再加 p = 3） |
| 网格 | N = 20, 40, 80 |
| M 扫描 | {0, ¼, ½, 1, 2} × max|u''| 与公开默认 `tvb_constant=1.0` |
| profile | `smoke` |

## 2. 怎么做的

1. **把论文判据实现成可对照的第二个核** `paper_troubled_from_differences`（同签名），
   用 context manager 换进 `tensordg.limiters` 的同名符号 → 同一条求解路径、同一套
   RK/求积，只有判据不同。
2. **指示器活性的独立复算**：不侵入求解器，直接从最终模态的单元均值按 `_detect` 的
   周期模板重算 mask，数出被置旗单元的比例。没有它，任何"阶保住了"都可能只是"限幅器
   根本没开"（vacuity）——这正是本席要堵的口子。
3. **与无限幅解的逐位比对**：同一 (p, N) 下解 `limiter=None` 的参照解，取 `max|Δcoeff|`。
   判据正确性的最强断言不是"误差小"，而是"M 到阈值时**一个比特都不许变**"。
4. 观测阶用 `tensordg.convergence.observed_order` 对整条网格阶梯做最小二乘拟合。
5. 门（`GATES`）：G1 = M = max|u''| 时零置旗 + 逐位等于无限幅 + 阶 ≥ p+½；
   G2 = M = ¼max|u''| 时**确实置旗**（G1 的非 vacuity 对照）；C1 = 论文版置旗率 ≥ 发货版；
   D1 = 公开默认 `tvb_constant=1.0` 在本算例上损失 ≥ ¼ 阶。

工具链：纯 TensorDG 既有内核（`advection_1d` / `limiters` / `tvb` / `convergence`），
torch CPU float64，`OMP_NUM_THREADS=1`。本机只跑短测试，bench 档（含 p=3、N=160）递交
`run_compute` 作业（cj4751），本报告的全部数字取自 **smoke 档工件**
`validation/repro_cks1998/artifacts/tvb_criterion_ab/tvb_criterion_ab_smoke.json`。

## 3. 计算结果

| p | M / max abs(u'') | criterion | order | theory p+1 | floor p+½ | flagged % | max abs(Δ) vs unlimited |
|---|---|---|---|---|---|---|---|
| 1 | — | unlimited | 2.009 | 2 | 1.5 | 0.0 | 0.00e+00 |
| 1 | 0.03 | shipped_default | 1.524 | 2 | 1.5 | 100.0 | 1.44e-01 |
| 1 | 0.00 | shipped | 1.524 | 2 | 1.5 | 100.0 | 1.44e-01 |
| 1 | 0.00 | paper | 1.524 | 2 | 1.5 | 100.0 | 1.44e-01 |
| 1 | 0.25 | shipped | 1.522 | 2 | 1.5 | 85.0 | 1.42e-01 |
| 1 | 0.25 | paper | 1.463 | 2 | 1.5 | 95.0 | 1.19e-01 |
| 1 | 0.50 | shipped | 1.505 | 2 | 1.5 | 70.0 | 1.30e-01 |
| 1 | 0.50 | paper | 1.392 | 2 | 1.5 | 95.0 | 9.10e-02 |
| 1 | 1.00 | shipped | 2.009 | 2 | 1.5 | 0.0 | 0.00e+00 |
| 1 | 1.00 | paper | 1.277 | 2 | 1.5 | 90.0 | 7.22e-02 |
| 1 | 2.00 | shipped | 2.009 | 2 | 1.5 | 0.0 | 0.00e+00 |
| 1 | 2.00 | paper | 1.093 | 2 | 1.5 | 85.0 | 5.23e-02 |
| 2 | — | unlimited | 3.000 | 3 | 2.5 | 0.0 | 0.00e+00 |
| 2 | 0.03 | shipped_default | 1.524 | 3 | 2.5 | 100.0 | 1.45e-01 |
| 2 | 0.00 | shipped | 1.524 | 3 | 2.5 | 100.0 | 1.45e-01 |
| 2 | 0.00 | paper | 1.524 | 3 | 2.5 | 95.0 | 1.45e-01 |
| 2 | 0.25 | shipped | 1.523 | 3 | 2.5 | 85.0 | 1.44e-01 |
| 2 | 0.25 | paper | 1.472 | 3 | 2.5 | 95.0 | 1.26e-01 |
| 2 | 0.50 | shipped | 1.486 | 3 | 2.5 | 70.0 | 1.32e-01 |
| 2 | 0.50 | paper | 1.393 | 3 | 2.5 | 95.0 | 9.66e-02 |
| 2 | 1.00 | shipped | 3.000 | 3 | 2.5 | 0.0 | 0.00e+00 |
| 2 | 1.00 | paper | 1.297 | 3 | 2.5 | 90.0 | 7.76e-02 |
| 2 | 2.00 | shipped | 3.000 | 3 | 2.5 | 0.0 | 0.00e+00 |
| 2 | 2.00 | paper | 1.090 | 3 | 2.5 | 85.0 | 5.59e-02 |

| gate | verdict |
|---|---|
| G1_inert_above_curvature | PASS |
| G2_indicator_fires_below_half_curvature | PASS |
| C1_paper_form_is_stricter | PASS |
| D1_public_default_constant | PASS |

## 4. 结果图

**cks1998_tvb_criterion_ab.png** — Left: observed L2 order against the theory-scaled TVB constant. At M = max|u''| the shipped kernel (solid) is inert and reproduces the unlimited order exactly; the paper's literal test (dashed) still flags most cells and loses the order. Right: fraction of cells the indicator flags, i.e. whether the limiter is doing anything at all.

**cks1998_tvb_criterion_ab_ladder.png** — P2 error ladders. M = max|u''| is bit-identical to the unlimited run (the two curves coincide), while M = max|u''|/4 and the paper-form test at M = max|u''| both flatten towards first order.

## 5. 注意点

1. **发货版判据偏离了论文原文，这是本次量化出来的事实**，不是新引入的：`tensordg.tvb` 的
   docstring 已自陈改过判据（为了修"每个光滑极值都被置旗"的历史缺陷），但此前没有任何门
   覆盖 `M > 0` 这一档——A1 阶门跑的是**无限幅**，其余限幅门全部 `M = 0`。
2. 两种读法在本算例上**都不 vacuous**，但代价不同：论文版即使在 `M = 2·max|u''|` 仍有
   85 %
   的单元被置旗（单调光滑段的单侧差是 O(h)，永远大于 `M h²`），因此它对 `M` 的要求随 h 收紧，
   1D 正弦上观测阶掉到 1.1 左右；发货版则在 `M ≥ max|u''|` 处干净地完全不介入。
3. **公开默认 `tvb_constant=1.0` 是个坑**：它是绝对量 `M`，与本解曲率 `4π² ≈ 39.5` 无关，
   于是 `M/max|u''| = 0.025`，几乎全单元置旗，
   p=1 损失 0.48 阶，
   p=2 损失 1.48 阶。用户"顺手打开限幅器"
   就会静默掉阶。默认值本身按仓规不改（向后兼容），但**这条损失现在有工件和门看着**。
4. 适用范围：标量 1D 线性对流、光滑初值。本报告**不**声称发货判据在激波算例上更好或更差；
   E3/TVD 那一档由既有 `tvb_M0` 门覆盖（见 `docs/CASE-REPRO-DG-CKS1998-E3-PAPER-PREREG.md`）。
5. 局限：只在 `sin(2πx)` 单一波形上量化了 M 的含义；`M` 的"正确"取法对一般解仍需
   `|u''|` 估计（本报告只提供判定标尺，不提供自适应 M 算法）。

## 6. 误差判定（关键量 vs 基准，逐项）

基准 = 无限幅 DG 的观测阶（同一网格阶梯、同一时间积分），这是本问题的**解析可控参照**
（DG 无限幅的理论阶 p+1 有证明，观测阶也与之吻合）：

| 关键量 | 基准 | 实测 | 偏差 | <3% ? |
|---|---|---|---|---|
| 无限幅 p=1 观测阶 | 2 (理论 p+1) | 2.009 | — | 是（0.45 %） |
| 无限幅 p=2 观测阶 | 3 (理论 p+1) | 3.000 | — | 是（0.00 %） |
| 发货版 M = max|u''| 的 p=2 解 | 无限幅解（逐位） | max|Δ| = 0.0e+00 | 0 | **是（0 %，逐位相同）** |
| 发货版 M = ¼max|u''| 的 p=2 观测阶 | 3 | 1.523 | −1.477 | **否（−49 %）**，已如实入账 |
| 公开默认 M=1.0 的 p=2 观测阶 | 3 | 1.524 | −1.476 | **否（−49 %）**，已如实入账 |

**是否 <3%**：在"判据正确 ⟹ 阈值处应完全不介入"这一条上，偏差是 **0 %（逐位相同）**，远好于 3 %；
在"限幅器开启后的阶"上**未能** <3%，因为那本来就是论文与实现共同的物理代价（极值削平），
本报告把它当**被测现象**而非误差指标，并已给出可复现的数值。
