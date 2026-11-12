# CASE-DG-PREFINEMENT-EXPONENTIAL: P-type Exponential Convergence Benchmark

## 1. Problem Statement & Reference

**Case anchor:** `CASE-DG-PREFINEMENT-EXPONENTIAL`

**Reference:** Babuska, I. & Suri, M. (1994). "The p- and h-p versions of the finite element method, basic principles and properties." *SIAM Review*, 36(4), 578-632. DOI: [10.21236/ada260237](https://doi.org/10.21236/ada260237)

**Key theoretical result (Theorem 3.1, Section 3.2):** For analytic solutions, the DG/FEM error decays exponentially in the polynomial degree:

$$\|u - u_h\|_{L^2} \leq C \exp(-\sigma p)$$

where $\sigma > 0$ depends on the analyticity strip of the solution. For entire functions (analytic everywhere), convergence is particularly rapid.

**Problem:** 1D linear advection equation
$$u_t + u_x = 0, \quad x \in [0,1], \text{ periodic BC}$$

**Initial condition:** $u(x,0) = \sin(2\pi x)$ — an entire function, analytic on the whole complex plane.

**Exact solution:** $u(x,t) = \sin(2\pi(x - t))$. At $T = 1.0$ (one full period), $u(x,1) = \sin(2\pi x)$ — identical to the initial condition.

**Tolerance criterion:** Semi-log error-vs-p curve slope $< -0.5$ per unit $p$ (i.e., decay rate $\sigma > 0.5$).

## 2. Method & Implementation

**Solver:** TensorDG 1D modal DG with Legendre basis, upwind numerical flux, classical RK4 time integration.

**Mesh:** Fixed at $N = 16$ cells (uniform, $h = 1/16$) to isolate p-refinement from h-refinement.

**Polynomial degrees:** $p = 1, 2, 3, 4, 5, 6$

**Temporal convergence:** For each $p$, the RK4 CFL number is halved iteratively until the L2 error changes by less than 1% between successive refinements. This ensures the measured error is dominated by spatial discretization, not temporal error.

**Error metric:** Quadrature-based L2 norm against the exact translated solution.

**Implementation files:**
- `benchmarks/p_refinement_exponential.py` — benchmark driver
- `tests/test_p_refinement_exponential.py` — pytest verification gates
- `tensordg/convergence.py:p_decay_diagnostics()` — spectral convergence diagnostics

## 3. Parameter Table

| Parameter | Value |
|---|---|
| Equation | $u_t + u_x = 0$ |
| Domain | $[0, 1]$ periodic |
| Initial condition | $\sin(2\pi x)$ |
| Final time | $T = 1.0$ (one period) |
| Wave speed | $c = 1.0$ |
| Mesh | $N = 16$ uniform cells, $h = 0.0625$ |
| Polynomial degrees | $p = 1, 2, 3, 4, 5, 6$ |
| Basis | Legendre polynomials on reference element $[-1, 1]$ |
| Numerical flux | Upwind |
| Time integrator | Classical RK4 (temporally converged) |
| Quadrature | Gauss-Legendre, $n_q = \max(2p+3, 4)$ points |
| Arithmetic | Float64 (torch.float64) |
| Platform | LoongArch64 (loong64), Python 3.11.6, PyTorch 2.7.1 |

## 4. Results: Reference vs Computed

### L2 Error Table

| $p$ | DOFs | L2 Error | Reduction Factor | Decay Rate (pairwise) |
|---|---|---|---|---|
| 1 | 32 | $7.464 \times 10^{-3}$ | — | — |
| 2 | 48 | $2.089 \times 10^{-4}$ | 35.7× | 3.576 |
| 3 | 64 | $5.042 \times 10^{-6}$ | 41.4× | 3.724 |
| 4 | 80 | $9.732 \times 10^{-8}$ | 51.8× | 3.948 |
| 5 | 96 | $1.593 \times 10^{-9}$ | 61.1× | 4.113 |
| 6 | 112 | $2.228 \times 10^{-11}$ | 71.5× | 4.269 |

### Convergence Diagnostics

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Least-squares decay rate $\sigma$ | **3.927** | $> 0.5$ | ✅ PASS |
| Monotone decreasing | True | Required | ✅ PASS |
| Superlinear convergence | True | Expected | ✅ PASS |
| Error ratio $e(p=1)/e(p=6)$ | $3.35 \times 10^8$ | — | — |

### Babuska & Suri Reference Comparison

Babuska & Suri (1994) establish the exponential bound $\|e\| \leq C \exp(-\sigma p)$. For the sine function (entire), the theoretical $\sigma$ depends on the distance to the nearest singularity in the complex plane — which is infinite for sine, implying arbitrarily fast convergence in exact arithmetic. In practice, the observed $\sigma = 3.93$ is limited by:

1. **Quadrature precision:** Gauss-Legendre with $2p+3$ points is exact for polynomials up to degree $4p+5$, but the sine function is not polynomial.
2. **Floating-point arithmetic:** At $p=6$, the error reaches $2.2 \times 10^{-11}$, approaching the limits of float64 precision for this problem size.
3. **Temporal discretization:** Though temporally converged to 1%, the RK4 method introduces a small residual temporal error.

The measured decay rate $\sigma = 3.93$ is consistent with the exponential convergence theory: the error drops by nearly 9 orders of magnitude from $p=1$ to $p=6$, far exceeding the $\sigma > 0.5$ threshold.

## 5. Convergence Analysis & Sensitivity

### 5.1 Semi-log Convergence Plot

The semi-log plot of L2 error vs polynomial degree shows a clear exponential decay trend:

![Semi-log convergence](docs/benchmarks/results/p-refinement-exponential/p_refinement_semilog.png)

The exponential fit (red dashed line) has decay rate $\sigma = 3.927$, well above the threshold $\sigma = 0.5$ (green dotted line).

### 5.2 Log10 Error vs p

![Log10 convergence](docs/benchmarks/results/p-refinement-exponential/p_refinement_log10.png)

The linear fit in log10 space has slope $-1.705$, meaning each increment in $p$ reduces the error by approximately $10^{1.7} \approx 50\times$.

### 5.3 Sensitivity Observations

1. **Increasing reduction factors:** The error reduction factor increases with $p$ (35.7× → 71.5×), confirming the superlinear nature of exponential convergence. This is a hallmark of spectral methods: higher degrees capture more of the analytic solution's structure.

2. **Pairwise decay rates:** The pairwise decay rates increase monotonically (3.58 → 4.27), indicating the solution is entering the asymptotic exponential regime. For lower $p$, the error is still influenced by pre-asymptotic effects; for higher $p$, the exponential bound dominates.

3. **DOF efficiency:** At $p=6$ with only 112 DOFs, the error is $2.2 \times 10^{-11}$ — comparable to what would require thousands of cells with $p=1$. This demonstrates the efficiency of p-refinement for smooth problems.

## 6. Failure Modes & Artifacts

### Potential Failure Modes

| Failure Mode | Likelihood | Mitigation |
|---|---|---|
| Temporal error masking spatial convergence | Low | CFL refinement loop ensures <1% temporal error |
| Quadrature error for high $p$ | Low | $n_q = 2p+3$ is sufficient for Legendre-DG |
| Floating-point underflow at very high $p$ | Medium | Limited to $p \leq 6$; error at $p=6$ is $2.2 \times 10^{-11}$, well above machine epsilon |
| Non-smooth initial condition | N/A | Sine is entire; no discontinuities |

### Artifacts

| File | Description |
|---|---|
| `docs/benchmarks/results/p-refinement-exponential/p_refinement_results.json` | Full JSON report with all numerical data |
| `docs/benchmarks/results/p-refinement-exponential/p_refinement_table.csv` | CSV table for spreadsheet import |
| `docs/benchmarks/results/p-refinement-exponential/p_refinement_semilog.png` | Semi-log convergence plot |
| `docs/benchmarks/results/p-refinement-exponential/p_refinement_log10.png` | Log10 error vs p plot |
| `benchmarks/p_refinement_exponential.py` | Benchmark driver script |
| `tests/test_p_refinement_exponential.py` | 12 pytest verification gates |

### Reproduction Command

```bash
# Full benchmark (p=1..6, ~85 seconds)
python -m benchmarks.p_refinement_exponential

# Quick smoke test (p=1..3, ~19 seconds)
python -m benchmarks.p_refinement_exponential --quick

# Verification tests
pytest -q tests/test_p_refinement_exponential.py
```

### Verdict: **PASS**

All predeclared criteria met:
- ✅ Decay rate $\sigma = 3.927 > 0.5$
- ✅ Monotonically decreasing errors
- ✅ Superlinear convergence confirmed
- ✅ All 12 pytest gates pass

## 7. 独立复核（m16499 verifier 席，2026-11-12）

本算例的已发布数值已由 `docs/AUDIT-m16499-CASE-DG-PREFINEMENT-EXPONENTIAL.md` 独立复核：

- **可复现**：用当前代码整跑 `python -m benchmarks.p_refinement_exponential`（79.5 s）得到与上文
  逐位一致的六个误差值，σ = 3.9271 不变。工件现已带 `provenance.json`（sha256 + git commit + 环境）。
- **测量独立**：`benchmarks/p_refinement_independent_check.py` 用自建 NumPy Legendre 基与四套 Gauss
  阶数（2p+4/16/32/64）重测 L2，相对离散度 p≤3 ≤4.1e-12、p=6 为 1.3e-6（p=6 已接近 float64 差分地板）；
  另以 L1/L∞ 与 SSP-RK3 交叉，结论不随测量路径改变。
- **门禁已加固**：旧 `tests/test_p_refinement_exponential.py` 的 12 道闸门**只读工件**（0.14 s、零 solver
  调用），属自证；新增 `tests/test_p_refinement_exponential_live.py` 的 10 道活闸门在 N=16/cfl=0.08 下
  重算 p=1..3 并独立求积核对，与工件比对相对差 ≤3.6e-9。
- **判据收紧建议（重要）**：预登记判据 `σ > 0.5` **不具鉴别力**——证伪对照显示代数误差律 p⁻¹ 给出
  σ=0.549、p⁻² 给出 1.099，都能通过；而真指数 σ=0.5 恰好被卡在边界。本案指数收敛的**鉴别证据**是
  σ=3.93 且相邻衰减率单调递增（3.576→4.269，代数律必递减）。后续算例建议预登记复合判据：
  `σ > 1` 且相邻衰减率不减且 `e(p_min)/e(p_max) > 10³`。
- **适用范围**：一维周期线性对流光滑解；p≥7 的验证受 roundoff 限制（`MAX_DEGREE=8` 但测量先失效），
  二维/变系数未覆盖。

## 7. 判据收紧与测量地板量化（m16918，关闭审计 m16499 F4/F5）

本节推翻了 §1 中“Tolerance criterion: slope < -0.5（σ>0.5）”作为**充分**判据的用法。

**7.1 复合预登记判据（现驱动 verdict）**

```
sigma > 1  AND  逐段衰减率不减  AND  e(p_min)/e(p_max) > 1e3  AND  误差单调下降
```

`benchmarks/p_refinement_exponential.py:check_discriminating_criterion()` 实现，`checks.overall_pass`
= 遗留 σ>0.5 子项 ∧ 复合判据；每次运行内嵌 `falsification_control`（10 条已知衰减律的裁决表），
门 `gate_pass` 必须为真。实测本算例：σ=3.9271，逐段衰减率 3.576/3.724/3.948/4.113/4.269（递增），
总降幅 3.35e8，四条件全过。

**7.2 为什么 σ 门槛必须被替换（真实求解器反例）**

同一求解器、同一网格，仅换初值：

| 初值 | 光滑性 | σ | 旧判据 | 新判据 |
|---|---|---|---|---|
| sin(2πx) | 整函数 | 3.9271 | PASS | PASS |
| 三角波（C⁰ 折点） | 代数收敛 | 0.544 | **PASS（假阳性）** | 拒 |
| 方波 ±1 | 间断 | −0.008 | fail | 拒 |

且代数律 p⁻³/⁻⁴/⁻⁵ 的 σ 达 1.028/1.371/1.714（>1），p⁻⁴/⁻⁵ 总降幅 1296×/7776× 也过 1e3 门——
**唯一决定性条件是“逐段衰减率不减”**，故三条同时保留。

**7.3 测量地板（F5 量化）**

`benchmarks/p_refinement_floor.py`：三路独立测量（生产 float64 / 另一求积阶 / `np.longdouble`）。
地板 = `eps64·||u_h||` = **1.570e-16**；p=5/p=6 实测误差 1.59e-09 / 2.23e-11，信噪比 1.0e7 / 1.4e5，
路径相对离散 ≤8.4e-07。按 σ=3.927 外推：p=7 信噪比 2796、p=8 约 55（仍可测），**p=9 起（信噪比 1.1）进入地板**。
故 `MAX_DEGREE=8` 尚可验证，p≥9 不可——这是量化结论，p=7/8 的**实测**仍未做（p=7 单点 >120 s）。

详见 `docs/CLOSEOUT-m16918-prefinement-criterion.md`（表格由 `tools/render_precision_closeout.py` 从工件程序化生成）。
