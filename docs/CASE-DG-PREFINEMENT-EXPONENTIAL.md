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
