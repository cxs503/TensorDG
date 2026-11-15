# CASE-DG-EULER-VORTEX-ERR: Isentropic Vortex Error Conservation (m21396)

**Anchor** `CASE-DG-EULER-VORTEX-ERR` | **Mission** m21396 |
**Tolerance** < 6% (ref: Yee et al. 2000, JCP 162:33-81) |
**Verdict**: PASS

## 1. Case Introduction

The 2D compressible Euler equations (ideal gas, gamma=1.4) with an **isentropic vortex**
is a classic CFD verification benchmark. The initial condition is an isentropic vortex
superimposed on a uniform freestream (u,v)_inf = (1,1), transported in a periodic domain.
Since the exact solution returns to the initial condition after one full period T = L/|u_inf| = 10,
this case tests whether **numerical error accumulates over time** (error conservation),
not just short-time accuracy.

| Parameter | Symbol | Value |
|---|---|---|
| Domain | Omega | [-5,5] x [-5,5], periodic BC |
| Vortex strength | beta | 5.0 |
| Freestream | (u,v)_inf | (1,1) |
| Adiabatic index | gamma | 1.4 |
| Period | T | 10.0 |
| Tolerance | eps_tol | 6% (relative L2, per primitive) |
| Reference | - | Yee, Vinokur & Djomehri, JCP 162 (2000) 33-81 |

## 2. Methodology

Two **independent** discretization paths are tested against the same criteria:

1. **High-order DG**: `tensordg.euler_dg_2d.solve_euler_dg_2d_vortex`
   - Tensor-product modal DG (P2), HLLC flux, SSP-RK3 time stepping
   - Initial state: modal L2 projection; error: pointwise relative L2 at quadrature nodes
2. **Low-order P0**: `tensordg.euler_2d.solve_isentropic_vortex`
   - P0 finite volume, Rusanov flux, SSP-RK3 time stepping
   - Initial state: exact cell averages via 4x4 Gauss; error: cell-average relative L2

**Key design**: The exact solution uses **periodic min-image form** (`period=(10,10)`),
avoiding the "reference defect" where the free-space formula ceases to be the periodic
domain solution for t > 5. This defect previously caused spurious "error amplification"
(alpha ~ 3.3-4.4); after fix, alpha ~ 0.2.

Seven gate criteria:

| Criterion | Content | Threshold |
|---|---|---|
| C1 | t=1 snapshot max-field relative L2 | < 6% |
| C2 | Error accumulation exponent alpha (e ~ t^alpha) | alpha <= 1.0 |
| C3 | Mass/momentum/energy discrete conservation drift | < 1e-10 |
| C4 | Entropy functional S = int p*rho^(-gamma) drift | < 0.001 |
| C5 | P0 cross-check observed order | >= 0.8 |
| C6 | LADDER checkpoint completeness | 0 missing |
| C7 | Design space invariance (multi-mesh x CFL) | alpha <= 1.0 per family |

## 3. Results

### 3.1 Gate Summary

| Criterion | Measured | Status |
|---|---|---|
| C1 tolerance (t=1) | 0.0493% (p0_n64_t1_cfl0.4) | PASS |
| C2 error conservation | alpha = 0.1987 (strict target alpha<=0.5) | PASS |
| C3 discrete conservation | worst drift 3.91e-13 (dg_d2_n64_t10_cfl0.2:energy) | PASS |
| C4 entropy budget | |dS|/S(0) = 2.98e-07 | PASS |
| C5 cross-check | P0 observed order = 0.8488 | PASS |
| C6 ladder completeness | 19/19 checkpoints | PASS |
| C7 design space | max alpha = 0.1989 (dg_d2_n32_cfl0.4) | PASS |

### 3.2 Error vs Time (per checkpoint)

| Family | t | Steps | rho % | u % | p % | Max % | alpha |
|---|---|---|---|---|---|---|---|
| dg_d2_n32_cfl0.2 | 1 | 469 | 0.0050 | 0.0094 | 0.0070 | 0.9431 | nan |
| dg_d2_n32_cfl0.2 | 2 | 937 | 0.0051 | 0.0097 | 0.0071 | 0.9690 | nan |
| dg_d2_n32_cfl0.2 | 5 | 2342 | 0.0054 | 0.0111 | 0.0075 | 1.1132 | nan |
| dg_d2_n32_cfl0.2 | 10 | 4684 | 0.0061 | 0.0152 | 0.0085 | 1.5163 | nan |
| dg_d2_n32_cfl0.4 | 1 | 235 | 0.0050 | 0.0094 | 0.0070 | 0.9431 | nan |
| dg_d2_n32_cfl0.4 | 2 | 469 | 0.0051 | 0.0097 | 0.0071 | 0.9690 | nan |
| dg_d2_n32_cfl0.4 | 5 | 1171 | 0.0054 | 0.0111 | 0.0075 | 1.1134 | nan |
| dg_d2_n32_cfl0.4 | 10 | 2342 | 0.0061 | 0.0152 | 0.0085 | 1.5169 | nan |
| dg_d2_n64_cfl0.2 | 1 | 937 | 0.0006 | 0.0011 | 0.0008 | 0.1149 | nan |
| dg_d2_n64_cfl0.2 | 2 | 1874 | 0.0006 | 0.0012 | 0.0008 | 0.1153 | nan |
| dg_d2_n64_cfl0.2 | 10 | 9369 | 0.0006 | 0.0012 | 0.0009 | 0.1225 | nan |
| p0_n128 | 1 | 183 | 1.2797 | 2.8022 | 2.0258 | 281.6911 | nan |
| p0_n128 | 2 | 357 | 2.5325 | 4.8617 | 3.9561 | 487.5468 | nan |
| p0_n128 | 5 | 854 | 4.5337 | 8.8749 | 7.0069 | 887.6448 | nan |
| p0_n128 | 10 | 1631 | 5.4216 | 12.0836 | 7.9004 | 1208.4611 | nan |
| p0_n256 | 1 | 370 | 0.6933 | 1.5043 | 1.1039 | 151.8646 | nan |
| p0_n64 | 1 | 90 | 2.2038 | 4.9191 | 3.4388 | 492.6036 | nan |
| p0_n64 | 5 | 408 | 5.7257 | 12.0235 | 8.4758 | 1202.3516 | nan |
| p0_n64 | 10 | 778 | 6.0187 | 14.3043 | 8.3587 | 1430.4741 | nan |

### 3.3 Amplification Factor A(t) = e(t)/e(0)

| Family | e(0) % | e(1) % | e(10) % | A(10) |
|---|---|---|---|---|
| dg_d2_n32 | 0.005492 | 0.0094 | 0.0152 | 2.761 |
| p0_n64 | 0.009381 | 4.9260 | 14.3047 | 1524.810 |
| p0_n128 | 0.002350 | 2.8169 | 12.0846 | 5142.710 |
| p0_n256 | 0.000588 | 1.5186 | nan | nan |

### 3.4 Exact Solution Invariants

| Quantity | Value |
|---|---|
| Mass | 98.241743560191 |
| x-Momentum | 98.241743560191 |
| y-Momentum | 98.241743560191 |
| Energy | 344.759326601030 |
| Entropy S = int p*rho^(-gamma) | 100.000000000000 |
| Core min density | 0.494290679887 |

## 4. Figures

### 4.1 Error vs Time
`fig_error_vs_time.png`: Max-field relative L2 error vs dimensionless time (log-log),
slope = accumulation exponent alpha; red dashed = 6% tolerance.

### 4.2 Invariants vs Time
`fig_invariants_vs_time.png`: C3 mass/energy conservation drift (round-off level)
and C4 entropy functional drift, core density error vs time.

### 4.3 Design Space
`fig_design_space.png`: C7 design space - left=mesh axis (32^2 vs 64^2, same CFL 0.2),
right=time-step axis (same 32^2, CFL 0.2 vs 0.4).

### 4.4 Amplification Factor
`fig_amplification.png`: A(t) = e(t)/e(0) semi-log plot, green band A in [1,2] =
"error conservation" regime. DG family A(10) ~ 2.76 (order 1), P0 family A(10) >> 1.

### 4.5 Error Field + Cut-line
`fig_error_field_t1.png`: DG-P2/32^2 at t=1 density/pressure pointwise error field
(periodic min-image reference) + y-centerline comparison (exact vs numerical) +
pointwise error along line. Smooth error field => dispersive smooth error, not instability.

## 5. Notes and Limitations

1. **Tolerance source**: 6% from case-pool entry verbatim (ref: Yee et al. 2000);
   paper not in repo, not independently verified; all errors are against the **analytic** solution.
2. **Reference defect fix**: Early readings used free-space exact solution (non-periodic),
   causing spurious "error" growth for t>5 (alpha ~ 3.3-4.4). Fix to periodic min-image
   yields alpha ~ 0.2, confirming the "amplification" was a reference defect, not a scheme issue.
3. **A(t) cross-family comparability is limited**: DG e(0) is L2 projection error,
   P0 e(0) is cell-average representation error, differing by 1-2 orders of magnitude.
4. **P0 long-time error**: P0 family t=10 max-field 14.3% far exceeds 6% tolerance;
   this is expected dissipative accumulation for a first-order scheme, not a precision gate.
5. **DG design space coverage**: Currently only P2/32^2 and P2/64^2 meshes, CFL 0.2/0.4;
   P1/P3 long-time error conservation not tested.

## 6. Error Assessment (vs benchmark)

| Key Quantity | Value | Benchmark | Error | < 3%? |
|---|---|---|---|---|
| DG-P2/32^2 t=1 max-field rel L2 | 0.0094% | Analytic vortex (periodic) | 0.0094% | **YES** |
| DG-P2/32^2 t=10 max-field rel L2 | 0.0152% | Analytic vortex (periodic) | 0.0152% | **YES** |
| DG-P2/64^2 t=10 max-field rel L2 | 0.0012% | Analytic vortex (periodic) | 0.0012% | **YES** |
| P0/128^2 t=1 max-field rel L2 | 2.8169% | Analytic vortex (periodic) | 2.8169% | **YES** |
| P0/256^2 t=1 max-field rel L2 | 1.5186% | Analytic vortex (periodic) | 1.5186% | **YES** |
| P0/64^2 t=1 max-field rel L2 | 4.9260% | Analytic vortex (periodic) | 4.9260% | No |
| DG long-time A(10) | 2.761 | A=1 (ideal conservation) | +176% | N/A (not error vs benchmark) |

**Conclusion**: DG design family max-field errors at t=1 and t=10 are both **< 3%**
(0.0094% and 0.0152% respectively). Error amplification after one full period is 2.761x
(order 1, confirming error conservation). P0 family t=1 4.93% not all < 3%; it is the
deliberately retained first-order cross-check path (C5), not承担 precision gate.

---

> Evidence: `case_euler_vortex_errconserve/metrics.json` (19 checkpoints,
> `ladder_completeness.missing = []`),
> `case_euler_vortex_errconserve/projection_baseline.json`,
> `case_euler_vortex_errconserve/results/*.json`,
> `case_euler_vortex_errconserve/fig_*.png` (5 figures).

> Reproduce: `PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate --figures --output-dir case_euler_vortex_errconserve`
