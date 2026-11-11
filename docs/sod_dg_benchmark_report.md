# Sod Shock Tube via DG (TensorDG) — Benchmark Report

## ① Problem & Reference

**Case**: Sod shock tube (Sod, 1978) — the canonical 1D Riemann problem for the Euler equations.

**Reference**: Sod, G. A. (1978). "A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws." *Journal of Computational Physics*, 27(1), 1–31. DOI: 10.1016/0021-9991(78)90023-2.

**Initial conditions** (γ = 1.4):
| Region | ρ | u | p |
|--------|------|------|------|
| Left (x < 0.5) | 1.0 | 0.0 | 1.0 |
| Right (x ≥ 0.5) | 0.125 | 0.0 | 0.1 |

**Domain**: [0, 1], transmissive boundaries, evaluated at t = 0.2.

**Exact solution**: Computed via the exact Riemann solver (Toro, 2009, §4.3) with guarded Newton iteration for star pressure:
- Star pressure p* = 0.303130178051
- Star velocity u* = 0.927452620049

The solution contains a left-going rarefaction, a contact discontinuity, and a right-going shock.

## ② Method & Implementation

**Solver**: TensorDG modal discontinuous Galerkin (DG) with:
- Spatial discretization: Legendre modal basis, degree P0–P3
- Numerical flux: HLLC (Toro, 2009) with Einfeldt wave speeds
- Time integration: SSP-RK3 (Shu & Osher, 1988)
- Positivity: Local positivity filter (Zhang & Shu, 2010)
- Limiter: None (positivity-only) for high-order; TVB available but degrades accuracy

**Implementation files**:
- `tensordg/euler_dg_1d.py` — high-order DG solver
- `tensordg/euler_1d.py` — P0 DG / finite-volume solver, HLLC flux
- `benchmarks/sod_exact.py` — exact solution wrapper
- `benchmarks/riemann_exact.py` — exact Riemann solver (Newton iteration)
- `benchmarks/sod_dg_benchmark.py` — benchmark harness (new)

**Reference solution**: Cell-averaged exact solution via 8-point Gauss-Legendre quadrature per cell (`benchmarks/sod_exact.cell_averages`).

## ③ Parameters

| Parameter | P0 (FV) | P1 DG | P2 DG | P3 DG |
|-----------|---------|-------|-------|-------|
| Cells (N) | 100–400 | 50 | 30–50 | 20–40 |
| DOF | N | 2N | 3N | 4N |
| CFL | 0.4 | 0.12 | 0.12 | 0.12 |
| Limiter | N/A (Rusanov) | None | None | None |
| Limiter stage | N/A | final | final | final |
| Flux | Rusanov | HLLC | HLLC | HLLC |
| Time steps | 103–431 | 540 | 523–933 | 517–1042 |

**Dimensionless numbers**:
- Mach number (post-shock): ~2.0
- Pressure ratio: p_L/p_R = 10
- Density ratio: ρ_L/ρ_R = 8

## ④ Results: Reference vs Computed

### Best Configuration: DG P3, N=40, no limiter

| Quantity | Reference (cell-avg) | DG P3 (N=40) | Rel. L2 Error | <3%? |
|----------|---------------------|--------------|---------------|------|
| Density ρ | exact cell averages | 0.5961% | **0.5961%** | ✓ |
| Velocity u | exact cell averages | 1.6761% | **1.6761%** | ✓ |
| Pressure p | exact cell averages | 0.3968% | **0.3968%** | ✓ |
| **Max** | — | — | **1.6761%** | **✓ PASS** |

### Convergence Summary

| Method | N | DOF | ρ error | u error | p error | Max | Pass? |
|--------|------|------|---------|---------|---------|------|-------|
| P0 (FV) | 100 | 100 | 5.51% | 16.11% | 6.35% | 16.11% | ✗ |
| P0 (FV) | 200 | 200 | 4.26% | 11.61% | 4.43% | 11.61% | ✗ |
| P0 (FV) | 400 | 400 | 3.14% | 7.84% | 2.96% | 7.84% | ✗ |
| P1 DG | 50 | 100 | 2.86% | 6.29% | 3.08% | 6.29% | ✗ |
| P2 DG | 30 | 90 | 0.59% | 5.27% | 0.78% | 5.27% | ✗ |
| P2 DG | 50 | 150 | 0.51% | 4.11% | 0.63% | 4.11% | ✗ |
| P3 DG | 20 | 80 | 0.65% | 3.91% | 0.96% | 3.91% | ✗ |
| **P3 DG** | **40** | **160** | **0.60%** | **1.68%** | **0.40%** | **1.68%** | **✓** |

**Key finding**: Only DG P3 at N=40 (160 DOF) achieves <3% error in all three fields. The velocity field is the most challenging due to the contact discontinuity.

## ⑤ Convergence & Sensitivity

### Convergence Rate
- P0 (first-order): Slow convergence; velocity error remains >7% even at N=400
- P1–P3: Density and pressure converge rapidly (<1% at modest DOF); velocity converges more slowly due to contact discontinuity smearing
- P3 at N=40 achieves the target with only 160 DOF, compared to P0 which would need N>3200 (estimated)

### Limiter Sensitivity
- **No limiter** (positivity only): Best accuracy; P3 N=40 achieves 1.68% max error
- **TVB limiter** (M=1, stage=final): Degrades accuracy significantly; P2 N=30 velocity error increases from 5.27% to 9.01%
- **TVB limiter** (stage=every): Worst accuracy; destroys high-order content in smooth regions

### CFL Sensitivity
- CFL=0.12: Stable for all degrees; P3 N=40 uses 1042 steps
- Higher CFL may reduce steps but risks positivity violations near shocks

## ⑥ Failure Modes & Artifacts

### Known Failure Modes
1. **P0 at high resolution**: N=3200 takes >120s locally; requires cluster for full convergence study
2. **TVB limiter**: Aggressively reduces high-order modes, destroying accuracy in smooth regions
3. **Contact discontinuity**: Velocity field is hardest to capture; requires higher polynomial degree or more cells
4. **Positivity violations**: Can occur with high CFL or aggressive limiting; mitigated by positivity filter

### Artifacts
| File | Description |
|------|-------------|
| `artifacts/sod_dg/metrics_quick.json` | Error metrics for P3 N=40 |
| `artifacts/sod_dg/profiles.csv` | Numerical + reference data (x, ρ, u, p) |
| `artifacts/sod_dg/sod_profiles.png` | Comparison plot: DG P3 vs exact |
| `artifacts/sod_dg/convergence.png` | Convergence study: error vs DOF |
| `benchmarks/sod_dg_benchmark.py` | Full benchmark harness (submitted to cluster cj1020) |

### Reproduction Command
```bash
# Quick verification (P3 N=40, ~13s)
python -c "
from tensordg.euler_dg_1d import solve_euler_dg_1d
from benchmarks.sod_exact import cell_averages
import numpy as np
x, coeff, meta = solve_euler_dg_1d(
    (1.0, 0.0, 1.0), (0.125, 0.0, 0.1),
    ncells=40, degree=3, final_time=0.2, cfl=0.12, limiter=None
)
avg = coeff[:, 0].detach().cpu().numpy()
rho, vel, pres = avg[:, 0], avg[:, 1]/avg[:, 0], 0.4*(avg[:, 2] - 0.5*avg[:, 1]**2/avg[:, 0])
_, ref = cell_averages(40, 0.2)
for k, n in [('density', rho), ('velocity', vel), ('pressure', pres)]:
    err = np.sqrt(np.mean((n - ref[k])**2)) / np.sqrt(np.mean(ref[k]**2))
    print(f'{k}: {100*err:.4f}% {\"PASS\" if err < 0.03 else \"FAIL\"}')
"
```

### Ecological Friction
- **Cluster queue depth**: 234 jobs ahead; local execution limited to ~60s timeout
- **P0 at N=3200**: Existing test `test_documented_sod_case_matches_exact_solution_within_three_percent` times out locally (>120s); requires cluster
- **No GPU acceleration**: All runs on CPU (loong64); PyTorch CUDA not available
