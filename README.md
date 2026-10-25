# TensorDG

**TensorDG** is a PyTorch-based discontinuous Galerkin (DG) solver project in
the Tensor scientific-computing ecosystem. The repository grows through
verified, reproducible numerical benchmarks rather than an unvalidated broad
feature set.

## Verified benchmarks

| Benchmark | Numerical method | Reference | Documented relative L2 error |
|---|---|---|---:|
| 1D periodic linear advection | P2 Legendre DG, upwind, RK4 | Exact translated sine wave | 0.000236% |
| 2D periodic linear advection | P1 tensor-product DG, upwind, RK4 | Exact translated profile | 0.281233% |
| Euler Sod shock tube | P0 DG / finite volume, Rusanov, SSP-RK3 | Exact Riemann cell averages | max 2.517658% |
| 2D Poisson | P1 SIPG | Manufactured sine solution | 0.380026% |
| Smooth Burgers | P2 DG, Rusanov, SSP-RK3 | Exact characteristic solution | 0.001820% |
| 1D heat equation | P2 SIPG, Crank--Nicolson | Exact decaying sine mode | 0.003604% |
| Euler Lax shock tube | P3 DG, HLLC, positivity fallback | Exact Riemann solution | max 2.769420% |
| 2D isentropic vortex | P0 DG / finite volume, Rusanov | Analytical moving vortex | max 1.972990% |

The documented configurations all have relative errors below 3%. For Sod,
Lax, and the isentropic vortex, the maximum is over all reported primitive
fields. These are verification baselines, not a claim of production-ready
multidimensional shock capturing.

**Chinese tutorials with methods, commands, contours, curves, and measured errors:**

- [Classic DG benchmarks](docs/tutorials/classic-dg-benchmarks.md)
- [Nonlinear and time-dependent benchmarks](docs/tutorials/nonlinear-dg-benchmarks.md)

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest -q

python -m benchmarks.linear_advection_1d --degree 2 --cells 10 20 40 80
python -m benchmarks.linear_advection_2d --cells 24 --degree 1
python -m benchmarks.compare_euler_sod_1d --cells 3200 --time 0.2
python -m benchmarks.poisson_2d --cells 16 --degree 1
python -m benchmarks.burgers_1d --cells 16 32 64 --degree 2
python -m benchmarks.heat_1d --cells 16 --degree 2
python -m benchmarks.lax_1d --cells 400 --degree 3
python -m benchmarks.isentropic_vortex_2d --cells 192
```

Add `--output-dir PATH` to the benchmark commands to write JSON metrics, CSV
data, and PNG plots. See `docs/benchmarks/` for method specifications and
`docs/benchmarks/results/` for the curated outputs shown in the tutorial.

## Current scope

- Tensor-based DG operators on uniform one- and two-dimensional Cartesian meshes.
- Modal Legendre bases and Gauss--Legendre projection/error quadrature.
- High-order smooth advection and Burgers, plus SIPG Poisson/heat verification.
- General P0 Euler Riemann solvers and high-order HLLC DG with local positivity fallback.
- TVB (Cockburn--Shu 1989) and WENO (Qiu--Shu 2005) shock limiters for 1-D Euler.
- Two-dimensional Euler isentropic-vortex verification.
- Regression tests for conservation, convergence, physical admissibility,
  exact-reference comparisons, and the documented 3% error threshold.

Sparse/matrix-free elliptic solvers, unstructured meshes, high-order
multidimensional Euler, limiter performance optimisation, and additional
Euler/Navier--Stokes benchmarks remain future work.

## Shock limiters

TensorDG includes TVB and WENO limiters for 1-D compressible Euler problems
to suppress spurious oscillations near discontinuities.

**TVB limiter** (Total Variation Bounded, Cockburn & Shu 1989):
- Detects troubled cells via minmod-based indicator
- Replaces high-order polynomial with linear reconstruction in troubled cells
- Preserves cell averages exactly
- Configurable TVB constant M (default M=1.0)

**WENO limiter** (Weighted Essentially Non-Oscillatory, Qiu & Shu 2005):
- Same troubled cell detection as TVB
- Reconstructs using WENO-weighted combination of neighboring stencils
- Higher resolution near discontinuities than TVB
- Slightly higher computational cost

**Usage**:
```python
from tensordg.euler_dg_1d import solve_euler_dg_1d

# Baseline (positivity filter only)
x, coeff, meta = solve_euler_dg_1d(left, right, ncells=200, degree=2)

# With TVB limiter
x, coeff, meta = solve_euler_dg_1d(
    left, right, ncells=200, degree=2,
    limiter="tvb", tvb_constant=1.0
)

# With WENO limiter
x, coeff, meta = solve_euler_dg_1d(
    left, right, ncells=200, degree=2,
    limiter="weno", tvb_constant=1.0
)
```

**Benchmark**:
```bash
python -m benchmarks.sod_with_limiter --cells 200 --limiter tvb
```

**Performance characteristics**:
- Limiter overhead: ~80-120% (pure Python implementation)
- Recommended for problems with strong shocks where positivity filter alone is insufficient
- For smooth problems or weak discontinuities, baseline solver is more efficient

## Development principles

Numerical claims should be backed by executed tests and recorded output. A
profile that merely looks plausible is not sufficient: smooth cases should
measure convergence, while discontinuous cases should compare with a trusted
reference and state the error norm.
