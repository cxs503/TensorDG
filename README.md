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
- Two-dimensional Euler isentropic-vortex verification.
- Regression tests for conservation, convergence, physical admissibility,
  exact-reference comparisons, and the documented 3% error threshold.

Robust TVB/WENO shock limiting, sparse/matrix-free elliptic solvers,
unstructured meshes, high-order multidimensional Euler, and additional
Euler/Navier--Stokes benchmarks remain future work.

## Development principles

Numerical claims should be backed by executed tests and recorded output. A
profile that merely looks plausible is not sufficient: smooth cases should
measure convergence, while discontinuous cases should compare with a trusted
reference and state the error norm.
