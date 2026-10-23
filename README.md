# TensorDG

**TensorDG** is the discontinuous Galerkin (DG) solver project in the Tensor scientific-computing ecosystem. Its initial implementation uses PyTorch and starts with a verified, reproducible numerical benchmark rather than a broad but unvalidated solver feature set.

## First benchmark

The first target is **1D periodic linear advection** with a smooth sine wave, modal Legendre basis, upwind flux, and RK4 time integration. It has a closed-form solution and tests conservation plus mesh-convergence order.

```bash
python -m pip install torch numpy pytest
python benchmarks/linear_advection_1d.py --degree 1 --cells 20 40 80 160
pytest -q
```

See [benchmark specification and method](docs/benchmarks/linear-advection-1d.md).

## Initial scope

- PyTorch tensor-based DG operator and time stepping.
- Polynomial degrees 0–3 on a uniform 1D periodic mesh.
- L2 error and observed convergence order reporting.
- Regression tests for conservation, finite values, and convergence.

## Development principles

Numerical claims should be backed by executed tests and recorded output. The current benchmark implementation is an initial baseline; it is not yet a production-ready multidimensional DG framework.

## Next benchmark: Euler Sod shock tube

A first-order P0 Euler baseline with Rusanov flux and SSP-RK3 is available as `benchmarks/euler_sod_1d.py`. It checks compressible-flow fluxes and physical admissibility before higher-order limiting is introduced. See [Sod benchmark notes](docs/benchmarks/euler-sod-1d.md).
