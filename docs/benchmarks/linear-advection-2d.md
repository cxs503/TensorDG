# Benchmark: 2D periodic linear advection (tensor-product DG)

## Problem

This benchmark solves `u_t + ax*u_x + ay*u_y = 0` on the periodic unit square.
The initial condition is `1 + sin(2*pi*x)*sin(2*pi*y)`, so the exact solution
is the same profile translated by `(ax*t, ay*t)`. The nonzero mean makes the
mass-conservation check nontrivial.

## Method

- Uniform Cartesian mesh with tensor-product modal Legendre bases P0--P3.
- Dimension-by-dimension weak DG volume terms and upwind face fluxes.
- Classical RK4 with `dt = CFL / ((2*p+1)*(|ax|/hx+|ay|/hy))`.
- Gauss--Legendre initial projection and L2-error quadrature.
- PyTorch tensors for the state and semi-discrete operator.

The documented case uses P1 on a 24x24 mesh, velocity `(1,0.5)`, CFL 0.08,
and final time 1. Its relative L2 error must be below 3%; the regression
suite also checks second-order convergence and negative velocity components.

## Run

```bash
python -m pip install -e ".[dev]"
python -m benchmarks.linear_advection_2d --cells 24 --degree 1 \
  --output-dir docs/benchmarks/results/linear-advection-2d
```

The command writes exact/numerical/error contours, a non-degenerate y=0.25
centerline comparison, CSV data, and a JSON metrics record. See the
[Chinese classic-benchmark tutorial](../tutorials/classic-dg-benchmarks.md)
for equations, measured results, and images.
