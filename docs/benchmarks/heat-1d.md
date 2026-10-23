# Benchmark: 1D heat equation with SIPG and Crank--Nicolson

Solves `u_t=u_xx` with homogeneous Dirichlet boundaries and
`u(x,0)=sin(pi*x)`. The exact solution is
`sin(pi*x)*exp(-pi^2*t)`.

The method-of-lines system uses the same symmetric interior-penalty ideas as
the Poisson benchmark: volume stiffness, interior/boundary consistency and
penalty terms, and a global mass matrix. Crank--Nicolson avoids the explicit
diffusion time-step restriction while providing second-order temporal
accuracy. The default dt is approximately `0.02*h`.

The documented P2, 16-cell, t=0.1 case uses 80 steps and has relative L2 error
below 3%. P1 convergence and matrix symmetry are also tested.

```bash
python -m benchmarks.heat_1d --cells 16 --degree 2 \
  --output-dir docs/benchmarks/results/heat-1d
```

Outputs include a fixed-dt numerical/exact/error space-time contour and a
final-time profile. See the Chinese nonlinear-benchmark tutorial for results.
