# Benchmark: smooth periodic inviscid Burgers equation

Solves `u_t+(u^2/2)_x=0` with `u0=0.2+0.8*sin(2*pi*x)` at the pre-shock
time `t=0.1`. The exact solution satisfies `u=u0(x-u*t)` and is computed by
Newton iteration on the characteristic relation.

The solver uses modal Legendre DG (P0--P3), quadrature of the nonlinear volume
flux, Rusanov interface fluxes, and SSP-RK3. No limiter is needed for this
smooth case. The documented P2 16/32/64-cell sequence checks near-third-order
convergence, nonzero-mean conservation, and relative L2 error below 3%.

```bash
python -m benchmarks.burgers_1d --cells 16 32 64 --degree 2 \
  --output-dir docs/benchmarks/results/burgers-1d
```

Outputs: profile/error and convergence curves, CSV data, and JSON metrics.
See `docs/tutorials/nonlinear-dg-benchmarks.md` for the Chinese tutorial.
