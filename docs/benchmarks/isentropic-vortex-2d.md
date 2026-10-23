# Benchmark: 2D Euler isentropic vortex

A smooth strength-5 isentropic vortex with unit far-field density and pressure
is advected by background velocity `(1,1)` on `[-5,5]^2`. At t=1 its center is
near `(1,1)`, far from the periodic boundaries, where the vortex exponential
is negligible.

The current solver is a P0 DG / first-order finite-volume baseline with
x/y Rusanov fluxes and SSP-RK3. Initial states and reference data use
four-point-per-direction cell quadrature. Density, both velocity components,
and pressure are all compared with the analytical moving vortex.

The documented 192x192, CFL=0.4 case requires every relative L2 error to be
below 3%. Tests also check initial-projection accuracy, conservation, and
physical admissibility.

```bash
python -m benchmarks.isentropic_vortex_2d --cells 192 \
  --output-dir docs/benchmarks/results/isentropic-vortex-2d
```

Outputs include numerical/exact/error density contours, a curve through the
translated vortex center, CSV data, and JSON metrics. This is a smooth
verification case, not a high-order two-dimensional shock-capturing claim.
