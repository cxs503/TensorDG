# Benchmark: Euler Lax shock tube with high-order DG

The standard Lax initial states are `(rho,u,p)=(0.445,0.698,3.528)` on the
left and `(0.5,0,0.571)` on the right, with the diaphragm at x=0.5 and final
time 0.13. The general exact Riemann solver gives
`p*=2.466097919207` and `u*=1.528723026633`.

The documented numerical method is P3 modal DG with HLLC interface fluxes,
SSP-RK3, and a local positivity fallback. Cells whose reconstructed traces or
quadrature states have negative density/pressure revert to their cell average.
This is not a TVD or oscillation-free guarantee; the plotted high-frequency
components are intentionally not hidden by post-processing.

Primitive-variable errors are evaluated with 8-point quadrature against the
exact Riemann solution. The 400-cell, CFL=0.35 case requires every field's
relative L2 error to be below 3%.

```bash
python -m benchmarks.lax_1d --cells 400 --degree 3 --cfl 0.35 \
  --output-dir docs/benchmarks/results/lax-1d
```

A P0 Euler/Rusanov or HLLC solver is also available in `tensordg.euler_1d`;
Sod and Lax share the general exact Riemann implementation.
