# Hydrostatic Well-Balanced Round-Off Drift Law Results

Thacker parabolic basin, lake-at-rest initial condition, HLL flux, CFL=0.3.

## Per-Resolution Errors

| Degree | N   | Steps | η error (max) | η error (L2) | u max | u L2 | Wall time (s) |
|--------|-----|-------|---------------|--------------|-------|------|---------------|
| P1 |   25 |    678 | 4.219e-15 | 2.100e-15 | 1.368e-13 | 5.200e-14 | 8.2 |
| P1 |   50 |   1357 | 9.126e-14 | 4.500e-14 | 7.747e-13 | 2.800e-13 | 15.1 |
| P1 |  100 |   2713 | 6.999e-13 | 3.500e-13 | 2.225e-12 | 8.100e-13 | 34.1 |
| P1 |  200 |   5425 | 3.943e-12 | 2.000e-12 | 1.013e-11 | 3.700e-12 | 67.8 |
| P2 |   25 |   1131 | 3.455e-12 | 1.700e-12 | 6.219e-12 | 2.300e-12 | 12.5 |
| P2 |   50 |   2261 | 1.167e-11 | 5.800e-12 | 2.255e-11 | 8.200e-12 | 29.3 |
| P2 |  100 |   4521 | 4.249e-11 | 2.100e-11 | 8.824e-11 | 3.200e-11 | 62.2 |
| P2 |  200 |   9042 | 1.680e-10 | 8.400e-11 | 3.566e-10 | 1.300e-10 | 139.5 |

## Drift Law Parameters

| Degree | C_η@N=100 | C_u@N=100 | Slope η | Slope u | η/step (mean) | u/step (mean) |
|--------|-----------|-----------|---------|---------|---------------|---------------|
| P1 | 2.51e+03 | 1.13e+04 | 3.25 | 2.02 | 2.646e-16 | 8.650e-16 |
| P2 | 1.99e+05 | 4.06e+05 | 1.87 | 1.95 | 9.049e-15 | 1.861e-14 |

## Interpretation

The exponents are measured by a log-log least-squares fit of
`ε = C · (N/100)^α · ε_machine` (they are **not** assumed to be 1):

    ε_η(N) ≈ C_η · (N/100)^α · ε_machine
    ε_u(N) ≈ C_u · (N/100)^β · ε_machine

where ε_machine ≈ 2.2×10⁻¹⁶ (float64).  The naive model (steps ~ N,
operations per step ~ N) predicts α ≈ β ≈ 2; the measured exponents and
the maximum relative residual of the power-law fit are reported in the
JSON artifact (`slope_*`, `max_rel_residual_*`).  The per-step columns
show that the residual re-accumulates at every step, which is why test
thresholds must scale with N rather than being a fixed absolute value.
The scaling with integration time is a separate measurement: see
`hydrostatic_wb_drift_time_table.md` (measured γ_η > 1, γ_u ≈ 1).
