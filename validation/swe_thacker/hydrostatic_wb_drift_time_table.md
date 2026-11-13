# Hydrostatic Well-Balanced Round-Off Drift: Time Scaling

Thacker parabolic basin (H0 = 2.0 m, L = 1.0 m, eta0 = 1.0 m), lake-at-rest initial condition, HLL flux, CFL = 0.3; resolution fixed, integration time scanned.

## Measured Drift vs Integration Time

| Degree | N | T (s) | Steps | η error (max) | η / ε_machine | u max (m/s) | u / ε_machine | Wall time (s) |
|--------|---|-------|-------|---------------|---------------|-------------|---------------|---------------|
| P1 |   50 | 0.50 |   679 | 2.665e-14 |           120 | 2.704e-13 |          1218 | 11.2 |
| P1 |   50 | 1.00 |  1357 | 9.126e-14 |           411 | 7.747e-13 |          3489 | 24.4 |
| P1 |   50 | 2.00 |  2713 | 1.172e-12 |          5279 | 1.570e-12 |          7072 | 39.3 |
| P1 |   50 | 4.00 |  5425 | 6.185e-12 |         27857 | 3.388e-12 |         15257 | 77.6 |
| P1 |  100 | 0.50 |  1357 | 1.135e-13 |           511 | 8.834e-13 |          3978 | 21.7 |
| P1 |  100 | 1.00 |  2713 | 6.999e-13 |          3152 | 2.225e-12 |         10020 | 39.8 |
| P1 |  100 | 2.00 |  5425 | 4.522e-12 |         20364 | 5.257e-12 |         23674 | 76.9 |
| P1 |  100 | 4.00 | 10850 | 2.156e-11 |         97116 | 1.145e-11 |         51587 | 165.9 |
| P2 |   50 | 0.50 |  1131 | 3.003e-12 |         13525 | 1.281e-11 |         57688 | 16.4 |
| P2 |   50 | 1.00 |  2261 | 1.209e-11 |         54464 | 2.343e-11 |        105501 | 38.3 |
| P2 |   50 | 2.00 |  4521 | 4.992e-11 |        224829 | 4.786e-11 |        215552 | 57.5 |
| P2 |   50 | 4.00 |  9042 | 2.028e-10 |        913322 | 9.539e-11 |        429581 | 117.6 |

## Fitted Time Law

`ε(T) = C · (T/1 s)^γ`, fitted per fixed resolution (γ measured, not assumed).

| Degree | N | γ_η | C_η @1s (ε_mach) | γ_u | C_u @1s (ε_mach) | max rel resid η | max rel resid u | η growth T_min→T_max |
|--------|---|-----|------------------|-----|------------------|-----------------|-----------------|----------------------|
| P1 |   50 | 2.73 | 638 | 1.20 | 3.06e+03 | 0.55 | 0.12 | 232.14 |
| P1 |  100 | 2.54 | 3.11e+03 | 1.23 | 9.69e+03 | 0.11 | 0.04 | 190.05 |
| P2 |   50 | 2.03 | 5.49e+04 | 0.97 | 1.1e+05 | 0.01 | 0.04 | 67.53 |

## Competing Hypotheses (same points, prescribed exponent)

| Degree | N | γ = 1 rms(η) | γ = 0.5 rms(η) | γ = 1 max(η) | γ = 0.5 max(η) | steps ratio | preferred | linear-in-T verdict | γ=1 off by (η, T_max) |
|--------|---|--------------|----------------|--------------|--------------|--------------|-----------|---------------------|----------------------|
| P1 |   50 | 2.18 | 3.80 | 3.84 | 7.13 | 8.0 | coherent (gamma=1) | rejected | x6.0 |
| P1 |  100 | 2.18 | 3.94 | 4.20 | 7.74 | 8.0 | coherent (gamma=1) | rejected | x4.6 |
| P2 |   50 | 1.04 | 2.02 | 1.90 | 3.88 | 8.0 | coherent (gamma=1) | rejected | x2.9 |

## Cross-Check Against the Mesh Study (T = 1 s)

| Degree | N | η (this study) | η (hydrostatic_wb_drift.json) | rel diff | u (this study) | u (artifact) | rel diff |
|--------|---|----------------|-------------------------------|----------|----------------|--------------|----------|
| P1 |   50 | 9.126e-14 | 9.126e-14 | 3.64e-06 | 7.747e-13 | 7.747e-13 | 2.04e-05 |
| P1 |  100 | 6.999e-13 | 6.999e-13 | 2.20e-05 | 2.225e-12 | 2.225e-12 | 4.22e-05 |
| P2 |   50 | 1.209e-11 | 1.167e-11 | 3.63e-02 | 2.343e-11 | 2.255e-11 | 3.88e-02 |

## Interpretation

`γ` is the measured exponent of the residual against integration time at
fixed mesh and degree.  Two hypotheses bracket the physics: a per-step
residual that is coherently biased gives `γ = 1`, an unbiased random walk
gives `γ = 0.5`.  Both are evaluated with the same four points, so the
table above shows the cost of getting the exponent wrong: a threshold
calibrated from the wrong model is off by `(T/1 s)^(1-γ)`.
The per-step means in the JSON (`eta_per_step_mean`) are derived from the
same runs; they are only meaningful when `γ ≈ 1`.

The measured `γ` is also the verdict on the linear-in-T model that the
drift-law document previously asserted: the last column of the table above
gives the factor by which a `γ = 1` threshold would mis-predict the longest
run.  A model that is off by more than a factor ~2 at the longest
integration time is recorded as `rejected`.
