# Hydrostatic Well-Balanced Round-Off Drift Law

## Overview

When the TensorDG SWE solver preserves a lake-at-rest equilibrium
(η = h + z_b = const, u = 0) with `well_balanced=True`, the numerical
solution does not remain *exactly* at the initial state: it acquires a
round-off-level residual in both the free-surface elevation η and the
velocity u.  This residual **grows** with mesh refinement, which is the
opposite of what one expects from a discretisation error.

This document explains the drift law, quantifies it, and provides
guidance for setting acceptance thresholds in automated tests.  Every number
here is generated from `validation/swe_thacker/hydrostatic_wb_drift.json` by
`validation/swe_thacker/run_hydrostatic_wb_drift.py`; the fits are derived by
least squares from the measured runs, never typed in by hand, and
`tests/test_swe_wb_drift_law_artifact.py` re-derives them independently and
fails if they disagree.

## The Drift Law

### Statement

For a lake-at-rest initial condition integrated to fixed final time T
with fixed CFL number, the maximum error in η and u follows a power law in
the number of cells N:

```
ε_η(N) ≈ C_η · (N/100)^α · ε_machine
ε_u(N) ≈ C_u · (N/100)^β · ε_machine
```

where:

- N = number of cells;
- ε_machine = 2.2204460492503131e-16 (float64);
- C_η, C_u = the fitted round-off level at N = 100 cells, in units of
  ε_machine (a directly readable "magnitude at a typical resolution");
- α, β = **measured** exponents from an ordinary least-squares fit of
  log(ε/ε_machine) against log(N).

The constants are reported at a reference resolution rather than as an
abstract intercept so that "how large is the drift?" can be answered without
re-doing the arithmetic.

### Why the drift grows with N

A naive model predicts ε ∝ N (α = 1): each time step adds O(ε_machine) error,
and the number of steps is proportional to N (since dt ~ dx ~ 1/N at fixed
CFL and final time).  In reality the round-off injected per step also grows
with N, because every RK stage evaluates an O(N) residual and the flux/source
cancellation that makes the scheme well balanced is the difference of two
O(1) floating-point expressions — its relative error is O(ε_machine), but the
absolute residual accumulates over all cells and stages:

1. **Step count** ~ N: dt ~ dx ~ 1/N, so the number of accepted steps is
   proportional to N (also visible in the table above).
2. **Residual per step** ~ N: every stage touches N cells, and the
   hydrostatic flux/source cancellation leaves an O(N·ε_machine) absolute
   round-off.
3. **Total drift** = steps × per-step residual ~ N², i.e. α ≈ β ≈ 2 in the
   simplest model.

The measured exponents are α = 3.25, β = 2.02 for P1 and
α = 1.87, β = 1.95 for P2.  P2 sits close to the N² baseline.  P1 is
steeper because its coarse-mesh residual is exceptionally clean (η error
4.2e-15 m ≈ 19 ε_machine at N = 25), so the four-point sequence bends in
log-log space: the reported `max_rel_residual_eta` for P1 is
0.45, i.e. the single-exponent law describes that series only to
within ~45%.  This is stated rather than hidden: see "Caveat" below.

### Measured Values

Thacker parabolic basin (H0 = 2 m, L = 1 m, η0 = 1 m), lake-at-rest
initial condition, HLL flux, CFL = 0.3, final_time = 1 s, reconstruction
with `zb_center = zb_modes[:, 0]`:

| Degree | N | Steps | η error (max) | u error (max) | η / ε_machine | u / ε_machine |
|--------|---|-------|---------------|---------------|---------------|---------------|
| P1 |  25 |   678 | 4.22e-15 m | 1.37e-13 m/s |     19 |    616 |
| P1 |  50 |  1357 | 9.13e-14 m | 7.75e-13 m/s |    411 |   3489 |
| P1 | 100 |  2713 | 7.00e-13 m | 2.22e-12 m/s |   3152 |  10021 |
| P1 | 200 |  5425 | 3.94e-12 m | 1.01e-11 m/s |  17758 |  45621 |
| P2 |  25 |  1131 | 3.46e-12 m | 6.22e-12 m/s |  15560 |  28008 |
| P2 |  50 |  2261 | 1.17e-11 m | 2.25e-11 m/s |  52557 | 101556 |
| P2 | 100 |  4521 | 4.25e-11 m | 8.82e-11 m/s | 191358 | 397398 |
| P2 | 200 |  9042 | 1.68e-10 m | 3.57e-10 m/s | 756605 | 1605984 |

Both errors are many orders of magnitude below any physical tolerance and
they *increase* with N — the signature of accumulated floating-point noise,
not of a discretisation error.

### Fitted Law

| Degree | C_η (N=100) | C_u (N=100) | Slope α | Slope β | max rel. residual (η) | max rel. residual (u) |
|--------|-------------|-------------|---------|---------|-----------------------|-----------------------|
| P1 | 2.51e+03 | 1.13e+04 | 3.25 | 2.02 | 0.45 | 0.20 |
| P2 | 1.99e+05 | 4.06e+05 | 1.87 | 1.95 | 0.04 | 0.03 |

`C_η`, `C_u` are in units of ε_machine; `max_rel_residual` is the largest
relative departure of the fitted power law from the four measured points of
that degree.

### Caveat: one exponent is a summary, not a law of nature

The P2 series (η residual 0.04, u residual 0.03) is well described by a
single exponent.  The P1 η series is not: at N = 25 the residual is only
~19 ε_machine, i.e. the cancellation is unusually exact there, and the
points from N = 50 upwards are steeper than the single fit.  For threshold
selection use the *upper envelope*: evaluate the law and multiply by a safety
factor (see below).  The figure `figures/swe_wb_drift_law.png` shows the
measured points, the fitted power law, and the N·ε_machine reference.

## Practical Implications

### Test Threshold Selection

Automated well-balanced tests must use N-dependent thresholds.  A fixed
absolute threshold will either be too loose at low N or spuriously fail at
high N:

```python
# WRONG: fixed threshold does not follow the drift law
assert eta_error < 1e-10

# CORRECT: threshold from the fitted drift law, with a safety factor
eps = np.finfo(float).eps
C_eta, alpha = 2511.8, 3.25   # P1 η, from the artifact
expected_drift = C_eta * (N / 100.0) ** alpha * eps
assert eta_error < 10.0 * expected_drift
```

Fitted drift levels (safety factor 1, i.e. the fitted law itself) for common
resolutions:

| N | P1 η | P2 η | P1 u | P2 u |
|---|------|------|------|------|
| 32 | 1.4e-14 m | 5.3e-12 m | 2.5e-13 m/s | 9.8e-12 m/s |
| 50 | 5.8e-14 m | 1.2e-11 m | 6.2e-13 m/s | 2.3e-11 m/s |
| 100 | 5.6e-13 m | 4.4e-11 m | 2.5e-12 m/s | 9.0e-11 m/s |
| 200 | 5.3e-12 m | 1.6e-10 m | 1.0e-11 m/s | 3.5e-10 m/s |

Recommendation: use `max(1e-14, 10 × fitted)` as an acceptance bound.  Do not
tighten below the fitted level: that would turn a correct scheme red.

### Long-Time Integration

At fixed CFL the number of steps is proportional to the final time, and the
residual re-accumulates at every step, so the drift is expected to scale
approximately linearly with T:

```
ε_η(N, T) ≈ ε_η(N, T=1) · (T / 1 s)
```

A simulation run 10× longer will have ≈10× larger round-off residuals, even
though the physical solution is exactly stationary.  (This linear-in-T
scaling is the model behind the per-step columns in
`hydrostatic_wb_drift_table.md`; it is not part of the fitted N-law.)

### When to Worry

The drift-law errors are always at the round-off level and never affect
physical accuracy.  They only matter for:

1. **Automated testing**: thresholds must account for the drift.
2. **Conservation checks**: mass conservation is exact to round-off, but the
   round-off level grows with N.
3. **Regression detection**: a change in the measured exponent (α or β) or in
   the constant C at fixed N is a signal that the operation ordering of the
   well-balanced residual changed — e.g. a new flux/source cancellation path.
   `tests/test_swe_wb_drift_law_artifact.py` guards exactly this by re-deriving
   the fits from the committed runs and by reproducing one artifact point with
   a fresh solve.

## Reproducing the Artifact

```bash
# Full study (8 solves, several minutes on one core)
PYTHONPATH=. python validation/swe_thacker/run_hydrostatic_wb_drift.py

# Re-derive the fits, table and figure from the committed runs (no PDE solves)
PYTHONPATH=. python validation/swe_thacker/run_hydrostatic_wb_drift.py --refit

# Guard: artifact, table and figure must agree with an independent refit
PYTHONPATH=. python -m pytest tests/test_swe_wb_drift_law_artifact.py
```

Provenance (`generated_at_utc`, command, interpreter/library versions, host)
is written into the JSON by the script: `runs` records the solve environment,
and, when `--refit` is used, `fits_recomputed_at_utc` /
`fits_recomputed_by` record that only the derived quantities were refreshed.

## References

- Higham, N. J. (1996). *Accuracy and Stability of Numerical Algorithms*, §2.7.
- Xing, Y. & Shu, C.-W. (2005). "Efficient and high order accurate well-balanced
  schemes for the shallow water equations."
- Thacker, W. C. (1981). "Some exact solutions of the nonlinear shallow-water
  equations." *Journal of Fluid Mechanics*, 107, 499-508.
- TensorDG validation data: `validation/swe_thacker/hydrostatic_wb_drift.json`
