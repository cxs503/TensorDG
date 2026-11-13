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

At fixed CFL the step count is proportional to the final time, so a longer run
performs proportionally more floating-point work.  Earlier versions of this
document asserted, without measuring it, that the residual therefore grows
*linearly* in `T` (`ε_η(N, T) ≈ ε_η(N, T=1) · (T / 1 s)`).  That statement has
now been measured and **it does not hold for η**.

The measurement (`validation/swe_thacker/hydrostatic_wb_drift_time.json`) scans
`T ∈ {0.5, 1, 2, 4} s` at fixed resolution, three series (P1/N=50, P1/N=100,
P2/N=50), and fits

```
ε(T) = C · (T / 1 s)^γ        (γ measured by log-log least squares, never assumed)
```

| Degree | N | γ_η | C_η @ T=1 s (ε_machine) | γ_u | C_u @ T=1 s (ε_machine) | `γ=1` model off by |
|--------|---|-----|-------------------------|-----|-------------------------|--------------------|
| P1 |  50 | 2.73 | 6.4e+02 | 1.20 | 3.1e+03 | ×6.0 |
| P1 | 100 | 2.54 | 3.1e+03 | 1.23 | 9.7e+03 | ×4.6 |
| P2 |  50 | 2.03 | 5.5e+04 | 0.97 | 1.1e+05 | ×2.9 |

Read the last column as: with a `γ = 1` threshold, the longest run in the scan
(`T = 4 s`, 8× the steps of the shortest) would be mis-predicted by that
factor.  Concretely, the η drift grows 68× (P2/N=50) to 232× (P1/N=50) over an
8× longer integration, i.e. **two orders of magnitude where a linear law
predicts one order**; the fitted single exponent also under-describes the
P1/N=50 series (max relative residual 0.55, versus 0.01 for P2/N=50).

The spurious velocity is the exception: `γ_u = 0.97 … 1.23`, and the `γ = 1`
model is within a factor 1.2 of the measured `T = 4 s` value for every series.
That matters because rest-state tests usually assert on `max|u|`; for that
quantity the linear-in-`T` picture survives, for `η` it does not.

Practical rule — scale a long-run threshold with the *measured* exponent, taken
per series from the artifact:

```python
fit = artifact["series"][0]["time_fit"]          # pick the matching (degree, N)
expected = fit["C_eta"] * (T / 1.0) ** fit["gamma_eta"]
assert eta_error < 10.0 * expected               # same 10× safety factor as above
```

The mechanism behind `γ_η > 1` is *not* established by these four points per
series; what is established is that `γ_η ≈ 1` is wrong by a factor 3-6 at
`T = 4 s` and that thresholds must not assume it.  A per-step mean quote
(`eta_per_step_mean` in `hydrostatic_wb_drift.json`) is only meaningful as a
summary when `γ ≈ 1`, i.e. for the velocity, not for `η`.

Cross-check: the `T = 1 s` point of every series was re-solved while producing
this artifact and compared with `hydrostatic_wb_drift.json` — P1 agrees to
`4e-5` relative, P2/N=50 to `3.6e-2` (thread-count dependent reduction order in
the round-off; the artifact records the solver environment in `provenance`).

### When to Worry

The drift-law errors are always at the round-off level and never affect
physical accuracy.  They only matter for:

1. **Automated testing**: thresholds must account for the drift.
2. **Conservation checks**: mass conservation is exact to round-off, but the
   round-off level grows with N.
3. **Regression detection**: a change in the measured exponent (α or β) or in
   the constant C at fixed N — or in the `T`-exponents γ_η / γ_u — is a signal
   that the operation ordering of the well-balanced residual changed — e.g. a
   new flux/source cancellation path.
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

# Time-scaling law: 12 solves (~12 min single core), then the same refit/guard pair
PYTHONPATH=. python validation/swe_thacker/run_hydrostatic_wb_drift_time.py
PYTHONPATH=. python validation/swe_thacker/run_hydrostatic_wb_drift_time.py --refit
PYTHONPATH=. python -m pytest tests/test_swe_wb_drift_law_time_scaling.py
```

`tests/test_swe_wb_drift_law_time_scaling.py` re-derives every `γ` from the
committed runs, checks that the step count tracks `T`, that the drift really
grows with integration time, that both competing exponents are scored on the
same points, that the markdown table and the figure agree with the JSON, and
reproduces one point (`P1`, `N=50`, `T=0.5 s`) with a fresh solve.

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
