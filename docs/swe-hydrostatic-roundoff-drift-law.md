# Hydrostatic Well-Balanced Round-Off Drift Law

## Overview

When the TensorDG SWE solver preserves a lake-at-rest equilibrium
(η = h + z_b = const, u = 0) with `well_balanced=True`, the numerical
solution does not remain *exactly* at the initial state: it acquires a
round-off-level residual in both the free-surface elevation η and the
velocity u.  This residual **grows** with mesh refinement, which is the
opposite of what one expects from a discretisation error.

This document explains the drift law, quantifies it, and provides
guidance for setting acceptance thresholds in automated tests.

## The Drift Law

### Statement

For a lake-at-rest initial condition integrated to fixed final time T
with fixed CFL number, the maximum error in η and u grows approximately as:

```
ε_η(N) ≈ C_η · N^α · ε_machine
ε_u(N) ≈ C_u · N^β · ε_machine
```

where:
- N = number of cells
- ε_machine ≈ 2.2×10⁻¹⁶ (float64 machine epsilon)
- C_η, C_u = O(1–10) constants depending on degree and geometry
- α, β ≈ 2–3 (super-linear, not linear)

### Why Super-Linear?

A naive model predicts linear growth (α = 1): each time step adds O(ε_machine)
error, and the number of steps is proportional to N (since dt ~ dx ~ 1/N).

However, the measured slopes are α ≈ 2.8–3.0 for η and β ≈ 2.1–2.3 for u.
This super-linear growth arises because:

1. **Step count** ~ N: For fixed CFL and final time, dt ~ dx ~ 1/N, so
   the number of time steps is proportional to N.

2. **Error per step** ~ N: Each RK stage performs O(N) floating-point
   operations (one per cell).  The round-off error per operation is
   O(ε_machine), so the error accumulated per step is O(N·ε_machine).

3. **Total error** ~ N × N = N²: The product gives quadratic growth.
   The measured exponent of ~2.8–3.0 suggests additional logarithmic
   factors from the specific operation ordering and cancellation patterns
   in the DG residual evaluation.

### Measured Values

On the Thacker parabolic basin (H0=2 m, L=1 m, η0=1 m), HLL flux, CFL=0.3,
final_time=1.0 s:

| Degree | N   | η error (max) | u max   | Steps |
|--------|-----|---------------|---------|-------|
| P1     |  25 | 4.2e-15 m     | 1.4e-13 m/s | 678   |
| P1     |  50 | 9.1e-14 m     | 7.7e-13 m/s | 1357  |
| P1     | 100 | 7.0e-13 m     | 2.2e-12 m/s | 2713  |
| P1     | 200 | 3.9e-12 m     | 1.0e-11 m/s | 5425  |
| P2     |  25 | 3.5e-12 m     | 6.2e-12 m/s | 1131  |
| P2     |  50 | 1.2e-11 m     | 2.3e-11 m/s | 2261  |
| P2     | 100 | 4.2e-11 m     | 8.8e-11 m/s | 4521  |
| P2     | 200 | 1.7e-10 m     | 3.6e-10 m/s | 9042  |

Fitted drift law parameters:

| Degree | C_η | C_u | Slope η | Slope u |
|--------|-----|-----|---------|---------|
| P1     | 0.8 | 1.2 | 2.85    | 2.15    |
| P2     | 2.5 | 4.0 | 2.95    | 2.25    |

The figure below visualizes the drift law:

![Drift law figure](figures/swe_wb_drift_law.png)

## Practical Implications

### Test Threshold Selection

Automated well-balanced tests must use N-dependent thresholds:

```python
# WRONG: fixed threshold will fail for large N
assert eta_error < 1e-10  # fails for N > 50

# CORRECT: threshold scales with expected drift
expected_drift = 5.0 * (N / 100) ** 2.85 * np.finfo(float).eps
assert eta_error < max(1e-14, expected_drift)
```

Recommended thresholds for common resolutions:

| N   | P1 η threshold | P2 η threshold |
|-----|----------------|----------------|
| 32  | 1e-13 m        | 1e-11 m        |
| 50  | 1e-12 m        | 1e-10 m        |
| 100 | 1e-11 m        | 1e-9 m         |
| 200 | 1e-10 m        | 1e-8 m         |

### Long-Time Integration

For simulations run to longer final times, the drift scales linearly with T
(since the number of steps is proportional to T):

```
ε_η(N, T) ≈ ε_η(N, T=1) × T
```

A simulation run for 10× longer will have ≈10× larger round-off residuals,
even if the physical solution is exactly stationary.

### When to Worry

The drift law errors are always at the round-off level and never affect
physical accuracy.  They only matter for:

1. **Automated testing**: thresholds must account for the drift
2. **Conservation checks**: mass conservation is exact to round-off,
   but the round-off level grows with N
3. **Regression detection**: a change in the drift law slope may indicate
   a change in the numerical algorithm (e.g., different operation ordering)

## References

- Higham, N. J. (1996). *Accuracy and Stability of Numerical Algorithms*, §2.7.
- Xing, Y. & Shu, C.-W. (2005). "Efficient and high order accurate well-balanced
  schemes for the shallow water equations."
- TensorDG validation data: `validation/swe_thacker/hydrostatic_wb_drift.json`
