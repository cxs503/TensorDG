# Conservation-law verification with TensorDG

This tutorial explains how to *measure* whether a TensorDG run conserves what it
should, whether its time integrator delivers the requested order, and how to
tell a numerical defect from physics.  It complements
[`convergence-verification.md`](convergence-verification.md), which measures the
*rate* of error decay.

The measurement helpers live in `tests/_conservation_tools.py`, the complete
study (with committed artifacts) in `tests/conservation_study.py`, and the
machine-checked criteria in `tests/test_conservation.py`.

## 1. Why a rate study is not enough

A p/h convergence study answers "how fast does the error go away".  It does not
answer two questions that decide whether a solution is a *solution*:

* **Does the scheme respect the conservation law?**  A flux-difference (or DG)
  discretisation conserves a quantity exactly when the interface fluxes cancel
  pairwise in a telescope.  If the implementation forgets one face, or applies a
  different flux on the boundary, the total drifts — often smoothly, so the
  convergence rate still looks fine.
* **Does it converge to the right weak solution?**  Across a shock, the
  convergence rate is meaningless (the solution is not smooth).  The correct
  statement is the Lax–Wendroff one: a conservative, converging scheme produces
  the discontinuity speed fixed by the Rankine–Hugoniot jump conditions.

## 2. The discrete invariant

In a modal DG state the `k = 0` Legendre coefficient *is* the cell average:
`P_0 = 1`, and every higher mode integrates to zero over the element.  So

```python
from _conservation_tools import cell_averages_1d, integral_1d, scaled_drift

centers, coeff, meta = solve_advection(ncells=32, degree=2, final_time=1.0)
averages = cell_averages_1d(coeff)        # shape (ncells,)
mass = integral_1d(averages, length=1.0)  # h * sum(u_bar)
```

Two practical warnings:

1. **Never normalise by the invariant itself.**  `sin(2*pi*x)` has zero mean, so
   the discrete mass is machine-small and `|M(T)-M(0)|/max|M|` reports a
   meaningless "100 % drift".  Pass an explicit scale — the discrete `L1` norm
   `h*sum|u|` is the natural one — as `scaled_drift(history, scale=...)` does.
2. A conserved quantity drifts only at round-off level, and that level must not
   grow with refinement: accumulating `~2e-16` per step over `O(1/h)` steps gives
   an `O(eps/h)`-bounded drift.  A drift that grows like `h^p` is a defect.

## 3. Global budgets for a system of conservation laws

For Euler, integrate the conservative variables directly:

```python
from _conservation_tools import conserved_totals, boundary_increment

totals = conserved_totals(fields["conservative"], cell_width=1.0 / ncells)
predicted = boundary_increment(left_state, right_state, final_time)
```

`boundary_increment` is the analytic statement

```
int_0^1 U(t) dx - int_0^1 U(0) dx = t * (F(U_L) - F(U_R))
```

which holds *exactly* while the waves have not reached the ends: the transmissive
boundary keeps the initial states on the mesh ends, so the boundary fluxes are
time-independent.  This is the sharpest cheap test of the boundary bookkeeping —
for the Sod problem it predicts invariant mass and energy, and a momentum
integral growing at the pressure imbalance `p_L - p_R`.  A wall-like boundary or
a dropped face flux misses it by `O(100 %)`.

## 4. Rankine–Hugoniot shock speed

```python
from _conservation_tools import rankine_hugoniot_speed, shock_position, linear_slope
```

* `rankine_hugoniot_speed(upstream, downstream)` returns
  `(rho2*u2 - rho1*u1)/(rho2 - rho1)`.  Feed it the exact star state from
  `benchmarks.riemann_exact.star_state` and you get an analytic reference that
  does not depend on the solver at all.  Cross-check it with the momentum jump:
  both estimates must agree to round-off, which validates the star state and the
  helper together.
* `shock_position(x, values, level)` finds the first left-to-right crossing of a
  level by linear interpolation.  **Calibrate the sensor on the exact profile
  first**: a sensor with an `O(dx)` bias would otherwise be mistaken for a
  solver error.  Use `level = (rho_star + rho_right)/2`.
* Fit `x_s(t) = x_0 + W t` with `linear_slope(times, positions)`; compare `W`
  with the RH value.  The error decreases under refinement — it is a
  representation error of the smeared shock, not a modelling error.

## 5. Time-integrator order without a spatial error floor

Comparing two solutions on **the same mesh** with two different time steps
cancels the spatial error, so the residual difference is purely temporal:

```python
from _conservation_tools import relative_difference, fit_order

diffs = [relative_difference(u_coarse[i + 1], u_coarse[i]) for i in range(len(u_coarse) - 1)]
order = fit_order(dts, diffs)          # ~4 for RK4, ~3 for SSP-RK3
```

This is the correct way to verify a time integrator: an error-against-exact
study would be polluted by the spatial floor as soon as `dt` becomes small.

## 6. Reading the results

| symptom | diagnosis |
|---|---|
| periodic mass drift `~1e-15`, mesh independent | correct: round-off only |
| mass drift `~h^p`, `p > 0` | a face flux is missing or asymmetric |
| momentum increment misses `t*(p_L-p_R)` | boundary flux not the physical flux, or a dropped end face |
| shock speed off by several per cent, mesh independent | non-conservative formulation (e.g. a quasi-linear update) |
| temporal order measurably below the RK order | stage coefficients wrong, or `dt` not the one reported in metadata |
| far-field noise *ahead* of the first characteristic | boundary closure defect — double the domain and re-measure to confirm |

## 7. Why the invariant checks need `ncells >= 400` (measured, not assumed)

A shock-capturing scheme leaves a small residual *ahead* of the smeared front.
For the Sod problem at `t = 0.2` the exact shock sits at `x_s = 0.850` and the
exact solution is *exactly* the uniform right state beyond it, so any signal in
`x in [0.9, 1.0]` (ahead of the front) is that residual.  Two measurements identify it:

* **Repeat the window on a domain of twice the length at the same `dx`.**  The
  value is unchanged (relative gap at round-off), so the transmissive boundary is
  *not* the cause — the residual is an interior object.
* **Fit `log|rho*u|` against the cell index behind the front.**  The tail decays
  by a factor `e` per cell, i.e. its width is `O(1)` cells while `dx` shrinks:
  it is the discrete shock profile, not physical diffusion.

Consequence: at a fixed physical distance from the front, the residual amplitude
drops super-algebraically with `dx` (the fitted rate in
`docs/benchmarks/results/conservation/metrics.json` is far above `h^2`).  On
coarse meshes the tail still reaches the last cell, so the boundary sees a
genuine outflow and the *global* integral legitimately changes; that is not a
conservation failure.  Assert invariants at the round-off level only for
`ncells >= 400`, where the tail has decayed below the round-off floor
(`tests/conservation_study.py::study_shock_precursor`).

The same logic applies to the 3 % field-error band: it is a claim about the
*documented* resolution (3200 cells, first-order `P0`).  `python
tests/conservation_study.py --quick` therefore reports
`sod_field_error_vs_exact` as FAIL by design — it runs 400–800 cells, where the
smeared shock is simply not resolved.  Quick mode also refuses to write
artifacts so a smoke run can never overwrite the committed full-mode numbers.
