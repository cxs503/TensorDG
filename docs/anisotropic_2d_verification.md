# Anisotropic-cell verification of the 2-D Euler DG solver (TensorDG)

> Type: **verification debt** -- an already-merged solver (`tensordg/euler_dg_2d.py`)
> re-measured against an analytic benchmark on meshes it has never been run on.
> Evidence: `validation/anisotropic_2d/verify.json` (generated 2026-11-04T11:28:19Z .. 2026-11-04T11:37:16ZZ).
> Regenerate: `PYTHONPATH=. python -m benchmarks.verify_anisotropic_2d` then
> `PYTHONPATH=. python -m benchmarks.report_anisotropic_2d`.
> Tests: `PYTHONPATH=. python -m pytest tests/test_euler_dg_2d_anisotropic.py`.

## 1. Why this study exists

Every convergence and limiter number in the repository was measured on **square**
cells: `validation/euler_dg_2d/design_order.json` runs `ncells = n^2` on
`[-5,5]^2`, and `validation/limiter_2d` inherits that mesh. On a square mesh
`dx == dy`, so the two places where the axes can be silently confused are
invisible:

1. the `1/(2dx)` / `1/(2dy)` scaling of the face terms in `_rhs`;
2. the per-axis `spacing` handed to the troubled-cell test, threaded through in
   mission m2549 -- the fix that made the TVB test scale consistent.

Both are exercised here on 2:1 stretched cells against the analytic isentropic
vortex (an exact solution of the 2-D Euler equations advected across a periodic
box), so the reference is analytic.

## 2. Setup

| item | value |
|---|---|
| domain | [[-5.0, 5.0], [-5.0, 5.0]] |
| final time | 1.0 |
| CFL | 0.2 |
| TVB constant | 1.0 |
| stretched ladder | [[8, 16], [12, 24], [16, 32]] (dy = dx/2) |
| isotropic control | [[8, 8], [12, 12], [16, 16]] (same `n_x`) |
| transposed twin | [[16, 8], [24, 12], [32, 16]] (x/y exchanged) |
| degrees | [1, 2] |
| threads | 2 |

## 3. Design order on stretched cells

### P1 (design order 2)

| ladder | cells | h | steps | rho | u | v | p | EOC rho | EOC worst |
|---|---|---|---|---|---|---|---|---|---|
| anisotropic | 8x16 | 0.884 | 106 | 0.011 | 0.020 | 0.026 | 0.015 | nan | nan |
| anisotropic | 12x24 | 0.589 | 159 | 0.005 | 0.009 | 0.012 | 0.007 | 1.77 | 1.77 |
| anisotropic | 16x32 | 0.442 | 212 | 0.003 | 0.005 | 0.007 | 0.004 | 2.08 | 1.80 |
| isotropic_control | 8x8 | 1.250 | 70 | 0.014 | 0.031 | 0.031 | 0.021 | nan | nan |
| isotropic_control | 12x12 | 0.833 | 106 | 0.007 | 0.014 | 0.014 | 0.010 | 1.78 | 1.78 |
| isotropic_control | 16x16 | 0.625 | 142 | 0.004 | 0.008 | 0.008 | 0.005 | 2.06 | 1.81 |
| transposed_control | 16x8 | 0.884 | 105 | 0.010 | 0.026 | 0.019 | 0.015 | nan | nan |
| transposed_control | 24x12 | 0.589 | 159 | 0.005 | 0.012 | 0.009 | 0.007 | 1.75 | 1.75 |
| transposed_control | 32x16 | 0.442 | 212 | 0.003 | 0.007 | 0.005 | 0.004 | 2.08 | 1.81 |

Least-squares order over the stretched ladder: `density` 1.89, `velocity_x` 2.00, `velocity_y` 1.92, `pressure` 1.92; isotropic control: `density` 1.88, `velocity_x` 1.88, `velocity_y` 1.93, `pressure` 1.94.

### P2 (design order 3)

| ladder | cells | h | steps | rho | u | v | p | EOC rho | EOC worst |
|---|---|---|---|---|---|---|---|---|---|
| anisotropic | 8x16 | 0.884 | 176 | 0.002 | 0.003 | 0.005 | 0.003 | nan | nan |
| anisotropic | 12x24 | 0.589 | 264 | 6.616e-04 | 9.997e-04 | 0.002 | 8.866e-04 | 3.16 | 2.60 |
| anisotropic | 16x32 | 0.442 | 352 | 2.850e-04 | 4.151e-04 | 7.189e-04 | 4.016e-04 | 2.93 | 2.75 |
| isotropic_control | 8x8 | 1.250 | 117 | 0.003 | 0.006 | 0.005 | 0.004 | nan | nan |
| isotropic_control | 12x12 | 0.833 | 176 | 9.139e-04 | 0.002 | 0.002 | 0.001 | 3.15 | 2.62 |
| isotropic_control | 16x16 | 0.625 | 234 | 3.976e-04 | 8.195e-04 | 8.186e-04 | 5.625e-04 | 2.89 | 2.73 |
| transposed_control | 16x8 | 0.884 | 175 | 0.002 | 0.005 | 0.003 | 0.003 | nan | nan |
| transposed_control | 24x12 | 0.589 | 264 | 6.607e-04 | 0.002 | 0.001 | 8.857e-04 | 3.15 | 2.59 |
| transposed_control | 32x16 | 0.442 | 352 | 2.847e-04 | 7.185e-04 | 4.150e-04 | 4.013e-04 | 2.93 | 2.75 |

Least-squares order over the stretched ladder: `density` 3.07, `velocity_x` 2.77, `velocity_y` 2.76, `pressure` 2.96; isotropic control: `density` 3.05, `velocity_x` 2.77, `velocity_y` 2.74, `pressure` 2.94.

**Transpose invariance.** The transposed twin reproduces the stretched ladder
with `u`/`v` exchanged:

| degree | cells | transposed | rho rel. diff | u/v swap rel. diff | steps |
|---|---|---|---|---|---|
| P1 | 8x16 | 16x8 | 1.487 % | 1.023 % | 106 / 105 |
| P1 | 12x24 | 24x12 | 0.612 % | 0.447 % | 159 / 159 |
| P1 | 16x32 | 32x16 | 0.436 % | 0.540 % | 212 / 212 |
| P2 | 8x16 | 16x8 | 0.455 % | 0.312 % | 176 / 175 |
| P2 | 12x24 | 24x12 | 0.133 % | 0.120 % | 264 / 264 |
| P2 | 16x32 | 32x16 | 0.097 % | 0.012 % | 352 / 352 |

The residual differences are at the 1 % level and track the one-step
difference in `steps` (the time step is adaptive), i.e. the discretisation is
transpose symmetric and the two runs differ only through their time grid.

![design order on stretched cells](docs/figures/anisotropic_2d/design_order_anisotropic.png)

## 4. Is the per-axis spacing in the troubled-cell test load bearing?

Mesh 8x64 (aspect 8.0:1, dx = 1.250, dy = 0.156), smooth profile
`u = 1 + A cos(2 pi (y-5)/5)`, amplitude calibrated so the data-owned TVB
deviation `|d - minmod|` sits at `0.049` -- between the
tolerance of the correct axis (`M dy^2 = 0.024`)
and of the wrong axis (`M dx^2 = 1.562`).

| spacing passed for the y axis | flagged cells |
|---|---|
| `dy` (correct) | 160 |
| `dx` (mutation) | 0 |

The verdict flips (`verdict_flips = True`), so the per-axis
threading of m2549 is observable; the discriminating TVB window is M in (0.031, 2.000).

Scale consistency of the flagging on a stretched refinement ladder (P2, smooth
vortex, M = 1):

| cells | h | flagged x | flagged y | y fraction | limiter modified | modified fraction |
|---|---|---|---|---|---|---|
| 8x16 | 0.884 | 0 | 2 | 1.56 % | 2 | 1.56 % |
| 12x24 | 0.589 | 2 | 4 | 1.39 % | 12 | 4.17 % |
| 16x32 | 0.442 | 2 | 5 | 0.98 % | 22 | 4.30 % |

## 5. Limiter on the smooth vortex (policy measurement)

| degree | ladder | limiter | cells | rho | EOC rho | EOC worst |
|---|---|---|---|---|---|---|
| P1 | anisotropic | tvb | 8x16 | 0.013 | nan | nan |
| P1 | anisotropic | tvb | 12x24 | 0.013 | -0.09 | -0.32 |
| P1 | anisotropic | tvb | 16x32 | 0.009 | 1.48 | 1.10 |
| P1 | anisotropic | weno | 8x16 | 0.014 | nan | nan |
| P1 | anisotropic | weno | 12x24 | 0.015 | -0.11 | -0.11 |
| P1 | anisotropic | weno | 16x32 | 0.010 | 1.36 | 0.88 |
| P1 | isotropic_control | tvb | 8x8 | 0.016 | nan | nan |
| P1 | isotropic_control | tvb | 12x12 | 0.014 | 0.34 | 0.05 |
| P1 | isotropic_control | tvb | 16x16 | 0.010 | 1.29 | 0.32 |
| P1 | isotropic_control | weno | 8x8 | 0.015 | nan | nan |
| P1 | isotropic_control | weno | 12x12 | 0.018 | -0.47 | -0.47 |
| P1 | isotropic_control | weno | 16x16 | 0.012 | 1.40 | 1.09 |
| P2 | anisotropic | tvb | 8x16 | 0.011 | nan | nan |
| P2 | anisotropic | tvb | 12x24 | 0.014 | -0.50 | -1.59 |
| P2 | anisotropic | tvb | 16x32 | 0.009 | 1.53 | 0.93 |
| P2 | anisotropic | weno | 8x16 | 0.015 | nan | nan |
| P2 | anisotropic | weno | 12x24 | 0.018 | -0.39 | -0.50 |
| P2 | anisotropic | weno | 16x32 | 0.012 | 1.23 | 0.94 |
| P2 | isotropic_control | tvb | 8x8 | 0.018 | nan | nan |
| P2 | isotropic_control | tvb | 12x12 | 0.015 | 0.40 | -0.88 |
| P2 | isotropic_control | tvb | 16x16 | 0.011 | 1.25 | 0.51 |
| P2 | isotropic_control | weno | 8x8 | 0.018 | nan | nan |
| P2 | isotropic_control | weno | 12x12 | 0.021 | -0.39 | -0.39 |
| P2 | isotropic_control | weno | 16x16 | 0.016 | 1.02 | 0.63 |

![limiter on stretched cells](docs/figures/anisotropic_2d/limiter_anisotropic.png)

## 6. Conservation and operator symmetry

| check | degree | measured |
|---|---|---|
| residual invariance under x<->y (u<->v) | P1 | 5.779e-16 |
| residual invariance under x<->y (u<->v) | P2 | 5.756e-16 |
| residual invariance under x<->y (u<->v) | P3 | 1.240e-15 |
| mass drift on 8x16 | P2 | 6.654e-15 |
| energy drift on 8x16 | P2 | 7.255e-15 |
| mass drift on 12x24 | P2 | 9.836e-15 |
| energy drift on 12x24 | P2 | 1.088e-14 |

## 7. Criteria

24/32 passed -- overall **FAIL**.

| criterion | measured | threshold | verdict | what it checks |
|---|---:|---:|:--:|---|
| `P1_anisotropic_lsq_order` | 1.891 | >= 1.700 | PASS | worst least-squares EOC over (rho,u,v,p), anisotropic ladder |
| `P1_isotropic_control_lsq_order` | 1.877 | >= 1.700 | PASS | worst least-squares EOC over (rho,u,v,p), isotropic_control ladder |
| `P1_transposed_control_lsq_order` | 1.875 | >= 1.700 | PASS | worst least-squares EOC over (rho,u,v,p), transposed_control ladder |
| `P1_anisotropic_finest_3pct` | 0.697 | <= 3.000 | PASS | worst relative L2 error on the finest stretched mesh, percent |
| `P1_transpose_density_8x16` | 0.015 | <= 0.050 | PASS | density error of the stretched ladder vs its transposed twin |
| `P1_transpose_density_12x24` | 0.006 | <= 0.050 | PASS | density error of the stretched ladder vs its transposed twin |
| `P1_transpose_density_16x32` | 0.004 | <= 0.050 | PASS | density error of the stretched ladder vs its transposed twin |
| `P2_anisotropic_lsq_order` | 2.758 | >= 2.700 | PASS | worst least-squares EOC over (rho,u,v,p), anisotropic ladder |
| `P2_isotropic_control_lsq_order` | 2.738 | >= 2.700 | PASS | worst least-squares EOC over (rho,u,v,p), isotropic_control ladder |
| `P2_transposed_control_lsq_order` | 2.758 | >= 2.700 | PASS | worst least-squares EOC over (rho,u,v,p), transposed_control ladder |
| `P2_anisotropic_finest_3pct` | 0.072 | <= 3.000 | PASS | worst relative L2 error on the finest stretched mesh, percent |
| `P2_transpose_density_8x16` | 0.005 | <= 0.050 | PASS | density error of the stretched ladder vs its transposed twin |
| `P2_transpose_density_12x24` | 0.001 | <= 0.050 | PASS | density error of the stretched ladder vs its transposed twin |
| `P2_transpose_density_16x32` | 9.677e-04 | <= 0.050 | PASS | density error of the stretched ladder vs its transposed twin |
| `limiter_wrong_axis_verdict_differs` | 1.000 | >= 1.000 | PASS | troubled-cell verdict must depend on the per-axis spacing (mutation probe) |
| `limiter_flagged_fraction_not_growing` | 0.010 | <= 0.016 | PASS | flagged fraction on the fine axis must not grow under refinement at fixed M (m2549 scale consistency on stretched cells) |
| `limiter_flagged_fraction_bounded` | 0.010 | <= 0.500 | PASS | at M = 1 the flagged fraction on a resolved smooth flow stays a minority |
| `limiter_tvb_anisotropic_P1_order` | 0.232 | >= 1.500 | FAIL | worst EOC with the tvb limiter active on the smooth vortex (anisotropic) -- the limiter must not destroy the smooth order |
| `limiter_weno_anisotropic_P1_order` | 0.463 | >= 1.500 | FAIL | worst EOC with the weno limiter active on the smooth vortex (anisotropic) -- the limiter must not destroy the smooth order |
| `limiter_tvb_isotropic_control_P1_order` | 0.310 | >= 1.500 | FAIL | worst EOC with the tvb limiter active on the smooth vortex (isotropic_control) -- the limiter must not destroy the smooth order |
| `limiter_weno_isotropic_control_P1_order` | 0.257 | >= 1.500 | FAIL | worst EOC with the weno limiter active on the smooth vortex (isotropic_control) -- the limiter must not destroy the smooth order |
| `limiter_tvb_anisotropic_P2_order` | -0.608 | >= 2.500 | FAIL | worst EOC with the tvb limiter active on the smooth vortex (anisotropic) -- the limiter must not destroy the smooth order |
| `limiter_weno_anisotropic_P2_order` | 0.059 | >= 2.500 | FAIL | worst EOC with the weno limiter active on the smooth vortex (anisotropic) -- the limiter must not destroy the smooth order |
| `limiter_tvb_isotropic_control_P2_order` | -0.189 | >= 2.500 | FAIL | worst EOC with the tvb limiter active on the smooth vortex (isotropic_control) -- the limiter must not destroy the smooth order |
| `limiter_weno_isotropic_control_P2_order` | 0.156 | >= 2.500 | FAIL | worst EOC with the weno limiter active on the smooth vortex (isotropic_control) -- the limiter must not destroy the smooth order |
| `operator_transpose_P1` | 5.779e-16 | <= 1.000e-14 | PASS | residual invariance under x<->y with u<->v (exact algebraic symmetry) |
| `operator_transpose_P2` | 5.756e-16 | <= 1.000e-14 | PASS | residual invariance under x<->y with u<->v (exact algebraic symmetry) |
| `operator_transpose_P3` | 1.240e-15 | <= 1.000e-14 | PASS | residual invariance under x<->y with u<->v (exact algebraic symmetry) |
| `conservation_mass_8x16` | 6.654e-15 | <= 1.000e-12 | PASS | relative mass drift on a periodic stretched mesh (round-off) |
| `conservation_energy_8x16` | 7.255e-15 | <= 1.000e-12 | PASS | relative energy drift on a periodic stretched mesh (round-off) |
| `conservation_mass_12x24` | 9.836e-15 | <= 1.000e-12 | PASS | relative mass drift on a periodic stretched mesh (round-off) |
| `conservation_energy_12x24` | 1.088e-14 | <= 1.000e-12 | PASS | relative energy drift on a periodic stretched mesh (round-off) |

### Honest FAIL

The following criteria failed and are **not** whitewashed:

* `limiter_tvb_anisotropic_P1_order`
* `limiter_weno_anisotropic_P1_order`
* `limiter_tvb_isotropic_control_P1_order`
* `limiter_weno_isotropic_control_P1_order`
* `limiter_tvb_anisotropic_P2_order`
* `limiter_weno_anisotropic_P2_order`
* `limiter_tvb_isotropic_control_P2_order`
* `limiter_weno_isotropic_control_P2_order`

Interpretation: the TVB/WENO limiter is *not* inert on a resolved smooth
flow at `M = 1` with limiting applied at every SSP-RK stage -- it removes
higher modes wherever the local cell-average curvature exceeds `M h^2`.
This is a **policy** defect of the limiter settings, not an anisotropic
wiring defect: the isotropic control ladder degrades in the same way and
the stretched ladder tracks it, which is exactly the discrimination the
design-order and transpose checks above provide.

## 8. Notes and limitations

* `p = 3` was not swept on the stretched ladder (budget: the shared box was at
  load ~110/128 cores while this ran); P1 and P2 bracket the design-order
  claim, and the exact algebraic transpose test covers `p = 3` at zero cost.
* The stretched meshes keep the refinement ratio fixed at 2:1; strongly
  stretched cells (aspect > 4) are untested for the *order* claim, though the
  mutation probe runs at aspect 8:1.
* The error norm is the Gauss-node `L2` norm of the primitive fields
  (`benchmarks.verify_euler_dg_2d._vortex_error`); a cell-centre-value norm
  would floor at `O(h^2)` and was rejected for exactly that reason.

