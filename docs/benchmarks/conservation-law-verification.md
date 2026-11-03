# Conservation-law and temporal-order verification (TensorDG)

Machine-readable record: [`results/conservation/metrics.json`](results/conservation/metrics.json)
(`2026-11-02T16:57:57Z`, `full` mode).  Regenerate with
`python tests/conservation_study.py`; the criteria below are re-asserted by
`python -m pytest tests/test_conservation.py`.

## Why this study exists

The S1 round verified the *rate* of error decay (p/h refinement + Roache GCI).
A rate says nothing about whether the discretisation respects the conservation
law it discretises, nor about the time integrator.  This study adds four
independent measurement families and one reference comparison:

| family | falsifiable statement | reference |
|---|---|---|
| discrete invariants | periodic mass drift stays at float64 round-off and does not grow with refinement | floating point |
| Euler boundary budget | `int U(t)-int U(0) = t*(F(U_L)-F(U_R))` for a Sod tube whose waves stay inside | analytic PDE statement |
| shock speed | the captured discontinuity moves at the Rankine-Hugoniot speed (Lax-Wendroff) | exact Riemann star state |
| time integrator | dt-only self-convergence reproduces the RK order on a fixed mesh | RK theory (4 and 3) |
| field error | density/velocity/pressure inside the documented 3 % band | exact Riemann solution |

## Criteria

17/17 passed — overall **PASS**.

| criterion | measured | threshold | verdict | what it checks |
|---|---:|---:|:--:|---|
| `periodic_mass_drift_scaled` | 7.405203e-15 | <= 1.000e-12 | PASS | worst relative mass drift, periodic advection 1-D/2-D + Burgers |
| `periodic_drift_grid_independence` | 1.604962e+00 | <= 1.000e+01 | PASS | round-off drift of the coarsest vs the finest mesh |
| `euler_mass_drift` | 4.204045e-14 | <= 1.000e-12 | PASS | Sod mass integral vs initial mass (waves stay inside) |
| `euler_energy_drift` | 4.021026e-14 | <= 1.000e-12 | PASS | Sod total-energy integral vs initial energy |
| `euler_momentum_budget` | 5.535695e-14 | <= 1.000e-06 | PASS | measured momentum increment vs analytic t*(p_L-p_R) |
| `rh_jump_consistency` | 3.801796e-16 | <= 1.000e-10 | PASS | RH speed from the mass jump vs from the momentum jump |
| `shock_sensor_bias` | 1.176105e-05 | <= 1.000e-03 | PASS | level-crossing sensor on the exact profile vs analytic shock path |
| `shock_speed_error_finest` | 5.416509e-05 | <= 1.000e-02 | PASS | finest mesh shock speed vs Rankine-Hugoniot speed |
| `shock_speed_error_coarsest` | 4.509403e-04 | <= 1.000e-02 | PASS | coarsest mesh shock speed vs Rankine-Hugoniot speed |
| `shock_speed_error_not_worse_under_refinement` | 1.000000e+00 | >= 1.000e+00 | PASS | shock-speed error at the finest mesh <= coarsest mesh |
| `precursor_amplitude_fine_mesh` | 4.796108e-09 | <= 1.000e-06 | PASS | residual \|rho*u\| in an untouched window, meshes >= 400 cells |
| `precursor_domain_independence` | 4.267933e-16 | <= 1.000e-06 | PASS | same window on a doubled domain at the same dx (rules out the wall) |
| `precursor_decay_rate` | 1.616622e+01 | >= 2.000e+00 | PASS | fitted decay rate of the residual amplitude in h |
| `precursor_tail_width` | 1.066706e+00 | <= 5.000e+00 | PASS | exponential tail width in cells (O(1) cells = discrete-shock artefact) |
| `sod_field_error_vs_exact` | 2.886883e-02 | <= 3.000e-02 | PASS | worst relative L2 of density/velocity/pressure vs the exact Riemann solution, at the finest computed resolution |
| `rk4_temporal_order` | 3.999994e+00 | >= 3.500e+00 | PASS | measured RK4 order on a fixed mesh (theory 4) |
| `ssp_rk3_temporal_order` | 2.993931e+00 | >= 2.600e+00 | PASS | measured SSP-RK3 order on a fixed mesh (theory 3) |

## 1. Periodic discrete invariants (relative mass drift)

The invariant is the `P_0` modal integral `h*sum(u_bar)`; the drift is normalised
by the discrete `L1` norm so that zero-mean data stays comparable.

| case | degree | cells | mass(0) | mass(T) | absolute drift | scaled drift |
|---|---:|---:|---:|---:|---:|---:|
| linear_advection_1d | 1 | 16 | 0.000000e+00 | 9.194034e-17 | 9.194e-17 | 1.452e-16 |
| linear_advection_1d | 1 | 32 | -6.071532e-18 | 1.847481e-16 | 1.908e-16 | 2.999e-16 |
| linear_advection_1d | 1 | 64 | 1.572093e-17 | -4.185020e-17 | 5.757e-17 | 9.044e-17 |
| linear_advection_1d | 2 | 16 | 1.734723e-17 | -7.285839e-17 | 9.021e-17 | 1.417e-16 |
| linear_advection_1d | 2 | 32 | 1.170938e-17 | 5.204170e-17 | 4.033e-17 | 6.335e-17 |
| linear_advection_1d | 2 | 64 | 6.830474e-18 | -5.995638e-17 | 6.679e-17 | 1.049e-16 |
| linear_advection_1d | 3 | 16 | 1.040834e-17 | 5.724587e-17 | 4.684e-17 | 7.357e-17 |
| linear_advection_1d | 3 | 32 | 8.239937e-18 | 1.218643e-16 | 1.136e-16 | 1.785e-16 |
| linear_advection_1d | 3 | 64 | -6.938894e-18 | -1.194791e-16 | 1.125e-16 | 1.768e-16 |
| linear_advection_2d | 1 | 16 | 1.000000e+00 | 1.000000e+00 | 1.110e-16 | 1.110e-16 |
| linear_advection_2d | 1 | 24 | 1.000000e+00 | 1.000000e+00 | 0.000e+00 | 0.000e+00 |
| linear_advection_2d | 2 | 16 | 1.000000e+00 | 1.000000e+00 | 0.000e+00 | 0.000e+00 |
| linear_advection_2d | 2 | 24 | 1.000000e+00 | 1.000000e+00 | 4.441e-16 | 4.441e-16 |
| burgers_1d | 2 | 32 | 2.000000e-01 | 2.000000e-01 | 1.915e-15 | 3.653e-15 |
| burgers_1d | 2 | 64 | 2.000000e-01 | 2.000000e-01 | 3.886e-15 | 7.405e-15 |

Worst scaled drift `7.405e-15`; coarse/fine ratio
`1.60` (round-off noise, not a systematic trend).

## 2. Euler boundary-flux budget (Sod, waves stay inside the domain)

analytic increments at `t = 0.2`:
mass `0.000e+00`,
momentum `0.180000`,
energy `0.000e+00`.

| flux | cells | steps | mass drift | energy drift | momentum increment | analytic | budget error |
|---|---:|---:|---:|---:|---:|---:|---:|
| hllc | 400 | 434 | 1.382e-14 | 1.276e-14 | 0.1800000000 | 0.1800000000 | 1.326e-14 |
| hllc | 800 | 872 | 2.408e-14 | 2.115e-14 | 0.1800000000 | 0.1800000000 | 2.791e-14 |
| hllc | 1600 | 1749 | 4.204e-14 | 4.021e-14 | 0.1800000000 | 0.1800000000 | 5.536e-14 |
| rusanov | 400 | 431 | 1.421e-14 | 1.373e-14 | 0.1800000000 | 0.1800000000 | 1.295e-14 |
| rusanov | 800 | 869 | 2.309e-14 | 2.374e-14 | 0.1800000000 | 0.1800000000 | 2.899e-14 |
| rusanov | 1600 | 1746 | 3.967e-14 | 3.940e-14 | 0.1800000000 | 0.1800000000 | 5.536e-14 |

Rival hypotheses scored against the measured momentum increment:
`transmissive_analytic` = 3.207e-14, `reflecting_wall` = 1.000e+00 —
only the transmissive prediction survives, so the criterion discriminates.

## 3. Rankine-Hugoniot shock speed

Exact star state: `rho* = 0.265574`,
`u* = 0.927453`, `p* = 0.303130`.
Shock speed from the mass jump `1.752156`, from the
momentum jump `1.752156`
(relative gap `3.80e-16`).  Sensor calibrated on the exact
profile: bias `1.18e-05`.

| cells | shock speed | RH analytic | relative error | positions x_s(t) |
|---:|---:|---:|---:|---|
| 400 | 1.752946 | 1.752156 | 4.509e-04 | 0.67750, 0.74764, 0.81776, 0.88786 |
| 800 | 1.752357 | 1.752156 | 1.146e-04 | 0.67640, 0.74650, 0.81659, 0.88669 |
| 1600 | 1.752251 | 1.752156 | 5.417e-05 | 0.67582, 0.74591, 0.81600, 0.88609 |

## 4. The discrete-shock tail ahead of the front

At `t = 0.2` the exact shock sits at
`x_s = 0.8504` and the exact
solution is *identical* to the uniform right state beyond it, so the window
`x in [0.90, 1.00]`
carries no physical signal; whatever is measured there is numerical.  Doubling the domain at the same `dx` (wall moved to
`x = 2`) leaves the value unchanged: the residual is an interior,
mesh-scale object, not a boundary defect.

| cells | dx | walled domain | doubled domain (same dx) | relative gap | tail width [cells] |
|---:|---:|---:|---:|---:|---:|
| 100 | 1.000e-02 | 4.065e-03 | 4.065e-03 | 4.3e-16 | 1.00 |
| 200 | 5.000e-03 | 4.833e-05 | 4.833e-05 | 0.0e+00 | 1.04 |
| 400 | 2.500e-03 | 4.796e-09 | 4.796e-09 | 0.0e+00 | 1.07 |
| 800 | 1.250e-03 | 5.269e-18 | 5.269e-18 | 0.0e+00 | 1.07 |

Fitted decay of the residual amplitude with `dx`: `16.17`;
worst relative gap between the two domains `4.3e-16`.

## 5. Fields vs the exact Riemann solution

| cells | steps | density rel. L2 | velocity rel. L2 | pressure rel. L2 |
|---:|---:|---:|---:|---:|
| 1600 | 1749 | 1.4752% | 3.5440% | 1.0693% |
| 3200 | 3502 | 1.1844% | 2.8869% | 0.7480% |

## 6. Time-integrator order (dt refinement, mesh fixed)

| integrator | dt | self-difference | successive orders | fitted order |
|---|---|---|---|---:|
| RK4 | linear advection P2, 40 cells | 2.000e-03, 1.000e-03 | 1.224e-09, 7.651e-11 | 4.00 | **4.000** |
| SSP-RK3 | Burgers P2, 64 cells | 6.241e-04, 3.116e-04 | 4.968e-09, 6.210e-10 | 2.99 | **2.994** |

## Figures

- `results/conservation/periodic_drift.png` — invariant drift per case with the criterion line.
- `results/conservation/euler_budget.png` — momentum integral vs the analytic line; mass/energy drift.
- `results/conservation/shock_speed.png` — measured shock path vs RH analytic; speed error vs cells.
- `results/conservation/shock_precursor.png` — residual amplitude vs `dx`, walled vs doubled domain.
- `results/conservation/sod_fields_vs_exact.png` — density/velocity/pressure vs the exact solution.
- `results/conservation/temporal_order.png` — dt self-convergence vs the theoretical slope.

## Artifact note

`study_stdout.log` is the raw console record of the generating run; it predates a
rename of `sod_profiles.png` -> `sod_fields_vs_exact.png` (the repository
`.gitignore` still matches local benchmark outputs named `sod_profiles.png`, so
the committed figure uses the longer name).  Content is unchanged.

## Scope and limitations

* Flat 1-D/2-D Cartesian meshes only, as in the rest of the release.
* The budget identity assumes the waves do not reach the ends before `t`;
  `study_euler_budget` uses `t = 0.2` where the fastest characteristic is at `x = 0.77`.
* Round-off level assertions require `ncells >= 400`; below that the
  discrete-shock tail of section 4 reaches the boundary cell and the outflow is
  genuine (see the S3 dev report).
* Entropy-production budget: **delivered** in `docs/entropy_verification.md`
  (`benchmarks/entropy_budget_1d.py`, `tests/test_entropy.py`) -- second law holds
  for every limiter setting, the budget converges to the Rankine-Hugoniot value
  but not yet to the 3 % gate on the meshes tested (honest FAIL recorded there).
