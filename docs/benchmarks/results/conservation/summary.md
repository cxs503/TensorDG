# Conservation-law and temporal-order verification (TensorDG, S3 round 2)

- generated: `2026-11-02T16:57:57Z` (full mode)
- criteria passed: **17/17**
- overall: **PASS**

## Criteria

| criterion | value | threshold | result | note |
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
| `precursor_amplitude_fine_mesh` | 4.796108e-09 | <= 1.000e-06 | PASS | residual |rho*u| in an untouched window, meshes >= 400 cells |
| `precursor_domain_independence` | 4.267933e-16 | <= 1.000e-06 | PASS | same window on a doubled domain at the same dx (rules out the wall) |
| `precursor_decay_rate` | 1.616622e+01 | >= 2.000e+00 | PASS | fitted decay rate of the residual amplitude in h |
| `precursor_tail_width` | 1.066706e+00 | <= 5.000e+00 | PASS | exponential tail width in cells (O(1) cells = discrete-shock artefact) |
| `sod_field_error_vs_exact` | 2.886883e-02 | <= 3.000e-02 | PASS | worst relative L2 of density/velocity/pressure vs the exact Riemann solution, at the finest computed resolution |
| `rk4_temporal_order` | 3.999994e+00 | >= 3.500e+00 | PASS | measured RK4 order on a fixed mesh (theory 4) |
| `ssp_rk3_temporal_order` | 2.993931e+00 | >= 2.600e+00 | PASS | measured SSP-RK3 order on a fixed mesh (theory 3) |

## Periodic discrete invariants

| case | degree | cells | mass(0) | mass(T) | scaled drift |
|---|---:|---:|---:|---:|---:|
| linear_advection_1d | 1 | 16 | 0.000000e+00 | 9.194034e-17 | 1.452e-16 |
| linear_advection_1d | 1 | 32 | -6.071532e-18 | 1.847481e-16 | 2.999e-16 |
| linear_advection_1d | 1 | 64 | 1.572093e-17 | -4.185020e-17 | 9.044e-17 |
| linear_advection_1d | 2 | 16 | 1.734723e-17 | -7.285839e-17 | 1.417e-16 |
| linear_advection_1d | 2 | 32 | 1.170938e-17 | 5.204170e-17 | 6.335e-17 |
| linear_advection_1d | 2 | 64 | 6.830474e-18 | -5.995638e-17 | 1.049e-16 |
| linear_advection_1d | 3 | 16 | 1.040834e-17 | 5.724587e-17 | 7.357e-17 |
| linear_advection_1d | 3 | 32 | 8.239937e-18 | 1.218643e-16 | 1.785e-16 |
| linear_advection_1d | 3 | 64 | -6.938894e-18 | -1.194791e-16 | 1.768e-16 |
| linear_advection_2d | 1 | 16 | 1.000000e+00 | 1.000000e+00 | 1.110e-16 |
| linear_advection_2d | 1 | 24 | 1.000000e+00 | 1.000000e+00 | 0.000e+00 |
| linear_advection_2d | 2 | 16 | 1.000000e+00 | 1.000000e+00 | 0.000e+00 |
| linear_advection_2d | 2 | 24 | 1.000000e+00 | 1.000000e+00 | 4.441e-16 |
| burgers_1d | 2 | 32 | 2.000000e-01 | 2.000000e-01 | 3.653e-15 |
| burgers_1d | 2 | 64 | 2.000000e-01 | 2.000000e-01 | 7.405e-15 |

## Euler boundary-flux budget (Sod, waves stay inside the domain)

analytic increments at t = 0.2: mass 0.000000e+00, momentum 1.800000e-01, energy 0.000000e+00

| flux | cells | steps | mass drift | energy drift | momentum increment | analytic | budget error |
|---|---:|---:|---:|---:|---:|---:|---:|
| hllc | 400 | 434 | 1.382e-14 | 1.276e-14 | 0.180000000 | 0.180000000 | 1.326e-14 |
| hllc | 800 | 872 | 2.408e-14 | 2.115e-14 | 0.180000000 | 0.180000000 | 2.791e-14 |
| hllc | 1600 | 1749 | 4.204e-14 | 4.021e-14 | 0.180000000 | 0.180000000 | 5.536e-14 |
| rusanov | 400 | 431 | 1.421e-14 | 1.373e-14 | 0.180000000 | 0.180000000 | 1.295e-14 |
| rusanov | 800 | 869 | 2.309e-14 | 2.374e-14 | 0.180000000 | 0.180000000 | 2.899e-14 |
| rusanov | 1600 | 1746 | 3.967e-14 | 3.940e-14 | 0.180000000 | 0.180000000 | 5.536e-14 |

Rival hypotheses scored against the measured momentum increment: `transmissive_analytic` = 3.207e-14, `reflecting_wall` = 1.000e+00 (only the transmissive prediction survives).

## Rankine-Hugoniot shock speed

star state: rho* = 0.265574, u* = 0.927453, p* = 0.303130; W_RH = 1.752156 (momentum-jump cross-check 1.752156)

sensor level = 0.195287; sensor bias on the exact profile = 1.176e-05

| cells | shock speed | relative error vs RH | shock positions |
|---:|---:|---:|---|
| 400 | 1.752946 | 4.509e-04 | 0.677500, 0.747644, 0.817755, 0.887855 |
| 800 | 1.752357 | 1.146e-04 | 0.676403, 0.746499, 0.816593, 0.886686 |
| 1600 | 1.752251 | 5.417e-05 | 0.675820, 0.745914, 0.816003, 0.886090 |

## Discrete-shock tail ahead of the front

window x in [0.90, 1.00] at t = 0.2: every characteristic is still at x < 0.77, so the
window holds no physical signal; the doubled domain (same dx, wall at x = 2) isolates wall effects.

| cells | dx | walled domain | doubled domain (same dx) | relative gap | tail width [cells] |
|---:|---:|---:|---:|---:|---:|
| 100 | 1.000e-02 | 4.065e-03 | 4.065e-03 | 4.3e-16 | 1.00 |
| 200 | 5.000e-03 | 4.833e-05 | 4.833e-05 | 0.0e+00 | 1.04 |
| 400 | 2.500e-03 | 4.796e-09 | 4.796e-09 | 0.0e+00 | 1.07 |
| 800 | 1.250e-03 | 5.269e-18 | 5.269e-18 | 0.0e+00 | 1.07 |

fitted decay of the residual amplitude with dx: **16.17**; the wall makes no difference (relative gap <= 4.3e-16), so the residual is an interior, mesh-scale object, not a boundary defect.

## Time-integrator order (dt refinement, mesh fixed)

| integrator | dt | self-difference | successive orders | fitted order |
|---|---|---|---|---:|
| RK4 | linear advection P2, 40 cells | 2.000e-03, 1.000e-03 | 1.224e-09, 7.651e-11 | 4.00 | **4.000** |
| SSP-RK3 | Burgers P2, 64 cells | 6.241e-04, 3.116e-04 | 4.968e-09, 6.210e-10 | 2.99 | **2.994** |

## Reproduce

```bash
python tests/conservation_study.py          # regenerate metrics/CSV/PNG/summary
python -m pytest tests/test_conservation.py -v
```

Figures: `periodic_drift.png`, `euler_budget.png`, `shock_speed.png`, `sod_profiles.png`, `shock_precursor.png`, `temporal_order.png`
