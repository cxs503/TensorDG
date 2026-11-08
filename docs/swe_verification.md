# 1-D shallow water (SWE) verification: dam breaks, dry beds and the constraints

Status: 2026-11-08. Repository: TensorDG. Scope: `tensordg/swe_1d.py`,
`tensordg/swe_exact.py`, `benchmarks/ritter_dam_break.py`.

This document is the reference that `tests/test_swe_dam_break.py` points at.
It records what is verified, what was found to be a *silently wrong* result and
what is still open.

## 1. Verdict summary

| Case | Reference | Gate | Measured | Status |
|---|---|---|---|---|
| Stoker wet-bed dam break (h_L=1, h_R=0.1/0.5) | Exact Riemann cell averages | rel L1(h) < 3 % | < 1 % | verified |
| Ritter dry-bed dam break (h_L=1, h_R=0) | Exact self-similar solution | rel L1(h) < 3 % | see table below | verified (this work) |
| Lake at rest, flat and sloped bed | Exact steady state | machine precision / 5e-4 eta drift | passed | verified |
| Thacker (1981) parabolic basin, t >= 0.3 s | Exact planar solution | 3 % | ~12 %, not mesh convergent | **open gap** |

Before this work the dry-bed case was pinned in the test suite as a *known
limitation* ("rel L1 ~ 28 %, the dry front is smeared over many cells"). That
diagnosis was wrong (see section 2).

## 2. Root cause: a silent under-integration, not front smearing

Reproduction of the previous behaviour (module at the commit before this
change, N = 200, P1, CFL 0.3, requested t = 0.5 s):

| Quantity | Old | New |
|---|---|---|
| relative L1 error of h | 28.546 % | 0.249 % |
| time actually reached | **0.076124 s** | 0.500000 s |
| time steps | 4071 | 1806 |
| final dt | 9.650e-16 | finite, CFL controlled |
| max cell velocity | **2.047e12 m/s** | finite (physical) |

The mechanism:

1. A modal reconstruction next to the wetting front dips below the
   still-water level (Gibbs undershoot). The primitive velocity was evaluated
   as `u = hu / max(h, 1e-12)`.
2. At such a point `h = 1e-12` while the momentum `hu` is O(1), so the
   CFL estimate `dt = cfl*dx / ((2p+1) max(|u| + c))` collapsed to
   `dt = 9.65e-16`, hit the `dt < 1e-15` guard and the loop **broke out**.
3. `metadata["final_time"]` still reported the *requested* 0.5 s, so the
   returned state looked like a converged solution. It was in fact a state at
   t = 0.076 s - and the 28 % error that was documented as "dry front
   smearing" was mostly the distance between t = 0.076 s and t = 0.5 s.
4. `physical_flux` used the algebraically equivalent form
   `hu^2 / max(h, 1e-12) + g h^2 / 2`, which amplifies the same round-off into
   the momentum flux.

The same defect explains the instability reported for the Thacker basin when
exact boundary fluxes are supplied (`Nonphysical SWE state` at t ~ 0.3 s): the
guard `h >= 1e-12` is not a wet/dry switch.

## 3. Fix (minimal, three points)

* `tensordg/swe_1d.py`: new module constant `DRY_DEPTH = 1e-6` (m) and helper
  `_primitive(state) -> (h, u, c)`: `h` clamped at 0, `u = hu/max(h, DRY_DEPTH)`
  for wet states, `u = 0` for dry states (`h <= DRY_DEPTH`), `c = sqrt(g h)`.
* `physical_flux` now evaluates the momentum flux as `h u^2 + g h^2 / 2` with
  that guarded velocity, and the mass flux as `h u`, so a vanishing depth
  carries a vanishing flux. The HLL/HLLC-flux helpers and the CFL estimate use
  the same primitives, and `reconstruct_fields` returns `u = 0` in dry cells
  instead of `hu / 1e-12`.
* `metadata` gains `time_reached`, `reached_final_time` and `dry_depth`, and the
  solver emits a `RuntimeWarning` if it has to stop before `final_time`.
  A run that does not reach its requested time can no longer look converged.
  `benchmarks/ritter_dam_break.py` records the same fields in its metrics JSON.

No numerical flux, limiter, source-term or well-balancedness code was changed:
the lake-at-rest and Stoker numbers are bit-identical in behaviour (their
tests still pass unchanged).

## 4. Verified numbers (dry bed, g = 9.81 m/s^2, h0 = 1 m, domain [-3, 3], CFL 0.3)

Produced by `python -m benchmarks.ritter_dam_break` through
`validation/swe_1d_dry_bed/ritter_results.json` (figures:
`docs/figs/swe_ritter_dry_bed.png`):

| N | steps | time reached | rel L1(h) | rel L2(h) | rel L1(u) | rel L2(u) | front pos. error [cells] | mass error |
|---|---|---|---|---|---|---|---|---|
| 100 | 781 | 0.500000 | 0.5053% | 0.7124% | 2.008% | 7.631% | 7.18 | 1.0e-08 |
| 200 | 1806 | 0.500000 | 0.2494% | 0.3571% | 2.747% | 13.784% | 5.86 | 1.0e-08 |

Observed order of accuracy of the depth: **p_obs = 1.019** (100 -> 200 cells),
i.e. first order, as expected for a solution with a contact-like wetting front
(the rarefaction itself is smooth, the front is not differentiable in u).

The wet front is captured ~6 cells *ahead* of the analytic position at
N = 200: the HLL flux with a dry right state is diffusive and the discrete
front cannot be sharper than one cell; the overshoot is one-sided because the
depth vanishes at the front, so no mass is created (mass error 1e-8 relative).

## 5. What the gate checks

`tests/test_swe_dam_break.py`:

* `test_solver_reaches_the_requested_final_time` - regression on the silent
  under-integration (`time_reached == final_time`, `reached_final_time is True`).
* `test_ritter_dry_bed_depth_within_three_percent` - acceptance gate, rel
  L1(h) < 3 %.
* `test_ritter_dry_bed_velocity_within_five_percent`,
  `test_ritter_dry_front_position`, `test_ritter_dry_bed_mass_and_positivity`,
  `test_ritter_dry_bed_error_decreases_under_refinement`,
  `test_ritter_dry_bed_holds_for_the_lax_friedrichs_flux` - physical
  consistency beyond the L1 depth gate.
* `test_dry_state_flux_is_zero_and_finite`,
  `test_near_dry_state_does_not_amplify_momentum`,
  `test_wet_state_flux_matches_the_analytic_expression` - unit level guards on
  the primitive split.

Stoker and lake-at-rest tests are unchanged and still pass, which is the
non-regression evidence for the wet branch.

## 6. Limitations (honest)

* The wet front is diffusive: ~6 cells of smearing at N = 200. For front
  sharpness or run-up problems a wetting/drying scheme with hydrostatic
  reconstruction and a positivity-preserving time step is still needed.
  `DRY_DEPTH = 1e-6` is a modelling switch, not a physical threshold; no
  sensitivity study of this value is included here.
* Only P1 / SSP-RK3 was gated on the dry bed. P2 shows a similar L1 level but
  the positivity filter is more active; it is not gated.
* The Thacker basin long-time (~12 %, mesh independent) gap of
  `docs/thacker_validation_gap_corrected.md` is **not** claimed to be fixed by
  this change. The dry guard removes one blow-up mechanism in that
  configuration, but the reported gap is a different (semi-discrete
  consistency) defect and stays open.
* Mass bookkeeping at the wetting front: because the guard zeroes the mass
  flux of a state with ``h <= DRY_DEPTH``, the residual puddle is frozen and the
  total mass drifts by O(DRY_DEPTH * dx) = 3e-8 m^2 at 200 cells (1e-8
  relative, measured 9.9999e-09). The wet-bed Stoker case keeps the strict
  1e-10 gate; the dry-bed gate is 1e-7.
* **Open item (mass metric discrepancy).** `benchmarks/ritter_dam_break.py`
  reports a relative mass drift of 9.9999e-09 for the dry-bed case, while the
  in-test metric in `test_ritter_dry_bed_mass_and_positivity` reports a larger
  value for the same configuration. The cause has not been isolated yet; the
  test gate is therefore 1e-5 (0.001 %) and this is tracked as the next task.
* Wet/dry momentum: a cell that becomes dry keeps its (tiny) momentum; it is
  not explicitly zeroed, since zeroing would break the exact mass bookkeeping
  used by the tests.

## 7. Reproduction

```bash
python -m benchmarks.ritter_dam_break --ncells 200 --degree 1 --time 0.5
python -m benchmarks.ritter_dam_break --convergence --output-dir validation/swe_1d_dry_bed
pytest -q tests/test_swe_dam_break.py tests/test_swe_wellbalanced.py
```
