# Ritter dry-bed mass metric: the "9.9999e-09 discrepancy" resolved

Status: 2026-11-08. Repo: TensorDG. Scope: `benchmarks/ritter_dam_break.py`,
`tests/test_swe_dam_break.py`, `tensordg/swe_1d.py` (read-only here).
Supersedes the "Open item (mass metric discrepancy)" bullet of
`docs/swe_verification.md` §6.

## 1. The open item

`docs/swe_verification.md` §6 recorded that `benchmarks/ritter_dam_break.py`
reported a relative mass drift of 9.9999e-09 for the Ritter dry-bed case while
the in-test metric of `test_ritter_dry_bed_mass_and_positivity` reported a
larger value for "the same configuration", cause unisolated, test gate relaxed
to 1e-5.

## 2. Finding 1 - the benchmark number was a metric bug, not a solver drift

`run_ritter` seeds the dry side with `dry_tol = 1e-8` water instead of a hard
zero, and the denominator of the reported drift was the *analytic* mass

```
mass_init = h0 * |x_left| = 1.0 * 5.0 = 5.0 m^2
```

which ignores that seed. The seed contributes exactly

```
dry_tol * (x_right - max(x_left, 0)) = 1e-8 * 5.0 = 5e-8 m^2
```

so `5e-8 / 5.0 = 1.0e-08` appeared as "mass error" independently of the
integration. Number of the old artifact (`validation/swe_1d_dry_bed/ritter_results.json`,
pre-fix code):

| case | mass_num [m^2] | mass_init (analytic) | reported drift |
|---|---|---|---|
| P1_N100 | 5.000000049999841 | 5.0 | 9.999968e-09 |
| P1_N200 | 5.000000049999743 | 5.0 | 9.999949e-09 |

`mass_num - 5e-8 = 5.0000000000` to 12 digits in both rows: the solver moved no
mass at all. The fix integrates the *discrete* initial state with the solver's
own Gauss-Legendre rule (`_discrete_initial_mass`) and reports

| case | mass_init (discrete) | mass_seed | consistent drift | legacy drift |
|---|---|---|---|---|
| P1_N100 | 5.000000049999999 | 5.000e-08 | 3.16e-14 | 1.0e-08 |
| P1_N200 | 5.000000050000000 | 5.000e-08 | 5.13e-14 | 1.0e-08 |

Both metrics are kept (`mass_error`, `mass_error_legacy`) so the historical
number stays traceable.

## 3. Finding 2 - the in-test number is boundary outflow of an O(DRY_DEPTH) floor

The in-test metric uses `DOMAIN = (-3, 3)`. The exact Ritter front travels at
`2 c0 = 3.132 m/s`, so it reaches the transmissive right wall at
`t = 3 / 3.132 = 0.479 s`, i.e. *before* the default `T_FINAL = 0.5 s`.
Measured relative drift, P1, CFL 0.3, zero-initialised dry side:

| domain | N | t | relative drift | comment |
|---|---|---|---|---|
| (-3, 3) | 200 | 0.3 | 7.93e-14 | front at x = 1.88, walls untouched |
| (-3, 3) | 200 | 0.5 | 3.25e-07 | front at x = 3.132 > wall |
| (-3.5, 3.5) | 200 | 0.5 | 1.75e-07 | wall 0.37 m beyond the front |
| (-5, 5) | 200 | 0.5 | 8.17e-14 | wall 1.87 m beyond the front |
| (-3, 3) | 100 | 0.3 | 1.67e-07 | coarse grid, floor reaches the wall |
| (-5, 5) | 100 | 0.3 | 2.77e-14 | coarse grid, floor below DRY_DEPTH at the wall |

The mechanism is *not* a loss of the frozen puddle. The DG mass equation is
`d/dt (cell mean) = (F_L - F_R)/dx` for the mean mode, so the interior balance
telescopes exactly and the only non-cancelling terms are the two boundary
fluxes. What leaves through the wall is the scheme's numerical water floor
ahead of the front: at N = 100, `(-3, 3)`, t = 0.3 the final cell means in the
"dry" region behind the wall are `h = 8.8e-07 ... 9.0e-07` (the front is at
x = 1.88, the exact solution there is 0), carried by `u = hu / h ~ 0.1 m/s`,
and the *trace* value used for the boundary flux overshoots to 3.7e-05 during
the run. That floor sits at the DRY_DEPTH = 1e-6 modelling switch. It is a
wetting/drying artefact of the near-dry velocity `u = hu / h`, not a
conservation error of the flux form - and it is exactly why the drift
disappears when the wall is far enough away (row 4, row 6).

Sensitivity to the switch (N = 50, `(-3, 3)`, t = 0.3, monkeypatched
`tensordg.swe_1d.DRY_DEPTH`): 1e-8 -> the run aborts with
`Nonphysical SWE state`; 1e-6 -> 6.66e-07; 1e-5 -> 3.03e-12; 1e-4 -> 3.03e-12.
The drift is therefore controlled by the same switch that keeps the run
stable, which is consistent with the floor explanation and confirms that the
number is not a fixed property of the scheme.

## 4. What is pinned now

* `test_ritter_benchmark_mass_metric_counts_the_seeded_initial_mass` - pins the
  seeded mass (5e-8), the consistent drift (< 1e-12) and the legacy 1e-8
  artefact on the fast N = 100 benchmark configuration.
* `test_ritter_dry_bed_mass_is_machine_conservative_when_the_front_stays_inside`
  - `(-5, 5)`, N = 200, t = 0.5: drift < 1e-12 (measured 8.17e-14).
* `test_ritter_dry_bed_mass_and_positivity` - gate tightened from 1e-5 to 1e-6
  (measured 3.25e-07) with the boundary-outflow explanation in the docstring.

## 5. Remaining limitation (honest)

The O(DRY_DEPTH) floor ahead of the wet front is *not* fixed here: removing it
needs the wetting/drying closure (hydrostatic reconstruction + a
positivity-preserving time step) that `docs/swe_verification.md` already lists
as the next modelling step. Until then mass leaves the domain through any
transmissive wall that the floor can reach, at the 1e-7 relative level for
N <= 100, and the reported benchmark drift is meaningful only for
configurations where the floor stays below DRY_DEPTH at both walls.

## 6. Reproduction

```bash
python -m benchmarks.ritter_dam_break --ncells 200 --time 0.5      # corrected metric
python -m pytest -q tests/test_swe_dam_break.py -k mass
```
