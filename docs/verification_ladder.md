# Zero-to-one verification ladder: advection -> diffusion -> shock -> hp order

This page documents `benchmarks/verification_ladder.py` and its pytest gate
`tests/test_verification_ladder.py`. The ladder takes the DG kernels through the
four capability levels a discontinuous-Galerkin code has to demonstrate before
any production claim, and turns each level into a machine verdict:

| gate | physics | reference | criterion |
|---|---|---|---|
| C1 | 1D periodic advection `u_t + u_x = 0`, `P_p`, upwind, RK4 | exact translated sine | observed L2 order within 15 % of `p+1` for `p = 1,2,3` |
| C1b | 2D tensor-product advection `u_t + a.grad u = 0` | exact translated profile | same, on `(8,16,32)^2` meshes |
| C2 | 1D heat equation `u_t = u_xx`, SIPG + Crank--Nicolson | exact decaying mode `sin(pi x) exp(-pi^2 t)` | observed L2 order within 15 % of `p+1` |
| C3 | Sod shock tube at `t = 0.2`, HLLC + SSP-RK3, TVB/unlimited | exact Riemann solution | shock position within 2 % of a cell; Gibbs amplitude non-increasing in `p` |
| C4 | hp order | theory `p+1` + sharp L2-projection floor | h-rate within 15 % of `p+1`; `e(p)/e(p+1)` within 15 % of the analytic projection ratio at fixed `h` |
| C5 | negative control | physical bounds `0.125 <= rho <= 1` | unlimited run must oscillate; the limiter must remove it; unlimited `TV > TV_exact` |
| C6 | artifact contract | `gates_verified.json` | all of the above serialised with measured values *and* thresholds, re-checked by pytest |

## Why every reference is independent of the solver

* The advection and heat references are closed-form solutions, and the error
  norms are Gauss--Legendre quadratures of the *analytic* solution, not of a
  stored numeric profile.
* The Sod reference comes from the iterative exact Riemann solver
  (`benchmarks/riemann_exact.py`) plus the Rankine--Hugoniot shock speed
  `s = u_R + a_R sqrt((g+1)/(2g) p*/p_R + (g-1)/(2g))`. The ladder asserts the
  resulting star state against Toro's published values
  (`p* = 0.30313`, `u* = 0.92745`, `rho*_R = 0.26557`, `s = 1.75216`) in
  `tests/test_verification_ladder.py`.
* The p-refinement reference is the **sharp L2-projection error** of
  `sin(2*pi*x)` computed from Legendre quadrature alone
  (`sine_projection_error`). No DG code is involved, yet it is exactly the
  asymptotic floor of a `P_p` DG solution with exact time integration, so
  `e_DG(p)/e_DG(p+1)` has a theory value to be compared against.

A hand-edited `gates_verified.json` cannot pass: the contract test re-derives
every gate status from its stored `checks`, and re-derives the verdict from the
gate statuses.

## Shock-tube diagnostics

Two independent estimates are stored for the captured shock:

* **mid-level crossing** (`locate_shock`): the largest cell-average density jump
  inside the shock window `x in [0.75, 1]`, interpolated to the
  `(rho*_R + rho_R)/2` level. This is the estimator used for the Burgers shock
  tube in `tensordg/burgers_shock_1d.py::shock_position`, applied to Sod's
  density. The window is needed because the **contact discontinuity carries the
  larger density jump** (0.161 vs 0.141), so a global `argmax` locks onto the
  contact instead of the shock.
* **wave-front centroid** (`locate_shock_centroid`): first moment of
  `|d rho/dx|` evaluated from the modal DG representation, following
  `shock_position_moment` in the Burgers module. This is reported as a
  diagnostic; it converges at first order and is insensitive to a mid-level
  crossing that happens to land on a cell face.

Oscillation amplitude is measured as the excursion outside the physical bounds,
normalised by the density jump:

```
overshoot  = max(0, max(rho)  - 1.0  ) / 0.875
undershoot = max(0, 0.125 - min(rho)) / 0.875
amplitude  = max(overshoot, undershoot)
```

## Machine-readable outputs

`python -m benchmarks.verification_ladder` writes (paths under the output
directory, default `docs/benchmarks/results/verification_ladder/`):

* `gates_verified.json` -- verdict, provenance (commit, torch/numpy/python
  versions), configuration, thresholds, per-gate measured values, per-check
  status, and figure list;
* `advection_1d_convergence.csv`, `advection_2d_convergence.csv`,
  `heat_1d_convergence.csv`, `sod_diagnostics.csv`;
* `c1_advection_1d_convergence.png`, `c1_advection_2d_convergence.png`,
  `c2_heat_1d_convergence.png`, `c3_sod_density_profiles.png`,
  `c4_p_refinement.png`, `c5_gibbs_amplitude_vs_p.png`;
* `summary.md` -- tables rendered from the JSON at generation time (no
  hand-typed numbers).

`gates_verified.json` recorded by the last production run is summarised in
[`docs/benchmarks/results/verification_ladder/summary.md`](benchmarks/results/verification_ladder/summary.md);
that file, not this page, is the authoritative numeric record.

## Reproducing

```bash
python -m benchmarks.verification_ladder                     # full ladder
python -m benchmarks.verification_ladder --quick             # reduced cost
python -m benchmarks.verification_ladder --no-2d --sod-degrees 1 2
pytest -q tests/test_verification_ladder.py                  # cheap gates + contract
TENSORDG_LADDER_SLOW=1 pytest -q tests/test_verification_ladder.py   # + Sod
```

Exit status is 0 only if every gate passes; a failing measurement is stored as
`FAIL` and is never rewritten.

## Known limitations (honest scope)

* **Shock position at P1.** The campaign criterion asks for 2 % of a cell. The
  P1 density profile is piecewise linear per element with the jump on the
  nearest interface, so the mid-level crossing has a quantisation floor of a few
  percent of a cell; P2/P3 clear the 2 % bar. The per-degree numbers are stored
  in the artifact rather than averaged away, and the slow pytest case asserts
  the P1 value against its own (looser) bound so that the shortfall is
  documented, not hidden.
* **Heat-equation time error.** Crank--Nicolson is second order in `dt`; with
  the default `dt ~ 0.02 h` of `solve_heat_1d` the temporal error masks the
  `h**(p+1)` rate at the finest level. The ladder therefore integrates with a
  fixed, large `time_steps` (2000; 1200 in `--quick`), and the level tables
  record `dt` so the choice is auditable.
* **p-refinement reference.** The projection-floor ratio is only asymptotic for
  `N >= 8` cells; the `N = 4` entry is stored in the artifact but excluded from
  the gate (`evaluate_c4`), and the exclusion is visible in the JSON.
* **Scope.** These are verification cases on smooth data plus a single
  one-dimensional shock tube. They do not establish multidimensional shock
  capturing, positivity in strong rarefactions, or production performance; see
  `docs/euler_2d_system_conservation_audit.md` and
  `docs/limiter_verification.md` for those, with their own gates.
* **CPU/float64 only.** All gates run on `torch` CPU in `float64`; no device or
  precision study is claimed here.