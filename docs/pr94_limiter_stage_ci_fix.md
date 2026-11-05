# PR #94 CI red: `limiter_stage` in the 2-D vortex driver

Mission m4355, round 1. Branch `agent/m3627` (PR #94) was red on GitHub Actions.
This note records the local reproduction, the root cause and the fix.

## 1. Reproduction (local, before the fix)

PR #94 head contains exactly two files (main + this diff):

* `tensordg/euler_dg_2d.py`: new `limiter_stage` option of `solve_euler_dg_2d`
  (`"every"` = limit after every SSP-RK3 sub-step, `"final"` = limit only the
  end-of-step state), echoed in the metadata;
* `tests/test_euler_dg_2d_limiter_stage.py`: new regression file.

Running the new file on the PR head reproduces the CI failure exactly
(`artifacts/pr94_fix/red_before_fix.log`):

```
FAILED tests/test_euler_dg_2d_limiter_stage.py::test_limiter_stage_inert_without_limiter
FAILED tests/test_euler_dg_2d_limiter_stage.py::test_limiter_stage_preserves_design_order_on_vortex
FAILED tests/test_euler_dg_2d_limiter_stage.py::test_limiter_stage_recorded_in_metadata
3 failed, 1 passed
```

i.e. three `test=failure` checks, matching the GitHub status of PR #94.

## 2. Root cause (two independent defects)

**(a) The option never reaches the vortex driver.**
`test_limiter_stage_*` drive the benchmark through `solve_euler_dg_2d_vortex`,
but the PR only added `limiter_stage` to `solve_euler_dg_2d`. Every call with
the new keyword therefore raised

```
TypeError: solve_euler_dg_2d_vortex() got an unexpected keyword argument 'limiter_stage'
```

which is what the three failing tests hit.

**(b) The order assertion contradicts the physics of the chosen case.**
`test_limiter_stage_preserves_design_order_on_vortex` additionally asserted
`EOC("every") < EOC("final")` on the smooth vortex with `tvb_constant = 10`.
That constant is scale consistent with the flow, so the TVB detector never
flags a smooth cell: the limiter is *inert* and both policies are bit-identical
(`artifacts/pr94_fix/sweep.json`, section `sweep`). The strict inequality can
therefore never hold; the historic `"every"` defect (re-projecting flagged
smooth cells at every sub-step) simply cannot be exhibited at `c = 10`.

## 3. Fix (two files, minimal)

1. `tensordg/euler_dg_2d.py`: `solve_euler_dg_2d_vortex` accepts
   `limiter_stage="every"` (documented) and forwards it to
   `solve_euler_dg_2d`. No behaviour change for existing callers — the default
   is the historic policy and the new keyword sits after `tvb_constant`, so all
   positional-argument call sites (all of which are keyword anyway) are
   unaffected.
2. `tests/test_euler_dg_2d_limiter_stage.py`: the accuracy gate is stated per
   policy at the scale-consistent constant (measured EOC 3.042 > 2.7, design
   order `p + 1 = 3` preserved), and the "does the flag reach the sub-steps?"
   claim is moved to the configuration that can actually show it: strict TVD
   `c = 0`, where the two policies differ by ~5.6e-3 relative
   (`artifacts/pr94_fix/sweep.json`, `T = 0.05`, `N = 8`).

## 4. Measurements (P2 isentropic vortex, density relative L2 error)

| configuration | N = 8 | N = 16 | EOC |
|---|---|---|---|
| c = 10, `"every"`, T = 1 | 3.275e-3 | 3.976e-4 | 3.042 |
| c = 10, `"final"`, T = 1 | 3.275e-3 | 3.976e-4 | 3.042 |
| c = 0, `"every"`, T = 1 | 4.267e-2 | 2.027e-2 | 1.074 |
| c = 0, `"final"`, T = 1 | 4.266e-2 | 2.039e-2 | 1.065 |
| c = 0, `"every"`, N = 8, T = 0.05 | 3.1607e-2 | — | — |
| c = 0, `"final"`, N = 8, T = 0.05 | 3.1430e-2 | — | — |

Without a limiter the two policies are bitwise identical (degree 1 and 2), as
they must be. Raw data: `artifacts/pr94_fix/sweep.json`
(`python artifacts/pr94_fix/measure.py`, `torch 2.7.1`).

Accuracy judgement: the fixed test keeps the only quantitative gate for which a
reference exists in this repo, namely the design order. `"final"` reproduces
`p + 1 = 3` to 1.4 % (3.042 vs 3.0), well inside the 3 % gate; no experimental
or literature reference is involved in this configuration, so no further
error statement is possible. The strict-TVD runs are deliberately *not* an
accuracy claim (order collapses to ~1 for both policies, as expected once every
curved cell is clipped).

## 5. Verification

* `python -m pytest -q tests/test_euler_dg_2d_limiter_stage.py` →
  `5 passed in 131.77s` (`artifacts/pr94_fix/green_after_fix.log`), the red log
  above is the same file before the fix;
* blast radius (Euler/limiter/design-order/vortex/anisotropic/temporal-order
  test modules) → `artifacts/pr94_fix/blast_radius.log`;
* the shipped default path is untouched: `limiter_stage="every"` reproduces
  the previous code path bit-for-bit, since `_apply_limiter` +
  `_positivity_filter` are chained in exactly the historic order.

## 6. Scope

Only the two PR files plus evidence under `artifacts/pr94_fix/` and this note
are touched. No solver algorithm, flux, limiter or threshold was changed; the
red was a plumbing defect plus a mis-specified assertion.