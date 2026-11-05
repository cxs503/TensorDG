# PR #75 CI red — diagnosis and repair (mission m4084, round 2)

## 1. What CI actually reported

PR #75 (`agent/m3714`, 2-D TVB/WENO limiter-order verification) was red on the
check run produced for head `4c4138f` (workflow run `41878529743`, job
`126744598725`, 2026-11-04). The failing step is step 6 `pytest -q`
(the later benchmark steps were skipped). Tail of the job log:

```
FAILED tests/test_ns_heat_conduction.py::test_shipped_wrapper_runs_at_its_documented_defaults - [XPASS(strict)]
FAILED tests/test_ns_heat_conduction.py::test_exact_steady_state_is_approximately_preserved - [XPASS(strict)]
FAILED tests/test_ns_heat_wall_closure.py::test_interior_energy_residual_decays_but_wall_rows_do_not - assert 1.1254006471155698 < 1.1
FAILED tests/test_ns_heat_wall_closure.py::test_conduction_entry_point_still_crashes_on_the_documented_default - Failed: DID NOT RAISE FloatingPointError
4 failed, 391 passed, 4 skipped, 7 xfailed in 9664.28s
```

## 2. Root cause: stale base, not a defect introduced by the branch

All four failures are *negative controls* of the NS heat-conduction wall
closure, i.e. tests that pinned a **known defect** and were therefore expected
to fail (or to raise):

* two `xfail(strict=True)` pins on the shipped wrapper / exact steady state —
  they XPASSed because the m3139 wall repair (one-sided DG wall derivative as
  the default + positivity filter + pressure guard) had already cured the
  symptom;
* `..._still_crashes_on_the_documented_default` — the `FloatingPointError` it
  asserted no longer fires, for the same reason;
* `..._wall_rows_do_not` — the bound was `1.1`, which sits 2.3 % *below* the
  measured 16/32 wall-residual ratio `1.1254`.

The PR head was built on `619167f0` (main, 2026-11-04 14:12). Every one of the
four has since been re-adjudicated on main (ratchet flips dated 2026-11-04,
`tests/test_ns_heat_conduction.py` / `tests/test_ns_heat_wall_closure.py`), the
last one by moving the bound to `1.2` (6 % margin over the measured 1.1254)
while keeping the physical claim (`|R_E|` stagnates at an O(1) floor, the two
`xfail(strict=True)` residual pins stay xfail).

So the red check is the **merge of an up-to-date-or-not branch with an older
main**: the branch itself only adds
`tests/test_limiter_2d_order.py`, `benchmarks/limiter_2d_order_study.py`,
`benchmarks/report_limiter_2d_order.py`, validation JSON, figures and a doc —
none of which touches the NS conduction path.

## 3. Repair

`git merge agent/m3714` into current `main` (`efd020d`, yielding merge commit
`5351b9d`) — the merge is clean, keeps every deliverable of PR #75, and picks
up the four ratchet flips that make the previously red pins consistent again.
No source file of the branch was modified.

## 4. Local verification on the merged tree

* `PYTHONPATH=. pytest -q -n 12 --timeout=600` on `5351b9d`:
  `485 passed, 4 skipped, 7 xfailed, 1 failed in 780s`.
  The single failure is **not an assertion failure**: the traceback ends in
  `Failed: Timeout (>600.0s) from pytest-timeout` inside
  `tensordg/euler_dg_2d.py:523 _positivity_filter`, i.e. the run of
  `test_shipped_wrapper_runs_at_its_documented_defaults`
  (`final_time=40`, documented defaults) was starved on a 128-core node that
  was simultaneously running other work. CI applies no per-test timeout, so the
  pin simply runs to completion there (it did in the pre-flip form: XPASS).
* The three other formerly red node IDs pass in that same run:
  `test_exact_steady_state_is_approximately_preserved`,
  `test_interior_energy_residual_decays_but_wall_rows_do_not`, and
  `test_conduction_entry_point_completes_and_stays_physical` (which replaced
  the deleted `..._still_crashes_...` pin).
* The branch's own deliverable, `tests/test_limiter_2d_order.py`, is unchanged
  and green (the inertness / negative-control / measure pins).

## 5. Limits of this evidence

CI status itself could not be re-read by the agent (the re-run for head
`4c4138f`, workflow run `42001594742`, was still queued behind the shared
runner pool). The claim supported here is: *the four red node IDs are
consistent on current main, reproduced locally on the merge commit*, not "CI is
green" — that requires the platform's pr-gate to observe the next run.
