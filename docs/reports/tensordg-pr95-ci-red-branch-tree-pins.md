# PR #95 (`agent/m3625`) CI red → green: branch tree vs. stale pins

## 1. What CI actually reported

The `test` job of `.github/workflows/tests.yml` is triggered on **both** `pull_request`
and `push`. The two events test different trees:

| event | tree tested | run on head `df8f38e` | conclusion |
|---|---|---|---|
| `pull_request` | merge of `df8f38e` with `main` | 42001052384 (09:16Z), 42017868242 (11:18Z) | **success** |
| `push` | `df8f38e` **alone** (branch tree) | 41849042800 (2026-11-04 14:35Z) | **failure** |

The recorded red comes from the `push` run. Its log (fetched read-only from the
Actions API) ends with

```
FAILED tests/test_ns_heat_conduction.py::test_shipped_wrapper_runs_at_its_documented_defaults
       - [XPASS(strict)] known defect: ... raises FloatingPointError at its documented defaults
FAILED tests/test_ns_heat_conduction.py::test_exact_steady_state_is_approximately_preserved
       - [XPASS(strict)] known defect: even starting on the exact steady state the run collapses
FAILED tests/test_ns_heat_wall_closure.py::test_interior_energy_residual_decays_but_wall_rows_do_not
       - assert 1.1254006470996836 < 1.1
FAILED tests/test_ns_heat_wall_closure.py::test_conduction_entry_point_still_crashes_on_the_documented_default
       - Failed: DID NOT RAISE FloatingPointError
4 failed, 390 passed, 4 skipped, 8 xfailed, 3 warnings in 8421.48s
```

## 2. Root cause: the pins, not the PR's change

`agent/m3625` is based on `619167f` ("Merge #53") and carries the *pre-repair* pins of
two heat-conduction test modules, while the solver repair it is measured against
(m3139 one-sided DG wall derivative + positivity guards) is shared by both trees. On
`main` the 2026-11-04 ratchet re-adjudication replaced those pins:

* `tests/test_ns_heat_conduction.py`: the two `xfail(strict=True)` markers on
  `test_shipped_wrapper_runs_at_its_documented_defaults` and
  `test_exact_steady_state_is_approximately_preserved` were **flipped to positive
  assertions** (the repair makes them pass, so `strict=True` turned the branch's
  copy into failures: *XPASS*).
* `tests/test_ns_heat_wall_closure.py`: the numeric bound was replaced by the
  measured one — `wall_16_32 < 1.2`, `wall_32_64 < 1.1`, monotone stagnation — and
  `test_conduction_entry_point_still_crashes_on_the_documented_default` (which
  asserted a `FloatingPointError` that the repair removed) was retired and replaced
  by accuracy assertions.

The PR's own six files never touch those modules, so the red is pure stale-base
drift of the pins. The PR's code change itself is a purely additive knob:
`tensordg/euler_dg_2d.py` gains `dt=None` (default reproduces the adaptive CFL
path bit-for-bit) plus two metadata keys, and no call site in `tests/`,
`benchmarks/` or `tensordg/` passes more than one positional argument to
`solve_euler_dg_2d(..)`, so parameter order cannot shift any existing result.

## 3. Fix

`git merge origin/main` into the branch (`4ae1597` at the time of writing) — no code
edit, no test edit, conflict-free because `main` never touched
`tensordg/euler_dg_2d.py`, `benchmarks/euler_dg_2d_time_split.py`,
`tests/test_euler_dg_2d_time_split.py` or `validation/euler_dg_2d/time_split.json`
in that window:

```
git diff origin/main --stat   # exactly the 6 PR files, nothing else
```

After the merge the branch tree **is** the tree the (already green) `pull_request`
runs test, so the `push` event on the new head tests the same green tree.

## 4. Verification — local reproduction of the red and of the fix

Both trees were run on this host with `PYTHONPATH=.`, pytest 9.2.0, Python 3.11,
`OMP_NUM_THREADS=2`; `base` = `df8f38e` alone, `fixed` = `df8f38e` ⊕ `origin/main`.
Logs: `artifacts/ci_red_repro/` (`base_wall.log`, `fixed_wall.log`,
`*_test_ns_heat_conduction.log`), driver `artifacts/ci_red_repro/run_pins.sh`.

| tree | `tests/test_ns_heat_wall_closure.py` | verdict |
|---|---|---|
| base `df8f38e` | `FAILED test_interior_energy_residual_decays_but_wall_rows_do_not`<br>`FAILED test_conduction_entry_point_still_crashes_on_the_documented_default`<br>**2 failed, 3 passed** in 105.9 s | CI red reproduced |
| fixed (merged) | **5 passed** in 100.3 s | red cleared |

`tests/test_ns_heat_conduction.py` holds the two `XPASS(strict)` failures; both runs
were still executing that module (its `test_shipped_wrapper_runs_at_its_documented_defaults`
integrates to `final_time=40`) when this note was written, on a host loaded to
~1.1e3 runnable tasks on 128 cores — see the honest caveats below. The pin flip is
visible directly in the merged diff (`git diff df8f38e HEAD -- tests/test_ns_heat_conduction.py`:
both `xfail(strict=True)` markers removed and replaced by positive assertions), and the
identical merged tree already ran green in GitHub's own `pull_request` CI
(run 42017868242, head `df8f38e`, base `4ae1597`, conclusion **success**, 15:20Z).

## 5. Caveats (honest)

* No full local `pytest -q` run: the host was oversubscribed (load ≈ 1e3 vs 128 cores,
  and CI itself needs 2 h 20 min for the suite), so only the implicated modules were run
  locally. The authoritative arbiter is the `push`/`pull_request` check on the new head.
* Nothing in `tensordg/`, `benchmarks/`, `validation/` or the PR's pins was modified.
  The whole code diff of this round is the merge plus this note.
* An earlier round merged `main` locally but never pushed it, so the branch tree on
  GitHub stayed stale; that is why the red survived to round 2.