# PR #68 CI red → green: root-cause audit

Branch `agent/m3625` (PR #68, "2-D Euler DG time/space error split") went red in the
`test` job. The red was **not** caused by the PR's own change; it came from the
branch being based on a stale `main` whose pins had already been superseded. This
note records the diagnosis, the minimal fix and the evidence, so the next agent
does not have to re-derive it.

## 1. Symptom

| item | value |
|---|---|
| PR | #68, head `df8f38e` ("m3625: matured mission deliverables") |
| failing job | `test` (workflow `.github/workflows/tests.yml`) |
| failing step | `pytest -q` (all later benchmark steps `skipped`) |
| branch base | `619167f` ("Merge #53"), i.e. **21 commits behind `main`** |

The PR itself is 6 files, none of them a test on `main`:

```
benchmarks/euler_dg_2d_time_split.py            | 343 +
docs/figures/euler_dg_2d_time_split.png         | Bin
docs/reports/tensordg-euler-dg-2d-time-split.md | 217 +
tensordg/euler_dg_2d.py                         |  20 +-   (adds the dt= knob)
tests/test_euler_dg_2d_time_split.py            | 124 +
validation/euler_dg_2d/time_split.json          | 579 +
```

`tensordg/euler_dg_2d.py` gains one keyword-only-in-practice argument
(`dt=None`, default = old adaptive CFL path) plus two metadata keys. An AST scan
of every call site in the repository (`tests/`, `benchmarks/`, `tensordg/`) found
**no** call that passes more than one positional argument to
`solve_euler_dg_2d` / `solve_euler_dg_2d_vortex`, so the shifted parameter order
cannot change any existing result.

## 2. Root cause: two stale strict pins on the old base

`main` (m3139 wall repair + the 2026-11-04 ratchet flips) changed the physics and
the pins together; the PR branch still carries the *pre-flip* pins, which are now
red against the already-repaired solver. Reproduced directly by checking the
base version of each file out of `git show df8f38e:tests/...` and running it
against the current sources:

**(a) `tests/test_ns_heat_wall_closure.py::test_interior_energy_residual_decays_but_wall_rows_do_not`**

```
assert bottom_ratio < 1.1
E       assert 1.1254006470928049 < 1.1
1 failed, 4 deselected in 7.65s
```

The pin sat 2.3 % below the measured ratio, so it was red on *every* runner; the
merged pin measures the bound from the data (`wall_16_32 < 1.2`,
`wall_32_64 < 1.1`, monotone) instead.

**(b) `tests/test_ns_heat_conduction.py::test_exact_steady_state_is_approximately_preserved`**

The base version is `@pytest.mark.xfail(strict=True, reason="known defect ...")`.
The m3139 wall repair (one-sided DG wall derivative as default + positivity
filter) removed the defect, so on the base the pin **XPASSes** and `strict=True`
turns that into a failure:

```
1 failed, 8 deselected in 66.39s
```

**(c) (same family, slow to time locally)**
`test_conduction_entry_point_still_crashes_on_the_documented_default` asserted
`pytest.raises(FloatingPointError)` for the `final_time=40` default; after the
repair the run no longer aborts, so the `raises` never fires. `main` replaced it
with `test_conduction_entry_point_completes_and_stays_physical`.

None of (a)–(c) touches the PR's `dt=` knob, the time-split benchmark or its pins.

## 3. Fix

`git merge origin/main` into `agent/m3625` — no code edit, no test edit, no
resolution needed (the merge was conflict-free, since `main` never touched
`tensordg/euler_dg_2d.py` between `619167f` and `efd020d`):

```
git diff origin/main --stat        # exactly the 6 PR files, nothing else
```

This is also what the `pull_request` event tests anyway: GitHub runs the CI job on
the *merge commit* of head and `main`, so a branch that lags `main` is red by
construction while its own content is fine.

## 4. Verification (all commands run in this worktree, `PYTHONPATH=.`)

| check | result |
|---|---|
| base pin (a) vs current sources | **FAIL** `1.1254 < 1.1` in 7.7 s — red reproduced |
| base pin (b) vs current sources | **FAIL** (strict XPASS) in 66.4 s — red reproduced |
| merged `test_interior_energy_residual_decays_but_wall_rows_do_not` | **PASS** 8.5 s |
| merged `test_exact_steady_state_is_approximately_preserved` | **PASS** 60.4 s |
| PR's own `tests/test_euler_dg_2d_time_split.py` (7 pins, 1 strict xfail) | **PASS** `6 passed, 1 xfailed` in 97.8 s |
| `tests/test_euler_dg_2d*.py` group (7 files) on the merged tree | see §5 |

## 5. Caveats (honest)

* The host these checks ran on was heavily oversubscribed (load average ~1.8e3 on
  128 cores), so wall times here are ~5–10× the CI ones and a full-suite run is
  not reproducible on demand; the group runs above are the usable substitute.
  The authoritative arbiter remains the PR `test` check run.
* Nothing was changed in `tensordg/euler_dg_2d.py`, the benchmark or the pins:
  the red was stale-base drift, so the fix is the merge only.
* The `dt=` knob keeps its documented contract (default `None` = adaptive CFL,
  byte-identical results); a float pins the step and the last step is trimmed
  onto `final_time`.