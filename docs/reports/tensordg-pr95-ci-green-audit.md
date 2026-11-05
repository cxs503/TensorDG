# PR #95 (`agent/m3625`) — CI red triage and green evidence

**Scope.** Mission m4356, round 1: clear the reported `test=failure` on PR #95
(TensorDG, branch `agent/m3625`) without touching unrelated files.

**Head under audit.** `df8f38ecba614dc901673f967f016a600686c20e` ("m3625: matured
mission deliverables"), one commit on top of `619167f` (merge #53). Its payload
is the `dt=` verification knob for `solve_euler_dg_2d`, the
`benchmarks/euler_dg_2d_time_split.py` benchmark, `tests/test_euler_dg_2d_time_split.py`
and `validation/euler_dg_2d/time_split.json` — six files in total, no test or
solver file removed or loosened.

## 1. What the red actually is

The check-run history of that exact commit (`validation/pr95_ci_audit.json`,
rebuilt by `tools/pr95_ci_audit.py` from the GitHub check-runs API):

| # | event context | status | conclusion | started (UTC) | wall time |
|---|---|---|---|---|---|
| 126640072223 | `push` | completed | **failure** | 2026-11-04 14:57:47 | 141.8 min |
| 127156049851 | `pull_request` | completed | success | 2026-11-05 09:59:25 | 141.0 min |
| 127212831198 | `pull_request` | completed | success | 2026-11-05 14:20:10 | 60.2 min |
| 127355438354 | `pull_request` | **queued** | — | 2026-11-05 15:50:43 | — |

All four runs are attached to the *same* `head_sha`. In the failing run the only
red step was `Run pytest -q` (steps 7–17, the benchmark smoke runs, were
skipped as a consequence).

The decisive fact: **the same tree passes the very same suite** as soon as it is
checked out as the `pull_request` merge commit against `main`. The two green
runs are not a different code state — they are the same commit with
`main` merged in. So the red is **event-context / stale-base drift**, not a
defect introduced by this PR:

* a `push` run tests the branch tree alone, i.e. base `619167f` (30 commits
  behind `main`);
* the `pull_request` run tests `merge(head, main)` — the tree that would land;
* on the stale base two pins of the base tree are red (this branch family's
  earlier audit measured both):
  * `tests/test_ns_heat_wall_closure.py::test_interior_energy_residual_decays_but_wall_rows_do_not`
    — `assert bottom_ratio < 1.1` vs measured `1.1254` (the merged `main`
    replaced the hand-set bound with one measured from the data);
  * `tests/test_ns_heat_conduction.py::test_exact_steady_state_is_approximately_preserved`
    — `@pytest.mark.xfail(strict=True)` for a wall defect that `main`'s
    one-sided DG wall derivative + positivity filter has since repaired, so it
    XPASSes and `strict=True` turns that into a failure.

Neither file is touched by this PR, so the PR cannot have caused them; they are
pure base drift.

## 2. Fix applied

`git merge origin/main` (`4ae1597`) into the `agent/m3625` lineage — conflict
free, and provably content-neutral for the PR's own payload:

```
git diff --stat df8f38e <merge> -- \
    benchmarks/euler_dg_2d_time_split.py tensordg/euler_dg_2d.py \
    tests/test_euler_dg_2d_time_split.py validation/euler_dg_2d/time_split.json \
    docs/reports/tensordg-euler-dg-2d-time-split.md
# (empty)
```

The merge commit makes the PR head identical to the merge tree CI already
verified, so the next `pull_request` run has no drift left to rediscover.

## 3. Local verification on the merge tree

All commands run in this worktree with `PYTHONPATH=.` and `python -m pytest -q`.
Host note: shared 128-core box under load average ≈ 1.5e3 (other agents'
suite runs), so wall times below are 3–10× the CI ones.

| check | result |
|---|---|
| historic red pins: `test_ns_heat_wall_closure.py::test_interior_energy_residual_decays_but_wall_rows_do_not` + `test_ns_heat_conduction.py::test_exact_steady_state_is_approximately_preserved` | **PASS** (part of the 8 below) |
| PR payload: `tests/test_euler_dg_2d_time_split.py` (6 pins + 1 strict xfail) | **PASS**, xfail stays xfail |
| that trio combined | `8 passed, 1 xfailed in 220.46 s` |
| flake probe: random-input limiter group (`test_limiters.py`, `test_limiter_2d.py`, `test_limiter_kernel_budget.py`, `test_limiter_validation.py`, `test_euler_dg_2d_anisotropic.py`, `test_euler_dg_2d_limiter.py`, `test_differentiability.py`) × 3 repetitions | `80 passed`, 0 failed (no flaky outcome across repeats) |
| full suite `pytest -q` | run started; **47 passed + 1 xfailed, 0 failed, 0 error when the session's wall clock ran out** (48/526 tests, ~20 s/test here). Honest limitation: a complete local pass takes ≈2 h at this load, so the full-suite arbiter remains the PR check run. |

Figure: `docs/figures/pr95_ci_audit.png` — left panel: the four check runs of
`df8f38e` in time order (red = the 11-04 push failure, green = the two
`pull_request` successes on the same sha, grey = the run still queued behind the
Actions backlog); right panel: the local pass/fail counts above.

## 4. Why the PR is not merged yet

`merge_governor` requires *every* check-run conclusion on the head to be
`success`. The run enqueued when the PR was opened (127355438354, 15:50:43Z) is
still `queued` — GitHub Actions has been backlogged since ≈12:40Z across the
whole repository (eight PRs queued at 15:50). The PR is `mergeable: true` and
already carries two green `test` conclusions on its head; merging is waiting on
the runner, not on code.

Deliberate decision: the branch ref was **not** force-updated from this session.
Pushing the merge commit would restart the 2 h check run and discard the two
green conclusions that already exist on this exact head — strictly worse than
leaving the proven-green head in place. This session therefore commits the
merge + evidence on its own worktree branch and lets the push/merge line decide.

## 5. Residual risk

* The 11-04 failure has no retrievable log (`actions/jobs/.../logs` needs admin
  rights on the repo), so the per-test attribution above is reconstructed from
  the base-tree audit plus the green merge runs, not from the failed job's
  trace. It is corroborated, not proven.
* Two of the four check runs took ≈141 min while one took 60 min for the same
  suite — the CI job duration is dominated by queue/runner contention, which is
  also the most likely reason a single run went red.
* The full local suite was not completed (see §3); no claim of a full local
  green is made.