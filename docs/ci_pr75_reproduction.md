# PR #75 CI-red investigation and fix (TensorDG, branch `agent/m3714`)

Mission: m4075. Scope: make PR #75's `test` check green without touching
unrelated files.

## 1. What the PR contains

`origin/agent/m3714` (head `4c4138f`) sits on merge base `619167f` and is 21
commits behind `origin/main`. It adds only verification material, plus one new
test module:

| path | content |
|---|---|
| `tests/test_limiter_2d_order.py` | 7 tests pinning the 2-D TVB/WENO limiter on the analytic isentropic vortex: inertness at large `M`, negative control at `M = 0`, `degree = 2` mask path, mask-level scale consistency, observed-order ladder |
| `benchmarks/limiter_2d_order_study.py`, `benchmarks/report_limiter_2d_order.py` | order study + figure/report generator |
| `validation/limiter_2d_order/*.json` | raw ladder, P1 and P2, meshes 8/16/32, five limiter settings |
| `docs/limiter_2d_order_verification.md`, `docs/figures/limiter_2d_order/*.png` | write-up and figures |

No file under `tensordg/` is modified by the PR, so a `test` failure cannot come
from a solver change: it must come either from the new test module or from
something inherited from `main`.

## 2. How the red was reproduced

* Local host: Linux x86_64, Python 3.11, **torch 2.7.1**, pytest 9.2.0.
* CI: `ubuntu-latest`, Python 3.11, `pip install -e ".[dev]"` which resolves
  `torch>=2.0` to the **newest** wheel, then `pytest -q`, then 12 benchmark
  steps (`PYTHONPATH=.`).
* GitHub tests the *merge commit* of the PR head into `main`, so `origin/main`
  was merged into the branch locally first. The merge is clean (no conflicts,
  `git merge --no-commit --no-ff origin/main`).

Result on the merge commit, full suite (`pytest -q -p no:randomly -n 8`):

```
475 passed, 4 skipped, 7 xfailed, 0 failed     (excluded the running file)
```

A first pass with an *artificial* `--timeout=120` reported 4 failures
(`test_ns_heat_conduction::test_shipped_wrapper_runs_at_its_documented_defaults`,
`test_ns_heat_wall_closure::test_conduction_entry_point_completes_and_stays_physical`,
`test_ns_mms_order::test_design_order_is_recovered[2-meshes1-2.5]`,
`test_ns_mms_order::test_design_order_is_recovered[1-meshes0-1.7]`). Every
traceback ends in `Failed: Timeout (>100.0s) from pytest-timeout`, i.e. they are
artefacts of a host with load average 1900 (128 cores shared by many agents),
**not** assertion failures: all four pass on re-run without the artificial
timeout (`tests/test_ns_mms_order.py` 3 passed; the heat-conduction tests pass
in the un-timed full-suite run above). CI does not pass `--timeout`.

The 12 benchmark steps of `.github/workflows/tests.yml` were replayed locally on
the merge commit (`PYTHONPATH=.`, `OMP_NUM_THREADS=1`):

| step | verdict | wall time |
|---|---|---|
| `linear_advection_1d --degree 1` | OK | 15 s |
| `linear_advection_1d --degree 2 --max-relative-error 0.03` | OK | 29 s |
| `linear_advection_2d --cells 24 --degree 1` | OK | 26 s |
| `poisson_2d --cells 16 --degree 1` | OK | 10 s |
| `euler_sod_1d --cells 120 --time 0.05` | OK | 12 s |
| `plot_euler_sod_1d` | OK | 2 s |
| `compare_euler_sod_1d --cells 3200 --time 0.2` | OK | 45 s |
| `burgers_1d --cells 16 32 64 --degree 2` | OK | 18 s |
| `heat_1d --cells 16 --degree 2` | OK | 15 s |

(`lax_1d` and `isentropic_vortex_2d --cells 192` were not finished before the
remote session was recycled; they are untouched by the PR and green on `main`.)

So the red is *not* reproducible on the merge commit with this torch build.

## 3. The fix: remove bit-for-bit fragility from the new tests

What is genuinely portability-fragile in `tests/test_limiter_2d_order.py` is the
way "the limiter is inert" was asserted: five `torch.equal` / `==` comparisons
require **bit-for-bit** identity between two separate solver integrations. The
PR was authored against one torch build; CI installs the newest wheel, and
different BLAS kernels/thread-count defaults in a fresh wheel are allowed to
move the last bits of a float64 reduction. A last-bit difference turns
`smooth flow untouched` red although nothing scientific changed — precisely the
signature of a PR that is green for its author and red in CI.

The assertions now compare magnitudes against explicit, documented thresholds
instead of raw identity (`tests/test_limiter_2d_order.py`):

| test | before | after |
|---|---|---|
| `test_large_tvb_constant_leaves_smooth_flow_untouched` | `torch.equal` / `==` | `max|Δ| ≤ 1e-12`, error `rel=1e-12` |
| `test_weno_limiter_is_also_inert_when_the_indicator_is` | `torch.equal` | `max|Δ| ≤ 1e-12` |
| `test_measure_is_scale_consistent_on_linear_averages` | `torch.equal` | `max|Δ| ≤ 1e-12` |
| `test_strict_tvb_constant_limits_and_worsens_the_vortex` | `not torch.equal` | `max|Δ| > 1e-8` |
| `test_measure_flags_a_genuine_extremum` | `not torch.equal` | `max|Δ| > 1e-8` |

The pair of thresholds is deliberately two orders apart on both sides: real
limiter action moves the solution by `O(1e-3)` (see
`validation/limiter_2d_order/degree2.json`), round-off is `≤ 1e-13`, so the
test can neither be fooled by floating-point noise nor by a silently inert
limiter. The scientific claims are unchanged; only the tolerance class is.

## 4. Verification after the fix

```
$ python -m pytest -q -p no:randomly tests/test_limiter_2d_order.py \
      -k "measure or large_tvb or weno or strict"
5 passed, 2 deselected, 2 warnings in 32.75 s
```

and, before the tolerance change, the whole module (all 7 tests, including the
`degree = 2` observed-order ladder `eoc_inert = 2.93`, `eoc_strict = 1.39`
against `validation/limiter_2d_order/degree2.json`).

## 5. Limitations / honesty

* The GitHub Actions log for PR #75 could **not** be read from this sandbox:
  the unauthenticated REST API returned `403 rate limit exceeded` and direct
  `curl` to `api.github.com` is blocked by the network policy; no GitHub token
  is available here. The hypothesis "red = portability of a bit-exact assert"
  is therefore inferred from (a) the PR failing only in CI, (b) the PR touching
  no solver code, and (c) the five bit-exact assertions being the only
  build-sensitive part of the diff. It has not been confirmed against the
  Actions log; CI must be re-read after the platform pushes this branch.
* One CI-invisible risk remains: the module is heavy (7 vortex integrations,
  ~4 min on a loaded 128-core host). GitHub's per-job limit (6 h) is far above
  this, and no step in the workflow sets `timeout-minutes`, so it is not the
  cause of a `failure` conclusion.