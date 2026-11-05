# Gate-power audit — sabotage-control hardening (PR #72 follow-up)

Scope: `tests/test_gate_power.py` only. No change to the audit driver
(`benchmarks/audit_gate_power.py`), its artifacts (`validation/gate_power*`) or
the report.

## Why

`test_sipg_penalty_below_stability_threshold_breaks_the_heat_gate` used to
assert a single numeric outcome for the sabotaged SIPG path:

```python
sabotaged = run_heat_1d(16, penalty=0.2)["u"]
assert sabotaged > 1e10
assert math.isfinite(sabotaged)
```

Both lines assume *how* a non-positive discrete operator surfaces. It does not
depend on the audit, it depends on the solver stack: the penalty-0.2 operator is
singular, so

| backend behaviour | measured value | old assertion |
|---|---|---|
| round-off amplified (this dev box, torch 2.7.1) | `4.63e+57` | pass |
| solve returns `inf` / `nan` | `inf`, `nan` | `math.isfinite` fails |
| solve raises `torch._C._LinAlgError` (a `RuntimeError`) | exception | test errors |

Only the first is a statement about the gate; the other two are a statement
about which BLAS/torch build the job ran on. A negative control must report
"the gate is broken" for all three, otherwise the audit's own evidence turns
red on a different machine while the physics is unchanged.

## What changed

A single helper normalises the three failure modes:

```python
def _sabotaged_heat_error(cells=16, penalty=0.2):
    try:
        value = float(run_heat_1d(cells, penalty=penalty)["u"])
    except (RuntimeError, ArithmeticError):
        return math.inf
    return value if math.isfinite(value) else math.inf
```

and the assertion becomes `sabotaged >= SABOTAGE_FLOOR` (1e10) plus
`sabotaged > healthy`. The scientific content is unchanged: the documented run
measures `3.6e-05`, so a 15-order-of-magnitude blow-up is still required before
the control counts as "the gate has teeth"; a small error still fails the test.

## Verification (run locally, torch 2.7.1 / numpy 2.4.6)

```
$ python -m pytest tests/test_gate_power.py -q
4 passed in 8.11s

# failure-mode simulation: force the two backend behaviours we do not see here
raise -> inf  True        # monkeypatched runner raising RuntimeError
nan   -> inf  True        # monkeypatched runner returning nan
measured -> 4.628893739777192e+57
```

Margins of the remaining one-sided asserts on this box (so that a
hardware-dependent regression is distinguishable from a real one):

| control | value | threshold | margin |
|---|---|---|---|
| documented 1-D advection error | `1.50e-03` vs gate `3e-02` | `< 0.03` | 20x |
| same case, 8x coarser (5 cells) | `8.93e-02` | `> 0.03` | 3.0x |
| `GATE / documented` | `20.0` | `> 10.0` | 2.0x |
| time-shifted metric (t=0.375 vs t=0.25) | `7.65e-01` | `> 0.03`, `> 100 x locked` | 25x / 5.1x |

## Limitations

* The GitHub Actions log of the failing run needs an authenticated token
  (unauthenticated `actions/jobs/<id>/logs` returns HTTP 403), so the failing
  test id of run `41867343629` could not be read from this environment. The
  change above targets the only new assertion in the PR that is
  platform-dependent; it is a robustness fix, not a proven repair of an
  identified assertion. The rest of the PR (`benchmarks/audit_gate_power.py`,
  `tools/render_gate_power_report.py`, `validation/gate_power*`) is
  byte-identical to the PR head and deterministic (float64, no RNG).
* The full local suite (494 tests) does not fit in the sandbox' 120 s command
  budget; the ~430 fast tests pass, the ~60 heavy ones were not re-run here.