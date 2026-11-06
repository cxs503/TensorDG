# 2-D Euler system conservation audit (per-step ledger)

Generated 2026-11-06 05:19 UTC by `python tests/euler_2d_conservation_report.py`.

## Case

Smooth isentropic vortex on a doubly periodic box $[-5,5]^2$, P0 finite volume with Rusanov fluxes and SSP-RK3, CFL 0.4, background velocity (1, 1).  Invariants sampled once per accepted step through the solver's `observer` hook.

| case | cells | dx | dy | steps | $t_{final}$ |
|---|---|---|---|---|---|
| square_32 | 32x32 | 0.3125 | 0.3125 | 43 | 1.00 |
| anisotropic_40x20 | 40x20 | 0.2500 | 0.5000 | 40 | 1.00 |
| square_32_long | 32x32 | 0.3125 | 0.3125 | 158 | 4.00 |

## Global invariants (relative to the initial integral)

| case | mass | x-momentum | y-momentum | energy | max \|mean\|/rms |
|---|---|---|---|---|---|
| square_32 | 2.75e-15 | 0.00e+00 | 2.89e-16 | 1.65e-15 | 0.056 |
| anisotropic_40x20 | 1.30e-15 | 2.60e-15 | 2.31e-15 | 1.65e-15 | 0.081 |
| square_32_long | 1.07e-14 | 9.40e-15 | 7.96e-15 | 9.40e-15 | 0.064 |

Criteria: drift <= 1e-13, bias ratio <= 0.5, closure residual <= 1e-15.  Measured maxima: drift 1.070e-14, bias 0.081, closure 0.0e+00.

## Negative control (injected non-conservative source)

| leak L | steps | expected $LT/|M_0|$ | measured drift | recovery error | bias ratio | red? |
|---|---|---|---|---|---|---|
| 1e-03 | 43 | 1.018e-05 | 1.018e-05 | 2.2e-10 | 0.999 | YES |
| 1e-06 | 43 | 1.018e-08 | 1.018e-08 | 3.3e-07 | 0.999 | YES |
| 1e-09 | 43 | 1.018e-11 | 1.018e-11 | 3.4e-04 | 0.999 | YES |
| 1e-12 | 43 | 1.018e-14 | 7.088e-15 | 3.0e-01 | 0.153 | no |

## Figures

* `euler_2d_system_drift.png` -- invariant drift vs step count (left) and the leak response curve (right).
* `euler_2d_system_density.png` -- density field of the audited run (64x64 cells, 90 steps, rho in [0.673, 1.000]).

## Reproduce

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest tests/test_conservation_2d_system.py -q
python tests/euler_2d_conservation_report.py            # regenerate this file
```

The in-suite negative control (`test_injected_cell_source_turns_the_ledger_red`, `test_the_criterion_catches_a_leak_far_below_the_clean_drift`) injects the source through `injected_mass_source`; the independent mutation evidence (a scaled periodic ghost state in `tensordg/euler_2d._pad`, which makes 5 of the 7 tests fail) is in `artifacts/euler2d_conservation_negative_control.log`.

## Notes

* The audit judges *conservation only*: an accuracy bug that keeps the scheme in flux-difference form (e.g. dropping the y flux entirely) leaves every invariant at round-off and is not caught here -- see `artifacts/euler2d_scope_probe.log`.  Accuracy is covered by the vortex/field-error tests.
* Per-step statistics, not just the end-state difference, are what separate round-off noise (`|mean|/rms ~ 0.03-0.08`) from a one-sided leak (`|mean|/rms > 0.9`).
