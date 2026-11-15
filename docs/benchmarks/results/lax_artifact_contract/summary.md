# CASE-LAX-DG · cross-artifact reconciliation (contract C1-C6)

generated: 2026-11-15T03:37:29Z

| check | verdict | worst |
|---|---|---|
| C1-C4 step-size semantics (`scripts/attach_lax_dt_semantics.py --check`) | PASS | - |
| C5 cross-artifact agreement (identical definitions) | PASS | 0.000e+00 |
| C6 live reproduction @ 1e-12 (pre-registered) | FAIL | 1.248e-05 |
| C6 live reproduction @ 0.001 (cross-platform floor, post-hoc) | PASS | 1.248e-05 |
| R3 cross-definition gap (diagnostic, not drift) | N/A | 9.927e-01 |
| R4 doc-quoted dt values belong to one run | PASS | - |

Falsified: the residual-gap item 'regenerate lax_asymptotic_probe' -- the 12 % 'dt difference' is dt_last vs dt_mean of one run; not solver drift.

