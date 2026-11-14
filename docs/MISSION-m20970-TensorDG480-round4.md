# Mission m20970: TensorDG#480 Round 4 — Periodic Reference Fix for Euler Vortex Error Conservation

**Status**: ✅ PASS (all gates)
**Original PR**: #480 (closed 2026-11-05, reopened as round 4)

## Summary

This mission consolidates the periodic reference fix for the isentropic vortex error conservation case (`CASE-DG-EULER-VORTEX-ERR`). The core issue was that long-time error measurements used the free-space vortex formula as reference, which is not the solution of the periodic problem once the vortex centre leaves the box. The fix adds a `period` parameter to `vortex_primitive()` that folds displacements into `[-L/2, L/2)` (min-image convention).

## Key Changes

| File | Change |
|---|---|
| `tensordg/euler_2d.py` | Added `period` parameter to `vortex_primitive()` and `vortex_error()`; `_wrap_shift()` helper |
| `benchmarks/verify_euler_dg_2d.py` | `_vortex_error()` forwards `period` to reference |
| `benchmarks/case_euler_vortex_errconserve.py` | `PERIOD` constant; both metric paths use periodic reference |
| `tests/test_vortex_periodic_reference.py` | 12 tests pinning the metric-level invariant |
| `scratch/diag_vortex_err.py` | Diagnostic: solver error vs periodic vs free-space reference |
| `scratch/diag_vortex_corner.py` | Corner probe: analytic evidence of reference defect |
| `docs/CASE-DG-EULER-VORTEX-ERR-round4.md` | Paper-quality report with all results |
| `docs/fig_vortex_refdefect.png` | Field cloud + error decomposition plot |

## Gate Results

| Gate | Criterion | Measured | Status |
|---|---|---|---|
| C1 | t=1 max-field < 6% | 4.926% (p0_n64) | PASS |
| C2 | DG α ≤ 1.0 | 0.199 | PASS |
| C3 | Invariant drift < 1e-10 | 1.95e-13 | PASS |
| C4 | Entropy drift < 1e-3 | 2.98e-07 | PASS |
| C5 | P0 order ≥ 0.8 | 0.8488 | PASS |

## Verification

- `pytest tests/test_vortex_periodic_reference.py -v`: 12/12 PASS
- Quick benchmark: P0 n=64 t=1 → 4.926%, DG P2 n=32 t=1 → 0.0094%
- All numbers match `case_euler_vortex_errconserve/metrics.json`

## Reproduction

```bash
PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --shard 0/1 --output-dir case_euler_vortex_errconserve
PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate --figures --output-dir case_euler_vortex_errconserve
PYTHONPATH=. python3 -m pytest tests/test_vortex_periodic_reference.py -v
```
