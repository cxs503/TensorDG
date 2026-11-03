# TensorDG TVB/WENO Limiter Development Report

**Mission**: m274  
**Goal**: EXPAND-DEV:tensordg-tvb-weno-limiter  
**Date**: 2026-10-25  
**Status**: Functional implementation with known limitations

## Implementation Summary

### Files Added
1. `tensordg/limiters.py` - TVB and WENO limiter implementations
2. `tests/test_limiters.py` - 13 unit tests for limiters
3. `benchmarks/sod_with_limiter.py` - Sod shock tube with limiter demonstration
4. `benchmarks/compare_limiter.py` - Comparison benchmark (no limiter vs TVB vs WENO)

### Files Modified
1. `tensordg/euler_dg_1d.py` - Added `limiter` and `tvb_constant` parameters to `solve_euler_dg_1d()`
2. `tensordg/__init__.py` - Exported limiter functions

### Features Implemented
- **TVB limiter** (Cockburn & Shu, 1989): Detects troubled cells via modified minmod test with tolerance M·dx². Reduces troubled cells to linear reconstruction with minmod-limited slope.
- **WENO limiter** (Qiu & Shu, 2005): Same troubled cell detection, replaces high-order polynomial with WENO reconstruction from three candidate stencils.
- **Troubled cell indicator**: Based on cell-average monotonicity + TVB threshold
- **Integration**: Limiters applied after each SSP-RK3 sub-step, before positivity filter
- **Runtime selection**: `limiter` parameter accepts `None`, `"tvb"`, or `"weno"`

### API Usage
```python
from tensordg.euler_dg_1d import solve_euler_dg_1d

# No limiter (baseline)
x, coeff, meta = solve_euler_dg_1d(left, right, ncells=100, degree=2)

# TVB limiter
x, coeff, meta = solve_euler_dg_1d(left, right, ncells=100, degree=2, 
                                     limiter="tvb", tvb_constant=1.0)

# WENO limiter
x, coeff, meta = solve_euler_dg_1d(left, right, ncells=100, degree=2,
                                     limiter="weno", tvb_constant=1.0)
```

## Test Results

### Unit Tests
- **Total tests**: 34 (21 existing + 13 new limiter tests)
- **Passed**: 34/34 (100%)
- **Failed**: 0

### Limiter-Specific Tests (13 tests)
1. ✅ TVB preserves cell average
2. ✅ WENO preserves cell average
3. ✅ TVB does not modify smooth monotone solutions
4. ✅ TVB reduces extremum oscillations
5. ✅ WENO reduces extremum oscillations
6. ✅ Troubled cell indicator detects extrema
7. ✅ Troubled cell indicator ignores monotone regions
8. ✅ Limiter dispatcher routes correctly
9. ✅ P0 (nmodes=1) is no-op
10. ✅ Multivariate handling
11. ✅ Dtype consistency (float32/float64)
12. ✅ Shock profile handling
13. ✅ WENO smoothness weights

## Performance Benchmarks

### Sod Shock Tube (100 cells, P2, t=0.2)

| Limiter | Time (s) | Steps | Density L2 (%) | Velocity L2 (%) | Pressure L2 (%) | Overhead |
|---------|----------|-------|----------------|-----------------|-----------------|----------|
| None    | 1.38     | 752   | 1.0020         | 3.7523          | 0.7871          | baseline |
| TVB M=1 | 2.57     | 763   | 2.1592         | 5.2444          | 2.3024          | 85.6%    |
| WENO M=1| 3.08     | 729   | 2.2113         | 6.7268          | 1.8776          | 122.9%   |

### Lax Shock Tube (100 cells, P2, t=0.13)

| Limiter | Time (s) | Steps | Density L2 (%) | Velocity L2 (%) | Pressure L2 (%) | Overhead |
|---------|----------|-------|----------------|-----------------|-----------------|----------|
| None    | 1.99     | 1064  | 6.6011         | 4.3156          | 1.8918          | baseline |
| TVB M=10| 3.51     | 1044  | 9.6290         | 6.4965          | 3.3305          | 76.4%    |

## Acceptance Criteria Evaluation

| Criterion | Status | Value | Notes |
|-----------|--------|-------|-------|
| All existing tests pass | ✅ PASS | 34/34 | No regressions |
| Limiter unit tests pass | ✅ PASS | 13/13 | All new tests pass |
| Limiter preserves cell average | ✅ PASS | Verified | Unit test confirmed |
| Limiter detects troubled cells | ✅ PASS | Verified | Unit test confirmed |
| Smooth solution accuracy preserved | ✅ PASS | Verified | Unit test: monotone solutions unchanged |
| Sod/Lax errors < 3% | ⚠️ PARTIAL | Baseline <3%, limiter 2-7% | Limiter increases errors |
| Limiter overhead < 30% | ❌ FAIL | 76-123% | Pure Python implementation |

## Limitations and Known Issues

### 1. Performance Overhead (FAIL)
- **Target**: <30% overhead
- **Actual**: 76-123% overhead
- **Cause**: Pure Python/PyTorch implementation with per-cell loops
- **Mitigation**: Future work should vectorize operations and/or move to C++/CUDA

### 2. Accuracy Degradation (FAIL)
- **Observation**: Limiter increases errors instead of reducing them
- **Cause**: TVB constant M=1.0 is too conservative; limiter activates in too many cells
- **Analysis**: Baseline solver (positivity filter only) already handles shocks reasonably well for moderate cases
- **Mitigation**: Problem-specific tuning of M; more sophisticated indicators needed

### 3. Over-Limiting
- **Issue**: Limiter reduces high-order modes even in smooth regions near discontinuities
- **Impact**: Loss of accuracy in contact discontinuities and expansion waves
- **Mitigation**: Hierarchical limiters, multi-dimensional indicators

### 4. Scope
- **Current**: 1D Cartesian meshes only
- **Missing**: 2D/unstructured mesh support, characteristic-based limiting
- **Future work**: Extension to multi-dimensional cases

## Recommendations

### For Current Use
- **Continue using baseline solver** (no limiter) for Sod/Lax and similar moderate shock problems
- The existing positivity filter is sufficient for verification benchmarks
- Limiter is not yet production-ready

### For Future Development
1. **Tune TVB constant**: Problem-specific M values (e.g., M=10-100 for strong shocks)
2. **Optimize performance**: Vectorize troubled cell detection and limiting operations
3. **Improve indicator**: Use entropy-based or multi-dimensional indicators
4. **Hierarchical WENO**: Implement higher-order WENO reconstructions
5. **2D extension**: Extend to tensor-product and unstructured meshes

## Conclusion

The TVB and WENO limiters are **functionally implemented** with correct API, comprehensive unit tests, and proper integration into the solver. However, the implementation has **two major limitations**:

1. **Performance**: 76-123% overhead vs <30% target
2. **Accuracy**: Limiter increases errors instead of reducing them

These limitations stem from the basic nature of the implementation (pure Python, simple indicator, conservative TVB constant). The baseline solver with positivity filter remains the recommended choice for current use cases.

**Verdict**: Functional but not production-ready. Requires tuning and optimization before deployment.

## Deliverables Checklist

- [x] `tensordg/limiters.py`: TVB + WENO limiter implementation
- [x] `tests/test_limiters.py`: 13 unit tests (all passing)
- [x] `benchmarks/sod_with_limiter.py`: Demonstration benchmark
- [x] `benchmarks/compare_limiter.py`: Comparison benchmark
- [x] Integration with `solve_euler_dg_1d()`
- [x] API documentation
- [x] Validation report
- [x] Dev report (this file)
- [ ] Performance target (<30% overhead): **NOT MET**
- [ ] Accuracy target (errors <3% with limiter): **NOT MET**


---

## Post-merge verification (2026-11-03, m2192) — correction of two statements

This report's "Accuracy Degradation" section attributed the error growth to
`M = 1.0` being too conservative. That attribution is **falsified**: on the
Sod problem the troubled-cell detector flags 129/200 cells at M = 1 and
54/200 at M = 300, yet the velocity error changes only from 4.7886% to
4.7914%. A real defect was found instead: `apply_tvb_limiter` wrote the
mode-1 coefficient as `minmod(d_left, d_right)` instead of
`minmod/2` (= physical slope * dx / 2), doubling the reconstructed slope; the
scaling ratio was 2.000 before and 1.000 after the one-line fix, which is now
covered by `tests/test_limiters.py`. With the scale fixed the limiter still
degrades accuracy *inside* the SSP-RK loop (4.79% vs 3.20% unlimited), while a
single application to the unlimited final state improves every field
(2.45%), so the remaining defect is the per-substage limiting policy, not the
TVB constant. Details, artifacts and the reproduction commands:
`docs/limiter_verification.md`.
