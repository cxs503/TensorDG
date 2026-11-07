# TensorDG Validation Status Report

**Mission**: m6315  
**Date**: 2026-11-06  
**Scope**: Comprehensive validation audit of TensorDG solvers

## Executive Summary

This report audits the validation status of TensorDG solvers against analytical benchmarks and literature references. The audit identifies **one critical validation gap** in the 1-D shallow water equation solver and confirms that other major solvers have adequate validation coverage.

## Validation Matrix

### ✅ Well-Validated Modules

| Module | Benchmark | Error | Status |
|--------|-----------|-------|--------|
| **Linear Advection 1D** | Periodic sine wave | 0.000236% (P2, 80 cells) | ✅ Design order (EOC=3) |
| **Linear Advection 2D** | Solid body rotation | <3% | ✅ Mass conservation verified |
| **Heat Equation 1D** | Analytical Fourier solution | <3% | ✅ SIPG P1 EOC=2 |
| **Poisson 2D** | Manufactured solution | <3% | ✅ SIPG P2 EOC=3 |
| **Euler 1D (Sod)** | Exact Riemann solution | max 2.52% | ✅ All fields <3% |
| **Euler 1D (Lax)** | Exact Riemann solution | <3% | ✅ Positivity preserved |
| **Isentropic Vortex 2D** | Exact transport solution | <3% | ✅ Conservation verified |
| **Burgers 1D** | Exact characteristic solution | <3% | ✅ Pre-shock regime |
| **Advection-Diffusion 1D** | Exponential layer (exact) | <3% | ✅ Flux consistency verified |
| **Matrix-free SIPG** | vs dense backend | <1e-10 | ✅ All backends match |

### ❌ Validation Gaps Identified

| Module | Benchmark | Issue | Severity |
|--------|-----------|-------|----------|
| **SWE 1D** | Thacker (1981) parabolic basin | O(100%) error, solution diverges | 🔴 CRITICAL |

## Detailed Findings

### 1. Thacker (1981) Parabolic Basin - CRITICAL GAP

**Problem**: Oscillating water in parabolic basin with analytical solution
```
z_b(x) = -h₀(1 - (x/L)²)
η(x,t) = h₀ + A·cos(ωt)·x/L
u(x,t) = -Aω·sin(ωt)/g
```

**Test Results**:
- Initial condition: ✅ Correct
- Time evolution: ❌ Errors O(100-10¹²)%
- Convergence: ❌ No mesh refinement improvement
- Velocity field: ❌ Spatially varying (should be uniform)

**Root Cause**: Boundary condition mismatch. The solver uses reflective/periodic BCs incompatible with the Thacker solution's free-oscillation requirement.

**Impact**: The well-balanced property is limited to simple cases (flat bed lake-at-rest). Strongly varying topography with oscillatory flow fails.

**Recommendation**: 
1. Implement hydrostatic reconstruction (Audusse et al. 2004)
2. Add characteristic/radiation boundary conditions
3. Document as known limitation

**Files Created**:
- `benchmarks/thacker_parabolic_basin.py` - Benchmark script
- `tests/test_thacker_benchmark.py` - Test suite (currently failing)
- `docs/thacker_validation_gap.md` - Detailed analysis

### 2. SWE 1D - Partial Validation

**Passing**:
- ✅ Lake-at-rest (flat bed): machine precision
- ✅ Smooth wave propagation: physical solutions
- ✅ Positivity preservation: h ≥ 0

**Failing**:
- ❌ Thacker parabolic basin (non-flat bed, oscillatory)
- ❌ Dam break with complex topography (not tested)

**Conclusion**: SWE solver is suitable for simple cases but not validated for complex topography or strongly oscillatory flows.

## Validation Coverage by Physics

| Physics | Coverage | Confidence |
|---------|----------|------------|
| Linear advection | ✅ Comprehensive | High |
| Nonlinear advection (Burgers) | ✅ Pre-shock | High |
| Compressible flow (Euler) | ✅ Riemann problems | High |
| Diffusion (heat/Poisson) | ✅ SIPG verified | High |
| Advection-diffusion | ✅ Boundary layers | High |
| Shallow water (simple) | ✅ Lake-at-rest | Medium |
| Shallow water (complex) | ❌ Thacker fails | Low |

## Recommendations

### Immediate Actions
1. **Document SWE limitations** in module docstring and user guide
2. **Add warning** when using SWE with strongly varying topography
3. **Create issue** for Thacker benchmark implementation

### Medium-term
1. Implement hydrostatic reconstruction for SWE
2. Add characteristic boundary conditions
3. Validate against additional SWE benchmarks:
   - Ritter dam break (flat bed, dry downstream)
   - Stationary shock (with bed step)
   - Tsunami runup on slope

### Long-term
1. Extend Thacker to 2-D (circular parabolic basin)
2. Add moving boundary / wetting-drying validation
3. Validate against experimental data (if available)

## Test Suite Status

**Total tests**: 80+ (across all modules)  
**Passing**: 78+  
**Failing**: 6 (Thacker benchmark tests - intentionally failing to document gap)

**Coverage**:
- Conservation laws: ✅ Verified
- Design order convergence: ✅ Verified for all smooth problems
- Positivity preservation: ✅ Verified where applicable
- Flux consistency: ✅ Verified for advection-diffusion
- Well-balanced property: ⚠️ Limited to flat bed

## Conclusion

TensorDG has **strong validation coverage** for smooth problems (advection, diffusion, compressible flow, advection-diffusion). The **critical gap** is in the shallow water equations for complex topography and oscillatory flows.

The Thacker benchmark failure reveals a fundamental limitation in the current SWE implementation that should be addressed before the solver is used for practical coastal/estuarine applications.

**Overall validation status**: 🟡 **PARTIALLY VALIDATED** (strong for smooth problems, weak for SWE with complex topography)

## References

- Thacker, W. C. (1981). "Some exact solutions of the nonlinear shallow-water equations."
- Audusse, E., et al. (2004). "A fast and stable well-balanced scheme with hydrostatic reconstruction."
- Roos, Stynes & Tobiska. "Robust Numerical Methods for Singularly Perturbed Differential Equations."
- Cockburn & Shu (1989). TVB Runge-Kutta local projection discontinuous Galerkin method.
- Qiu & Shu (2005). WENO limiting for DG methods.
