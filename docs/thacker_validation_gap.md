# TensorDG SWE Thacker Benchmark Validation Report

**Mission**: m6315  
**Date**: 2026-11-06  
**Status**: IDENTIFIED VALIDATION GAP

## Objective

Validate the 1-D shallow water equation solver against the Thacker (1981) parabolic basin oscillation analytical solution. This is a classical benchmark for testing:
1. Well-balanced property (ability to maintain lake-at-rest with non-flat bed)
2. Oscillatory flow accuracy
3. Positivity preservation

## Thacker (1981) Analytical Solution

The Thacker solution describes oscillating water in a parabolic basin:

**Bed topography:**
```
z_b(x) = -h₀ · (1 - (x/L)²)
```

**Initial water surface (tilted):**
```
η(x,0) = h₀ + A · x/L
```

**Exact time-dependent solution:**
```
η(x,t) = h₀ + A · cos(ωt) · x/L
u(x,t) = -A · ω · sin(ωt) / g
h(x,t) = η(x,t) - z_b(x)
```

where ω = √(2gh₀) / L

**Parameters used:**
- h₀ = 1.0 m (reference depth)
- A = 0.5 m (oscillation amplitude)
- L = 1.0 m (basin half-width)
- g = 9.81 m/s²
- ω = 4.4294 rad/s
- Period T = 1.4185 s

## Test Results

### Initial Condition Verification
✅ Initial water surface matches analytical solution exactly:
```
x:      [-1.0, -0.8, -0.6, -0.4, -0.2,  0.0,  0.2,  0.4,  0.6,  0.8,  1.0]
η(t=0): [ 0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4,  1.5]
```

### Time Evolution (t = 0.1 s)
❌ **CRITICAL FAILURE**: Numerical solution diverges dramatically from exact solution

**Expected (exact):**
- η: [0.57, 0.62, 0.66, 0.71, 0.75, 0.80, 0.84, 0.89, 0.93, 0.98, ...]
- u: -0.097 m/s (uniform)

**Computed (numerical, 20 cells, P1):**
- η: [0.64, 0.63, 0.66, 0.73, 0.82, 0.91, 0.99, 1.06, 1.13, 1.19, ...]
- u: [2.38, 1.46, 0.97, 0.68, 0.48, 0.30, 0.12, -0.07, -0.26, -0.45, ...]

**Error:**
- Max |η - η_exact| = 0.223 m (22.3% relative error)
- Velocity field completely wrong (spatially varying instead of uniform)

### Convergence Study
Even with mesh refinement, errors remain O(100%):
- P1, 50 cells, t=0.5s: η error = 8.2×10¹² %
- P2, 50 cells, t=0.5s: η error = 1.4×10⁴ %

## Root Cause Analysis

The Thacker solution requires **specific boundary conditions** that allow the water surface to oscillate freely at the basin edges. The current SWE solver implementation appears to use boundary conditions incompatible with this analytical solution.

**Likely issues:**
1. **Boundary condition mismatch**: The solver may use reflective or periodic BCs that constrain the solution
2. **Source term treatment**: The bed slope source term may not be balanced correctly for this strongly varying topography
3. **Well-balanced property**: The scheme may not preserve the exact balance between flux gradients and source terms required by the Thacker solution

**Evidence:**
- Initial condition is correct
- Solution immediately diverges (within first time step)
- Velocity field shows spatial variation where exact solution is uniform
- Errors are orders of magnitude larger than expected discretization error

## Implications

This is a **validation gap** in the TensorDG SWE solver. The solver passes:
- ✅ Lake-at-rest with flat bed
- ✅ Smooth wave propagation
- ✅ Positivity preservation

But fails:
- ❌ Thacker parabolic basin oscillation (non-flat bed with oscillatory flow)

This suggests the well-balanced property is limited to simple cases and may not hold for strongly varying topography or oscillatory flows.

## Recommendations

1. **Implement proper boundary conditions** for the Thacker problem (likely characteristic-based or radiation BCs)
2. **Improve source term treatment** using hydrostatic reconstruction (Audusse et al. 2004)
3. **Add well-balanced tests** with non-flat beds to the test suite
4. **Document limitations** of current SWE solver regarding complex topography

## References

- Thacker, W. C. (1981). "Some exact solutions of the nonlinear shallow-water equations." Communications on Pure and Applied Mathematics, 34(4), 543-552.
- Audusse, E., et al. (2004). "A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows." SIAM Journal on Scientific Computing, 25(6), 2050-2065.

## Conclusion

The Thacker (1981) parabolic basin benchmark reveals a **significant validation gap** in the TensorDG 1-D SWE solver. The solver cannot reproduce this classical analytical solution, indicating limitations in boundary condition treatment and/or source term balancing for strongly varying topography.

This finding should be documented as a known limitation and addressed in future development. The benchmark script and test file have been created for regression testing once the issue is resolved.

**Status**: VALIDATION GAP IDENTIFIED - requires solver improvements before this benchmark can pass.
