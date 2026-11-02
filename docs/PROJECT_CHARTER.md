# TensorDG — discontinuous Galerkin solver charter

**Canonical ecosystem vision:** [Tensor Ecosystem: Vision for the AI Era](https://github.com/cxs503/TensorPlatform/blob/main/docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md)

## Mission
Provide independently testable discontinuous-Galerkin numerical methods and reproducible benchmarks, with clear formulation and stability limits.

## Owns
- DG basis, numerical fluxes, quadrature, mass/stiffness operators, limiters, time integration and supported PDE formulations.
- DG-specific convergence, stability and conservation diagnostics.
- Runnable analytical and canonical benchmark cases.

## Does not own
- Generic research-agent runtime or cross-repository orchestration.
- Silently conflating DG, DG-LBM and other hybrid methods; each formulation must be named and documented.
- Treating a script that runs as proof of expected convergence or physical fidelity.

## Integration
Publish typed problem/mesh/field inputs, solver configuration, diagnostics and result artifacts. TensorPhysics may coordinate DG with other domain solvers; TensorPlatform remains the shared generic infrastructure layer.

## Quality gates
Demonstrate expected-order convergence on smooth problems, shock/limiter behavior on nonsmooth cases, conservation/stability where applicable, and comparison with analytical or trusted reference solutions. Record tolerances and failed cases.

## Agent rules
Read this charter and README; explain repository ownership and affected contracts; avoid private imports; include reproducible tests and benchmarks; report exact outcomes and limitations; distinguish code execution from numerical verification.

## Maturity
This charter defines intended scope; use actual code and measured benchmark evidence to determine maturity.
