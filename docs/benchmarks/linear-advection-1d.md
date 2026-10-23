# Benchmark: 1D periodic linear advection (PyTorch DG)

## Purpose

This is TensorDG's first verification benchmark. Smooth periodic linear advection is a standard DG accuracy test: it isolates the spatial discretization, numerical flux, and time integration while providing an exact solution and a clean mesh-convergence measurement. Published DG studies routinely use this problem to verify design-order accuracy (for example, [RKDG accuracy tests](https://pmc.ncbi.nlm.nih.gov/articles/PMC4234454/)).

## Governing equation

\[
\partial_t u + a\,\partial_x u = 0,\qquad x\in[0,1],\quad a=1,
\]
with periodic boundary conditions and
\[
u(x,0)=\sin(2\pi x),\qquad u_{exact}(x,t)=\sin(2\pi(x-at)).
\]
At final time \(T=1\), the exact solution returns to its initial profile.

## Numerical method

- Uniform 1D mesh; modal Legendre basis of degree \(p=0,1,2,3\).
- Element-local weak DG formulation with an upwind numerical flux.
- Classical four-stage Runge–Kutta time integration.
- PyTorch tensors for the state and numerical operator; CPU by default, with CUDA selectable when available.
- Gauss–Legendre quadrature for projection and error evaluation.

For smooth solutions, the expected spatial L2 convergence rate is approximately \(p+1\), provided temporal error is sufficiently smaller than spatial error. The benchmark reports observed order of convergence (EOC) from successive mesh refinements.

## Run

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m benchmarks.linear_advection_1d --degree 1 --cells 20 40 80 160
python -m benchmarks.linear_advection_1d --degree 2 --cells 10 20 40 80 \
  --output-dir docs/benchmarks/results/linear-advection-1d \
  --max-relative-error 0.03
pytest -q
```

To request GPU execution, pass `--device cuda` on a CUDA-enabled PyTorch installation.

## Acceptance criteria

1. All solution coefficients and reported errors are finite.
2. Periodic total mass is conserved to numerical precision.
3. Smooth-solution L2 errors decrease under mesh refinement.
4. Degree-one EOC approaches 2 and degree-two EOC approaches 3 once spatial error dominates time-integration error.

## Measured results

The P2 10/20/40/80-cell run gives absolute L2 errors of 8.56563e-4,
1.06970e-4, 1.33721e-5, and 1.67157e-6, with successive EOC values 3.001,
3.000, and 3.000. The finest relative L2 error is 0.000236%, below the 3%
acceptance threshold. Profile and convergence plots, CSV data, and JSON
metrics are stored in `docs/benchmarks/results/linear-advection-1d/`.

See the [Chinese classic-benchmark tutorial](../tutorials/classic-dg-benchmarks.md)
for the complete run record and images.

## Follow-on benchmarks

Implemented follow-ons include smooth Burgers, Sod and Lax Riemann problems,
2D Cartesian advection, SIPG Poisson/heat, and a 2D isentropic vortex. See the
[nonlinear benchmark tutorial](../tutorials/nonlinear-dg-benchmarks.md).
Unstructured triangular meshes, Shu--Osher, robust TVB/WENO limiting, and
Navier--Stokes benchmarks remain future work.
