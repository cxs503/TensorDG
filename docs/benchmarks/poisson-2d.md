# Benchmark: 2D Poisson with symmetric interior-penalty DG

## Problem

The benchmark solves `-Delta(u) = f` on the unit square with homogeneous
Dirichlet data. The manufactured solution is `u=sin(pi*x)*sin(pi*y)`, hence
`f=2*pi^2*sin(pi*x)*sin(pi*y)`. It verifies volume derivatives, interior-face
coupling, boundary conditions, and the linear solve without requiring a
fluid-dynamics limiter.

## Method

- Tensor-product modal Legendre bases P1--P3 on a uniform Cartesian mesh.
- Symmetric interior-penalty (SIPG) volume, consistency, symmetry, and penalty
  terms on interior and boundary faces.
- Penalty scaling `penalty*(p+1)^2/h`, with `penalty=20` by default.
- Dense `torch.linalg.solve`, appropriate for these small verification cases.
- Gauss--Legendre load integration and exact-solution L2-error evaluation.

The documented P1, 16x16 case has 1024 degrees of freedom and must have
relative L2 error below 3%. Tests also check matrix symmetry, approximately
second-order P1 convergence, and approximately third-order P2 convergence.

## Run

```bash
python -m pip install -e ".[dev]"
python -m benchmarks.poisson_2d --cells 16 --degree 1 \
  --output-dir docs/benchmarks/results/poisson-2d
```

Outputs include numerical/exact/error contours, a y=0.5 centerline curve,
CSV data, and JSON metrics. The dense matrix is a verification baseline, not
a scalable solver; sparse or matrix-free operators are future work. See the
[Chinese tutorial](../tutorials/classic-dg-benchmarks.md) for measured output.
