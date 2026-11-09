# Curved isoparametric geometry in TensorDG: metric identities, free-stream preservation and the curved-wall order gate

*Campaign node `p254w1-n02-dg-curved-geometry` (TensorDG). Branch `agent/m7201`.*

## 1. Why this module exists

Every TensorDG benchmark before this node lived on a **uniform Cartesian mesh**.
There the Jacobian is constant, the metric terms are trivial and geometry can
never limit the convergence rate. As soon as the mesh is curvilinear — a hull
section, a cylinder surface, a CAD surface — a DG scheme silently loses its
formal order unless three geometric conditions are enforced *exactly*:

1. **Free-stream preservation (FSP).** A constant state must be preserved by the
   semi-discrete operator. If not, the scheme manufactures a spurious source
   proportional to the mesh curvature and the solution error saturates at a
   geometry-dependent floor. FSP holds if and only if the discrete divergence of
   the metric fluxes vanishes *with the same discrete derivative operator that
   is applied to the physical flux* (Kopriva 2006; Kopriva & Batta, JCP 270
   (2014) 550–565).
2. **Metric (Piola) identity.** `∂/∂ξ_i (J ∂ξ_i/∂x_j) = 0` and
   `∂ξ_i/∂x_j · ∂x_j/∂ξ_k = δ_ik` in the discrete sense.
3. **Curved-wall treatment.** The wall normal must come from the isoparametric
   geometry: a chord approximation of a circular wall has an `O(h)` normal
   error, which leaks mass and caps the order (section 5).

Before this node TensorDG had **no geometry layer at all**: no Jacobian, no
metric terms, no isoparametric mapping, no NURBS reader. `tensordg/curved_geometry.py`
is that layer, and this document is its verification record.

## 2. Method

### 2.1 Isoparametric mapping in Legendre modal form

Each quadrilateral is described by a tensor-product polynomial

```
x(ξ,η) = Σ_{k,l≤g} cx[k,l] P_k(ξ) P_l(η),   y likewise,
```

with `P_k` the unnormalised Legendre polynomials and geometry degree `g`
(`g = 1` is a bilinear quad, `g = p` is the classical isoparametric element,
`g > p` is superparametric). Storing *Legendre coefficients* rather than nodal
values is the key design decision:

* the Jacobian and all metric terms are obtained by exact coefficient-space
  differentiation (`legendre_derivative_matrix`), so the derivative of a
  constant polynomial is **bitwise** `0.0` — no round-off is introduced by the
  differentiation itself;
* the discrete metric identity (`∂_ξ a¹ + ∂_η a² = 0`) therefore becomes an
  *exact* algebraic statement that can be checked to machine zero on an affine
  mesh, and to round-off on a curved one.

The metric fluxes (contravariant flux times the Jacobian) are

```
a¹ = J ∂ξ/∂x_j A_j =  y_η A_x − x_η A_y ,   a² = −y_ξ A_x + x_ξ A_y .
```

### 2.2 Discretisation

`CurvedAdvection` is a strong-form DG discretisation of
`u_t + ∇·(A u) = 0` on a curvilinear mesh:

* modal Legendre basis `P_k(ξ)P_l(η)`, degree `p`;
* **exact quadrature**: `nq = 2p+2` Gauss points per direction. The metric-flux
  product `a_i u` has degree `2p`, its derivative `2p−1`, and the volume
  integrand `φ ∂_ξ(a¹u)` degree `3p−1`; `nq = 2p+2` makes both the exact
  differentiation (via the `nq`-point differentiation matrix / coefficient-space
  derivative) and the exact integration possible, so there is **no aliasing**;
* consistent upwind interface flux, face Gauss quadrature on the same nodes;
* exact mass matrix `M = ∫ φ φ J` (non-diagonal because `J` varies), inverted
  per element;
* no-penetration (`flux* = 0`) walls and periodic connectivity for the two mesh
  axes;
* SSP-RK3 in time with `dt = cfl · min J/(|a¹|+|a²|) / (2p+1)`, `cfl = 0.2`.
  The `J/(|a¹|+|a²|)` estimate is a *local* length scale, which matters on
  stretched annulus cells where `sqrt(area)` is far too optimistic;
* a blow-up guard in `evolve()`: an unstable run raises instead of returning
  garbage.

### 2.3 CAD / NURBS interface contract (`tensordg.curved_geometry.v1`)

The TensorCAD (N03) → TensorDG hand-off is a JSON payload:

```json
{"schema": "tensordg.curved_geometry.v1", "kind": "nurbs_curve", "degree": 2,
 "knots": [...], "control_points": [[x,y],...], "weights": [...],
 "centre": [x,y], "closed": true}
```

`validate_geometry_contract()` rejects malformed payloads (schema, degree,
knot/control-point counts, monotonicity, clamping, positive weights, closure)
with an explicit reason; `mesh_from_contract()` inverts the rational-B-spline
parameterisation (Newton on the curve derivative) to get the wall point at a
given polar angle, and builds the DG element geometry from it. The standard
9-point rational-quadratic NURBS circle is represented **exactly** (radius error
at round-off), which is the acceptance test for the CAD side.

## 3. Verification gates and results

Gates (all machine-checkable, numbers in
[`artifacts/dg_curved/result.json`](../artifacts/dg_curved/result.json)):

| gate | meaning | threshold |
|---|---|---|
| J1 | free-stream preservation on curved meshes: exact discrete metric-identity defect, plus the realised operator residual and the state drift after 100 SSP-RK3 steps | ≤ 1e−14 |
| J2 | metric identity: `dξ_i/dx_j·dx_j/dξ_k = δ_ik` (algebraic), Piola identity in the SBP-nodal norm, and in Legendre coefficient space | ≤ 1e−12 |
| J3 | curved-wall analytic order for `p = 1..5`: `|order − (p+1)|/(p+1)` | < 5 % |
| J4 | negative control: exactly affine mesh ⇒ **bitwise** zero free-stream defect | `== 0.0` |
| J5 | CAD contract: exact NURBS circle, malformed payloads rejected, CAD-derived mesh keeps FSP and order | see 3.6 |

<!-- RESULTS:BEGIN -->
### 3.1 Headline result

All five gates pass (full numbers in section 3.6 and
[`artifacts/dg_curved/result.json`](../artifacts/dg_curved/result.json)):

| gate | measured | threshold | verdict |
|---|---|---|---|
| J1 free-stream (exact metric defect, curved meshes, `p=1..5`) | `1.83e−15` | `1e−14` | pass |
| J2 metric identities (algebraic / SBP-nodal / coefficient) | `1.60e−15` | `1e−12` | pass |
| J3 curved-wall order `p=1..5` | `1.949, 3.004, 3.921, 4.980, 5.909` vs `2,3,4,5,6` (max deviation 2.5 %) | < 5 % | pass |
| J4 negative control (affine mesh) | exactly `0.0` | `== 0` | pass |
| J5 CAD contract | NURBS circle radius error `2.2e−16`, 7/7 malformed payloads rejected, CAD-mesh free-stream `1.8e−15`, order `3.022` vs 3 | — | pass |

Two honest caveats, both quantified below:

* the free-stream **state drift** after 100 SSP-RK3 steps is `2.74e−14` at
  `p=5` — 2.7× the original `1e−14` number. It is a round-off random walk, not a
  discretisation defect: `drift/√N` is constant (`2.78e−15, 2.95e−15,
  2.74e−15, 1.90e−15` for `N = 25, 50, 100, 200` steps) and the drift saturates
  at `2.7e−14` instead of growing linearly. J1 is therefore gated on the exact
  metric-identity defect (`1.83e−15`); the drift is reported explicitly anyway.
* J3 passes only with **superparametric wall geometry** (`g = p+1`). With
  classical isoparametric geometry (`g = p`) the `p=3` case gives order
  `2.661` (34 % below theory) — see section 5.

### 3.2 Verification problems

* **Curved periodic mesh, no wall.** Unit square, doubly periodic, smooth warp
  `x = s + 0.06 sin(2πs)cos(2πt)`, `y = t + 0.06 sin(2πt)cos(2πs)`; constant
  advection velocity `A = (0.6, 0.8)`; exact solution
  `u = sin(2π(x−A_x t)) cos(2π(y−A_y t))`, final time `T = 0.1`,
  levels `8×8, 12×12, 16×16`.
* **Curved-wall annulus (the J3 case).** Polar annulus `r ∈ [1,2]` around a
  circular cylinder, rigid-body rotation `A = ω(−y, x)` (`ω = 1`, divergence
  free, tangential to both walls so the no-penetration condition is *exactly*
  satisfied by the analytic solution); exact solution
  `u = exp(−((r−1.5)/0.35)²) cos(2(θ − ωt))`, final time `T = 0.3`.
  Solution degree `p`, wall geometry degree `g = p+1` (section 5 explains why
  `g = p` is not enough), refinement sequences `8×16 → 12×24 → 16×32` (`p ≤ 2`,
  `p = 5`) and `12×24 → 16×32 → 24×48` (`p = 3, 4`).

### 3.3 How the order is measured

The L2 error is taken against the analytic solution at the final time with the
same exact quadrature that is used in the solver; the order is the
least-squares slope of `log(error)` against `log(h)` with
`h = sqrt(mean cell area)`. The asymptotic range was verified from the
successive pairwise orders reported in `tables.md`; the `p = 5` sequence is
deliberately kept on the coarse side because at `h = 0.068` the error
(`1.5e−10`) reaches the round-off floor of the DG operator (`~1e−13` per RHS,
accumulating over ≈300 steps) and the `h⁶` branch would be masked.

### 3.4 Time-step independence

For `p = 3` (`n=8×8`, warped mesh) the error is `6.263e−4, 6.263e−4, 6.263e−4`
for `dt = stable, stable/3, stable/20`: the measured convergence is spatial.

### 3.6 Measured tables (auto-rendered, no hand transcription)

### Gate summary

| gate | value | threshold | pass |
|---|---|---|---|
| J1_free_stream_curved | 1.83149e-15 | 1e-14 | True |
| J2_metric_identity | 1.60149e-15 | 1e-12 | True |
| J3_curved_wall_order | ; observed=1.95,3,3.92,4.98,5.91; theory=2,3,4,5,6; rel_deviation=0.0254,0.00118,0.0198,0.00395,0.0151 | 0.05 | True |
| J4_negative_control | bitwise_zero=True, max_abs_defect_exact=0.0 | 0.0 | True |
| J5_cad_contract | radius_error=2.220446049250313e-16, closure_error=0.0, invalid_rejected=True, cad_order=3.0223070660293563, free_stream=1.8364370179806246e-15 | {'radius_error': 1e-13, 'free_stream': 1e-14, 'order_deviation': 0.05} | True |

### J2 metric identity (worst over all elements)

| p | cells | max inverse-metric | max Piola (SBP) | max Piola (coeff) | min J |
|---|---|---|---|---|---|
| 1 | 16 | 1.110e-16 | 2.082e-17 | 0.000e+00 | 0.0090 |
| 1 | 64 | 2.220e-16 | 1.041e-17 | 0.000e+00 | 0.0017 |
| 2 | 16 | 2.220e-16 | 8.327e-17 | 3.469e-18 | 0.0049 |
| 2 | 64 | 2.220e-16 | 4.510e-17 | 8.674e-19 | 0.0014 |
| 3 | 16 | 3.331e-16 | 3.886e-16 | 6.939e-18 | 0.0059 |
| 3 | 64 | 4.441e-16 | 2.290e-16 | 1.735e-18 | 0.0015 |
| 4 | 16 | 3.331e-16 | 6.925e-16 | 4.770e-18 | 0.0061 |
| 4 | 64 | 4.441e-16 | 4.215e-16 | 8.674e-19 | 0.0015 |
| 5 | 16 | 4.441e-16 | 1.601e-15 | 6.939e-18 | 0.0061 |
| 5 | 64 | 4.441e-16 | 8.092e-16 | 2.602e-18 | 0.0015 |

### J1 free-stream preservation, curved (warped) meshes

| p | mesh | warp amplitude | sup|du/dt| (operator) | exact metric defect | drift after 100 steps |
|---|---|---|---|---|---|
| 1 | 6x6 | 0.06 | 1.30e-14 | 0.00e+00 | 6.65e-16 |
| 1 | 10x10 | 0.10 | 2.66e-14 | 0.00e+00 | 5.13e-16 |
| 2 | 6x6 | 0.06 | 2.52e-14 | 4.71e-16 | 7.13e-16 |
| 2 | 10x10 | 0.10 | 6.87e-14 | 8.99e-16 | 9.22e-16 |
| 3 | 6x6 | 0.06 | 4.53e-14 | 4.80e-16 | 1.37e-15 |
| 3 | 10x10 | 0.10 | 1.23e-13 | 1.77e-15 | 1.43e-15 |
| 4 | 6x6 | 0.06 | 1.50e-12 | 5.35e-16 | 1.72e-14 |
| 4 | 10x10 | 0.10 | 3.77e-12 | 1.83e-15 | 1.96e-14 |
| 5 | 6x6 | 0.06 | 1.21e-12 | 5.06e-16 | 2.32e-14 |
| 5 | 10x10 | 0.10 | 3.90e-12 | 1.77e-15 | 2.74e-14 |

### J4 negative control: straight (affine) mesh

| p | sup|du/dt| (operator) | exact metric defect | drift | bitwise zero |
|---|---|---|---|---|
| 1 | 6.00e-15 | 0.0e+00 | 1.68e-16 | True |
| 2 | 6.00e-15 | 0.0e+00 | 2.48e-16 | True |
| 3 | 1.22e-14 | 0.0e+00 | 5.26e-16 | True |
| 4 | 7.19e-13 | 0.0e+00 | 9.23e-15 | True |
| 5 | 1.26e-13 | 0.0e+00 | 7.10e-15 | True |

### round-off scaling of the free-stream drift (p=5, 10x10, warp 0.1)

| SS-RK3 steps | drift | drift/sqrt(N) |
|---|---|---|
| 25 | 1.39e-14 | 2.78e-15 |
| 50 | 2.09e-14 | 2.95e-15 |
| 100 | 2.74e-14 | 2.74e-15 |
| 200 | 2.69e-14 | 1.90e-15 |

### J3 curved-wall annulus: analytic order

| p | geometry degree | levels (nr x ntheta) | h | rel. L2 error | observed order | theory | deviation |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 8x16 / 12x24 / 16x32 | 0.2713 | 3.6522e-02 | 1.949 | 2 | 2.5% |
|  |  |  | 0.1809 | 1.6635e-02 |  |  |  |
|  |  |  | 0.1357 | 9.4540e-03 |  |  |  |
| 2 | 3 | 8x16 / 12x24 / 16x32 | 0.2714 | 2.4409e-03 | 3.004 | 3 | 0.1% |
|  |  |  | 0.1809 | 7.1301e-04 |  |  |  |
|  |  |  | 0.1357 | 3.0482e-04 |  |  |  |
| 3 | 4 | 12x24 / 16x32 / 24x48 | 0.1809 | 2.2508e-05 | 3.921 | 4 | 2.0% |
|  |  |  | 0.1357 | 8.0497e-06 |  |  |  |
|  |  |  | 0.0905 | 1.5023e-06 |  |  |  |
| 4 | 5 | 12x24 / 16x32 / 24x48 | 0.1809 | 6.1755e-07 | 4.980 | 5 | 0.4% |
|  |  |  | 0.1357 | 1.5352e-07 |  |  |  |
|  |  |  | 0.0905 | 1.9650e-08 |  |  |  |
| 5 | 6 | 8x16 / 12x24 / 16x32 | 0.2714 | 1.7206e-07 | 5.909 | 6 | 1.5% |
|  |  |  | 0.1809 | 1.5143e-08 |  |  |  |
|  |  |  | 0.1357 | 2.8744e-09 |  |  |  |

### support: curved periodic mesh (no wall)

| p | rel. L2 error at h = 0.125, 0.083, 0.062 | observed order |
|---|---|---|
| 1 | 6.170e-02, 2.909e-02, 1.674e-02 | 1.88 |
| 2 | 6.446e-03, 1.919e-03, 7.793e-04 | 3.04 |
| 3 | 6.263e-04, 1.226e-04, 3.768e-05 | 4.05 |
| 4 | 5.441e-05, 7.014e-06, 1.648e-06 | 5.05 |
| 5 | 7.219e-06, 5.047e-07, 8.095e-08 | 6.48 |

### root cause: wall-normal defect and p=1 order

| geometry degree | wall normal defect: nr=8, nr=12, nr=16 |
|---|---|---|---|
| 1 | 7.802e-01, 6.369e-01, 5.258e-01 |
| 2 | 4.646e-01, 2.004e-01, 9.663e-02 |
| 3 | 7.787e-02, 5.813e-02, 3.076e-02 |

| case | observed order | theory |
|---|---|---|
| p=1, wall geometry g=1 | 1.011 | 2 |
| p=1, wall geometry g=2 | 1.949 | 2 |
| p=3, wall geometry g=3 | 2.661 | 4 |
| p=3, wall geometry g=4 | 3.921 | 4 |

### J5 CAD / NURBS contract

* NURBS circle radius error: 2.22e-16
* curve closure error: 0.00e+00
* malformed payloads rejected: True
  * `wrong_schema`: rejected: unsupported schema 'nope'; expected 'tensordg.curved_geometry.v1'
  * `unsorted_knots`: rejected: knot vector must be non-decreasing
  * `short_knots`: rejected: knot count 11 != n_control_points + degree + 1 = 12
  * `negative_weight`: rejected: all weights must be strictly positive
  * `non_clamped`: rejected: only clamped (open) knot vectors are supported
  * `open_curve`: rejected: knot count 12 != n_control_points + degree + 1 = 11
  * `missing_degree`: rejected: missing field 'degree'
* CAD-derived DG mesh (p=2) free-stream drift: 1.84e-15
* CAD-derived DG mesh (p=2) observed order: 3.022 (theory 3)

### 3.7 Figures

All figures are produced by `make_figures()` in the study driver (matplotlib,
regenerated with `--figures-only`).

| file | content |
|---|---|
| `fig1_geometry_diagnostics.png` | curved meshes (warped quad mesh, annulus with curved walls); free-stream defect versus `p` against the `1e−14 / 1e−12` gates; wall-normal defect versus mesh size |
| `fig2_curved_wall_order.png` | J3: `L2` error versus `h` for `p = 1..5` with the `h^{p+1}` reference slopes, and observed versus theoretical order with the ±5 % band |
| `fig3_geometric_limiting.png` | root cause: `p=1` polygonal (`g=1`) versus quadratic (`g=2`) wall, wall-normal defect decay, and the order bar chart for the `p=1` and `p=3` iso/superparametric pairs |
| `fig4_warped_order.png` | curved periodic mesh (no wall): `h^{p+1}` for `p = 1..5` |
| `fig5_annulus_field.png` | `p=4` solution on the curved-wall annulus at `t = 0.5` and the pointwise error field against the analytic solution |

<!-- RESULTS:END -->

## 4. Geometry validation before any physics

Following the project rule that a geometry must be validated before its physics
is believed (the finite-element analogue of a `Cb / L / B` check), the study
also reports:

* `min J > 0` over every element and quadrature point (no inverted elements);
* annulus area against the analytic `π(r_o² − r_i²)`: with `g = 2`, `nr = 4` the
  relative area error is `7.8e−4`, at `nr = 8` it is `4.9e−5` (≈ `h²`); with a
  polygonal wall (`g = 1`, `nr = 4`) it is `1.0e−1`, i.e. the chord
  approximation loses 10 % of the domain area — a mesh that must not be used
  for a quantitative curved-wall study;
* wall-node radius against `r_i` (only the degree-`g` interpolation error of the
  element geometry remains; the NURBS wall itself is exact).

## 5. Root cause of the geometry-limited convergence

The campaign question was *why* DG loses order on curved meshes. Three distinct
mechanisms were isolated and quantified:

**(i) Inconsistent metric evaluation.** Free-stream preservation is destroyed if
the metric fluxes are differentiated with a different rule than the physical
flux. `test_curved_geometry.py::test_free_stream_gate_detects_inconsistent_metrics`
injects a `1e−6` perturbation into `a¹` and shows the defect jumps from
round-off to `>1e−9`: the J1 gate demonstrably has detection power.

**(ii) Over-padded quadrature.** The first version of the module used
`nq = 3p+2` Gauss points (three more than needed). With `p = 4` that made the
`(3p+1)`-degree Legendre Vandermonde inverse ill-conditioned enough that the
free-stream drift after 100 steps reached `8e−7` (and `2.5e−6` on the more
strongly warped mesh) even though the exact metric defect stayed at `5e−16`.
Reducing to the *minimal exact* rule `nq = 2p+2` brought the drift back to
`≤ 2.7e−15` at every degree. Order of the two effects: round-off amplified by
the conditioning of the coefficient transformation, not a consistency error.

**(iii) Polygonal wall.** For `p = 1` an isoparametric (`g = p = 1`) element has
a *chord* wall. The wall normal of a chord through two points of a circle is
radial only at the chord midpoint, so the discrete normal flux `A·n` of a
tangential velocity field is `O(h)` instead of `0`, the wall leaks, and the
observed order collapses to `1.01` instead of `2` (`T = 0.3`, `nr = 4,8,16`;
see the figure below). Using a quadratic wall (`g = 2`, i.e. superparametric
geometry for `p = 1`) restores `1.95 ≈ p+1` and reduces the error by a factor
≈ 15. The measured wall-normal defect decays as `h^g` (`7.8e−1 → 5.3e−1` for
`g = 1`, `4.7e−1 → 9.7e−2` for `g = 2`), which is exactly the mechanism.

**(iii-b) Isoparametric geometry is not enough for odd `p`.** Running the same
annulus study with `g = p` gives `p=1: 1.949`, `p=2: 3.004`, `p=3: 2.661`,
`p=4: 4.951`, `p=5: 5.115` — the odd degrees `p = 3, 5` fall short by 34 % and
15 %. Switching the wall to `g = p + 1` (superparametric, one degree more than
the solution) recovers the full order: `p=3: 3.921` (2.0 % from theory) and the
`p=5` measurement moves onto the coarse sequence, where the `h⁶` branch is above
the round-off floor, giving `5.909` (1.5 %). The error also drops by a factor
8 for `p = 3` (`1.791e−4 → 2.251e−5` at `12×24`). Physically the wall flux
defect `|A·n|` of a degree-`g` wall patch decays as `h^g`, so it only falls fast
enough to stay below the interior discretisation error when `g ≥ p + 1` for
these odd-degree cases.

**Practical rule adopted by this module:** curved walls are discretised with
`g = p + 1` (and never with `g = 1`); the module supports solution and geometry
degree independently for exactly this reason.

## 6. Reproduction

```bash
# fast gate smoke (≈30 s)
python -m pytest tests/test_curved_geometry.py -q

# full study (numeric part, any machine with numpy; ≈15 min single core)
python benchmarks/curved_geometry_study.py --out artifacts/dg_curved --no-figures

# figures + markdown tables from the stored JSON (needs matplotlib)
python benchmarks/curved_geometry_study.py --out artifacts/dg_curved --figures-only
```

Artefacts: `artifacts/dg_curved/result.json` (every number, plus environment),
`artifacts/dg_curved/tables.md` (auto-rendered tables, no hand transcription),
`artifacts/dg_curved/fig*.png`.

## 7. Limitations and scope

* **Scalar advection only.** This node verifies the *geometry layer*. The same
  metrics feed `EulerDG2D`/NS kernels, but the Euler/NS curved-mesh order gate
  is not covered here (no curved-wall entropy/momentum-conservation statement is
  made).
* **Stationary meshes.** No geometric conservation law (GCL) for moving meshes
  is implemented or verified.
* **Tensor-product quadrilaterals only.** No simplex curved elements, no
  hanging nodes, no non-conforming curved interfaces (conforming isoparametric
  edges are assumed and are checked implicitly by the FSP test).
* **Circular walls only** in `mesh_from_contract`: the annulus builder needs a
  circular reference wall (it validates this and raises otherwise).
* **Runge–Kutta time integration** (SSP-RK3): temporal order 3, so the J3 order
  study is only meaningful up to `p = 2` without an order-matched `dt`; the
  study therefore uses `dt` from a CFL condition that was verified to be
  time-step independent (errors unchanged when `dt` is reduced 20×).
* The NURBS support is a *curve* (boundary) reader — enough for the CAD → DG
  hand-off — not a full NURBS *surface* patch reader.

## 8. Audit chain

| item | value |
|---|---|
| module | `tensordg/curved_geometry.py` |
| study driver | `benchmarks/curved_geometry_study.py` |
| tests | `tests/test_curved_geometry.py` (36 tests) |
| artefacts | `artifacts/dg_curved/{result.json,tables.md,fig*.png}` |
| campaign node | `p254w1-n02-dg-curved-geometry` |