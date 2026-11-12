# DG advection benchmark registry — ice-edge transport reference (IN12-P5)

**Owner repository:** TensorDG (DG formulations and their numerical verification).
**Campaign node:** `IN12-P5` (纵贯军 P5 · ice-edge transport high-order reference).
**Artifact:** `validation/dg_advection_bench/dg_advection_bench.json` (machine-readable record).
**Report:** `docs/reports/ice_edge_transport_dg_bench_IN12-P5.md` (paper-level).
**Tests:** `tests/test_dg_advection_bench.py` (15 assertions, ~50 s with `-n 8`).

## 1. Scope and status

This node fixes a **route-level verification reference** for the transport
operator used by an ice-condition assimilation cycle (`A2`) and by passive
frazil transport: a divergence-free advection problem on the periodic unit
square, discretised with high-order discontinuous Galerkin.

* Evidence level: **REAL** — a working DG solver (`tensordg.advection_2d_var`),
  an independently cross-checked operator, measured orders of accuracy against
  analytic/reentrant solutions, and mass conservation at round-off.
* What this node does **not** claim: no ice thermodynamics, no frazil source
  terms, no assimilation coupling, no operational validation. It is a numerical
  reference to be cited when a transport scheme is upgraded (e.g. p = 1 → p = 3)
  inside an assimilation scheme.
* Physics domain: rotating front / convection-dominated transport
  (solid-body rotation peak preservation, reentrant deformational vortex).

## 2. Software interface

| Module | Public API | Purpose |
|---|---|---|
| `tensordg.advection_2d_var` | `DGAdvection2D(degree, ncells)`, `.project()`, `.rhs()`, `.solve()`, `.l2_error()`, `.l2_difference()`, `.mass()`, `.peak()` | variable-coefficient (`a = a(x,y,t)`) divergence-free DG advection, upwind flux, RK4 |
| `tensordg.advection_2d_var` | `solid_body_rotation()`, `reentrant_vortex()`, `trig_profile()`, `gaussian_bump()`, `rotated_profile()` | reference velocity fields and initial/exact profiles |
| `tensordg.ice_edge_bench` | `CASES`, `CRITERIA`, `run_order_study()`, `run_revolution_check()`, `run_vortex_peak_check()`, `mass_balance_trace()`, `observed_order()` | benchmark case definitions, metrics and acceptance criteria (single source of truth) |
| `tools/run_dg_advection_bench.py` | CLI (`--quick`, `--jobs N`) | campaign run → artifact + figures |
| `tools/render_bench_tables.py` | CLI | markdown tables injected into the docs (no hand-copied numbers) |

Discretisation (identical normalisation to `tensordg.advection_2d`, mode
`[0,0]` = cell average):

```
dc_kl/dt = (2k+1)(2l+1)/(h_x h_y) * ( ∫_K (a u)·∇(φ_k φ_l) dV − ∫_∂K (a·n) û φ_k φ_l dS )
time: classical RK4,  dt = cfl / ((2p+1)(|a_x|_max/h_x + |a_y|_max/h_y)),  cfl = 0.3
```

## 3. Benchmark cases

| key | velocity | initial profile | exact solution | final time | norm |
|---|---|---|---|---|---|
| `shear_transport` | `a = (−2π(y−1/2), 0)` (sign-changing upwind) | `1 + sin 2πx sin 2πy` | analytic: `u0(x − a_x(y)t, y)` | 0.05 | full periodic domain |
| `solid_body_rotation_front` | `a = 2π(−(y−1/2), x−1/2)` | Gaussian ice-edge front, centre (0.5, 0.35), σ = 0.075 | analytic: initial field rotated by `−2πt` | 0.5 (order), 1.0 (peak) | disk r ≤ 0.28 about (0.5, 0.5) |
| `reentrant_deformational_vortex` | `ψ = (1/π) sin²πx sin²πy cos πt`, `a = (−ψ_y, ψ_x)` | `1 + sin 2πx sin 2πy` | reentrant: `u(T) = u(0)` exactly | 1.0 | full periodic domain |

Rotation speed `ω = 2π` ⇒ one revolution per unit time, so `t = 1` is a
**reentrant** check in which the exact solution is the initial field.

## 4. Acceptance criteria (machine checkable)

| criterion | value | where enforced |
|---|---|---|
| observed order of accuracy | `≥ p + 0.9` for p = 1, 2 (p = 3 reported) | `CRITERIA["order_margin"]`, `verdict.orders_pass` |
| global mass conservation | relative drift ≤ `1e-12` | `CRITERIA["mass_tolerance"]`, `verdict.mass_pass` |
| peak retention, one revolution | `≥ 0.99` (p = 3) | `CRITERIA["peak_tolerance"]`, `verdict.peak_pass` |
| pytest | `≥ 5` tests | `tests/test_dg_advection_bench.py` (15 collected) |

Observed order = least-squares slope of `log(err)` vs `log(n_cells)`; pairwise
rates are also stored for diagnostics. The acceptance test uses the least-squares
slope because it is the standard ("observed order") estimate and is less
sensitive to pre-asymptotic behaviour on the coarsest mesh.

## 5. Results (injected from `validation/dg_advection_bench/dg_advection_bench.json`)

<!-- BEGIN GENERATED TABLES （由 tools/render_bench_tables.py 生成） -->

### Observed order of accuracy (least-squares fit of log err vs log n)

| case | p | n cells | L2 errors | observed order | required (p+0.9) | verdict |
|---|---|---|---|---|---|---|
| periodic shear (analytic) | 1 | [12, 16, 24] | 1.031e-02 / 5.859e-03 / 2.624e-03 | **1.98** | 1.9 | PASS |
| periodic shear (analytic) | 2 | [8, 12, 16] | 1.738e-03 / 5.094e-04 / 2.142e-04 | **3.02** | 2.9 | PASS |
| periodic shear (analytic) | 3 | [8, 12, 16] | 9.975e-05 / 1.956e-05 / 6.183e-06 | **4.01** | 3.9 | PASS (reported) |
| rotating front (disk norm) | 1 | [16, 24, 32] | 1.206e-02 / 4.795e-03 / 2.403e-03 | **2.32** | 1.9 | PASS |
| rotating front (disk norm) | 2 | [12, 16, 24] | 2.203e-03 / 8.569e-04 / 2.332e-04 | **3.24** | 2.9 | PASS |
| rotating front (disk norm) | 3 | [12, 16, 24] | 2.484e-04 / 7.946e-05 / 1.547e-05 | **4.01** | 3.9 | PASS (reported) |
| reentrant vortex | 1 | [12, 16, 24] | 1.764e-02 / 8.745e-03 / 3.257e-03 | **2.44** | 1.9 | PASS |
| reentrant vortex | 2 | [12, 16, 24] | 1.558e-03 / 7.030e-04 / 1.969e-04 | **2.99** | 2.9 | PASS |
| reentrant vortex | 3 | [10, 14, 20] | 3.918e-04 / 9.885e-05 / 2.456e-05 | **4.00** | 3.9 | PASS (reported) |

### Pairwise rates, mass drift and step counts

| case | p | pairwise orders | max relative mass drift | RK4 steps |
|---|---|---|---|---|
| periodic shear (analytic) | 1 | 1.97 / 1.98 | 0.00e+00 | 19, 25, 37 |
| periodic shear (analytic) | 2 | 3.03 / 3.01 | 2.22e-16 | 21, 31, 41 |
| periodic shear (analytic) | 3 | 4.02 / 4.00 | 1.11e-16 | 29, 43, 57 |
| rotating front (disk norm) | 1 | 2.27 / 2.40 | 3.93e-16 | 482, 723, 964 |
| rotating front (disk norm) | 2 | 3.28 / 3.21 | 1.96e-16 | 603, 803, 1205 |
| rotating front (disk norm) | 3 | 3.96 / 4.04 | 3.93e-16 | 843, 1124, 1686 |
| reentrant vortex | 1 | 2.44 / 2.44 | 3.33e-16 | 237, 316, 474 |
| reentrant vortex | 2 | 2.77 / 3.14 | 2.22e-16 | 395, 527, 790 |
| reentrant vortex | 3 | 4.09 / 3.90 | 4.44e-16 | 461, 645, 922 |

### Peak preservation after one revolution (exact solution = initial field)

| case | p | n | steps | peak initial | peak final | retention | rel. L2 | mass drift |
|---|---|---|---|---|---|---|---|---|
| rotating front | 1 | 24 | 1446 | 1.01607 | 0.95365 | **0.93856** | 5.953e-02 (disk) | 3.9e-16 |
| rotating front | 2 | 16 | 1606 | 0.99924 | 0.99324 | **0.99399** | 8.700e-03 (disk) | 2.0e-16 |
| rotating front | 3 | 16 | 2248 | 0.99815 | 0.99906 | **1.00091** | 6.875e-04 (disk) | 3.9e-16 |
| rotating front, res. study | 3 | 8 | 1124 | 1.00223 | 0.99507 | **0.99286** | 1.149e-02 (disk) | 5.9e-16 |
| rotating front, res. study | 3 | 12 | 1686 | 0.99982 | 0.99722 | **0.99739** | 2.232e-03 (disk) | 9.8e-16 |
| rotating front, res. study | 3 | 20 | 2810 | 0.99951 | 0.99979 | **1.00028** | 2.791e-04 (disk) | 0.0e+00 |
| reentrant vortex | 2 | 12 | 395 | 0.99247 | 0.99245 | **0.99998** | 6.017e-03 (global) | 2.2e-16 |
| reentrant vortex | 3 | 12 | 553 | 0.99727 | 0.99685 | **0.99958** | 6.238e-04 (global) | 4.4e-16 |
| reentrant vortex | 3 | 16 | 738 | 1.00019 | 1.00026 | **1.00007** | 2.023e-04 (global) | 2.2e-16 |

### Global mass balance (all runs)

Max relative drift over every run of the campaign: **9.817e-16** (acceptance 1e-12).
Traced case: p=2, n=12, 395 RK4 steps, final drift 2.220e-16.

### Environment and provenance

```
{"python": "3.11.6", "torch": "2.7.1", "numpy": "2.4.6", "device": "cpu",
 "dtype": "float64", "machine": "loongarch64",
 "platform": "Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38"}
commit ee93147 | config full | generated 2026-11-07T01:04:35+00:00
```

<!-- END GENERATED TABLES -->

## 6. Reproduce

```bash
# full campaign run (~90 s wall with 28 workers; writes artifact + 4 figures)
python tools/run_dg_advection_bench.py --jobs 28
# fast smoke configuration
python tools/run_dg_advection_bench.py --quick --jobs 8
# acceptance tests
python -m pytest tests/test_dg_advection_bench.py -q -n 8
# regenerate the tables above from the artifact
python tools/render_bench_tables.py
```

`validation/dg_advection_bench/dg_advection_bench.json` schema (top level):
`benchmark, campaign_node, repo, branch, commit, generated_utc, config,
environment, purpose, discretisation, criteria, cases{case}{degree},
order_summary[], peak_preservation{}, mass_balance{}, verdict{},
figures[], limitations[]`.

## 7. Known limitations

1. **Non-periodic rotation seam.** `a = 2π(−(y−1/2), x−1/2)` is discontinuous at
   the periodic seam (`a_y` jumps by `2π`), which creates a thin O(1) error layer
   there. The rotation order is therefore measured inside the disk `r ≤ 0.28`
   (characteristics are circles, so seam noise stays at its own radius); the
   full-domain norm of the same run is stored alongside
   (`relative_l2_global` in `peak_preservation`).
2. **Smooth fronts only.** All profiles are C^∞. Sharp (C⁰/discontinuous) ice
   edges need limiters (`tensordg.limiters`, TVB) and are out of scope here.
3. **Transport operator only.** No frazil thermodynamics, no ice concentration
   coupling, no data assimilation; this is a numerical reference for the
   advection step, not an engineering model.
4. **Fixed flow amplitude.** The vortex uses the classical LeVeque/Nair
   amplitude `A = 1/π` (max speed 1); a stronger (`A = 1`) variant is not part of
   the acceptance set.
5. **Order estimator.** The reported order is a least-squares slope over three
   resolutions; the finest pairwise rate is `2.77` for vortex p = 2 and only
   reaches `3.14` on the finest pair — evidence that the coarse members are
   pre-asymptotic. Both numbers are published.

## 8. References

* LeVeque, R. J. (1996). High-resolution conservative algorithms for advection in
  incompressible flow. *SIAM J. Numer. Anal.* 33(2), 627–665.
* Nair, R. D., & Machenhauer, B. (2002). The mass-conservative cell-integrated
  semi-Lagrangian advection scheme on the sphere. *Mon. Wea. Rev.* 130, 649–667.
* Williamson, D. L. et al. (1992). A standard test set for numerical
  approximations to the shallow water equations. *J. Comput. Phys.* 102, 211–224
  (solid-body rotation and deformational-flow standards).
* Cockburn, B., & Shu, C.-W. (1998). The local discontinuous Galerkin method for
  time-dependent convection-diffusion systems. *SIAM J. Numer. Anal.* 35, 2440–2463
  (upwind flux, RK4 stability bound used here).