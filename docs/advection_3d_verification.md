# 3-D tensor-product DG advection: verification page

This page documents `tensordg/advection_3d.py`, the 3-D extension of the
shipped 1-D/2-D periodic advection verification surface, and the machine-checked
ladder that certifies it.

## 1. What is verified

| item | value |
|---|---|
| PDE | `u_t + a·grad u = 0` on the periodic unit cube |
| discretisation | modal tensor-product Legendre DG, weak form, upwind face flux, classical RK4 |
| mesh | uniform Cartesian hexahedra, `(nx, ny, nz)` independently chosen |
| exact solution | `1 + sin(2π(x-a_x t)) sin(2π(y-a_y t)) sin(2π(z-a_z t))` |
| mode convention | `[0,0,0]` is the cell average (mass = mean of that mode) |

The reference is a closed form evaluated by Gauss quadrature, not a stored
numeric profile: a regression that merely rescales the solution cannot pass.

## 2. How it is checked

`python -m benchmarks.advection_3d` runs a refinement ladder per polynomial
degree and writes, under `docs/validation/advection_3d/`:

* `metrics.json` — configuration, provenance (commit, torch/numpy/python),
  per-level `h`, relative/absolute L2 error, mass drift, time step and observed
  order, plus every gate with its threshold and status;
* `convergence.csv` — the same numbers in flat form;
* `summary.md` — tables rendered from the measurements at generation time
  (**this file, not this page, is the authoritative numeric record**);
* `advection_3d_convergence.png` — log-log error vs `h` with the design slopes
  and the mid-plane error map of the finest run.

Gates (exit status is 0 only if all pass):

| gate | threshold |
|---|---|
| `design_order_P{p}` | observed order of the finest pair within 15 % of `p + 1` |
| `mass_conservation_P{p}` | relative drift of the analytic integral (`1`) below `1e-12`, i.e. round-off, for every velocity sign |
| `documented_case_below_3pct_P{p}` | the documented `8×8×8` P1/P2 runs below the ecosystem's 3 % error gate |

`pytest tests/test_advection_3d.py` re-checks the same invariants on a cheaper
configuration: mass conservation for positive and sign-changing velocities,
P1/P2 design order, design order of the initial L2 projection alone
(`final_time = 0`), the documented 3 % case, P0 first-order behaviour on a
stretched mesh, the zero-velocity short circuit, the closed-form mean/translation
identities, and argument validation.

## 3. Why it is worth having

* Every production DG solver runs on 3-D meshes; the previous verification
  surface stopped at 2-D, so 3-D mode indexing (the classic place where a
  swapped mode axis silently degrades the order) had no machine-checked
  coverage at all.
* Conservation in 3-D depends on the face traces telescoping in all three
  directions; the mass gate is sensitive to a missing roll in any single axis.
* The separable construction of the initial condition is `O(N(p+1))` instead of
  `O((N(p+1))³)`, and the error norm is accumulated in `x`-slabs, so the ladder
  is usable on modest hardware (a full ladder runs in about a minute on one
  CPU core).

## 4. Known limitations

* Uniform Cartesian hexahedral meshes only — no curved or unstructured 3-D
  geometry (curved 2-D geometry is covered by `tensordg/curved_geometry.py`).
* Linear advection only; 3-D Euler/Navier–Stokes remain future work.
* `p ≤ 3` (the modal operators are built for that range), CPU/float64 gates.
* The P0 first-order check is deliberately loose (`0.6 < EOC < 1.4`) because a
  first-order upwind scheme on the coarse stretched ladder is still
  pre-asymptotic; the gate exists to catch a *second*-order regression, not to
  certify the asymptotic constant.
* Time integration is classical RK4 with a `cfl = 0.08` default; the ladder
  records `dt` and `steps` per level, so a temporal-order regression is visible
  in the artifact even though it is not a separate gate.