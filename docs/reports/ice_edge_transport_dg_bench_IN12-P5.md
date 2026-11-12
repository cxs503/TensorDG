# Ice-edge transport high-order reference: rotating front and reentrant vortex in DG (IN12-P5)

**Node:** `IN12-P5` (纵贯军 P5) · **Repository:** TensorDG · **Branch:** `agent/m4933`
**Artifact:** `validation/dg_advection_bench/dg_advection_bench.json` · **Registry:** `docs/benchmarks/dg_advection_bench.md`
**Tests:** `tests/test_dg_advection_bench.py` (15 collected, all passing)
**Evidence level:** REAL (measured convergence orders, analytic/reentrant references, round-off mass balance)
**Scope statement:** route-level verification of a *transport operator*. No ice
thermodynamics, no frazil source terms, no assimilation coupling, no operational claim.

---

## 1. Case introduction (physical background, geometry, parameters)

An ice-condition assimilation cycle (A2) advances an ice-edge field with a
passive transport step; frazil is likewise advected passively. The two numerical
properties that decide whether such a step can be trusted are (i) the *order of
accuracy* of the advection operator on a smooth front and (ii) *exact
conservation of the transported quantity* (ice area / frazil mass). This node
builds the corresponding standards on the periodic unit square `[0,1]²` with a
divergence-free velocity field and a high-order nodal DG discretisation.

| parameter | value |
|---|---|
| domain | periodic unit square, uniform Cartesian mesh `n × n` |
| polynomial degree | `p = 1, 2, 3` (modal tensor-product Legendre) |
| quadrature | Gauss–Legendre, `nq = 2p+3` per direction |
| numerical flux | upwind, selected node-wise from the sign of the face-normal velocity |
| time integration | classical RK4, `dt = cfl/((2p+1)(|a_x|/h_x+|a_y|/h_y))`, `cfl = 0.3` |
| `shear_transport` | `a = (−2π(y−½), 0)`, `u0 = 1 + sin2πx sin2πy`, `t = 0.05` |
| `solid_body_rotation_front` | `a = 2π(−(y−½), x−½)`, Gaussian front centre (0.5, 0.35), `σ = 0.075`, `ω = 2π` (T = 1), `t = 0.5` (order) / `t = 1` (one revolution) |
| `reentrant_deformational_vortex` | `ψ = (1/π) sin²πx sin²πy · cos πt/T`, `T = 1`, `t = 1` |

Why these three: the shear case carries an **analytic** solution (periodic, with
the upwind direction changing sign inside the domain) and validates the
variable-coefficient operator; the **rotating front** is the classical
solid-body-rotation peak-preservation standard (convection dominated, reentrant
after one revolution); the **reentrant deformational vortex** (LeVeque / Nair)
is the strongest reentrant deformation standard whose exact flow map over
`[0,T]` is the identity, hence `u(T) ≡ u(0)` gives an exact error norm.

## 2. Method

1. Implemented `tensordg.advection_2d_var.DGAdvection2D`: weak conservative DG,
   mode `[0,0]` = cell average, upwind flux with node-wise sign selection, RK4.
   Semi-discrete form actually assembled:
   `dc_kl/dt = (2k+1)(2l+1)/(h_x h_y) [ ∫_K (a·u)∇(φ_kφ_l) − ∫_∂K (a·n)û φ_kφ_l ]`.
2. **Operator validation before any physics:**
   * constant-velocity degeneration against the pre-existing, independently
     written `tensordg.advection_2d.solve_advection_2d` → agreement `8.9e-16`
     (p=1), `1.6e-15` (p=2), `1.2e-15` (p=3);
   * uniform field is an exact invariant under divergence-free variable flow
     (`|rhs(constant)| = 8.5e-14`);
   * single-direction analytic shear tests with sign change (orders 1.99 / 2.99 /
     4.00) — these isolates located two implementation defects during
     development (a transposed face-velocity layout and a wrong face offset),
     both fixed before the reported numbers were produced.
3. Order study with three resolutions per degree, observed order = least-squares
   slope of `log(err)` vs `log(n)`; pairwise rates stored as diagnostics.
4. Peak retention `= peak(final)/peak(initial)` after exactly one revolution
   (exact solution = initial field).
5. Global mass `Σ c₀₀ h_x h_y` recorded at every RK4 step for one representative
   run and at the end of **every** run.
6. Everything is machine-produced: `tools/run_dg_advection_bench.py` writes the
   artifact and the four figures; `tools/render_bench_tables.py` injects the
   tables into this document (no hand-copied numbers). Parallel execution over
   worker processes (`--jobs`), 90 s wall for the full campaign.

## 3. Results

### 3.1 Convergence (criterion: observed order ≥ p + 0.9 for p = 1, 2)

| case | p | n cells | L2 errors | observed order | required | verdict |
|---|---|---|---|---|---|---|
| periodic shear (analytic) | 1 | [12,16,24] | 1.031e-02 / 5.859e-03 / 2.624e-03 | **1.98** | 1.9 | PASS |
| periodic shear (analytic) | 2 | [8,12,16] | 1.738e-03 / 5.094e-04 / 2.142e-04 | **3.02** | 2.9 | PASS |
| periodic shear (analytic) | 3 | [8,12,16] | 9.975e-05 / 1.956e-05 / 6.183e-06 | **4.01** | 3.9 | PASS (reported) |
| rotating front (disk norm) | 1 | [16,24,32] | 1.206e-02 / 4.795e-03 / 2.403e-03 | **2.32** | 1.9 | PASS |
| rotating front (disk norm) | 2 | [12,16,24] | 2.203e-03 / 8.569e-04 / 2.332e-04 | **3.24** | 2.9 | PASS |
| rotating front (disk norm) | 3 | [12,16,24] | 2.484e-04 / 7.946e-05 / 1.547e-05 | **4.01** | 3.9 | PASS (reported) |
| reentrant vortex | 1 | [12,16,24] | 1.764e-02 / 8.745e-03 / 3.257e-03 | **2.44** | 1.9 | PASS |
| reentrant vortex | 2 | [12,16,24] | 1.558e-03 / 7.030e-04 / 1.969e-04 | **2.99** | 2.9 | PASS |
| reentrant vortex | 3 | [10,14,20] | 3.918e-04 / 9.885e-05 / 2.456e-05 | **4.00** | 3.9 | PASS (reported) |

All nine studies reach the design order `p+1`: p = 1 → 1.98–2.44, p = 2 →
2.99–3.24, p = 3 → 4.00–4.01.

### 3.2 Peak preservation (rotating ice-edge front, one revolution)

| p | n | steps | peak initial | peak final | retention | rel. L2 (disk r ≤ 0.28) |
|---|---|---|---|---|---|---|
| 1 | 24 | 1446 | 1.01607 | 0.95365 | **0.93856** | 5.953e-02 |
| 2 | 16 | 1606 | 0.99924 | 0.99324 | **0.99399** | 8.700e-03 |
| 3 | 16 | 2248 | 0.99815 | 0.99906 | **1.00091** | 6.875e-04 |
| 3 | 8 | 1124 | 1.00223 | 0.99507 | 0.99286 | 1.149e-02 |
| 3 | 12 | 1686 | 0.99982 | 0.99722 | 0.99739 | 2.232e-03 |
| 3 | 20 | 2810 | 0.99951 | 0.99979 | **1.00028** | 2.791e-04 |

Under the reentrant deformational vortex (strong deformation, full cycle) the
peak is retained to `0.99998` (p=2, n=12), `0.99958` (p=3, n=12) and `1.00007`
(p=3, n=16): the upwind DG transport is dissipative but, at p ≥ 2, damps a
smooth front by less than 0.05 %.

### 3.3 Conservation

Global mass drift is at **round-off**: maximum relative drift over all 38 runs of
the campaign = `9.8e-16` (acceptance `1e-12`), and the step-by-step trace for the
vortex case (p=2, n=12, 395 RK4 steps) never exceeds `2.2e-16`. This is
structural, not accidental: the face flux is single-valued for both adjacent
cells, so the flux sums telescope exactly on the periodic mesh.

### 3.4 Figures

| figure | content |
|---|---|
| `docs/figures/dg_advection_convergence.png` | L2 error vs `n` for the three cases × p = 1,2,3 with `p+1` reference slopes |
| `docs/figures/dg_advection_peak_preservation.png` | peak retention after one revolution (left) and vs resolution (right) |
| `docs/figures/dg_advection_mass_balance.png` | relative mass drift per RK4 step with the `1e-12` line and double-precision `eps` |
| `docs/figures/dg_advection_rotation_fields.png` | initial front, DG p=3 solution after one revolution, numerical−exact field |

## 4. Error judgement against baselines (explicit "≤ 3 %?" answer)

| key quantity | computed | baseline | deviation | ≤ 3 %? |
|---|---|---|---|---|
| order, shear p=1 | 1.98 | theory `p+1 = 2` | 1.0 % | **yes** |
| order, shear p=2 | 3.02 | 3 | 0.7 % | **yes** |
| order, vortex p=1 | 2.44 | 2 | 22 % | **no** (superconvergent on these meshes; direction is favourable, criterion `≥1.9` still met) |
| order, vortex p=2 | 2.99 | 3 | 0.3 % | **yes** |
| order, rotating front p=1 | 2.32 | 2 | 16 % | **no** (favourable; disk-norm subdomain) |
| order, rotating front p=2 | 3.24 | 3 | 8 % | **no** (favourable; asymptotic set `[12,16,24]`, finest pair 3.21) |
| order, p=3 (all three cases) | 4.00 / 4.01 / 4.01 | 4 | ≤ 0.25 % | **yes** |
| peak retention, rotating front p=3, n=16 | 1.00091 | 1 (exact reentrancy) | 0.09 % | **yes** |
| peak retention, vortex p=3, n=16 | 1.00007 | 1 | 0.007 % | **yes** |
| L2 error, rotating front p=3, n=24 | 1.547e-05 | analytic solution (reference norm 1.33e-01) | 1.2e-04 relative | **yes** |
| global mass drift | 9.8e-16 relative | 0 (exact discrete conservation) | machine precision | **yes** |

The two "no" entries are *superconvergence* relative to the theoretical slope, not
accuracy failures: the acceptance criterion is one-sided (`order ≥ p + 0.9`) and
is met in all required degrees (p = 1, 2). They are listed rather than smoothed
over because the deviation from theory is larger than 3 %.

## 5. Notes: pitfalls, limitations, applicability

1. **Periodic seam of the rigid rotation (main pitfall).** `a = 2π(−(y−½), x−½)`
   is *not* periodic: `a_y` jumps from `−π` to `+π` across `x = 1`. A global L2
   norm then saturates at ~13 % relative error regardless of `p` and `n` (this was
   observed and diagnosed here). Since characteristics of a rigid rotation are
   circles, the seam layer stays at its own radius, so the error is measured in
   the disk `r ≤ 0.28` (front trajectory `r = 0.15`, seam `r ≥ 0.5`). The
   full-domain norm is stored in the artifact for transparency. **Do not quote
   the global norm of a rotating-front run on a periodic square.**
2. **Two real bugs found by the cross-checks** (kept as lessons): the face
   velocity arrays were laid out as `(ny, nx, nq)` instead of `(nx, ny, nq)`
   (invisible for constant or single-direction velocity, fatal for rotation), and
   the right/top face velocity was sampled at the *left/bottom* face
   (harmless for the transverse-independent rotation, wrong in general). The
   constant-velocity cross-check, the uniform-field invariant and the
   single-direction analytic shears are exactly the tests that expose both.
3. **Smooth profiles only.** Gaussian and trigonometric fronts (C^∞). A sharp
   ice edge (C⁰) requires limiters (TVB/`tensordg.limiters`), loses the formal
   order and must be validated separately — outside this node's scope.
4. **No physics coupling.** Passive transport of a scalar. Frazil
   thermodynamics, ice concentration source terms and assimilation increments are
   not modelled.
5. **Order estimate on three meshes.** Least-squares slope over three
   resolutions; vortex p=2 pairwise rates `2.77 → 3.14` show that the coarse
   pair is still pre-asymptotic. Both numbers are published.
6. **Architecture note.** All runs were executed on `loongarch64`
   (Python 3.11.6, torch 2.7.1, float64, CPU); results are deterministic
   (no RNG, no threading in the numerics), so re-runs reproduce bit-comparable
   numbers.

## 6. Reproduction

```bash
python tools/run_dg_advection_bench.py --jobs 28    # artifact + figures, ~90 s
python -m pytest tests/test_dg_advection_bench.py -q -n 8   # 15 passed, ~50 s
python tools/render_bench_tables.py                # regenerate the tables above
```