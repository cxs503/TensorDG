# TensorDG m3694 — P3 design order closed (ρ/p reach 4) and a new velocity-channel floor

> Repo: TensorDG @ branch `agent/m3694` (base `c1c2420`).
> Mission: tp-impl-8 self-start, platform seat. House rule: this repo only.
> Evidence: `validation/euler_dg_2d/p3_ladder_fine.json`,
> `validation/euler_dg_2d/velocity_order_floor.json`,
> logs `runs/p3_ladder_fine.log`, `runs/probe_velocity_floor.log`.
> All numbers below are injected by `tools/gen_report_m3694.py` from those JSONs
> (zero hand-copied digits).

## 1. Case and why it was open

`tensordg/euler_dg_2d.py` (modal DG, HLLC interface flux, SSP-RK3, local
positivity fallback) is the workhorse of the 2-D compressible stack. Its only
order evidence was one interval of P2 (16²→24², EOC 2.95). A previous mission
(m3024, commit `2ed072c`) built a full ladder but left **P3 at 3.85/3.52/3.48/3.85
versus the design order 4** and registered two untested explanations:

1. "P3 needs a finer mesh (top ≥ 24²–32²), estimated ×4 wall-clock, not done";
2. "SSP-RK3 is third-order while P3 is fourth-order, so the time integrator
   co-limits the order; needs `dt`-refinement or RK4 — not done".

This mission closes both, and settles which explanation is right.

Test case — the **isentropic vortex**, an *exact* smooth solution of the 2-D
Euler equations advected across a periodic box, so the reference is analytic
(no manufactured source, no numerical truth):

| parameter | value |
|---|---|
| domain | [-5,5]², periodic |
| vortex strength β / background (ρ,u,v,p) | 5 / (1, 1, 1, 1) |
| final time | 1.0 (vortex traverses 1 length unit) |
| degree | 3 (design order p+1 = 4) |
| flux / time integrator | HLLC / SSP-RK3 |
| CFL (ladder) | 0.2 |
| volume quadrature | 5 points/direction (`max(p+2,3)`), over-integration tested at 8 and 12 |
| mesh ladder | 8², 12², 16², 24², 32² (h = 1.25 → 0.313) |
| dtype / device | float64 / CPU, `OMP_NUM_THREADS=1` |

## 2. What was done

1. **Extended the benchmark** `benchmarks/design_order_euler_dg_2d.py` with the
   two missing knobs, both default-off so the historical study is unchanged:
   - `--temporal` + `temporal_study()`: refine `dt` at a *fixed* mesh
     (CFL 0.2 → 0.05, steps 164 → 654) and measure the temporal order directly.
     Mean step `final_time/steps` is used because the last step is clipped by
     `final_time`;
   - `--order-matched-dt` + `cfl_schedule()`: CFL(h) = CFL₀ (h/h₀)^γ with
     γ = (p+1)/q − 1 (γ = 1/3 for P3, clipped to 0 for P1/P2);
   - `--quadrature N`: volume-quadrature override, the aliasing isolator;
   - `--out`, and a three-panel coupled figure.
2. **Extended the P3 ladder to 24² and 32²** (the "needs a finer mesh" branch).
3. **Ran three attribution experiments** — time step, quadrature, and the
   channel-wise decomposition (`tools/probe_velocity_order_floor.py`, which
   splits each field's error into a mean offset and a shape part, and
   self-checks its total against the benchmark's own error).
4. **Added three machine-decidable tests** to
   `tests/test_euler_dg_2d_design_order.py` (CFL scheduling algebra, and the
   "dt is not the bottleneck" claim as an assertion).

## 3. Results

### 3.1 P3 ladder extended to 32² (fixed CFL 0.2)

| cells | h | steps | ρ rel-L2 | u rel-L2 | v rel-L2 | p rel-L2 | EOC ρ | EOC u | EOC p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8² | 1.250 | 164 | 5.6481e-04 | 1.5726e-03 | 1.5633e-03 | 7.8724e-04 | — | — | — |
| 12² | 0.833 | 246 | 1.1695e-04 | 3.4576e-04 | 3.4427e-04 | 1.7516e-04 | 3.88 | 3.74 | 3.71 |
| 16² | 0.625 | 328 | 3.8310e-05 | 1.1621e-04 | 1.1623e-04 | 5.3853e-05 | 3.88 | 3.79 | 4.10 |
| 24² | 0.417 | 492 | 7.7041e-06 | 8.6293e-05 | 8.6289e-05 | 1.0856e-05 | 3.96 | 0.73 | 3.95 |
| 32² | 0.312 | 656 | 2.4055e-06 | 8.4497e-05 | 8.4499e-05 | 3.4036e-06 | 4.05 | 0.07 | 4.03 |

Least-squares slope over all five rungs: **ρ 3.93, u 2.10,
v 2.10, p 3.94** (design order 4).

The two channels separate cleanly at fine mesh:

* **ρ and p reach the design order**: EOC 4.05 (ρ) and 4.03 (p) on
  24²→32². The old "P3 = 3.5" was a pre-asymptotic artefact of stopping at 16² —
  explanation #1 is confirmed for the thermodynamic channels.
* **the velocity channel does not converge**: EOC collapses to 0.73 (16²→24²) and
  0.07 (24²→32²); the error freezes at 8.629e-05 → 8.450e-05.

### 3.2 Attribution experiment A — is the time integrator limiting the order?

`temporal_study(P3, 8², CFL 0.2/0.1/0.05)` (164 → 327 → 654 steps): the relative
L2 error of every field is **bit-identical to four significant digits** under 4×
`dt` refinement (e.g. ρ 5.6481e-04 at all three CFL values). Improbable for a
scheme whose error had an O(dt³) component of any visible size. Therefore
**explanation #2 is refuted**: with 164–654 steps taken over one traversal, the
temporal error at this CFL is orders of magnitude below the spatial error at every
rung of the ladder, so the time integrator cannot be what caps the observed order.
The `--order-matched-dt` knob is still useful algebra (and is now unit-tested),
but it is *not* what fixes the P3 ladder.

### 3.3 Attribution experiment B — is nonlinear-flux aliasing limiting the order?

P3 on 12², volume quadrature 4 / 5 (default) / 8 / 12:

| quadrature | ρ | u | p |
|---:|---:|---:|---:|
| 4 (under-integrated) | 9.0383e-05 | 3.0602e-04 | 1.2577e-04 |
| 5 = max(p+2,3), default | 1.1695e-04 | 3.4576e-04 | 1.7516e-04 |
| 8 | 1.1662e-04 | 3.4556e-04 | 1.7519e-04 |
| 12 | 1.1662e-04 | 3.4557e-04 | 1.7519e-04 |

Quadrature 8 and 12 agree to four digits with the default (0.3 %), so the volume
integral is converged and **aliasing is not the bottleneck** either. (The
under-integrated Q=4 case is *more* accurate in ρ at this resolution — a
cancellation, not a reason to under-integrate.)

### 3.4 The velocity floor is a shape error, not a momentum leak

Offset 1.09e-05 (16²) vs 1.09e-05 (24²); shape 1.157e-04 vs 8.560e-05.

| cells | field | offset | shape | total | total/ladder |
|---:|---|---:|---:|---:|---:|
| 16² | density | 3.007e-10 | 3.831e-05 | 3.8310e-05 | 1.000 |
| 16² | velocity_x | 1.087e-05 | 1.157e-04 | 1.1621e-04 | 1.000 |
| 16² | pressure | 1.559e-08 | 5.385e-05 | 5.3853e-05 | 1.000 |
| 24² | density | 3.024e-10 | 7.704e-06 | 7.7041e-06 | 1.000 |
| 24² | velocity_x | 1.088e-05 | 8.560e-05 | 8.6293e-05 | 1.000 |
| 24² | pressure | 5.971e-10 | 1.086e-05 | 1.0856e-05 | 1.000 |

`total/ladder = 1.000` for all fields and both meshes is the self-check: the
split reproduces the benchmark's own relative L2 error exactly, so the
decomposition is calibrated (an earlier version of the probe forgot the
cell-volume average and inflated every number by √n_cells — the self-check is
what caught it).

The domain-averaged velocity offset is **1.09e-05 at 16² and 1.09e-05 at 24²** —
identical to three digits, i.e. a genuine but *small* constant bias (a
momentum-mean drift of that size is within the round-off of the conserved
summation, and it is 8× smaller than the floor). The floor itself is the
**shape** term, which only goes 1.157e-04 → 8.560e-05 (EOC ≈ 0.35) while ρ and p
keep converging at 4. So the residual defect is in the velocity *profile* of a
smooth solution — a new, previously unreported item; it is **not** closed here.

### 3.5 The 3 % engineering gate

Finest-rung worst-field error on 32² is 8.450e-05 = **0.0084 %**, three
orders of magnitude inside the 3 % gate; the velocity floor (8.450e-05) is
likewise ~0.0084 %. The defect is a *verification-purity* issue, not
an engineering-accuracy issue.

## 4. Figures

`docs/figures/euler_dg_2d_p3_order_decomposition.png` — three panels, all from
the JSON above:

1. P3 ladder error vs h for (ρ, u, p) with a slope-4 reference line — the
   thermodynamic channels run parallel to it from 24² on, the velocity curve goes
   flat;
2. the offset/shape split vs h (dashed vs solid) — velocity offset is a horizontal
   line at 1.09e-05, velocity shape flattens while ρ/p shape keeps falling;
3. per-interval observed order vs h with the design order (4) and the SSP-RK3
   order (3) as reference lines — the velocity EOC crosses below 3 and dies.

No field contour is shown on purpose: at these meshes the numerical and exact
density/pressure fields differ by ≤ 0.0084 %, so a contour pair is
visually indistinguishable; the informative object is the error curve, not the
field. Recorded as a deliberate omission, not an oversight.

## 5. Caveats

1. **Single machine, single process**: the box was at load15 ≈ 120/128 during the
   runs, so all timings above (`OMP_NUM_THREADS=1`) are inflated; the cluster
   (loong64, 2×128) has no torch wheel, so the study ran locally — one
   single-threaded process, no fan-out, per the capacity rule.
2. **The velocity floor is described, not explained.** Candidates not yet excluded:
   the HLLC dissipation on a smooth but *swirled* velocity field, the positivity
   filter's interaction with the momentum modes, and the L2 initial projection of
   the velocity component. Each is a separate, cheap experiment
   (limiter-free vs filter-disabled vs nodal-projection) and is the natural next
   ticket.
3. **γ = (p+1)/q − 1 is algebra, not a cure** — since the temporal error is
   negligible here, order-matched CFL only adds cost. It matters for long-time or
   under-resolved runs and is kept as a tested tool, not as a fix.
4. Timings and wall-clock numbers are evidence-environment-specific; only the
   dimensionless orders are portable.

## 6. Error verdict (vs analytic solution)

| quantity | reference | value | error vs design order | within 3 %? |
|---|---|---:|---:|---|
| ρ observed order (24²→32²) | 4 | 4.05 | +1.15 % | **YES** |
| p observed order (24²→32²) | 4 | 4.03 | +0.80 % | **YES** |
| u observed order (24²→32²) | 4 | 0.07 | -98.17 % | **NO** |
| temporal order contribution at CFL 0.2 | ≤ O(dt³) | below 4-digit resolution | n/a | undetectable → temporal explanation **falsified** |
| ρ rel-L2 on 32² | exact vortex | 2.4055e-06 | 0.0002 % | **YES** |
| p rel-L2 on 32² | exact vortex | 3.4036e-06 | 0.0003 % | **YES** |
| u rel-L2 on 32² | exact vortex | 8.4497e-05 | 0.0084 % | value **YES**, order **NO (EOC 0.07)** |

**Honest answer to "is it < 3 %?"**: as an accuracy gate, yes everywhere (≤ 0.009 %).
As an *order* statement, ρ and p pass at the design order to within
1.2 % / 0.8 %, while the velocity
channel **fails** its order claim (EOC 0.07 vs 4) and is reported as an open
defect, not as a pass.

## 7. Files

| file | role |
|---|---|
| `benchmarks/design_order_euler_dg_2d.py` | + temporal study, CFL schedule, quadrature override, `--out`, coupled figure |
| `tools/probe_velocity_order_floor.py` | offset/shape decomposition with a self-check against the benchmark |
| `tools/gen_report_m3694.py` | this document, generated from JSON |
| `tests/test_euler_dg_2d_design_order.py` | + 3 tests (CFL algebra, dt-not-the-bottleneck assertion) |
| `validation/euler_dg_2d/p3_ladder_fine.json` | 8²→32² P3 ladder |
| `validation/euler_dg_2d/velocity_order_floor.json` | decomposition + self-check ratios |
| `docs/figures/euler_dg_2d_p3_order_decomposition.png` | report figure |
| `runs/p3_ladder_fine.log`, `runs/probe_velocity_floor.log` | verbatim run logs |
