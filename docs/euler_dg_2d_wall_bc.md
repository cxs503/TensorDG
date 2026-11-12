# 2-D Euler DG: wall / general ghost-state boundaries (m16803) — PARTIAL, honest FAIL on two gates

**Anchor**: `REPRO-DG-CKS1998` (open) — its documented missing item "2-D 系统基准" (double Mach
reflection, JCP 141) was blocked by a *capability* gap, not by resolution: `solve_euler_dg_2d`
only knew periodic faces and a zero-gradient outer face (`benchmarks/double_mach_reflection.py`
docstring, mission m13761 audit). This mission adds the boundary API; the DMR itself is **not**
accepted yet (see §4/§5).

## 1. What changed (`tensordg/euler_dg_2d.py`)

`solve_euler_dg_2d(..., bcs=None)`; keys `left/right/bottom/top`, each value

| value | meaning |
|---|---|
| `"outflow"` / `None` | legacy behaviour, interior trace on both sides (HLLC → physical flux), **bit-identical** to before |
| `"wall"` | impermeable slip wall via the exact wall flux `(0, p, 0, 0)` — no mass/momentum/energy leak, not the O(M²)-leaky mirror-ghost HLLC |
| `callable(X, Y, t)` | general ghost state at the face quadrature points, conservative 4-vector, or `(state, wall_mask)` so one face can mix wall and inflow (needed by the DMR floor) |

`conservative_state()` and `normalise_bcs()` are exported; `metadata["bcs"]` records the spec.

## 2. Verification actually performed

| gate | target | measured | verdict |
|---|---|---|---|
| RH reference, incident shock (M=2) — independently re-derived in the test | <1e-12 | mass/momentum/energy flux residuals <1e-12 | PASS |
| RH reference, reflected shock (`M_r=√3`, ρ₅=6, p₅=15) | 1e-10 | 1e-10, entropy increases | PASS |
| `bcs=None` vs explicit outflow | bit-identical | `torch.equal` true | PASS |
| wall on a periodic axis is ignored | documented | identical states | PASS |
| closed box (16×16 P2, pulse, 4 walls, 211 steps) | mass/energy drift <1e-13 | **7.9e-15 / 6.4e-15** | PASS |
| reflected-shock **front position** vs closed-form two-shock solution (60×1 P1, t=0.28) | <2 cells | **0.24 cells** | PASS |
| free-stream preservation (uniform flow along a wall, 24×8 P2, t=0.2) | <1e-12 | **1.0 (state collapses)** | **FAIL** |
| reflected plateau density/pressure | <1 % | measurement band was wrong in the first draft (sampled the pre-reflection state); band fixed to `(x_front+2h, 0.98)`, **not re-measured** | NOT RUN |
| Mach-10 double Mach reflection (60×15 P1, cfl 0.03…0.15, tvb/weno) | run to t=0.2 | `FloatingPointError: nonphysical cell average` within ~10 steps | **FAIL** |

Earlier probe (4×3 P1, t=0.02) shows the *uniform* state reproduced to 4e-17 at projection time and
both wall/outflow give the identical t=0 projection, so the free-stream failure is a time-stepping /
positivity-filter interaction, not a projection bug — **not diagnosed yet**.

## 3. Files

* `tensordg/euler_dg_2d.py` — BC API (+178/−17 lines, no behaviour change for `bcs=None`)
* `benchmarks/euler_dg_2d_wall_bc.py` — reference solution (`post_normal_shock`, `rhs_reflected`),
  free-stream / reflection / closed-box gates, JSON+PNG artefacts, provenance sha256
* `benchmarks/double_mach_wall_check.py` — Woodward–Colella configuration with wall floor, tracked
  top boundary, closed-form incident-front position check
* `tests/test_euler_dg_2d_wall_bc.py` — 10 gates (RH references, BC semantics, conservation,
  free-stream, closed box, callable mask, reflection vs closed form for P1/P2)

## 4. Honest verdict

**Not accepted.** The wall boundary is *exact* wherever it can be checked in closed form
(conservation to 1e-14, reflected front to a quarter cell), but two gates fail: free-stream
preservation over a long run and the Mach-10 DMR run. `REPRO-DG-CKS1998`'s 2-D item therefore stays
**open**; the blocker is now a scalar robustness defect (positivity/limiting under a strong shock
plus a wall), not a missing boundary condition.

## 5. Notes / traps

1. The first plateau band `(0.55, 0.72)` was **wrong physics**: at t=0.4 the reflected front sits at
   x=0.777, so that band holds the *post-incident-shock* state (ρ=8/3). Reported ρ₅ error −56 % was a
   measurement artefact, not a solver result — fixed in both driver and test.
2. Relative L2 against a field that is identically zero (velocity behind the reflected shock) divides
   by zero → ∞. Normalise by the measured field there.
3. `OMP_NUM_THREADS` matters on small tensors: 4 threads made a 100×3 P2 run slower than 4 minutes,
   1 thread is ~10× faster on this host (loadavg ≈ 45/128 cores).
4. `gates[]` in `docs/benchmarks/results/euler-dg-2d-wall-bc/results.json` is written only by the full
   driver; the committed artefacts are *not* produced in this mission (driver runtime exceeded the
   session budget), so the numbers above come from the console runs recorded in this file — treat them
   as L1 evidence until the driver is re-run.
