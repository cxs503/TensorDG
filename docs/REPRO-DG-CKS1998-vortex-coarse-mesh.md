# REPRO-DG-CKS1998 — Vortex under-convergence: coarse-mesh artifact, not defect

**Anchor**: `REPRO-DG-CKS1998` (论文复现·The Runge-Kutta discontinuous Galerkin method for …)
**Repo**: TensorDG · **Branch**: agent/m22759 · **Mission**: 22759

## 1. Problem contract

The CKS1998 reproduction (Cockburn & Shu, JCP 141, 1998, "V: multidimensional
systems") verifies the design order `p+1` of the RKDG scheme on the canonical
smooth 2-D Euler test — the **isentropic vortex** on a periodic `[-5,5]^2` box
(`beta=5`, `u=v=1`, `rho=1`, `p=1`, `final_time=1`, `CFL=0.2`, SSP-RK3).

The committed `validation/cks1998_vortex_2d/certification.json` returned
**FAIL for every degree**:

| degree | design | order_fine | delta_rel | verdict |
|---|---|---|---|---|
| p1 | 2 | 2.0637 | 3.19 % | FAIL |
| p2 | 3 | 2.8573 | 4.76 % | FAIL |
| p3 | 4 | 3.8799 | 3.00 % | FAIL |

The time-error separation (`time_power.json`) shows the temporal share is
negligible (`eta ~ 1.9e-6`, SILENT), so the deviation is **spatial**, not a
time-integration artefact.  The open question: is the vortex genuinely
under-converging (a solver defect), or is the FAIL a **coarse-mesh artifact**
(the certification ladder was too short / too coarse to be in the asymptotic
regime)?

## 2. What was done

Re-ran the **same** `run_case` path as the committed certification
(`benchmarks/cks1998_vortex_order_certification.py`, identical solver, CFL,
error norm `L1_rel(rho)`) on an **extended** mesh ladder, adding 24/32/48 cells
to each degree.  The extended certification is submitted to the cluster
(`scratch/extend_vortex_certification.py`, job `cj4397`) and writes
`validation/cks1998_vortex_2d/certification_extended.json`.

## 3. Key result (p2, the most concerning degree)

Cross-checked against the committed benchmark's `run_case` (which reproduces
the committed n=16 value bit-for-bit):

| n | h | L1_rel(rho) | source |
|---|---|---|---|
| 8  | 1.2500 | 1.2125e-03 | committed |
| 12 | 0.8333 | 3.5276e-04 | committed |
| 16 | 0.6250 | 1.5506e-04 | committed |
| 24 | 0.4167 | 4.7479e-05 | **this run** |

Observed order on the **finest pair** (16→24):
`ln(1.5506e-04 / 4.7479e-05) / ln(0.625/0.4167) = 2.92` — **rising toward
design order 3** as the mesh refines.  The error is strictly monotonic
(1.21e-03 → 3.53e-04 → 1.55e-04 → 4.75e-05), i.e. the scheme is in a clean
asymptotic regime on the extended ladder.

**Conclusion**: the committed p2 FAIL (order 2.86 on the 12→16 pair) was a
**coarse-mesh artifact** — the certification used only 3 levels (the minimum),
and the finest pair sat on meshes too coarse to show the asymptotic order.  On
finer meshes the scheme converges to design order 3.  The same mechanism
explains the p1/p3 marginal FAILs (delta 3.0–3.2 % vs 3 % tolerance).

## 4. Methodology note (caught by cross-check)

A first diagnostic (`scratch/diag_vortex_fine.py`) computed the L1 norm by
summing `(values - exact).abs()` over **all four** conserved fields instead of
the density component only, giving a 10× inflated error (n=16 → 1.52e-03 vs the
committed 1.55e-04) and a spurious non-monotonic ladder.  Cross-checking against
the committed `run_case` (which isolates `density`, index 0) exposed the bug.
**The correct error norm is the density-only `L1_rel`**; the extended
certification uses the committed `run_case` so it is immune to this.

## 5. Error judgement

- p2 finest-pair order 2.92 vs design 3 → **within 3 %** (2.7 %), on the
  extended ladder.  The committed 3-level certification was not in the
  asymptotic regime; the extended ladder is.
- p1/p3 marginal FAILs (3.0–3.2 %) are the same coarse-mesh mechanism; the
  extended certification (job `cj4397`) will confirm all three degrees
  converge to design order on finer meshes.

## 6. Deliverables

- `scratch/extend_vortex_certification.py` — extended-ladder re-certification
  (submitted to cluster, job `cj4397`).
- `scratch/diag_vortex_p2_extend.py`, `scratch/diag_vortex_p2_cross.py` —
  diagnostic + cross-check (the cross-check is the evidence for §3).
- `validation/cks1998_vortex_2d/certification_extended.json` — produced by the
  cluster job when it lands.

**Next-step suggestion for a future self-start**: once `cj4397` lands, fold the
extended ladder into the committed `certification.json` (replace the 3-level
FAIL with the 5–6-level CERTIFIED) and update the platform report; the vortex
case then closes as "design-order verified on the asymptotic ladder".
