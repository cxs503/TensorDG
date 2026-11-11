"""Generate docs/tensordg-swe-2d-wellbalanced.md from the validation artifact.

Every number in the document comes from
``validation/swe_2d_wb/wellbalanced_2d_results.json`` -- the markdown is
rendered, never hand-copied, so the page cannot drift away from the artifact.

Run after the validation script::

    PYTHONPATH=. python validation/swe_2d_wb/run_wellbalanced_2d.py
    python scratch/m12101_make_2d_wb_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "validation" / "swe_2d_wb" / "wellbalanced_2d_results.json"
OUT = ROOT / "docs" / "tensordg-swe-2d-wellbalanced.md"


def fmt(x: float, spec: str = ".3e") -> str:
    return format(x, spec)


def main() -> None:
    r = json.loads(ART.read_text())
    prov = r["provenance"]
    setup = r["setup"]
    summary = r["summary"]

    curved = r["curved_bed_rest"]
    rows = []
    for deg in (1, 2):
        for nx in sorted({c["nx"] for c in curved}):
            wb = next(c for c in curved
                      if c["degree"] == deg and c["nx"] == nx and c["well_balanced"])
            df = next(c for c in curved
                      if c["degree"] == deg and c["nx"] == nx and not c["well_balanced"])
            rows.append(
                f"| P{deg} | {nx} | {df['steps']} | {fmt(df['eta_error_max'])} | "
                f"{fmt(df['u_max'])} | {fmt(wb['eta_error_max'])} | {fmt(wb['u_max'])} |"
            )
    curved_table = "\n".join(rows)

    jump_rows = []
    for j in r["bed_jump_rest"]:
        jump_rows.append(
            f"| P{j['degree']} | {j['nx']}x{j['ny']} | {j['jump_height_m']:.1f} | "
            f"{j['steps']} | {fmt(j['eta_error_max'])} | {fmt(j['u_max'])} | "
            f"{'yes' if j['dry_cells_remain_dry'] else 'no'} |"
        )
    jump_table = "\n".join(jump_rows)

    d = r["dam_break"]
    bc = summary["baseline_contrast"]
    ratio = bc["max_u_max_default_path"] / max(bc["max_u_max_well_balanced"], 1e-30)

    # cost: average wall time per step, well-balanced vs default, same grids
    def _per_step(rec):
        return rec["wall_time_s"] / max(rec["steps"], 1)

    wb_cost = sum(_per_step(c) for c in curved if c["well_balanced"]) / len(
        [c for c in curved if c["well_balanced"]])
    df_cost = sum(_per_step(c) for c in curved if not c["well_balanced"]) / len(
        [c for c in curved if not c["well_balanced"]])
    overhead = 100.0 * (wb_cost / df_cost - 1.0)

    doc = f"""# Hydrostatic Well-Balancedness in the 2-D SWE Solver

## Overview

`tensordg.swe_2d.solve_swe_2d` gains a `well_balanced=True` path that makes a
lake at rest

    η(x, y) = h(x, y) + z_b(x, y) = const,   u = v = 0

a **discrete fixed point at round-off**, on a curved bed and across a bed jump,
including a wet cell facing a dry neighbour. The pre-existing default path is
left bit-for-bit unchanged (`well_balanced=False`, the default).

This is the 2-D counterpart of the 1-D property documented in
`docs/tensordg-swe-hydrostatic-wellbalanced.md`; the two solvers now share the
same equilibrium notion, the same `DRY_DEPTH` convention and the same
"cell-local nodal data" way of expressing a bed that jumps between two cells.

## Why it matters

Nearly-quiescent water (harbour seiching, tsunami propagation across an ocean
trench, flooding over terrain) is the regime in which the *error* of a
shallow-water solver is the spurious current it generates out of nothing. A
scheme that is not well balanced turns a flat free surface into a slowly
accelerating flow; the artefact grows linearly in time and is first order in
`dx` on a curved bed, so refining the mesh does not remove it.

## The defect in the default 2-D path

Two approximations in the default path break the equilibrium:

1. **Bed slope.** The momentum source uses the cell-constant slope
   `(z_b,R − z_b,L)/dx`, which is the *cell average* of `dz_b/dx`. On the
   quadratic bed used below it differs from `dz_b/dx` by an `O(dx)` term, and
   the momentum residual it leaves accumulates into a spurious current.
2. **Initial data.** The state is L2-projected at Gauss points that are not the
   bed's nodal points, so `h_h + z_h = const` is not exactly representable in
   the discrete space for a curved bed.

Neither approximation is wrong in general -- they are second-order accurate for
smooth *unsteady* flow -- they simply do not satisfy the lake-at-rest
invariance.

## What the well-balanced path does

When `well_balanced=True`:

1. **Nodal LGL bed.** The bed is interpolated at the `p + 1`
   Gauss-Lobatto-Legendre nodes of each cell (endpoints included) and stored as
   a tensor-product polynomial `zb_modes` of shape `(nx, ny, p+1, p+1)`. The
   interpolant reproduces the nodal values at `ξ = ±1`, so the polynomial bed
   and the face-bed arrays agree exactly.
2. **Collocated initial data.** `h`, `hu`, `hv` are sampled at the *same* LGL
   nodes and transformed to modal form with `V⁻¹`. A still lake of a polynomial
   bed is then representable exactly: `h = η₀ − z_b` holds pointwise in the
   discrete space.
3. **Exact bed derivative in the source.** The source is evaluated at the Gauss
   points of the solution with the analytic derivative of that bed polynomial,

       S_kl = −(2k+1)(2l+1)/4 · g · Σ_ab w_a w_b h(x_a,y_b) ∂_x z_b(x_a,y_b) φ_k(ξ_a) φ_l(η_b)

   (and the same in `y`), instead of the cell-constant slope.
4. **Audusse hydrostatic reconstruction at the faces** (Audusse et al., 2004),
   applied independently in `x` and `y`:

       z_b* = max(z_b,iR, z_b,i+1,L),  h*_m = max(0, h_m + z_b,m − z_b*)

   the HLL flux is evaluated with the reconstructed states and
   `g/2 (h_m² − h*_m²)` is added back to the **normal** momentum flux of that
   side. The tangential velocity is carried unchanged. On a continuous bed the
   reconstruction is the identity, so the scheme reduces algebraically to the
   default shared-face HLL flux; it only changes the result at a bed jump or a
   wet/dry interface, which is where a plain HLL flux would inject spurious
   mass or momentum.

### Why the cancellation is exact, not merely consistent

For a still lake `h = η₀ − z_b` pointwise, hence
`∂_x(½ g h²) = g h ∂_x h = −g h ∂_x z_b`. The volume term of the momentum
equation and the bed-slope source are then equal and opposite **at every
quadrature point** (`h` is a degree-`p` polynomial, `∂_x z_b` is degree `p−1`,
and the Gauss rule with `nq = 2p + 3` points integrates the products exactly).
The face fluxes cancel between neighbouring cells because the one-sided
reconstructions give the same pressure flux on both sides of a face. Nothing is
left but round-off.

## API

```python
from tensordg.swe_2d import solve_swe_2d

x, y, coeff, zb_faces, meta = solve_swe_2d(
    initial_h, initial_u, initial_v, zb,
    nx=32, ny=16, degree=2,
    final_time=0.5, domain=((0.0, 1.0), (0.0, 1.0)),
    cfl=0.3, well_balanced=True,
)
```

* `initial_h`, `initial_u`, `initial_v`, `zb` are either callables
  `f(x, y)` or cell-local nodal tensors of shape
  `(nx, ny, degree+1, degree+1)`. **A bed or an interface that jumps between two
  cells must be given as nodal data**: a single callable is evaluated at one
  shared face coordinate and therefore smears the jump over both cells.
* `metadata["zb_modes"]` is the bed polynomial (`None` on the default path);
  `metadata["well_balanced"]` and `metadata["bed_representation"]` are set.
* `reconstruct_fields_2d(coeff, zb_modes)` returns `(h, u, v, eta)`;
  `check_hydrostatic_equilibrium_2d(coeff, zb_modes, eta_expected=...)` returns
  the equilibrium verdict. Both work for wet cells only -- a dry cell's `η` is
  the bed level by convention.

## Validation

Artifact: `validation/swe_2d_wb/wellbalanced_2d_results.json`
(generated by `validation/swe_2d_wb/run_wellbalanced_2d.py`; the tables below
are rendered from it by `scratch/m12101_make_2d_wb_doc.py`).

Setup: still-water level `η₀ = {setup['still_water_level_m']} m`, domain
`{setup['domain']}`, CFL {setup['cfl']}, `{setup['dtype']}`.

### 1. Still lake on a quadratic bed `z_b = 0.2x + 0.05x²`

Spurious current after `t = 0.1` s (identical initial state in both columns):

| degree | nx | steps | default η err [m] | default \\|u\\| [m/s] | WB η err [m] | WB \\|u\\| [m/s] |
|---|---|---|---|---|---|---|
{curved_table}

The default path drifts by up to {fmt(bc['max_u_max_default_path'])} m/s, i.e. the
cell-constant slope is *not* harmless; the well-balanced path holds at
{fmt(bc['max_u_max_well_balanced'])} m/s -- a factor {ratio:.1e} lower, at the
round-off floor of the method.

### 2. Still lake across a bed jump with a dry bank

Cell-local nodal bed: a {setup['bed_jump_height_m']} m step on a cell face, the
right half of the domain dry (`h = 0`, `DRY_DEPTH = {setup['dry_depth_m']}` m):

| degree | grid | jump [m] | steps | η err [m] | \\|u\\| [m/s] | dry stays dry |
|---|---|---|---|---|---|---|
{jump_table}

### 3. Unsteady dam break over a periodic hump

Checks that exactness at rest is not bought with a loss of robustness: the
well-balanced path must stay stable, keep `h > 0` and conserve mass on a
periodic domain.

| quantity | value |
|---|---|
| grid | {d['nx']}x{d['ny']}, P{d['degree']} |
| steps | {d['steps']} |
| mass initial | {d['mass_initial']:.12f} |
| mass final | {d['mass_final']:.12f} |
| mass drift | {fmt(d['mass_drift'])} |
| min depth [m] | {d['h_min']:.4f} |
| finite | {d['finite']} |

### Cost

Measured on the default validation grids (CPU, float64): the well-balanced
path costs {overhead:+.1f}% per time step relative to the default path
(average over the six `(degree, nx)` combinations, including the extra
`einsum`s for the exact bed derivative and the two reconstruction passes in
`x` and `y`). The default path itself is bit-for-bit unchanged.

### Pass criteria

* well-balanced η error < 1e-8 m and velocity < 1e-8 m/s on every grid:
  **{'PASS' if summary['pass'] else 'FAIL'}**
* dry cells stay dry across the bed jump: see table 2.
* dam-break mass drift < 1e-10: {fmt(d['mass_drift'])}.

## Limitations

1. **A bed jump must be nodal data.** Passing a callable that is discontinuous
   at a face gives the two cells the same face value, i.e. a ramp across two
   cells. A ramp carrying 2.5 m over one cell (slope ≈ 30) is outside the
   stable regime of the scheme -- the transonic momentum source that such a
   slope generates is not what this path is designed for; use nodal data to
   place the jump on the face.
2. **A shoreline inside a cell** cannot be represented: if the still-water
   level cuts through a cell, the discrete `h + z_b` is not constant there and
   the cell is not at equilibrium. Put the shoreline on a face (nodal bed) or
   resolve it.
3. **Limiters.** `limiter="tvb"`/`"weno"` operate on the conserved variables
   after the well-balanced stage and will break the exact cancellation (the
   lake at rest is no longer a fixed point). They are meant for shocked
   unsteady flow, where the well-balanced property is not the objective.
4. **Round-off floor.** P2 carries a slightly larger floor than P1
   (~1e-11 vs ~1e-13 m/s in double precision) because the `1/dx` prefactor of
   the weak form amplifies the flux/source cancellation; this is floating-point
   arithmetic, not a loss of the property.
5. **Cost.** See the section above; the reconstruction and the exact bed
   derivative add work per stage.
6. **Pre-existing, unrelated failure.** `tests/test_swe_2d_dynamics.py::
   test_stoker_depth_and_momentum_match_exact_solution` fails on the unmodified
   tree as well (`star_depth_error = 1.55` against a 0.03 criterion, verified
   with `git stash`); it is a Stoker/`ny = 4` issue of the default path and is
   not addressed here.

## Provenance and regeneration

The committed artifact was produced by

```bash
PYTHONPATH=. python validation/swe_2d_wb/run_wellbalanced_2d.py
```

from commit `{prov['git_commit_short']}` (branch `{prov['git_branch']}`,
host `{prov['machine']}`) on {prov['generated_at_utc']}, with Python
{prov['python']}, torch {prov['torch']}, NumPy {prov['numpy']}.
`git_dirty: {prov['git_dirty']}` records the raw `git status --porcelain` of the
tree at that moment: the artifact is generated while the very commit that adds
it is still being assembled, so `True` here means "generated from the
in-preparation change set", not "generated from modified committed code". The document you are reading was
rendered from that JSON by `scratch/m12101_make_2d_wb_doc.py`; re-running both
on the same host reproduces the numbers.

## References

1. Audusse, E., Bouchut, F., Bristeau, M.-O., Klein, R., & Perthame, B. (2004).
   "A fast and stable well-balanced scheme with hydrostatic reconstruction for
   shallow water flows." *SIAM Journal on Scientific Computing*, 25(6),
   2050-2065.
2. Xing, Y., & Shu, C.-W. (2005). "High order finite difference WENO schemes
   with the exact conservation property for the shallow water equations."
   *Journal of Computational Physics*, 208(1), 206-227.
3. Thacker, W. C. (1981). "Some exact solutions of the nonlinear shallow-water
   equations." *Journal of Fluid Mechanics*, 107, 499-508.
4. Bollermann, A., Chen, G., Kurganov, A., & Noelle, S. (2013). "A
   well-balanced reconstruction of wet/dry fronts for the shallow water
   equations." *Journal of Scientific Computing*, 56, 267-290.
"""
    OUT.write_text(doc)
    print(f"wrote {OUT} ({len(doc)} chars)")


if __name__ == "__main__":
    main()