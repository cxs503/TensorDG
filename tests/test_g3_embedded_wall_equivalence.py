"""G3 attribution: an *embedded* solid wall must be identical to a domain wall.

``docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md`` §4(a) recorded the shock reflection
off an internal solid block as an honest FAIL and left two suspects undistinguished:

* the embedded-wall flux in ``tensordg.euler_dg_2d._solid_interface`` is wrong, or
* the reflected plateau on a 30x1 / 40x1 mesh is simply a numerical artifact of a
  shock that has nowhere to form (geometry degeneracy).

This module separates them with an invariant that does not need the analytic
reference at all: put the wall at the same cell face in two ways --

* **boundary arm** — the mesh ends at the wall, ``bcs={"right": "wall"}``;
* **solid arm** — the mesh continues, the cells behind the wall are a frozen solid
  block, so the wall is an interior face handled by ``_solid_interface``.

Both arms carry the *same* ``nx_fluid`` fluid cells, the *same* ``dx`` and the
*same* initial data, so the fluid cells are compared one-to-one.  The two arms
must agree to round-off: any disagreement is the embedded-wall flux, because
nothing else differs.  ``limiter=None`` is used so the interface is the only code
path under test (a limiter would also see the frozen cells as a jump).
"""
from __future__ import annotations

import torch

from benchmarks.euler_dg_2d_wall_bc import WallReflection
from tensordg.euler_dg_2d import solve_euler_dg_2d

INCIDENT_MACH = 2.0
NX_FLUID = 16       # fluid cells in front of the wall (both arms)
NX_SOLID = 3        # frozen cells behind the wall (solid arm only)
XMAX_FLUID = 0.32   # domain length of the boundary arm == wall position
DX = XMAX_FLUID / NX_FLUID
FINAL_TIME = 0.08   # wall_time ~ 0.051 -> the reflection has started
YMAX = 0.5


def _reference():
    # the shock is snapped to a face of the fluid mesh so both arms start identically
    x_shock = round(0.2 / DX) * DX
    x_wall = round(XMAX_FLUID / DX) * DX
    return WallReflection(x_shock=x_shock, x_wall=x_wall, incident_mach=INCIDENT_MACH)


def _initial(wr):
    pre_rho, pre_u, pre_p = wr.pre
    post_rho, post_u, post_p = wr.post

    def initial(x, y):
        ahead = x < wr.x_zero
        return (
            torch.where(ahead, torch.full_like(x, pre_rho), torch.full_like(x, post_rho)),
            torch.where(ahead, torch.full_like(x, pre_u), torch.full_like(x, post_u)),
            torch.zeros_like(x),
            torch.where(ahead, torch.full_like(x, pre_p), torch.full_like(x, post_p)),
        )

    return initial


def _run(wall_mode, degree=1, nx_solid=NX_SOLID):
    wr = _reference()
    nx = NX_FLUID if wall_mode == "boundary" else NX_FLUID + nx_solid
    xmax = nx * DX
    solid = None
    if wall_mode == "solid":
        solid = torch.zeros(nx, 1, dtype=torch.bool)
        solid[NX_FLUID:] = True
    x, y, coeff, meta = solve_euler_dg_2d(
        _initial(wr), ncells=(nx, 1), degree=degree, final_time=FINAL_TIME,
        cfl=0.12, domain=((0.0, xmax), (0.0, YMAX)), periodic=(False, False),
        limiter=None, solid=solid, track_boundary_flux=True,
        bcs={"left": "outflow", "right": "wall", "bottom": "wall", "top": "wall"},
    )
    return coeff[:NX_FLUID], meta


def test_reflection_actually_started():
    """Guard: the comparison is only meaningful once the incident shock has hit the
    wall, otherwise both arms are trivially equal."""
    wr = _reference()
    assert FINAL_TIME > wr.wall_time
    assert wr.reflected_speed < 0.0  # the reflected shock runs back into the gas


def test_embedded_wall_matches_boundary_wall_to_roundoff():
    coeff_bnd, meta_bnd = _run("boundary")
    coeff_sol, meta_sol = _run("solid")
    scale = coeff_bnd.abs().max().item()
    delta = (coeff_bnd - coeff_sol).abs().max().item()
    assert delta <= 1e-9 * scale, (
        f"embedded wall differs from the boundary wall by {delta / scale:.3e} "
        f"(relative) on identical fluid cells")
    # both arms took the same number of steps: the solid cells do not shrink dt
    assert meta_sol["steps"] == meta_bnd["steps"]


def test_embedded_wall_moves_the_shock_back():
    """The reflected plateau must appear: density near the wall rises above the
    post-incident-shock value, and the whole-domain mass is conserved."""
    wr = _reference()
    coeff, meta = _run("solid")
    x = (torch.arange(NX_FLUID, dtype=torch.float64) + 0.5) * DX
    rho = coeff[..., 0, 0, 0][:, 0]
    front = wr.front(FINAL_TIME)
    behind_wall_shock = rho[(x > front) & (x > wr.x_wall - 4 * DX)]
    assert behind_wall_shock.numel() >= 1
    assert behind_wall_shock.numel() >= 1
    assert behind_wall_shock.max().item() > wr.post[0], (
        "no compression between the reflected front and the wall")
    assert abs(meta["conservation_residual_total"]["mass"]["relative"]) < 1e-12