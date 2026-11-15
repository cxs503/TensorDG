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

# ---------------------------------------------------------------------------
# (m22062) 1. the conserved-quantity ledger must be *evaluable* on both arms
# ---------------------------------------------------------------------------
def test_conservation_ledger_is_finite_exactly_when_no_obstacle_is_present():
    """``conservation_residual_total`` is the strongest identity the solver reports
    (``change == boundary_flux + obstacle_impulse``), and both arms must be able to
    evaluate it.

    Regression for the defect recorded in
    ``docs/CASE-REPRO-DG-CKS1998-G3-EMBEDDED-WALL-ATTRIBUTION.md`` §5.3: the
    identity was gated on ``solid_balance is not None``, so the *obstacle-free*
    arm returned ``nan`` for every component -- and ``abs(nan) < tol`` is
    ``False``, i.e. a conservation gate silently degraded to "cannot be judged"
    on exactly the arm that needs no obstacle bookkeeping at all.  A ``nan``
    residual is now a failure in its own right, not a non-answer.
    """
    import math

    _, meta_bnd = _run("boundary")
    _, meta_sol = _run("solid")

    # no obstacle on the boundary arm -> the impulse term is exactly zero, not nan
    assert meta_bnd["conservation_residual_total"]["mass"]["has_obstacle"] is False
    assert meta_bnd["obstacle_impulse"]["mass"] == 0.0
    assert meta_sol["conservation_residual_total"]["mass"]["has_obstacle"] is True

    for arm, meta in (("boundary", meta_bnd), ("solid", meta_sol)):
        for name in ("mass", "x_momentum", "y_momentum", "energy"):
            entry = meta["conservation_residual_total"][name]
            for key in ("change", "boundary_flux", "obstacle_impulse", "residual", "relative"):
                assert math.isfinite(entry[key]), (arm, name, key, entry[key])
            assert abs(entry["relative"]) < 1e-12, (arm, name, entry["relative"])


# ---------------------------------------------------------------------------
# (m22062) 2. the *y*-direction embedded wall (axis = 1) is the same invariant
# ---------------------------------------------------------------------------
# ``_solid_interface`` has an ``axis=1`` branch that no earlier case exercised:
# every embedded-wall experiment so far (G3, the forward-facing step) put the
# obstacle behind a *x*-face, which only reaches the ``axis=0`` branch.  The
# step's vertical face ``x = 0.6, y in [0, 0.2]`` is therefore still unproven
# (recorded as limitation 5 of the attribution doc).  Rotating the G3 experiment
# by 90 degrees exercises it with the same invariant, so no second analytic
# reference is introduced: the two arms must still agree to round-off.
def _run_y(wall_mode, degree=1, ny_solid=NX_SOLID):
    wr = _reference()
    ny = NX_FLUID if wall_mode == "boundary" else NX_FLUID + ny_solid
    ymax = ny * DX
    solid = None
    if wall_mode == "solid":
        solid = torch.zeros(1, ny, dtype=torch.bool)
        solid[:, NX_FLUID:] = True
    pre_rho, pre_u, pre_p = wr.pre
    post_rho, post_u, post_p = wr.post

    def initial(x, y):
        # the same two-shock Riemann problem, propagated along +y instead of +x
        ahead = y < wr.x_zero
        return (
            torch.where(ahead, torch.full_like(x, pre_rho), torch.full_like(x, post_rho)),
            torch.zeros_like(x),
            torch.where(ahead, torch.full_like(x, pre_u), torch.full_like(x, post_u)),
            torch.where(ahead, torch.full_like(x, pre_p), torch.full_like(x, post_p)),
        )

    x, y, coeff, meta = solve_euler_dg_2d(
        initial, ncells=(1, ny), degree=degree, final_time=FINAL_TIME,
        cfl=0.12, domain=((0.0, YMAX), (0.0, ymax)), periodic=(False, False),
        limiter=None, solid=solid, track_boundary_flux=True,
        bcs={"left": "wall", "right": "wall", "bottom": "outflow", "top": "wall"},
    )
    return coeff[:, :NX_FLUID], meta


def test_embedded_wall_in_the_y_direction_matches_the_boundary_wall():
    """The ``axis = 1`` solid-interface branch reproduces the domain-wall path."""
    coeff_bnd, meta_bnd = _run_y("boundary")
    coeff_sol, meta_sol = _run_y("solid")
    scale = coeff_bnd.abs().max().item()
    delta = (coeff_bnd - coeff_sol).abs().max().item()
    assert delta <= 1e-9 * scale, (
        f"y-direction embedded wall differs from the boundary wall by {delta / scale:.3e}")
    assert meta_sol["steps"] == meta_bnd["steps"]
    # the reflected plateau exists in the y-direction too
    wr = _reference()
    rho = coeff_sol[0, ..., 0, 0, 0]
    assert rho.max().item() > wr.post[0]
    for name in ("mass", "energy"):
        assert abs(meta_sol["conservation_residual_total"][name]["relative"]) < 1e-12
