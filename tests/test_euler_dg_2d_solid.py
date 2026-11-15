"""Internal solid obstacles (``solid=``) in ``solve_euler_dg_2d``.

The capability lets a grid-aligned body sit *inside* the mesh -- the forward-facing
step of Cockburn & Shu, JCP 141 (1998) has no representation without it, because
``bcs`` only addresses the four outer faces.  Each test below binds a claim of the
implementation to an independent reference:

* the mask API rejects masks that are not a full ``(nx, ny)`` cell array;
* a stream parallel to every embedded face is preserved (exact, round-off);
* a closed box with an interior obstacle keeps mass and energy (exact);
* an *embedded* wall and an outer ``bcs right="wall"`` produce the same solution
  at equal dx (the matched A/B that isolates the wall flux from shock capturing
  error; see ``docs/CASE-REPRO-DG-CKS1998-EMBEDDED-WALL-PARITY.md``);
* the Mach 3 step case runs, leaves the upstream state untouched and produces
  the obstacle drag through ``obstacle_impulse``.

Runtime is kept below ~2 min by using small grids; the production-resolution
numbers come from ``python -m benchmarks.forward_facing_step``.
"""
import math

import pytest
import torch

from tensordg.euler_dg_2d import conservative_state, normalise_solid, solve_euler_dg_2d
from benchmarks.euler_dg_2d_wall_bc import WallReflection

GAMMA = 1.4


def _uniform(x, y):
    return (torch.ones_like(x), 2.0 * torch.ones_like(x), torch.zeros_like(x), torch.ones_like(x))


def _pressure(field):
    rho = field[..., 0]
    return (GAMMA - 1.0) * (
        field[..., 3] - 0.5 * (field[..., 1] ** 2 + field[..., 2] ** 2) / rho.clamp_min(1e-30)
    )


def test_solid_mask_validation():
    """A mask that is not one entry per cell must be rejected, not broadcast."""
    with pytest.raises(ValueError):
        normalise_solid(torch.zeros(4, 3, dtype=torch.bool), 4, 4, torch.device("cpu"))
    with pytest.raises(ValueError):
        normalise_solid(torch.ones(4, 4, dtype=torch.bool), 4, 4, torch.device("cpu"))
    assert normalise_solid(None, 4, 4, torch.device("cpu")) is None
    assert normalise_solid(torch.zeros(4, 4, dtype=torch.bool), 4, 4, torch.device("cpu")) is None
    mask = normalise_solid(torch.eye(4, dtype=torch.int64), 4, 4, torch.device("cpu"))
    assert mask is not None and mask.dtype == torch.bool and int(mask.sum()) == 4


def test_free_stream_preserved_across_embedded_wall():
    """Exact gate: a stream parallel to every solid face must not be disturbed."""
    nx, ny = 20, 10
    solid = torch.zeros(nx, ny, dtype=torch.bool)
    solid[:, ny // 3: ny // 3 + 2] = True  # horizontal strip: all its faces are y-normal
    _, _, ref, _ = solve_euler_dg_2d(_uniform, ncells=(nx, ny), degree=2, final_time=0.0)
    _, _, got, meta = solve_euler_dg_2d(
        _uniform, ncells=(nx, ny), degree=2, final_time=0.03, cfl=0.2,
        periodic=(False, False), solid=solid, bcs={"bottom": "wall"},
    )
    deviation = float((got - ref).abs().max()) / float(ref.abs().max())
    assert meta["solid_cells"] == 2 * nx
    assert deviation < 1e-10, f"free stream disturbed by {deviation:.3e}"


def test_closed_box_with_obstacle_conserves_mass_and_energy():
    """Exact gate: no mass or energy crosses a wall, internal walls included."""
    nx = 16
    solid = torch.zeros(nx, nx, dtype=torch.bool)
    solid[nx // 3: 2 * nx // 3, nx // 3: nx // 2] = True

    def bump(x, y):
        r = (x - 0.5) ** 2 + (y - 0.5) ** 2
        return (1.0 + 0.3 * torch.exp(-40.0 * r), torch.zeros_like(x), torch.zeros_like(x),
                1.0 + 0.5 * torch.exp(-40.0 * r))

    _, _, _, meta = solve_euler_dg_2d(
        bump, ncells=(nx, nx), degree=1, final_time=0.06, cfl=0.15,
        periodic=(False, False), solid=solid, track_boundary_flux=True,
        bcs={"left": "wall", "right": "wall", "bottom": "wall", "top": "wall"},
    )
    total = meta["conservation_residual_total"]
    assert abs(total["mass"]["relative"]) < 1e-12
    assert abs(total["energy"]["relative"]) < 1e-12
    # momentum needs the obstacle impulse; the solid must actually push the gas
    assert abs(meta["obstacle_impulse"]["y_momentum"]) > 1e-6
    assert abs(total["y_momentum"]["relative"]) < 1e-12


def test_exact_two_shock_orientation():
    """The closed-form reference and the driver must agree on which side is post-shock.

    ``WallReflection`` describes a shock running *right*, so the post-shock state
    sits to the LEFT of the front and the gas moves right, into the wall.  The
    helper this file used to carry initialised the mirrored state (pre-shock on
    the left), which put the gas moving *away* from the wall and made the
    "reflected plateau" assertion fail for a reason that had nothing to do with
    the wall flux.  This test pins the orientation down; it needs no solver.
    """
    from benchmarks.cks1998_embedded_wall_parity import _initial

    wr = WallReflection(x_shock=0.5, x_wall=1.0, incident_mach=2.0)
    import numpy as np

    left_rho, left_u, left_p = wr.state(np.array([0.3]), 0.0)
    right_rho, right_u, right_p = wr.state(np.array([0.7]), 0.0)
    assert float(left_rho[0]) == pytest.approx(wr.post[0]) and wr.post[0] > wr.pre[0]
    assert float(right_rho[0]) == pytest.approx(wr.pre[0])
    assert float(left_u[0]) > 0.0  # post-shock gas moves toward +x, i.e. into the wall
    assert float(right_u[0]) == pytest.approx(wr.pre[1])
    assert float(left_p[0]) == pytest.approx(wr.post[2])

    # the driver's own initial data must agree, or the reference scores nothing
    x = torch.tensor([0.3, 0.7])
    y = torch.zeros_like(x)
    rho, u, v, p = _initial(wr, 0.5)(x, y)
    assert float(rho[0]) == pytest.approx(wr.post[0])
    assert float(rho[1]) == pytest.approx(wr.pre[0])
    assert float(p[0]) == pytest.approx(wr.post[2])
    assert float(u[0]) > 0.0


def test_embedded_wall_matches_boundary_wall():
    """An embedded wall and an outer wall produce the same solution at equal dx.

    Gate G3 of ``docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md`` was recorded as a
    failure with two candidate causes: a wrong embedded-wall flux, or DG shock
    smearing.  This is the matched A/B that separates them: the same reflection,
    the same cell size, the same limiter, the wall in the same place -- only the
    treatment (frozen solid cells vs ``bcs right="wall"``) differs.  The two arms
    must agree to round-off; whatever error both of them carry is shock capturing
    error, not a wall defect.
    """
    from benchmarks.cks1998_embedded_wall_parity import parity, run_arm

    solid = run_arm(12, 1, "solid", extra_time=0.05)
    boundary = run_arm(10, 1, "boundary", extra_time=0.05)
    assert solid["solid_cells"] > 0
    assert solid["steps"] == boundary["steps"]  # same dx, same CFL -> same march
    gap = parity(solid, boundary)
    assert math.isclose(gap["dx"], 0.1, rel_tol=1e-12)
    assert gap["whole_domain_l1_relative"] < 1e-10, gap["whole_domain_l1_relative"]


def test_forward_facing_step_upstream_untouched():
    """The Mach 3 step leaves the region the shock never reaches exactly uniform."""
    from benchmarks.forward_facing_step import (
        P_INF, RHO_INF, U_INF, free_stream, step_mask,
    )

    ncells, final_time = (24, 8), 0.12
    solid = step_mask(ncells)
    inflow = conservative_state(
        torch.tensor(RHO_INF), torch.tensor(U_INF), torch.tensor(0.0), torch.tensor(P_INF)
    )
    x, y, coeff, meta = solve_euler_dg_2d(
        free_stream, ncells=ncells, degree=1, domain=((0.0, 3.0), (0.0, 1.0)),
        final_time=final_time, cfl=0.2, periodic=(False, False), solid=solid,
        limiter="tvb", tvb_constant=1.0, track_boundary_flux=True,
        bcs={"left": lambda X, Y, t: inflow, "bottom": "wall", "top": "outflow",
             "right": "outflow"},
    )
    assert meta["solid_cells"] == 12 * 2  # 0.6/0.125 x 0.2/0.125 columns
    rho = coeff[..., 0, 0, 0]
    upstream = (x[:, None] < 0.4) & (y[None, :] > 0.6)
    assert float((rho[upstream] / RHO_INF - 1.0).abs().max()) < 1e-10
    assert torch.isfinite(coeff).all()
    # the obstacle has started to resist the stream
    assert meta["obstacle_impulse"]["x_momentum"] < 0.0
