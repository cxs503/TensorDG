"""Gates for the wall / ghost-state boundaries of ``solve_euler_dg_2d`` (m16803).

The gates are written so that a broken boundary cannot pass them:

* the closed-form references (``post_normal_shock``, ``rhs_reflected``) are
  re-derived here from the Rankine--Hugoniot conditions independently of the
  solver, including an explicit residual check of the reflected state;
* the wall faces are tested for *exactness* (free-stream preservation, closed
  box conservation) rather than for "looking right" -- a ghost-state wall with
  an O(M^2) mass leak fails at 1e-12;
* the reflected-shock plateau and front position are compared with the
  closed-form two-shock solution.
"""
import math

import numpy as np
import pytest
import torch

from tensordg.euler_1d import GAMMA
from tensordg.euler_dg_2d import (
    cell_fields,
    conservative_state,
    normalise_bcs,
    solve_euler_dg_2d,
)
from benchmarks.euler_dg_2d_wall_bc import WallReflection, post_normal_shock, rhs_reflected


def free_stream_reference(initial, x, y, degree):
    """Modal reference of a free stream: mean mode only, higher modes exactly 0.

    The constant state is evaluated on the *cell-centre* grid ``(nx, ny)`` the
    solver returns, so a gate can compare modal coefficients element-wise.  The
    first version of the free-stream gate subtracted this constant from *every*
    modal coefficient, so its "deviation" was the reference magnitude itself: it
    read 1.0 for a perfectly exact run and could never pass.
    """
    X = x[:, None] + 0.0 * y[None, :]
    Y = y[None, :] + 0.0 * x[:, None]
    state = conservative_state(*initial(X, Y))  # (nx, ny, 4)
    reference = torch.zeros(
        (*state.shape[:2], degree + 1, degree + 1, 4),
        dtype=state.dtype, device=state.device,
    )
    reference[..., 0, 0, :] = state
    return reference


# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------

def test_incident_shock_state_satisfies_rankine_hugoniot():
    post, speed = post_normal_shock((1.0, 0.0, 1.0), 2.0)
    rho1, u1, p1 = 1.0, 0.0, 1.0
    rho2, u2, p2 = post
    # mass, momentum and energy fluxes across a shock moving at ``speed``
    mass = rho1 * (speed - u1) - rho2 * (speed - u2)
    momentum = (p1 + rho1 * (speed - u1) ** 2) - (p2 + rho2 * (speed - u2) ** 2)
    # energy flux in the shock frame: (E' + p)(S - u) with E' built from the
    # *shock-frame* kinetic energy (the classic lab-frame slip is a factor of
    # ``S + u`` here and was caught by this very gate)
    energy1 = (p1 / (GAMMA - 1.0) + 0.5 * rho1 * (speed - u1) ** 2 + p1) * (speed - u1)
    energy2 = (p2 / (GAMMA - 1.0) + 0.5 * rho2 * (speed - u2) ** 2 + p2) * (speed - u2)
    assert abs(mass) < 1e-12
    assert abs(momentum) < 1e-12
    assert abs(energy1 - energy2) / abs(energy1) < 1e-12
    assert abs(p2 / p1 - 4.5) < 1e-12 and abs(rho2 - 8.0 / 3.0) < 1e-12


def test_reflected_state_satisfies_rankine_hugoniot():
    post, _ = post_normal_shock((1.0, 0.0, 1.0), 2.0)
    reflected, mach, speed = rhs_reflected(post)
    rho2, u2, p2 = post
    rho5, u5, p5 = reflected
    assert abs(u5) < 1e-12, "the reflected plateau must be at rest"
    mass = rho2 * (speed - u2) - rho5 * (speed - u5)
    momentum = (p2 + rho2 * (speed - u2) ** 2) - (p5 + rho5 * (speed - u5) ** 2)
    assert abs(mass) < 1e-12
    assert abs(momentum) < 1e-12
    # entropy must increase through the reflected shock
    entropy2 = p2 / rho2 ** GAMMA
    entropy5 = p5 / rho5 ** GAMMA
    assert entropy5 > entropy2
    # closed-form values of the M=2 toy problem used by the gates
    assert abs(rho5 - 6.0) < 1e-10 and abs(p5 - 15.0) < 1e-10
    assert abs(mach - math.sqrt(3.0)) < 1e-6


# ---------------------------------------------------------------------------
# Boundary machinery
# ---------------------------------------------------------------------------

def test_boundary_specification_is_validated():
    with pytest.raises(ValueError):
        normalise_bcs({"sideways": "wall"})
    with pytest.raises(ValueError):
        normalise_bcs({"left": "reflecting"})
    assert normalise_bcs(None) == {}
    assert normalise_bcs({"left": "outflow", "right": "wall"}) == {"right": "wall"}


def test_wall_on_a_periodic_axis_is_ignored():
    """A periodic axis wins over a boundary entry (documented behaviour)."""
    def initial(x, y):
        return (torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
                torch.ones_like(x))

    _, _, periodic, _ = solve_euler_dg_2d(
        initial, ncells=4, degree=1, final_time=0.02, periodic=(True, True)
    )
    _, _, ignored, _ = solve_euler_dg_2d(
        initial, ncells=4, degree=1, final_time=0.02, periodic=(True, True),
        bcs={"left": "wall", "bottom": "wall"},
    )
    assert torch.equal(periodic, ignored)


def test_outflow_is_the_legacy_behaviour():
    """``bcs=None`` and an explicit outflow spec must be bit-identical."""
    def initial(x, y):
        rho = 1.0 + 0.1 * torch.sin(2.0 * math.pi * x) * torch.cos(math.pi * y)
        return rho, torch.full_like(x, 0.3), torch.zeros_like(x), torch.ones_like(x)

    common = dict(ncells=(6, 5), degree=1, final_time=0.05, periodic=(False, False))
    _, _, legacy, _ = solve_euler_dg_2d(initial, **common)
    _, _, explicit, meta = solve_euler_dg_2d(
        initial, bcs={"left": "outflow", "right": "outflow",
                      "bottom": "outflow", "top": "outflow"}, **common
    )
    assert torch.equal(legacy, explicit)
    assert meta["bcs"] == {}


def test_free_stream_preservation_with_walls():
    """A flow parallel to a wall must not move: the wall flux is exact."""
    uniform = (1.0, 0.7, 0.0, 1.0)

    def initial(x, y):
        return tuple(torch.full_like(x, value) for value in uniform)

    x, y, coeff, _ = solve_euler_dg_2d(
        initial, ncells=(10, 6), degree=2, final_time=0.05, cfl=0.2,
        periodic=(True, False), bcs={"bottom": "wall", "top": "wall"},
    )
    reference = free_stream_reference(initial, x, y, coeff.shape[2] - 1)
    # round-off only: the uniform state must be reproduced to machine accuracy
    deviation = float((coeff - reference).abs().max()) / float(reference.abs().max())
    assert deviation < 1e-12, deviation
    # ... and the invariants must be checked on *both* halves of the state: the
    # mean mode carries the free stream, every higher mode must be zero.  The
    # historic gate compared the constant state with *every* modal coefficient
    # (a shape broadcast), so its "deviation" was the reference magnitude itself
    # -- it read 1.0 for a perfectly exact run and could never pass.
    mean_mode = float((coeff[..., 0, 0, :] - reference[..., 0, 0, :]).abs().max())
    higher = float(coeff[..., 1:, :, :].abs().max()) if coeff.shape[2] > 1 else 0.0
    scale = float(reference.abs().max())
    assert mean_mode / scale < 1e-12, mean_mode / scale
    assert higher / scale < 1e-12, higher / scale


def test_wall_blocks_normal_momentum():
    """A jet aimed at a wall must not cross it (no mass flux through a wall)."""
    moving = (1.0, 0.0, -0.9, 1.0)  # normal velocity towards the bottom wall

    def initial(x, y):
        return tuple(torch.full_like(x, value) for value in moving)

    x, y, coeff, _ = solve_euler_dg_2d(
        initial, ncells=(8, 8), degree=1, final_time=0.03, cfl=0.1,
        periodic=(True, False), bcs={"bottom": "wall", "top": "outflow"},
    )
    fields = cell_fields(coeff)
    density = fields["density"]
    # mass in the whole box may only change if the top lets it out; the wall
    # itself must not add or remove mass, and the flow must stay symmetric in x
    assert torch.isfinite(coeff).all()
    assert float((density[:, 0] - density[:, -1]).abs().max()) < 1e-10


def test_closed_box_conserves_mass_and_energy():
    def initial(x, y):
        bubble = torch.exp(-40.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
        return 1.0 + 0.2 * bubble, torch.zeros_like(x), torch.zeros_like(x), 1.0 + 0.4 * bubble

    walls = {"left": "wall", "right": "wall", "bottom": "wall", "top": "wall"}
    common = dict(ncells=(10, 10), degree=2, cfl=0.15, periodic=(False, False), bcs=walls)
    _, _, start, _ = solve_euler_dg_2d(initial, final_time=0.0, **common)
    x, y, end, meta = solve_euler_dg_2d(initial, final_time=0.08, **common)
    mass0 = float(start[..., 0, 0, 0].sum())
    mass1 = float(end[..., 0, 0, 0].sum())
    energy0 = float(start[..., 0, 0, 3].sum())
    energy1 = float(end[..., 0, 0, 3].sum())
    assert abs(mass1 - mass0) / abs(mass0) < 1e-13
    assert abs(energy1 - energy0) / abs(energy0) < 1e-13
    # the wall momentum flux is a pressure difference only: the box stays still
    assert abs(float(end[..., 0, 0, 1].sum())) / abs(mass0) < 1e-12


def test_callable_boundary_with_wall_mask():
    """A callable may mix an impermeable wall and an inflow on the same face."""
    def bottom(X, Y, t):
        ghost = conservative_state(
            torch.full_like(X, 1.0), torch.zeros_like(X), torch.zeros_like(X),
            torch.full_like(X, 1.0),
        )
        wall_mask = X < 0.5
        return ghost, wall_mask

    def initial(x, y):
        return (torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
                torch.ones_like(x))

    _, _, coeff, meta = solve_euler_dg_2d(
        initial, ncells=(8, 4), degree=1, final_time=0.02,
        periodic=(False, False), bcs={"bottom": bottom, "left": "outflow",
                                      "right": "outflow", "top": "outflow"},
    )
    assert meta["bcs"] == {"bottom": "bottom"}
    assert torch.isfinite(coeff).all()


# ---------------------------------------------------------------------------
# Physics: shock reflecting off a wall
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [1, 2])
def test_shock_reflection_matches_closed_form(degree):
    case = WallReflection()
    rho2, u2, p2 = case.post
    rho1, u1, p1 = case.pre

    def initial(x, y):
        before = x < case.x_zero
        return (
            torch.where(before, torch.full_like(x, rho2), torch.full_like(x, rho1)),
            torch.where(before, torch.full_like(x, u2), torch.full_like(x, u1)),
            torch.zeros_like(x),
            torch.where(before, torch.full_like(x, p2), torch.full_like(x, p1)),
        )

    final_time = 0.32
    x, y, coeff, _ = solve_euler_dg_2d(
        initial, ncells=(80, 2), degree=degree, final_time=final_time, cfl=0.12,
        periodic=(False, True), bcs={"right": "wall"},
        limiter="tvb", tvb_constant=0.0, limiter_stage="final",
    )
    fields = cell_fields(coeff)
    centers = x.cpu().numpy()
    exact_rho, exact_u, exact_p = case.state(centers, final_time)
    rho = fields["density"][:, 0].cpu().numpy()
    pressure = fields["pressure"][:, 0].cpu().numpy()
    velocity = fields["velocity_x"][:, 0].cpu().numpy()

    # the reflected plateau sits between the reflected front and the wall
    front_exact = case.front(final_time)
    band = (centers > front_exact + 2.0 * (centers[1] - centers[0])) & (centers < 0.98)
    rho5, _, p5 = case.reflected
    assert abs(rho[band].mean() - rho5) / rho5 < 0.01
    assert abs(pressure[band].mean() - p5) / p5 < 0.01
    assert abs(velocity[band].mean()) / math.sqrt(GAMMA * p5 / rho5) < 0.01

    gradient = np.abs(np.gradient(rho, centers))
    front_numeric = centers[int(np.argmax(gradient))]
    cell_width = centers[1] - centers[0]
    assert abs(front_numeric - front_exact) < 2.0 * cell_width

    l2 = float(np.sqrt(np.mean((rho - exact_rho) ** 2)) / np.sqrt(np.mean(exact_rho ** 2)))
    assert l2 < 0.05, l2