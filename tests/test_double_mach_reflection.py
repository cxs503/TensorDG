"""Double Mach reflection: exact references, orientation and front tracking.

Every threshold here is a *derived* quantity, not a measured-and-pinned number:

* the post-shock state is re-derived from the one-dimensional Rankine--Hugoniot
  relations inside the test and the jump residuals must vanish (``< 1e-12``);
* the undisturbed part of the incident front is an exact plane convecting with
  the normal shock speed ``W = M a_1``, so ``x_front(y, t)`` is analytic;
* the direction of propagation is asserted (the front must advance towards
  ``x = 4``).  This is the assertion that the pre-m20397 time-reversed set-up
  cannot satisfy: it put the shocked gas *ahead* of the front with a velocity
  pointing up-left, so the front left the domain at ``t = 0.0144`` and no wall
  reflection ever formed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.double_mach_reflection import (
    FRONT_SLOPE, FRONT_X0, NORMAL, POST_SHOCK, PRE_SHOCK, SHOCK_ANGLE_DEG,
    SHOCK_MACH, SHOCK_SPEED, analytic_front_x, double_mach_bcs,
    initial_double_mach, measure_primary_front, normal_shock_downstream,
    rh_residuals, solve_double_mach,
)
from tensordg.euler_dg_2d import cell_fields, conservative_state

DX_COARSE = 4.0 / 64.0


def test_post_shock_state_satisfies_rankine_hugoniot():
    residuals = rh_residuals(PRE_SHOCK, POST_SHOCK)
    assert residuals["shock_speed"] == pytest.approx(SHOCK_SPEED, rel=1e-12)
    for key in ("mass", "normal_momentum", "tangential_momentum", "energy"):
        assert abs(residuals[key]) < 1e-12, (key, residuals[key])


def test_post_shock_state_matches_independent_reshock_formulae():
    """The module state must equal the closed-form Mach-10 air shock values."""
    rho, u, v, p = POST_SHOCK
    gamma = 1.4
    m = SHOCK_MACH
    assert rho == pytest.approx(1.4 * (gamma + 1) * m ** 2
                               / ((gamma - 1) * m ** 2 + 2), rel=1e-12)   # 8.0
    assert p == pytest.approx(1.0 + 2 * gamma / (gamma + 1) * (m ** 2 - 1),
                              rel=1e-12)                                  # 116.5
    assert math.hypot(u, v) == pytest.approx(8.25, rel=1e-10)
    # velocity points along the front normal, i.e. down-right (into the wall)
    assert u * NORMAL[0] + v * NORMAL[1] == pytest.approx(8.25, rel=1e-10)
    assert u > 0.0 and v < 0.0


def test_analytic_front_is_exact_at_t0_and_convects_right():
    ys = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(analytic_front_x(ys, 0.0), FRONT_X0 + ys / FRONT_SLOPE,
                       rtol=0, atol=0)
    assert analytic_front_x(ys, 0.0)[0] == pytest.approx(1.0 / 6.0, rel=1e-15)
    speed = (analytic_front_x(ys, 0.01) - analytic_front_x(ys, 0.0)) / 0.01
    assert np.allclose(speed, SHOCK_SPEED / math.sin(math.radians(SHOCK_ANGLE_DEG)),
                       rtol=1e-12)
    assert np.all(speed > 0.0)


def test_initial_data_puts_shocked_gas_behind_the_front():
    """Post-shock up-left of the front, undisturbed air down-right."""
    x = np.array([0.02, 0.5, 1.0, 3.0])
    y = np.array([0.95, 0.7, 0.2, 0.1])
    rho, u, v, p = initial_double_mach(__import__("torch").as_tensor(x),
                                       __import__("torch").as_tensor(y))
    # (0.02, 0.95) and (0.5, 0.7) are behind the front -> shocked gas
    assert all(rho[i].item() == pytest.approx(POST_SHOCK[0], rel=1e-14) for i in (0, 1))
    # (1.0, 0.2) and (3.0, 0.1) are ahead of it -> air
    assert all(rho[i].item() == pytest.approx(PRE_SHOCK[0], rel=1e-14) for i in (2, 3))
    assert u[0].item() > 0.0 and v[0].item() < 0.0


def test_boundary_set_is_the_standard_one():
    bcs = double_mach_bcs()
    assert bcs["bottom"] == "wall"
    assert bcs["right"] == "outflow"
    assert callable(bcs["left"]) and callable(bcs["top"])


@pytest.fixture(scope="module")
def coarse_run():
    x, y, coeff, meta = solve_double_mach(
        ncells=(64, 16), degree=1, final_time=0.015, cfl=0.15,
        limiter="tvb", tvb_constant=100.0,
    )
    return x, y, cell_fields(coeff), meta


def test_front_advances_towards_the_wall_and_matches_exact_law(coarse_run):
    x, y, fields, meta = coarse_run
    rho = fields["density"].cpu().numpy()
    front = measure_primary_front(rho, x, y, y_levels=(0.75, 0.85))
    assert len(front) == 2, front
    for y_level, x_meas in front.items():
        x_exact = float(analytic_front_x(y_level, 0.015))
        # direction: the front must be *right* of its t=0 position
        assert x_meas > FRONT_X0 + y_level / FRONT_SLOPE, (y_level, x_meas)
        # magnitude: exact plane-front law, tolerance 1.5 cells
        assert abs(x_meas - x_exact) <= 1.5 * DX_COARSE, (y_level, x_meas, x_exact)


def test_shocked_plateau_behind_front_matches_rankine_hugoniot(coarse_run):
    """Cell averages a few cells behind the front must be the RH state."""
    x, y, fields, meta = coarse_run
    t = 0.015
    rho = fields["density"].cpu().numpy()
    p = fields["pressure"].cpu().numpy()
    ux = fields["velocity_x"].cpu().numpy()
    uy = fields["velocity_y"].cpu().numpy()
    xc, yc = np.asarray(x).ravel(), np.asarray(y).ravel()
    offset = max(5.0 * DX_COARSE, 0.05)
    j = int(np.argmin(np.abs(yc - 0.85)))
    i = int(np.argmin(np.abs(xc - (float(analytic_front_x(0.85, t)) - offset))))
    assert rho[i, j] == pytest.approx(POST_SHOCK[0], rel=0.10), rho[i, j]
    assert p[i, j] == pytest.approx(POST_SHOCK[3], rel=0.10), p[i, j]
    speed = math.hypot(ux[i, j], uy[i, j])
    assert speed == pytest.approx(math.hypot(POST_SHOCK[1], POST_SHOCK[2]), rel=0.05)


def test_reflection_structure_forms(coarse_run):
    """The Mach stem must produce densities well above the post-shock value."""
    x, y, fields, meta = coarse_run
    rho = fields["density"].cpu().numpy()
    assert float(rho.max()) > POST_SHOCK[0] * 1.15, float(rho.max())
    assert float(rho.min()) == pytest.approx(PRE_SHOCK[0], rel=0.10)


def test_uniform_state_is_preserved_through_the_boundary_set():
    """A uniform post-shock flow with the DMR walls must not be disturbed.

    Guards the wall / inflow ghost-state implementation: a state that is
    admissible and divergence free up to the boundaries has to stay uniform
    (``cell_average`` projection, 16x8 P1, one short march).
    """

    def initial(x, y):
        return tuple(__import__("torch").full_like(x, value) for value in PRE_SHOCK)

    from tensordg.euler_dg_2d import solve_euler_dg_2d
    x, y, coeff, meta = solve_euler_dg_2d(
        initial, ncells=(16, 8), degree=1, domain=((0.0, 4.0), (0.0, 1.0)),
        final_time=0.002, cfl=0.1, limiter=None, periodic=(False, False),
        initial_projection="l2",
        bcs={"bottom": "wall", "left": "outflow", "right": "outflow",
             "top": "outflow"},
    )
    fields = cell_fields(coeff)
    assert torch_absmax(fields["density"] - PRE_SHOCK[0]) < 1e-12


def torch_absmax(tensor):
    return float(tensor.abs().max())