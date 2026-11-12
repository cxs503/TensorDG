"""Verification of the *transcritical* (choked) bump steady state.

Case: Goutal & Maurel (1997) test 3 / Delestre et al. (2013) SWASHES "Bump with
shock" -- ``z_b = 0.2 (1 - ((x-10)/2)^2)`` on ``[8, 12]``, discharge
``q = 0.18 m^2/s`` and outlet depth ``h(25) = 0.33 m``.  The flow is choked: the
subcritical branch of :func:`tensordg.swe_steady_bump.bump_steady_state` does
not exist, and the solution is subcritical -> critical at the crest ->
supercritical -> **stationary hydraulic jump** -> subcritical.

The assertions here are deliberately independent of the branch-inversion code
used by the module: the smooth parts are re-integrated as the *steady ODE*

    dh/dx = -z_b'(x) / (1 - q^2/(g h^3))

(RK4, starting at the crest where ``h = h_c`` exactly), which never inverts the
Bernoulli cubic, and the jump is checked with the Rankine-Hugoniot conditions.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg.swe_steady_bump import (
    BUMP_HEIGHT,
    BUMP_SHOCK_DEPTH_OUT,
    BUMP_SHOCK_DISCHARGE,
    DOMAIN,
    GRAVITY,
    bump_bed,
    bump_bed_slope,
    bump_steady_state,
    bump_transcritical_state,
    momentum_flux,
    shock_heads,
    shock_position,
)

Q = BUMP_SHOCK_DISCHARGE
HOUT = BUMP_SHOCK_DEPTH_OUT


def _heads():
    return shock_heads(Q, HOUT, GRAVITY, BUMP_HEIGHT)


def _bernoulli(h, head, z):
    return h + Q * Q / (2.0 * GRAVITY * h * h) + z - head


def _integrate_steady_ode(x0: float, h0: float, x1: float, steps: int = 20000):
    """RK4 integration of the steady ODE from ``x0`` to ``x1`` (h0 = h(x0))."""
    xs = torch.linspace(x0, x1, steps + 1, dtype=torch.float64)

    def rhs(xv, hv):
        zb_p = float(bump_bed_slope(torch.tensor([xv], dtype=torch.float64))[0])
        denom = 1.0 - Q * Q / (GRAVITY * hv ** 3)
        return -zb_p / denom

    h = float(h0)
    dx = float(xs[1] - xs[0])
    for k in range(steps):
        xv = float(xs[k])
        k1 = rhs(xv, h)
        k2 = rhs(xv + 0.5 * dx, h + 0.5 * dx * k1)
        k3 = rhs(xv + 0.5 * dx, h + 0.5 * dx * k2)
        k4 = rhs(xv + dx, h + dx * k3)
        h = h + dx / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return h


# ---------------------------------------------------------------------------
# 1. Regime separation: the choked case really has no smooth subcritical branch
# ---------------------------------------------------------------------------
def test_choked_case_is_outside_the_subcritical_branch():
    with pytest.raises(ValueError, match="chokes"):
        bump_steady_state(torch.tensor([0.0, 10.0, 25.0]), discharge=Q, depth_out=HOUT)
    # the subcritical case of the same geometry is still fine
    assert bump_steady_state(torch.tensor([10.0]))["regime"] == "subcritical"


def test_heads_and_loss_are_physical():
    heads = _heads()
    hc = heads["critical_depth"]
    assert heads["head_left"] == pytest.approx(BUMP_HEIGHT + 1.5 * hc, rel=1e-14)
    assert heads["head_right"] == pytest.approx(
        HOUT + Q * Q / (2 * GRAVITY * HOUT ** 2), rel=1e-14)
    # a stationary shock dissipates energy: the head must *drop* through it
    assert heads["head_loss"] > 0.0
    assert heads["head_loss"] == pytest.approx(
        heads["head_left"] - heads["head_right"], rel=1e-14)
    # and it is not choked for a shallower bump
    with pytest.raises(ValueError, match="not a choked"):
        shock_heads(Q, 5.0, GRAVITY, BUMP_HEIGHT)


# ---------------------------------------------------------------------------
# 2. Smooth parts: Bernoulli residual + the independent steady-ODE integration
# ---------------------------------------------------------------------------
def test_smooth_parts_conserve_head_and_discharge():
    x = torch.cat([torch.linspace(0.0, 9.999, 200), torch.linspace(10.001, 24.999, 400)],
                  dim=0)
    st = bump_transcritical_state(x)
    z = bump_bed(x)
    head = torch.where(x <= 10.0,
                       torch.full_like(x, st["head_left"]),
                       torch.where(x <= st["x_shock"],
                                   torch.full_like(x, st["head_left"]),
                                   torch.full_like(x, st["head_right"])))
    resid = (st["h"] + Q * Q / (2 * GRAVITY * st["h"] ** 2) + z - head).abs()
    # ``head`` is flat in h near the crest (dH/dh = 0 at critical), so the
    # bisection inversion leaves an O(1e-8) head residual there; away from it the
    # residual is at the inversion tolerance.
    assert float(resid.max()) < 1e-7, float(resid.max())
    assert float((st["h"] * st["u"] - Q).abs().max()) < 1e-12


def test_crest_is_critical():
    st = bump_transcritical_state(torch.tensor([10.0]))
    hc = _heads()["critical_depth"]
    assert float(st["h"][0]) == pytest.approx(hc, rel=1e-7)
    # Froude number one: u = sqrt(g h)
    # Froude number one to the accuracy of the inversion (the head is flat there)
    assert float(st["u"][0] / math.sqrt(GRAVITY * float(st["h"][0]))) == pytest.approx(1.0, rel=1e-6)


def _asymptotic_depth_at(x: float, hc: float) -> float:
    """Depth a distance ``delta`` off the crest, from the local energy balance.

    ``E(h) = 1.5 h_c + (3/(2 h_c)) (h - h_c)^2 + O((h-h_c)^3)`` and the specific
    energy at the crest section is ``1.5 h_c + 0.05 delta^2`` (parabolic bed), so
    ``h - h_c = -+sqrt(2 h_c * 0.05 / 3) * delta``.  This is elementary analysis
    of the ODE's own coefficients -- it does not invert the Bernoulli cubic --
    and it is only used to *start* the RK4 away from the critical point, where
    ``1 - q^2/(g h^3) = 0`` makes the ODE singular.
    """
    delta = abs(x - 10.0)
    slope = math.sqrt(2.0 * hc * 0.05 / 3.0)
    return hc - slope * delta if x > 10.0 else hc + slope * delta


def test_analytic_branches_agree_with_independent_ode_integration():
    """The closed-form branches must equal the RK4 solution of the steady ODE.

    The ODE is integrated *away* from the crest on both sides and backwards
    from the outlet control, so no branch value of the module is used as an
    anchor except through the inlet/outlet data of the case itself.
    """
    st = bump_transcritical_state(torch.tensor([11.4, 12.5, 5.0]))
    hc = _heads()["critical_depth"]
    # forward along the lee slope (supercritical), started 1 cm past the crest
    h_ode_lee = _integrate_steady_ode(10.01, _asymptotic_depth_at(10.01, hc), 11.4)
    assert h_ode_lee == pytest.approx(float(st["h"][0]), rel=5e-4)
    # backward from the outlet control (h(25) = 0.33 is case data, not a branch)
    h_ode_down = _integrate_steady_ode(25.0, HOUT, 12.5)
    assert h_ode_down == pytest.approx(float(st["h"][1]), rel=2e-5)
    # backward along the upstream subcritical branch, started 1 cm before the crest
    h_ode_up = _integrate_steady_ode(9.99, _asymptotic_depth_at(9.99, hc), 5.0)
    assert h_ode_up == pytest.approx(float(st["h"][2]), rel=5e-4)


# ---------------------------------------------------------------------------
# 3. The jump: Rankine-Hugoniot (mass + momentum) and the energy drop
# ---------------------------------------------------------------------------
def test_stationary_jump_satisfies_rankine_hugoniot():
    st = bump_transcritical_state(torch.tensor([25.0]))
    h_l, h_r = st["shock_depths"]
    assert h_l < h_r                                     # lee side is the thin one
    # mass: q is the same on both sides by construction
    assert h_l * (Q / h_l) == pytest.approx(h_r * (Q / h_r), rel=1e-12)
    # momentum: the flux the jump must conserve
    m_l = float(momentum_flux(h_l))
    m_r = float(momentum_flux(h_r))
    assert m_l == pytest.approx(m_r, rel=1e-9), (m_l, m_r)
    # energy: *not* conserved across a jump -- the loss equals the head drop
    loss_jump = (h_l + Q * Q / (2 * GRAVITY * h_l ** 2)) - (h_r + Q * Q / (2 * GRAVITY * h_r ** 2))
    assert loss_jump > 0.0
    assert math.isclose(loss_jump, st["head_loss"], rel_tol=1e-6)
    # Froude numbers bracket one the right way round (Lax entropy condition)
    assert st["froude_left"] > 1.0 > st["froude_right"]


def test_shock_position_is_the_running_momentum_balance_root():
    x_s = shock_position()
    assert 11.25 < x_s < 12.0                     # on the lee slope, inside the bump
    def residual_at(x: float) -> float:
        st = bump_transcritical_state(torch.tensor([x]))
        return float(momentum_flux(st["h"][0]) - momentum_flux(st["shock_depths"][1]))

    # at the root the running momentum-flux residual vanishes ...
    assert abs(residual_at(x_s - 1e-6)) < 1e-5
    assert abs(residual_at(x_s + 1e-6)) < 1e-5
    # ... while 5 cm away it is O(1e-2): the location is discriminating
    assert abs(residual_at(x_s - 0.05)) > 5e-3
    assert abs(residual_at(x_s + 0.05)) > 5e-3


def test_profiles_are_piecewise_and_continuous_upstream():
    x = torch.linspace(*DOMAIN, 5001)
    st = bump_transcritical_state(x)
    h = st["h"]
    assert float(h.min()) > 0.0
    # monotone drop from inlet to the crest, jump up, then flat outlet value
    up = h[x <= 10.0]
    assert bool((up.diff() <= 1e-12).all())
    assert float(h[x >= 12.0].min()) == pytest.approx(HOUT, rel=1e-9)
    # the jump is the only place where the depth increases downstream of the crest
    lee = h[(x > 10.0) & (x < st["x_shock"])]
    assert bool((lee.diff() <= 1e-12).all())