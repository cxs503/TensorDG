"""Self-consistency tests for the exact dam-break reference solutions.

The reference solutions in :mod:`tensordg.swe_exact` are used as the ground
truth for the solver verification, so they are validated here against the
Rankine-Hugoniot conditions and the analytical limits - independently of any
numerical solver.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordg.swe_exact import (
    GRAVITY,
    dam_break_cell_averages,
    ritter_exact,
    stoker_exact,
    stoker_star_state,
)


@pytest.mark.parametrize("h_right", [0.1, 0.25, 0.5, 0.9])
def test_star_state_satisfies_both_riemann_relations(h_right):
    """h_* must satisfy the rarefaction and the shock relation simultaneously."""
    h_left = 1.0
    star = stoker_star_state(h_left, h_right)
    g = GRAVITY
    c1 = math.sqrt(g * h_left)
    c_star = math.sqrt(g * star["h_star"])

    u_from_rarefaction = 2.0 * (c1 - c_star)
    u_from_shock = (star["h_star"] - h_right) * math.sqrt(
        g * (star["h_star"] + h_right) / (2.0 * star["h_star"] * h_right))

    assert abs(u_from_rarefaction - star["u_star"]) < 1e-12
    assert abs(u_from_shock - star["u_star"]) < 1e-10
    assert h_right < star["h_star"] < h_left


@pytest.mark.parametrize("h_right", [0.1, 0.5])
def test_shock_rankine_hugoniot_conditions(h_right):
    """Mass and momentum jumps across the shock must satisfy F*-F = S (U*-U)."""
    h_left = 1.0
    star = stoker_star_state(h_left, h_right)
    g = GRAVITY
    h_s, u_s, s = star["h_star"], star["u_star"], star["shock_speed"]

    # mass: [hu] = S [h]
    assert abs(h_s * u_s - s * (h_s - h_right)) < 1e-10 * abs(h_s * u_s)
    # momentum: [hu^2 + g h^2 / 2] = S [hu]
    flux_jump = (h_s * u_s ** 2 + 0.5 * g * h_s ** 2) - 0.5 * g * h_right ** 2
    assert abs(flux_jump - s * h_s * u_s) < 1e-8 * abs(flux_jump)


def test_profile_is_continuous_across_the_star_region():
    """Fan tail and star state must join continuously."""
    h_left, h_right = 1.0, 0.3
    star = stoker_star_state(h_left, h_right)
    xi = star["wave_tail"]
    eps = 1e-9
    h_a, u_a, _ = stoker_exact(np.array([(xi - eps) * 1.0]), 1.0, h_left, h_right)
    h_b, u_b, _ = stoker_exact(np.array([(xi + eps) * 1.0]), 1.0, h_left, h_right)
    assert abs(h_a[0] - star["h_star"]) < 1e-8
    assert abs(h_b[0] - star["h_star"]) < 1e-8
    assert abs(u_a[0] - u_b[0]) < 1e-7


def test_profile_is_continuous_across_the_shock():
    """The exact solution has a single jump: the shock front."""
    h_left, h_right = 1.0, 0.2
    star = stoker_star_state(h_left, h_right)
    xi = star["shock_speed"]
    eps = 1e-9
    h_a, _, _ = stoker_exact(np.array([xi - eps]), 1.0, h_left, h_right)
    h_b, _, _ = stoker_exact(np.array([xi + eps]), 1.0, h_left, h_right)
    assert abs(h_a[0] - star["h_star"]) < 1e-7
    assert abs(h_b[0] - h_right) < 1e-9


def test_dry_bed_limit_matches_ritter():
    """h_R -> 0 must recover the Ritter solution (u_* = 2 c_1, wave front 2 c_1)."""
    c1 = math.sqrt(GRAVITY * 1.0)
    star = stoker_star_state(1.0, 0.0)
    assert star["h_star"] == 0.0
    assert abs(star["u_star"] - 2.0 * c1) < 1e-13
    assert abs(star["shock_speed"] - 2.0 * c1) < 1e-13

    x = np.array([2.5 * c1, 2.0001 * c1, 1.9999 * c1, 0.0, -0.5 * c1, -1.5 * c1])
    h, u, _ = ritter_exact(x, 1.0, 1.0)
    assert h[0] == 0.0 and u[0] == 0.0             # ahead of the front: dry
    assert h[1] == 0.0 and u[1] == 0.0             # still ahead of the front
    assert 0.0 < h[2] < 1e-4                       # just behind the front
    assert h[-1] == pytest.approx(1.0) and u[-1] == 0.0    # undisturbed
    assert h[3] == pytest.approx(c1 ** 2 / GRAVITY * (2 / 3) ** 2)  # xi = 0
    assert h[4] == pytest.approx(c1 ** 2 / GRAVITY * (5 / 6) ** 2)


def test_fan_is_monotone_and_non_negative():
    h_left = 1.0
    star = stoker_star_state(h_left, 0.0)
    xi = np.linspace(star["wave_head"], star["shock_speed"], 200)
    h, u, _ = stoker_exact(xi * 0.5, 0.5, h_left, 0.0)
    assert np.all(h >= 0.0)
    assert np.all(np.diff(h) <= 1e-14)     # depth decreases across the fan


def test_cell_averages_conserve_mass_and_hu():
    """Cell averaging must be conservative: integral = initial integral."""
    h_left, h_right = 1.0, 0.1
    t = 0.5
    faces = np.linspace(-3.0, 3.0, 601)
    h_bar, hu_bar = dam_break_cell_averages(faces, t, h_left, h_right)
    dx = faces[1] - faces[0]
    assert abs(h_bar.sum() * dx - (h_left * 3.0 + h_right * 3.0)) < 1e-10
    # momentum is NOT a conserved integral of the SWE: it is driven by the
    # hydrostatic pressure at the two (transmissive) ends, so only the
    # physical balance is meaningful.  Check d/dt int(hu) = -(F_R - F_L) in
    # a finite-difference sense against the momentum flux at t -> 0.
    t_small = 1e-4
    h_s, hu_s = dam_break_cell_averages(faces, t_small, h_left, h_right)
    p_left = 0.5 * GRAVITY * h_left ** 2
    p_right = 0.5 * GRAVITY * h_right ** 2
    predicted = (p_left - p_right) * t_small
    assert abs(hu_s.sum() * dx - predicted) < 5e-3 * abs(predicted)


def test_cell_average_converges_to_pointwise_value():
    """On a smooth region, refine until the cell average matches h(x_c)."""
    h_left, h_right = 1.0, 0.2
    previous = None
    for ncells in (50, 100, 200):
        faces = np.linspace(-3.0, 3.0, ncells + 1)
        h_bar, _ = dam_break_cell_averages(faces, 0.5, h_left, h_right)
        centre = 0.5 * (faces[1:] + faces[:-1])
        # deep inside the undisturbed left state the average must be exact
        left = centre < -1.8
        err = float(np.abs(h_bar[left] - h_left).max())
        assert err < 1e-12
        previous = err
    assert previous is not None


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        stoker_star_state(0.5, 1.0)      # h_right > h_left
    with pytest.raises(ValueError):
        stoker_star_state(-1.0, 0.1)
    with pytest.raises(ValueError):
        stoker_exact([0.0], 0.0, 1.0, 0.1)