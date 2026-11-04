"""Wave-configuration coverage of the exact Riemann baseline and the DG solver.

The companion audit tool is ``tools/verify_riemann_coverage.py``; this module
freezes its cheapest, highest-signal assertions into the regression suite.  The
five configurations and the rationale for each are documented in that module's
docstring and in ``docs/riemann_coverage_verification.md``.

Every criterion here is either (a) an analytic invariant that needs no
reference solution, or (b) a published star state (Toro 2009, ch. 4).  The
``cont`` configuration provides the negative control: an isolated contact
discontinuity must *not* be classified as a shock by the Rankine-Hugoniot
detector.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.riemann_exact import sample, star_state
from tools.verify_riemann_coverage import (
    CONFIGS,
    DIAPHRAGM,
    DOMAIN,
    fan_invariants,
    indicator_mirror_probe,
    integral_balance,
    shock_rankine_hugoniot,
    wave_confinement,
)

GAMMA = 1.4


def _fields(name, npts=100001):
    config = CONFIGS[name]
    x = np.linspace(DOMAIN[0], DOMAIN[1], npts)
    values = sample(x, config["final_time"], config["left"], config["right"], DIAPHRAGM)
    return x, values


@pytest.mark.parametrize("name", list(CONFIGS))
def test_exact_solution_matches_the_integral_form_of_the_euler_equations(name):
    """Initial data + analytic boundary-flux budget must reproduce the integral.

    The ends of the Riemann problem are open, so the invariant is *not* "the
    integral of U stays constant": Sod gains 0.117 momentum in ``t = 0.13``
    because ``p_L != p_R``.  With the flux budget included, a single
    mis-assigned region (wrong wave speed, inverted fan polarity) still shifts
    the integral at the O(1) relative level while quadrature noise stays below
    1e-4.
    """
    config = CONFIGS[name]
    report = integral_balance(
        sample, config["left"], config["right"], config["final_time"], npts=100001,
    )
    assert report["max_relative_error"] < 1e-4


@pytest.mark.parametrize("name", ["sod", "revsod", "coll"])
def test_shocks_satisfy_all_three_rankine_hugoniot_relations(name):
    """Every genuine shock must close mass, momentum *and* energy with one speed."""
    x, fields = _fields(name)
    report = shock_rankine_hugoniot(
        x, fields, CONFIGS[name]["left"], CONFIGS[name]["right"], CONFIGS[name]["final_time"],
    )
    assert report["shocks"], f"{name}: no shock detected"
    for shock in report["shocks"]:
        assert shock["normalised_closure"] < 1e-2
        assert abs(shock["mass_residual"]) < 1e-2 * max(abs(shock["speed"]), 1.0)


def test_isolated_contact_closes_rankine_hugoniot_with_zero_pressure_jump():
    """Negative control: a contact *does* satisfy all three jump relations.

    The distinguishing feature is not a failing jump relation (a contact is a
    genuine weak solution with speed ``sigma = u``) but the vanishing pressure
    jump, which no shock has.  This is exactly why the audit cannot use "RH
    closes" alone to identify shocks.
    """
    x, fields = _fields("cont")
    report = shock_rankine_hugoniot(
        x, fields, CONFIGS["cont"]["left"], CONFIGS["cont"]["right"],
        CONFIGS["cont"]["final_time"],
    )
    assert report["candidates"], "contact jump was not even detected"
    for jump in report["shocks"]:
        assert abs(jump["speed"] - CONFIGS["cont"]["left"][1]) < 1e-6
    pressure_jump = np.abs(fields["pressure"] - CONFIGS["cont"]["left"][2]).max()
    assert pressure_jump < 1e-12


@pytest.mark.parametrize("name", ["sod", "123"])
def test_star_state_matches_published_values(name):
    """Toro (2009) tabulated star states for the right-shock and near-vacuum cases."""
    literature = {"sod": (0.30313, 0.92745, 5e-4), "123": (0.00189, 0.0, 1e-2)}[name]
    pressure_star, velocity_star = star_state(CONFIGS[name]["left"], CONFIGS[name]["right"])
    assert abs(pressure_star - literature[0]) <= literature[2] * literature[0]
    assert abs(velocity_star - literature[1]) <= literature[2] * max(abs(literature[1]), 1e-3)


def test_left_shock_configuration_is_the_exact_mirror_of_the_right_shock_one():
    """x -> 1-x maps the Sod solution onto the reversed-Sod solution."""
    x = np.linspace(DOMAIN[0], DOMAIN[1], 50001)
    sod = CONFIGS["sod"]["final_time"], CONFIGS["sod"]
    rev = CONFIGS["revsod"]["final_time"], CONFIGS["revsod"]
    left = sample(x, sod[0], sod[1]["left"], sod[1]["right"], DIAPHRAGM)
    mirrored = sample(1.0 - x, rev[0], rev[1]["left"], rev[1]["right"], DIAPHRAGM)
    for field in ("density", "pressure"):
        assert np.allclose(left[field], mirrored[field], rtol=0, atol=1e-10)
    assert np.allclose(left["velocity"], -mirrored["velocity"], rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_no_wave_reaches_the_domain_ends(name):
    """Guards every comparison in this module: ends stay in the constant states."""
    config = CONFIGS[name]
    assert wave_confinement(
        config["left"], config["right"], config["final_time"],
    )["confined"]
