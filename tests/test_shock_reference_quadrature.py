"""The cell-average reference instrument: exactness, bias, and gate consequences.

Companion to ``benchmarks/shock_reference_quadrature.py``.  The tests pin four
things:

1. the breakpoint-split reference is **exact** -- independent of the quadrature
   rule once the cells are split (so it carries no tolerance knob);
2. its wave positions are the analytic ones (checked against the exact solution
   itself and against the Rankine-Hugoniot relation at the shock);
3. the legacy fixed-rule reference has a bias that **does not vanish** under
   refinement (this is the defect the audit is about -- it is asserted, not
   asserted-away);
4. the documented Sod gates still hold when judged against the exact reference
   (numbers read from the committed artifact, so the gate table cannot drift
   away from the evidence).
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchmarks import shock_reference_quadrature as srq
from benchmarks.riemann_exact import sample
from benchmarks.sod_exact import cell_averages as legacy_cell_averages

SOD = ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1))
LAX = ((0.445, 0.698, 3.528), (0.5, 0.0, 0.571))
SOD_TIME = 0.2
LAX_TIME = 0.13
ARTIFACT = (Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "results"
            / "shock_reference_quadrature" / "metrics.json")


# --------------------------------------------------------------------------- #
# 1. the corrected reference is exact
# --------------------------------------------------------------------------- #
def test_exact_reference_is_independent_of_the_quadrature_rule():
    _, reference = srq.cell_averages(400, SOD_TIME, *SOD, quadrature_order=64)
    for order in (4, 8, 16, 40):
        _, other = srq.cell_averages(400, SOD_TIME, *SOD, quadrature_order=order)
        for key in srq.FIELDS:
            assert np.max(np.abs(other[key] - reference[key])) < 1e-13, (order, key)


def test_exact_cell_averages_match_brute_force_integration():
    """Independent cross-check: midpoint sum on a very fine sub-grid of one cell."""
    ncells, order = 40, 8
    centers, exact = srq.cell_averages(ncells, SOD_TIME, *SOD, quadrature_order=order)
    dx = 1.0 / ncells
    sub = 20001
    worst = 0.0
    for index in (0, int(0.35 * ncells), int(0.7 * ncells), ncells - 1):
        left = centers[index] - 0.5 * dx
        points = left + dx * (np.arange(sub) + 0.5) / sub
        values = sample(points, SOD_TIME, *SOD)
        for key in srq.FIELDS:
            brute = float(np.mean(values[key]))
            worst = max(worst, abs(brute - exact[key][index]))
    assert worst < 1e-9, worst


def test_exact_reference_is_conservative_in_the_integral_sense():
    """The cell averages must sum to the analytic integral of the exact solution."""
    ncells = 200
    centers, exact = srq.cell_averages(ncells, SOD_TIME, *SOD)
    dx = 1.0 / ncells
    finite = 400001
    points = (np.arange(finite) + 0.5) / finite
    values = sample(points, SOD_TIME, *SOD)
    for key in ("density", "velocity", "pressure"):
        integral_cells = float(np.sum(exact[key]) * dx)
        integral_direct = float(np.mean(values[key]))
        # the fine midpoint sum carries its own O(1/n_sub) kink error
        assert abs(integral_cells - integral_direct) < 1e-6, (key, integral_cells, integral_direct)


# --------------------------------------------------------------------------- #
# 2. wave positions are the analytic ones
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case, final_time", [(SOD, SOD_TIME), (LAX, LAX_TIME)])
def test_wave_positions_localise_the_jumps_of_the_exact_solution(case, final_time):
    """Every jump of the exact density profile sits at a reported position."""
    positions = srq.wave_positions(*case, final_time)
    grid = np.linspace(0.0, 1.0, 400001)
    density = sample(grid, final_time, *case)["density"]
    jumps = np.abs(np.diff(density))
    # only the two genuine discontinuities survive a 4e-6 grid; the rarefaction
    # edges are kinks and stay well below 10 % of the smallest jump.
    threshold = 0.2 * np.max(jumps)
    indices = np.where(jumps > threshold)[0]
    located = sorted({float(grid[i]) for i in indices})
    # cluster consecutive indices
    clusters = []
    for value in located:
        if not clusters or value - clusters[-1][-1] > 1e-4:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    found = sorted(float(np.mean(cluster)) for cluster in clusters)
    assert len(found) == 2, found
    expected = sorted([positions["contact"], positions["front"]])
    for got, want in zip(found, expected):
        assert abs(got - want) < 1e-4, (got, want)


def test_shock_position_satisfies_rankine_hugoniot():
    """The reported shock speed is the RH speed of the states it separates."""
    positions = srq.wave_positions(*SOD, SOD_TIME)
    gamma = 1.4
    speed = (positions["front"] - 0.5) / SOD_TIME
    eps = 1e-6
    behind = sample(np.array([positions["front"] - eps]), SOD_TIME, *SOD)
    ahead = sample(np.array([positions["front"] + eps]), SOD_TIME, *SOD)

    def flux(state):
        rho = float(state["density"][0])
        u = float(state["velocity"][0])
        p = float(state["pressure"][0])
        return np.array([rho * u, rho * u * u + p, u * (p * gamma / (gamma - 1.0) + 0.5 * rho * u * u)])

    def state_vector(state):
        rho = float(state["density"][0])
        u = float(state["velocity"][0])
        p = float(state["pressure"][0])
        return np.array([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u * u])

    rh = (flux(behind) - flux(ahead)) / (state_vector(behind) - state_vector(ahead))
    assert abs(float(np.mean(rh)) - speed) < 1e-6, (rh, speed)


def test_contact_carries_equal_pressure_and_velocity():
    positions = srq.wave_positions(*SOD, SOD_TIME)
    eps = 1e-6
    left = sample(np.array([positions["contact"] - eps]), SOD_TIME, *SOD)
    right = sample(np.array([positions["contact"] + eps]), SOD_TIME, *SOD)
    assert abs(float(left["pressure"][0]) - float(right["pressure"][0])) < 1e-9
    assert abs(float(left["velocity"][0]) - float(right["velocity"][0])) < 1e-9
    assert abs(float(left["density"][0]) - float(right["density"][0])) > 0.1  # real jump


# --------------------------------------------------------------------------- #
# 3. the legacy reference's bias does not vanish
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case, final_time", [(SOD, SOD_TIME), (LAX, LAX_TIME)])
def test_legacy_reference_bias_does_not_decay_under_refinement(case, final_time):
    """Pinned defect: the per-cell error of the fixed rule is O(jump), not O(dx)."""
    coarse = srq.reference_bias(400, final_time, *case)
    fine = srq.reference_bias(3200, final_time, *case)
    # 8x refinement shrinks dx by 8: an O(dx) instrument would show ~1/8 the error.
    assert fine["fields"]["density"]["max_abs"] > 0.5 * coarse["fields"]["density"]["max_abs"], (
        coarse["fields"]["density"]["max_abs"], fine["fields"]["density"]["max_abs"])
    # ... and the induced relative-L2 bias stays measurable at the finest gate mesh
    assert fine["fields"]["velocity"]["rms_rel"] > 1e-3, fine["fields"]["velocity"]


def test_more_quadrature_points_do_not_remove_the_bias():
    """A bigger fixed rule helps only the first few orders -- this is aliasing."""
    biases = [srq.reference_bias(800, SOD_TIME, *SOD, quadrature_order=order)
              for order in (4, 8, 16)]
    values = [row["fields"]["velocity"]["rms_rel"] for row in biases]
    assert min(values) > 1e-4, values


# --------------------------------------------------------------------------- #
# 4. the documented Sod gates under the exact reference
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not ARTIFACT.exists(), reason="audit artifact not committed")
def test_committed_gate_table_survives_the_exact_reference():
    metrics = json.loads(ARTIFACT.read_text())
    assert metrics["gates"], "no gates recorded"
    assert metrics["verdict"]["verdict_flips"] == [], metrics["verdict"]
    for gate in metrics["gates"]:
        if gate.get("status") != "measured":
            assert gate["reason"]          # a not_run gate must say why
            continue
        for key, field in gate["fields"].items():
            assert field["exact_rel_l2"] < metrics["tolerance"], (gate["key"], key, field)
            # the instrument shift must stay far below the gate itself
            assert abs(field["delta_pp"]) < 1.0, (gate["key"], key, field)


@pytest.mark.skipif(not ARTIFACT.exists(), reason="audit artifact not committed")
def test_committed_bias_ladder_matches_a_fresh_computation():
    metrics = json.loads(ARTIFACT.read_text())
    fresh = srq.reference_bias(400, SOD_TIME, *SOD)
    committed = next(row for row in metrics["bias"]["sod"] if row["ncells"] == 400)
    for key in srq.FIELDS:
        assert math.isclose(fresh["fields"][key]["rms_rel"],
                            committed["fields"][key]["rms_rel"], rel_tol=1e-12, abs_tol=1e-15)

def test_sod_conserved_cell_averages_match_analytic_integrals():
    # Waves are still interior. The boundary momentum flux is the pressure,
    # and both the mass and energy fluxes vanish at the two ends.
    _, ref = srq.cell_averages(40, SOD_TIME, *SOD, conserved=True)
    totals = [ref[field].mean() for field in ("density", "momentum", "energy")]
    np.testing.assert_allclose(totals, [0.5625, 0.18, 1.375], rtol=0., atol=2e-14)

    # The cell cut by the right shock contains exactly two constant states.
    n = 40
    front = srq.wave_positions(*SOD, SOD_TIME)["front"]
    index = int(n * front)
    fraction = n * front - index
    sides = sample(np.array([index / n, (index + 1) / n]), SOD_TIME, *SOD)
    rho, u, p = (sides[key] for key in srq.FIELDS)
    for key, pair in (("density", rho), ("momentum", rho*u),
                      ("energy", p/(srq.GAMMA-1) + 0.5*rho*u**2)):
        expected = fraction*pair[0] + (1-fraction)*pair[1]
        assert ref[key][index] == pytest.approx(expected, abs=2e-14, rel=0.)
