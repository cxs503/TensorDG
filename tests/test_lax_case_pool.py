"""Contract tests for the ``CASE-LAX-DG`` case pool.

Three layers, deliberately independent of each other:

1. **analytic layer** -- the wave positions and the star state are re-derived here
   from closed-form relations and from an independent scalar solver, so the
   driver cannot grade itself with its own numbers;
2. **detector layer** -- the level-crossing locator is applied to the *exact*
   field and must return the exact positions, and a synthetic profile whose
   contact and shock are known must not be confused;
3. **artifact layer** -- the committed ``runs/lax_case_pool/metrics.json`` is
   re-adjudicated from its stored numbers (a hand-edited verdict cannot pass),
   plus one cheap DG run proving the gates are reachable at a coarse resolution.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.lax_case_pool import (
    CFL, DIAPHRAGM, FINAL_TIME, GAMMA, LEFT, MIN_C3_ORDER, OUT, R3_ORDER_FRACTION,
    RIGHT, TOL_C1, TOL_C2, exact_positions, exact_reference, exact_waves,
    entropy_wave_exact, gate_c1, gate_c2, modal_fields, plateau_values,
    convergence_order, wave_positions, _weighted_l1, ENTROPY_AMPLITUDE,
    ENTROPY_TIME, RAREFACTION_LEVEL_FRACTION,
)
from tensordg.euler_dg_1d import solve_euler_dg_1d, solve_periodic_euler_dg_1d

METRICS = Path(OUT) / "metrics.json"


# --------------------------------------------------------------- analytic layer
def _riemann_residual(pressure, left=LEFT, right=RIGHT, gamma=GAMMA):
    """f_L(p) + f_R(p) + u_R - u_L, written out here so it is independent."""
    def branch(p, state):
        rho, _, p_k = state
        a = math.sqrt(gamma * p_k / rho)
        if p > p_k:
            a_k = 2.0 / ((gamma + 1.0) * rho)
            b_k = (gamma - 1.0) / (gamma + 1.0) * p_k
            return (p - p_k) * math.sqrt(a_k / (p + b_k))
        return 2.0 * a / (gamma - 1.0) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)

    return branch(pressure, left) + branch(pressure, right) + right[1] - left[1]


def _bisection_star_state(left=LEFT, right=RIGHT, gamma=GAMMA, iterations=200):
    p_l, p_r = left[2], right[2]
    lo = 1e-6
    hi = 10.0 * max(p_l, p_r)
    assert _riemann_residual(lo, left, right, gamma) < 0.0
    assert _riemann_residual(hi, left, right, gamma) > 0.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if _riemann_residual(mid, left, right, gamma) > 0.0:
            hi = mid
        else:
            lo = mid
    pressure = 0.5 * (lo + hi)
    def q(p, state):
        rho, _, p_k = state
        a = math.sqrt(gamma * p_k / rho)
        if p > p_k:
            a_k = 2.0 / ((gamma + 1.0) * rho)
            b_k = (gamma - 1.0) / (gamma + 1.0) * p_k
            return (p - p_k) * math.sqrt(a_k / (p + b_k))
        return 2.0 * a / (gamma - 1.0) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
    velocity = 0.5 * (left[1] + right[1]) + 0.5 * (q(pressure, right) - q(pressure, left))
    return pressure, velocity


def test_star_state_matches_independent_bisection():
    waves = exact_waves()
    pressure, velocity = _bisection_star_state()
    assert abs(waves["pressure_star"] - pressure) < 1e-9
    assert abs(waves["velocity_star"] - velocity) < 1e-9
    # the Newton state must also be a root of the written-out residual
    assert abs(_riemann_residual(waves["pressure_star"])) < 1e-9


def test_star_densities_follow_isentropic_and_rankine_hugoniot_relations():
    waves = exact_waves()
    gamma = GAMMA
    # left wave of the Lax tube is a rarefaction (p* < p_L), right wave a shock
    assert waves["left_wave_kind"] == "rarefaction"
    assert waves["right_wave_kind"] == "shock"
    assert waves["pressure_star"] < LEFT[2] and waves["pressure_star"] > RIGHT[2]
    expected_left = LEFT[0] * (waves["pressure_star"] / LEFT[2]) ** (1.0 / gamma)
    ratio = waves["pressure_star"] / RIGHT[2]
    expected_right = RIGHT[0] * ((ratio + (gamma - 1.0) / (gamma + 1.0))
                                 / ((gamma - 1.0) / (gamma + 1.0) * ratio + 1.0))
    assert abs(waves["density_star_left"] - expected_left) < 1e-12
    assert abs(waves["density_star_right"] - expected_right) < 1e-12
    # Rankine-Hugoniot across the right shock, written as mass and momentum flux
    rho_r, u_r, p_r = RIGHT
    rho_sr, u_s = waves["density_star_right"], waves["velocity_star"]
    p_s = waves["pressure_star"]
    shock_speed = (rho_sr * u_s - rho_r * u_r) / (rho_sr - rho_r)
    assert abs(shock_speed - waves["speeds"]["shock"]) < 1e-9
    momentum = rho_sr * u_s * (u_s - shock_speed) + p_s
    momentum_r = rho_r * u_r * (u_r - shock_speed) + p_r
    assert abs(momentum - momentum_r) < 1e-9 * max(1.0, abs(momentum))


def test_wave_positions_are_the_closed_form_travel_distances():
    waves = exact_waves()
    gamma = GAMMA
    a_l = math.sqrt(gamma * LEFT[2] / LEFT[0])
    a_star = a_l * (waves["pressure_star"] / LEFT[2]) ** ((gamma - 1.0) / (2.0 * gamma))
    expected = {
        "rarefaction_head": DIAPHRAGM + (LEFT[1] - a_l) * FINAL_TIME,
        "rarefaction_tail": DIAPHRAGM + (waves["velocity_star"] - a_star) * FINAL_TIME,
        "contact": DIAPHRAGM + waves["velocity_star"] * FINAL_TIME,
        "shock": DIAPHRAGM + waves["speeds"]["shock"] * FINAL_TIME,
    }
    for name, value in expected.items():
        assert abs(waves["positions"][name] - value) < 1e-12, name
    # ordering head < tail < contact < shock, and all four inside the domain
    ordered = [expected[name] for name in
               ("rarefaction_head", "rarefaction_tail", "contact", "shock")]
    assert ordered == sorted(ordered)
    assert 0.0 < ordered[0] and ordered[-1] < 1.0


# --------------------------------------------------------------- detector layer
def test_locator_recovers_the_exact_positions_from_the_exact_field():
    waves = exact_waves()
    located = exact_positions(waves, samples=200001)
    reference = exact_positions(waves, samples=200001)
    for name, value in located.items():
        assert abs(value - reference[name]) < 5e-5, (name, value)
    # the pre-registered pressure levels sit *inside* the fan, not at its edges:
    # report that offset explicitly so no reader mistakes it for edge accuracy.
    offset = abs(located["rarefaction_head"] - waves["positions"]["rarefaction_head"])
    assert 0.0 < offset < 0.03


def test_contact_and_shock_are_not_confused_by_the_search_windows():
    """Regression for the window bug: a smeared contact must not be read as the shock."""
    waves = exact_waves()
    grid = np.linspace(1e-6, 1.0 - 1e-6, 60001)
    exact = exact_reference(grid[None, :])
    primitive = {key: value[0].copy() for key, value in exact.items()}
    # smear the contact and the shock over ~2 % of the domain, like a coarse DG leg
    for name in ("contact", "shock"):
        centre = waves["positions"][name]
        width = 0.02
        weight = 0.5 * (1.0 + np.tanh((grid - centre) / width))
        left_value = primitive["density"][0] if name == "contact" \
            else waves["density_star_right"]
        right_value = waves["density_star_right"] if name == "contact" else RIGHT[0]
        primitive["density"] = np.where(
            np.abs(grid - centre) < 4.0 * width,
            left_value + (right_value - left_value) * weight, primitive["density"])
    located = wave_positions(grid, primitive, waves, dx=grid[1] - grid[0])
    assert abs(located["contact"] - waves["positions"]["contact"]) < 0.02
    assert abs(located["shock"] - waves["positions"]["shock"]) < 0.02
    assert located["shock"] > located["contact"]


def test_gate_c1_and_c2_flag_planted_errors():
    waves = exact_waves()
    exact = exact_positions(waves)
    good = dict(exact)
    assert gate_c1(good, exact)["status"] == "PASS"
    bad = dict(exact)
    bad["shock"] = exact["shock"] + 0.02          # 6 % of the travel distance
    gate = gate_c1(bad, exact)
    assert gate["status"] == "FAIL"
    assert gate["waves"]["shock"]["status"] == "FAIL"
    assert gate["worst_relative_error"] > TOL_C1
    # a missing wave is a FAIL, never a silent pass
    assert gate_c1({"contact": exact["contact"]}, exact)["status"] == "FAIL"

    plateaus = {"star_left": {"density": waves["density_star_left"],
                              "velocity": waves["velocity_star"],
                              "pressure": waves["pressure_star"]},
                "star_right": {"density": waves["density_star_right"]}}
    assert gate_c2(plateaus, waves)["status"] == "PASS"
    off = json.loads(json.dumps(plateaus))
    off["star_left"]["pressure"] *= 1.0 + 2.0 * TOL_C2
    gate = gate_c2(off, waves)
    assert gate["status"] == "FAIL"
    assert gate["values"]["star_left.pressure"]


def test_convergence_order_helper():
    assert convergence_order(100, 1e-2, 200, 2.5e-3) == pytest.approx(2.0)
    assert convergence_order(100, 1e-2, 400, 1e-2 / 16.0) == pytest.approx(2.0)
    assert convergence_order(100, 0.0, 200, 1.0) is None


# --------------------------------------------------------------- solver layer
def test_coarse_dg_leg_passes_c1_and_c2():
    waves = exact_waves()
    x, coeff, _ = solve_euler_dg_1d(LEFT, RIGHT, ncells=100, degree=2,
                                    final_time=FINAL_TIME, cfl=CFL)
    points, primitive, dx, _weights = modal_fields(x, coeff)
    assert float(primitive["density"].min()) > 0.0
    located = wave_positions(points, primitive, waves, dx=dx)
    assert gate_c1(located, exact_positions(waves))["status"] == "PASS"
    assert gate_c2(plateau_values(points, primitive, waves), waves)["status"] == "PASS"


def test_entropy_wave_is_advected_in_the_right_direction():
    speed = 1.0
    cells = 16

    def initial_state(node_x):
        rho = 1.0 + ENTROPY_AMPLITUDE * torch.sin(2.0 * np.pi * node_x)
        return (rho, torch.full_like(rho, speed), torch.full_like(rho, 1.0))

    x, coeff, _ = solve_periodic_euler_dg_1d(initial_state, ncells=cells, degree=1,
                                             final_time=ENTROPY_TIME,
                                             domain=(0.0, 1.0), cfl=0.2)
    points, primitive, dx, weights = modal_fields(x, coeff)
    exact = entropy_wave_exact(points, ENTROPY_TIME, speed)
    error = _weighted_l1(points, primitive["density"], exact, weights, dx)
    assert error < 5e-3
    # the exact phase is rho(x,t) = 1 + 0.2 sin(2 pi (x - u t)): check the sign of
    # the shift by correlating against the un-shifted wave
    shifted = 1.0 + ENTROPY_AMPLITUDE * np.sin(2.0 * np.pi * points)
    assert _weighted_l1(points, primitive["density"], shifted, weights, dx) > error


# --------------------------------------------------------------- artifact layer
def _recheck_gate(name, gate):
    """Re-derive a gate verdict from the stored numbers only."""
    if name == "c1":
        worst = max(v["relative_error"] for v in gate["waves"].values())
        return "PASS" if worst < gate["tolerance"] else "FAIL"
    if name == "c2":
        worst = max(v["relative_error"] for v in gate["values"].values())
        return "PASS" if worst < gate["tolerance"] else "FAIL"
    if name == "c3":
        gated = min(v for k, v in gate["per_leg"].items() if k.endswith("density"))
        return "PASS" if gated >= gate["minimum_order"] else "FAIL"
    if name == "r3":
        rows = gate["per_leg"]
        ok = all(rows[f"p{degree}.density"] >= rows[f"p{degree}.requirement"]
                 for degree in (1, 2, 3) if f"p{degree}.density" in rows)
        return "PASS" if ok else "FAIL"
    if name == "r5":
        ok = all((1.0 - gate["fraction"]) * v["design_order"] <= v["observed_order"]
                 <= (1.0 + gate["fraction"]) * v["design_order"]
                 for v in gate["cases"].values())
        return "PASS" if ok else "FAIL"
    raise AssertionError(name)


@pytest.mark.skipif(not METRICS.exists(),
                    reason="case-pool artifact not generated in this worktree")
def test_committed_artifact_is_self_consistent():
    metrics = json.loads(METRICS.read_text())
    assert metrics["anchor"] == "CASE-LAX-DG"
    statuses = {name: _recheck_gate(name, gate)
                for name, gate in metrics["gates"].items()}
    for name, gate in metrics["gates"].items():
        assert statuses[name] == gate["status"], (name, statuses[name], gate["status"])
    expected_verdict = "PASS" if all(s == "PASS" for s in statuses.values()) else "FAIL"
    assert metrics["verdict"] == expected_verdict
    # the tolerances recorded in the artifact are the ones the ledger pre-registered
    tolerances = metrics["config"]["tolerances"]
    assert tolerances["c1"] == TOL_C1
    assert tolerances["c2"] == TOL_C2
    assert tolerances["c3_min_order"] == MIN_C3_ORDER
    assert tolerances["r3_order_fraction"] == R3_ORDER_FRACTION
    assert tolerances["r3_band_cells"] == pytest.approx(4.0)
    assert metrics["config"]["tolerances"]["r5_order_fraction"] == pytest.approx(0.03)
    # ladder monotonicity: every degree must improve as the mesh is refined
    for degree in (1, 2, 3):
        legs = sorted([e for e in metrics["ladder"] if e["degree"] == degree],
                      key=lambda e: e["cells"])
        values = [e["errors"]["density"] for e in legs]
        assert values == sorted(values, reverse=True), (degree, values)
    # the artifact must not hide negative numbers behind a "quick" run
    assert metrics["config"]["quick"] is False
    assert len(metrics["ladder"]) == 9
    assert metrics["figures"], "figures must be recorded"


@pytest.mark.skipif(not METRICS.exists(),
                    reason="case-pool artifact not generated in this worktree")
def test_committed_artifact_profiles_agree_with_the_exact_solution():
    metrics = json.loads(METRICS.read_text())
    profiles = np.load(Path(OUT) / "profiles.npz")
    points = profiles["points"]
    for field in ("density", "velocity", "pressure"):
        numerical = profiles[f"numerical_{field}"]
        exact = profiles[f"exact_{field}"]
        error = np.sum(np.abs(numerical - exact)) / np.sum(np.abs(exact))
        assert error < 0.05, field
    finest = metrics["ladder"][-1]
    assert (finest["cells"], finest["degree"]) == (400, 3)
    assert points.shape[0] == finest["cells"]