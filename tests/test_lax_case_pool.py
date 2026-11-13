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
    CFL, DIAPHRAGM, FINAL_TIME, GAMMA, LEFT, MIN_C3_ORDER, OUT, R3_BAND_WIDTH,
    R3_LEGACY_BAND_CELLS, R3_ORDER_FRACTION, RIGHT, TIME_CFL_DIVISORS, TIME_LEGS,
    TOL_C1, TOL_C2, TOL_TIME_ERROR_SHARE, exact_positions, exact_reference,
    exact_waves, entropy_wave_exact, gate_c1, gate_c2, modal_fields,
    plateau_values, convergence_order, temporal_order, temporal_share,
    wave_positions, _weighted_l1, ENTROPY_AMPLITUDE, ENTROPY_TIME,
    RAREFACTION_LEVEL_FRACTION,
)
from tensordg.euler_dg_1d import solve_euler_dg_1d, solve_periodic_euler_dg_1d

# A gate pinned to ``runs/`` never fires on ``main``: the merge gate drops that
# tree (0 tracked files), so the module skipped forever instead of asserting.
# Resolve from ``__file__`` and keep the record on the tracked ``docs/`` path.
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / OUT / "metrics.json"
PROFILES = ARTIFACT.parent / "profiles.npz"


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
    if name == "time":
        shares = [v["worst_share"] for v in gate["per_leg"].values()]
        ok = all(s is not None and s < gate["tolerance"] for s in shares)
        return "PASS" if ok else "FAIL"
    if name == "r5":
        ok = all((1.0 - gate["fraction"]) * v["design_order"] <= v["observed_order"]
                 <= (1.0 + gate["fraction"]) * v["design_order"]
                 for v in gate["cases"].values())
        return "PASS" if ok else "FAIL"
    raise AssertionError(name)


def test_committed_artifact_is_self_consistent():
    assert ARTIFACT.exists(), (
        f"missing committed record {ARTIFACT.relative_to(ROOT)}; regenerate with "
        "`python -m benchmarks.lax_case_pool`")
    metrics = json.loads(ARTIFACT.read_text())
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
    assert tolerances["r3_band_width"] == pytest.approx(R3_BAND_WIDTH)
    assert tolerances["r3_band_cells_legacy"] == pytest.approx(R3_LEGACY_BAND_CELLS)
    assert tolerances["time_error_share"] == pytest.approx(TOL_TIME_ERROR_SHARE)
    assert metrics["gates"]["r3"]["band"]["width"] == pytest.approx(R3_BAND_WIDTH)
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


def test_committed_artifact_profiles_agree_with_the_exact_solution():
    assert PROFILES.exists(), (
        f"missing committed record {PROFILES.relative_to(ROOT)}; regenerate with "
        "`python -m benchmarks.lax_case_pool`")
    metrics = json.loads(ARTIFACT.read_text())
    profiles = np.load(PROFILES)
    points = profiles["points"]
    for field in ("density", "velocity", "pressure"):
        numerical = profiles[f"numerical_{field}"]
        exact = profiles[f"exact_{field}"]
        error = np.sum(np.abs(numerical - exact)) / np.sum(np.abs(exact))
        assert error < 0.05, field
    finest = metrics["ladder"][-1]
    assert (finest["cells"], finest["degree"]) == (400, 3)
    assert points.shape[0] == finest["cells"]

# ------------------------------------------- region-rule invariant (m16606 F1)
def test_r3_region_rule_is_nested_across_the_ladder():
    """The R3 band must not move with the mesh, or the measured order is bogus.

    A ``4 dx`` band shrinks as the mesh is refined, so the retained measure
    *grows* with N (m16606 measured 0.680 -> 0.920, +35 %) and the analytic-region
    error then rises with refinement.  The fixed physical band is nested: the
    retained measure is the same region at every resolution.
    """
    assert ARTIFACT.exists(), f"missing committed record {ARTIFACT}"
    metrics = json.loads(ARTIFACT.read_text())
    nesting = metrics["diagnostics"]["r3_nesting"]
    for degree, data in nesting.items():
        assert data["nested_spread"] < 2e-3, (degree, data["nested_fractions"])
        # the retired rule is kept as the counter-example, and must stay one:
        # if a future edit made it nested again, this test flags the stale claim
        assert data["legacy_grows_with_refinement"] is True, (degree, data)
        assert data["legacy_spread"] > 0.1, (degree, data["legacy_fractions"])


def test_stored_nested_errors_reproduce_the_r3_verdict():
    """Recompute R3 from the stored nested-band errors, never from the verdict."""
    metrics = json.loads(ARTIFACT.read_text())
    rows = metrics["gates"]["r3"]["per_leg"]
    for degree in (1, 2, 3):
        legs = sorted([e for e in metrics["ladder"] if e["degree"] == degree],
                      key=lambda e: e["cells"])
        coarse, fine = legs[-2], legs[-1]
        order = convergence_order(
            coarse["cells"], coarse["analytic_errors"]["density"],
            fine["cells"], fine["analytic_errors"]["density"])
        assert order == pytest.approx(rows[f"p{degree}.density"], rel=1e-9)
    # the retired rule is adjudicated side by side and is *not* allowed to pass
    # silently: it is recorded, and it disagrees with the published rule
    assert "r3_legacy_band" in metrics["diagnostics"]
    assert metrics["diagnostics"]["r3_legacy_band"]["status"] in ("PASS", "FAIL")


# --------------------------------------------- temporal-error separation (F5)
def test_temporal_order_is_exact_for_a_known_dt_decay():
    """E(dt) = E0 + C dt**q with exact halvings -> q recovered in closed form."""
    for e0, c, q in ((0.0, 0.4, 2.0), (0.01, 0.3, 3.0), (0.05, 0.2, 1.5)):
        dts = (0.04, 0.02, 0.01)
        errors = [e0 + c * dt ** q for dt in dts]
        assert temporal_order(errors) == pytest.approx(q, rel=1e-9)
        if e0 == 0.0:
            # pure temporal error: the share of the coarse error is all of it
            assert temporal_share(errors, temporal_order(errors)) == pytest.approx(1.0)


def test_temporal_order_refuses_a_plateau_or_a_reversal():
    assert temporal_order([1e-3, 1e-3, 1e-3]) is None      # nothing to split
    assert temporal_order([1e-3, 2e-3, 4e-3]) is None      # error grows with dt->0
    assert temporal_share([1e-3, 1e-3, 1e-3], None) is None


def test_temporal_share_never_exceeds_the_total_error():
    errors = [2.0e-2, 1.4e-2, 1.1e-2]
    share = temporal_share(errors, temporal_order(errors))
    assert share is not None
    assert 0.0 < share < 1.0


def test_committed_time_refinement_separates_the_dt_term():
    """The ladder order is only meaningful if ``dt`` is not doing the work."""
    assert ARTIFACT.exists(), f"missing committed record {ARTIFACT}"
    metrics = json.loads(ARTIFACT.read_text())
    assert metrics["config"]["quick"] is False
    rows = metrics["time_refinement"]
    assert rows, "the dt-refinement study must be part of the committed record"
    assert [tuple(leg) for leg in metrics["config"]["time_study"]["legs"]] == list(TIME_LEGS)
    assert tuple(metrics["config"]["time_study"]["cfl_divisors"]) == TIME_CFL_DIVISORS
    for row in rows:
        levels = row["levels"]
        assert len(levels) == len(TIME_CFL_DIVISORS)
        # only dt moves: the mesh is held fixed for the whole leg
        assert len({row["cells"] for _ in levels}) == 1
        dts = [level["dt"] for level in levels]
        assert all(a > b for a, b in zip(dts, dts[1:])), dts  # strictly refined
        for field in ("density", "velocity", "pressure"):
            series = row["fields"][field]["relative_l1"]
            assert [level["relative_l1"][field] for level in levels] == series
            assert all(math.isfinite(v) and v > 0.0 for v in series)
            order = temporal_order(series)
            assert row["fields"][field]["temporal_order"] == order
            share = row["fields"][field]["temporal_share"]
            bound = row["fields"][field]["temporal_bound"]
            assert row["fields"][field]["temporal_bound"] == bound
            assert share is not None or bound is not None
            if order is not None:
                assert share is not None and 0.0 < share < 1.0
            else:
                # order not identifiable (flat / non-monotone): the 4x-dt
                # sensitivity is the honest upper bound on the temporal term
                assert share is None and bound is not None
    gate = metrics["gates"]["time"]
    assert _recheck_gate("time", gate) == gate["status"]
    assert gate["tolerance"] == pytest.approx(TOL_TIME_ERROR_SHARE)


def test_rendered_followup_doc_matches_the_artifact():
    """The follow-up note must be a pure function of the committed metrics."""
    from scripts.render_lax_case_pool_doc import DOC, render

    assert ARTIFACT.exists(), f"missing committed record {ARTIFACT}"
    assert DOC.exists(), (
        f"missing {DOC.relative_to(ROOT)}; regenerate with "
        "`python scripts/render_lax_case_pool_doc.py`")
    assert DOC.read_text() == render(json.loads(ARTIFACT.read_text())), (
        f"{DOC.name} drifted from the artifact; re-run "
        "`python scripts/render_lax_case_pool_doc.py`")
