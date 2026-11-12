"""Tests of the canonical ``CASE-DG-DALEMBERT-ELASTIC-1D`` case module.

The case module (:mod:`tensordg.cases.elastic_rod_1d`) is the single source of
truth for the benchmark, the tests and downstream callers, so it is verified
here as a unit: parameter consistency, closed-form anchors, the three scenarios
(pulse / standing / reflection) and the benchmark driver end-to-end.
"""
import json
import math

import pytest
import torch

from tensordg.cases import elastic_rod_1d as case
from tensordg.elastic_1d import elastic_energy, elastic_wave_speed, \
    relative_error, stress_profile


def test_parameters_are_consistent_with_the_solver():
    p = case.parameters()
    assert p["case"] == case.CASE_KEY == "dalembert-elastic-rod-1d"
    assert case.C == pytest.approx(elastic_wave_speed(case.RHO, case.YOUNG))
    assert case.C == pytest.approx(1.0)
    assert p["nondimensional_groups"]["pulse_width_over_length"] == pytest.approx(
        case.PULSE_WIDTH / case.LENGTH)
    # dimensional cross-check: steel bar speed must sit in the handbook band
    c_steel = p["material_cross_check"]["steel"]["c"]
    assert 5100.0 < c_steel < 5200.0
    assert 5000.0 < p["material_cross_check"]["aluminium"]["c"] < 5150.0


def test_closed_form_anchors_are_the_textbook_values():
    anchors = {row["name"]: row["ref"] for row in case.anchor_table(case.PULSE_TIME)}
    assert anchors["bar wave speed c"] == pytest.approx(1.0)
    assert anchors["right-going packet position"] == pytest.approx(0.7)
    assert anchors["left-going packet position"] == pytest.approx(0.3)
    assert anchors["peak stress of one packet sigma_max"] == pytest.approx(0.5)
    assert anchors["clamped-free first eigenfrequency f1"] == pytest.approx(0.25)
    assert anchors["wall stress amplification (fixed end)"] == pytest.approx(2.0)
    assert case.probe_arrival_time(0.8) == pytest.approx(0.3)
    # exact energy of the Gaussian velocity pulse: 0.5 rho w sqrt(pi/2)
    assert anchors["elastic energy at t (periodic, conserved)"] == pytest.approx(
        0.5 * case.PULSE_WIDTH * math.sqrt(math.pi / 2.0))


def test_exact_pulse_obeys_the_characteristic_invariants():
    """The closed form must be a superposition of +-c travelling waves."""
    x = torch.linspace(0.0, 1.0, 4001, dtype=torch.float64)
    t = case.PULSE_TIME
    v, s = case.exact_pulse(x, t)
    c, rho = case.C, case.RHO
    v0 = lambda z: torch.exp(-((z - case.PULSE_CENTER) / case.PULSE_WIDTH) ** 2)
    # w+ = sigma - rho c v is advected with +c, w- = sigma + rho c v with -c
    assert torch.allclose(s - rho * c * v, -rho * c * v0(x - c * t), atol=1e-12)
    assert torch.allclose(s + rho * c * v, rho * c * v0(x + c * t), atol=1e-12)


def test_pulse_scenario_matches_the_closed_form_below_two_percent():
    x, _v, s, meta = case.sample("pulse", 200, 2, case.PULSE_TIME, case.DEFAULT_CFL)
    _v_ex, s_ex = case.exact_pulse(x, case.PULSE_TIME)
    assert meta["bc"] == "periodic" and meta["c"] == pytest.approx(1.0)
    assert relative_error(x, s, s_ex) < 0.02


def test_standing_scenario_hits_the_exact_eigenfrequency():
    """Wall-stress zero crossing -> f1 = c/(4L) = 0.25 (bisection, N=20 P2)."""
    def wall_sigma(t):
        _c, coeff, _m = case.solve_case("standing", 20, 2, t, 0.3)
        return float(case.boundary_values(coeff)[0][1])

    lo, hi = 0.7, 1.4
    f_lo = wall_sigma(lo)
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        if wall_sigma(mid) * f_lo > 0.0:
            lo, f_lo = mid, wall_sigma(mid)
        else:
            hi = mid
    f_got = 1.0 / (4.0 * 0.5 * (lo + hi))
    f_ref = case.clamped_free_frequency(1)
    assert abs(f_got - f_ref) / f_ref < 0.02


def test_reflection_scenario_doubles_the_wall_stress():
    _c, coeff, _m = case.solve_case("reflection", 100, 2,
                                    case.REFLECTION_TIME, 0.3)
    _left, right = case.boundary_values(coeff)
    got = float(right[1])
    ref = 2.0 * case.REFLECTION_AMPLITUDE
    assert abs(got - ref) / ref < 0.02


def test_unknown_scenario_and_bad_bc_inputs_are_rejected():
    with pytest.raises(ValueError):
        case.solve_case("torsion", 10)
    with pytest.raises(ValueError):
        case.boundary_condition("torsion")
    with pytest.raises(ValueError):
        case.initial_state("torsion")


def test_cases_registry_lists_the_elastic_rod_case():
    import tensordg.cases as cases_pkg
    assert cases_pkg.CASES["elastic_rod_1d"] == "dalembert-elastic-rod-1d"
    assert cases_pkg.solve_elastic_rod_case is case.solve_case


def test_benchmark_driver_writes_reference_vs_computed_artifacts(tmp_path):
    """End-to-end: the entry point must run and all anchor rows must pass."""
    from benchmarks.elastic_1d_dalembert import main

    rc = main([
        "--meshes", "40", "80", "--standing-meshes", "40",
        "--degrees", "1", "2", "--cfls", "0.3",
        "--probe-cells", "25", "--freq-cells", "16",
        "--output-dir", str(tmp_path),
    ])
    assert rc == 0
    for name in ("elastic_1d_dalembert.json", "benchmark_ref_vs_got.json",
                 "sigma_profile_ref_vs_got.csv", "run_manifest.json",
                 "elastic_1d_dalembert.png"):
        assert (tmp_path / name).is_file(), name

    bench = json.loads((tmp_path / "benchmark_ref_vs_got.json").read_text())
    assert bench["case"] == case.CASE_KEY
    assert bench["tolerance_pct"] == 2.0
    assert len(bench["rows"]) >= 8
    assert bench["all_rows_pass"], [r for r in bench["rows"] if r["verdict"] != "PASS"]
    assert bench["worst_rel_error_pct"] < 2.0
    assert "benchmarks.elastic_1d_dalembert" in bench["repro_cmd"]

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["elapsed_seconds"] >= 0.0
    assert set(manifest["artifacts"]) >= {
        "benchmark_ref_vs_got.json", "sigma_profile_ref_vs_got.csv"}

    payload = json.loads((tmp_path / "elastic_1d_dalembert.json").read_text())
    assert payload["gate"]["passed"] is True
    assert payload["degree_sweep"] and payload["cfl_sweep"]