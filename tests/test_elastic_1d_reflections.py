"""Tests for the long-time (multi-reflection) d'Alembert reference.

The case ``CASE-DG-DALEMBERT-ELASTIC-1D`` grades the DG solver against the
*single-image* closed form, which is only exact before the packets wrap around
the periodic seam.  These tests pin the image (reflection-series) reference that
remains exact for arbitrary time, and -- equally important -- pin the property
that the metric rejects a wrong-sign reflection series, i.e. the gate is not
vacuous.
"""

import json
import math

import pytest
import torch

import benchmarks.elastic_1d_reflections as bench
from tensordg.elastic_1d import (
    _dalembert_fixed_free_images,
    _dalembert_periodic_images,
    _seam_limit,
    dalembert_pulse,
    gaussian_pulse,
    relative_error,
    stress_profile,
)

RHO = YOUNG = 1.0
C = 1.0
LENGTH = 1.0
CENTER = 0.5
WIDTH = 0.05


def _grid(n=4001):
    return torch.linspace(0.001, 0.999, n, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# the reference itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t", [0.0, 0.1, 0.2])
def test_image_sum_reduces_to_the_single_image_form_before_the_seam(t):
    """Inside the old validity window the two references must coincide."""
    x = _grid()
    v_ref, s_ref = dalembert_pulse(x, t, RHO, YOUNG, CENTER, WIDTH, LENGTH)
    v_img, s_img = _dalembert_periodic_images(x, t, LENGTH, RHO, YOUNG,
                                              CENTER, WIDTH)
    assert torch.allclose(v_ref, v_img, atol=1e-12)
    assert torch.allclose(s_ref, s_img, atol=1e-12)


def test_finite_pulse_width_shrinks_the_practical_validity_window():
    """The tail crosses the seam *before* the packet centre does.

    Measured (not assumed): at ``c t = 0.30`` the single-image form is still
    exact to ~1e-7, while at ``c t = 0.40`` -- still inside the nominal
    ``min(center, L-center) = 0.5`` window -- it is already off by ~1.7%, which
    is at the 2% acceptance tolerance.  The image sum has neither problem.
    """
    scan = bench.reference_validity_scan()
    by_ct = {round(r["c_t_over_L"], 3): r for r in scan["scan"]}
    assert by_ct[0.3]["max_stress_deviation"] < 1e-6      # 4.8e-8 measured
    assert by_ct[0.4]["deviation_pct"] > 1.0
    # 5.5e-5 absolute deviation at c t = 0.35 -> still "exact to 1e-4"
    assert scan["measured_practical_limit_c_t"] == pytest.approx(0.35)
    assert scan["three_width_rule_c_t"] == pytest.approx(CENTER - 3 * WIDTH)
    assert scan["first_c_t_at_or_above_tolerance"] is not None
    assert scan["first_c_t_at_or_above_tolerance"] < scan["nominal_limit_c_t"]


def test_periodic_reassembly_at_one_transit_doubles_velocity_and_kills_stress():
    """At ``t = L/c`` both packets are back at the centre: exact, non-trivial."""
    x = _grid()
    v, s = _dalembert_periodic_images(x, LENGTH / C, LENGTH, RHO, YOUNG,
                                      CENTER, WIDTH)
    assert float(v.max()) == pytest.approx(1.0, rel=1e-12)      # 0.5 + 0.5
    assert float(s.abs().max()) < 1e-12                          # g+ == g-
    # ...and the single-image form cannot reproduce it (it decays instead).
    v_plain, _ = dalembert_pulse(x, LENGTH / C, RHO, YOUNG, CENTER, WIDTH,
                                 LENGTH, strict=False)
    assert float(v_plain.max()) < 0.05


@pytest.mark.parametrize("t", [0.5, 1.5, 3.5, 7.5])
def test_clamped_free_image_sum_satisfies_both_boundary_conditions(t):
    """Clamped end ``v=0`` (odd image), free end ``sigma=0`` (even image)."""
    x = torch.tensor([0.0, LENGTH], dtype=torch.float64)
    v, s = _dalembert_fixed_free_images(x, t, LENGTH, RHO, YOUNG, CENTER, WIDTH)
    assert abs(float(v[0])) < 1e-12, "clamped end must hold v = 0"
    assert abs(float(s[1])) < 1e-12, "free end must hold sigma = 0"


@pytest.mark.parametrize("bc_kind", ["periodic", "clamped-free"])
def test_image_reference_initial_state_is_stress_free_and_velocity_pulse(bc_kind):
    """Both references start from ``v = V(x)``, ``sigma = 0`` -- the DG case IC."""
    x = _grid()
    sol = (_dalembert_periodic_images if bc_kind == "periodic"
           else _dalembert_fixed_free_images)
    v, s = sol(x, 0.0, LENGTH, RHO, YOUNG, CENTER, WIDTH)
    assert torch.allclose(v, gaussian_pulse(x, CENTER, WIDTH), atol=1e-12)
    assert float(s.abs().max()) < 1e-12


def test_seam_limit_is_the_minimum_distance_to_the_closest_end():
    assert _seam_limit(LENGTH, CENTER) == pytest.approx(0.5)
    assert _seam_limit(LENGTH, 0.3) == pytest.approx(0.3)
    assert _seam_limit(2.0, 0.5) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# the defect: the old form was a *silently wrong* reference past the seam
# --------------------------------------------------------------------------- #
def test_single_image_form_refuses_to_be_a_silent_wrong_reference():
    x = _grid(201)
    t_bad = 1.5 * _seam_limit(LENGTH, CENTER)      # c t = 0.75 > 0.5
    with pytest.raises(ValueError, match="only exact for"):
        dalembert_pulse(x, t_bad, RHO, YOUNG, CENTER, WIDTH, LENGTH)
    # the old behaviour stays reachable, but only on explicit request
    v, s = dalembert_pulse(x, t_bad, RHO, YOUNG, CENTER, WIDTH, LENGTH,
                           strict=False)
    v_ok, _ = _dalembert_periodic_images(x, t_bad, LENGTH, RHO, YOUNG,
                                         CENTER, WIDTH)
    assert float((v - v_ok).abs().max()) > 0.1 * float(v_ok.abs().max())


def test_existing_short_time_call_sites_stay_inside_the_seam_limit():
    """The guard must not invalidate the shipped case (t = 0.2 << 0.5)."""
    from tensordg.cases import elastic_rod_1d as case

    assert case.C * case.PULSE_TIME <= _seam_limit(case.LENGTH, case.PULSE_CENTER)
    v, s = case.exact_pulse(_grid(), case.PULSE_TIME)   # must not raise
    assert float(v.abs().max()) > 0.0


# --------------------------------------------------------------------------- #
# end-to-end: DG vs the image reference in the multi-reflection regime
# --------------------------------------------------------------------------- #
def test_periodic_solver_matches_the_image_reference_after_one_transit():
    """covers the wrapped regime the single-image form cannot grade"""
    m = bench.measure("periodic", ncells=80, final_time=1.0, degree=2)
    assert m["transits"] > 0.9
    assert m["stress_is_degenerate"] is True      # exact sigma == 0 exactly
    assert m["energy_norm_error"] < 0.02


def test_clamped_free_solver_matches_the_image_reference_after_reflections():
    """mixed reflection signs at the two ends, 1.5 transits"""
    m = bench.measure("clamped-free", ncells=100, final_time=1.5, degree=2)
    assert m["stress_is_degenerate"] is False
    assert m["energy_norm_error"] < 0.02
    assert m["rel_l2_velocity"] < 0.02
    assert m["rel_l2_stress"] < 0.02


def test_energy_norm_metric_is_finite_where_relative_l2_error_is_undefined():
    """At integer transits sigma_exact == 0: plain relative L2 must fail, the
    energy norm must not."""
    m = bench.measure("periodic", ncells=80, final_time=1.0, degree=2)
    x, v, s, _v_ex, s_ex = m["_profile"]
    assert float(s_ex.abs().max()) < 1e-12
    # The plain relative L2 error of sigma is either undefined (exactly zero
    # denominator -> ValueError) or numerically meaningless; either way it may
    # not be used to grade this time.
    try:
        bad = relative_error(x, s, s_ex)
    except ValueError:
        bad = float("inf")
    assert bad == float("inf") or bad > 10.0, bad
    assert math.isfinite(m["energy_norm_error"])
    assert m["energy_norm_error"] < 0.02


def test_gate_rejects_a_wrong_sign_reflection_series():
    """Non-vacuity: the same numerical solution graded against a *wrong* image
    series (free instead of clamped reflection at x=0) must be rejected."""
    m = bench.measure("clamped-free", ncells=100, final_time=1.5, degree=2)
    x, v, s, _v_ex, _s_ex = m["_profile"]
    v_wrong, s_wrong = bench.wrong_symmetry_reference(x, m["final_time"])
    err = bench.energy_norm_error(x, v, s, v_wrong, s_wrong)
    assert err > 0.2, f"wrong reflection sign detected only at {err:.3f}"
    assert err > 50 * m["energy_norm_error"]


def test_error_grows_with_the_number_of_transits_but_stays_below_tolerance():
    """Cumulative dispersion: error must be monotone-ish in transits, tiny."""
    errs = [bench.measure("periodic", ncells=80, final_time=t, degree=2)
            ["energy_norm_error"] for t in (0.2, 1.0)]
    assert errs[1] >= errs[0] * 0.5
    assert max(errs) < 0.02


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def test_benchmark_driver_writes_artifacts_and_all_rows_pass(tmp_path):
    from benchmarks.elastic_1d_reflections import main

    rc = main([
        "--cells", "80", "--degree", "2",
        "--tracked-transits", "0.2", "1.0",
        "--clamped-transits", "0.5", "1.5",
        "--conv-cells", "40", "80",
        "--output-dir", str(tmp_path),
    ])
    assert rc == 0
    for name in ("elastic_1d_reflections.json", "benchmark_ref_vs_got.json",
                 "reflection_profiles.csv", "run_manifest.json",
                 "elastic_1d_reflections.png"):
        assert (tmp_path / name).is_file(), name

    bench_json = json.loads((tmp_path / "benchmark_ref_vs_got.json").read_text())
    assert bench_json["tolerance_pct"] == 2.0
    assert bench_json["all_rows_pass"], [
        r for r in bench_json["rows"] if r["verdict"] != "PASS"]
    assert bench_json["worst_rel_error_pct"] < 2.0
    assert "benchmarks.elastic_1d_reflections" in bench_json["repro_cmd"]

    payload = json.loads((tmp_path / "elastic_1d_reflections.json").read_text())
    assert payload["reference"]["clamped-free"].startswith("4L-periodic")
    assert payload["fault_injection"]["detected"] is True
    assert payload["convergence"]["eoc"] is not None
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["artifacts"]["elastic_1d_reflections.json"]["sha256"]
    assert manifest["elapsed_seconds"] >= 0.0