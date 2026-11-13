"""Independent-reference cross-check and arrival-time estimator tests.

Case pool anchor: CASE-DG-DALEMBERT-ELASTIC-1D (TensorDG).

The committed case verifies the DG solver against an analytic solution that
lives in the same module as the solver.  These tests pin down the layer that
makes that trustworthy:

* the three independent references (spectral exact propagator, method of
  characteristics, Richardson-extrapolated staggered leapfrog) agree with each
  other and with the shipped ``dalembert_pulse`` to well below the 2 % gate, so
  a transcription error in any single closed form cannot go unnoticed;
* the DG solution agrees with every reference, i.e. the anchor's verdict does
  not depend on which reference is chosen;
* the touchy part of R3 (the half-cell offset of the staggered pair) is
  asserted through its convergence *order* and the absence of a residual
  offset, not through a stored number;
* the arrival-time estimator upgrade is asserted against the exact
  ``t* = |x_p - L/2| / c`` and the defect of the committed estimator is
  reproduced, so the report's diagnosis is executable and not prose.
"""
import numpy as np
import pytest
import torch

from benchmarks.elastic_1d_crosscheck import (
    PROBE,
    T_CHECK,
    _yee_raw,
    arrival_time_committed,
    arrival_time_new,
    crosscheck_references,
    probe_series,
    ref_characteristics,
    ref_spectral,
    ref_yee,
    yee_convergence,
)
from tensordg.cases import elastic_rod_1d as case
from tensordg.elastic_1d import (
    dalembert_pulse,
    elastic_wave_speed,
    solve_elastic_1d,
    stress_profile,
)

RHO, E = case.RHO, case.YOUNG
C = elastic_wave_speed(RHO, E)


@pytest.fixture(scope="module")
def grid():
    x = np.linspace(0.02, 0.98, 1201)
    _v, sigma = dalembert_pulse(torch.as_tensor(x), T_CHECK, RHO, E)
    return x, sigma.numpy()


def _rel_pct(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)) * 100.0)


def test_spectral_reference_matches_the_shipped_analytic(grid):
    """R1 (Fourier exact propagator) reproduces the analytic solution."""
    x, sigma = grid
    rel = _rel_pct(ref_spectral(x, T_CHECK), sigma)
    assert rel < 1e-8, f"spectral reference deviates by {rel:.3e} %"


def test_characteristics_reference_matches_the_shipped_analytic(grid):
    """R2 (method of characteristics) reproduces the analytic solution."""
    x, sigma = grid
    rel = _rel_pct(ref_characteristics(x, T_CHECK), sigma)
    assert rel < 1e-10, f"characteristics reference deviates by {rel:.3e} %"


def test_yee_reference_is_second_order_and_therefore_usable(grid):
    """R3 must converge at order ~2 and land far below the 2 % gate.

    The order test is what catches the half-cell offset trap of the staggered
    pair: with the initial velocity sampled on the stress nodes the same code
    converges at order 1 and the reference silently loses three digits.
    """
    x, _sigma = grid
    conv = yee_convergence(x, T_CHECK)
    orders = [c["observed_order"] for c in conv[1:]]
    assert all(o is not None for o in orders)
    for o in orders:
        assert 1.85 < o < 2.2, f"R3 not second order: observed {o:.3f}"
    rel = _rel_pct(ref_yee(x, T_CHECK), ref_spectral(x, T_CHECK))
    assert rel < 5e-3, f"Richardson-extrapolated R3 deviates by {rel:.3e} %"


def test_staggered_reference_has_no_residual_half_cell_offset(grid):
    """The h/2 offset trap is closed: shifting by +-h/2 must make it worse."""
    x, _sigma = grid
    h = 1.0 / 1024
    ref = ref_characteristics(x, T_CHECK)
    e0 = _rel_pct(_yee_raw(x, T_CHECK, 1024), ref)
    ep = _rel_pct(_yee_raw(x + 0.5 * h, T_CHECK, 1024), ref)
    em = _rel_pct(_yee_raw(x - 0.5 * h, T_CHECK, 1024), ref)
    assert e0 < 0.2 * min(ep, em), (
        f"reference still carries a sub-cell offset: err(0)={e0:.3e} %, "
        f"err(+h/2)={ep:.3e} %, err(-h/2)={em:.3e} %")


def test_dg_solution_agrees_with_every_independent_reference(grid):
    """The anchor's verdict does not depend on the choice of reference."""
    x, sigma = grid
    centers, coeff, _meta = case.solve_case("pulse", 400, 2, T_CHECK, 0.3)
    x_num, _v, s_num = stress_profile(centers, coeff, nsample=6)
    o = np.argsort(x_num.numpy())
    x_np, s_np = x_num.numpy()[o], s_num.numpy()[o]
    keep = (x_np > 0.02) & (x_np < 0.98)
    for name, ref in (("analytic", sigma),
                      ("spectral", ref_spectral(x, T_CHECK)),
                      ("characteristics", ref_characteristics(x, T_CHECK)),
                      ("yee", ref_yee(x, T_CHECK))):
        rel = _rel_pct(s_np[keep], np.interp(x_np[keep], x, ref))
        assert rel < 2.0, f"DG vs {name}: {rel:.4e} % exceeds the 2 % gate"
        assert rel < 0.01, f"DG vs {name}: {rel:.4e} % is unexpectedly large"


def test_crosscheck_payload_is_self_consistent():
    """Machine-checked report table: every pair passes, R3 is the loosest."""
    cc = crosscheck_references(npts=601)
    for row in cc["reference_agreement"] + cc["dg_vs_references"]:
        assert row["verdict"] == "PASS", row
    pairs = {r["pair"]: r["rel_linf_pct"] for r in cc["reference_agreement"]}
    assert pairs["R0_tensordg_analytic vs R2_characteristics"] < 1e-10
    assert pairs["R0_tensordg_analytic vs R1_spectral_exact"] < 1e-8


def test_probe_series_reproduces_the_shipped_solver():
    """The dense probe stepper must be the shipped solver, not a clone."""
    t_end, ncells, degree = 0.1, 50, 2
    times, values, meta, (centers, coeff) = probe_series(
        ncells, degree, 0.3, PROBE, t_end, return_state=True)
    _c, coeff_lib, meta_lib = solve_elastic_1d(ncells=ncells, degree=degree,
                                               final_time=t_end, cfl=0.3)
    assert meta["steps"] == meta_lib["steps"]
    assert meta["dt"] == pytest.approx(meta_lib["dt"], rel=0.0, abs=0.0)
    assert torch.allclose(coeff, coeff_lib, rtol=0.0, atol=1e-14)
    assert times.shape == values.shape == (meta["steps"] + 1,)


def test_dense_probe_trace_peaks_at_the_exact_arrival_time():
    """Sanity of the probe evaluation itself, before any peak refinement."""
    t_star = case.probe_arrival_time(PROBE)
    times, values, _m = probe_series(200, 2, 0.3, PROBE, t_star + 0.05)
    j = int(np.argmax(np.abs(values)))
    assert abs(times[j] - t_star) < 5e-3
    assert abs(abs(values[j]) - 0.5) < 5e-3       # half-amplitude packet


def test_arrival_time_estimator_meets_the_anchor_accuracy():
    """Upgraded estimator: sub-0.01 % agreement with t* = |x_p - L/2| / c."""
    r = arrival_time_new(200, 2)
    assert abs(r["rel_error_pct"]) < 0.01, r
    r3 = arrival_time_new(100, 3)
    assert abs(r3["rel_error_pct"]) < 0.01, r3


def test_committed_estimator_defect_is_reproduced_not_masked():
    """The 0.602 % row is real and mesh-limited; the fix must be visible.

    Reproducing the committed number keeps the report honest (no silent
    baseline change); asserting that the upgraded estimator is at least 50x
    better proves the defect was actually removed.
    """
    old = arrival_time_committed(nt=11)
    assert old["rel_error_pct"] == pytest.approx(-0.6022, abs=5e-3)
    assert abs(old["probe_snap_x"] - PROBE) > 1e-3      # O(h) probe snapping
    new = arrival_time_new(200, 2)
    assert abs(new["rel_error_pct"]) * 50.0 < abs(old["rel_error_pct"])


@pytest.mark.parametrize("young_scale", [1.02 ** 2])
def test_arrival_time_measurement_detects_a_wrong_wave_speed(young_scale):
    """Fault injection: a 2 % fast rod must show up as a ~2 % early arrival."""
    r = arrival_time_new(100, 2, young=E * young_scale)
    assert r["rel_error_pct"] == pytest.approx(-2.0, abs=0.2)