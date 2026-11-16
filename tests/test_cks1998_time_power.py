"""Regression gates for the CKS1998 time-attribution power calibration.

All tests are synthetic: no solver run, no cluster job.  They pin the
*instrument* (estimators + verdict logic), because the numeric payload is
produced by ``benchmarks/cks1998_vortex_time_power.py`` on the cluster and is
published with its raw records.
"""
from __future__ import annotations

import math

import pytest

from benchmarks.cks1998_vortex_time_power import (
    ADMISSIBLE_CFL_MAX,
    BETA,
    CFL_BASE,
    ETA_MAX,
    assemble,
    cfl_at_eta,
    classify_leg,
    eta_share,
    etas_from_records,
    loglog_fit,
    order_drift_leg,
    power_verdict,
    second_refinement,
    stability_envelope,
)


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def test_eta_share_is_the_certified_leg():
    assert eta_share(1.0, 0.9) == pytest.approx(0.1)
    # direction does not matter: only magnitude of the movement
    assert eta_share(1.0, 1.1) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        eta_share(0.0, 1.0)


def test_loglog_fit_recovers_a_known_power_law():
    amplitude, exponent = 3.0e-6, 2.5
    cfls = [0.05, 0.1, 0.2, 0.4]
    etas = [amplitude * c ** exponent for c in cfls]
    a, s, r2 = loglog_fit(cfls, etas)
    assert a == pytest.approx(amplitude, rel=1e-10)
    assert s == pytest.approx(exponent, rel=1e-10)
    assert r2 == pytest.approx(1.0)


def test_loglog_fit_ignores_non_positive_and_needs_two_points():
    assert loglog_fit([0.1, 0.0, 0.2], [1e-4, 1e-9, 4e-4])[1] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        loglog_fit([0.1], [1e-4])


def test_cfl_at_eta_inverts_the_fit():
    a, s, target = 2.0e-6, 2.0, ETA_MAX
    crossing = cfl_at_eta(a, s, target)
    assert a * crossing ** s == pytest.approx(target, rel=1e-12)
    with pytest.raises(ValueError):
        cfl_at_eta(1.0, 0.0, 0.05)          # a flat law never reaches the target


def test_classify_leg_is_strict_and_flags_nan():
    assert classify_leg(ETA_MAX) == "SILENT"          # equality is not a fire
    assert classify_leg(ETA_MAX * 1.0001) == "FIRE"
    assert classify_leg(float("nan")) == "NOT_MEASURABLE"


# ---------------------------------------------------------------------------
# verdict logic (the point of the whole module)
# ---------------------------------------------------------------------------
def _scaling(exponent, crossing, r2=0.99):
    return {"exponent": exponent, "cfl_at_eta_max": crossing, "r2": r2}


def test_vacuous_when_the_crossing_is_outside_the_admissible_window():
    assert power_verdict(_scaling(2.0, 34.0), ADMISSIBLE_CFL_MAX) == \
        "GATE_VACUOUS_IN_ENVELOPE"


def test_informative_when_the_instrument_fires_inside_the_window():
    assert power_verdict(_scaling(2.0, 0.3), ADMISSIBLE_CFL_MAX) == \
        "GATE_INFORMATIVE"


@pytest.mark.parametrize("scaling", [
    _scaling(-1.0, 1.0),            # decreasing law: never fires with rising CFL
    _scaling(2.0, 0.5, r2=0.3),     # fit not credible
    {"exponent": None, "cfl_at_eta_max": None, "r2": None},
])
def test_no_verdict_without_a_credible_power_law(scaling):
    assert power_verdict(scaling, ADMISSIBLE_CFL_MAX) == "SCALING_UNRESOLVED"


def test_no_verdict_without_a_window():
    assert power_verdict(_scaling(2.0, 0.3), None) == "SCALING_UNRESOLVED"


# ---------------------------------------------------------------------------
# envelope / sample construction
# ---------------------------------------------------------------------------
def test_stability_envelope_splits_blowup_and_non_finite():
    records = [{"cfl": c, "error": e} for c, e in
               [(0.05, 1.0e-3), (0.2, 1.0e-3), (0.4, 1.1e-3),
                (1.6, 5.0e3), (3.2, float("nan"))]]
    env = stability_envelope(records, reference_cfl=0.2, blowup=1.0e3)
    assert env["status"] == "OK"
    assert env["stable_cfl_max"] == pytest.approx(0.4)
    assert [u["cfl"] for u in env["lost"]] == [1.6, 3.2]


def test_stability_envelope_without_reference_refuses():
    env = stability_envelope([{"cfl": 0.1, "error": 1.0}], reference_cfl=0.2)
    assert env["status"] == "NO_REFERENCE" and env["stable_cfl_max"] is None


def test_etas_are_paired_with_their_beta_partner_only():
    records = [{"cfl": c, "error": e} for c, e in
               [(CFL_BASE, 1.0e-3), (CFL_BASE / BETA, 1.0e-3 * (1 - 1e-3)),
                (0.4, 2.0e-3)]]           # 0.4 has no 0.1 partner -> no sample
    samples = etas_from_records(records)
    assert [s["cfl"] for s in samples] == [CFL_BASE]
    assert samples[0]["production_cfl"] is True
    assert samples[0]["eta"] == pytest.approx(abs(1.0e-3 - 1.0e-3 * (1 - 1e-3))
                                              / 1.0e-3, rel=1e-12)


def test_unstable_runs_never_enter_the_eta_arithmetic():
    records = [{"cfl": CFL_BASE, "error": 1e-3, "unstable": False},
               {"cfl": CFL_BASE / BETA, "error": float("nan"), "unstable": True}]
    assert etas_from_records(records) == []


# ---------------------------------------------------------------------------
# the certified study's declared gap: second-level refinement
# ---------------------------------------------------------------------------
def test_second_refinement_reports_converged_and_unconverged_cases():
    converged = [{"cfl": c, "error": e} for c, e in [
        (CFL_BASE, 1.0e-3),
        (CFL_BASE / BETA, 1.0e-3 * (1 - 4.0e-6)),
        (CFL_BASE / BETA ** 2, 1.0e-3 * (1 - 4.001e-6)),
    ]]
    out = second_refinement(converged)
    assert out["status"] == "OK"
    assert out["temporal_asymptote_reached"] is True
    assert out["omega_second_level"] < out["eta_one_level"]

    unconverged = [{"cfl": c, "error": e} for c, e in [
        (CFL_BASE, 1.0e-3), (CFL_BASE / BETA, 9.0e-4),
        (CFL_BASE / BETA ** 2, 7.0e-4),
    ]]
    out = second_refinement(unconverged)
    assert out["temporal_asymptote_reached"] is False


def test_second_refinement_refuses_incomplete_levels():
    assert second_refinement([{"cfl": CFL_BASE, "error": 1.0}])["status"] == \
        "INSUFFICIENT_LEVELS"


# ---------------------------------------------------------------------------
# drift leg
# ---------------------------------------------------------------------------
def _mesh_records(order_base, order_refined, degree=1):
    """Two meshes at the base CFL and its refined partner, with prescribed orders."""
    h_c, h_f = 0.1, 0.05
    ratio = h_c / h_f
    out = []
    for cfl, order in ((CFL_BASE, order_base), (CFL_BASE / BETA, order_refined)):
        e_f = 1.0e-4
        e_c = e_f * ratio ** order
        out.append({"cfl": cfl, "mesh": 10, "h": h_c, "error": e_c, "kind": "ladder"})
        out.append({"cfl": cfl, "mesh": 20, "h": h_f, "error": e_f, "kind": "ladder"})
    return out


def test_drift_leg_fires_only_when_the_order_moves():
    quiet = order_drift_leg(_mesh_records(2.0, 2.005), degree=1)
    assert quiet["status"] == "OK" and quiet["leg"] == "SILENT"
    moving = order_drift_leg(_mesh_records(2.0, 2.5), degree=1)
    assert moving["leg"] == "FIRE" and moving["drift"] > moving["drift_max"]


# ---------------------------------------------------------------------------
# end to end on synthetic records
# ---------------------------------------------------------------------------
def _synthetic_records(exponent, amplitude, stable_max=0.4, degree=1, mesh=32):
    """errors following e(CFL) = e0 * (1 + A*CFL**s) -- an exact power law in eta."""
    e0 = 2.0e-3
    cfll = sorted({0.05, 0.1, 0.2, 0.4, 0.5, CFL_BASE / BETA ** 2})
    recs = []
    for cfl in cfll:
        e = e0 * (1.0 + amplitude * cfl ** exponent)
        if cfl > stable_max:
            e = e0 * 1.0e6                      # lost stability
        recs.append({"degree": degree, "mesh": mesh, "h": 10.0 / mesh,
                     "cfl": cfl, "error": e, "steps": 1, "dt": cfl,
                     "kind": "ladder"})
    # coarse mesh of the production pair for the drift leg (same CFLs only)
    for cfl in (CFL_BASE, CFL_BASE / BETA):
        recs.append({"degree": degree, "mesh": mesh // 2, "h": 20.0 / mesh,
                     "cfl": cfl, "error": 2.0e-3 * 2.0 ** 2, "steps": 1,
                     "dt": cfl, "kind": "ladder"})
    return recs


def test_assemble_flags_a_vacuous_gate_on_a_slow_power_law():
    payload = assemble(_synthetic_records(exponent=2.0, amplitude=4.0e-4))
    bundle = payload["degrees"]["p1"]
    assert bundle["scaling"]["exponent"] == pytest.approx(2.0, rel=1e-3)
    assert bundle["verdict"] == "GATE_VACUOUS_IN_ENVELOPE"
    assert bundle["envelope"]["stable_cfl_max"] == pytest.approx(0.4)


def test_assemble_flags_an_informative_gate_on_a_steep_law():
    # eta crosses 5 % around CFL ~ 0.4 -> inside the admissible window
    payload = assemble(_synthetic_records(exponent=4.0, amplitude=2.0))
    assert payload["degrees"]["p1"]["verdict"] == "GATE_INFORMATIVE"


def test_assemble_never_silently_reports_a_verdict_without_runs():
    payload = assemble([])
    assert payload["degrees"] == {} and payload["verdicts"] == {}


def test_assemble_keeps_raw_runs_for_the_report():
    payload = assemble(_synthetic_records(exponent=2.0, amplitude=4.0e-4))
    bundle = payload["degrees"]["p1"]
    runs = bundle["runs"]
    assert len(runs) >= 7
    assert all("error" in r and "cfl" in r for r in runs)
    # the lost run is preserved (not filtered) in the raw records, so the
    # envelope decision stays auditable, while the eta samples exclude it
    assert bundle["envelope"]["lost"]
    assert all(s["cfl"] not in {x["cfl"] for x in bundle["envelope"]["lost"]}
               for s in bundle["eta_samples"])