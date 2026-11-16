"""Pure-function regression gates for the power-calibration closure (CKS1998 §5.3).

Every gate here is synthetic: the numbers come from the analytic error model
``e(CFL) = S + tau * CFL**q`` (``S`` the CFL-independent spatial error), never
from the solver, so the gates are fast, deterministic and independent of the
cluster payload.  What they assert is *behaviour*, not code execution: a
three-level probe must recover the order that was put in, a jackknife must not
certify a claim its leave-one-out fits disagree on, and no refused/diverged
configuration may ever reach an arithmetic step.
"""
from __future__ import annotations

import math

import pytest

from benchmarks.cks1998_vortex_time_power_closure import (
    PROPOSED_ETA_BOUND,
    window_sup,
    ADMISSIBLE_CFL_MAX,
    BETA,
    CLOSURE_OUT,
    ETA_MAX,
    EXTENDED_CFLS,
    EXTENDED_FIT_CFLS,
    MIN_SAMPLES,
    ORDER_TOL,
    assemble,
    closure_verdict,
    fit_law,
    level_triples,
    measurable_by_cfl,
    order_agreement,
    power_jackknife,
)

S_DEFAULT = 4.0e-3
TAU_DEFAULT = 1.0e-6
Q_TRUE = 3.0


def err(cfl: float, spatial: float = S_DEFAULT, tau: float = TAU_DEFAULT,
        q: float = Q_TRUE) -> float:
    """Analytic error model: spatial part + temporally dominated part."""
    return spatial + tau * float(cfl) ** q


def ladder(mesh: int = 32, cfls=EXTENDED_CFLS, spatial: float = S_DEFAULT,
           tau: float = TAU_DEFAULT, q: float = Q_TRUE, degree: int = 1):
    return [{"degree": degree, "mesh": mesh, "kind": "ladder", "cfl": c,
             "error": err(c, spatial, tau, q), "steps": 1,
             "h": 10.0 / mesh, "dt": 1.0, "dofs": 1, "wall_seconds": 0.0}
            for c in cfls]


def samples_from(records):
    from benchmarks.cks1998_vortex_time_power import etas_from_records
    return etas_from_records(records)


# ---------------------------------------------------------------------------
# Three-level probe
# ---------------------------------------------------------------------------
def test_three_level_recovers_temporal_order():
    triples = level_triples(ladder())
    prod = [t for t in triples if t["production_cfl"]]
    assert len(prod) == 1, "the production CFL must yield exactly one triple"
    q_obs = prod[0]["q_obs"]
    assert q_obs is not None
    # bias is O(eta) ~ 1e-6: the estimator must reproduce the injected order
    assert abs(q_obs - Q_TRUE) < 1e-2, f"q_obs={q_obs} vs true {Q_TRUE}"


def test_three_level_eta_and_omega_are_shares():
    prod = [t for t in level_triples(ladder()) if t["production_cfl"]][0]
    e0, e1, e2 = (prod["errors"]["base"], prod["errors"]["refined"],
                  prod["errors"]["refined2"])
    assert prod["eta"] == pytest.approx(abs(e0 - e1) / e0, rel=0, abs=0)
    assert prod["omega"] == pytest.approx(abs(e1 - e2) / e1, rel=0, abs=0)
    assert 0.0 < prod["eta"] < ETA_MAX  # spatially dominated configuration


@pytest.mark.parametrize("flag", ["unstable", "rejected"])
def test_lost_records_never_enter_arithmetic(flag):
    records = ladder()
    for r in records:
        if r["cfl"] == 0.05:
            r[flag] = True
            r["error"] = float("nan")
    m = measurable_by_cfl(records)
    assert 0.05 not in m
    for t in level_triples(records):
        assert all(math.isfinite(v) for v in t["errors"].values())


def test_three_level_needs_all_three_levels():
    # drop the beta**2 partner of the production CFL: no order may be claimed
    cfls = tuple(c for c in EXTENDED_CFLS if c != 0.0125)
    triples = level_triples(ladder(cfls=cfls))
    assert not [t for t in triples if t["production_cfl"]], \
        "no beta**2 level => no production-CFL triple at all (never a partial one)"
    # the *other* triples still resolve -- the missing level is local, not global
    assert any(t["q_obs"] is not None for t in triples)
    # and the eta leg is unaffected: 0.05/0.0125 is gone, 0.1/0.025 still pairs
    assert len(samples_from(ladder(cfls=cfls))) == 3


def test_zero_eta_gives_no_order():
    records = ladder()
    for r in records:                    # CFL-independent error => eta == 0
        r["error"] = S_DEFAULT
    triples = level_triples(records)
    for t in triples:
        assert t["eta"] == 0.0
        assert t["q_obs"] is None and t["reason"] == "ETA_ZERO"


def test_zero_omega_gives_no_order():
    records = ladder()
    for r in records:                    # time error only below the base CFL
        if r["cfl"] <= 0.05:
            r["error"] = S_DEFAULT + TAU_DEFAULT * 0.05 ** Q_TRUE
    triples = [t for t in level_triples(records) if t["production_cfl"]][0]
    assert triples["omega"] == 0.0
    assert triples["q_obs"] is None and triples["reason"] == "OMEGA_ZERO"


def test_roundoff_scale_difference_never_becomes_an_order():
    """A sub-ulp refined level collapses ``omega`` to zero, not to a fake order."""
    records = ladder()
    for r in records:
        if r["cfl"] == 0.0125:
            r["error"] = err(0.05) * (1 - 1e-18)   # below double resolution
    triples = [t for t in level_triples(records) if t["production_cfl"]][0]
    assert triples["omega"] == 0.0
    assert triples["q_obs"] is None and triples["reason"] == "OMEGA_ZERO"


# ---------------------------------------------------------------------------
# Power law + jackknife
# ---------------------------------------------------------------------------
def test_fit_excludes_nonpositive_samples():
    samples = [{"cfl": 0.05, "eta": 0.0}, {"cfl": 0.1, "eta": -1.0},
               {"cfl": 0.2, "eta": 1.0e-6}, {"cfl": 0.4, "eta": 1.0e-5}]
    fit = fit_law(samples)
    assert fit["status"] == "OK" and fit["n_points"] == 2
    assert fit["cfls"] == [0.2, 0.4]


def test_fit_needs_two_points():
    assert fit_law([{"cfl": 0.2, "eta": 1e-6}])["status"] == "INSUFFICIENT_SAMPLES"


def test_jackknife_certifies_vacuity_when_every_leave_one_out_agrees():
    power = power_jackknife(samples_from(ladder()))
    assert power["n_points"] == 4, "the partner-complete ladder must give 4 samples"
    assert power["verdict"] == "GATE_VACUOUS_CERTIFIED"
    assert power["point_verdict"] == "GATE_VACUOUS_IN_ENVELOPE"
    assert power["crossing_range"][0] > ADMISSIBLE_CFL_MAX
    assert power["full"]["r2"] > 0.9


def test_jackknife_certifies_informativeness_for_a_strong_time_error():
    # tau tuned so eta reaches 5 % just inside the admissible window: this is the
    # branch that would *overturn* m22636, so it must be reachable and certified
    power = power_jackknife(samples_from(ladder(tau=2.6e-3)))
    assert power["point_verdict"] == "GATE_INFORMATIVE"
    assert power["verdict"] == "GATE_INFORMATIVE_CERTIFIED"
    assert power["crossing_range"][1] <= ADMISSIBLE_CFL_MAX


def test_jackknife_underpowered_below_min_samples():
    cfls = (0.0125, 0.05, 0.1, 0.2)
    power = power_jackknife(samples_from(ladder(cfls=cfls)))
    assert power["n_points"] == 2 < MIN_SAMPLES
    assert power["verdict"] == "CLOSURE_UNDERPOWERED"
    assert power["reason"] == "INSUFFICIENT_SAMPLES"


def test_jackknife_refuses_to_certify_when_leave_one_out_straddles():
    """Sweep a poisoned 0.4-point until the leave-one-out fits disagree.

    The invariant under test is not "some factor breaks it" but the weaker, exact
    one: a certified verdict is only ever issued when *every* leave-one-out fit
    lands on the same side of the window edge.
    """
    straddled = 0
    for factor in (1.1, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
        records = ladder()
        for r in records:
            if r["cfl"] == 0.4:               # poison the top rung of the ladder
                r["error"] = err(0.4) * 1.0 + S_DEFAULT * (factor - 1.0)
        power = power_jackknife(samples_from(records))
        if power["verdict"] == "CLOSURE_UNDERPOWERED":
            straddled += 1
            crossings = [j["cfl_at_eta_max"] for j in power["jackknife"]]
            assert min(crossings) <= ADMISSIBLE_CFL_MAX < max(crossings)
            assert power["all_leave_one_out_agree"] is False
        else:
            crossings = [j["cfl_at_eta_max"] for j in power["jackknife"]]
            assert all(c > ADMISSIBLE_CFL_MAX for c in crossings) or \
                all(c <= ADMISSIBLE_CFL_MAX for c in crossings)
    assert straddled >= 1, "the straddle branch must be exercised"


# ---------------------------------------------------------------------------
# Order agreement / closure fold
# ---------------------------------------------------------------------------
def test_order_agreement_two_way():
    ok = order_agreement(3.0, 3.0)
    assert ok["status"] == "CONSISTENT" and ok["relative_gap"] == 0.0
    bad = order_agreement(2.0, 3.0)
    assert bad["status"] == "DISAGREE"
    assert bad["relative_gap"] == pytest.approx(1 / 3)
    edge = order_agreement(3.0 * (1 + ORDER_TOL * 0.999), 3.0)
    assert edge["status"] == "CONSISTENT"          # band is inclusive
    over = order_agreement(3.0 * (1 + 2 * ORDER_TOL), 3.0)
    assert over["status"] == "DISAGREE"


@pytest.mark.parametrize("s,q,reason", [(None, 3.0, "MISSING_ESTIMATE"),
                                        (3.0, None, "MISSING_ESTIMATE"),
                                        (float("nan"), 3.0, "NON_FINITE"),
                                        (3.0, 0.0, "NON_FINITE")])
def test_order_agreement_never_invents_a_verdict(s, q, reason):
    out = order_agreement(s, q)
    assert out["status"] == "UNRESOLVED" and out["reason"] == reason


def test_closure_verdict_flags_order_inconsistency():
    power = {"verdict": "GATE_VACUOUS_CERTIFIED"}
    assert closure_verdict(power, [{"status": "CONSISTENT"}]) == \
        "GATE_VACUOUS_CERTIFIED"
    assert closure_verdict(power, [{"status": "DISAGREE"}]) == \
        "GATE_VACUOUS_CERTIFIED_ORDER_INCONSISTENT"
    assert closure_verdict({"verdict": "CLOSURE_UNDERPOWERED"},
                           [{"status": "DISAGREE"}]) == "CLOSURE_UNDERPOWERED"


# ---------------------------------------------------------------------------
# Window supremum: what a vacuous gate can be replaced by
# ---------------------------------------------------------------------------
def test_window_sup_takes_the_max_inside_the_window_only():
    samples = samples_from(ladder())
    out = window_sup(samples)
    assert out["status"] == "OK"
    assert out["at_cfl"] == max(s["cfl"] for s in samples
                               if s["cfl"] <= ADMISSIBLE_CFL_MAX)
    assert out["eta_sup"] == max(s["eta"] for s in samples
                                 if s["cfl"] <= ADMISSIBLE_CFL_MAX)
    assert out["n_admissible"] == len(samples)


def test_window_sup_ignores_out_of_window_samples():
    samples = [{"cfl": 0.2, "eta": 1e-6}, {"cfl": 0.4, "eta": 2e-5},
               {"cfl": 7.4, "eta": 0.9}]          # extrapolated crossing, not a run
    out = window_sup(samples)
    assert out["eta_sup"] == 2e-5 and out["at_cfl"] == 0.4
    assert out["below_proposed_1e4"] is True
    assert out["sup_over_eta_max"] == pytest.approx(2e-5 / ETA_MAX)


def test_window_sup_declares_absence_of_admissible_samples():
    out = window_sup([{"cfl": 7.4, "eta": 0.9}])
    assert out["status"] == "NO_ADMISSIBLE_SAMPLES" and "eta_sup" not in out
    assert PROPOSED_ETA_BOUND < ETA_MAX


# ---------------------------------------------------------------------------
# Assembly: mesh grouping, mesh independence, honest degradation
# ---------------------------------------------------------------------------
def test_assemble_measures_mesh_independence():
    records = ladder(mesh=10) + ladder(mesh=32)
    payload = assemble(records)
    bundle = payload["degrees"]["p1"]
    assert bundle["meshes"] == [10, 32]
    assert bundle["primary_mesh"] == 32, "the certified mesh is the primary one"
    assert bundle["mesh_independent"] is True
    assert payload["verdicts"]["p1"] == "GATE_VACUOUS_CERTIFIED"
    for mesh in ("10", "32"):
        m = bundle["per_mesh"][mesh]
        assert m["power"]["n_points"] == 4
        assert m["order_agreement"]["status"] == "CONSISTENT"


def test_assemble_degrades_honestly_without_the_second_refinement():
    cfls = tuple(c for c in EXTENDED_CFLS if c != 0.0125)
    payload = assemble(ladder(mesh=32, cfls=cfls))
    m = payload["degrees"]["p1"]["per_mesh"]["32"]
    assert m["second_refinement"]["status"] == "INSUFFICIENT_LEVELS"
    assert (m["production_triple"] or {}).get("q_obs") is None
    assert m["order_agreement"]["status"] == "UNRESOLVED"
    # the power leg still stands on its own samples -- no false certification
    assert m["power"]["n_points"] == 3
    assert payload["verdicts"]["p1"] in {"GATE_VACUOUS_CERTIFIED",
                                         "CLOSURE_UNDERPOWERED"}


def test_assemble_keeps_refused_probes_out_of_every_fit():
    records = ladder(mesh=32)
    records.append({"degree": 1, "mesh": 32, "kind": "refused_probe",
                    "cfl": 0.6, "rejected": True,
                    "reason": "invalid cells, final_time or CFL"})
    payload = assemble(records)
    m = payload["degrees"]["p1"]["per_mesh"]["32"]
    assert m["refused_probes"] == [{"cfl": 0.6,
                                   "reason": "invalid cells, final_time or CFL"}]
    assert all(s["cfl"] <= ADMISSIBLE_CFL_MAX for s in m["eta_samples"])
    assert max(s["cfl"] for s in m["eta_samples"]) in EXTENDED_FIT_CFLS


def test_assemble_records_the_contract_it_checks_against():
    payload = assemble(ladder(mesh=32))
    assert payload["anchor"] == "REPRO-DG-CKS1998"
    assert payload["beta"] ** 2 == payload["beta2"] == BETA ** 2
    assert payload["eta_max"] == ETA_MAX
    assert payload["order_tol"] == ORDER_TOL
    assert CLOSURE_OUT.name == "time_power_closure.json"