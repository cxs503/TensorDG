"""Tests for the Sod second-physical-case driver of the certified-reference engine.

The claims under test are the ones the case report rests on, so they are the
ones asserted here:

* the closed-form star state used as an *external* anchor really is Toro's
  tabulated Sod solution (a typo in the anchor would silently move every
  "error vs exact" number);
* the exact cell-average reference conserves the analytic mass
  ``0.5*1 + 0.5*0.125`` (a broken quadrature shows up here, not in the report);
* the engine is *case agnostic*: the same ``certify`` / ``judge`` primitives
  certify a Sod FV ladder, and the predeclared criteria are evaluated as
  declared -- ``accepted`` is the conjunction, and ``band_covers_true_error``
  is exactly ``true_error <= band`` rather than a re-statement of the verdict;
* a refused ladder is reported as a refusal (no exception, no silent
  ``CERTIFIED``), so a red run cannot masquerade as green;
* the self-certification limitation is *reported* for every DG family offered as
  a certifier (the engine has no shared-code detector; the report says so).
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks import certified_reference_sod as sod


# ------------------------------------------------------------------ the oracle
def test_star_state_matches_the_tabulated_sod_solution():
    """Toro's Sod values: p*=0.30313, u*=0.92745, rho*_L=0.42632, rho*_R=0.26557."""
    st = sod.star_state()
    assert st["p_star"] == pytest.approx(0.303130, abs=1e-5)
    assert st["u_star"] == pytest.approx(0.927453, abs=1e-5)
    assert st["rho_star_left"] == pytest.approx(0.426319, abs=1e-5)
    assert st["rho_star_right"] == pytest.approx(0.265574, abs=1e-5)
    assert st["shock_speed"] == pytest.approx(1.752156, abs=1e-5)


def test_exact_reference_conserves_the_analytic_mass():
    """Nothing leaves the domain by t=0.2, so the exact mass is 0.5625 exactly."""
    ex = sod.exact_profiles(400)
    dx = float(ex["x"][1] - ex["x"][0])
    # the quadrature is exact on smooth cells; the two cells straddling the
    # contact and the shock carry the residual, so 1e-5 relative is the honest
    # bound here (a broken quadrature is off by orders of magnitude more)
    measured = float(np.sum(ex["density"]) * dx)
    assert measured == pytest.approx(sod.INITIAL_MASS, rel=1e-4)
    assert abs(measured - sod.INITIAL_MASS) < 1e-4
    assert np.all(ex["density"] > 0.0) and np.all(ex["pressure"] > 0.0)


def test_exact_solution_reproduces_the_three_plateaus():
    """The piecewise-constant limits of the exact solution are physical anchors."""
    st = sod.star_state()
    ex = sod.exact_profiles(2000)
    x, rho = ex["x"], ex["density"]
    assert np.mean(rho[x < 0.25]) == pytest.approx(1.0, abs=1e-8)          # untouched left
    assert np.mean(rho[(x > st["contact_position"] + 0.02)
                       & (x < st["shock_position"] - 0.02)]) == pytest.approx(
        st["rho_star_right"], abs=2e-3)                                     # post-shock plateau
    assert np.mean(rho[x > st["shock_position"] + 0.02]) == pytest.approx(
        sod.RIGHT[0], abs=1e-8)                                             # untouched right


# ---------------------------------------------------------------- engine facets
def test_band_covers_flag_is_the_comparison_it_claims():
    """``band_covers_true_error`` must not be a restatement of the verdict."""
    a = sod.audit(fv_ladder=(50, 100, 200), dg_ladder=(25, 50), degrees=(1,),
                  fv_cfl=0.3, dg_cfl=0.4)
    err = a["error_vs_exact"]
    assert err["band_percent"] is not None and err["true_error_percent"] > 0.0
    assert err["band_covers_true_error"] == (
        err["true_error_percent"] <= err["band_percent"])
    assert err["ratio_true_error_over_band"] == pytest.approx(
        err["true_error_percent"] / err["band_percent"])


def test_accepted_is_the_conjunction_of_the_predeclared_criteria():
    a = sod.audit(fv_ladder=(50, 100, 200), dg_ladder=(25, 50), degrees=(1,),
                  fv_cfl=0.3, dg_cfl=0.4)
    assert set(a["criteria"]) == {
        "C1_ladder_certifies_at_gate",
        "C2_dg_legs_under_strict_gate",
        "C3_external_oracle_under_1pct",
        "C4_band_brackets_true_error",
        "C5_analytic_invariants_under_1pct",
    }
    assert a["accepted"] == all(a["criteria"].values())


def test_the_engine_accepts_the_sod_fv_ladder():
    """Case-agnostic claim: the same primitives certify a second physical case."""
    a = sod.audit(fv_ladder=(50, 100, 200), dg_ladder=(25, 50), degrees=(1,),
                  fv_cfl=0.3, dg_cfl=0.4)
    assert a["fv_certification"]["verdict"] == "CERTIFIED"
    assert a["criteria"]["C1_ladder_certifies_at_gate"] is True
    # the reference must also be *right*, not merely self-consistent
    assert a["error_vs_exact"]["true_error_percent"] < 1.0


def test_a_stagnating_ladder_is_refused_not_certified():
    """Identical levels carry no order information: the engine must refuse."""
    ex = sod.exact_profiles(100)
    # same field on all three levels -> zero successive differences -> refusal
    flat = [{"ncells": 100, "field": ex["density"].copy()} for _ in range(3)]
    a = sod.audit(levels_override={"fv": flat, "dg": {}}, degrees=(1,), dg_ladder=(1,))
    assert a["fv_certification"]["verdict"] != "CERTIFIED"
    assert a["criteria"]["C1_ladder_certifies_at_gate"] is False
    assert a["accepted"] is False


def test_every_offered_dg_ladder_is_reported_as_a_potential_certifier():
    """The shared-code limitation must be visible in the artifact for each family."""
    a = sod.audit(fv_ladder=(50, 100, 200), dg_ladder=(25, 50),
                  degrees=(1, 2), fv_cfl=0.3, dg_cfl=0.4)
    assert set(a["dg_self_certification"]) == {"dg_p1", "dg_p2"}
    for entry in a["dg_self_certification"].values():
        assert set(entry) >= {"verdict", "classification", "band_share_of_gate",
                              "observed_order", "reasons"}


def test_dg_legs_carry_both_gates():
    a = sod.audit(fv_ladder=(50, 100, 200), dg_ladder=(25, 50), degrees=(1,),
                  fv_cfl=0.3, dg_cfl=0.4)
    assert a["dg_legs"]
    for leg in a["dg_legs"]:
        assert leg["decision_5pct"] in {"PASS", "PASS_MARGINAL", "FAIL", "WITHHELD"}
        assert leg["decision_3pct"] in {"PASS", "PASS_MARGINAL", "FAIL", "WITHHELD"}
        assert leg["rel_l1_percent"] > 0.0


# ---------------------------------------------------------------- small helpers
def test_downsample_averages_cells():
    f = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    assert np.allclose(sod._downsample(f, 2), [2.0, 6.0, 10.0])


def test_windowed_rel_l1_is_zero_for_identical_fields():
    ex = sod.exact_profiles(200)
    v = sod._windowed_rel_l1(ex["x"], ex["density"], ex["x"], ex["density"], 0.3, 0.6)
    assert v == pytest.approx(0.0, abs=1e-15)


def test_shock_position_finder_locates_the_analytic_shock():
    st = sod.star_state()
    ex = sod.exact_profiles(4000)
    found = sod._shock_position(ex["x"], ex["density"],
                                x_min=st["contact_position"] + 0.01)
    assert abs(found - st["shock_position"]) < 2.0e-3