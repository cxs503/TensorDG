"""Gates on the stability polynomials of the explicit RK tableaus.

Every assertion here is an *independent* anchor on
:mod:`tensordg.rk_stability`, not a restatement of its implementation:

``A1 two forms``    the coefficient form ``R(z) = sum_k (b^T A^{k-1} 1) z^k`` and
                    the solve form ``1 + z b^T (I - z A)^{-1} 1`` must agree to
                    ``1e-12`` for every registered tableau.
``A2 order``        ``|R - T_p| <= 1e-8 * z^(p+1)`` at ``z = 1e-3``, with ``T_p``
                    the Taylor series of ``exp``: the order in the tableau
                    metadata is the order ``R`` actually has (this is the
                    Butcher-barrier statement in polynomial form).
``A3 axis anchors`` textbook stability intervals -- RK4 real ``2.7852935634``,
                    RK4 imaginary ``2 sqrt 2``, Kutta-3 imaginary ``sqrt 3``,
                    Kutta-3 real ``2.5127453267``, Heun/Euler real ``2``.
``A4 cross-impl``   the Butcher form of the solver's TVD-RK3 must reproduce the
                    real-axis interval that
                    ``benchmarks/cfl_stability_limit.py`` measured by replaying
                    the *hand-coded* stage recurrence
                    (``validation/cfl_stability/results.json``); two independent
                    transcriptions of the scheme, one number.
``A5 solver path``  :func:`tensordg.runge_kutta.rk_step` driven by the registered
                    ``ssp_rk3`` tableau must reproduce, to round-off,
                    ``solve_advection(time_integrator="ssp_rk3")`` -- i.e. the
                    registered tableau describes the branch the solver runs.
``A6 certificate``  the ``cfl = 0.2`` / ``0.4`` values of the round-2 accuracy
                    study are inside the certified region of every matched rung
                    with margin ``>= 2``, recomputed here from the assembled
                    operator (not read from the artifact).
``A7 selection``    ``tableau_for_degree`` keeps its round-2 mapping; the new
                    ``ssp_rk3`` registration must not take over the ``p = 2``
                    rung.

Run: ``PYTHONPATH=. pytest -q tests/test_rk_stability.py``
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest
import torch

from tensordg.advection_1d import _operators, _rhs, solve_advection
from tensordg.dispersion_analysis import assemble_operator, symbol_matrix
from tensordg.rk_stability import (amplification_order, cfl_limit,
                                   imaginary_axis_interval, is_stable,
                                   max_stable_dt, real_axis_interval,
                                   stability_function, stability_function_dense)
from tensordg.runge_kutta import (SSP_RK3, TABLEAUS, rk_step,
                                  tableau_for_degree)

STABILITY_ARTIFACT = pathlib.Path("validation/advection_order_stability/metrics.json")
REPLAY_ARTIFACT = pathlib.Path("validation/cfl_stability/results.json")

Z_PROBE = np.array([0.37 + 0.62j, -1.13, 2.02 - 0.41j, 1e-3, -2.7 + 1.4j])
TWO_FORM_TOL = 1e-12
ORDER_Z = 1e-3
ORDER_TOL = 1e-8

#: ``cfl`` of the round-2 accuracy ladder and of its dt-order probe.
CFL_LADDER = 0.2
CFL_PROBE = 0.4
MARGIN_MIN = 2.0


@pytest.mark.parametrize("name", sorted(TABLEAUS))
def test_two_forms_of_the_amplification_agree(name):
    """A1: coefficient form vs ``(I - z A)`` solve form."""
    tab = TABLEAUS[name]
    diff = np.abs(stability_function(tab, Z_PROBE)
                  - stability_function_dense(tab, Z_PROBE)).max()
    assert diff <= TWO_FORM_TOL, (name, diff)


@pytest.mark.parametrize("name", sorted(TABLEAUS))
def test_amplification_reproduces_the_tableau_order(name):
    """A2: ``R`` matches ``exp`` through ``z^p`` and not through ``z^(p+1)``."""
    tab = TABLEAUS[name]
    err_p, err_p1 = amplification_order(tab, ORDER_Z)
    assert err_p <= ORDER_TOL * ORDER_Z ** (tab.order + 1), (name, err_p)
    # the free (p+1)-term is only pinned when the tableau *is* of order p+1
    if tab.order + 1 <= 4:
        assert err_p1 > 0.5 * ORDER_Z ** (tab.order + 1) / math.factorial(tab.order + 1)


def test_rk4_axis_intervals_match_the_textbook_values():
    """A3 (part 1): RK4 -- the classical anchors."""
    tab = TABLEAUS["rk4"]
    assert abs(real_axis_interval(tab) - 2.785293563405282) <= 1e-8
    assert abs(imaginary_axis_interval(tab) - 2.0 * math.sqrt(2.0)) <= 1e-8


def test_kutta3_axis_intervals_match_the_closed_form_values():
    """A3 (part 2): Kutta-3 -- ``|R(iy)|^2 = 1 - y^4/12 + y^6/36`` peaks at ``sqrt 3``."""
    tab = TABLEAUS["rk3"]
    assert abs(imaginary_axis_interval(tab) - math.sqrt(3.0)) <= 1e-8
    assert abs(real_axis_interval(tab) - 2.5127453266791484) <= 1e-8


def test_heun_and_euler_real_axis_interval_is_two():
    """A3 (part 3): both have ``R(-2) = 1`` exactly and diverge beyond."""
    for name in ("euler", "rk2"):
        assert abs(real_axis_interval(TABLEAUS[name]) - 2.0) <= 1e-8, name


def test_butcher_form_of_ssp_rk3_matches_the_hand_coded_replay():
    """A4: registered ``ssp_rk3`` vs the independent stage replay of the solver."""
    if not REPLAY_ARTIFACT.exists():  # pragma: no cover - artifact always shipped
        pytest.skip("validation/cfl_stability/results.json missing")
    replayed = json.loads(REPLAY_ARTIFACT.read_text())["anchors"]["ssp_rk3_real_axis_interval"]
    assert abs(real_axis_interval(SSP_RK3) - replayed) <= 1e-8


def test_registered_ssp_rk3_reproduces_the_solvers_ssp_rk3_branch():
    """A5: the registered tableau and the solver's own branch are the same scheme."""
    ncells, degree, cfl, t_end = 8, 2, 0.2, 0.3
    _, coeff_ref, info = solve_advection(
        ncells=ncells, degree=degree, final_time=t_end, cfl=cfl,
        time_integrator="ssp_rk3")
    volume, phi_l, phi_r = _operators(degree, torch.device("cpu"),
                                      torch.float64)[3:]
    _, u, _ = solve_advection(ncells=ncells, degree=degree, final_time=0.0)
    rhs = lambda w: _rhs(w, 1.0 / ncells, 1.0, volume, phi_l, phi_r)  # noqa: E731
    for _ in range(int(info["steps"])):
        u = rk_step(rhs, u, info["dt"], SSP_RK3)
    assert torch.allclose(u, coeff_ref, atol=1e-13, rtol=0.0), (
        float((u - coeff_ref).abs().max()), info)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_certified_cfl_of_the_matched_rung_covers_the_accuracy_ladder(degree):
    """A6: ``cfl = 0.2`` and ``0.4`` are inside the region, with margin >= 2."""
    tab = tableau_for_degree(degree)
    assert tab.stages == degree + 1
    matrix = assemble_operator(degree, 16, 1.0)
    eig = np.concatenate([np.linalg.eigvals(symbol_matrix(matrix, degree, theta, 16))
                          for theta in np.linspace(0.0, math.pi, 64)])
    rec = cfl_limit(tab, eig, degree)
    assert rec["cfl_max"] >= MARGIN_MIN * CFL_LADDER, (degree, tab.name, rec["cfl_max"])
    assert rec["cfl_max"] > CFL_PROBE, (degree, tab.name, rec["cfl_max"])
    # the two CFL values the studies use, expressed as a step through the solver rule
    assert is_stable(tab, eig, CFL_LADDER / (2 * degree + 1))
    assert is_stable(tab, eig, CFL_PROBE / (2 * degree + 1))


def test_tableau_selection_is_unchanged_by_the_ssp_rk3_registration():
    """A7: the round-2 rung mapping must survive the extra registration."""
    assert [tableau_for_degree(p).name for p in range(5)] == [
        "euler", "rk2", "rk3", "rk4", "ssp_rk54"]
    assert tableau_for_degree(5).name == "rk5"
    assert SSP_RK3.name == "ssp_rk3" and SSP_RK3.stages == 3


def test_max_stable_dt_on_a_single_eigenvalue_reduces_to_the_axis_interval():
    """Sanity: one eigenvalue on the negative real axis reproduces ``|R(-x)|<=1``."""
    tab = TABLEAUS["rk4"]
    dt_max, corrected = max_stable_dt(tab, [-1.0], hi=8.0)
    assert abs(dt_max - real_axis_interval(tab)) <= 1e-6
    assert corrected is False
    dt_max2, _ = max_stable_dt(tab, [-2.0], hi=8.0)
    assert abs(dt_max2 - 0.5 * real_axis_interval(tab)) <= 1e-6


def test_stored_artifact_verdict_is_pass_and_margins_hold():
    """Regression lock on the shipped certificate (``validation/.../metrics.json``)."""
    if not STABILITY_ARTIFACT.exists():  # pragma: no cover - artifact always shipped
        pytest.skip("validation/advection_order_stability/metrics.json missing")
    art = json.loads(STABILITY_ARTIFACT.read_text())
    assert art["verdict"] == "PASS", art.get("failed")
    assert art["checks"]["S3_cfl_used"] and art["checks"]["S4_probe_cfl"]
    for row in art["matched"]:
        assert row["stages_matched"]
        assert row["margin_at_used"] >= MARGIN_MIN, row
        assert row["probe_inside"], row
        # the stored binding eigenvalue reproduces the decision it was used for:
        # stable just inside dt_max, unstable just outside it
        tab = TABLEAUS[row["tableau"]]
        lam = complex(*row["binding_lambda"])
        assert is_stable(tab, [lam], 0.99 * row["dt_max_h_over_a"]), row
        assert not is_stable(tab, [lam], 1.01 * row["dt_max_h_over_a"]), row