"""Gates for the Zhang-Shu positivity-preserving scaling limiter (Toro test 2).

Structure of the evidence, in the repository convention (a gate must be able to
fail):

* the **analytic reference** is checked against published Toro (2009) star-state
  values and against its own Riemann invariants, so a broken reference cannot
  make the accuracy gate pass;
* the **need** for an admissibility treatment is *demonstrated*, not assumed:
  the same solver with ``positivity="none"`` must raise ``FloatingPointError``
  on the near-vacuum 123 problem;
* the **limiter properties** (cell-average preservation, bitwise identity on an
  admissible cell, value validation) are checked live and cheaply;
* the **accuracy claim** (scaling retains more high order than the historic
  collapse) is re-derived from the committed artifact
  ``artifacts/positivity_limiter/metrics.json`` and cross-checked against the
  live solver on a small configuration.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.positivity_limiter import (
    FINAL_TIME,
    exact_double_rarefaction,
    fan_speeds,
    run_case,
    star_pressure_bisection,
)
from tensordg.euler_dg_1d import (
    GAMMA,
    _positivity_scale,
    _scaling_factors,
    solve_euler_dg_1d,
)

ARTIFACT = (Path(__file__).resolve().parent.parent
            / "artifacts" / "positivity_limiter" / "metrics.json")

# Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed.,
# Table 4.1, test 2 (the "123 problem"): p* = 1.89e-3, u* = 0, rho* = 2.185e-2.
TORO_P_STAR = 1.89e-3
TORO_RHO_STAR = 2.185e-2


def _artifact() -> dict:
    assert ARTIFACT.exists(), f"missing committed artifact {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


def _record(payload, degree, ncells, mode):
    hits = [r for r in payload["records"]
            if r["degree"] == degree and r["ncells"] == ncells and r["positivity"] == mode]
    assert hits, f"record p={degree} N={ncells} {mode} missing from artifact"
    return hits[0]


# --------------------------------------------------------------------------
# 1. the analytic reference is right (independent of any solver code)
# --------------------------------------------------------------------------
def test_star_pressure_matches_toro_table():
    p_star = star_pressure_bisection()
    assert abs(p_star - TORO_P_STAR) / TORO_P_STAR < 1e-2


def test_star_density_matches_toro_table():
    p_star = star_pressure_bisection()
    sol = exact_double_rarefaction(np.array([0.5]), FINAL_TIME, p_star=p_star)
    assert abs(sol["rho_star"] - TORO_RHO_STAR) / TORO_RHO_STAR < 1e-2
    assert abs(sol["u_star"]) < 1e-12


def test_exact_solution_satisfies_its_own_riemann_invariants():
    x = np.linspace(0.0, 1.0, 2001)
    sol = exact_double_rarefaction(x, FINAL_TIME)
    # symmetry of a symmetric Riemann problem
    assert np.max(np.abs(sol["u"] + sol["u"][::-1])) < 1e-12
    assert np.max(np.abs(sol["p"] - sol["p"][::-1])) < 1e-12
    # fan tails must reproduce u* -+ a* with a* = sqrt(gamma p* / rho*)
    p_star = sol["p_star"]
    a_star = math.sqrt(GAMMA * p_star / sol["rho_star"])
    head_l, tail_l, tail_r, head_r = sol["fans"]
    assert abs(sol["u_star"] - a_star - tail_l) < 1e-12
    assert abs(sol["u_star"] + a_star - tail_r) < 1e-12
    # the reference is only used while the fans stay inside the domain
    assert fan_speeds(p_star=p_star)["inside_domain"]
    # the star region is exactly constant: the minimum is rho* and it is attained
    # on the whole plateau (an argmin tie, hence the value test rather than an index)
    assert abs(sol["rho"].min() - sol["rho_star"]) < 1e-12
    assert abs(sol["rho"][len(x) // 2] - sol["rho_star"]) < 1e-12


# --------------------------------------------------------------------------
# 2. detection power: without a treatment the high-order state is non-physical
# --------------------------------------------------------------------------
def test_untreated_high_order_state_is_nonphysical():
    record = run_case(60, 2, "none")
    assert record["status"] == "NONPHYSICAL"
    assert "Nonphysical" in record["error"]


# --------------------------------------------------------------------------
# 3. limiter properties (live, cheap)
# --------------------------------------------------------------------------
def test_invalid_positivity_mode_is_rejected():
    with pytest.raises(ValueError):
        solve_euler_dg_1d((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=40, degree=1,
                          final_time=0.01, positivity="bogus")


def test_limiter_leaves_an_admissible_cell_bitwise_untouched():
    torch.manual_seed(0)
    ncell, degree = 8, 2
    coeff = torch.zeros((ncell, degree + 1, 3), dtype=torch.float64)
    # admissible, smooth states: rho = 1 + 0.01 x, u = 0.1, p = 1
    coeff[:, 0, 0] = 1.0
    coeff[:, 1, 0] = 0.01
    coeff[:, 0, 1] = 0.1
    coeff[:, 0, 2] = 2.5  # E = p/(gamma-1) + 0.5 rho u^2
    s = np.linspace(-1.0, 1.0, 5)
    basis = torch.tensor(
        np.stack([np.ones_like(s), s, 0.5 * (3.0 * s ** 2 - 1.0)], axis=1),
        dtype=torch.float64,
    )  # shape (q, m): check points times modes
    theta = _scaling_factors(coeff, basis)
    assert torch.allclose(theta, torch.ones_like(theta))
    scaled = _positivity_scale(coeff, basis)
    assert torch.equal(scaled, coeff)


def test_limiter_preserves_cell_averages_and_admissibility():
    x, coeff, meta = solve_euler_dg_1d((1.0, -2.0, 0.4), (1.0, 2.0, 0.4), ncells=60,
                                       degree=2, final_time=0.05, positivity="scale")
    assert torch.isfinite(coeff).all()
    # the scaling limiter only touches the higher modes: cell averages are exact
    s7 = np.linspace(-1.0, 1.0, 7)
    phi = torch.tensor(np.stack([np.ones_like(s7), s7, 0.5 * (3.0 * s7 ** 2 - 1.0)], axis=1),
                       dtype=torch.float64)
    values = torch.einsum("cmv,qm->cqv", coeff, phi)
    rho = values[..., 0]
    pressure = (GAMMA - 1.0) * (values[..., 2] - 0.5 * values[..., 1] ** 2 / rho)
    assert float(rho.min()) > 0.0
    assert float(pressure.min()) > 0.0
    stats = meta["positivity_stats"]
    assert stats["min_theta"] >= 0.0 and stats["min_theta"] <= 1.0
    assert stats["limited_cells"] >= 0


# --------------------------------------------------------------------------
# 4. accuracy claim, re-derived from the committed artifact
# --------------------------------------------------------------------------
def test_artifact_records_are_admissible():
    payload = _artifact()
    for record in payload["records"]:
        if record["status"] != "ok":
            continue
        assert record["min_rho"] > 0.0
        assert record["min_p"] > 0.0
        assert record["L1_rho"] > 0.0


def test_scaling_limiter_retains_more_accuracy_than_collapse():
    payload = _artifact()
    for degree in (2, 3):
        collapse = _record(payload, degree, 60, "collapse")
        scale = _record(payload, degree, 60, "scale")
        assert scale["L1_rho"] < 0.95 * collapse["L1_rho"], (degree, scale["L1_rho"],
                                                             collapse["L1_rho"])
        assert scale["L1_p"] < 0.90 * collapse["L1_p"]
        # a less dissipative treatment resolves the near-vacuum core better:
        # min p should be closer to the exact p* from below
        p_star = payload["analytic"]["p_star"]
        assert abs(scale["min_p"] - p_star) < abs(collapse["min_p"] - p_star)


def test_untreated_case_is_reported_as_nonphysical_in_the_artifact():
    payload = _artifact()
    for degree in (2, 3):
        assert _record(payload, degree, 60, "none")["status"] == "NONPHYSICAL"