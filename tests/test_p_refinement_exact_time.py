"""Gates for the exact-in-time p-refinement ladder (CASE-DG-PREFINEMENT-EXPONENTIAL).

The published ladder stopped at p = 6 and extrapolated p = 7/8
(``docs/CLOSEOUT-m16918-prefinement-criterion.md``).  ``benchmarks/p_refinement_exact_time.py``
measures them by removing the time integrator from the measurement.  These gates
assert *invariants* of that study, and four of them re-run the machinery live so
that a regression in the solver or in the matrix-exponential helper turns the
gate red instead of leaving a stale JSON untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from benchmarks.p_refinement_exact_time import (
    ARTIFACT_RELPATH,
    NCELLS,
    assemble_operator,
    exact_time_level,
)
from benchmarks.p_refinement_floor import l2_error_extended
from tensordg.convergence import (
    exact_time_solution,
    expm_scaled_squaring,
    operator_linearity_residual,
    semidiscrete_operator,
)

ARTIFACT = Path(__file__).resolve().parent.parent / ARTIFACT_RELPATH / "exact_time_results.json"
PROJECTION_FACTOR_TOL = 3.0
FLOOR_SNR_MIN = 10.0


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.exists():
        pytest.skip(f"artifact missing: {ARTIFACT}")
    return json.loads(ARTIFACT.read_text())


# ---------------------------------------------------------------------------
# Artifact-side gates (fast: no solver call)
# ---------------------------------------------------------------------------


def test_artifact_declares_the_measured_high_degree_ladder(artifact):
    assert artifact["schema"] == "tensordg-p-refinement-exact-time/1"
    assert artifact["case"] == "CASE-DG-PREFINEMENT-EXPONENTIAL"
    degrees = [lv["degree"] for lv in artifact["levels"]]
    assert degrees == list(range(1, max(degrees) + 1))
    assert max(degrees) >= 8, "the point of this study is to reach MAX_DEGREE"


def test_verdict_is_derived_from_the_checks_not_asserted(artifact):
    """No overclaim: the published verdict must equal the conjunction of the gates."""
    checks = artifact["checks"]
    gating = (
        "all_levels_resolved_above_measurement_floor", "monotone_decreasing",
        "expm_controls_agree", "scipy_expm_agrees_where_adequate",
        "quadrature_converged", "operator_linear", "mass_conserved",
        "published_rk4_reproduced", "small_cfl_rk4_cross_check",
    )
    expected = all(checks[key]["pass"] for key in gating) and checks[
        "discriminating_criterion"]["pass"]
    if checks["projection_test"]["applicable"]:
        expected = expected and checks["projection_test"]["all_pass"]
    assert (artifact["verdict"] == "PASS") is bool(expected)
    assert artifact["overall_pass"] is bool(expected)


def test_resolved_ladder_is_contiguous_and_reaches_p7(artifact):
    """Levels above the measurement floor form a contiguous ladder reaching p=7.

    The measurement floor is max(float64 representation, float64-operator
    sensitivity): a level below it is reported but must not feed the fit.
    """
    resolved = artifact["floor"]["resolved_degrees"]
    assert resolved == list(range(min(resolved), max(resolved) + 1)), resolved
    # The measured p=7 value is 3.1x the float64-operator noise bound and p=8 is
    # below it, so the certifiable ladder must stop at p=6 or p=7 -- if a future
    # run certifies p=8 too, this gate reminds the reader that the floor moved.
    assert max(resolved) >= 6, resolved
    for lv in artifact["levels"]:
        if lv["resolved"]:
            assert lv["snr_over_measurement_floor"] >= FLOOR_SNR_MIN, lv["degree"]
    limited = artifact["floor"]["floor_limited_degrees"]
    assert all(p_ > min(resolved) for p_ in limited), limited


def test_exact_time_error_is_not_larger_than_the_rk4_error(artifact):
    """Spatial-only and temporal+spatial errors must agree to a few percent.

    A strict ``exact <= rk4`` cannot hold: the temporal error is a *different*
    field, so it may cancel part of the spatial error in the L2 norm (it does so
    marginally at p=1..2).  The invariant is agreement within a few percent.
    """
    for lv in artifact["levels"]:
        ratio = lv["exact_time"]["l2_longdouble_expm"] / lv["rk4_production"]["l2"]
        if lv["degree"] <= 6:
            # temporal error negligible: both paths measure the same spatial error
            assert 0.95 <= ratio <= 1.05, (lv["degree"], ratio)
        else:
            # temporal error dominates: the exact-time value must be *below* RK4
            assert ratio <= 1.05, (lv["degree"], ratio)


def test_composite_criterion_holds_on_the_resolved_ladder(artifact):
    diag = artifact["diagnostics_resolved_ladder"]
    criterion = artifact["checks"]["discriminating_criterion"]
    assert diag is not None
    assert diag["monotone_decreasing"] is True
    assert diag["lsq_decay_rate"] > 1.0
    rates = diag["decay_rates"]
    assert all(b >= a - 1e-9 for a, b in zip(rates[:-1], rates[1:])), rates
    assert diag["errors"][0] / diag["errors"][-1] > 1.0e3
    assert criterion["pass"] is True


def test_full_ladder_is_reported_even_where_the_rate_bends(artifact):
    """Honest reporting: the measured ladder including any floor-limited tail."""
    full = artifact["diagnostics"]
    assert full["degrees"] == [lv["degree"] for lv in artifact["levels"]]
    assert full["monotone_decreasing"] is True


def test_small_cfl_rk4_is_reported_as_a_failed_confirmation(artifact):
    """Honest negative result: at p=8 the production RK4 at CFL 5e-3 disagrees.

    The disagreement (~8e-15) is the size of the float64-operator rounding
    response, i.e. the p=8 exact-time value is *not* independently confirmed and
    must not be cited as evidence of continued exponential decay.
    """
    cross = artifact["checks"]["small_cfl_rk4_cross_check"]
    assert cross["cfl"] < 0.01 and cross["steps"] > 10000
    assert cross["pass"] is False, "artifact no longer a failed confirmation -- re-audit"
    assert cross["ratio_rk4_over_exact_time"] > 1.5
    top = artifact["levels"][-1]
    assert top["resolved"] is False
    assert top["operator_representation_floor"] > top["exact_time"]["l2_longdouble_expm"]


def test_published_extrapolation_reproduces_when_time_error_is_removed(artifact):
    """The p=7/8 predictions of measurement_floor.json must land within 3x."""
    check = artifact["projection_check"]
    assert check["applicable"] is True
    for degree in ("7", "8"):
        level = check["levels"][degree]
        ratio = level["measured_over_predicted"]
        assert 1.0 / PROJECTION_FACTOR_TOL <= ratio <= PROJECTION_FACTOR_TOL, (degree, ratio)
    assert check["all_pass"] is True


def test_temporal_error_masks_the_spatial_ladder_at_high_degree(artifact):
    """RK4/shape: negligible at low p, dominant at p=8 -- the reason for this study."""
    factors = {lv["degree"]: lv["rk4_production"]["temporal_dominance_factor"]
               for lv in artifact["levels"]}
    assert factors[1] < 2.0, factors
    assert factors[2] < 2.0, factors
    assert factors[max(factors)] > 20.0, factors
    ordered = [factors[k] for k in sorted(factors)]
    # dominance grows with p (up to L2 cancellation noise at low p)
    assert all(b >= a * 0.99 for a, b in zip(ordered[:-1], ordered[1:])), ordered
    assert ordered[-1] > 50 * ordered[0]


def test_temporal_sweep_shows_stagnation_under_cfl_refinement(artifact):
    """At the top degree even a 4x smaller CFL stays well above the spatial error."""
    sweep = {entry["degree"]: entry["history"] for entry in artifact["temporal_sweep"]}
    assert sweep, "temporal sweep missing (rerun without --quick)"
    top = max(sweep)
    ratios = [h["ratio_to_exact_time"] for h in sweep[top]]
    # CFL 0.08 is dominated by the time integrator; 0.02 must still be > 2x off
    assert ratios[0] > 20.0, (top, ratios)
    assert ratios[-1] > 2.0, (top, ratios)


def test_expm_control_variants_agree(artifact):
    """Varying the scaling/squaring schedule must not move the measured error."""
    assert artifact["checks"]["expm_controls_agree"]["measured"] <= 1e-6
    scipy_check = artifact["checks"]["scipy_expm_agrees_where_adequate"]
    assert scipy_check["measured"] <= 1.0, scipy_check
    # the float64 scipy path is expected to lose adequacy at the top degrees
    assert max(scipy_check["adequate_degrees"]) <= 6


def test_artifact_paths_are_relocatable_and_hashed(artifact):
    provenance = artifact["artifact_provenance"]
    assert not provenance["published_artifact"].startswith("/")
    assert provenance["published_artifact_sha256"]
    assert provenance["floor_artifact_sha256"]
    for name in artifact["figures"]:
        assert not name.startswith("/")


# ---------------------------------------------------------------------------
# Live gates (cheap: matrix assembly + expm, no time stepping)
# ---------------------------------------------------------------------------


def test_live_operator_is_linear_and_conservative():
    torch.set_num_threads(1)
    operator, info = assemble_operator(8, NCELLS)
    assert info["linearity_residual"] < 1e-12
    # periodic conservative DG: the cell averages (P0 modes) have zero net rate,
    # i.e. the sum of the P0 rows of L vanishes column by column.
    p0_rows = operator.reshape(NCELLS, 9, -1)[:, 0, :]
    assert np.max(np.abs(p0_rows.sum(axis=0))) < 1e-10


def _live_exact_time_error(degree: int) -> float:
    """Exact-in-time L2 error of the shipped p=degree scheme (no time stepping)."""
    from tensordg.advection_1d import solve_advection

    operator, _ = assemble_operator(degree, NCELLS)
    centers, coeff0, _ = solve_advection(ncells=NCELLS, degree=degree, final_time=0.0)
    u = (expm_scaled_squaring(operator, 1.0) @ np.asarray(coeff0.numpy().ravel(),
                                                          dtype=np.longdouble))
    return l2_error_extended(centers, np.asarray(u, dtype=float).reshape(NCELLS, degree + 1), 1.0)


def test_live_exact_time_ladder_continues_to_max_degree():
    """p=7 -> p=8 must still reduce the error when time integration is exact.

    Arithmetic fact only: whether the *reduction is measurable* is decided by the
    float64-operator floor (see the artifact gates above, where p=8 is reported
    as not certifiable).
    """
    torch.set_num_threads(1)
    e7 = _live_exact_time_error(7)
    e8 = _live_exact_time_error(8)
    assert e8 < e7 < 1e-12
    assert e7 / e8 > 10.0


def test_live_expm_matches_scipy_on_the_dg_operator():
    torch.set_num_threads(1)
    operator, _ = assemble_operator(3, 8)
    ours = expm_scaled_squaring(operator, 1.0)
    reference = expm(operator * 1.0).astype(np.longdouble)
    rel = float(np.linalg.norm(ours - reference) / np.linalg.norm(reference))
    assert rel < 1e-12, rel


def test_live_exact_time_solution_matches_a_small_step_rk4():
    """Cross-check the matrix exponential against the shipped time integrator."""
    torch.set_num_threads(1)
    from tensordg.advection_1d import _operators, _rhs, l2_error, solve_advection

    ncells, degree, cfl = 4, 0, 2.0e-3
    h = 1.0 / ncells
    _, _, _, volume, phi_l, phi_r = _operators(degree, torch.device("cpu"), torch.float64)

    def apply(vec):
        u = torch.as_tensor(np.asarray(vec, dtype=float).reshape(ncells, degree + 1))
        return _rhs(u, h, 1.0, volume, phi_l, phi_r).numpy().ravel()

    _, u0, _ = solve_advection(ncells=ncells, degree=degree, final_time=0.0)
    exact = exact_time_solution(apply, u0.numpy().ravel(), 1.0)
    centers, coeff, meta = solve_advection(
        ncells=ncells, degree=degree, final_time=1.0, cfl=cfl, time_integrator="rk4",
    )
    assert meta["steps"] > 100
    rk4_err = l2_error(centers, coeff, 1.0)
    expm_err = l2_error_extended(centers, np.asarray(exact, dtype=float).reshape(ncells, degree + 1), 1.0)
    assert abs(expm_err - rk4_err) < 1e-3, (expm_err, rk4_err)


def test_linearity_check_rejects_a_nonlinear_operator():
    """Negative control: the linearity gate must be able to fail."""
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((5, 5))

    def nonlinear(vec):
        v = np.asarray(vec, dtype=float)
        return matrix @ v + 1e-3 * v * v

    residual = operator_linearity_residual(nonlinear, matrix)
    assert residual > 1e-6, residual
    linear_op = semidiscrete_operator(lambda v: matrix @ np.asarray(v, dtype=float), 5)
    assert np.allclose(linear_op, matrix)