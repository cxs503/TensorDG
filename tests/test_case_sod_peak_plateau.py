"""CASE-DG-SOD-SHOCK: density peak position / plateau verification.

Reference: Sod (1978) J. Comput. Phys. 27(1):1-31; wave structure as in Toro
(2009) 3rd ed. Sec. 4.3.  The published star-state constants below are used
only to *cross-check* the repository's closed-form evaluation -- every
pass/fail value in the case is computed, not copied.
"""
import numpy as np
import pytest

from benchmarks.case_sod_peak_plateau import (
    DIAPHRAGM, FINAL_TIME, GAMMA, LEFT, RIGHT, TOLERANCE,
    compare_features, exact_features, measure_density_features, run_single,
)
from benchmarks.riemann_exact import sample
from tensordg.euler_dg_1d import solve_euler_dg_1d


def test_exact_features_match_published_sod_star_state():
    feat = exact_features()
    published = {
        "p_star": 0.303130,
        "u_star": 0.927453,
        "rho_peak": 0.426319,        # rho_L*, left-star density (density peak)
        "rho_plateau": 0.265574,     # rho_R*, plateau behind the contact
        "shock_speed": 1.752156,
    }
    for key, value in published.items():
        assert feat[key] == pytest.approx(value, rel=1e-5), key
    # the shock speed must agree with the Rankine-Hugoniot mass jump
    assert feat["shock_speed"] == pytest.approx(feat["shock_speed_rankine_hugoniot"], rel=1e-12)
    # wave ordering head < tail < contact < shock, and the exact widths
    assert feat["x_head"] < feat["x_tail"] < feat["x_contact"] < feat["x_shock"]
    assert feat["x_tail"] == pytest.approx(0.4859454, abs=1e-6)
    assert feat["x_contact"] == pytest.approx(0.6854905, abs=1e-6)
    assert feat["x_shock"] == pytest.approx(0.8504311, abs=1e-6)


def test_extractor_recovers_exact_features_from_exact_profile():
    """The measurement definition must be unbiased when fed the exact solution."""
    feat = exact_features()
    x = np.linspace(0.0, 1.0, 40001)
    exact = sample(x, FINAL_TIME, LEFT, RIGHT, DIAPHRAGM, GAMMA)
    meas = measure_density_features(x, exact["density"], feat)
    rows = {row["metric"]: row for row in compare_features(meas, feat)}
    for key in ("rho_peak", "rho_plateau", "x_contact", "x_shock", "plateau_width"):
        assert rows[key]["rel_err_pct"] < 0.1, (key, rows[key]["rel_err_pct"])
    # peak onset is defined with a 1 % level, so a small bias is expected there
    assert rows["x_peak"]["rel_err_pct"] < 0.5
    assert meas["rho_peak_overshoot_pct"] == pytest.approx(0.0, abs=1e-9)


def test_modal_evaluation_matches_solver_basis():
    """Guards the Legendre basis convention used for crossing detection."""
    from benchmarks.case_sod_peak_plateau import pointwise_primitive
    from tensordg.curved_geometry import legendre_eval_matrix

    x, coeff, _ = solve_euler_dg_1d(LEFT, RIGHT, ncells=40, degree=2, final_time=0.01, cfl=0.3)
    coeff_np = coeff.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()

    # Gauss quadrature of the reconstruction must return the first modal
    # coefficient (the cell average) to machine precision.
    nodes, weights = np.polynomial.legendre.leggauss(6)
    L = legendre_eval_matrix(2, nodes)
    point = np.einsum("cdv,sd->csv", coeff_np, L)
    cell_mean = np.einsum("csv,s->cv", point, weights) / 2.0
    assert np.allclose(cell_mean, coeff_np[:, 0], rtol=0, atol=1e-13)

    # the uniform-subsample evaluator used by the case agrees to sampling error
    _, prim = pointwise_primitive(x_np, coeff_np, degree=2, subsamples=8)
    rho = prim["density"]
    assert rho.size == 40 * 8
    assert (rho > 0.0).all() and np.isfinite(prim["pressure"]).all()
    assert np.allclose(rho.reshape(-1, 8).mean(axis=1), coeff_np[:, 0, 0], rtol=1e-2, atol=0)


def test_documented_configuration_meets_four_percent():
    """Predeclared case gate: every gated feature within 4 % at P2 / 100 cells."""
    run = run_single(degree=2, ncells=100, cfl=0.15)
    assert run["pass"], [(row["metric"], row["rel_err_pct"]) for row in run["table"]
                         if row["pass"] is False]
    assert run["max_rel_err_pct"] < 100 * TOLERANCE