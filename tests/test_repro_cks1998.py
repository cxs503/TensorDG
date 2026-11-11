"""Regression gates for the CKS1998 reproduction (mission REPRO-DG-CKS1998).

These are the *fast* variants of the three anchor gates; the full sweeps live in
``validation/repro_cks1998/repro_cks1998.py`` (profile ``full``) and are stored
under ``runs/cks1998/``.  Each test names the anchor tolerance it defends.
"""
import numpy as np
import torch

from tensordg.advection_1d import l1_error, l2_error, solve_advection
from tensordg.convergence import observed_order
from tensordg.limiters import apply_tvb_limiter


def _square(x):
    xm = torch.remainder(x, 1.0)
    return ((xm > 0.25) & (xm < 0.75)).to(x.dtype)


# ---------------------------------------------------------------------------
# plumbing added for CKS1998: limiter + SSP-RK3 on the scalar advection solver
# ---------------------------------------------------------------------------
def test_default_integrator_and_limiter_are_backwards_compatible():
    """Default call must be bit-identical to the explicit historical path."""
    c1, a1, m1 = solve_advection(ncells=24, degree=2, final_time=0.2)
    c2, a2, m2 = solve_advection(ncells=24, degree=2, final_time=0.2,
                                 limiter=None, time_integrator="rk4")
    assert torch.equal(a1, a2)
    assert m1["limiter"] == "none" and m2["time_integrator"] == "rk4"


def test_unknown_troubled_policy_is_rejected():
    try:
        solve_advection(ncells=8, degree=1, final_time=0.1, limiter="tvb",
                        troubled_policy="everything")
    except ValueError as exc:
        assert "troubled_policy" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_unknown_integrator_is_rejected():
    try:
        solve_advection(ncells=8, degree=1, final_time=0.1, time_integrator="euler")
    except ValueError as exc:
        assert "ssp_rk3" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_ssp_rk3_matches_rk4_on_smooth_data():
    """Both integrators must converge to the same smooth solution."""
    c, a_rk4, _ = solve_advection(ncells=40, degree=2, final_time=0.2, cfl=0.1)
    _, a_rk3, _ = solve_advection(ncells=40, degree=2, final_time=0.2, cfl=0.1,
                                  time_integrator="ssp_rk3")
    e4 = l2_error(c, a_rk4, 0.2)
    e3 = l2_error(c, a_rk3, 0.2)
    assert abs(e4 - e3) < 0.05 * max(e4, e3) + 1e-12


# ---------------------------------------------------------------------------
# anchor item 1: linear advection L2 order >= p + 1/2
# ---------------------------------------------------------------------------
def test_linear_advection_l2_order_p1_meets_half_order_tolerance():
    cells = [16, 32, 64]
    errs = []
    for n in cells:
        c, coeff, _ = solve_advection(ncells=n, degree=1, final_time=0.3, cfl=0.3)
        errs.append(l2_error(c, coeff, 0.3))
    order = observed_order(cells, errs)
    assert order >= 1.5, order          # p + 0.5 = 1.5


def test_periodic_tvb_limit_matches_rolled_interior_limiting():
    """periodic=True must equal limiting after rolling the seam into the interior."""
    ncells, degree = 12, 2
    coeff = torch.zeros(ncells, degree + 1, 1, dtype=torch.float64)
    rng = np.random.default_rng(0)
    coeff[:, 0, 0] = torch.as_tensor(rng.random(ncells)) * 0.0 + 1.0
    # jump between the last and the first cell (the periodic seam)
    coeff[0, 0, 0] = 0.0
    coeff[1, 1, 0] = 0.4
    coeff[-1, 1, 0] = -0.4
    dx = 1.0 / ncells
    periodic = apply_tvb_limiter(coeff, dx, tvb_constant=0.0, periodic=True)
    k = 5
    rolled = apply_tvb_limiter(torch.roll(coeff, k, 0), dx, tvb_constant=0.0)
    rolled = torch.roll(rolled, -k, 0)
    assert torch.allclose(periodic, rolled, atol=1e-12)


def test_limiter_removes_square_wave_overshoot():
    """anchor item 3: TVD bound, no new extrema beyond 1 % (strict policy)."""
    ncells, degree = 40, 1
    kw = dict(ncells=ncells, degree=degree, final_time=1.0, cfl=0.2,
              initial=_square, tvb_constant=0.0, time_integrator="ssp_rk3")
    _, unlimited, _ = solve_advection(**kw)
    _, limited, _ = solve_advection(limiter="tvb", troubled_policy="tvb", **kw)
    _, strict, _ = solve_advection(limiter="tvb", troubled_policy="all", **kw)
    over = lambda c: max(0.0, float(c[:, 0].max() - 1.0), float(-c[:, 0].min()))
    assert over(unlimited) > 0.01      # the control does violate the bound
    assert over(limited) <= 0.02       # published indicator: O(h) bulge, ~1.2 %
    assert over(strict) <= 0.01        # declared gate: strict minmod every cell
    assert over(strict) < over(limited)


def test_limited_square_wave_has_half_order_l1_convergence():
    """anchor item: limited advection of a discontinuity converges at O(h^{1/2})."""
    cells = [40, 80]
    errs = []
    for n in cells:
        c, coeff, _ = solve_advection(ncells=n, degree=1, final_time=0.5, cfl=0.2,
                                      initial=_square, limiter="tvb",
                                      tvb_constant=0.0, time_integrator="ssp_rk3")
        errs.append(l1_error(c, coeff, 0.5, exact=_square, jumps=(0.25, 0.75)))
    order = observed_order(cells, errs)
    assert order >= 0.5, (order, errs)


# ---------------------------------------------------------------------------
# anchor item 2: Sod via DG vs exact Riemann solution within 3 %
# ---------------------------------------------------------------------------
def test_sod_via_dg_p2_matches_exact_riemann_solution_within_three_percent():
    """Gate case = finest declared mesh (N=400): P2 at N=200 is 4.17 % (reported)."""
    from benchmarks.sod_exact import cell_averages
    from tensordg.euler_1d import GAMMA
    from tensordg.euler_dg_1d import solve_euler_dg_1d

    ncells, t = 400, 0.2
    _, coeff, meta = solve_euler_dg_1d(
        (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=ncells, degree=2,
        final_time=t, cfl=0.12, limiter="tvb", tvb_constant=1.0,
        limiter_stage="final",
    )
    avg = coeff[:, 0]
    rho = avg[:, 0].numpy()
    vel = (avg[:, 1] / avg[:, 0]).numpy()
    pre = ((GAMMA - 1.0) * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / avg[:, 0])).numpy()
    _, ref = cell_averages(ncells, t)
    for key, got in (("density", rho), ("velocity", vel), ("pressure", pre)):
        rel = np.sqrt(np.mean((got - ref[key]) ** 2)) / np.sqrt(np.mean(ref[key] ** 2))
        assert rel < 0.03, (key, rel)
    assert meta["limiter"] == "tvb"

# ---------------------------------------------------------------------------
# L3 integration: the benchmark entry point (non-test caller of the new API)
# ---------------------------------------------------------------------------
def test_benchmark_module_integration_probe():
    """``benchmarks.cks1998_benchmark`` is the non-test caller of the new path."""
    from benchmarks.cks1998_benchmark import integration_calls

    probe = integration_calls(ncells=40, degree=1)
    assert probe["probe_pass"], probe
    solver = probe["calls"][0]
    assert solver["time_integrator"] == "ssp_rk3" and solver["limiter"] == "tvb"
    assert solver["overshoot_pct"] <= 1e-9          # strict policy: no bulge
    lim = probe["calls"][1]
    assert lim["cells_modified"] > 0
    assert lim["edge_cell_modified"] and lim["last_cell_modified"]  # periodic seam


def test_benchmark_ref_vs_got_table_from_committed_artifacts():
    """The committed run artifacts must regenerate a FAIL-free ref-vs-got table."""
    import json
    import pathlib

    import pytest
    from benchmarks.cks1998_benchmark import build_ref_vs_got

    metrics_path = (pathlib.Path(__file__).resolve().parents[1]
                    / "validation" / "repro_cks1998" / "artifacts" / "quick"
                    / "metrics.json")
    if not metrics_path.exists():
        pytest.skip("run artifacts not present")
    metrics = json.loads(metrics_path.read_text())
    table = build_ref_vs_got(metrics, "quick")
    assert table["counts"]["total"] > 20
    assert table["counts"]["FAIL"] == 0
    assert any(r["ref_source"] == "analytic" and r["ref"] is not None
               for r in table["rows"])
    assert any(r["ref_source"] == "paper_theorem" for r in table["rows"])
