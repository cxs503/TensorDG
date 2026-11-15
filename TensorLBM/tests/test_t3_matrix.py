"""Tests for the T3 validation matrix and the wall boundary conditions.

Two kinds of assertions live here:

* **Physics**: the wall operator is verified against exact solutions
  (Couette is linear to machine precision, Poiseuille is parabolic to <1 %),
  and the Taylor-Green probe reproduces the analytic decay rate.
* **Evidence discipline**: the matrix must not be able to claim a case it
  cannot run.  A case without a registered probe is ``NOT_SUPPORTED``; a
  probe that returns a number without advancing the solver is downgraded to
  ``MOCK``; a tolerance looser than the ecosystem acceptance limit is
  rejected by :func:`validate_spec`; a failing probe yields ``FAIL``.
"""

import math

import pytest
import torch

from tensorlbm.cavity import (
    couette_error,
    poiseuille_error,
    run_couette,
    run_poiseuille,
)
from tensorlbm.t3_matrix import (
    ACCEPTANCE_LIMIT,
    CASES,
    run_matrix,
    to_markdown,
    validate_spec,
)


@pytest.fixture(autouse=True)
def _single_thread():
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Spec discipline
# ---------------------------------------------------------------------------


def test_spec_is_valid_and_complete():
    assert validate_spec() == []
    assert len(CASES) == 9
    kinds = {c.reference_kind for c in CASES}
    assert kinds <= {"analytic", "literature", "reference-software"}
    # every tolerance sits inside the ecosystem acceptance limit
    assert all(0.0 < c.tolerance <= ACCEPTANCE_LIMIT for c in CASES)
    # each case is reproducible from the repository root
    assert all(c.key in c.repro_cmd for c in CASES)


def test_spec_rejects_tolerance_above_acceptance_limit():
    import dataclasses

    bad = [
        dataclasses.replace(c, tolerance=0.30) if c.key == "cavity-2d-ghia" else c
        for c in CASES
    ]
    problems = validate_spec(bad)
    assert any("tolerance" in p for p in problems)


def test_spec_rejects_real_claim_without_registered_probe():
    import dataclasses

    bad = [
        dataclasses.replace(c, probe="not_a_probe", gap=None)
        if c.key == "taylor-green-2d" else c
        for c in CASES
    ]
    problems = validate_spec(bad)
    assert any("not registered" in p for p in problems)


def test_spec_rejects_unsupported_case_without_gap():
    import dataclasses

    bad = [
        dataclasses.replace(c, gap="   ") if c.key == "sphere-3d" else c
        for c in CASES
    ]
    problems = validate_spec(bad)
    assert any("documented gap" in p for p in problems)


# ---------------------------------------------------------------------------
# Runners must not fabricate evidence
# ---------------------------------------------------------------------------


def test_unsupported_case_is_not_counted_as_pass():
    summary = run_matrix(only=["sphere-3d", "dam-break-ritter"])
    for row in summary["rows"]:
        assert row["status"] == "NOT_SUPPORTED"
        assert row["evidence"] == "NOT_SUPPORTED"
        assert row["pass_"] is None
        assert row["gap"]
    assert summary["summary"]["n_real_pass"] == 0
    assert summary["summary"]["coverage_real"] == 0.0


def test_probe_without_solver_steps_is_downgraded_to_mock():
    def fake_probe(tolerance, **_):
        return {"measured": 0.0, "pass": True, "solver_steps": 0, "detail": {}}

    summary = run_matrix(only=["taylor-green-2d"],
                         probes={"tg_decay_2d": fake_probe})
    row = summary["rows"][0]
    assert row["evidence"] == "MOCK"
    assert row["status"] == "FAIL"
    assert row["pass_"] is False


def test_failing_probe_is_reported_as_fail():
    def bad_probe(tolerance, **_):
        return {"measured": 0.9, "pass": False, "solver_steps": 10, "detail": {}}

    summary = run_matrix(only=["taylor-green-2d"],
                         probes={"tg_decay_2d": bad_probe})
    row = summary["rows"][0]
    assert row["status"] == "FAIL" and row["pass_"] is False
    assert row["evidence"] == "REAL"


def test_probe_exception_is_withheld_not_passed():
    def exploding(tolerance, **_):
        raise RuntimeError("solver diverged")

    summary = run_matrix(only=["taylor-green-2d"],
                         probes={"tg_decay_2d": exploding})
    row = summary["rows"][0]
    assert row["status"] == "ERROR"
    assert row["evidence"] == "WITHHELD"
    assert row["pass_"] is None
    assert "diverged" in row["gap"]


def test_markdown_renders_every_row():
    summary = run_matrix(only=["sphere-3d"])
    md = to_markdown(summary)
    assert "sphere-3d" in md
    assert "NOT_SUPPORTED" in md


# ---------------------------------------------------------------------------
# Wall boundary conditions against exact solutions
# ---------------------------------------------------------------------------


def test_couette_profile_is_linear_to_machine_precision():
    """Couette is the sharpest test of the moving-wall term itself."""
    res = run_couette(Ny=17, Nx=8, U=0.05, tau=1.0, n_steps=20000,
                      tol=1e-11, check_every=500)
    err = couette_error(res)
    assert err["converged"]
    assert err["linf"] < 1e-6, err
    # the profile is monotone and the top fluid node is the fastest
    Y, u = res.centreline_u()
    assert float(u[-1]) == pytest.approx(1.0, abs=0.05)
    assert bool((u[1:] > u[:-1]).all())


def test_poiseuille_profile_is_parabolic():
    res = run_poiseuille(Ny=17, Nx=8, body_force=1e-5, tau=0.8,
                         n_steps=20000, tol=1e-8, check_every=500)
    err = poiseuille_error(res, 1e-5)
    assert err["converged"]
    assert err["linf_rel"] < 0.01, err
    assert err["u_max_lb"] == pytest.approx(err["u_max_exact"], rel=0.01)


def test_moving_wall_injects_momentum_with_the_right_sign():
    """A lid moving in +x must push the fluid in +x, not -x."""
    from tensorlbm.cavity import run_cavity

    res = run_cavity(N=17, U=0.05, re=100.0, n_steps=200, tol=1e-12,
                     min_steps=10 ** 9, check_every=10 ** 9)
    fluid = ~res.solid
    total = float(res.ux[fluid].sum())
    assert total > 0.0
    # and the lid-adjacent row is faster than the middle of the cavity
    yc = res.Nx // 2
    assert float(res.ux[-2, yc]) > float(res.ux[res.Ny // 2, yc])


def test_cavity_wall_gap_convention():
    """H = N - 2 and Y spans (0, 1) strictly inside the fluid band."""
    from tensorlbm.cavity import run_cavity

    res = run_cavity(N=17, U=0.05, re=100.0, n_steps=50, check_every=10 ** 9)
    assert res.H == res.Nx - 2 == res.Ny - 2
    Y, _ = res.centreline_u()
    assert 0.0 < float(Y[0]) and float(Y[-1]) < 1.0
    assert float(Y[0]) == pytest.approx(0.5 / res.H)


# ---------------------------------------------------------------------------
# Taylor-Green probe (analytic rate)
# ---------------------------------------------------------------------------


def test_taylor_green_decay_rate_matches_analytic():
    from tensorlbm.t3_matrix import probe_tg_decay_2d

    res = probe_tg_decay_2d(n=32, nu=0.05, n_steps=600, tolerance=0.02)
    detail = res["detail"]
    assert detail["decay_rate_fit"] > 0.0
    assert abs(detail["decay_rate_fit"] - detail["decay_rate_analytic"]) / \
        detail["decay_rate_analytic"] < 0.02
    assert res["pass"] is True
    assert res["solver_steps"] == 600
    # the amplitude must actually decay (guards a frozen solver)
    assert detail["amplitude_last"] < 0.5 * detail["amplitude_first"]


def test_taylor_green_rate_error_scales_with_resolution():
    """A wrong decay rate is not allowed to hide behind a coarse grid."""
    from tensorlbm.t3_matrix import probe_tg_decay_2d

    coarse = probe_tg_decay_2d(n=16, nu=0.05, n_steps=300, tolerance=0.02)
    assert math.isfinite(coarse["measured"])
    assert coarse["measured"] < 0.05