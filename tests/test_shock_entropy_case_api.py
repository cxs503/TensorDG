"""Integration gates for the library-owned Shu--Osher case API.

The community-facing contract of the case is *not* the benchmark script but the
library: :mod:`tensordg.cases.shock_entropy` plus the ``tensordg`` namespace
exports and the solver hook :func:`tensordg.verify_shock_entropy`.  These tests
keep that contract honest and cheap (small grids only; the resolution ladder and
the independent reference solver are exercised by
``tests/test_shu_osher_case_pool.py``).
"""
from __future__ import annotations

import numpy as np
import pytest

import tensordg
from tensordg.cases.shock_entropy import (
    DOMAIN,
    LEFT_STATE,
    coarsen,
    exact_mass_budget,
    gate_report,
    rankine_hugoniot_anchor,
    relative_l1_error,
    solve_dg_case,
)

SMALL = dict(ncells=48, degree=1, cfl=0.4)


def test_package_namespace_exposes_the_case():
    """The library (not the benchmark dir) is the owner users import from."""
    for name in ("solve_shock_entropy_dg", "shock_entropy_anchor",
                 "shock_entropy_gate_report", "verify_shock_entropy"):
        assert name in tensordg.__all__, f"{name} missing from tensordg.__all__"
        assert callable(getattr(tensordg, name))


def test_solver_hook_runs_the_case_and_respects_positivity():
    """``verify_shock_entropy`` is the main solver's self-check hook."""
    report = tensordg.verify_shock_entropy(**SMALL)
    assert report["rho_positive"] is True
    assert report["rho_min"] > 0.0
    assert report["steps"] > 0
    # closed-form anchors: plateau = left state, exact mass budget
    assert report["plateau_dev_percent"] < 1.0
    # coarse-grid initial projection of the sine is O(h^2); at N>=200 this is <1e-9
    assert report["mass_error_percent_of_initial"] < 1.0
    # the sine ahead of the transmitted shock is still the initial data
    # (4e-2 % at N=800, 3.5e-2 % at N=200; only ~4 % at this deliberately tiny N)
    assert report["preshock_sine_rel_l1_percent"] < 10.0


def test_gate_report_without_reference_reports_anchors_only():
    report = gate_report(**SMALL)
    assert report["passes_gate"] is None, "no reference -> no relative-error verdict"
    for key in ("rho_min", "rho_max", "plateau_rho_mean", "mass_t1p8", "shock_position"):
        assert key in report
    assert abs(report["plateau_target"] - LEFT_STATE[0]) == 0.0


def test_gate_report_against_a_coarse_reference_is_well_formed():
    """A reference twice as fine must be coarsened, not broadcast."""
    out = solve_dg_case(SMALL["ncells"], degree=SMALL["degree"], cfl=SMALL["cfl"])
    # build a reference on a 2x finer grid by midpoint replication of the profile
    x_ref = np.linspace(DOMAIN[0], DOMAIN[1], 2 * SMALL["ncells"], endpoint=False) \
        + 0.5 * (DOMAIN[1] - DOMAIN[0]) / (2 * SMALL["ncells"])
    rho_ref = np.repeat(out["rho"], 2)
    report = gate_report(SMALL["ncells"], SMALL["degree"], cfl=SMALL["cfl"],
                         reference=(x_ref, rho_ref))
    assert report["reference"]["coarsen_factor"] == 2
    assert report["rel_l1_percent"] == pytest.approx(0.0, abs=1e-12)
    assert report["passes_gate"] is True


def test_relative_l1_error_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        relative_l1_error(np.zeros(10), np.zeros(9))


def test_coarsen_is_a_volume_average():
    values = np.arange(8, dtype=float)
    np.testing.assert_allclose(coarsen(values, 2), [0.5, 2.5, 4.5, 6.5])


def test_anchors_are_self_consistent():
    """The left state must be reproducible from a Mach-3 shock (no transcribed data)."""
    anchor = rankine_hugoniot_anchor()
    assert anchor["hugoniot_relative_error"]["rho"] < 1e-5
    assert anchor["hugoniot_relative_error"]["u"] < 1e-5
    assert anchor["hugoniot_relative_error"]["p"] < 1e-5
    assert anchor["shock_position_analytic"] > 0.0
    budget = exact_mass_budget()
    assert budget["mass_final_predicted"] > budget["mass_initial"] > 0.0