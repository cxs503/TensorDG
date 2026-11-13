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
from pathlib import Path

from tensordg.cases.shock_entropy import (
    DIAPHRAGM,
    DOMAIN,
    LEFT_STATE,
    PLATEAU_MARGIN,
    SHOCK_LEVEL,
    UNDISTURBED_MARGIN,
    WAVE_AMPLITUDE,
    WAVE_NUMBER,
    coarsen,
    exact_mass_budget,
    gate_report,
    rankine_hugoniot_anchor,
    relative_l1_error,
    resample_cell_averages,
    shock_position,
    solve_dg_case,
)

COMMITTED_REFERENCE = Path("benchmarks/data/shu_osher_reference.npz")

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

# ---------------------------------------------------------------------------
# reference comparison across grids that do not share a refinement ratio
def _smooth_profile(ncells, amplitude=0.3, wavenumber=5.0):
    """A smooth density profile sampled as cell averages on ``ncells`` cells."""
    x = DOMAIN[0] + (DOMAIN[1] - DOMAIN[0]) / ncells * (np.arange(ncells) + 0.5)
    return x, 1.0 + amplitude * np.sin(wavenumber * x)


def test_resampling_preserves_mass_for_any_ratio():
    """Conservative re-averaging must neither create nor destroy mass."""
    rng = np.random.default_rng(0)
    for n_src, n_tgt in ((300, 200), (200, 300), (300, 13), (7, 3)):
        values = 0.5 + rng.random(n_src)
        out = resample_cell_averages(values, n_tgt)
        assert out.shape == (n_tgt,)
        mass_src = values.sum() * (DOMAIN[1] - DOMAIN[0]) / n_src
        mass_tgt = out.sum() * (DOMAIN[1] - DOMAIN[0]) / n_tgt
        assert mass_tgt == pytest.approx(mass_src, rel=1e-14)
    # a constant field is a fixed point of the remap, whatever the ratio
    np.testing.assert_allclose(resample_cell_averages(np.full(300, 2.5), 13), 2.5,
                               rtol=1e-12, atol=1e-12)


def test_relative_l1_error_remaps_when_no_refinement_ratio_exists():
    """400 vs 300 cells: the documented command used to raise ValueError."""
    x_c, rho_c = _smooth_profile(400)
    _, rho_r = _smooth_profile(300)
    err = relative_l1_error(rho_c, rho_r, domain=DOMAIN)
    assert err["comparison"]["method"] == "remap"
    assert err["comparison"]["common_grid_cells"] == 300
    # both are cell averages of the same smooth field: only the O(h^2)
    # projection difference is left, well under a percent
    assert 0.0 <= err["rel_l1_percent"] < 1.0
    # without the domain the mismatch stays a hard error (no silent broadcast)
    with pytest.raises(ValueError):
        relative_l1_error(rho_c, rho_r)


def test_gate_report_accepts_the_committed_reference_on_another_grid():
    """The documented library entry must report, not crash, on the shipped file."""
    if not COMMITTED_REFERENCE.exists():  # pragma: no cover - data file missing
        pytest.skip("committed Shu--Osher reference profile is absent")
    blob = np.load(COMMITTED_REFERENCE, allow_pickle=True)
    assert blob["rho"].size != SMALL["ncells"], "the point is the grid mismatch"
    report = gate_report(SMALL["ncells"], SMALL["degree"], cfl=SMALL["cfl"],
                         reference=(blob["x"], blob["rho"]))
    ref = report["reference"]
    assert ref["ncells"] == blob["rho"].size
    assert ref["comparison"]["method"] in ("coarsen", "remap")
    assert ref["comparison"]["common_grid_cells"] == min(blob["rho"].size,
                                                         SMALL["ncells"])
    assert isinstance(report["passes_gate"], bool)
    assert report["rel_l1_percent"] > 0.0


# ---------------------------------------------------------------------------
# measurement windows and the shock locator are derived, not hand-picked
def test_shock_position_is_interpolated_between_cell_centres():
    ncells = 200
    x = DOMAIN[0] + (DOMAIN[1] - DOMAIN[0]) / ncells * (np.arange(ncells) + 0.5)
    exact = rankine_hugoniot_anchor()["shock_position_analytic"]
    rho = np.where(x < exact, LEFT_STATE[0],
                   1.0 + WAVE_AMPLITUDE * np.sin(WAVE_NUMBER * x))
    located = shock_position(x, rho)
    dx = (DOMAIN[1] - DOMAIN[0]) / ncells
    nearest = float(x[np.argmax(x >= exact)])
    assert abs(located - exact) < abs(nearest - exact), "interpolation must beat rounding"
    assert abs(located - exact) < 0.5 * dx
    # an explicit level is honoured, and a profile that never crosses it reports None
    assert shock_position(x, rho, level=3.0) is not None
    assert shock_position(x, np.full(ncells, 1.0)) is None


def test_measurement_windows_follow_the_analytic_wave_skeleton():
    anchor = rankine_hugoniot_anchor()
    report = gate_report(**SMALL)
    assert report["windows"]["plateau"] == [DOMAIN[0], DIAPHRAGM + PLATEAU_MARGIN]
    assert report["windows"]["undisturbed"][0] == pytest.approx(
        anchor["undisturbed_window"][0])
    assert report["windows"]["undisturbed"][0] == pytest.approx(
        anchor["shock_position_analytic"] + UNDISTURBED_MARGIN)
    assert report["shock_level"] == pytest.approx(SHOCK_LEVEL)
    # the undisturbed window sits ahead of the shock and is sampled at all
    assert report["windows"]["undisturbed"][0] > anchor["shock_position_analytic"]
    assert report["preshock_sine_cells"] > 0
    assert report["preshock_sine_rel_l1_percent"] is not None


def test_exact_mass_budget_is_closed_form():
    """No quadrature: the closed form must agree with an independent midpoint rule."""
    budget = exact_mass_budget()
    samples = 400_000
    x = DOMAIN[0] + (DOMAIN[1] - DOMAIN[0]) / samples * (np.arange(samples) + 0.5)
    rho0 = np.where(x < DIAPHRAGM, LEFT_STATE[0],
                    1.0 + WAVE_AMPLITUDE * np.sin(WAVE_NUMBER * x))
    quadrature = float(rho0.sum() * (DOMAIN[1] - DOMAIN[0]) / samples)
    assert budget["mass_initial"] == pytest.approx(quadrature, rel=1e-10)
    # the legacy ``samples`` argument is accepted and cannot change the answer
    assert exact_mass_budget(samples=200_000)["mass_initial"] == budget["mass_initial"]
    assert budget["quadrature"].startswith("closed form")
