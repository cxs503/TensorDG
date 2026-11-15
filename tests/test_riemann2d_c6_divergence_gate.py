"""C6 (physical-magnitude / divergence) gate for the CKS1998 2-D Riemann case.

Why this file exists
--------------------
``docs/CASE-DG-CKS1998-2D-RIEMANN-CONSERVATION.md`` §3 recorded an honest FAIL:
under the *declared* configuration (Lax & Liu 1998 config 3, CFL = 0.2,
t = 0.25) the solution **diverges** -- total mass grows to +4.8e+26 on a 32x32
mesh -- while positivity (C1), the discrete flux-balance identity (C4) and the
periodic control (C5) all still hold, because a diverging run keeps closing the
bookkeeping identities to round-off.  The document listed the missing gate as
"待办 C6" and required that any threshold be pre-registered and calibrated
against a CFL/M scan before being scored.

These tests pin the gate down:

* the bounds are *a priori* (built from the initial data and geometry only);
* the gate is sensitive: it must fail on the recorded divergent numbers while
  C1/C4/C5 stay green, and pass on a physical run;
* a row that cannot be scored is reported ``not_run``, never ``passed``.

No tolerance in ``benchmarks.riemann_2d_cks1998.GATES`` is modified here.
"""
from __future__ import annotations

import math

import pytest
import torch

from benchmarks.riemann_2d_cks1998 import (
    EXTREMUM_SAFETY,
    GATES,
    MASS_FLUX_SAFETY,
    a_priori_bounds,
    c6_extremum_factor,
    c6_mass_bound_ratio,
    c6_row_is_scorable,
    judge,
    run_riemann_2d,
)

CONSERVED = ("mass", "x_momentum", "y_momentum", "energy")


# ---------------------------------------------------------------------------
# 1. The bounds are a priori (derived from the initial data, never from a run)
# ---------------------------------------------------------------------------
def test_a_priori_bounds_are_recomputable_from_the_initial_data_alone():
    bounds = a_priori_bounds(0.05)

    # Lax & Liu config 3 quadrant states, transcribed here independently.
    states = [(0.138, 1.206, 1.206, 0.3), (0.5323, 0.0, 1.206, 0.3),
              (0.5323, 1.206, 0.0, 0.3), (1.5, 0.0, 0.0, 1.5)]
    rho_max0 = max(s[0] for s in states)
    p_max0 = max(s[3] for s in states)
    gamma = 1.4
    speed_max0 = max(
        math.hypot(s[1], s[2]) + math.sqrt(gamma * s[3] / s[0]) for s in states
    )

    assert bounds["rho_max0"] == pytest.approx(rho_max0)
    assert bounds["p_max0"] == pytest.approx(p_max0)
    assert bounds["speed_max0"] == pytest.approx(speed_max0, rel=1e-12)
    assert bounds["perimeter"] == pytest.approx(4.0)  # unit square
    assert bounds["mass0"] == pytest.approx(0.25 * sum(s[0] for s in states))
    assert bounds["mass_flux_bound"] == pytest.approx(
        MASS_FLUX_SAFETY * rho_max0 * speed_max0 * 4.0 * 0.05, rel=1e-12
    )
    assert bounds["rho_cap"] == pytest.approx(EXTREMUM_SAFETY * rho_max0)
    assert bounds["p_cap"] == pytest.approx(EXTREMUM_SAFETY * p_max0)


def test_bounds_are_deterministic_and_time_scaled():
    """Same inputs -> bit-identical bounds; the flux bound grows linearly in t."""
    first = a_priori_bounds(0.05)
    second = a_priori_bounds(0.05)
    assert first == second
    doubled = a_priori_bounds(0.10)
    assert doubled["mass_flux_bound"] == pytest.approx(2.0 * first["mass_flux_bound"])
    assert doubled["rho_cap"] == first["rho_cap"]  # no time dependence


# ---------------------------------------------------------------------------
# 2. Sensitivity: the gate sees exactly what C1/C4/C5 cannot
# ---------------------------------------------------------------------------
def _row(final_time=0.05, mass_change=0.0, rho_max=1.5, p_max=1.5,
         rho_min=0.13, p_min=0.3, finite=True, flux_residual=1e-15,
         periodic=(False, False)):
    bounds = a_priori_bounds(final_time)
    return {
        "ncells": 32,
        "degree": 1,
        "final_time": final_time,
        "periodic": periodic,
        "total_mass": bounds["mass0"] + mass_change,
        "initial_mass": bounds["mass0"],
        "c6_bounds": bounds,
        "max_flux_balance_residual": flux_residual,
        "finite": finite,
        "positivity": {
            "rho_min": rho_min, "rho_max": rho_max,
            "p_min": p_min, "p_max": p_max,
            "rho_positive": rho_min > 0.0, "p_positive": p_min > 0.0,
            "rho_positive_mean": rho_min > 0.0, "p_positive_mean": p_min > 0.0,
        },
        "conservation": {
            name: {"change": mass_change if name == "mass" else 0.0,
                   "boundary_flux": mass_change if name == "mass" else 0.0,
                   "residual": 0.0, "relative_residual": flux_residual}
            for name in CONSERVED
        },
    }


def test_judge_flags_the_recorded_divergent_run_while_c1_c4_c5_stay_green():
    """The +2.209e+07 mass of the recorded 16x16 / CFL=0.2 run must fail C6 only."""
    divergent = _row(mass_change=2.209e07)
    verdicts = judge([divergent])

    assert verdicts["C6_physical_magnitude"]["passed"] is False
    assert verdicts["C6_physical_magnitude"]["worst_mass_bound_ratio"] > 1.0
    # ... and the pre-existing gates really were blind to it:
    assert verdicts["C1_positivity"]["passed"] is True
    assert verdicts["C4_flux_balance"]["passed"] is True


def test_judge_passes_a_physical_row():
    physical = _row(mass_change=0.05, rho_max=1.62, p_max=1.62)
    verdicts = judge([physical])
    c6 = verdicts["C6_physical_magnitude"]
    assert c6["passed"] is True
    assert c6["worst_mass_bound_ratio"] < 1.0
    assert c6["mass_tolerance"] == 1.0
    assert c6["extremum_tolerance"] == GATES["C6_extremum_factor"]


def test_judge_flags_polynomial_bulge_even_when_the_mass_is_fine():
    """Candidate (ii) of the divergence: shock cells bulge above the initial max."""
    bulged = _row(mass_change=1e-3, rho_max=40.0, p_max=40.0)
    verdicts = judge([bulged])
    assert verdicts["C6_physical_magnitude"]["passed"] is False
    assert verdicts["C6_physical_magnitude"]["worst_mass_bound_ratio"] < 1.0
    assert verdicts["C6_physical_magnitude"]["worst_extremum_factor"] > GATES["C6_extremum_factor"]


def test_gate_scales_with_final_time_not_with_the_mesh():
    """A mass change that is physical at t=0.05 is not at t=0.004 (bound shrinks)."""
    change = 1.0
    late = judge([_row(final_time=0.5, mass_change=change)])["C6_physical_magnitude"]
    early = judge([_row(final_time=0.004, mass_change=change)])["C6_physical_magnitude"]
    assert late["passed"] is True
    assert early["passed"] is False


# ---------------------------------------------------------------------------
# 3. Non-vacuity: unscorable rows are not_run, never passed
# ---------------------------------------------------------------------------
def test_row_without_bounds_is_not_run_not_pass():
    bare = _row()
    del bare["c6_bounds"]
    assert c6_row_is_scorable(bare) is False
    verdict = judge([bare])["C6_physical_magnitude"]
    assert verdict["passed"] is False
    assert verdict["check_state"] == "not_run"


def test_row_with_non_finite_mass_is_not_scorable():
    broken = _row()
    broken["conservation"]["mass"]["change"] = float("nan")
    assert c6_row_is_scorable(broken) is False
    assert judge([broken])["C6_physical_magnitude"]["check_state"] == "not_run"


def test_scorable_row_reports_both_ratios():
    row = _row(mass_change=-0.5, rho_max=2.0, p_max=1.8)
    assert c6_row_is_scorable(row) is True
    assert c6_mass_bound_ratio(row) == pytest.approx(0.5 / row["c6_bounds"]["mass_flux_bound"])
    assert c6_extremum_factor(row) == pytest.approx(2.0 / 1.5)


# ---------------------------------------------------------------------------
# 4. End-to-end: a short real run is scored and passes
# ---------------------------------------------------------------------------
def test_short_real_run_is_scored_and_passes_c6():
    torch.set_num_threads(1)
    record = run_riemann_2d(ncells=4, degree=1, final_time=0.01, cfl=0.1)
    assert record["initial_mass"] == pytest.approx(record["c6_bounds"]["mass0"])
    verdicts = judge([record])
    c6 = verdicts["C6_physical_magnitude"]
    assert "check_state" not in c6, f"row should be scorable, got {c6}"
    assert c6["passed"] is True, c6
    # The short run is nowhere near the a-priori ceiling, with a wide margin.
    assert verdicts["C6_physical_magnitude"]["worst_mass_bound_ratio"] < 0.5