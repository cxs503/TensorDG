"""Acceptance tests for the ice-edge transport DG benchmarks (IN12-P5).

The tests assert the *same* machine-checkable criteria as the campaign artifact
``artifacts/dg_advection_bench.json`` (written by
``tools/run_dg_advection_bench.py``), on a reduced configuration so that the
module stays fast (roughly 1-2 minutes single-threaded):

* the variable-coefficient operator reproduces the constant-velocity reference
  solver of :mod:`tensordg.advection_2d` to round-off;
* global mass is conserved to round-off for divergence-free variable flow;
* a constant field is an exact invariant (consistency of volume + face terms);
* observed order ``>= p + 0.9`` for p = 1, 2 on the periodic shear (analytic),
  the reentrant vortex (reentrant) and the rotating front (restricted norm);
* peak retention of the rotating front after one revolution;
* the artifact contract (paths, criteria numbers, verdict) of the published
  benchmark record.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensordg import ice_edge_bench as bench
from tensordg.advection_2d import solve_advection_2d
from tensordg.advection_2d_var import (
    DGAdvection2D,
    gaussian_bump,
    reentrant_vortex,
    rotated_profile,
    solid_body_rotation,
    trig_profile,
)

OMEGA = 2.0 * math.pi
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "dg_advection_bench.json"


def _order(case: str, degree: int, cells, cfl: float = 0.3) -> dict:
    study = bench.run_order_study(case, degree, cells, cfl=cfl)
    table = study.as_dict()
    assert table["observed_order"] is not None
    return table


# ---------------------------------------------------------------------------
# 1. Operator cross-check against the constant-velocity reference solver
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_constant_velocity_matches_reference_solver(degree: int) -> None:
    """Variable-velocity machinery must degenerate to the reference solver."""
    n, t_end, cfl = 16, 0.25, 0.25
    solver = DGAdvection2D(degree, n)
    velocity = lambda x, y, t: (torch.ones_like(x), 0.5 * torch.ones_like(y))  # noqa: E731
    result = solver.solve(solver.project(trig_profile), velocity, t_end, cfl=cfl)
    _, _, reference, _ = solve_advection_2d(
        ncells=n, degree=degree, final_time=t_end, velocity=(1.0, 0.5), cfl=cfl)
    assert float((result["coeff"] - reference).abs().max()) < 1e-12


# ---------------------------------------------------------------------------
# 2. Global mass conservation (machine precision)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case_key", ["reentrant_deformational_vortex",
                                      "solid_body_rotation_front"])
def test_global_mass_conserved_to_roundoff(case_key: str) -> None:
    """Upwind DG fluxes telescope on a periodic mesh: mass is exact."""
    spec = bench.CASES[case_key]
    solver = DGAdvection2D(2, 12)
    result = solver.solve(solver.project(spec.profile), spec.velocity,
                          min(spec.final_time, 0.25), cfl=0.3, record_mass=True)
    meta = result["meta"]
    assert meta["mass_drift_rel"] < bench.CRITERIA["mass_tolerance"]
    assert meta["mass_drift_max_rel_history"] < bench.CRITERIA["mass_tolerance"]
    assert meta["steps"] >= 50  # the check must span many RK4 steps


# ---------------------------------------------------------------------------
# 3. Constant field is an exact invariant of the divergence-free operator
# ---------------------------------------------------------------------------
def test_constant_field_is_preserved() -> None:
    """Consistency of volume and face terms: a uniform field must not move."""
    solver = DGAdvection2D(2, 12)
    coeff = torch.zeros(12, 12, 3, 3, dtype=torch.float64)
    coeff[:, :, 0, 0] = 1.0
    residual = solver.rhs(coeff, solid_body_rotation(OMEGA), 0.0)
    assert float(residual.abs().max()) < 1e-11
    result = solver.solve(coeff, solid_body_rotation(OMEGA), 0.25, cfl=0.3)
    assert float((result["coeff"] - coeff).abs().max()) < 1e-11


# ---------------------------------------------------------------------------
# 4.-6. Observed order of accuracy, p = 1, 2 (criterion: order >= p + 0.9)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("degree,cells", [(1, [12, 16, 24]), (2, [8, 12, 16])])
def test_order_periodic_shear_analytic(degree: int, cells) -> None:
    """Periodic sign-changing shear with an analytic solution (full domain)."""
    table = _order("shear_transport", degree, cells)
    assert table["norm"] == "global"
    assert table["observed_order"] >= degree + bench.CRITERIA["order_margin"], table


@pytest.mark.parametrize("degree,cells", [(1, [12, 16, 24]), (2, [12, 16, 24])])
def test_order_reentrant_vortex(degree: int, cells) -> None:
    """Reentrant deformational vortex: exact solution at T is the initial field."""
    table = _order("reentrant_deformational_vortex", degree, cells)
    assert table["observed_order"] >= degree + bench.CRITERIA["order_margin"], table
    assert max(table["mass_drift_rel"]) < bench.CRITERIA["mass_tolerance"]


@pytest.mark.parametrize("degree,cells", [(1, [12, 16]), (2, [8, 12])])
def test_order_solid_body_rotation_disk_norm(degree: int, cells) -> None:
    """Rotating front: order measured inside the disk that excludes the seam."""
    table = _order("solid_body_rotation_front", degree, cells)
    assert table["norm"].startswith("disk")
    assert table["observed_order"] >= degree + bench.CRITERIA["order_margin"], table


# ---------------------------------------------------------------------------
# 7. Peak preservation of the rotating ice-edge front
# ---------------------------------------------------------------------------
def test_peak_preservation_rotating_front() -> None:
    """After one exact revolution the Gaussian front peak must be retained."""
    check = bench.run_revolution_check(degree=3, cells=16, cfl=0.3, revolutions=1.0)
    assert check["peak_retention"] >= bench.CRITERIA["peak_tolerance"], check
    assert check["mass_drift_rel"] < bench.CRITERIA["mass_tolerance"]
    # the exact solution after one revolution is the initial field
    assert abs(check["revolutions"] - 1.0) < 1e-12


def test_peak_preservation_under_deformation() -> None:
    """A Gaussian bump survives one reentrant deformation cycle."""
    check = bench.run_vortex_peak_check(degree=3, cells=12, cfl=0.3)
    assert check["peak_retention"] >= bench.CRITERIA["peak_tolerance"], check


# ---------------------------------------------------------------------------
# 8. Published benchmark record contract
# ---------------------------------------------------------------------------
def test_artifact_contract() -> None:
    """The campaign artifact exists, declares the criteria and passes them."""
    if not ARTIFACT.exists():
        pytest.skip("artifacts/dg_advection_bench.json not generated yet")
    payload = json.loads(ARTIFACT.read_text())
    assert payload["benchmark"] == "dg_advection_bench"
    assert payload["campaign_node"] == "IN12-P5"
    assert payload["repo"] == "TensorDG"
    criteria = payload["criteria"]
    assert criteria["order_margin"] == 0.9
    assert criteria["order_degrees_required"] == [1, 2]
    assert criteria["mass_tolerance"] == 1e-12
    assert criteria["pytest_min_tests"] >= 5
    verdict = payload["verdict"]
    assert verdict["orders_pass"] and verdict["mass_pass"] and verdict["peak_pass"], verdict
    assert verdict["overall"] == "PASS"
    # every required degree must clear its own threshold with the reported number
    checked = [row for row in payload["order_summary"] if row["check_applied"]]
    assert {row["degree"] for row in checked} == {1, 2}
    for row in checked:
        assert row["observed_order"] >= row["required_order"] - 1e-12, row
    # mass numbers are machine precision, not just "small"
    assert payload["mass_balance"]["max_relative_drift_observed"] < 1e-12
    for cell in ("shear_transport", "solid_body_rotation_front",
                 "reentrant_deformational_vortex"):
        assert cell in payload["cases"]
    # figures are registered by name (may be absent on figure-less runs)
    for fig in payload["figures"]:
        assert (ROOT / "docs" / "figures" / fig["name"]).exists(), fig["name"]