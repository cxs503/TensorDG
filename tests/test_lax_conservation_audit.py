"""Conservation-budget verification for the ``CASE-LAX-DG`` case pool.

The case pool's existing gates grade *accuracy* against the exact Riemann
solution.  These tests pin a different, accuracy-independent property: the
global conservation law integrated in closed form against the end fluxes.

Four things are asserted, and the third/fourth are the pair that makes the gate
falsifiable in both directions:

1. **estimator self-test** -- the closed-form budget is the analytic
   ``t*(F_L - F_R)`` (and it is *not* zero for the Lax case, whose left state
   moves);
2. **reference validation** -- the exact solution, integrated piecewise with
   Gauss quadrature and without any solver call, satisfies that same budget, so
   a defect in the closed-form Riemann algebra would show up here;
3. **solver closes the budget** -- mass, momentum and energy bookkeeping of the
   DG run matches the closed-form prediction to the pre-registered tolerance;
4. **the naive invariant fails** -- "``int U`` is constant" is violated by an
   O(1) amount, so a tolerance that passed it would be blind to the very
   bookkeeping error under audit;
5. **artifact honesty** -- the committed ``metrics.json`` is re-adjudicated from
   its own numbers, so a hand-edited verdict cannot pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import lax_conservation_audit as audit
from benchmarks.lax_case_pool import FINAL_TIME, GAMMA, LEFT, RIGHT
from tests._conservation_tools import boundary_increment, conserved_totals, physical_flux

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / audit.OUT / "metrics.json"


# --------------------------------------------------------------- self-tests
def test_budget_is_the_analytic_end_flux_increment():
    """The budget must be ``t*(F_L - F_R)``, not an assumed zero."""
    flux_left = physical_flux(LEFT, GAMMA)
    flux_right = physical_flux(RIGHT, GAMMA)
    increment = boundary_increment(LEFT, RIGHT, FINAL_TIME, GAMMA)
    for index, name in enumerate(audit.FIELDS):
        expected = FINAL_TIME * (flux_left[index] - flux_right[index])
        assert increment[name] == pytest.approx(expected, rel=1e-14, abs=1e-14)
    # The Lax left state moves, so the naive "no change" budget is wrong here.
    assert increment["mass"] > 1e-2
    assert increment["momentum"] > 1e-1
    assert increment["energy"] > 1e-1


def test_initial_cell_averages_are_exact_for_a_straddled_diaphragm():
    """A diaphragm strictly inside one cell must give the fraction-weighted mean."""
    ncells, diaphragm = 64, 0.51
    averages = audit.initial_cell_averages(ncells, diaphragm=diaphragm)
    # cell 32 spans [0.5, 0.515625]; 0.01/0.015625 = 0.64 of it is the left state
    fraction = (diaphragm - 0.5) / (1.0 / ncells)
    assert fraction == pytest.approx(0.64)
    mixed = averages[32]
    left = audit.conserved(*LEFT)
    right = audit.conserved(*RIGHT)
    assert np.allclose(mixed, fraction * left + (1.0 - fraction) * right, rtol=1e-14)
    # a diaphragm on a face reproduces the piecewise-constant totals exactly
    face = audit.initial_cell_averages(100)
    assert audit.start_totals(face)["mass"] == pytest.approx(0.4725, rel=1e-15)


def test_conserved_totals_scales_with_the_cell_width():
    """Guards the integration weight used on every leg."""
    states = np.tile(np.array([[1.0, 0.5, 2.0]]), (10, 1))
    totals = conserved_totals(states, 0.1)
    assert totals == pytest.approx(
        {"mass": 1.0, "momentum": 0.5, "energy": 2.0}, rel=1e-15
    )


# ------------------------------------------------------- reference validation
def test_exact_reference_obeys_the_global_budget():
    """The closed-form Riemann solution must satisfy the same identity.

    No solver is involved: the exact field is integrated piecewise on the five
    analytic pieces, so this is an independent check of the reference algebra
    that every case-pool accuracy gate is graded against.
    """
    residual = audit.reference_residual()
    for name in audit.FIELDS:
        assert residual["relative"][name] < 1e-12, (name, residual["relative"])
    # the reference check is not vacuous: the increments are O(1e-1)
    assert abs(residual["increment"]["mass"]) > 1e-2


# --------------------------------------------------------- solver behaviour
@pytest.mark.parametrize("limiter", [None, "tvb"])
def test_solver_closes_the_budget_on_the_stress_leg(limiter):
    """Coarsest ladder leg (largest round-off residual) must still close."""
    entry = audit.measure_leg(50, 1, limiter=limiter)
    for name in audit.FIELDS:
        assert entry["budget_residual"][name] < audit.TOLERANCE, (
            name,
            entry["budget_residual"],
        )


def test_naive_invariant_is_violated_by_the_same_run():
    """The falsifiability pair: the wrong invariant fails, the budget passes."""
    entry = audit.measure_leg(50, 1, limiter=None)
    assert entry["naive_residual"]["mass"] > 1e-2
    assert entry["naive_residual"]["momentum"] > 1.0
    assert entry["naive_residual"]["energy"] > 1e-1
    assert max(entry["budget_residual"].values()) < audit.TOLERANCE
    reported = audit.gate([entry])
    assert reported["verdict"] == "PASS"
    assert reported["detection_ratio"] > 1e4


def test_sensitivity_resolves_a_one_percent_flux_defect():
    """A 1 % end-flux bookkeeping defect must be far above the tolerance."""
    entry = audit.measure_leg(50, 1, limiter=None)
    probe = audit.sensitivity(entry, relative_flux_error=1.0e-2)
    assert probe["detection_margin"] > 100.0


# -------------------------------------------------------------- artifact
@pytest.mark.skipif(not ARTIFACT.exists(), reason="conservation artifact not committed")
def test_committed_artifact_is_re_adjudicated_from_its_own_numbers():
    """A hand-edited verdict cannot pass: recompute it from the rows."""
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    gate = report["gate"]
    assert gate["tolerance"] == pytest.approx(audit.TOLERANCE)
    rows = gate["rows"]
    assert rows, "artifact carries no measured rows"
    worst = max(row["budget_residual"] for row in rows)
    assert gate["worst"]["value"] == pytest.approx(worst, rel=1e-12)
    expected = "PASS" if all(row["budget_residual"] <= gate["tolerance"] for row in rows) else "FAIL"
    assert gate["verdict"] == expected
    # the naive arm must stay far above the graded tolerance in the artifact too
    assert max(row["naive_residual"] for row in rows) > 1e-2
    # every ladder leg / limiter / positivity arm is present
    legs = {(row["leg"], row["limiter"], row["positivity"]) for row in rows}
    assert len(legs) == len(report["entries"])
    # the reference check is stored and machine-precise
    assert report["exact_reference"]["worst"] < 1e-12


@pytest.mark.skipif(not ARTIFACT.exists(), reason="conservation artifact not committed")
def test_committed_artifact_records_the_production_leg():
    """The ledger production leg must be present, and its numbers must be read honestly.

    The registered tolerance is round-off limited; the production leg is *not*
    required to sit under it (it is reported as FAIL if it does not, see the
    artifact).  What is required is that the leg is present and that its residual
    stays in the round-off band, i.e. orders of magnitude below the naive arm and
    below the 3 % ledger accuracy bar.
    """
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    production = [
        entry
        for entry in report["entries"]
        if (entry["cells"], entry["degree"]) == (400, 3)
    ]
    assert production, "the ledger production leg p3.N400 is missing from the artifact"
    for entry in production:
        worst = max(entry["budget_residual"].values())
        assert worst < 1e-3, worst
        assert worst * 1e3 < max(entry["naive_residual"].values())