"""Gates for the instrument census: the reference rule cannot decay silently.

Run: ``PYTHONPATH=. pytest -q tests/test_cks1998_instrument_census.py``

What is asserted (and why each one can fail):

1. **Every call site of the fixed rule is declared.**  :func:`discover` reparses the
   package; an undeclared site raises, so a new tool cannot inherit a non-convergent
   reference without someone writing down what it is for.
2. **Every declared site still exists.**  The registry cannot accumulate fictitious
   entries, and the three files this change moved off the fixed rule must stay moved.
3. **The three switched sites score against the exact instrument.**  Their gated number
   changes hands; the test re-derives it from the source and from the modules.
4. **The instrument bias is what the census claims.**  Measured here from both
   instruments, not read from the artifact: zero on a wave-free cell set, ``O(1e-2)``
   on a cell that contains a breakpoint, and not monotone in ``N`` (a converged
   reference would go to zero -- that is the whole point).
5. **The gate bound is a bound, not a hope.**  Any 3 % gate with a margin larger than the
   largest measured bias cannot flip; the one recorded tight margin (extruded 2-D P0)
   is reported with the margin it keeps, and is not declared safe by assertion.
6. **The swap does not move a verdict on a runnable configuration.**  One cheap DG run:
   the gate-margin row computed under both instruments agrees on PASS/FAIL and differs
   by less than the measured bias bound.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import shock_reference_quadrature as srq                 # noqa: E402
from benchmarks import sod_exact                                        # noqa: E402
from validation.repro_cks1998 import instrument_census as IC            # noqa: E402
from validation.repro_cks1998 import station_reference as SR            # noqa: E402

CENSUS_JSON = REPO / "validation" / "cks1998_instrument_census" / "census.json"


# --------------------------------------------------------------------------- #
# 1-2. the census is closed in both directions                                #
# --------------------------------------------------------------------------- #
def test_every_fixed_rule_call_site_is_declared():
    payload = IC.census()                      # raises UndeclaredInstrumentSite otherwise
    assert payload["calls"] > 0 and payload["files"] > 0
    assert payload["declared_without_site"] == []


def test_an_undeclared_site_is_rejected():
    """Negative control: dropping one declaration must make the census fail closed."""
    registry = [dict(e) for e in IC.REGISTRY
                if e["path"] != "tests/test_repro_cks1998.py"]
    with pytest.raises(IC.UndeclaredInstrumentSite) as info:
        IC.census(registry)
    assert "tests/test_repro_cks1998.py" in str(info.value)


def test_switched_sites_no_longer_call_the_fixed_rule():
    discovered = {s["path"] for s in IC.discover()}
    for entry in IC.SWITCHED_BY_THIS_CENSUS:
        assert entry["path"] not in discovered, entry["path"]
        source = (REPO / entry["path"]).read_text()
        assert "exact_cell_averages" in source or "sod_cell_averages" in source


def test_switched_sites_call_the_breakpoint_split_instrument():
    """The two anchor drivers must name the exact instrument, not the legacy one."""
    for path in ("validation/repro_cks1998/repro_cks1998.py",
                 "validation/repro_cks1998/sod_l1_ladder.py"):
        tree = ast.parse((REPO / path).read_text())
        imported = {alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    and (node.module or "").endswith("station_reference")
                    for alias in node.names}
        assert "exact_cell_averages" in imported, path
    gate_margin = (REPO / "validation/repro_cks1998/sod_gate_margin.py").read_text()
    assert "exact_cell_averages" in gate_margin
    assert "fixed_rule_cell_averages" in gate_margin   # kept to record the before/after


# --------------------------------------------------------------------------- #
# 3. the instrument identity of the registry itself                           #
# --------------------------------------------------------------------------- #
def test_registry_roles_are_from_the_declared_vocabulary():
    assert {e["role"] for e in IC.REGISTRY} <= {"gating", "diagnostic", "plot",
                                               "instrument_measurement"}
    assert {e["status"] for e in IC.REGISTRY} <= {"at_risk_legacy", "n/a"}


# --------------------------------------------------------------------------- #
# 4. the bias is measured, not asserted                                       #
# --------------------------------------------------------------------------- #
def test_bias_is_zero_where_there_is_no_wave_and_large_where_there_is():
    """A wave-free cell set makes both instruments coincide; a wave cell does not."""
    centres, exact_wave = SR.exact_cell_averages(400)
    _, fixed_wave = SR.fixed_rule_cell_averages(400)
    rho_exact = np.asarray(exact_wave["density"])
    rho_fixed = np.asarray(fixed_wave["density"])
    cells = IC.wave_cells(400)["cells"]
    assert cells, "the Sod snapshot must contain waves"
    mask = np.ones(400, dtype=bool)
    mask[cells] = False
    # wave-free cells: the two instruments agree to rounding (2.1e-14 measured here;
    # the residual is the rule order, 8 vs 40, on the smooth pieces -- not a bias)
    assert np.max(np.abs(rho_fixed[mask] - rho_exact[mask])) < 1e-13
    # wave cells: the fixed rule is wrong at the 1e-2 level (a property of the rule:
    # measured rho 5.5e-3, u 9.9e-3, p 2.2e-3 absolute on those four cells)
    worst = float(np.max(np.abs(rho_fixed[~mask] - rho_exact[~mask])))
    assert worst > 1e-3, worst
    _, fixed_wave_u = SR.fixed_rule_cell_averages(400)
    worst_u = float(np.max(np.abs(np.asarray(fixed_wave_u["velocity"])[~mask]
                                  - np.asarray(exact_wave["velocity"])[~mask])))
    assert worst_u > 5e-3, worst_u


def test_bias_does_not_vanish_under_refinement():
    """The legacy bias is non-monotone in N -- refinement cannot remove it."""
    rows = IC.bias_table((100, 200, 400, 800, 1600, 3200))
    shifts = [row["worst_shift_pp"] for row in rows]
    assert min(shifts) > 0.0
    assert not all(b < a for a, b in zip(shifts, shifts[1:])), shifts
    # and it stays far above machine precision at the finest mesh
    assert shifts[-1] > 1e-2, shifts


def test_exact_instrument_has_no_knob():
    """Converged in the rule order: the exact reference is unique."""
    _, base = srq.cell_averages(400, 0.2, sod_exact.LEFT, sod_exact.RIGHT)
    for order in (4, 8, 16, 40):
        _, other = srq.cell_averages(400, 0.2, sod_exact.LEFT, sod_exact.RIGHT,
                                     quadrature_order=order)
        for key in srq.FIELDS:
            assert np.allclose(base[key], other[key], rtol=0.0, atol=1e-12), (order, key)


# --------------------------------------------------------------------------- #
# 5. the bound the census is allowed to claim                                 #
# --------------------------------------------------------------------------- #
def test_gating_bound_is_computed_from_the_measured_bias():
    payload = IC.build_payload()
    bound = payload["gating_bound"]
    measured = max(row["worst_rel_l2"] for row in payload["bias"])
    assert bound["worst_bias_pp"] == pytest.approx(100.0 * measured)
    assert bound["tolerance_3pct_pp"] == 3.0
    # the tight margin this repository recorded must be reported with its arithmetic
    risk = bound["gating_sites"]["tests/test_euler_2d_p0_verification.py"]["risk"]
    assert risk["margin_pp"] == pytest.approx(0.5)
    assert risk["margin_with_exact_pp"] == pytest.approx(0.5 - 100.0 * measured)
    assert risk["at_risk"] is False            # survives ... by this much, not by faith
    assert 0.0 < risk["margin_with_exact_pp"] < 0.2


def test_gate_risk_flips_when_the_margin_is_below_the_bias():
    assert IC.gate_risk(0.5, 0.4)["at_risk"] is False
    assert IC.gate_risk(0.2, 0.4)["at_risk"] is True
    assert IC.gate_risk(None, 0.4)["verdict"] == "unknown_margin"


# --------------------------------------------------------------------------- #
# 6. one runnable configuration: the swap moves no verdict                    #
# --------------------------------------------------------------------------- #
def test_swap_is_inert_on_a_runnable_gate_row():
    """P1 N=48 (smoke-class cost): same verdict, shift bounded by the measured bias."""
    from validation.repro_cks1998.sod_gate_margin import run_case

    row = run_case(1, 48, 0.12, 1.0)
    assert set(row) >= {"rel_l2", "rel_l2_legacy", "pass", "pass_legacy",
                        "instrument_delta_pp", "verdict_flip"}
    assert row["verdict_flip"] is False
    assert row["pass"] == row["pass_legacy"]
    bound = 100.0 * max(r["worst_rel_l2"] for r in IC.bias_table((48, 100, 200, 300,
                                                                 400, 600, 800, 3200)))
    assert abs(row["instrument_delta_pp"]) <= bound + 1e-9


# --------------------------------------------------------------------------- #
# 7. the published artifact stays in step with the code                       #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not CENSUS_JSON.exists(), reason="census artifact not generated")
def test_artifact_matches_the_live_census():
    artifact = json.loads(CENSUS_JSON.read_text())
    live = IC.build_payload()
    assert artifact["calls"] == live["calls"]
    assert artifact["files"] == live["files"]
    assert [s["path"] for s in artifact["sites"]] == [s["path"] for s in live["sites"]]
    assert artifact["bias"] == json.loads(json.dumps(live["bias"], default=float))