"""Guards for the dual-instrument measurement of the two censused gate rows (m23108).

The committed artifact ``validation/cks1998_gate_row_swap/rows.json`` is the measured
answer to the question ``instrument_census`` (m22969) left ``not_run``: at the two
production 3 %-gate rows, does swapping the reference instrument (fixed 8-point Gauss
-> breakpoint-split exact) move the verdict?  These tests keep that answer honest:

* the registered rows are the *actual* production configurations (constants are read
  off the owning modules, not restated);
* the cheap row is re-computed from scratch and its verdict is stable;
* the committed JSON is complete, internally consistent, and still reproduces the
  published 2.368 %;
* every row's measured instrument displacement stays inside the row's own margin --
  the property the census could only bound is now asserted against measurement.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from validation.repro_cks1998.gate_row_swap import (
    GATE,
    ROWS,
    build_payload,
    measure_row,
    relative_l2,
    render,
)

ARTIFACT = Path("validation/cks1998_gate_row_swap/rows.json")


def _payload():
    return json.loads(ARTIFACT.read_text())


# ---------------------------------------------------------------------------
# registry: the rows must be the production rows, not convenient ones
# ---------------------------------------------------------------------------
def test_the_registered_rows_are_the_two_the_census_left_not_run():
    assert set(ROWS) == {"E2_dg_p2_n400", "extruded2d_p0_n3200"}


def test_the_gate_is_read_off_the_owning_modules():
    from benchmarks.verify_euler_2d_p0 import GATE as GATE_2D
    from benchmarks.sod_dg_benchmark import TOLERANCE as TOLERANCE_1D

    assert GATE == GATE_2D == TOLERANCE_1D


def test_the_dg_row_is_the_configuration_the_anchor_gate_uses():
    row = ROWS["E2_dg_p2_n400"]
    assert (row["ncells"], row["degree"], row["cfl"]) == (400, 2, 0.12)
    assert (row["limiter"], row["tvb_constant"], row["limiter_stage"]) == ("tvb", 1.0, "final")
    assert row["final_time"] == 0.2


def test_the_extruded_row_is_the_configuration_its_gate_uses():
    row = ROWS["extruded2d_p0_n3200"]
    assert (row["nx"], row["ny"], row["cfl"]) == (3200, 4, 0.4)
    assert tuple(row["periodic"]) == (False, True)
    assert row["final_time"] == 0.2


# ---------------------------------------------------------------------------
# the instrument machinery
# ---------------------------------------------------------------------------
def test_relative_l2_is_the_norm_the_gates_use():
    x = np.linspace(0.0, 1.0, 11)
    assert relative_l2(x, x) == 0.0
    assert relative_l2(x + 0.1 * x, x) == pytest.approx(0.1, rel=1e-12)
    with pytest.raises(ValueError):
        relative_l2(x, np.zeros_like(x))


def test_the_two_instruments_disagree_only_in_wave_cells():
    """The exact instrument is the definition; the legacy one errs where a wave sits."""
    from validation.repro_cks1998.station_reference import wave_positions

    ncells, t = 48, 0.2
    from validation.repro_cks1998.station_reference import (
        exact_cell_averages,
        fixed_rule_cell_averages,
    )

    x, _ = exact_cell_averages(ncells, t)
    _x, legacy = fixed_rule_cell_averages(ncells, t)
    _x, exact = exact_cell_averages(ncells, t)
    dx = 1.0 / ncells
    distance = np.min(np.abs(x[:, None] - wave_positions(t)[None, :]), axis=1) / dx
    err = np.abs(legacy["density"] - exact["density"])
    assert err[distance > 1.0].max() < 1e-12          # wave-free cells: exact agreement
    assert err[distance < 0.5].max() > 1e-6           # wave-bearing cells: the bias


def test_a_runnable_gate_row_keeps_its_verdict_under_either_instrument():
    """Cheap twin of the two production rows: same solvers, coarser mesh."""
    spec = dict(ROWS["E2_dg_p2_n400"])
    spec.update(ncells=64, recorded_max_percent=None)
    row = measure_row("probe_p2_n64", spec)
    assert row["verdict_exact"] == row["verdict_legacy"]
    assert abs(row["instrument_displacement_pp"]) < 1.0
    for field in ("density", "velocity", "pressure"):
        assert math.isfinite(row["fields"][field]["percent_exact"])


# ---------------------------------------------------------------------------
# the committed measurement
# ---------------------------------------------------------------------------
def test_the_artifact_exists_and_covers_both_rows():
    assert ARTIFACT.is_file(), "run the module with --json to regenerate the artifact"
    payload = _payload()
    assert {row["row"] for row in payload["rows"]} == set(ROWS)
    assert payload["verdict_flips"] == []
    assert payload["all_rows_pass_under_exact_instrument"] is True


def test_the_artifact_reproduces_the_published_dg_row():
    payload = _payload()
    row = next(r for r in payload["rows"] if r["row"] == "E2_dg_p2_n400")
    assert row["max_percent_exact"] == pytest.approx(2.368, rel=1e-3)
    assert row["verdict_exact"] == "PASS"
    assert row["reproduces_recorded"] is None or row["reproduces_recorded"] is True


def test_no_row_loses_its_margin_to_the_instrument():
    """The census could only bound this; here it is measured, row by row."""
    payload = _payload()
    for row in payload["rows"]:
        assert row["margin_pp_exact"] > 0.0, row["row"]
        assert row["instrument_displacement_pp"] < row["margin_pp_exact"], row["row"]


def test_artifact_bookkeeping_is_internally_consistent():
    payload = _payload()
    for row in payload["rows"]:
        worst = max(row["fields"][f]["rel_l2_exact"] for f in row["fields"])
        assert row["max_percent_exact"] == pytest.approx(100.0 * worst, rel=1e-12)
        assert row["margin_pp_exact"] == pytest.approx(
            100.0 * (GATE - worst), rel=1e-12
        )
        displacement = max(
            abs(row["fields"][f]["delta_pp"]) for f in row["fields"]
        )
        assert row["instrument_displacement_pp"] == pytest.approx(displacement, rel=1e-12)
        for stats in row["fields"].values():
            assert stats["delta_pp"] == pytest.approx(
                100.0 * (stats["rel_l2_exact"] - stats["rel_l2_legacy"]), abs=1e-12
            )
    assert payload["max_instrument_displacement_pp"] == pytest.approx(
        max(row["instrument_displacement_pp"] for row in payload["rows"]), rel=1e-12
    )


def test_render_covers_every_field_of_every_row():
    payload = _payload()
    table = render(payload)
    for row in payload["rows"]:
        for field in row["fields"]:
            assert f"| {row['row']} | {field} |" in table