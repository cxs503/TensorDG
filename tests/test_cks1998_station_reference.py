"""Gates for the E2 station metrics: which reference instrument, which cell.

Run: ``PYTHONPATH=. pytest -q tests/test_cks1998_station_reference.py``

The four things asserted here are the ones the anchor's metrics silently relied on:

1. the station -> cell map is a *stated* rule and does not depend on the last bit of
   ``x * ncells`` (before the fix, ``x = 0.70`` was scored on the left cell at
   ``ncells in {90, 170, 180, ...}`` and on the right cell at ``{80, 100, 120, ...}``);
2. the E2 station reference must be the breakpoint-split *exact* cell average,
   because ``benchmarks.riemann_exact.cell_averages`` documents itself as unfit for
   tolerance comparisons;
3. the swap is numerically inert on the cells the anchor actually scores (so no
   rebaselining happens behind the change);
4. the wave-free invariant of a station cell is *guarded*: it has a discriminating
   negative control, and the guard fires exactly where the invariant is violated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import riemann_exact                                    # noqa: E402
from validation.repro_cks1998 import bench_workload                    # noqa: E402
from validation.repro_cks1998 import station_reference as SR           # noqa: E402

WORKLOAD = REPO / "validation" / "repro_cks1998" / "bench_workload.py"


# --------------------------------------------------------------------------- #
# 1. the station -> cell rule                                                 #
# --------------------------------------------------------------------------- #
def test_station_table_is_the_anchor_table():
    assert SR.ANCHOR_STATIONS == bench_workload.SOD_STATIONS
    assert SR.ANCHOR_RESOLUTIONS["smoke"] == bench_workload.PROFILES["smoke"]["sod"]
    assert SR.ANCHOR_RESOLUTIONS["bench"] == bench_workload.PROFILES["bench"]["sod"]


def test_index_rule_does_not_depend_on_the_last_bit_of_the_product():
    """Every analytic face case scores the **right** cell, for every resolution.

    This is the regression the fix buys: ``int(x * ncells)`` truncates, so a product
    that comes out one ulp below the integer (``0.7 * 90 == 62.99999999999999``)
    scored the left cell, while ``0.7 * 100 == 70.0`` scored the right one.
    """
    for x in (0.30, 0.55, 0.70):
        for ncells in range(4, 1201):
            raw = x * ncells
            if abs(raw - round(raw)) > 1e-9:
                continue                                   # not a face case
            cell = SR.station_cell(ncells, x)
            assert cell["index"] == int(round(raw)), (x, ncells, cell)
            assert cell["tie_class"] == "on_face_right", (x, ncells, cell)


def test_the_old_expression_really_did_flip_and_the_new_one_does_not():
    """Concrete instances the old rule got wrong, pinned so the fix cannot regress."""
    flipped = [n for n in range(4, 700) if 0.70 * n < round(0.70 * n)
               and abs(0.70 * n - round(0.70 * n)) < 1e-9]
    assert 90 in flipped and len(flipped) >= 10, flipped
    for ncells in flipped:
        assert min(int(0.70 * ncells), ncells - 1) == int(round(0.70 * ncells)) - 1  # old
        assert SR.station_cell(ncells, 0.70)["index"] == int(round(0.70 * ncells))   # new


def test_anchor_indices_are_unchanged_by_the_stricter_rule():
    """No published metric moves: the anchor's own cells are interior or faces-right."""
    assert [SR.station_cell(48, x)["index"] for x in (0.30, 0.70, 0.55)] == [14, 33, 26]
    assert [SR.station_cell(400, x)["index"] for x in (0.30, 0.70, 0.55)] == [120, 280, 220]


# --------------------------------------------------------------------------- #
# 2./3. the reference instrument                                              #
# --------------------------------------------------------------------------- #
def test_workload_scores_the_exact_reference_not_the_fixed_gauss_rule():
    source = WORKLOAD.read_text()
    start = source.index("# ---- E2")
    end = source.index("    rel_l2 = {")
    assert start < end
    e2 = source[start:end]
    assert "exact_cell_averages(ncells, SOD_T)" in e2
    assert "from benchmarks.sod_exact import cell_averages" not in source
    assert "station_cell(ncells, xs)[" in source


def test_the_two_instruments_agree_on_every_cell_the_anchor_scores():
    """The swap is numerically inert where it is used -> no rebaselining."""
    for degree, ncells in SR.ANCHOR_RESOLUTIONS.values():
        _, exact = SR.exact_cell_averages(ncells)
        _, fixed = SR.fixed_rule_cell_averages(ncells)
        for row in SR.assert_station_cells_wave_free(ncells):
            assert row["instrument_rel_delta"] < 1e-12, (ncells, row["metric"])
            assert exact[row["field"]][row["index"]] == pytest.approx(
                fixed[row["field"]][row["index"]], rel=1e-12)


def test_the_fixed_rule_is_not_merely_a_different_number_it_is_biased_on_a_wave_cell():
    """The instrument policy is not cosmetic: on the contact cell it is off by 2.3 %."""
    ncells = 24                                    # smoke/2: the rho(0.70) cell straddles x_c
    row = SR.station_cell_diagnosis(ncells, 0.70, "density")
    assert row["contains_breakpoint"] is True
    assert row["instrument_rel_delta"] > 0.02      # 76 % of the 3 % tolerance
    # and the bias is *not* cured by refinement on that cell -- it is a rule constant
    coarse = SR.station_cell_diagnosis(24, 0.70, "density")["instrument_rel_delta"]
    fine = SR.station_cell_diagnosis(400, 0.70, "density")["instrument_rel_delta"]
    assert coarse > 2e-2 and fine < 1e-12, (coarse, fine)


# --------------------------------------------------------------------------- #
# 4. the wave-free guard                                                      #
# --------------------------------------------------------------------------- #
def test_guard_passes_at_every_resolution_the_anchor_scores():
    for name, (degree, ncells) in SR.ANCHOR_RESOLUTIONS.items():
        table = SR.assert_station_cells_wave_free(ncells)
        assert not any(row["contains_breakpoint"] for row in table)
        assert all(row["instrument_rel_delta"] < 1e-12 for row in table), name


def test_guard_discloses_the_tightest_margin_rather_than_hiding_it():
    """``rho(x=0.70)`` at the smoke resolution clears the contact by 9.6 % of a cell."""
    table = {row["metric"]: row for row in SR.assert_station_cells_wave_free(48)}
    assert table["sod_rho_x070"]["gap_to_breakpoint_dx"] == pytest.approx(0.096, abs=5e-3)
    assert min(row["gap_to_breakpoint_dx"] for row in table.values()) < 0.1


def test_guard_fires_when_a_station_cell_straddles_a_breakpoint():
    with pytest.raises(SR.StationCellContainsWave) as exc:
        SR.assert_station_cells_wave_free(24)
    assert "sod_rho_x070" in str(exc.value)


def test_guard_fires_for_a_station_placed_on_the_contact_itself():
    """Negative control: pin a station onto x_c = 0.685490524... and the guard must trip."""
    cut = float(SR.wave_positions()[2])
    with pytest.raises(SR.StationCellContainsWave):
        SR.assert_station_cells_wave_free(48, {"probe": ("density", cut)})


def test_wave_positions_match_the_exact_solver():
    """The breakpoints are the exact Riemann wave abscissae, not a hard-coded list."""
    cuts = SR.wave_positions()
    assert cuts.shape == (4,)
    assert np.all(np.diff(cuts) > 0.0)
    rho_star, u_star = riemann_exact.star_state(SR.SOD_LEFT, SR.SOD_RIGHT)
    assert u_star == pytest.approx(0.92745262, rel=1e-8)
    # the contact sits at x_diaphragm + u_star * t -- the breakpoint the guard protects
    assert 0.5 + u_star * SR.SOD_T == pytest.approx(float(cuts[2]), rel=1e-12)