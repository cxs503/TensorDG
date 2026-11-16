"""E2 station metrics of the CKS1998 anchor: which reference instrument, and which cell?

Two questions that the anchor's platform metrics never answered explicitly, and that
:mod:`benchmarks.riemann_exact` answers *against* the way they were being used.

1. **The reference instrument.**  ``benchmarks.riemann_exact.cell_averages`` evaluates
   the exact Riemann solution on a *fixed* 8-point Gauss rule.  Its own docstring
   says so and says more: a cell that contains a wave is integrated inexactly, the
   error has the size ``rho_q * jump`` with ``rho_q`` a property of the rule, it does
   not shrink under refinement, and the breakpoint-split
   ``benchmarks.shock_reference_quadrature.cell_averages`` is the number to use
   "when the number is compared against a tolerance".  The anchor's E2 station
   metrics are compared against a 3 % tolerance, so the fixed-rule reference is the
   wrong instrument *by the repository's own documented policy* -- it just happens to
   be harmless at the two resolutions the anchor uses, because the station cells
   there are (barely) wave-free.  This module makes that a property to be **checked**
   rather than a coincidence to be relied on.

2. **The station cell.**  ``bench_workload`` picks ``idx = min(int(x * ncells), ncells - 1)``
   and reports the metric value as ``got[field][idx]`` against ``ref[field][idx]``.
   Nothing records which cell that is, how far it sits from the nearest exact
   breakpoint, or whether it contains one.  At ``ncells = 48`` the ``sod_rho_x070``
   cell clears the contact discontinuity by **0.096 dx``; at ``ncells = 24`` it does
   not clear it at all, and the fixed-rule reference is then biased by 2.3 %.

Both facts are cheap to compute and cheap to guard, so they are computed and guarded
here.  Nothing in this module changes a tolerance, a station position or a grid: the
reference swap is numerically inert on the anchor's actual cells (measured difference
<= 2.2e-14 relative), which is the point -- the *policy* was wrong, not the numbers.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np

from benchmarks import riemann_exact
from benchmarks import sod_exact
from benchmarks.shock_reference_quadrature import breakpoints
from benchmarks.shock_reference_quadrature import cell_averages as _exact_cell_averages

#: Sod test bed, single source: ``benchmarks.sod_exact`` (same numbers as the workload).
SOD_LEFT = sod_exact.LEFT
SOD_RIGHT = sod_exact.RIGHT
SOD_T = 0.2

#: Station table of the anchor (field, x).  Duplicated on purpose from
#: ``bench_workload.SOD_STATIONS`` so this module can be audited without importing the
#: workload; a test pins the two tables equal.
ANCHOR_STATIONS: dict[str, tuple[str, float]] = {
    "sod_rho_x030": ("density", 0.30),
    "sod_rho_x070": ("density", 0.70),
    "sod_vel_x055": ("velocity", 0.55),
}

#: The resolutions the anchor actually scores (``bench_workload.PROFILES[*]["sod"]``).
ANCHOR_RESOLUTIONS: dict[str, tuple[int, int]] = {"smoke": (1, 48), "bench": (2, 400)}


#: Relative tolerance for "the station lies exactly on a cell face".  ``x * ncells``
#: is evaluated in binary floating point, so a station that is a face analytically can
#: land one ulp either side of the integer: for ``x = 0.70``, 166 of the 200 resolutions
#: in ``4 <= ncells <= 2000`` that put the station on a face produce ``x * ncells``
#: *above* the integer and are scored on the right cell, while
#: ``ncells in {90, 170, 180, 330, 340, 350, 360, 650, ...}`` produce it *below* and are
#: scored on the left cell -- 34 resolutions with no convention behind the difference.
#: The rule below is therefore stated once, explicitly: **an exact face scores the
#: right-hand cell**, and :data:`TIE_EPS` is what makes "exact" mean the analytic face
#: and not the last bit of the product.
TIE_EPS = 1e-9


class StationCellContainsWave(ValueError):
    """A station cell contains an exact breakpoint, so a fixed-rule reference is biased."""


def exact_cell_averages(ncells: int, final_time: float = SOD_T):
    """Breakpoint-split exact cell averages -- the reference *definition*."""
    return _exact_cell_averages(ncells, final_time, SOD_LEFT, SOD_RIGHT)


def fixed_rule_cell_averages(ncells: int, final_time: float = SOD_T, quadrature_order: int = 8):
    """The fixed Gauss rule the anchor used to use (kept only to measure its bias)."""
    return riemann_exact.cell_averages(
        ncells, final_time, SOD_LEFT, SOD_RIGHT, diaphragm=0.5,
        domain=(0.0, 1.0), quadrature_order=quadrature_order,
    )


def wave_positions(final_time: float = SOD_T) -> np.ndarray:
    """Exact breakpoint abscissae (rarefaction head/tail, contact, shock)."""
    return np.asarray(
        breakpoints(SOD_LEFT, SOD_RIGHT, final_time, riemann_exact.GAMMA, 0.5, (0.0, 1.0)),
        dtype=float,
    )


def station_cell(ncells: int, x: float) -> dict:
    """The cell the workload scores, and the arithmetic that chose it.

    ``tie`` is ``x * ncells - floor(x * ncells)``: the position of the station inside
    its cell, in units of a cell (``0`` = on the left face, ``1`` = on the right
    face).  The rule is **stated, not inherited from ``int()``**: an exact face scores
    the right-hand cell, and the exactness test carries :data:`TIE_EPS` so that the
    choice does not depend on the last bit of ``x * ncells``.  ``tie_class`` discloses
    which case a station fell into.
    """
    if ncells < 1:
        raise ValueError("ncells must be positive")
    raw = float(x) * ncells
    index = min(int(math.floor(raw + TIE_EPS * max(1.0, abs(raw)))), ncells - 1)
    fractional = raw - math.floor(raw)
    on_face = abs(fractional) <= TIE_EPS * max(1.0, abs(raw)) or abs(fractional - 1.0) <= TIE_EPS
    dx = 1.0 / ncells
    return {
        "index": int(index),
        "raw": raw,
        "tie": float(fractional),
        "tie_class": "on_face_right" if on_face else "interior",
        "x_left": index * dx,
        "x_right": (index + 1) * dx,
        "x_centre": (index + 0.5) * dx,
        "ncells": int(ncells),
        "station_x": float(x),
    }


def station_cell_diagnosis(ncells: int, x: float, field: str,
                           final_time: float = SOD_T) -> dict:
    """Everything needed to decide whether scoring this station is *valid*.

    Returns the coded cell, the distance from that cell to the nearest exact
    breakpoint (in cells), whether the cell contains one, the exact reference, the
    fixed-rule reference, and the relative bias of the latter.
    """
    cell = station_cell(ncells, x)
    dx = 1.0 / ncells
    cuts = wave_positions(final_time)
    inside = cuts[(cuts > cell["x_left"]) & (cuts < cell["x_right"])]
    gap = min(np.min(np.abs(cuts - cell["x_left"])), np.min(np.abs(cuts - cell["x_right"])))
    _, ref_exact = exact_cell_averages(ncells, final_time)
    _, ref_fixed = fixed_rule_cell_averages(ncells, final_time)
    value_exact = float(ref_exact[field][cell["index"]])
    value_fixed = float(ref_fixed[field][cell["index"]])
    cell.update({
        "field": field,
        "contains_breakpoint": bool(inside.size),
        "breakpoints_inside": [float(c) for c in inside],
        "gap_to_breakpoint_dx": float(gap / dx),
        "ref_exact": value_exact,
        "ref_fixed_rule": value_fixed,
        "instrument_rel_delta": abs(value_fixed - value_exact) / abs(value_exact),
    })
    return cell


def station_table(ncells: int, stations: Mapping[str, tuple[str, float]] | None = None,
                  final_time: float = SOD_T) -> list[dict]:
    """Diagnosis of every anchor station at one resolution."""
    table = []
    for name, (field, x) in (stations or ANCHOR_STATIONS).items():
        row = station_cell_diagnosis(ncells, x, field, final_time)
        row["metric"] = name
        table.append(row)
    return table


def assert_station_cells_wave_free(ncells: int, stations: Mapping[str, tuple[str, float]] | None = None,
                                   final_time: float = SOD_T) -> list[dict]:
    """Raise if any station cell contains a breakpoint; return the diagnosis table.

    The guard exists because the failure is silent and large: at ``ncells = 24`` the
    ``rho(x = 0.70)`` cell straddles the contact and a fixed-rule reference is biased
    by 2.3 %, i.e. 76 % of the 3 % tolerance, with no visible symptom other than a
    metric that looks slightly worse.
    """
    table = station_table(ncells, stations, final_time)
    bad = [row for row in table if row["contains_breakpoint"]]
    if bad:
        detail = ", ".join(
            f"{row['metric']} cell=[{row['x_left']:.6g},{row['x_right']:.6g}] "
            f"contains {row['breakpoints_inside']} (fixed-rule bias {row['instrument_rel_delta']:.3e})"
            for row in bad
        )
        raise StationCellContainsWave(
            f"ncells={ncells}: station cell contains an exact breakpoint -- {detail}. "
            "Move the station, or use the breakpoint-split reference; do not score a "
            "fixed-rule cell average that straddles a wave."
        )
    return table


def sweeps_breakpoint(ncells_values: Iterable[int], x: float) -> list[dict]:
    """Show, per resolution, how far the station is from the nearest breakpoint."""
    out = []
    for ncells in ncells_values:
        cell = station_cell(ncells, x)
        cuts = wave_positions()
        dx = 1.0 / ncells
        gap = min(np.min(np.abs(cuts - cell["x_left"])), np.min(np.abs(cuts - cell["x_right"])))
        cell.update({"gap_to_breakpoint_dx": float(gap / dx),
                     "contains_breakpoint": bool(((cuts > cell["x_left"]) & (cuts < cell["x_right"])).any())})
        out.append(cell)
    return out