"""Exact invariants of the 2-D shallow-water DG solver (verification debt).

Companion of ``benchmarks/verify_swe_2d_symmetry.py`` (report:
``docs/reports/tensordg-swe-2d-symmetry.md``).  These are *algebraic* invariants
of the discretisation, so they are asserted at round-off scale rather than at a
truncation-error tolerance:

* C1 a y-invariant problem on a 2-D Cartesian mesh reproduces the 1-D solver;
* C2 such a state acquires no y-variation at all (regression guard for the m6386
  cross-mode defect, which made every unsteady 2-D flow blow up);
* C3 a radially symmetric problem is invariant under index transposition
  ``(nx, ny) -> (ny, nx)`` together with ``hu <-> hv``;
* C4 still water stays still when the bed is representable by the piecewise
  constant cell gradients (linear bed, kink exactly on a face).

The configurations are deliberately small so that the file costs ~15 s.
"""

import torch

from benchmarks.verify_swe_2d_symmetry import (
    _recon_1d,
    _recon_2d,
    case_cross_solver,
    case_transposition,
    case_well_balanced,
)
from tensordg.swe_1d import solve_swe_1d
from tensordg.swe_2d import solve_swe_2d

TOL = 1.0e-11


def test_y_invariant_2d_reproduces_1d_solver():
    res = case_cross_solver(nx=8, ny=4, degree=1, final_time=0.01)
    assert res["pass_c1"], res
    # one full 1-D column of the 2-D run, node by node, matches the 1-D solver
    assert res["h_rel_diff_2d_vs_1d"] <= TOL
    assert res["hu_rel_diff_2d_vs_1d"] <= TOL


def test_y_invariant_state_has_no_y_variation():
    res = case_cross_solver(nx=8, ny=4, degree=1, final_time=0.01)
    assert res["pass_c2"], res
    assert res["y_spread_quad"] <= TOL and res["y_spread_cells"] <= TOL


def test_transposition_symmetry_of_radial_problem():
    res = case_transposition((8, 12), degree=1, final_time=0.02)
    assert res["pass_c3"], res
    # momentum components must be exchanged, not compared in place
    assert res["rel_diff"]["hu"] <= TOL and res["rel_diff"]["hv"] <= TOL


def test_lake_at_rest_linear_bed_is_stationary():
    res = case_well_balanced("linear", n=8, degree=1, final_time=0.02)
    assert res["pass_c4"], res


def test_lake_at_rest_kinked_bed_is_stationary():
    """Kink on a cell face -> the per-cell gradient is exact -> still water."""
    res = case_well_balanced("kinked", n=8, degree=1, final_time=0.02)
    assert res["pass_c4"], res


def test_curved_bed_is_not_exactly_well_balanced_but_second_order():
    """Documented limitation (report section 5): a curved bed is *not* exactly
    well balanced; the still-water residual decays at the bed-representation
    order.  Pinned here so the limitation cannot silently disappear (e.g. by a
    hidden special case) or be forgotten.
    """
    coarse = case_well_balanced("smooth", n=8, degree=1, final_time=0.02,
                                gating=False)
    fine = case_well_balanced("smooth", n=16, degree=1, final_time=0.02,
                              gating=False)
    assert coarse["rel_h_error"] > 1.0e-4          # not machine precision
    rate = torch.log2(torch.tensor(coarse["rel_h_error"] / fine["rel_h_error"]))
    # pre-asymptotic on these tiny meshes (measured 0.77); the asymptotic rate
    # 1.76 (2-D) / 1.79 (1-D) is what the artifact records at 16 -> 32 cells.
    assert float(rate) > 0.5, (coarse["rel_h_error"], fine["rel_h_error"])