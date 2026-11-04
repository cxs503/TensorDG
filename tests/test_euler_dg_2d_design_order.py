"""Design-order pin for the 2-D high-order compressible Euler DG solver.

Before this file the repository only had *one* convergence interval for
``tensordg/euler_dg_2d.py`` (P2, 16^2 -> 24^2, EOC 2.95 in
``validation/euler_dg_2d/evidence.json``), which cannot distinguish the design
order ``p + 1`` from super-convergence or from a pre-asymptotic window.  The
full ladder (P1/P2/P3, four meshes each) lives in
``benchmarks/design_order_euler_dg_2d.py`` -> ``validation/euler_dg_2d/
design_order.json``; the tests here pin the cheap corner of that sweep so a
future change to the residual, the initial projection or the time stepping that
degrades the order turns the suite red without needing the whole study.

Reference: the isentropic vortex is an *exact* solution of the 2-D Euler
equations (no manufactured source), so the comparison is against analysis.
"""
import math

import pytest

import tensordg.euler_dg_2d as dg2
from benchmarks.verify_euler_dg_2d import _vortex_error

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))


def _vortex(ncells, degree, final_time=1.0):
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
        ncells=ncells, degree=degree, final_time=final_time
    )
    return _vortex_error(coeff, basis, nodes, DOMAIN, final_time), metadata


@pytest.mark.parametrize("field", ["density", "pressure"])
def test_degree_one_reaches_the_design_order(field):
    """P1 must converge at (close to) h^2 on the smooth analytic vortex."""
    coarse, _ = _vortex(8, 1)
    fine, metadata = _vortex(16, 1)
    assert metadata["steps"] > 0
    eoc = math.log(coarse[field] / fine[field]) / math.log(2.0)
    assert 1.6 < eoc < 2.3, (field, eoc, coarse[field], fine[field])


def test_degree_one_meets_the_three_percent_gate():
    """The 16^2 P1 vortex stays far inside the 3 % engineering gate."""
    errors, _ = _vortex(16, 1)
    worst = max(errors.values())
    assert worst < 0.03, errors
    # regression pin on the value reported in
    # docs/reports/tensordg-euler-dg-2d-design-order.md
    assert errors["density"] == pytest.approx(3.8091e-03, rel=0.05)


def test_degree_two_beats_degree_one_on_the_same_mesh():
    """p-refinement must pay: P2 on 8^2 is more accurate than P1 on 16^2."""
    p1, _ = _vortex(16, 1)
    p2, _ = _vortex(8, 2)
    assert p2["density"] < p1["density"], (p2["density"], p1["density"])


def test_high_order_degrees_use_the_full_modal_basis():
    """A smooth field must populate every mode, independently of the degree."""
    for degree in (2, 3):
        _, _, coeff, _ = dg2.solve_euler_dg_2d_vortex(
            ncells=4, degree=degree, final_time=0.0
        )
        for axis in (2, 3):
            slice_ = coeff[..., 1:, :, :] if axis == 2 else coeff[..., :, 1:, :]
            assert float(slice_.abs().max()) > 1e-4, (degree, axis)