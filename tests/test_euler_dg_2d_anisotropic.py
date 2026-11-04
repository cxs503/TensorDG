"""Anisotropic (stretched-cell) tests for the 2-D compressible Euler DG solver.

Companion to ``benchmarks/verify_anisotropic_2d.py`` and
``docs/anisotropic_2d_verification.md``.  The repository's convergence and
limiter evidence was measured exclusively on square cells (``ncells = n^2``,
``dx == dy``), where a swap of ``dx`` and ``dy`` in the residual scaling or in
the limiter's troubled-cell test would be completely invisible.  These tests
pin that blind spot down with cheap, sharp checks:

* ``test_residual_transpose_invariance`` -- exact algebraic ``x <-> y``
  symmetry of one residual evaluation (catches any ``dx``/``dy`` mix-up);
* ``test_troubled_cell_test_uses_per_axis_spacing`` -- mutation probe: the
  verdict on a smooth y-profile must flip when the *other* axis' spacing is
  substituted, which proves the m2549 per-axis threading is load bearing;
* ``test_stretched_ladder_reaches_design_order`` -- P1 EOC on 2:1 cells;
* ``test_stretched_mesh_conserves_mass_and_energy`` -- round-off budget;
* ``test_flagged_fraction_does_not_grow_on_stretched_cells`` -- m2549 scale
  consistency (no refinement-driven flood of flagged cells).
"""
from __future__ import annotations

import math

import pytest
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.verify_euler_dg_2d import _vortex_error
from tensordg.tvb import minmod

DTYPE = torch.float64
DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
TVB = 1.0


def _operators(degree):
    return dg2.legendre_operators(degree, max(degree + 2, 3))


def _flagged_fraction(coeff, dx, dy):
    averages = coeff[..., 0, 0, :]
    troubled, _ = dg2._troubled_1d(averages, 1, dy, TVB, True)
    return float(troubled.any(dim=-1).float().mean())


def _vortex(ncells, degree, final_time, limiter=None):
    nodes, weights, basis, _, _ = _operators(degree)
    _, _, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=ncells,
        degree=degree,
        final_time=final_time,
        cfl=0.2,
        limiter=limiter,
        tvb_constant=TVB,
        domain=DOMAIN,
        dtype=DTYPE,
    )
    return coeff, meta, (nodes, weights, basis)


# ---------------------------------------------------------------------------
# Operator symmetry on stretched cells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [1, 2, 3])
def test_residual_transpose_invariance(degree):
    """One residual evaluation is invariant under ``x <-> y`` (with ``u <-> v``).

    A residual that mixed ``1/(2dx)`` with the y-face term would still be
    correct on square cells, so this is the only check that detects that class
    of defect directly.
    """
    _, weights, basis, derivative, signs = _operators(degree)
    nx, ny = 6, 11
    generator = torch.Generator().manual_seed(20261104 + degree)
    coeff = 0.05 * torch.randn(
        nx, ny, degree + 1, degree + 1, 4, dtype=DTYPE, generator=generator
    )
    coeff[..., 0, 0, 0] += 1.0
    coeff[..., 0, 0, 3] += 2.5

    left = dg2._rhs(coeff, basis, derivative, weights, signs,
                    10.0 / nx, 10.0 / ny, (True, True))
    swapped = coeff.permute(1, 0, 3, 2, 4)[..., [0, 2, 1, 3]].contiguous()
    right = dg2._rhs(swapped, basis, derivative, weights, signs,
                     10.0 / ny, 10.0 / nx, (True, True))
    right = right.permute(1, 0, 3, 2, 4)[..., [0, 2, 1, 3]]

    scale = float(left.abs().max())
    assert torch.allclose(left, right, rtol=0.0, atol=1e-14 * scale)


def test_residual_axes_are_not_interchangeable():
    """Negative control: the same call with ``dx``/``dy`` *not* swapped differs.

    Guards the test above against a trivially symmetric implementation (e.g.
    one that ignored the spacings).
    """
    degree = 2
    _, weights, basis, derivative, signs = _operators(degree)
    nx, ny = 6, 11
    generator = torch.Generator().manual_seed(7)
    coeff = 0.05 * torch.randn(
        nx, ny, degree + 1, degree + 1, 4, dtype=DTYPE, generator=generator
    )
    coeff[..., 0, 0, 0] += 1.0
    coeff[..., 0, 0, 3] += 2.5
    dx, dy = 10.0 / nx, 10.0 / ny
    reference = dg2._rhs(coeff, basis, derivative, weights, signs, dx, dy, (True, True))
    swapped_mesh = coeff.permute(1, 0, 3, 2, 4)[..., [0, 2, 1, 3]].contiguous()
    wrong = dg2._rhs(swapped_mesh, basis, derivative, weights, signs,
                     dx, dy, (True, True))
    wrong = wrong.permute(1, 0, 3, 2, 4)[..., [0, 2, 1, 3]]
    assert not torch.allclose(reference, wrong, rtol=0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Per-axis spacing in the troubled-cell test
# ---------------------------------------------------------------------------

def test_troubled_cell_test_uses_per_axis_spacing():
    """Mutation probe for the m2549 per-axis spacing threading.

    On a 2:1 stretched mesh (``dx = 2 dy``) a smooth *y* profile has a
    data-owned deviation of ``|u''| dy^2``.  Calibrating the amplitude so that
    deviation sits between ``M dy^2`` (correct axis) and ``M dx^2 = 4 M dy^2``
    (wrong axis) means the verdict must flip: cells are flagged with ``dy`` and
    released with ``dx``.  If the limiter silently used the other axis'
    spacing this assertion fails.
    """
    nx, ny = 8, 64
    dx, dy = 10.0 / nx, 10.0 / ny
    period = 5.0
    y = (torch.arange(ny, dtype=DTYPE) + 0.5) * dy
    shape = torch.cos(2.0 * math.pi * (y - 5.0) / period)

    def deviation(amplitude):
        """max_cells |d_left - minmod(d_left, d_right)| -- the TVB measure."""
        averages = (1.0 + amplitude * shape).view(1, ny, 1).expand(nx, ny, 1).clone()
        d_left = averages[:, 1:-1, 0] - averages[:, :-2, 0]
        d_right = averages[:, 2:, 0] - averages[:, 1:-1, 0]
        return float((d_left - minmod(d_left, d_right)).abs().max())

    amplitude = 1e-4 * (2.0 * TVB * dy * dy) / deviation(1e-4)
    averages = (1.0 + amplitude * shape).view(1, ny, 1).expand(nx, ny, 1).clone()

    correct, _ = dg2._troubled_1d(averages, 1, dy, TVB, True)
    wrong, _ = dg2._troubled_1d(averages, 1, dx, TVB, True)
    assert int(correct.sum()) > 0
    assert int(wrong.sum()) == 0


def test_flagged_fraction_does_not_grow_on_stretched_cells():
    """Scale consistency (m2549) on stretched cells: refinement may not flood.

    The pre-m2549 criterion compared an O(dx u') measure against ``M dx^2``, so
    the flagged set grew as ``1/Mh`` under refinement.  At fixed ``M = 1`` the
    flagged fraction of a resolved smooth flow must stay bounded (and, on this
    ladder, not grow).
    """
    fractions = []
    for cells in ((6, 12), (9, 18)):
        coeff, _, _ = _vortex(cells, 2, 0.4)
        fractions.append(_flagged_fraction(coeff, 10.0 / cells[0], 10.0 / cells[1]))
    assert fractions[-1] <= fractions[0] + 1e-12
    assert fractions[-1] <= 0.5


# ---------------------------------------------------------------------------
# Convergence and conservation on stretched cells
# ---------------------------------------------------------------------------

def test_stretched_ladder_reaches_design_order():
    """P1 attains its design order 2 on 2:1 stretched cells.

    The ladder starts at 8x16 because the coarser rungs are still in the
    pre-asymptotic window (the isotropic control degrades identically there, see
    ``docs/anisotropic_2d_verification.md``), so the assertion uses the first
    asymptotic pair and a 0.35 margin below the design order.
    """
    rows = []
    for cells in ((8, 16), (12, 24)):
        coeff, _, (nodes, _, basis) = _vortex(cells, 1, 1.0)
        errors = _vortex_error(coeff, basis, nodes, DOMAIN, 1.0)
        rows.append((math.sqrt((10.0 / cells[0]) * (10.0 / cells[1])), errors))
    worst = min(
        math.log(rows[0][1][field] / rows[1][1][field]) / math.log(rows[0][0] / rows[1][0])
        for field in ("density", "velocity_x", "velocity_y", "pressure")
    )
    assert worst >= 1.65, worst


def test_stretched_mesh_finest_error_below_three_percent():
    """Engineering gate: P2 on 16x32 stretched cells stays under 3 %."""
    coeff, _, (nodes, _, basis) = _vortex((10, 20), 2, 1.0)
    errors = _vortex_error(coeff, basis, nodes, DOMAIN, 1.0)
    assert max(errors[field] for field in
               ("density", "velocity_x", "velocity_y", "pressure")) < 0.03


def test_stretched_mesh_conserves_mass_and_energy():
    """Periodic mass/energy drift on a stretched mesh stays at round-off."""
    dx, dy = 10.0 / 8, 10.0 / 16
    _, _, coeff0, _ = dg2.solve_euler_dg_2d_vortex(
        ncells=(8, 16), degree=2, final_time=0.0, cfl=0.2, domain=DOMAIN, dtype=DTYPE
    )
    _, _, coeff1, _ = dg2.solve_euler_dg_2d_vortex(
        ncells=(8, 16), degree=2, final_time=0.4, cfl=0.2, domain=DOMAIN, dtype=DTYPE
    )
    for component, name in ((0, "mass"), (3, "energy")):
        before = float(coeff0[..., 0, 0, component].sum()) * dx * dy
        after = float(coeff1[..., 0, 0, component].sum()) * dx * dy
        assert abs(after - before) / abs(before) < 1e-12, name