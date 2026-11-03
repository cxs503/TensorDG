"""Regression tests for the limiter *application policy* and TVB indicator.

Guards the two defects measured in ``docs/limiter_policy_verification.md``:

1. the TVB troubled-cell test must be scale-consistent, i.e. it compares the
   distance from a cell average to the neighbouring extrema -- an O(dx^2 u'')
   quantity -- against ``M dx^2``.  The historic test fired in every concave
   stretch of a smooth profile and therefore flagged ~half of any smooth wave;
2. the limiter must not destroy the design order ``p + 1 = 3`` of the P2
   scheme on smooth periodic data, and it must not leak conserved quantities.
"""
from __future__ import annotations

import numpy as np
import torch

from tensordg.euler_dg_1d import solve_euler_dg_1d, solve_periodic_euler_dg_1d
from tensordg.limiters import _troubled_cell_indicator

SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)
AMPLITUDE = 0.2
FINAL_TIME = 0.1


def _wave_state(x):
    rho = 1.0 + AMPLITUDE * torch.sin(2.0 * np.pi * x)
    return (
        rho,
        torch.ones_like(x),
        torch.ones_like(x),
    )


def _wave_error(x, coeff, time=0.1, order=8):
    values = coeff.detach().cpu().numpy()
    centres = x.detach().cpu().numpy()
    cells, modes, _ = values.shape
    dx = (centres[-1] - centres[0]) / (cells - 1)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    phi = np.stack(
        [np.polynomial.legendre.legval(nodes, np.pad([1.0], (k, 0)))
         for k in range(modes)], axis=1)
    numerical = np.einsum("cmv,qm->cqv", values, phi)[..., 0]
    points = centres[:, None] + 0.5 * dx * nodes[None, :]
    exact = 1.0 + AMPLITUDE * np.sin(2.0 * np.pi * (points - time))
    error2 = 0.5 * dx * np.sum(weights[None, :] * (numerical - exact) ** 2)
    norm2 = 0.5 * dx * np.sum(weights[None, :] * exact ** 2)
    return float(100.0 * np.sqrt(error2 / norm2))


def test_tvb_indicator_is_mesh_independent_on_a_smooth_wave():
    """The TVB constant must have O(dx^2) semantics, i.e. be mesh-independent.

    This wave has |u''| dx^2 = 3.95 dx^2, so ``M = 10`` must leave every cell
    alone at *every* resolution.  The historic test compared interface jumps
    O(dx u') against ``M dx^2``, which pushed the required M to O(1/dx): at
    M = 100 it flagged 0, 0, 3 and 53 of 20/40/80/160 cells respectively, so no
    single M was inert on a mesh sequence.
    """
    for cells in (20, 40, 80, 160):
        _, coeff, _ = solve_periodic_euler_dg_1d(
            _wave_state, ncells=cells, degree=2, final_time=0.0)
        dx = 1.0 / cells
        counts = {M: int(_troubled_cell_indicator(coeff, dx, M).sum())
                  for M in (1.0, 10.0, 100.0)}
        assert counts[10.0] == 0 and counts[100.0] == 0, (cells, counts)
        # A strict TVD constant may flag the steep stretches, never everything.
        assert counts[1.0] < cells - 2, (cells, counts)


def test_tvb_indicator_still_flags_discontinuities():
    """Shock data must be flagged even with a large TVB constant."""
    cells = 100
    dx = 1.0 / cells
    x = (np.arange(cells) + 0.5) * dx
    left = x < 0.5
    coeff = torch.zeros(cells, 3, 3, dtype=torch.float64)
    coeff[:, 0, 0] = torch.as_tensor(np.where(left, 1.0, 0.125))
    coeff[:, 0, 2] = torch.as_tensor(np.where(left, 2.5, 0.25))
    for M in (1.0, 100.0):
        troubled = _troubled_cell_indicator(coeff, dx, M).numpy()
        assert troubled.any(), M
        flagged = np.flatnonzero(troubled)
        # Only the two cells adjacent to the interface may be flagged.
        assert set(flagged.tolist()) <= {49, 50}, flagged


def test_limiter_stage_is_validated_and_inert_without_limiter():
    with np.testing.assert_raises(ValueError):
        solve_euler_dg_1d(SOD_LEFT, SOD_RIGHT, ncells=40, degree=2,
                          final_time=0.01, limiter_stage="sometimes")
    _, every, _ = solve_euler_dg_1d(SOD_LEFT, SOD_RIGHT, ncells=40, degree=2,
                                    final_time=0.01, limiter_stage="every")
    _, final, _ = solve_euler_dg_1d(SOD_LEFT, SOD_RIGHT, ncells=40, degree=2,
                                    final_time=0.01, limiter_stage="final")
    assert torch.equal(every, final)


def test_periodic_entropy_wave_converges_at_design_order():
    """Exact entropy-wave solution: P2 mode DG must reach order p + 1 = 3."""
    errors, drifts = [], []
    for cells in (20, 40, 80):
        x, coeff, meta = solve_periodic_euler_dg_1d(
            _wave_state, ncells=cells, degree=2, final_time=FINAL_TIME)
        errors.append(_wave_error(x, coeff))
        drifts.append(max(meta["conservation_drift"].values()))
    orders = [
        float(np.log(errors[i] / errors[i + 1]) / np.log(2.0))
        for i in range(len(errors) - 1)
    ]
    assert min(orders) > 2.7, orders
    assert errors[-1] < 1e-2, errors
    assert max(drifts) < 1e-12, drifts


def test_limiter_preserves_design_order_on_smooth_wave():
    """TVB with a scale-consistent constant keeps the third-order rate."""
    errors = []
    for cells in (20, 40, 80):
        x, coeff, _ = solve_periodic_euler_dg_1d(
            _wave_state, ncells=cells, degree=2, final_time=FINAL_TIME,
            limiter="tvb", tvb_constant=10.0, limiter_stage="final")
        errors.append(_wave_error(x, coeff))
    orders = [
        float(np.log(errors[i] / errors[i + 1]) / np.log(2.0))
        for i in range(len(errors) - 1)
    ]
    assert min(orders) > 2.7, (orders, errors)
    assert errors[-1] < 1e-2, errors