"""P3 smooth-flow design order -- the R5 gate's missing top rung.

Why this test exists
--------------------
``docs/CASE-LAX-DG.md`` reports the smooth entropy-wave gate (R5) for ``p1`` and
``p2`` only: ``benchmarks/lax_case_pool.py`` registers ``ENTROPY_DEGREES =
(1, 2)``.  At the same time the pool's order gates (C3/R3) FAIL *only* on the
``p3`` rung of the Lax ladder (measured order ``0.48`` / ``0.27``).  Without a
p3 smooth-flow measurement there is no way to tell two very different stories
apart:

* the p3 **spatial operator is wrong**, or
* the p3 **shock solution is polluted** (Gibbs oscillations leaking out of the
  contact and the shock into the plateaus, which is what an unlimited scheme
  does and what an order gate on a discontinuous field is allowed to punish).

This test pins the first story down on smooth data through the *same* kernels
(``solve_periodic_euler_dg_1d`` -> HLLC flux + modal volume term + SSP-RK3 +
Zhang-Shu positivity), on the periodic entropy wave the R5 gate already uses:
``rho = 1 + 0.2 sin(2 pi x)``, ``p = 1``, ``t = 0.25``, ``cfl = 0.2``.

The measured orders on the committed configuration are 4.0097 (N=32 -> 64) and
3.9997 (N=64 -> 128); the bar below is the R5 reading (within 3 % of ``p + 1``)
applied to the 32 -> 64 pair so the test stays a few seconds long.
"""
from __future__ import annotations

import math

import pytest
import torch

from benchmarks.lax_case_pool import (
    DOMAIN,
    ENTROPY_AMPLITUDE,
    ENTROPY_CFL,
    ENTROPY_PRESSURE,
    ENTROPY_TIME,
    _weighted_l1,
    entropy_wave_exact,
    modal_fields,
)
from tensordg.euler_dg_1d import solve_periodic_euler_dg_1d


def _initial_state(node_x):
    rho = 1.0 + ENTROPY_AMPLITUDE * torch.sin(2.0 * math.pi * node_x)
    return (rho, torch.ones_like(rho), torch.full_like(rho, ENTROPY_PRESSURE))


def _l1(cells, degree):
    x, coeff, _ = solve_periodic_euler_dg_1d(
        _initial_state, ncells=cells, degree=degree, final_time=ENTROPY_TIME,
        domain=DOMAIN, cfl=ENTROPY_CFL,
    )
    points, primitive, dx, weights = modal_fields(x, coeff)
    exact = entropy_wave_exact(points, ENTROPY_TIME, 1.0)
    return _weighted_l1(points, primitive["density"], exact, weights, dx)


def _order(coarse_cells, fine_cells, degree):
    coarse, fine = _l1(coarse_cells, degree), _l1(fine_cells, degree)
    return math.log(coarse / fine) / math.log(fine_cells / coarse_cells)


def test_p3_smooth_entropy_wave_hits_design_order():
    """The p3 operator must converge at 4th order on smooth data (R5 top rung)."""
    order = _order(32, 64, 3)
    design = 4.0
    assert order == pytest.approx(design, rel=0.03), (
        f"p3 smooth order {order:.4f} deviates from the design order {design} by "
        f"more than the R5 band; a p3 operator defect, not a shock effect"
    )


def test_p3_is_not_worse_than_p2_on_smooth_data():
    """A higher degree must buy accuracy where the flow is smooth."""
    p2 = _l1(64, 2)
    p3 = _l1(64, 3)
    assert p3 < p2 / 10.0, (
        f"p3 L1 {p3:.3e} is not at least a decade below p2 L1 {p2:.3e} at N=64; "
        "the degree-3 path is not delivering its design accuracy"
    )