"""Regression tests for the 2-D limiter's behaviour on *smooth* flow.

The repository only ever measured the 2-D TVB/WENO limiter on shocked data, so
the accuracy side of the Cockburn-Shu TVB criterion was untested in 2-D: does a
large enough ``tvb_constant`` leave smooth data alone (and hence keep the design
order), and does a small one really cap the order?  These tests pin the
mechanism on the analytic isentropic vortex
(``benchmarks/limiter_design_order_2d.py`` holds the full study):

* the troubled-cell fraction is non-increasing in ``M`` and vanishes for large
  ``M`` on smooth data,
* for large ``M`` the limited run is *bit-identical* to ``limiter=None``,
* for ``M = 1`` cells are flagged, the observed order drops, and the cell
  averages (hence mass) are still conserved exactly,
* the unlimited solver recovers the design order on a two-mesh pair, which is
  what makes the drop above attributable to the limiter rather than the solver.
"""
from __future__ import annotations

import pytest
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.limiter_design_order_2d import flagged_fraction

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
FINAL_TIME = 0.2
STRENGTH = 5.0


def _vortex(degree, ncells, limiter, tvb_constant, final_time=FINAL_TIME):
    return dg2.solve_euler_dg_2d_vortex(
        ncells=ncells,
        degree=degree,
        final_time=final_time,
        cfl=0.2,
        strength=STRENGTH,
        background_velocity=(1.0, 1.0),
        limiter=limiter,
        tvb_constant=tvb_constant,
        domain=DOMAIN,
    )


def _density_error(degree, ncells, limiter, tvb_constant, final_time=FINAL_TIME):
    _, _, coeff, _ = _vortex(degree, ncells, limiter, tvb_constant, final_time)
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    from benchmarks.verify_euler_dg_2d import _vortex_error

    return coeff, _vortex_error(
        coeff, basis, nodes, DOMAIN, final_time, strength=STRENGTH, background=(1.0, 1.0)
    )["density"]


def _mass(coeff, ncells):
    dx = (DOMAIN[0][1] - DOMAIN[0][0]) / ncells
    return float(coeff[..., 0, 0, 0].sum().item()) * dx * dx


@pytest.fixture(scope="module")
def initial_state():
    """L2 projection of the vortex at t = 0 (no time stepping)."""
    _, _, coeff, _ = _vortex(degree=2, ncells=16, limiter=None, tvb_constant=0.0, final_time=0.0)
    return coeff


@pytest.fixture(scope="module")
def order_pairs():
    """(error, coeff) for two meshes and three limiter settings, computed once."""
    data = {}
    for config, limiter, tvb_constant in (
        ("none", None, 0.0),
        ("tvb_M1", "tvb", 1.0),
        ("tvb_M1000", "tvb", 1000.0),
    ):
        for ncells in (12, 24):
            coeff, error = _density_error(1, ncells, limiter, tvb_constant)
            data[(config, ncells)] = (coeff, error)
    return data


def test_troubled_fraction_is_non_increasing_in_the_tvb_constant(initial_state):
    """On smooth data the flagged fraction must fall as M grows, and reach zero."""
    fractions = [
        flagged_fraction(initial_state, 10.0 / 16, 10.0 / 16, m)
        for m in (0.1, 1.0, 10.0, 100.0, 1.0e4)
    ]
    assert fractions[0] > 0.0, "a tiny TVB constant must flag smooth data"
    for coarser, finer in zip(fractions, fractions[1:]):
        assert finer <= coarser + 1.0e-12, f"fraction grew with M: {fractions}"
    assert fractions[-1] == 0.0, f"M=1e4 still flagged cells: {fractions}"


def test_large_tvb_constant_is_bit_identical_to_no_limiter(order_pairs):
    """M large enough to release every smooth extremum => the limiter is a no-op."""
    for ncells in (12, 24):
        reference = order_pairs[("none", ncells)][0]
        limited = order_pairs[("tvb_M1000", ncells)][0]
        assert flagged_fraction(reference, 10.0 / ncells, 10.0 / ncells, 1000.0) == 0.0
        assert torch.equal(reference, limited), (
            f"limiter changed the solution with M=1000 at {ncells}^2"
        )


def test_unlimited_solver_recovers_the_design_order(order_pairs):
    """Baseline: p+1 = 2 on the analytic vortex for a 12 -> 24 mesh pair (P1)."""
    coarse = order_pairs[("none", 12)][1]
    fine = order_pairs[("none", 24)][1]
    observed = torch.log(torch.tensor(coarse / fine)).item() / torch.log(torch.tensor(2.0)).item()
    assert observed > 1.75, f"unlimited P1 observed order {observed:.2f} is below the 2 order"


def test_small_tvb_constant_caps_the_observed_order(order_pairs):
    """M = 1 flags smooth cells and measurably degrades the order (documented debt)."""
    coarse = order_pairs[("tvb_M1", 12)][1]
    fine = order_pairs[("tvb_M1", 24)][1]
    observed = torch.log(torch.tensor(coarse / fine)).item() / torch.log(torch.tensor(2.0)).item()
    unlimited = torch.log(
        torch.tensor(order_pairs[("none", 12)][1] / order_pairs[("none", 24)][1])
    ).item() / torch.log(torch.tensor(2.0)).item()
    assert fine > order_pairs[("none", 24)][1], "M=1 should cost accuracy on smooth flow"
    assert observed < unlimited - 0.3, (
        f"expected the order to drop from {unlimited:.2f}; measured {observed:.2f}"
    )


def test_limiter_never_touches_the_cell_average_or_the_mass(order_pairs):
    """Active limiting (M = 1) must leave mass conserved: mode 0 is never modified."""
    for ncells in (12, 24):
        reference = _mass(order_pairs[("none", ncells)][0], ncells)
        limited = _mass(order_pairs[("tvb_M1", ncells)][0], ncells)
        assert abs(limited - reference) <= 1.0e-12 * abs(reference), (
            f"mass drift {abs(limited - reference):.3e} with M=1 at {ncells}^2"
        )


def test_tvb_constant_1_flags_cells_on_a_resolved_smooth_solution():
    """Sanity: with M=1 and h=10/24 the detector does fire on a smooth field."""
    _, _, coeff, _ = _vortex(degree=1, ncells=24, limiter="tvb", tvb_constant=1.0)
    fraction = flagged_fraction(coeff, 10.0 / 24, 10.0 / 24, 1.0)
    assert fraction > 0.0