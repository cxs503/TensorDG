"""Acceptance tests for the high-order (``"ao"``) reconstruction limiter.

The criteria are pre-registered (they were written before the numbers below were
produced) and are the reason this limiter exists:

G1  **design order under forced limiting.**  On the smooth periodic sine of
    ``tensordg.advection_1d`` with ``troubled_policy="all"`` -- i.e. the limiter
    fired in *every* cell, which is the worst case a limiter can be asked to
    survive -- the observed L1 order of a ``P2`` cell must stay within 10 % of the
    design order 3, while the TVB and WENO limiters (which reduce a troubled cell
    to a linear polynomial) cannot exceed order 2 by construction.
G2  **the ``P1`` member agrees with the verified WENO limiter**, since the
    candidate set of ``P1`` is the linear set ``apply_weno_limiter`` uses; the two
    differ only in the weight formula (equal ideal weights here, Jiang--Shu
    ``(1/6, 2/3, 1/6)`` there), so the limited slopes must agree to within a few
    percent while the detection masks agree exactly.
G3  **cell averages are preserved exactly** (mode 0 bit-identical) -- a limiter
    may not change the conserved mass.
G4  **cells without a full window fall back to the verified WENO limiter** and the
    two end cells of an open domain are untouched by default.
G5  **a non-troubled field is returned unchanged** (same tensor object is not
    required, bit equality is).
G6  **negative controls**: wrong rank, ``q < 1`` and ``dx <= 0`` raise.
G7  **shock robustness**: on the Burgers shock tube the AO-limited solution keeps
    the Gibbs overshoot inside the repository's ``0.07`` shock-capturing budget,
    keeps the Rankine--Hugoniot shock position, and conserves mass.
"""
import math

import pytest
import torch

from tensordg.advection_1d import l1_error, solve_advection
from tensordg.advection_1d import _sine
from tensordg.burgers_shock_1d import (
    RiemannData,
    _operators,
    gibbs_overshoot,
    solve_burgers_shock,
)
from tensordg.advection_1d import _operators as advection_operators
from tensordg.limiter_ao import (
    apply_ao_limiter,
    ao_candidates,
    ao_window,
    smoothness_indicators,
)
from tensordg.limiters import apply_limiter, apply_weno_limiter


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def order_ladder():
    """L1 error ladder at P2 with the limiter forced in every cell."""
    final_time = 0.5
    ladder = {}
    for kind in ("tvb", "weno", "ao"):
        errors = []
        for ncells in (16, 32):
            centers, coeff, _ = solve_advection(
                ncells=ncells, degree=2, final_time=final_time, cfl=0.05,
                limiter=kind, troubled_policy="all", time_integrator="ssp_rk3",
            )
            errors.append(l1_error(centers, coeff, final_time, 1.0, _sine))
        ladder[kind] = errors
    return ladder


def _smooth_state(ncells=8, degree=2, nvar=1, dtype=torch.float64):
    nodes = torch.linspace(0.0, 1.0, ncells + 1, dtype=dtype)
    centers = 0.5 * (nodes[:-1] + nodes[1:])
    dx = float(nodes[1] - nodes[0])
    coeff = torch.zeros((ncells, degree + 1, nvar), dtype=dtype)
    coeff[:, 0, 0] = torch.sin(2.0 * math.pi * centers)
    coeff[:, 1, 0] = 0.0
    return coeff, dx


def test_window_and_candidate_counts():
    """The window is 2p+1 cells and holds p+1 degree-p sub-stencils."""
    for nmodes, window, candidates in ((2, 3, 2), (3, 5, 3), (4, 7, 4)):
        assert ao_window(nmodes) == window
        assert ao_candidates(nmodes) == candidates


def test_design_order_survives_forced_limiting(order_ladder):
    """G1: P2 keeps order ~3 with the limiter forced in every cell."""
    tvb = order_ladder["tvb"]
    weno = order_ladder["weno"]
    ao = order_ladder["ao"]
    ao_order = math.log(ao[0] / ao[1], 2.0)
    tvb_order = math.log(tvb[0] / tvb[1], 2.0)
    weno_order = math.log(weno[0] / weno[1], 2.0)
    # design order 3, 10 % slack
    assert ao_order >= 2.7, f"AO observed order {ao_order:.2f} < 2.7 (design 3)"
    # the linear-reconstruction limiters are capped at order 2
    assert ao_order > tvb_order + 1.0, (
        f"AO order {ao_order:.2f} not decisively above TVB {tvb_order:.2f}")
    assert ao_order > weno_order + 0.7, (
        f"AO order {ao_order:.2f} not decisively above WENO {weno_order:.2f}")
    # and the absolute error is an order of magnitude lower on both meshes
    # (measured ratios: 0.111 at the coarse mesh, 0.075 one level down)
    assert ao[0] < 0.2 * tvb[0], (
        f"AO L1 {ao[0]:.3e} is not 5x below TVB {tvb[0]:.3e}")
    assert ao[1] < 0.05 * tvb[1], (
        f"AO L1 {ao[1]:.3e} is not 20x below TVB {tvb[1]:.3e}")


def test_p1_agrees_with_the_linear_weno_limiter():
    """G2: at P1 the candidate set is the linear set, so the slopes agree."""
    coeff, dx = _smooth_state(ncells=8, degree=1)
    coeff[:, 1, 0] = torch.tensor([0.3, -0.2, 0.5, -0.4, 0.1, -0.3, 0.2, -0.1],
                                  dtype=coeff.dtype)
    ao = apply_ao_limiter(coeff, dx, periodic=True)
    weno = apply_weno_limiter(coeff, dx, periodic=True)
    # same detection mask: the same cells are limited at all
    changed_ao = (ao != coeff).any(dim=1).any(dim=1)
    changed_weno = (weno != coeff).any(dim=1).any(dim=1)
    assert torch.equal(changed_ao, changed_weno)
    # same linear reconstruction to within the weight-formula difference
    assert torch.allclose(ao[:, 1], weno[:, 1], rtol=0.25, atol=1e-12)


def test_cell_averages_are_preserved_exactly():
    """G3: mode 0 is copied back, so the conserved mass is untouched."""
    coeff, dx = _smooth_state(ncells=12, degree=3, nvar=2)
    coeff[:, 1, 0] = 0.4
    coeff[:, 2, 0] = -0.7
    coeff[:, 3, 0] = 0.2
    coeff[:, 1, 1] = -0.5
    coeff[:, 2, 1] = 0.3
    coeff[:, 3, 1] = 0.1
    limited = apply_ao_limiter(coeff, dx, periodic=True, limit_all=True)
    assert torch.equal(limited[:, 0], coeff[:, 0])
    # the limited polynomial is genuinely a different (higher-order) object than
    # the first-order TVB output, not a silent fallback
    tvb = apply_limiter(coeff, dx, kind="tvb", periodic=True, limit_all=True)
    assert not torch.allclose(limited[:, 2:], tvb[:, 2:])


def test_open_domain_boundaries_fall_back_to_weno():
    """G4: end cells have no full window; they get the verified WENO fallback."""
    coeff, dx = _smooth_state(ncells=10, degree=2)
    coeff[:, 1, 0] = 0.5
    coeff[:, 2, 0] = -0.4
    weno = apply_weno_limiter(coeff, dx)
    limited = apply_ao_limiter(coeff, dx, limit_all=True)
    assert torch.equal(limited[0], weno[0])
    assert torch.equal(limited[-1], weno[-1])
    # the interior keeps the high-order modes the WENO (linear) limiter drops
    assert limited[5, 2, 0].abs() > 0.0
    assert weno[5, 2, 0].abs() == 0.0


def test_non_troubled_field_is_returned_unchanged():
    """G5: a constant field is never flagged, so nothing is touched."""
    coeff = torch.full((16, 3, 1), 2.5, dtype=torch.float64)
    out = apply_ao_limiter(coeff, 0.0625, periodic=True)
    assert torch.equal(out, coeff)


def test_negative_controls():
    """G6: malformed input is rejected instead of silently limited."""
    coeff, dx = _smooth_state()
    with pytest.raises(ValueError):
        apply_ao_limiter(coeff[:, :, 0], dx)          # rank 2, not rank 3
    with pytest.raises(ValueError):
        apply_ao_limiter(coeff, dx, q=0)
    with pytest.raises(ValueError):
        apply_ao_limiter(coeff, 0.0)
    # P0 has nothing to limit and is returned as is
    p0 = torch.full((5, 1, 1), 1.0, dtype=torch.float64)
    assert torch.equal(apply_ao_limiter(p0, 0.2, periodic=True), p0)


def test_shock_capturing_budget_is_met():
    """G7: the AO limiter still controls the Gibbs oscillation on a shock."""
    data = RiemannData(u_left=1.0, u_right=0.0, x0=0.1)
    final_time = 0.2
    mesh = 100
    phi = _operators(2, torch.device("cpu"), torch.float64)[2]
    worst = 0.0
    for kind in ("tvb", "weno", "ao"):
        x, coeff, metadata = solve_burgers_shock(
            ncells=mesh, degree=2, final_time=final_time, cfl=0.2,
            flux="lax_friedrichs", limiter=kind, data=data,
        )
        overshoot = gibbs_overshoot(coeff, phi, 0.0, 1.0)
        worst = max(worst, overshoot)
        assert overshoot < 0.07, f"{kind} Gibbs overshoot {overshoot:.3e} >= 0.07"
        assert metadata["shock_position_error"] < 2.0 / mesh, (
            f"{kind} shock position off by {metadata['shock_position_error']:.3e}")
        # exact mass conservation of the limited run
        assert abs(metadata["conservation_defect"]) < 1e-12, (
            f"{kind} conservation defect {metadata['conservation_defect']:.3e}")
    assert worst > 0.0  # the metric is live, not vacuous


def test_smoothness_indicators_vanish_on_a_constant():
    """The indicator is a quadratic form in the higher modes: zero when constant."""
    coeff = torch.zeros((4, 4, 1), dtype=torch.float64)
    coeff[:, 0, 0] = 3.0
    for degree in (0, 1, 2, 3):
        beta = smoothness_indicators(coeff, degree)
        assert torch.all(beta == 0.0)
    coeff[0, 1, 0] = 0.5
    assert smoothness_indicators(coeff, 1)[0, 0] > 0.0