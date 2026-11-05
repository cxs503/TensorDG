"""Direct unit tests for the shared TVB troubled-cell test in ``tensordg/tvb.py``.

The ``troubled_from_differences`` function is the single implementation of the
scale-consistent TVB criterion used by both the 1-D and 2-D limiters (extracted
in m2549).  All existing limiter tests exercise it indirectly through
``_troubled_cell_indicator``; these tests verify the core function in isolation.

Reference: Cockburn & Shu (1989) for the TVB criterion;
docs/limiter_policy_verification.md for the scale-consistency fix (m2500).
"""
from __future__ import annotations

import pytest
import torch

from tensordg.tvb import minmod, troubled_from_differences


# ---------------------------------------------------------------------------
# minmod helper
# ---------------------------------------------------------------------------

def test_minmod_same_sign_selects_smaller():
    """minmod(a, b) = sign(a) * min(|a|, |b|) when a, b have the same sign."""
    a = torch.tensor([2.0, -3.0, 5.0])
    b = torch.tensor([4.0, -1.0, 2.0])
    result = minmod(a, b)
    expected = torch.tensor([2.0, -1.0, 2.0])
    assert torch.allclose(result, expected)


def test_minmod_opposite_sign_returns_zero():
    """minmod(a, b) = 0 when a and b have opposite signs."""
    a = torch.tensor([2.0, -3.0, 0.5])
    b = torch.tensor([-1.0, 4.0, -0.3])
    result = minmod(a, b)
    assert torch.allclose(result, torch.zeros_like(a))


def test_minmod_one_zero_returns_zero():
    """minmod(a, 0) = 0 and minmod(0, b) = 0."""
    a = torch.tensor([2.0, 0.0, -3.0])
    b = torch.tensor([0.0, 5.0, 0.0])
    result = minmod(a, b)
    assert torch.allclose(result, torch.zeros_like(a))


# ---------------------------------------------------------------------------
# troubled_from_differences: affine data (should never be troubled)
# ---------------------------------------------------------------------------

def test_affine_data_never_troubled():
    """For a linear profile u = a*x + b, the deviation is zero → never troubled."""
    # u_i = i, so d_left = d_right = 1.0 for all interior cells
    d_left = torch.ones(10, 1, dtype=torch.float64)
    d_right = torch.ones(10, 1, dtype=torch.float64)
    troubled, slope = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=0.0)
    assert not troubled.any(), "affine data must not be flagged as troubled"
    # minmod(1, 1) = 1, so slope should be 1.0
    assert torch.allclose(slope, torch.ones_like(slope))


def test_affine_data_with_nonzero_M_still_not_troubled():
    """Even with M > 0, affine data has zero deviation → never troubled."""
    d_left = torch.ones(10, 1, dtype=torch.float64) * 2.0
    d_right = torch.ones(10, 1, dtype=torch.float64) * 2.0
    troubled, slope = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=100.0)
    assert not troubled.any()


# ---------------------------------------------------------------------------
# troubled_from_differences: smooth extremum (should be troubled at M=0, released at large M)
# ---------------------------------------------------------------------------

def test_smooth_extremum_troubled_at_strict_tvd():
    """At a smooth extremum d_left and d_right have opposite signs → minmod = 0,
    deviation = |d_left| > 0 → troubled when M = 0."""
    # Simulate a smooth peak: u = [..., 0.8, 1.0, 0.8, ...]
    # d_left = 0.2, d_right = -0.2
    d_left = torch.tensor([[0.2]], dtype=torch.float64)
    d_right = torch.tensor([[-0.2]], dtype=torch.float64)
    troubled, slope = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=0.0)
    assert troubled.item(), "smooth extremum must be flagged at M=0"
    # minmod(0.2, -0.2) = 0
    assert slope.item() == 0.0


def test_smooth_extremum_released_at_large_M():
    """With M large enough, the smooth extremum should not be troubled.

    The deviation is |d_left - minmod(d_left, d_right)| = |0.2 - 0| = 0.2.
    The threshold is M * dx^2.  For dx = 0.1, M * 0.01 > 0.2 → M > 20.
    """
    d_left = torch.tensor([[0.2]], dtype=torch.float64)
    d_right = torch.tensor([[-0.2]], dtype=torch.float64)
    troubled, _ = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=25.0)
    assert not troubled.item(), "smooth extremum should be released at M=25, dx=0.1"


def test_smooth_extremum_threshold_boundary():
    """Verify the exact threshold: M * dx^2 = deviation → not troubled (strict >)."""
    d_left = torch.tensor([[0.2]], dtype=torch.float64)
    d_right = torch.tensor([[-0.2]], dtype=torch.float64)
    # deviation = 0.2, threshold = M * 0.01
    # M = 20 → threshold = 0.2, deviation == threshold → not troubled (strict >)
    troubled, _ = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=20.0)
    assert not troubled.item()
    # M = 19 → threshold = 0.19, deviation > threshold → troubled
    troubled, _ = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=19.0)
    assert troubled.item()


# ---------------------------------------------------------------------------
# troubled_from_differences: discontinuity (should always be troubled)
# ---------------------------------------------------------------------------

def test_discontinuity_always_troubled():
    """At a jump, the deviation is O(dx * u') which dominates M * dx^2 for any
    reasonable M, so the cell should always be troubled."""
    # Jump: u = [..., 0.0, 1.0, 1.0, ...]
    # d_left = 1.0, d_right = 0.0
    d_left = torch.tensor([[1.0]], dtype=torch.float64)
    d_right = torch.tensor([[0.0]], dtype=torch.float64)
    troubled, slope = troubled_from_differences(d_left, d_right, spacing=0.01, tvb_constant=1000.0)
    assert troubled.item(), "discontinuity must be flagged even at large M"
    # minmod(1.0, 0.0) = 0.0
    assert slope.item() == 0.0


def test_discontinuity_with_small_M():
    """A discontinuity with small M should definitely be troubled."""
    d_left = torch.tensor([[0.5]], dtype=torch.float64)
    d_right = torch.tensor([[-0.3]], dtype=torch.float64)
    troubled, slope = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=1.0)
    assert troubled.item()
    # minmod(0.5, -0.3) = 0
    assert slope.item() == 0.0


# ---------------------------------------------------------------------------
# troubled_from_differences: smooth decelerating data (should be troubled at M=0)
# ---------------------------------------------------------------------------

def test_smooth_decelerating_troubled_at_M_zero():
    """For smooth decelerating data, d_left != d_right but same sign.
    deviation = |d_left - minmod(d_left, d_right)| = |d_left - d_right| = |u''| dx^2.
    At M=0, any non-zero deviation triggers troubled."""
    # u'' < 0: d_left = 0.3, d_right = 0.1 (decelerating rise)
    d_left = torch.tensor([[0.3]], dtype=torch.float64)
    d_right = torch.tensor([[0.1]], dtype=torch.float64)
    troubled, slope = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=0.0)
    assert troubled.item(), "decelerating smooth data must be flagged at M=0"
    # minmod(0.3, 0.1) = 0.1
    assert slope.item() == pytest.approx(0.1)


def test_smooth_decelerating_released_at_appropriate_M():
    """For smooth decelerating data, deviation = |d_left - d_right| = 0.2.
    Threshold = M * dx^2 = M * 0.01.  M > 20 → not troubled."""
    d_left = torch.tensor([[0.3]], dtype=torch.float64)
    d_right = torch.tensor([[0.1]], dtype=torch.float64)
    troubled, _ = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=25.0)
    assert not troubled.item()


# ---------------------------------------------------------------------------
# troubled_from_differences: multi-variable (ncells, nvar)
# ---------------------------------------------------------------------------

def test_multi_variable_troubled_reduces_across_vars():
    """The troubled mask should be True if *any* variable is troubled."""
    # 5 cells, 3 variables
    d_left = torch.zeros(5, 3, dtype=torch.float64)
    d_right = torch.zeros(5, 3, dtype=torch.float64)
    # Variable 0: affine (not troubled)
    d_left[:, 0] = 1.0
    d_right[:, 0] = 1.0
    # Variable 1: extremum at cell 2 (troubled)
    d_left[2, 1] = 0.2
    d_right[2, 1] = -0.2
    # Variable 2: affine (not troubled)
    d_left[:, 2] = 0.5
    d_right[:, 2] = 0.5

    troubled, slope = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=0.0)
    assert troubled.shape == (5,)
    assert troubled[2].item(), "cell 2 must be troubled due to variable 1"
    assert not troubled[0].item()
    assert not troubled[1].item()
    assert not troubled[3].item()
    assert not troubled[4].item()


def test_multi_variable_slope_shape():
    """The slope tensor should have the same shape as d_left/d_right."""
    d_left = torch.randn(8, 4, dtype=torch.float64)
    d_right = torch.randn(8, 4, dtype=torch.float64)
    troubled, slope = troubled_from_differences(d_left, d_right, spacing=0.05, tvb_constant=1.0)
    assert slope.shape == d_left.shape
    assert troubled.shape == (8,)


# ---------------------------------------------------------------------------
# troubled_from_differences: round-off floor for affine data
# ---------------------------------------------------------------------------

def test_roundoff_floor_does_not_trigger_on_near_affine():
    """For nearly affine data, the deviation is a few ulp and should not trigger
    the troubled flag due to the round-off floor (1e-12 * scale)."""
    # Exactly affine: d_left = d_right = 1.0
    d_left = torch.ones(10, 1, dtype=torch.float64)
    d_right = torch.ones(10, 1, dtype=torch.float64)
    # Add tiny perturbation (well below round-off floor)
    d_left[5, 0] += 1e-14
    troubled, _ = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=0.0)
    assert not troubled.any(), "near-affine data must not be flagged due to round-off"


# ---------------------------------------------------------------------------
# troubled_from_differences: dtype preservation
# ---------------------------------------------------------------------------

def test_dtype_preserved():
    """The function should preserve the input dtype."""
    d_left = torch.ones(5, 2, dtype=torch.float32)
    d_right = torch.ones(5, 2, dtype=torch.float32)
    troubled, slope = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=0.0)
    assert slope.dtype == torch.float32
    assert troubled.dtype == torch.bool


def test_dtype_float64():
    """Float64 inputs should produce float64 slope."""
    d_left = torch.ones(5, 2, dtype=torch.float64)
    d_right = torch.ones(5, 2, dtype=torch.float64)
    troubled, slope = troubled_from_differences(d_left, d_right, spacing=0.1, tvb_constant=0.0)
    assert slope.dtype == torch.float64


# ---------------------------------------------------------------------------
# Scale consistency: the key property that makes M grid-independent
# ---------------------------------------------------------------------------

def test_scale_consistency_smooth_wave():
    """For a smooth wave u = sin(2*pi*x), the deviation scales as O(dx^2 * u'').
    With the scale-consistent criterion, M can be chosen grid-independently."""
    import numpy as np

    # u = sin(2*pi*x), u'' = -4*pi^2 * sin(2*pi*x)
    # At x = 0.25, u'' = -4*pi^2 ≈ -39.48
    # deviation ≈ |u''| * dx^2 = 39.48 * dx^2
    # For M > 39.48, the cell should not be troubled

    for ncells in (40, 80, 160):
        dx = 1.0 / ncells
        centres = (np.arange(ncells) + 0.5) * dx
        u = np.sin(2.0 * np.pi * centres)
        d_left = torch.tensor(np.diff(u, prepend=u[0]), dtype=torch.float64).unsqueeze(-1)
        d_right = torch.tensor(np.diff(u, append=u[-1]), dtype=torch.float64).unsqueeze(-1)

        # At M = 100, all cells should be released (100 > 39.48)
        troubled, _ = troubled_from_differences(d_left, d_right, spacing=dx, tvb_constant=100.0)
        n_troubled = troubled.sum().item()
        # Some cells near the extrema may still be flagged due to discretisation
        # effects, but the count should not grow with refinement
        assert n_troubled < ncells * 0.1, (
            f"ncells={ncells}: {n_troubled} cells troubled at M=100, expected < 10%"
        )
