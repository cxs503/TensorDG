"""Tests for TVB and WENO limiters."""
import numpy as np
import torch

from tensordg.limiters import (
    apply_limiter,
    apply_tvb_limiter,
    apply_weno_limiter,
    _troubled_cell_indicator,
)


def test_tvb_limiter_preserves_cell_average():
    """TVB limiter must preserve mode-0 (cell average)."""
    torch.manual_seed(42)
    coeff = torch.randn(20, 3, 3, dtype=torch.float64)
    dx = 0.05
    limited = apply_tvb_limiter(coeff, dx, tvb_constant=1.0)
    # Cell averages should be unchanged
    assert torch.allclose(limited[:, 0], coeff[:, 0])


def test_weno_limiter_preserves_cell_average():
    """WENO limiter must preserve mode-0 (cell average)."""
    torch.manual_seed(42)
    coeff = torch.randn(20, 3, 3, dtype=torch.float64)
    dx = 0.05
    limited = apply_weno_limiter(coeff, dx, tvb_constant=1.0)
    # Cell averages should be unchanged
    assert torch.allclose(limited[:, 0], coeff[:, 0])


def test_tvb_limiter_does_not_modify_smooth_solution():
    """TVB limiter should not modify a smooth monotone solution."""
    ncells = 30
    dx = 1.0 / ncells
    # Create a smooth linear function: u(x) = x
    x = (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    # For a linear function, only mode-0 and mode-1 are non-zero
    coeff = torch.zeros(ncells, 3, 3, dtype=torch.float64)
    coeff[:, 0, 0] = x  # density = x
    coeff[:, 1, 0] = dx / 2.0  # slope in mode-1
    coeff[:, 0, 1] = 2.0 * x  # velocity = 2x
    coeff[:, 1, 1] = dx  # slope
    coeff[:, 0, 2] = x ** 2  # energy = x^2
    coeff[:, 1, 2] = 2.0 * x * dx  # slope
    
    limited = apply_tvb_limiter(coeff, dx, tvb_constant=1.0)
    # Smooth monotone solution should not be modified
    assert torch.allclose(limited, coeff, atol=1e-12)


def test_tvb_limiter_reduces_extremum():
    """TVB limiter should reduce a non-monotone extremum."""
    ncells = 20
    dx = 1.0 / ncells
    # Create a function with a local maximum at cell 10
    coeff = torch.zeros(ncells, 3, 1, dtype=torch.float64)
    # Parabolic profile with maximum exactly at cell 10
    for i in range(ncells):
        coeff[i, 0, 0] = 1.0 - ((i - 10) * dx) ** 2
    # Add spurious high-order modes
    torch.manual_seed(123)
    coeff[:, 2, 0] = 0.1 * torch.randn(ncells, dtype=torch.float64)
    
    limited = apply_tvb_limiter(coeff, dx, tvb_constant=0.0)
    # At the extremum (cell 10), higher modes should be reduced
    assert limited[10, 2, 0].abs() < coeff[10, 2, 0].abs()


def test_weno_limiter_reduces_extremum():
    """WENO limiter should reduce a non-monotone extremum."""
    ncells = 20
    dx = 1.0 / ncells
    # Create a function with a local maximum at cell 10
    coeff = torch.zeros(ncells, 3, 1, dtype=torch.float64)
    # Parabolic profile with maximum exactly at cell 10
    for i in range(ncells):
        coeff[i, 0, 0] = 1.0 - ((i - 10) * dx) ** 2
    # Add spurious high-order modes
    torch.manual_seed(123)
    coeff[:, 2, 0] = 0.1 * torch.randn(ncells, dtype=torch.float64)
    
    limited = apply_weno_limiter(coeff, dx, tvb_constant=0.0)
    # At the extremum (cell 10), higher modes should be reduced
    assert limited[10, 2, 0].abs() < coeff[10, 2, 0].abs()


def test_troubled_cell_indicator_detects_extremum():
    """Indicator should flag cells at extrema."""
    ncells = 20
    dx = 1.0 / ncells
    coeff = torch.zeros(ncells, 2, 1, dtype=torch.float64)
    # Parabolic profile with maximum exactly at cell 10
    for i in range(ncells):
        coeff[i, 0, 0] = 1.0 - ((i - 10) * dx) ** 2
    
    troubled = _troubled_cell_indicator(coeff, dx, tvb_constant=0.0)
    # Cell 10 should be flagged as troubled
    assert troubled[10].item() is True
    # Boundary cells should never be troubled
    assert troubled[0].item() is False
    assert troubled[-1].item() is False


def test_troubled_cell_indicator_ignores_monotone():
    """Indicator should not flag monotone regions."""
    ncells = 20
    dx = 1.0 / ncells
    coeff = torch.zeros(ncells, 2, 1, dtype=torch.float64)
    x = (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    # Linear monotone profile
    coeff[:, 0, 0] = x
    
    troubled = _troubled_cell_indicator(coeff, dx, tvb_constant=0.0)
    # No cells should be troubled
    assert troubled.any().item() is False


def test_apply_limiter_dispatcher():
    """Dispatcher should route to correct limiter."""
    coeff = torch.randn(10, 3, 3, dtype=torch.float64)
    dx = 0.1
    
    # None should return unchanged
    result = apply_limiter(coeff, dx, kind=None)
    assert torch.equal(result, coeff)
    
    # "none" should return unchanged
    result = apply_limiter(coeff, dx, kind="none")
    assert torch.equal(result, coeff)
    
    # "tvb" should call TVB limiter
    result = apply_limiter(coeff, dx, kind="tvb", tvb_constant=1.0)
    assert result.shape == coeff.shape
    
    # "weno" should call WENO limiter
    result = apply_limiter(coeff, dx, kind="weno", tvb_constant=1.0)
    assert result.shape == coeff.shape
    
    # Invalid should raise
    try:
        apply_limiter(coeff, dx, kind="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unknown limiter" in str(e)


def test_limiter_p0_is_noop():
    """P0 (nmodes=1) should be unchanged by any limiter."""
    coeff = torch.randn(10, 1, 3, dtype=torch.float64)
    dx = 0.1
    
    limited_tvb = apply_tvb_limiter(coeff, dx)
    assert torch.equal(limited_tvb, coeff)
    
    limited_weno = apply_weno_limiter(coeff, dx)
    assert torch.equal(limited_weno, coeff)


def test_limiter_handles_multivariate():
    """Limiters should handle multiple variables correctly."""
    ncells = 15
    dx = 1.0 / ncells
    coeff = torch.zeros(ncells, 3, 3, dtype=torch.float64)
    x = (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    
    # Different profiles for each variable
    coeff[:, 0, 0] = x  # density
    coeff[:, 0, 1] = 2.0 * x  # momentum
    coeff[:, 0, 2] = x ** 2  # energy
    
    limited = apply_tvb_limiter(coeff, dx, tvb_constant=1.0)
    # Cell averages should be preserved for all variables
    assert torch.allclose(limited[:, 0], coeff[:, 0])
    assert limited.shape == coeff.shape


def test_limiter_dtype_consistency():
    """Limiters should preserve dtype."""
    coeff_f64 = torch.randn(10, 3, 3, dtype=torch.float64)
    coeff_f32 = torch.randn(10, 3, 3, dtype=torch.float32)
    dx = 0.1
    
    limited_f64 = apply_tvb_limiter(coeff_f64, dx)
    assert limited_f64.dtype == torch.float64
    
    limited_f32 = apply_tvb_limiter(coeff_f32, dx)
    assert limited_f32.dtype == torch.float32


def test_limiter_with_shock_profile():
    """Test limiter on a step function (shock-like profile)."""
    ncells = 40
    dx = 1.0 / ncells
    coeff = torch.zeros(ncells, 3, 1, dtype=torch.float64)
    x = (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    
    # Step function at x=0.5
    coeff[:, 0, 0] = torch.where(x < 0.5, torch.tensor(1.0), torch.tensor(0.125))
    
    limited = apply_tvb_limiter(coeff, dx, tvb_constant=0.0)
    # Limiter should activate near the discontinuity
    # Cell averages should still be preserved
    assert torch.allclose(limited[:, 0], coeff[:, 0])
    # Shape should be unchanged
    assert limited.shape == coeff.shape


def test_weno_smoothness_weights():
    """WENO weights should favor smooth stencils."""
    from tensordg.limiters import _weno_weights
    
    # Smooth case: all betas similar
    beta0 = torch.tensor([1.0, 1.0, 1.0])
    beta1 = torch.tensor([1.0, 1.0, 1.0])
    beta2 = torch.tensor([1.0, 1.0, 1.0])
    w0, w1, w2 = _weno_weights(beta0, beta1, beta2)
    # Weights should sum to 1
    assert torch.allclose(w0 + w1 + w2, torch.ones(3), atol=1e-10)
    # With equal betas, w1 (center stencil) should be largest (ideal weight 2/3)
    assert (w1 > w0).all()
    assert (w1 > w2).all()


# ---------------------------------------------------------------------------
# Regression tests for the mode-1 scaling convention
# (fixed in the limiter-verification pass: c1 = physical slope * dx / 2)
# ---------------------------------------------------------------------------

def _step_coefficients():
    """Cell averages with a mild non-monotone slope so the detector fires."""
    ncells, dx = 6, 0.1
    coeff = torch.zeros(ncells, 3, 1, dtype=torch.float64)
    coeff[:, 0, 0] = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.6, 2.6], dtype=torch.float64)
    return coeff, dx


def test_tvb_limited_slope_equals_half_minmod():
    """Mode-1 must be minmod(d_left, d_right) / 2, not minmod itself.

    The normalised Legendre mode-1 on [-1, 1] is xi, so a physical slope s is
    stored as c1 = s * dx / 2 = minmod / 2.  Writing c1 = minmod doubles the
    reconstructed slope and makes the limiter anti-diffusive.
    """
    from tensordg.limiters import _minmod

    coeff, dx = _step_coefficients()
    avg = coeff[:, 0]
    padded = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    target = _minmod(avg - padded[:-2], padded[2:] - avg) / 2.0

    limited = apply_tvb_limiter(coeff, dx, tvb_constant=1e-9)
    # Cell 3 carries the non-zero minmod slope; all other cells stay at 0.
    assert target[3, 0].item() > 0.0
    assert torch.allclose(limited[3, 1], target[3], atol=1e-12)
    assert torch.allclose(limited[:, 0], coeff[:, 0])


def test_tvb_limiter_does_not_amplify_slope():
    """Limited slope magnitude must not exceed the largest one-sided slope."""
    torch.manual_seed(7)
    ncells, dx = 25, 0.04
    coeff = torch.zeros(ncells, 3, 1, dtype=torch.float64)
    avg = torch.cumsum(torch.rand(ncells, dtype=torch.float64), dim=0)
    avg[10:] += 2.0 * torch.rand(ncells - 10, dtype=torch.float64)
    coeff[:, 0, 0] = avg

    padded = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    d_left = (avg - padded[:-2])[:, None]
    d_right = (padded[2:] - avg)[:, None]
    bound = 0.5 * torch.maximum(d_left.abs(), d_right.abs())

    limited = apply_tvb_limiter(coeff, dx, tvb_constant=0.0)
    assert bool((limited[:, 1].abs() <= bound + 1e-12).all())


def test_limited_interface_values_stay_within_neighbour_bounds():
    """A limited polynomial must not overshoot into the neighbouring cell."""
    coeff, dx = _step_coefficients()
    limited = apply_tvb_limiter(coeff, dx, tvb_constant=1e-9)
    # Legendre P1 = xi, so u(x_{i+1/2}) = c0 + c1 and u(x_{i-1/2}) = c0 - c1.
    avg = limited[:, 0, 0]
    c1 = limited[:, 1, 0]
    right_edge = avg + c1
    left_edge = avg - c1
    upper = torch.maximum(avg[:-1], avg[1:])
    lower = torch.minimum(avg[:-1], avg[1:])
    # Physical slope s = 2 * c1 / dx; the interface values are c0 +/- c1.
    for i in range(1, len(avg) - 1):
        assert left_edge[i] >= lower[i - 1] - 1e-12
        assert right_edge[i] <= upper[i] + 1e-12


def test_tvb_and_weno_share_mode1_scale():
    """Both limiters must write mode-1 in the same (half-slope) units."""
    coeff, dx = _step_coefficients()
    tvb = apply_tvb_limiter(coeff, dx, tvb_constant=1e-9)
    weno = apply_weno_limiter(coeff, dx, tvb_constant=1e-9)
    # Cell 1: cell averages are constant, so both slopes must vanish.
    assert tvb[1, 1, 0].abs().item() < 1e-12
    assert weno[1, 1, 0].abs().item() < 1e-12
    # Cell 3: the WENO convex combination stays inside the one-sided bounds
    # that the TVB header also respects, i.e. |c1| <= max(|d_left|, |d_right|)/2.
    padded = torch.cat([coeff[:1, 0], coeff[:, 0], coeff[-1:, 0]], dim=0)
    bound = 0.5 * max(
        abs(float(padded[4, 0] - padded[2, 0])),
        abs(float(padded[3, 0] - padded[4, 0])),
    )
    assert weno[3, 1, 0].abs().item() <= bound + 1e-12
    assert tvb[3, 1, 0].abs().item() <= bound + 1e-12
