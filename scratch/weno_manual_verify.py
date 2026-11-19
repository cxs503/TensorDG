"""Independent verification of WENO limiter numerical properties.

This script performs hand-calculated checks on two key properties:
1. Odd symmetry: L(-coeff) == -L(coeff)
2. Cell average conservation: mode-0 is never modified

The checks are done independently of the test suite, using direct
torch operations to verify the limiter's internal logic.
"""
import torch
import numpy as np

torch.set_default_dtype(torch.float64)

from tensordg.limiters import apply_weno_limiter, apply_tvb_limiter, _weno_weights


def test_odd_symmetry_weno():
    """Verify L(-coeff) == -L(coeff) for WENO limiter.
    
    The WENO weights depend on smoothness indicators beta ~ s^2, which are
    sign-invariant. The candidate slopes flip sign when coeff flips sign.
    Therefore the limited slope flips sign but the troubled mask stays the same.
    """
    print("=" * 60)
    print("CHECK 1: WENO Limiter Odd Symmetry")
    print("=" * 60)
    
    torch.manual_seed(42)
    ncells = 20
    coeff = torch.randn(ncells, 3, 3)  # 3 modes, 3 variables
    
    # Apply limiter to original and negated data
    limited_pos = apply_weno_limiter(coeff, 0.05, tvb_constant=1.0)
    limited_neg = apply_weno_limiter(-coeff, 0.05, tvb_constant=1.0)
    neg_limited = -limited_pos
    
    # Check odd symmetry
    diff = (limited_neg - limited_neg).abs().max()
    diff_actual = (limited_neg - (-limited_pos)).abs().max()
    
    print(f"Max difference |L(-u) - (-L(u))|: {diff_actual:.2e}")
    print(f"Mode-0 preserved in positive run: {torch.equal(limited_pos[:, 0], coeff[:, 0])}")
    print(f"Mode-0 preserved in negative run: {torch.equal(limited_neg[:, 0], -coeff[:, 0])}")
    
    # Verify the troubled mask is the same for both
    from tensordg.limiters import _troubled_cell_indicator
    mask_pos = _troubled_cell_indicator(coeff, 0.05, 1.0)
    mask_neg = _troubled_cell_indicator(-coeff, 0.05, 1.0)
    print(f"Troubled mask identical: {torch.equal(mask_pos, mask_neg)}")
    print(f"Number of troubled cells: {mask_pos.sum().item()}")
    
    assert torch.allclose(limited_neg, -limited_pos, atol=1e-12), "Odd symmetry failed!"
    print("✓ Odd symmetry VERIFIED\n")


def test_cell_average_conservation_weno():
    """Verify that WENO limiter preserves cell averages exactly.
    
    The limiter only modifies modes >= 1, so mode-0 (cell average) must
    be unchanged bit-for-bit.
    """
    print("=" * 60)
    print("CHECK 2: WENO Limiter Cell Average Conservation")
    print("=" * 60)
    
    # Create test data with varying cell averages
    ncells = 16
    coeff = torch.zeros(ncells, 3, 2)
    # Varying cell averages
    coeff[:, 0, 0] = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 2.8, 2.6, 2.4, 
                                    2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8])
    coeff[:, 0, 1] = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.8, 0.75,
                                    0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35])
    # Non-trivial higher modes
    coeff[:, 1, :] = 0.3 * coeff[:, 0, :]
    coeff[:, 2, :] = -0.1 * coeff[:, 0, :]
    
    averages_original = coeff[:, 0, :].clone()
    limited = apply_weno_limiter(coeff, 1.0/ncells, tvb_constant=0.0)
    
    # Check conservation
    avg_preserved = torch.equal(limited[:, 0], averages_original)
    max_avg_diff = (limited[:, 0] - averages_original).abs().max()
    
    print(f"Cell averages preserved exactly: {avg_preserved}")
    print(f"Max mode-0 difference: {max_avg_diff:.2e}")
    
    # Check that higher modes were actually modified (limiter is active)
    higher_modes_changed = (limited[:, 1:] != coeff[:, 1:]).any()
    print(f"Higher modes modified (limiter active): {higher_modes_changed}")
    
    assert avg_preserved, "Cell average conservation failed!"
    print("✓ Cell average conservation VERIFIED\n")


def test_weno_weights_invariance():
    """Verify WENO-Z weight invariance under rescaling of smoothness indicators.
    
    WENO-Z uses ratios of indicators, so w(c*beta) == w(beta).
    JS weights are NOT invariant because epsilon fixes an absolute scale.
    """
    print("=" * 60)
    print("CHECK 3: WENO-Z Weight Scale Invariance")
    print("=" * 60)
    
    torch.manual_seed(1)
    beta = (torch.rand(3, 32) + 0.05) * 1e-3
    
    # Test with different scales
    scales = [1e-8, 1e-4, 1e4, 1e8]
    ideal = (1/6, 2/3, 1/6)
    
    for scale in scales:
        z_weights = _weno_weights(beta[0], beta[1], beta[2], variant="z")
        z_scaled = _weno_weights(*(scale * b for b in beta), variant="z")
        
        z_diff = max((a - b).abs().max().item() for a, b in zip(z_weights, z_scaled))
        
        js_weights = _weno_weights(beta[0], beta[1], beta[2], variant="js")
        js_scaled = _weno_weights(*(scale * b for b in beta), variant="js")
        js_diff = max((a - b).abs().max().item() for a, b in zip(js_weights, js_scaled))
        
        print(f"Scale {scale:8.0e}: Z-diff={z_diff:.2e}, JS-diff={js_diff:.2e}")
        
        assert z_diff < 1e-10, f"Z weights not invariant at scale {scale}"
        if scale != 1.0:
            assert js_diff > 1e-3, f"JS weights should drift at scale {scale}"
    
    print("✓ WENO-Z scale invariance VERIFIED\n")


def test_weno_slope_boundedness():
    """Verify that WENO limited slope is bounded by the larger one-sided increment.
    
    For a troubled cell, the WENO slope is a convex combination of three candidate
    slopes, so it must satisfy |s_weno| <= max(|s_left|, |s_right|).
    In modal form: |c_1| <= max(|d_left|, |d_right|) / 2.
    """
    print("=" * 60)
    print("CHECK 4: WENO Limiter Slope Boundedness")
    print("=" * 60)
    
    ncells = 25
    dx = 0.04
    coeff = torch.zeros(ncells, 3, 1)
    
    # Create a step function with a discontinuity
    avg = torch.cumsum(torch.rand(ncells), dim=0)
    avg[10:] += 2.0 * torch.rand(ncells - 10)
    coeff[:, 0, 0] = avg
    
    # Compute one-sided increments
    padded = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    d_left = (avg - padded[:-2])[:, None]
    d_right = (padded[2:] - avg)[:, None]
    bound = 0.5 * torch.maximum(d_left.abs(), d_right.abs())
    
    limited = apply_weno_limiter(coeff, dx, tvb_constant=0.0)
    
    # Check boundedness for all cells
    bounded = (limited[:, 1].abs() <= bound + 1e-12).all()
    max_ratio = (limited[:, 1].abs() / bound.clamp(min=1e-15)).max().item()
    
    print(f"All cells respect minmod bound: {bounded}")
    print(f"Max |c_1| / bound ratio: {max_ratio:.4f}")
    
    assert bounded, "Slope boundedness failed!"
    print("✓ Slope boundedness VERIFIED\n")


if __name__ == "__main__":
    test_odd_symmetry_weno()
    test_cell_average_conservation_weno()
    test_weno_weights_invariance()
    test_weno_slope_boundedness()
    
    print("=" * 60)
    print("ALL CHECKS PASSED - WENO limiter properties confirmed")
    print("=" * 60)
