"""Well-balanced (Audusse hydrostatic reconstruction) mode of the 1-D SWE solver.

The default 1-D SWE discretisation carries a *bed-shape imbalanced* source term:
the momentum source uses a quadrature of ``dz_b/dx`` that is inconsistent with
the pressure flux of the weak form, so a lake at rest on a curved bed is not a
discrete fixed point (measured drift at N=64, t=1 s: 3.2e-4 m at P1 and 0.645 m
= 32 % of the still-water depth at P2 -- see
``docs/tensordg-swe-bed-slope-source-fix.md`` sec. 5/6, where the missing
capability is named as Audusse hydrostatic reconstruction).

``solve_swe_1d(..., well_balanced=True)`` implements that reconstruction:

* every interior face is reconstructed onto ``b* = max(b_L, b_R)`` with
  ``h* = max(0, h + b - b*)`` and the one-sided hydrostatic pressure correction
  ``g/2 (h^2 - h*^2)`` is added back to that side's momentum flux;
* the bed-slope source is discretised with the derivative of the *per-cell bed
  polynomial* sampled at the same Gauss points as ``h``.

Consequence: for a lake at rest ``h + z_b = const, hu = 0`` the semi-discrete
residual vanishes identically at every polynomial degree, so the discrete state
is stationary to round-off (measured: 1.8e-12 at P1, 9.6e-11 at P2 for
N = 64, t = 1 s, 777/1294 RK3 steps).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg.swe_1d import reconstruct_fields, solve_swe_1d
from tensordg import swe_exact

G = 9.81
ETA0 = 2.0


def _curved_bed(xx: torch.Tensor) -> torch.Tensor:
    return 0.4 * torch.sin(math.pi * xx / 2.0)


def _lake_at_rest(degree: int, ncells: int = 64, final_time: float = 1.0,
                  well_balanced: bool = True):
    return solve_swe_1d(
        lambda xx: ETA0 - _curved_bed(xx),
        lambda xx: torch.zeros_like(xx),
        _curved_bed,
        ncells=ncells, degree=degree, final_time=final_time,
        domain=(-2.0, 2.0), cfl=0.3, well_balanced=well_balanced,
    )


@pytest.mark.parametrize("degree", [1, 2])
def test_curved_bed_lake_at_rest_is_a_discrete_fixed_point(degree):
    """Rest state must be stationary to round-off -- not merely bounded."""
    coeff0 = _lake_at_rest(degree, final_time=0.0)[1]
    coeff1 = _lake_at_rest(degree, final_time=1.0)[1]

    mass_change = float((coeff1[:, :, 0] - coeff0[:, :, 0]).abs().max())
    momentum = float(coeff1[:, :, 1].abs().max())

    # Round-off floor: 1.8e-12 (P1) / 9.6e-11 (P2) over ~1e3 RK3 stages.
    assert mass_change < 1e-9, f"depth drifted by {mass_change:.3e}"
    assert momentum < 1e-9, f"spurious momentum {momentum:.3e}"


def test_default_mode_is_unchanged_and_still_drifts_at_p2():
    """Characterisation of the default path (regression guard, not a fix).

    The default 1-D discretisation still lacks the reconstruction, so the P2
    rest state on this bed is not a fixed point.  This test pins the *status quo*
    so that a future default-on switch of the well-balanced mode is a deliberate,
    visible change rather than a silent one.
    """
    momentum = float(_lake_at_rest(2, well_balanced=False)[1][:, :, 1].abs().max())
    assert momentum > 1e-2, f"default path unexpectedly balanced ({momentum:.3e})"


def test_piecewise_linear_bed_reduces_to_the_default_discretisation():
    """On a linear bed the reconstruction is the identity -> identical solver.

    ``b_L = b_R`` at interior faces and the polynomial bed slope equals the
    cell-constant slope, so every term of the residual reduces algebraically to
    the default branch.  Agreement is required to round-off so that enabling the
    mode cannot silently change well-resolved flat/linear-bed results.
    """
    def zb(xx):
        return 0.25 * xx

    kw = dict(ncells=32, degree=1, final_time=0.05, domain=(-2.0, 2.0), cfl=0.3)
    args = (lambda xx: ETA0 - zb(xx), lambda xx: 0.1 * torch.ones_like(xx), zb)
    c_def = solve_swe_1d(*args, **kw)[1]
    c_wb = solve_swe_1d(*args, well_balanced=True, **kw)[1]
    assert float((c_def - c_wb).abs().max()) < 1e-11


def test_zb_slope_and_well_balanced_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        solve_swe_1d(
            lambda xx: ETA0 - _curved_bed(xx), lambda xx: torch.zeros_like(xx),
            _curved_bed, ncells=16, degree=1, final_time=0.01,
            domain=(-2.0, 2.0), well_balanced=True,
            zb_slope=lambda xx: 0.4 * (math.pi / 2.0) * torch.cos(math.pi * xx / 2.0),
        )


@pytest.mark.parametrize("rk_order", [2, 3])
def test_analytic_boundary_flux_is_used_at_every_rk_stage(rk_order):
    """Thacker basin: the water volume may only change through the boundary.

    With the analytic boundary flux imposed the numerical volume must equal the
    analytic value ``2 eta_c + 4/3 h0 L`` to round-off.  The SSP-RK2 stage-2
    residual used to be evaluated with ``time = 0`` and *without* the boundary
    callbacks (it silently fell back to the transmissive closure), which broke
    volume conservation at O(dt) -- this test locks the fix for both orders.
    """
    h0, amp, half = 1.0, 0.5, 1.0
    t_end = 0.2

    def bed(xx):
        return torch.as_tensor(swe_exact.thacker_parabolic_bed(xx.numpy(), h0, half),
                               dtype=xx.dtype)

    def left(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(
            t, h0, amp, half, G)[0], dtype=torch.float64)

    def right(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(
            t, h0, amp, half, G)[1], dtype=torch.float64)

    x, coeff, zb_faces, meta = solve_swe_1d(
        lambda xx: torch.as_tensor(
            swe_exact.thacker_planar_basin_initial(xx.numpy(), h0, amp, half)[0],
            dtype=xx.dtype),
        lambda xx: torch.zeros_like(xx), bed,
        ncells=20, degree=1, final_time=t_end, domain=(-half, half),
        cfl=0.3, rk_order=rk_order, bc_left_flux=left, bc_right_flux=right,
    )
    dx = 2.0 * half / 20
    volume = float(coeff[:, 0, 0].sum()) * dx
    omega = math.sqrt(2.0 * G * h0) / half
    eta_c = h0 + amp ** 2 / (4.0 * h0) * math.sin(omega * t_end) ** 2
    volume_exact = 2.0 * eta_c + (4.0 / 3.0) * h0 * half
    # Tolerances are stage-quadrature limited: the Heun flux average is a
    # trapezoid rule for the boundary flux integral over the step (O(dt^2) per
    # step -> 5.6e-7 relative over this run), SSP-RK3 is Simpson-like (1e-13).
    # The defect being guarded against is O(dt) = 1e-3 relative, because the
    # stage-2 residual used the *transmissive* closure instead of the analytic
    # flux.
    tol = 1e-5 if rk_order == 2 else 1e-10
    assert abs(volume - volume_exact) / volume_exact < tol, (
        f"rk{rk_order}: volume {volume:.12f} vs exact {volume_exact:.12f}")