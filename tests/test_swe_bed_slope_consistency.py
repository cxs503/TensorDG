"""Bed-slope source term: cell-constant face difference vs exact quadrature.

Defect under test
-----------------
``_rhs`` historically used a **cell-constant** bed slope
``dbdx = (z_b,R - z_b,L)/dx`` for the momentum source ``-g h dz_b/dx``.
For a piecewise-linear bed that is exact, but for a *curved* bed it is only the
cell average of ``dz_b/dx``; the missing linear part of the slope multiplies the
DG solution and leaves an ``O(dx)`` error in the discrete force that does **not**
shrink with the polynomial degree ``p``.  On the Thacker (1981) planar sloshing
benchmark the resulting force error shifts the sloshing frequency, so the
long-time error saturates instead of converging:

    t = 0.5 s, P1, N = 20: eta rel. L2 = 11.93 %   (legacy)
    t = 0.5 s, P1, N = 20: eta rel. L2 =  0.13 %   (exact, this option)

Mission m6377 (`docs/thacker_validation_gap_corrected.md` §6) had already
observed the fingerprint -- "mass residual machine zero, momentum residual O(dx)"
-- and left the cause open; these tests pin the cause and the fix.

The fix is an opt-in keyword ``solve_swe_1d(..., zb_slope=callable)`` which
samples ``dz_b/dx`` at the same Gauss points as ``h`` so the source quadrature is
exact for polynomial data.  For a *constant* slope it reduces algebraically to
``-g dbdx c_k``, i.e. to the legacy branch (tested below), so the change is a
strict generalisation rather than a re-normalisation.

Boundary closure: all Thacker runs below use the analytic boundary fluxes
(``bc_left_flux``/``bc_right_flux`` from ``swe_exact``), i.e. the verification
configuration.  With the default transmissive closure the exact-source run loses
positivity before ``t = 0.5`` s, the same boundary/truncation-state interaction
mission m6377 reported for the legacy path (§6c); it is a separate defect and is
explicitly *not* claimed fixed here.

Known limitation, pinned by test 5: the exact-quadrature source does **not**
preserve the discrete lake-at-rest cancellation on a curved bed (the legacy
constant source is what accidentally provides it).  Enabling it is therefore a
choice for smooth-bed *unsteady* verification; the well-balanced/consistent
unification requires Audusse hydrostatic reconstruction at the interfaces
(see ``docs/tensordg-swe-bed-slope-source-fix.md`` §6).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg import swe_exact
from tensordg.swe_1d import _operators, _rhs, reconstruct_fields, solve_swe_1d

G, H0, AMP, L = 9.81, 1.0, 0.5, 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _thacker_run(ncells, degree, t_end, exact_source, exact_bc=True):
    def init_h(x):
        return torch.as_tensor(
            swe_exact.thacker_planar_basin_initial(x.numpy(), H0, AMP, L)[0], dtype=x.dtype)

    def zb(x):
        return torch.as_tensor(swe_exact.thacker_parabolic_bed(x.numpy(), H0, L), dtype=x.dtype)

    def zb_slope(x):
        return torch.as_tensor(
            swe_exact.thacker_parabolic_bed_slope(x.numpy(), H0, L), dtype=x.dtype)

    kw = {}
    if exact_source:
        kw["zb_slope"] = zb_slope
    if exact_bc:
        # Analytic boundary fluxes: the verification configuration.  The default
        # transmissive closure is a separate defect (m6377 §6c) and is not what
        # this test is about.
        kw["bc_left_flux"] = lambda t: torch.as_tensor(
            swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)[0])
        kw["bc_right_flux"] = lambda t: torch.as_tensor(
            swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)[1])
    x, coeff, zb_faces, meta = solve_swe_1d(
        init_h, lambda z: torch.zeros_like(z), zb, ncells=ncells, degree=degree,
        final_time=t_end, domain=(-L, L), cfl=0.3, flux_kind="hll", limiter=None,
        rk_order=3, dtype=torch.float64, **kw)
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    h_e, u_e, eta_e = swe_exact.thacker_planar_basin(x.numpy(), t_end, H0, AMP, L, G)
    exact = tuple(torch.as_tensor(a, dtype=torch.float64) for a in (h_e, u_e, eta_e))
    rel = lambda a, b: float(torch.sqrt(torch.mean((a - b) ** 2)) / b.abs().max()) * 100.0
    return rel(eta, exact[2]), rel(h, exact[0]), rel(u, exact[1]), meta


# ---------------------------------------------------------------------------
# 1. constant slope: the new branch must reproduce the legacy branch exactly
# ---------------------------------------------------------------------------

def test_exact_branch_reduces_to_legacy_for_constant_slope():
    torch.manual_seed(0)
    ncells, degree = 6, 2
    nodes, weights, phi, dphi, phi_left, phi_right = _operators(
        degree, torch.device("cpu"), torch.float64)
    coeff = torch.randn(ncells, degree + 1, 2, dtype=torch.float64)
    coeff[:, 0, 0] = 1.0 + 0.1 * torch.rand(ncells, dtype=torch.float64)  # keep h > 0
    zb_jump = torch.full((ncells,), 0.37, dtype=torch.float64)
    slope = zb_jump / 0.5
    quad = torch.full((ncells, nodes.shape[0]), float(slope[0]), dtype=torch.float64)

    legacy = _rhs(coeff, 0.5, weights, phi, dphi, phi_left, phi_right, zb_jump, periodic=True)
    exact = _rhs(coeff, 0.5, weights, phi, dphi, phi_left, phi_right, zb_jump, periodic=True,
                 zb_slope_quad=quad)
    assert float((legacy - exact).abs().max()) < 1e-12


# ---------------------------------------------------------------------------
# 2. p-consistency: the exact source is what makes the P2 solution exact
# ---------------------------------------------------------------------------

def test_p2_uniform_velocity_is_represented_when_source_is_exact():
    """The Thacker solution has a *spatially uniform* velocity field.

    With ``p = 2`` the exact solution (quadratic ``h``, uniform ``u``) lies in
    the discrete space, so the numerical-flux dissipation vanishes and the only
    remaining error source is the bed-slope quadrature.  Legacy: ~22 % velocity
    error at 0.1 s, non-vanishing.  Exact: round-off.
    """
    _, _, u_legacy, _ = _thacker_run(20, 2, 0.1, exact_source=False)
    eta_exact, h_exact, u_exact, meta = _thacker_run(20, 2, 0.1, exact_source=True)

    assert u_legacy > 5.0, f"legacy uniform-velocity error {u_legacy:.3f} %"
    assert u_exact < 1e-3, f"exact-source uniform-velocity error {u_exact:.3e} %"
    assert eta_exact < 0.5, f"exact-source eta error {eta_exact:.3f} %"
    assert meta["exact_bed_slope"] is True


# ---------------------------------------------------------------------------
# 3. long-time acceptance gate on the benchmark mission m6377 could not close
# ---------------------------------------------------------------------------

def test_thacker_long_time_meets_three_percent_with_exact_source():
    """t = 0.5 s (~1/3 period) was the open ~12 % non-converging gap."""
    eta_legacy, _, u_legacy, _ = _thacker_run(20, 1, 0.5, exact_source=False)
    eta_exact, h_exact, u_exact, _ = _thacker_run(20, 1, 0.5, exact_source=True)

    assert eta_legacy > 5.0, f"legacy eta error {eta_legacy:.3f} % (expected > 5 %)"
    assert eta_exact < 3.0, f"exact-source eta error {eta_exact:.3f} %"
    assert h_exact < 3.0, f"exact-source h error {h_exact:.3f} %"
    assert u_exact < 3.0, f"exact-source u error {u_exact:.3f} %"


# ---------------------------------------------------------------------------
# 4. h-refinement must reduce the error (legacy does not)
# ---------------------------------------------------------------------------

def test_exact_source_restores_grid_convergence():
    e20 = _thacker_run(20, 1, 0.5, exact_source=True)[0]
    e40 = _thacker_run(40, 1, 0.5, exact_source=True)[0]
    ratio = e20 / e40
    order = math.log(ratio) / math.log(2.0)
    assert ratio > 2.5, f"error ratio {ratio:.2f} (EOC {order:.2f}) -- expected ~4 for P1"

    # legacy: either the run blows up (m6377 §6c) or the error barely moves
    try:
        l20 = _thacker_run(20, 1, 0.5, exact_source=False)[0]
        l40 = _thacker_run(40, 1, 0.5, exact_source=False)[0]
        assert l20 / l40 < 1.5, f"legacy ratio {l20 / l40:.2f} should not converge"
    except FloatingPointError:
        pass  # legacy path loses positivity on the finer grid -- also a defect


# ---------------------------------------------------------------------------
# 5. honest limitation: exact source breaks lake-at-rest on a curved bed
# ---------------------------------------------------------------------------

def _lake_at_rest_drift(degree, exact_source):
    eta0 = 2.0
    zb = lambda xx: 0.4 * torch.sin(math.pi * xx / 2.0)
    zb_slope = lambda xx: 0.4 * (math.pi / 2.0) * torch.cos(math.pi * xx / 2.0)
    kw = {"zb_slope": zb_slope} if exact_source else {}
    _, coeff, zb_faces, _ = solve_swe_1d(
        lambda xx: eta0 - zb(xx), lambda xx: torch.zeros_like(xx), zb,
        ncells=64, degree=degree, final_time=1.0, domain=(-2.0, 2.0), cfl=0.3, **kw)
    _, _, eta = reconstruct_fields(coeff, zb_faces)
    return float((eta - eta0).abs().max())


def test_exact_source_does_not_preserve_lake_at_rest_on_curved_bed():
    """Pins the trade-off so the option is never enabled blindly.

    The legacy constant source is the mechanism behind the (approximate)
    rest balance; replacing it by exact quadrature destroys that cancellation.
    """
    drift_legacy = _lake_at_rest_drift(1, exact_source=False)
    drift_exact = _lake_at_rest_drift(1, exact_source=True)
    assert drift_legacy < 5e-3, f"legacy rest drift {drift_legacy:.3e} m"
    assert drift_exact > drift_legacy, "exact source must not be claimed well-balanced"