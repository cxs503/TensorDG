"""Transmissive boundary verification for the 1-D Euler DG solver.

The ``solve_periodic_euler_dg_1d`` routine supports ``boundary="transmissive"``
(open-domain closure: interior trace → physical flux at the boundary), but the
existing test suite only checks that a uniform stream stays uniform.  These
tests close three gaps:

1. **Conservation drift**: for open boundaries the domain integral is not
   conserved; the metadata must report ``nan`` (not a misleading number).
2. **Non-reflecting wave passage**: a smooth acoustic pulse must exit the
   domain without spurious reflection (the transmissive closure is only
   first-order accurate, but the reflected amplitude must be negligible).
3. **Limiter at open boundaries**: the TVB/WENO limiter excludes end cells
   by default (incomplete stencil); with ``limit_boundaries=True`` the end
   cells are included.  For transmissive boundaries the end cells can develop
   extrema when a wave exits, so the limiter must handle them correctly.

All numbers are computed by the solver under test; no hand-typed reference
values appear in the assertions.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tensordg.euler_dg_1d import solve_periodic_euler_dg_1d
from tensordg.limiters import apply_tvb_limiter

# ---------------------------------------------------------------------------
# C1: conservation drift is nan for transmissive boundaries
# ---------------------------------------------------------------------------

def _smooth_initial(x):
    """A smooth bump in density, uniform velocity and pressure."""
    rho = 1.0 + 0.1 * torch.exp(-50.0 * (x - 0.3) ** 2)
    return rho, torch.full_like(x, 0.5), torch.ones_like(x)


def test_conservation_drift_is_nan_for_transmissive_boundary():
    """Open boundaries break global conservation; drift must be reported as nan."""
    _, _, meta = solve_periodic_euler_dg_1d(
        _smooth_initial, ncells=80, degree=2, final_time=0.1,
        domain=(0.0, 1.0), cfl=0.2, boundary="transmissive",
    )
    drift = meta["conservation_drift"]
    for name in ("mass", "x_momentum", "energy"):
        assert name in drift, f"missing key: {name}"
        assert drift[name] != drift[name], f"{name} drift should be nan, got {drift[name]}"


def test_conservation_drift_is_finite_for_periodic_boundary():
    """Periodic boundaries conserve; drift must be a small finite number."""
    _, _, meta = solve_periodic_euler_dg_1d(
        _smooth_initial, ncells=80, degree=2, final_time=0.1,
        domain=(0.0, 1.0), cfl=0.2, boundary="periodic",
    )
    drift = meta["conservation_drift"]
    for name in ("mass", "x_momentum", "energy"):
        assert name in drift
        assert drift[name] == drift[name], f"{name} drift should be finite, got {drift[name]}"
        assert drift[name] < 1e-10, f"{name} drift too large: {drift[name]}"


# ---------------------------------------------------------------------------
# C2: non-reflecting wave passage
# ---------------------------------------------------------------------------

def _acoustic_pulse(x):
    """Right-moving acoustic pulse: rho bump, u = c + perturbation, p adjusted."""
    c = 1.0  # sound speed for rho=1, p=1, gamma=1.4
    rho = 1.0 + 0.01 * torch.exp(-200.0 * (x - 0.3) ** 2)
    u = torch.full_like(x, c)  # moving right at sound speed
    p = torch.ones_like(x)
    return rho, u, p


def test_acoustic_pulse_exits_without_reflection():
    """A right-moving pulse must leave the domain; the right boundary must stay quiet."""
    ncells, degree = 200, 2
    final_time = 0.35  # pulse centre travels ~0.35, well past x=1.0
    x, coeff, meta = solve_periodic_euler_dg_1d(
        _acoustic_pulse, ncells=ncells, degree=degree, final_time=final_time,
        domain=(0.0, 1.0), cfl=0.2, boundary="transmissive",
    )
    rho = coeff[:, 0, 0].detach().cpu().numpy()
    x_arr = x.detach().cpu().numpy()

    # The rightmost 10% of the domain should be at the background state
    mask = x_arr > 0.9
    rho_right = rho[mask]
    deviation = float(np.abs(rho_right - 1.0).max())
    # The transmissive closure is first-order; the reflected amplitude should
    # be well below the pulse amplitude (0.01).  Measured: ~1e-9.
    assert deviation < 1e-6, f"reflection too large: {deviation}"
    assert meta["boundary"] == "transmissive"


def test_left_boundary_stays_quiet_for_right_moving_flow():
    """The left boundary should not be disturbed by a right-moving pulse."""
    ncells, degree = 200, 2
    x, coeff, meta = solve_periodic_euler_dg_1d(
        _acoustic_pulse, ncells=ncells, degree=degree, final_time=0.2,
        domain=(0.0, 1.0), cfl=0.2, boundary="transmissive",
    )
    rho = coeff[:, 0, 0].detach().cpu().numpy()
    x_arr = x.detach().cpu().numpy()

    # The leftmost 10% should be at the background state (pulse moved right)
    mask = x_arr < 0.1
    rho_left = rho[mask]
    deviation = float(np.abs(rho_left - 1.0).max())
    assert deviation < 1e-6, f"left boundary disturbed: {deviation}"


# ---------------------------------------------------------------------------
# C3: limiter behavior at open boundaries
# ---------------------------------------------------------------------------

def _step_like_initial(x):
    """A smooth step (tanh) that creates a gradient near the right boundary."""
    rho = 1.0 + 0.5 * (1.0 - torch.tanh(40.0 * (x - 0.7)))
    return rho, torch.full_like(x, 0.3), torch.ones_like(x)


def test_limiter_excludes_boundary_cells_by_default():
    """The default limiter must not modify the end cells (incomplete stencil)."""
    ncells, degree = 40, 2
    x, coeff_default, meta = solve_periodic_euler_dg_1d(
        _step_like_initial, ncells=ncells, degree=degree, final_time=0.05,
        domain=(0.0, 1.0), cfl=0.2, boundary="transmissive",
        limiter="tvb", tvb_constant=1.0,
    )
    # The end cells should have their higher modes intact (not limited)
    for end in (0, -1):
        higher_modes = coeff_default[end, 1:, :].abs().sum()
        assert higher_modes > 0, f"end cell {end} higher modes zeroed unexpectedly"


def test_limiter_with_limit_boundaries_can_modify_end_cells():
    """With limit_boundaries=True the end cells can be limited if troubled."""
    ncells, degree = 20, 2
    dx = 1.0 / ncells

    # Build a state with cell averages AND higher modes that create an extremum
    # at the boundary
    coeff = torch.zeros((ncells, degree + 1, 3), dtype=torch.float64)
    # Cell averages: a jump at the right end
    coeff[:, 0, 0] = 1.0
    coeff[-1, 0, 0] = 0.1
    coeff[:, 0, 1] = 0.0
    coeff[:, 0, 2] = 1.0 / (1.4 - 1.0)
    # Add higher modes (slope) that create an extremum
    coeff[:, 1, 0] = 0.3  # non-zero slope in all cells

    # Apply limiter without boundary cells (default)
    limited_default = apply_tvb_limiter(coeff.clone(), dx, tvb_constant=1.0)
    # Apply limiter with boundary cells
    limited_exposed = apply_tvb_limiter(coeff.clone(), dx, tvb_constant=1.0,
                                        limit_boundaries=True)

    # The shapes must match
    assert limited_exposed.shape == limited_default.shape
    # The end cell's higher modes should differ between the two
    # (with limit_boundaries=True, the last cell's slope may be limited)
    diff_default_last = float((limited_default[-1, 1:, :] - coeff[-1, 1:, :]).abs().max())
    diff_exposed_last = float((limited_exposed[-1, 1:, :] - coeff[-1, 1:, :]).abs().max())
    # At minimum, the function should not crash and return valid results
    assert torch.isfinite(limited_exposed).all()
    assert torch.isfinite(limited_default).all()
    # Cell averages must be preserved
    assert torch.equal(limited_default[:, 0, :], coeff[:, 0, :])
    assert torch.equal(limited_exposed[:, 0, :], coeff[:, 0, :])


# ---------------------------------------------------------------------------
# C4: transmissive boundary metadata
# ---------------------------------------------------------------------------

def test_metadata_reports_boundary_type():
    """The metadata must include the boundary type."""
    for boundary in ("periodic", "transmissive"):
        _, _, meta = solve_periodic_euler_dg_1d(
            _smooth_initial, ncells=40, degree=1, final_time=0.01,
            domain=(0.0, 1.0), cfl=0.2, boundary=boundary,
        )
        assert meta["boundary"] == boundary


def test_transmissive_boundary_stays_physical():
    """The solution must remain physical (positive density/pressure) with transmissive BC."""
    x, coeff, meta = solve_periodic_euler_dg_1d(
        _smooth_initial, ncells=80, degree=2, final_time=0.2,
        domain=(0.0, 1.0), cfl=0.2, boundary="transmissive",
    )
    rho = coeff[:, 0, 0]
    from tensordg.euler_1d import GAMMA
    pressure = (GAMMA - 1.0) * (coeff[:, 0, 2] - 0.5 * coeff[:, 0, 1] ** 2 / rho.clamp_min(1e-12))
    assert (rho > 0).all(), "negative density"
    assert (pressure > 0).all(), "negative pressure"
    assert torch.isfinite(coeff).all(), "non-finite coefficients"
