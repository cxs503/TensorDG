"""Well-balancedness regression tests for the SWE bed-slope source term.

These tests pin the bed-slope source normalization of both solvers.  The
pre-2026-11 implementation carried a spurious ``2/(2k+1)`` modal factor (and,
in 2-D, a further ``1/2`` / ``dx <-> dy`` mix-up in the modal prefactor), which
made the momentum source term wrong by up to 2x on the mean mode: a lake at
rest on a sloping bed then diverged exponentially (|eta - eta_0| ~ 5e12 after
t = 2 s) instead of staying at rest.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg.swe_1d import reconstruct_fields, solve_swe_1d
from tensordg.swe_2d import solve_swe_2d

G = 9.81


def _lake_at_rest_1d(ncells=64, final_time=1.0, degree=1, sloped=True):
    eta0 = 2.0
    zb = (lambda xx: 0.4 * torch.sin(math.pi * xx / 2.0)) if sloped \
        else (lambda xx: torch.zeros_like(xx))
    x, coeff, zb_faces, meta = solve_swe_1d(
        lambda xx: eta0 - zb(xx), lambda xx: torch.zeros_like(xx), zb,
        ncells=ncells, degree=degree, final_time=final_time,
        domain=(-2.0, 2.0), cfl=0.3)
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    return eta, coeff, meta, eta0


@pytest.mark.parametrize("degree", [1, 2])
def test_flat_bed_lake_at_rest_is_machine_precise(degree):
    eta, coeff, _, eta0 = _lake_at_rest_1d(final_time=1.0, degree=degree, sloped=False)
    assert float((eta - eta0).abs().max()) < 1e-11
    assert float(coeff[:, 0, 1].abs().max()) < 1e-11


@pytest.mark.parametrize("degree", [1, 2])
def test_sloped_bed_lake_at_rest_stays_finite(degree):
    """Regression guard for the 2/(2k+1) source factor.

    With the buggy normalization this test diverges to ~1e12 (and then to
    inf); with the fix the state stays bounded and the drift is below 0.05%
    of the still-water depth (exact well-balancedness on a non-flat bed would
    need Audusse hydrostatic reconstruction, not implemented yet).
    """
    eta, coeff, meta, eta0 = _lake_at_rest_1d(final_time=1.0, degree=degree)
    drift = float((eta - eta0).abs().max())
    assert np.isfinite(drift)
    assert drift < 5e-4 * eta0, f"eta drift {drift:.3e} m"
    assert float(coeff[:, 0, 1].abs().max()) < 1e-2, "spurious momentum"


def test_sloped_bed_drift_does_not_grow_between_two_and_one_times():
    """Drift must stay bounded in time (blow-up guard)."""
    eta_a, _, _, eta0 = _lake_at_rest_1d(final_time=0.5)
    eta_b, _, _, _ = _lake_at_rest_1d(final_time=1.0)
    drift_a = float((eta_a - eta0).abs().max())
    drift_b = float((eta_b - eta0).abs().max())
    assert drift_b < 5e-4 * eta0
    assert drift_b < 50.0 * max(drift_a, 1e-12) or drift_b < 1e-6


def test_2d_flat_bed_lake_at_rest_is_machine_precise():
    """The 2-D solver had a broken projection/trace/prefactor; this pins it."""
    eta0 = 2.0
    flat = lambda X, Y: torch.zeros_like(X + Y)
    x, y, coeff, zb_faces, meta = solve_swe_2d(
        lambda X, Y: eta0 - flat(X, Y), lambda X, Y: torch.zeros_like(X + Y),
        lambda X, Y: torch.zeros_like(X + Y), flat,
        nx=16, ny=16, degree=1, final_time=0.5,
        domain=((-2.0, 2.0), (-2.0, 2.0)), cfl=0.2)
    eta = coeff[:, :, 0, 0, 0] + 0.5 * (zb_faces["x"][0] + zb_faces["x"][1])
    assert float((eta - eta0).abs().max()) < 1e-11
    assert float(coeff[:, :, :, :, 1].abs().max()) < 1e-11
    assert meta["steps"] > 10


def test_2d_constant_state_has_zero_residual():
    """A constant state must be an exact steady state of the 2-D residual."""
    from tensordg.swe_2d import _rhs_2d, legendre_operators_2d

    nx = ny = 6
    degree = 1
    nodes, w, phi, dphi, pl, pr = legendre_operators_2d(degree)
    nm = degree + 1
    coeff = torch.zeros(nx, ny, nm, nm, 3, dtype=torch.float64)
    coeff[:, :, 0, 0, 0] = 2.0
    zeros = torch.zeros(nx, ny, dtype=torch.float64)
    r = _rhs_2d(coeff, 1.0 / nx, 1.0 / ny, w, phi, dphi, pl, pr, zeros, zeros,
                (False, False))
    assert float(r.abs().max()) < 1e-10


def test_2d_uniform_state_projects_onto_the_mean_mode_only():
    """The L2 projection must not populate higher modes for a constant field."""
    from tensordg.swe_2d import legendre_operators_2d

    nx = ny = 5
    nodes, w, phi, dphi, pl, pr = legendre_operators_2d(1)
    coeff = torch.zeros(nx, ny, 2, 2, 3, dtype=torch.float64)
    h0 = torch.full((nx, ny, len(nodes), len(nodes)), 2.0, dtype=torch.float64)
    for k in range(2):
        for l in range(2):
            coeff[:, :, k, l, 0] = ((2 * k + 1) / 2) * ((2 * l + 1) / 2) * torch.einsum(
                "xyab,a,b,a,b->xy", h0, w, w, phi[:, k], phi[:, l])
    assert torch.allclose(coeff[:, :, 0, 0, 0], torch.full((nx, ny), 2.0,
                                                           dtype=torch.float64))
    assert float(coeff[:, :, 1, 1, 0].abs().max()) < 1e-12