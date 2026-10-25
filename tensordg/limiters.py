"""TVB and WENO limiters for high-order DG Euler solvers.

Implements two shock-capturing strategies for modal DG:

1. **TVB limiter** (Cockburn & Shu, 1989): detects troubled cells via a
   modified minmod test with a tolerance ``M * dx**2``.  Troubled cells are
   reduced to a linear reconstruction whose slope is limited by minmod of
   the neighbouring cell-average differences.

2. **WENO limiter** (Qiu & Shu, 2005): detects troubled cells via the same
   jump indicator, then replaces the high-order polynomial inside the
   troubled cell by a WENO reconstruction built from the cell averages of
   the cell and its neighbours.

Both limiters preserve the cell average (mode-0 coefficient) and only
modify higher modes.  They are designed to be called after each SSP-RK
sub-step, together with the existing positivity filter.
"""
from __future__ import annotations

import torch

from .euler_1d import GAMMA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise minmod: same-sign selection with smallest magnitude."""
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(a.abs(), b.abs())


def _troubled_cell_indicator(
    coeff: torch.Tensor,
    dx: float,
    tvb_constant: float = 0.0,
) -> torch.Tensor:
    """Return a boolean mask [ncells] of cells that need limiting.

    The indicator fires when the cell-average sequence is *not* monotone
    (i.e. a local extremum) **and** the variation exceeds the TVB
    tolerance ``M * dx**2``.  With ``M = 0`` this reduces to the strict
    TVD test; positive ``M`` allows smooth extrema to pass.
    """
    ncells = coeff.shape[0]
    avg = coeff[:, 0]  # [ncells, nvar]
    # Pad with copies of boundary cells for one-sided stencils
    avg_pad = torch.cat([avg[:1], avg, avg[-1:]], dim=0)  # [ncells+2, nvar]
    d_left = avg - avg_pad[:-2]   # u_i - u_{i-1}
    d_right = avg_pad[2:] - avg   # u_{i+1} - u_i

    # Variation measure: max absolute jump across the two interfaces
    variation = torch.maximum(d_left.abs(), d_right.abs()).max(dim=-1).values  # [ncells]

    # Not-monotone: minmod(d_left, d_right) != d_left for at least one variable
    mm = _minmod(d_left, d_right)
    not_monotone = ~((mm - d_left).abs().max(dim=-1).values < 1e-14)

    threshold = tvb_constant * dx * dx
    troubled = not_monotone & (variation > threshold)
    # Boundary cells are never troubled (no full stencil)
    troubled[0] = False
    troubled[-1] = False
    return troubled


# ---------------------------------------------------------------------------
# TVB limiter
# ---------------------------------------------------------------------------

def apply_tvb_limiter(
    coeff: torch.Tensor,
    dx: float,
    tvb_constant: float = 1.0,
) -> torch.Tensor:
    """Apply the TVB limiter to a modal DG coefficient tensor.

    Parameters
    ----------
    coeff : (ncells, nmodes, nvar)
    dx : cell width
    tvb_constant : M in the TVB criterion; M=0 is strict TVD.

    Returns
    -------
    limited : same shape as *coeff*, with higher modes reduced in troubled
    cells.
    """
    if coeff.shape[1] <= 1:
        return coeff  # P0 -- nothing to limit

    troubled = _troubled_cell_indicator(coeff, dx, tvb_constant)
    if not troubled.any():
        return coeff

    avg = coeff[:, 0]  # [ncells, nvar]
    avg_pad = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    d_left = avg - avg_pad[:-2]
    d_right = avg_pad[2:] - avg

    # Minmod slope (in reference space the slope coefficient equals the
    # physical slope * dx/2 for the normalised Legendre basis on [-1,1]).
    slope = _minmod(d_left, d_right)  # [ncells, nvar]

    limited = coeff.clone()
    # In troubled cells: keep mode-0, replace mode-1 with minmod slope,
    # zero out all higher modes.
    t = troubled  # [ncells]
    limited[t, 1] = slope[t]
    if coeff.shape[1] > 2:
        limited[t, 2:] = 0.0
    return limited


# ---------------------------------------------------------------------------
# WENO limiter
# ---------------------------------------------------------------------------

def _weno_weights(
    beta0: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    epsilon: float = 1e-6,
    ideal: tuple = (1 / 6, 2 / 3, 1 / 6),
) -> tuple:
    """Compute non-linear WENO weights from smoothness indicators."""
    d0, d1, d2 = ideal
    alpha0 = d0 / (epsilon + beta0) ** 2
    alpha1 = d1 / (epsilon + beta1) ** 2
    alpha2 = d2 / (epsilon + beta2) ** 2
    total = alpha0 + alpha1 + alpha2
    return alpha0 / total, alpha1 / total, alpha2 / total


def apply_weno_limiter(
    coeff: torch.Tensor,
    dx: float,
    tvb_constant: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Apply a WENO reconstruction limiter to troubled cells.

    For each troubled cell *i*, the high-order polynomial is replaced by
    a WENO combination of three candidate linear polynomials built from
    the cell averages of stencils {i-1, i}, {i, i+1}, and {i-1, i, i+1}.
    The cell average is preserved exactly.

    Parameters
    ----------
    coeff : (ncells, nmodes, nvar)
    dx : cell width
    tvb_constant : M for the troubled-cell detector
    epsilon : WENO regularisation
    """
    if coeff.shape[1] <= 1:
        return coeff

    troubled = _troubled_cell_indicator(coeff, dx, tvb_constant)
    if not troubled.any():
        return coeff

    ncells, nmodes, nvar = coeff.shape
    avg = coeff[:, 0]  # [ncells, nvar]

    # Pad averages for stencil access
    avg_pad = torch.cat([avg[:1], avg, avg[-1:]], dim=0)  # [ncells+2]
    # In padded indexing, cell i maps to index i+1
    u_im1 = avg_pad[:-2]  # [ncells]
    u_i   = avg
    u_ip1 = avg_pad[2:]

    # Three candidate slopes (in physical units)
    slope0 = (u_i - u_im1) / dx   # stencil {i-1, i}
    slope1 = (u_ip1 - u_i) / dx   # stencil {i, i+1}
    slope2 = (u_ip1 - u_im1) / (2.0 * dx)  # stencil {i-1, i, i+1}

    # Smoothness indicators (classical WENO3)
    beta0 = dx * slope0 ** 2
    beta1 = dx * slope1 ** 2
    # Standard WENO3 beta2
    beta2 = (
        (1.0 / 3.0) * (u_im1 - 2.0 * u_i + u_ip1) ** 2 / dx
        + dx * slope2 ** 2
    )

    # Reduce across variables: use max smoothness per cell
    beta0_scalar = beta0.max(dim=-1).values
    beta1_scalar = beta1.max(dim=-1).values
    beta2_scalar = beta2.max(dim=-1).values

    w0, w1, w2 = _weno_weights(beta0_scalar, beta1_scalar, beta2_scalar, epsilon)

    # WENO combined slope
    weno_slope = (
        w0[..., None] * slope0
        + w1[..., None] * slope1
        + w2[..., None] * slope2
    )

    # Build limited coefficients: mode-0 = cell average, mode-1 = weno_slope * dx/2
    # (the normalised Legendre mode-1 on [-1,1] has slope 1 at the boundary,
    # so the coefficient equals the half-cell physical slope).
    limited = coeff.clone()
    t = troubled
    limited[t, 1] = weno_slope[t] * (dx / 2.0)
    if nmodes > 2:
        limited[t, 2:] = 0.0
    return limited


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

_LIMITERS = {
    "tvb": apply_tvb_limiter,
    "weno": apply_weno_limiter,
}


def apply_limiter(
    coeff: torch.Tensor,
    dx: float,
    kind: str = "tvb",
    tvb_constant: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Dispatch to the requested limiter by name."""
    if kind is None or kind == "none":
        return coeff
    if kind not in _LIMITERS:
        raise ValueError(f"unknown limiter {kind!r}; choose from {list(_LIMITERS)}")
    return _LIMITERS[kind](coeff, dx, tvb_constant=tvb_constant, **kwargs)
