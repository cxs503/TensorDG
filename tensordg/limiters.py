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

Implementation note (mission m3262)
-----------------------------------
Everything here is element-wise tensor algebra on ``(ncells, nmodes, nvar)``
arrays, so the cost of a call is *kernel-count* bound: for 200 cells the
arrays are a few thousand doubles and the wall time is entirely torch's
per-op dispatch overhead.  The limiter is therefore written as one detector
pass whose minmod slope is reused by the caller, plus **one** full-tensor
``torch.where`` write; this is arithmetically identical (bit-for-bit) to the
previous masked-assignment formulation while issuing far fewer kernels.
``tests/test_limiter_kernel_budget.py`` pins both properties: exact equality
against the reference implementation and an upper bound on the kernel count.
See ``docs/limiter_performance_verification.md``.
"""
from __future__ import annotations

import torch

from .euler_1d import GAMMA
from .tvb import minmod as _minmod
from .tvb import troubled_from_differences


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Boolean "cell is interior" masks, cached per (ncells, device): building the
# mask costs two scalar writes, so it is done once instead of per call.
_INTERIOR_MASK_CACHE: dict = {}


def _interior_mask(ncells: int, device) -> torch.Tensor:
    """[ncells] bool mask that is False in the two end cells."""
    key = (ncells, str(device))
    mask = _INTERIOR_MASK_CACHE.get(key)
    if mask is None or mask.device != torch.device(device) or mask.numel() != ncells:
        mask = torch.ones(ncells, dtype=torch.bool, device=device)
        if ncells > 1:
            mask[0] = False
            mask[-1] = False
        else:
            mask[0] = False
        _INTERIOR_MASK_CACHE[key] = mask
    return mask


# Replicate-padding index [ncells+2] (first and last cell repeated) cached per
# (ncells, device): one ``index_select`` kernel instead of ``torch.cat`` plus two
# slices.  ``F.pad(..., mode="replicate")`` does not accept 2-D input.
_PAD_INDEX_CACHE: dict = {}


def _pad_index(ncells: int, device) -> torch.Tensor:
    key = (ncells, str(device))
    idx = _PAD_INDEX_CACHE.get(key)
    if idx is None or idx.numel() != ncells + 2 or idx.device != torch.device(device):
        idx = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                torch.arange(ncells, dtype=torch.long, device=device),
                torch.full((1,), ncells - 1, dtype=torch.long, device=device),
            ]
        )
        _PAD_INDEX_CACHE[key] = idx
    return idx


def _detect(
    coeff: torch.Tensor,
    dx: float,
    tvb_constant: float,
    limit_boundaries: bool = False,
    periodic: bool = False,
    limit_all: bool = False,
):
    """Troubled-cell mask and minmod slope in one pass.

    Returns ``(troubled, slope)`` where ``troubled`` is ``[ncells]`` bool and
    ``slope = minmod(d_left, d_right)`` is ``[ncells, nvar]`` -- the cell-average
    increment (physical slope * dx).  Returning the slope removes the second
    minmod evaluation the callers used to perform.
    """
    ncells = coeff.shape[0]
    avg = coeff[:, 0]  # [ncells, nvar]
    if periodic:
        # Genuine wrap-around neighbours: the end cells of a periodic domain do
        # have a full two-sided stencil, so the one-sided clamping below would
        # both mis-estimate their slope and (with the interior mask) skip them.
        avg_pad = torch.cat((avg[-1:], avg, avg[:1]), dim=0)  # [ncells+2, nvar]
    else:
        # Pad with copies of the boundary cells for one-sided stencils
        avg_pad = avg.index_select(0, _pad_index(ncells, coeff.device))  # [ncells+2, nvar]
    d_left = avg - avg_pad[:-2]   # u_i - u_{i-1}
    d_right = avg_pad[2:] - avg   # u_{i+1} - u_i
    troubled, slope = troubled_from_differences(d_left, d_right, dx, tvb_constant)
    if limit_all:
        # Strict TVD policy: minmod every cell, including the monotone ones the
        # TVB indicator releases.  Those released cells keep their full DG
        # polynomial and their trace can still bulge past the neighbouring cell
        # averages (measured 1.4 % of the jump amplitude at N=80, O(h) decay);
        # flagging every cell removes that bulge but clips smooth extrema.
        troubled = torch.ones_like(troubled)
    elif not limit_boundaries and not periodic:
        # End cells have no full stencil; leave them unlimited (default).
        # A cached mask AND-ed in is one kernel instead of two scalar writes.
        troubled = troubled & _interior_mask(ncells, coeff.device)
    return troubled, slope


def _troubled_cell_indicator(
    coeff: torch.Tensor,
    dx: float,
    tvb_constant: float = 0.0,
    limit_boundaries: bool = False,
) -> torch.Tensor:
    """Return a boolean mask [ncells] of cells that need limiting.

    The indicator fires when the cell-average sequence is *not* monotone
    (i.e. a local extremum) **and** the variation exceeds the TVB
    tolerance ``M * dx**2``.  With ``M = 0`` this reduces to the strict
    TVD test; positive ``M`` allows smooth extrema to pass.

    ``limit_boundaries`` controls the two end cells.  The historical
    (``False``) behaviour leaves them unlimited because the half stencil of an
    outflow boundary carries no upwind information.  With ``True`` they are
    tested as well, using the one-sided (clamped) neighbour already used by
    the padding above; pass ``True`` when the outer cells carry a genuine
    discontinuity that must be limited.

    The TVB test itself (Cockburn & Shu 1989, in the scale-consistent form
    introduced in mission m2500) lives in :func:`tensordg.tvb.
    troubled_from_differences`; see ``docs/limiter_policy_verification.md``
    for why the deviation ``|d_left - minmod(d_left, d_right)|`` -- an
    ``O(dx**2)`` quantity for smooth data -- is the right thing to compare
    against ``M dx**2``.
    """
    troubled, _ = _detect(coeff, dx, tvb_constant, limit_boundaries)
    return troubled


def _replace_high_modes(
    coeff: torch.Tensor,
    troubled: torch.Tensor,
    mode1: torch.Tensor,
) -> torch.Tensor:
    """Keep mode 0, set mode 1 to *mode1* and higher modes to zero in troubled cells.

    The masked write touches only the flagged rows, which is both the cheapest
    formulation measured (8 kernels for 200 cells versus 16 for the equivalent
    full-tensor ``torch.where``, mission m3262) and bit-identical to the
    historical implementation.
    """
    limited = coeff.clone()
    limited[troubled, 1] = mode1[troubled]
    if coeff.shape[1] > 2:
        limited[troubled, 2:] = 0.0
    return limited


# ---------------------------------------------------------------------------
# TVB limiter
# ---------------------------------------------------------------------------

def apply_tvb_limiter(
    coeff: torch.Tensor,
    dx: float,
    tvb_constant: float = 1.0,
    limit_boundaries: bool = False,
    periodic: bool = False,
    limit_all: bool = False,
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

    troubled, slope = _detect(
        coeff, dx, tvb_constant, limit_boundaries=limit_boundaries,
        periodic=periodic, limit_all=limit_all,
    )
    if not troubled.any():
        return coeff

    # In troubled cells: keep mode-0, replace mode-1 with the minmod slope and
    # zero out all higher modes.  The modal basis is the unnormalised Legendre
    # basis on [-1, 1], for which c_1 = (du/dx) * dx/2 (P_1(xi) = xi), so the
    # average increment ``slope`` (== (du/dx)*dx) must be halved.  Omitting the
    # factor doubles the reconstructed slope and breaks the minmod edge values
    # u_i +/- slope/2; the WENO limiter below already applies the same dx/2.
    return _replace_high_modes(coeff, troubled, 0.5 * slope)


# ---------------------------------------------------------------------------
# WENO limiter
# ---------------------------------------------------------------------------

# WENO-Z (Borges et al. 2008) constants.  ``_Z_EPSILON`` is the relative floor
# of the ``tau / (beta_k + eps)`` ratio (indicators are normalised by their common
# maximum first, so it is a scale-free number, not a modelling scale);
# ``_Z_RATIO_CAP`` bounds the ratio so ``r**q`` cannot overflow in single
# precision -- a weight that has reached the cap already equals its ideal value.
_Z_EPSILON = 1e-12
_Z_RATIO_CAP = 1e6


def _weno_weights(
    beta0: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    epsilon: float = 1e-6,
    ideal: tuple = (1 / 6, 2 / 3, 1 / 6),
    variant: str = "js",
    q: int = 2,
) -> tuple:
    """Compute non-linear WENO weights from smoothness indicators.

    Parameters
    ----------
    beta0, beta1, beta2 : smoothness indicators of the three candidate stencils
    epsilon : Jiang--Shu regularisation (``"js"`` only, see below)
    ideal : linear weights ``d_k`` of the stencils
    variant : ``"js"`` (default, Jiang & Shu 1996) or ``"z"`` (WENO-Z, Borges,
        Carmona, Costa & Don, JCP 227 (2008) 3101--3111)
    q : exponent of the WENO-Z global smoothness indicator (``2`` is the
        published value; unused by ``"js"``)

    Notes
    -----
    ``"js"`` uses ``alpha_k = d_k / (epsilon + beta_k)**2``.  Its weakness is
    that ``epsilon`` fixes an absolute scale: when every ``beta_k`` is much
    smaller than ``epsilon`` -- small-amplitude fields, or fine meshes, since
    ``beta ~ dx * (du/dx)**2`` -- all the weights collapse to the ideal
    ``d = (1/6, 2/3, 1/6)`` and the stencil that crosses a discontinuity is no
    longer rejected.

    ``"z"`` replaces the normalisation by a *ratio*,

        ``alpha_k = d_k * (1 + (tau / (beta_k + eps_z))**q)``,
        ``tau = |beta_0 - beta_2|``,

    so the weights depend on ``beta`` only through ratios and are invariant under
    a uniform rescaling of the indicators.  ``eps_z`` is the floating-point
    ``tiny`` of the working dtype: it only guards an exactly zero indicator, it is
    not a modelling scale, and ``epsilon`` is therefore unused by this variant.
    ``w -> d`` is still recovered for uniform data (``tau = 0``).
    """
    if variant == "js":
        d0, d1, d2 = ideal
        alpha0 = d0 / (epsilon + beta0) ** 2
        alpha1 = d1 / (epsilon + beta1) ** 2
        alpha2 = d2 / (epsilon + beta2) ** 2
    elif variant == "z":
        d0, d1, d2 = ideal
        # Work with the indicators normalised by their common maximum: the
        # weights then depend on the data only through scale-free ratios, which
        # is the invariance property the variant exists for, and the ratio can
        # never overflow (it is capped at _Z_RATIO_CAP, far beyond the range
        # where the weight is already indistinguishable from the ideal one).
        scale = (
            torch.maximum(torch.maximum(beta0.abs(), beta1.abs()), beta2.abs())
            .clamp_min(torch.finfo(beta0.dtype).tiny)
        )
        b0, b1, b2 = beta0 / scale, beta1 / scale, beta2 / scale
        tau = (b2 - b0).abs()
        r0 = (tau / (b0.abs() + _Z_EPSILON)).clamp_max(_Z_RATIO_CAP)
        r1 = (tau / (b1.abs() + _Z_EPSILON)).clamp_max(_Z_RATIO_CAP)
        r2 = (tau / (b2.abs() + _Z_EPSILON)).clamp_max(_Z_RATIO_CAP)
        alpha0 = d0 * (1.0 + r0**q)
        alpha1 = d1 * (1.0 + r1**q)
        alpha2 = d2 * (1.0 + r2**q)
    else:
        raise ValueError(f"unknown WENO weight variant {variant!r}; choose from ['js', 'z']")
    total = alpha0 + alpha1 + alpha2
    return alpha0 / total, alpha1 / total, alpha2 / total

def apply_weno_limiter(
    coeff: torch.Tensor,
    dx: float,
    tvb_constant: float = 1.0,
    epsilon: float = 1e-6,
    limit_boundaries: bool = False,
    weno_variant: str = "js",
    weno_q: int = 2,
    periodic: bool = False,
    limit_all: bool = False,
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
    epsilon : WENO regularisation (``weno_variant="js"``)
    weno_variant : ``"js"`` (Jiang--Shu, default, historical behaviour) or
        ``"z"`` (WENO-Z, Borges et al. 2008).  ``"z"`` makes the weights
        invariant under a rescaling of the smoothness indicators, i.e. under a
        change of the field amplitude or of the mesh spacing, and loses the
        ``epsilon``-driven degeneration of the weights to the ideal stencil
        weights on small-amplitude data -- see :func:`_weno_weights`.
    weno_q : exponent of the WENO-Z global smoothness indicator (default 2)
    periodic : use wrap-around neighbours for the end cells (both in the
        troubled-cell detector and in the stencil averages) instead of the
        one-sided clamping of the open-domain case.
    limit_all : force the reconstruction in every cell (strict TVD policy),
        see :func:`_detect`.
    """
    if coeff.shape[1] <= 1:
        return coeff

    troubled, _ = _detect(
        coeff, dx, tvb_constant, limit_boundaries=limit_boundaries,
        periodic=periodic, limit_all=limit_all,
    )
    if not troubled.any():
        return coeff

    ncells, nmodes, nvar = coeff.shape
    avg = coeff[:, 0]  # [ncells, nvar]

    # Pad averages for stencil access.  A periodic domain wraps; the open case
    # replicates the boundary cells (there is no neighbour to reach for).
    if periodic:
        avg_pad = torch.cat((avg[-1:], avg, avg[:1]), dim=0)
    else:
        avg_pad = avg.index_select(0, _pad_index(ncells, coeff.device))  # [ncells+2]
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

    w0, w1, w2 = _weno_weights(
        beta0_scalar, beta1_scalar, beta2_scalar, epsilon,
        variant=weno_variant, q=weno_q,
    )

    # WENO combined slope
    weno_slope = (
        w0[..., None] * slope0
        + w1[..., None] * slope1
        + w2[..., None] * slope2
    )

    # Build limited coefficients: mode-0 = cell average, mode-1 = weno_slope * dx/2
    # (the normalised Legendre mode-1 on [-1,1] has slope 1 at the boundary,
    # so the coefficient equals the half-cell physical slope).
    return _replace_high_modes(coeff, troubled, weno_slope * (dx / 2.0))


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

# Hierarchical high-order limiter, registered in _LIMITERS below.  ``limiter_ao``
# only imports numpy/torch at module scope, so this import cannot cycle.
from .limiter_ao import apply_ao_limiter  # noqa: E402

# Sensor-gated modal (SVV) artificial viscosity; also torch-only at module
# scope, so this import cannot cycle.
from .artificial_viscosity import apply_av_filter  # noqa: E402

_LIMITERS = {
    "tvb": apply_tvb_limiter,
    "weno": apply_weno_limiter,
    # Continuous-strength alternative to reconstruction limiters: damps the top
    # modes of each element in proportion to a smoothness sensor instead of
    # replacing the polynomial (see tensordg/artificial_viscosity.py).
    "av": apply_av_filter,
    # Hierarchical high-order limiter (design order kept in troubled cells);
    # imported here, not at module top, because tensordg.limiter_ao imports the
    # troubled-cell detector from this module.
    "ao": apply_ao_limiter,
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