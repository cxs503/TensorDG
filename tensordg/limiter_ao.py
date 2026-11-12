"""High-order (``"ao"``) WENO reconstruction limiter for modal DG in 1-D.

Motivation
----------
The two limiters already in :mod:`tensordg.limiters` both *reduce a troubled
cell to a linear polynomial*, so they cannot keep design order ``p + 1`` when
the troubled-cell detector fires in a smooth region (at smooth extrema, on coarse
meshes, or always under the strict ``limit_all`` policy):

* TVB/minmod (Cockburn--Shu) sets ``c_1`` to a minmod slope and zeroes every
  higher mode -- first order in the limited cell;
* the WENO limiter combines three *linear* candidates -- second order at best.

``tensordg/burgers_shock_1d.py`` therefore declares a "WENO-type high-order
limiter" as an open roadmap item; this module is that item.

What this module adds
---------------------
``apply_ao_limiter`` replaces the polynomial of a troubled cell by a WENO
combination of **degree-``p``** candidate reconstructions, each obtained by
exactly matching the ``p+1`` cell averages of a contiguous sub-stencil of the
``(2p+1)``-cell window ``i-p .. i+p`` (there are ``p+1`` such sub-stencils).
This is the Qiu & Shu (SIAM J. Sci. Comput. 27 (2005) 995) limiter idea with
``r = p+1`` cell stencils instead of ``r = 2``, and it is the natural companion of
the existing :func:`tensordg.limiters.apply_weno_limiter`, which is the ``r = 2``
case.

The nonlinear weights are WENO-Z type (Borges et al., JCP 227 (2008) 3101):
indicators are normalised by their common maximum and enter through the
scale-free ratio ``tau / (beta_k + tiny)`` with ``tau = max_k beta_k - min_k
beta_k``, so an amplitude rescaling or a mesh refinement never pushes the
weights to the ideal ones (the failure mode of the Jiang--Shu ``epsilon``).

Why design order survives
-------------------------
A degree-``p`` reconstruction that matches ``p+1`` cell averages has pointwise
error ``O(dx**(p+1))``, and **every** candidate here is such a reconstruction.
A convex combination of ``O(dx**(p+1))`` approximations is ``O(dx**(p+1))``, so
the limited polynomial keeps design order **for any** convex weights; the
nonlinear weights only decide which sub-stencil dominates across a
discontinuity.  Measured, not assumed: ``tests/test_limiter_ao.py`` forces the
limiter on every cell (``limit_all``) of a smooth periodic sine and compares the
observed order against the TVB and WENO limiters on the same meshes.

Robustness
----------
* The weights are WENO-Z, so a sub-stencil crossing a jump gets a vanishing
  weight; the surviving sub-stencils still carry the design order.
* Cells without a full ``(2p+1)``-cell window (open-domain boundaries) fall back
  row-wise to the verified :func:`~tensordg.limiters.apply_weno_limiter`, so the
  routine is a drop-in replacement for ``kind="weno"``.
* Cell averages are preserved **exactly**: every candidate matches the target
  cell average (the target cell belongs to every sub-stencil) and mode 0 of the
  output is copied back from the input.
* Components are limited independently (component-wise limiting); characteristic
  limiting for systems is **not** part of this module.

Conventions match :mod:`tensordg.limiters`: unnormalised Legendre modes on
``xi in [-1, 1]``, ``coeff`` of shape ``(ncells, nmodes, nvar)``, ``c_1 =
(du/dx) * dx / 2``.
"""
from __future__ import annotations

import functools

import numpy as np
import torch

__all__ = ["apply_ao_limiter", "ao_window", "ao_candidates", "smoothness_indicators"]

# WENO-Z exponent q and the ratio cap that keeps ``r**q`` inside float32.
_Q = 2
_RATIO_CAP = 1e6


def ao_window(nmodes: int) -> int:
    """Width ``2p+1`` of the reconstruction window of a ``P_p`` modal cell."""
    degree = nmodes - 1
    return 2 * degree + 1 if degree >= 1 else 1


def ao_candidates(nmodes: int) -> int:
    """Number of degree-``p`` candidate sub-stencils inside the window (``p+1``)."""
    degree = nmodes - 1
    return max(degree, 0) + 1


@functools.lru_cache(maxsize=None)
def _stencil_matrix(size: int, position: int, degree: int) -> np.ndarray:
    """Inverse average-matching matrix of one contiguous sub-stencil.

    Rows are the cells of the sub-stencil; columns are the modal coefficients
    ``c_0 .. c_{size-1}`` of the target-cell polynomial.  Entry ``(k, j)`` is the
    mean of the Legendre mode ``j`` over the ``k``-th stencil cell, i.e. over the
    reference interval shifted by ``2 * (position - degree + k)``.
    """
    nodes, weights = np.polynomial.legendre.leggauss(2 * degree + 3)
    width = 2 * degree + 1
    means = np.empty((width, size))
    for row, m in enumerate(range(-degree, degree + 1)):
        for j in range(size):
            polynomial = np.zeros(j + 1)
            polynomial[-1] = 1.0
            means[row, j] = 0.5 * np.sum(
                weights * np.polynomial.legendre.legval(nodes + 2.0 * m, polynomial)
            )
    return np.linalg.inv(means[position:position + size, :])


@functools.lru_cache(maxsize=None)
def _smoothness_matrix(degree: int) -> np.ndarray:
    """Jiang--Shu smoothness indicator as a quadratic form ``c^T B c``.

    ``beta = sum_{l=1}^{d} (dx)^{2l-1} int_cell (d^l u / dx^l)^2 dx`` reduces, in
    the reference cell, to ``sum_l 2^{2l-1} int_{-1}^{1} (d^l q / dxi^l)^2 dxi``
    -- free of ``dx``, so indicators from different stencils are comparable.
    """
    size = degree + 1
    nodes, weights = np.polynomial.legendre.leggauss(2 * degree + 3)
    matrix = np.zeros((size, size))
    for order in range(1, degree + 1):
        lth = []
        for j in range(size):
            polynomial = np.zeros(j + 1)
            polynomial[-1] = 1.0
            derived = polynomial
            for _ in range(order):
                derived = np.polynomial.legendre.legder(derived)
            lth.append(
                np.polynomial.legendre.legval(nodes, derived)
                if len(derived) else np.zeros_like(nodes)
            )
        lth = np.stack(lth)                      # (size, nq)
        weight = (2.0 ** (2 * order - 1)) * weights
        matrix += (lth * weight[None, :]) @ lth.T
    return matrix


def smoothness_indicators(coeff: torch.Tensor, degree: int) -> torch.Tensor:
    """Jiang--Shu indicator ``beta`` of each row of ``coeff`` (modes ``0..degree``)."""
    size = degree + 1
    matrix = torch.as_tensor(_smoothness_matrix(degree), dtype=coeff.dtype,
                             device=coeff.device)
    block = coeff[..., :size, :]                 # (..., size, nvar)
    # beta[..., v] = sum_{j,k} block[..., j, v] B[j, k] block[..., k, v]
    contracted = torch.einsum("...jv,jk->...kv", block, matrix)
    return (contracted * block).sum(dim=-2)


def _reconstruct(
    averages: torch.Tensor,
    nmodes: int,
    position: int,
    size: int,
    cells: torch.Tensor,
) -> torch.Tensor:
    """Degree ``size-1`` polynomial on a sub-stencil of the target window.

    ``cells`` holds, for every target cell, the global indices of the window
    cells ``i-p .. i+p`` in order; the sub-stencil is
    ``cells[:, position : position + size]``.  Modes ``>= size`` are zero.
    """
    degree = nmodes - 1
    gathered = torch.stack(
        [averages.index_select(0, cells[:, position + k]) for k in range(size)],
        dim=1)                                    # (ncells, size, nvar)
    inverse = torch.as_tensor(_stencil_matrix(size, position, degree),
                              dtype=averages.dtype, device=averages.device)
    block = torch.einsum("km,cmv->ckv", inverse, gathered)
    out = averages.new_zeros((averages.shape[0], nmodes, averages.shape[1]))
    out[:, :size] = block
    return out


def _weno_z_weights(betas: torch.Tensor, q: int) -> torch.Tensor:
    """WENO-Z weights along dim 0 of ``betas`` (shape ``(n, ...)``).

    Ideal weights are equal (``1/n``): with every candidate carrying the design
    order, equal ideal weights are order-optimal and avoid the asymmetric
    upwind bias of a one-sided optimal-weight set.
    """
    n = betas.shape[0]
    scale = betas.abs().amax(dim=0).clamp_min(torch.finfo(betas.dtype).tiny)
    normalised = betas / scale
    tau = normalised.amax(dim=0) - normalised.amin(dim=0)
    tiny = torch.finfo(betas.dtype).tiny
    alphas = (1.0 / n) * (
        1.0 + (tau / (normalised.abs() + tiny)).clamp_max(_RATIO_CAP) ** q
    )
    return alphas / alphas.sum(dim=0, keepdim=True)


def apply_ao_limiter(
    coeff: torch.Tensor,
    dx: float,
    tvb_constant: float = 1.0,
    periodic: bool = False,
    limit_all: bool = False,
    limit_boundaries: bool = False,
    q: int = _Q,
) -> torch.Tensor:
    """High-order WENO limiter; see the module docstring.

    Parameters
    ----------
    coeff : (ncells, nmodes, nvar) modal coefficients, unnormalised Legendre.
    dx : cell width (uniform mesh).
    tvb_constant : ``M`` of the troubled-cell indicator.
    periodic : wrap the reconstruction window around the domain.
    limit_all : limit every cell (strict policy; the order-preservation test uses
        it to force the limiter on smooth data).
    limit_boundaries : limit the two end cells of an open domain through the
        one-sided WENO fallback (they have no full window).
    q : WENO-Z exponent.

    Returns
    -------
    A tensor with the same shape and dtype as ``coeff``; untouched cells are
    returned bit-identically.
    """
    if coeff.dim() != 3:
        raise ValueError(f"coeff must be (ncells, nmodes, nvar), got {tuple(coeff.shape)}")
    if q < 1:
        raise ValueError(f"q must be >= 1, got {q!r}")
    if dx <= 0.0:
        raise ValueError(f"dx must be positive, got {dx!r}")
    nmodes = coeff.shape[1]
    degree = nmodes - 1
    if degree < 1:
        return coeff  # P0: the cell average is the whole polynomial

    from .limiters import _detect, apply_weno_limiter  # local import: no cycle

    troubled, _ = _detect(coeff, dx, tvb_constant,
                          limit_boundaries=limit_boundaries,
                          periodic=periodic, limit_all=limit_all)
    if not troubled.any():
        return coeff

    ncells, _, nvar = coeff.shape
    half = degree                                  # window = i-p .. i+p
    indices = torch.arange(ncells, device=coeff.device)
    shifts = torch.arange(-half, half + 1, device=coeff.device)
    raw = indices[:, None] + shifts[None, :]
    if periodic:
        cells = torch.remainder(raw, ncells)
        full = torch.ones(ncells, dtype=torch.bool, device=coeff.device)
    else:
        full = ((raw >= 0) & (raw < ncells)).all(dim=1)
        cells = raw.clamp(0, ncells - 1)

    averages = coeff[:, 0]                          # (ncells, nvar)
    size = degree + 1
    candidates = []
    for position in range(ao_candidates(nmodes)):
        poly = _reconstruct(averages, nmodes, position, size, cells)
        candidates.append((poly, smoothness_indicators(poly, degree)))
    betas = torch.stack([candidate[1] for candidate in candidates], dim=0)
    weights = _weno_z_weights(betas, q)
    high_order = candidates[0][0] * weights[0].unsqueeze(1)
    for k in range(1, len(candidates)):
        high_order = high_order + candidates[k][0] * weights[k].unsqueeze(1)

    limited = coeff.clone()
    rows = troubled & full
    if rows.any():
        limited[rows] = high_order[rows]
    fallback = troubled & ~full
    if fallback.any():
        weno = apply_weno_limiter(coeff, dx, tvb_constant=tvb_constant,
                                  periodic=periodic, limit_all=False)
        limited[fallback] = weno[fallback]
    limited[:, 0] = coeff[:, 0]                     # exact cell averages
    return limited