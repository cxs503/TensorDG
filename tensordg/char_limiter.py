"""Local characteristic-variable TVB limiting for the 2-D Euler system.

Why this module exists
----------------------
``tensordg.euler_dg_2d.apply_tvb_limiter_2d`` limits the **conserved** modal
coefficients component by component: the minmod slope of ``rho``, ``rho u``,
``rho v`` and ``E`` is taken independently.  For a scalar conservation law the
Cockburn--Shu TVB test (Cockburn & Shu, *Math. Comp.* 52 (1989) 411-435) has a
provable TVD/TVB property; for a **system** the scalar argument only transfers
to the *characteristic* fields, because the total variation is not a
coordinate-free functional of a vector of conserved densities.  The
characteristic-space version of the same limiter is the textbook construction
(``U = R W`` pointwise, limit each component of ``W`` with the scalar
troubled-cell test, transform back with ``R``) and it is what makes the
limiter's decision differ between a genuinely non-monotone acoustic wave and a
contact/shear wave that is multi-valued in the conserved variables but
perfectly smooth in its own characteristic field.

Nothing in this repository did that: :func:`apply_tvb_limiter_2d` and
:func:`apply_weno_limiter_2d` both act componentwise, so the 2-D Euler tests of
the ``REPRO-DG-CKS1998`` anchor were all run with a conservative-variable
limiter.  This module adds ``limiter="char_tvb"`` to
:func:`tensordg.euler_dg_2d.solve_euler_dg_2d` so the difference becomes
*measurable* instead of asserted (see ``benchmarks/char_limiter_2d.py`` and
``docs/CASE-DG-CKS1998-CHAR-LIMITER.md``).

Construction
------------
For one cell with cell-average state ``U = (rho, rho u, rho v, E)``:

* build the right-eigenvector matrix ``R_n`` of the flux Jacobian
  ``A_n = dF_n/dU`` for the axis direction ``n`` (analytic, standard form with
  the entropy / shear / two acoustic columns);
* transform every modal coefficient of the cell, ``W = R_n^{-1} U_h``;
* run the *existing, unchanged* Cockburn--Shu troubled-cell test along ``n`` on
  the characteristic cell averages ``W_0`` and replace the mode-1 coefficient of
  the offending direction with the minmod slope in ``W`` (higher modes of that
  direction dropped), exactly as the componentwise limiter does;
* transform back, ``U_h = R_n W``.

The cell average is restored verbatim after the transform (it cannot be
modified by the limiter anyway), so **conservation is bit-identical** to the
unlimited state, not merely close.

Two approximations are made deliberately and are documented rather than hidden:

1. ``R_n`` is frozen at the cell-average state for the whole cell (the usual
   "local characteristic decomposition" of the CKS series; it is what makes the
   transform a single constant matrix per cell).
2. The x-pass uses ``R_x`` and the y-pass uses ``R_y`` (dimension-by-dimension
   decomposition), not a 2-D eigen-decomposition of ``n_x F + n_y G``; this is
   the same directional structure as the componentwise limiter, so the two can
   be compared on equal footing.

Degenerate cells (``rho <= 0``, ``p <= 0``, non-finite, or a numerically
singular ``R``) fall back to the identity transformation, i.e. to the
componentwise limiter, and are counted in the returned diagnostics.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .euler_1d import GAMMA

__all__ = [
    "euler_eigenvectors_2d",
    "apply_characteristic_tvb_limiter_2d",
]

#: Floor on the sound speed so that a state that is admissible up to round-off
#: cannot produce ``0/0`` in the eigenvector columns.
_SOUND_FLOOR = 1e-8


def _admissible_state(state: torch.Tensor, gamma: float):
    """Split a conservative state into ``(rho, u, v, p, a, H, ok)``."""
    rho = state[..., 0]
    safe_rho = rho.clamp_min(1e-30)
    velocity_x = state[..., 1] / safe_rho
    velocity_y = state[..., 2] / safe_rho
    pressure = (gamma - 1.0) * (
        state[..., 3]
        - 0.5 * (state[..., 1] ** 2 + state[..., 2] ** 2) / safe_rho
    )
    sound = torch.sqrt((gamma * pressure.clamp_min(0.0) / safe_rho).clamp_min(_SOUND_FLOOR ** 2))
    enthalpy = (state[..., 3] + pressure) / safe_rho
    ok = (
        (rho > 0.0) & (pressure > 0.0) & torch.isfinite(state).all(-1)
        & torch.isfinite(enthalpy)
    )
    return rho, velocity_x, velocity_y, pressure, sound, enthalpy, ok


def euler_eigenvectors_2d(
    state: torch.Tensor, axis: int, gamma: float = GAMMA
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-eigenvector matrix ``R`` and its inverse for ``dF_axis/dU``.

    Parameters
    ----------
    state : (..., 4)
        Conservative state ``(rho, rho u, rho v, E)``, one state per leading
        index combination (cells, quadrature points, ...).
    axis : {0, 1}
        Flux direction: 0 -> x, 1 -> y.

    Returns
    -------
    (R, R_inv) : both ``(..., 4, 4)``.
        ``R`` has the **right** eigenvectors as *columns*, ordered
        ``(u_n - a, u_n, u_n, u_n + a)`` where ``u_n`` is the velocity along
        ``axis``; the third (shear/entropy) column is the tangential velocity
        perturbation, the second is the entropy wave.  Hence
        ``R_inv A_n R = diag(lambda)``: the rows of ``R_inv`` are the left
        eigenvectors, so ``W = R_inv U`` is the characteristic amplitude vector.
        ``R_inv`` is the numerical inverse; for cells where the
        state is not admissible (or the inverse is not finite) both matrices are
        replaced by the identity, so the caller degrades gracefully to the
        componentwise limiter.

    The columns satisfy ``R_inv A_n R = diag(lambda)`` with
    ``lambda = (u_n - a, u_n, u_n, u_n + a)``; this is verified against an
    autograd Jacobian of the flux in
    ``tests/test_char_limiter_2d.py::test_eigenvectors_diagonalise_the_flux_jacobian``
    rather than trusted from the textbook.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (x) or 1 (y)")
    if state.shape[-1] != 4:
        raise ValueError("state must have a trailing component axis of length 4")

    rho, u, v, p, a, H, ok = _admissible_state(state, gamma)
    zeros = torch.zeros_like(rho)
    ones = torch.ones_like(rho)
    kinetic = 0.5 * (u ** 2 + v ** 2)
    if axis == 0:
        un = u
        columns = (
            (ones, u - a, v, H - u * a),
            (ones, u, v, kinetic),
            (zeros, zeros, ones, v),
            (ones, u + a, v, H + u * a),
        )
    else:
        un = v
        columns = (
            (ones, u, v - a, H - v * a),
            (ones, u, v, kinetic),
            (zeros, ones, zeros, u),
            (ones, u, v + a, H + v * a),
        )
    # Unused here but kept explicit: the eigenvalues are (un-a, un, un, un+a)
    # and the callers rely on that ordering (mode-1 column replacement is applied
    # to all four fields at once, so no per-wave bookkeeping is needed).
    del un

    shape = rho.shape
    # Assemble explicitly as (column, component, *batch) and move the two matrix
    # axes last: ``torch.stack`` with a negative ``dim`` is ambiguous for 1-D
    # inputs (it inserts before the only axis), which silently transposed the
    # matrix in the first draft of this module.
    assembled = torch.stack(
        [torch.stack(column, dim=0) for column in columns], dim=0
    )
    R = assembled.permute(*range(2, 2 + len(shape)), 1, 0).contiguous()
    identity = torch.eye(4, dtype=R.dtype, device=R.device).expand_as(R)
    R = torch.where(ok[..., None, None], R, identity)
    flat = R.reshape(-1, 4, 4)
    try:
        inverse = torch.linalg.inv(flat)
    except RuntimeError:  # pragma: no cover - singular batches fall back below
        inverse = torch.zeros_like(flat)
    bad = ~torch.isfinite(inverse).all(-1).all(-1)
    if bool(bad.any()):
        inverse[bad] = torch.eye(4, dtype=R.dtype, device=R.device)
        flat[bad] = torch.eye(4, dtype=R.dtype, device=R.device)
    return flat.reshape(R.shape), inverse.reshape(R.shape)


def _transform(matrix: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """Apply a per-cell 4x4 matrix to the component axis of a modal tensor."""
    return torch.einsum("...ij,...klj->...kli", matrix, coeff)


from .registry import capability
@capability(
    name="char-limiter-2d",
    category="limiter",
    maturity="planned",
    evidence={"feature_ledger": "CASE-SHU-OSHER-DG"},
    description="2D characteristic TVB limiter",
    tags=['limiter', 'characteristic', '2d']
)
def apply_characteristic_tvb_limiter_2d(
    coeff: torch.Tensor,
    dx: float,
    dy: float,
    tvb_constant: float = 1.0,
    periodic=(True, True),
    gamma: float = GAMMA,
    return_diagnostics: bool = False,
):
    """Characteristic-space TVB limiting of a tensor-product DG expansion.

    Parameters
    ----------
    coeff : (nx, ny, kx, ky, 4)
        Modal coefficients of the conservative state; ``coeff[..., 0, 0, :]`` is
        the cell average.
    dx, dy : float
        Cell spacings.
    tvb_constant : float
        ``M`` in ``|d_left - minmod(d_left, d_right)| > M h^2``; ``M = 0`` is the
        strict TVD test.
    periodic : (bool, bool)
        Wrap-around vs edge replication for the neighbour stencil, per axis.
    gamma : float
        Ratio of specific heats of the ideal gas.

    Returns
    -------
    limited : same shape as ``coeff``.
        Cell averages are bit-identical to the input (see module docstring).
    diagnostics : dict, only with ``return_diagnostics=True``
        ``{"troubled_x", "troubled_y", "degenerate_cells"}`` counts, so a caller
        can tell "limiter inactive" from "limiter active but degenerate".
    """
    # Imported lazily: ``euler_dg_2d`` imports this module for the solver hook.
    from .euler_dg_2d import _troubled_1d

    if coeff.ndim != 5 or coeff.shape[-1] != 4:
        raise ValueError("coeff must have shape (nx, ny, kx, ky, 4)")
    diagnostics = {
        "troubled_x": 0,
        "troubled_y": 0,
        "degenerate_cells": 0,
    }
    if coeff.shape[2] <= 1 and coeff.shape[3] <= 1:
        if return_diagnostics:
            return coeff, diagnostics
        return coeff

    averages = coeff[..., 0, 0, :]
    _, _, _, _, _, _, ok = _admissible_state(averages, gamma)
    diagnostics["degenerate_cells"] = int((~ok).sum().item())

    limited = coeff
    for axis, spacing in ((0, dx), (1, dy)):
        if coeff.shape[2 + axis] <= 1:
            continue
        matrix, inverse = euler_eigenvectors_2d(averages, axis, gamma)
        characteristic = _transform(inverse, limited)
        troubled, slope = _troubled_1d(
            characteristic[..., 0, 0, :],
            axis,
            spacing,
            tvb_constant,
            bool(periodic[axis]),
        )
        diagnostics["troubled_x" if axis == 0 else "troubled_y"] = int(
            troubled.sum().item()
        )
        if not troubled.any():
            continue
        mask = troubled.unsqueeze(-1)  # over the 4 characteristic fields
        mask_modes = troubled.view(*troubled.shape, 1, 1, 1)
        if axis == 0:
            characteristic[..., 1, 0, :] = torch.where(
                mask, 0.5 * slope, characteristic[..., 1, 0, :]
            )
            if coeff.shape[2] > 2:
                characteristic[..., 2:, :, :] = torch.where(
                    mask_modes,
                    torch.zeros_like(characteristic[..., 2:, :, :]),
                    characteristic[..., 2:, :, :],
                )
        else:
            characteristic[..., 0, 1, :] = torch.where(
                mask, 0.5 * slope, characteristic[..., 0, 1, :]
            )
            if coeff.shape[3] > 2:
                characteristic[..., :, 2:, :] = torch.where(
                    mask_modes,
                    torch.zeros_like(characteristic[..., :, 2:, :]),
                    characteristic[..., :, 2:, :],
                )
        limited = _transform(matrix, characteristic)
        # The limiter never touches the (0, 0) mode, so the average is unchanged
        # in characteristic space; restoring it verbatim makes conservation
        # exact instead of "exact up to the R R^{-1} round-off".
        limited[..., 0, 0, :] = averages
    if return_diagnostics:
        return limited, diagnostics
    return limited