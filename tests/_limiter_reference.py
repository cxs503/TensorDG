"""Reference (pre-optimisation) limiter formulations -- test oracle only.

Verbatim arithmetic of ``tensordg/limiters.py`` as of commit ``ed8f38b``
(before mission m3262):

* own padding, own minmod, own troubled-cell test,
* ``clone`` + two boolean-mask ``index_put`` writes for the limited state,
* two scalar writes to clear the boundary cells of the mask.

The kernel-budget/equivalence tests compare the shipped implementation against
this reference with ``torch.equal`` (bit-for-bit), so any accidental change of
rounding or of the limiting policy fails loudly.
"""
from __future__ import annotations

import torch


def ref_minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise minmod: same-sign selection with smallest magnitude."""
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(a.abs(), b.abs())


def ref_troubled_from_differences(d_left, d_right, spacing, tvb_constant):
    mm = ref_minmod(d_left, d_right)
    deviation = (d_left - mm).abs().max(dim=-1).values
    threshold = tvb_constant * spacing * spacing
    scale = torch.maximum(d_left.abs(), d_right.abs()).max(dim=-1).values
    troubled = deviation > threshold + 1e-12 * scale.clamp_min(1.0)
    return troubled, mm


def ref_troubled_cell_indicator(coeff, dx, tvb_constant=0.0, limit_boundaries=False):
    ncells = coeff.shape[0]
    avg = coeff[:, 0]
    avg_pad = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    d_left = avg - avg_pad[:-2]
    d_right = avg_pad[2:] - avg
    troubled, _ = ref_troubled_from_differences(d_left, d_right, dx, tvb_constant)
    if not limit_boundaries:
        troubled[0] = False
        troubled[-1] = False
    return troubled


def ref_tvb_limiter(coeff, dx, tvb_constant=1.0, limit_boundaries=False):
    if coeff.shape[1] <= 1:
        return coeff
    troubled = ref_troubled_cell_indicator(
        coeff, dx, tvb_constant, limit_boundaries=limit_boundaries
    )
    if not troubled.any():
        return coeff
    avg = coeff[:, 0]
    avg_pad = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    d_left = avg - avg_pad[:-2]
    d_right = avg_pad[2:] - avg
    slope = ref_minmod(d_left, d_right)
    limited = coeff.clone()
    t = troubled
    limited[t, 1] = 0.5 * slope[t]
    if coeff.shape[1] > 2:
        limited[t, 2:] = 0.0
    return limited


def ref_weno_weights(beta0, beta1, beta2, epsilon=1e-6, ideal=(1 / 6, 2 / 3, 1 / 6)):
    d0, d1, d2 = ideal
    alpha0 = d0 / (epsilon + beta0) ** 2
    alpha1 = d1 / (epsilon + beta1) ** 2
    alpha2 = d2 / (epsilon + beta2) ** 2
    total = alpha0 + alpha1 + alpha2
    return alpha0 / total, alpha1 / total, alpha2 / total


def ref_weno_limiter(coeff, dx, tvb_constant=1.0, epsilon=1e-6, limit_boundaries=False):
    if coeff.shape[1] <= 1:
        return coeff
    troubled = ref_troubled_cell_indicator(
        coeff, dx, tvb_constant, limit_boundaries=limit_boundaries
    )
    if not troubled.any():
        return coeff
    ncells, nmodes, nvar = coeff.shape
    avg = coeff[:, 0]
    avg_pad = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    u_im1 = avg_pad[:-2]
    u_i = avg
    u_ip1 = avg_pad[2:]
    slope0 = (u_i - u_im1) / dx
    slope1 = (u_ip1 - u_i) / dx
    slope2 = (u_ip1 - u_im1) / (2.0 * dx)
    beta0 = dx * slope0 ** 2
    beta1 = dx * slope1 ** 2
    beta2 = (1.0 / 3.0) * (u_im1 - 2.0 * u_i + u_ip1) ** 2 / dx + dx * slope2 ** 2
    beta0_scalar = beta0.max(dim=-1).values
    beta1_scalar = beta1.max(dim=-1).values
    beta2_scalar = beta2.max(dim=-1).values
    w0, w1, w2 = ref_weno_weights(beta0_scalar, beta1_scalar, beta2_scalar, epsilon)
    weno_slope = (
        w0[..., None] * slope0 + w1[..., None] * slope1 + w2[..., None] * slope2
    )
    limited = coeff.clone()
    t = troubled
    limited[t, 1] = weno_slope[t] * (dx / 2.0)
    if nmodes > 2:
        limited[t, 2:] = 0.0
    return limited