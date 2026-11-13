"""EXPERIMENTAL / NOT VERIFIED -- do not use for production work.

See ``docs/CASE-DG-ELASTIC-2D-PSV.md``: the plane-wave error does not converge,
the closed-box elastic energy grows and the semi-discrete operator has positive
real eigenvalues (the face-flux assembly is still anti-dissipative).  The
module is kept in the tree because the failure is instrumented by gates in
``tests/test_elastic_2d_psv.py`` and the diagnosis is reusable.

Two-dimensional linear elastodynamics (P-SV) with a tensor-product modal DG.

The medium is homogeneous and isotropic, described by the first-order
velocity--stress system

.. math::

   \rho\,\dot v_x = \sigma_{xx,x} + \sigma_{xy,y}, \qquad
   \rho\,\dot v_y = \sigma_{xy,x} + \sigma_{yy,y},

   \dot\sigma_{xx} = (\lambda+2\mu) v_{x,x} + \lambda v_{y,y}, \qquad
   \dot\sigma_{yy} = \lambda v_{x,x} + (\lambda+2\mu) v_{y,y}, \qquad
   \dot\sigma_{xy} = \mu (v_{x,y} + v_{y,x}),

i.e. :math:`U_t + F(U)_x + G(U)_y = 0` with
:math:`U = (v_x, v_y, \sigma_{xx}, \sigma_{yy}, \sigma_{xy})`.

Features
--------
* Modal Legendre DG of degree ``p`` on Cartesian cells with the ``p + 1``
  Gauss nodes per direction (standard modal DG quadrature: products of degree
  ``2p`` are integrated exactly, so the discrete operators satisfy the
  summation-by-parts identity and the scheme is conservative).
* Exact upwind (characteristic) interface flux for axis-aligned faces, built
  from the P- (:math:`\pm c_p`) and S- (:math:`\pm c_s`) Riemann invariants
  with the acoustic impedances :math:`Z_p = \rho c_p`, :math:`Z_s = \rho c_s`.
  This is the 2-D analogue of :func:`tensordg.elastic_1d.upwind_flux`.
* Traction-free (``free``) and rigid (``rigid``) characteristic boundary
  conditions, plus periodic ends.  For the free surface the incoming P/S
  characteristics are chosen so that :math:`\sigma_{nn} = \sigma_{nt} = 0` on
  the boundary, which reproduces the classical velocity doubling
  :math:`|v_n| = 2|v_n^{\text{inc}}|` at normal incidence.
* SSP-RK3 (Cockburn--Shu) time integration.

Exact solutions used for verification
-------------------------------------
* :func:`plane_wave` -- a single P or S plane wave
  :math:`U = A\,\hat d \sin(\boldsymbol k\cdot\boldsymbol x - \omega t)` in a
  periodic box.  For a P wave the stress follows
  :math:`\sigma_{nn} = -Z_p v_n`, for an S wave :math:`\sigma_{nt} = -Z_s v_t`;
  both are exact solutions of the PDE, so any deviation is numerical.

References
----------
* Graff, K. F. (1975), *Wave Motion in Elastic Solids*, Ch. 1--2 (P and S
  waves, free-surface reflection, velocity doubling).
* del Aguila, L. & Kaeser, M. (2008), *Accuracy of the spectral-element method
  for elastic wave propagation*, Geophys. J. Int. 173 (characteristic
  decomposition of the 2-D velocity--stress system).
* Wilcox, L. C. et al. (2010), *A high-order discontinuous Galerkin method for
  elastic wave propagation*, J. Comput. Phys. 229 (upwind DG for
  velocity--stress elastodynamics).
* Cockburn, S. & Shu, C.-W. (1998), J. Comput. Phys. 141 (SSP-RK).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "ElasticMedium",
    "legendre_operators_2d",
    "flux_x",
    "flux_y",
    "riemann_flux_x",
    "riemann_flux_y",
    "solve_elastic_2d",
    "elastic_energy",
    "plane_wave",
    "modal_to_quadrature",
    "quadrature_grid",
    "relative_l2_error",
    "fourier_mode_analysis",
    "Elastic2DResult",
]

# supported boundary-condition labels
BC_X = ("periodic",)
BC_Y = ("periodic", "free-rigid")

# state vector layout
_VX, _VY, _SXX, _SYY, _SXY = 0, 1, 2, 3, 4


class ElasticMedium:
    """Homogeneous isotropic elastic medium (Lam\'e parameters + density)."""

    def __init__(self, rho: float = 1.0, lam: float = 1.0, mu: float = 1.0):
        if rho <= 0 or mu <= 0:
            raise ValueError("rho and mu must be positive")
        if lam + 2.0 * mu / 3.0 <= 0:
            raise ValueError("bulk modulus must be positive (lam > -2 mu / 3)")
        self.rho = float(rho)
        self.lam = float(lam)
        self.mu = float(mu)

    @property
    def cp(self) -> float:
        """P-wave speed ``sqrt((lam + 2 mu) / rho)``."""
        return math.sqrt((self.lam + 2.0 * self.mu) / self.rho)

    @property
    def cs(self) -> float:
        """S-wave speed ``sqrt(mu / rho)``."""
        return math.sqrt(self.mu / self.rho)

    @property
    def zp(self) -> float:
        """P impedance ``rho c_p``."""
        return self.rho * self.cp

    @property
    def zs(self) -> float:
        """S impedance ``rho c_s``."""
        return self.rho * self.cs

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"ElasticMedium(rho={self.rho}, lam={self.lam}, mu={self.mu}, "
                f"cp={self.cp:.6f}, cs={self.cs:.6f})")


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
def legendre_operators_2d(degree: int, device="cpu", dtype=torch.float64):
    """Return the 1-D modal DG operators of degree ``degree``.

    Returns ``dict`` with

    ``nodes``    ``(p+1,)``    Gauss-Legendre nodes on ``[-1, 1]``
    ``weights``  ``(p+1,)``    matching quadrature weights
    ``phi``      ``(p+1, p+1)``  ``phi[a, k] = P_k(xi_a)``
    ``dphi``     ``(p+1, p+1)``  ``dphi[a, k] = P_k'(xi_a)``
    ``proj``     ``(p+1, p+1)``  nodal -> modal (``= M^{-1} phi^T W``)
    ``dmat``     ``(p+1, p+1)``  modal derivative matrix (``proj @ dphi``)
    ``svec``     ``(p+1, p+1)``  ``svec[l, b] = w_b P_l(xi_b)``
    ``phi_left``, ``phi_right``  ``(p+1,)``  basis at ``xi = -1`` / ``+1``
    ``mass_inv`` ``(p+1, p+1)``  inverse 2-D mass diagonal
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    k = degree + 1
    nodes_np, weights_np = np.polynomial.legendre.leggauss(k)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)

    values = np.stack([np.polynomial.legendre.legval(nodes_np, e) for e in np.eye(k)],
                      axis=1)
    derivs = np.stack([
        np.polynomial.legendre.legval(
            nodes_np,
            np.polynomial.legendre.legder(e) if k > 1 else np.zeros(1))
        for e in np.eye(k)], axis=1)
    phi = torch.as_tensor(values, device=device, dtype=dtype)
    dphi = torch.as_tensor(derivs, device=device, dtype=dtype)

    mass = torch.as_tensor(2.0 / (2.0 * np.arange(k) + 1.0), device=device, dtype=dtype)
    svec = (phi * weights[:, None]).T                    # (k, nq)
    proj = svec / mass[:, None]                          # (k, nq)
    dmat = proj @ dphi                                   # (k, k)
    phi_left = torch.as_tensor([(-1.0) ** i for i in range(k)], device=device, dtype=dtype)
    phi_right = torch.ones(k, device=device, dtype=dtype)
    mass_inv = torch.outer(1.0 / mass, 1.0 / mass)       # (k, k)
    return {
        "degree": degree,
        "nodes": nodes,
        "weights": weights,
        "phi": phi,
        "dphi": dphi,
        "proj": proj,
        "dmat": dmat,
        "svec": svec,
        "phi_left": phi_left,
        "phi_right": phi_right,
        "mass": mass,
        "mass_inv": mass_inv,
    }


# ---------------------------------------------------------------------------
# physical fluxes (axis aligned faces only)
# ---------------------------------------------------------------------------
def flux_x(u: torch.Tensor, medium: ElasticMedium) -> torch.Tensor:
    """Physical flux in ``x``: ``F(U)`` with ``U = (vx, vy, sxx, syy, sxy)``."""
    rho, lam = medium.rho, medium.lam
    two_mu_lam = lam + 2.0 * medium.mu
    out = torch.empty_like(u)
    out[..., _VX] = -u[..., _SXX] / rho
    out[..., _VY] = -u[..., _SXY] / rho
    out[..., _SXX] = -two_mu_lam * u[..., _VX]
    out[..., _SYY] = -lam * u[..., _VX]
    out[..., _SXY] = -medium.mu * u[..., _VY]
    return out


def flux_y(u: torch.Tensor, medium: ElasticMedium) -> torch.Tensor:
    """Physical flux in ``y``: ``G(U)``."""
    rho, lam = medium.rho, medium.lam
    two_mu_lam = lam + 2.0 * medium.mu
    out = torch.empty_like(u)
    out[..., _VX] = -u[..., _SXY] / rho
    out[..., _VY] = -u[..., _SYY] / rho
    out[..., _SXX] = -lam * u[..., _VY]
    out[..., _SYY] = -two_mu_lam * u[..., _VY]
    out[..., _SXY] = -medium.mu * u[..., _VX]
    return out


def riemann_flux_x(left: torch.Tensor, right: torch.Tensor,
                   medium: ElasticMedium) -> torch.Tensor:
    """Upwind flux ``F* . e_x`` between ``left`` (at ``-x``) and ``right``.

    Exact Godunov flux of the linear system: the P/S Riemann invariants of the
    upwind side are imposed, i.e.
    ``v*_n = (v_n^L + v_n^R)/2 + (s_nn^L - s_nn^R)/(2 Z)``.
    """
    zp, zs = medium.zp, medium.zs
    vx = 0.5 * (left[..., _VX] + right[..., _VX]) + \
        (right[..., _SXX] - left[..., _SXX]) / (2.0 * zp)
    sxx = 0.5 * (left[..., _SXX] + right[..., _SXX]) + \
        0.5 * zp * (right[..., _VX] - left[..., _VX])
    vy = 0.5 * (left[..., _VY] + right[..., _VY]) + \
        (right[..., _SXY] - left[..., _SXY]) / (2.0 * zs)
    sxy = 0.5 * (left[..., _SXY] + right[..., _SXY]) + \
        0.5 * zs * (right[..., _VY] - left[..., _VY])

    lam, two_mu_lam = medium.lam, medium.lam + 2.0 * medium.mu
    out = torch.empty_like(left)
    out[..., _VX] = -sxx / medium.rho
    out[..., _VY] = -sxy / medium.rho
    out[..., _SXX] = -two_mu_lam * vx
    out[..., _SYY] = -lam * vx
    out[..., _SXY] = -medium.mu * vy
    return out


def riemann_flux_y(left: torch.Tensor, right: torch.Tensor,
                   medium: ElasticMedium) -> torch.Tensor:
    """Upwind flux ``F* . e_y`` between ``left`` (at ``-y``) and ``right``."""
    zp, zs = medium.zp, medium.zs
    vy = 0.5 * (left[..., _VY] + right[..., _VY]) + \
        (right[..., _SYY] - left[..., _SYY]) / (2.0 * zp)
    syy = 0.5 * (left[..., _SYY] + right[..., _SYY]) + \
        0.5 * zp * (right[..., _VY] - left[..., _VY])
    vx = 0.5 * (left[..., _VX] + right[..., _VX]) + \
        (right[..., _SXY] - left[..., _SXY]) / (2.0 * zs)
    sxy = 0.5 * (left[..., _SXY] + right[..., _SXY]) + \
        0.5 * zs * (right[..., _VX] - left[..., _VX])

    lam, two_mu_lam = medium.lam, medium.lam + 2.0 * medium.mu
    out = torch.empty_like(left)
    out[..., _VX] = -sxy / medium.rho
    out[..., _VY] = -syy / medium.rho
    out[..., _SXX] = -lam * vy
    out[..., _SYY] = -two_mu_lam * vy
    out[..., _SXY] = -medium.mu * vx
    return out


def _boundary_flux_y(state: torch.Tensor, side: str, bc: str,
                     medium: ElasticMedium) -> torch.Tensor:
    """Outward numerical flux ``F* . n`` on a ``y`` boundary.

    ``side='top'`` (outward normal ``+e_y``) with ``bc='free'``: traction free,
    ``sigma_yy = sigma_xy = 0``; the outgoing characteristics
    ``sigma_nn - Z_p v_n`` and ``sigma_nt - Z_s v_t`` are conserved, hence
    ``v_n* = v_n - sigma_nn/Z_p`` (velocity doubling for an incident wave) and
    ``v_t* = v_t - sigma_nt/Z_s``.

    ``side='bottom'`` (outward normal ``-e_y``) with ``bc='rigid'``: ``v = 0``
    and ``sigma_nn* = sigma_nn + Z_p v_n``, ``sigma_nt* = sigma_nt + Z_s v_t``.
    """
    zp, zs = medium.zp, medium.zs
    lam, two_mu_lam = medium.lam, medium.lam + 2.0 * medium.mu
    out = torch.empty_like(state)

    if side == "top" and bc == "free":
        # normal +e_y: v_n = v_y, s_nn = s_yy, s_nt = -s_xy
        vy_star = state[..., _VY] - state[..., _SYY] / zp
        vx_star = state[..., _VX] - state[..., _SXY] / zs
        out[..., _VX] = torch.zeros_like(vx_star)          # -s_xy*/rho with s_xy* = 0
        out[..., _VY] = torch.zeros_like(vy_star)          # -s_yy*/rho with s_yy* = 0
        out[..., _SXX] = -lam * vy_star
        out[..., _SYY] = -two_mu_lam * vy_star
        out[..., _SXY] = -medium.mu * vx_star
        return out

    if side == "bottom" and bc == "rigid":
        # normal -e_y: v_n = -v_y, s_nn = s_yy, s_nt = -s_xy ; v* = 0
        syy_star = state[..., _SYY] - zp * state[..., _VY]
        sxy_star = state[..., _SXY] - zs * state[..., _VX]
        out[..., _VX] = sxy_star / medium.rho
        out[..., _VY] = syy_star / medium.rho
        out[..., _SXX] = torch.zeros_like(syy_star)        # lam * vy* with vy* = 0
        out[..., _SYY] = torch.zeros_like(syy_star)
        out[..., _SXY] = torch.zeros_like(syy_star)
        return out

    raise ValueError(f"unsupported y boundary: side={side!r}, bc={bc!r}")


# ---------------------------------------------------------------------------
# solution container
# ---------------------------------------------------------------------------
class Elastic2DResult:
    """Result of :func:`solve_elastic_2d`.

    ``coeff`` holds the final modal coefficients of shape
    ``(nx, ny, p+1, p+1, 5)``; ``snapshots`` maps a requested time to the
    coefficients at that time (``final_time`` is always included).
    """

    def __init__(self, coeff, medium, ops, geometry, times, snapshots,
                 bc_x, bc_y, cfl):
        self.coeff = coeff
        self.medium = medium
        self.ops = ops
        self.nx = geometry["nx"]
        self.ny = geometry["ny"]
        self.length_x = geometry["length_x"]
        self.length_y = geometry["length_y"]
        self.hx = geometry["hx"]
        self.hy = geometry["hy"]
        self.times = times
        self.snapshots = snapshots
        self.bc_x = bc_x
        self.bc_y = bc_y
        self.cfl = cfl
        self.degree = ops["degree"]

    def fields(self, coeff=None) -> torch.Tensor:
        """State at the quadrature points, shape ``(nx, ny, p+1, p+1, 5)``."""
        c = self.coeff if coeff is None else coeff
        return modal_to_quadrature(c, self.ops["phi"])

    def quadrature_grid(self):
        """Physical quadrature coordinates ``(x, y)`` of shape ``(nx, ny, p+1, p+1)``."""
        return quadrature_grid(self.nx, self.ny, self.hx, self.hy, self.ops["nodes"])

    def energy(self, coeff=None) -> torch.Tensor:
        c = self.coeff if coeff is None else coeff
        return elastic_energy(c, self.ops, self.medium, self.hx, self.hy)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def quadrature_grid(nx: int, ny: int, hx: float, hy: float,
                    nodes: torch.Tensor):
    xi = nodes[None, None, :, None]
    eta = nodes[None, None, None, :]
    x = hx * (torch.arange(nx, device=nodes.device, dtype=nodes.dtype)[:, None, None, None]
              + 0.5 * (1.0 + xi))
    y = hy * (torch.arange(ny, device=nodes.device, dtype=nodes.dtype)[None, :, None, None]
              + 0.5 * (1.0 + eta))
    x = x.expand(nx, ny, nodes.numel(), nodes.numel()) if False else \
        hx * (torch.arange(nx, device=nodes.device, dtype=nodes.dtype)[:, None, None, None]
              + 0.5 * (1.0 + xi)).expand(nx, ny, nodes.numel(), nodes.numel())
    y = hy * (torch.arange(ny, device=nodes.device, dtype=nodes.dtype)[None, :, None, None]
              + 0.5 * (1.0 + eta)).expand(nx, ny, nodes.numel(), nodes.numel())
    return x, y


def modal_to_quadrature(coeff: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Evaluate modal coefficients at the Gauss points of every cell."""
    return torch.einsum("ak,bl,ijklm->ijabm", phi, phi, coeff)


def _trace(coeff: torch.Tensor, phi_face: torch.Tensor, axis: int,
           phi: torch.Tensor) -> torch.Tensor:
    """One-sided trace on an ``axis`` face, at the other direction's Gauss nodes.

    ``axis=0`` returns the field at ``xi = +-1`` for every ``eta`` Gauss node,
    shape ``(nx, ny, nq_eta, 5)``; ``axis=1`` returns ``(nx, ny, nq_xi, 5)``.
    Both the face direction (basis evaluated at the end point) and the
    in-face direction (basis evaluated at the Gauss nodes) are handled here, so
    no modal index survives in the returned tensor.
    """
    if axis == 0:
        return torch.einsum("k,bl,ijklm->ijbm", phi_face, phi, coeff)
    return torch.einsum("l,ak,ijklm->ijam", phi_face, phi, coeff)


def elastic_energy(coeff: torch.Tensor, ops, medium: ElasticMedium,
                   hx: float, hy: float) -> torch.Tensor:
    """Total elastic energy ``int (1/2 rho |v|^2 + 1/2 sigma:eps) dA``."""
    u = modal_to_quadrature(coeff, ops["phi"])
    w = ops["weights"]
    w2 = torch.outer(w, w).reshape(1, 1, w.numel(), w.numel(), 1)
    vx, vy = u[..., _VX], u[..., _VY]
    sxx, syy, sxy = u[..., _SXX], u[..., _SYY], u[..., _SXY]
    kinetic = 0.5 * medium.rho * (vx ** 2 + vy ** 2)
    mu, lam = medium.mu, medium.lam
    strain = (sxx ** 2 + syy ** 2 + 2.0 * sxy ** 2) / (4.0 * mu) \
        - lam * (sxx + syy) ** 2 / (4.0 * mu * (3.0 * lam + 2.0 * mu))
    cell = (kinetic + strain) * w2
    return cell.sum() * (hx * hy / 4.0)


def relative_l2_error(numerical: torch.Tensor, exact: torch.Tensor,
                      weights: torch.Tensor) -> torch.Tensor:
    """Quadrature-weighted relative ``L2`` error (cell scaling cancels)."""
    w = weights
    w2 = torch.outer(w, w).reshape(1, 1, w.numel(), w.numel(), 1)
    num = ((numerical - exact) ** 2 * w2).sum()
    den = (exact ** 2 * w2).sum()
    return torch.sqrt(num / den.clamp_min(1e-300))


def plane_wave(mode: str, kx: float, ky: float, amplitude: float,
               medium: ElasticMedium, phase: float = 0.0) -> Callable:
    """Exact single plane wave of the 2-D elastic system.

    Returns ``f(x, y, t) -> U`` with ``U`` of shape ``(..., 5)``.

    * ``mode='P'``: polarisation along ``n = k/|k|``, speed ``c_p``,
      ``sigma_nn = -Z_p v_n`` and ``sigma_tt = -(lambda A / c_p) s`` (the
      isotropic part of the P-wave stress is *not* zero -- omitting it makes
      the reference fail to solve the PDE).
    * ``mode='S'``: polarisation along ``t = (-n_y, n_x)``, speed ``c_s``,
      ``sigma_nt = -Z_s v_t``.

    Both are exact solutions of the velocity--stress system, i.e. the
    reference has no physical dispersion.
    """
    knorm = math.hypot(kx, ky)
    if knorm == 0:
        raise ValueError("wavenumber must be non-zero")
    nx, ny = kx / knorm, ky / knorm
    tx, ty = -ny, nx
    mode = mode.upper()
    if mode == "P":
        speed, impedance = medium.cp, medium.zp
        pol = (nx, ny)
        normal_stress = True
    elif mode == "S":
        speed, impedance = medium.cs, medium.zs
        pol = (tx, ty)
        normal_stress = False
    else:
        raise ValueError("mode must be 'P' or 'S'")
    omega = speed * knorm

    def exact(x, y, t=0.0):
        s = torch.sin(kx * x + ky * y - omega * t + phase)
        vx = amplitude * pol[0] * s
        vy = amplitude * pol[1] * s
        amp = -impedance * amplitude
        if normal_stress:
            # sigma = sigma_nn n (x) n + sigma_tt t (x) t with
            # sigma_nn = -Z_p A s and sigma_tt = -(lambda A / c_p) s
            two_mu_lam = medium.lam + 2.0 * medium.mu
            coef_nn = -amplitude * two_mu_lam / medium.cp
            coef_tt = -amplitude * medium.lam / medium.cp
            sxx = (coef_nn * nx * nx + coef_tt * tx * tx) * s
            syy = (coef_nn * ny * ny + coef_tt * ty * ty) * s
            sxy = (coef_nn * nx * ny + coef_tt * tx * ty) * s
        else:
            # sigma_nt = -Z_s A sin ; sigma = sigma_nt (n (x) t + t (x) n)
            sxx = amp * 2.0 * nx * tx * s
            syy = amp * 2.0 * ny * ty * s
            sxy = amp * (nx * ty + tx * ny) * s
        out = torch.empty(*s.shape, 5, device=s.device, dtype=s.dtype)
        out[..., _VX] = vx
        out[..., _VY] = vy
        out[..., _SXX] = sxx
        out[..., _SYY] = syy
        out[..., _SXY] = sxy
        return out

    exact.omega = omega          # type: ignore[attr-defined]
    exact.speed = speed          # type: ignore[attr-defined]
    exact.knormal = (nx, ny)     # type: ignore[attr-defined]
    return exact


def fourier_mode_analysis(coeff_or_fields: torch.Tensor, x: torch.Tensor,
                          y: torch.Tensor, kx: float, ky: float,
                          weights: torch.Tensor) -> complex:
    """Complex amplitude of the Fourier mode ``exp(-i(kx x + ky y))``.

    Used to measure numerical phase lag and amplitude damping of a plane wave
    (numerical dispersion / dissipation) without any fitting.
    """
    u = coeff_or_fields
    w2 = torch.outer(weights, weights).reshape(1, 1, weights.numel(), weights.numel(), 1)
    phase = torch.exp(-1j * (kx * x + ky * y))
    # combine the two velocity components so that both P and S waves register
    signal = (u[..., _VX] + u[..., _VY]) * w2
    val = (signal.to(torch.complex128) * phase).sum()
    return complex(val.item())


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
def _rhs(coeff: torch.Tensor, ops, medium: ElasticMedium, hx: float, hy: float,
         bc_x: str, bc_y: str) -> torch.Tensor:
    """Semi-discrete DG right-hand side in modal coefficients.

    Strong form with exact face corrections:

    .. math::

       \dot c_{kl} = -M^{-1}_{kl}\Big[\int \phi_k\phi_l \nabla\cdot F\Big]
       + M^{-1}_{kl}\Big[\phi_k(\pm1)\,(\phi_l, F^*_n - F_n)\Big]
    """
    phi, proj, dmat = ops["phi"], ops["proj"], ops["dmat"]
    svec, mass_inv = ops["svec"], ops["mass_inv"]
    phi_left, phi_right = ops["phi_left"], ops["phi_right"]
    k = phi.shape[1]

    u = modal_to_quadrature(coeff, phi)
    fx = flux_x(u, medium)
    fy = flux_y(u, medium)
    cfx = torch.einsum("ka,lb,ijabm->ijklm", proj, proj, fx)
    cfy = torch.einsum("ka,lb,ijabm->ijklm", proj, proj, fy)

    # ---- volume (exact for polynomial fluxes of degree <= p) ---------------
    rhs = -(2.0 / hx) * torch.einsum("kn,ijnlm->ijklm", dmat, cfx) \
          -(2.0 / hy) * torch.einsum("ln,ijknm->ijklm", dmat, cfy)

    # Face contributions carry the reference->physical mapping factor 2/h of
    # the face direction (same factor as the volume term); omitting it leaves
    # the numerical-flux correction under-weighted by O(h), which destroys
    # consistency and skew-symmetry (see docs/CASE-DG-ELASTIC-2D-PSV.md).
    face_x = 2.0 / hx
    face_y = 2.0 / hy
    mass = mass_inv.view(1, 1, k, k, 1)
    pr = phi_right.view(1, 1, k, 1, 1)
    pl = phi_left.view(1, 1, k, 1, 1)

    # ---- x faces ----------------------------------------------------------
    u_l = _trace(coeff, phi_left, 0, phi)     # (nx, ny, nq_eta, 5)
    u_r = _trace(coeff, phi_right, 0, phi)
    if bc_x != "periodic":  # pragma: no cover - only periodic x supported
        raise ValueError(f"unsupported x boundary condition {bc_x!r}")
    right_neighbour_left = torch.roll(u_l, shifts=-1, dims=0)   # trace of cell i+1
    star_r = riemann_flux_x(right_neighbour_left, u_r, medium)  # F*.e_x at i+1/2
    g_r = star_r - flux_x(u_r, medium)
    proj_r = torch.einsum("lb,ijbm->ijlm", svec, g_r)           # (nx,ny,K_l,5)
    rhs = rhs - face_x * mass * pr * proj_r.view(proj_r.shape[0], proj_r.shape[1], 1,
                                        proj_r.shape[2], proj_r.shape[3])

    left_neighbour_right = torch.roll(u_r, shifts=1, dims=0)    # trace of cell i-1
    star_l = riemann_flux_x(left_neighbour_right, u_l, medium)  # F*.e_x at i-1/2
    # outward normal is -e_x: F*_n = -star_l, F_n^int = -flux_x(u_l)
    g_l = (-star_l) + flux_x(u_l, medium)
    proj_l = torch.einsum("lb,ijbm->ijlm", svec, g_l)
    rhs = rhs - face_x * mass * pl * proj_l.view(proj_l.shape[0], proj_l.shape[1], 1,
                                        proj_l.shape[2], proj_l.shape[3])

    # ---- y faces ----------------------------------------------------------
    u_b = _trace(coeff, phi_left, 1, phi)     # (nx, ny, nq_xi, 5)
    u_t = _trace(coeff, phi_right, 1, phi)
    # outward-normal numerical flux on the top (+) and bottom (-) faces
    if bc_y == "periodic":
        above_bottom = torch.roll(u_b, shifts=-1, dims=1)
        star_t = riemann_flux_y(above_bottom, u_t, medium)      # F*.e_y at j+1/2
        below_top = torch.roll(u_t, shifts=1, dims=1)
        star_b = -riemann_flux_y(below_top, u_b, medium)        # outward -e_y
    elif bc_y == "free-rigid":
        star_t = riemann_flux_y(torch.zeros_like(u_t), u_t, medium)
        star_t[:, -1] = _boundary_flux_y(u_t[:, -1], "top", "free", medium)
        star_b = -riemann_flux_y(torch.zeros_like(u_b), u_b, medium)
        star_b[:, 0] = _boundary_flux_y(u_b[:, 0], "bottom", "rigid", medium)
    else:  # pragma: no cover
        raise ValueError(f"unsupported y boundary condition {bc_y!r}")

    g_t = star_t - flux_y(u_t, medium)

    proj_t = torch.einsum("ka,ijam->ijkm", svec, g_t)           # (nx,ny,K_k,5)
    pr_l = phi_right.view(1, 1, 1, k, 1)
    rhs = rhs - face_y * mass * pr_l * proj_t.view(proj_t.shape[0], proj_t.shape[1],
                                          proj_t.shape[2], 1, proj_t.shape[3])

    g_b = star_b + flux_y(u_b, medium)   # F*_n - F_n^int with F_n^int = -flux_y
    proj_b = torch.einsum("ka,ijam->ijkm", svec, g_b)
    pl_l = phi_left.view(1, 1, 1, k, 1)
    rhs = rhs - face_y * mass * pl_l * proj_b.view(proj_b.shape[0], proj_b.shape[1],
                                          proj_b.shape[2], 1, proj_b.shape[3])
    return rhs


def solve_elastic_2d(initial_state: Callable, *, nx: int = 16, ny: int = 16,
                     degree: int = 2, final_time: float = 0.2,
                     length_x: float = 1.0, length_y: float = 1.0,
                     medium: Optional[ElasticMedium] = None,
                     cfl: float = 0.2, bc_x: str = "periodic",
                     bc_y: str = "periodic",
                     snapshot_times: Sequence[float] = (),
                     dtype=torch.float64, device="cpu") -> Elastic2DResult:
    """March the 2-D velocity--stress system in time with DG + SSP-RK3.

    ``initial_state(x, y)`` must return a tensor of shape ``(..., 5)`` with the
    components ``(vx, vy, sxx, syy, sxy)``; ``final_time`` and the entries of
    ``snapshot_times`` are always included in the returned snapshots.
    """
    if bc_x not in BC_X:
        raise ValueError(f"bc_x must be one of {BC_X}")
    if bc_y not in BC_Y:
        raise ValueError(f"bc_y must be one of {BC_Y}")
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    if cfl <= 0:
        raise ValueError("cfl must be positive")
    medium = medium or ElasticMedium()

    ops = legendre_operators_2d(degree, device=device, dtype=dtype)
    hx, hy = length_x / nx, length_y / ny

    x, y = quadrature_grid(nx, ny, hx, hy, ops["nodes"])
    u0 = initial_state(x, y)
    if u0.shape != (nx, ny, degree + 1, degree + 1, 5):
        u0 = u0.expand(nx, ny, degree + 1, degree + 1, 5)
    coeff = torch.einsum("ka,lb,ijabm->ijklm", ops["proj"], ops["proj"], u0)

    wavespeed = medium.cp  # fastest characteristic speed
    dt_max = cfl * min(hx, hy) / ((2.0 * degree + 1.0) * wavespeed)
    n_steps = max(1, int(math.ceil(final_time / dt_max)))
    dt = final_time / n_steps

    phys = {"medium": medium, "hx": hx, "hy": hy}

    def rhs(c):
        return _rhs(c, ops, medium, hx, hy, bc_x, bc_y)

    wanted = sorted({float(t) for t in snapshot_times} | {float(final_time)})
    snapshots = {}
    t = 0.0
    idx = 0
    if wanted and wanted[0] <= 1e-15:
        snapshots[wanted[0]] = coeff.clone()
        idx = 1
    for step in range(n_steps):
        # SSP-RK3 (Shu & Osher 1988), the standard DG time integrator
        u1 = coeff + dt * rhs(coeff)
        u2 = 0.75 * coeff + 0.25 * (u1 + dt * rhs(u1))
        coeff = coeff / 3.0 + (2.0 / 3.0) * (u2 + dt * rhs(u2))
        t = (step + 1) * dt
        while idx < len(wanted) and t >= wanted[idx] - 1e-12:
            snapshots[wanted[idx]] = coeff.clone()
            idx += 1

    geometry = {"nx": nx, "ny": ny, "length_x": length_x, "length_y": length_y,
                "hx": hx, "hy": hy}
    del phys
    return Elastic2DResult(coeff, medium, ops, geometry, (dt, n_steps),
                           snapshots, bc_x, bc_y, cfl)