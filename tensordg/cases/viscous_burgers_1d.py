"""CASE-DG-VISCOUS-BURGERS-COLEHOPF: periodic viscous Burgers vs the Cole--Hopf exact solution.

Why this case exists
--------------------
TensorDG verifies its two halves separately and (for the *steady, linear* layer
problem) jointly:

* nonlinear hyperbolic flux -- :mod:`tensordg.burgers_1d` (inviscid, ``nu = 0``),
  :mod:`tensordg.euler_dg_1d`, :mod:`tensordg.swe_1d`;
* symmetric interior penalty diffusion -- :mod:`tensordg.heat_1d`,
  :mod:`tensordg.sipg_sparse`, :mod:`tensordg.sipg_matrixfree`;
* both at once -- :mod:`tensordg.advection_diffusion_1d`, but only on a
  **steady linear** advection--diffusion layer with homogeneous Dirichlet data.

Nothing in the repository ever exercised a **nonlinear flux together with a
diffusive operator in an unsteady march**, which is exactly the operator
combination every Navier--Stokes/ice-transport step uses.  The classical
viscous Burgers equation

    u_t + (u^2/2)_x = nu u_xx,    x in (0, 1) periodic,   nu > 0,          (1)

closes that gap *and* has an exact solution, so the joint operator is compared
against analysis rather than against itself.

Reference (Cole--Hopf)
----------------------
Cole (1951) and Hopf (1950): with the substitution ``u = -2 nu theta_x / theta``
equation (1) is equivalent to the linear heat equation ``theta_t = nu theta_xx``
with

    theta(x, 0) = exp( -(1/(2 nu)) * int_0^x u_0(s) ds ).                   (2)

For the periodic sine data used here, ``u_0(x) = sin(2 pi x)``, the inner
integral is elementary, ``int_0^x u_0 = (1 - cos 2 pi x)/(2 pi)``, so

    theta_0(x) = exp( -(1 - cos 2 pi x) / (4 pi nu) ),                       (3)

and the periodic heat kernel propagates it::

    theta(x, t) = int_0^1 K_p(x - s, t) theta_0(s) ds,
    K_p(y, t)   = sum_{n in Z} (4 pi nu t)^(-1/2) exp( -(y + n)^2 / (4 nu t) ).

``theta`` is positive for all ``t > 0`` (a maximum principle on the heat
equation), so ``u`` is smooth: the inviscid shock of ``u_0`` is regularised into
an internal layer of width ``O(nu)``.  The reference is evaluated with the
midpoint (= trapezoid for a periodic integrand) rule, which is spectrally
accurate here, and cross-checked against a **second, independent
implementation** (:func:`exact_solution_fourier`: FFT of ``theta_0`` followed by
the exact-in-time heat decay ``exp(-nu (2 pi k)^2 t)``).  The two agree to
round-off; the tests pin that agreement, so a coding error in one of them cannot
masquerade as "the reference".

Discretisation
--------------
The semi-discrete operator is *assembled from the production kernels*, not
re-derived:

* advective part -- :func:`tensordg.burgers_1d._rhs` (modal Legendre weak DG
  with a Rusanov interface flux) is called verbatim;
* diffusive part -- :func:`periodic_sipg` which is the **same face stencil** as
  :func:`tensordg.heat_1d.assemble_sipg_1d` (``-{grad u}.[phi] - {grad phi}.[u]
  + sigma [u][phi]`` with ``sigma = penalty (p+1)^2 / h``), closed into a ring
  instead of being cut off at a Dirichlet boundary.  ``tests`` assert that the
  ring operator's interior rows are *identical* to the production matrix's
  interior rows, that it reproduces ``nu u_xx`` exactly on global polynomials,
  and that it is SPD.

Time integration is classical RK4.  The step is chosen from the measured
generalised spectrum of ``M^{-1} K`` (``M`` = modal mass matrix, ``K`` = SIPG
stiffness) so the diffusive restriction, which dominates at these ``nu``, is met
without guess-work; see :func:`stable_timestep`.

References
----------
* J. D. Cole (1951), "On a quasi-linear parabolic equation occurring in
  aerodynamics", *Quart. Appl. Math.* 9, 225-236.
* E. Hopf (1950), "The partial differential equation u_t + u u_x = mu u_xx",
  *Comm. Pure Appl. Math.* 3, 201-230.
* G. B. Whitham (1974), *Linear and Nonlinear Waves*, Wiley, section 4.1
  (the Cole--Hopf transform and the regularised shock).
* D. A. Koubogiannis & P. Chaviaropoulos / B. Cockburn (any DG reference) for
  the SIPG face form used by :mod:`tensordg.heat_1d`.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from ..burgers_1d import _operators as _flux_operators
from ..burgers_1d import _rhs as _burgers_rhs
from ..heat_1d import _operators as _diffusion_operators

#: Registry key of this case (kept stable for report/benchmark plumbing).
CASE_KEY = "DG-VISCOUS-BURGERS-COLEHOPF"

#: Documented parameter set (see ``docs/CASE-DG-VISCOUS-BURGERS-COLEHOPF.md``).
NU = 0.05
FINAL_TIME = 0.2
DEGREE = 2
CELLS = 16
PENALTY = 20.0

#: Acceptance tolerance on the relative L2 error of the documented run.
TOLERANCE = 0.03

#: RK4 stability radius on the negative real axis (``|R(z)| <= 1`` for
#: ``z in [-2.785, 0]``); the diffusive eigenvalues of ``M^{-1}K`` are real
#: negative, so this is the relevant bound.
RK4_NEGATIVE_REAL_BOUND = 2.785

__all__ = [
    "CASE_KEY",
    "NU",
    "FINAL_TIME",
    "DEGREE",
    "CELLS",
    "PENALTY",
    "TOLERANCE",
    "initial_profile",
    "theta_initial",
    "exact_solution",
    "exact_solution_fourier",
    "periodic_sipg",
    "stable_timestep",
    "solve_viscous_burgers_1d",
    "l2_error",
    "total_mass",
    "total_energy",
    "viscous_burgers_parameters",
]


def viscous_burgers_parameters() -> dict:
    """The documented parameter set (single source of truth for docs/tests)."""
    return {
        "case_key": CASE_KEY,
        "nu": NU,
        "final_time": FINAL_TIME,
        "degree": DEGREE,
        "cells": CELLS,
        "penalty": PENALTY,
        "tolerance": TOLERANCE,
        "initial_condition": "sin(2*pi*x)",
        "domain": (0.0, 1.0),
        "boundary": "periodic",
    }


# ---------------------------------------------------------------------------
# analytic reference (Cole--Hopf)
# ---------------------------------------------------------------------------
def initial_profile(x) -> np.ndarray:
    """Periodic sine initial data ``u_0(x) = sin(2 pi x)``."""
    return np.sin(2.0 * math.pi * np.asarray(x, dtype=float))


def theta_initial(x, nu: float = NU) -> np.ndarray:
    """Cole--Hopf datum (3): ``theta_0 = exp(-(1 - cos 2 pi x)/(4 pi nu))``."""
    if nu <= 0.0:
        raise ValueError("nu must be positive for the Cole-Hopf reference")
    return np.exp(-(1.0 - np.cos(2.0 * math.pi * np.asarray(x, dtype=float))) / (4.0 * math.pi * nu))


def exact_solution(x, final_time: float = FINAL_TIME, nu: float = NU,
                   samples: int = 512, images: int = 6) -> np.ndarray:
    """Cole--Hopf solution of (1) with the periodic sine datum.

    ``u(x, t) = -2 nu theta_x / theta`` with ``theta`` the periodic heat-kernel
    convolution of :func:`theta_initial`.  The integral is evaluated with the
    midpoint rule on ``samples`` uniform points (spectrally accurate for this
    analytic periodic integrand; the tests show the value is stable to 1e-15
    from 128 to 4096 points) and the kernel image sum is truncated where
    ``exp(-(y+n)^2/(4 nu t))`` has underflowed.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if final_time <= 0.0:
        return initial_profile(x)
    if nu <= 0.0:
        raise ValueError("the Cole-Hopf reference needs nu > 0")
    if samples < 8:
        raise ValueError("samples must be at least 8")
    s = (np.arange(samples) + 0.5) / samples
    ds = 1.0 / samples
    y = x[:, None] - s[None, :]
    y = y - np.round(y)                       # wrap onto the periodic ring
    sigma2 = 4.0 * nu * final_time
    kernel = np.zeros_like(y)
    kernel_derivative = np.zeros_like(y)
    for n in range(-images, images + 1):
        gauss = np.exp(-(y + n) ** 2 / sigma2) / math.sqrt(math.pi * sigma2)
        kernel += gauss
        kernel_derivative += -(y + n) / (2.0 * nu * final_time) * gauss
    theta0 = theta_initial(s, nu)
    theta = (kernel * theta0[None, :]).sum(axis=1) * ds
    theta_x = (kernel_derivative * theta0[None, :]).sum(axis=1) * ds
    return -2.0 * nu * theta_x / theta


def exact_solution_fourier(x, final_time: float = FINAL_TIME, nu: float = NU,
                           modes: int = 256) -> np.ndarray:
    """Independent implementation of the same reference (cross-check).

    ``theta_0`` is transformed by FFT, advanced **exactly in time** by the heat
    symbol ``exp(-nu (2 pi k)^2 t)`` and differentiated spectrally.  Different
    quadrature, different differentiation, same mathematics: agreement with
    :func:`exact_solution` is evidence that the analytic formula was coded
    correctly.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if final_time <= 0.0:
        return initial_profile(x)
    if nu <= 0.0:
        raise ValueError("the Cole-Hopf reference needs nu > 0")
    s = np.arange(modes) / modes                 # nodal samples: no half-cell shift
    wavenumber = np.fft.fftfreq(modes, d=1.0 / modes)
    theta_hat = np.fft.fft(theta_initial(s, nu)) * np.exp(-nu * (2.0 * math.pi * wavenumber) ** 2 * final_time)
    basis = np.exp(2j * math.pi * np.outer(x, wavenumber))
    theta = (basis @ theta_hat).real / modes
    theta_x = (basis @ (theta_hat * 2j * math.pi * wavenumber)).real / modes
    return -2.0 * nu * theta_x / theta


# ---------------------------------------------------------------------------
# production-kernel assembly
# ---------------------------------------------------------------------------
def _project(cells: int, degree: int, function: Callable[[np.ndarray], np.ndarray],
             dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """L2 projection of the *physical* field ``function(x)`` onto the DG space.

    ``function`` is called with physical coordinates of shape ``[cells, nq]``
    (uniform cells on ``[0, 1]``); the returned modal coefficients are
    ``[cells, degree + 1]`` in the reference Legendre basis.
    """
    nodes, weights, phi, *_ = _flux_operators(degree, torch.device("cpu"), dtype)
    nodes_np = nodes.numpy()
    weights_np = weights.numpy()
    phi_np = phi.numpy()
    h = 1.0 / cells
    centers = (np.arange(cells) + 0.5) * h
    x = centers[:, None] + 0.5 * h * nodes_np[None, :]
    values = np.broadcast_to(np.asarray(function(x), dtype=float), x.shape)
    coeff = np.empty((cells, degree + 1), dtype=float)
    for k in range(degree + 1):
        coeff[:, k] = (2 * k + 1) * 0.5 * np.sum(weights_np[None, :] * values * phi_np[None, :, k], axis=1)
    return torch.as_tensor(coeff, dtype=dtype)


def periodic_sipg(ncells: int, degree: int, penalty: float = PENALTY,
                  device: str = "cpu", dtype: torch.dtype = torch.float64):
    """Periodic SIPG stiffness ``K`` and modal mass diagonal on ``[0, 1]``.

    The face block is copied from :func:`tensordg.heat_1d.assemble_sipg_1d`
    (same ``jump``, ``average_derivative`` and penalty scaling); only the
    boundary cutoff is replaced by ``cell -> (cell + 1) % ncells`` wrapping.
    Returns ``(K, mass_diag, h)`` with ``mass_diag`` the diagonal of the modal
    mass matrix ``M`` (``M_kk = h / (2k + 1)``), so that ``M du/dt = -nu K u``.
    """
    if ncells < 2:
        raise ValueError("ncells must be at least 2")
    if degree < 1:
        raise ValueError("the periodic SIPG operator needs degree >= 1")
    if penalty <= 0.0:
        raise ValueError("penalty must be positive")
    dev = torch.device(device)
    (_, _, _, _, stiffness, phi_left, phi_right,
     dphi_left, dphi_right) = _diffusion_operators(degree, dev, dtype)
    modes = degree + 1
    h = 1.0 / ncells
    ndofs = ncells * modes
    stiffness_matrix = torch.zeros((ndofs, ndofs), device=dev, dtype=dtype)

    local = torch.arange(modes, device=dev)
    dofs = lambda cell: cell * modes + local

    volume = (2.0 / h) * stiffness
    for cell in range(ncells):
        idx = dofs(cell)
        stiffness_matrix[idx[:, None], idx[None, :]] += volume

    jump = torch.cat((phi_right, -phi_left))
    average_derivative = 0.5 * (2.0 / h) * torch.cat((dphi_right, dphi_left))
    consistency = -torch.outer(jump, average_derivative)
    face = (consistency + consistency.T
            + penalty * modes * modes / h * torch.outer(jump, jump))
    for cell in range(ncells):
        idx = torch.cat((dofs(cell), dofs((cell + 1) % ncells)))
        stiffness_matrix[idx[:, None], idx[None, :]] += face

    mass_diag = torch.cat([
        torch.full((1,), h / (2 * k + 1), device=dev, dtype=dtype) for k in range(modes)
    ]).repeat(ncells)
    return stiffness_matrix, mass_diag, h


def stable_timestep(ncells: int, degree: int, nu: float, cfl: float = 0.2,
                    penalty: float = PENALTY, safety: float = 0.9,
                    stiffness=None, mass_diag=None) -> float:
    """Largest timestep that respects both the RK4 diffusive and advective limits.

    The diffusive restriction uses the *measured* largest eigenvalue of
    ``M^{-1}K`` (symmetrised as ``M^{-1/2} K M^{-1/2}``, whose spectrum is real
    and non-negative): RK4 is stable on ``z in [-2.785, 0]``, hence
    ``dt <= 2.785 / (nu lambda_max)``.  The advective limit is ``cfl * h`` with
    the wave speed bounded by ``1`` for the sine datum (it is attained by the
    initial profile, so no estimate is needed).
    """
    if stiffness is None or mass_diag is None:
        stiffness, mass_diag, h = periodic_sipg(ncells, degree, penalty)
    else:
        h = 1.0 / ncells
    if nu > 0.0:
        inverse_sqrt_mass = torch.diag(1.0 / torch.sqrt(mass_diag))
        lambda_max = float(torch.linalg.eigvalsh(inverse_sqrt_mass @ stiffness @ inverse_sqrt_mass).max())
        diffusive = safety * RK4_NEGATIVE_REAL_BOUND / (nu * lambda_max)
    else:
        lambda_max = 0.0
        diffusive = math.inf
    advective = cfl * h
    return min(diffusive, advective), lambda_max


def _rk4(rhs: Callable[[torch.Tensor], torch.Tensor], u: torch.Tensor,
         dt: float, steps: int) -> torch.Tensor:
    for _ in range(steps):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return u


def rhs_operator(ncells: int, degree: int, nu: float, penalty: float = PENALTY,
                 dtype: torch.dtype = torch.float64):
    """Semi-discrete operator ``du/dt`` = production Burgers flux + ``-nu M^{-1} K u``."""
    (_, weights, phi, dphi, phi_left, phi_right) = _flux_operators(
        degree, torch.device("cpu"), dtype)
    stiffness, mass_diag, h = periodic_sipg(ncells, degree, penalty, dtype=dtype)
    diffusion = nu * stiffness / mass_diag[:, None]

    def rhs(u: torch.Tensor) -> torch.Tensor:
        advective = _burgers_rhs(u, h, weights, phi, dphi, phi_left, phi_right)
        return advective - (diffusion @ u.reshape(-1)).reshape(u.shape)

    return rhs, stiffness, mass_diag, h


def solve_viscous_burgers_1d(ncells: int = CELLS, degree: int = DEGREE,
                             final_time: float = FINAL_TIME, nu: float = NU,
                             penalty: float = PENALTY, cfl: float = 0.2,
                             safety: float = 0.9, dtype: torch.dtype = torch.float64):
    """March (1) to ``final_time`` and return the field plus its diagnostics."""
    if final_time < 0.0:
        raise ValueError("final_time must be non-negative")
    if nu < 0.0:
        raise ValueError("nu must be non-negative")
    rhs, stiffness, mass_diag, h = rhs_operator(ncells, degree, nu, penalty, dtype=dtype)
    timestep, lambda_max = stable_timestep(
        ncells, degree, nu, cfl=cfl, penalty=penalty, safety=safety,
        stiffness=stiffness, mass_diag=mass_diag,
    )
    steps = 1 if final_time == 0.0 else int(math.ceil(final_time / timestep))
    dt = final_time / steps if steps else 0.0
    centers = (np.arange(ncells) + 0.5) * h
    u = _project(ncells, degree, initial_profile, dtype=dtype)
    u = _rk4(rhs, u, dt, steps)
    return {
        "case_key": CASE_KEY,
        "ncells": ncells,
        "degree": degree,
        "final_time": final_time,
        "nu": nu,
        "penalty": penalty,
        "h": h,
        "dt": dt,
        "steps": steps,
        "lambda_max": lambda_max,
        "centers": centers,
        "coefficients": u,
        "dtype": dtype,
    }


# ---------------------------------------------------------------------------
# functionals
# ---------------------------------------------------------------------------
def _quadrature_field(result) -> tuple[np.ndarray, np.ndarray]:
    degree = result["degree"]
    h = result["h"]
    nodes, weights, phi, *_ = _flux_operators(degree, torch.device("cpu"), result["dtype"])
    nodes_np = nodes.numpy()
    weights_np = weights.numpy()
    phi_np = phi.numpy()
    coefficients = result["coefficients"].numpy()
    x = result["centers"][:, None] + 0.5 * h * nodes_np[None, :]
    values = coefficients @ phi_np.T
    return x.reshape(-1), values.reshape(-1), h, weights_np


def l2_error(result, reference: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> float:
    """Relative L2 error of the marched field against the Cole--Hopf reference."""
    x, values, h, weights = _quadrature_field(result)
    reference = reference or (lambda z: exact_solution(z, result["final_time"], result["nu"]))
    exact = np.asarray(reference(x), dtype=float)
    mask = np.tile(weights, result["ncells"])
    squared = float(np.sum(mask * (values - exact) ** 2))
    norm = float(np.sum(mask * exact ** 2))
    if norm == 0.0:
        raise ValueError("reference has zero L2 norm; a relative error is undefined")
    return math.sqrt(squared / norm)


def total_mass(result) -> float:
    """``int_0^1 u dx`` (mode-0 coefficient is the cell mean)."""
    return float(result["h"] * result["coefficients"][:, 0].sum())


def total_energy(result) -> float:
    """``int_0^1 u^2/2 dx`` from the Gauss quadrature of the discrete field."""
    _, values, h, weights = _quadrature_field(result)
    mask = np.tile(weights, result["ncells"])
    # int_0^1 u^2/2 dx = sum_cells (h/2) sum_q w_q u^2 / 2
    return float(0.25 * h * np.sum(mask * values ** 2))