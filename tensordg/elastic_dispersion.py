"""Eigenvalue (dispersion / dissipation) spectrum of the 1-D DG elastodynamics operator.

Verification debt this module closes
------------------------------------
Anchor ``CASE-DG-DALEMBERT-ELASTIC-1D`` grades the 1-D velocity--stress solver
with **one** number: the relative L2 error of the stress profile against the
d'Alembert closed form (<2% at P3, N=200).  Its acceptance text also claims
"no physical dispersion" -- true for the *continuum* bar model, but it says
nothing about how much dispersion the *scheme* adds, at which wavenumber the 2%
gate is actually met, or how the resolution budget behaves with polynomial
degree.  Every existing elastic-module gate is an error-norm study; none of them
looks at the semi-discrete operator's spectrum.

This module computes the exact spectrum of the semi-discrete operator that
:mod:`tensordg.elastic_1d` really assembles and derives from it

* the **phase-speed error** and the **dissipation rate** of every degree
  ``p = 0..4`` as a function of the non-dimensional wavenumber
  ``theta = k h`` (``h`` = element size),
* the **super-convergence exponents**, measured from the computed spectrum
  (not quoted from the literature),
* the **points-per-wavelength (PPW) budget** needed for a 1% / 2% phase error,
  i.e. the number the d'Alembert case is implicitly spending,
* a structural check that the "symbol matrix" language is exact here
  (block-circulant assembly), so the spectrum below is a property of the
  shipped code rather than of a re-derived idealization.

Method (no hand copies, no analytic black boxes)
-----------------------------------------------
1. **Assembly from the production kernel.**  Unit modal vectors are pushed
   through :func:`tensordg.elastic_1d._rhs` -- the very routine the time
   integrator calls -- on a periodic mesh of ``N`` elements with ``h = 1``.  The
   result is the ``(N (p+1) 2) x (N (p+1) 2)`` matrix ``S`` of the code under
   test.  State layout is ``coeff[cell, mode, var]`` with ``var = (v, sigma)``,
   identical to :func:`tensordg.elastic_1d.solve_elastic_1d`.
2. **Block symbol.**  ``S`` is block-circulant (translation invariance on a
   uniform periodic mesh), so a single ``2(p+1) x 2(p+1)`` *symbol matrix*
   ``T(theta)`` carries the whole spectrum.  The identity

       spec(S) = union over m of spec(T(2 pi m / N))

   is asserted numerically (:func:`symbol_union_error`) -- that is the
   structural check that "symbol matrix" is the right language and that two
   independent eigenvalue computations agree.
3. **Closed-form anchor at p = 0.**  For ``p = 0`` the scheme is the
   characteristic-upwind finite-volume operator, whose symbol follows in closed
   form from the flux split ``F* = A^+ U_L + A^- U_R`` (with
   ``A^+ = (A + c I)/2``, ``A^- = (A - c I)/2``).  On cell ``i``

       dU_i/dt = (1/h) [ A^+(U_{i-1} - U_i) + A^-(U_i - U_{i+1}) ],

   so for ``U_i ~ exp(i theta i)``

       T(theta) = (1/h) [ A^+(e^{-i theta} - 1) - A^-(e^{i theta} - 1) ]
                = -(i sin(theta)/h) A - (2 c sin^2(theta/2)/h) I,

   and, since ``A`` has eigenvalues ``+/- c``,

       lambda_{+-}(theta) = (c/h) [ -2 sin^2(theta/2) -/+ i sin(theta) ].

   Both parts are compared against the assembled p=0 spectrum to roundoff
   (:func:`p0_symbol_error`) -- an implementation-independent anchor for the
   whole pipeline.  Note the *dissipation* ``2 c sin^2(theta/2)/h`` is
   first-order in ``h`` (classic upwind smearing) while the phase speed
   ``c sin(theta)/theta`` is second-order accurate: the exponents below must
   reproduce that asymmetry at ``p = 0``.
4. **Branch tracking.**  The propagating ("physical") mode is tracked as the
   least-damped eigenvalue with non-zero imaginary part, continued in
   ``theta`` station by station (:func:`physical_branch`); the spurious
   branches of the DG spectrum are strongly damped and are never selected at
   small ``theta``.

Sign convention
---------------
The flux split above is the one the shipped ``_rhs`` implements (it advects
``w+ = sigma - rho c v`` with ``+c``); the assembled spectrum consequently
appears either as ``{A + iB, A - iB}`` or as its conjugate, depending on where
``exp(+i theta i)`` or ``exp(-i theta i)`` is inserted.  All comparisons here
are made on the **conjugation-invariant set** ``{Re, |Im|}``, so a global sign
flip cannot turn a gate green.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch

from .advection_1d import _operators
from .elastic_1d import _rhs, elastic_wave_speed

__all__ = [
    "ASSEMBLY_CELLS",
    "assemble_operator",
    "system_symbol",
    "symbol_union_error",
    "p0_closed_form",
    "p0_symbol_error",
    "physical_branch",
    "phase_and_dissipation",
    "exponent_details",
    "fit_exponent",
    "dominant_theta",
    "resolution_table",
    "analyse",
]

#: Element count of the assembly mesh.  ``N`` must be > 3 so that the widest
#: stencil of the block-circulant operator (one cell each side) does not wrap
#: onto itself; 8 keeps the assembly exact and cheap.
ASSEMBLY_CELLS = 8

#: Degrees covered by the published tables.
DEGREES = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def _legendre_derivatives(degree: int, xi: np.ndarray,
                          dtype: torch.dtype) -> torch.Tensor:
    """Derivative matrix ``[quadrature, modes]`` at the reference nodes."""
    derivs = []
    for k in range(degree + 1):
        coeff = np.zeros(k + 1)
        coeff[-1] = 1.0
        dcoeff = np.polynomial.legendre.legder(coeff)
        vals = (np.polynomial.legendre.legval(xi, dcoeff)
                if len(dcoeff) else np.zeros_like(xi))
        derivs.append(torch.as_tensor(vals, dtype=dtype))
    return torch.stack(derivs, dim=1)


def assemble_operator(degree: int, ncells: int = ASSEMBLY_CELLS,
                      rho: float = 1.0, young: float = 1.0, h: float = 1.0,
                      dtype: torch.dtype = torch.float64) -> np.ndarray:
    """Semi-discrete matrix of the shipped DG operator on a periodic mesh.

    ``S[a, b] = d(coeff_flattened[a]) / dt`` for the unit modal state ``b``,
    where the flattening is ``cell -> mode -> var`` (``var = (v, sigma)``),
    exactly the layout of :func:`tensordg.elastic_1d.solve_elastic_1d`.
    """
    if ncells < 4:
        raise ValueError("ncells must be >= 4 to avoid periodic stencil wrap")
    if not 0 <= degree <= 8:
        raise ValueError("degree must be in [0, 8]")
    if rho <= 0.0 or young <= 0.0 or h <= 0.0:
        raise ValueError("rho, young and h must be positive")

    dev = torch.device("cpu")
    xq, wq, phi, _volume, phi_l, phi_r = _operators(degree, dev, dtype)
    dphi = _legendre_derivatives(degree, xq.detach().cpu().numpy(), dtype)
    c = elastic_wave_speed(rho, young)

    nmodes = degree + 1
    nvar = 2
    n = nmodes * nvar
    size = ncells * n

    S = np.zeros((ncells, n, ncells, n), dtype=float)
    for j in range(size):
        probe = torch.zeros(ncells, nmodes, nvar, dtype=dtype)
        probe.view(-1)[j] = 1.0
        column = _rhs(probe, h, rho, young, c, phi, dphi, phi_l, phi_r,
                      "periodic", wq)
        S[:, :, j // n, j % n] = column.reshape(ncells, n).numpy()
    return S.reshape(size, size)


def _blocks(matrix: np.ndarray, ncells: int, n: int) -> np.ndarray:
    """``B[d]`` = operator block coupling cell ``i`` to cell ``i + d``."""
    four = matrix.reshape(ncells, n, ncells, n)
    row0 = four[0].transpose(1, 0, 2)          # [d, n, n]
    return row0



def _symbol(blocks: np.ndarray, theta: float) -> np.ndarray:
    """Block symbol ``T(theta) = sum_d B[d] exp(i theta d')`` (signed offset).

    ``d' = d`` for ``d <= N/2`` and ``d - N`` beyond, i.e. the block coupling
    cell ``i`` to cell ``i + d (mod N)`` enters with the *signed* shift.  Using
    the raw index ``d`` instead of ``d'`` silently breaks the last block row of
    the circulant (``exp(7 i theta) != exp(-i theta)``), which is visible as a
    spurious eigenvalue in the p=0 symbol.
    """
    ncells = blocks.shape[0]
    n = blocks.shape[1]
    T = np.zeros((n, n), dtype=complex)
    for d in range(ncells):
        signed = d if d <= ncells // 2 else d - ncells
        T += blocks[d] * np.exp(1j * theta * signed)
    return T

def _is_block_circulant(matrix: np.ndarray, ncells: int, n: int,
                        atol: float = 1e-12) -> bool:
    four = matrix.reshape(ncells, n, ncells, n)
    ref = four[0]
    for i in range(1, ncells):
        for j in range(ncells):
            if not np.allclose(four[i, :, j, :], ref[:, (j - i) % ncells, :],
                               atol=atol):
                return False
    return True


def system_symbol(degree: int, theta: float, matrix: np.ndarray | None = None,
                  ncells: int = ASSEMBLY_CELLS, **kwargs) -> np.ndarray:
    """Symbol matrix ``T(theta)`` (``2(p+1) x 2(p+1)``) of the elastic operator.

    ``T(theta)`` acts on the cell-local modal vector ``(v_0..v_p, s_0..s_p)``
    and reproduces the whole operator spectrum through
    ``spec(S) = union_m spec(T(2 pi m / ncells))``.
    """
    n = 2 * (degree + 1)
    if matrix is None:
        matrix = assemble_operator(degree, ncells=ncells, **kwargs)
    blocks = _blocks(matrix, ncells, n)
    return _symbol(blocks, theta)


def symbol_union_error(degree: int, ncells: int = ASSEMBLY_CELLS,
                       matrix: np.ndarray | None = None, **kwargs) -> float:
    """Max distance between ``spec(S)`` and ``union_m spec(T(2 pi m / N))``.

    A mismatch means ``S`` is not block-circulant, i.e. the symbol matrix is
    the wrong language for this operator and every spectrum number below would
    be meaningless.
    """
    n = 2 * (degree + 1)
    if matrix is None:
        matrix = assemble_operator(degree, ncells=ncells, **kwargs)
    blocks = _blocks(matrix, ncells, n)
    if not _is_block_circulant(matrix, ncells, n):
        return float("inf")
    direct = np.linalg.eigvals(matrix)
    union = []
    for m in range(ncells):
        union.append(np.linalg.eigvals(_symbol(blocks, 2.0 * math.pi * m / ncells)))
    union = np.concatenate(union)
    scale = max(1.0, float(np.max(np.abs(direct))))
    worst = 0.0
    remaining = list(union)
    for lam in sorted(direct, key=lambda z: -abs(z)):
        dist, idx = min(((abs(lam - u), k) for k, u in enumerate(remaining)),
                        key=lambda t: t[0])
        worst = max(worst, dist)
        remaining.pop(idx)
    return worst / scale


# ---------------------------------------------------------------------------
# closed-form anchor (p = 0)
# ---------------------------------------------------------------------------
def p0_closed_form(theta: float, rho: float = 1.0, young: float = 1.0,
                   h: float = 1.0) -> tuple[complex, complex]:
    """Characteristic-upwind FV symbol pair, derived in the module docstring.

    ``lambda_{+-} = (c/h) [-2 sin^2(theta/2) -/+ i sin(theta)]``.
    """
    c = elastic_wave_speed(rho, young)
    real = -2.0 * math.sin(0.5 * theta) ** 2 * c / h
    imag = math.sin(theta) * c / h
    return complex(real, imag), complex(real, -imag)


def p0_symbol_error(theta: float = 0.7, rho: float = 1.0, young: float = 1.0,
                    h: float = 1.0, ncells: int = ASSEMBLY_CELLS) -> float:
    """Distance between the assembled p=0 spectrum and :func:`p0_closed_form`."""
    matrix = assemble_operator(0, ncells=ncells, rho=rho, young=young, h=h)
    got = np.linalg.eigvals(_symbol(_blocks(matrix, ncells, 2), theta))
    exact = p0_closed_form(theta, rho=rho, young=young, h=h)
    scale = max(abs(exact[0]), abs(exact[1]), 1e-30)
    return max(min(abs(g - exact[0]), abs(g - exact[1])) for g in got) / scale


# ---------------------------------------------------------------------------
# branches
# ---------------------------------------------------------------------------
def _least_damped_propagating(values: np.ndarray) -> complex:
    """Physical branch at one ``theta``: least damped eigenvalue with Im != 0.

    The physical (propagating) mode is the least-damped one -- the spurious DG
    modes are O(c/h) damped.  Among the two propagating eigenvalues (a
    conjugate pair) the one with positive imaginary part is kept.
    """
    propagating = [v for v in values if abs(v.imag) > 1e-12]
    if not propagating:
        return min(values, key=lambda v: -v.real)   # theta = 0: all zero
    best = min(propagating, key=lambda v: -v.real)
    return best if best.imag > 0 else best.conjugate()


def physical_branch(degree: int, thetas: Sequence[float],
                    ncells: int = ASSEMBLY_CELLS, rho: float = 1.0,
                    young: float = 1.0, h: float = 1.0,
                    matrix: np.ndarray | None = None) -> np.ndarray:
    """Least-damped propagating eigenvalue at each ``theta`` (positive Im).

    Branch tracking is by **continuation**: the physical mode is the eigenvalue
    that stays closest to the previous station, seeded at the smallest ``theta``
    by the least-damped eigenvalue with non-zero imaginary part.  No eigenvalue
    is picked by "largest |Im|" -- near ``theta = pi`` that rule jumps onto a
    spurious DG mode (which is exactly the artefact this module has to be able
    to see in the *spurious* branches rather than in the physical one).
    """
    n = 2 * (degree + 1)
    if matrix is None:
        matrix = assemble_operator(degree, ncells=ncells, rho=rho,
                                   young=young, h=h)
    blocks = _blocks(matrix, ncells, n)
    theta_arr = np.asarray(thetas, dtype=float)
    order = np.argsort(theta_arr)
    out = np.zeros(theta_arr.size, dtype=complex)
    previous = None
    for pos, idx in enumerate(order):
        values = np.linalg.eigvals(_symbol(blocks, float(theta_arr[idx])))
        if previous is None:
            lam = _least_damped_propagating(values)
        else:
            lam = values[int(np.argmin(np.abs(values - previous)))]
        if lam.imag < 0.0:
            lam = np.conjugate(lam)
        out[idx] = lam
        previous = lam
    return out


def phase_and_dissipation(degree: int, thetas: Sequence[float],
                          rho: float = 1.0, young: float = 1.0,
                          h: float = 1.0, ncells: int = ASSEMBLY_CELLS,
                          matrix: np.ndarray | None = None) -> dict:
    """Relative phase-speed error and dimensionless dissipation per ``theta``.

    ``omega_num = |Im lambda|``, ``omega_exact = c k = c theta / h``,
    ``phase_error = omega_num / omega_exact - 1`` (signed: positive means the
    discrete wave runs fast), ``dissipation = -Re(lambda) h / c`` (>= 0 for a
    stable upwind discretisation, ``= 0`` in the continuum).
    """
    c = elastic_wave_speed(rho, young)
    lam = physical_branch(degree, thetas, ncells=ncells, rho=rho, young=young,
                          h=h, matrix=matrix)
    theta = np.asarray(thetas, dtype=float)
    k = theta / h
    exact = c * k
    with np.errstate(invalid="ignore", divide="ignore"):
        phase_error = np.where(exact > 0, np.abs(lam.imag) / exact - 1.0, 0.0)
    dissipation = -lam.real * h / c
    return {
        "theta": theta,
        "phase_speed_ratio": np.where(exact > 0, np.abs(lam.imag) / np.where(
            exact > 0, exact, 1.0), 1.0),
        "phase_error": phase_error,
        "dissipation": dissipation,
        "eigenvalue": lam,
    }


def _fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = (x > 0) & (y > 0) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def exponent_details(degree: int, metric: str = "phase_error",
                     theta_lo: float = 0.15, theta_hi: float = 1.0,
                     samples: int = 40, noise_floor: float = 1e-10,
                     **kwargs) -> dict:
    """Measured log-log exponent of ``metric`` together with its fit window.

    Two guards make the number honest:

    * the **window** is ``[theta_lo, theta_hi]`` -- moderate wavenumbers, where
      the asymptotic law is established but the wave is still resolved.  At
      small ``theta`` a super-convergent scheme hits the double-precision floor
      of the eigenvalue solve (``~1e-13`` here, see the tables in
      ``docs/elastic-dispersion-analysis.md``), which would flatten the fit and
      fake a *lower* exponent;
    * points below ``noise_floor`` are dropped, and the count of retained
      points is reported so a degenerate fit cannot hide.
    """
    thetas = np.geomspace(theta_lo, theta_hi, samples)
    data = phase_and_dissipation(degree, thetas, **kwargs)
    values = np.abs(np.asarray(data[metric], dtype=float))
    mask = np.isfinite(values) & (values > noise_floor)
    exponent = float("nan")
    if np.count_nonzero(mask) >= 3:
        exponent = float(np.polyfit(np.log(thetas[mask]),
                                    np.log(values[mask]), 1)[0])
    return {
        "metric": metric,
        "exponent": exponent,
        "window": [theta_lo, theta_hi],
        "points_used": int(np.count_nonzero(mask)),
        "error_at_theta_lo": float(values[0]),
        "error_at_theta_hi": float(values[-1]),
    }


def fit_exponent(degree: int, metric: str = "phase_error",
                 theta_lo: float = 0.15, theta_hi: float = 1.0,
                 samples: int = 40, **kwargs) -> float:
    """Measured log-log decay exponent of ``metric`` (see :func:`exponent_details`)."""
    return exponent_details(degree, metric, theta_lo=theta_lo,
                            theta_hi=theta_hi, samples=samples,
                            **kwargs)["exponent"]


def dominant_theta(width: float, h: float) -> float:
    """Non-dimensional wavenumber of a Gaussian of physical width ``width``.

    ``g(x) = exp(-(x/width)^2)`` has transform ``~ exp(-(k width / 2)^2)``, i.e.
    its spectrum falls to ``1/e`` at ``k = 2/width``.  With ``theta = k h`` this
    is the wavenumber the scheme must actually resolve for a pulse of that
    width on elements of size ``h`` -- the number that decides whether a
    profile-comparison gate is dispersion-limited or not.
    """
    if width <= 0.0 or h <= 0.0:
        raise ValueError("width and h must be positive")
    return 2.0 * h / width


def resolution_table(degree: int, tolerances: Iterable[float] = (0.01, 0.02),
                     theta_max: float = math.pi, samples: int = 2000,
                     **kwargs) -> dict:
    """Points-per-wavelength needed to keep the phase error below ``tolerance``.

    ``PPW(theta) = 2 pi / theta``; the returned value is the *worst* (smallest)
    PPW at which the phase error is still within tolerance, i.e. the resolution
    budget for a wave that is resolved no better than the tolerance allows.
    """
    thetas = np.linspace(1e-4, theta_max, samples)
    data = phase_and_dissipation(degree, thetas, **kwargs)
    err = np.abs(np.asarray(data["phase_error"], dtype=float))
    out = {}
    for tol in tolerances:
        ok = np.nonzero(err <= tol)[0]
        out[f"ppw_{tol:g}"] = (2.0 * math.pi / thetas[ok.max()]
                              if ok.size else float("nan"))
        out[f"diss_{tol:g}"] = (float(np.asarray(data["dissipation"])[ok.max()])
                               if ok.size else float("nan"))
    out["max_phase_error"] = float(err.max())
    return {k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in out.items()}


def analyse(degrees: Sequence[int] = DEGREES, ncells: int = ASSEMBLY_CELLS,
            **kwargs) -> dict:
    """Full spectrum report: structural checks, exponents, resolution budgets."""
    thetas = np.geomspace(1e-3, math.pi, 60)
    rows = []
    for p in degrees:
        data = phase_and_dissipation(p, thetas, ncells=ncells, **kwargs)
        rows.append({
            "degree": p,
            "symbol_union_error": symbol_union_error(p, ncells=ncells,
                                                     **kwargs),
            "p0_anchor_error": (p0_symbol_error(ncells=ncells, **kwargs)
                                if p == 0 else None),
            "phase": exponent_details(p, "phase_error", ncells=ncells,
                                      **kwargs),
            "dissipation": exponent_details(p, "dissipation", ncells=ncells,
                                            **kwargs),
            "resolution": resolution_table(p, ncells=ncells, **kwargs),
            "max_dissipation": float(np.asarray(data["dissipation"]).max()),
        })
    return {
        "schema": "tensordg-elastic-dispersion/1",
        "anchor": "CASE-DG-DALEMBERT-ELASTIC-1D",
        "assembly_cells": ncells,
        "fit_window": [1e-2, 1e-1],
        "rows": rows,
    }