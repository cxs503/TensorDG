"""Exact (closed-form) solution of the DG-0 upwind discretisation of 1-D linear advection.

Motivation
----------
For ``degree = 0`` the DG method with an upwind flux reduces to the classical
first-order upwind finite-volume scheme on a uniform periodic mesh.  Its
semidiscrete operator is **circulant**, so it is diagonalised in closed form by
the discrete Fourier transform.  That makes the semidiscrete solution -- and
hence the *purely spatial* discretisation error -- available at machine
precision for any mesh size ``N``, without any time-stepping, reference grid or
hand-copied number.

With this reference the observed ``DG0`` convergence rate can be audited
honestly: the rate measured on a coarse ladder (``N = 20 ... 320``) is
``~0.85``, not the asymptotic ``1``.  :func:`l2_error_semidiscrete` shows that
this is a property of the *exact* semidiscrete error curve (multi-term
asymptotics at ``N = 20`` already has ``exp(-2*pi^2/N) = 0.37`` amplitude
decay), not a solver defect, and :func:`asymptotic_order` recovers
``1.00`` once the ladder reaches the asymptotic regime.

Analysis
--------
For ``u_t + c u_x = 0`` on ``[0, 1)`` periodic with cell averages ``u_j``::

    du_j/dt = -(c/h) * (u_j - u_{j-1}),      h = 1/N

is circulant with symbol ``lambda_k = -(c/h) * (1 - exp(-i*theta_k))``,
``theta_k = 2*pi*k/N`` (the sign convention used by
:func:`tensordg.advection_1d.solve_advection`; checked in
``tests/test_p0_semidiscrete_exact.py``).  Hence ``uhat_k(t) = exp(lambda_k t)
uhat_k(0)``.

For the benchmark initial condition ``u(x, 0) = sin(2*pi*x)`` the initial cell
average is analytic, and so is the ``L2`` error of the piecewise-constant
reconstruction against ``sin(2*pi*(x - c t))``.  Per cell the error splits
*orthogonally*::

    (1/h) * int_cell (u_j - u_exact)^2 = (u_j - m1_j)^2 + (m2_j - m1_j^2)

with ``m1``/``m2`` the analytic cell mean and mean square of the exact sine.
The first term is the *scheme* error (amplitude damping ``2*pi^2/N`` plus a
phase error of ``4*pi^3/(3 N^2)`` over one period), the second is the
``O(h^2)`` **projection floor** of piecewise-constant data
(``pi^2 h^2 / 6``).  Hence::

    E(N) -> sqrt(2*pi^4 + pi^2/6) / N = pi*sqrt(2*pi^2 + 1/6)/N ~ 14.0165/N

i.e. the closed-form leading constant is **not** ``pi^2*sqrt(2) = 13.958``
(the scheme term alone) but ``14.0165``; the 0.4% gap is the projection floor
and is a common source of spurious "constant off by 0.4%" claims in
verification scripts.  :func:`asymptotic_constant` measures ``N*E(N)`` and the
tests assert both the two-term closed form and the limiting value.

Everything here is analytic and dimensionless: ``c = 1``, ``L = 1``, ``t`` in
units of ``L/c``.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "P0_ASYMPTOTIC_CONSTANT",
    "P0_SCHEME_CONSTANT",
    "P0_PROJECTION_CONSTANT",
    "p0_eigenvalues",
    "p0_initial_cell_averages",
    "p0_exact_cell_averages",
    "l2_error_semidiscrete",
    "l2_error_fully_discrete",
    "observed_order",
    "asymptotic_order",
    "asymptotic_constant",
]

#: Closed-form leading constant of the DG0/upwind L2 error (scheme part only),
#: ``pi^2*sqrt(2)``: amplitude damping ``2*pi^2/N`` over one period divided by
#: the ``sqrt(2)`` RMS of a unit sine.
P0_SCHEME_CONSTANT = math.pi ** 2 * math.sqrt(2.0)

#: ``O(h^2)`` projection floor of piecewise-constant data, ``pi/sqrt(6)`` per
#: unit ``h`` (mean square ``pi^2 h^2/6`` for ``sin(2*pi*x)``).
P0_PROJECTION_CONSTANT = math.pi / math.sqrt(6.0)

#: Closed-form leading constant of the total L2 error,
#: ``sqrt(scheme^2 + projection^2) = pi*sqrt(2*pi^2 + 1/6) ~ 14.0165``.
P0_ASYMPTOTIC_CONSTANT = math.hypot(P0_SCHEME_CONSTANT, P0_PROJECTION_CONSTANT)


def p0_eigenvalues(ncells: int, speed: float = 1.0) -> np.ndarray:
    """Eigenvalues of the circulant DG0 upwind operator, length ``ncells``.

    ``lambda_k = -(speed/h) * (1 - exp(-i*theta_k))`` with
    ``theta_k = 2*pi*k/N`` for ``k = 0 ... N-1`` (FFT ordering).
    """
    if ncells < 2:
        raise ValueError("ncells must be >= 2")
    h = 1.0 / ncells
    theta = 2.0 * math.pi * np.arange(ncells) / ncells
    return -(speed / h) * (1.0 - np.exp(-1j * theta))


def p0_initial_cell_averages(ncells: int) -> np.ndarray:
    """Exact cell averages of ``sin(2*pi*x)`` on the uniform periodic mesh.

    ``mean_j = [cos(2*pi*a_j) - cos(2*pi*b_j)] / (2*pi*h)`` with ``b_j = a_j + h``.
    """
    if ncells < 2:
        raise ValueError("ncells must be >= 2")
    h = 1.0 / ncells
    a = np.arange(ncells) * h
    b = a + h
    return (np.cos(2.0 * math.pi * a) - np.cos(2.0 * math.pi * b)) / (2.0 * math.pi * h)


def p0_exact_cell_averages(ncells: int, final_time: float = 1.0,
                           speed: float = 1.0) -> np.ndarray:
    """Exact semidiscrete DG0 cell averages at ``final_time`` (float64 round-off)."""
    u0 = p0_initial_cell_averages(ncells)
    lam = p0_eigenvalues(ncells, speed)
    uhat = np.fft.fft(u0) * np.exp(lam * final_time)
    return np.real(np.fft.ifft(uhat))


def _cell_means_sine(ncells: int, final_time: float, speed: float):
    """Analytic per-cell mean and mean square of ``sin(2*pi*(x - speed*t))``."""
    h = 1.0 / ncells
    a = np.arange(ncells) * h
    b = a + h
    shift = speed * final_time
    two_pi = 2.0 * math.pi
    four_pi = 4.0 * math.pi
    m1 = (np.cos(two_pi * (a - shift)) - np.cos(two_pi * (b - shift))) / (two_pi * h)
    m2 = 0.5 - (np.sin(four_pi * (b - shift)) - np.sin(four_pi * (a - shift))) / (four_pi * h)
    return m1, m2


def l2_error_semidiscrete(ncells: int, final_time: float = 1.0,
                          speed: float = 1.0) -> float:
    """Exact L2 error of the *semidiscrete* DG0 solution at ``final_time``.

    No time integration and no quadrature: the per-cell constant solution is
    integrated against the analytic sine exactly, so the result is accurate to
    float64 round-off.  This is the spatial discretisation error of DG0 alone.
    """
    u = p0_exact_cell_averages(ncells, final_time, speed)
    m1, m2 = _cell_means_sine(ncells, final_time, speed)
    h = 1.0 / ncells
    cell_err2 = u * u - 2.0 * u * m1 + m2      # (1/h) * int_cell (u_j - exact)^2
    return float(math.sqrt(max(h * float(np.sum(cell_err2)), 0.0)))


def l2_error_fully_discrete(cell_averages, final_time: float = 1.0,
                            speed: float = 1.0) -> float:
    """Exact L2 error of a *given* DG0 cell-average vector (no quadrature).

    Use this to measure a solver's fully discrete solution against the same
    analytic sine, so the only difference from
    :func:`l2_error_semidiscrete` is the time-integration error.
    """
    u = np.asarray(cell_averages, dtype=float)
    ncells = u.size
    if ncells < 2:
        raise ValueError("need at least 2 cells")
    m1, m2 = _cell_means_sine(ncells, final_time, speed)
    h = 1.0 / ncells
    cell_err2 = u * u - 2.0 * u * m1 + m2
    return float(math.sqrt(max(h * float(np.sum(cell_err2)), 0.0)))


def observed_order(ncells, errors) -> float:
    """Least-squares slope of ``log(E)`` vs ``log(N)`` (same estimator as the case pool)."""
    n = np.asarray(ncells, dtype=float)
    e = np.asarray(errors, dtype=float)
    if n.size != e.size or n.size < 2:
        raise ValueError("need >= 2 matched samples")
    keep = e > 0.0
    return float(-np.polyfit(np.log(n[keep]), np.log(e[keep]), 1)[0])


def asymptotic_order(ncells: int, final_time: float = 1.0,
                     speed: float = 1.0) -> float:
    """Local order of the analytic error curve at ``N`` via ``N`` and ``2N``.

    ``log(E(N)/E(2N))/log(2)`` -- the local (two-point) order, which is the
    honest way to locate the asymptotic regime of a multi-term error curve.
    """
    e1 = l2_error_semidiscrete(ncells, final_time, speed)
    e2 = l2_error_semidiscrete(2 * ncells, final_time, speed)
    return float(math.log(e1 / e2) / math.log(2.0))


def asymptotic_constant(ncells: int, final_time: float = 1.0,
                        speed: float = 1.0) -> float:
    """Measure ``N * E(N)``; approaches ``pi*sqrt(2*pi^2 + 1/6)`` as ``N -> inf``."""
    return ncells * l2_error_semidiscrete(ncells, final_time, speed)