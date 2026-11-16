"""Stability polynomials of the explicit Runge-Kutta tableaus.

Verification debt this module closes
------------------------------------
:mod:`tensordg.runge_kutta` supplies the tableaus that advance the DG
semi-discrete ODE, and ``CASE-DG-ADVECT-ORDER`` (round 2) selects one per
polynomial degree so that the ``p + 1`` stage count of the RKDG accuracy tables
is matched.  What was *asserted* rather than measured is that the ``cfl`` those
studies run at is inside the linear stability region of each of those
integrators: the round-2 benchmark carried the comment "``0.4`` stays inside the
linear DG stability limit for every degree in the study", while
``validation/cfl_stability`` computed limits only for the two legacy
integrators ``rk4`` / ``ssp_rk3``.

This module supplies the missing piece on the *analytic* route.  For an explicit
tableau in Butcher form the amplification of the scalar model problem
``u' = z u`` (``z = dt * lambda``) is the polynomial

    R(z) = 1 + z b^T (I - z A)^{-1} 1 = 1 + sum_{k>=1} (b^T A^{k-1} 1) z^k,   (1)

so the stability region is ``{z : |R(z)| <= 1}``.  Nothing is transcribed: the
coefficients come from the same :class:`~tensordg.runge_kutta.RKTableau`
objects the solvers integrate with, so a tableau edit can never leave the
stability analysis behind.  Eq. (1) holds for any explicit tableau, including
the five-stage fourth-order SSP scheme, whose stage recurrence is *not* the
classical ``k_i = f(u + dt sum_j a_ij k_j)`` composition written out by hand --
which is exactly why an independent route is worth having.

The two algebraically identical forms are both provided: :func:`stability_polynomial`
(the coefficient form used for the scans, vectorised) and
:func:`stability_function_dense` (the ``(I - z A)`` solve, used to cross-check
the coefficients).  Only the *linear* statement is made: a frozen spectrum sees
a linear, constant-coefficient operator.  Nonlinear or limited schemes are not
covered there -- the relevant object is then the norm of the stage map, not
``|R|``.
"""
from __future__ import annotations

import math

import numpy as np

from .runge_kutta import RKTableau

#: ``|R| <= 1 + TOL`` counts as stable.  Not cosmetic: the assembled DG
#: operator carries a neutral eigenvalue at round-off distance from the origin
#: where ``|R|`` evaluates to ``1 + O(1e-16)``; a strict ``> 1`` test then
#: reports a spurious, mesh-dependent limit (the same trap documented in
#: ``benchmarks/cfl_stability_limit.py``).
STABILITY_TOL = 1e-10


def butcher_arrays(tableau: RKTableau) -> tuple[np.ndarray, np.ndarray]:
    """``(A, b)`` as dense arrays, with the strictly lower-triangular rows padded."""
    s = tableau.stages
    a = np.zeros((s, s))
    for i, row in enumerate(tableau.a):
        a[i, :len(row)] = row
    return a, np.asarray(tableau.b, dtype=float)


def stability_polynomial(tableau: RKTableau) -> np.ndarray:
    """Coefficients ``[c_0, ..., c_s]`` of ``R`` from eq. (1); ``c_0 = 1``."""
    a, b = butcher_arrays(tableau)
    coeffs = [1.0]
    vec = np.ones(tableau.stages)
    for _ in range(tableau.stages):
        coeffs.append(float(b @ vec))
        vec = a @ vec
    return np.asarray(coeffs)


def stability_function(tableau: RKTableau, z) -> np.ndarray:
    """``R(z)`` of eq. (1), vectorised over an array of complex ``z``."""
    coeffs = stability_polynomial(tableau)
    zs = np.atleast_1d(np.asarray(z, dtype=complex))
    return np.polynomial.polynomial.polyval(zs, coeffs)


def stability_function_dense(tableau: RKTableau, z) -> np.ndarray:
    """``R(z)`` from the ``(I - z A)`` solve -- the other form of eq. (1)."""
    a, b = butcher_arrays(tableau)
    eye = np.eye(tableau.stages)
    ones = np.ones(tableau.stages)
    zs = np.atleast_1d(np.asarray(z, dtype=complex))
    out = np.empty(zs.shape, dtype=complex)
    for i, zi in enumerate(zs):
        out[i] = 1.0 + zi * (b @ np.linalg.solve(eye - zi * a, ones))
    return out


def amplification_order(tableau: RKTableau, z: float) -> tuple[float, float]:
    """``(|R - T_p|, |R - T_{p+1}|)`` at ``z``; ``T_p`` = Taylor series of ``exp``.

    An explicit tableau of order ``p`` reproduces ``exp(z)`` through ``z^p``, so
    the first entry vanishes as ``z^(p+1)`` while the second is the *free* term a
    method of order ``p`` may get wrong.  The pair pins the order claim of a
    tableau without a convergence study, and it is what keeps the ``p = 4``
    five-stage rung honest: order five would need ``|R - T_5|`` to vanish for
    every tableau of that size, which Butcher's barrier forbids.
    """
    order = tableau.order
    r = stability_function(tableau, [z])[0]
    t_p = sum(z ** k / math.factorial(k) for k in range(order + 1))
    t_p1 = t_p + z ** (order + 1) / math.factorial(order + 1)
    return abs(r - t_p), abs(r - t_p1)


def growth_envelope(tableau: RKTableau, eigenvalues, dts) -> np.ndarray:
    """``G(dt) = max_lambda |R(dt lambda)|`` for every ``dt`` in ``dts``."""
    lam = np.asarray(eigenvalues, dtype=complex)
    dts = np.atleast_1d(np.asarray(dts, dtype=float))
    coeffs = stability_polynomial(tableau)
    out = np.empty(dts.shape, dtype=float)
    for i, dt in enumerate(dts):
        out[i] = float(np.abs(
            np.polynomial.polynomial.polyval(dt * lam, coeffs)).max())
    return out


def is_stable(tableau: RKTableau, eigenvalues, dt: float,
              tol: float = STABILITY_TOL) -> bool:
    """True when ``dt`` leaves every scaled eigenvalue inside the region."""
    return bool(growth_envelope(tableau, eigenvalues, [dt])[0] <= 1.0 + tol)


def axis_interval(tableau: RKTableau, direction: complex,
                  hi: float = 12.0, iters: int = 200) -> float:
    """Largest ``x >= 0`` with ``|R(x * direction)| <= 1`` (bisection).

    ``direction = -1`` is the classical negative-real-axis interval and
    ``direction = 1j`` the imaginary-axis one.  Both have textbook values for the
    classical schemes (RK4: ``2.7852935634`` and ``2 sqrt(2)``; Kutta-3:
    ``2.5127453267`` and ``sqrt(3)``), so they anchor eq. (1) independently of
    this module.
    """
    def ok(x: float) -> bool:
        return abs(stability_function(tableau, [direction * x])[0]) <= 1.0 + STABILITY_TOL

    lo, up = 0.0, hi
    if ok(up):  # pragma: no cover - no registered tableau has a 12-wide interval
        return up
    for _ in range(iters):
        mid = 0.5 * (lo + up)
        if ok(mid):
            lo = mid
        else:
            up = mid
    return 0.5 * (lo + up)


def real_axis_interval(tableau: RKTableau, hi: float = 12.0) -> float:
    """Largest ``x`` with ``|R(-x)| <= 1``."""
    return axis_interval(tableau, -1.0, hi=hi)


def imaginary_axis_interval(tableau: RKTableau, hi: float = 12.0) -> float:
    """Largest ``y`` with ``|R(i y)| <= 1``."""
    return axis_interval(tableau, 1j, hi=hi)


def max_stable_dt(tableau: RKTableau, eigenvalues, hi: float = 4.0,
                  iters: int = 80, passes: int = 3) -> tuple[float, bool]:
    """Largest ``dt`` with ``|R(dt lambda)| <= 1 + TOL`` for *every* eigenvalue.

    Bracketed bisection on the envelope ``G(dt)``, then an audit of ``(0, dt_max]``
    because ``G`` need not be monotone: one mode can fall back below the
    threshold while another is still growing, and a plain bisection would then
    return an optimistic later crossing.  The audit moves the boundary to the
    *earliest* crossing it finds, so the returned step is never later than the
    first failing one.  Returns ``(dt_max, audit_corrected)``.
    """
    lam = np.asarray(eigenvalues, dtype=complex)
    if growth_envelope(tableau, lam, [1e-8])[0] > 1.0 + STABILITY_TOL:
        raise RuntimeError("operator unstable at dt -> 0: check the spectrum")
    if growth_envelope(tableau, lam, [hi])[0] <= 1.0 + STABILITY_TOL:
        return hi, False
    lo, up = 0.0, hi
    for _ in range(iters):
        mid = 0.5 * (lo + up)
        if growth_envelope(tableau, lam, [mid])[0] <= 1.0 + STABILITY_TOL:
            lo = mid
        else:
            up = mid
    dt_max, corrected = 0.5 * (lo + up), False
    for _ in range(passes):
        audit = np.linspace(dt_max / 400.0, dt_max, 400)
        bad = growth_envelope(tableau, lam, audit) > 1.0 + STABILITY_TOL
        if not bad.any():
            break
        idx = int(np.argmax(bad))
        lo2 = float(audit[idx - 1]) if idx else 0.0
        up2 = float(audit[idx])
        for _ in range(60):
            mid = 0.5 * (lo2 + up2)
            if growth_envelope(tableau, lam, [mid])[0] <= 1.0 + STABILITY_TOL:
                lo2 = mid
            else:
                up2 = mid
        dt_max, corrected = 0.5 * (lo2 + up2), True
    return dt_max, corrected


def binding_index(tableau: RKTableau, eigenvalues, dt_max: float) -> int:
    """Index of the eigenvalue that sets the limit (probed just inside it)."""
    lam = np.asarray(eigenvalues, dtype=complex)
    amp = np.abs(stability_function(tableau, (1.0 + 1e-3) * dt_max * lam))
    return int(np.argmax(amp))


def cfl_limit(tableau: RKTableau, eigenvalues, degree: int,
              hi: float = 4.0) -> dict:
    """Stable step converted into the step rule the solvers actually use.

    Every TensorDG solver sets ``dt = cfl * h / ((2p + 1) |a|)``, so the ``cfl``
    a user passes maps to ``dt = cfl / (2p + 1)`` in units of ``h / |a|``.  The
    returned ``cfl_max`` is the inverse image of ``dt_max`` under that rule,
    i.e. *the largest ``cfl`` the solver can be handed* -- the only form in which
    the number can be compared with a case file's ``cfl``.
    """
    lam = np.asarray(eigenvalues, dtype=complex)
    dt_max, corrected = max_stable_dt(tableau, lam, hi=hi)
    j = binding_index(tableau, lam, dt_max)
    return {
        "tableau": tableau.name,
        "order": tableau.order,
        "stages": tableau.stages,
        "degree": int(degree),
        "dt_max_h_over_a": float(dt_max),
        "cfl_max": float(dt_max) * (2 * degree + 1),
        "binding_lambda": [float(lam[j].real), float(lam[j].imag)],
        "max_abs_lambda": float(np.abs(lam).max()),
        "n_modes": int(lam.size),
        "audit_corrected": bool(corrected),
    }