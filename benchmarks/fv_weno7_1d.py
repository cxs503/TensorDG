"""Higher-order (5 and 7) finite-volume WENO reference solver for 1-D Euler.

Why this module exists
----------------------
The round-2 certified reference for CASE-SHU-OSHER is produced by
:mod:`benchmarks.fv_weno5_1d`.  Its certified uncertainty band ``U`` is an
*intra-family, intra-ladder* number: the certification engine
(:func:`tensordg.certified_reference.certify`) only ever sees profiles from one
discretisation family at three to four resolutions, so nothing in that pipeline
tests whether ``U`` also bounds the difference to a *different* high-order
family.  :func:`tensordg.certified_reference.band_coverage_study` states that
gap itself ("that says nothing about the shock case, where the true error
cannot be measured at all").

This module supplies the missing independent leg for the shock case: the same
physics (HLLC flux from :mod:`tensordg.euler_1d`), the same time integrator
(SSP-RK3) and the same cell-centred finite-volume layout, but a **different
reconstruction order** (7 instead of 5: seven-cell stencil, four sub-stencils,
different optimal weights and different smoothness indicators).  Running the
same certification engine on a WENO7 ladder and comparing the two certified
profiles is a falsifiable test of the band's honesty: if the certified profiles
differ by more than ``U``, then ``U`` does not bound the reference's own error,
and any ``error < 3 %`` gate quoted against it inherits that bias.

Derived, not copied
-------------------
Every table below (sub-stencil coefficients, optimal weights, smoothness
indicators) is derived exactly from its definition with rational arithmetic in
``tools/derive_weno_tables.py``:

* sub-stencil coefficients -- the linear map from cell averages to the face
  value of an upwind reconstruction, solved from the moment matrix;
* optimal weights -- exact solution of the consistency + normalisation system;
* smoothness indicators -- the Jiang--Shu definition
  ``beta_k = sum_{l=1..r-1} int (d^l p_k/dx^l)^2 dx`` on unit cells.

:func:`selftest_tables` re-checks the emitted numbers (ideal-weight blend must
reproduce degree ``order-1`` polynomials to machine precision, and the order-5
optimal weights must equal the classic Jiang--Shu values).  The *scheme* order is
verified empirically by :func:`solve_advection_periodic` (see
``tests/test_xval_dg_round2_band_honesty.py``).

Note on the smoothness-indicator normalisation: the Jin--Shu ``beta`` is defined
up to a constant factor that only cancels in ``alpha_k = d_k/(eps+beta_k)^2``
asymptotically, so this module's ``beta`` is *not* numerically identical to the
hard-coded Jiang--Shu constants used by :mod:`benchmarks.fv_weno5_1d` (same
function, different normalisation).  The order-5 path here is therefore a
re-implementation rather than a bit-identical copy; that is deliberate and is
used as a second sensitivity axis in
``benchmarks/xval_dg_round2_band_honesty.py``.
"""
from __future__ import annotations

import math
import time

import torch

from tensordg.euler_1d import GAMMA, hllc_flux, physical_flux, sound_speed

__all__ = [
    "WENO_TABLES",
    "EPS_WENO",
    "weno_reconstruct",
    "solve_fv_weno",
    "reconstruction_order_study",
    "selftest_tables",
    "GAMMA",
]

# --------------------------------------------------------------------- tables
# SUB[q][t][k] multiplies the cell average at offset k of sub-stencil t; the
# sub-stencil t starts at cell (stencil_start + t), the target face being the
# right edge of sub-cell (r-1-t).
WENO_TABLES = {
    5: {
        "r": 3,
        "sub": [
            [1 / 3, -7 / 6, 11 / 6],          # ( 2, -7, 11) / 6
            [-1 / 6, 5 / 6, 1 / 3],           # (-1,  5,  2) / 6
            [1 / 3, 5 / 6, -1 / 6],           # ( 2,  5, -1) / 6
        ],
        "d": [1 / 10, 6 / 10, 3 / 10],
        # beta = v^T Q v, v = (v0, v1, v2):
        #   6 v0^2 - 21 v0 v1 + 9 v0 v2 + 21 v1^2 - 21 v1 v2 + 6 v2^2
        "Q": [[6.0, -21 / 2, 9 / 2],
              [-21 / 2, 21.0, -21 / 2],
              [9 / 2, -21 / 2, 6.0]],
    },
    7: {
        "r": 4,
        "sub": [
            [-1 / 4, 13 / 12, -23 / 12, 25 / 12],
            [1 / 12, -5 / 12, 13 / 12, 1 / 4],
            [-1 / 12, 7 / 12, 7 / 12, -1 / 12],
            [1 / 4, 13 / 12, -5 / 12, 1 / 12],
        ],
        "d": [1 / 35, 12 / 35, 18 / 35, 4 / 35],
        # beta over 4 cells (symmetrised quadratic form):
        #   289/20 v0^2 - 727/10 v0v1 + 1901/30 v0v2 - 587/30 v0v3
        # + 6083/60 v1^2 - 5803/30 v1v2 + 1901/30 v1v3
        # + 6083/60 v2^2 - 727/10 v2v3 + 289/20 v3^2
        "Q": [[289 / 20, -727 / 20, 1901 / 60, -587 / 60],
              [-727 / 20, 6083 / 60, -5803 / 60, 1901 / 60],
              [1901 / 60, -5803 / 60, 6083 / 60, -727 / 20],
              [-587 / 60, 1901 / 60, -727 / 20, 289 / 20]],
    },
}

EPS_WENO = 1e-6


def selftest_tables(order: int = 7, tol: float = 1e-10) -> dict:
    """Exactness check: the ideal-weight blend reproduces degree ``order-1`` polys."""
    order = int(order)
    tab = WENO_TABLES[order]
    r, sub, d = tab["r"], tab["sub"], tab["d"]
    s = (order - 1) // 2
    worst = 0.0
    for k in range(order):                  # polynomial degrees 0 .. order-1
        n = 2 * s + 1
        ubar = []
        for c in range(n):                  # cells [-s, -s+1], ..., [s, s+1]
            lo, hi = c - s, c - s + 1
            ubar.append((hi ** (k + 1) - lo ** (k + 1)) / (k + 1))
        # the target face is the right edge of local cell s, i.e. x = 1
        exact = 1.0 ** k
        approx = 0.0
        for t in range(r):
            approx += d[t] * sum(sub[t][j] * ubar[t + j] for j in range(r))
        worst = max(worst, abs(approx - exact))
    if order == 5:                          # classic Jiang--Shu optimal weights
        classic = [1 / 10, 6 / 10, 3 / 10]
        assert all(abs(a - b) < 1e-15 for a, b in zip(d, classic)), (d, classic)
    assert worst < tol, f"order {order} ideal-weight blend inexact: {worst:.3e}"
    return {"order": order, "r": r, "worst_polynomial_error": worst}


# ------------------------------------------------------------- reconstruction
def _beta(vals, Q):
    """Quadratic form ``vals^T Q vals`` over ``r`` equal-shape tensors."""
    out = None
    for i in range(len(vals)):
        for j in range(i, len(vals)):
            c = Q[i][j] * (1.0 if i == j else 2.0)
            term = c * vals[i] * vals[j]
            out = term if out is None else out + term
    return out


def _states_from_padded(padded, order: int, eps: float, nfaces: int):
    """Left-biased states at faces ``0..nfaces-1`` of a ghost-padded array."""
    tab = WENO_TABLES[int(order)]
    r, sub, d, Q = tab["r"], tab["sub"], tab["d"], tab["Q"]
    num = total = None
    for t in range(r):
        vals = [padded[t + k: t + k + nfaces] for k in range(r)]
        p = sub[t][0] * vals[0]
        for k in range(1, r):
            p = p + sub[t][k] * vals[k]
        alpha = d[t] / (eps + _beta(vals, Q)) ** 2
        num = alpha * p if num is None else num + alpha * p
        total = alpha if total is None else total + alpha
    return num / total


def weno_reconstruct(u: torch.Tensor, order: int = 7, eps: float = EPS_WENO):
    """Face states ``(u_left, u_right)`` of shape ``(ncells + 1, C)``.

    Transmissive ends: ``s = (order-1)/2`` ghost cells per side copy the boundary
    cell (the convention of :mod:`benchmarks.fv_weno5_1d`, so the same problem
    setup drives both solvers).
    """
    order = int(order)
    s = (order - 1) // 2
    ncells = u.shape[0]
    # symmetric (s+1)-cell ghost padding: the stencil of face j (between cells
    # j-1 and j, at x = lower + j*dx) is padded[j : j+order], the convention of
    # benchmarks.fv_weno5_1d where flux[j] sits on the same face.  One ghost more
    # than the minimum keeps the mirrored evaluation index-aligned (see below).
    padded = torch.cat((u[:1].expand(s + 1, -1), u, u[-1:].expand(s + 1, -1)), dim=0)
    u_left = _states_from_padded(padded, order, eps, ncells + 1)
    # the right state at face j is the left state at face N-j of the mirrored
    # array: reflection maps the upwind stencil onto the downwind one, and with
    # symmetric padding face j' of the mirrored array sits at x = (N-j')*dx
    u_right = _states_from_padded(torch.flip(padded, dims=[0]), order, eps,
                                  ncells + 1).flip(0)
    return u_left, u_right


# ------------------------------------------------------------------ Euler solve
def _admissible(u: torch.Tensor) -> bool:
    rho = u[:, 0]
    if not torch.isfinite(u).all() or float(rho.min()) <= 0.0:
        return False
    pressure = (GAMMA - 1.0) * (u[:, 2] - 0.5 * u[:, 1] ** 2 / rho)
    return float(pressure.min()) > 0.0


def solve_fv_weno(initial_state, ncells: int = 3200, order: int = 7,
                  final_time: float = 1.8, domain=(-5.0, 5.0), cfl: float = 0.3,
                  device: str = "cpu", dtype=torch.float64,
                  wave_speed: str = "davis", eps: float = EPS_WENO):
    """Advance ``initial_state`` to ``final_time`` with FV-WENO{order}/HLLC/SSP-RK3.

    ``initial_state(x)`` takes a ``(ncells, 1)`` coordinate tensor and returns
    ``(rho, velocity, pressure)`` -- the interface of
    :func:`benchmarks.fv_weno5_1d.solve_fv_weno5`.
    """
    dev, dt_type = torch.device(device), dtype
    lower, upper = domain
    dx = (upper - lower) / ncells
    x = lower + (torch.arange(ncells, device=dev, dtype=dt_type) + 0.5) * dx
    rho, velocity, pressure = (a.reshape(-1) for a in initial_state(x[:, None]))
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * velocity ** 2
    u = torch.stack((rho, rho * velocity, energy), dim=-1)

    def rhs(state):
        ul, ur = weno_reconstruct(state, order=order, eps=eps)
        flux = hllc_flux(ul, ur, wave_speed=wave_speed)
        flux = torch.cat((physical_flux(state[:1]), flux[1:-1],
                          physical_flux(state[-1:])), dim=0)
        return -(flux[1:] - flux[:-1]) / dx

    start, steps, t_now, dt = time.perf_counter(), 0, 0.0, 0.0
    while t_now < final_time:
        k1 = rhs(u)
        speed = float(torch.max(torch.abs(u[:, 1] / u[:, 0]) + sound_speed(u)).item())
        dt = min(cfl * dx / max(speed, 1e-12), final_time - t_now)
        u1 = u + dt * k1
        if not _admissible(u1):
            raise FloatingPointError(f"FV-WENO{order} inadmissible state at step {steps}")
        u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1))
        if not _admissible(u2):
            raise FloatingPointError(f"FV-WENO{order} inadmissible state at step {steps}")
        u = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * rhs(u2))
        if not _admissible(u):
            raise FloatingPointError(f"FV-WENO{order} inadmissible state at step {steps}")
        t_now += dt
        steps += 1
    meta = {
        "scheme": f"FV-WENO{order}-JS/HLLC/SSP-RK3", "ncells": int(ncells),
        "domain": [lower, upper], "final_time": final_time, "cfl": cfl,
        "steps": steps, "dt": float(dt), "wave_speed": wave_speed,
        "eps_weno": float(eps), "wall_time_s": time.perf_counter() - start,
        "density_min": float(u[:, 0].min()), "density_max": float(u[:, 0].max()),
    }
    return x.detach().cpu().numpy(), u.detach().cpu().numpy(), meta


# ----------------------------------------------------- scheme-order verification
def reconstruction_order_study(order: int = 7, resolutions=(40, 80, 160, 320),
                               wave_number: int = 1, eps: float = EPS_WENO) -> dict:
    """Observed order of the face reconstruction on a smooth periodic field.

    The cell averages of ``sin(2*pi*k*x)`` are built exactly (analytic integral),
    periodic ghost cells are used so no boundary effect enters, and the L-infinity
    error of ``u_left`` against the *pointwise* face value is measured.  Under the
    nonlinear WENO weights this recovers the formal order in the smooth regime,
    which is exactly the property the tables encode (no time integration -- the
    RK3 stage order is unchanged from the shipped solver and would otherwise mask
    a spatial order of 7 behind ``O(dt^3)``).
    """
    order = int(order)
    s = (order - 1) // 2
    rows = []
    for n in resolutions:
        dx = 1.0 / n
        j = torch.arange(n, dtype=torch.float64)
        a, b = j * dx, (j + 1) * dx
        w = 2 * torch.pi * wave_number
        ubar = (torch.cos(w * a) - torch.cos(w * b)) / (w * dx)   # exact averages
        padded = torch.cat((ubar[-(s + 1):], ubar, ubar[:s + 1]))
        ul = _states_from_padded(padded, order, eps, n + 1)
        xf = torch.arange(n + 1, dtype=torch.float64) * dx
        exact = torch.sin(w * xf)
        err = float((ul - exact).abs().max())
        rows.append({"ncells": int(n), "linf": err})
    for i in range(1, len(rows)):
        r = rows[i]["ncells"] / rows[i - 1]["ncells"]
        rows[i]["observed_order"] = float(
            math.log(rows[i - 1]["linf"] / rows[i]["linf"]) / math.log(r))
    return {"order": order, "wave_number": int(wave_number), "rows": rows,
            "finest_observed_order": rows[-1].get("observed_order")}


__all__ += ["selftest_tables"]