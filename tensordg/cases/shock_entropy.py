"""Shu--Osher (1988) shock / entropy-wave interaction: canonical case definition.

Reference
---------
G. S. Jiang & C.-W. Shu, *Efficient implementation of weighted ENO schemes*,
J. Comput. Phys. **126** (1996) 202--228 uses this problem as its headline
one-dimensional test, and it originates in C.-W. Shu & S. Osher,
*Efficient implementation of essentially non-oscillatory shock-capturing
schemes*, J. Comput. Phys. **77** (1988) 439--471 (restated as test case 3 of
Shu & Osher, J. Comput. Phys. **83** (1989) 32--78).

Physics
-------
A Mach-3 right-moving normal shock at ``x = -4`` in a ``gamma = 1.4`` ideal gas
sweeps over a stationary sinusoidal density wave ``rho = 1 + 0.2 sin(5x)``.
Behind the shock the sine wave is converted into a fan of high-frequency entropy
and acoustic waves.  The case has no closed-form solution; what *is* closed-form
is the wave skeleton (shock speed, contact position, the head of the quiet zone)
and the global mass budget, and those are what this module provides as anchors.

Why the left state is not free data
-----------------------------------
``rho = 3.857143``, ``u = 2.629369``, ``p = 10.33333`` are the Rankine--Hugoniot
post-shock state of a Mach-3 shock into ``rho = p = 1`` (see
:func:`rankine_hugoniot_anchor`).  Computing them back from the jump conditions,
rather than trusting the literals, is what keeps this file free of transcribed
numbers: the literals are *checked*, not copied.

Usage
-----
>>> from tensordg.cases.shock_entropy import solve_dg_case
>>> out = solve_dg_case(ncells=200, degree=2, cfl=0.4)      # doctest: +SKIP
>>> out["rho"].shape, out["meta"]["steps"]                  # doctest: +SKIP
((200,), 4712)
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from tensordg.euler_dg_1d import solve_periodic_euler_dg_1d

CASE_KEY = "shu-osher-1988-shock-entropy"
GAMMA = 1.4
DOMAIN = (-5.0, 5.0)
DIAPHRAGM = -4.0
FINAL_TIME = 1.8
#: (rho, u, p) of the uniform post-shock state on ``x < -4``
LEFT_STATE = (3.857143, 2.629369, 10.33333)
#: pre-shock state on ``x >= -4``: ``rho = 1 + amplitude*sin(wavenumber*x)``
WAVE_AMPLITUDE = 0.2
WAVE_NUMBER = 5.0
RIGHT_STATE = (1.0, 0.0, 1.0)
#: Mach number of the incident shock (into the ``rho = p = 1`` still gas)
SHOCK_MACH = 3.0


def initial_state(x):
    """Case initial data as ``(rho, u, p)`` on the coordinates ``x``."""
    x = torch.as_tensor(x, dtype=torch.float64) if not torch.is_tensor(x) else x
    left_rho, left_u, left_p = LEFT_STATE
    rho = torch.where(x < DIAPHRAGM, torch.full_like(x, left_rho),
                      torch.ones_like(x) + WAVE_AMPLITUDE * torch.sin(WAVE_NUMBER * x))
    u = torch.where(x < DIAPHRAGM, torch.full_like(x, left_u), torch.zeros_like(x))
    p = torch.where(x < DIAPHRAGM, torch.full_like(x, left_p), torch.ones_like(x))
    return rho, u, p


def conserved_state(x):
    """Case initial data in conserved variables ``(rho, rho u, rho E)``."""
    rho, u, p = initial_state(x)
    e = p / (GAMMA - 1.0) + 0.5 * rho * u ** 2
    return torch.stack((rho, rho * u, e), dim=-1)


def solve_dg_case(ncells: int, degree: int = 2, limiter: str = "weno", cfl: float = 0.4,
                  final_time: float = FINAL_TIME, device: str = "cpu") -> dict:
    """Run the case with the modal DG solver and return cell-averaged fields.

    Returns a dict with ``x`` (cell centres), the three conserved cell averages
    ``rho``, ``rho_u``, ``rho_e`` (each 1-D, length ``ncells``), and the solver
    ``meta`` plus the wall time.  Keeping the three columns apart is deliberate:
    the density is column 0 of the solver's ``(ncells, 3)`` state, and treating
    the whole state as "density" corrupts every profile error (``rho E`` is ~39
    on the post-shock plateau, ten times the density).
    """
    start = time.perf_counter()
    x, coeff, meta = solve_periodic_euler_dg_1d(
        initial_state, ncells=ncells, degree=degree, final_time=final_time,
        domain=DOMAIN, cfl=cfl, device=device, limiter=limiter,
        boundary="transmissive",
    )
    meta = dict(meta)
    meta["wall_time_s"] = time.perf_counter() - start
    # the solver returns (ncells, degree + 1, 3) modal coefficients; the k = 0
    # coefficient is the cell average, i.e. the conservative projection
    state = coeff[:, 0, :].detach().cpu().numpy()
    if state.ndim != 2 or state.shape[1] != 3:
        raise ValueError(f"expected conserved cell averages (ncells, 3), got {state.shape}")
    return {
        "x": x.detach().cpu().numpy(),
        "rho": state[:, 0],
        "rho_u": state[:, 1],
        "rho_e": state[:, 2],
        "meta": meta,
    }


def rankine_hugoniot_anchor() -> dict:
    """Wave skeleton of the case from the jump conditions (no literature table).

    The shock speed follows from the mass jump ``rho_L (u_L - S) = rho_R (u_R - S)``
    with ``u_R = 0``; the left state is regenerated from a Mach-3 shock into
    ``rho = p = 1`` and compared with the literals in :data:`LEFT_STATE`, so the
    case definition verifies itself.
    """
    rho_l, u_l, p_l = LEFT_STATE
    rho_r, u_r, p_r = RIGHT_STATE
    shock_speed = (rho_r * u_r - rho_l * u_l) / (rho_r - rho_l)
    c_r = math.sqrt(GAMMA * p_r / rho_r)
    s_from_mach = SHOCK_MACH * c_r
    rho_l_rh = rho_r * ((GAMMA + 1.0) * SHOCK_MACH ** 2) / (
        (GAMMA - 1.0) * SHOCK_MACH ** 2 + 2.0)
    u_l_rh = s_from_mach * (1.0 - rho_r / rho_l_rh)
    p_l_rh = p_r * (1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (SHOCK_MACH ** 2 - 1.0))
    return {
        "shock_speed_analytic": shock_speed,
        "shock_speed_from_mach3": s_from_mach,
        "contact_speed_analytic": u_l,
        "shock_position_analytic": DIAPHRAGM + shock_speed * FINAL_TIME,
        "contact_position_analytic": DIAPHRAGM + u_l * FINAL_TIME,
        "hugoniot_left_state_from_mach3": {"rho": rho_l_rh, "u": u_l_rh, "p": p_l_rh},
        "hugoniot_left_state_case": {"rho": rho_l, "u": u_l, "p": p_l},
        "hugoniot_relative_error": {
            "rho": abs(rho_l_rh - rho_l) / rho_l,
            "u": abs(u_l_rh - u_l) / u_l,
            "p": abs(p_l_rh - p_l) / p_l,
        },
        # the incident perturbation starts at x = -4, so the slowest family it
        # sheds (left-going acoustic waves, lab speed u_L - c_L) only reaches
        # x = -4 + (u_L - c_L) t; everything to the left stays at the left state
        "quiet_zone_head_analytic": DIAPHRAGM + (
            u_l - math.sqrt(GAMMA * p_l / rho_l)) * FINAL_TIME,
        "post_shock_plateau": {"rho": rho_l, "window": [DOMAIN[0], -3.0]},
    }


def exact_mass_budget(samples: int = 2_000_000) -> dict:
    """Exact mass in the domain at ``t = 1.8`` (analytic, no discretisation).

    No wave ever reaches ``x = -5`` (everything moves right), so the boundary
    flux is the constant ``rho_L u_L`` and ``M(t) = M(0) + rho_L u_L t`` with the
    inflow counted positive.
    """
    rho_l, u_l, _ = LEFT_STATE
    length = DOMAIN[1] - DOMAIN[0]
    x = np.linspace(DOMAIN[0], DOMAIN[1], samples, endpoint=False) + 0.5 * length / samples
    rho0 = np.where(x < DIAPHRAGM, rho_l,
                    RIGHT_STATE[0] + WAVE_AMPLITUDE * np.sin(WAVE_NUMBER * x))
    mass0 = float(np.sum(rho0) * length / samples)
    flux = rho_l * u_l
    return {
        "mass_initial": mass0,
        "left_boundary_mass_flux": flux,
        "mass_inflow": flux * FINAL_TIME,
        "mass_final_predicted": mass0 + flux * FINAL_TIME,
        "mass_inflow_relative_to_initial": flux * FINAL_TIME / mass0,
        "closure": "inflow through x=-5 only; no wave reaches the boundary",
    }