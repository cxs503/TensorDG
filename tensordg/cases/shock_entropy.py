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
                  final_time: float = FINAL_TIME, device: str = "cpu",
                  tvb_constant: float = 1.0, limiter_stage: str = "every") -> dict:
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
        tvb_constant=tvb_constant, limiter_stage=limiter_stage,
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


# ------------------------------------------------------------ error measurement
def coarsen(values, factor: int):
    """Volume-average a uniform-grid profile by an integer ``factor``.

    Only cell averages are compared across grids, so a coarse grid relates to a
    fine one by a plain mean (the conservative projection of the ``k = 0`` mode).
    """
    import numpy as _np

    values = _np.asarray(values, dtype=float)
    if factor < 1 or values.shape[0] % factor:
        raise ValueError(f"factor {factor} does not divide {values.shape[0]}")
    return values.reshape(values.shape[0] // factor, factor).mean(axis=1)


def relative_l1_error(candidate, reference, factor: int = 1) -> dict:
    """Relative L1/L2/Linf of ``candidate`` against ``reference`` (same grid after coarsening).

    ``factor`` is the refinement ratio of the *reference* grid to the candidate
    grid (``reference.size // candidate.size``); it is applied to the reference,
    never to the candidate.
    """
    import numpy as _np

    cand = _np.asarray(candidate, dtype=float)
    ref = coarsen(reference, factor) if factor > 1 else _np.asarray(reference, dtype=float)
    if cand.shape != ref.shape:
        raise ValueError(f"profile shapes differ: {cand.shape} vs {ref.shape}")
    diff = cand - ref
    l1 = float(_np.sum(_np.abs(diff)) / _np.sum(_np.abs(ref)))
    l2 = float(_np.sqrt(_np.sum(diff ** 2) / _np.sum(ref ** 2)))
    linf = float(_np.max(_np.abs(diff)) / _np.max(_np.abs(ref)))
    return {"rel_l1": l1, "rel_l1_percent": 100.0 * l1,
            "rel_l2_percent": 100.0 * l2, "linf_percent": 100.0 * linf}


def gate_report(ncells: int, degree: int = 2, limiter: str = "weno", cfl: float = 0.4,
                reference=None, reference_cells: int | None = None,
                tolerance_percent: float = 5.0) -> dict:
    """Solve the case and report the analytic anchors (library-side acceptance).

    ``reference`` is an optional ``(x, rho)`` reference profile (e.g. the
    converged FV-WENO5 solution shipped in ``benchmarks/data``); when given, the
    relative L1 error against it is added and compared with ``tolerance_percent``.
    Without a reference the report still carries the closed-form anchors, which
    need no external data: the post-shock plateau, the exact unshocked sine, the
    global mass budget and the Rankine--Hugoniot shock position.
    """
    import numpy as _np

    out = solve_dg_case(ncells, degree=degree, limiter=limiter, cfl=cfl)
    x, rho = out["x"], out["rho"]
    anchors = rankine_hugoniot_anchor()
    budget = exact_mass_budget()
    plateau = rho[(x >= DOMAIN[0]) & (x <= -3.0)]
    unshocked = x >= 2.6
    sine = 1.0 + WAVE_AMPLITUDE * _np.sin(WAVE_NUMBER * x[unshocked])
    mass = float(_np.sum(rho) * (DOMAIN[1] - DOMAIN[0]) / rho.size)
    # rightmost upward crossing of rho = 2.0 is the transmitted shock
    level = 2.0
    xs = None
    for i in range(rho.size - 1, 0, -1):
        if rho[i - 1] > level >= rho[i]:
            xs = x[i]
            break
    report = {
        "case": CASE_KEY,
        "configuration": {"ncells": ncells, "degree": degree, "limiter": limiter, "cfl": cfl},
        "steps": out["meta"]["steps"],
        "wall_time_s": out["meta"]["wall_time_s"],
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "rho_positive": bool(rho.min() > 0.0),
        "plateau_rho_mean": float(plateau.mean()),
        "plateau_target": LEFT_STATE[0],
        "plateau_dev_percent": 100.0 * abs(float(plateau.mean()) - LEFT_STATE[0]) / LEFT_STATE[0],
        "preshock_sine_rel_l1_percent": 100.0 * float(
            _np.sum(_np.abs(rho[unshocked] - sine)) / _np.sum(_np.abs(sine))),
        "mass_t1p8": mass,
        "mass_analytic": budget["mass_final_predicted"],
        "mass_error_percent_of_initial": 100.0 * abs(
            mass - budget["mass_final_predicted"]) / budget["mass_initial"],
        "shock_position": xs,
        "shock_position_analytic": anchors["shock_position_analytic"],
        "tolerance_percent": tolerance_percent,
    }
    if xs is not None:
        report["shock_position_percent_error"] = 100.0 * abs(
            xs - anchors["shock_position_analytic"]) / abs(anchors["shock_position_analytic"])
    if reference is not None:
        ref_x, ref_rho = reference
        factor = len(ref_rho) // ncells
        err = relative_l1_error(rho, ref_rho, factor)
        report["reference"] = {"ncells": int(len(ref_rho)), "coarsen_factor": int(factor),
                               "reference_cells": reference_cells or int(len(ref_rho))}
        report.update(err)
        report["passes_gate"] = bool(err["rel_l1_percent"] < tolerance_percent)
    else:
        report["passes_gate"] = None
    return report


def main(argv=None) -> int:
    """``python -m tensordg.cases.shock_entropy`` - run the case from the library.

    Prints the acceptance report as JSON; exit code 0 iff every check passes.
    With ``--reference`` an ``.npz`` holding ``x`` and ``rho`` (a converged
    reference profile, e.g. ``benchmarks/data/shu_osher_reference.npz``) the
    relative error against it is evaluated too.
    """
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ncells", type=int, default=400)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--limiter", default="weno")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--tolerance-percent", type=float, default=5.0)
    ap.add_argument("--reference", default=None,
                    help="optional .npz with x/rho of a converged reference profile")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    reference = None
    reference_cells = None
    if args.reference:
        loaded = np.load(args.reference, allow_pickle=True)
        reference = (loaded["x"], loaded["rho"])
        reference_cells = int(reference[0].size)
    report = gate_report(args.ncells, args.degree, args.limiter, args.cfl,
                         reference=reference, reference_cells=reference_cells,
                         tolerance_percent=args.tolerance_percent)
    text = _json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        from pathlib import Path as _Path
        _Path(args.json_out).write_text(text + "\n")
    ok = (report["rho_positive"]
          and report["plateau_dev_percent"] < 1.0
          and report["preshock_sine_rel_l1_percent"] < 0.5
          and report["mass_error_percent_of_initial"] < 0.1
          and (report["passes_gate"] is not False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
