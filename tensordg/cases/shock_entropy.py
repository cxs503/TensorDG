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
#: margin past the diaphragm whose left side is still the uniform left state.
#: Every signal the sine sheds is *right-going* in the lab frame
#: (``u_L - c_L = 0.69 > 0``), so this window is only a buffer for the shock
#: ramp, not a physics cut-off.
PLATEAU_MARGIN = 1.0
#: margin ahead of the analytic shock front that is still undisturbed initial
#: data; the sine there has not been reached by any wave yet.
UNDISTURBED_MARGIN = 0.2
#: density level used to locate the transmitted shock: half way between the
#: post-shock plateau and the largest pre-shock density (``1 + amplitude``)
SHOCK_LEVEL = 0.5 * (LEFT_STATE[0] + RIGHT_STATE[0] + WAVE_AMPLITUDE)


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
        "post_shock_plateau": {"rho": rho_l,
                               "window": [DOMAIN[0], DIAPHRAGM + PLATEAU_MARGIN]},
        # the undisturbed window is anchored to the *analytic* front, never to a
        # hand-picked abscissa: if the final time or the Mach number changes the
        # window follows the physics instead of silently measuring the shock.
        "undisturbed_window": [DIAPHRAGM + shock_speed * FINAL_TIME + UNDISTURBED_MARGIN,
                               DOMAIN[1]],
    }


def exact_mass_budget(samples: int | None = None) -> dict:
    """Exact mass in the domain at ``t = 1.8`` (closed form, no discretisation).

    No wave ever reaches ``x = -5`` (everything moves right), so the boundary
    flux is the constant ``rho_L u_L`` and ``M(t) = M(0) + rho_L u_L t`` with the
    inflow counted positive.  ``M(0)`` is integrated in closed form,

        M(0) = rho_L (x_d - x_0) + rho_R (x_1 - x_d)
               + A [cos(k x_d) - cos(k x_1)] / k,

    so the budget carries no quadrature error at all: the closure residual of a
    run is then a statement about the solver, not about the reference value.
    ``samples`` is accepted for call-site compatibility and ignored.
    """
    rho_l, u_l, _ = LEFT_STATE
    x0, x1 = DOMAIN
    amplitude, wavenumber = WAVE_AMPLITUDE, WAVE_NUMBER
    mass_left = rho_l * (DIAPHRAGM - x0)
    mass_right = RIGHT_STATE[0] * (x1 - DIAPHRAGM) + amplitude * (
        math.cos(wavenumber * DIAPHRAGM) - math.cos(wavenumber * x1)) / wavenumber
    mass0 = mass_left + mass_right
    flux = rho_l * u_l
    return {
        "mass_initial": mass0,
        "mass_initial_left_of_diaphragm": mass_left,
        "mass_initial_right_of_diaphragm": mass_right,
        "left_boundary_mass_flux": flux,
        "mass_inflow": flux * FINAL_TIME,
        "mass_final_predicted": mass0 + flux * FINAL_TIME,
        "mass_inflow_relative_to_initial": flux * FINAL_TIME / mass0,
        "closure": "inflow through x=-5 only; no wave reaches the boundary",
        "quadrature": "closed form (sin integral), no numerical integration",
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


def resample_cell_averages(values, target_ncells: int, domain=DOMAIN):
    """Volume-average a uniform-grid cell-average profile onto another uniform grid.

    Both grids cover the same ``domain``.  A cell average is, by definition, the
    mean of a piecewise-constant field over the cell, so the mass integral of the
    source profile is known exactly and the target average is

        <u>_j = (1/h_j) int_{cell j} u

    with the fractional overlap taken into account.  Mass is therefore preserved to
    round-off for *any* resolution ratio, integer or not - which is what lets a
    400-cell solution be compared with the committed 300-cell reference (neither
    grid divides the other).
    """
    import numpy as _np

    values = _np.asarray(values, dtype=float).ravel()
    n_src = values.size
    if target_ncells < 1:
        raise ValueError(f"target_ncells must be positive, got {target_ncells}")
    if target_ncells == n_src:
        return values.copy()
    x0, x1 = domain
    h_src = (x1 - x0) / n_src
    edges_src = x0 + h_src * _np.arange(n_src + 1)
    cumulative = _np.concatenate(([0.0], _np.cumsum(values * h_src)))

    def antiderivative(edge):
        """``int_{x0}^{edge} u``  for the piecewise-constant profile."""
        index = _np.clip(_np.searchsorted(edges_src, edge, side="right") - 1, 0, n_src - 1)
        return cumulative[index] + (edge - edges_src[index]) * values[index]

    edges_tgt = x0 + (x1 - x0) / target_ncells * _np.arange(target_ncells + 1)
    mass_tgt = antiderivative(edges_tgt[1:]) - antiderivative(edges_tgt[:-1])
    return mass_tgt * (target_ncells / (x1 - x0))


def relative_l1_error(candidate, reference, factor: int = 1, domain=None) -> dict:
    """Relative L1/L2/Linf of ``candidate`` against ``reference``.

    Three comparison modes, in order of preference:

    * ``factor > 1``: ``reference`` lives on a grid ``factor`` times finer and is
      volume-averaged down with :func:`coarsen` (the fast, exactly conservative
      path when one resolution divides the other);
    * equal shapes: compared cell by cell;
    * otherwise, if ``domain`` is given, both profiles are conservatively
      re-averaged onto the coarser of the two grids (see
      :func:`resample_cell_averages`); without ``domain`` a shape mismatch is an
      error rather than a silent broadcast.

    The chosen mode is echoed under ``comparison`` so a reported number always
    says which grids were actually compared.
    """
    import numpy as _np

    cand = _np.asarray(candidate, dtype=float).ravel()
    ref = _np.asarray(reference, dtype=float).ravel()
    comparison = {"method": "direct", "candidate_cells": int(cand.size),
                  "reference_cells": int(ref.size), "common_grid_cells": int(cand.size)}
    if factor > 1:
        ref = coarsen(ref, factor)
        comparison.update(method="coarsen", coarsen_factor=int(factor),
                          common_grid_cells=int(ref.size))
    if cand.shape != ref.shape:
        if domain is None:
            raise ValueError(
                f"profile shapes differ: {cand.shape} vs {ref.shape}; pass "
                "domain=(x0, x1) to compare two uniform profiles that do not "
                "share a refinement ratio")
        common = min(cand.size, ref.size)
        cand = resample_cell_averages(cand, common, domain)
        ref = resample_cell_averages(ref, common, domain)
        comparison.update(method="remap", common_grid_cells=int(common))
    diff = cand - ref
    l1 = float(_np.sum(_np.abs(diff)) / _np.sum(_np.abs(ref)))
    l2 = float(_np.sqrt(_np.sum(diff ** 2) / _np.sum(ref ** 2)))
    linf = float(_np.max(_np.abs(diff)) / _np.max(_np.abs(ref)))
    return {"rel_l1": l1, "rel_l1_percent": 100.0 * l1,
            "rel_l2_percent": 100.0 * l2, "linf_percent": 100.0 * linf,
            "comparison": comparison}


def shock_position(x, rho, level: float | None = None):
    """Interpolated ``x`` of the transmitted shock (rightmost downward crossing).

    ``rho`` falls through ``level`` from the post-shock plateau to the pre-shock
    wave, so the shock is the *last* crossing when scanned from the right; the
    crossing is linearly interpolated between the two straddling cell centres,
    which removes the ``O(dx)`` quantisation of a nearest-cell pick.
    ``level`` defaults to :data:`SHOCK_LEVEL`.
    """
    import numpy as _np

    x = _np.asarray(x, dtype=float).ravel()
    rho = _np.asarray(rho, dtype=float).ravel()
    if x.shape != rho.shape:
        raise ValueError(f"x and rho differ in shape: {x.shape} vs {rho.shape}")
    level = SHOCK_LEVEL if level is None else float(level)
    for i in range(rho.size - 1, 0, -1):
        if rho[i - 1] > level >= rho[i]:
            r0, r1 = rho[i - 1], rho[i]
            if r1 == r0:
                return float(x[i])
            return float(x[i - 1] + (level - r0) * (x[i] - x[i - 1]) / (r1 - r0))
    return None


def gate_report(ncells: int, degree: int = 2, limiter: str = "weno", cfl: float = 0.4,
                reference=None, reference_cells: int | None = None,
                tolerance_percent: float = 5.0, reference_meta: dict | None = None,
                include_profile: bool = False) -> dict:
    """Solve the case and report the analytic anchors (library-side acceptance).

    ``reference`` is an optional ``(x, rho)`` reference profile (e.g. the 300-cell
    DG profile shipped in ``benchmarks/data/shu_osher_reference.npz``); when given,
    the relative L1 error against it is added and compared with
    ``tolerance_percent``.  The reference does **not** have to share a refinement
    ratio with ``ncells``: profiles are conservatively re-averaged onto the
    coarser grid (see :func:`relative_l1_error`) and the grid pair actually
    compared is echoed under ``report["reference"]["comparison"]``.

    Without a reference the report still carries the closed-form anchors, which
    need no external data: the post-shock plateau, the exact unshocked sine, the
    global mass budget and the Rankine--Hugoniot shock position.

    ``include_profile=True`` adds the solved cell averages under ``"profile"``
    (``x``, ``rho``, ``rho_u``, ``rho_e``) so that a caller can measure the *same*
    field another way without paying for a second solve.  ``main`` strips it before
    printing, so the CLI record stays plain JSON.
    """
    import numpy as _np

    out = solve_dg_case(ncells, degree=degree, limiter=limiter, cfl=cfl)
    x, rho = out["x"], out["rho"]
    anchors = rankine_hugoniot_anchor()
    budget = exact_mass_budget()
    # every measurement window is derived from the analytic wave skeleton, so the
    # report keeps measuring the plateau / the undisturbed sine if the case
    # parameters (Mach number, final time) are ever changed.
    plateau_lo, plateau_hi = anchors["post_shock_plateau"]["window"]
    undisturbed_lo, undisturbed_hi = anchors["undisturbed_window"]
    plateau = rho[(x >= plateau_lo) & (x <= plateau_hi)]
    unshocked = (x >= undisturbed_lo) & (x <= undisturbed_hi)
    n_unshocked = int(_np.count_nonzero(unshocked))
    if n_unshocked:
        sine = 1.0 + WAVE_AMPLITUDE * _np.sin(WAVE_NUMBER * x[unshocked])
        sine_error_percent = 100.0 * float(
            _np.sum(_np.abs(rho[unshocked] - sine)) / _np.sum(_np.abs(sine)))
    else:
        # a grid too coarse to put a cell in the undisturbed window: report the
        # absence explicitly instead of a silent 0 %
        sine_error_percent = None
    mass = float(_np.sum(rho) * (DOMAIN[1] - DOMAIN[0]) / rho.size)
    xs = shock_position(x, rho)
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
        "preshock_sine_rel_l1_percent": sine_error_percent,
        "preshock_sine_cells": n_unshocked,
        "mass_t1p8": mass,
        "mass_analytic": budget["mass_final_predicted"],
        "mass_error_percent_of_initial": 100.0 * abs(
            mass - budget["mass_final_predicted"]) / budget["mass_initial"],
        "shock_position": xs,
        "shock_position_analytic": anchors["shock_position_analytic"],
        "shock_level": SHOCK_LEVEL,
        "shock_position_cell_step": float((DOMAIN[1] - DOMAIN[0]) / rho.size),
        "windows": {"plateau": [plateau_lo, plateau_hi],
                    "undisturbed": [undisturbed_lo, undisturbed_hi]},
        "tolerance_percent": tolerance_percent,
    }
    if xs is not None:
        report["shock_position_percent_error"] = 100.0 * abs(
            xs - anchors["shock_position_analytic"]) / abs(anchors["shock_position_analytic"])
    if include_profile:
        report["profile"] = {"x": x, "rho": rho, "rho_u": out["rho_u"],
                             "rho_e": out["rho_e"]}
    if reference is not None:
        ref_x, ref_rho = reference
        ref_ncells = int(_np.asarray(ref_rho).size)
        factor = ref_ncells // int(ncells)
        if factor < 1 or ref_ncells % int(ncells):
            factor = 1  # no integer refinement ratio: let the remap path do the work
        err = relative_l1_error(rho, ref_rho, factor, domain=DOMAIN)
        comparison = err.pop("comparison")
        report["reference"] = {"ncells": ref_ncells,
                               "reference_cells": reference_cells or ref_ncells,
                               "coarsen_factor": int(comparison.get("coarsen_factor", 1)),
                               "comparison": comparison,
                               "meta": reference_meta}
        report.update(err)
        report["passes_gate"] = bool(err["rel_l1_percent"] < tolerance_percent)
    else:
        report["passes_gate"] = None
    return report


def main(argv=None) -> int:
    """``python -m tensordg.cases.shock_entropy`` - run the case from the library.

    Prints the acceptance report as JSON; exit code 0 iff every check passes.
    With ``--reference`` an ``.npz`` holding ``x`` and ``rho`` the relative error
    against it is evaluated too.  The grids need not share a refinement ratio:
    ``benchmarks/data/shu_osher_reference.npz`` is a *300-cell DG P3* profile, so
    the comparison is done on the coarser grid after conservative re-averaging.
    Whatever the file says about itself (its ``meta``) is echoed under
    ``report["reference"]["meta"]`` so a published number always states which
    reference it was measured against.
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
    reference_meta = None
    if args.reference:
        loaded = np.load(args.reference, allow_pickle=True)
        reference = (loaded["x"], loaded["rho"])
        reference_cells = int(reference[0].size)
        if "meta" in loaded.files:
            try:
                reference_meta = _json.loads(str(loaded["meta"]))
            except ValueError:  # a non-JSON meta must not take the gate down
                reference_meta = {"raw": str(loaded["meta"])}
    report = gate_report(args.ncells, args.degree, args.limiter, args.cfl,
                         reference=reference, reference_cells=reference_cells,
                         tolerance_percent=args.tolerance_percent,
                         reference_meta=reference_meta)
    text = _json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        from pathlib import Path as _Path
        _Path(args.json_out).write_text(text + "\n")
    sine_ok = (report["preshock_sine_rel_l1_percent"] is not None
               and report["preshock_sine_rel_l1_percent"] < 0.5)
    ok = (report["rho_positive"]
          and report["plateau_dev_percent"] < 1.0
          and sine_ok
          and report["mass_error_percent_of_initial"] < 0.1
          and (report["passes_gate"] is not False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
