"""CASE-DG-ADVECT-ORDER / M5_dt_order: temporal-order ladder vs the exact semidiscrete solution.

Why this exists
---------------
The case-pool anchor ``CASE-DG-ADVECT-ORDER`` (1-D periodic linear advection,
upwind DG, modal Legendre basis) is judged on *spatial* convergence rates with
a time integrator whose order is slaved to the polynomial degree
(``tableau_for_degree``: order ``p+1``, the RKDG convention).  That design rests
on one unverified premise: **the time-integration error is negligible at the
CFL used by the rate sweeps**.  Round 2 of this anchor flagged the missing piece
as ``M5_dt_order = not_run``, and round 1/2's attempt to measure it by refining
``dt`` at fixed ``N`` failed for an instrument reason, not a scheme reason: once
``dt`` is small the measured L2 error stops falling because it is pinned to the
*fixed* spatial error of that ``N``.

This driver measures the temporal order the only way that is free of that trap:
compare the numerical solution to the **exact solution of the semi-discrete
ODE system** ``du/dt = L u``, i.e. ``expm(L T) u0``, with ``L`` assembled from
the shipped right-hand side ``tensordg.advection_1d._rhs``.  The spatial error
cancels exactly in that comparison, so refining ``dt`` must reveal a clean
``dt^q`` ladder, ``q`` = the tableau order.  The remaining floor is float64
round-off, which is quantified per level (``roundoff_floor`` / signal-to-noise)
instead of being mistaken for a scheme property.

A second experiment ("space floor") runs the shipped time loop
``solve_advection`` with a refined ``dt`` ladder at fixed ``N`` and shows the
plateau explicitly, documenting *why* the round-2 probe was classed as an
instrument defect.

Outputs (all program-generated, no hand-transcribed numbers):
    artifacts/case_dg_advect_order_time_order.json   machine-readable tables
    docs/figures/case_dg_advect_order_time_order.png figure
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Callable

import numpy as np
import torch

from tensordg import advection_1d as A
from tensordg import convergence as C
from tensordg.runge_kutta import TABLEAUS, rk_step, tableau_for_degree

FINAL_TIME = 1.0
SPEED = 1.0
DEFAULT_LEVELS = 4
MAX_LEVELS = 10
DECADE_TARGET = 1e3        # refine until the error has dropped by this factor
FLOOR_GUARD = 3.0          # accept a level only if err > FLOOR_GUARD * accumulated floor
STEP_BUDGET = 400_000      # hard cap on total time steps per ladder (instrument guard)


# --------------------------------------------------------------------------- #
# setup helpers
# --------------------------------------------------------------------------- #
def project_sine(ncells: int, degree: int) -> torch.Tensor:
    """Modal Legendre coefficients of ``sin(2*pi*x)`` on ``ncells`` cells."""
    nq = max(2 * degree + 3, 4)
    xi, w = np.polynomial.legendre.leggauss(nq)
    h = 1.0 / ncells
    centres = (np.arange(ncells) + 0.5) * h
    x = centres[:, None] + 0.5 * h * xi[None, :]
    u = np.sin(2.0 * math.pi * x)
    # orthogonality: int_{-1}^{1} P_k P_k dxi = 2/(2k+1)
    coeff = np.empty((ncells, degree + 1))
    for k in range(degree + 1):
        c = np.zeros(k + 1)
        c[-1] = 1.0
        phi = np.polynomial.legendre.legval(xi, c)
        coeff[:, k] = (2 * k + 1) / 2.0 * (u @ (w * phi))
    return torch.as_tensor(coeff, dtype=torch.float64)


def make_apply(ncells: int, degree: int):
    """Flat-vector version of the shipped DG right-hand side."""
    _, _, _, volume, phi_l, phi_r = A._operators(degree, torch.device("cpu"), torch.float64)
    h = 1.0 / ncells
    shape = (ncells, degree + 1)

    def apply(flat) -> np.ndarray:
        u = torch.as_tensor(np.asarray(flat, dtype=np.float64).reshape(shape))
        out = A._rhs(u, h, SPEED, volume, phi_l, phi_r)
        return out.numpy().reshape(-1)

    return apply, shape


def modal_l2(coeff: np.ndarray, shape: tuple[int, int]) -> float:
    """L2 norm of the modal field: ``h * sum_k u_k^2 / (2k+1)`` per cell."""
    u = np.asarray(coeff, dtype=np.float64).reshape(shape)
    ncells, nmodes = u.shape
    h = 1.0 / ncells
    k = np.arange(nmodes)
    return float(np.sqrt(h * np.sum(u ** 2 / (2 * k + 1))))


def accumulated_roundoff_floor(exact_norm: float, nsteps: int, stages: int) -> float:
    """Round-off floor of ``nsteps`` explicit stages on a field of size ``||u||``.

    Each step is a difference of O(``||u||``) float64 numbers, so the random-walk
    accumulation over ``nsteps`` stages is ``eps * ||u|| * sqrt(nsteps * stages)``.
    A plain ``eps * ||u||`` floor (``convergence.roundoff_floor``) is far too
    optimistic here: at the fine end of a ladder the *integrator's own* rounding
    dominates the truncation error, which is exactly the failure that made the
    round-2 probe report a negative slope.
    """
    return C.FLOAT64_EPS * exact_norm * math.sqrt(max(nsteps, 1) * max(stages, 1))


def temporal_ladder(ncells: int, degree: int, tableau_name: str,
                    levels: int = DEFAULT_LEVELS) -> dict:
    """Refine ``dt`` against the exact semi-discrete solution; return the ladder."""
    apply, shape = make_apply(ncells, degree)
    ndof = ncells * (degree + 1)
    operator = C.semidiscrete_operator(apply, ndof)
    u0 = project_sine(ncells, degree).numpy().reshape(-1)
    exact = np.asarray(C.expm_scaled_squaring(operator, FINAL_TIME), dtype=np.float64) @ u0
    exact_norm = modal_l2(exact, shape)

    rho = float(np.max(np.abs(np.linalg.eigvals(operator))))
    tableau = TABLEAUS[tableau_name]
    nsteps0 = max(4, int(math.ceil(rho * FINAL_TIME)))   # dt0 * rho ~ 1: inside every stability region

    # The matrix is assembled from the *shipped* right-hand side above, so
    # stepping through it is the shipped operator, not a re-derivation.
    operator_t = torch.as_tensor(operator, dtype=torch.float64)
    rhs = lambda v: operator_t @ v            # noqa: E731  (flat DOF vector)
    u0_t = torch.as_tensor(u0, dtype=torch.float64)

    rows = []
    total_steps = 0
    for j in range(MAX_LEVELS):
        nsteps = nsteps0 * 2 ** j
        if j >= levels:
            # Beyond the base ladder, keep refining only while the error is still
            # comfortably above the round-off floor and large compared with the
            # target: that is what turns the round-2 plateau into a real ladder.
            if rows and (rows[0]["error"] >= DECADE_TARGET * rows[-1]["error"]
                         or not rows[-1]["resolved"]):
                break
            if total_steps + nsteps > STEP_BUDGET:
                break
        total_steps += nsteps
        dt = FINAL_TIME / nsteps
        u = u0_t
        for _ in range(nsteps):
            u = rk_step(rhs, u, dt, tableau)
        err = float(np.linalg.norm(u.numpy() - exact))
        floor = accumulated_roundoff_floor(exact_norm, nsteps, tableau.stages)
        rows.append({
            "nsteps": nsteps,
            "dt": dt,
            "error": err,
            "snr": float(exact_norm / err) if err > 0 else float("inf"),
            "floor": floor,
            "resolved": bool(err > FLOOR_GUARD * floor),
        })
        if not rows[-1]["resolved"] and j >= levels - 1:
            break           # floor reached: further refinement measures rounding, not order

    resolved = [r for r in rows if r["resolved"]]
    orders = C.pairwise_orders([r["nsteps"] for r in resolved],
                               [r["error"] for r in resolved])
    return {
        "ncells": ncells,
        "degree": degree,
        "tableau": tableau_name,
        "nominal_order": tableau.order,
        "stages": tableau.stages,
        "spectral_radius": rho,
        "exact_norm": exact_norm,
        "roundoff_floor": C.roundoff_floor(exact_norm),
        "accumulated_floor_last": rows[-1]["floor"],
        "levels": rows,
        "pairwise_orders": orders,
        "observed_order": (float(np.median(orders[-2:])) if len(orders) >= 2
                           else (orders[0] if len(orders) == 1 else None)),
        "fit_levels": len(resolved),
    }


def space_floor_ladder(ncells: int = 40, degree: int = 2, integrator: str = "rk3",
                       steps: tuple[int, ...] = (50, 100, 200, 400, 800)) -> list[dict]:
    """Fixed-``N``, refined-``dt`` ladder of the shipped time loop (the trap).

    The matched tableau (``rk3`` for ``p=2``) is used on purpose: at the coarse
    end its time error is still visible while ``O(dt^3)`` decay reaches the
    spatial error quickly, so the ladder shows decay, knee and plateau in one
    picture.  Refining past the knee measures nothing but the fixed spatial
    error, which is why a fixed-``N`` probe cannot yield ``q`` at all.
    """
    rows = []
    for nsteps in steps:
        dt = FINAL_TIME / nsteps
        centres, coeff, meta = A.solve_advection(
            ncells=ncells, degree=degree, final_time=FINAL_TIME, speed=SPEED,
            time_integrator=integrator, cfl=dt * SPEED / (1.0 / ncells))
        err = A.l2_error(centres, coeff, FINAL_TIME, SPEED)
        rows.append({"nsteps": nsteps, "dt": dt, "error": err,
                     "semidiscrete_error": None, "meta_steps": meta["steps"]})
    return rows


def schedule(ncells_list=(8, 16)) -> list[tuple[int, int, str, bool]]:
    """(N, p, tableau, is_matched) work list.

    The matched pairing is swept over every degree and both resolutions -- that
    is the configuration the anchor's spatial rates actually use.  Cross-checks
    (deliberately mismatched order, and a high-order tableau on a low-order
    space) are run once each: a tableau order is a property of the tableau, so
    repeating it for every ``p`` would only cost time steps, not evidence.
    """
    work = []
    for ncells in ncells_list:
        for degree in range(0, 5):
            work.append((ncells, degree, tableau_for_degree(degree).name, True))
    extras = [(8, 1, "rk4", False), (8, 4, "rk2", False), (16, 1, "rk5", False)]
    work += [w for w in extras if w[0] in ncells_list]
    return work


def main(ncells_list=(8, 16)) -> dict:
    temporal = []
    for ncells, degree, name, matched in schedule(ncells_list):
        row = temporal_ladder(ncells, degree, name)
        row["matched"] = matched
        temporal.append(row)

    floor = space_floor_ladder()
    orders = C.pairwise_orders([1.0 / r["dt"] for r in floor],
                               [r["error"] for r in floor])

    summary = []
    for row in temporal:
        summary.append({
            "N": row["ncells"], "p": row["degree"], "integrator": row["tableau"],
            "matched": row["matched"],
            "levels": len(row["levels"]),
            "error_span": (row["levels"][0]["error"] / row["levels"][-1]["error"]),
            "nominal": row["nominal_order"], "observed": row["observed_order"],
            "deviation": (None if row["observed_order"] is None
                          else row["observed_order"] - row["nominal_order"]),
        })

    payload = {
        "case": "CASE-DG-ADVECT-ORDER",
        "criterion": "M5_dt_order (temporal order vs exact semidiscrete solution)",
        "final_time": FINAL_TIME,
        "speed": SPEED,
        "temporal": temporal,
        "summary": summary,
        "space_floor": floor,
        "space_floor_orders": orders,
    }

    os.makedirs("artifacts", exist_ok=True)
    tag = os.environ.get("OUT_TAG")
    out = (f"artifacts/case_dg_advect_order_time_order_{tag}.json" if tag
           else "artifacts/case_dg_advect_order_time_order.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"wrote {out}")

    header = "| N | p | integrator | matched | nominal q | observed q | deviation | levels |"
    lines = [header, "|--:|--:|---|:--:|--:|--:|--:|--:|"]
    for s in summary:
        dev = "—" if s["deviation"] is None else f"{s['deviation']:+.3f}"
        obs = "—" if s["observed"] is None else f"{s['observed']:.4f}"
        lines.append(f"| {s['N']} | {s['p']} | {s['integrator']} | "
                     f"{'yes' if s['matched'] else 'no'} | {s['nominal']} | {obs} | {dev} | "
                     f"{s['levels']} |")
    print("\n".join(lines))
    print("\nspace-floor ladder (p=2, N=40): "
          + ", ".join(f"{r['error']:.3e}" for r in floor))
    print("space-floor orders: " + ", ".join(f"{o:.3f}" for o in orders))
    return payload


def merge_chunks(chunks: tuple[str, ...] = ("8", "16")) -> dict:
    """Fold per-resolution chunk files into the canonical artifact."""
    base = None
    rows = []
    for tag in chunks:
        with open(f"artifacts/case_dg_advect_order_time_order_{tag}.json") as fh:
            part = json.load(fh)
        base = base or {k: v for k, v in part.items() if k not in ("temporal", "summary")}
        rows.extend(part["temporal"])
    payload = dict(base or {})
    payload["temporal"] = rows
    payload["summary"] = [{
        "N": r["ncells"], "p": r["degree"], "integrator": r["tableau"],
        "matched": r["matched"], "nominal": r["nominal_order"],
        "observed": r["observed_order"],
        "deviation": (None if r["observed_order"] is None
                      else r["observed_order"] - r["nominal_order"]),
        "levels": len(r["levels"]), "fit_levels": r["fit_levels"],
    } for r in rows]
    with open("artifacts/case_dg_advect_order_time_order.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return payload


if __name__ == "__main__":
    grids = (8,) if "--only-8" in sys.argv else ((16,) if "--only-16" in sys.argv else (8, 16))
    main(grids)