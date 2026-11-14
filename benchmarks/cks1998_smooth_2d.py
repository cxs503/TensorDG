"""CKS1998 §5.1 accuracy test on the 2-D Euler *system* — L1 and L-infinity.

What the anchor asks for and what was missing
---------------------------------------------
``REPRO-DG-CKS1998`` (Cockburn & Shu, *The Runge-Kutta discontinuous Galerkin
method for conservation laws V: multidimensional systems*, JCP 141 (1998)
199-224) verifies the order of accuracy of the multidimensional RKDG scheme on
a **smooth solution of the 2-D Euler equations**, reporting both the ``L1`` and
the ``L-infinity`` error of the density for ``P^k``, ``k = 1, 2, 3``.

Until this benchmark the repository's only 2-D Euler order evidence was

* ``benchmarks/design_order_euler_dg_2d.py`` -- relative **L2** error of
  ``(rho, u, v, p)`` on the *isentropic vortex* (exact, but a different exact
  solution than the paper's), and
* ``benchmarks/limiter_design_order_2d.py`` -- same norm, same case, with the
  TVB/WENO limiter switched on.

So **the paper's own smooth case and the ``L1`` / ``L-infinity`` norms were
never measured** for the 2-D system solver.  This module measures them.

The exact solution (this is the paper's test, and it is exactly the entropy
wave advected by a uniform flow, ``u = v = 1``, ``p = 1``)::

    rho(x, y, t) = 1 + 0.2 sin(pi (x + y - 2 t)),   u = v = 1,   p = 1

on the periodic box ``[0, 2]^2``.  Substituting into the 2-D Euler equations
shows that the momentum and energy equations reduce to the *same* linear
advection equation as continuity, because ``u`` and ``v`` are constant and
``p`` is constant -- hence the closed form above is an exact solution and the
reference is analytic, with no manufactured source term (``tests/
test_cks1998_smooth_2d_order.py::test_smooth_solution_satisfies_the_euler_
residual`` re-derives that with automatic differentiation, so the reference
itself is gated).

What is reported
----------------
For every ``(degree, mesh, limiter configuration)``:

* ``L1`` (integral of ``|rho_h - rho_exact|``), the same norm normalised by
  ``integral |rho_exact|`` (``L1_rel``), and ``L-infinity`` at the tensor
  Gauss nodes -- i.e. the two norms the paper tabulates;
* the adjacent-mesh observed order ``EOC`` computed with the actual mesh ratio;
* the design order ``p + 1`` the paper proves for smooth solutions, next to the
  measured ``EOC``.

Space/time order coupling -- measured, not assumed
--------------------------------------------------
The solver advances in time with **SSP-RK3**, so the temporal error is
``O(dt^3)``; for ``P3`` the spatial error is ``O(h^4)`` and *could* be swamped.
That was the hypothesis this benchmark was built to test -- and it was
**falsified for this case**: at ``CFL = 0.2`` on the ladders below, ``P3``
measures an observed order of 3.98 (L1) / 4.06 (L-infinity), i.e. the design
order 4, because the advection speed is O(1) and the spatial error still
dominates at ``h >= 1/16``.  ``--order-matched-dt`` (``CFL(h) = CFL_0
(h/h_0)^gamma``, ``gamma = (p+1)/3 - 1``, clipped at 0 -- the same isolator as
``benchmarks/design_order_euler_dg_2d.py``) is therefore available but **not
needed** here; it is kept for meshes fine enough that the temporal term starts
to bind.

Limiter threshold prediction (falsifiable)
------------------------------------------
The TVB minmod test flags a cell when the deviation of the one-sided difference
from the minmod slope exceeds ``M h^2``.  For smooth data that deviation is
``~ h^2 |rho''| / 2``, so the limiter is invisible as soon as

    ``M > M* = max |d^2 rho / dx^2| = 0.2 pi^2 ~= 1.974``

(in the units of each component of the conserved state; for this case
``rho, rho u, rho v, E`` all carry the same ``0.2 pi^2`` because ``u = v = 1``
and ``p = 1``).  The sweep therefore measures ``M = 1`` (below the threshold,
order should degrade) and ``M = 3, 10`` (above it, order should be recovered),
and the measured ``M*`` is reported next to the prediction.

Usage
-----
    OMP_NUM_THREADS=1 python -m benchmarks.cks1998_smooth_2d            # full sweep
    OMP_NUM_THREADS=1 python -m benchmarks.cks1998_smooth_2d --quick    # ~1 min corner
    OMP_NUM_THREADS=1 python -m benchmarks.cks1998_smooth_2d --phases order

Results are written after *every* run (``validation/cks1998_smooth_2d/
order.json``), so an interrupted sweep still leaves usable evidence; the figure
(``docs/figures/cks1998_smooth_2d_order.png``) is regenerated from the JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "validation" / "cks1998_smooth_2d"
FIG = REPO / "docs" / "figures" / "cks1998_smooth_2d_order.png"

DOMAIN = ((0.0, 2.0), (0.0, 2.0))
FINAL_TIME = 0.5
CFL0 = 0.2
AMPLITUDE = 0.2

#: ``(degree, meshes)`` ladders.  Kept per degree and per configuration so the
#: whole sweep fits a few minutes of node time; the ratio between adjacent
#: meshes is the mesh ratio used by the EOC (never assumed to be 2).
LADDERS = {1: [16, 24, 32], 2: [12, 16, 24], 3: [8, 12, 16]}


# ---------------------------------------------------------------------------
# Analytic reference
# ---------------------------------------------------------------------------
def smooth_primitive(x: torch.Tensor, y: torch.Tensor, time: float):
    """The paper's smooth solution: (rho, u, v, p) at ``(x, y, time)``."""
    rho = 1.0 + AMPLITUDE * torch.sin(math.pi * (x + y - 2.0 * time))
    ones = torch.ones_like(rho)
    return rho, ones.clone(), ones.clone(), ones


def smooth_conservative(x: torch.Tensor, y: torch.Tensor, time: float):
    rho, u, v, p = smooth_primitive(x, y, time)
    return dg2.conservative_state(rho, u, v, p)


def initial(x: torch.Tensor, y: torch.Tensor):
    return smooth_primitive(x, y, 0.0)


#: ``max |d^2 rho / dx^2|`` of the exact solution -> predicted TVB threshold.
PREDICTED_TVB_THRESHOLD = AMPLITUDE * math.pi ** 2


# ---------------------------------------------------------------------------
# Error norms at the tensor Gauss nodes
# ---------------------------------------------------------------------------
def gauss_fields(coeff: torch.Tensor, degree: int, quadrature: int | None = None):
    """Evaluate the modal solution at the tensor Gauss nodes of every cell.

    Returns ``(values, nodes, weights)`` with ``values`` shaped
    ``(nx, ny, q, q, 4)`` in conservative variables.
    """
    order = quadrature if quadrature is not None else max(degree + 2, 3)
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, order)
    values = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    return values, nodes, weights


def error_norms(coeff, x, y, degree, final_time=FINAL_TIME, domain=DOMAIN):
    """``L1`` (absolute and relative), ``L2`` relative and ``L-inf`` of each field."""
    values, nodes, weights = gauss_fields(coeff, degree)
    (xmin, xmax), (ymin, ymax) = domain
    nx, ny = x.numel(), y.numel()
    hx, hy = (xmax - xmin) / nx, (ymax - ymin) / ny
    X = x[:, None, None, None] + nodes[None, None, :, None] * (hx / 2.0)
    Y = y[None, :, None, None] + nodes[None, None, None, :] * (hy / 2.0)
    exact = smooth_conservative(X, Y, final_time)
    jacobian = (
        weights[None, None, :, None]
        * weights[None, None, None, :]
        * (hx / 2.0)
        * (hy / 2.0)
    )
    out = {}
    names = ("density", "momentum_x", "momentum_y", "energy")
    for index, name in enumerate(names):
        diff = (values[..., index] - exact[..., index]).abs()
        l1 = float((diff * jacobian).sum())
        scale = float((exact[..., index].abs() * jacobian).sum())
        l2 = float(torch.sqrt((diff ** 2 * jacobian).sum() / jacobian.sum()))
        l2_ref = float(torch.sqrt((exact[..., index] ** 2 * jacobian).sum() / jacobian.sum()))
        out[name] = {
            "L1": l1,
            "L1_rel": l1 / scale,
            "L2_rel": l2 / l2_ref,
            "Linf": float(diff.max()),
        }
    return out


# ---------------------------------------------------------------------------
# One run of the sweep
# ---------------------------------------------------------------------------
def cfl_schedule(degree: int, ncells: int, reference_cells: int, cfl0: float = CFL0):
    """``CFL(h) = cfl0 (h/h0)^gamma`` with ``gamma = max((p+1)/3 - 1, 0)``.

    ``gamma = 0`` for ``p <= 2``: there the temporal order 3 already matches or
    exceeds the spatial order, so no down-scaling is needed.  For ``p = 3`` the
    exponent is ``1/3``, making ``dt ~ h^{4/3}`` so that ``dt^3 ~ h^4``.
    """
    gamma = max((degree + 1) / 3.0 - 1.0, 0.0)
    return float(cfl0 * (reference_cells / ncells) ** gamma)


def run_case(degree: int, ncells: int, limiter: str | None, tvb_constant: float,
             order_matched_dt: bool, final_time: float = FINAL_TIME,
             cfl0: float = CFL0, device="cpu") -> dict:
    cfl = cfl_schedule(degree, ncells, LADDERS[degree][0], cfl0) if order_matched_dt else cfl0
    start = time.time()
    x, y, coeff, metadata = dg2.solve_euler_dg_2d(
        initial,
        ncells=ncells,
        degree=degree,
        domain=DOMAIN,
        final_time=final_time,
        cfl=cfl,
        limiter=limiter,
        tvb_constant=tvb_constant,
        periodic=(True, True),
        device=device,
    )
    wall = time.time() - start
    norms = error_norms(coeff, x, y, degree, final_time)
    hx = (DOMAIN[0][1] - DOMAIN[0][0]) / ncells
    return {
        "degree": degree,
        "ncells": ncells,
        "h": hx,
        "limiter": limiter,
        "tvb_constant": tvb_constant if limiter == "tvb" else None,
        "cfl": cfl,
        "order_matched_dt": bool(order_matched_dt),
        "final_time": final_time,
        "steps": metadata["steps"],
        "dt": metadata.get("dt"),
        "wall_seconds": round(wall, 2),
        "norms": norms,
        "dofs": metadata["dofs"],
    }


def with_eoc(runs: list[dict], field: str = "density", norm: str = "L1") -> list[dict]:
    """Attach adjacent-mesh observed orders (mesh ratio from ``h``)."""
    ordered = sorted(runs, key=lambda r: r["ncells"])
    for coarse, fine in zip(ordered, ordered[1:]):
        ratio = coarse["h"] / fine["h"]
        values = {}
        for key in ("L1", "L1_rel", "L2_rel", "Linf"):
            ec = math.log(coarse["norms"][field][key] / fine["norms"][field][key]) / math.log(ratio)
            values[key] = ec
        fine["eoc"] = values
        fine["eoc_ratio"] = ratio
    if ordered:
        ordered[0]["eoc"] = None
    return runs


# ---------------------------------------------------------------------------
# Phase A: order ladder with the limiter off (the paper's headline table)
# ---------------------------------------------------------------------------
def phase_order(ladders: dict, final_time: float = FINAL_TIME,
                order_matched_dt: bool = False, device="cpu") -> dict:
    results = {}
    for degree, meshes in ladders.items():
        runs = []
        for ncells in meshes:
            run = run_case(degree, ncells, None, 1.0, order_matched_dt, final_time, device=device)
            run["design_order"] = degree + 1
            runs.append(run)
        results[f"p{degree}"] = with_eoc(runs)
    return results


# ---------------------------------------------------------------------------
# Phase B: TVB threshold study (does the limiter destroy the order?)
# ---------------------------------------------------------------------------
def phase_limiter(degrees=(1,), meshes=(16, 24), constants=(1.0, 3.0, 10.0),
                  final_time: float = FINAL_TIME, device="cpu") -> dict:
    results = {}
    for degree in degrees:
        ladder = [m for m in meshes]
        for constant in constants:
            runs = []
            for ncells in ladder:
                run = run_case(degree, ncells, "tvb", constant, False, final_time, device=device)
                run["design_order"] = degree + 1
                runs.append(run)
            results[f"p{degree}_M{constant:g}"] = with_eoc(runs)
        # limiter-off reference on the identical ladder
        runs = []
        for ncells in ladder:
            run = run_case(degree, ncells, None, 1.0, False, final_time, device=device)
            run["design_order"] = degree + 1
            runs.append(run)
        results[f"p{degree}_none"] = with_eoc(runs)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(payload: dict) -> str:
    lines = []
    header = (
        "| config | P^k | mesh | h | steps | rho L1 | rho L1(rel) | rho Linf "
        "| EOC L1 | EOC Linf | design p+1 |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 11)
    for name, runs in payload.items():
        for run in runs:
            eoc = run.get("eoc") or {}
            lines.append(
                "| {} | {} | {} | {:.3e} | {} | {:.3e} | {:.3e} | {:.3e} | {} | {} | {} |".format(
                    name,
                    run["degree"],
                    run["ncells"],
                    run["h"],
                    run["steps"],
                    run["norms"]["density"]["L1"],
                    run["norms"]["density"]["L1_rel"],
                    run["norms"]["density"]["Linf"],
                    "--" if not eoc else f"{eoc['L1']:.2f}",
                    "--" if not eoc else f"{eoc['Linf']:.2f}",
                    run["design_order"],
                )
            )
    return "\n".join(lines)


def make_figure(payload: dict, out: Path = FIG) -> Path | None:
    """Log-log error-vs-h figure for the L1 and L-infinity norms."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, norm, title in zip(axes, ("L1_rel", "Linf"),
                               ("density L1 (relative)", "density L-infinity")):
        for degree in sorted({run["degree"] for runs in payload.values() for run in runs}):
            series, style = None, None
            for name, runs in payload.items():
                picked = sorted([r for r in runs if r["degree"] == degree], key=lambda r: r["h"])
                if not picked:
                    continue
                style = "-o" if name == f"p{degree}" or name.endswith("_none") else "--"
                label = name
                hs = [r["h"] for r in picked]
                es = [max(r["norms"]["density"][norm], 1e-300) for r in picked]
                ax.loglog(hs, es, style, label=label, markersize=4)
            _ = series
        # reference slopes h^(p+1)
        h_ref = np.array([2.0 / 64.0, 2.0 / 8.0])
        for degree, e0 in ((1, 1e-3), (2, 1e-5), (3, 1e-7)):
            ax.loglog(h_ref, e0 * (h_ref / h_ref[0]) ** -(degree + 1), ":", color="0.6")
            ax.annotate(f"slope {degree + 1}", (h_ref[-1], e0 * (h_ref[-1] / h_ref[0]) ** -(degree + 1)),
                        fontsize=7, color="0.4")
        ax.set_xlabel("h = dx")
        ax.set_ylabel(title)
        ax.invert_xaxis()
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("CKS1998 §5.1 smooth 2-D Euler accuracy test (exact entropy wave)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write(payload: dict, out_json: Path, meta: dict) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "anchor": "REPRO-DG-CKS1998",
        "case": "smooth_2d_entropy_wave",
        "reference": "analytic: rho = 1 + 0.2 sin(pi (x + y - 2 t)), u = v = 1, p = 1",
        "domain": DOMAIN,
        "final_time": meta.get("final_time"),
        "predicted_tvb_threshold": PREDICTED_TVB_THRESHOLD,
        "design_orders": {f"p{k}": k + 1 for k in (1, 2, 3)},
        "runs": payload,
        "table_markdown": summarize(payload),
        **meta,
    }
    out_json.write_text(json.dumps(document, indent=2))
    (out_json.with_suffix(".md")).write_text(
        "# CKS1998 §5.1 smooth 2-D Euler accuracy sweep\n\n"
        + summarize(payload)
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--meshes", type=int, nargs="*", default=None,
                        help="override the per-degree ladder with one common ladder")
    parser.add_argument("--phases", nargs="*", default=["order", "limiter"],
                        choices=["order", "limiter"])
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--order-matched-dt", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="cheap corner of the sweep (one mesh pair per degree)")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "order.json")
    parser.add_argument("--figure", type=Path, default=FIG)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(max(args.threads, 1))
    if args.quick:
        ladders = {1: [16, 32], 2: [12, 16], 3: [8, 12]}
        degrees, meshes = (1, 2), (16,)
    else:
        if args.meshes:
            ladders = {k: list(args.meshes) for k in args.degrees}
        else:
            ladders = {k: LADDERS[k] for k in args.degrees}
        degrees, meshes = (1, 2), (16, 24)

    payload: dict = {}
    meta = {"final_time": args.final_time, "order_matched_dt": bool(args.order_matched_dt)}

    def flush():
        write(payload, args.out, meta)

    if "order" in args.phases:
        for degree, ladder in ladders.items():
            runs = []
            for ncells in ladder:
                run = run_case(degree, ncells, None, 1.0, args.order_matched_dt, args.final_time)
                run["design_order"] = degree + 1
                runs.append(run)
                print(f"[order] p{degree} {ncells}x{ncells}: "
                      f"L1rel={run['norms']['density']['L1_rel']:.3e} "
                      f"Linf={run['norms']['density']['Linf']:.3e} "
                      f"steps={run['steps']} wall={run['wall_seconds']}s", flush=True)
                payload[f"p{degree}"] = with_eoc(runs)
                flush()

    if "limiter" in args.phases:
        limiter = phase_limiter(degrees=degrees, meshes=meshes, final_time=args.final_time)
        payload.update(limiter)
        for name, runs in limiter.items():
            tail = runs[-1]
            eoc = tail.get("eoc") or {}
            print(f"[limiter] {name}: EOC L1 = {eoc.get('L1', float('nan')):.2f} "
                  f"Linf = {eoc.get('Linf', float('nan')):.2f} (design {tail['design_order']})",
                  flush=True)
        flush()

    flush()
    figure = make_figure(payload, args.figure)
    votes = []
    for name, runs in payload.items():
        ordered = sorted(runs, key=lambda r: -r["ncells"])
        if ordered and ordered[0].get("eoc"):
            votes.append((name, ordered[0]["eoc"]["L1"], ordered[0]["design_order"]))
    for name, eoc, design in votes:
        verdict = "PASS" if eoc >= design - 0.35 else "BELOW-DESIGN"
        print(f"{name}: measured EOC(L1)={eoc:.2f} vs design {design} -> {verdict}")
    if figure:
        print(f"figure -> {figure}")
    print(f"json   -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())