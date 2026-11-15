"""CKS1998 §5.3 accuracy test on the 2-D Euler *system* — isentropic vortex.

Reproduces the smooth 2-D Euler accuracy test from Cockburn & Shu (JCP 141,
1998, "V: multidimensional systems"), which uses the isentropic vortex as the
canonical smooth solution of the 2-D Euler equations to verify the design order
of the RKDG scheme.

What the paper reports
----------------------
The paper tabulates the ``L1`` and ``L-infinity`` error of the density for
``P^k``, ``k = 1, 2, 3``, on a sequence of meshes.  The exact solution is the
isentropic vortex advected by a uniform flow on a periodic box.

Exact solution (isentropic vortex, far-field rho=1, p=1, u=v=1)
---------------------------------------------------------------
    rho(x, y, t) = [1 - (gamma-1) beta^2 / (8 gamma pi^2)
                     exp(1 - ((x-x0-ut)^2 + (y-y0-vt)^2))]^{1/(gamma-1)}
    u(x, y, t)   = 1 - beta/(2 pi) exp(0.5(1-r^2)) (y-y0-vt)
    v(x, y, t)   = 1 + beta/(2 pi) exp(0.5(1-r^2)) (x-x0-ut)
    p(x, y, t)   = rho^gamma

where beta is the vortex strength, (x0, y0) is the initial centre, and
r^2 = (x-x0-ut)^2 + (y-y0-vt)^2.

This is the *same* exact solution used in ``tensordg.euler_2d.vortex_primitive``
and ``benchmarks/design_order_euler_dg_2d.py``, but this benchmark reports the
``L1`` and ``L-infinity`` norms (the paper's format) rather than the relative
L2 norm used elsewhere.

What is reported
----------------
For every ``(degree, mesh)``:

* ``L1`` (integral of ``|rho_h - rho_exact|``), normalised by
  ``integral |rho_exact|`` (``L1_rel``);
* ``L-infinity`` at the tensor Gauss nodes;
* the adjacent-mesh observed order ``EOC``;
* the design order ``p + 1`` the paper proves for smooth solutions.

Usage
-----
    OMP_NUM_THREADS=1 python -m benchmarks.cks1998_vortex_2d            # full sweep
    OMP_NUM_THREADS=1 python -m benchmarks.cks1998_vortex_2d --quick    # ~1 min corner

Results are written to ``validation/cks1998_vortex_2d/order.json`` and the
figure to ``docs/figures/cks1998_vortex_2d_order.png``.
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
from tensordg.euler_1d import GAMMA

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "validation" / "cks1998_vortex_2d"
FIG = REPO / "docs" / "figures" / "cks1998_vortex_2d_order.png"

#: Domain: [-5, 5]^2, the standard vortex box.
DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
FINAL_TIME = 1.0
CFL0 = 0.2
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)

#: ``(degree, meshes)`` ladders — four rungs per degree for a reliable EOC.
LADDERS = {1: [12, 16, 24, 32], 2: [8, 12, 16, 24], 3: [6, 8, 12, 16]}


# ---------------------------------------------------------------------------
# Analytic reference
# ---------------------------------------------------------------------------
def vortex_primitive(x, y, time):
    """The paper's smooth solution: (rho, u, v, p) at ``(x, y, time)``."""
    return dg2.vortex_primitive(x, y, time, STRENGTH, BACKGROUND)


def vortex_conservative(x, y, time):
    """Conservative state (rho, rho*u, rho*v, E) at ``(x, y, time)``."""
    rho, u, v, p = vortex_primitive(x, y, time)
    return dg2.conservative_state(rho, u, v, p)


def initial(x, y):
    return vortex_primitive(x, y, 0.0)


# ---------------------------------------------------------------------------
# Error norms at the tensor Gauss nodes
# ---------------------------------------------------------------------------
def gauss_fields(coeff, degree, quadrature=None):
    """Evaluate the modal solution at the tensor Gauss nodes of every cell."""
    order = quadrature if quadrature is not None else max(degree + 2, 3)
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, order)
    values = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    return values, nodes, weights


def error_norms(coeff, x, y, degree, final_time=FINAL_TIME, domain=DOMAIN):
    """L1 (absolute and relative), L2 relative and L-inf of each field."""
    values, nodes, weights = gauss_fields(coeff, degree)
    (xmin, xmax), (ymin, ymax) = domain
    nx, ny = x.numel(), y.numel()
    hx, hy = (xmax - xmin) / nx, (ymax - ymin) / ny
    X = x[:, None, None, None] + nodes[None, None, :, None] * (hx / 2.0)
    Y = y[None, :, None, None] + nodes[None, None, None, :] * (hy / 2.0)
    exact = vortex_conservative(X, Y, final_time)
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
def run_case(degree, ncells, final_time=FINAL_TIME, cfl=CFL0, device="cpu"):
    nx = ncells if isinstance(ncells, int) else ncells
    ny = ncells if isinstance(ncells, int) else ncells
    start = time.time()
    x, y, coeff, metadata = dg2.solve_euler_dg_2d(
        initial,
        ncells=(nx, ny),
        degree=degree,
        domain=DOMAIN,
        final_time=final_time,
        cfl=cfl,
        periodic=(True, True),
        device=device,
    )
    wall = time.time() - start
    norms = error_norms(coeff, x, y, degree, final_time)
    hx = (DOMAIN[0][1] - DOMAIN[0][0]) / nx
    return {
        "degree": degree,
        "ncells": nx,
        "h": hx,
        "cfl": cfl,
        "final_time": final_time,
        "steps": metadata["steps"],
        "dt": metadata.get("dt"),
        "wall_seconds": round(wall, 2),
        "norms": norms,
        "dofs": metadata["dofs"],
    }


def with_eoc(runs, field="density", norm="L1"):
    """Attach adjacent-mesh observed orders."""
    ordered = sorted(runs, key=lambda r: r["ncells"])
    for coarse, fine in zip(ordered, ordered[1:]):
        ratio = coarse["h"] / fine["h"]
        eoc = math.log(
            coarse["norms"][field][norm] / fine["norms"][field][norm]
        ) / math.log(ratio)
        fine["eoc"] = eoc
    return runs


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def make_figure(payload, fig_path):
    """Log-log error decay figure with design-order reference lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    for idx, key in enumerate(sorted(payload.get("by_degree", {}).keys())):
        runs = payload["by_degree"][key]
        degree = int(key[1:])  # "p1" -> 1
        hs = [r["h"] for r in runs]
        errs = [r["norms"]["density"]["L1_rel"] for r in runs]
        ax.loglog(hs, errs, marker=markers[idx], color=colors[idx],
                  label=f"P{degree} (measured)", linewidth=2, markersize=8)

        # Design order reference line through the finest mesh point
        if len(runs) >= 2:
            h_ref = hs[-1]
            err_ref = errs[-1]
            p_plus_1 = degree + 1
            h_line = np.logspace(np.log10(hs[0]), np.log10(hs[-1]), 50)
            err_line = err_ref * (h_line / h_ref) ** p_plus_1
            ax.loglog(h_line, err_line, "--", color=colors[idx], alpha=0.4,
                      label=f"P{degree} (O(h^{p_plus_1}))")

    ax.set_xlabel("h (cell size)", fontsize=12)
    ax.set_ylabel("L1 relative error (density)", fontsize=12)
    ax.set_title("CKS1998 §5.3 — 2D isentropic vortex: design order verification",
                 fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------
def make_table(payload):
    """Markdown table in the paper's format."""
    lines = ["| config | P^k | mesh | h | steps | rho L1 | rho L1(rel) | rho Linf | EOC L1 | EOC Linf | design p+1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(payload.get("by_degree", {}).keys()):
        runs = payload["by_degree"][key]
        degree = int(key[1:])  # "p1" -> 1
        for r in runs:
            eoc_l1 = f"{r.get('eoc_l1', '--'):.2f}" if isinstance(r.get('eoc_l1'), (int, float)) else "--"
            eoc_linf = f"{r.get('eoc_linf', '--'):.2f}" if isinstance(r.get('eoc_linf'), (int, float)) else "--"
            lines.append(
                f"| {key} | {degree} | {r['ncells']} | {r['h']:.3e} | "
                f"{r['steps']} | {r['norms']['density']['L1']:.3e} | "
                f"{r['norms']['density']['L1_rel']:.3e} | "
                f"{r['norms']['density']['Linf']:.3e} | "
                f"{eoc_l1} | {eoc_linf} | {degree + 1} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degrees", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--meshes", type=int, nargs="*", default=None,
                    help="override the per-degree ladder")
    ap.add_argument("--final-time", type=float, default=FINAL_TIME)
    ap.add_argument("--cfl", type=float, default=CFL0)
    ap.add_argument("--quick", action="store_true",
                    help="cheap corner: two meshes per degree")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "order.json")
    ap.add_argument("--figure", type=Path, default=FIG)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)

    torch.set_num_threads(max(args.threads, 1))

    if args.quick:
        ladders = {1: [16, 32], 2: [12, 16], 3: [8, 12]}
    elif args.meshes:
        ladders = {k: list(args.meshes) for k in args.degrees}
    else:
        ladders = {k: LADDERS[k] for k in args.degrees}

    payload = {
        "anchor": "REPRO-DG-CKS1998",
        "case": "vortex_2d",
        "reference": "analytic: isentropic vortex, beta=5, u_inf=(1,1), rho_inf=1, p_inf=1",
        "domain": [list(DOMAIN[0]), list(DOMAIN[1])],
        "final_time": args.final_time,
        "design_orders": {f"p{k}": k + 1 for k in args.degrees},
        "by_degree": {},
    }

    def flush():
        payload["table_markdown"] = make_table(payload)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))

    for degree in args.degrees:
        ladder = ladders.get(degree, LADDERS.get(degree, [8, 12, 16]))
        runs = []
        for ncells in ladder:
            run = run_case(degree, ncells, args.final_time, args.cfl)
            run["design_order"] = degree + 1
            runs.append(run)
            print(f"[vortex] p{degree} {ncells}x{ncells}: "
                  f"L1rel={run['norms']['density']['L1_rel']:.3e} "
                  f"Linf={run['norms']['density']['Linf']:.3e} "
                  f"steps={run['steps']} wall={run['wall_seconds']}s", flush=True)

        runs = with_eoc(runs, "density", "L1")
        # Also compute L-inf EOC
        ordered = sorted(runs, key=lambda r: r["ncells"])
        for coarse, fine in zip(ordered, ordered[1:]):
            ratio = coarse["h"] / fine["h"]
            eoc_linf = math.log(
                coarse["norms"]["density"]["Linf"] / fine["norms"]["density"]["Linf"]
            ) / math.log(ratio)
            fine["eoc_linf"] = eoc_linf

        for r in runs:
            if "eoc" in r:
                r["eoc_l1"] = r.pop("eoc")

        payload["by_degree"][f"p{degree}"] = runs
        flush()

    # Generate figure
    make_figure(payload, args.figure)
    print(f"\nFigure: {args.figure}")
    print(f"Table:\n{payload['table_markdown']}")

    # Summary verdict
    all_pass = True
    for degree in args.degrees:
        runs = payload["by_degree"][f"p{degree}"]
        if len(runs) >= 2:
            finest = runs[-1]
            eoc_l1 = finest.get("eoc_l1", 0)
            design = degree + 1
            margin = eoc_l1 - (design - 0.5)  # paper tolerance: p+0.5
            ok = eoc_l1 >= design - 0.5
            all_pass = all_pass and ok
            print(f"P{degree}: EOC L1 = {eoc_l1:.3f} (design {design}, "
                  f"tolerance >= {design - 0.5}) -> {'PASS' if ok else 'FAIL'}")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
