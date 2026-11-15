"""Componentwise vs characteristic-variable TVB limiting on the anchor's 2-D cases.

Two measurements, both anchored in ``REPRO-DG-CKS1998`` (Cockburn & Shu, JCP 141
(1998) 199-224):

``sod``
    The 1-D Sod shock tube extruded in ``y`` (``y``-invariant data, ``nx x 2``
    cells, outflow in ``x``).  The reference is the **exact Riemann solution**
    averaged over the same cells (``benchmarks.sod_exact.cell_averages``), so
    the error is a real discretisation error, not a self-comparison.  Reported:
    ``L1(rho)``, the overshoot above the left state, the undershoot below the
    right state, and the number of cells in which the density is non-monotone
    in the plateau -- i.e. the quantities the limiter is supposed to control.

``smooth``
    The paper's own smooth 2-D Euler solution ``rho = 1 + 0.2 sin(pi(x+y-2t))``,
    ``u = v = 1``, ``p = 1`` on ``[0,2]^2`` at two TVB constants.  Reported:
    ``L1(rho)`` and the adjacent-mesh EOC, so it is visible whether the
    characteristic transform costs the design order.

The question the benchmark answers is deliberately narrow and falsifiable:
*does limiting in characteristic variables change the error at all, and in
which direction?*  The null hypothesis is "no difference beyond the
transformation round-off", which is what the componentwise limiter would
predict if the coupling between the conserved variables did not matter.

Usage
-----
    OMP_NUM_THREADS=1 python -m benchmarks.char_limiter_2d          # full, ~5 min
    OMP_NUM_THREADS=1 python -m benchmarks.char_limiter_2d --quick  # smaller ladders

Writes ``validation/char_limiter_2d/results.json`` and
``docs/figures/cks1998_char_limiter.png``.
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
OUT_DIR = REPO / "validation" / "char_limiter_2d"
FIG = REPO / "docs" / "figures" / "cks1998_char_limiter.png"

SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)
DIAPHRAGM = 0.5
SOD_TIME = 0.2


# ---------------------------------------------------------------------------
# Case set-ups
# ---------------------------------------------------------------------------
def sod_initial(x, y):
    rho = torch.where(
        x < DIAPHRAGM,
        torch.as_tensor(SOD_LEFT[0], dtype=x.dtype),
        torch.as_tensor(SOD_RIGHT[0], dtype=x.dtype),
    )
    pressure = torch.where(
        x < DIAPHRAGM,
        torch.as_tensor(SOD_LEFT[2], dtype=x.dtype),
        torch.as_tensor(SOD_RIGHT[2], dtype=x.dtype),
    )
    zero = torch.zeros_like(rho)
    return rho, zero.clone(), zero.clone(), pressure


def smooth_initial(x, y):
    rho = 1.0 + 0.2 * torch.sin(math.pi * (x + y))
    ones = torch.ones_like(rho)
    return rho, ones.clone(), ones.clone(), ones


def smooth_exact(x, y, time):
    rho = 1.0 + 0.2 * torch.sin(math.pi * (x + y - 2.0 * time))
    return rho


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def run(case, *, nx, ny, degree, limiter, tvb_constant, final_time, cfl,
        periodic, domain):
    started = time.time()
    x, y, coeff, metadata = dg2.solve_euler_dg_2d(
        sod_initial if case == "sod" else smooth_initial,
        ncells=(nx, ny),
        degree=degree,
        final_time=final_time,
        cfl=cfl,
        periodic=periodic,
        domain=domain,
        limiter=limiter,
        tvb_constant=tvb_constant,
    )
    fields = dg2.cell_fields(coeff)
    return {
        "x": x,
        "y": y,
        "coeff": coeff,
        "fields": fields,
        "metadata": metadata,
        "wall_seconds": time.time() - started,
    }


def sod_metrics(result, nx):
    from benchmarks.sod_exact import cell_averages

    centers, exact = cell_averages(nx, SOD_TIME)
    density = result["fields"]["density"][:, 0].cpu().numpy()
    dx = 1.0 / nx
    l1 = float(np.sum(np.abs(density - exact["density"])) * dx)
    reference = float(np.sum(np.abs(exact["density"])) * dx)
    # Overshoot/undershoot are measured against the two admissible bounds of the
    # exact solution, not against the run's own extrema.
    overshoot = float(max(0.0, density.max() - SOD_LEFT[0]))
    undershoot = float(max(0.0, SOD_RIGHT[0] - density.min()))
    # Non-monotonicity inside the plateau behind the shock: count sign changes of
    # the first difference where the exact solution is constant to 1e-12.
    plateau = np.abs(exact["density"] - np.round(exact["density"], 4)) < 1e-12
    slopes = np.diff(density)[:-1]
    mask = plateau[1:-1] & plateau[:-2]
    signs = np.sign(slopes[mask])
    wiggles = int(np.sum(signs[1:] * signs[:-1] < 0)) if signs.size > 1 else 0
    return {
        "nx": nx,
        "L1_density": l1,
        "L1_density_rel": l1 / reference,
        "overshoot_above_left": overshoot,
        "undershoot_below_right": undershoot,
        "plateau_wiggles": wiggles,
        "max_abs_error": float(np.max(np.abs(density - exact["density"]))),
    }


def smooth_metrics(result, nx, ny, final_time, domain):
    from benchmarks.cks1998_smooth_2d import error_norms

    norms = error_norms(
        result["coeff"], result["x"], result["y"],
        result["metadata"]["degree"], final_time=final_time, domain=domain,
    )
    return {
        "nx": nx,
        "L1_density": norms["density"]["L1"],
        "L1_density_rel": norms["density"]["L1_rel"],
        "Linf_density": norms["density"]["Linf"],
    }


def eoc(coarse, fine, h_coarse, h_fine):
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return float(math.log(coarse / fine) / math.log(h_coarse / h_fine))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller ladders")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    limiters = (None, "tvb", "char_tvb")
    sod_nx = [50, 100] if args.quick else [100, 200]
    smooth_meshes = [10, 14] if args.quick else [12, 16]
    smooth_time = 0.1
    smooth_domain = ((0.0, 2.0), (0.0, 2.0))

    results = {
        "anchor": "REPRO-DG-CKS1998",
        "case_domain": "TensorDG 2-D Euler DG (HLLC, SSP-RK3), limiter in "
                        "conserved vs local-characteristic variables",
        "sod": {"final_time": SOD_TIME, "left": SOD_LEFT, "right": SOD_RIGHT,
                "reference": "exact Riemann solution cell-averaged "
                             "(benchmarks.sod_exact.cell_averages)",
                "runs": {}},
        "smooth": {"final_time": smooth_time, "domain": smooth_domain,
                   "reference": "analytic rho = 1 + 0.2 sin(pi (x+y-2t))",
                   "runs": {}},
    }

    # --- Sod, extruded in y -------------------------------------------------
    for degree in (1, 2):
        for tvb_constant in (1.0,):
            rows = {}
            for nx in sod_nx:
                for limiter in limiters:
                    key = "none" if limiter is None else limiter
                    result = run(
                        "sod", nx=nx, ny=2, degree=degree, limiter=limiter,
                        tvb_constant=tvb_constant, final_time=SOD_TIME, cfl=0.2,
                        periodic=(False, True), domain=((0.0, 1.0), (0.0, 0.1)),
                    )
                    row = sod_metrics(result, nx)
                    row["wall_seconds"] = result["wall_seconds"]
                    row["steps"] = result["metadata"]["steps"]
                    rows.setdefault(key, []).append(row)
                    print(f"sod  P{degree} nx={nx:<4d} {key:<9s} "
                          f"L1={row['L1_density_rel']:.4e} "
                          f"over={row['overshoot_above_left']:.2e} "
                          f"under={row['undershoot_below_right']:.2e} "
                          f"wiggles={row['plateau_wiggles']} ({row['wall_seconds']:.1f}s)")
            results["sod"]["runs"][f"p{degree}_M{tvb_constant:g}"] = rows

    # --- The paper's smooth 2-D Euler accuracy case -------------------------
    for degree in (1, 2):
        for tvb_constant in (1.0, 3.0):
            rows = {}
            for nx in smooth_meshes:
                for limiter in ("tvb", "char_tvb"):
                    result = run(
                        "smooth", nx=nx, ny=nx, degree=degree, limiter=limiter,
                        tvb_constant=tvb_constant, final_time=smooth_time,
                        cfl=0.2, periodic=(True, True), domain=smooth_domain,
                    )
                    row = smooth_metrics(result, nx, nx, smooth_time, smooth_domain)
                    row["wall_seconds"] = result["wall_seconds"]
                    rows.setdefault(limiter, []).append(row)
                    print(f"smth P{degree} M={tvb_constant:<3g} n={nx:<3d} {limiter:<9s} "
                          f"L1={row['L1_density']:.4e} Linf={row['Linf_density']:.4e} "
                          f"({row['wall_seconds']:.1f}s)")
            for limiter, series in rows.items():
                series.sort(key=lambda r: r["nx"])
                for coarse, fine in zip(series, series[1:]):
                    coarse["eoc_L1"] = eoc(
                        coarse["L1_density"], fine["L1_density"],
                        2.0 / coarse["nx"], 2.0 / fine["nx"],
                    )
                series[-1]["eoc_L1"] = float("nan")
            results["smooth"]["runs"][f"p{degree}_M{tvb_constant:g}"] = rows

    path = OUT_DIR / "results.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"wrote {path}")
    plot(results)
    return 0


def plot(results) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    key = sorted(results["sod"]["runs"])[0]
    series = results["sod"]["runs"][key]
    labels = {"none": "no limiter", "tvb": "componentwise TVB M=1",
              "char_tvb": "characteristic TVB M=1"}
    for name, rows in series.items():
        row = rows[-1]
        ax.plot(row["nx"], 100.0 * row["L1_density_rel"], "o-", label=labels.get(name, name))
    ax.set_xlabel("cells in x")
    ax.set_ylabel(r"$L_1(\rho)$ [% of $\int|\rho_{\rm exact}|$]")
    ax.set_title(f"Sod (extruded), {key}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for cfg, rows in sorted(results["smooth"]["runs"].items()):
        for limiter, series in sorted(rows.items()):
            series = sorted(series, key=lambda r: r["nx"])
            ax.loglog(2.0 / np.array([r["nx"] for r in series]),
                      [r["L1_density"] for r in series], "o-",
                      label=f"{cfg} {limiter}")
    ax.set_xlabel("h = 2/n")
    ax.set_ylabel(r"$L_1(\rho)$")
    ax.set_title("paper smooth case: limiter arms")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())