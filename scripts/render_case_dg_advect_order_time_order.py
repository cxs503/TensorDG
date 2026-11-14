"""Render the temporal-order ladder figures for CASE-DG-ADVECT-ORDER / M5_dt_order.

Two panels, both driven by ``artifacts/case_dg_advect_order_time_order.json``
(no hand-copied numbers):

* left  -- error vs number of time steps against the exact semi-discrete
  solution ``expm(L T) u0`` for every Runge-Kutta tableau, with ``dt^q``
  reference slopes: the measured slopes must equal the tableau order.
* right -- the round-2 trap: refining ``dt`` at *fixed* ``N`` in the shipped
  time loop plateaus on the spatial error, so it can never expose ``q``.

Usage: ``PYTHONPATH=. python scripts/render_case_dg_advect_order_time_order.py``
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path("artifacts/case_dg_advect_order_time_order.json")
OUT = Path("docs/figures/case_dg_advect_order_time_order.png")
COLOURS = {"euler": "#d62728", "rk2": "#ff7f0e", "rk3": "#2ca02c",
           "rk4": "#1f77b4", "rk5": "#6a3d9a"}


def main() -> None:
    data = json.loads(ART.read_text())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    panel = [r for r in data["temporal"] if r["ncells"] == 16 and r["degree"] == 2]
    for row in panel:
        name = row["tableau"]
        steps = [lvl["nsteps"] for lvl in row["levels"]]
        errs = [lvl["error"] for lvl in row["levels"]]
        ax.loglog(steps, errs, "o-", color=COLOURS.get(name, "k"), lw=1.4, ms=4,
                  label=f"{name}: measured {row['observed_order']:.3f} "
                        f"(nominal {row['nominal_order']})")
    ref_steps = np.array([panel[0]["levels"][0]["nsteps"], panel[0]["levels"][-1]["nsteps"]])
    base = panel[0]["levels"][0]["error"]
    for q, style in ((1, ":"), (2, "--"), (3, "-."), (4, "--"), (5, ":")):
        ax.loglog(ref_steps, base * (ref_steps / ref_steps[0]) ** (-q), style,
                  color="grey", lw=0.9, alpha=0.7)
        ax.annotate(rf"$dt^{{{q}}}$", (ref_steps[1], base * (ref_steps[1] / ref_steps[0]) ** (-q)),
                    textcoords="offset points", xytext=(4, -2), fontsize=8, color="grey")
    ax.set_xlabel("time steps to T = 1")
    ax.set_ylabel(r"$\|u_h - \exp(LT)u_0\|_2$")
    ax.set_title("temporal order vs exact semi-discrete solution (N=16, p=2)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower left")

    floor = data["space_floor"]
    steps = [r["nsteps"] for r in floor]
    errs = [r["error"] for r in floor]
    bx.semilogx(steps, errs, "s-", color="#1f77b4", lw=1.4, ms=5,
                label="shipped loop, fixed N=40, p=2 (space error fixed)")
    bx.axhline(errs[-1], color="grey", ls="--", lw=1.0,
               label=f"plateau (spatial error floor) = {errs[-1]:.3e}")
    bx.set_xlabel("time steps to T = 1")
    bx.set_ylabel(r"$L^2$ error vs analytic sine")
    bx.set_title("why refining dt at fixed N cannot reveal the time order")
    bx.grid(True, which="both", ls=":", alpha=0.4)
    bx.legend(fontsize=8)

    fig.suptitle("CASE-DG-ADVECT-ORDER · M5_dt_order (temporal convergence ladder)", fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()