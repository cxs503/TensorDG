#!/usr/bin/env python3
"""m8749 report figures: Thacker hydrostatic well-balanced validation.

Reads validation/swe_thacker/hydrostatic_wb_results.json and emits
scratch/m8749_fig1.png (convergence) and scratch/m8749_fig2.png (basin state).
All numbers are read programmatically from the JSON (zero hand-copying).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch"
data = json.load(open(ROOT / "validation/swe_thacker" / "hydrostatic_wb_results.json"))

# ---- Fig 1: eta-error vs N for P1/P2 (log-log) -------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
colors = {1: "tab:blue", 2: "tab:red"}
markers = {1: "o", 2: "s"}
for run in data["convergence_study"]["runs"]:
    deg = run["degree"]
    ns = [r["ncells"] for r in run["results"]]
    errs = [r["eta_error_max"] for r in run["results"]]
    ax.loglog(ns, errs, marker=markers[deg], color=colors[deg],
              label=f"P{deg}  $\\eta$ error (max)")
    us = [r["u_max"] for r in run["results"]]
    ax.loglog(ns, us, marker=markers[deg], color=colors[deg], ls="--",
              label=f"P{deg}  $u$ max")
ax.axhline(1e-8, color="gray", lw=0.8, ls=":",
           label="pass criterion 1e-8 m")
ax.set_xlabel("cells N")
ax.set_ylabel("max error (m, m/s)")
ax.set_title("Thacker parabolic basin, lake at rest, $t_f$ = "
             f"{data['convergence_study']['final_time']} s\n"
             "well_balanced=True, nodal LGL bed, "
             "$z_b$-center = mode-0 (cell-average bed)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "m8749_fig1.png", dpi=140)
plt.close(fig)

# ---- Fig 2: basin geometry + lake-at-rest state -------------------------
H0, L, eta0 = (data["single_run"][k] for k in ("H0", "L", "eta0"))
x = np.linspace(-L, L, 400)
zb = -H0 + (H0 / L**2) * x**2
h = eta0 - zb
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.fill_between(x, zb, eta0, color="tab:blue", alpha=0.35, label="water")
ax.plot(x, zb, "k-", lw=1.5, label="$z_b(x) = -H_0 + (H_0/L^2)x^2$")
ax.plot(x, np.full_like(x, eta0), "b--", lw=1.2,
        label=f"$\\eta = h + z_b = {eta0}$ (const)")
ax.set_xlabel("x")
ax.set_ylabel("z (m)")
ax.set_title(f"Thacker basin: $H_0$={H0}, $L$={L}, $\\eta_0$={eta0}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "m8749_fig2.png", dpi=140)
plt.close(fig)

# ---- console summary for the report --------------------------------------
s = data["single_run"]
print("single_run:", {k: s[k] for k in
      ("ncells", "degree", "final_time", "eta_error_max", "u_max",
       "h_error_max", "steps", "wall_time_s")})
for run in data["convergence_study"]["runs"]:
    for r in run["results"]:
        print(f"P{run['degree']} N={r['ncells']}: eta={r['eta_error_max']:.3e} "
              f"u={r['u_max']:.3e} steps={r['steps']} t={r['wall_time_s']}s")
print("pass:", data["summary"]["pass"])
