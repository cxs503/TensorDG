#!/usr/bin/env python3
"""Figures for the m21003 audit (data: audit/artifacts/poiseuille_convergence.json)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path(__file__).resolve().parent / "artifacts"
d = json.loads((ART / "poiseuille_convergence.json").read_text())
c = d["case"]
Ny, F, nu, H = c["Ny"], c["F"], c["nu"], c["H"]
cycles = [k["steps"] for k in d["checkpoints"]]
e_nc = [100 * k["rel_l2_node_centred"] for k in d["checkpoints"]]
e_bb = [100 * k["rel_l2_halfway_bb"] for k in d["checkpoints"]]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
y = range(Ny)
for k, lab, style in ((0, "delivered config: 2000 steps (11.0%)", "o-"),
                      (-1, "continued to 16000 steps (2.45%)", "s-")):
    ax[0].plot(d["checkpoints"][k]["profile"], y, style, ms=3, lw=1.2, label=lab)
yy = list(y)
ax[0].plot([F / (2 * nu) * v * (H - v) for v in yy], yy, "k--", lw=1.2,
           label="analytic $u(y)=F y (H-y)/(2\\nu)$")
ax[0].plot([F / (2 * nu) * (v - 0.5) * (H - 0.5 - v) for v in yy], yy, "k:", lw=1.2,
           label="analytic, half-way BB walls")
ax[0].set_xlabel("$u_x$ (lattice units)")
ax[0].set_ylabel("y (lattice node)")
ax[0].set_title("Poiseuille profile, BGK, 64×32, $\\tau$=0.8")
ax[0].legend(fontsize=7)
ax[0].grid(alpha=.3)

ax[1].plot(cycles, e_nc, "o-", label="rel-L2 vs node-centred analytic")
ax[1].plot(cycles, e_bb, "s-", label="rel-L2 vs half-way-BB analytic")
ax[1].axhline(3.0, color="r", ls="--", lw=1, label="3% acceptance bar")
ax[1].axvline(2000, color="0.6", ls=":", lw=1)
ax[1].annotate("delivered n_steps=2000\n(11.0% reported)", xy=(2000, e_nc[0]),
               xytext=(3500, 8.5), fontsize=7,
               arrowprops=dict(arrowstyle="->", lw=.8))
ax[1].set_xlabel("LBM steps")
ax[1].set_ylabel("relative L2 error [%]")
ax[1].set_title("Error is dominated by under-convergence, not by the scheme")
ax[1].legend(fontsize=7)
ax[1].grid(alpha=.3)
fig.tight_layout()
out = ART / "fig_poiseuille_convergence.png"
fig.savefig(out, dpi=140)
print(out)
