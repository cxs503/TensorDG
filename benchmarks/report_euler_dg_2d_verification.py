"""Collect the report figures/numbers for mission m2400 (budget-bounded run)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from tensordg.euler_2d import vortex_primitive
from benchmarks.sod_exact import cell_averages as sod_cell_averages

OUT = Path("validation/euler_dg_2d")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
report = {}
started = time.time()

# ---- 1. extruded Sod, P1, 100 cells vs exact Riemann --------------------
_, reference = sod_cell_averages(100, 0.2)
x, _, coeff, meta = dg2.solve_euler_dg_2d_sod(ncells=100, degree=1, final_time=0.2, cfl=0.2)
fields = dg2.cell_fields(coeff)
sod = {"steps": meta["steps"], "dt": meta["dt"], "cells": 100, "degree": 1, "errors": {}}
for name in ("density", "pressure"):
    numerical = fields[name][:, 0].detach().cpu().numpy()
    exact = np.asarray(reference[name], dtype=float)
    absolute = float(np.sqrt(np.mean((numerical - exact) ** 2)))
    norm = float(np.sqrt(np.mean(exact ** 2)))
    sod["errors"][name] = {"relative_l2": absolute / norm,
                           "relative_l2_percent": 100.0 * absolute / norm}
    sod[name] = numerical.tolist()
sod["x"] = np.asarray(x).tolist()
sod["exact_density"] = np.asarray(reference["density"], dtype=float).tolist()
sod["exact_pressure"] = np.asarray(reference["pressure"], dtype=float).tolist()
report["extruded_sod"] = sod
print("sod", sod["errors"], f"{time.time()-started:.1f}s", flush=True)

# ---- 2. vortex, P2, two meshes -> EOC + field map ----------------------
nodes, weights, basis, _, _ = dg2.legendre_operators(2, 4)
vortex = {"degree": 2, "final_time": 1.0, "rows": []}
for ncells in (16, 24):
    t0 = time.time()
    xc, yc, coeff, vmeta = dg2.solve_euler_dg_2d_vortex(
        ncells=ncells, degree=2, final_time=1.0, cfl=0.2
    )
    dx = 10.0 / ncells
    xq = xc[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = yc[None, :, None, None] + 0.5 * dx * nodes[None, None, None, :]
    exact = vortex_primitive(xq, yq, 1.0, 5.0, (1.0, 1.0))
    state = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    rho, u, v, p = dg2._primitive(state)
    w = np.polynomial.legendre.leggauss(nodes.numel())[1]
    weight = torch.as_tensor(w, dtype=torch.float64)
    weight = weight[:, None] * weight[None, :]
    row = {"cells": ncells, "h": dx, "steps": vmeta["steps"], "dofs": vmeta["dofs"],
           "wall": round(time.time() - t0, 1), "errors": {}}
    for name, numerical, ref in (("density", rho, exact[0]), ("pressure", p, exact[3])):
        num = float(torch.sqrt(((numerical - ref) ** 2 * weight).sum()
                               / (ref ** 2 * weight).sum()))
        row["errors"][name] = num
    vortex["rows"].append(row)
    print("vortex", ncells, row["errors"], row["wall"], flush=True)
    if ncells == 24:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), constrained_layout=True)
        import matplotlib
        mesh = axes[0].pcolormesh(
            np.asarray(xc), np.asarray(yc), coeff[..., 0, 0, 0].numpy().T,
            shading="nearest", cmap="viridis",
        )
        axes[0].set_title(r"DG P2 numerical $\bar\rho$, $t=1$")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
        fig.colorbar(mesh, ax=axes[0], shrink=0.9)
        radius = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(radius, radius)
        exact_field = vortex_primitive(torch.as_tensor(X), torch.as_tensor(Y), 1.0, 5.0, (1.0, 1.0))[0].numpy()
        axes[1].contourf(X, Y, exact_field, levels=20, cmap="viridis")
        axes[1].set_title(r"exact $\rho$, $t=1$")
        axes[1].set_xlabel("x")
        fig.savefig(FIG / "vortex_field_p2_24.png", dpi=95)
        plt.close(fig)

h = [row["h"] for row in vortex["rows"]]
for field in ("density", "pressure"):
    e = [row["errors"][field] for row in vortex["rows"]]
    vortex.setdefault("eoc", {})[field] = float(math.log(e[0] / e[1]) / math.log(h[0] / h[1]))
report["vortex"] = vortex

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
axes[0].loglog(h, [row["errors"]["density"] for row in vortex["rows"]], "o-", label=r"$\rho$ error")
axes[0].loglog(h, [row["errors"]["pressure"] for row in vortex["rows"]], "s-", label="p error")
href = np.asarray(h, dtype=float)
axes[0].loglog(href, vortex["rows"][-1]["errors"]["density"] * (href / href[-1]) ** 3,
               "k--", label=r"slope 3 (design order)")
axes[0].invert_xaxis()
axes[0].set_xlabel("h"); axes[0].set_ylabel("relative L2 error")
axes[0].set_title("isentropic vortex, DG P2")
axes[0].legend(fontsize=8); axes[0].grid(True, which="both", alpha=0.3)
sod_x = np.asarray(sod["x"])
axes[1].plot(sod_x, sod["exact_density"], "k-", lw=1.6, label="exact Riemann")
axes[1].plot(sod_x, sod["density"], "r.", ms=2.5, label="DG P1 extruded 2-D")
axes[1].set_xlabel("x"); axes[1].set_ylabel(r"$\rho$")
axes[1].set_title("Sod tube extruded to 2-D, t=0.2")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
fig.savefig(FIG / "verification_summary.png", dpi=95)
plt.close(fig)

report["environment"] = {"torch": torch.__version__, "numpy": np.__version__,
                         "wall_seconds": round(time.time() - started, 1)}
(OUT / "evidence.json").write_text(json.dumps(report, indent=2))
print("wrote", OUT / "evidence.json", report["environment"], flush=True)