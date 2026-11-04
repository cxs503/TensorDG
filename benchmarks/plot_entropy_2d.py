"""Figure for the two-dimensional discrete entropy budget (m3430).

Reads ``validation/entropy_2d/summary.json`` (written by
:mod:`benchmarks.entropy_budget_2d`) and produces

(a) ``|Delta int int rho s dA|`` versus mesh size for the exact 2-D smooth
    advection wave, for the unlilmited scheme and the two shock limiters, with an
    ``h**2`` reference slope;
(b) the extruded-Sod entropy production against the analytic Rankine-Hugoniot
    budget (with the 3 % acceptance band) for the same three settings;
(c) the pointwise entropy density defect ``rho s - rho s(0)`` of the smooth wave
    at the final time, unlimited versus WENO-limited -- the spatial signature of
    the entropy destruction reported for the WENO limiter.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.entropy_budget_2d import (
    entropy_density_field_2d,
    smooth_initial,
    total_entropy_2d,
)
from tensordg.euler_dg_2d import solve_euler_dg_2d

LABELS = {"none": "no limiter", "tvb": "TVB (M=1)", "weno": "WENO"}


def _panel_a(axis, smooth):
    for limiter in ("none", "tvb", "weno"):
        rows = [entry for entry in smooth if entry["limiter"] == limiter]
        rows.sort(key=lambda entry: entry["ncells"])
        h = [1.0 / entry["ncells"] for entry in rows]
        drift = [max(abs(entry["entropy_drift"]), 1e-16) for entry in rows]
        sign = [-1.0 if entry["entropy_drift"] < 0 else 1.0 for entry in rows]
        axis.loglog(h, drift, "o-" if limiter != "weno" else "s--",
                    label=LABELS[limiter])
        for hi, di, si in zip(h, drift, sign):
            if si < 0:
                axis.annotate("entropy\ndestruction", (hi, di), textcoords="offset points",
                              xytext=(-6, -26), fontsize=7, color="tab:red",
                              ha="right")
    h = np.array([1.0 / entry["ncells"] for entry in smooth])
    ref = 2.0e-4 * (h / h.min()) ** 2
    axis.loglog(h[h > 0], ref[h > 0], ":", color="0.6", label="$h^2$ reference")
    axis.set_xlabel("mesh size $h$")
    axis.set_ylabel(r"$|\Delta\int\!\!\int \rho s\,dA|$")
    axis.set_title("(a) smooth 2-D advection: entropy invariant")
    axis.legend(fontsize=8)
    axis.grid(True, which="both", alpha=0.3)


def _panel_b(axis, sod, budget):
    meshes = sorted({entry["ncells"] for entry in sod})
    width = 0.26
    for index, limiter in enumerate(("none", "tvb", "weno")):
        rows = {entry["ncells"]: entry for entry in sod if entry["limiter"] == limiter}
        values = [rows[m]["entropy_production"] for m in meshes]
        positions = np.arange(len(meshes)) + (index - 1) * width
        axis.bar(positions, values, width=width, label=LABELS[limiter])
    axis.axhline(budget["production_total"], color="k", lw=1.2,
                 label="analytic RH budget")
    axis.axhspan(0.97 * budget["production_total"], 1.03 * budget["production_total"],
                 color="k", alpha=0.12, label="3 % band")
    axis.set_xticks(np.arange(len(meshes)))
    axis.set_xticklabels([f"{m}x{m}" for m in meshes])
    axis.set_ylabel(r"$\Delta\int\!\!\int \rho s\,dA$")
    axis.set_title("(b) extruded Sod: entropy production")
    axis.legend(fontsize=7)
    axis.grid(True, axis="y", alpha=0.3)


def _panel_c(axis, ncells=32, final_time=0.05, degree=1):
    initial = smooth_initial()
    common = dict(ncells=ncells, degree=degree, final_time=final_time, cfl=0.4)
    _, _, coeff_0, _ = solve_euler_dg_2d(initial, final_time=0.0,
                                        **{k: v for k, v in common.items()
                                           if k != "final_time"})
    dx = dy = 1.0 / ncells
    reference = entropy_density_field_2d(coeff_0, dx, dy, degree)
    panels = []
    for limiter in (None, "weno"):
        _, _, coeff_t, _ = solve_euler_dg_2d(initial, limiter=limiter, **common)
        panels.append(entropy_density_field_2d(coeff_t, dx, dy, degree) - reference)
    unlimited, limited = panels
    induced = limited - unlimited
    limit = max(float(np.abs(induced).max()), 1e-16)
    # collapse the (nx, ny, q, q) quadrature grid onto cell-centred samples
    coarse = induced[:, :, induced.shape[2] // 2, induced.shape[3] // 2]
    print(f"(c) unlimited: max|rho s(t) - rho s(0)| = {np.abs(unlimited).max():.3e}")
    print(f"(c) weno:      max|rho s(t) - rho s(0)| = {np.abs(limited).max():.3e}")
    print(f"(c) limiter-induced defect: min = {induced.min():+.3e}, "
          f"max = {induced.max():+.3e}")
    image = axis.imshow(coarse.T, origin="lower", cmap="coolwarm",
                        vmin=-limit, vmax=limit, extent=(0, 1, 0, 1))
    axis.set_title(r"(c) WENO-induced $\rho s$ defect at $t=%.2f$" % final_time)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    return image


def make_figure(summary_path="validation/entropy_2d/summary.json",
                output_path="docs/figures/entropy_budget_2d.png"):
    with open(summary_path) as handle:
        summary = json.load(handle)
    torch.set_num_threads(1)
    figure = plt.figure(figsize=(13.0, 4.2))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0])
    _panel_a(figure.add_subplot(grid[0, 0]), summary["cases"]["smooth_advection"])
    _panel_b(figure.add_subplot(grid[0, 1]), summary["cases"]["extruded_sod"],
             summary["analytic_budget"])
    image = _panel_c(figure.add_subplot(grid[0, 2]))
    if image is not None:
        figure.colorbar(image, ax=figure.axes[-1], fraction=0.046, pad=0.02,
                        label=r"$\rho s(t) - \rho s(0)$")
    figure.suptitle("TensorDG: 2-D discrete entropy budget (Euler DG, P1, HLLC, SSP-RK3)")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=150)
    print(f"wrote {output_path}")
    return output_path


if __name__ == "__main__":
    make_figure()