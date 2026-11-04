"""Figures for the Riemann wave-configuration coverage audit."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.riemann_dg_error import riemann_dg_errors
from benchmarks.riemann_exact import sample
from tools.verify_riemann_coverage import CONFIGS, DIAPHRAGM, DOMAIN


def main(out="docs/figures/riemann_coverage.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))

    # (a) exact density of all five wave patterns in the self-similar variable
    x = np.linspace(DOMAIN[0], DOMAIN[1], 4001)
    for name, config in CONFIGS.items():
        exact = sample(x, config["final_time"], config["left"], config["right"], DIAPHRAGM)
        axes[0, 0].plot(x, exact["density"], lw=1.6,
                        label=f"{name}: {config['pattern'].split(' (')[0]}")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel(r"$\rho$ (exact)")
    axes[0, 0].set_title("(a) five wave patterns at their own final time")
    axes[0, 0].legend(fontsize=7)

    # (b) DG solution vs exact for the newly covered left-shock case
    from tensordg.euler_dg_1d import solve_euler_dg_1d

    config = CONFIGS["revsod"]
    xc, coeff, meta = solve_euler_dg_1d(
        config["left"], config["right"], ncells=60, degree=2,
        final_time=config["final_time"], diaphragm=DIAPHRAGM, domain=DOMAIN,
        cfl=0.1, limiter="tvb",
    )
    xf = np.linspace(DOMAIN[0], DOMAIN[1], 8001)
    exact = sample(xf, config["final_time"], config["left"], config["right"], DIAPHRAGM)
    axes[0, 1].plot(xf, exact["density"], "k-", lw=1.5, label="exact")
    xc_np = xc.detach().cpu().numpy()
    axes[0, 1].plot(xc_np, coeff.detach().cpu().numpy()[:, 0, 0], "o", ms=2.5,
                    label="DG P2 cell average")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel(r"$\rho$")
    axes[0, 1].set_title(f"(b) reversed Sod (left shock), {meta['steps']} steps")
    axes[0, 1].legend(fontsize=8)

    # (c) grid convergence of the newly covered configurations
    with open("runs/riemann_coverage.json", "r", encoding="utf-8") as handle:
        data = json.load(handle)
    for name, entry in data["convergence"].items():
        series = [item for item in entry["series"] if item["status"] == "ok"]
        if len(series) < 2:
            continue
        grids = [item["ncells"] for item in series]
        values = [item["relative_l2_percent"]["density"] for item in series]
        axes[1, 0].loglog(grids, values, "o-", label=name)
    axes[1, 0].set_xlabel("cells")
    axes[1, 0].set_ylabel(r"$L^2$ error in $\rho$ [%]")
    axes[1, 0].set_title("(c) error vs grid (P2, TVB limiter)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, which="both", alpha=0.3)

    # (d) mirror symmetry of the indicator
    probe = data.get("indicator_symmetry", {})
    labels, values = [], []
    for case in ("smooth", "jump"):
        for variant in ("historic", "symmetric"):
            entry = probe.get(case, {}).get(variant)
            if entry is not None:
                labels.append(f"{case}\n{variant}")
                values.append(entry["mirror_mismatches"])
    axes[1, 1].bar(range(len(values)), values, color=["#b2182b"] * 2 + ["#2166ac"] * 2)
    axes[1, 1].set_xticks(range(len(values)))
    axes[1, 1].set_xticklabels(labels, fontsize=7)
    axes[1, 1].set_ylabel("mirror-mismatched cells")
    axes[1, 1].set_title("(d) troubled-cell indicator reflection symmetry")

    figure.tight_layout()
    figure.savefig(out, dpi=110)
    print("wrote", out)
    return out


if __name__ == "__main__":
    main()