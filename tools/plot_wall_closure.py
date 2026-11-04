"""Figures for the recovered Dirichlet wall closure (docs/reports/).

Reads ``validation/wall_closure_probe.json`` (written by
``tools/probe_wall_closure.py``) and renders the two convergence figures:
wall heat-flux error and steady-state residual, legacy vs recovered closure.

Run: PYTHONPATH=. python tools/plot_wall_closure.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "docs/figures"
LABELS = {
    "one_sided|p1|sigma_w=None": "legacy one-sided, $\\sigma_w=\\sigma$",
    "recovered|p1|sigma_w=None": "recovered, $\\sigma_w=\\sigma$",
    "recovered|p1|sigma_w=1.0": "recovered, $\\sigma_w=1$",
    "recovered|p1|sigma_w=0.0": "recovered, $\\sigma_w=0$",
}


def main():
    data = json.load(open("validation/wall_closure_probe.json"))
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for key, block in data["closures"].items():
        ny = sorted(int(n) for n in block["wall_flux_error"])
        h = [1.0 / n for n in ny]
        flux = [block["wall_flux_error"][str(n)] for n in ny]
        resid = [block["residual"][str(n)] for n in ny]
        style = dict(marker="o", label=LABELS.get(key, key))
        axes[0].loglog(h, flux, **style)
        axes[1].loglog(h, resid, **style)

    # reference slopes anchored on the finest recovered point
    h_ref = [1.0 / 32, 1.0 / 8]
    axes[0].loglog(h_ref, [7.28e-6, 7.28e-6 * 16.0], "k--", lw=1,
                   label="slope 1 (first order)")
    axes[0].loglog(h_ref, [7.28e-6, 7.28e-6 * 256.0], "k:", lw=1,
                   label="slope 2 (design order)")

    axes[0].set_xlabel("cell size $h = 1/n_y$")
    axes[0].set_ylabel("|q_wall - q_analytic|")
    axes[0].set_title("Steady Couette wall heat flux error")
    axes[1].set_xlabel("cell size $h = 1/n_y$")
    axes[1].set_ylabel("max |DG residual| at the exact state")
    axes[1].set_title("Strong-form residual of the analytic steady state")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ns_wall_closure_convergence.png"), dpi=150)
    print("wrote", os.path.join(OUT, "ns_wall_closure_convergence.png"))
    print(json.dumps({k: v["wall_flux_order"] for k, v in data["closures"].items()},
                     indent=2))


if __name__ == "__main__":
    main()