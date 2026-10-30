"""Generate convergence plots from existing JSON artifacts."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    os.makedirs("artifacts", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Poisson convergence ----
    ax = axes[0]
    for deg in [1, 2, 3]:
        data = load_json(f"artifacts/converge_poisson_p{deg}.json")
        hs = [e["h"] for e in data["entries"]]
        errs = [e["error_l2"] for e in data["entries"]]
        obs = data["entries"][-1]["order_l2"]
        ax.loglog(hs, errs, "o-", label=f"SIPG p={deg} (obs={obs:.2f})")
        # Reference slope
        h_ref = [hs[0], hs[-1]]
        e_ref = [errs[0], errs[0] * (hs[-1] / hs[0]) ** (deg + 1)]
        ax.loglog(h_ref, e_ref, "--", alpha=0.4, label=f"O(h^{deg+1})")
    ax.set_xlabel("Mesh size h")
    ax.set_ylabel("L2 Error")
    ax.set_title("SIPG Poisson Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Advection convergence ----
    ax = axes[1]
    for lim in ["none", "tvb", "weno"]:
        data = load_json(f"artifacts/converge_advection_{lim}_p2.json")
        hs = [e["h"] for e in data["entries"]]
        errs = [e["error_l2"] for e in data["entries"]]
        obs = data["entries"][-1]["order_l2"]
        ax.loglog(hs, errs, "o-", label=f"{lim.upper()} p=2 (obs={obs:.2f})")
    h_ref = [hs[0], hs[-1]]
    e_ref = [errs[0], errs[0] * (hs[-1] / hs[0]) ** 3]
    ax.loglog(h_ref, e_ref, "--", alpha=0.4, label="O(h³)")
    ax.set_xlabel("Mesh size h")
    ax.set_ylabel("L2 Error")
    ax.set_title("Advection DG Convergence (p=2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("artifacts/convergence_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved artifacts/convergence_plots.png")


if __name__ == "__main__":
    main()
