#!/usr/bin/env python3
"""Generate fig_amplification.png for CASE-DG-EULER-VORTEX-ERR.

A(t) = e(t)/e(0) semi-log plot showing error amplification factor.
Green band A in [1,2] indicates "error conservation" regime.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("case_euler_vortex_errconserve")

def main():
    baseline = json.loads((OUT / "projection_baseline.json").read_text())

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    colors = {"dg_d2_n32": "steelblue", "p0_n64": "coral",
              "p0_n128": "seagreen", "p0_n256": "purple"}
    markers = {"dg_d2_n32": "o", "p0_n64": "s", "p0_n128": "^", "p0_n256": "D"}

    for b in baseline["baselines"]:
        key = b["task_key"]
        amp = b.get("amplification", [])
        if not amp:
            continue
        times = [a["final_time"] for a in amp]
        values = [a["amplification"] for a in amp]
        color = colors.get(key, "gray")
        marker = markers.get(key, "o")
        ax.semilogy(times, values, f"{marker}-", color=color,
                    label=f"{key} (A(10)={values[-1]:.2f})")

    # Green band for "error conservation" regime
    ax.fill_between([0.5, 12], 1, 2, alpha=0.15, color="green",
                    label="A in [1,2]: error conservation regime")
    ax.axhline(1, color="green", ls="-", alpha=0.4)
    ax.axhline(2, color="green", ls="--", alpha=0.4)

    # Red line at A=5 (growth limit)
    ax.axhline(5, color="crimson", ls=":", alpha=0.6, label="growth limit A=5")

    ax.set_xlabel("dimensionless time t  (period = 10)")
    ax.set_ylabel("amplification factor A(t) = e(t) / e(0)")
    ax.set_title("Error amplification: A(t) = e(t)/e(0)\n"
                 "DG: A(10) ~ 2.8 (conservation); P0: A(10) >> 1 (dissipative accumulation)")
    ax.set_xlim(0.5, 11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_amplification.png", dpi=150)
    plt.close(fig)
    print(f"-> {OUT / 'fig_amplification.png'}")

if __name__ == "__main__":
    main()
