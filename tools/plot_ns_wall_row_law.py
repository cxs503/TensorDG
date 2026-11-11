"""Plot the strong-form wall-row residual law from
``validation/ns_heat_conduction/wall_row_law.json`` (numbers are read from the
artifact, never re-typed).

Two panels: (a) residual vs mesh with the ``h**(p-1)`` reference slope, which is
the rate the two ``xfail`` pins of ``tests/test_ns_heat_conduction.py`` fail to
meet; (b) the same residuals scaled by ``h**(p-1)`` -- bounded across an 8x
refinement, i.e. no ``h**-1`` blow-up.

    PYTHONPATH=. python tools/plot_ns_wall_row_law.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAW = Path("validation/ns_heat_conduction/wall_row_law.json")
FLUX = Path("validation/ns_heat_conduction/wall_flux_order.json")
OUT = Path("docs/figures/ns_wall_row_law.png")


def main():
    law = json.loads(LAW.read_text())
    table = law["table"]
    flux = json.loads(FLUX.read_text())["orders"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    colors = {"p1_taylor": "C0", "p1_face": "C0", "p2_taylor": "C1", "p2_face": "C1"}
    styles = {"taylor": "-o", "face": "--s"}
    for key, entry in table.items():
        degree = int(key[1])
        gradient = key.split("_")[1]
        ny = [c["ny"] for c in entry["cases"]]
        res = [c["cold_wall_row"] for c in entry["cases"]]
        order = entry["cold_wall_order"]
        ax.loglog(ny, res, styles[gradient], color=colors[key], ms=4,
                  label=f"P{degree} {gradient}: order {order:+.2f}")
    ref_x = np.array([16.0, 128.0])
    ax.loglog(ref_x, 0.9 * (ref_x / 16.0) ** 0.0, "k:", lw=1.0, label="h^0 (P1 reference)")
    ax.loglog(ref_x, 0.08 * (ref_x / 16.0) ** -1.0, "k-.", lw=1.0, label="h^1 (P2 reference)")
    pin_a = table["p1_taylor"]["cases"][0]["cold_wall_row"] * 0.25
    ax.axhline(pin_a, color="crimson", lw=1.0, ls=(0, (1, 2)),
               label="pin A target at ny=128 (0.25 x ny=16)")
    ax.set_xlabel("cells in y (2 x ny mesh)")
    ax.set_ylabel("cold-wall energy residual  max|R_E|")
    ax.set_title("strong-form wall row decays at p-1, not stalled")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    for key, entry in table.items():
        degree = int(key[1])
        gradient = key.split("_")[1]
        ny = np.array([c["ny"] for c in entry["cases"]], dtype=float)
        res = np.array([c["cold_wall_row"] for c in entry["cases"]])
        ax.semilogx(ny, res / (1.0 / ny) ** (degree - 1.0), styles[gradient],
                    color=colors[key], ms=4,
                    label=f"P{degree} {gradient}")
    ax.set_xlabel("cells in y (2 x ny mesh)")
    ax.set_ylabel("residual / h^(p-1)  (bounded = O(h^(p-1)))")
    ax.set_title("scaled residual is mesh-independent: no 1/h amplification")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)

    for ax, key in zip(axes, ("p1_taylor", "p2_taylor")):
        ax.text(0.02, 0.03,
                f"wall-flux order: P1 {flux['p1_taylor']['wall_flux_error']:.2f}, "
                f"P2 {flux['p2_taylor']['wall_flux_error']:.2f} (design order)",
                transform=ax.transAxes, fontsize=7, color="0.3")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()