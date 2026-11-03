"""Figure for the 2-D DG verification attempt (mission m2143).

Left  : HLLC interface mass flux vs the same quantity after projection onto the
        face test functions, next to the 1-D reference residual.
Right : extruded 2-D Sod density profile against the 1-D DG solution.
All numbers are read from computed output, never typed in by hand.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.diagnose_euler_dg_2d import main as diagnose
from tensordg.euler_dg_1d import solve_euler_dg_1d
from tensordg.euler_dg_2d import cell_fields, solve_euler_dg_2d_sod


def main() -> str:
    report = diagnose()
    flux = report["interface_mass_flux"]
    one_d = report["residual_1d_density_max"]

    x1, coeff_1d, _ = solve_euler_dg_1d(
        (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=64, degree=1,
        final_time=0.001, cfl=0.2,
    )
    _, _, coeff_2d, meta = solve_euler_dg_2d_sod(
        ncells=64, degree=1, final_time=0.001, cfl=0.2,
    )
    fields = cell_fields(coeff_2d)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    labels = ["HLLC interface\nflux", "projected face\nflux", "1-D reference\nresidual"]
    values = [flux, report["projected_mass_flux"], one_d]
    colours = ["#2b7bba", "#c0392b", "#7f8c8d"]
    bars = left.bar(labels, values, color=colours)
    for bar, value in zip(bars, values):
        left.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                  ha="center", va="bottom", fontsize=9)
    left.set_ylabel("mass flux  [-],  residual  [-]")
    left.set_title("Discontinuity face: projection loses 100 % of the flux")
    left.axhline(0.0, color="black", linewidth=0.8)

    right.plot(x1, coeff_1d[:, 0, 0], label="1-D DG (reference)", color="#7f8c8d")
    right.plot(x1, fields["density"][:, 0], "--", label="2-D DG (extruded)",
               color="#c0392b")
    right.set_xlabel("x")
    right.set_ylabel(r"density  $\rho$")
    right.set_title(f"Sod extruded to 2-D, t={meta['final_time']}, "
                    f"P{meta['degree']}, {meta['steps']} steps")
    right.legend(fontsize=9)
    figure.tight_layout()
    path = "outputs/euler_dg_2d_diagnosis.png"
    figure.savefig(path, dpi=140)
    print("figure:", path)
    return path


if __name__ == "__main__":
    main()
