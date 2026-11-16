"""Figures for the property-database report (matplotlib, no hand-typed data).

Writes ``outputs/figures/props_db_v1_*.png``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.props_solver_integration import (  # noqa: E402
    ittc57_friction,
    plate_buckling,
)
from tensordg.props import load_db  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"


def fig_properties(db) -> Path:
    T = np.linspace(0.0, 30.0, 121)
    nu_fw = np.array([db.get("water", "nu", T=t) for t in T]) * 1e6
    nu_sw = np.array([db.get("sea_water", "nu", t * 1.0) if False else
                      db.get("sea_water", "nu", T=t) for t in T]) * 1e6
    rho_fw = np.array([db.get("water", "rho", T=t) for t in T])
    rho_sw = np.array([db.get("sea_water", "rho", T=t) for t in T])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    ax = axes[0]
    ax.plot(T, nu_fw, label="fresh water (ITTC Table 2)")
    ax.plot(T, nu_sw, "--", label="sea water S=35 (ITTC Table 3)")
    nodes = db._subs["sea_water"].properties["viscosity_kinematic"].points
    ax.plot([p[0] for p in nodes], [p[1] * 1e6 for p in nodes], "o",
            ms=4, mfc="none", label="table nodes")
    ax.set_xlabel("T [degC]")
    ax.set_ylabel("nu [mm^2/s = cSt]")
    ax.set_title("(a) kinematic viscosity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(T, rho_fw, label="fresh water")
    ax.plot(T, rho_sw, "--", label="sea water S=35")
    ax.set_xlabel("T [degC]")
    ax.set_ylabel("rho [kg/m^3]")
    ax.set_title("(b) density (ITTC tables)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    Ta = np.linspace(-20.0, 40.0, 121)
    mu_air = np.array([db.get("air", "mu", T=t) for t in Ta]) * 1e5
    a_air = np.array([db.get("air", "c", T=t) for t in Ta])
    ax.plot(Ta, mu_air, label="air mu [1e-5 Pa*s]")
    ax.plot(Ta, a_air / 10.0, "--", label="air a / 10 [m/s]")
    ax.set_xlabel("T [degC]")
    ax.set_title("(c) air: Sutherland viscosity, sound speed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle("TensorDG property database v1 - water / air (SI base, "
                 "provenance: ITTC 7.5-02-01-03, White, ISO 2533)", fontsize=9)
    fig.tight_layout()
    path = OUT / "props_db_v1_fluids.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_validation(db) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    # (a) DB (rho*nu) vs Vogel correlation
    A, B, C = 2.414e-05, 247.8, 140.0
    T = np.linspace(0.0, 30.0, 61)
    dev = []
    for t in T:
        mu_db = db.get("water", "mu", T=t)
        mu_v = A * 10.0 ** (B / ((t + 273.15) - C))
        dev.append(100.0 * (mu_db - mu_v) / mu_db)
    ax = axes[0]
    ax.plot(T, dev, marker="o", ms=3)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-1.0, 1.0, color="tab:green", alpha=0.12, label="+/-1% band")
    ax.set_xlabel("T [degC]")
    ax.set_ylabel("(mu_DB - mu_Vogel)/mu_DB  [%]")
    ax.set_title("(a) water: table rho*nu vs Vogel correlation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # (b) ITTC-1957 friction line with DB-driven operating points
    Re = np.logspace(5, 10, 200)
    ax = axes[1]
    cf_vec = 0.075 / (np.log10(Re) - 2.0) ** 2   # ITTC-1957 line, vectorised
    ax.loglog(Re, cf_vec)
    nu = db.get("sea_water", "nu", T=15.0)
    for label, L, U, mk in (("model 1:40", 230.0 / 40, 12.0 / math.sqrt(40), "s"),
                            ("ship", 230.0, 12.0, "^")):
        r = U * L / nu
        ax.plot([r], [ittc57_friction(r)], mk, ms=8, label=f"{label} Re={r:.2e}")
    ax.set_xlabel("Re")
    ax.set_ylabel("C_F")
    ax.set_title("(b) ITTC-1957 line, seawater nu(15 degC) from DB")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    # (c) plate buckling stress vs slenderness for DB materials
    ax = axes[2]
    ratios = np.linspace(0.002, 0.02, 40)
    for material in ("AH36", "alu_5083_H116", "frp_glass_polyester"):
        sig = [plate_buckling(db, material, t=r * 0.8, b=0.8)["sigma_cr_Pa"] / 1e6
               for r in ratios]
        ax.plot(ratios * 100.0, sig, label=material)
    ax.set_xlabel("t/b [%]")
    ax.set_ylabel("sigma_cr [MPa]")
    ax.set_title("(c) DNV-RP-C201 plate buckling from DB E, nu")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle("TensorDG property database v1 - validation and solver "
                 "consumption", fontsize=9)
    fig.tight_layout()
    path = OUT / "props_db_v1_validation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    db = load_db()
    for path in (fig_properties(db), fig_validation(db)):
        print(path)


if __name__ == "__main__":
    main()