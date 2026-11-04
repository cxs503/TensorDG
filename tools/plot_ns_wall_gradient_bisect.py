"""Figure for the m3139 wall-gradient bisection (``docs/figures``).

Reads ``validation/ns_dg_2d_wall/bisect_wall_gradient.json`` only, so every
plotted number is a recorded measurement -- nothing is transcribed by hand.

Left : wall heat-flux error vs mesh size for P1/P2/P3, one-sided DG derivative
       (``face``, the default) against the PR #36 extrapolation (``taylor``).
Right: residual of the exact analytic steady state, same two estimators.

Usage::

    PYTHONPATH=. python -m tools.plot_ns_wall_gradient_bisect
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation" / "ns_dg_2d_wall" / "bisect_wall_gradient.json"
FIG = ROOT / "docs" / "figures" / "ns_dg_2d_wall_gradient_bisect.png"

STYLE = {
    "face": dict(marker="o", ls="-", color="#1f77b4"),
    "taylor": dict(marker="s", ls="--", color="#d62728"),
}
NAME = {"face": "face (default)", "taylor": "taylor (opt-in)"}


def _series(rows, degree, key):
    picked = sorted((r for r in rows if r["degree"] == degree), key=lambda r: r["ny"])
    return [1.0 / r["ny"] for r in picked], [r[key] for r in picked]


def main():
    payload = json.loads(DATA.read_text())
    run = next(r for r in payload["runs"] if r["label"].startswith("main@fix"))
    modes = run["modes"]

    fig, (ax_flux, ax_res) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for degree in (1, 2, 3):
        for mode in modes:
            style = STYLE[mode]
            ax = ax_flux
            h, err = _series(run["flux"][mode], degree, "energy_error_bottom")
            ax.loglog(h, err, lw=1.6, ms=4.5, **style,
                      label=f"{NAME[mode]}, P{degree} (order "
                            f"{run['summary'][mode][f'p{degree}']['observed_order']:.2f})")
            h, res = _series(run["residual"][mode], degree, "rhs_max")
            ax_res.loglog(h, res, lw=1.6, ms=4.5, **style,
                          label=f"{NAME[mode]}, P{degree}")

    ax_flux.set_xlabel("cell size h = 1/ny")
    ax_flux.set_ylabel(r"$|F_{E,\mathrm{bottom}} - F_{E,\mathrm{exact}}|$")
    ax_flux.set_title("wall heat flux, exact Brinkman/Couette state")
    ax_flux.grid(True, which="both", alpha=0.25)
    ax_flux.legend(fontsize=6.4, ncol=1, loc="lower left")

    ax_res.set_xlabel("cell size h = 1/ny")
    ax_res.set_ylabel(r"$\max|R_{\mathrm{exact}}|$")
    ax_res.set_title("residual of the exact steady state")
    ax_res.grid(True, which="both", alpha=0.25)
    ax_res.legend(fontsize=6.4, loc="lower right")

    fig.suptitle(
        "TensorDG ns_dg_2d wall operator, m3139 bisection "
        f"({payload['arch']['uname_machine']}, float64): design order p is kept only by the default",
        fontsize=9.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=170)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()