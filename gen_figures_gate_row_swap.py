"""Figures for the dual-instrument gate-row measurement (m23108).

Reads only committed artifacts -- ``validation/cks1998_gate_row_swap/rows.json`` and the
per-row ``profiles/<row>.npz`` dumped by ``gate_row_swap.py --profiles`` -- and writes
``docs/figures/cks1998_gate_row_swap.png``.  No number is typed by hand.

Panels
------
(a) the two gate rows' cell-average profiles against the *exact* reference, so the
    reader can see which cells carry a wave (the only cells where the instruments
    can differ);
(b) ``|legacy - exact|`` per cell on a log scale, with the wave abscissae drawn:
    the bias lives in ``distance < 0.5 dx`` and is zero elsewhere;
(c) the gated quantity itself: max relative L2 per row under both instruments, with
    the 3 % bar and each row's margin after the displacement.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.shock_reference_quadrature import breakpoints
from benchmarks.sod_exact import LEFT, RIGHT

ARTIFACT = Path("validation/cks1998_gate_row_swap/rows.json")
PROFILE_DIR = Path("validation/cks1998_gate_row_swap/profiles")
OUT = Path("docs/figures/cks1998_gate_row_swap.png")

ROW_LABEL = {"E2_dg_p2_n400": "1-D DG P2, N=400", "extruded2d_p0_n3200": "extruded 2-D P0, nx=3200"}


def main() -> None:
    payload = json.loads(ARTIFACT.read_text())
    t = 0.2
    cuts = np.asarray(breakpoints(LEFT, RIGHT, t), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.6))
    colors = {"E2_dg_p2_n400": "#1f77b4", "extruded2d_p0_n3200": "#d62728"}

    # (a) profiles vs the exact reference
    ax = axes[0]
    for row in payload["rows"]:
        data = np.load(PROFILE_DIR / f"{row['row']}.npz")
        x = data["x"]
        ax.plot(x, data["numerical_density"], color=colors[row["row"]], lw=1.4,
                label=f"{ROW_LABEL[row['row']]} (DG)")
        ax.plot(x, data["exact_density"], color="k", lw=0.9, ls="--", alpha=0.75,
                label="exact cell averages" if row["row"] == payload["rows"][0]["row"] else None)
    for c in cuts:
        ax.axvline(c, color="grey", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("x")
    ax.set_ylabel(r"cell-average $\rho$")
    ax.set_title("(a) the two gated rows vs the exact reference\n(dotted: analytic breakpoints, t=0.2)")
    ax.legend(fontsize=8)

    # (b) per-cell instrument difference
    ax = axes[1]
    for row in payload["rows"]:
        data = np.load(PROFILE_DIR / f"{row['row']}.npz")
        x = data["x"]
        diff = np.abs(data["legacy_density"] - data["exact_density"])
        ax.semilogy(x, np.maximum(diff, 1e-17), color=colors[row["row"]], lw=1.2,
                    label=ROW_LABEL[row["row"]])
    for c in cuts:
        ax.axvline(c, color="grey", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\rho_{\mathrm{legacy}}-\rho_{\mathrm{exact}}|$")
    ax.set_title("(b) the legacy fixed-rule instrument errs only\nin cells that contain a wave")
    ax.set_ylim(1e-17, 1e-1)
    ax.legend(fontsize=8)

    # (c) the gated quantity
    ax = axes[2]
    width = 0.36
    xs = np.arange(len(payload["rows"]))
    exact = [row["max_percent_exact"] for row in payload["rows"]]
    legacy = [row["max_percent_legacy"] for row in payload["rows"]]
    ax.bar(xs - width / 2, exact, width, color="#2ca02c", label="exact instrument")
    ax.bar(xs + width / 2, legacy, width, color="#ff7f0e", label="legacy fixed rule")
    ax.axhline(payload["gate_percent"], color="k", ls="--", lw=1.0, label="3 % gate")
    for xi, (e, l) in enumerate(zip(exact, legacy)):
        ax.text(xi - width / 2, e + 0.05, f"{e:.3f}%", ha="center", fontsize=8)
        ax.text(xi + width / 2, l + 0.05, f"{l:.3f}%", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([ROW_LABEL[row["row"]] for row in payload["rows"]], fontsize=8)
    ax.set_ylabel("max relative L2 [%]  (gated quantity)")
    ax.set_ylim(0.0, max(max(exact), max(legacy)) * 1.35)
    margins = ", ".join(f"{row['margin_pp_exact']:.2f} pp" for row in payload["rows"])
    ax.set_title(f"(c) verdict identical under both instruments\nmargins left: {margins}")
    ax.legend(fontsize=8)

    fig.suptitle("CKS1998 gate rows: dual-instrument measurement (m23108)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()