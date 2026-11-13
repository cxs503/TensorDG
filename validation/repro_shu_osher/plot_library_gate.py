"""Figures for the Shu--Osher library-acceptance record.

Reads ``docs/shu_osher_library_gate.json`` (written by ``library_gate.py``) and
one DG solve, and draws

* (a) the density profile of the DG run against the committed reference profile,
* (b) a zoom on the post-shock entropy-wave train,
* (c) the shock-locator error vs cell size, pre-fix baseline next to the shipped
  locator (the numbers come from the JSON, not from this file).

Usage::

    python validation/repro_shu_osher/plot_library_gate.py --out-dir /tmp/shu_figs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensordg.cases.shock_entropy import (  # noqa: E402
    rankine_hugoniot_anchor,
    solve_dg_case,
)

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "docs" / "shu_osher_library_gate.json"
REFERENCE = ROOT / "benchmarks" / "data" / "shu_osher_reference.npz"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="/tmp/shu_figs")
    ap.add_argument("--ncells", type=int, default=400)
    ap.add_argument("--degree", type=int, default=2)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    record = json.loads(RECORD.read_text())
    anchor = rankine_hugoniot_anchor()
    solved = solve_dg_case(args.ncells, degree=args.degree, limiter="weno", cfl=0.4)
    x, rho = solved["x"], solved["rho"]
    ref = np.load(REFERENCE, allow_pickle=True)
    x_ref, rho_ref = ref["x"], ref["rho"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))

    ax = axes[0]
    ax.step(x, rho, where="mid", lw=1.1, color="C0",
            label=f"DG P{args.degree} N={args.ncells} (weno, this work)")
    ax.step(x_ref, rho_ref, where="mid", lw=0.9, color="C3", alpha=0.75,
            label="committed reference (DG P3 N=300)")
    for pos, style, name in (
        (anchor["shock_position_analytic"], "--", "analytic shock"),
        (anchor["contact_position_analytic"], ":", "analytic contact"),
    ):
        ax.axvline(pos, color="k", ls=style, lw=0.9)
        ax.annotate(name, xy=(pos, ax.get_ylim()[1]), xytext=(2, -10),
                    textcoords="offset points", fontsize=7, rotation=90, va="top")
    ax.set_xlabel("x")
    ax.set_ylabel("density (cell average)")
    ax.set_title("(a) Shu-Osher, t = 1.8")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    zoom = (x > 0.6) & (x < 3.2)
    zoom_ref = (x_ref > 0.6) & (x_ref < 3.2)
    ax.step(x[zoom], rho[zoom], where="mid", lw=1.3, color="C0", label="DG")
    ax.step(x_ref[zoom_ref], rho_ref[zoom_ref], where="mid", lw=1.0, color="C3",
            alpha=0.8, label="reference")
    ax.axvline(anchor["contact_position_analytic"], color="k", ls=":", lw=0.9)
    ax.axvline(anchor["shock_position_analytic"], color="k", ls="--", lw=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("(b) entropy-wave train behind the shock")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[2]
    cells = np.array([c["configuration"]["ncells"] for c in record["configs"]], dtype=float)
    dx = np.array([c["shock_locator"]["cell_step"] for c in record["configs"]])
    legacy = np.array([c["shock_locator"]["legacy"]["abs_error_percent"]
                       for c in record["configs"]])
    shipped = np.array([c["shock_locator"]["shipped"]["abs_error_percent"]
                        for c in record["configs"]])
    order = np.argsort(dx)
    ax.loglog(dx[order], legacy[order], "o--", color="C3", label="pre-fix: nearest cell")
    ax.loglog(dx[order], shipped[order], "s-", color="C2",
              label="shipped: interpolated mid-level")
    for d, err, n in zip(dx[order], shipped[order], cells[order]):
        ax.annotate(f"N={int(n)}", (d, err), textcoords="offset points",
                    xytext=(4, 4), fontsize=6)
    ax.set_xlabel("cell size dx")
    ax.set_ylabel("|x_shock - x_analytic| / x_analytic  [%]")
    ax.set_title("(c) shock-locator error vs resolution")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    path = out / "shu_osher_library_gate.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())