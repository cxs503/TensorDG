"""Produce the DG-side time axis artifact for XVAL-DG-ROUND2-CERTIFIED-REFERENCE.

Why this driver exists
----------------------
``benchmarks/xval_dg_round2_time_pollution.py`` audits the *reference* for time
pollution (T0-T5).  Its real-data branch is still withheld: six CFL-refined
FV-WENO5 profiles are missing (submitted as compute jobs, none dispatched).  The
**DG legs** carry a second, independent time error that no report ever quantified:
nine of the eleven published legs are quoted at CFL=0.4 and nothing says the
quoted number is a space error rather than space *plus* time error.

Three of those legs -- P2, N=400 at CFL 0.4/0.2/0.1 -- are already committed and
differ *only* in the time step, so the axis separates with **zero new
computation**.  This driver runs that measurement and writes it as a first-class
artifact with its own predeclared criteria (D0-D3, defined in the module).

Usage::

    python validation/xval_dg_round2/gen_dg_time_axis.py
    python validation/xval_dg_round2/gen_dg_time_axis.py --rawdir <dir> --outdir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import xval_dg_round2_time_pollution as tp  # noqa: E402

DEFAULT_OUT = "validation/xval_dg_round2/artifacts"


def figure(axis: dict, cert: dict, rawdir: Path | str, outdir: Path | str) -> str:
    """One PNG: leg error vs CFL against the certified band, and the time steps."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / "dg_time_axis.png"
    cfls = [lg["cfl"] for lg in axis["legs"]]
    errs = [lg["rel_l1_percent"] for lg in axis["legs"]]
    band = axis["criteria"]["D0"]["band_percent"]
    gate = axis["tolerance_percent"]
    dts = [d["rel_l1_percent"] for d in axis["differences"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax1.plot(cfls, errs, "o-", color="#1f4e79", label="DG P2 N=400 leg error")
    ax1.axhline(gate, ls="--", color="#c00000", label=f"{gate:g} % gate")
    ax1.fill_between([min(cfls), max(cfls)], band, gate, color="#c00000", alpha=0.06)
    ax1.set_xscale("log")
    ax1.set_xlabel("CFL (fixed grid, fixed degree: time step only)")
    ax1.set_ylabel("rel L1 vs certified reference  [%]")
    ax1.set_title("leg error barely moves with the time step")
    ax1.legend(loc="center right", fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    labels = [f"{d['pair'][0]:g}→{d['pair'][1]:g}" for d in axis["differences"]]
    ax2.bar(labels, dts, color="#2e75b6", width=0.5, label="temporal difference")
    ax2.axhline(band, ls="--", color="#c00000",
                label=f"certified band U = {band:.4f} %")
    ax2.set_ylabel("rel L1  [%]")
    ax2.set_title("time step change is below the reference band")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
    for i, v in enumerate(dts):
        ax2.annotate(f"{v:.4f}%", (i, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8)
    fig.suptitle("XVAL-DG-ROUND2 — DG-side time axis (P2 N=400, CFL 0.4/0.2/0.1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return str(png)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rawdir", default=tp.DEFAULT_RAW)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    cert = tp.certify([tp.load_level(args.rawdir, n, 0.3)
                       for n in (*tp.BASE_LEVELS, *tp.EXTRA_RATIO_LEVELS)],
                      tolerance_percent=tp.GATE_PERCENT)
    axis = tp.dg_temporal_axis(args.rawdir, cert)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dg_time_axis.json").write_text(
        json.dumps(tp._jsonable(axis), indent=2, sort_keys=True) + "\n")
    md = "\n".join([
        "# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — DG-side time axis",
        "",
        f"Question: {axis['question']}",
        "",
        tp.markdown_dg_axis(axis),
        "## Predeclared thresholds",
        "",
        f"* `POLLUTION_MAX = {tp.POLLUTION_MAX}` (D1 CLEAN), "
        f"`POLLUTION_MARGINAL = {tp.POLLUTION_MARGINAL}` (D1 MARGINAL ceiling)",
        f"* `LEG_STABILITY_MAX_PP = {tp.LEG_STABILITY_MAX_PP}` (D2), "
        f"`P_MIN_T = {tp.P_MIN_T}` (D3)",
        "* `band` = Roache GCI of the certified reference (Fs = 1.25), i.e. the "
        "reference's own uncertainty; D0 asks whether the time step change is "
        "large enough to be distinguishable from it at all.",
        "",
    ])
    (out / "dg_time_axis.md").write_text(md + "\n")
    figs = [] if args.no_figure else [figure(axis, cert, args.rawdir, out)]
    print(md)
    for f in figs:
        print(f"wrote {f}")
    print(f"wrote {out/'dg_time_axis.json'}")
    return 0 if axis["accepted"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())