"""Root-cause scan for the CKS1998 2-D Riemann divergence (gate C6 study).

Background (honest bookkeeping)
-------------------------------
``docs/CASE-DG-CKS1998-2D-RIEMANN-CONSERVATION.md`` recorded that the declared
configuration of ``benchmarks.riemann_2d_cks1998`` (Lax & Liu 1998 config 3,
CFL = 0.2, t = 0.25) **diverges**: total mass grows to +4.8e+26 by t = 0.25 on a
32x32 mesh while positivity and the discrete flux-balance identity still hold to
round-off, so none of the then-existing gates (C1/C3/C4/C5) could see it.  It
also stated the root cause was *not* isolated and named two candidates:

  (i)  ``CFL = 0.2`` is simply past the 2-D P1 DG + SSP-RK3 stability limit;
  (ii) the TVB indicator with ``M = 1`` releases shock cells, letting the
       polynomial bulge (over/undershoot feeding back into the solution).

This script separates them with a two-factor scan over ``cfl`` and the TVB
constant ``M`` at a *short* final time (``t = 0.05``, the "stable window" used by
the conservation audit), and scores every run with the new pre-registered gate
C6 (``benchmarks.riemann_2d_cks1998.a_priori_bounds``).

Usage (single thread, ~8 min on one core):

    OMP_NUM_THREADS=1 python -m benchmarks.riemann_2d_divergence_scan \
        --out validation/riemann2d_c6

The output JSON is written incrementally, so an interrupted scan still leaves
the finished rows on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from benchmarks.riemann_2d_cks1998 import (  # noqa: E402
    EXTREMUM_SAFETY,
    GATES,
    MASS_FLUX_SAFETY,
    a_priori_bounds,
    c6_extremum_factor,
    c6_mass_bound_ratio,
    run_riemann_2d,
)

DEFAULT_CFLS = (0.1, 0.15, 0.2, 0.25)
DEFAULT_TVB = (0.0, 1.0)


def _slim_result(record, cfl, tvb_constant, ncells, degree):
    return {
        "degree": degree,
        "ncells": ncells,
        "cfl": cfl,
        "tvb_constant": tvb_constant,
        "final_time": record["final_time"],
        "steps": record["steps"],
        "wall_seconds": record["wall_seconds"],
        "mass0": record["c6_bounds"]["mass0"],
        "mass_change": record["conservation"]["mass"]["change"],
        "mass_flux_bound": record["c6_bounds"]["mass_flux_bound"],
        "c6_mass_bound_ratio": c6_mass_bound_ratio(record),
        "c6_extremum_factor": c6_extremum_factor(record),
        "rho_min": record["positivity"]["rho_min"],
        "rho_max": record["positivity"]["rho_max"],
        "p_min": record["positivity"]["p_min"],
        "p_max": record["positivity"]["p_max"],
        "finite": record["finite"],
        "flux_balance_residual": record["max_flux_balance_residual"],
        "c6_passed": bool(
            c6_mass_bound_ratio(record) <= 1.0
            and c6_extremum_factor(record) <= GATES["C6_extremum_factor"]
        ),
    }


def _failed_row(exc, cfl, tvb_constant, ncells, degree, final_time):
    """A run that raised (e.g. FloatingPointError on negative pressure)."""
    bounds = a_priori_bounds(final_time)
    return {
        "degree": degree,
        "ncells": ncells,
        "cfl": cfl,
        "tvb_constant": tvb_constant,
        "final_time": final_time,
        "steps": None,
        "wall_seconds": None,
        "mass0": bounds["mass0"],
        "mass_change": None,
        "mass_flux_bound": bounds["mass_flux_bound"],
        "c6_mass_bound_ratio": None,
        "c6_extremum_factor": None,
        "rho_min": None,
        "rho_max": None,
        "p_min": None,
        "p_max": None,
        "finite": False,
        "flux_balance_residual": None,
        "c6_passed": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def scan(cfls=DEFAULT_CFLS, tvb_constants=DEFAULT_TVB, ncells=16, degree=1,
         final_time=0.05, out_dir=Path("validation/riemann2d_c6")):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    payload = {
        "benchmark": "riemann_2d_divergence_scan",
        "case": "Lax_Liu_1998_config_3",
        "gates": GATES,
        "safety_constants": {
            "MASS_FLUX_SAFETY": MASS_FLUX_SAFETY,
            "EXTREMUM_SAFETY": EXTREMUM_SAFETY,
        },
        "a_priori_bounds": a_priori_bounds(final_time),
        "rows": rows,
    }
    target = out_dir / "scan.json"

    def flush():
        target.write_text(json.dumps(payload, indent=2, default=str))

    total = len(cfls) * len(tvb_constants)
    for cfl in cfls:
        for tvb in tvb_constants:
            label = f"P{degree} n={ncells} cfl={cfl} M={tvb}"
            print(f"[{len(rows) + 1}/{total}] {label} ...", flush=True)
            started = time.time()
            try:
                record = run_riemann_2d(
                    ncells=ncells, degree=degree, final_time=final_time,
                    cfl=cfl, tvb_constant=tvb,
                )
                row = _slim_result(record, cfl, tvb, ncells, degree)
            except Exception as exc:  # noqa: BLE001 - divergence is a datum here
                row = _failed_row(exc, cfl, tvb, ncells, degree, final_time)
            row["judge_seconds"] = round(time.time() - started, 2)
            rows.append(row)
            flush()
            print(
                "      steps={steps} mass_change={mc} ratio={ratio} "
                "extremum={ext} pass={p}".format(
                    steps=row["steps"],
                    mc=None if row["mass_change"] is None else f"{row['mass_change']:.6e}",
                    ratio=(
                        None if row["c6_mass_bound_ratio"] is None
                        else f"{row['c6_mass_bound_ratio']:.4e}"
                    ),
                    ext=(
                        None if row["c6_extremum_factor"] is None
                        else f"{row['c6_extremum_factor']:.4f}"
                    ),
                    p=row["c6_passed"],
                ),
                flush=True,
            )
    flush()

    # Whole-scan judge: one aggregate verdict over every row that produced data.
    scored = [r for r in rows if r["mass_change"] is not None]
    payload["scan_summary"] = {
        "rows": len(rows),
        "diverged_rows": [
            f"cfl={r['cfl']},M={r['tvb_constant']}" for r in rows if not r["c6_passed"]
        ],
        "worst_mass_bound_ratio": max(
            (r["c6_mass_bound_ratio"] for r in scored), default=None
        ),
        "worst_extremum_factor": max(
            (r["c6_extremum_factor"] for r in scored), default=None
        ),
    }
    flush()
    return payload


def plot_scan(payload, output_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in payload["rows"] if r["c6_mass_bound_ratio"] is not None]
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for tvb in sorted({r["tvb_constant"] for r in rows}):
        subset = sorted(
            (r for r in rows if r["tvb_constant"] == tvb), key=lambda r: r["cfl"]
        )
        cfls = [r["cfl"] for r in subset]
        axes[0].plot(cfls, [r["c6_mass_bound_ratio"] for r in subset], "o-",
                     label=f"TVB M={tvb:g}")
        axes[1].plot(cfls, [r["c6_extremum_factor"] for r in subset], "s-",
                     label=f"TVB M={tvb:g}")
    axes[0].axhline(1.0, color="k", ls="--", lw=1, label="C6 limit (=1)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("CFL")
    axes[0].set_ylabel(r"$|\Delta M| \,/\, M_{\rm flux\ bound}$")
    axes[0].set_title("C6 mass-amplification ratio (log scale)")
    axes[1].axhline(GATES["C6_extremum_factor"], color="k", ls="--", lw=1,
                    label=f"C6 limit (={GATES['C6_extremum_factor']:g})")
    axes[1].set_xlabel("CFL")
    axes[1].set_ylabel(r"$\max(\rho_{\max}/\rho_{\max,0},\ p_{\max}/p_{\max,0})$")
    axes[1].set_title("C6 extremum factor")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "CKS1998 2-D Riemann (Lax-Liu cfg 3), P{deg} {n}x{n}, t={t}: "
        "CFL x TVB scan".format(
            deg=rows[0]["degree"], n=rows[0]["ncells"], t=rows[0]["final_time"]
        )
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="validation/riemann2d_c6")
    parser.add_argument("--ncells", type=int, default=16)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--time", type=float, default=0.05)
    parser.add_argument("--cfls", type=float, nargs="+", default=list(DEFAULT_CFLS))
    parser.add_argument("--tvb", type=float, nargs="+", default=list(DEFAULT_TVB))
    args = parser.parse_args()

    torch.set_num_threads(1)
    payload = scan(
        cfls=tuple(args.cfls),
        tvb_constants=tuple(args.tvb),
        ncells=args.ncells,
        degree=args.degree,
        final_time=args.time,
        out_dir=Path(args.out),
    )
    png = plot_scan(payload, Path(args.out) / "riemann2d_c6_scan.png")
    print(json.dumps(payload["scan_summary"], indent=2))
    print("figure:", png)


if __name__ == "__main__":
    main()