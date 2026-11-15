"""Measure the triple point of the double Mach reflection and compare it with the
three-shock (von Neumann) theory reference -- anchor REPRO-DG-CKS1998.

This closes the gap that ``docs/CASE-REPRO-DG-CKS1998-DOUBLE-MACH.md`` §5-1
records: the triple-point trajectory had no analytic/semi-analytic reference.

Every number written to ``metrics.json`` / ``summary.json`` is produced by this
script (no hand copying).  Run one configuration:

    PYTHONPATH=. python scripts/run_dmr_triple_point.py --nx 256 --ny 64 \
        --degree 1 --time 0.2 --out validation/dmr/triple_point/p1_256x64_t020

or several at once with ``--configs`` (a JSON list of dicts) in one cluster job.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from tensordg.euler_dg_2d import cell_fields
from benchmarks.double_mach_reflection import analytic_front_x, solve_double_mach
import benchmarks.dmr_three_shock as TS


def run_case(nx: int, ny: int, degree: int, final_time: float, cfl: float,
             limiter: str = "tvb", tvb_constant: float = 100.0) -> Dict[str, object]:
    t0 = time.time()
    x, y, coeff, meta = solve_double_mach(
        ncells=(nx, ny), degree=degree, final_time=final_time, cfl=cfl,
        limiter=None if limiter == "none" else limiter,
        tvb_constant=tvb_constant, limiter_stage="final",
    )
    wall = time.time() - t0
    fields = cell_fields(coeff)
    rho = fields["density"].cpu().numpy()
    xc = np.asarray(x, dtype=float).ravel()
    yc = np.asarray(y, dtype=float).ravel()
    tp = TS.measure_triple_point(rho, xc, yc)
    out: Dict[str, object] = {
        "grid": [nx, ny], "degree": degree, "final_time": final_time,
        "cfl": cfl, "limiter": limiter, "steps": meta.get("steps"),
        "wall_seconds": wall,
        "dx": 4.0 / nx, "dy": 1.0 / ny,
        "density_min": float(rho.min()), "density_max": float(rho.max()),
        "front_x_exact_at_y": {
            f"{yy:.2f}": float(analytic_front_x(yy, final_time))
            for yy in (0.2, 0.4, 0.6, 0.8)
        },
    }
    if "x" in tp:
        chi_meas = TS.trajectory_angle_from_point(tp["x"], tp["y"])
        out["triple_point"] = {k: v for k, v in tp.items() if k != "locus"}
        out["triple_point"]["chi_deg"] = chi_meas
        out["triple_point"]["x_hinge_applied"] = TS.X_HINGE
    else:
        out["triple_point"] = tp
    # keep the raw front locus for the figure / independent re-analysis
    out["front_locus"] = tp.get("locus", {})
    return out, rho, xc, yc


def compare_with_theory(cases: List[Dict[str, object]], theory: Dict[str, object]
                        ) -> Dict[str, object]:
    branches = theory["branches"]
    comp = {}
    for case in cases:
        tp = case.get("triple_point", {})
        if "x" not in tp:
            comp[f"{case['degree']}_{case['grid'][0]}x{case['grid'][1]}"] = {
                "note": "no triple point resolved"}
            continue
        entry = {"measured": {"x": tp["x"], "y": tp["y"],
                              "chi_deg": tp["chi_deg"]},
                 "theory": {}}
        for label, br in branches.items():
            pred = TS.trajectory_point(br, case["final_time"])
            dist = math.hypot(tp["x"] - pred["x"], tp["y"] - pred["y"])
            entry["theory"][label] = {
                "chi_deg": br["chi_deg"], "x": pred["x"], "y": pred["y"],
                "chi_error_deg": tp["chi_deg"] - br["chi_deg"],
                "distance": dist,
                "distance_cells": dist / math.hypot(case["dx"], case["dy"]),
            }
        comp[f"P{case['degree']}_{case['grid'][0]}x{case['grid'][1]}_"
             f"t{case['final_time']:.3f}"] = entry
    return comp


def figure(cases, theory, out: Path, label: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case = cases[-1]
    rho = case["_rho"]
    xc, yc = case["_xc"], case["_yc"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))

    ax = axes[0]
    mesh = ax.pcolormesh(xc, yc, rho.T, cmap="viridis", shading="auto")
    fig.colorbar(mesh, ax=ax, label=r"$\rho$")
    locus = case.get("front_locus", {})
    if locus:
        ys = np.array(sorted(float(k) for k in locus))
        xs = np.array([locus[f"{y}"] if f"{y}" in locus else locus[y] for y in ys])
        ax.plot(xs, ys, "w-", lw=0.8, label="traced front")
    tp = case.get("triple_point", {})
    if "x" in tp:
        ax.plot([tp["x"]], [tp["y"]], "r*", ms=13, label="measured triple point")
    for lab, br in theory["branches"].items():
        p = TS.trajectory_point(br, case["final_time"])
        ax.plot([p["x"]], [p["y"]], "o", mfc="none", mec="w", ms=9,
                label=f"theory {lab} (chi={br['chi_deg']:.1f} deg)")
    ax.set_xlim(0.5, 4.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"dMR density, P{case['degree']} {case['grid'][0]}x{case['grid'][1]}, "
                 f"t={case['final_time']:.2f}")
    ax.legend(fontsize=6, loc="lower left")

    ax = axes[1]
    if locus:
        resid = np.array([float(r) for r in case["triple_point"].get("residual_cells", [])])
        ys = np.array(sorted(float(k) for k in locus))
        n = min(resid.size, ys.size)
        ax.plot(resid[:n], ys[:n], lw=1.0)
        ax.axvline(1.5, ls=":", color="0.4", label="kink threshold (1.5 cells)")
        ax.axhline(tp.get("y", np.nan), ls="--", color="r", lw=0.8,
                   label=f"measured y_T={tp.get('y', float('nan')):.3f}")
        ax.set_xlabel("front deviation from the straight incident line [cells]")
        ax.set_ylabel("y")
        ax.set_title("triple-point detection")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / f"{label}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--time", type=float, default=0.2)
    ap.add_argument("--cfl", type=float, default=0.2)
    ap.add_argument("--limiter", default="tvb")
    ap.add_argument("--configs", default=None,
                    help="JSON list of {nx,ny,degree,time,cfl} run back to back")
    ap.add_argument("--theory-json", default=None,
                    help="reuse a theory summary instead of re-solving")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    theory = (json.loads(Path(args.theory_json).read_text())
              if args.theory_json else TS.summary())

    configs = (json.loads(args.configs) if args.configs
               else [dict(nx=args.nx, ny=args.ny, degree=args.degree,
                          time=args.time, cfl=args.cfl, limiter=args.limiter)])

    cases = []
    for cfg in configs:
        t0 = time.time()
        res, rho, xc, yc = run_case(nx=cfg["nx"], ny=cfg["ny"],
                                    degree=cfg.get("degree", 1),
                                    final_time=cfg["time"],
                                    cfl=cfg.get("cfl", 0.2),
                                    limiter=cfg.get("limiter", "tvb"))
        print(f"[{time.time()-t0:7.1f}s] finished {cfg}", flush=True)
        cases.append(res)
        np.savez_compressed(out / f"density_p{cfg.get('degree',1)}_"
                                  f"{cfg['nx']}x{cfg['ny']}_t{cfg['time']:.3f}.npz",
                            density=rho, x=xc, y=yc)
        res["_rho"], res["_xc"], res["_yc"] = rho, xc, yc

    comparison = compare_with_theory(cases, theory)
    summary = {
        "case": "double_mach_reflection triple point vs three-shock theory",
        "anchor": "REPRO-DG-CKS1998",
        "theory": {k: v for k, v in theory.items() if k != "theory"},
        "theory_full": theory["theory"],
        "cases": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cases],
        "comparison": comparison,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for c in cases:
        (out / f"metrics_p{c['degree']}_{c['grid'][0]}x{c['grid'][1]}"
               f"_t{c['final_time']:.3f}.json").write_text(json.dumps(
                   {k: v for k, v in c.items() if not k.startswith("_")}, indent=1) + "\n")
    if not args.no_figure:
        for c in cases:
            lab = f"dmr_triple_p{c['degree']}_{c['grid'][0]}x{c['grid'][1]}_t{c['final_time']:.3f}"
            print("figure ->", figure([c], theory, out, lab))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()