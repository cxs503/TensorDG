"""Cross-validation of the LIFE-style static immersed-marker obstacles.

Runs the marker-only cylinder (no solid mask, no bounce-back on the body) over
a refinement ladder ``D = 16, 24, 32`` cells per diameter plus a marker-density
study at ``D = 24``, and writes every number to
``outputs/cylinder_ibm/xval_summary.json``.  Nothing is hand-typed afterwards:
the report generator reads that JSON.

Reference (pinned in ``tensorlbm.t3_matrix.CASES['cylinder-2d-re100']``):
``Cd = 1.32 ... 1.38`` and ``St = 0.164 ... 0.166`` at Re = 100
(Park, Kwon & Choi 1998; Schaefer & Turek 1996 benchmark collection).
Acceptance: relative error <= 15 % (ecosystem limit), the T3 criterion for
this case is the tighter 10 %.

Usage:
    python benchmarks/cylinder_ibm_xval.py            # full ladder
    python benchmarks/cylinder_ibm_xval.py --quick     # D=12 only, smoke
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.cylinder import run_cylinder  # noqa: E402

CD_REF = 1.35
CD_RANGE = (1.32, 1.38)
ST_REF = 0.165
ST_RANGE = (0.164, 0.166)
ACCEPTANCE = 0.15
T3_TOLERANCE = 0.10
OUT = ROOT / "outputs" / "cylinder_ibm"


def _rel(value: float, ref: float) -> float:
    return abs(value - ref) / abs(ref)


def _summarise(res, label: str) -> dict:
    cd_err = _rel(res.cd_mean, CD_REF)
    st_rel = _rel(res.st, ST_REF) if res.st == res.st else float("nan")   # NaN-safe
    return {
        "label": label,
        "d_cells": res.d_cells,
        "re": res.re,
        "u0": res.u0,
        "nu": res.nu,
        "tau": res.tau,
        "nx": res.nx,
        "ny": res.ny,
        "blockage": res.blockage,
        "steps": res.steps,
        "sample_every": res.sample_every,
        "average_from_step": res.average_from_step,
        "n_markers": res.marker_stats["n_markers"],
        "marker_spacing_realised": res.marker_stats["spacing_realised"],
        "marker_spacing_requested": res.marker_stats.get("marker_spacing_requested"),
        "marker_weight_max_deviation": res.marker_weights["max_deviation_from_one"],
        "cd_mean": res.cd_mean,
        "cd_amp": res.cd_amp,
        "cl_mean": res.cl_mean,
        "cl_rms": res.cl_rms,
        "st": res.st,
        "st_zero_crossing": res.st_zero_crossing,
        "n_periods": res.n_periods,
        "cd_rel_error": cd_err,
        "st_rel_error": st_rel,
        "cd_rel_error_vs_range": max(_rel(res.cd_mean, CD_RANGE[0]), _rel(res.cd_mean, CD_RANGE[1])),
        "st_rel_error_vs_range": max(_rel(res.st, ST_RANGE[0]), _rel(res.st, ST_RANGE[1])) if res.st == res.st else float("nan"),
        "slip_after": res.slip_after,
        "slip_rel_to_u0": res.slip_rel_to_u0,
        "mass_drift": res.mass_drift,
        "rho_min": res.rho_min,
        "rho_max": res.rho_max,
        "momentum_budget": res.momentum_budget,
        "wall_seconds": res.wall_seconds,
        "wake_periodic": res.converged_wake,
        "status": "ok",
    }


def _history(res) -> dict:
    return {
        "t_over_D": [round(v, 5) for v in res.history["t_over_D"]],
        "cd": [round(v, 6) for v in res.history["cd"]],
        "cl": [round(v, 6) for v in res.history["cl"]],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="16,24,32")
    ap.add_argument("--spacing-study-d", type=int, default=24)
    ap.add_argument("--spacing", default="1.0,0.5")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps-per-diameter", type=float, default=150.0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.quick:
        ladder = [12]
        spacings = []
        spd = 120.0
    else:
        ladder = [int(v) for v in args.ladder.split(",")]
        spacings = [float(v) for v in args.spacing.split(",")]
        spd = args.steps_per_diameter

    summary = {
        "reference": {
            "cd": CD_REF, "cd_range": list(CD_RANGE),
            "st": ST_REF, "st_range": list(ST_RANGE),
            "re": 100.0,
            "source": "Park, Kwon & Choi 1998 JCP 139, 401; Schaefer & Turek 1996 benchmark set",
            "acceptance_limit": ACCEPTANCE,
            "t3_tolerance": T3_TOLERANCE,
        },
        "convention": {
            "markers": "equal arc length, floor(2 pi r / dx) markers, prune < 0.5 dx",
            "kernel": "4pt cosine Peskin delta",
            "forcing": "multi-direct forcing (epsilon=1, 3 sweeps) + Guo source",
            "boundaries": "velocity inlet / pressure outlet (neq extrapolation), no-slip walls",
        },
        "ladder": [],
        "spacing_study": [],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for d in ladder:
        n_steps = int(round(spd * d))
        print(f"[ladder] D={d} steps={n_steps}", flush=True)
        try:
            res = run_cylinder(d_cells=d, re=100.0, n_steps=n_steps, average_from=0.55,
                               sample_every=2, threads=args.threads,
                               marker_spacing=1.0, field_snapshot=(d == ladder[-1]))
            rec = _summarise(res, f"D={d} dx=1.0")
            rec["history"] = _history(res)
            if d == ladder[-1] and res.snapshot is not None:
                import numpy as np
                np.savez_compressed(
                    OUT / f"field_D{d}.npz",
                    vorticity=res.snapshot["vorticity"],
                    ux=res.snapshot["ux"], uy=res.snapshot["uy"], rho=res.snapshot["rho"],
                    marker_x=res.snapshot["marker_x"], marker_y=res.snapshot["marker_y"],
                    solid=res.snapshot["solid"],
                )
            summary["ladder"].append(rec)
        except Exception as exc:            # honest: record, do not fake a pass
            summary["ladder"].append({"label": f"D={d} dx=1.0", "d_cells": d,
                                      "status": f"failed: {type(exc).__name__}: {exc}"})
        (OUT / "xval_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    for sp in spacings:
        d = args.spacing_study_d
        n_steps = int(round(spd * d))
        print(f"[spacing] D={d} spacing={sp} steps={n_steps}", flush=True)
        try:
            res = run_cylinder(d_cells=d, re=100.0, n_steps=n_steps, average_from=0.55,
                               sample_every=2, threads=args.threads,
                               marker_spacing=sp, field_snapshot=False)
            rec = _summarise(res, f"D={d} dx={sp}")
            rec["history"] = _history(res)
            summary["spacing_study"].append(rec)
        except Exception as exc:
            summary["spacing_study"].append({"label": f"D={d} dx={sp}", "d_cells": d,
                                             "status": f"failed: {type(exc).__name__}: {exc}"})
        (OUT / "xval_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    ok = [r for r in summary["ladder"] if r.get("status") == "ok"]
    fine = ok[-1] if ok else None
    summary["verdict"] = {
        "finest_d_cells": fine["d_cells"] if fine else None,
        "cd": fine["cd_mean"] if fine else None,
        "st": fine["st"] if fine else None,
        "cd_rel_error": fine["cd_rel_error"] if fine else None,
        "st_rel_error": fine["st_rel_error"] if fine else None,
        "wake_periodic": fine["wake_periodic"] if fine else None,
        "within_15pct": bool(fine and fine["cd_rel_error"] <= ACCEPTANCE
                             and fine["st_rel_error"] <= ACCEPTANCE) if fine else False,
        "within_t3_10pct": bool(fine and fine["cd_rel_error"] <= T3_TOLERANCE
                                and fine["st_rel_error"] <= T3_TOLERANCE) if fine else False,
    }
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT / "xval_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["verdict"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())