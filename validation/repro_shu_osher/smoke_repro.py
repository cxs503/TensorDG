"""Fast end-to-end reproducibility check for the Shu--Osher DG benchmark.

The full ladder in ``run_all.sh`` takes ~30 min.  This script re-runs *one*
configuration from scratch (default: DG P1 N=200, ~25 s), rebuilds every derived
number from that fresh run, and compares it against the numbers tabulated in
``runs/shu_osher_case_pool/metrics.json``.  It is the cheap proof that the
committed artifacts are still produced by the committed code, not only stored.

It also re-checks the closed-form anchors the benchmark leans on:

* post-shock plateau density = ``rho_L`` of the left state,
* unshocked sine ``1 + 0.2 sin(5x)`` downstream of the transmitted shock,
* global mass at ``t = 1.8`` against the analytic inflow budget,
* transmitted-shock position against the Rankine--Hugoniot speed.

Usage::

    python validation/repro_shu_osher/smoke_repro.py              # P1 / N=200
    python validation/repro_shu_osher/smoke_repro.py --degree 2 --ncells 400
    python validation/repro_shu_osher/smoke_repro.py --tol-percent 1e-3

Exit code 0 iff the fresh run reproduces the tabulated entry (within
``--tol-percent``, relative) and every analytic anchor holds.  The JSON record
lands in ``validation/repro_shu_osher/artifacts/repro_smoke.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow ``python validation/repro_shu_osher/smoke_repro.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.shu_osher_case_pool import coarsen, profile_metrics, rankine_hugoniot_anchor
from benchmarks.shu_osher_1d import solve_case
from tensordg.cases.shock_entropy import DOMAIN, GAMMA, LEFT_STATE, exact_mass_budget

RUN = Path("runs/shu_osher_case_pool")
OUT = Path("validation/repro_shu_osher/artifacts")
TOLERANCE_PERCENT = 5.0


def load_reference(ncells: int):
    path = RUN / "raw" / f"fv_cfl0p3_n{ncells}.npz"
    if not path.exists():
        raise SystemExit(f"missing reference profile {path}; run the 'reference' stage first")
    d = np.load(path, allow_pickle=True)
    return d["x"], d["rho"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ncells", type=int, default=200)
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--limiter", default="weno")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--reference-cells", type=int, default=3200)
    ap.add_argument("--tol-percent", type=float, default=1e-3,
                    help="allowed relative drift of rel_L1 vs the tabulated value, in percent")
    args = ap.parse_args(argv)

    tab = {(r["ncells"], r["degree"], r["limiter"], round(float(r["cfl"]), 6)): r
           for r in json.loads((RUN / "metrics.json").read_text())["ladder"]}
    key = (args.ncells, args.degree, args.limiter, round(float(args.cfl), 6))
    if key not in tab:
        raise SystemExit(f"configuration {key} is not in metrics.json (have {sorted(tab)})")
    expected = tab[key]

    start = time.perf_counter()
    x, state, meta = solve_case(args.ncells, args.degree, args.limiter, cfl=args.cfl)
    wall = time.perf_counter() - start
    state = np.asarray(state, dtype=float)
    rho = state[:, 0]

    x_ref, rho_ref = load_reference(args.reference_cells)
    if x_ref.shape[0] % rho.shape[0]:
        raise SystemExit(f"reference grid {x_ref.shape[0]} not a multiple of {rho.shape[0]}")
    ref_coarse = coarsen(rho_ref, x_ref.shape[0] // rho.shape[0])
    diff = rho - ref_coarse
    rel_l1 = 100.0 * float(np.sum(np.abs(diff)) / np.sum(np.abs(ref_coarse)))
    rel_l2 = 100.0 * float(np.sqrt(np.sum(diff ** 2) / np.sum(ref_coarse ** 2)))
    linf = 100.0 * float(np.max(np.abs(diff)) / np.max(np.abs(ref_coarse)))

    m = profile_metrics(x, rho)
    anchors = rankine_hugoniot_anchor()
    budget = exact_mass_budget()
    mass = float(np.sum(rho) * (DOMAIN[1] - DOMAIN[0]) / rho.size)
    mass_err = 100.0 * abs(mass - budget["mass_final_predicted"]) / budget["mass_initial"]
    shock_err = 100.0 * abs(m["shock_position"] - anchors["shock_position_analytic"]) / abs(
        anchors["shock_position_analytic"])

    drift = abs(rel_l1 - expected["rel_l1_percent"])
    checks = {
        "rel_l1_reproduces_tabulated": bool(drift <= args.tol_percent),
        "rel_l1_under_5pct": bool(rel_l1 < TOLERANCE_PERCENT),
        "density_positive": bool(rho.min() > 0.0),
        "plateau_matches_left_state": bool(
            abs(m["plateau_rho_mean"] - LEFT_STATE[0]) / LEFT_STATE[0] * 100.0 < 1.0),
        "unshocked_sine_exact": bool(m["preshock_sine_rel_l1_percent"] < 0.1),
        "mass_budget_closed": bool(mass_err < 0.1),
        "shock_position_vs_rh": bool(shock_err < 2.0),
    }
    record = {
        "case": "CASE-SHU-OSHER-DG",
        "configuration": {"ncells": args.ncells, "degree": args.degree,
                          "limiter": args.limiter, "cfl": args.cfl},
        "reference": {"solver": "benchmarks/fv_weno5_1d.py FV-WENO5/HLLC/SSP-RK3",
                      "ncells": args.reference_cells},
        "steps": int(meta.get("steps", -1)),
        "wall_time_s": round(wall, 3),
        "got": {"rel_l1_percent": round(rel_l1, 6), "rel_l2_percent": round(rel_l2, 6),
                "linf_percent": round(linf, 6), "rho_min": round(float(rho.min()), 6),
                "rho_max": round(float(rho.max()), 6),
                "shock_position": round(float(m["shock_position"]), 6)},
        "tabulated": {"rel_l1_percent": expected["rel_l1_percent"],
                      "rel_l2_percent": expected["rel_l2_percent"],
                      "steps": expected["steps"],
                      "wall_time_s": expected["wall_time_s"]},
        "drift_percent_points": round(drift, 9),
        "tolerance_percent": args.tol_percent,
        "anchors": {
            "plateau_rho_mean": round(float(m["plateau_rho_mean"]), 9),
            "plateau_target": LEFT_STATE[0],
            "preshock_sine_rel_l1_percent": m["preshock_sine_rel_l1_percent"],
            "mass_t1p8": round(mass, 9),
            "mass_analytic": budget["mass_final_predicted"],
            "mass_error_percent_of_initial": round(mass_err, 9),
            "shock_position_vs_rh_percent": round(shock_err, 6),
            "gamma": GAMMA,
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "generated_by": "validation/repro_shu_osher/smoke_repro.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "repro_smoke.json"
    out.write_text(json.dumps(record, indent=2) + "\n")

    print(f"[smoke] DG P{args.degree} N={args.ncells} steps={record['steps']} "
          f"wall={wall:.1f}s")
    print(f"[smoke] rel_L1 = {rel_l1:.6f}%  (tabulated {expected['rel_l1_percent']:.6f}%, "
          f"drift {drift:.2e} pp)  gate <{TOLERANCE_PERCENT:.0f}% -> "
          f"{'PASS' if rel_l1 < TOLERANCE_PERCENT else 'FAIL'}")
    for k, v in checks.items():
        print(f"[smoke] {'OK  ' if v else 'FAIL'} {k}")
    print(f"[smoke] wrote {out}")
    return 0 if record["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())