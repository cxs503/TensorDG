"""Refinement ladders for the 1-D shock tubes used by the verification chain.

Writes ``validation/verification_chain/shock_tube_refinement.json`` with the
measured cell-average errors (against the exact Riemann cell averages) at
several resolutions plus the observed pairwise convergence orders.  Every
number is produced here; nothing is hand-copied.

Run:  PYTHONPATH=. python3 validation/verification_chain/shock_tube_refinement.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from benchmarks.riemann_exact import cell_averages
from tensordg.euler_1d import GAMMA
from tensordg.euler_dg_1d import solve_euler_dg_1d

FIELDS = ("density", "velocity", "pressure")

CASES = {
    "sod": {"left": (1.0, 0.0, 1.0), "right": (0.125, 0.0, 0.1),
            "final_time": 0.2, "degree": 2, "resolutions": [50, 100, 200, 400]},
    "lax": {"left": (0.445, 0.698, 3.528), "right": (0.5, 0.0, 0.571),
            "final_time": 0.13, "degree": 2, "resolutions": [100, 200, 400]},
}


def _order(resolutions, errors):
    h = [1.0 / n for n in resolutions]
    return [float(np.log(errors[i] / errors[i + 1]) / np.log(h[i] / h[i + 1]))
            for i in range(len(errors) - 1)]


def run_case(name, spec):
    rows = []
    for ncells in spec["resolutions"]:
        t0 = time.time()
        _, coeff, meta = solve_euler_dg_1d(
            spec["left"], spec["right"], ncells=ncells, degree=spec["degree"],
            final_time=spec["final_time"], cfl=0.15,
            limiter="tvb", limiter_stage="final",
        )
        avg = coeff[:, 0]
        rho = avg[:, 0].clamp_min(1e-12)
        pressure = (GAMMA - 1.0) * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / rho)
        numeric = {"density": avg[:, 0].cpu().numpy(),
                   "velocity": (avg[:, 1] / rho).cpu().numpy(),
                   "pressure": pressure.cpu().numpy()}
        _, exact = cell_averages(ncells, spec["final_time"], spec["left"], spec["right"])
        rel = {k: float(np.sqrt(np.mean((numeric[k] - exact[k]) ** 2))
                        / np.sqrt(np.mean(exact[k] ** 2))) for k in FIELDS}
        rows.append({"ncells": ncells, "relative_l2": rel,
                     "steps": int(meta["steps"]), "wall_seconds": round(time.time() - t0, 2)})
        print(f"{name} ncells={ncells} " + " ".join(f"{k}={rel[k]*100:.4f}%" for k in FIELDS),
              flush=True)
    order = {k: _order(spec["resolutions"], [r["relative_l2"][k] for r in rows]) for k in FIELDS}
    return {"left": list(spec["left"]), "right": list(spec["right"]),
            "final_time": spec["final_time"], "degree": spec["degree"],
            "cfl": 0.15, "limiter": "tvb", "limiter_stage": "final",
            "reference": "exact Riemann cell averages (benchmarks/riemann_exact.cell_averages, 8-point Gauss)",
            "rows": rows, "pairwise_order": order,
            "mean_order": {k: float(np.mean(v)) for k, v in order.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(CASES),
                        help="comma separated subset of " + ",".join(CASES))
    parser.add_argument("--resolutions", default="",
                        help="optional override, e.g. 50,100 (applied to every case)")
    args = parser.parse_args(argv)
    cases = {}
    for name in args.cases.split(","):
        spec = dict(CASES[name])
        if args.resolutions:
            spec["resolutions"] = [int(v) for v in args.resolutions.split(",")]
        cases[name] = run_case(name, spec)
    out = {"cases": cases,
           "provenance": {"generated_by": "validation/verification_chain/shock_tube_refinement.py",
                          "argv": [args.cases, args.resolutions]}}
    path = pathlib.Path(__file__).with_name("shock_tube_refinement.json")
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()