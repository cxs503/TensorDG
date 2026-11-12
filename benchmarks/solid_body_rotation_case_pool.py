"""CASE-DG-SOLID-BODY-ROTATION-2D -- case-pool benchmark driver.

Anchor: ``CASE-DG-SOLID-BODY-ROTATION-2D`` (算例池·二维余弦钟刚体旋转输运).

Physics
-------
Advection of the LeVeque cosine bell (C1, compact support radius R=0.15,
centre (0.5, 0.75)) by rigid rotation

    a(x,y) = omega * (-(y - 1/2), x - 1/2),   omega = 2*pi  =>  period T = 1.

After exactly one period every characteristic is a closed circle, so the exact
solution is ``u(T) = u(0)``: the initial bell is recovered to round-off.  This
makes the case a *self-anchored* reference (no third-party solver needed).

Why the error norm is a disk and not the whole square
-----------------------------------------------------
The rigid-rotation velocity is **not** periodic across the seam
``x=0==1`` / ``y=0==1``: it jumps by ``omega`` there.  The characteristics are
circles about the centre, so a particle never crosses the seam, but the DG
numerical flux on the seam faces does exchange data with a velocity that
disagrees by ``omega``.  That injects an O(1) error layer at the domain edges;
a global norm is therefore polluted and only converges while the transported
bell happens to stay away from the edges.  All *gate* metrics below are
restricted to the disk ``|x - (1/2,1/2)| <= 0.45``, which contains the whole
bell trajectory (annulus r in [0.10, 0.40]) and excludes only the corner/edge
strips.  The global norm is still computed and reported (never gated) and the
``coverage`` field records exactly which share of the analytic L1 mass the disk
keeps, so the masking is auditable rather than silent.

Pre-registration
----------------
Every tolerance in ``GATES`` was fixed before the declared mesh was run (see
git history: the driver lands with these constants, the artifact JSON is
produced afterwards from it).  Nothing is tuned to the result.

Usage
-----
    python benchmarks/solid_body_rotation_case_pool.py            # full ladder
    python benchmarks/solid_body_rotation_case_pool.py --only 100:2
    python benchmarks/solid_body_rotation_case_pool.py --shard 0/2
    python benchmarks/solid_body_rotation_case_pool.py --merge
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from tensordg.advection_2d_var import (
    DGAdvection2D,
    rotated_profile,
    solid_body_rotation,
)
from tensordg.advection_diffusion import cosine_bell

# ---------------------------------------------------------------------------
# Case definition (fixed, not CLI-tunable: the anchor is a *named* case)
# ---------------------------------------------------------------------------
ANCHOR = "CASE-DG-SOLID-BODY-ROTATION-2D"
OMEGA = 2.0 * math.pi          # period T = 1
CENTER = (0.5, 0.5)
BELL_CENTER = (0.5, 0.75)
BELL_RADIUS = 0.15
FINAL_TIME = 1.0               # exactly one revolution
CFL = 0.5
DISK_RADIUS = 0.45             # contains the trajectory annulus [0.10, 0.40]
OUT_DEFAULT = "validation/solid_body_rotation_2d/results.json"

# Mesh ladder: coarse -> declared.  (16,1) doubles as the negative control.
LADDER = [(16, 1), (25, 1), (50, 1), (100, 1), (16, 2), (25, 2), (50, 2), (100, 2)]
DECLARED = (100, 2)

# ---------------------------------------------------------------------------
# Pre-registered gates
# ---------------------------------------------------------------------------
GATES = {
    "G1_DECLARED": {
        "description": "declared acceptance case (100x100, P2, CFL=0.5) disk-relative L1 error",
        "config": DECLARED,
        "metric": "relative_l1_disk",
        "op": "<",
        "threshold": 0.05,
    },
    "G2_MASS": {
        "description": "global mass drift over one revolution (all configs)",
        "metric": "mass_drift_rel",
        "op": "<",
        "threshold": 1e-12,
    },
    "G3_PEAK": {
        "description": "peak retention after one revolution at the declared mesh",
        "config": DECLARED,
        "metric": "peak_retention",
        "op": ">",
        "threshold": 0.95,
    },
    "G4_ORDER_P2": {
        "description": "observed L1 convergence order, P2 over n=25->100 (L1 design order p+1=3)",
        "metric": "order_p2",
        "op": ">=",
        "threshold": 2.0,
    },
    "G5_ORDER_P1": {
        "description": "observed L1 convergence order, P1 over n=25->100 (L1 design order p+1=2)",
        "metric": "order_p1",
        "op": ">=",
        "threshold": 1.2,
    },
    "G6_SEAM": {
        "description": "global norm must exceed the disk norm at the declared mesh "
                       "(documents the seam pollution; if it ever inverts, the geometry "
                       "assumption behind the disk norm is broken)",
        "config": DECLARED,
        "metric": "global_minus_disk_rel_l1",
        "op": ">",
        "threshold": 0.0,
    },
    "G7_NEGCTRL": {
        "description": "negative control: the coarse P1 mesh must FAIL the 5% line",
        "config": (16, 1),
        "metric": "relative_l1_disk",
        "op": ">",
        "threshold": 0.05,
    },
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


def _norm_pair(ncells: int, degree: int, cfl: float = CFL,
               dtype=torch.float64) -> dict:
    """Run one (ncells, degree) configuration and return its metrics record."""
    bell = cosine_bell(BELL_CENTER, BELL_RADIUS)
    profile = lambda x, y, t=0.0: bell(x, y)             # noqa: E731
    velocity = solid_body_rotation(OMEGA, CENTER)
    exact = rotated_profile(profile, OMEGA, CENTER)

    solver = DGAdvection2D(degree=degree, ncells=ncells, dtype=dtype)
    coeff0 = solver.project(profile)
    t0 = time.perf_counter()
    out = solver.solve(coeff0, velocity, final_time=FINAL_TIME, cfl=cfl)
    wall = time.perf_counter() - t0
    coeff, meta = out["coeff"], out["meta"]

    disk_l1 = solver.l1_error(coeff, exact, FINAL_TIME, radius=DISK_RADIUS, center=CENTER)
    disk_l2 = solver.l2_error(coeff, exact, FINAL_TIME, radius=DISK_RADIUS, center=CENTER)
    glob_l1 = solver.l1_error(coeff, exact, FINAL_TIME, radius=None, center=CENTER)
    proj_l1 = solver.l1_error(coeff0, exact, 0.0, radius=DISK_RADIUS, center=CENTER)
    max_cell, max_node = solver.peak(coeff)
    peak_ref = float(bell(torch.tensor([[BELL_CENTER[0]]]), torch.tensor([[BELL_CENTER[1]]])).item())

    return {
        "ncells": int(ncells),
        "degree": int(degree),
        "cfl": float(cfl),
        "steps": int(meta["steps"]),
        "dt": float(meta["dt"]),
        "dofs": int(meta["dofs"]),
        "wall_s": round(wall, 3),
        "metrics": {
            "relative_l1_disk": disk_l1["relative_l1"],
            "absolute_l1_disk": disk_l1["absolute_l1"],
            "reference_l1_disk": disk_l1["reference_l1"],
            "coverage": disk_l1["coverage"],
            "relative_l2_disk": disk_l2["relative_l2"],
            "relative_l1_global": glob_l1["relative_l1"],
            "global_minus_disk_rel_l1": glob_l1["relative_l1"] - disk_l1["relative_l1"],
            "projection_relative_l1_disk": proj_l1["relative_l1"],
            "mass_drift_rel": float(meta["mass_drift_rel"]),
            "mass_drift_abs": float(meta["mass_drift_abs"]),
            "peak_cell_average": max_cell,
            "peak_nodal": max_node,
            "peak_retention": max_node / peak_ref if peak_ref else 0.0,
        },
    }


def _observed_order(runs: list[dict], degree: int) -> dict:
    """Least-squares slope of log(relative_l1_disk) vs log(ncells), and n=25->100 pair float."""
    pts = sorted((r["ncells"], r["metrics"]["relative_l1_disk"])
                 for r in runs if r["degree"] == degree and r["metrics"]["relative_l1_disk"] > 0)
    out = {"degree": degree, "points": [[n, e] for n, e in pts], "n_points": len(pts)}
    if len(pts) >= 2:
        xs = [math.log(n) for n, _ in pts]
        ys = [math.log(e) for _, e in pts]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        out["fit_slope"] = -num / den if den else float("nan")
    pair = {n: e for n, e in pts}
    if 25 in pair and 100 in pair and pair[100] > 0:
        out["pair_slope_25_100"] = math.log(pair[25] / pair[100]) / math.log(100.0 / 25.0)
    return out


def _lookup(runs: list[dict], config: tuple[int, int]) -> dict | None:
    for r in runs:
        if (r["ncells"], r["degree"]) == config:
            return r
    return None


def _value(runs: list[dict], metric: str, orders: dict) -> float | None:
    if metric in ("order_p1", "order_p2"):
        deg = int(metric[-1])
        block = orders.get(deg, {})
        # Preferred estimator: least-squares slope over the whole measured ladder
        # (>= 3 points).  Falls back to the n=25->100 pair slope when only two
        # points exist.  Both are recorded in the artifact under "orders".
        if block.get("n_points", 0) >= 3:
            return block.get("fit_slope")
        return block.get("pair_slope_25_100")
    return None


OPS = {"<": lambda a, b: a < b, ">": lambda a, b: a > b,
       ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b}


def _evaluate(runs: list[dict]) -> dict:
    orders = {d: _observed_order(runs, d) for d in sorted({r["degree"] for r in runs})}
    results = {}
    for name, gate in GATES.items():
        metric = gate["metric"]
        if metric in ("order_p1", "order_p2"):
            got = _value(runs, metric, orders)
        else:
            run = _lookup(runs, tuple(gate["config"])) if "config" in gate else None
            if run is None and name == "G2_MASS":
                # all configs must respect it
                worst = max(runs, key=lambda r: r["metrics"]["mass_drift_rel"], default=None)
                got = worst["metrics"]["mass_drift_rel"] if worst else None
                run = worst
            elif run is None:
                got = None
            else:
                got = run["metrics"][metric]
        if got is None:
            results[name] = {"status": "NOT_RUN", "metric": metric,
                             "threshold": gate["threshold"], "op": gate["op"],
                             "description": gate["description"]}
            continue
        ok = bool(OPS[gate["op"]](got, gate["threshold"]))
        results[name] = {"status": "PASS" if ok else "FAIL", "metric": metric,
                         "got": got, "threshold": gate["threshold"], "op": gate["op"],
                         "description": gate["description"]}
    ran = [k for k, v in results.items() if v["status"] != "NOT_RUN"]
    measured_pass = bool(ran) and all(results[k]["status"] == "PASS" for k in ran)
    # A gate that was never executed is NOT a pass: the full verdict needs every
    # gate to be measured.  Both flags are reported so that a partial ladder can
    # never be quoted as "all gates pass".
    all_pass = measured_pass and len(ran) == len(results)
    return {"orders": {str(k): v for k, v in orders.items()},
            "gate_results": results,
            "verdict": {"all_gates_pass": all_pass,
                        "measured_gates_pass": measured_pass,
                        "n_measured": len(ran),
                        "any_not_run": any(v["status"] == "NOT_RUN" for v in results.values()),
                        "n_gates": len(results)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default=None, help="i/n -- run only shard i of n")
    ap.add_argument("--only", default=None, help="n:p single configuration, e.g. 100:2")
    ap.add_argument("--configs", default=None,
                    help="comma-separated n:p list, e.g. 16:1,25:1,50:2")
    ap.add_argument("--merge", action="store_true", help="merge results_shard*.json")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--outdir", default="validation/solid_body_rotation_2d")
    args = ap.parse_args(argv)

    out = Path(args.out)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        runs: list[dict] = []
        paths = [p for p in sorted(glob.glob(str(outdir / "results*.json")))
                 if Path(p).name != out.name]
        if not paths:
            raise SystemExit(f"no shard artifacts under {outdir}")
        for path in paths:
            data = json.loads(Path(path).read_text())
            runs.extend(data["runs"])
        runs.sort(key=lambda r: (r["degree"], r["ncells"]))
        return _write(out, runs, argv)

    configs = LADDER
    if args.configs:
        configs = [tuple(int(x) for x in tok.split(":")) for tok in args.configs.split(",")]
    if args.only:
        n, p = args.only.split(":")
        configs = [(int(n), int(p))]

    shard_tag = ""
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        configs = [c for k, c in enumerate(configs) if k % n == i]
        shard_tag = f"_shard{i}of{n}"
        out = outdir / f"results_shard{i}of{n}.json"

    runs = []
    for ncells, degree in configs:
        print(f"[run] n={ncells:4d} p={degree} cfl={CFL} T={FINAL_TIME}", flush=True)
        rec = _norm_pair(ncells, degree)
        m = rec["metrics"]
        print(f"  steps={rec['steps']:6d} wall={rec['wall_s']:8.2f}s "
              f"L1disk={m['relative_l1_disk']:.6e} L1glob={m['relative_l1_global']:.6e} "
              f"cov={m['coverage']:.4f} peak={m['peak_retention']:.6f}", flush=True)
        runs.append(rec)

    if shard_tag or args.configs or args.only:
        payload = {"anchor": ANCHOR,
                   "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "provenance": {"git_commit": _git_commit(), "host": socket.gethostname(),
                                  "torch": torch.__version__, "platform": platform.platform()},
                   "runs": runs}
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[ok] shard written to {out}")
        return 0
    return _write(out, runs, argv)


def _write(out: Path, runs: list[dict], argv) -> int:
    runs.sort(key=lambda r: (r["degree"], r["ncells"]))
    evaluation = _evaluate(runs)
    payload = {
        "anchor": ANCHOR,
        "case": {
            "equation": "u_t + div(a u) = 0 (periodic unit square)",
            "velocity": "rigid rotation a = omega*(-(y-1/2), x-1/2)",
            "omega": OMEGA,
            "final_time": FINAL_TIME,
            "cfl": CFL,
            "initial_condition": "LeVeque cosine bell, centre (0.5,0.75), R=0.15",
            "exact_solution": "u(T=1) = u(0) (closed characteristics, one full revolution)",
            "norm": f"relative L1 on the disk |x-(0.5,0.5)| <= {DISK_RADIUS}",
            "disk_radius": DISK_RADIUS,
            "declared_config": list(DECLARED),
            "ladder": [list(c) for c in LADDER],
        },
        "gates_preregistered": GATES,
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "argv": argv if argv is not None else sys.argv[1:],
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "nproc": os.cpu_count(),
            "dtype": "float64",
        },
        "runs": runs,
        **evaluation,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    v = evaluation["verdict"]
    print(f"[ok] written {out} | all_gates_pass={v['all_gates_pass']} any_not_run={v['any_not_run']}")
    for k, g in evaluation["gate_results"].items():
        print(f"   {k:14s} {g['status']:8s} {g.get('got')}")
    return 0 if v["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())