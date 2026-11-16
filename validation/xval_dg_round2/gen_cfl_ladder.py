"""Generate the CFL-refinement levels the time-pollution audit (T0-T4) consumes.

Context
-------
``benchmarks/xval_dg_round2_time_pollution.py`` audits how much of the round-2
reference band is *temporal* discretisation error, but its real-data branch has
never been fed: the published ladder ``fv_cfl0p3_n800/1600/3200/6400.npz`` was
produced at a **single** CFL, and the CFL-refined profiles were submitted to the
cluster (cj4305-cj4308) and never dispatched.  The audit therefore stands at
``WITHHELD`` -- the engine and its synthetic ground truth are certified, the
numbers about the real reference do not exist.

This driver is the missing producer.  It runs the *same* call path that produced
the committed ladder (``benchmarks.shu_osher_case_pool.run_fv`` ->
``benchmarks.fv_weno5_1d.solve_fv_weno5``, CFL 0.15/0.075 instead of 0.3) so the
new levels are ladder-consistent by construction, and names them exactly as
:func:`benchmarks.xval_dg_round2_time_pollution.level_path` expects
(``fv_cfl<cfl tag>_n<ncells>.npz``).

Sharding
--------
One level costs 70 s (N=800, CFL=0.15) to ~22 min (N=3200, CFL=0.075), i.e. the
whole table is ~1.2 CPU-hours: a cluster job, not a local one.  ``--shard i
--nshards n`` runs the cases ``i, i+n, i+2n, ...`` of :data:`CASES`, so the table
can be dispatched as independent single-core jobs.

Cases already on disk with a finite, positive density profile are skipped, so
re-dispatching a partial shard is safe and idempotent.

Usage::

    python validation/xval_dg_round2/gen_cfl_ladder.py --list
    python validation/xval_dg_round2/gen_cfl_ladder.py --shard 0 --nshards 6
    python validation/xval_dg_round2/gen_cfl_ladder.py --ncells 200 --cfl 0.3 \
        --out /tmp/smoke.npz          # local smoke, ~10 s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "validation" / "repro_shu_osher" / "artifacts" / "raw"

#: ``(ncells, cfl)`` table -- exactly the levels the T0-T4 audit needs.
#: Sorted coarsest/loosest first so a partial run still leaves useful data.
CASES: list[tuple[int, float]] = [
    (800, 0.075),
    (1600, 0.15),
    (1600, 0.075),
    (3200, 0.15),
    (3200, 0.075),
    (6400, 0.15),
]


def cfl_tag(cfl: float) -> str:
    """File-name tag of a CFL number (``0.075`` -> ``0p075``); mirrors the audit."""
    return f"{float(cfl):g}".replace(".", "p")


def target_path(ncells: int, cfl: float, rawdir: Path = RAW) -> Path:
    return Path(rawdir) / f"fv_cfl{cfl_tag(cfl)}_n{int(ncells)}.npz"


def level_is_usable(path: Path) -> bool:
    """A level is reusable when it parses and carries a positive density field."""
    if not path.exists():
        return False
    try:
        blob = np.load(path, allow_pickle=True)
        rho = np.asarray(blob["rho"], dtype=float)
    except Exception:
        return False
    return bool(rho.size > 0 and np.all(np.isfinite(rho)) and rho.min() > 0.0)


def solve_level(ncells: int, cfl: float):
    """Run the reference solver on one (ncells, cfl) pair; returns (x, u, meta)."""
    from benchmarks.fv_weno5_1d import solve_fv_weno5
    from benchmarks.shu_osher_case_pool import DOMAIN, FINAL_TIME, initial_state

    t0 = time.perf_counter()
    x, u, meta = solve_fv_weno5(initial_state, ncells=int(ncells), cfl=float(cfl),
                                final_time=FINAL_TIME, domain=DOMAIN)
    return np.asarray(x, dtype=float), np.asarray(u, dtype=float), {
        "scheme": "FV-WENO5-JS/HLLC/SSP-RK3",
        "ncells": int(ncells), "cfl": float(cfl),
        "final_time": float(FINAL_TIME), "domain": list(DOMAIN),
        "steps": int(meta.get("steps", -1)),
        "wall_time_s": round(time.perf_counter() - t0, 3),
    }


def run_one(ncells: int, cfl: float, out: Path) -> dict:
    x, u, meta = solve_level(ncells, cfl)
    rho, rho_u, rho_e = u[:, 0], u[:, 1], u[:, 2]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, x=x, rho=rho, rho_u=rho_u, rho_e=rho_e, meta=json.dumps(meta))
    return {"out": str(out), "meta": meta,
            "rho_min": float(rho.min()), "rho_max": float(rho.max()),
            "admissible": bool(np.all(np.isfinite(rho)) and rho.min() > 0.0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ncells", type=int, default=None)
    ap.add_argument("--cfl", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rawdir", default=str(RAW))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--list", action="store_true", help="print the case table and exit")
    args = ap.parse_args(argv)

    if args.list:
        print(json.dumps([{"ncells": n, "cfl": c,
                           "path": str(target_path(n, c, Path(args.rawdir))),
                           "present": level_is_usable(target_path(n, c, Path(args.rawdir)))}
                          for n, c in CASES], indent=2))
        return 0

    if args.ncells is not None:  # ad-hoc single level (smoke test / manual)
        out = Path(args.out) if args.out else target_path(args.ncells,
                                                         0.3 if args.cfl is None else args.cfl,
                                                         Path(args.rawdir))
        print(json.dumps(run_one(args.ncells, 0.3 if args.cfl is None else args.cfl, out),
                         indent=2, sort_keys=True))
        return 0

    rawdir = Path(args.rawdir)
    done, skipped = [], []
    for i, (n, c) in enumerate(CASES):
        if i % max(int(args.nshards), 1) != int(args.shard):
            continue
        out = target_path(n, c, rawdir)
        if level_is_usable(out):
            skipped.append({"ncells": n, "cfl": c, "out": str(out), "status": "present"})
            continue
        done.append(run_one(n, c, out))
    print(json.dumps({"shard": int(args.shard), "nshards": int(args.nshards),
                      "produced": done, "skipped": skipped}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())