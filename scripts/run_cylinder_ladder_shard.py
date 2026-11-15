"""Shard runner for ``benchmarks.cylinder_stag_ladder`` (cluster job entry point).

Runs the ladder cases with ``index %% shards == shard %% shards`` so that a
``shards``-way split covers the whole ladder exactly once regardless of whether
the scheduler numbers the shard argument from 0 or from 1.

    PYTHONPATH=. python scripts/run_cylinder_ladder_shard.py --shard {i} --shards {n}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.cylinder_stag_ladder import OUT, default_ladder, run_case


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--max-time", type=float, default=8.0)
    ap.add_argument("--steady-tol", type=float, default=1e-3)
    args = ap.parse_args()
    cases = default_ladder()
    mine = [c for i, c in enumerate(cases) if i % args.shards == args.shard % args.shards]
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"shard": args.shard, "shards": args.shards,
                      "cases": [f"m{c['mach']:g}-p{c['p']}-{c['nr']}x{c['ntheta']}"
                                for c in mine]}), flush=True)
    for case in mine:
        t0 = time.perf_counter()
        row = run_case(max_time=args.max_time, steady_tol=args.steady_tol, **case)
        (OUT / f"row_{row['tag']}.json").write_text(json.dumps(row, indent=1))
        print(json.dumps({"tag": row["tag"], "wall_seconds": row["wall_seconds"],
                          "steady": row["steady"], "cp_stag": row["cp_stag"],
                          "target": row["cp_stag_target"],
                          "rel_error": row["cp_stag_rel_error"],
                          "symmetry_error": row["symmetry_error"],
                          "mass_drift_rel": row["mass_drift_rel"],
                          "residual_ratio": row["residual_ratio"],
                          "driver_seconds": time.perf_counter() - t0}), flush=True)


if __name__ == "__main__":
    main()