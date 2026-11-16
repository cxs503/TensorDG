"""Solve one CASE-LAX-DG hold-out leg and cache its leg record verbatim.

Why this exists: the hold-out measurement is a ~90-210 s single-core solve, i.e.
longer than a local short-test window and (on 2026-11-16) longer than the cluster
queue was willing to dispatch.  Splitting the ladder across processes keeps every
completed leg on disk, so an interrupted run never throws away a solve that was
already paid for.  ``entries``/``format_legs`` are the driver's own formatting
path, so a cached leg cannot be shaped differently from an in-process one.

Usage::

    python3 scripts/solve_holdout_leg.py 600 2      # p2.N600
    python3 scripts/solve_holdout_leg.py 600 3      # p3.N600
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.lax_case_pool import run_ladder          # noqa: E402
from benchmarks.lax_certified_holdout import format_legs  # noqa: E402

CACHE = ROOT / "runs" / "holdout_legs.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", type=int)
    parser.add_argument("degree", type=int)
    args = parser.parse_args(argv)

    started = time.time()
    _, _, entries, _, _ = run_ladder(ladder=[(args.cells, args.degree)])
    rows = format_legs(entries)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else []
    cache = [row for row in cache if row["leg"] != rows[0]["leg"]] + rows
    cache.sort(key=lambda row: (row["dof"], row["degree"]))
    CACHE.write_text(json.dumps(cache, indent=1) + "\n")
    print(json.dumps(rows[0], indent=1))
    print(f"[leg] cached {rows[0]['leg']} -> {CACHE} "
          f"(process wall {time.time() - started:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())