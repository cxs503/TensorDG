"""Which committed Lax artifact is stale: ``lax_case_pool`` or ``lax_asymptotic_probe``?

The two committed artifacts record the *same* legs (``p3.N400``, ``p3.N800``) but
disagree in whole-field relative L1 by 1.06e-04 / 7.14e-03 relative.  The
resolution-requirement note blamed the recorded ``dt``; that reading is wrong --
``tensordg.euler_dg_1d`` returns ``dt = dt_last`` (the *time-truncated* final
step), explicitly documented as not representative.  So the disagreement has to
be adjudicated on the fields, by re-running the legs with the current code.

This script prints, for each requested leg, the readings of the current code
next to both committed artifacts.  It writes nothing into the repository.

Usage::

    PYTHONPATH=. python scripts/diag_lax_probe_staleness.py --legs 400:3,800:3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.lax_asymptotic_probe import measure_leg
from benchmarks.lax_case_pool import OUT as POOL_OUT, exact_waves

PROBE_OUT = Path("docs/benchmarks/results/lax_asymptotic_probe")


def _load(path):
    return json.loads(Path(path).read_text())


def _find(entries, cells, degree):
    return next((e for e in entries if e["cells"] == cells and e["degree"] == degree), None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legs", default="400:3,800:3")
    args = parser.parse_args()
    legs = [tuple(int(v) for v in part.split(":")) for part in args.legs.split(",")]

    pool = _load(Path(POOL_OUT) / "metrics.json")
    probe = _load(PROBE_OUT / "metrics.json")
    waves = exact_waves()

    rows = []
    for cells, degree in legs:
        leg, _ = measure_leg(cells, degree, waves)
        row = {
            "cells": cells, "degree": degree,
            "steps": leg["steps"], "dt_last": leg.get("dt_last", leg.get("dt")),
            "dt_mean": 0.13 / leg["steps"],
            "errors": leg["errors"], "analytic_errors": leg["analytic_errors"],
        }
        for name, book in (("pool", pool), ("probe", probe)):
            entry = _find(book["ladder"], cells, degree)
            if entry is None:
                row[name] = None
                continue
            row[name] = {
                "errors": entry["errors"],
                "steps": entry["steps"], "dt_last": entry.get("dt_last", entry.get("dt")),
                "rel_diff_vs_current": {
                    f: abs(leg["errors"][f] - entry["errors"][f]) / abs(entry["errors"][f])
                    for f in ("density", "velocity", "pressure")},
            }
        rows.append(row)
        print(json.dumps(row, indent=1, sort_keys=True), flush=True)

    print(json.dumps({"rows": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())