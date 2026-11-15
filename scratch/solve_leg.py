"""Solve ONE hold-out leg and cache it verbatim (operational helper, not a deliverable).

Usage:  python3 scratch/solve_leg.py <cells> <degree>
Writes scratch/leg_<p><N>.json with the raw ``run_ladder`` entry.
The recorded-leg formatting stays in benchmarks/lax_certified_holdout.format_legs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmarks.lax_case_pool import run_ladder

cells, degree = int(sys.argv[1]), int(sys.argv[2])
_, _, entries, _, _ = run_ladder(ladder=[(cells, degree)])
entry = entries[0]
out = Path("scratch") / f"leg_p{degree}N{cells}.json"
out.write_text(json.dumps(entry, indent=1) + "\n")
print("cached", out, "worst", max(entry["l2_errors"].values()))