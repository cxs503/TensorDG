"""Compare an independent re-run of the CASE-EULER-VORTEX records with the stored ones.

Usage::

    python3 tools/compare_rerun.py case_euler_vortex audit_repro

Prints one line per configuration shared by both result directories and exits
non-zero if any re-run error differs from the stored value by more than
``RTOL`` (the numbers in the report must be reproducible, not merely plausible).
"""

from __future__ import annotations

import glob
import json
import os
import sys

RTOL = 1e-6


def load(directory: str) -> dict:
    out = {}
    for path in glob.glob(os.path.join(directory, "results", "*.json")):
        rec = json.load(open(path))
        if rec.get("task") == "residual":
            continue
        key = (rec["task"], rec["degree"], rec["cells"], round(rec["final_time"], 9), round(rec["cfl"], 9))
        out[key] = rec
    return out


def main() -> int:
    stored_dir = sys.argv[1] if len(sys.argv) > 1 else "case_euler_vortex"
    rerun_dir = sys.argv[2] if len(sys.argv) > 2 else "audit_repro"
    stored, rerun = load(stored_dir), load(rerun_dir)
    shared = sorted(set(stored) & set(rerun))
    worst = 0.0
    bad = 0
    print(f"{'config':34s} {'stored_max':>14s} {'rerun_max':>14s} {'rel_diff':>10s}")
    for key in shared:
        a, b = stored[key], rerun[key]
        rel = abs(a["max_relative_l2"] - b["max_relative_l2"]) / abs(a["max_relative_l2"])
        worst = max(worst, rel)
        bad += rel > RTOL
        label = b.get("task_key") or "-".join(str(x) for x in key)
        print(f"{label:34s} {a['max_relative_l2']:14.9e} {b['max_relative_l2']:14.9e} {rel:10.2e}")
    print(f"shared={len(shared)} worst_rel_diff={worst:.3e} rtol={RTOL:.0e} mismatches={bad}")
    print(f"rerun_only={sorted(set(rerun) - set(stored))}")
    return 0 if shared and bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())