#!/usr/bin/env python3
"""Attach the ledger acceptance gate A0 to the committed CASE-LAX-DG artifact.

Why post-processing instead of a fresh run
------------------------------------------
A0 is a **pure function of committed numbers**:

* the ``ladder_l1`` arm reads ``metrics.json["ladder"][*]["errors"]``;
* the ``production_l2`` arm re-measures the finest leg out of the committed
  ``profiles.npz`` (primitive fields and exact solution already stored on the
  element Gauss nodes).

So no solver is re-run -- and no number is invented.  The script also records a
consistency check between the two sources: the relative L1 recomputed from
``profiles.npz`` must reproduce the ladder's own relative L1 for the same leg,
otherwise the arm is refused.

Usage::

    python scripts/attach_lax_acceptance.py            # write
    python scripts/attach_lax_acceptance.py --check     # non-zero if absent/stale
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

from benchmarks.lax_case_pool import (  # noqa: E402
    gate_a0, ledger_acceptance_tolerance, production_errors_from_profiles,
    render_acceptance_figure, render_summary)

OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool"
METRICS = OUT / "metrics.json"
PROFILES = OUT / "profiles.npz"
FIGDIR = OUT / "figures"
FIELDS = ("density", "velocity", "pressure")
#: the profiles-based re-measurement must reproduce the ladder's own L1
CONSISTENCY_TOL = 1e-6


def attach(metrics: dict, production: dict) -> dict:
    ladder = metrics["ladder"]
    finest = ladder[-1]
    # Both norms are re-measured from profiles.npz, so both are checked against
    # the ladder's own numbers for the same leg: the L1 arm and the ledger's L2
    # arm must come out of one and the same field array.
    mismatch = {}
    for f in FIELDS:
        mismatch[f] = {
            "l1": abs(production["l1"][f] - finest["errors"][f]) / finest["errors"][f],
            "l2": (abs(production["l2"][f] - finest["l2_errors"][f])
                   / finest["l2_errors"][f]),
        }
    worst = max(v for row in mismatch.values() for v in row.values())
    if worst > CONSISTENCY_TOL:
        raise SystemExit(
            f"profiles.npz does not reproduce the ladder's errors for the finest "
            f"leg (worst relative mismatch {worst:.3e} > {CONSISTENCY_TOL:g}); "
            "refusing to attach an acceptance verdict")
    acceptance = gate_a0(ladder, tolerance=ledger_acceptance_tolerance(),
                         production_leg=finest, production_l2=production["l2"])
    metrics = dict(metrics)
    metrics["acceptance"] = acceptance
    metrics["verdicts"] = {
        "acceptance": acceptance["status"], "verification": metrics["verdict"],
        "reading": ("acceptance = ledger CASE-LAX-DG, relative L2 vs exact < "
                    "registered tolerance; verification = supplementary order "
                    "gates C1/C2/C3/R3/R5/TIME")}
    tolerances = dict(metrics["config"]["tolerances"])
    tolerances["a0_acceptance"] = acceptance["tolerance"]
    config = dict(metrics["config"])
    config["tolerances"] = tolerances
    metrics["config"] = config
    metrics["acceptance_provenance"] = {
        "attached_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "production_leg": acceptance["production_leg"],
        "arm_source": acceptance["arms"]["production_l2"]["source"],
        "profiles_file": str(PROFILES.relative_to(ROOT)),
        "method": ("A0 recomputed from the committed artifact only: ladder L1 arm "
                   "from metrics.json, production L2 arm from profiles.npz; no "
                   "solver re-run, no transcribed number"),
        "profile_consistency": {
            "worst_relative_mismatch": float(worst),
            "worst_relative_mismatch_l1": float(max(r["l1"] for r in mismatch.values())),
            "worst_relative_mismatch_l2": float(max(r["l2"] for r in mismatch.values())),
            "tolerance": CONSISTENCY_TOL,
            "reading": ("both norms re-measured from profiles.npz vs the ladder's "
                        "own numbers for the same leg; the ledger arm is refused "
                        "above the tolerance"),
            "per_field": mismatch},
        "production_errors": production,
    }
    return metrics


def render_outputs(metrics: dict) -> None:
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    (OUT / "summary.md").write_text(render_summary(metrics))
    render_acceptance_figure(metrics["ladder"], metrics["acceptance"], FIGDIR)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    metrics = json.loads(METRICS.read_text())
    if args.check:
        if "acceptance" not in metrics:
            print("no acceptance block in the artifact; run without --check",
                  file=sys.stderr)
            return 1
        production = production_errors_from_profiles(PROFILES)
        rebuilt = attach(metrics, production)
        problems = []
        for key in ("acceptance",):
            if rebuilt[key] != metrics[key]:
                problems.append(key)
        if rebuilt["verdicts"] != metrics["verdicts"]:
            problems.append("verdicts")
        for problem in problems:
            print(f"[lax-a0] stale {problem}", file=sys.stderr)
        return 1 if problems else 0
    production = production_errors_from_profiles(PROFILES)
    metrics = attach(metrics, production)
    render_outputs(metrics)
    acceptance = metrics["acceptance"]
    print(f"[lax-a0] A0={acceptance['status']} ledger arm "
          f"{acceptance['arms']['production_l2']['worst']:.5e} "
          f"({acceptance['production_leg']}, source="
          f"{acceptance['arms']['production_l2']['source']}), L1 arm "
          f"{acceptance['arms']['ladder_l1']['worst']:.5e}; verification "
          f"{metrics['verdict']}; wrote {METRICS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())