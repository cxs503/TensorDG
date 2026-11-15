#!/usr/bin/env python3
"""Aggregate the AO arm into the scheme_variants metrics after the cluster job completes.

Usage::

    python scripts/aggregate_ao_arm.py          # re-aggregate if ao.json exists
    python scripts/aggregate_ao_arm.py --check  # verify consistency

This script is idempotent: if ao.json is already aggregated, it re-aggregates
from scratch (all legs) to ensure consistency.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGS = ROOT / "docs" / "benchmarks" / "results" / "lax_scheme_variants" / "legs"
METRICS = ROOT / "docs" / "benchmarks" / "results" / "lax_scheme_variants" / "metrics.json"
DOC = ROOT / "docs" / "CASE-LAX-DG-scheme-variants.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify consistency only")
    args = parser.parse_args()

    ao_leg = LEGS / "ao.json"
    if not ao_leg.exists():
        print(f"ERROR: {ao_leg} not found. The cluster job (cj4072) may still be running.",
              file=sys.stderr)
        sys.exit(1)

    # Re-aggregate using the benchmark module
    sys.path.insert(0, str(ROOT))
    from benchmarks import lax_scheme_variants as study

    if args.check:
        # Just verify the existing metrics are consistent with the legs
        try:
            study.check()
            print("OK: metrics.json is consistent with leg files.")
        except AssertionError as e:
            print(f"FAIL: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Re-aggregate
    print(f"Re-aggregating with AO leg from {ao_leg}...")
    metrics = study.aggregate(allow_missing=False)

    # Write metrics
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Updated {METRICS}")

    # Re-generate doc
    study.report(metrics=metrics)
    print(f"Updated {DOC}")

    # Print summary
    ao_row = next(r for r in metrics["rows"] if r["variant"] == "ao")
    verdict = metrics["verdict"]
    print(f"\nAO arm results:")
    print(f"  steps: {ao_row['steps']}")
    print(f"  worst_l2: {ao_row['worst_l2']:.6f} ({ao_row['worst_field']})")
    print(f"  L2 density: {ao_row['l2']['density']:.6f}")
    print(f"  L2 velocity: {ao_row['l2']['velocity']:.6f}")
    print(f"  L2 pressure: {ao_row['l2']['pressure']:.6f}")
    print(f"  admissible: {ao_row['decision']['admissible']}")
    print(f"\nVerdict:")
    print(f"  admissible_arms: {verdict['admissible_arms']}")
    print(f"  arms_not_run: {verdict['arms_not_run']}")
    print(f"  arms_errored: {verdict['arms_errored']}")
    print(f"  scope_complete: {verdict['scope_complete']}")


if __name__ == "__main__":
    main()
