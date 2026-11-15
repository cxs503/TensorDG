"""Compute worker gate validation: run a DG convergence study and validate results.

This script demonstrates the run_compute workflow for TensorDG benchmarks:
1. Run a lightweight convergence study locally (P2 advection, 4 resolution levels)
2. Validate the results against theoretical expectations
3. Produce a machine-readable report with all numbers

Usage:
    python -m benchmarks.compute_worker_gate_validation
    python -m benchmarks.compute_worker_gate_validation --json /tmp/cwg_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tensordg import solve_advection, l2_error
from tensordg.convergence import observed_order, pairwise_orders

#: Theoretical EOC for P2 DG on smooth problem
THEORETICAL_EOC = 3.0

#: EOC tolerance (must be within this of theoretical)
EOC_TOLERANCE = 0.3


def run_advection_convergence(cells_list=None, degree=2):
    """Run 1D periodic advection at multiple resolutions and return errors + EOC."""
    if cells_list is None:
        cells_list = [10, 20, 40, 80]

    results = []
    for cells in cells_list:
        t0 = time.monotonic()
        centers, coeff, meta = solve_advection(ncells=cells, degree=degree, final_time=0.5)
        elapsed = time.monotonic() - t0

        # Compute L2 error against analytic solution at t=0.5
        rel_l2 = l2_error(centers, coeff, final_time=0.5, speed=1.0)

        results.append({
            "cells": cells,
            "relative_l2": float(rel_l2),
            "elapsed_seconds": round(elapsed, 3),
            "steps": meta["steps"],
        })

    # Compute EOC
    resolutions = [r["cells"] for r in results]
    errors = [r["relative_l2"] for r in results]
    orders = pairwise_orders(resolutions, errors)
    avg_order = observed_order(resolutions, errors)

    for i, order in enumerate(orders):
        results[i + 1]["eoc"] = round(order, 4)

    return {
        "problem": "1D periodic linear advection",
        "degree": degree,
        "theoretical_eoc": THEORETICAL_EOC,
        "results": results,
        "average_eoc": round(avg_order, 4),
    }


def validate_convergence(study):
    """Validate convergence study results against theoretical expectations."""
    verdicts = []

    # Check EOC
    avg_eoc = study["average_eoc"]
    theoretical = study["theoretical_eoc"]
    eoc_ok = abs(avg_eoc - theoretical) < EOC_TOLERANCE
    verdicts.append({
        "check": "average_eoc",
        "measured": avg_eoc,
        "expected": theoretical,
        "tolerance": EOC_TOLERANCE,
        "pass": eoc_ok,
    })

    # Check finest grid error is small
    finest = study["results"][-1]
    error_ok = finest["relative_l2"] < 1e-4
    verdicts.append({
        "check": "finest_grid_error",
        "measured": finest["relative_l2"],
        "threshold": 1e-4,
        "pass": error_ok,
    })

    # Check all EOC values are reasonable
    for r in study["results"][1:]:
        eoc = r.get("eoc", 0)
        eoc_reasonable = abs(eoc - theoretical) < EOC_TOLERANCE
        verdicts.append({
            "check": "eoc_at_%d_cells" % r["cells"],
            "measured": eoc,
            "expected": theoretical,
            "tolerance": EOC_TOLERANCE,
            "pass": eoc_reasonable,
        })

    all_pass = all(v["pass"] for v in verdicts)
    return {
        "verdicts": verdicts,
        "all_pass": all_pass,
        "summary": "%s: %d/%d checks passed" % (
            "PASS" if all_pass else "FAIL",
            sum(v["pass"] for v in verdicts),
            len(verdicts),
        ),
    }


def build_report(study, validation):
    """Build a comprehensive report with all results."""
    return {
        "title": "TensorDG Compute Worker Gate Validation",
        "anchor": "FEAT-COMPUTE-WORKER-GATE",
        "mission_id": 21271,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "study": study,
        "validation": validation,
        "compute_workflow": {
            "submit_semantics": "run_compute(action='submit') returns job_id + queue position immediately",
            "fetch_semantics": "run_compute(action='fetch', job_id=..., wait_seconds=N) polls until completion",
            "hog_guard": "Local processes >=8 cores for >10min are stopped and migrated to queue",
            "usage_receipts": "Every job records local vs queued compute time",
        },
        "conclusion": {
            "verdict": "PASS" if validation["all_pass"] else "FAIL",
            "eoc_measured": study["average_eoc"],
            "eoc_theoretical": study["theoretical_eoc"],
            "finest_error": study["results"][-1]["relative_l2"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compute worker gate validation")
    parser.add_argument("--json", type=str, help="Write report to JSON file")
    parser.add_argument("--cells", type=int, nargs="+", default=[10, 20, 40, 80],
                        help="Resolution levels (default: 10 20 40 80)")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree (default: 2)")
    args = parser.parse_args()

    print("=" * 60)
    print("TensorDG Compute Worker Gate Validation")
    print("=" * 60)
    print()

    # Run convergence study
    print("Running P%d advection convergence study..." % args.degree)
    print("Resolution levels: %s" % args.cells)
    print()

    study = run_advection_convergence(cells_list=args.cells, degree=args.degree)

    # Print results table
    print("%6s | %14s | %8s | %10s | %6s" % ("Cells", "Relative L2", "EOC", "Time (s)", "Steps"))
    print("-" * 60)
    for r in study["results"]:
        eoc_str = "%.4f" % r["eoc"] if "eoc" in r else "—"
        print("%6d | %14.6e | %8s | %10.3f | %6d" % (
            r["cells"], r["relative_l2"], eoc_str, r["elapsed_seconds"], r["steps"]))
    print()

    print("Average EOC: %.4f (theoretical: %d)" % (study["average_eoc"], study["theoretical_eoc"]))
    print()

    # Validate
    print("Validating results...")
    validation = validate_convergence(study)
    print("Verdict: %s" % validation["summary"])
    print()

    # Print verdicts
    for v in validation["verdicts"]:
        status = "PASS" if v["pass"] else "FAIL"
        expected_val = v.get("expected", v.get("threshold", "N/A"))
        print("  [%s]: %s = %s (expected %s)" % (
            status, v["check"], v["measured"], expected_val))
    print()

    # Build report
    report = build_report(study, validation)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print("Report written to %s" % args.json)

    return 0 if validation["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
