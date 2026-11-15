#!/usr/bin/env python3
"""Run all pending CASE-DG-EULER-VORTEX-ERR yee ladder configs.

Usage:
    python run_yee_ladder.py --config-index I  # run single config I
    python run_yee_ladder.py --all              # run all 13 pending configs sequentially
    python run_yee_ladder.py --aggregate        # aggregate results + generate figures
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.case_euler_vortex_yee import (
    CONFIGS, TOLERANCE, PERIOD, GROWTH_LIMIT, REFERENCE,
    run_config, aggregate
)

OUTPUT_DIR = Path("case_euler_vortex_yee")
DOCS_DIR = Path("docs")

# Pending config indices (0-based in CONFIGS list)
# Already done: 4 (dg_d1_n16), 6 (dg_d3_n8)
PENDING_INDICES = [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14]


def run_single(idx: int) -> dict:
    config = CONFIGS[idx]
    print(f"[{idx}/{len(CONFIGS)}] Running {config['id']}...")
    return run_config(config, OUTPUT_DIR)


def run_all():
    results = []
    for idx in PENDING_INDICES:
        started = time.time()
        try:
            rec = run_single(idx)
            elapsed = time.time() - started
            print(f"  DONE in {elapsed:.1f}s: max={rec.get('max_percent', 'N/A')}%, pass={rec.get('gate_pass', 'N/A')}")
            results.append(rec)
        except Exception as e:
            elapsed = time.time() - started
            print(f"  FAILED after {elapsed:.1f}s: {e}")
    print(f"\nCompleted {len(results)}/{len(PENDING_INDICES)} configs")
    return results


def run_aggregate():
    """Aggregate all results and generate figures + report."""
    ret = aggregate(OUTPUT_DIR, docs=DOCS_DIR)
    print(f"Aggregate returned: {ret}")
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-index", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.config_index is not None:
        run_single(args.config_index)
    elif args.all:
        run_all()
    elif args.aggregate:
        run_aggregate()
    else:
        parser.print_help()
