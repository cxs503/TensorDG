#!/usr/bin/env python3
"""Replay the 7 unverified C11 provenance records and generate certificates.

Unverified records (from case_euler_vortex_errconserve/metrics.json C11):
  dg_d2_n32_t10_cfl0.2, dg_d2_n32_t10_cfl0.4,
  dg_d2_n32_t2_cfl0.2, dg_d2_n32_t5_cfl0.2,
  dg_d2_n64_t10_cfl0.2, dg_d2_n64_t1_cfl0.2, dg_d2_n64_t2_cfl0.2

Each is re-run and compared against the stored result.  A certificate is
emitted when the replay delta is within PROVENANCE_REPLAY_TOLERANCE (1e-9).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

import tensordg.euler_dg_2d as dg2
from tensordg.euler_2d import (_cell_averages, minimum_images,
                               vortex_error, vortex_primitive)
from benchmarks.verify_euler_dg_2d import _reconstruct, _vortex_error as dg_vortex_error
from benchmarks.case_euler_vortex_errconserve import (
    BACKGROUND, DOMAIN, FIELDS, PERIOD, STRENGTH, REFERENCE_MODEL,
    PROVENANCE_REPLAY_TOLERANCE, PROVENANCE_CERTIFICATES,
    _operators, _initial_dg_state, _dg_measurements, _reference_kwargs,
)

RESULTS_DIR = Path("case_euler_vortex_errconserve/results")
CERT_DIR = Path("case_euler_vortex_errconserve")

#: The 7 unverified task keys and their configs
UNVERIFIED = [
    {"task_key": "dg_d2_n32_t10_cfl0.2", "task": "dg", "degree": 2, "cells": 32, "final_time": 10.0, "cfl": 0.2},
    {"task_key": "dg_d2_n32_t10_cfl0.4", "task": "dg", "degree": 2, "cells": 32, "final_time": 10.0, "cfl": 0.4},
    {"task_key": "dg_d2_n32_t2_cfl0.2",  "task": "dg", "degree": 2, "cells": 32, "final_time": 2.0,  "cfl": 0.2},
    {"task_key": "dg_d2_n32_t5_cfl0.2",  "task": "dg", "degree": 2, "cells": 32, "final_time": 5.0,  "cfl": 0.2},
    {"task_key": "dg_d2_n64_t10_cfl0.2", "task": "dg", "degree": 2, "cells": 64, "final_time": 10.0, "cfl": 0.2},
    {"task_key": "dg_d2_n64_t1_cfl0.2",  "task": "dg", "degree": 2, "cells": 64, "final_time": 1.0,  "cfl": 0.2},
    {"task_key": "dg_d2_n64_t2_cfl0.2",  "task": "dg", "degree": 2, "cells": 64, "final_time": 2.0,  "cfl": 0.2},
]


def run_dg_replay(config: dict) -> dict:
    """Run a DG configuration and return errors (same logic as run_dg in the benchmark)."""
    started = time.time()
    degree = config["degree"]
    strength = config.get("strength", STRENGTH)
    background = tuple(config.get("background_velocity", BACKGROUND))
    final_time = config.get("final_time", 1.0)
    _x, _y, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=config["cells"],
        degree=degree,
        final_time=final_time,
        cfl=config.get("cfl", 0.2),
        strength=strength,
        background_velocity=background,
        domain=DOMAIN,
    )
    nodes, weights, basis, _, _ = _operators(degree)
    reference_kwargs = _reference_kwargs(config, final_time)
    errors = dg_vortex_error(coeff, basis, nodes, DOMAIN, final_time, strength,
                            background, **reference_kwargs)
    return {
        "errors": {name: float(errors[name]) for name in FIELDS},
        "max_relative_l2": max(float(errors[name]) for name in FIELDS),
        "steps": meta["steps"],
        "wall_seconds": time.time() - started,
    }


def load_stored(task_key: str) -> dict | None:
    """Load the stored result from results/."""
    path = RESULTS_DIR / f"{task_key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def compute_delta(stored: dict, replay: dict) -> float:
    """Compute max relative delta between stored and replay errors."""
    deltas = []
    for field in FIELDS:
        s = stored.get("errors", {}).get(field, 0.0)
        r = replay.get("errors", {}).get(field, 0.0)
        denom = max(abs(s), abs(r), 1e-30)
        deltas.append(abs(s - r) / denom)
    # Also compare max_relative_l2
    s_max = stored.get("max_relative_l2", 0.0)
    r_max = replay.get("max_relative_l2", 0.0)
    denom = max(abs(s_max), abs(r_max), 1e-30)
    deltas.append(abs(s_max - r_max) / denom)
    return max(deltas)


def main():
    cert_path = CERT_DIR / PROVENANCE_CERTIFICATES
    if cert_path.exists():
        existing = json.loads(cert_path.read_text())
    else:
        existing = {"certificates": {}, "reference_model": REFERENCE_MODEL, "tolerance": PROVENANCE_REPLAY_TOLERANCE}

    certs = existing.get("certificates", existing)  # handle both formats

    results = []
    for cfg in UNVERIFIED:
        task_key = cfg["task_key"]
        print(f"\n=== Replaying {task_key} ===")

        stored = load_stored(task_key)
        if stored is None:
            print(f"  SKIP: no stored result for {task_key}")
            continue

        replay = run_dg_replay(cfg)
        delta = compute_delta(stored, replay)
        verified = delta <= PROVENANCE_REPLAY_TOLERANCE

        print(f"  Stored max_err: {stored['max_relative_l2']:.6e}")
        print(f"  Replay max_err: {replay['max_relative_l2']:.6e}")
        print(f"  Delta: {delta:.6e}  (tolerance={PROVENANCE_REPLAY_TOLERANCE:.0e})")
        print(f"  Verified: {verified}")

        cert = {
            "config": {k: cfg[k] for k in ("task", "degree", "cells", "final_time", "cfl")},
            "reference_model": REFERENCE_MODEL,
            "stored_errors": {f: stored["errors"][f] for f in FIELDS},
            "replay_errors": replay["errors"],
            "replay_delta": delta,
            "verified": verified,
            "wall_seconds": replay["wall_seconds"],
            "certified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        certs[task_key] = cert
        results.append((task_key, delta, verified))

    # Write updated certificates
    output = {
        "certificates": certs,
        "reference_model": REFERENCE_MODEL,
        "tolerance": PROVENANCE_REPLAY_TOLERANCE,
    }
    cert_path.write_text(json.dumps(output, indent=2))
    print(f"\nCertificates written to {cert_path}")

    # Summary
    print("\n=== Summary ===")
    for tk, delta, verified in results:
        status = "PASS" if verified else "FAIL"
        print(f"  {tk}: delta={delta:.6e} [{status}]")

    all_verified = all(v for _, _, v in results)
    print(f"\nAll verified: {all_verified}")
    return 0 if all_verified else 1


if __name__ == "__main__":
    exit(main())
