"""Certify the provenance of result files that predate the ``reference_model`` field.

Why this exists
---------------
``benchmarks/case_euler_vortex_errconserve.py`` writes the reference model it
measured against into every record.  The bundle under
``case_euler_vortex_errconserve/results/`` also contains records written before
that field existed, and the aggregator used to fill the hole with the *default*
model -- an assumption.  That exact assumption is what went wrong in rounds 1-3
of this case: the early runs were measured against the free-space formula, which
stops being the periodic solution once the vortex wraps (t > 5), and the
"error amplification" published from them was a reference defect.

This tool converts the assumption into evidence: for every legacy record it
re-runs the *same* checkpoint (configuration read from the record itself) against
the current reference, compares the four field errors with the stored ones, and
writes a certificate.  Certificates are data, not prose; C11
(``provenance_audit``) reads them and refuses records whose replay does not
reproduce.

Usage
-----
    PYTHONPATH=. python3 tools/provenance_certify.py --list          # what is legacy
    PYTHONPATH=. python3 tools/provenance_certify.py --keys p0_n64_t1_cfl0.4
    PYTHONPATH=. python3 tools/provenance_certify.py --all-legacy --max-seconds 540

``--max-seconds`` keeps one invocation inside the local short-test budget: the
remaining keys are simply left uncertified (and stay ``unverified`` in the gate),
never silently certified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks import case_euler_vortex_errconserve as ec

CASE_DIR = Path("case_euler_vortex_errconserve")
ERROR_FIELDS = ("density", "velocity_x", "velocity_y", "pressure")


def load_records(results_dir: Path) -> list:
    return [json.loads(path.read_text()) for path in sorted(results_dir.glob("*.json"))]


def legacy_keys(records: list) -> list:
    """Records with no recorded reference model -- the ones needing a replay."""
    return [record["task_key"] for record in records
            if record.get("reference_model") is None]


def replay(record: dict) -> dict:
    """Re-run one checkpoint against the current reference and compare errors."""
    config = dict(record["config"])
    config["reference"] = ec.REFERENCE_MODEL
    started = time.time()
    fresh = ec.run_task(config)
    wall = time.time() - started
    deltas = {field: abs(fresh["errors"][field] - record["errors"][field])
              for field in ERROR_FIELDS if field in record["errors"]}
    scale = max(abs(record["errors"][field]) for field in deltas) or 1.0
    return {
        "task_key": record["task_key"],
        "reference_model": fresh["reference_model"],
        "config": ec.canonical_config(config),
        "stored_max_percent": record["max_percent"],
        "replayed_max_percent": fresh["max_percent"],
        "absolute_delta": deltas,
        "replay_delta": max(deltas.values()) / scale,
        "wall_seconds": wall,
        "steps": fresh["steps"],
        "certified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tolerance": ec.PROVENANCE_REPLAY_TOLERANCE,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(CASE_DIR))
    parser.add_argument("--keys", nargs="*", default=None)
    parser.add_argument("--all-legacy", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=540.0,
                        help="stop starting new replays after this budget")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(max(1, args.threads))

    output_dir = Path(args.output_dir)
    records = load_records(output_dir / "results")
    by_key = {record["task_key"]: record for record in records}
    legacy = legacy_keys(records)

    if args.list:
        print(json.dumps({"present": len(records), "legacy": legacy}, indent=2))
        return 0

    if args.all_legacy:
        keys = [key for key in legacy if key in by_key]
    else:
        keys = list(args.keys or [])
    unknown = [key for key in keys if key not in by_key]
    if unknown:
        raise SystemExit(f"no such record(s): {unknown}")
    if not keys:
        raise SystemExit("nothing to certify: pass --keys ... or --all-legacy")

    certificate_path = output_dir / ec.PROVENANCE_CERTIFICATES
    bundle = (json.loads(certificate_path.read_text())
              if certificate_path.exists() else {"certificates": {}})
    certificates = bundle.setdefault("certificates", {})

    started = time.time()
    for key in keys:
        if time.time() - started > args.max_seconds:
            print(f"budget exhausted, not attempted: "
                  f"{[k for k in keys if k not in certificates]}", flush=True)
            break
        certificate = replay(by_key[key])
        certificates[key] = certificate
        bundle["reference_model"] = ec.REFERENCE_MODEL
        bundle["tolerance"] = ec.PROVENANCE_REPLAY_TOLERANCE
        certificate_path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
        verdict = ("REPRODUCED" if certificate["replay_delta"]
                   <= ec.PROVENANCE_REPLAY_TOLERANCE else "MISMATCH")
        print(f"{key}: stored {certificate['stored_max_percent']:.12g}% -> "
              f"replayed {certificate['replayed_max_percent']:.12g}% "
              f"(rel {certificate['replay_delta']:.2e}) {verdict} "
              f"[{certificate['wall_seconds']:.1f} s]", flush=True)

    remaining = [key for key in legacy if key not in certificates]
    print(f"certificates: {len(certificates)}; legacy still unverified: {remaining}")
    return 0


if __name__ == "__main__":
    sys.exit(main())