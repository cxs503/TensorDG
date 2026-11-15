"""CASE-LAX-DG · how far can a committed artifact be *reproduced* on another stack?

Why this probe exists
---------------------
The committed ``lax_case_pool`` / ``lax_asymptotic_probe`` artifacts record the
whole-field relative L1/L2 errors of the Lax (Toro test 2) DG ladder, and the
ledger arm of ``CASE-LAX-DG`` is one of those numbers
(``p3.N400`` whole-field relative **L2** density = ``3.11556e-02``).  Until now
"the artifact is reproducible" was argued by comparing **two committed files**
with each other -- a check that cannot fail on a *new* machine, because a
committed file never re-runs.

Re-measuring the same leg on the development box (``torch 2.7.1`` /
``numpy 2.4.6``) against the artifact (``torch 2.13.0`` / ``numpy 2.0.1``,
loongarch64 node) gives a different number, and the difference grows with the
number of time steps:

============================  ===============  ==========  ==============
leg                           artifact         rerun       rel. drift
============================  ===============  ==========  ==============
``p3.N100``                   ``1.58470e-02``  see json    ~1e-07
``p3.N400``                   ``8.68554e-03``  see json    ~1e-04
============================  ===============  ==========  ==============

This module measures that band instead of assuming it is zero, and it also
separates the two candidate causes that the previous audit could not separate:

* **thread count** -- ``torch.set_num_threads`` changes reduction order.  Tested
  directly: the same leg is measured at two thread counts and the fields are
  compared **bit for bit**.
* **stack** -- torch / numpy / python versions and the CPU records.  Every leg
  is stamped with an :func:`tensordg.reproducibility.environment_fingerprint`.

Pre-registered reads
--------------------
* **T1** if the two thread counts give bit-identical fields on one machine, the
  thread hypothesis is falsified and any residual drift is stack-attributable.
* **T2** the drift is expected to grow with the step count (a stiffening wave
  field amplifies last-bit reduction differences); the table is reported per
  leg, no model is fitted.
* **T3** the ledger reading is *not* re-judged here: ``p3.N400`` L2 stays at its
  FAIL.  What is measured is how tightly the number can be relied on.

Usage::

    PYTHONPATH=. python -m benchmarks.lax_reproducibility_band --legs 100:3,200:3
    PYTHONPATH=. python -m benchmarks.lax_reproducibility_band --legs 400:3 --threads 1:8
    PYTHONPATH=. python -m benchmarks.lax_reproducibility_band --write      # merge into artifact
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.lax_asymptotic_probe import measure_leg
from benchmarks.lax_case_pool import (
    FINAL_TIME,
    OUT as POOL_OUT,
    exact_waves,
    modal_fields,
)
from tensordg.reproducibility import environment_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROBE_METRICS = ROOT / "docs/benchmarks/results/lax_asymptotic_probe/metrics.json"
POOL_METRICS = ROOT / "docs/benchmarks/results/lax_case_pool/metrics.json"
OUT = ROOT / "docs/benchmarks/results/lax_reproducibility_band"
QUADRATURE_ORDER = 10
#: ledger arm (``feature_ledger`` CASE-LAX-DG): p3.N400 whole-field relative L2
LEDGER_LEG = (400, 3)


def _l2(points, values, exact, weights, dx, mask=None):
    """Relative L2, identical normalisation to ``lax_case_pool``'s L2 path."""
    cell = 0.5 * dx * weights[None, :] * np.ones_like(values)
    if mask is not None:
        cell = cell * mask
    num = float(np.sum(cell * (values - exact) ** 2))
    den = float(np.sum(cell * exact ** 2))
    return math.sqrt(num / den)


def _dx(points):
    """Cell width exactly as ``modal_fields`` computes it (span / (ncells - 1))."""
    centres = 0.5 * (points[:, 0] + points[:, -1])
    return float((centres[-1] - centres[0]) / (len(centres) - 1))


def _committed(leg_key):
    """Committed readings for ``leg_key`` from both artifacts (first hit wins)."""
    out = {}
    cells, degree = leg_key
    for tag, path, entry_key in (
        ("pool", POOL_METRICS, "ladder"),
        ("probe", PROBE_METRICS, "ladder"),
    ):
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for entry in payload.get(entry_key, []):
            if entry.get("cells") == cells and entry.get("degree") == degree:
                out[tag] = {
                    "l1_density": entry["errors"]["density"],
                    "steps": entry.get("steps"),
                    "environment": payload.get("environment"),
                }
                break
    return out


def measure(cells, degree, waves):
    """One leg at the *current* thread setting, with both norms."""
    started = time.time()
    leg, fields = measure_leg(cells, degree, waves)
    points = fields["points"]
    weights = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)[1]
    dx = _dx(points)
    exact = fields["exact"]["density"]
    numerical = fields["numerical"]["density"]
    return {
        "cells": cells,
        "degree": degree,
        "steps": leg["steps"],
        "torch_threads": int(torch.get_num_threads()),
        "l1_density": leg["errors"]["density"],
        "l2_density": _l2(points, numerical, exact, weights, dx),
        "density_min": leg["density_min"],
        "pressure_min": leg["pressure_min"],
        "wall_seconds": time.time() - started,
        "fields_sha256": _fields_digest(numerical),
    }, numerical


def _fields_digest(values):
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def rel_drift(value, reference):
    if reference in (None, 0.0):
        return None
    return abs(value / reference - 1.0)


def run(legs, threads):
    waves = exact_waves()
    rows = []
    digests = {}
    for cells, degree in legs:
        for nthreads in threads:
            torch.set_num_threads(nthreads)
            row, field = measure(cells, degree, waves)
            committed = _committed((cells, degree))
            reference = committed.get("probe", committed.get("pool"))
            row["committed"] = committed
            if reference is not None:
                row["drift_l1"] = rel_drift(row["l1_density"], reference["l1_density"])
                row["steps_agree"] = (
                    None if reference["steps"] is None else row["steps"] == reference["steps"]
                )
            row["stack"] = _stack_tag(environment_fingerprint())
            rows.append(row)
            digests.setdefault(f"p{degree}.N{cells}", {})[str(nthreads)] = row.pop("fields_sha256")
            print(
                f"[band] p{degree}.N{cells} threads={nthreads} steps={row['steps']} "
                f"L1={row['l1_density']:.6e} L2={row['l2_density']:.6e} "
                f"drift={row.get('drift_l1')} t={row['wall_seconds']:.1f}s",
                flush=True,
            )
    thread_tests = {}
    for key, by_threads in digests.items():
        values = list(by_threads.values())
        thread_tests[key] = {
            "threads": sorted(int(t) for t in by_threads),
            "fields_bit_identical": len(set(values)) == 1,
            "fields_sha256": by_threads,
        }
    return {"rows": rows, "thread_tests": thread_tests}


def payload(legs, threads, result, extra_env=None):
    env_local = environment_fingerprint()
    if extra_env:
        env_local = {**env_local, **extra_env}
    drifts = [r["drift_l1"] for r in result["rows"] if r.get("drift_l1") is not None]
    worst = max(drifts) if drifts else None
    thread_falsified = all(t["fields_bit_identical"] for t in result["thread_tests"].values())
    return {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "question": "how tightly is a committed CASE-LAX-DG artifact reproducible on another stack?",
        "legs": [list(leg) for leg in legs],
        "threads": list(threads),
        "config": {
            "final_time": FINAL_TIME,
            "quadrature_order": QUADRATURE_ORDER,
            "metrics": "whole-field relative L1 and L2 of density",
            "ledger_leg": list(LEDGER_LEG),
        },
        "environment": env_local,
        "rows": result["rows"],
        "thread_tests": result["thread_tests"],
        "worst_drift_l1": worst,
        "thread_hypothesis_falsified": thread_falsified,
        "ledger_arm_rejudged": False,
    }


def write(payload_dict, merge=True):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "metrics.json"
    merged = payload_dict
    if merge and path.exists():
        old = json.loads(path.read_text())
        old_env = old.get("environment", {}) or {}
        old_tag = _stack_tag(old_env)
        by_key = {}
        for row in old.get("rows", []):
            # artifacts written before the stack tag existed belong to the stack
            # the artifact itself records -- that is where the numbers came from.
            row = {**row, "stack": row.get("stack") or old_tag}
            by_key[(row["cells"], row["degree"], row["torch_threads"], row["stack"])] = row
        for row in payload_dict["rows"]:
            by_key[(row["cells"], row["degree"], row["torch_threads"],
                    row.get("stack"))] = row
        merged = {**payload_dict}
        merged["rows"] = sorted(by_key.values(),
                                key=lambda r: (r["cells"], r["degree"], r["torch_threads"], str(r.get("stack"))))
        merged["thread_tests"] = {}
        for source in (old.get("thread_tests", {}), payload_dict["thread_tests"]):
            for key, entry in source.items():
                slot = merged["thread_tests"].setdefault(
                    key, {"threads": [], "fields_sha256": {}, "fields_bit_identical": True})
                slot["fields_sha256"].update(entry["fields_sha256"])
                slot["threads"] = sorted(set(slot["threads"]) | set(entry["threads"]))
                slot["fields_bit_identical"] = (
                    slot["fields_bit_identical"] and entry["fields_bit_identical"]
                    and len(set(slot["fields_sha256"].values())) == 1)
        merged["stacks"] = old.get("stacks", [])
        if old_env and old_env != payload_dict["environment"]:
            tag = _stack_tag(old_env)
            if tag not in [s.get("tag") for s in merged["stacks"]]:
                merged["stacks"].append({"tag": tag, "environment": old_env})
        drifts = [r["drift_l1"] for r in merged["rows"] if r.get("drift_l1") is not None]
        merged["worst_drift_l1"] = max(drifts) if drifts else None
        merged["thread_hypothesis_falsified"] = all(
            t["fields_bit_identical"] for t in merged["thread_tests"].values()
        )
    tag = _stack_tag(payload_dict["environment"])
    merged.setdefault("stacks", [])
    if tag not in [s.get("tag") for s in merged["stacks"]]:
        merged["stacks"].append({"tag": tag, "environment": payload_dict["environment"]})
    path.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    print(f"[band] wrote {path}")
    return merged


def _stack_tag(env):
    torch_version = (env or {}).get("torch", "?")
    numpy_version = (env or {}).get("numpy", "?")
    machine = (env or {}).get("machine", "?")
    return f"torch{torch_version}/numpy{numpy_version}/{machine}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legs", default="100:3,200:3", help="comma list cells:degree")
    parser.add_argument("--threads", default="1", help="comma list of torch thread counts")
    parser.add_argument("--write", action="store_true", help="merge into the artifact")
    args = parser.parse_args(argv)
    legs = [tuple(int(v) for v in item.split(":")) for item in args.legs.split(",")]
    threads = [int(v) for v in args.threads.split(",")]
    result = run(legs, threads)
    data = payload(legs, threads, result)
    print(json.dumps({k: data[k] for k in ("worst_drift_l1", "thread_hypothesis_falsified")}))
    if args.write:
        write(data)


if __name__ == "__main__":
    main()