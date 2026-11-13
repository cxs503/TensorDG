"""Merge the per-mechanism CASES-LAX-DG contrast runs into one metrics bundle.

The study is run as one process per shock-capturing mechanism (each process
owns its own ladder, figures and part JSON) so the wall clock is bounded by
the slowest single mechanism instead of the sum.  This script stitches the
parts together with the committed ``none`` ladder from the pool, re-runs the
cheap reference-reuse leg, applies the pool's gate functions and writes::

    validation/lax_limiter_study/metrics.json
    validation/lax_limiter_study/metrics.sha256
    validation/lax_limiter_study/summary.md

Usage::

    PYTHONPATH=. python -m scripts.merge_lax_limiter_parts
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.lax_limiter_study import (
    LADDER,
    MECHANISMS,
    OUT,
    CFL,
    FINAL_TIME,
    LEFT,
    RIGHT,
    R3_BAND_WIDTH,
    load_reference_ladder,
    mechanism_gates,
    order_by_mechanism,
    reference_definition_diagnostic,
    reference_reuse_check,
    render_summary,
    residual_windows,
)
from benchmarks.lax_case_pool import exact_waves


def merge(out: Path = OUT) -> dict:
    torch.set_num_threads(1)
    started = time.time()
    waves = exact_waves()
    entries, figures, environment, per_mechanism_config = [], [], {}, {}
    _, reference_entries = load_reference_ladder()
    entries.extend(reference_entries)
    for part in sorted((out / "parts").glob("*/metrics.json")):
        data = json.loads(part.read_text())
        entries.extend(data["ladder"])
        environment = data.get("environment", environment)
        per_mechanism_config.update(data["config"].get("mechanicsm", {}))
        figures.extend(str(path) for path in sorted((part.parent / "figures").glob("*.png")))

    gates = {"none": mechanism_gates(reference_entries)}
    by_mechanism = {}
    for entry in entries:
        by_mechanism.setdefault(entry["mechanism"], []).append(entry)
    for mechanism, mechanism_entries in by_mechanism.items():
        if mechanism == "none":
            continue
        gates[mechanism] = mechanism_gates(mechanism_entries)

    payload = {
        "case": "lax-1954-shock-tube",
        "anchor": "CASE-LAX-DG",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"ladder": LADDER, "cfl": CFL, "final_time": FINAL_TIME,
                   "left_state": LEFT, "right_state": RIGHT,
                   "mechanicsm": {name: MECHANISMS[name]
                                  for name in per_mechanism_config or {}},
                   "quick": False, "r3_band_width": R3_BAND_WIDTH,
                   "residual_windows": residual_windows(waves)},
        "waves": waves["positions"],
        "environment": environment or {"python": platform.python_version(),
                                       "numpy": np.__version__,
                                       "torch": torch.__version__,
                                       "platform": platform.platform()},
        "reference_definition": reference_definition_diagnostic(waves),
        "reference_reuse_check": reference_reuse_check(waves),
        "ladder": entries,
        "gates": gates,
        "orders": order_by_mechanism(entries),
        "figures": figures,
        "wall_seconds": float(time.time() - started),
    }
    payload["verdict"] = ("PASS" if gates and all(g["verdict"] == "PASS"
                                                 for g in gates.values())
                          else "FAIL")
    metrics = out / "metrics.json"
    metrics.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(metrics.read_bytes()).hexdigest()
    (out / "metrics.sha256").write_text(f"{digest}  metrics.json\n")
    render_summary(payload)
    print(f"[merge] mechanisms={list(gates)} verdict={payload['verdict']} "
          f"sha256={digest[:16]}")
    return payload


if __name__ == "__main__":
    merge()