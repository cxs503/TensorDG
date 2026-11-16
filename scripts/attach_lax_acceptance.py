"""Attach the ledger acceptance (A0) to the committed Lax case-pool artifact.

Why this script exists
----------------------
``feature_ledger`` grades ``CASE-LAX-DG`` as *relative L2 against the exact
Riemann solution < 3 %*.  A whole-field relative L2 on a solution with two true
jumps converges at roughly ``h**0.4``, so **which leg** the bar is read on is
part of the registration.  Historically the pool read it on "the finest ladder
leg" (``p3.N400`` -> 3.116 %, FAIL) while the certified-configuration closure
published ``p2.N800`` (2.570 %, PASS): two artifacts, one ledger, no single
source of truth -- exactly the drift an adversarial review of the A0 chain flags
as blocking.

The leg is now registered in the case module
(``tensordg.cases.lax_1d.lax_parameters()["production_leg"]``) and this script is
the only place that turns that registration into a verdict:

* the registered leg is **measured** with the pool instrument
  (``benchmarks.lax_case_pool.measure_leg``, same Gauss nodes, same q = 10
  rule as every ladder leg) into ``production_leg.json`` +
  ``profiles_production.npz``;
* the record is re-measured offline from the committed profiles and refused if
  the two disagree (no hand-editable number reaches the verdict);
* the record leg must equal the registry leg, otherwise nothing is attached;
* the ``ladder_l1`` arm still covers every ladder leg, and the ladder's own
  ``p3.N400`` row keeps its honest sub-bar verdict in the same table.

Usage::

    python scripts/attach_lax_acceptance.py --measure   # solve + write + attach
    python scripts/attach_lax_acceptance.py             # attach from record
    python scripts/attach_lax_acceptance.py --check     # non-zero if stale

``--check`` never runs a solver.
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
    gate_a0, ledger_acceptance_tolerance, measure_leg, production_errors_from_profiles,
    registered_production_leg, render_acceptance_figure, render_summary)

OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool"
METRICS = OUT / "metrics.json"
PROFILES = OUT / "profiles.npz"                 # ladder-finest leg (legacy consumer)
PROD_PROFILES = OUT / "profiles_production.npz"  # registered (ledger) leg
PROD_RECORD = OUT / "production_leg.json"
FIGDIR = OUT / "figures"
FIELDS = ("density", "velocity", "pressure")
#: record <-> profiles re-measurement agreement (relative)
CONSISTENCY_TOL = 1e-6


def measure_registered_leg() -> dict:
    """Solve the registered leg with the pool instrument and write its record."""
    leg = registered_production_leg()
    waves_entry_waves = None  # exact wave positions are needed by measure_leg
    from benchmarks.lax_case_pool import exact_waves
    waves_entry_waves = exact_waves()
    entry, bundle = measure_leg(leg["cells"], leg["degree"], waves_entry_waves)
    import numpy as np
    np.savez(PROD_PROFILES, points=bundle["points"],
             **{f"numerical_{k}": v for k, v in bundle["primitive"].items()},
             **{f"exact_{k}": v for k, v in bundle["exact"].items()})
    record = {
        "case": "CASE-LAX-DG",
        "leg": f"p{leg['degree']}.N{leg['cells']}",
        "cells": leg["cells"],
        "degree": leg["degree"],
        "registration": {"source": "tensordg.cases.lax_1d.lax_parameters()",
                         "leg": leg,
                         "tolerance": ledger_acceptance_tolerance()},
        "instrument": ("benchmarks.lax_case_pool.measure_leg: modal DG fields on "
                       "the element Gauss nodes, q = 10 Legendre weights, whole "
                       "field, denominator = exact-solution norm"),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": _environment(),
        "steps": entry["steps"],
        "dt": entry["dt"],
        "wall_seconds": entry["wall_seconds"],
        "density_min": entry["density_min"],
        "errors": entry["errors"],
        "l2_errors": entry["l2_errors"],
        "profiles": str(PROD_PROFILES.relative_to(ROOT)),
    }
    PROD_RECORD.write_text(json.dumps(record, indent=2) + "\n")
    return record


def _environment() -> dict:
    import platform

    env = {"python": platform.python_version(), "machine": platform.machine(),
           "processor": platform.processor() or "unknown"}
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_threads"] = int(torch.get_num_threads())
    except Exception:  # pragma: no cover - torch is a hard dep of the solver
        env["torch"] = "unavailable"
    return env


def attach(metrics: dict, record: dict, production: dict) -> dict:
    """Rebuild the acceptance block from the record; refuse on any drift."""
    leg = registered_production_leg()
    if (record["cells"], record["degree"]) != (leg["cells"], leg["degree"]):
        raise SystemExit(
            f"production_leg.json records p{record['degree']}.N{record['cells']} "
            f"but the case module registers p{leg['degree']}.N{leg['cells']}: "
            "refusing to attach a verdict for a leg nobody registered")
    mismatch, worst = {}, 0.0
    for f in FIELDS:
        row = {"l1": abs(production["l1"][f] - record["errors"][f])
               / record["errors"][f],
               "l2": abs(production["l2"][f] - record["l2_errors"][f])
               / record["l2_errors"][f]}
        mismatch[f] = row
        worst = max(worst, row["l1"], row["l2"])
    if worst > CONSISTENCY_TOL:
        raise SystemExit(
            f"{PROD_PROFILES.name} does not reproduce production_leg.json "
            f"(worst relative mismatch {worst:.3e} > {CONSISTENCY_TOL:g}); "
            "refusing to attach an acceptance verdict")
    ladder = metrics["ladder"]
    entry = {"cells": record["cells"], "degree": record["degree"],
             "errors": record["errors"], "l2_errors": record["l2_errors"],
             "steps": record["steps"], "wall_seconds": record["wall_seconds"],
             "density_min": record["density_min"],
             "role": "registered production leg (ledger A0)"}
    acceptance = gate_a0(ladder, tolerance=ledger_acceptance_tolerance(),
                         production_leg=entry, production_l2=record["l2_errors"])
    metrics = dict(metrics)
    metrics["acceptance"] = acceptance
    metrics["production_leg"] = {
        "leg": record["leg"], "cells": record["cells"],
        "degree": record["degree"], "role": "registered production leg",
        "registration": record["registration"],
        "steps": record["steps"], "wall_seconds": record["wall_seconds"],
        "density_min": record["density_min"], "errors": record["errors"],
        "l2_errors": record["l2_errors"],
        "profiles": record["profiles"],
        "record": str(PROD_RECORD.relative_to(ROOT)),
    }
    metrics["verdicts"] = {
        "acceptance": acceptance["status"], "verification": metrics["verdict"],
        "reading": ("acceptance = ledger CASE-LAX-DG, relative L2 vs exact < "
                    "registered tolerance at the *registered* production leg "
                    "(p3.N400 ladder row keeps its own sub-bar verdict); "
                    "verification = supplementary order gates C1/C2/C3/R3/R5/TIME")}
    tolerances = dict(metrics["config"]["tolerances"])
    tolerances["a0_acceptance"] = acceptance["tolerance"]
    config = dict(metrics["config"])
    config["tolerances"] = tolerances
    config["production_leg"] = record["registration"]["leg"]
    metrics["config"] = config
    metrics["acceptance_provenance"] = {
        "attached_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "production_leg": acceptance["production_leg"],
        "production_leg_source": acceptance.get("production_leg_source"),
        "arm_source": acceptance["arms"]["production_l2"]["source"],
        "record_file": str(PROD_RECORD.relative_to(ROOT)),
        "profiles_file": str(PROD_PROFILES.relative_to(ROOT)),
        "registry": record["registration"],
        "method": ("the registered leg is measured by the same instrument as the "
                   "ladder, re-measured offline from its committed profiles, and "
                   "the verdict recomputed from the record; the ladder_finest "
                   "profiles.npz is left untouched for its existing consumers"),
        "profile_consistency": {
            "worst_relative_mismatch": float(worst),
            "worst_relative_mismatch_l1": float(max(r["l1"]
                                                    for r in mismatch.values())),
            "worst_relative_mismatch_l2": float(max(r["l2"]
                                                    for r in mismatch.values())),
            "tolerance": CONSISTENCY_TOL,
            "reading": ("record numbers vs re-measurement from "
                        "profiles_production.npz; the ledger arm is refused above "
                        "the tolerance"),
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
    parser.add_argument("--measure", action="store_true",
                        help="(re)solve the registered production leg first")
    args = parser.parse_args(argv)
    metrics = json.loads(METRICS.read_text())
    if args.measure and not args.check:
        record = measure_registered_leg()
    else:
        if not PROD_RECORD.exists() or not PROD_PROFILES.exists():
            print("no production-leg record; run with --measure", file=sys.stderr)
            return 1
        record = json.loads(PROD_RECORD.read_text())
    production = production_errors_from_profiles(PROD_PROFILES)
    rebuilt = attach(metrics, record, production)
    if args.check:
        problems = [key for key in ("acceptance", "production_leg")
                    if rebuilt[key] != metrics.get(key)]
        if rebuilt["verdicts"] != metrics.get("verdicts"):
            problems.append("verdicts")
        for problem in problems:
            print(f"[lax-a0] stale {problem}", file=sys.stderr)
        return 1 if problems else 0
    render_outputs(rebuilt)
    acceptance = rebuilt["acceptance"]
    print(f"[lax-a0] A0={acceptance['status']} ledger arm "
          f"{acceptance['arms']['production_l2']['worst']:.5e} "
          f"({acceptance['production_leg']}, source="
          f"{acceptance['arms']['production_l2']['source']}), L1 arm "
          f"{acceptance['arms']['ladder_l1']['worst']:.5e}; verification "
          f"{rebuilt['verdict']}; wrote {METRICS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())