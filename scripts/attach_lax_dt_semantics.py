"""Repair the step-size key drift in the committed CASE-LAX-DG artifacts.

The defect
----------
``solve_euler_dg_1d`` returns ``meta["dt"]`` = the *time-truncated final step*
(``min(cfl*dx/..., final_time - time)``) and ``meta["dt_mean"] = final_time/steps``.
Publishers disagree about which one belongs under the bare key ``dt``:

===================  ==========================================  =====================
artifact             bare ``dt`` holds                          ``dt_semantics``?
===================  ==========================================  =====================
lax_case_pool        ``dt_mean`` (+ ``dt_last_step``)           missing
lax_asymptotic_probe ``dt_last``/``dt_mean`` (explicit keys)    present
lax_resolution_*     ``dt_mean``                                missing
lax_certified_*      ``dt_mean``                                missing
lax_limiter_arms     **``dt_last``**  <- drift                  missing
===================  ==========================================  =====================

Consequence in the record: on ``p3.N400`` the last step is ``2.2296e-05`` while the
mean step is ``2.5445e-05`` (12 % apart; on ``p1.N100`` 94 % apart), so reading
``dt`` across artifacts produces a "discrepancy" that has nothing to do with the
solver.  ``docs/CASE-LAX-DG-residual-gap.md`` consumed exactly such a comparison
as evidence that the probe artifact was no longer reproducible.

What this script does
---------------------
* ``lax_case_pool_limiter_arms``: the recorded value is the *last* step.  It is
  kept, but under the key that means it (``dt_last``); ``dt`` is set to the mean
  step ``final_time/steps`` -- which is exact, since ``steps`` is recorded on the
  same leg.  **The solver is not re-run**: the repair only moves and recomputes
  metadata that the artifact already carries, and the old value stays visible as
  ``dt_last`` plus in the ``dt_repair`` provenance block.
* the other artifacts are **left untouched**: their bare ``dt`` already holds the
  mean step, which the contract verifies numerically (C2) on every leg, so
  rewriting them would change file bytes (and the sha256 declared in
  ``docs/CASE-LAX-DG-certified-configuration.md``) without changing anything a
  reader learns.  Semantics is *resolved*, not merely *labelled*.

Usage::

    PYTHONPATH=. python3 scripts/attach_lax_dt_semantics.py            # repair
    PYTHONPATH=. python3 scripts/attach_lax_dt_semantics.py --check    # gate
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from benchmarks.lax_artifact_contract import (  # noqa: E402
    PUBLISHERS,
    TOL,
    extract,
    load,
)
from benchmarks.lax_case_pool import FINAL_TIME  # noqa: E402

SEMANTICS_TEXT = ("dt = final_time/steps (mean step); a time-truncated final "
                  "step, when recorded, is kept under dt_last*")
REPAIR_NOTE = ("dt was the solver's time-truncated final step (meta['dt']); the "
               "recorded value is preserved under dt_last and dt is recomputed as "
               "final_time/steps. Solver was not re-run: both numbers were already "
               "carried by the same leg record.")


REPAIR_TARGETS = ("lax_case_pool_limiter_arms",)


def _detect_indent(path: Path) -> int:
    """Numeric indent of a committed JSON artifact (its own style, not ours)."""
    with path.open() as handle:
        handle.readline()
        second = handle.readline()
    return max(len(second) - len(second.lstrip(" ")), 1)


def _relative(a: float, b: float) -> float:
    return float(abs(a - b) / max(abs(a), abs(b), 1e-300))


def repair(root: Path = ROOT) -> list[dict]:
    """Attach ``dt_semantics`` everywhere and un-drift the limiter-arms artifact."""
    reports: list[dict] = []
    for spec in PUBLISHERS:
        payload = load(spec, root)
        if payload is None:
            reports.append({"publisher": spec.name, "status": "SKIP_ABSENT"})
            continue
        if spec.name not in REPAIR_TARGETS:
            reports.append({"publisher": spec.name, "status": "UNTOUCHED",
                            "reason": "bare dt is already the mean step (C2 PASS)"})
            continue
        touched, moved = 0, []
        for _json_path, leg in spec.leg_records(payload):
            if spec.name == "lax_case_pool_limiter_arms":
                steps = int(leg["steps"])
                mean = FINAL_TIME / steps if steps else 0.0
                legacy = float(leg["dt"])
                # ``legacy`` is the last step: it must be <= the mean step, which
                # is the invariant the repair relies on instead of a re-run.
                if legacy > mean * (1.0 + TOL):
                    raise SystemExit(
                        f"[attach-dt] {spec.name} leg p{leg['degree']}.N{leg['cells']}: "
                        f"recorded dt={legacy:.6e} exceeds the mean step {mean:.6e}; "
                        "the recorded value is not a final step -- refusing to move it"
                    )
                leg["dt_legacy_last_step"] = legacy
                leg["dt_last"] = legacy
                leg["dt"] = mean
                moved.append({
                    "leg": f"p{leg['degree']}.N{leg['cells']}",
                    "steps": steps,
                    "legacy_dt": legacy,
                    "dt_mean": mean,
                    "relative_difference": _relative(legacy, mean),
                })
            leg.setdefault("dt_semantics", SEMANTICS_TEXT)
            touched += 1
        if spec.name == "lax_case_pool_limiter_arms":
            payload["dt_repair"] = {
                "repaired_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "reason": REPAIR_NOTE,
                "solver_rerun": False,
                "legs": moved,
                "worst_relative_difference": max((row["relative_difference"]
                                                  for row in moved), default=None),
            }
        path = root / spec.artifact
        # keep the artifact's own formatting so the diff shows only the repair
        indent = _detect_indent(path)
        path.write_text(json.dumps(payload, indent=indent, sort_keys=False) + "\n")
        reports.append({"publisher": spec.name, "status": "REPAIRED",
                        "legs": touched, "paths": path.as_posix()})
    return reports


def check(root: Path = ROOT) -> dict:
    """Gate: the contract must hold on the committed artifacts."""
    from benchmarks.lax_artifact_contract import audit

    result = audit(root)
    details = {}
    for spec in PUBLISHERS:
        table = extract(spec, root)
        details[spec.name] = {
            "legs": len(table),
            "semantics_declared": all(record["dt_semantics"] for record in table.values()),
            "status": result["publishers"][spec.name]["C2_mean_identity"],
            "worst_mean_relative_error": result["publishers"][spec.name]["worst_mean_relative_error"],
        }
    return {"verdict": result["verdict"], "publishers": details, "audit": result}


def main() -> int:
    parser = argparse.ArgumentParser(description="attach dt semantics / repair drift")
    parser.add_argument("--check", action="store_true", help="verify instead of repair")
    args = parser.parse_args()
    if args.check:
        result = check()
        print(json.dumps(result["publishers"], indent=1))
        print(f"contract verdict: {result['verdict']}")
        return 0 if result["verdict"] == "PASS" else 1
    for row in repair():
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())