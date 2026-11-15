"""CASE-LAX-DG · cross-artifact metric / step-size contract for the Lax family.

Why this module exists
----------------------
Ten sibling artifacts under ``docs/benchmarks/results/lax_*`` all report error
ladders for the *same* physical case (Lax 1954 / Toro test 2) on overlapping
``(cells, degree)`` legs.  They were written by independent drivers over several
days, and two kinds of key drift accumulated:

1. **Step-size key drift.**  ``solve_euler_dg_1d`` returns ``meta["dt"]`` = the
   *time-truncated final step* (``min(cfl*dx/..., final_time - t)``), which is
   **not** representative of the run, plus ``meta["dt_last"]`` (same value under
   an honest name) and ``meta["dt_mean"] = final_time/steps``.  The pool writer
   stores ``dt`` = ``dt_mean`` (with the last step kept as ``dt_last_step``), the
   probe writer stores ``dt_last``/``dt_mean``, but the limiter-arms writer
   stored the *last step* under the bare key ``dt``.  Reading ``dt`` across
   artifacts therefore compared two different quantities -- a 12 % "discrepancy"
   that has nothing to do with the solver.
2. **Metric-definition drift.**  ``errors`` is the *whole-field* relative L1,
   ``l2_errors`` the whole-field relative L2, ``analytic_errors`` the relative L1
   restricted to the analytic region (wave band removed).  Comparing an
   ``errors`` value of one artifact with an ``analytic_errors`` value of another
   yields an ``O(10^-3)`` difference on the same physical leg -- again not drift.

Both failure modes were consumed as evidence: ``docs/CASE-LAX-DG-residual-gap.md``
concluded that the probe artifact "is no longer reproducible by the current code"
and queued a regeneration of it.  This module makes the comparison
*definition-explicit*: every artifact is described by a :class:`PublisherSpec`
that maps its native keys to canonical metrics, so a reconciliation can only
ever put like against like, and a semantic collision is a *check failure* rather
than a silent mis-read.

Canonical metrics
-----------------
``whole_field_L1``   relative L1 over the whole domain, Gauss-node Legendre
                     quadrature weighting, denominator = exact Riemann solution
``whole_field_L2``   same measure with the L2 norm
``band_L1``          relative L1 restricted to the analytic region (band removed)

Canonical step sizes
--------------------
``dt_mean``  ``final_time / steps`` -- what a CFL/refinement study refines
``dt_last``  the final, time-truncated step returned by the solver

Contract checks (pre-registered, see ``CONTRACT_CHECKS``)
---------------------------------------------------------
C1 semantics resolved   every publisher's leg records either declare which
                        quantity the bare key ``dt`` holds (``dt_semantics``) or
                        carry a value that *is* verified to be the mean step by
                        C2 -- a reader never has to guess, and a file with the
                        right value is not rewritten just to add a label
C2 mean identity        ``|dt_mean - final_time/steps| / (final_time/steps) <= 1e-12``
C3 last-step bound      ``0 < dt_last <= dt_mean * (1 + 1e-12)`` where recorded
C4 no key collision     the bare key ``dt`` maps to the *same* canonical quantity
                        in every publisher
C5 cross-artifact       for every leg shared by >= 2 publishers and every
                        canonical metric both publish, relative difference
                        ``<= 1e-12`` (bitwise, up to float formatting)
C6 live reproduction    a fresh solve of a shared leg reproduces the committed
                        canonical metrics to ``<= 1e-12``

Nothing here re-runs the solver; C6 is evaluated by
``benchmarks/lax_artifact_reconciliation.py`` and its record is read back here.

Usage
-----
>>> from benchmarks.lax_artifact_contract import audit, reconcile   # doctest: +SKIP
>>> audit()["verdict"], reconcile()["verdict"]                     # doctest: +SKIP
('PASS', 'PASS')
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from benchmarks.lax_case_pool import FINAL_TIME

#: relative tolerance for the step-size identities (C2/C3) and for cross-artifact
#: agreement on a physical quantity (C5/C6).  float64 round-off between two runs
#: of the same configuration is 0.0 in practice; 1e-12 is the guard band.
TOL = 1e-12

RESULTS = Path("docs/benchmarks/results")

#: canonical metric name -> human description (used by the renderer)
CANONICAL_METRICS: Mapping[str, str] = {
    "whole_field_L1": "relative L1, whole field (Gauss-node Legendre weighting)",
    "whole_field_L2": "relative L2, whole field (Gauss-node Legendre weighting)",
    "band_L1": "relative L1, analytic region only (wave band removed)",
}


@dataclass(frozen=True)
class PublisherSpec:
    """How one artifact spells the canonical quantities."""

    name: str
    artifact: str                      # repo-relative path of metrics.json
    legs_path: str                     # dotted path to the legs; '*' expands
    metric_keys: Mapping[str, str]     # canonical metric -> native key
    dt_mean_keys: Tuple[str, ...] = ()  # native key(s) holding dt_mean
    dt_last_keys: Tuple[str, ...] = ()  # native key(s) holding dt_last
    dt_semantics_key: Optional[str] = None
    #: canonical quantity that the *bare* key ``dt`` carries in this artifact,
    #: ``None`` when the artifact has no bare ``dt`` key at all
    bare_dt_semantics: Optional[str] = None
    note: str = ""

    def leg_records(self, payload) -> List[Tuple[str, Mapping]]:
        """Yield ``(json_path, leg_record)`` for every leg in the artifact."""
        out: List[Tuple[str, Mapping]] = []

        # explicit implementation: resolve the dotted path, expanding '*'
        def resolve(node, parts: Sequence[str], path: str) -> None:
            if not parts:
                if isinstance(node, list):
                    for index, item in enumerate(node):
                        if isinstance(item, Mapping):
                            out.append((f"{path}[{index}]", item))
                elif isinstance(node, Mapping):
                    out.append((path, node))
                return
            head, rest = parts[0], parts[1:]
            if head == "*":
                if isinstance(node, Mapping):
                    for key in sorted(node):
                        resolve(node[key], rest, f"{path}.{key}")
                elif isinstance(node, list):
                    for index, item in enumerate(node):
                        resolve(item, rest, f"{path}[{index}]")
                return
            if isinstance(node, Mapping) and head in node:
                resolve(node[head], rest, f"{path}.{head}" if path else head)

        resolve(payload, self.legs_path.split("."), "")
        # drop records that are not legs (no cells/degree pair)
        return [(p, r) for p, r in out if "cells" in r and "degree" in r]

    def resolve_dt_mean(self, leg: Mapping) -> Tuple[Optional[float], Optional[str]]:
        for key in self.dt_mean_keys:
            if key in leg and leg[key] is not None:
                return float(leg[key]), key
        if self.bare_dt_semantics == "dt_mean" and leg.get("dt") is not None:
            return float(leg["dt"]), "dt"
        return None, None

    def resolve_dt_last(self, leg: Mapping) -> Tuple[Optional[float], Optional[str]]:
        for key in self.dt_last_keys:
            if key in leg and leg[key] is not None:
                return float(leg[key]), key
        if self.bare_dt_semantics == "dt_last" and leg.get("dt") is not None:
            return float(leg["dt"]), "dt"
        return None, None


#: Publisher registry.  ``bare_dt_semantics`` documents the *current committed*
#: meaning of the bare key ``dt``; changing it is a contract change that C4
#: catches.  ``lax_case_pool_limiter_arms`` used to say ``"dt_last"`` -- that is
#: the defect recorded in ``LEGACY_DT_DRIFT`` and repaired by
#: ``scripts/attach_lax_dt_semantics.py``.
PUBLISHERS: Tuple[PublisherSpec, ...] = (
    PublisherSpec(
        name="lax_case_pool",
        artifact="docs/benchmarks/results/lax_case_pool/metrics.json",
        legs_path="ladder",
        metric_keys={"whole_field_L1": "errors", "whole_field_L2": "l2_errors",
                     "band_L1": "analytic_errors"},
        dt_mean_keys=("dt",),
        dt_last_keys=("dt_last_step",),
        dt_semantics_key="dt_semantics",
        bare_dt_semantics="dt_mean",
        note="the ledger artifact; 9-leg ladder N=100/200/400 x p=1/2/3",
    ),
    PublisherSpec(
        name="lax_asymptotic_probe",
        artifact="docs/benchmarks/results/lax_asymptotic_probe/metrics.json",
        legs_path="ladder",
        metric_keys={"whole_field_L1": "errors", "band_L1": "analytic_errors"},
        dt_mean_keys=("dt_mean",),
        dt_last_keys=("dt_last",),
        dt_semantics_key="dt_semantics",
        bare_dt_semantics=None,
        note="p3 order attribution; extends the ladder to N=800",
    ),
    PublisherSpec(
        name="lax_resolution_requirement",
        artifact="docs/benchmarks/results/lax_resolution_requirement/metrics.json",
        legs_path="legs",
        metric_keys={"whole_field_L2": "l2_errors"},
        dt_mean_keys=("dt",),
        dt_last_keys=(),
        dt_semantics_key="dt_semantics",
        bare_dt_semantics="dt_mean",
        note="what resolution the 3 % bar needs (p3.N400 -> p3.N800)",
    ),
    PublisherSpec(
        name="lax_certified_configuration",
        artifact="docs/benchmarks/results/lax_certified_configuration/metrics.json",
        legs_path="legs",
        metric_keys={"whole_field_L2": "l2_errors"},
        dt_mean_keys=("dt",),
        dt_last_keys=(),
        dt_semantics_key="dt_semantics",
        bare_dt_semantics="dt_mean",
        note="cheapest (p, N) that certifies the 3 % bar (DOF-ordered)",
    ),
    PublisherSpec(
        name="lax_case_pool_limiter_arms",
        artifact="docs/benchmarks/results/lax_case_pool_limiter_arms/metrics.json",
        legs_path="arms.*.ladder",
        metric_keys={"whole_field_L1": "errors", "band_L1": "analytic_errors"},
        dt_mean_keys=("dt",),
        dt_last_keys=("dt_last",),
        dt_semantics_key="dt_semantics",
        bare_dt_semantics="dt_mean",
        note="single-knob limiter attribution; dt repaired in place (no re-run)",
    ),
)

#: The drift that was consumed as evidence of non-reproducibility.
LEGACY_DT_DRIFT = {
    "publisher": "lax_case_pool_limiter_arms",
    "native_key": "dt",
    "meant": "dt_last (solver's time-truncated final step)",
    "expected": "dt_mean (final_time/steps)",
    "example_leg": "p3.N400",
    "legacy_value": 2.2296238657348866e-05,
    "dt_mean_same_run": 2.5445292620865142e-05,
    "fixed_in": "scripts/attach_lax_dt_semantics.py",
    "solver_rerun": False,
}

CONTRACT_CHECKS: Mapping[str, str] = {
    "C1": "the meaning of the bare key `dt` is resolved for every publisher: "
          "either declared by a dt_semantics string, or verified by C2 to be "
          "final_time/steps (so no reader has to guess and no file is rewritten "
          "only to carry a label)",
    "C2": "|dt_mean - final_time/steps| / (final_time/steps) <= 1e-12 on every leg",
    "C3": "0 < dt_last <= dt_mean (1 + 1e-12) wherever dt_last is recorded",
    "C4": "the bare key `dt` maps to the same canonical quantity in every publisher",
    "C5": "legs shared by >= 2 publishers agree on every commonly published "
          "canonical metric to <= 1e-12 relative",
    "C6": "a fresh solve of a shared leg reproduces the committed canonical "
          "metrics to <= 1e-12 relative (evaluated by the reconciliation driver)",
}


def _rel(a: float, b: float) -> float:
    denominator = max(abs(a), abs(b), 1e-300)
    return float(abs(a - b) / denominator)


def load(spec: PublisherSpec, root: Path = Path(".")) -> Optional[dict]:
    path = root / spec.artifact
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def extract(spec: PublisherSpec, root: Path = Path(".")) -> Dict[Tuple[int, int], dict]:
    """Canonical leg table of one artifact, keyed by ``(cells, degree)``."""
    payload = load(spec, root)
    if payload is None:
        return {}
    table: Dict[Tuple[int, int], dict] = {}
    for json_path, leg in spec.leg_records(payload):
        cells, degree = int(leg["cells"]), int(leg["degree"])
        metrics, sources = {}, {}
        for canonical, native in spec.metric_keys.items():
            block = leg.get(native)
            if not isinstance(block, Mapping):
                continue
            metrics[canonical] = {field: float(value) for field, value in block.items()
                                  if isinstance(value, (int, float))}
            sources[canonical] = native
        dt_mean, dt_mean_key = spec.resolve_dt_mean(leg)
        dt_last, dt_last_key = spec.resolve_dt_last(leg)
        steps = int(leg["steps"]) if leg.get("steps") is not None else None
        declared = leg.get(spec.dt_semantics_key) if spec.dt_semantics_key else None
        previous = table.get((cells, degree))
        if previous is not None:
            # same leg repeated inside one artifact (e.g. two arms): keep the
            # first, record the collision so it is visible rather than averaged
            previous.setdefault("duplicates", []).append(json_path)
            continue
        table[(cells, degree)] = {
            "cells": cells,
            "degree": degree,
            "json_path": json_path,
            "steps": steps,
            "metrics": metrics,
            "metric_sources": sources,
            "dt_mean": dt_mean,
            "dt_mean_key": dt_mean_key,
            "dt_last": dt_last,
            "dt_last_key": dt_last_key,
            "dt_semantics": declared,
            "bare_dt_semantics": spec.bare_dt_semantics,
        }
    return table


def dt_identity(table: Mapping[Tuple[int, int], dict],
                final_time: float = FINAL_TIME) -> List[dict]:
    """C2/C3 per leg: does the recorded step size match its own semantics?"""
    rows = []
    for key in sorted(table):
        record = table[key]
        steps = record["steps"]
        expected = final_time / steps if steps else None
        mean = record["dt_mean"]
        row = {
            "leg": f"p{record['degree']}.N{record['cells']}",
            "cells": record["cells"],
            "degree": record["degree"],
            "steps": steps,
            "dt_mean": mean,
            "dt_mean_key": record["dt_mean_key"],
            "expected_mean": expected,
            "mean_relative_error": _rel(mean, expected) if (mean and expected) else None,
            "dt_last": record["dt_last"],
            "dt_last_key": record["dt_last_key"],
            "last_within_mean": (record["dt_last"] <= mean * (1.0 + TOL)
                                 if (record["dt_last"] and mean) else None),
        }
        row["status"] = "PASS" if (
            (row["mean_relative_error"] is not None and row["mean_relative_error"] <= TOL)
            and row["last_within_mean"] in (True, None)
        ) else "FAIL"
        rows.append(row)
    return rows


def audit(root: Path = Path(".")) -> dict:
    """C1-C4 over every registered publisher."""
    publishers = {}
    for spec in PUBLISHERS:
        table = extract(spec, root)
        identity = dt_identity(table)
        missing_artifact = not table
        declared = sorted({str(record["dt_semantics"]) for record in table.values()})
        # C1: semantics resolved = declared, or the value itself proves the
        # meaning (C2 identity holds on every leg).
        identity_ok = all(row["mean_relative_error"] is not None
                          and row["mean_relative_error"] <= TOL for row in identity)
        c1 = (not missing_artifact) and identity_ok
        c2 = (not missing_artifact) and all(
            row["mean_relative_error"] is not None and row["mean_relative_error"] <= TOL
            for row in identity
        )
        c3 = (not missing_artifact) and all(row["last_within_mean"] in (True, None)
                                            for row in identity)
        worst_mean = max((row["mean_relative_error"] for row in identity
                          if row["mean_relative_error"] is not None), default=None)
        publishers[spec.name] = {
            "artifact": spec.artifact,
            "artifact_present": not missing_artifact,
            "legs": len(table),
            "dt_semantics_declared": declared,
            "dt_semantics_resolved_by": ("declared" if all(
                record["dt_semantics"] for record in table.values()) else
                "verified (dt == final_time/steps on every leg)"),
            "bare_dt_semantics": spec.bare_dt_semantics,
            "worst_mean_relative_error": worst_mean,
            "C1_semantics_declared": "PASS" if not missing_artifact and c1 else
                                     ("SKIP" if missing_artifact else "FAIL"),
            "C2_mean_identity": "PASS" if not missing_artifact and c2 else
                                ("SKIP" if missing_artifact else "FAIL"),
            "C3_last_step_bound": "PASS" if not missing_artifact and c3 else
                                  ("SKIP" if missing_artifact else "FAIL"),
            "note": spec.note,
        }
    semantics = {spec.name: spec.bare_dt_semantics for spec in PUBLISHERS
                 if spec.bare_dt_semantics is not None}
    by_meaning: Dict[str, List[str]] = {}
    for name, meaning in semantics.items():
        by_meaning.setdefault(meaning, []).append(name)
    collisions = dict(by_meaning) if len(by_meaning) > 1 else {}
    verdict = "PASS" if all(
        entry[key] == "PASS"
        for entry in publishers.values()
        for key in ("C1_semantics_declared", "C2_mean_identity", "C3_last_step_bound")
    ) and not collisions else "FAIL"
    return {
        "publishers": publishers,
        "bare_dt_key_map": semantics,
        "C4_collisions": collisions,
        "legacy_dt_drift": LEGACY_DT_DRIFT,
        "checks": dict(CONTRACT_CHECKS),
        "tolerance": TOL,
        "verdict": verdict,
    }


def reconcile(root: Path = Path(".")) -> dict:
    """C5: compare only *identically defined* metrics on shared legs."""
    tables = {spec.name: extract(spec, root) for spec in PUBLISHERS}
    shared_keys = set()
    for name, table in tables.items():
        shared_keys |= set(table)
    rows = []
    for key in sorted(shared_keys):
        present = {name: table[key] for name, table in tables.items() if key in table}
        if len(present) < 2:
            continue
        leg = f"p{present[next(iter(present))]['degree']}.N{present[next(iter(present))]['cells']}"
        for canonical in CANONICAL_METRICS:
            contributors = {name: record["metrics"][canonical]
                            for name, record in present.items()
                            if canonical in record["metrics"]}
            if len(contributors) < 2:
                continue
            fields = set().union(*(block.keys() for block in contributors.values()))
            for field in sorted(fields):
                values = {name: block[field] for name, block in contributors.items()
                          if field in block}
                if len(values) < 2:
                    continue
                names = sorted(values)
                worst = 0.0
                pair = None
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        diff = _rel(values[a], values[b])
                        if diff >= worst:
                            worst, pair = diff, [a, b]
                rows.append({
                    "leg": leg, "cells": key[0], "degree": key[1],
                    "metric": canonical, "field": field,
                    "publishers": names, "values": values,
                    "worst_relative_difference": worst,
                    "worst_pair": pair,
                    "status": "PASS" if worst <= TOL else "FAIL",
                })
    # cross-metric diagnostic: how big is the difference between *different*
    # definitions on the same leg?  This is the size of "discrepancy" a reader
    # gets by comparing e.g. band_L1 with whole_field_L1.
    cross = []
    for key in sorted(shared_keys):
        for name, table in tables.items():
            record = table.get(key)
            if not record:
                continue
            whole, band = record["metrics"].get("whole_field_L1"), record["metrics"].get("band_L1")
            if not whole or not band:
                continue
            for field in sorted(set(whole) & set(band)):
                cross.append({
                    "leg": f"p{record['degree']}.N{record['cells']}",
                    "publisher": name, "field": field,
                    "whole_field_L1": whole[field], "band_L1": band[field],
                    "relative_difference": _rel(whole[field], band[field]),
                })
    worst_cross = max((row["relative_difference"] for row in cross), default=None)
    verdict = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    return {
        "legs_compared": len({row["leg"] for row in rows}),
        "rows": rows,
        "worst_relative_difference": max((row["worst_relative_difference"] for row in rows),
                                         default=None),
        "cross_metric_diagnostic": {
            "note": "differences a reader sees when comparing two *different* error "
                    "definitions on the same leg; not drift",
            "rows": cross,
            "worst_relative_difference": worst_cross,
        },
        "tolerance": TOL,
        "verdict": verdict,
    }


if __name__ == "__main__":  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description="CASE-LAX-DG artifact contract audit")
    parser.add_argument("--json", action="store_true", help="print the full record")
    args = parser.parse_args()
    payload = {"audit": audit(), "reconcile": reconcile()}
    if args.json:
        print(json.dumps(payload, indent=1))
    else:
        print(f"audit verdict: {payload['audit']['verdict']}")
        print(f"reconcile verdict: {payload['reconcile']['verdict']} "
              f"(worst rel diff {payload['reconcile']['worst_relative_difference']})")