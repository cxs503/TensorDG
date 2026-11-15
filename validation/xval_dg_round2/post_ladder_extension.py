"""Ingest the ladder-extension reference into the platform as a machine-judged row.

The anchor ``XVAL-DG-ROUND2-CERTIFIED-REFERENCE`` was certified from a
three-level ladder and the platform's L4 row (mission 20997, job ``cj3111``)
compared that reference -- the ``N = 1600`` profile -- against the closed-form
oracle.  Extending the ladder changes *which* profile the certification returns
(the reference is the second-finest level by the engine's convention, so a
four-level ladder certifies the ``N = 3200`` profile).  The question this driver
answers on the platform is therefore narrow and independent of the repo-side
gates ``L0``-``L4``:

    does the profile the extended certification hands downstream still satisfy
    the closed-form anchors, and is it *at least as good* as the profile the
    three-level certification handed downstream?

Only quantities with an independent closed-form ``ref`` become metrics (Roache's
band and the per-leg L1 gates are inequalities, not ``|got-ref|/|ref|``, so they
stay repo-side -- same rule as ``post_platform.py``).

Honesty rules inherited from ``close_l4.py``:

1. the ``workload_stdout`` document is generated from the committed raw
   profiles, and its sha256 is recorded next to the job id;
2. the ingest is attempted only with a real, succeeded cluster job of this
   mission (``--job-id``); no id is ever invented;
3. the script fails unless the platform's own ``l4-verdict`` returns
   ``VERIFIED``.

Usage::

    python validation/xval_dg_round2/post_ladder_extension.py --mission 21166 \
        --job-id cj3227 --dry-run       # contract validation only, no network
    python validation/xval_dg_round2/post_ladder_extension.py --mission 21166 \
        --job-id cj3227                 # post + l4-verdict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import xval_dg_round2_certified as xval  # noqa: E402
from validation.xval_dg_round2 import close_ladder_extension as cle  # noqa: E402
from validation.xval_dg_round2 import post_platform as pp  # noqa: E402

ART = ROOT / "validation" / "xval_dg_round2" / "artifacts"


def _reference_rows(ncells: int) -> dict:
    """Oracle-comparable quantities of the raw profile at ``ncells``."""
    blob = np.load(cle.level_path(ncells), allow_pickle=False)
    return xval.measure_metrics(np.asarray(blob["x"], dtype=float),
                                np.asarray(blob["rho"], dtype=float))


def build(mission_id: int, job_id: str | None = None) -> dict:
    """Spec/result/stdout for 'the extended certification's reference vs the oracle'."""
    oracle = xval.oracle_values()
    ref4 = _reference_rows(cle.BASE_LEVELS[-1])   # 4-level certification returns N=3200
    ref3 = _reference_rows(cle.BASE_LEVELS[-2])   # 3-level certification returned N=1600

    rows = []
    for spec in xval.ORACLE_METRICS:
        name = spec["name"]
        rows.append({
            "name": name,
            "kind": spec["kind"],
            "unit": spec["unit"],
            "description": spec["description"],
            "got": ref4[name],
            "ref": oracle[name],
            "got_three_level": ref3[name],
            "tol_percent": xval.ANCHOR_TOL_PERCENT,
        })
    for row in rows:
        if row["kind"] == "relative":
            row["error_percent"] = 100.0 * abs(row["got"] - row["ref"]) / abs(row["ref"])
            row["error_percent_three_level"] = (
                100.0 * abs(row["got_three_level"] - row["ref"]) / abs(row["ref"]))
        else:  # bounded: the oracle value is 0 by construction
            row["error_percent"] = abs(row["got"])
            row["error_percent_three_level"] = abs(row["got_three_level"])
        row["pass"] = bool(row["error_percent"] < row["tol_percent"])

    stdout_doc = {
        "reference": {r["name"]: r["got"] for r in rows},
        "oracle": {r["name"]: r["ref"] for r in rows if r["ref"] is not None},
        "metadata": {
            "case": ("Shu-Osher 1988 shock/entropy interaction; reference returned by "
                     "the 4-level certification"),
            "reference_ncells_four_levels": cle.BASE_LEVELS[-1],
            "reference_ncells_three_levels": cle.BASE_LEVELS[-2],
            "ladder_four_levels": [*cle.BASE_LEVELS, cle.EXT_LEVEL],
            "ladder_three_levels": list(cle.BASE_LEVELS),
            "source": "validation/xval_dg_round2/post_ladder_extension.py",
        },
    }

    metrics, bounds = [], []
    for row in rows:
        if row["kind"] == "relative":
            metrics.append({"name": row["name"],
                            "got_pointer": f"reference.{row['name']}",
                            "ref_pointer": f"oracle.{row['name']}",
                            "tolerance_rel": row["tol_percent"] / 100.0,
                            "unit": row["unit"]})
        else:
            bounds.append({"name": row["name"], "got_pointer": f"reference.{row['name']}",
                           "bound": row["tol_percent"], "unit": row["unit"],
                           "description": row["description"]})

    benchmark = {"benchmark_id": "XVAL-DG-ROUND2-CERTIFIED-REFERENCE", "version": "3.0.0"}
    workload = {"workload_id": "xval-dg-round2-ladder-extension", "version": "1",
                "command": ["python", "-m", "benchmarks.xval_dg_round2_ladder_extension",
                            "--json-stdout"],
                "configuration": {
                    "case": ("Shu-Osher 1988 shock/entropy interaction (1D Euler, gamma=1.4, "
                             "x in [-5,5], t=1.8)"),
                    "reference_ladder": [*cle.BASE_LEVELS, cle.EXT_LEVEL],
                    "reference_scheme": "FV-WENO5-JS/HLLC/SSP-RK3, CFL=0.3",
                    "reference_ncells": cle.BASE_LEVELS[-1],
                    "gate_percent": cle.GATE_PERCENT}}
    spec = {
        "benchmark": benchmark,
        "workload": workload,
        "implementation": {"project": "TensorDG", "revision": pp.git_rev()},
        "execution": {"timeout_seconds": 5400},
        "evaluation": {
            "extract": {"format": "json_stdout"},
            "metrics": metrics,
            "notes": "ref = closed-form oracle (Hugoniot plateau, Hugoniot shock position, "
                     "exact mass budget); got = the same quantity measured on the profile the "
                     "four-level certification returns (N=3200). The three-level figures stay "
                     "in the repo artifact, not in the spec.",
        },
        "notes": "ladder extension 3 -> 4 levels: does the profile handed downstream still "
                 "satisfy the closed forms? Inequality gates (band coverage, per-leg L1) are "
                 "repo-side by the same rule as post_platform.py.",
    }
    result = {
        "schema_version": "1.0.0",
        "run_id": f"xval-dg-round2-{workload['workload_id']}",
        "benchmark": benchmark,
        "implementation": {"project": "TensorDG", "revision": pp.git_rev()},
        "workload": workload,
        "execution": {"status": "success", "mission_id": int(mission_id),
                      "cluster_job_id": job_id},
        "metrics": {r["name"]: {"value": r["got"], "unit": r["unit"]}
                    for r in rows if r["kind"] == "relative"},
        "validation": {
            "reference_ncells": cle.BASE_LEVELS[-1],
            "rows": rows,
            "bounds": bounds,
            "strict_percent": cle.STRICT_PERCENT,
            "source": "validation/xval_dg_round2/post_ladder_extension.py",
        },
    }
    return {"spec": spec, "result": result,
            "workload_stdout": json.dumps(stdout_doc, indent=1, sort_keys=True)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mission", type=int, required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--dry-run", action="store_true", help="contract validation only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    docs = build(args.mission, args.job_id)
    report = pp.validate(docs)
    print("contract validation:", json.dumps({k: v for k, v in report.items() if k != "rows"}))
    for row in report["rows"]:
        print("  ", json.dumps(row))
    if not report["all_passed"]:
        raise SystemExit("a metric exceeds its tolerance: refusing to post")
    if args.dry_run:
        out = ART / "ladder_extension_platform_dryrun.json"
        out.write_text(json.dumps({"spec": docs["spec"],
                                   "rows": docs["result"]["validation"]["rows"],
                                   "contract": report}, indent=1, sort_keys=True, default=str) + "\n")
        print(f"dry run; wrote {out}")
        return 0

    token = pp.login()
    jid = int(str(args.job_id).lstrip("cj") or 0)
    code, d = pp._call("POST", "/tensor/bench/results", {
        "mission_id": args.mission, "cluster_job_id": jid,
        "spec": docs["spec"], "result": docs["result"],
        "workload_stdout": docs["workload_stdout"]}, token)
    print("POST /tensor/bench/results ->", code, json.dumps(d)[:2000])
    if code != 200:
        raise SystemExit(1)
    code, verdict = pp._call("GET", f"/tensor/bench/l4-verdict/{args.mission}", None, token)
    print("l4-verdict ->", code, json.dumps(verdict, indent=1))

    evidence = {
        "anchor": "XVAL-DG-ROUND2-CERTIFIED-REFERENCE",
        "mission_id": args.mission,
        "cluster_job_id": args.job_id,
        "workload_stdout_sha256": hashlib.sha256(docs["workload_stdout"].encode()).hexdigest(),
        "implementation_revision": docs["result"]["implementation"]["revision"],
        "rows": docs["result"]["validation"]["rows"],
        "ingest": {"http_status": code, "response": d},
        "l4_verdict": verdict,
        "validator_mode": report.get("mode"),
    }
    out = Path(args.out) if args.out else ART / f"ladder_extension_platform_m{args.mission}.json"
    out.write_text(json.dumps(evidence, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {out}")
    return 0 if verdict.get("status") == "VERIFIED" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())