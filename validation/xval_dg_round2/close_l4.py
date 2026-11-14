"""Close the L4 machine gate for XVAL-DG-ROUND2-CERTIFIED-REFERENCE on one cluster job.

The anchor is registered (``tensor_benchmark_cases`` id 79) but the platform's
L4 rule (``tools/maturity_gate/benchmark_source.py``) needs a
``tensorbench_results`` row that the platform can recompute, bound to a
*succeeded cluster job of the same mission*.  ``docs/AUDIT-RESPONSE-XVAL-DG-ROUND2-B1.md``
§4 refused to invent that job id -- correctly -- so the item stayed open.

This driver is the ingest half of the closure, with three honesty rules:

1. the ``workload_stdout`` posted is the **verbatim bytes** of the cluster job's
   stdout (path given by ``--stdout``), never a locally rebuilt document; its
   sha256 is recorded in the evidence file next to the job id;
2. ``spec``/``result`` are built by :mod:`validation.xval_dg_round2.post_platform`
   from the same raw profiles the job produced (``--rawdir``), so the numbers the
   platform recomputes are the job's numbers;
3. the ingest is attempted **only** when the job actually succeeded
   (``--job-id``), and the script exits non-zero unless the platform's own
   ``l4-verdict`` comes back ``VERIFIED``.

Usage::

    python validation/xval_dg_round2/close_l4.py --mission 20997 --job-id cj3111 \
        --rawdir runs/shu_osher_case_pool/raw --stdout runs/xval_round2_cluster/stdout.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.xval_dg_round2 import post_platform as pp  # noqa: E402

ART = ROOT / "validation" / "xval_dg_round2" / "artifacts"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mission", type=int, required=True)
    ap.add_argument("--job-id", required=True, help="cj<N> of the succeeded cluster job")
    ap.add_argument("--rawdir", required=True)
    ap.add_argument("--stdout", required=True, help="verbatim stdout of that cluster job")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    stdout_text = Path(args.stdout).read_text()
    stdout_sha = hashlib.sha256(stdout_text.encode()).hexdigest()
    docs = pp.build(args.mission, args.job_id, rawdir=args.rawdir)
    docs["workload_stdout"] = stdout_text

    report = pp.validate(docs)
    print("contract validation:", json.dumps({k: v for k, v in report.items() if k != "rows"}))
    for row in report["rows"]:
        print("  ", json.dumps(row))
    if not report["all_passed"]:
        raise SystemExit("a metric exceeds its tolerance: refusing to post")

    token = pp.login()
    jid = int(str(args.job_id).lstrip("cj") or 0)  # the API wants the bare integer id
    code, d = pp._call("POST", "/tensor/bench/results", {
        "mission_id": args.mission, "cluster_job_id": jid,
        "spec": docs["spec"], "result": docs["result"],
        "workload_stdout": stdout_text}, token)
    print("POST /tensor/bench/results ->", code, json.dumps(d)[:2000])
    if code != 200:
        raise SystemExit(1)
    code, verdict = pp._call("GET", f"/tensor/bench/l4-verdict/{args.mission}", None, token)
    print("l4-verdict ->", code, json.dumps(verdict, indent=1))

    evidence = {
        "anchor": "XVAL-DG-ROUND2-CERTIFIED-REFERENCE",
        "mission_id": args.mission,
        "cluster_job_id": args.job_id,
        "cluster_job_id_int": jid,
        "workload_stdout_sha256": stdout_sha,
        "rawdir": args.rawdir,
        "implementation_revision": docs["result"]["implementation"]["revision"],
        "ingest": {"http_status": code, "response": d},
        "l4_verdict": verdict,
        "recomputed_rows": report["rows"],
        "validator_mode": report.get("mode"),
    }
    out = Path(args.out) if args.out else ART / f"l4_closure_m{args.mission}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0 if verdict.get("status") == "VERIFIED" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())