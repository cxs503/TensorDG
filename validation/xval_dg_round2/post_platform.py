"""Post the XVAL-DG-ROUND2 benchmark row to the platform (L4 machine gate).

Reads ``artifacts/xval_dg_round2_certified/platform_documents.json`` (written by
``benchmarks/xval_dg_round2_certified``), validates the documents with the
platform's *own* contract module when it is importable, posts them to
``POST /tensor/bench/results`` bound to a **succeeded** cluster job of the same
mission (the platform refuses otherwise -- anti-fabrication binding), then reads
back ``GET /tensor/bench/l4-verdict/<mission>``.

Usage::

    python validation/xval_dg_round2/post_platform.py --mission 17606 --job-id <cjN> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "artifacts/xval_dg_round2_certified/platform_documents.json"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")


def call(method, path, payload=None, token=None, timeout=180):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "ignore")[:4000]}


def login():
    user, pwd = os.environ.get("TENSOR_AGENT_USER"), os.environ.get("TENSOR_AGENT_PASS")
    if not (user and pwd):
        raise SystemExit("TENSOR_AGENT_USER/TENSOR_AGENT_PASS not set")
    code, d = call("POST", "/auth/login", {"username": user, "password": pwd})
    if code != 200 or not d.get("access_token"):
        raise SystemExit(f"login failed: {code} {d}")
    return d["access_token"]


def validate(docs):
    """Validate against the platform contract when that module is reachable."""
    for cand in ("/root/tensor-ecosystem/TensorPlatform",):
        p = Path(cand)
        if (p / "tools/tensorbench_eval/contract.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            break
    try:
        from tools.tensorbench_eval import contract
    except Exception as exc:  # pragma: no cover
        return {"contract_available": False, "reason": f"{type(exc).__name__}: {exc}"}
    parsed = contract.parse_spec(docs["spec"])
    meta = contract.validate_result(docs["result"], parsed)
    doc = contract.extract_document(docs["workload_stdout"])
    rows = contract.evaluate_metrics(parsed, doc)
    return {"contract_available": True, "result_meta": meta, "rows": rows,
            "all_passed": all(r["passed"] for r in rows)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mission", type=int, required=True)
    ap.add_argument("--job-id", type=int, required=True)
    ap.add_argument("--docs", default=str(DOCS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    docs = json.loads(Path(args.docs).read_text())
    report = validate(docs)
    print("contract:", json.dumps(report, default=str)[:1500])
    record = {"mission_id": args.mission, "cluster_job_id": args.job_id,
              "documents": str(args.docs), "validation": report}
    if args.dry_run:
        print(json.dumps(record, indent=1, default=str)[:2000])
        return 0
    token = login()
    code, d = call("POST", "/tensor/bench/results", {
        "mission_id": args.mission, "cluster_job_id": args.job_id,
        "spec": docs["spec"], "result": docs["result"],
        "workload_stdout": docs["workload_stdout"]}, token)
    record["post_status"] = code
    record["post_response"] = d
    code_v, v = call("GET", f"/tensor/bench/l4-verdict/{args.mission}", None, token)
    record["l4_verdict_status"] = code_v
    record["l4_verdict"] = v
    out = Path(args.docs).parent / "platform_post.json"
    out.write_text(json.dumps(record, indent=1, default=str) + "\n")
    print("POST /tensor/bench/results ->", code, json.dumps(d, default=str)[:1500])
    print("l4-verdict ->", code_v, json.dumps(v, default=str)[:1500])
    print("wrote", out)
    return 0 if code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
