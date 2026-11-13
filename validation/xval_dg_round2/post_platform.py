"""Platform evidence loop for XVAL-DG-ROUND2-CERTIFIED-REFERENCE.

Why this file exists
--------------------
The round-2 audit (``docs/AUDIT-m17539-XVAL-DG-ROUND2.md``, finding B2) recorded
the one gap that kept the case from being *machine*-verifiable: the anchor had
**zero rows** in ``tensor_benchmark_cases`` and no ``tensorbench_results`` row,
so the platform's own ``GET /tensor/bench/l4-verdict/<mission>`` answered
``CLAIMED_NOT_PROVEN`` no matter how good the numbers were.  The fix has two
halves; this file is the second one (the first is the registered anchor row):

* **build**    -- turn ``benchmarks.xval_dg_round2_certified`` into the platform's
  strict documents: ``spec`` (top-level keys limited to ``{benchmark, workload,
  execution, evaluation, implementation, notes}``), ``result`` (the eight
  required keys of ``benchmark-result.schema.json`` v1.0.0) and the workload
  **stdout document** the JSON pointers index.
* **validate** -- run the platform's *own* ``tools.tensorbench_eval.contract``
  parser/validator/metric evaluator on those documents, so a schema rejection
  is found here instead of as a 422 in production.
* **post**     -- ``POST /tensor/bench/results`` for a mission, bound to a
  ``cj<N>`` cluster job of the *same* mission.  The platform refuses the ingest
  otherwise (anti-fabrication binding), and this driver refuses to invent a job
  id: without ``--job-id`` it stops before the network.

Nothing is hand-typed: every number in the documents is computed from the
committed raw profiles by the benchmark module, and the platform re-computes
them again from the stdout bytes.

Usage::

    python validation/xval_dg_round2/post_platform.py --mission 18730
    python validation/xval_dg_round2/post_platform.py --mission 18730 --post --job-id cj1234
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import xval_dg_round2_certified as xval  # noqa: E402

ART = ROOT / "validation" / "xval_dg_round2" / "artifacts"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
#: Where the platform's TensorBench contract module may be checked out.
PLATFORM_CANDIDATES = ("/root/tensor-ecosystem/TensorPlatform",)


def git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - only when git is unavailable
        return "unknown"


# ------------------------------------------------------------------ documents
def build(mission_id: int, job_id: str | None = None,
          rawdir: str | Path = xval.DEFAULT_RAW) -> dict:
    """Build ``{spec, result, workload_stdout}`` from the reference audit."""
    audit = xval.cross_validate(rawdir)
    block = xval.platform_block(audit)
    stdout_doc = block["stdout_document"]
    metrics = {row["name"]: {"value": row["got"], "unit": row["unit"]}
               for row in audit["D3_reference_vs_oracle"]["rows"]
               if row["kind"] == "relative"}
    spec = {
        "benchmark": dict(block["benchmark"]),
        "workload": dict(block["workload"]),
        "implementation": {"project": "TensorDG", "revision": git_rev()},
        "execution": {"timeout_seconds": 3600},
        "evaluation": {
            "extract": {"format": "json_stdout"},
            # the platform contract allows exactly {name, got_pointer, ref_pointer,
            # tolerance_rel, unit, better}; the human descriptions stay in the
            # repo-side audit artifact, not in the spec
            "metrics": [{k: m[k] for k in ("name", "got_pointer", "ref_pointer",
                                           "tolerance_rel", "unit")}
                        for m in block["metrics"]],
            "notes": "ref = closed-form oracle (Rankine-Hugoniot plateau, Hugoniot "
                     "shock position, exact mass budget) evaluated on the certified "
                     "reference profile; got = the same quantities measured on the "
                     "reference. Pointers index the workload stdout document.",
        },
        "notes": "round-2 xval (m18730): cross-validates the certified reference against "
                 "sources that do not share its discretisation; per-leg relative L1 gates "
                 "stay repo-side (an L1 error is not expressible as |got-ref|/|ref|).",
    }
    result = {
        "schema_version": "1.0.0",
        "run_id": f"xval-dg-round2-{block['workload']['workload_id']}",
        "benchmark": dict(block["benchmark"]),
        "implementation": {"project": "TensorDG", "revision": git_rev()},
        "workload": dict(block["workload"]),
        "execution": {"status": "success", "mission_id": int(mission_id),
                      "cluster_job_id": job_id},
        "metrics": metrics,
        "validation": {
            "reference_ncells": audit["D1_reference_usable"]["reference_ncells"],
            "band_percent": audit["D1_reference_usable"]["band_percent"],
            "band_share_of_gate": audit["D1_reference_usable"]["band_share_of_gate"],
            "strict_percent": audit["thresholds"]["strict_percent"],
            "worst_leg_rel_l1_percent": audit["D2_legs_under_strict_gate"]["worst_rel_l1_percent"],
            "accepted": audit["accepted"],
            "bounds": block["bounds"],
            "source": "validation/xval_dg_round2/post_platform.py",
        },
    }
    return {"spec": spec, "result": result,
            "workload_stdout": json.dumps(stdout_doc, indent=1, sort_keys=True)}


# ------------------------------------------------------------------ validation
def _platform_contract():
    for cand in PLATFORM_CANDIDATES:
        if (Path(cand) / "tools/tensorbench_eval/contract.py").exists():
            if cand not in sys.path:
                sys.path.insert(0, cand)
            from tools.tensorbench_eval import contract  # noqa: PLC0415
            return contract
    return None


def validate(docs: dict) -> dict:
    """Platform contract first; a local pointer/schema check as the fallback."""
    contract = _platform_contract()
    if contract is not None:
        parsed = contract.parse_spec(docs["spec"])
        meta = contract.validate_result(docs["result"], parsed)
        doc = contract.extract_document(docs["workload_stdout"])
        rows = contract.evaluate_metrics(parsed, doc)
        return {"mode": "platform_contract", "result_meta": meta, "rows": rows,
                "all_passed": all(r["passed"] for r in rows)}
    # ---- fallback: resolve every pointer in the stdout document, locally
    doc = json.loads(docs["workload_stdout"])
    rows = []
    for metric in docs["spec"]["evaluation"]["metrics"]:
        def resolve(pointer: str):
            node = doc
            for part in pointer.split("."):
                node = node[part]
            return node
        got, ref = resolve(metric["got_pointer"]), resolve(metric["ref_pointer"])
        err = abs(got - ref) / abs(ref) if ref else float("inf")
        rows.append({"name": metric["name"], "got": got, "ref": ref,
                     "rel_error": err, "tolerance_rel": metric["tolerance_rel"],
                     "passed": bool(err <= metric["tolerance_rel"])})
    return {"mode": "local_pointer_check", "rows": rows,
            "all_passed": all(r["passed"] for r in rows)}


# ------------------------------------------------------------------ HTTP
def _call(method: str, path: str, payload=None, token: str | None = None, timeout=180):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "ignore")[:2000]}


def login() -> str:
    user, pwd = os.environ.get("TENSOR_AGENT_USER"), os.environ.get("TENSOR_AGENT_PASS")
    if not (user and pwd):
        raise SystemExit("TENSOR_AGENT_USER/TENSOR_AGENT_PASS not set")
    code, d = _call("POST", "/auth/login", {"username": user, "password": pwd})
    if code != 200 or not d.get("access_token"):
        raise SystemExit(f"login failed: {code} {d}")
    return d["access_token"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mission", type=int, required=True)
    ap.add_argument("--job-id", default=None,
                    help="cj<N> of a succeeded cluster job of the same mission (needed for --post)")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--out", default=str(ART / "xval_round2_bench_payload.json"))
    args = ap.parse_args(argv)

    docs = build(args.mission, args.job_id)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(docs, indent=1, sort_keys=True) + "\n")
    report = validate(docs)
    print(json.dumps(report, indent=1))
    print(f"wrote {args.out}", file=sys.stderr)
    if not report["all_passed"]:
        print("WARNING: at least one metric exceeds its tolerance", file=sys.stderr)
    if not args.post:
        return 0
    if not args.job_id:
        raise SystemExit("--post needs --job-id of a succeeded cluster job of this mission")
    token = login()
    code, d = _call("POST", "/tensor/bench/results", {
        "mission_id": args.mission, "cluster_job_id": args.job_id,
        "spec": docs["spec"], "result": docs["result"],
        "workload_stdout": docs["workload_stdout"]}, token)
    print("POST /tensor/bench/results ->", code, json.dumps(d)[:1500])
    code, v = _call("GET", f"/tensor/bench/l4-verdict/{args.mission}", None, token)
    print("l4-verdict ->", code, json.dumps(v)[:1500])
    return 0 if code == 200 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())