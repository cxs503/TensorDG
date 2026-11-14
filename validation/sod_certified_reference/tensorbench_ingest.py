"""TensorBench ingest for the Sod second physical case of the certified-reference engine.

The audit of the anchor closure (``docs/AUDIT-m17539-XVAL-DG-ROUND2.md``, B2)
left one blocking institutional gap: the ``XVAL-DG-ROUND2-CERTIFIED-REFERENCE``
anchor had no ``tensorbench_results`` row, so the platform's own
``GET /tensor/bench/l4-verdict/<mission>`` answered ``CLAIMED_NOT_PROVEN`` no
matter how good the numbers were.  A row cannot be *typed in*: the platform
binds it to a **succeeded cluster job of the same mission** and re-evaluates the
metrics from the raw workload stdout bytes.  This driver therefore only turns the
cluster job's own JSON stdout into the spec/result documents and posts them.

Metrics are physical quantities with an *analytic* reference (the closed-form
Riemann solution), never a relative error dressed up as ``|got-ref|/|ref|``
against a non-zero reference -- that would be a fabricated number:

============================  =========================================  ============
metric                        reference (closed form)                     tolerance
============================  =========================================  ============
``plateau_density``           rho*_R = 0.2655737 (post-shock plateau)     3 %
``star_left_density``         rho*_L = 0.4263194                          3 %
``shock_position``            x0 + W t = 0.8504311                        3 %
``conserved_mass``            0.5*1 + 0.5*0.125 = 0.5625                  1 %
============================  =========================================  ============

All four are *measured on the certified reference profile* (Richardson
extrapolation of the FV-WENO5 ladder) inside the workload, so the platform
re-derives them itself.

Usage
-----
    python validation/sod_certified_reference/tensorbench_ingest.py --validate-only
    python validation/sod_certified_reference/tensorbench_ingest.py --post --job-id 2705 --mission 20420
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
ART = ROOT / "artifacts/sod_certified_reference"
BENCH_ID, BENCH_VER = "XVAL-DG-ROUND2-CERTIFIED-REFERENCE-SOD", "1"
WORKLOAD_ID, WORKLOAD_VER = "sod-certified-reference-ladder", "1"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")

#: (metric, pointer suffix inside the audit document, tolerance, unit, description)
METRICS = [
    ("plateau_density", "analytic_anchors/plateau_density", 0.03, "-",
     "post-shock plateau density vs the closed-form Rankine-Hugoniot value"),
    ("star_left_density", "analytic_anchors/star_left_density", 0.03, "-",
     "star-left plateau density vs the closed-form star state"),
    ("shock_position", "analytic_anchors/shock_position", 0.03, "-",
     "shock location vs x0 + W t from the exact Riemann solution"),
    ("conserved_mass", "analytic_anchors/conserved_mass", 0.01, "-",
     "global mass integral vs the analytic 0.5625 (nothing has left the domain)"),
]


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # pragma: no cover
        return "unknown"


def build(mission_id: int, job_id: int) -> tuple[dict, dict, str]:
    stdout_path = ART / "run_stdout.json"
    if not stdout_path.exists():
        raise SystemExit(f"missing {stdout_path}: run the workload on the cluster first")
    stdout = stdout_path.read_text()
    doc = json.loads(stdout)
    if not doc.get("ladders", {}).get("fv"):
        raise SystemExit("workload stdout is not an audit document (no ladders block)")

    spec = {
        "benchmark": {"benchmark_id": BENCH_ID, "version": BENCH_VER},
        "workload": {
            "workload_id": WORKLOAD_ID, "version": WORKLOAD_VER,
            "command": ["python", "-m", "benchmarks.certified_reference_sod",
                        "--figures", "--json-stdout"],
            "cwd": ".", "precision": "float64",
            "configuration": {"case": "Sod 1978 shock tube", "gamma": 1.4,
                              "domain": [0.0, 1.0], "final_time": 0.2,
                              "certifier": "FV-WENO5-JS/HLLC/SSP-RK3",
                              "certifier_ladder": doc["ladders"]["fv"],
                              "certifier_cfl": doc["ladders"]["fv_cfl"],
                              "dg_ladder": doc["ladders"]["dg"],
                              "dg_cfl": doc["ladders"]["dg_cfl"],
                              "dg_degrees": doc["ladders"]["degrees"]},
        },
        "implementation": {"project": "TensorDG", "revision": _git_rev()},
        "execution": {"timeout_seconds": 5400},
        "evaluation": {
            "extract": {"format": "json_stdout"},
            "metrics": [
                {"name": name, "got_pointer": f"/{ptr}/measured",
                 "ref_pointer": f"/{ptr}/exact", "tolerance_rel": tol, "unit": unit}
                for name, ptr, tol, unit, _ in METRICS
            ],
            "notes": "ref = closed-form Riemann solution (external oracle, no shared "
                     "discretisation); got = value measured on the certified "
                     "reference profile inside the workload.",
        },
        "notes": "second physical case for XVAL-DG-ROUND2-CERTIFIED-REFERENCE: the "
                 "engine is exercised on a case with a known exact solution and its "
                 "GCI band is tested against the true error.",
    }
    result = {
        "schema_version": "1.0.0",
        "run_id": "sod-certified-reference-ladder",
        "benchmark": {"benchmark_id": BENCH_ID, "version": BENCH_VER},
        "implementation": {"project": "TensorDG", "revision": _git_rev()},
        "workload": {"workload_id": WORKLOAD_ID, "version": WORKLOAD_VER},
        "execution": {"status": "success", "mission_id": int(mission_id),
                      "cluster_job_id": int(job_id) if job_id else 0},
        "metrics": {name: {"value": doc["analytic_anchors"][ptr.split("/")[-1]]["measured"],
                           "unit": unit}
                    for name, ptr, _tol, unit, _ in METRICS},
        "validation": {
            "criteria": doc.get("criteria"),
            "accepted": doc.get("accepted"),
            "true_error_percent": doc["error_vs_exact"].get("true_error_percent"),
            "band_percent": doc["error_vs_exact"].get("band_percent"),
            "band_covers_true_error": doc["error_vs_exact"].get("band_covers_true_error"),
            "source": "benchmarks/certified_reference_sod.py",
        },
    }
    return spec, result, stdout


def _platform():
    for cand in ("/root/tensor-ecosystem/TensorPlatform", "/root/tp-agent/work/m9356"):
        p = Path(cand)
        if (p / "tools/tensorbench_eval/contract.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            break
    from tools.tensorbench_eval import contract  # noqa: E402
    return contract


def validate_only(spec: dict, result: dict, stdout: str) -> dict:
    contract = _platform()
    parsed = contract.parse_spec(spec)
    meta = contract.validate_result(result, parsed)
    doc = contract.extract_document(stdout)
    rows = contract.evaluate_metrics(parsed, doc)
    return {"parsed_ok": True, "result_meta": meta, "metrics": rows,
            "all_passed": all(r["passed"] for r in rows)}


def _call(method: str, path: str, payload=None, token: str | None = None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=180) as r:
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
    ap.add_argument("--mission", type=int, default=20420)
    ap.add_argument("--job-id", type=int, default=0)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--out", default=str(ART / "tensorbench_payload.json"))
    args = ap.parse_args(argv)

    spec, result, stdout = build(args.mission, args.job_id)
    Path(args.out).write_text(json.dumps({"spec": spec, "result": result,
                                          "workload_stdout": stdout}, indent=1))
    report = validate_only(spec, result, stdout)
    print(json.dumps(report, indent=1))
    if not report["all_passed"]:
        print("WARNING: at least one metric exceeds its tolerance", file=sys.stderr)
    if not args.post:
        return 0
    if not args.job_id:
        raise SystemExit("--post needs --job-id of a succeeded cluster job of this mission")
    token = login()
    code, d = _call("POST", "/tensor/bench/results", {
        "mission_id": args.mission, "cluster_job_id": args.job_id, "spec": spec,
        "result": result, "workload_stdout": stdout}, token)
    print("POST /tensor/bench/results ->", code, json.dumps(d)[:1500])
    code, v = _call("GET", f"/tensor/bench/l4-verdict/{args.mission}", None, token)
    print("l4-verdict ->", code, json.dumps(v)[:1500])
    return 0 if code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())