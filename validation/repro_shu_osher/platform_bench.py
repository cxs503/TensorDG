"""Register / query the TensorDG Shu--Osher case on the platform bench API.

Two platform-side integration actions (no repo writes, no secrets in output):

  1. ``POST /tensor/bench/evaluations`` - submit the case as a benchmark spec for
     this mission (ref source, metric, tolerance, repro command).
  2. ``GET  /tensor/bench/l4-verdict/<mission>`` - read back the platform's own
     L4 verdict for the mission.

Usage::

    python validation/repro_shu_osher/platform_bench.py --mission 15480
    python validation/repro_shu_osher/platform_bench.py --dry-run   # print payload, no network

Reads ``metrics.json`` from ``runs/shu_osher_case_pool/`` when the full study has
been re-run, otherwise from the git-tracked bundle (see :mod:`bundle`) - so the
documented command also works on a clean checkout.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

try:  # both ``python validation/.../platform_bench.py`` and ``-m`` must work
    from . import bundle
except ImportError:  # pragma: no cover - plain-script invocation
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bundle

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
OUT = bundle.CURATED
BENCH = bundle.BUNDLE / "benchmark_ref_vs_got.json"


def call(method: str, path: str, payload=None, token=None, timeout=120):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        return e.code, {"error": body[:2000]}


def login() -> str:
    user = os.environ.get("TENSOR_AGENT_USER")
    pwd = os.environ.get("TENSOR_AGENT_PASS")
    if not (user and pwd):
        raise SystemExit("TENSOR_AGENT_USER/TENSOR_AGENT_PASS not set")
    code, d = call("POST", "/auth/login", {"username": user, "password": pwd})
    if code != 200 or not d.get("access_token"):
        raise SystemExit(f"login failed: {code} {d}")
    return d["access_token"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mission", type=int, default=15480)
    ap.add_argument("--dry-run", action="store_true", help="print the payload only")
    args = ap.parse_args(argv)

    metrics, metrics_path, metrics_source = bundle.load_metrics()
    bench = json.loads(BENCH.read_text())
    best = bench["best_configuration"]
    spec = {
        "anchor_slug": "CASE-SHU-OSHER-DG",
        "case_name": "Shu-Osher shock/entropy-wave interaction (Mach 3 shock into "
                     "rho=1+0.2 sin(5x), gamma=1.4, t=1.8)",
        "repo": "TensorDG",
        "ref_source": "Shu & Osher, J. Comput. Phys. 77 (1988) 439-471 (restated as "
                      "test case 3, J. Comput. Phys. 83 (1989) 32-78); no tabulated "
                      "reference values in the papers, so the numeric reference is an "
                      "independent FV-WENO5/HLLC/SSP-RK3 solution at N=3200 whose own "
                      "grid convergence is 0.101% L1 (N=1600 vs N=3200), plus closed-form "
                      "Rankine-Hugoniot / mass-budget anchors.",
        "metric": "relative L1 of the density profile vs the independent reference",
        "tolerance": "<5%",
        "repro_cmd": "bash validation/repro_shu_osher/run_all.sh",
        "reference_cells": bench["reference_provenance"]["independent_numeric_reference"],
        "configurations": len(bench["cfgs"]),
        "got": {"configuration": best["configuration"],
                "rel_l1_percent": best["got_rel_l1_percent"],
                "rel_l2_percent": best["got_rel_l2_percent"],
                "passes": best["passes"]},
        "reference_self_convergence_percent": bench["reference_self_convergence_percent"],
        "all_configs_pass": bench["all_configs_pass"],
        "ladder_rel_l1_percent_range": [min(r["got_rel_l1_percent"] for r in bench["cfgs"]),
                                        max(r["got_rel_l1_percent"] for r in bench["cfgs"])],
        # provenance of the submitted numbers: fresh study or tracked bundle
        "artifact_source": metrics_source,
    }
    if args.dry_run:
        print(json.dumps(spec, indent=2, ensure_ascii=False))
        return 0

    token = login()
    code, ev = call("POST", "/tensor/bench/evaluations",
                    {"mission_id": args.mission, "spec": spec}, token)
    code_v, verdict = call("GET", f"/tensor/bench/l4-verdict/{args.mission}", None, token)
    status = "ingested"
    if code == 422:
        # the TensorBench evaluation channel needs a *registered* benchmark/workload
        # pair (spec.benchmark.{benchmark_id,version} + spec.workload.workload_id);
        # this case is not in that registry, so the API rejects the spec.  Recorded
        # verbatim rather than dropped - see evidence.md for the friction note.
        status = "not_registered_in_tensorbench"
    record = {"mission_id": args.mission, "artifact_source": metrics_source,
              "metrics_path": str(metrics_path),
              "evaluation_status": code, "evaluation_status_note": status,
              "evaluation": ev,
              "l4_verdict_status": code_v, "l4_verdict": verdict, "spec": spec,
              "tabulated_rel_l1_percent": [r["got_rel_l1_percent"] for r in bench["cfgs"]],
              "metrics_case": metrics["case"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "platform_bench.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"evaluation_status": code, "evaluation": ev,
                      "l4_verdict_status": code_v, "l4_verdict": verdict},
                     indent=2, ensure_ascii=False)[:2000])
    print(f"wrote {OUT/'platform_bench.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())