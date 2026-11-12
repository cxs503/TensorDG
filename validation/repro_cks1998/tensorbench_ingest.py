"""REPRO-DG-CKS1998 → 平台 TensorBench 取证回路（ref/got 由平台复算，不采信自述）。

三次审计的共同欠账（`docs/audit_m15292_*.md`、`docs/AUDIT_m15345_*.md`、
`docs/REPRO-DG-CKS1998-round2.md` §6）：本锚点的判据只活在 Markdown/仓内 JSON，
``tensorbench_results`` 零行 ⇒ ``GET /tensor/bench/l4-verdict/<mid>`` 恒答
``CLAIMED_NOT_PROVEN``。本驱动把「集群作业原始 stdout → 平台复算 → 实证行 → L4 裁定」
接成一条可重跑的命令。

两档
----
``--validate-only``
    用 ``bench_spec.validate_documents``（与平台 ``contract.py`` 同算式的仓内等价实现）
    把 spec / result / stdout 三件文档全量自检：键集合、身份一致、指针可取、逐度量复算。
    不碰网络、不写数据库，可在任何机器（含集群节点）跑。

``--post --job-id N``
    同一批文档，``POST /tensor/bench/results``。平台侧会**再次**校验并按指针从
    ``workload_stdout`` 复算 ref/got/误差；``cluster_jobs`` 必须是同一 mission 的
    succeeded 作业（平台反造假绑定，本驱动无法绕过，故不提供 bypass 开关）。

用法
----
    python validation/repro_cks1998/tensorbench_ingest.py --emit-spec tensorbench/REPRO-DG-CKS1998.spec.json
    python validation/repro_cks1998/tensorbench_ingest.py --stdout runs/cks1998_tensorbench/stdout.txt --validate-only
    python validation/repro_cks1998/tensorbench_ingest.py --stdout ... --post --job-id 1301 --mission 17011
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from validation.repro_cks1998 import bench_spec                     # noqa: E402

API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASSWORD = os.environ.get("TENSOR_AGENT_PASS", "")
WORKLOAD_CMD = ["python", "validation/repro_cks1998/bench_workload.py", "--profile", "bench"]
CONFIGURATION = {"case": "CKS1998 1-D RKDG benchmark set (A1 smooth advection + E2 Sod)",
                 "A1": {"final_time": 0.5, "cfl": 0.3, "cells": [20, 40, 80, 160, 320],
                        "degrees": [1, 2, 3], "metric": "L2 order vs theory k+1"},
                 "E2": {"degree": 2, "cells": 400, "final_time": 0.2, "cfl": 0.12,
                        "limiter": "tvb M=1.0", "limiter_stage": "final",
                        "left": [1.0, 0.0, 1.0], "right": [0.125, 0.0, 0.1],
                        "ref": "exact Riemann solution (Toro), benchmarks.sod_exact"}}


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:                                    # pragma: no cover
        return os.environ.get("TENSORDG_REVISION", "unknown")


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def _get(url: str, token: str) -> dict:
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def login() -> str:
    if not PASSWORD:
        raise SystemExit("TENSOR_AGENT_PASS 未设置，无法取证")
    out = _post(f"{API}/auth/login", {"username": USER, "password": PASSWORD})
    return out.get("access_token") or out.get("token")


def render(rows, stream=sys.stdout) -> None:
    print(f"{'metric':22s} {'ref':>14s} {'got':>14s} {'rel_err':>10s} "
          f"{'tol':>8s}  verdict", file=stream)
    for r in rows:
        rel = "n/a" if r["rel_error"] is None else f"{r['rel_error']:.4%}"
        print(f"{r['metric']:22s} {r['ref_value']:14.6g} {r['got_value']:14.6g} "
              f"{rel:>10s} {r['tolerance_rel']:8.2%}  "
              f"{'PASS' if r['passed'] else 'FAIL'}", file=stream)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout", help="集群作业原始 stdout 文件（平台复算的唯一输入）")
    ap.add_argument("--mission", type=int, default=int(os.environ.get("MISSION_ID", "0")))
    ap.add_argument("--job-id", type=int, default=0)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--emit-spec", default=None, help="把 spec 写到指定路径")
    ap.add_argument("--out", default=None, help="把 spec/result 文档写到目录")
    args = ap.parse_args()

    spec = bench_spec.build_spec(revision=args.revision or git_revision(),
                                 command=WORKLOAD_CMD, configuration=CONFIGURATION,
                                 notes="REPRO-DG-CKS1998 platform evidence loop: "
                                       "ref=theory order / exact Riemann; got=DG measurement.")
    if args.emit_spec:
        path = Path(args.emit_spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n")
        print(f"spec → {path}")

    stdout = None
    if args.stdout:
        stdout = Path(args.stdout).read_text()
        result = bench_spec.build_result(json.loads(stdout), mission_id=args.mission or None)
        if args.out:
            outdir = Path(args.out)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1))
            (outdir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
        checked = bench_spec.validate_documents(spec, result, stdout)
        render(checked["rows"])
        print(f"passed: {checked['passed']}\nfailed: {checked['failed']}")
        if args.validate_only or not args.post:
            return 0 if not checked["failed"] else 1
        if not args.job_id:
            raise SystemExit("--post 需要 --job-id（平台要求 succeeded 集群作业绑定）")
        token = login()
        payload = {"mission_id": args.mission, "cluster_job_id": args.job_id,
                   "spec": spec, "result": result, "workload_stdout": stdout}
        try:
            out = _post(f"{API}/tensor/bench/results", payload, token)
        except urllib.error.HTTPError as exc:
            print(f"HTTP {exc.code}: {exc.read().decode()[:800]}", file=sys.stderr)
            return 2
        print(json.dumps(out, ensure_ascii=False, indent=1))
        verdict = _get(f"{API}/tensor/bench/l4-verdict/{args.mission}", token)
        print("l4-verdict:", json.dumps(verdict, ensure_ascii=False))
        return 0 if verdict.get("l4_benchmark_pass") else 3
    if args.emit_spec:
        return 0
    ap.error("需要 --stdout（或仅 --emit-spec）")


if __name__ == "__main__":
    raise SystemExit(main())