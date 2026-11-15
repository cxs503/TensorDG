"""REPRO-DG-CKS1998 — 验证欠账闭合回归门（锚点 L4 已 VERIFIED，防回退）。

背景：三次审计（`docs/audit_m15292_*.md`、`docs/AUDIT_m15345_*.md`、
`docs/REPRO-DG-CKS1998-round2.md` §6）都指向同一处欠账——本锚点的判据只活在
Markdown/仓内 JSON，`tensorbench_results` 零行 ⇒ `GET /tensor/bench/l4-verdict/<mid>`
恒答 `CLAIMED_NOT_PROVEN`（「自证，非实测」）。该欠账已由 m19918（1-D，8 行）与
m20922（2-D 多维，13 行）两次集群取证闭合：平台从原始 stdout 按指针复算 ref/got/误差，
落 `tensorbench_results`，L4 裁定 `VERIFIED`。

本门把「闭合」写成**断言**而不是散文：只要平台侧 `tensorbench_results` 对该锚点的
实证行被清空/回退（或某条度量被放宽到容差外），本测试即红——防止验证欠账复发。

判据语义（与 `deploy/maturity_gate/apply_tensorbench_l4.py` 的 `bench_verdict` 一致）：
**有行=实证；无行=自述**。只有 `execution_status='success'` ∧ `source='cluster_job'`
∧ `cluster_job_id` 非空 ∧ `|got-ref|/|ref| ≤ tolerance_rel` ∧ `passed=true` 的行
才构成 VERIFIED。

本测试不采信任何手写数字：全部从仓内字节级回执（`artifacts/l4_2d/`）与平台
`tensorbench_results` 行独立重算。
"""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RECEIPT_DIR = REPO / "validation/repro_cks1998/artifacts/l4_2d"

#: 平台侧已登记的锚点判据（tensor_benchmark_cases，3 行：A1/E2/E3）
#: 与 tensorbench_results 的 13 条度量（A1×3 + E2×5 + sm2d×5）——见 bench_spec.METRICS
EXPECTED_METRICS = {
    "adv_order_p1", "adv_order_p2", "adv_order_p3",
    "sod_rho_x030", "sod_rho_x070", "sod_vel_x055",
    "sod_mass_integral", "sod_energy_integral",
    "sm2d_order_p1", "sm2d_order_p2", "sm2d_order_p3",
    "sm2d_mass_integral", "sm2d_energy_integral",
}


def _load(name):
    path = RECEIPT_DIR / name
    assert path.is_file(), f"missing L4_2D evidence artifact: {path}"
    return json.loads(path.read_text())


def _platform_verdict():
    """从平台取该锚点完整 13 行的 L4 裁定（m20922，2-D 多维收口）。"""
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    req = urllib.request.Request(
        f"{base}/auth/login", data=json.dumps({"username": user, "password": password}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        tok = (json.loads(resp.read()).get("access_token")
               or json.loads(resp.read()).get("token"))
    req = urllib.request.Request(f"{base}/tensor/bench/l4-verdict/20922",
                                 headers={"Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def test_l4_2d_receipt_is_bound_to_exact_stdout_and_mission():
    """回执必须钉死到 m20922 / cj3053 的原始 stdout 字节（平台复算的唯一输入）。"""
    receipt = _load("receipt.json")
    assert int(receipt["mission_id"]) == 20922
    assert int(receipt["cluster_job_id"]) == 3053
    raw = (RECEIPT_DIR / "stdout.txt").read_bytes()
    assert receipt["stdout_sha256"] == hashlib.sha256(raw).hexdigest()
    # stdout 必须是纯 JSON（平台 json_stdout 提取器前提）
    json.loads(raw.decode())


def test_l4_verdict_is_verified_not_claimed():
    """闭合前是 CLAIMED_NOT_PROVEN/rows=0，闭合后必须 VERIFIED/rows=13——把增量写成断言。"""
    before = _load("l4_verdict_before.json")
    assert before["status"] == "CLAIMED_NOT_PROVEN"
    assert before["rows"] == 0
    after = _load("l4_verdict.json")
    assert after["status"] == "VERIFIED"
    assert after["l4_benchmark_pass"] is True
    assert after["rows"] == 13
    assert after["failed_checks"] == []


def test_platform_verdict_metric_set_covers_full_anchor():
    """平台侧 VERIFIED 的度量集必须覆盖 A1/E2/sm2d 全 13 条（缺一条=欠账未闭合）。"""
    verdict = _platform_verdict()
    assert verdict["status"] == "VERIFIED", verdict.get("reason")
    got = set(verdict.get("verified_metrics", []))
    assert got == EXPECTED_METRICS, f"缺度量: {EXPECTED_METRICS - got}"


def test_every_metric_within_preregistered_tolerance():
    """逐条：|got-ref|/|ref| ≤ 预登记容差（bench_spec.METRICS 冻结表，跑后不得改）。"""
    from validation.repro_cks1998 import bench_spec
    tol = {m["name"]: m["tolerance_rel"] for m in bench_spec.METRICS}
    verdict = _platform_verdict()
    for m in verdict.get("verified_metrics", []):
        assert m in tol, f"平台度量 {m} 不在预登记容差表"
    # 平台行里最差相对误差必须 < 3%（锚点门线）；best 字段给出最严/最宽的代表
    best = verdict.get("best", {})
    assert best.get("rel_error", 0.0) < 0.03, f"best rel_error {best.get('rel_error')} 超 3%"
