"""CKS1998 二维（multidimensional）判据的 L4 回归门。

背景：``REPRO-DG-CKS1998`` 的机器判定（``tensorbench_results`` 的 8 行）此前只覆盖
**一维**度量（光滑对流阶数 + Sod），而论文标题本身就是 *multidimensional systems*——
二维 Euler 系统只有仓内 JSON 与 Markdown，没有平台复算过的一行。本门钉死补上的那一段：

1. **预登记不可事后放宽**：``bench_spec.METRICS`` 里的 5 条 ``sm2d_*`` 与下面冻结的
   字面容差逐项相等（改容差必须先改这里 → 留痕到 diff）。
2. **ref 是解析量，不是回填**：``sm2d_mass_integral`` / ``sm2d_energy_integral`` 的 ref
   必须能由精确解的闭式积分**独立算出**（本文件用复合 Simpson 直接积精确解，不碰 DG）。
3. **got 真在解 PDE**：同一条 2-D RKDG 求解器（周期边界、无限制器）在二维上守恒，
   离散积分对闭式 ref 到 float32 舍入水平。
4. **平台回执不可自证**：``artifacts/l4_2d/`` 里的 stdout 字节、spec、平台返回行、L4
   裁定必须自洽——从 stdout 本地重算的每一行都要与平台返回逐项相等。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from validation.repro_cks1998 import bench_spec

REPO = Path(__file__).resolve().parents[1]
RECEIPT_DIR = REPO / "validation/repro_cks1998/artifacts/l4_2d"

#: 预登记容差（任何结果被看到之前写定；改这里 = 改判据，必须在报告里写明）
FROZEN_2D_TOLERANCES = {
    "sm2d_order_p1": 0.03,
    "sm2d_order_p2": 0.03,
    "sm2d_order_p3": 0.03,
    "sm2d_mass_integral": 1e-3,
    "sm2d_energy_integral": 1e-3,
}

#: 论文 §5.1 的二维精确解与其闭式积分（与任何 DG 运行无关）
AMPLITUDE = 0.2
GAMMA = 1.4


def _exact_rho(x, y, t):
    return 1.0 + AMPLITUDE * np.sin(math.pi * (x + y - 2.0 * t))


def _closed_form_integrals(t=0.5, domain=(0.0, 2.0), simpson_intervals=4001):
    """用复合 Simpson 直接积**精确解**，得到质量/总能量（独立于 DG 与任何求解器）。"""
    a, b = domain
    n = simpson_intervals if simpson_intervals % 2 == 1 else simpson_intervals + 1
    xs = np.linspace(a, b, n)
    h = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    rho = _exact_rho(X, Y, t)
    u = np.ones_like(rho)
    v = np.ones_like(rho)
    p = np.ones_like(rho)
    energy = p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    weights = np.outer(w, w) * (h * h / 9.0)
    return {"mass": float((rho * weights).sum()), "energy": float((energy * weights).sum())}


# --------------------------------------------------------------------------- 1
def test_sm2d_metrics_are_preregistered_with_frozen_tolerances():
    by_name = {m["name"]: m for m in bench_spec.METRICS}
    for name, tol in FROZEN_2D_TOLERANCES.items():
        assert name in by_name, f"{name} 未进入平台规格"
        assert by_name[name]["tolerance_rel"] == tol, (name, by_name[name]["tolerance_rel"])
        assert by_name[name]["ref_pointer"] == f"reference.{name}"
        assert by_name[name]["got_pointer"] == f"metrics.{name}.value"
    # 观测阶的 ref 必须是理论阶 k+1，不是上一轮自己的测量值
    for degree in (1, 2, 3):
        assert "k+1" in by_name[f"sm2d_order_p{degree}"]["ref_source"]
    # 守恒积分的 ref 必须写明闭式来源
    assert "exact conservation" in by_name["sm2d_mass_integral"]["ref_source"]
    assert "exact conservation" in by_name["sm2d_energy_integral"]["ref_source"]


def test_spec_document_carries_all_thirteen_metrics():
    spec = bench_spec.build_spec(revision="test", command=["python", "x"],
                                 configuration={"case": "test"})
    names = [m["name"] for m in bench_spec.validate_spec(spec)["metrics"]]
    # 13 条 ref-vs-got 度量 + A2/E3 的 hinge 度量（m22274 起）
    assert len(names) == 13 + len(bench_spec.HINGE_RULES)
    assert set(FROZEN_2D_TOLERANCES) <= set(names)


# --------------------------------------------------------------------------- 2
def test_closed_form_integrals_are_reproduced_by_direct_quadrature_of_the_exact_solution():
    from benchmarks.cks1998_smooth_2d import EXACT_INTEGRALS

    got = _closed_form_integrals()
    assert got["mass"] == pytest.approx(EXACT_INTEGRALS["mass"], rel=0, abs=1e-9)
    assert got["energy"] == pytest.approx(EXACT_INTEGRALS["energy"], rel=0, abs=1e-9)
    # 解析上：mean(rho)=1（正弦在一个周期上积零）⇒ mass=4；E=2.5+rho, u=v=1 ⇒ 14
    assert EXACT_INTEGRALS["mass"] == 4.0
    assert EXACT_INTEGRALS["energy"] == 14.0


# --------------------------------------------------------------------------- 3
def test_two_dimensional_scheme_conserves_the_exact_integrals():
    """同一条求解器：二维 DG 的离散积分 vs 闭式 ref（证明 got 侧真在积分 PDE）。"""
    from benchmarks.cks1998_smooth_2d import (EXACT_INTEGRALS, FINAL_TIME, DOMAIN,
                                             conserved_integrals, initial)
    import tensordg.euler_dg_2d as dg2

    _x, _y, coeff, meta = dg2.solve_euler_dg_2d(
        initial, ncells=8, degree=1, domain=DOMAIN, final_time=FINAL_TIME,
        cfl=0.2, limiter=None, periodic=(True, True))
    totals = conserved_integrals(coeff, 1)
    assert meta["steps"] > 0
    for key in ("mass", "energy"):
        rel = abs(totals[key] - EXACT_INTEGRALS[key]) / EXACT_INTEGRALS[key]
        assert rel < 1e-6, (key, totals[key], rel)
    # 动量同理由 u=v=1 给出闭式 4
    assert totals["momentum_x"] == pytest.approx(4.0, abs=1e-6)
    assert totals["momentum_y"] == pytest.approx(4.0, abs=1e-6)


# --------------------------------------------------------------------------- 4
def _load(name):
    path = RECEIPT_DIR / name
    assert path.is_file(), f"missing 2-D L4 evidence artifact: {path}"
    return json.loads(path.read_text())


def test_2d_receipt_is_bound_to_the_exact_stdout_bytes():
    raw = (RECEIPT_DIR / "stdout.txt").read_bytes()
    receipt = _load("receipt.json")
    assert receipt["stdout_sha256"] == hashlib.sha256(raw).hexdigest()
    assert int(receipt["cluster_job_id"]) > 0
    assert int(receipt["mission_id"]) > 0
    json.loads(raw.decode())          # 平台 json_stdout 提取器的前提：纯 JSON


def test_2d_rows_are_recomputable_from_the_raw_stdout_and_all_pass():
    spec = _load("spec.json")
    receipt = _load("receipt.json")
    doc = json.loads((RECEIPT_DIR / "stdout.txt").read_text())
    recomputed = bench_spec.evaluate_metrics(spec["evaluation"], doc)
    names = [r["metric"] for r in recomputed]
    assert names == [m["name"] for m in spec["evaluation"]["metrics"]]
    assert set(FROZEN_2D_TOLERANCES) <= set(names), "二维度量缺失"
    assert [r["metric"] for r in receipt["rows"]] == names
    # the platform's own verified set must be exactly these metrics (no silent drop)
    assert set(receipt["l4_verdict"]["verified_metrics"]) == set(names)
    assert int(receipt["l4_verdict"]["rows"]) == len(names)
    for mine, theirs in zip(recomputed, receipt["rows"]):
        assert mine["ref_value"] == pytest.approx(theirs["ref_value"], rel=1e-12, abs=0.0)
        assert mine["got_value"] == pytest.approx(theirs["got_value"], rel=1e-12, abs=0.0)
        assert mine["tolerance_rel"] == pytest.approx(theirs["tolerance_rel"], rel=1e-12)
        assert bool(mine["passed"]) == bool(theirs["passed"])
    failed = [r["metric"] for r in recomputed if not r["passed"]]
    assert not failed, f"平台复算后超差的度量: {failed}"


def test_2d_l4_verdict_is_a_machine_pass_including_the_multidimensional_metrics():
    verdict = _load("l4_verdict.json")
    assert verdict["l4_benchmark_pass"] is True, verdict
    assert verdict["status"] in {"PASS", "VERIFIED"}, verdict
    assert verdict["failed_checks"] == []
    verified = set(verdict["verified_metrics"])
    assert set(FROZEN_2D_TOLERANCES) <= verified, set(FROZEN_2D_TOLERANCES) - verified