"""A2/E3 不等式判据的**单侧 hinge 编码**：等价性证明 + 可审计性门（m22274）。

背景：`bench_spec.NOT_ENCODED` 曾把 A2（L1 阶 ≥ 1/2）与 E3（超调 ≤ 1 %）列为
「进不了平台规格」的不等式判据。理由是对的——硬写 ``ref = 界`` 的双侧相对误差会伪造
度量（超调 0 % 对 1 % 的界算成 100 % 误差）。本单不放弃判据，改用保序 hinge。

本文件的门**不采信任何自述数字**：
* 等价性：对大量 (raw, bound) 逐点对拍「hinge 判定的 PASS/FAIL」== 「原始不等式」；
* 必要性（否证）：用**朴素** ``ref=界`` 编码重放同一批数据，证明它会把"比要求更好"判红
  ——即 hinge 不是装饰，是让不等式在双侧契约下**可判**的唯一无损办法；
* 单调性：hinge 不可能把 FAIL 洗成 PASS（只朝通过方向饱和，且饱和值恰在界上）；
* 可审计：从 smoke 档 stdout 的 ``rows`` **重算** A2 观测阶，与 ``validation.hinge.raw``
  对上；``hinge == clamp(raw)`` 与 spec 里登记的界/类型逐项一致。
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from validation.repro_cks1998 import bench_spec                    # noqa: E402

WORKLOAD = REPO / "validation" / "repro_cks1998" / "bench_workload.py"
TOL = bench_spec.HINGE_TOL


@pytest.fixture(scope="module")
def smoke_doc() -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="1", TORCH_THREADS="1")
    out = subprocess.run([sys.executable, str(WORKLOAD), "--profile", "smoke"],
                         cwd=REPO, capture_output=True, text=True, timeout=900, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)


def _passes(got: float, ref: float, tol: float) -> bool:
    """平台判据（`contract.evaluate_metrics` 的同一算式，见 bench_spec.evaluate_metrics）。"""
    rel = abs(got - ref) / abs(ref) if ref != 0 else None
    return bool(rel is not None and rel <= tol)


@pytest.mark.parametrize("kind,bound", [("lower", 0.5), ("upper", 1.0), ("lower", 1e-3)])
def test_hinge_decision_is_equivalent_to_the_raw_inequality(kind, bound):
    """逐点等价：hinge 编码的 PASS/FAIL == 原始不等式判据（含界上、界下、远离界）。"""
    grid = [bound * f for f in (0.0, 1e-9, 0.5, 0.999999, 1.0, 1.000001, 1.5, 3.0, 8.0, 40.0)]
    grid += [-bound, 0.0, bound * 2, bound * 1e6]
    for raw in grid:
        got = bench_spec.hinge(raw, bound, kind)
        ref = 1.0
        assert _passes(got, ref, TOL) == bench_spec.hinge_holds(raw, bound, kind, TOL), (kind, raw)
        # 原始判据本体（不含浮点护栏）也必须一致——界上的等号归 PASS
        assert _passes(got, ref, TOL) == (
            raw >= bound if kind == "lower" else raw <= bound), (kind, raw)


def test_better_than_required_saturates_instead_of_being_rejected():
    """关键性质：越过界的**很好**结果不能被双侧规则反判红（朴素编码的致命缺陷）。"""
    # A2：L1 阶实测 4.0（远优于 1/2 的地板）——朴素 ref=0.5 编码会算成 700 % 误差
    naive_rel = abs(4.0 - 0.5) / 0.5
    assert naive_rel > TOL and not _passes(4.0, 0.5, TOL)          # 朴素编码：FAIL（错）
    assert _passes(bench_spec.hinge(4.0, 0.5, "lower"), 1.0, TOL)  # hinge：PASS（对）
    # E3：超调 0 %（严格 TVD 策略）——朴素 ref=1 % 编码会算成 100 % 误差
    assert not _passes(0.0, 1.0, TOL)
    assert _passes(bench_spec.hinge(0.0, 1.0, "upper"), 1.0, TOL)
    # 而真正越界的结果在两种编码下都必须 FAIL
    assert not _passes(bench_spec.hinge(1.389, 1.0, "upper"), 1.0, TOL)   # 1.389 % > 1 %
    assert not _passes(bench_spec.hinge(0.45, 0.5, "lower"), 1.0, TOL)    # 0.45 < 1/2


def test_hinge_cannot_launder_a_violation_into_a_pass():
    """单调性/防洗白：只有当原始不等式成立时才可能 PASS。"""
    for raw in (0.0, 0.1, 0.49, 0.4999999, 0.5, 0.51, 1.0, 9.0):
        hinge = bench_spec.hinge(raw, 0.5, "lower")
        assert (hinge < 1.0 - 1e-12) or (raw >= 0.5), raw
        if _passes(hinge, 1.0, TOL):
            assert raw >= 0.5 * (1 - TOL), raw
    for raw in (0.0, 1.0, 1.0000001, 1.5, 100.0):
        hinge = bench_spec.hinge(raw, 1.0, "upper")
        assert (hinge > 1.0 + 1e-12) or (raw <= 1.0), raw


def test_bad_bound_and_kind_are_rejected():
    with pytest.raises(bench_spec.DocError):
        bench_spec.hinge(1.0, 0.0, "lower")
    with pytest.raises(bench_spec.DocError):
        bench_spec.hinge(1.0, 1.0, "sideways")


def test_spec_registers_every_hinge_metric_with_bound_and_kind(smoke_doc):
    spec = bench_spec.validate_spec(bench_spec.build_spec(
        revision="test", command=["python", "x"], configuration={"case": "test"}))
    by_name = {m["name"]: m for m in spec["metrics"]}
    names = [r["metric"] for r in bench_spec.HINGE_RULES]
    assert names and set(names) <= set(by_name)
    assert not bench_spec.NOT_ENCODED, "hinge 之后不应再有「进不了规格」的判据"
    for rule in bench_spec.HINGE_RULES:
        assert by_name[rule["metric"]]["tolerance_rel"] == TOL
        assert by_name[rule["metric"]]["got_pointer"] == f"metrics.{rule['metric']}.value"
        # 文档里必须有 hinge 绑定（raw/bound/kind/hinge）
        bound = smoke_doc["validation"]["hinge"][rule["metric"]]
        assert bound["bound"] == rule["bound"] and bound["kind"] == rule["kind"]


def test_hinge_matches_raw_and_rows_recompute_the_order(smoke_doc):
    """可审计性：hinge == clamp(raw)；A2 的观测阶可由 rows 的 L1 误差**重算**。"""
    hinge = smoke_doc["validation"]["hinge"]
    for rule in bench_spec.HINGE_RULES:
        blk = hinge[rule["metric"]]
        assert blk["hinge"] == bench_spec.hinge(blk["raw"], rule["bound"], rule["kind"])
        assert smoke_doc["metrics"][rule["metric"]]["value"] == blk["hinge"]

    rows = [r for r in smoke_doc["rows"] if r.get("section") == "hinge"]
    for rule in bench_spec.HINGE_RULES:
        row = next(r for r in rows if r["metric"] == rule["metric"])
        assert row["raw"] == hinge[rule["metric"]]["raw"]

    # A2：从逐网格 L1 误差重算最小二乘观测阶，必须与 raw 一致
    for rule in bench_spec.HINGE_RULES:
        if not rule["metric"].startswith("a2_"):
            continue
        row = next(r for r in rows if r["metric"] == rule["metric"])
        cells, l1 = row["cells"], row["l1_errors"]
        n = len(cells)
        lx = [math.log(c) for c in cells]
        ly = [math.log(e) for e in l1]
        mx, my = sum(lx) / n, sum(ly) / n
        slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
        assert abs(abs(slope) - row["raw"]) < 5e-3, (rule["metric"], slope, row["raw"])

    # E3：非门例（unlimited / tvb_M0）必须在 rows 里可见，不得被吞
    non_gate = [r for r in smoke_doc["rows"] if r.get("section") == "E3_non_gate"]
    assert non_gate, "E3 对照/公开指示器的越界必须留在 rows"
    assert all("exceeds_1pct" in r for r in non_gate)
