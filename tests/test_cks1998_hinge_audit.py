"""hinge 编码的第三方交叉核验与判别力审计的回归门（m22343）。

被守护的不变量
--------------
1. **rows ⟶ raw ⟶ hinge ⟶ metrics 四方一致**：任何一个位置被改动（伪造 got、改 raw、
   改点值极值）都必须抛 ``DocError``，使 ``validate_documents`` / ingest 拒收。
2. **决定等价**：hinge 过门 ⟺ 原始不等式成立（不在"比要求更好"上误判）。
3. **判别力**：不限幅对照必须在同一判据下判 FAIL（否则该门无信息量）。
4. **审计披露**：A2 的局部阶/子梯次、E3 的分辨率扫描必须出现在报告里。
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from validation.repro_cks1998 import bench_spec            # noqa: E402
from validation.repro_cks1998 import hinge_audit           # noqa: E402

WORKLOAD = REPO / "validation" / "repro_cks1998" / "bench_workload.py"

# 2 阶律的合成梯次：order = 2，界 1/2 ⇒ 必然 PASS
_OK_CELLS = [20, 40, 80, 160]
_OK_ERRS = [0.64, 0.16, 0.04, 0.01]


def _doc(a2_order_errors=None, e3_pt=(0.0, 1.0), e3_raw=None):
    """合成一份最小可核验文档（结构同 bench_workload 的 stdout 文档，覆盖全部 hinge 度量）。"""
    errs = list(a2_order_errors or _OK_ERRS)
    order = hinge_audit.lsq_order(_OK_CELLS[: len(errs)], errs)
    pt_min, pt_max = e3_pt
    e3 = hinge_audit.pointwise_overshoot_pct(pt_min, pt_max)
    rows, hinge_block, metrics, reference = [], {}, {}, {}
    for rule in bench_spec.HINGE_RULES:
        name = rule["metric"]
        if name.startswith("a2_l1_order"):
            raw = order
            extra = {"case": "A2_square_wave", "degree": int(name[-1]),
                     "cells": _OK_CELLS[: len(errs)], "l1_errors": errs}
        else:
            raw = e3 if e3_raw is None else e3_raw
            extra = {"case": "E3_tvd", "degree": int(name[-1]),
                     "limiter": "strict_all", "ncells": 100,
                     "overshoot_pointwise_pct": e3, "pt_min": pt_min, "pt_max": pt_max}
        got = bench_spec.hinge(raw, rule["bound"], rule["kind"])
        rows.append({"section": "hinge", "metric": name, "raw": raw,
                     "bound": rule["bound"], "kind": rule["kind"], "hinge": got, **extra})
        hinge_block[name] = {"raw": raw, "bound": rule["bound"], "kind": rule["kind"],
                             "hinge": got, "criterion": "synthetic"}
        metrics[name] = {"value": got, "unit": "-"}
        reference[name] = 1.0
    return {
        "schema_version": "1.0.0", "run_id": "synthetic",
        "benchmark": {"benchmark_id": bench_spec.BENCH_ID, "version": bench_spec.BENCH_VER},
        "metrics": metrics, "reference": reference,
        "validation": {"hinge": hinge_block}, "rows": rows,
    }


def test_consistency_accepts_a_honest_document():
    report = hinge_audit.verify_hinge_consistency(_doc())
    assert set(report) == {r["metric"] for r in bench_spec.HINGE_RULES}
    for row in report.values():
        assert row["consistent"] and row["raw_rel_diff"] < 1e-12
        assert row["decision"] == "PASS"          # order 2 >= 0.5、超调 0 % <= 1 %
    assert report["a2_l1_order_p1"]["raw_rows"] == pytest.approx(2.0, rel=1e-12)


def test_consistency_rejects_a_forged_saturated_got():
    """把越界梯次伪造成 hinge=1.0（自述 raw=0.65）：必须被 rows 重算抓出来。"""
    bad = _doc(a2_order_errors=[0.1, 0.1 * 2 ** -0.2, 0.1 * 4 ** -0.2])
    assert hinge_audit.verify_hinge_consistency(bad)[
        "a2_l1_order_p1"]["decision"] == "FAIL"      # 诚实文档：阶 0.2 < 1/2
    for name in ("a2_l1_order_p1", "a2_l1_order_p2", "a2_l1_order_p3"):
        bad["validation"]["hinge"][name].update({"raw": 0.65, "hinge": 1.0})
        bad["metrics"][name]["value"] = 1.0
    with pytest.raises(bench_spec.DocError, match="rows 重算的 raw"):
        hinge_audit.verify_hinge_consistency(bad)


def test_consistency_rejects_raw_hinge_disagreement():
    doc = _doc()
    doc["validation"]["hinge"]["a2_l1_order_p1"]["raw"] = 0.40   # 自述 raw 变成越界
    with pytest.raises(bench_spec.DocError, match="不一致"):
        hinge_audit.verify_hinge_consistency(doc)


def test_consistency_rejects_tampered_pointwise_extrema():
    doc = _doc()
    next(r for r in doc["rows"] if r["metric"] == "e3_overshoot_p1")["pt_max"] = 1.05
    with pytest.raises(bench_spec.DocError, match="e3_overshoot_p1"):
        hinge_audit.verify_hinge_consistency(doc)


def test_consistency_rejects_registered_bound_drift():
    doc = _doc()
    doc["validation"]["hinge"]["e3_overshoot_p1"]["bound"] = 2.0
    with pytest.raises(bench_spec.DocError, match="与预登记"):
        hinge_audit.verify_hinge_consistency(doc)


def test_consistency_rejects_missing_hinge_block():
    doc = _doc()
    del doc["validation"]["hinge"]
    with pytest.raises(bench_spec.DocError, match="validation.hinge"):
        hinge_audit.verify_hinge_consistency(doc)


def test_violating_configuration_is_decided_fail_by_the_same_rule():
    """E3 上界：超调 5 % ⇒ hinge=5.0 ⇒ 相对误差 4.0 ≫ 1e-9 ⇒ FAIL（判据有判别力）。"""
    doc = _doc(e3_pt=(-0.05, 1.0))
    report = hinge_audit.verify_hinge_consistency(doc)
    assert report["e3_overshoot_p1"]["decision"] == "FAIL"
    assert report["e3_overshoot_p1"]["hinge_recomputed"] == pytest.approx(5.0)


@pytest.fixture(scope="module")
def real_doc():
    env = dict(os.environ, OMP_NUM_THREADS="1", TORCH_THREADS="1")
    out = subprocess.run([sys.executable, str(WORKLOAD), "--profile", "smoke"],
                         cwd=REPO, capture_output=True, text=True, timeout=900, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_real_workload_document_is_hinge_consistent(real_doc):
    doc = json.loads(real_doc)
    report = hinge_audit.verify_hinge_consistency(doc)
    assert len(report) == len(bench_spec.HINGE_RULES) == 7  # +论文配置并列门 2 条
    for name, row in report.items():
        assert row["consistent"], name
        assert row["decision"] == "PASS", (name, row)


def test_validate_documents_gates_on_hinge_consistency(real_doc):
    doc = json.loads(real_doc)
    spec = bench_spec.build_spec(revision="test", command=["python", "x.py"],
                                 configuration={"case": "smoke"})
    result = bench_spec.build_result(copy.deepcopy(doc), mission_id=22343)
    checked = bench_spec.validate_documents(spec, result, real_doc)
    assert set(checked["hinge"]) == {r["metric"] for r in bench_spec.HINGE_RULES}
    # 篡改点值极值 ⇒ validate_documents 必须抛（ingest 因此拒发）
    tampered = copy.deepcopy(doc)
    row = next(r for r in tampered["rows"] if r.get("metric") == "e3_overshoot_p1")
    row["pt_max"] = 1.5
    bad_result = bench_spec.build_result(copy.deepcopy(tampered), mission_id=22343)
    with pytest.raises(bench_spec.DocError, match="e3_overshoot_p1"):
        bench_spec.validate_documents(spec, bad_result, json.dumps(tampered))


def test_a2_ladder_robustness_discloses_pairs_and_subsets(real_doc):
    doc = json.loads(real_doc)
    out = hinge_audit.a2_ladder_robustness(doc)
    assert out and all(row["pairs"] for row in out.values())
    for row in out.values():
        assert row["min_pair"] >= 0.5, row        # 最小局部阶也在半阶地板之上


def test_discriminating_power_separates_gate_from_control():
    out = hinge_audit.discriminating_power(cells=(40, 80), degrees=(1,))
    by = out["by_resolution"]
    assert by["40"]["unlimited_passes"] is False      # 对照必须判红
    assert by["40"]["strict_all_passes"] is True
    assert by["40"]["unlimited_max_pct"] > by["40"]["strict_all_max_pct"]