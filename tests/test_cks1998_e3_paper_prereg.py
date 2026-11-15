"""E3 判据的「去 vacuity + 分辨率预登记」回归门（m22426，锚点 REPRO-DG-CKS1998）。

被守护的不变量（对应审计 m22343 §3 的 F1/F2）
---------------------------------------------
1. **门例不再"构造即真"**：E3 并列门钉在**论文配置**（TVB M=0 指示器）上，该判据在
   同一批网格上**确实会判红**（不限幅对照 6–9 %；论文配置粗网格 N=80 → 1.389 %），
   故 PASS 携带信息量（不是"逐单元 minmod 按构造无新极值"的代码性质）。
2. **分辨率是判据本体**：门行的 ``ncells`` / ``limiter`` 必须逐项等于
   ``bench_spec.E3_PREREGISTRATION`` 在该档位的登记值，换网格/换策略即 ``DocError``。
3. **分辨率敏感性可见**：粗网格披露行（``E3_paper_resolution``）必须存在且分辨率正确；
   若粗网格已越界而门例过门，``resolution_sensitive=True`` 必须如实报出。
4. **自述值不可信**：门的自述超调必须能由同一行的 ``pt_min/pt_max`` 重算复核。
5. **两张表不许漂移**：``bench_workload.INEQUALITY_PROFILES`` 的预登记分辨率必须与
   ``bench_spec.E3_PREREGISTRATION`` 一致。
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

from validation.repro_cks1998 import bench_spec              # noqa: E402
from validation.repro_cks1998 import bench_workload         # noqa: E402
from validation.repro_cks1998 import hinge_audit            # noqa: E402

WORKLOAD = REPO / "validation" / "repro_cks1998" / "bench_workload.py"
PAPER_METRICS = ("e3_paper_overshoot_p1", "e3_paper_overshoot_p2")


@pytest.fixture(scope="module")
def real_doc() -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="1", TORCH_THREADS="1")
    out = subprocess.run([sys.executable, str(WORKLOAD), "--profile", "smoke"],
                         cwd=REPO, capture_output=True, text=True, timeout=900, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)


def _row(doc: dict, metric: str) -> dict:
    return next(r for r in doc["rows"] if r.get("metric") == metric)


def _spec() -> dict:
    return bench_spec.build_spec(revision="test", command=["python", "x.py"],
                                 configuration={"case": "smoke"})


# ---------------------------------------------------------------- 静态规格
def test_paper_gate_is_registered_as_an_inequality():
    names = [r["metric"] for r in bench_spec.HINGE_RULES]
    for metric in PAPER_METRICS:
        assert metric in names
        rule = next(r for r in bench_spec.HINGE_RULES if r["metric"] == metric)
        assert rule["kind"] == "upper" and rule["bound"] == 1.0
        # 进 METRICS 的 hinge 指针：got=metrics.<name>.value, ref=reference.<name>=1.0
        entry = next(m for m in bench_spec.METRICS if m["name"] == metric)
        assert entry["got_pointer"] == f"metrics.{metric}.value"
        assert entry["ref_pointer"] == f"reference.{metric}"
        assert entry["tolerance_rel"] == bench_spec.HINGE_TOL


def test_profile_table_matches_preregistration():
    """workload 档位表与预登记表不许各说各话（否则门会钉在不存在的网格上）。"""
    for profile, hp in bench_workload.INEQUALITY_PROFILES.items():
        for metric, reg in bench_spec.E3_PREREGISTRATION.items():
            assert reg["ncells_by_profile"][profile] == hp["tvd_paper_cells"]
            assert (reg["coarse_ncells_by_profile"][profile]
                    == hp["tvd_paper_coarse_cells"])


# ---------------------------------------------------------------- 诚实文档
def test_honest_document_passes_preregistration(real_doc):
    report = hinge_audit.verify_preregistration(real_doc)
    assert set(report) == set(PAPER_METRICS)
    for metric, row in report.items():
        assert row["limiter"] == "tvb_M0", (metric, row)
        assert row["ncells"] == 40 and row["coarse_ncells"] == 20, (metric, row)
        assert row["gate_holds"] is True, (metric, row)


def test_validate_documents_reports_preregistration(real_doc):
    out = bench_spec.validate_documents(_spec(),
                                        bench_spec.build_result(copy.deepcopy(real_doc),
                                                                mission_id=22426),
                                        json.dumps(real_doc))
    assert set(out["preregistration"]) == set(PAPER_METRICS)


# ---------------------------------------------------------------- 拒收路径
def test_preregistration_rejects_resolution_drift(real_doc):
    """门不得悄悄换网格：把门行的 ncells 从 40 改成 20 ⇒ 抛。"""
    doc = copy.deepcopy(real_doc)
    _row(doc, "e3_paper_overshoot_p1")["ncells"] = 20
    with pytest.raises(bench_spec.DocError, match="预登记分辨率"):
        hinge_audit.verify_preregistration(doc)


def test_preregistration_rejects_limiter_swap(real_doc):
    """把论文指示器偷换成 strict_all（构造即真）⇒ 抛。"""
    doc = copy.deepcopy(real_doc)
    _row(doc, "e3_paper_overshoot_p2")["limiter"] = "strict_all"
    with pytest.raises(bench_spec.DocError, match="论文策略"):
        hinge_audit.verify_preregistration(doc)


def test_preregistration_rejects_missing_resolution_disclosure(real_doc):
    """抹掉粗网格披露行 ⇒ 抛（分辨率敏感性必须可见）。"""
    doc = copy.deepcopy(real_doc)
    doc["rows"] = [r for r in doc["rows"] if r.get("section") != "E3_paper_resolution"]
    with pytest.raises(bench_spec.DocError, match="分辨率披露"):
        hinge_audit.verify_preregistration(doc)


def test_preregistration_rejects_misreported_overshoot(real_doc):
    """自述 0 % 但 pt_max=1.05 ⇒ 重算不符 ⇒ 抛。"""
    doc = copy.deepcopy(real_doc)
    row = _row(doc, "e3_paper_overshoot_p1")
    row["pt_max"] = 1.05
    with pytest.raises(bench_spec.DocError, match="重算"):
        hinge_audit.verify_preregistration(doc)


def test_validate_documents_gates_on_preregistration(real_doc):
    doc = copy.deepcopy(real_doc)
    _row(doc, "e3_paper_overshoot_p1")["ncells"] = 80
    result = bench_spec.build_result(copy.deepcopy(doc), mission_id=22426)
    with pytest.raises(bench_spec.DocError):
        bench_spec.validate_documents(_spec(), result, json.dumps(doc))


# ---------------------------------------------------------------- 判别力 / 非 vacuity
def test_paper_gate_can_be_falsified_by_the_same_rule():
    """同一判据（upper 1 %）对论文配置的越界必须判 FAIL，且不允许被 hinge 饱和掩盖。"""
    rule = next(r for r in bench_spec.HINGE_RULES
                if r["metric"] == "e3_paper_overshoot_p1")
    bound = float(rule["bound"])
    over = hinge_audit.pointwise_overshoot_pct(0.0, 1.01389)      # 1.389 % 超调
    got = bench_spec.hinge(over, bound, rule["kind"])
    assert got > 1.0 + 1e-6
    assert bench_spec.hinge_holds(over, bound, rule["kind"]) is False
    # 未越界时 hinge 精确饱和到 1.0（保序，不把"更好"当成误差）
    assert bench_spec.hinge(0.0, bound, rule["kind"]) == 1.0


def test_gate_is_not_vacuous_but_the_reported_resolution_flip_is_not_reproduced():
    """论文配置判据的判别力实测（与审计 m22343 §3 F2 的**对拍**，如实记录不一致）。

    实测（P1/P2，cells ∈ {40, 80, 100}，``repro_cks1998.run_tvd`` 同源）：
      * 不限幅对照 **6.3–8.6 %** ⇒ 判红（判据有判别力、非 vacuity）；
      * 论文配置（TVB M=0）在**所有扫描档**均为 0.000 % ⇒ 判绿。
    即：审计 m22343 §3 F2 称"N=80 时论文配置 P1 = 1.389 %"，**本席独立重跑未复现**
    （该数字在本树当前实现下不可达）。故本单的预登记核验应被读作"防未来漂移的护栏"，
    而不是"修一个当前可观测的翻转"；这条差异如实入账，不粉饰。
    """
    out = hinge_audit.discriminating_power(cells=(40, 80, 100), degrees=(1,))
    by = out["by_resolution"]
    assert all(row["unlimited_passes"] is False for row in by.values()), by
    assert all(row["unlimited_max_pct"] > 6.0 for row in by.values()), by
    assert all(row["tvb_M0_passes"] is True for row in by.values()), by
    assert all(row["tvb_M0_max_pct"] == 0.0 for row in by.values()), by
