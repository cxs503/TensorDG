"""REPRO-DG-CKS1998 → TensorBench 取证回路的回归门。

钉死三件事（都是过去审计里真出过的失效）：

1. **规格与非零 ref 一致**：spec 的白名单键、metric 指针、容差表与
   ``bench_spec.METRICS`` 逐项相等（容差表冻结在本文件里，改容差必须先改这里 → 留痕）；
2. **workload 文档能被契约指针取到**：跑 ``--profile smoke``（秒级），验证 8 个 metric
   的 ref/got 指针都能解析、守恒积分落在机器精度、观测阶有限；
3. **平台会拒收的四种坏文档，本地校验器也会拒**：schema_version 错、benchmark 身份
   不一致、指针缺失、execution.status != success —— 防止"文档坏了但本地自检绿"。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from validation.repro_cks1998 import bench_spec                    # noqa: E402

WORKLOAD = REPO / "validation" / "repro_cks1998" / "bench_workload.py"
SPEC_FILE = REPO / "tensorbench" / "REPRO-DG-CKS1998.spec.json"

#: 预登记容差（跑后不得改；改这里=改判据，必须写进报告）
FROZEN_TOLERANCES = {
    "adv_order_p1": 0.05, "adv_order_p2": 0.05, "adv_order_p3": 0.05,
    "sod_rho_x030": 0.03, "sod_rho_x070": 0.03, "sod_vel_x055": 0.03,
    "sod_mass_integral": 1e-3, "sod_energy_integral": 1e-3,
}


@pytest.fixture(scope="module")
def smoke_doc() -> str:
    env = dict(os.environ, OMP_NUM_THREADS="1", TORCH_THREADS="1")
    out = subprocess.run([sys.executable, str(WORKLOAD), "--profile", "smoke"],
                         cwd=REPO, capture_output=True, text=True, timeout=900, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def _spec(**kw):
    return bench_spec.build_spec(revision="deadbeef",
                                 command=["python", "validation/repro_cks1998/bench_workload.py",
                                          "--profile", "bench"],
                                 configuration={"case": "test"}, **kw)


def test_spec_uses_only_contract_keys_and_full_metric_set():
    spec = _spec()
    parsed = bench_spec.validate_spec(spec)              # 未知键/坏键 → DocError
    assert set(spec) <= {"benchmark", "workload", "execution", "evaluation",
                         "implementation", "notes"}
    names = [m["name"] for m in parsed["metrics"]]
    assert names == list(FROZEN_TOLERANCES), "度量集与预登记表不一致"
    for m in parsed["metrics"]:
        assert m["tolerance_rel"] == FROZEN_TOLERANCES[m["name"]], m["name"]


def test_tolerances_are_preregistered_and_not_averaged_away():
    """容差必须是单侧上界比较（|got-ref|/|ref| <= tol），且没有 0 容差的伪度量。"""
    for m in bench_spec.METRICS:
        assert 0 < m["tolerance_rel"] <= 0.05, m["name"]
        assert m["ref_pointer"].startswith("reference.")
        assert m["got_pointer"].startswith("metrics.")


def test_committed_spec_matches_builder():
    assert SPEC_FILE.exists(), f"缺 {SPEC_FILE}（平台 evaluator_command 约定路径）"
    on_disk = json.loads(SPEC_FILE.read_text())
    parsed = bench_spec.validate_spec(on_disk)
    assert parsed["benchmark_id"] == bench_spec.BENCH_ID
    assert [m["name"] for m in parsed["metrics"]] == list(FROZEN_TOLERANCES)


def test_smoke_document_resolves_every_pointer_and_conserves_mass_and_energy(smoke_doc):
    doc = json.loads(smoke_doc)
    spec = _spec()
    rows = bench_spec.evaluate_metrics(bench_spec.validate_spec(spec), doc)
    assert len(rows) == len(FROZEN_TOLERANCES)
    by_name = {r["metric"]: r for r in rows}
    # 守恒量：DG 离散守恒 ⇒ 与解析积分到 float32 舍入水平（求解器默认 float32，
    # 实测 1.3e-8；阈值 1e-6 留一个数量级余量，仍比 1e-3 的度量容差严 3 个量级）
    for key in ("sod_mass_integral", "sod_energy_integral"):
        assert by_name[key]["rel_error"] < 1e-6, by_name[key]
        assert by_name[key]["passed"]
    # 观测阶必须有限且在合理区间（粗网格 smoke 档不了精度，只证明算出来了）
    for key in ("adv_order_p1", "adv_order_p2", "adv_order_p3"):
        assert 0.5 < by_name[key]["got_value"] < 6.0, key


def test_validate_documents_accepts_wellformed_bundle(smoke_doc):
    spec = _spec()
    result = bench_spec.build_result(json.loads(smoke_doc), mission_id=17011)
    checked = bench_spec.validate_documents(spec, result, smoke_doc)
    assert checked["result_meta"]["execution_status"] == "success"
    assert isinstance(checked["passed"], list) and isinstance(checked["failed"], list)


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r.update({"schema_version": "0.9.9"}), "schema_version"),
    (lambda r: r["benchmark"].update({"version": "1999.1"}), "benchmark"),
    (lambda r: r["workload"].update({"workload_id": "other-workload"}), "workload_id"),
    (lambda r: r["execution"].update({"status": "failed"}), "execution_status"),
])
def test_bad_result_documents_are_rejected(smoke_doc, mutate, needle):
    spec = _spec()
    result = bench_spec.build_result(json.loads(smoke_doc))
    mutate(result)
    with pytest.raises(bench_spec.DocError) as exc:
        bench_spec.validate_documents(spec, result, smoke_doc)
    assert needle in str(exc.value)


def test_missing_pointer_is_rejected_not_filled_with_zero(smoke_doc):
    spec = _spec()
    spec["evaluation"]["metrics"].append({
        "name": "ghost_metric", "got_pointer": "metrics.ghost_metric.value",
        "ref_pointer": "reference.ghost_metric", "tolerance_rel": 0.01, "unit": "-"})
    result = bench_spec.build_result(json.loads(smoke_doc))
    with pytest.raises(bench_spec.DocError) as exc:
        bench_spec.validate_documents(spec, result, smoke_doc)
    assert "pointer" in str(exc.value)


def test_stdout_without_json_document_is_rejected():
    spec = _spec()
    result = bench_spec.build_result({"schema_version": "1.0.0", "run_id": "x",
                                      "implementation": {"project": "TensorDG",
                                                         "revision": "deadbeef"},
                                      "execution": {"status": "success"},
                                      "metrics": {}, "validation": {}})
    with pytest.raises(bench_spec.DocError) as exc:
        bench_spec.validate_documents(spec, result, "Traceback (most recent call last): boom\n")
    assert "JSON" in str(exc.value)