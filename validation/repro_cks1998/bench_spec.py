"""REPRO-DG-CKS1998 — TensorBench 规格/结果文档（平台契约的仓内唯一真源）。

背景：三次审计（`docs/audit_m15292_REPRO-DG-CKS1998.md`、`docs/AUDIT_m15345_REPRO-DG-CKS1998.md`、
`docs/REPRO-DG-CKS1998-round2.md` §6）都指出同一处欠账——本锚点的判据只活在 Markdown 与
仓内 JSON 里，`tensor_benchmark_cases` / `tensorbench_results` **零行**，于是平台
``GET /tensor/bench/l4-verdict/<mid>`` 只能答 ``CLAIMED_NOT_PROVEN``。本模块把该锚点的
ref-vs-got 判据写成平台契约文档，使数字由**平台侧从集群作业原始 stdout 复算**。

平台契约要点（``tools/tensorbench_eval/contract.py``，本模块逐条对齐）：

* spec 顶层只允许 ``{benchmark, workload, execution, evaluation, implementation, notes}``；
* 每个 metric 必须有 ``got_pointer`` / ``ref_pointer``（**点号**指针，取 workload stdout
  文档里的值）与 ``tolerance_rel``；误差= ``|got-ref|/|ref|`` 由平台自己算；
* result 必须带 ``benchmark-result.schema.json`` v1.0.0 的 8 个必填顶层键，且
  benchmark / workload 身份与 spec 逐字一致。

度量口径（**先定后跑，跑后不改容差**）
--------------------------------------
``ref`` 一律是**与被测量无关的来源**：

* ``adv_order_p*``：ref = 论文的渐近阶 k+1（CKS1998 §2 的理论值），got = 由 5 级网格
  L2 误差最小二乘拟合出的观测阶；
* ``sod_*``：ref = 由 ``benchmarks.sod_exact``（Toro 精确 Riemann 求解器，**独立于 DG
  求解器**）在固定 x 台站上的单元均值 / 守恒积分，got = DG 解在**同一台站**的单元均值。

因此这两类都不是"自己比自己"：观测阶 vs 理论阶；DG 数值解 vs 精确解。
``A2``（限幅方波 L1 阶 ≥1/2）与 ``E3``（超调 ≤1 %）是**不等式**判据，不是 ref-vs-got，
硬编码成 ``|got-ref|/|ref|`` 会伪造度量（例如把 got=0 % 与 ref=1 % 相减得 100 % 误差），
故不进入本规格，继续留在仓内门 ``tests/test_repro_cks1998.py`` + 报告表中。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

# ---------------------------------------------------------------- 身份
BENCH_ID = "REPRO-DG-CKS1998"
BENCH_VER = "1998.1"
WORKLOAD_ID = "cks1998-dg-ladder"
WORKLOAD_VER = "1"
RESULT_SCHEMA_VERSION = "1.0.0"

#: 锚点出处（与 feature_ledger.slug = REPRO-DG-CKS1998 同一篇）
ANCHOR = ("Cockburn, B. & Shu, C.-W., The Runge-Kutta discontinuous Galerkin method "
          "for conservation laws V: multidimensional systems, JCP 141 (1998) 199-224 "
          "/ SIAM Review 43(1) (2001)")

#: 平台侧复算的度量表。容差在**任何结果被看到之前**写定（见 git 首次提交时间）。
METRICS: List[Dict[str, Any]] = [
    # --- A1 光滑对流：观测阶 vs 理论阶 k+1 --------------------------------
    {"name": "adv_order_p1", "got_pointer": "metrics.adv_order_p1.value",
     "ref_pointer": "reference.adv_order_p1", "tolerance_rel": 0.05, "unit": "-",
     "ref_source": "theory: L2 order = k+1 = 2 (CKS1998 §2)"},
    {"name": "adv_order_p2", "got_pointer": "metrics.adv_order_p2.value",
     "ref_pointer": "reference.adv_order_p2", "tolerance_rel": 0.05, "unit": "-",
     "ref_source": "theory: L2 order = k+1 = 3"},
    {"name": "adv_order_p3", "got_pointer": "metrics.adv_order_p3.value",
     "ref_pointer": "reference.adv_order_p3", "tolerance_rel": 0.05, "unit": "-",
     "ref_source": "theory: L2 order = k+1 = 4"},
    # --- E2 Sod：DG 单元均值 vs 精确 Riemann 解（3 % 锚点门） -------------
    {"name": "sod_rho_x030", "got_pointer": "metrics.sod_rho_x030.value",
     "ref_pointer": "reference.sod_rho_x030", "tolerance_rel": 0.03, "unit": "-",
     "ref_source": "exact Riemann (Toro, benchmarks.sod_exact) cell average"},
    {"name": "sod_rho_x070", "got_pointer": "metrics.sod_rho_x070.value",
     "ref_pointer": "reference.sod_rho_x070", "tolerance_rel": 0.03, "unit": "-",
     "ref_source": "exact Riemann cell average (post-shock plateau)"},
    {"name": "sod_vel_x055", "got_pointer": "metrics.sod_vel_x055.value",
     "ref_pointer": "reference.sod_vel_x055", "tolerance_rel": 0.03, "unit": "-",
     "ref_source": "exact Riemann cell average (star region)"},
    # --- E2 守恒量：精确守恒积分（机器精度级容差） ------------------------
    {"name": "sod_mass_integral", "got_pointer": "metrics.sod_mass_integral.value",
     "ref_pointer": "reference.sod_mass_integral", "tolerance_rel": 1e-3, "unit": "-",
     "ref_source": "exact conservation: 0.5*1.0 + 0.5*0.125 = 0.5625"},
    {"name": "sod_energy_integral", "got_pointer": "metrics.sod_energy_integral.value",
     "ref_pointer": "reference.sod_energy_integral", "tolerance_rel": 1e-3, "unit": "-",
     "ref_source": "exact conservation: 0.5*2.5 + 0.5*0.25 = 1.375"},
]

#: 未进入平台规格的判据（不等式），报告里必须显式说明理由（见模块 docstring）
NOT_ENCODED = {
    "A2_discontinuous_l1_order": "不等式判据（L1 阶 >= 1/2），无零附近可比的 ref 值",
    "E3_tvb_no_new_extrema": "不等式判据（超调 <= 1 %），ref=1 % 与 got=0 % 的差商无意义",
}


# ---------------------------------------------------------------- spec
def build_spec(*, revision: str, command: List[str], configuration: Mapping[str, Any],
               cwd: str = ".", timeout_seconds: float = 3600.0,
               notes: str = "") -> Dict[str, Any]:
    """构造平台 ``tensorbench-eval/1`` 规格（键集合与 contract.parse_spec 白名单一致）。"""
    return {
        "benchmark": {"benchmark_id": BENCH_ID, "version": BENCH_VER},
        "workload": {
            "workload_id": WORKLOAD_ID, "version": WORKLOAD_VER,
            "command": [str(c) for c in command], "cwd": cwd,
            "precision": "float64", "configuration": dict(configuration),
        },
        "implementation": {"project": "TensorDG", "revision": revision},
        "execution": {"timeout_seconds": float(timeout_seconds),
                      "env": {"OMP_NUM_THREADS": "1", "TORCH_THREADS": "1"}},
        "evaluation": {
            "extract": {"format": "json_stdout"},
            "metrics": [{"name": m["name"], "got_pointer": m["got_pointer"],
                         "ref_pointer": m["ref_pointer"],
                         "tolerance_rel": float(m["tolerance_rel"]), "unit": m["unit"]}
                        for m in METRICS],
            "notes": "; ".join(m["ref_source"] for m in METRICS),
        },
        "notes": notes or (f"{ANCHOR}. ref = theory order / exact Riemann solution; "
                           "got = measured DG quantities; errors recomputed by the platform."),
    }


# ---------------------------------------------------------------- result
def build_result(doc: Mapping[str, Any], *, mission_id: int | None = None) -> Dict[str, Any]:
    """从 workload stdout 文档构造 ``benchmark-result.schema.json`` v1.0.0 结果文档。"""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": str(doc["run_id"]),
        "benchmark": {"benchmark_id": BENCH_ID, "version": BENCH_VER},
        "implementation": {"project": str(doc["implementation"]["project"]),
                           "revision": str(doc["implementation"]["revision"])},
        "workload": {"workload_id": WORKLOAD_ID, "version": WORKLOAD_VER},
        "execution": {"status": str(doc["execution"]["status"]),
                      **({"mission_id": int(mission_id)} if mission_id else {})},
        "metrics": {k: {"value": float(v["value"]), "unit": v.get("unit", "-")}
                    for k, v in doc["metrics"].items()},
        "validation": dict(doc.get("validation") or {}),
    }


# ---------------------------------------------------------------- 仓内校验器
class DocError(ValueError):
    """文档不满足平台契约（配置/取证问题，不是「判据不过」）。"""


_SPEC_TOP = {"benchmark", "workload", "execution", "evaluation", "implementation", "notes"}
_RESULT_REQUIRED = ("schema_version", "run_id", "benchmark", "implementation", "workload",
                    "execution", "metrics", "validation")


def get_pointer(doc: Any, pointer: str) -> Any:
    """平台 ``tools.maturity_gate.evidence.get_pointer`` 的等价实现（点号 + ``a[0]``）。"""
    if pointer in ("", "."):
        return doc
    cur = doc
    for part in pointer.split("."):
        if part == "":
            raise DocError(f"pointer {pointer!r}: 空段")
        idx = None
        if part.endswith("]") and "[" in part:
            part, _, rest = part.partition("[")
            idx = int(rest[:-1])
        if part:
            if not isinstance(cur, Mapping) or part not in cur:
                raise DocError(f"pointer {pointer!r}: 缺键 {part!r}")
            cur = cur[part]
        if idx is not None:
            if not isinstance(cur, (list, tuple)) or idx >= len(cur):
                raise DocError(f"pointer {pointer!r}: 下标越界 [{idx}]")
            cur = cur[idx]
    return cur


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DocError(f"{where} 必须是数值，得到 {type(value).__name__}")
    return float(value)


def validate_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """对齐 ``contract.parse_spec``：未知键/缺键/空 metrics/坏指针 → ``DocError``。"""
    if not isinstance(spec, Mapping):
        raise DocError("spec 必须是对象")
    unknown = set(spec) - _SPEC_TOP
    if unknown:
        raise DocError(f"未知 spec 键: {sorted(unknown)}")
    bench = spec.get("benchmark") or {}
    if not bench.get("benchmark_id") or not bench.get("version"):
        raise DocError("spec.benchmark.{benchmark_id,version} 必填")
    work = spec.get("workload") or {}
    for key in ("workload_id", "version", "command"):
        if not work.get(key):
            raise DocError(f"spec.workload.{key} 必填")
    if not isinstance(work["command"], (list, tuple)) or not work["command"]:
        raise DocError("spec.workload.command 必须是命令数组")
    ev = spec.get("evaluation") or {}
    fmt = ((ev.get("extract") or {}).get("format")) or "json_stdout"
    if fmt != "json_stdout":
        raise DocError("只支持 extract.format='json_stdout'")
    metrics = ev.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise DocError("spec.evaluation.metrics 必须非空数组")
    seen: set[str] = set()
    for m in metrics:
        if not isinstance(m, Mapping):
            raise DocError("每个 metric 必须是对象")
        name = m.get("name")
        if not name:
            raise DocError("metric.name 必填")
        if name in seen:
            raise DocError(f"metric.name 重复: {name}")
        seen.add(name)
        for key in ("got_pointer", "ref_pointer"):
            if not m.get(key):
                raise DocError(f"metric {name}: {key} 必填")
        tol = m.get("tolerance_rel")
        if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol < 0:
            raise DocError(f"metric {name}: tolerance_rel 必须是非负数")
    return {"benchmark": {"benchmark_id": str(bench["benchmark_id"]),
                          "version": str(bench["version"])},
            "workload": {"workload_id": str(work["workload_id"]),
                         "version": str(work["version"])},
            "benchmark_id": str(bench["benchmark_id"]),
            "benchmark_version": str(bench["version"]),
            "workload_id": str(work["workload_id"]), "metrics": list(metrics),
            "raw": spec}


def validate_result(result: Mapping[str, Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
    """对齐 ``contract.validate_result``：必填键 + 身份一致性 + execution.status。"""
    missing = [k for k in _RESULT_REQUIRED if k not in (result or {})]
    if missing:
        raise DocError(f"result 缺必填键: {missing}")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise DocError(f"result.schema_version 必须是 {RESULT_SCHEMA_VERSION}")
    if (result.get("benchmark") or {}) != dict(spec["benchmark"]):
        raise DocError("result.benchmark 与 spec 不一致")
    if (result.get("workload") or {}).get("workload_id") != spec["workload"]["workload_id"]:
        raise DocError("result.workload.workload_id 与 spec 不一致")
    status = (result.get("execution") or {}).get("status")
    if not status:
        raise DocError("result.execution.status 必填")
    impl = result.get("implementation") or {}
    if not impl.get("project") or not impl.get("revision"):
        raise DocError("result.implementation.{project,revision} 必填")
    if not result.get("run_id"):
        raise DocError("result.run_id 必填")
    return {"execution_status": str(status), "run_id": str(result["run_id"])}


def evaluate_metrics(spec: Mapping[str, Any], doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """对齐 ``contract.evaluate_metrics``：误差由本侧复算（与平台同一算式）。

    用 **spec 的度量表**取值而不是 result 的自述值——与平台一致（平台从不信 result 数字）。
    """
    rows: List[Dict[str, Any]] = []
    for m in spec["metrics"]:
        ref = _number(get_pointer(doc, m["ref_pointer"]), f"metric {m['name']} ref")
        got = _number(get_pointer(doc, m["got_pointer"]), f"metric {m['name']} got")
        abs_err = abs(got - ref)
        rel = (abs_err / abs(ref)) if ref != 0 else None
        rows.append({"metric": m["name"], "unit": m.get("unit", "-"),
                     "ref_value": ref, "got_value": got, "abs_error": abs_err,
                     "rel_error": rel, "tolerance_rel": float(m["tolerance_rel"]),
                     "passed": bool(rel is not None and rel <= float(m["tolerance_rel"]))})
    return rows


def validate_documents(spec: Mapping[str, Any], result: Mapping[str, Any],
                       stdout: str) -> Dict[str, Any]:
    """离线全量自检（不碰数据库）：spec → result → stdout 文档 → 逐度量复算。"""
    parsed = validate_spec(spec)
    meta = validate_result(result, parsed)
    if meta["execution_status"] != "success":
        raise DocError(f"execution_status={meta['execution_status']!r}，平台要求 'success'")
    text = stdout if isinstance(stdout, str) else stdout.decode("utf-8", "replace")
    doc = None
    for candidate in (text.strip(), text[text.find("{"): text.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            doc = json.loads(candidate)
            break
        except ValueError:
            continue
    if doc is None:
        raise DocError("workload stdout 里取不到 JSON 文档（平台同样会拒收）")
    rows = evaluate_metrics(parsed, doc)
    return {"parsed": parsed, "result_meta": meta, "rows": rows,
            "passed": [r["metric"] for r in rows if r["passed"]],
            "failed": [r["metric"] for r in rows if not r["passed"]]}