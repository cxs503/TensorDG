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

* ``sm2d_*``：ref = 论文 §5.1 自己的**二维 Euler 光滑熵波**算例的理论阶 k+1（3 条）
  与其精确解的**闭式守恒积分**（质量 4、总能量 14，见 ``benchmarks.cks1998_smooth_2d.
  EXACT_INTEGRALS``）；got = 同一条 RKDG 求解器在 ``[0,2]^2`` 周期盒上的实测密度 L1
  观测阶与守恒积分。这一组把"多维系统"（论文标题里的 multidimensions）也纳入机器判定。

因此这些度量都不是"自己比自己"：观测阶 vs 理论阶；DG 数值解 vs 精确解。

不等式判据的**单侧 hinge 编码**（本模块 m22274 起）
---------------------------------------------------
``A2``（限幅方波 L1 收敛阶 ≥ 1/2）与 ``E3``（超调 ≤ 1 %）是论文的**不等式**判据，不是
ref-vs-got。直接写成 ``ref = 界`` 的双侧相对误差会伪造度量：超调 0 % 对 1 % 的界算成
100 % 误差（比超调 1.4 % 的 1.389 号结果"更差"），把越过界的很好结果也判红。

故改用**保序 hinge**：把不等式压缩成一个**有界、单调、只朝通过方向饱和**的比值，再交给
平台的双侧 ``|got-ref|/|ref| <= tol`` 规则：

* 下界（``x >= b``，A2）：``got = min(x, b) / b``，``ref = 1``；
  ``x >= b`` ⇒ ``got == 1`` 逐位（``min`` 与 ``/b`` 都是 IEEE 精确）⇒ 误差 0 ⇒ PASS；
  ``x < b`` ⇒ ``got < 1`` ⇒ 误差 ``(b-x)/b`` ⇒ 只有 ``x` 越过界才会红。
* 上界（``x <= b``，E3）：``got = max(x, b) / b``，``ref = 1``；对称成立。

**等价性**（``tests/test_cks1998_inequality_hinge.py`` 逐点验证）：对任意 ``x``，
hinge 编码的 PASS/FAIL 与原始不等式判据**完全相同**；"比要求更好"不再被双侧规则反判。

**可审计性**（防"自带裁判"）：hinge 由 workload 在打印前施加，故同一份 stdout 文档里
必须同时留下

1. ``validation.hinge.<metric> = {raw, bound, kind, hinge}``（原始测量值与界的绑定），
2. ``rows`` 里的原始网格梯次（A2 的 L1 误差、E3 的三种限幅器峰值），

第三方可从 ``rows`` **重算**原始量与观测阶，并核对 ``hinge == clamp(raw)``——回归门
``test_hinge_matches_raw_and_rows_recompute_the_order`` 就做这件事。hinge 只朝通过方向
饱和，不可能把 FAIL 洗成 PASS（变换单调且饱和值恰在界上）。

* ``tolerance_rel`` 取 ``1e-9``（**不是**放宽判据）：hinge 比值在界上逐位等于 1，理论误差
  为 0；该 1e-9 只吸收 JSON 往返可能带来的 1 ulp 抖动，比观测阶的有效数字低 6 个量级。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

# ---------------------------------------------------------------- 身份
BENCH_ID = "REPRO-DG-CKS1998"
#: 版本随**度量集**走（平台按 (benchmark_id, version) 归档行）：
#: 1998.1 = 13 条（A1/E2/B1/B2）；m22274 加 hinge 5 条未改版本（历史欠账）；
#: 1998.2 = +E3 论文配置并列门 2 条（本单 m22426，度量集 20 条，含预登记表）。
BENCH_VER = "1998.2"
WORKLOAD_ID = "cks1998-dg-ladder"
#: workload 程序版本（新增"论文配置并列门 + 粗网格分辨率披露"两段测量 ⇒ 2）。
WORKLOAD_VER = "2"
RESULT_SCHEMA_VERSION = "1.0.0"

#: 锚点出处（与 feature_ledger.slug = REPRO-DG-CKS1998 同一篇）
ANCHOR = ("Cockburn, B. & Shu, C.-W., The Runge-Kutta discontinuous Galerkin method "
          "for conservation laws V: multidimensional systems, JCP 141 (1998) 199-224 "
          "/ SIAM Review 43(1) (2001)")

#: hinge 编码的浮点护栏（**不是**判据放宽）：见模块 docstring 末段。定义在 METRICS 之前
#: 因为度量表在 import 时求值。
HINGE_TOL = 1e-9

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
    # --- B1 二维 Euler 光滑熵波（论文 §5.1 自己的算例）：观测阶 vs 理论阶 k+1 -
    # 精确解 rho = 1 + 0.2 sin(pi (x + y - 2t)), u = v = 1, p = 1（周期盒 [0,2]^2），
    # got = 密度 L1 误差在 3 级网格上最小二乘拟合出的观测阶（P^k => k+1）。
    {"name": "sm2d_order_p1", "got_pointer": "metrics.sm2d_order_p1.value",
     "ref_pointer": "reference.sm2d_order_p1", "tolerance_rel": 0.03, "unit": "-",
     "ref_source": "theory: 2-D Euler density L1 order = k+1 = 2 (CKS1998 §5.1)"},
    {"name": "sm2d_order_p2", "got_pointer": "metrics.sm2d_order_p2.value",
     "ref_pointer": "reference.sm2d_order_p2", "tolerance_rel": 0.03, "unit": "-",
     "ref_source": "theory: 2-D Euler density L1 order = k+1 = 3 (CKS1998 §5.1)"},
    {"name": "sm2d_order_p3", "got_pointer": "metrics.sm2d_order_p3.value",
     "ref_pointer": "reference.sm2d_order_p3", "tolerance_rel": 0.03, "unit": "-",
     "ref_source": "theory: 2-D Euler density L1 order = k+1 = 4 (CKS1998 §5.1)"},
    # --- B2 二维守恒量：精确解在 [0,2]^2 上的闭式积分（与 DG 无关） -----------
    {"name": "sm2d_mass_integral", "got_pointer": "metrics.sm2d_mass_integral.value",
     "ref_pointer": "reference.sm2d_mass_integral", "tolerance_rel": 1e-3, "unit": "-",
     "ref_source": "exact conservation: mean(rho)=1 over area 4 => 4"},
    {"name": "sm2d_energy_integral", "got_pointer": "metrics.sm2d_energy_integral.value",
     "ref_pointer": "reference.sm2d_energy_integral", "tolerance_rel": 1e-3, "unit": "-",
     "ref_source": "exact conservation: E = 2.5 + rho with u=v=1 => 2.5*4 + 4 = 14"},
    # --- A2/E3 不等式判据的 hinge 编码（见模块 docstring 的「单侧 hinge 编码」） ----
    {"name": "a2_l1_order_p1", "got_pointer": "metrics.a2_l1_order_p1.value",
     "ref_pointer": "reference.a2_l1_order_p1", "tolerance_rel": HINGE_TOL, "unit": "-",
     "ref_source": "hinge lower bound b=1/2 on the L1 order of the limited square wave "
                   "(A2, CKS1998 §3.2/§4): got=min(order,1/2)/(1/2), ref=1 => PASS iff order>=1/2"},
    {"name": "a2_l1_order_p2", "got_pointer": "metrics.a2_l1_order_p2.value",
     "ref_pointer": "reference.a2_l1_order_p2", "tolerance_rel": HINGE_TOL, "unit": "-",
     "ref_source": "hinge lower bound b=1/2 on the L1 order, P2 (A2): same encoding"},
    {"name": "a2_l1_order_p3", "got_pointer": "metrics.a2_l1_order_p3.value",
     "ref_pointer": "reference.a2_l1_order_p3", "tolerance_rel": HINGE_TOL, "unit": "-",
     "ref_source": "hinge lower bound b=1/2 on the L1 order, P3 (A2): same encoding"},
    {"name": "e3_overshoot_p1", "got_pointer": "metrics.e3_overshoot_p1.value",
     "ref_pointer": "reference.e3_overshoot_p1", "tolerance_rel": HINGE_TOL, "unit": "%",
     "ref_source": "hinge upper bound b=1 % on the pointwise overshoot of the strict "
                   "troubled-cell policy (E3 TVD, CKS1998 §3.2): got=max(over,1)/1, ref=1 "
                   "=> PASS iff overshoot<=1 %"},
    {"name": "e3_overshoot_p2", "got_pointer": "metrics.e3_overshoot_p2.value",
     "ref_pointer": "reference.e3_overshoot_p2", "tolerance_rel": HINGE_TOL, "unit": "%",
     "ref_source": "hinge upper bound b=1 % on the pointwise overshoot, P2 (E3): same encoding"},
    # --- E3 并列门：**论文公开指示器**（TVB M=0）上的同一条 TVD 判据 ----------------
    # 审计 m22343 §3 F1：旧门例钉在 strict_all（逐单元 minmod，"构造即真"），0.000 % 是
    # 代码性质而非数值证据。这里把门并到 CKS1998 §3.2 的 TVB 指示器上（limiter=tvb,
    # troubled_policy=tvb, tvb_constant=0），使判据**可能为假**（不限幅对照 6–9 %、
    # 粗网格 N=80 时 1.389 %，见 validation/repro_cks1998/hinge_audit.py 的扫描）。
    {"name": "e3_paper_overshoot_p1", "got_pointer": "metrics.e3_paper_overshoot_p1.value",
     "ref_pointer": "reference.e3_paper_overshoot_p1", "tolerance_rel": HINGE_TOL,
     "unit": "%",
     "ref_source": "hinge upper bound b=1 % on the pointwise overshoot at the **paper** "
                   "indicator (TVB M=0), P1, pinned to the preregistered resolution "
                   "(E3_PREREGISTRATION): got=max(over,1)/1, ref=1 => PASS iff overshoot<=1 %"},
    {"name": "e3_paper_overshoot_p2", "got_pointer": "metrics.e3_paper_overshoot_p2.value",
     "ref_pointer": "reference.e3_paper_overshoot_p2", "tolerance_rel": HINGE_TOL,
     "unit": "%", "ref_source": "hinge upper bound b=1 %, paper TVB M=0 indicator, P2"},
]

#: 不等式判据 → 单侧 hinge 的登记表（唯一真源；workload 与测试都从这里取）。
#: ``kind='lower'``：PASS 当且仅当 ``raw >= bound``；``kind='upper'``：``raw <= bound``。
HINGE_RULES: List[Dict[str, Any]] = [
    {"metric": "a2_l1_order_p1", "raw": "a2_l1_order_p1_raw", "kind": "lower",
     "bound": 0.5, "unit": "-",
     "criterion": "A2 limited square wave: L1 order >= 1/2 (CKS1998 half-order floor)"},
    {"metric": "a2_l1_order_p2", "raw": "a2_l1_order_p2_raw", "kind": "lower",
     "bound": 0.5, "unit": "-", "criterion": "A2 L1 order >= 1/2, P2"},
    {"metric": "a2_l1_order_p3", "raw": "a2_l1_order_p3_raw", "kind": "lower",
     "bound": 0.5, "unit": "-", "criterion": "A2 L1 order >= 1/2, P3"},
    {"metric": "e3_overshoot_p1", "raw": "e3_overshoot_p1_raw", "kind": "upper",
     "bound": 1.0, "unit": "%", "criterion": "E3 TVD: pointwise overshoot <= 1 %, P1"},
    {"metric": "e3_overshoot_p2", "raw": "e3_overshoot_p2_raw", "kind": "upper",
     "bound": 1.0, "unit": "%", "criterion": "E3 TVD: pointwise overshoot <= 1 %, P2"},
    # E3 并列门：论文配置（TVB M=0）。**分辨率是判据本体的一部分**（见 E3_PREREGISTRATION）：
    # 同一判据在 N=80 会判红（1.389 %）、N=100 判绿（0 %），故门必须声明它钉在哪一档。
    {"metric": "e3_paper_overshoot_p1", "raw": "e3_paper_overshoot_p1_raw", "kind": "upper",
     "bound": 1.0, "unit": "%",
     "criterion": "E3 TVD at the paper TVB M=0 indicator: pointwise overshoot <= 1 %, P1"},
    {"metric": "e3_paper_overshoot_p2", "raw": "e3_paper_overshoot_p2_raw", "kind": "upper",
     "bound": 1.0, "unit": "%",
     "criterion": "E3 TVD at the paper TVB M=0 indicator: pointwise overshoot <= 1 %, P2"},
]

#: E3 并列门的**分辨率预登记**（审计 m22343 §3 F1/F2 的机器化闭合）。
#:
#: 两处材料级欠账：(i) 旧门例 strict_all 是"构造即真"（逐单元 minmod 按构造无新极值），
#: 0.000 % 不承担判别任务；(ii) 单分辨率门在论文配置下 N=80→100 之间**翻转**
#: （1.389 % → 0.000 %）⇒ "超调 ≤ 1 %" 不是分辨率无关的性质，单点预登记会把
#: "网格选得好"当成"性质成立"。
#:
#: 本表把两件事写进判据本体：
#:   * ``ncells_by_profile``：门**钉住**的分辨率（该度量 rows 行的 ``ncells`` 与之不符 ⇒
#:     ``DocError``，防止"悄悄换网格"）；smoke 档只用于契约自检，故另钉一档并显式标注。
#:   * ``coarse_ncells_by_profile``：**粗网格披露点**（进 rows 的 ``E3_paper_resolution``
#:     节，只披露不判定）——它是"该判据对分辨率敏感"的可见证据，而不是第二个门。
E3_PREREGISTRATION: Dict[str, Dict[str, Any]] = {
    "e3_paper_overshoot_p1": {"limiter": "tvb_M0",
                              "ncells_by_profile": {"bench": 100, "smoke": 40},
                              "coarse_ncells_by_profile": {"bench": 80, "smoke": 20}},
    "e3_paper_overshoot_p2": {"limiter": "tvb_M0",
                              "ncells_by_profile": {"bench": 100, "smoke": 40},
                              "coarse_ncells_by_profile": {"bench": 80, "smoke": 20}},
}


def hinge(raw: float, bound: float, kind: str) -> float:
    """不等式 → 双侧可比值的**保序**编码（见模块 docstring）。

    ``kind='lower'``（x >= b）: ``min(x, b)/b``；``kind='upper'``（x <= b）: ``max(x, b)/b``。
    两侧都满足：PASS 当且仅当原始不等式成立，且"比要求更好"精确饱和到 1.0。
    """
    x = float(raw)
    b = float(bound)
    if not (b > 0.0):
        raise DocError(f"hinge: bound 必须为正，得到 {bound!r}")
    if kind == "lower":
        return min(x, b) / b
    if kind == "upper":
        return max(x, b) / b
    raise DocError(f"hinge: 未知 kind={kind!r}（只支持 'lower'/'upper'）")


def hinge_holds(raw: float, bound: float, kind: str, tol: float = 0.0) -> bool:
    """原始不等式判据本体（与 hinge 编码做等价性对拍用）。"""
    x, b = float(raw), float(bound)
    return (x >= b * (1.0 - tol)) if kind == "lower" else (x <= b * (1.0 + tol))


#: hinge 化之后已无「进不了平台」的判据；保留该登记表为**空**，以便后续新增时仍显式声明。
#: （历史：A2/E3 曾因不等式不可用相对误差编码而列在此处，m22274 用 hinge 闭合。）
NOT_ENCODED: Dict[str, str] = {}


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


def _verify_hinge(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """hinge 度量的三方一致性核验（rows → raw → hinge → metrics）。

    实现见 ``validation/repro_cks1998/hinge_audit.py``（延迟 import 以避免循环）。
    没有 hinge 度量时返回空表；有则**不一致即抛 DocError**，本函数因此把
    "workload 自述的 raw↔hinge" 从"自证"提升为"文档内可交叉重算"。
    """
    if not HINGE_RULES:
        return {}
    from validation.repro_cks1998.hinge_audit import verify_hinge_consistency
    return verify_hinge_consistency(doc)


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
    hinge_report = _verify_hinge(doc)
    from validation.repro_cks1998.hinge_audit import verify_preregistration
    prereg_report = verify_preregistration(doc)
    return {"parsed": parsed, "result_meta": meta, "rows": rows,
            "hinge": hinge_report, "preregistration": prereg_report,
            "passed": [r["metric"] for r in rows if r["passed"]],
            "failed": [r["metric"] for r in rows if not r["passed"]]}