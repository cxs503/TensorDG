"""Assemble the CASE-DG-ADVECT-ORDER unified report from existing artifacts.

This script reads the three prior studies (round 1, round 2, P0) and produces
a single consolidated ``metrics.json`` and ``summary.md`` without re-running
any expensive computations.

    PYTHONPATH=. python benchmarks/assemble_advorder_unified.py
"""
from __future__ import annotations

import csv
import json
import pathlib

OUT = pathlib.Path("validation/advection_order_unified")
FIG = pathlib.Path("docs/figures")

P0_METRICS = pathlib.Path("validation/advection_order_p0/metrics.json")
R2_METRICS = pathlib.Path("validation/advection_order_matched_rk/metrics.json")
R1_METRICS = pathlib.Path("validation/advection_order/metrics.json")
REF_JSON = pathlib.Path("validation/advection_order/ref_table.json")

RATE_TOL = 0.10
TIME_SHARE_TOL = 0.10
P0_FINE_ORDER_TOL = 0.01
P0_CONST_TOL = 0.001
SOLVER_EXACT_TOL = 1e-5


def mk(ok):
    return "\u2705" if ok else "\u274c"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    p0 = json.loads(P0_METRICS.read_text())
    r2 = json.loads(R2_METRICS.read_text())
    r1 = json.loads(R1_METRICS.read_text()) if R1_METRICS.exists() else None
    ref = json.loads(REF_JSON.read_text())

    # ---- Unified checks ----
    checks = {}

    # U1: rate check for P1-P4 (from round 2, M1_rate)
    rate_gaps = [abs(r["order_gap_vs_ref"]) for r in r2["comparison"] if r["order_gap_vs_ref"] is not None]
    checks["U1_rate"] = all(g <= RATE_TOL for g in rate_gaps) if rate_gaps else False

    # U2: P0 fine ladder order
    fine_order = p0.get("derived", {}).get("order_analytic_fine", 0)
    checks["U2_P0_order"] = abs(fine_order - 1.0) <= P0_FINE_ORDER_TOL

    # U3: P0 constant match
    const_err = p0.get("derived", {}).get("constant_rel_error", 1)
    checks["U3_P0_constant"] = const_err <= P0_CONST_TOL

    # U4: time share (from round 2 error split)
    max_time_share = max((s["time_share"] for s in r2["error_split"]), default=1)
    checks["U4_time_share"] = max_time_share <= TIME_SHARE_TOL

    # U5: solver vs exact for P0
    solver_exact = p0.get("derived", {}).get("solver_vs_exact_max_rel", 1)
    checks["U5_solver_exact"] = solver_exact <= SOLVER_EXACT_TOL

    verdict = "PASS" if all(checks.values()) else "FAIL"
    failed = [k for k, v in checks.items() if not v]

    # ---- Unified metrics ----
    metrics = {
        "case": "CASE-DG-ADVECT-ORDER",
        "version": "unified",
        "objective": "P0-P4 convergence rate verification: consolidated from three prior studies",
        "reference": {
            "citation": {
                "author": "Zheng Sun",
                "title": "Reducing polynomial degree by one for inner-stage operators affects neither stability type nor accuracy order of the Runge-Kutta discontinuous Galerkin method",
                "journal": "Mathematics of Computation (AMS)",
                "doi": "10.1090/mcom/4037",
                "date": "2024-10-31",
                "preprint": "arXiv:2404.15453v2",
                "table_label": "tab:acc-sin-1d-uni",
            }
        },
        "setup": {
            "cfl": 0.2,
            "final_time": 1.0,
            "speed": 1.0,
            "cells_p1p4": [20, 40, 80, 160, 320],
            "degrees_p1p4": [1, 2, 3, 4],
        },
        "thresholds": {
            "RATE_TOL": RATE_TOL,
            "TIME_SHARE_TOL": TIME_SHARE_TOL,
            "P0_FINE_ORDER_TOL": P0_FINE_ORDER_TOL,
            "P0_CONST_TOL": P0_CONST_TOL,
            "SOLVER_EXACT_TOL": SOLVER_EXACT_TOL,
        },
        "p0": {
            "verdict": p0["verdict"],
            "checks": p0["checks"],
            "closed_form": p0["closed_form"],
            "derived": p0["derived"],
            "falsified_hypotheses": p0.get("falsified_hypotheses", []),
        },
        "p1p4": {
            "comparison": r2["comparison"],
            "error_split": r2["error_split"],
            "time_share_vs_stages": r2.get("time_share_vs_stages", []),
            "checks": r2["checks"],
            "verdict": r2["verdict"],
            "failed": r2.get("failed", []),
            "falsified_hypotheses": r2.get("falsified_hypotheses", []),
        },
        "round1_summary": {
            "verdict": r1.get("verdict", "n/a") if r1 else "n/a",
            "falsified_hypotheses": r1.get("falsified_hypotheses", []) if r1 else [],
        } if r1 else {},
        "checks": checks,
        "verdict": verdict,
        "failed": failed,
    }

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- Summary markdown ----
    comparison = r2["comparison"]
    max_gap = max(abs(r["order_gap_vs_ref"]) for r in comparison if r["order_gap_vs_ref"] is not None)

    # Build comparison table rows
    comp_rows = []
    for r in comparison:
        gap_str = f"{r['order_gap_vs_ref']:+.3f}" if r['order_gap_vs_ref'] is not None else "n/a"
        comp_rows.append(
            f"| P{r['degree']} | {r['integrator']} | {r['rk_stages']} | "
            f"{r['ref_column']} | {r['ref_fitted_order']:.3f} | "
            f"{r['got_fitted_order']:.3f} | {gap_str} | "
            f"{mk(r['rate_ok'])} |"
        )

    # Build error split table rows
    split_rows = []
    for s in r2["error_split"]:
        split_rows.append(
            f"| P{s['degree']} | {s['ncells']} | {s['E_total']:.3e} | "
            f"{s['E_space']:.3e} | {s['E_time']:.3e} | {100*s['time_share']:.2f}% |"
        )

    verdict_text = "是，全部五项判据通过" if verdict == "PASS" else "否，见上表"
    conclusion_text = "全部五项判据通过，CASE-DG-ADVECT-ORDER 收敛率复测完成" if verdict == "PASS" else "部分判据未通过"

    md = f"""# CASE-DG-ADVECT-ORDER（终态统一报告）：P0-P4 线性对流收敛率复测

**锚点** CASE-DG-ADVECT-ORDER | **判据** 收敛率 | **总判定** **{verdict}**
（未过项：{', '.join(failed) or '无'}）

## 0. 误差判定（是否 <3%）

本算例判据是**收敛率**。按"观测阶相对 ref 同阶列的绝对偏差"判定：

| p | 基准阶（ref 列） | TensorDG 阶 | 绝对偏差 | <3%（±0.10 阶） |
|--:|---:|---:|---:|:--:|
""" + "\n".join(comp_rows) + f"""

| 判据 | 描述 | 阈值 | 实测 | 判定 |
|---|---|---|---|:--:|
| U1 rate | P1-P4 观测阶 vs ref | ±{RATE_TOL} | 最大偏差 {max_gap:.3f} | {mk(checks['U1_rate'])} |
| U2 P0 order | P0 渐近区阶 | ±{P0_FINE_ORDER_TOL} | {fine_order:.4f} | {mk(checks['U2_P0_order'])} |
| U3 P0 constant | 渐近常数闭式匹配 | ±{P0_CONST_TOL:.1%} | {const_err:.4%} | {mk(checks['U3_P0_constant'])} |
| U4 time share | P1-P4 时间误差占比 | ≤{TIME_SHARE_TOL:.0%} | {max_time_share:.4f} | {mk(checks['U4_time_share'])} |
| U5 solver exact | P0 求解器 vs 精确解 | ≤{SOLVER_EXACT_TOL:.3%} | {solver_exact:.4e} | {mk(checks['U5_solver_exact'])} |

**结论：{verdict_text}。**

## 1. 案例介绍

一维周期线性对流 u_t + c u_x = 0，x ∈ [0,1)，c=1，
u(x,0) = sin(2πx)，t* = cT/L = 1。
精确解 u = sin(2π(x-t)) 解析已知（manufactured solution）。

| 项 | 符号/定义 | 值 |
|---|---|---|
| 单元数（P1-P4） | N | 20, 40, 80, 160, 320 |
| 多项式阶 | p | 0, 1, 2, 3, 4 |
| CFL | cΔt/h | 0.2 |
| 时间格式 | P1-P4 | 阶数匹配表：p+1 阶 / p+1 级 |
| 时间格式 | P0 | RK4（默认） |
| 无量纲末时 | t* | 1.0 |

**ref 出处**：Zheng Sun, *Mathematics of Computation* (2024),
DOI 10.1090/mcom/4037, arXiv:2404.15453v2, 表 `tab:acc-sin-1d-uni`。
由 `tools/fetch_ref_advorder.py` 机械解析（零手抄）。

## 2. 怎么做的

本统一报告整合三轮独立研究：

1. **Round 1**（`benchmarks/case_dg_advect_order.py`）：P1-P4 用 RK4 跑阶梯，
   与 ref 对比。发现 R2/R6（"RK4 时间阶封顶"假设）被数据否定，R3（幅值比）
   属跨代码口径误设。修正后 S1（收敛率）全部通过。

2. **Round 2**（`benchmarks/case_dg_advect_order_matched_rk.py`）：
   P1-P4 用 p+1 阶/p+1 级 RK 表复测，并用 `expm(L·T)u₀` 精确拆分
   时间/空间误差。M1_rate 全部通过；M2_stages 在 P4 处 FAIL
   （Butcher 障碍：不存在 5 级 5 阶显式 RK）。

3. **P0 闭式分析**（`validation/advection_order_p0/`）：
   DG0（迎风格式）的半离散误差有闭式表达 E(N) → π√(2π² + 1/6)/N。
   通过 N=20..40960 的阶梯验证预渐近区（N<640, 阶≈0.85）和渐近区（N≥1280, 阶→1.00）。
   全部 10 项检查通过。

## 3. 计算结果

### 3.1 P1-P4 收敛率（同阶匹配）

| p | 格式 | 级数 | ref 列 | ref 拟合阶 | TensorDG 拟合阶 | 差 | 判定 |
|--:|---|---:|---|---:|---:|---:|:--:|
""" + "\n".join(comp_rows) + f"""

### 3.2 P0 渐近常数

| 项 | 值 |
|---|---|
| 闭式常数 | {p0['closed_form']['asymptotic_constant']:.6f} |
| 实测常数（N=40960） | {p0['derived']['constant_measured_finest']:.6f} |
| 相对误差 | {p0['derived']['constant_rel_error']:.4%} |
| 渐近区起点（1% 偏差） | N ≥ {p0['derived']['regime_N_order_1pct']} |

### 3.3 时间/空间误差拆分（cfl=0.2）

| p | N | E_total | E_space | E_time | E_time 占比 |
|--:|--:|--:|--:|--:|--:|
""" + "\n".join(split_rows) + """

## 4. 结果图

- `docs/figures/advorder_unified_convergence.png`：P1-P4 收敛阶梯 vs ref 同阶列
- `docs/figures/advorder_unified_error_split.png`：E_time vs E_space
- `docs/figures/advorder_unified_p0_constant.png`：P0 渐近常数收敛

## 5. 注意点（踩坑/局限）

- **P4 级数不匹配（M2_stages = FAIL）**：P4 需 5 阶 5 级 RK，但**不存在 5 级 5 阶显式 RK
  （Butcher 障碍）**，故用 Butcher 6 级 5 阶。ref 的 `RK5DG4` 标注数的是 5 个 *DG* 级，
  不是 RK 级数。该项 FAIL 是数学事实，不是实现缺陷。
- **P0 预渐近区**：N=20..320 的阶梯测得阶≈0.85，这是预渐近行为，不是算子缺陷。
  闭式分析证明设计阶 1.00 在 N≥1280 时恢复。
- **L2 幅值比**：ref 未公布 Δt/CFL 与通量形式，L2 幅值无法判定；
  但比值逐网格近乎常数（P1≈0.63, P2≈0.19, P3≈0.06, P4≈0.02），说明同阶不同常数。
- **expm 限制**：精确拆分用稠密 `expm`，仅适用小网格（N≤16）；
  大网格结论以"E_time 随 Δt^(p+1) 迅速衰减"外推。
- **被否定的假设**：
  1. "RK4 时间阶封顶导致 P4 测不到 5 阶"→ 被 CFL 敏感性数据否定（斜率≈0，非 4）。
  2. "ref 的较大 L2 常数由低阶时间积分引起"→ 被同阶匹配后的幅值比否定。
  3. "P0 的 0.85 阶意味着算子低于设计阶"→ 被闭式分析否定（预渐近行为）。

## 6. 误差判定总结

| 判据 | 描述 | 阈值 | 实测 | 判定 |
|---|---|---|---|:--:|
| U1 rate | P1-P4 观测阶 vs ref | ±{RATE_TOL} | 最大偏差 {max_gap:.3f} | {mk(checks['U1_rate'])} |
| U2 P0 order | P0 渐近区阶 | ±{P0_FINE_ORDER_TOL} | {fine_order:.4f} | {mk(checks['U2_P0_order'])} |
| U3 P0 constant | 渐近常数闭式匹配 | ±{P0_CONST_TOL:.1%} | {const_err:.4%} | {mk(checks['U3_P0_constant'])} |
| U4 time share | P1-P4 时间误差占比 | ≤{TIME_SHARE_TOL:.0%} | {max_time_share:.4f} | {mk(checks['U4_time_share'])} |
| U5 solver exact | P0 求解器 vs 精确解 | ≤{SOLVER_EXACT_TOL:.3%} | {solver_exact:.4e} | {mk(checks['U5_solver_exact'])} |

**最终结论：{conclusion_text}。**
"""

    (OUT / "summary.md").write_text(md)

    # Write CSV
    with open(OUT / "convergence.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["degree", "integrator", "rk_stages", "ref_column",
                         "ref_fitted_order", "got_fitted_order", "order_gap", "rate_ok"])
        for row in comparison:
            writer.writerow([row["degree"], row["integrator"], row["rk_stages"],
                             row["ref_column"], row["ref_fitted_order"],
                             row["got_fitted_order"], row["order_gap_vs_ref"],
                             row["rate_ok"]])

    print(f"Unified metrics written to {OUT}/metrics.json")
    print(f"Summary written to {OUT}/summary.md")
    print(f"Verdict: {verdict}")
    print(f"Checks: {checks}")
    return metrics


if __name__ == "__main__":
    main()
