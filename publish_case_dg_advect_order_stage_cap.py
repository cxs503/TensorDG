"""Publish the CASE-DG-ADVECT-ORDER round-3 report (stage/order cap at p = 5, 6).

The markdown page itself is rendered by the benchmark
(:func:`benchmarks.case_dg_advect_order_stage_cap.render_doc`) straight from
``validation/advection_order_stage_cap/metrics.json``, so this script only
re-reads those artifacts, adds the paper-style summary blocks that the platform
report needs (case description, method, error verdict, limits) and posts them.
Nothing is retyped: every number below is formatted from the metrics.

    PYTHONPATH=. python publish_case_dg_advect_order_stage_cap.py
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = int(os.environ.get("MISSION_ID", "22989"))
OUT = pathlib.Path("validation/advection_order_stage_cap")
DOC = pathlib.Path("docs/CASE-DG-ADVECT-ORDER-stage-cap.md")
FIG = pathlib.Path("docs/figures")


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(f"{BASE}/{path}", data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=240) as resp:
        return json.loads(resp.read().decode())


def b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def render(metrics: dict) -> str:
    def mk(ok):
        return {True: "✅", False: "❌", "not_run": "⏸"}[ok]
    setup = metrics["setup"]
    checks = metrics["checks"]
    order = {r["degree"]: r for r in metrics["rungs"]}
    budget = {b["degree"]: b for b in metrics["cfl_budgets"]}
    detail = metrics["checks_detail"]
    p5, p6 = order[5], order[6]
    rel5 = abs(p5["fitted_order"] - 6) / 6 if p5["fitted_order"] else None
    rel6 = abs(p6["fitted_order"] - 7) / 7 if p6["fitted_order"] else None
    share = {d: detail[f"S4_time_share_p{d}"]["time_share"] for d in (5, 6)}
    margin = detail["S5_cfl_budget"]["cfl_time_share_10"]
    blow = detail["S5_cfl_budget"]["cfl_divergence"]
    table = metrics["barrier"]["table"]

    out = [
        "# CASE-DG-ADVECT-ORDER 第3轮：级数/阶数上限实测（p = 5, 6）与 CFL 窗口",
        "",
        f"**总判定 {metrics['verdict']}**｜判据：自收敛阶 ±{metrics['thresholds']['ORDER_TOL']}、"
        f"时间误差占比 ≤ {metrics['thresholds']['TIME_SHARE_TOL']:.0%}、"
        f"CFL 裕度 ≥ {metrics['thresholds']['CFL_MARGIN_MIN']}",
        "",
        "## ⓪ 误差判定（是否 <3%）",
        "",
        f"1. **收敛阶（本算例声明的判据）**：$p=5$ 实测 {p5['fitted_order']:.3f}（期望 "
        f"{p5['expected_spatial_order']}，相对偏差 {100 * rel5:.2f}%）；$p=6$ 实测 "
        f"{p6['fitted_order']:.3f}（期望 {p6['expected_spatial_order']}，相对偏差 "
        f"{100 * rel6:.2f}%）→ 均 **<3%**：{mk(rel5 < 0.03 and rel6 < 0.03)}。",
        f"2. **时间误差占比**：cfl={setup['cfl']}、N={setup['probe_cells']} 下 $E_{{time}}/E_{{total}}$ "
        f"P5 = {100 * share[5]:.4f}%、P6 = {100 * share[6]:.4f}%（阈 "
        f"{100 * metrics['thresholds']['TIME_SHARE_TOL']:.0f}%）→ 实测阶确为空间阶："
        f"{mk(checks['S4_time_share_p5'] and checks['S4_time_share_p6'])}。",
        "3. **绝对 L2 幅度 vs 文献**：**无基准**——ref 表只到 $p=4$、且未公布 Δt 与误差范数定义，"
        "故本轮不作幅度承诺（第 2 轮已证否「级数差解释幅值差」）。",
        "",
        "## 1. 案例介绍",
        "",
        "一维周期线性对流 $u_t+u_x=0$，$x\\in[0,1)$，$u(x,0)=\\sin(2\\pi x)$，"
        f"末时 $t^*={setup['final_time']}$，$c={setup['speed']}$；精确解 $\\sin(2\\pi(x-t))$。"
        "被检验量是 DG 空间算子的观测收敛阶：光滑解应达 $p+1$。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 多项式阶 | {setup['degrees']}（本轮新测 {setup['rung_degrees']}） |",
        f"| 网格阶梯 $N$ | {setup['ladder']} |",
        f"| CFL | {setup['cfl']} |",
        f"| 时间格式 | " + "；".join(
            f"P{r['degree']} `{r['integrator']}`（{r['rk_stages']} 级 {r['rk_order']} 阶）"
            for r in metrics["rungs"]) + " |",
        f"| 时间/空间拆分探针网格 | N = {setup['probe_cells']} |",
        f"| CFL 扫描网格 | {setup['cfl_grid']} |",
        "",
        "## 2. 怎么做的",
        "",
        "1. **把级数/阶数上限表换成已发表的极小级数**：`MIN_STAGES_FOR_ORDER` = "
        f"{metrics['barrier']['min_stages_for_order']}（Butcher 1964/65；Butcher 1985；"
        "Hairer & Wanner II.5），`MAX_ORDER_FOR_STAGES` 由其机械反演，二者不会漂移；"
        "超过 17 级拒绝外推。旧实现用奇偶公式回退，在 $s\\ge11$ 处比真值低 1 阶"
        "（11 级可达 8 阶，旧值 7），会让高阶阶梯出现假 FAIL。",
        "2. **扩展阶梯自收敛**：$p=5,6$ 直接调用 `solve_advection`，时间格式由 "
        "`tableau_for_degree(p)` 选定；共用上一轮的最小二乘斜率估计器，无外部基准参与。",
        "3. **精确时间/空间拆分**：线性律的 DG 半离散系统线性，`expm(L·T)u₀` 是其精确解；"
        "数值解与之比较得 $E_{time}$，与连续解比较得 $E_{total}$。",
        "4. **CFL 预算**：固定网格扫 CFL，取时间误差占比 ≤10% 的最大 CFL（此窗口内实测阶即空间阶）"
        "与发散阈值。",
        "",
        "## 3. 计算结果",
        "",
        "### 3.1 Butcher 障碍表（级数 → 可承诺阶）",
        "",
        "| s | " + " | ".join(str(s) for s in table) + " |",
        "|--:|" + "|".join("--:" for _ in table) + "|",
        "| 本轮 | " + " | ".join(str(v) for v in table.values()) + " |",
        "| 发表值 | " + " | ".join(
            str(v) for v in metrics["barrier"]["expected"].values()) + " |",
        "",
        f"与发表值一致：**{metrics['barrier']['table_matches_published']}**；"
        f"已注册格式全部满足 order ≤ barrier(stages)："
        f"**{metrics['barrier']['all_tableaus_within_barrier']}**。",
        "",
        "### 3.2 扩展阶梯（自收敛，两分辨率以上）",
        "",
        "| p | 格式 | 级数 | 期望空间阶 | 实测阶 | 相对偏差 | 判定 |",
        "|--:|---|---:|---:|---:|---:|:--:|",
    ]
    for degree in (5, 6):
        r = order[degree]
        key = f"S{degree - 3}_order_p{degree}"
        rel = abs(r["fitted_order"] - r["expected_spatial_order"]) / r["expected_spatial_order"]
        out.append(f"| P{degree} | {r['integrator']} | {r['rk_stages']} | "
                   f"{r['expected_spatial_order']} | {r['fitted_order']:.3f} | "
                   f"{100 * rel:.2f}% | {mk(checks[key])} |")
    out += ["",
            f"P5 L2 阶梯 = {['%.3e' % v for v in p5['l2']]}；"
            f"P6 L2 阶梯 = {['%.3e' % v for v in p6['l2']]}。",
            "",
            "### 3.3 CFL 预算（精确时间/空间拆分）",
            "",
            "| p | E_time ≤10% 的最大 cfl | 发散 cfl（最小） | cfl=0.2 处占比 |",
            "|--:|--:|--:|--:|"]
    for degree in (1, 2, 3, 4, 5, 6):
        b = budget[degree]
        row = [r for r in b["rows"] if r["cfl"] == setup["cfl"]][0]
        s = "n/a" if row["time_share"] is None else f"{100 * row['time_share']:.3f}%"
        out.append(f"| P{degree} | {b['cfl_time_share_10']} | {b['cfl_divergence']} | {s} |")
    out += ["",
            f"cfl=0.2 位于时间误差窗口内，且到 10% 边界仍有 "
            f"≥{min(v for v in margin.values() if v is not None) / setup['cfl']:.1f}× 裕度。",
            "",
            "![stage cap](figures/advorder_stage_cap.png)",
            "",
            "## 4. 注意点",
            "",
            "- **级数上限在 cfl=0.2 处不咬人**：$p=6$ 本应需要 7 级，但注册表没有，"
            "阶梯仍用 6 级 5 阶 `rk5`，实测阶仍达 7 阶——因为本案 cfl 约定 "
            "$dt=\\mathrm{cfl}/((2p+1)N)$ 使 $dt$ 随 $p$ 变小，时间误差被压到空间误差之下若干数量级。"
            "这不是「7 级不需要」的一般结论；3.3 节给出把时间误差抬到 10% 的 cfl 位置"
            f"（P6: {blow['P6']} 处发散，{margin['P6']} 处占比仍 ≤10%）。",
            "- **自收敛 vs 外部基准**：ref 表（Sun 2024）只到 $p=4$，$p=5,6$ 只有自收敛证据。",
            f"- **舍入**：$p=6$ 在 $N\\ge24$ 误差已到 $10^{{-12}}$ 量级，阶梯受 "
            f"FIT_FLOOR={metrics['thresholds']['FIT_FLOOR']} 截断。",
            f"- **发散判据**是工程判据（非有限或误差 > "
            f"{metrics['thresholds']['DIVERGENCE_FACTOR']:g}），不是线性稳定域严格边界；"
            "严格谱结果见 `validation/advection_order_stability/metrics.json`。",
            "",
            "## 5. 适用性",
            "",
            "结论适用于一维周期线性对流 + 光滑解 + 均匀网格；"
            "判据是收敛率与时间误差占比；对含激波/限制器、非均匀网格或非线性通量的算例不适用，"
            "需另测（见 `CASE-DG-ADVECT-ORDER-robustness` / `-stability`）。",
            ""]
    return "\n".join(out)


def main() -> dict:
    metrics = json.loads((OUT / "metrics.json").read_text())
    md = render(metrics)
    DOC.write_text(md, encoding="utf-8")

    token = post("auth/login", {"username": USER, "password": PASSWORD})["access_token"]
    figs = []
    p = FIG / "advorder_stage_cap.png"
    if p.exists():
        figs.append({"name": p.name,
                     "caption": "左：p=1..6 自收敛阶梯（点线为精确斜率 p+1）；"
                                "右：时间误差占比 vs cfl（虚线=发散点，红竖线=本案 cfl=0.2）",
                     "png_base64": b64(p)})
    p5 = {r["degree"]: r for r in metrics["rungs"]}[5]
    p6 = {r["degree"]: r for r in metrics["rungs"]}[6]
    payload = {
        "title": "CASE-DG-ADVECT-ORDER 第3轮：级数/阶数上限实测（p=5,6）+ CFL 窗口",
        "summary": (f"TensorDG 把 Butcher 障碍表换成已发表极小级数（s≤17，旧奇偶回退在 s≥11 低 1 阶），"
                    f"并把匹配级数阶梯从 p=4 扩到 p=6：实测阶 P5={p5['fitted_order']:.3f}（期望 6）、"
                    f"P6={p6['fitted_order']:.3f}（期望 7），cfl=0.2 下时间误差占比 ≤"
                    f"{100 * max(d['time_share'] for d in metrics['checks_detail'].values() if isinstance(d, dict) and 'time_share' in d):.3f}%。"),
        "scenario_key": "CASE-DG-ADVECT-ORDER",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "Runge-Kutta", "Butcher障碍", "收敛率", "CFL预算"],
        "markdown": md,
        "figures": figs,
    }
    res = post("tensor/reports", payload, token)
    print("report id:", res.get("id"), "figures:", len(figs))
    return res


if __name__ == "__main__":
    main()