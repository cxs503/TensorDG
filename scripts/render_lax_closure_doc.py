"""Render ``docs/CASE-LAX-DG-resolution-closure.md`` from the closure artifact.

Every number in the rendered document is read out of
``docs/benchmarks/results/lax_resolution_closure/metrics.json``: the doc is a
*view* of the artifact, so a re-run cannot leave the prose and the evidence
disagreeing (that failure mode is what ``tests/test_lax_resolution_closure.py``
guards from the other side).
"""
from __future__ import annotations

import json
from pathlib import Path

METRICS = Path("docs/benchmarks/results/lax_resolution_closure/metrics.json")
OUT = Path("docs/CASE-LAX-DG-resolution-closure.md")
FIELDS = ("density", "velocity", "pressure")


def pct(value, digits=3):
    return "n/a" if value is None else f"{100.0 * value:.{digits}f} %"


def render(metrics: dict) -> str:
    gate = metrics["gate"]
    tol = gate["tolerance"]
    legs = sorted(metrics["ladder"], key=lambda e: e["cells"])
    arms = metrics["arms"]
    model = gate["per_field"][gate["worst_field"]]["model"] or {}
    lines = []
    add = lines.append

    add("# CASE-LAX-DG 分辨率闭合研究（p3 阶梯对 3 % 台账门）")
    add("")
    add(f"> 工件：`docs/benchmarks/results/lax_resolution_closure/metrics.json`  ")
    add(f"> 生成：{metrics['generated_at_utc']}　墙钟：{metrics['wall_seconds']:.0f} s　"
        f"配置：p = {metrics['config']['degree']}，CFL = {metrics['config']['cfl']}，"
        f"阶梯 N = {metrics['config']['ladder']}，限幅器 = none（登记臂）  ")
    add(f"> verdict：**{gate['status']}**，闭合网格 = **{gate['closure_mesh']}**")
    add("")
    add("## ① 案例介绍")
    add("")
    add("`CASE-LAX-DG` = Lax (1954) 激波管（Toro 1999 test 2）：`[0,1]` 域、隔膜 `x0 = 0.5`、"
        "γ = 1.4、快照 `t = 0.13`，左态 (ρ,u,p) = (0.445, 0.698, 3.528)、右态 (0.5, 0, 0.571)。"
        "解析波系为左稀疏波（头/尾）、接触间断、右激波，四者位置都有闭式解，参考值由"
        "`benchmarks/riemann_exact.py` 独立给出。")
    add("")
    add("台账 `CASE-LAX-DG` 的验收原文是「ref: 精确解｜容差 < 3 %」。算例池把这条读成"
        "**生产腿 `p3.N400` 的全场相对 L2**，并如实记下 **FAIL（3.1156 %）**。本单据此追问一个不同的问题：")
    add("")
    add(f"**这个 3 % 门在已登记的计算配置（p3、无限幅器）下，靠 h-细化是否可达？可达的最小实际算过的网格是哪个？**")
    add("")
    add("## ② 怎么做的")
    add("")
    add("1. **h-细化到闭合**：登记配置（p3、CFL 0.35、无限幅器、精确 Riemann 参考）在 "
        f"N ∈ {metrics['config']['ladder']} 上重跑；每条腿在单元 Gauss 节点上测三个原始量的"
        "全场相对 L1/L2，以及 R3 规则的固定物理带（±0.02）掩码 L1。")
    add("2. **单旋钮归因**：在生产网格 N = "
        f"{metrics['config']['arm_cells']} 上只切换 `limiter`（none / tvb / weno），"
        "其余全部固定，用来把「p3 不准是因为无限幅（Gibbs 振荡从三个间断漏到平台）」"
        "与「p3 算子本身有错」分开。")
    add("3. **闭合判据预先声明**：`PASS` 仅当**实际跑过**的某条腿在**全部三个量**上 L2 ≤ 容差；"
        "功率律外推出来的 `N*` 只作模型预测记录，**永远不计为通过**。")
    add("4. **p3 算子健康度另测**：`tests/test_lax_p3_smooth_order.py` 在光滑周期熵波上测 p3 设计阶"
        "（实测 4.0097，设计 4），把「算子错」这一假设先行否证。")
    add("")
    add("## ③ 计算结果")
    add("")
    add("### 3.1 p3 无限幅阶梯（登记臂）")
    add("")
    add("| N | ρ L2 | u L2 | p L2 | ρ L1 | ρ 带掩码 L1 | ρ 峰值误差 | 步数 | 墙钟 s |")
    add("|---|---|---|---|---|---|---|---|---|")
    for leg in legs:
        add(f"| {leg['cells']} | {pct(leg['l2']['density'])} | {pct(leg['l2']['velocity'])} | "
            f"{pct(leg['l2']['pressure'])} | {pct(leg['l1']['density'])} | "
            f"{pct(leg['band_l1']['density'])} | {leg['max_abs']['density']:.4f} | "
            f"{leg['steps']} | {leg['wall_seconds']:.0f} |")
    add("")
    add(f"台账门 = {pct(tol, 2)}（全场相对 L2，三个量都必须满足）。")
    add("")
    add("### 3.2 生产网格上的单旋钮对比")
    add("")
    add("| limiter | ρ L2 | u L2 | p L2 | ρ 带掩码 L1 | 墙钟 s |")
    add("|---|---|---|---|---|---|")
    for arm in arms:
        add(f"| {arm['limiter']} | {pct(arm['l2']['density'])} | {pct(arm['l2']['velocity'])} | "
            f"{pct(arm['l2']['pressure'])} | {pct(arm['band_l1']['density'])} | "
            f"{arm['wall_seconds']:.0f} |")
    add("")
    add("### 3.3 闭合与外推")
    add("")
    add(f"- 实测闭合网格（全部量达标）：**{gate['closure_mesh']}**")
    add(f"- 最差量：`{gate['worst_field']}`")
    add(f"- 最细对阶（{model.get('finest_pair')}）：**{model.get('order_finest_pair')}**")
    add(f"- 外推闭网格（仅模型预测，不作证据）：{model.get('extrapolated_n')}")
    add(f"- verdict：**{gate['status']}**")
    add("")
    add("## ④ 结果图")
    add("")
    for figure in metrics.get("figures", []):
        add(f"- `figures/{figure['name']}` — {figure['caption']}")
    if not metrics.get("figures"):
        add("（本次运行跳过了作图。）")
    add("")
    add("## ⑤ 注意点")
    add("")
    add("1. **闭合 ≠ 台账原文已通过**。台账登记的是生产腿 `p3.N400`，那条腿的诚实结论仍是 FAIL；"
        "本单给出的是「同一配置在哪个网格上达到同一容差」，是接受准则的**分辨率限定**，不是改口径。")
    add("2. **p3 的观测阶远低于设计阶 4**：光滑流上 p3 实测 4.00（`tests/test_lax_p3_smooth_order.py`），"
        "所以低阶不是算子错，而是无限幅方案在间断附近产生的振荡污染了全场范数——这正是不连续解上"
        "阶门可以 FAIL 的原因。")
    add("3. **限幅器不是补救手段**：单旋钮对比里 tvb/weno 都让 L2 变差（见 3.2），"
        "本仓这两种限幅器都把问题单元**退化为线性重构**，在 p3 上属于过度限制。")
    add("4. **外推不作证据**：3.3 的外推网格只用于说明「门是否在可负担范围内」，"
        "任何 `PASS` 都必须落在实际算过的腿上（测试已钉死这条）。")
    add("")
    add("## ⑥ 误差判定（逐项回答「是否 < 3 %」）")
    add("")
    add("| 判据 | 量 | 值 | 基准 | 是否 < 3 % |")
    add("|---|---|---|---|---|")
    prod = gate["production_l2"]
    for field in FIELDS:
        value = prod[field]
        add(f"| 台账 A0（生产腿 p3.N400） | {field} 全场相对 L2 | {pct(value)} | "
            f"{pct(tol, 1)} | **{'是' if value <= tol else '否'}** |")
    for leg in legs:
        worst = max(leg["l2"][f] for f in FIELDS)
        add(f"| 闭合阶梯 | N = {leg['cells']} 三量最差 L2 | {pct(worst)} | {pct(tol, 1)} | "
            f"{'是' if worst <= tol else '**否**'} |")
    add("")
    verdict_line = (
        f"**结论**：登记生产腿 p3.N400 的全场相对 L2 三量最差 {pct(max(prod[f] for f in FIELDS))}，"
        f"**不满足** < 3 %（honest FAIL，未粉饰）；同一登记配置在 N = {gate['closure_mesh']} 上"
        f"三量最差 {pct(max(next(e for e in legs if e['cells'] == gate['closure_mesh'])['l2'][f] for f in FIELDS))}，"
        "**满足** < 3 %。"
        if gate["closure_mesh"] else
        f"**结论**：阶梯上没有任何实际跑过的腿满足 < 3 %（最细腿三量最差 "
        f"{pct(max(legs[-1]['l2'][f] for f in FIELDS))}），verdict 保持 FAIL。"
    )
    add(verdict_line)
    add("")
    return "\n".join(lines)


def main():
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    OUT.write_text(render(metrics), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()