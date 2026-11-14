#!/usr/bin/env python3
"""Generate the m20486 audit report and platform submission."""
import json, base64, os, sys

WORK = "/root/tp-agent/work/m20486"
METRICS = os.path.join(WORK, "case_euler_vortex_errconserve", "metrics.json")

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def main():
    with open(METRICS) as f:
        metrics = json.load(f)

    gates = metrics["gates"]
    records = metrics["records"]
    exact = metrics["exact_invariants"]

    c1 = gates["C1_tolerance"]
    c2 = gates["C2_error_conservation"]
    c3 = gates["C3_discrete_conservation"]
    c4 = gates["C4_entropy_budget"]
    c5 = gates["C5_crosscheck_order"]

    # Build markdown
    lines = []
    lines.append("# CASE-DG-EULER-VORTEX-ERR：等熵涡移动误差守恒审计核验报告\n")
    lines.append("## ① 案例介绍\n")
    lines.append("**物理背景**：二维等熵涡在周期方盒 [-5,5]² 中匀速平移。精确解保持涡旋形状不变，离散误差应仅取决于网格/通量对，与输送时间无关。本算例检验「误差是否随时间守恒」这一核心问题。\n")
    lines.append("**几何与参数**：")
    lines.append("| 参数 | 值 |")
    lines.append("|---|---|")
    lines.append("| 计算域 | [-5, 5] × [-5, 5]，周期边界 |")
    lines.append("| 涡强度 β | 5.0 |")
    lines.append("| 背景速度 | (1, 1) |")
    lines.append("| 比热比 γ | 1.4 |")
    lines.append("| 精确熵 S = ∫p·ρ⁻ᵞ | %.6f（理论值 = 盒面积 = 100） |" % exact['entropy'])
    lines.append("| 精确核最小密度 | %.6f |" % exact.get('density_min', 0))
    lines.append("")
    lines.append("**基准来源**：算例池条目 EULER-VORTEX-ERR 引用 Yee, Sandham & Djomehri (2000)，容差 <6%；解析解形式见 Shu (1998) §4.4.2。\n")
    lines.append("**两条独立离散路径**：")
    lines.append("- DG-P2/32²：模态 DG，HLLC 通量 + SSP-RK3 时间积分")
    lines.append("- P0/128²（及 n=64/256）：有限体积，Rusanov 通量 + SSP-RK3\n")

    lines.append("## ② 方法\n")
    lines.append("**门禁判据（5 项）**：")
    lines.append("| 门禁 | 物理含义 | 阈值 |")
    lines.append("|---|---|---|")
    lines.append("| C1 快照容差 | t=1 时各族最大场相对 L2 误差 | < 6% |")
    lines.append("| C2 误差守恒 | DG-P2/32² 族误差累积指数 α（e ~ t^α） | α ≤ 1.0 |")
    lines.append("| C3 离散守恒 | 质量/动量/能量相对漂移（周期通量 telescoping） | < 1e-10 |")
    lines.append("| C4 熵预算 | |ΔS|/S(0)（等熵涡熵应不变） | < 1e-3 |")
    lines.append("| C5 交叉校核阶 | P0 族在 t=1 的观测收敛阶（n=64/128/256） | ≥ 0.8 |\n")
    lines.append("**复现命令**：`PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate`\n")

    lines.append("## ③ 计算结果\n")
    lines.append("**逐族逐检查点误差**：")
    lines.append("| 族 | t | 步数 | ρ误差%% | u误差%% | p误差%% | 最大场%% |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in records:
        lines.append("| %s | %s | %d | %.4f | %.4f | %.4f | %.4f |" % (
            r['family'], r['final_time'], r['steps'],
            r['errors']['density']*100, r['errors']['velocity_x']*100,
            r['errors']['pressure']*100, r['max_percent']))

    lines.append("")
    lines.append("**守恒与熵**：")
    lines.append("| 配置 | 质量漂移 | x动量漂移 | 能量漂移 | 熵漂移 |")
    lines.append("|---|---|---|---|---|")
    for r in records:
        d = r["invariant_drift"]
        lines.append("| %s | %.3e | %.3e | %.3e | %.3e |" % (
            r['task_key'], d['mass'], d['momentum_x'], d['energy'], r['entropy_drift']))

    lines.append("")
    lines.append("## ④ 结果图\n")
    lines.append("### 图1：误差随时间演化")
    lines.append("![Error vs Time](case_euler_vortex_errconserve/fig_error_vs_time.png)")
    lines.append("*等熵涡长时输送：误差是否累积？红色虚线 = 6%% 容差*\n")
    lines.append("### 图2：不变量随时间演化")
    lines.append("![Invariants vs Time](case_euler_vortex_errconserve/fig_invariants_vs_time.png)")
    lines.append("*质量/动量/能量/熵的离散守恒性*\n")

    lines.append("## ⑤ 注意点\n")
    lines.append("1. **C2 失败机制**：DG-P2/32² 在 t=1→5 的误差从 0.0127%% 跳变到 14.43%%，最小二乘指数 α=4.377 >> 1.0，强烈提示长时积分中的色散聚焦或界面放大机制。P0 族 α=0.367 表现正常。")
    lines.append("2. **C1x 信息项**：长时检查点原始误差（最高 14.43%%）仅作透明展示，不参与门禁判定。")
    lines.append("3. **容差来源**：6%% 按算例池条目 verbatim 采用，Yee et al. (2000) 原文未入库复核。")
    lines.append("4. **计算资源**：每配置 1 核，DG-P2/32²@t=5 耗时 ~28s。\n")

    lines.append("## ⑥ 误差判定\n")
    lines.append("| 门禁 | 实测值 | 阈值 | 是否通过 | 是否<3%%（额外参考） |")
    lines.append("|---|---|---|---|---|")

    c1_pass = "✅ 通过" if c1["passed"] else "❌ 失败"
    c1_3pct = "✅" if c1["measured"] < 0.03 else "❌ %.2f%% > 3%%" % (c1["measured"]*100)
    c1_worst = c1.get("worst_at", "N/A")
    lines.append("| C1 快照容差 | %.3f%% (%s) | < 6%% | %s | %s |" % (
        c1["measured"]*100, c1_worst, c1_pass, c1_3pct))

    c2_pass = "✅ 通过" if c2["passed"] else "❌ **失败**"
    lines.append("| C2 误差守恒 | α = %.3f | ≤ 1.0 | %s | N/A（指数非百分比） |" % (
        c2["exponent"], c2_pass))

    c3_pass = "✅ 通过" if c3["passed"] else "❌ 失败"
    c3_worst = c3.get("worst_at", "N/A")
    lines.append("| C3 离散守恒 | %.3e (%s) | < 1e-10 | %s | ✅ 远优于 |" % (
        c3["worst_drift"], c3_worst, c3_pass))

    c4_pass = "✅ 通过" if c4["passed"] else "❌ 失败"
    lines.append("| C4 熵预算 | %.3e | < 1e-3 | %s | ✅ 远优于 |" % (
        c4["measured"], c4_pass))

    c5_pass = "✅ 通过" if c5["passed"] else "❌ 失败"
    c5_cells = c5.get("cells", [])
    lines.append("| C5 交叉校核阶 | %.3f (n=%s) | ≥ 0.8 | %s | N/A |" % (
        c5["order"], c5_cells, c5_pass))

    lines.append("")
    lines.append("**总判定：%s**（C2 误差守恒未通过）\n" % gates["verdict"])
    lines.append("**诚实声明**：设计族 DG-P2/32² 的误差随输送时间超线性放大，在 32² 网格上不满足「误差守恒」。两种读法（α≤0.5 严格目标与 α≤1.0 物理上界）同向失败，不构成 p-hacking。t=2→t=5 的 58 倍跳变需进一步研究（可能涉及 SSP-RK3 长时稳定性或模态投影初值的色散特性）。")

    md = "\n".join(lines)

    # Build report JSON
    fig1_b64 = b64(os.path.join(WORK, "case_euler_vortex_errconserve", "fig_error_vs_time.png"))
    fig2_b64 = b64(os.path.join(WORK, "case_euler_vortex_errconserve", "fig_invariants_vs_time.png"))

    report = {
        "title": "CASE-DG-EULER-VORTEX-ERR 交差审计核验：等熵涡移动误差守恒",
        "summary": "审计 m20486 三件套（run_artifacts/benchmark/repro_cmd）：12 个结果 JSON + metrics.json + 2 PNG 齐全；C1 快照 4.93%%<6%% 通过，C2 误差守恒 α=4.377>1.0 失败，C3/C4/C5 通过；总判定 FAIL。16 项测试全 PASS。成熟度 L4（完整基准+误差分析+可复现，物理结果 FAIL 不降低实现成熟度）。",
        "scenario_key": "CASE-DG-EULER-VORTEX-ERR",
        "mission_id": 20542,
        "tags": ["board-audit", "m20486", "TensorDG", "euler-vortex", "error-conservation", "L4"],
        "markdown": md,
        "figures": [
            {"name": "fig_error_vs_time", "caption": "等熵涡长时输送误差演化：DG-P2/32² 超线性放大 (α=4.377)，P0 族亚线性 (α=0.367)", "png_base64": fig1_b64},
            {"name": "fig_invariants_vs_time", "caption": "离散不变量（质量/动量/能量/熵）随时间演化：守恒量漂移在舍入量级", "png_base64": fig2_b64}
        ]
    }

    # Write report JSON
    with open("/tmp/report_m20486.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Report written to /tmp/report_m20486.json")
    print("Markdown length: %d chars" % len(md))
    print("Figure 1: %d base64 chars" % len(fig1_b64))
    print("Figure 2: %d base64 chars" % len(fig2_b64))

if __name__ == "__main__":
    main()
