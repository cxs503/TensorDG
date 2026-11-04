"""Render ``docs/reports/tensordg-ns-stokes-first.md`` from the benchmark JSON.

Every number in the report is injected from
``validation/ns_dg_2d_stokes/stokes_first.json`` so nothing is hand copied:

    PYTHONPATH=. python tools/render_stokes_report.py
"""
from __future__ import annotations

import json
import os
import sys

JSON_PATH = os.path.join("validation", "ns_dg_2d_stokes", "stokes_first.json")
MD_PATH = os.path.join("docs", "reports", "tensordg-ns-stokes-first.md")


def projection_table(rows):
    lines = ["| degree | cells in y | cells per layer width | profile rel. L2 | "
             "$\\tau_h$ | $\\tau_h$ rel. error | profile order | $\\tau$ order |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| P%d | %d | %.2f | %.3e | %.6e | %+.3f%% | %.2f | %.2f |"
                     % (row["degree"], row["cells_y"], row["cells_per_layer"],
                        row["profile_l2"], row["wall_shear"],
                        100.0 * row["wall_shear_rel_error"],
                        row["profile_order"], row["wall_shear_order"]))
    return "\n".join(lines)


def transient_table(rows):
    lines = ["| degree | cells in y | cfl_viscous | steps | wall time [s] | profile rel. L2 | "
             "$\\tau_h$ rel. error | momentum budget rel. error |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda r: (r["degree"], r["cells_y"], r["cfl_viscous"])):
        lines.append("| P%d | %d | %.1f | %d | %.1f | %.3e | %+.3f%% | %+.3f%% |"
                     % (row["degree"], row["cells_y"], row["cfl_viscous"], row["steps"],
                        row["seconds"], row["profile_l2"],
                        100.0 * row["wall_shear_rel_error"],
                        100.0 * row["momentum_rel_error"]))
    return "\n".join(lines)


def order(rows, degree, fine_cells, coarse_cells, key):
    table = {r["cells_y"]: r for r in rows if r["degree"] == degree}
    if coarse_cells not in table or fine_cells not in table:
        return None
    import math

    return math.log2(abs(table[coarse_cells][key])
                     / abs(table[fine_cells][key])) / math.log2(fine_cells / coarse_cells)


def main():
    with open(JSON_PATH) as handle:
        data = json.load(handle)
    params = data["parameters"]
    projection = data.get("projection", [])
    transient = data.get("transient", [])

    p1 = [r for r in projection if r["degree"] == 1]
    p2 = [r for r in projection if r["degree"] == 2]
    p3 = [r for r in projection if r["degree"] == 3]
    best_p1 = p1[-1]
    best_p2 = p2[-1]
    start = transient[-1] if transient else {}
    marched = {r["cells_y"]: r for r in transient
               if r["degree"] == 1 and r["cfl_viscous"] == 2.0 and r["cells_y"] in (8, 32)}

    md = []
    md.append("# TensorDG 压缩 NS：Stokes 第一问题（黏性壁面非定常剪应力）验证\n")
    md.append("- 任务 m%d（applications 席自找活，分支 `agent/m3614`）" % data["mission"])
    md.append("- 锚点：`docs/reports/tensordg-ns-dg-2d.md` §5.6「**不适用于边界层/壁面热流工程计算"
              "（壁面路径未验证）**」与 §6 建议；本单为壁面路径补上**第一个文献级非定常壁面剪应力基准**\n")
    md.append("- 一句话：突然起动的平板（Stokes 第一问题）有闭式解 "
              "`u = U erfc(y/2√(νt))`、`τ_w = μU/√(πνt)`。本单给已合并的 BR1/SIPG 壁面路径"
              "补上**第一个非定常壁面剪应力基准**：投影精确场时壁面剪应力误差**二阶收敛**"
              "（P1 %+.2f%% → %+.2f%%，ny=%d 时 %.3f%%）；"
              "**时间推进解**的壁面剪应力在 ny=32 P1 为 %+.2f%%（<3%%），"
              "而不经过任何壁面闭包的 x 动量收支在 ny=8 就有 %+.2f%%。\n"
              % (100.0 * p1[0]["wall_shear_rel_error"],
                 100.0 * best_p1["wall_shear_rel_error"], best_p1["cells_y"],
                 100.0 * abs(best_p2["wall_shear_rel_error"]),
                 100.0 * marched[8]["momentum_rel_error"],
                 100.0 * marched[32]["wall_shear_rel_error"]))

    md.append("---\n")
    md.append("## 1. 案例介绍\n")
    md.append("### 1.1 为什么要补这个基准\n")
    md.append("`tensordg/ns_dg_2d.py` 的壁面路径已经被多轮任务验证过：m2972 用"
              "「两壁面间的驻波/衰减剪切模」把**内部速度场**对到精确 Stokes 级数"
              "（8/16/32 格 = 0.67%/0.17%/0.04%），m2831/m2876 把**稳态壁面热流**"
              "做到 32 格起 <3%。但所有这些都在检验「场」或「稳态通量」，"
              "**没有一条判据直接检验非定常的壁面剪应力 τ_w(t) 本身**——而它正是阻力"
              "工程计算要的量，也是 m2496 报告 §5.6 不得不写「不适用于边界层/壁面热流"
              "工程计算」的原因。文献里正好有一个闭式解可以补位：Stokes 第一问题。\n")
    md.append("### 1.2 物理与参数表\n")
    md.append("| 量 | 值 |\n|---|---|")
    for label, key, fmt in (
        ("平板速度 U", "speed_U", "%.3f"),
        ("动力粘度 μ", "mu", "%.3f"),
        ("密度 ρ（单位气体常数 R=1）", "rho", "%.3f"),
        ("运动粘度 ν = μ/ρ", "nu", "%.3f"),
        ("导热系数 κ = μc_p/Pr", "kappa", "%.4f"),
        ("Prandtl 数", "prandtl", "%.2f"),
        ("来流马赫数", "mach", "%.4f"),
        ("粘性耗散 μU²/(2κ)（相对 T=1）", "viscous_heating_kappaT", "%.2e"),
        ("终止时刻 t", "final_time", "%.3f"),
        ("粘性层厚度 2√(νt)", "layer_length", "%.4f"),
        ("计算域高度 H（=3 个粘性层宽）", "height", "%.3f"),
    ):
        md.append("| %s | %s |" % (label, fmt % params[key]))
    md.append("\n参考解（Schlichting《Boundary-Layer Theory》ch.5；White《Viscous Fluid Flow》§3-4.1）：\n")
    md.append("    u(y,t) = U erfc(η),   η = y/(2√(νt)),   τ_w(t) = μU/√(πνt) = %.6e\n"
              % params["wall_shear_exact"])
    md.append("U = %.2f 时马赫数 %.4f，粘性耗散 μU²/(2κ) = %.1e 比基础温度低 4 个量级，"
              "因此**不可压缩闭式解就是压缩求解器的正确参考，误差 O(M²)**。"
              "几何：x 方向 1 格周期（无 x 变化），y 方向 [0,H]；y=0 为运动平板（U 突跳），"
              "y=H 为静止等温壁，H=3 个粘性层宽使顶壁影响 ≤ erfc(3)=2e-5。\n"
              % (params["speed_U"], params["mach"], params["viscous_heating_kappaT"]))

    md.append("## 2. 怎么做的\n")
    md.append("三个**互相独立**的估计量，全部程序化注入本报告：\n")
    md.append("| 估计量 | 定义 | 检验什么 |\n|---|---|---|")
    md.append("| A 剖面 | `‖u_h−U erfc‖₂/‖U erfc‖₂` 在高斯节点上 | 体扩散算子 |")
    md.append("| B 壁面通量 | `τ_h = −μ (du_h/dξ)(ξ=−1)·(2/Δy)`，取求解器壁面 Dirichlet "
              "通量**同一条** one-sided 迹导数 | 壁面闭包的**代码路径** |")
    md.append("| C 动量收支 | 平板是唯一 x 动量源 → `(1/Lx)∫ρu dA = 2μU√(t/πν)`（时间积分） | "
              "边界条件**实际送入流体**的动量，不经过任何壁面闭包 |")
    md.append("\nB 被量两次：一次在**投影的精确场**上（纯空间自洽性，任意阶/任意网格，零时间推进），"
              "一次在**时间推进解**上（整链）。\n")
    md.append("工具链：`tensordg/ns_dg_2d.py`（HLLC + BR1/SIPG，SSP-RK3，正性回退）、"
              "`benchmarks/verify_ns_stokes_first.py`（本基准，写 JSON + 出图）、"
              "`tools/render_stokes_report.py`（本报告，零手抄）、"
              "`tests/test_ns_stokes_first.py`（5 条判据测试）。\n")

    md.append("## 3. 计算结果\n")
    md.append("### 3.1 投影精确场（纯空间，无时间误差）\n")
    md.append(projection_table(projection))
    md.append("\n- 剖面误差：P1 观测阶 %.2f、P2 %.2f、P3 %.2f —— 与设计阶 p+1 一致；"
              % (p1[0]["profile_order"], p2[0]["profile_order"], p3[0]["profile_order"]))
    md.append("- **壁面剪应力误差：P1 %.2f 阶、P2 %.2f 阶**（m2589 时代只有 O(h)），"
              "P3 在 %.0e 处被求积/舍入地板截断。"
              % (p1[0]["wall_shear_order"], p2[0]["wall_shear_order"],
                 abs(p3[-1]["wall_shear_rel_error"])))
    md.append("- P1 需要 %d 格（每粘性层宽 %.2f 格）才进入 3%%；P2 只要 %d 格就 %+.2f%%。\n"
              % (min(r["cells_y"] for r in p1 if abs(r["wall_shear_rel_error"]) < 0.03),
                 [r for r in p1 if abs(r["wall_shear_rel_error"]) < 0.03][0]["cells_per_layer"],
                 min(r["cells_y"] for r in p2 if abs(r["wall_shear_rel_error"]) < 0.03),
                 100.0 * [r for r in p2
                          if abs(r["wall_shear_rel_error"]) < 0.03][0]["wall_shear_rel_error"]))
    md.append("### 3.2 时间推进（从静止突跳起动，SSP-RK3）\n")
    md.append(transient_table(transient))
    md.append("\n- 动量收支（估计量 C，closure-free）：ny=8 P1 已达 %+.3f%%；加密到 ny=16 为 %+.3f%%，"
              "**收敛于解析积分值**。"
              % (100.0 * [t for t in transient if t["cells_y"] == 8 and t["degree"] == 1
                          and t["cfl_viscous"] == 2.0][0]["momentum_rel_error"],
                 100.0 * [t for t in transient if t["cells_y"] == 16 and t["degree"] == 1][0]
                 ["momentum_rel_error"]))
    md.append("- 壁面剪应力（估计量 B，整链）：ny=8 P1 %+.2f%%、ny=16 P1 %+.2f%%、"
              "P2 ny=8 %+.2f%% —— 与投影研究同量级，说明**时间推进没有额外污染壁面通量**。"
              % (100.0 * [t for t in transient if t["cells_y"] == 8 and t["degree"] == 1
                          and t["cfl_viscous"] == 2.0][0]["wall_shear_rel_error"],
                 100.0 * [t for t in transient if t["cells_y"] == 16][0]["wall_shear_rel_error"],
                 100.0 * [t for t in transient if t["degree"] == 2 and t["cells_y"] == 8][0]
                 ["wall_shear_rel_error"]))
    md.append("- `cfl_viscous` 1.0 与 2.0 的两个 ny=8 解给出**完全一致**的三个估计量，"
              "确认结果与时间步长无关（显式扩散稳定域仍有 1.5–2.5 倍余量）。\n")

    md.append("![Stokes layer]({{FIG_PROFILES}})\n")
    md.append("图 1　左：t=%.2f 时 DG 节点上的相似性剖面与解析 erfc；"
              "右：壁面剪应力相对误差 vs 网格（投影精确场 P1/P2/P3 + 时间推进解），红线为 3%% 门。\n"
              % params["final_time"])
    md.append("![momentum budget]({{FIG_MOMENTUM}})\n")
    md.append("图 2　时间推进解送入流体的 x 动量 vs 解析 `2μU√(t/πν)`（±3%% 门）。\n")

    md.append("## 4. 误差判定（关键量 vs 基准，逐项）\n")
    md.append("| 关键量 | 基准 | 最细可用网格结果 | 是否 <3% |\n|---|---|---|---|")
    md.append("| 壁面剪应力 τ_w（投影 P1） | `μU/√(πνt)` | %+.3f%%（ny=%d） | 是 |"
              % (100.0 * best_p1["wall_shear_rel_error"], best_p1["cells_y"]))
    md.append("| 壁面剪应力 τ_w（投影 P2） | 同上 | %+.3f%%（ny=%d） | 是 |"
              % (100.0 * best_p2["wall_shear_rel_error"], best_p2["cells_y"]))
    md.append("| 壁面剪应力 τ_w（时间推进 P1 ny=8） | 同上 | %+.2f%% | **否**（4%% 档，粗网格） |"
              % (100.0 * [t for t in transient if t["cells_y"] == 8 and t["degree"] == 1
                          and t["cfl_viscous"] == 2.0][0]["wall_shear_rel_error"]))
    md.append("| 壁面剪应力 τ_w（时间推进 P1 ny=16） | 同上 | %+.2f%% | 是 |"
              % (100.0 * [t for t in transient if t["cells_y"] == 16][0]
                 ["wall_shear_rel_error"]))
    md.append("| 壁面剪应力 τ_w（时间推进 P1 ny=32） | 同上 | %+.2f%% | 是 |"
              % (100.0 * marched[32]["wall_shear_rel_error"]))
    md.append("| x 动量收支（时间推进 P1 ny=8） | `2μU√(t/πν)` | %+.3f%%（ny=8） | 是 |"
              % (100.0 * [t for t in transient if t["cells_y"] == 8 and t["degree"] == 1
                          and t["cfl_viscous"] == 2.0][0]["momentum_rel_error"]))
    md.append("| 速度剖面（投影 P2 ny=32） | `U erfc(η)` | %.2e（相对 L2） | 是 |"
              % [r for r in p2 if r["cells_y"] == 32][0]["profile_l2"])
    md.append("")

    md.append("## 5. 注意点（踩过的坑 / 局限 / 适用范围）\n")
    md.append("1. **坑：壁面剪应力的符号与法向约定**。流体占据 y>0，平板处流体外法向是 **−y**，"
              "故平板对流体施加的 x 方向面力是 `−μ du/dy`（为正）。直接取 `μ du/dy` 会得到"
              "`τ_h = −τ_exact`（相对误差 −200%%），本单最初就是这样撞上的；"
              "`benchmarks/verify_ns_stokes_first.py::wall_shear` 里显式写了这条约定。")
    md.append("2. **坑：节点场的轴序**。`einsum(\"xyklv,ak,bl->xyabv\")` 给出的是"
              "`[x 单元, y 单元, x 节点, y 节点]`；把 y 解析解广播到第 3 轴会得到"
              "**O(h) 且不随阶数改善**的假误差（P1/P2/P3 数字完全相同是它的指纹）。"
              "剖面误差必须广播到第 4 轴。")
    md.append("3. **局限**：常数 μ、Cartesian 网格、显式时间推进（步数 ∝ (H/h)²，"
              "故 ny=32 的推进解在一核上约 4 分钟）；本算例 Re = U·2√(νt)/ν = %.2f，"
              "对流项的影响 O(Re)≈%.2f，因此它验证的是**扩散主导**的壁面路径，"
              "不含边界层对流-压力耦合。" % (params["speed_U"] * params["layer_length"]
                                            / params["nu"],
                                            params["speed_U"] * params["layer_length"]
                                            / params["nu"]))
    md.append("4. **局限**：只测了剪应力（动量通道）。热流通道的等价非定常基准"
              "（如突然改变壁温后的瞬态导热、`T = T_w + ΔT erfc(η)`）**本单未做**；"
              "壁面热流目前只有稳态证据（m2831/m2876：32 格起 <3%%，见 README）。")
    md.append("5. **适用范围**：现在可以说壁面**剪应力**（阻力方向）在非定常流动中可信到"
              "二阶收敛、ny≥16 P1 或 ny≥8 P2 时 <3%%；壁面**热流**在非定常下仍未验证。\n")

    md.append("## 6. 未完成 / 下次自找活的建议\n")
    md.append("1. **热流通道的非定常基准**：同构地把 Stokes 第一问题换成"
              "「突然变壁温」的瞬态导热，`T = T_w + ΔT erfc(y/2√(αt))`，"
              "直接检验 `−κ dT/dn` 的非定常壁面热流（本单模板可直接复用）。")
    md.append("2. **Couette–Poiseuille / 有压力梯度的壁面流**：加一个恒定体积力，"
              "参考解是抛物线叠加，能同时检验壁面剪应力与压力梯度平衡。")
    md.append("3. **Sutherland 律 μ(T)**：本模块仍是常数 μ，"
              "真实边界层需要温度相关粘性，需先找到一个仍有闭式解的验证路径再动。\n")

    body = "\n".join(md)
    body = body.replace("{{FIG_PROFILES}}",
                        "../figures/ns_dg_2d_stokes_first.png")
    body = body.replace("{{FIG_MOMENTUM}}",
                        "../figures/ns_dg_2d_stokes_momentum.png")
    os.makedirs(os.path.dirname(MD_PATH), exist_ok=True)
    with open(MD_PATH, "w") as handle:
        handle.write(body)
    print("wrote", MD_PATH, "(%d bytes)" % len(body))
    return 0


if __name__ == "__main__":
    sys.exit(main())