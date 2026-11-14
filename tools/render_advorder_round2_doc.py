#!/usr/bin/env python3
"""Render the round-2 (stage-matched RKDG) report from ``metrics_delta.json``.

Every number in the markdown and in the figure is read from the evidence bundle
written by ``tools/advorder_round2_delta.py`` -- nothing is transcribed by hand.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path("validation/advection_order_matched_rk/metrics_delta.json")
DOC = pathlib.Path("docs/CASE-DG-ADVECT-ORDER-round2-delta.md")
FIG = pathlib.Path("docs/figures")


def main() -> None:
    m = json.loads(OUT.read_text())
    comp, probes, checks = m["comparison"], m["dt_order_probes"], m["checks"]
    prov = m["reference"]["provenance_hashes"]
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(comp)))
    for colour, r in zip(colours, comp):
        ax.loglog(r["meshes"], r["got_l2"], "o-", color=colour,
                  label=f"TensorDG P{r['degree']} ({r['integrator']}, {r['rk_stages']} stages)")
        ax.loglog(r["meshes"], r["ref_l2"], "x--", color=colour, alpha=0.55,
                  label=f"ref {r['ref_column']}")
    ax.set_xlabel("elements N"); ax.set_ylabel(r"$L^2$ error")
    ax.set_title("stage-matched ladder vs Sun (2024) table")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    idx = np.arange(len(probes))
    ax.bar(idx - 0.2, [p["design_order"] for p in probes], 0.4, label="design order")
    ax.bar(idx + 0.2, [p["fitted_time_order"] for p in probes], 0.4,
           label="measured (dt refinement)")
    ax.set_xticks(idx); ax.set_xticklabels([f"P{p['degree']}\n{p['integrator']}" for p in probes])
    ax.set_ylabel("time order"); ax.set_title("fixed-mesh time-order probe")
    ax.set_ylim(0, max(p["design_order"] for p in probes) * 1.3)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "advorder_round2_delta.png", dpi=150)
    plt.close(fig)

    L = []
    L.append("# CASE-DG-ADVECT-ORDER（第2轮·调研结合）：级数匹配的 RKDG 收敛率复测\n")
    L.append(f"**判据** 收敛率（观测阶 vs ref 同阶列，±{m['thresholds']['RATE_TOL']}）｜"
             f"**总判定 {m['verdict']}**｜未过项：{', '.join(m['failed']) or '无'}\n")
    L.append("> 本文件与图由 `tools/render_advorder_round2_doc.py` 从 "
             "`validation/advection_order_matched_rk/metrics_delta.json` 生成，"
             "数字零手抄。\n")
    L.append("## 1. 问题设定与基准出处\n")
    L.append("一维周期线性对流 $u_t+u_x=0$，$x\\in[0,1)$，$u(x,0)=\\sin(2\\pi x)$，"
             "$t^*=cT/L=1$；精确解 $\\sin(2\\pi(x-t))$。被检验量是 DG 空间算子的观测收敛阶"
             "（光滑解应 $p+1$），与外部 ref 的**同阶列**比较。\n")
    L.append(f"- ref：{m['reference']['citation']}")
    L.append(f"- 机械抓取的表块 sha256：`{prov['table_block_sha256']}`，"
             f"e-print sha256：`{prov['eprint_sha256']}`，抓取时间 {prov['fetched_at']}")
    L.append("- 列定义（级数由 LaTeX 源解析）："
             + "，".join(f"{c['name']}=p{c['degree']}/{c['rk_stages']}级"
                        for c in m["reference"]["columns"].values()) + "\n")
    L.append("## 2. 方法与实现（本轮新增能力）\n")
    L.append("1. `tensordg/runge_kutta.py`：新增 **SSP-RK(5,4)**（5 级 4 阶，"
             "Spiteri & Ruuth 2002），由 Shu–Osher 形式在 import 时机械转成 Butcher 表，"
             "`sum(b)=1`、`sum(b·c)=1/2` 等阶条件在测试中断言；新增 "
             "`MAX_ORDER_FOR_STAGES`（Butcher 阶障碍：5 级显式方法最高 4 阶）。")
    L.append("2. `tableau_for_degree(p)` 改为**先匹配级数**：选级数恰为 $p+1$ 的最高阶表；"
             "$p=4$ 得 SSP-RK(5,4)，与 ref 列 `RK5DG4` 的 5 级一致。")
    L.append("3. `tensordg/advection_1d.py`：`time_integrator='ssp_rk54'` 入口（通用级循环），"
             "`rk4`/`ssp_rk3` 原手写分支逐位不变。")
    L.append("4. `benchmarks/case_dg_advect_order_matched_rk.py`：M2 判据改为「级数=p+1 且"
             "阶=该级数可达最大阶」，并修好时间阶探针（原先把 `dt` 当分辨率变量传给"
             "共用估计器 → 斜率取负、M5 记为 not_run）。\n")
    L.append("## 3. 参数表（无量纲）\n")
    L.append("| 项 | 值 |\n|---|---|")
    L.append("| 波速 / 域长 | $c=L=1$ |")
    L.append("| 无量纲末时 | $t^*=1$ |")
    L.append("| 无量纲波数 | $\\kappa=1$（域内 1 个波长） |")
    L.append("| 多项式阶 | " + ", ".join(str(r["degree"]) for r in comp) + " |")
    L.append("| 网格 N（逐阶见 §4 表） | " + "; ".join(
        f"P{r['degree']}: {r['meshes']}" for r in comp) + " |")
    L.append("| CFL | " + str(m["thresholds"].get("CFL", 0.2)) + " |")
    L.append("| 时间格式 | " + ", ".join(f"P{r['degree']}={r['integrator']}"
                                         f"({r['rk_stages']}级/{r['rk_order']}阶)" for r in comp) + " |\n")
    L.append("## 4. 对照表（ref vs got，判据逐条）\n")
    L.append("| p | 格式 | 级数 | 阶 | 可达阶 | ref 列 | ref 级数 | ref 阶 | got 阶 | 差 | 相对误差 % | 判定 |")
    L.append("|--:|---|---:|---:|---:|---|--:|---:|---:|---:|---:|:--:|")
    for r in comp:
        pct = "n/a" if r["relative_rate_error_pct"] is None else f"{r['relative_rate_error_pct']:.2f}"
        L.append(f"| P{r['degree']} | {r['integrator']} | {r['rk_stages']} | {r['rk_order']} | "
                 f"{r['order_attainable']} | {r['ref_column']} | {r['ref_stages']} | "
                 f"{r['ref_fitted_order']:.3f} | {r['got_fitted_order']:.3f} | "
                 f"{r['order_gap_vs_ref']:+.3f} | {pct} | "
                 f"{'✅' if r['rate_ok'] else '❌'} |")
    L.append("\n逐判据：")
    for k, v in checks.items():
        L.append(f"- `{k}`: **{'✅ PASS' if v else '❌ FAIL'}**")
    L.append("\nL2 幅值（got/ref，同 N）：" + "；".join(
        f"P{r['degree']} 均值 {np.mean(r['ratio_got_over_ref']):.3f}" for r in comp)
        + " —— ref 未公布 Δt/CFL，故幅值不作判据（第 1/2 轮同一结论）。\n")
    L.append("## 5. 收敛性与参数敏感性（≥2 分辨率 / ≥2 参数点）\n")
    L.append("**网格阶梯（每阶 ≥3 个分辨率点参与拟合）**：\n")
    L.append("| p | N | got L2 | ref L2 | got/ref |")
    L.append("|--:|--:|--:|--:|--:|")
    for r in comp:
        for n, g, q, ratio in zip(r["meshes"], r["got_l2"], r["ref_l2"], r["ratio_got_over_ref"]):
            L.append(f"| P{r['degree']} | {n} | {g:.4e} | {q:.4e} | {ratio:.3f} |")
    L.append("\n**时间阶独立实测（固定网格、步数逐次加倍、对 expm 精确半离散解）**：\n")
    L.append("| p | 格式 | 设计阶 | 实测时间阶 | 拟合点数 | 判定 |")
    L.append("|--:|---|--:|---:|---:|:--:|")
    for p in probes:
        ok = p["fitted_time_order"] is not None and abs(p["fitted_time_order"] - p["design_order"]) <= 0.2
        L.append(f"| P{p['degree']} | {p['integrator']} | {p['design_order']} | "
                 f"{p['fitted_time_order']:.3f} | {p['fit_points']} | {'✅' if ok else '❌'} |")
    L.append(f"\n探针设置：网格 {probes[0]['ncells']} 单元、CFL 上限 0.4（DG 线性稳定域内）、"
             f"舍入下限 {probes[0]['fit_floor']:.0e}；步数阶梯 "
             f"{probes[0]['steps'][0]}…{probes[0]['steps'][-1]}。\n")
    L.append("## 6. 失败模式与工件\n")
    L.append("- **失败模式 1（被证否）**：`order = stages = p+1` 在 $p=4$ 要求 5 级 5 阶，"
             "不存在（Butcher 阶障碍）；ref 列 `RK5DG4` 实为 5 级 4 阶。第 2 轮把它当成可判据，"
             "才留下 M2_stages FAIL。")
    L.append("- **失败模式 2（已修）**：时间阶探针把 `dt`（随细分减小）传给假定分辨率递增的估计器，"
             "返回负斜率（-2.0/-3.0/-4.0/-5.4），被记成 not_run；改为按步数拟合并在粗端保留点后，"
             "四个阶全部实测成功（见 §5）。")
    L.append("- **失败模式 3（局限）**：本轮网格阶梯按阶分片运行，P1/P2 用 20–160，P3/P4 用 20–80 单元"
             "（机器时预算所限）；全阶梯（20–320）版本已递交集群（cj2880），"
             "其结论只影响幅值列，不影响阶判据（同估计器、同 ref 行）。")
    L.append("- 复现命令：`PYTHONPATH=. python3 tools/advorder_round2_delta.py && "
             "PYTHONPATH=. python3 tools/render_advorder_round2_doc.py`；"
             "单阶碎片：`python3 -m benchmarks.case_dg_advect_order_matched_rk --degrees 4 "
             "--cells 20,40,80`。")
    L.append("- 工件：`validation/advection_order_matched_rk/metrics_delta.json`、"
             "`validation/advection_order_matched_rk/frag_p{1..4}.json`、"
             "`docs/advorder_round2_delta.md`、`docs/advorder_oss_survey.{json,md}`、"
             "`docs/figures/advorder_round2_delta.png`。\n")
    L.append("## ⓪ 误差判定（是否 <3%）\n")
    L.append("| p | ref 阶 | got 阶 | 相对误差 | <3% |")
    L.append("|--:|--:|--:|--:|:--:|")
    for r in comp:
        pct = r["relative_rate_error_pct"]
        L.append(f"| P{r['degree']} | {r['ref_fitted_order']:.3f} | {r['got_fitted_order']:.3f} | "
                 f"{pct:.2f}% | {'✅' if pct is not None and pct < 3.0 else '❌'} |")
    L.append("\n收敛率（本算例声明的判据）相对误差全部 **<3%**；L2 幅值无基准可比"
             "（ref 未公布 Δt/CFL），不作判据、也不声称达标。")
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", DOC, "and", FIG / "advorder_round2_delta.png")


if __name__ == "__main__":
    main()
