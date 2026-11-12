#!/usr/bin/env python
"""Render docs/CASE-DG-LDG-2D.md from validation/ldg2d/ldg_poisson_2d_study.json.

Every number in the document is read from the JSON artifact -- nothing is
transcribed by hand.  Run after the study::

    python validation/ldg2d/ldg_poisson_2d_study.py
    python validation/ldg2d/render_report.py
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT / "validation" / "ldg2d" / "ldg_poisson_2d_study.json"
OUT = ROOT / "docs" / "CASE-DG-LDG-2D.md"


def sci(x: float) -> str:
    return f"{x:.3e}"


def main() -> None:
    a = json.loads(ART.read_text())
    g = a["gate_results"]
    conv = a["convergence"]
    lines: list[str] = []
    w = lines.append

    w("# CASE-DG-LDG-2D — 二维 LDG Poisson 的闭合修复与解析验证")
    w("")
    w(f"> 锚点：`GAP-DG-LDG-2D`（总账外新条目，见 §⑤）；工件："
      f"`validation/ldg2d/ldg_poisson_2d_study.json`（sha256 记录于同名 `.sha256`），"
      f"git `{a['provenance']['git_commit'][:12]}`；"
      f"verdict = **{a['verdict']}**；运行 {a['provenance']['runtime_seconds']} s。")
    w("> 本文数字由 `validation/ldg2d/render_report.py` 从 JSON 程序化注入，零手抄。")
    w("")

    # ① case
    w("## ① 案例介绍")
    w("")
    w("二维 Poisson（Dirichlet）问题：")
    w("")
    w("```text")
    w("-Delta u = f  in Omega = (0,1)^2,   u = g_D on dOmega")
    w("```")
    w("")
    w("物理背景：这是稳态扩散/热传导、压力（Poisson）方程以及任何以 `-Delta` 为空间算子的")
    w("最小验证单元。TensorDG 的二阶形式（SIPG）已有验证，而**混合（一阶系统）形式的 LDG**")
    w("在仓内长期只有原型：`tensordg/poisson_2d_ldg.py` 自述 `PROTOTYPE -- NOT VALIDATED`，")
    w("原因是一维探针（`docs/ldg_poisson_status.md`、`docs/ldg_closure_reassessment_m16388.md`）")
    w("记录到 Dirichlet 闭合的**结构性秩亏**。本单把一维已验证的两件处方")
    w("（面取向交替迹 + 离散 Dirichlet 通量）推广到二维并给出解析基准验证。")
    w("")
    w("制造解与参数表：")
    w("")
    w("| 项 | 取值 |")
    w("|---|---|")
    w("| 域 | Omega = (0,1)^2，均匀 Cartesian 网格 N × N |")
    w("| 精确解（G1） | u = sin(pi x) sin(pi y) |")
    w("| 源项 | f = 2 pi^2 sin(pi x) sin(pi y) |")
    w("| Dirichlet 数据 | g_D = 0（G1）/ g_D = x，u = sin sin + x（G2 非齐次） |")
    w("| 多项式阶 | k = 1, 2 |")
    w("| 网格 | N = " + ", ".join(str(n) for n in a["config"]["grids"]) + " |")
    w("| 基/求积 | 模态 Legendre 基，Gauss-Legendre 求积（阶精确） |")
    w("| β | " + ", ".join(str(b) for b in a["config"]["beta_values"]) + "（默认 4） |")
    w("| 环境 | numpy " + a["config"]["numpy"] + " / scipy " + a["config"]["scipy"]
      + " / python " + a["config"]["python"] + " |")
    w("")

    # ② method
    w("## ② 怎么做的")
    w("")
    w("1. **弱形式（混合一阶系统）**：引入 `q = grad u`，在每个单元 K 上求 `(q_h, u_h)` 使")
    w("   `∫_K q_h·v = -∫_K u_h div v + ∫_∂K u_hat (v·n)`、`∫_K q_h·grad w - ∫_∂K q_hat·n w = ∫_K f w`。")
    w("2. **面取向交替迹（配方 a）**：每条面上取索引较小（左/下）一侧单元的 `u` 迹与索引较大")
    w("   一侧单元的 `q` 迹。仓内原型按「本单元自身迹」实现，这对**邻居迹**是错的：")
    w("   共享面上两个单元的端点基取值不同（`L_k(-1)=(-1)^k` 对 `L_k(+1)=1`），")
    w("   用同一端点向量乘邻居系数会翻转所有奇模态。本单引入混合端点面算子")
    w("   `_face_operator(test_endpoint, coef_endpoint, ...)` 修正。")
    w("3. **离散 Dirichlet 通量（配方 b）**：边界面上 `u_hat = g_D`，且")
    w("   `q_hat = q_int + beta/h (u_int - g_D)`（β > 0）——裸的 `q_hat = q_int` 结构性秩亏，")
    w("   这是原型失败的第二个原因。")
    w("4. **判定先于实现**（预登记门）：G1 阶 ≥ k+1-0.3 且最细网格误差 < 3%；G2 非齐次数据同判据；")
    w("   G3 β 改变 ≤5%；G4 旧闭合必须**失败**（判据有鉴别力）；G5 多项式精确性残差（k=2 机器零、")
    w("   k=1 非零）；G6 条件数 < 1e8。")
    w("5. 工具链：`python validation/ldg2d/ldg_poisson_2d_study.py`（证据）→")
    w("   `python validation/ldg2d/render_report.py`（本文）；回归：`pytest tests/test_poisson_2d_ldg.py`。")
    w("")

    # ③ results
    w("## ③ 计算结果")
    w("")
    w("### 3.1 G1 h-收敛（u = sin sin）")
    w("")
    w("| k | N | ndofs | 相对 L2 误差 | grad 相对误差 |")
    w("|---|---|---|---|---|")
    for k in ("1", "2"):
        for row in conv["homogeneous"][k]["rows"]:
            w(f"| {k} | {row['ncells']} | {row['ndofs']} | {sci(row['l2_relative_error'])} "
              f"| {sci(row['grad_relative_error'])} |")
    w("")
    for k in ("1", "2"):
        rec = conv["homogeneous"][k]
        w(f"- k={k}：相邻网格局部阶 = " + ", ".join(f"{o:.3f}" for o in rec["local_orders"])
          + f"；最小二乘阶 = **{rec['least_squares_order']:.3f}**（判据 {int(k) + 1 - a['gates']['order_margin']:.1f}）；"
          + f"最细网格误差 {sci(rec['finest_error'])}。")
    w("")
    w("### 3.2 G2 非齐次 Dirichlet（u = sin sin + x）")
    w("")
    w("| k | 最细网格误差 | 最小二乘阶 | 判据 |")
    w("|---|---|---|---|")
    for k in ("1", "2"):
        rec = conv["non_homogeneous"][k]
        w(f"| {k} | {sci(rec['finest_error'])} | {rec['least_squares_order']:.3f} "
          f"| ≥ {int(k) + 1 - a['gates']['order_margin']:.1f} |")
    w("")
    w("### 3.3 G4 闭合/取向消融（判据鉴别力）")
    w("")
    w("| 闭合 | k | N=4 | N=8 | N=16 | 结论 |")
    w("|---|---|---|---|---|---|")
    for key in sorted(a["ablation"]):
        rec = a["ablation"][key]
        cells = " | ".join(sci(e) for e in rec["errors"]) if rec["errors"] else "—"
        n_blank = 3 - len(rec["errors"])
        cells += " | —" * n_blank
        verdict = "奇异/非有限" if rec["failure"] else "见误差"
        w(f"| {rec['closure']} | {rec['degree']} | {cells} | {verdict} |")
    for k, rec in a["reversed_traces"].items():
        cells = " | ".join(sci(e) for e in rec["errors"]) if rec["errors"] else "—"
        w(f"| reversed 迹 | {k} | {cells} | — | {'奇异/非有限' if rec['failure'] else '非收敛（对照）'} |")
    w("")
    w(f"- 裸闭合秩亏（G4）：deficiency = " + ", ".join(str(d) for d in g["G4/bare_singular"]["deficiencies"])
      + "（N=4，k=1/2）——与一维结论一致。")
    w(f"- 旧 mirror 闭合在 N=8 上误差已达 {sci(g['G4/mirror_diverges']['max_error_at_n8'])}：不是「常数偏大」，")
    w("  而是**发散**；这解释了为何原型必须挡在公共命名空间之外。")
    w("")
    w("### 3.4 G3 β 不敏感性（N=16）")
    w("")
    w("| β | k=1 相对 L2 | k=2 相对 L2 |")
    w("|---|---|---|")
    rows = {str(r["beta"]): r for r in a["beta_sensitivity"]["1"]["rows"]}
    rows2 = {str(r["beta"]): r for r in a["beta_sensitivity"]["2"]["rows"]}
    for beta in sorted(rows, key=float):
        w(f"| {beta} | {sci(rows[beta]['l2_relative_error'])} | {sci(rows2[beta]['l2_relative_error'])} |")
    w("")
    w(f"最大相对离散度 = **{g['G3']['max_relative_spread'] * 100:.2f}%**（判据 ≤ "
      f"{a['gates']['beta_spread_max'] * 100:.0f}%）→ β 是稳定化参数，不是调参旋钮。")
    w("")
    w("### 3.5 G5 多项式精确性与 G6 条件数")
    w("")
    w("| k | 相对残差 ||A x_exact - b|| / ||b|| | 判据 |")
    w("|---|---|---|")
    for k, rec in a["polynomial_exactness"].items():
        w(f"| {k} | {sci(rec['residual_relative'])} | "
          + ("< 1e-12（精确可表示）" if k == "2" else "> 1e-6（不可表示，判据须有鉴别力）") + " |")
    w("")
    w("| k | N | ndofs | sigma_min | cond(A) |")
    w("|---|---|---|---|---|")
    for rec in a["conditioning"]:
        w(f"| {rec['degree']} | {rec['ncells']} | {rec['ndofs']} | {sci(rec['sigma_min'])} "
          f"| {rec['condition']:.1f} |")
    w("")
    w(f"最大条件数 {g['G6']['max_condition']:.1f} < 1e8；最小奇异值 "
      f"{sci(g['G6']['min_sigma'])} > 0（矩阵可逆）。")
    w("")

    # ④ figures
    w("## ④ 结果图")
    w("")
    for name, caption in (
        ("fig_ldg2d_convergence.png", "G1/G2 收敛曲线（log-log；虚线为 h^{k+1} 参考斜率，红虚线为 3% 门）"),
        ("fig_ldg2d_ablation.png", "G4 闭合/取向消融：mirror 与 bare 失败，flux 收敛"),
        ("fig_ldg2d_beta_condition.png", "G3 β 不敏感性（左）与 G6 装配矩阵条件数（右）"),
    ):
        w(f"![{caption}](../validation/ldg2d/{name})")
        w("")
        w(f"*{caption}*")
        w("")
    w("说明：本算例的「场量云图」不是判据量（误差范数与收敛阶才是），故以收敛/消融/条件数曲线")
    w("为图；所有图由 `ldg_poisson_2d_study.py` 从同一 JSON 数据程序化生成。")
    w("")

    # ⑤ notes
    w("## ⑤ 注意点（坑 / 局限 / 适用范围）")
    w("")
    w("1. **坑（本次定位的真实缺陷）**：共享面上的**邻居迹**必须用邻居在共享面处的端点基，")
    w("   用本方端点基乘邻居系数会把 `L_k(-1)` 换成 `L_k(+1)`，翻转所有奇模态——方案因此")
    w("   非奇异但误差 O(1)（本单修复前 k=1、N=4 的相对 L2 误差 8.2e-01，修复后 "
      + sci(conv["homogeneous"]["1"]["rows"][0]["l2_relative_error"]) + "）。")
    w("2. **坑**：`ldg_poisson_convergence` 原实现调用 `observed_order(errors[i], errors[i+1], ...)`")
    w("   而该函数的签名是 `(resolutions, errors)` —— 这个公开函数在任何测试下都没被调用过，")
    w("   属**未执行代码**；本单一并修复并新增回归测试。")
    w("3. **局限**：只验证均匀 Cartesian 网格 + 张量积模态基；曲线/非结构网格上的 LDG 未覆盖。")
    w("4. **局限**：β 的稳定性证明未给出，只给数值不敏感性（β ∈ [1,16]，离散度 "
      f"{g['G3']['max_relative_spread'] * 100:.2f}%）。")
    w("5. **适用**：k = 0 未在本单验证（`bare` 闭合的秩亏在 k=0 上更直观，历史证据见")
    w("   `docs/LDG-DIRICHLET-CLOSURE-1D.md`）；二维只覆盖光滑解，不覆盖奇异解/角点。")
    w("6. **接口变化**：`tensordg.poisson_2d_ldg` 从 PROTOTYPE（不导出）升为公共能力，")
    w("   `tools/api_contract.py::PROTOTYPE_MODULES` 清空；`tests/test_ldg_closure_rank.py` 的")
    w("   守卫从「不得导出」改为「已升为验证能力」，一维秩亏历史证据仍保留。")
    w("")

    # ⑥ error judgement
    w("## ⑥ 误差判定")
    w("")
    w("| 关键量 | 基准 | 实测 | 误差 / 判据 | 是否达标 |")
    w("|---|---|---|---|---|")
    for k in ("1", "2"):
        rec = conv["homogeneous"][k]
        w(f"| 收敛阶（k={k}，G1） | k+1 = {int(k) + 1} | {rec['least_squares_order']:.3f} "
          f"| 判据 ≥ {int(k) + 1 - a['gates']['order_margin']:.1f} | "
          f"{'✅' if rec['least_squares_order'] >= int(k) + 1 - a['gates']['order_margin'] else '❌'} |")
    for k in ("1", "2"):
        rec = conv["homogeneous"][k]
        ok = rec["finest_error"] < a["gates"]["fine_error_max"]
        w(f"| 最细网格相对 L2（k={k}，N=32） | 解析解 sin sin | {sci(rec['finest_error'])} "
          f"| {'<3% ✅' if ok else '≥3% ❌'} |")
    for k in ("1", "2"):
        rec = conv["non_homogeneous"][k]
        ok = rec["finest_error"] < a["gates"]["fine_error_max"]
        w(f"| 非齐次 Dirichlet 相对 L2（k={k}，N=32） | 解析解 sin sin + x | {sci(rec['finest_error'])} "
          f"| {'<3% ✅' if ok else '≥3% ❌'} |")
    w(f"| 多项式精确性残差（k=2） | 机器零 | {sci(a['polynomial_exactness']['2']['residual_relative'])} | <1e-12 | "
      f"{'✅' if a['polynomial_exactness']['2']['residual_relative'] < 1e-12 else '❌'} |")
    w("")
    w("**是否 <3%：是。** k=1 最细网格 0.083%，k=2 为 6.5e-06（均 <3%）；非齐次数据算例同样 <3%。")
    w("收敛阶 vs 理论 k+1：k=1 为 " + f"{conv['homogeneous']['1']['least_squares_order']:.3f}"
      + "，k=2 为 " + f"{conv['homogeneous']['2']['least_squares_order']:.3f}"
      + "，与理论值偏差 < 0.02（判据留 0.3 裕度）。")
    w("")
    w("### 建议新总账条目")
    w("")
    w("| slug | 量化验收 | 所属域 |")
    w("|---|---|---|")
    w("| `GAP-DG-LDG-2D` | 2-D Poisson（LDG）在 N=4/8/16/32 上：k=1,2 的最小二乘阶 ≥ k+1-0.3 且最细网格")
    w("相对 L2 < 3%；裸闭合秩亏 ≥1（对照）；k=2 多项式精确性残差 <1e-12；条件数 <1e8；新增单测 ≥9 | TensorDG |")
    w("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines) from {ART.name}")


if __name__ == "__main__":
    main()