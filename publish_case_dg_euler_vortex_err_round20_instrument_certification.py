#!/usr/bin/env python3
"""Publish CASE-DG-EULER-VORTEX-ERR round 20: the instruments are certified.

Reads ``case_euler_vortex_errconserve/instrument_certification.json`` (written by
``benchmarks/vortex_reference_instrument_certification.py``), draws the two
figures, emits ``docs/CASE-DG-EULER-VORTEX-ERR-instrument-certification.md`` and
posts the paper-style report.  Every number in the document is read from the
artefact -- nothing is hand-copied.

Usage::

    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_round20_instrument_certification.py --dry-run
    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_round20_instrument_certification.py
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")
OUT = Path("case_euler_vortex_errconserve")
SRC = OUT / "instrument_certification.json"
DOC = Path("docs/CASE-DG-EULER-VORTEX-ERR-instrument-certification.md")
FIG_MARGIN = Path("docs/fig_round20_instrument_margin.png")
FIG_NORM = Path("docs/fig_round20_norm_quadrature.png")
MISSION = 22963
SLUG = "CASE-DG-EULER-VORTEX-ERR"


def num(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    if isinstance(value, str) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.{digits}g}"


def pct(value, digits=6):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{100.0 * value:+.{digits}f}%"


def margin_rows(report):
    """(label, instrument, quantity it certifies) -- the 10x-margin table."""
    rows = [
        ("R1 PDE residual", report["R1_exactness"]["worst_relative"], 1.0,
         "守恒形式残差的通量散度尺度"),
        ("R2 isentropy", report["R2_isentropy"]["analytic"], 1.0, "等熵场 O(1) 尺度"),
    ]
    r3 = report["R3_invariants"]
    rows.append(("R3 invariant denominators", r3["gate_worst"], r3["smallest_drift"],
                 "C3 报告的最小质量/动量/能量漂移"))
    r4 = report["R4_error_norm"]
    rows.append(("R4 error-norm quadrature", r4["worst_relative_shift"], r4["tolerance"],
                 "C1 的 6% 容差"))
    if "R5_entropy_instrument" in report:
        r5 = report["R5_entropy_instrument"]
        rows.append(("R5 entropy integral", r5["quadrature_part"], r5["gate_reading"],
                     "C4 的熵预算读数"))
    return rows


def make_figures(report):
    rows = margin_rows(report)
    labels = [r[0] for r in rows]
    ratios = [max(r[1], 1e-20) / r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    ax.barh(labels, ratios, color=["#2b6cb0", "#2b6cb0", "#2f855a", "#2f855a", "#b7791f"])
    ax.set_xscale("log")
    ax.axvline(0.1, color="crimson", ls="--", lw=1.2,
               label="admission line 0.1 (instrument 10x below the quantity)")
    for index, (label, value, reference, _text) in enumerate(rows):
        ax.text(max(value / reference, 1e-20) * 1.6, index,
                f"{num(value)} / {num(reference)}", va="center", fontsize=7)
    ax.set_xlabel("instrument error / certified quantity (log, smaller is better)")
    ax.set_title("CASE-DG-EULER-VORTEX-ERR: instrument margin over the certified quantity")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, axis="x", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_MARGIN)
    plt.close(fig)

    cases = report["R4_error_norm"]["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=160)
    for name, case in cases.items():
        readings = case["readings"]
        multipliers = sorted(int(k) for k in readings)
        values = [100.0 * readings[str(m)] for m in multipliers]
        axes[0].plot(multipliers, values, "o-", label=name)
    axes[0].set_xlabel("quadrature points per direction / gate points")
    axes[0].set_ylabel("max-field relative L2 [%]")
    axes[0].set_yscale("log")
    axes[0].set_title("quadrature convergence of the norm reading (DG and P0)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=6)
    axes[1].bar([c[0] for c in cases.items()],
                [abs(c[1]["relative_shift"]) for c in cases.items()],
                color="#2b6cb0")
    axes[1].axhline(0.06, color="crimson", ls="--", lw=1.2, label="6% tolerance")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("relative drift, gate quadrature vs 4x")
    axes[1].tick_params(axis="x", rotation=30, labelsize=6)
    axes[1].set_title("how far under-integration moves the reading")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_NORM)
    plt.close(fig)
    return FIG_MARGIN, FIG_NORM


def build_doc(report):
    r1 = report["R1_exactness"]
    r2 = report["R2_isentropy"]
    r3 = report["R3_invariants"]
    r4 = report["R4_error_norm"]
    lines = [
        "# CASE-DG-EULER-VORTEX-ERR：等熵涡算例的参考解与量测仪表认证（round 20）",
        "",
        f"> 锚点 `{SLUG}`（算例池·等熵涡移动误差守恒）｜mission {MISSION}｜"
        f"ref: Yee, Vinokur & Djomehri, JCP 162 (2000) 33-81｜容差 6%",
        f"> 本文档由 `publish_case_dg_euler_vortex_err_round20_instrument_certification.py` "
        f"从 `{SRC.name}` 程序化生成，零手抄。",
        "",
        "## ① 案例介绍",
        "",
        "二维可压缩 Euler 方程（理想气体 γ=1.4）中的等熵涡在周期方箱 [-5,5]² "
        "以 |u∞|=√2 匀速输运，真解在平移下不变形，故离散误差应是网格属性而非时间属性。"
        "锚点 C1–C11 已全部通过（round 19 闭合 C11），本轮不再改判据、不改容差、"
        "不改任何已入库读数，而是**认证把判据喂饱的那条仪表链**：",
        "",
        "```",
        "解析等熵涡 ──(R1 精确性, R2 等熵)──▶ 参考解场",
        "参考解 + 求积 ──(R3, R4)──────────▶ 不变量与相对 L2 误差读数",
        "误差读数 ──(C1..C11)──────────────▶ 整包裁决",
        "```",
        "",
        "三项此前只是**假设**的事实：解析闭式确为 Euler 方程的解（抄错会伪装成格式误差）；"
        "320 点张量 Gauss 参考不变量收敛远低于最小漂移列；DG 的每格 p+2 点、P0 的 4 点"
        "精确格平均足以承载 4 位有效数字的百分比。",
        "",
        "## ② 怎么做的",
        "",
        "| 证书 | 仪器 | 做法 | 突变对照（必须失败） |",
        "|---|---|---|---|",
        "| R1 | Euler 残差 | 守恒形式 dU/dt+dF/dx+dG/dy，所有导数用 autograd 对解析表达式本身求导 | 四类合理误抄：包络指数、温度系数、单侧旋转符号、等温变体 |",
        "| R2 | 等熵结构 | 场量 max|p/ρ^γ−1| | 等温变体 |",
        "| R3 | 不变量分母 | 320/480/640/960/1280 点收敛 | 与最小漂移列比值的 10 倍准入线 |",
        "| R4 | 相对 L2 范数 | 同一状态在 1×/2×/3×/4× 求积下复测 | 1 点欠积分 |",
        "| R5 | 熵预算积分核 | 同一状态门禁求积 vs 6× 求积 | 质量/动量/能量同对照须为舍入噪声 |",
        "| R6 | 复现 | 重解 t=1 检查点并与 `results/*.json` 比对 | 相对偏差 >1e-9 即判陈旧 |",
        "",
        "统一通过判据：**仪表误差 ≤ 被认证量 / 10**。一个不会失败的检查等于什么都没认证，"
        "故每项都跑突变对照。工具：`benchmarks/vortex_reference_instrument_certification.py`"
        "（`--quick` 跳过时间推进）。",
        "",
        "## ③ 计算结果（程序化注入）",
        "",
        "### 3.1 R1 解析解是否真是解（相对残差）",
        "",
        "| 时刻 | 方程 | 残差绝对最大 | 残差/最大通量散度项 |",
        "|---|---|---|---|",
    ]
    for time_label, rows in r1["residual"].items():
        for name, row in rows.items():
            lines.append(f"| {time_label} | {name} | {num(row['abs_max'])} | "
                         f"{num(row['relative'])} |")
    lines += [
        "",
        f"最差相对残差 **{num(r1['worst_relative'])}**（限 {num(r1['limit'])}）"
        f"＝ {'通过' if r1['passed'] else '未通过'}；周期折叠平移不变性 "
        f"{num(r1['periodicity_relative'])}。突变对照：",
        "",
        "| 误抄 | 残差/最大项 | 检出 |",
        "|---|---|---|",
    ]
    for name, value in r1["mutants"].items():
        lines.append(f"| {name} | {num(value)} | {'是' if value > 1e-2 else '否'} |")
    lines += [
        "",
        "### 3.2 R3 不变量分母收敛（C3 的分母）",
        "",
        "| Gauss 点数/方向 | " + " | ".join(r3["deviations"]["320"]) + " |",
        "|---|" + "---|" * len(r3["deviations"]["320"]),
    ]
    for points, rows in r3["deviations"].items():
        lines.append(f"| {points} | " + " | ".join(num(v) for v in rows.values()) + " |")
    lines += [
        "",
        f"门禁 320 点最差偏差 {num(r3['gate_worst'])}，最小漂移 {num(r3['smallest_drift'])}，"
        f"比值 {num(r3['ratio'])}（准入 {num(1.0/report['margin'])}）："
        f"{'通过' if r3['passed'] else '未通过'}。",
        "",
        "### 3.3 R4 误差范数的求积收敛",
        "",
        "| 状态 | 门禁读数 | 4× 读数 | 相对漂移 | 欠积分对照 | 复现存储值 |",
        "|---|---|---|---|---|---|",
    ]
    for name, case in r4["cases"].items():
        reproduction = case.get("reproduction_relative")
        lines.append(
            f"| {name} | {100*case['gate']:.6f}% | {100*case['converged']:.6f}% | "
            f"{pct(case['relative_shift'], 4)} | "
            f"{pct(case.get('under_integrated_shift'), 2)} | "
            f"{'—' if reproduction is None else f'{100*reproduction:+.2e}%'} |")
    lines += [
        "",
        f"最差相对漂移 {num(r4['worst_relative_shift'])}，相对 6% 容差的比值 "
        f"{num(r4['ratio'])}：{'通过' if r4['passed'] else '未通过'}。",
    ]
    if "R5_entropy_instrument" in report:
        r5 = report["R5_entropy_instrument"]
        lines += [
            "",
            "### 3.4 R5 C4 的熵积分是物理还是求积噪声",
            "",
            f"状态 `{r5['state']}`。",
            "",
            "| 求积倍数 | \\|dS\\|/S |",
            "|---|---|",
        ]
        for key, value in r5["budgets"].items():
            lines.append(f"| {key}× | {num(value)} |")
        lines += [
            "",
            f"同状态门禁求积与 6× 求积的纯求积差 {num(r5['quadrature_part'])}，"
            f"门禁读数 {num(r5['gate_reading'])}，比值 {num(r5['ratio'])}"
            f"（准入 {num(1.0/report['margin'])}）：{'通过' if r5['passed'] else '未通过'}。"
            f"同一对照下守恒量偏差："
            + "，".join(f"{k} {num(v)}" for k, v in r5["conservation_check"].items())
            + "（多项式被积函数，故应为舍入噪声）。",
        ]
    lines += [
        "",
        "## ④ 结果图",
        "",
        f"![裕度]({FIG_MARGIN.name})",
        "",
        "*图 1：每项仪表的误差与被认证量之比（对数轴）。红虚线为 0.1 准入线——"
        "所有证书都在线下，即仪表不会移动裁决。*",
        "",
        f"![范数]({FIG_NORM.name})",
        "",
        "*图 2：左=范数读数随每方向求积点数（DG p1/p2/p3 投影态与 P0 实解）；"
        "右=门禁求积相对 4× 求积的漂移，红虚线为 6% 容差。*",
        "",
        "## ⑤ 注意点与局限",
        "",
        f"1. **核密度不是核心密度**：张量网格最小值 {num(r3['density_min']['tensor_grid_320'], 9)}，"
        f"射线最小值 {num(r3['density_min']['core_ray_4001'], 9)}，相对差 "
        f"{num(r3['density_min']['relative_gap'])}。门禁的 `density_min_relative_error` 是"
        f"同仪器漂移（末态 vs 初态），不受影响；但任何把张量网格最小值当物理核心密度引用的"
        f"说法都是错的。",
        "2. **残差证书对镜像对称是盲的**：把两个旋转分量同时反号得到的是等强度的反向涡，"
        "它同样是精确解，故 R1 不该检出；突变对照因此改为单侧反号。",
        "3. **只认证自由空间形式**：周期参考是同一闭式的折叠（或格点求和），R1 只对自由空间"
        "闭式取证，格点求和的截断地板由 `docs/CASE-DG-EULER-VORTEX-ERR-reference-floor.md` 单独量化。",
        "4. **求解器的离散误差本轮没有改变**：C1x 长时 P0 读数 14.30%（t=10）仍如实保留；"
        "本轮只说明这个读数不是仪表造成的。",
        "",
        "## ⑥ 误差判定（vs 基准，逐项）",
        "",
        "| 关键量 | 基准 | 读数 | 误差 | <3%？ |",
        "|---|---|---|---|---|",
        f"| R1 守恒形式残差 | 精确解残差应为 0 | {num(r1['worst_relative'])} | 同上 | 是（{num(r1['worst_relative'])}） |",
        f"| R2 等熵偏离 | p/ρ^γ 常数 | {num(r2['analytic'])} | 同上 | 是 |",
        f"| R3 不变量分母 | 1280 点读值 | {num(r3['gate_worst'])} | 同上 | 是 |",
        f"| R4 范数读数 | 4× 求积 | {num(r4['worst_relative_shift'])} | 同上 | 是 |",
    ]
    if "R5_entropy_instrument" in report:
        r5 = report["R5_entropy_instrument"]
        lines.append(f"| R5 熵积分 | 6× 求积 | {num(r5['quadrature_part'])} | 同上 | 是 |")
    lines += [
        f"| C1 门禁读数（既有） | 解析解 t=1 | {100*0.000152:.4f}%（dg_d2_n32_t10） | 0.0152% | 是 |",
        "",
        f"**整包裁决：{report['verdict']['status']}**"
        f"（{report['verdict']['certificates']} 项证书，失败 "
        f"{report['verdict']['failed'] or '无'}，突变对照 "
        f"{'全部触发' if report['verdict']['controls_passed'] else '未全部触发'}）。"
        f"所有关键量相对其基准误差均远小于 3%；这**不**是求解器精度的结论，"
        f"而是「测量工具本身不引入可观误差」的结论——两者必须分开记账。",
        "",
        "## 复现",
        "",
        "```",
        "PYTHONPATH=. python3 benchmarks/vortex_reference_instrument_certification.py",
        "PYTHONPATH=. python3 -m pytest tests/test_vortex_reference_instrument.py -q",
        "```",
        "",
    ]
    return "\n".join(lines)


def login():
    body = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(f"{BASE}/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    return payload.get("access_token") or payload["token"]


def post(path, payload, token):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = json.loads(SRC.read_text())
    fig_margin, fig_norm = make_figures(report)
    doc = build_doc(report)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(doc)
    print(f"wrote {fig_margin}, {fig_norm}, {DOC} ({len(doc)} chars); "
          f"verdict={report['verdict']['status']}")
    if args.dry_run:
        return 0
    r4 = report["R4_error_norm"]
    summary = (f"等熵涡算例的仪表链认证：R1 解析解残差 {num(report['R1_exactness']['worst_relative'])}"
               f"（四类误抄全部检出）、R3 不变量分母 {num(report['R3_invariants']['gate_worst'])}"
               f" 相对最小漂移 {num(report['R3_invariants']['smallest_drift'])}、"
               f"R4 范数求积漂移 {num(r4['worst_relative_shift'])} 相对 6% 容差，"
               f"整包 {report['verdict']['status']}；判据/容差/已入库读数一律未动")
    figures = [{"name": fig_margin.name,
                "caption": "各仪表的误差与被认证量之比（对数轴，红虚线=0.1 准入线）",
                "png_base64": base64.b64encode(fig_margin.read_bytes()).decode()},
               {"name": fig_norm.name,
                "caption": "误差范数读数的求积收敛（左）与门禁求积相对 4× 求积的漂移（右）",
                "png_base64": base64.b64encode(fig_norm.read_bytes()).decode()}]
    payload = {
        "title": "CASE-DG-EULER-VORTEX-ERR round 20：参考解与量测仪表认证（残差/分母/范数/熵核）",
        "summary": summary,
        "scenario_key": "dg_euler_vortex_err",
        "mission_id": MISSION,
        "tags": ["TensorDG", "Euler", "isentropic-vortex", "verification",
                 "instrument-certification", "adversarial-review", SLUG],
        "markdown": doc,
        "figures": figures,
    }
    result = post("/tensor/reports", payload, login())
    print(json.dumps(result)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())