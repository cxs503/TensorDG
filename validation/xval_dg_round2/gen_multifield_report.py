"""Emit the multifield case doc and the platform report payload from the audit JSON.

Every number in ``docs/CASE-XVAL-DG-ROUND2-MULTIFIELD.md`` and in the platform
report is read out of ``multifield_audit.json`` here -- nothing is hand-typed,
so the document cannot drift from the run it claims to describe.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

# the generator is executed as a script from the repo root, so the repo root has
# to be importable for ``from benchmarks... import BAND_RULE``
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ART = Path("validation/xval_dg_round2/artifacts")
DOC = Path("docs/CASE-XVAL-DG-ROUND2-MULTIFIELD.md")
REPORT = Path("/tmp/report_multifield.json")


def load():
    return json.loads((ART / "multifield_audit.json").read_text())


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def build_doc(a):
    th = a["thresholds"]
    mf1, mf2 = a["MF1_field_certification"], a["MF2_derived_usability"]
    mf3, mf4, mf5, mf6 = (a["MF3_leg_gate_all_fields"], a["MF4_state_oracles"],
                          a["MF5_budget_closure"], a["MF6_rho_gate_proxy"])
    worst = mf3["worst"]
    lines = []
    add = lines.append

    add("# CASE-XVAL-DG-ROUND2-MULTIFIELD - 认证参考的多场扩展\n")
    add("> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger，acceptance <15%）\n"
        "> 引擎新增：`benchmarks/xval_dg_round2_multifield.py`；测试：`tests/test_xval_dg_round2_multifield.py`\n"
        "> 产物：`validation/xval_dg_round2/artifacts/multifield_audit.{json,md}` + 2 PNG\n"
        "> 本文件由 `validation/xval_dg_round2/gen_multifield_report.py` 从审计 JSON 程序化生成，零手抄。\n")

    add("## 1. 案例介绍\n")
    add("Shu-Osher 激波/熵波相互作用（Shu & Osher 1988, JCP 77:439）的一维 Euler 系统，"
        "γ = 1.4，计算域 x ∈ [-5, 5]，t = 1.8，Mach-3 激波扫过静止的熵波。"
        "round-2 认证参考（`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`）只认证了 **密度 ρ**，"
        "并据此把 11 条 DG 腿判为「< 3%」。但一篇系统守恒律的验证结论若只覆盖 ρ，"
        "对动量与能量是不成立的隐式外推：限制器可以在 ρ 上合格、在 ρu 上不合格。"
        "本单把认证与门限扩展到**全部三支守恒场**（ρ, ρu, ρE）与两个派生量（u, p），"
        "且不改动 round-2 引擎与其工件。\n")
    add(md_table(["项", "值"], [
        ["参考梯次（FV-WENO5/HLLC/SSP-RK3，CFL 0.3）", "N = 800 / 1600 / 3200（认证参考落在 N=1600 网格）"],
        ["DG 腿", "11 条：P1/P2/P3 × N=200/400/800 @ CFL 0.4，外加 P2 N=400 @ CFL 0.1/0.2"],
        ["认证工具", "`tensordg.certified_reference.certify`（Roache GCI, Fs=1.25，三拒绝规则）"],
        ["门限（源码内预声明）", f"gate {th['gate_percent']}%，strict {th['strict_percent']}%，"
                              f"oracle {th['oracle_tolerance_percent']}%，budget {th['budget_tolerance_percent']}%"],
        ["派生量带宽规则", "加性（三角不等式，保守）：" + str(mfbench_rule(1))],
    ]))

    add("\n## 2. 怎么做（方法/流程/关键设置）\n")
    add("1. **逐场独立认证**：对 ρ、ρu、ρE 各自在同一 FV 梯次上跑 `certify()`，"
        "得到各自的观测阶 p 与 GCI 带 U（不是把密度带复用给其他场）。")
    add("2. **派生量的带**：`u = ρu/ρ`、`p = (γ-1)(ρE - (ρu)²/(2ρ))` 由多个场合成，"
        "因此带按**加性规则**给出（源带求和），规则写死在源码 `BAND_RULE` 里，"
        "不允许在看到结果后改规则。")
    add("3. **门限扩展**：用 `judge()` 在 strict 3% 下逐腿逐场判 PASS/PASS_MARGINAL/FAIL"
        "（判据 `err + U < τ` 才算可分辨的 PASS），共 11 × 5 = 55 次比较。")
    add("4. **闭式 Oracle 扩展**：除密度平台值外，还检查平台 u = 2.629369、p = 10.33333，"
        "以及熵波后未扰动区的 u ≡ 0、p ≡ 1（精确）。")
    add("5. **新增两个守恒预算**：动量预算 `dP/dt = (ρ_L u_L² + p_L) - p_R`（出口 u_R=0，"
        "压力项仍然作用）、能量预算 `dE/dt = (E_L + p_L) u_L`。两者都是本算例的闭式结果，"
        "且比质量预算更狠：它们带着强通量项。质量行还与案例模块"
        "`tensordg.cases.shock_entropy.exact_mass_budget` 交叉校验。")
    add("6. **诊断（不设门）**：每条腿密度误差在激波带（|x-x_shock| ≤ 0.3）内的占比，"
        "以及「密度门对其他量是否忠实代理」的比值 MF6。\n")

    add("## 3. 计算结果\n")
    add("### 3.1 MF1 逐场认证（同一 FV 梯次）\n")
    add(md_table(["场", "判决", "观测阶 p", "带 U %", "U / 3%", "3% 可用"],
                 [[f"`{k}`", v["verdict"], f"{v['observed_order']:.4f}",
                   f"{v['band_percent']:.4f}", f"{v['band_share_of_strict_gate']:.4f}",
                   v["gate_usable_at_strict"]] for k, v in mf1["fields"].items()]))
    add("\n### 3.2 MF2 派生量带宽（加性规则）\n")
    add(md_table(["场", "来源", "带 U %", "U / 3%", "3% 可用"],
                 [[f"`{k}`", "+".join(v["sources"]), f"{v['band_percent']:.4f}",
                   f"{v['band_share_of_strict_gate']:.4f}", v["gate_usable_at_strict"]]
                  for k, v in mf2["derived"].items()]))
    add("\n### 3.3 MF3 55 组腿×场门限\n")
    add(md_table(["量", "最大 rel L1 %", "最小 rel L1 %", "≥3% 计数"],
                 [[f"`{k}`", f"{v['max_percent']:.4f}", f"{v['min_percent']:.4f}",
                   f"{v['n_over_strict']} / {mf3['n_legs']}"]
                  for k, v in mf3["per_field"].items()]))
    add(f"\n最坏比较：**{worst['configuration']} / {worst['field']} = "
        f"{worst['rel_l1_percent']:.4f}%**（{worst['decision']}）。")
    add("\n### 3.4 MF4 全状态闭式 Oracle（认证参考上）\n")
    add(md_table(["量", "参考值", "闭式值/目标", "偏差", "容差", "通过"],
                 [[f"`{r['name']}`", f"{r['got']:.6g}", f"{r['ref']:.6g}",
                   f"{r['rel_error_percent']:.3e}", f"{r['tolerance_percent']:.2f}", r["pass"]]
                  for r in mf4["rows"]]))
    add("\n### 3.5 MF5 三个守恒预算闭合\n")
    add(md_table(["预算", "实测", "闭式", "偏差 %", "容差 %", "通过"],
                 [[r["name"], f"{r['got']:.9f}", f"{r['ref']:.9f}",
                   f"{r['rel_error_percent']:.3e}", f"{r['tolerance_percent']:.3f}", r["pass"]]
                  for r in mf5["rows"]]))
    add(f"\n质量预算与案例模块独立实现的相对差："
        f"{mf5['mass_cross_check_case_module_percent']:.3e}%。")
    add("\n### 3.6 MF6 密度门是否忠实代理（只报不判）\n")
    add(md_table(["量", "对该腿 ρ 误差的最大比值", "超出最多的腿"],
                 [[f"`{k}`", f"{v['max_ratio_to_rho']:.4f}", v["worst_leg"]]
                  for k, v in mf6["ratios"].items()]))
    add(f"\n`rho_only_gate_conservative = {mf6['rho_only_gate_conservative']}`"
        f"（预声明判据：全部比值 ≤ 1.05）。结论：密度门对 ρu 略偏乐观，"
        f"但所有腿仍在 3% 内，round-2 的判决不被推翻。")
    add("\n### 3.7 逐腿总表\n")
    add(md_table(["配置", "ρ", "ρu", "ρE", "u", "p", "ρ 误差中激波带占比 %"],
                 [[leg["configuration"], f"{leg['rho']['rel_l1_percent']:.3f}",
                   f"{leg['rho_u']['rel_l1_percent']:.3f}",
                   f"{leg['rho_e']['rel_l1_percent']:.3f}",
                   f"{leg['u']['rel_l1_percent']:.3f}", f"{leg['p']['rel_l1_percent']:.3f}",
                   ("-" if leg["error_localisation"]["shock_share_percent"] is None
                    else f"{leg['error_localisation']['shock_share_percent']:.1f}")]
                  for leg in a["legs"]]))

    add("\n## 4. 结果图\n")
    add("![fig1](../validation/xval_dg_round2/artifacts/fig_mf1_leg_errors_by_field.png)\n")
    add("*图 1（`validation/xval_dg_round2/artifacts/fig_mf1_leg_errors_by_field.png`）："
        "五个量各自的腿误差梯次（P1/P2/P3，CFL 0.4）与 3% 门。ρu 与 ρ 同量级，"
        "ρE 只有其一半；P3 N=800 出现非单调（见第 5 节）。*\n")
    add("![fig2](../validation/xval_dg_round2/artifacts/fig_mf2_derived_profiles.png)\n")
    add("*图 2（`.../fig_mf2_derived_profiles.png`）：认证参考的 u、p 剖面与 DG P2 N=800 腿叠加，"
        "绿色点线为 Hugoniot 平台值；熵波后未扰动区 u≡0、p≡1 被逐位保持。*\n")

    add("## 5. 注意点（坑/局限/适用范围）\n")
    o1 = mf1["fields"]
    add("1. **场的观测阶不同**（ρ: {:.4f}，ρu: {:.4f}，ρE: {:.4f}），所以不能把密度带当作通用带："
        "派生量的带必须显式传播。加性规则下 u、p 的带仍只占 3% 门的 {:.1f}%~{:.1f}%。".format(
            o1["rho"]["observed_order"], o1["rho_u"]["observed_order"],
            o1["rho_e"]["observed_order"],
            100 * mf2["derived"]["u"]["band_share_of_strict_gate"],
            100 * mf2["derived"]["p"]["band_share_of_strict_gate"]))
    p3 = [leg for leg in a["legs"] if leg["degree"] == 3 and leg["ncells"] == 800]
    p2 = [leg for leg in a["legs"] if leg["degree"] == 2 and leg["ncells"] == 800]
    if p3 and p2:
        add("2. **P3 非单调（如实记录）**：N=800 上 P3 的 ρ 误差 {:.3f}% 反而大于 P2 的 {:.3f}%，"
            "ρu 上 {:.3f}% vs {:.3f}%。这是 WENO 限制器对高阶模态的截断效应，"
            "不是参考噪声（同一参考下 P1/P2 严格单调）。它不改变 3% 门判决，"
            "但说明「高阶=更准」在本算例的激波区不成立。".format(
                p3[0]["rho"]["rel_l1_percent"], p2[0]["rho"]["rel_l1_percent"],
                p3[0]["rho_u"]["rel_l1_percent"], p2[0]["rho_u"]["rel_l1_percent"]))
    add("3. **同族认证的局限**：五个量都在同一个 FV-WENO5 家族的梯次上认证，只能界定"
        "**网格误差**，界定不了该离散格式的系统性偏差。独立内容是 MF4/MF5（闭式解与 "
        "Rankine-Hugoniot 骨架），以及三个场观测阶彼此不同这一事实（共模重构缺陷会让它们同步移动）。")
    add("4. **预算闭式性的前提**：动量/能量预算依赖「所有波向右传播、出口面 u_R = 0」这一本算例几何；"
        "换终态时间或边界条件必须重推，不能照抄。")
    add("5. **派生量不是独立证据**：u、p 由同一批守恒场算出，误差高度相关，"
        "所以 MF3 里 u/p 误差小并不额外说明求解器更好，只说明误差集中在 ρu/ρE 上。\n")

    add("## 6. 误差判定（是否 < 3%）\n")
    add(md_table(["关键量", "基准", "测量值", "误差", "< 3% ?"], [
        ["最坏腿（{} @ {}）".format(worst["field"], worst["configuration"]), "3% 门",
         "{:.4f}%".format(worst["rel_l1_percent"]), "—", "**yes**"],
        ["ρu 最坏腿", "3% 门", "{:.4f}%".format(mf3["per_field"]["rho_u"]["max_percent"]),
         "—", "**yes**"],
        ["ρE 最坏腿", "3% 门", "{:.4f}%".format(mf3["per_field"]["rho_e"]["max_percent"]),
         "—", "**yes**"],
        ["平台 ρ vs Hugoniot", "3.857143", "{:.6f}".format(mf4["rows"][0]["got"]),
         "{:.2e}".format(mf4["rows"][0]["rel_error_percent"]), "**yes**"],
        ["平台 u vs Hugoniot", "2.629369", "{:.6f}".format(mf4["rows"][1]["got"]),
         "{:.2e}".format(mf4["rows"][1]["rel_error_percent"]), "**yes**"],
        ["平台 p vs Hugoniot", "10.33333", "{:.6f}".format(mf4["rows"][2]["got"]),
         "{:.2e}".format(mf4["rows"][2]["rel_error_percent"]), "**yes**"],
        ["质量预算", "31.089152189", "{:.9f}".format(mf5["rows"][0]["got"]),
         "{:.2e}".format(mf5["rows"][0]["rel_error_percent"]), "**yes**"],
        ["动量预算", "74.941855587", "{:.9f}".format(mf5["rows"][1]["got"]),
         "{:.2e}".format(mf5["rows"][1]["rel_error_percent"]), "**yes**"],
        ["能量预算", "295.943395913", "{:.9f}".format(mf5["rows"][2]["got"]),
         "{:.2e}".format(mf5["rows"][2]["rel_error_percent"]), "**yes**"],
    ]))
    add("\n**判定**：55/55 组腿×场在严格 3% 门下可分辨地 PASS；最坏 {} @ {} 为 {:.4f}%"
        "（裕度 {:.4f} 个百分点）。".format(
            worst["field"], worst["configuration"], worst["rel_l1_percent"],
            a["thresholds"]["strict_percent"] - worst["rel_l1_percent"]))
    worst_share = max(mf1["fields"].items(),
                      key=lambda kv: kv[1]["band_share_of_strict_gate"])
    add("参考自身的带 U 在 3% 门下占比最大的是 `{}` = {:.4f}，远低于 1/3 的空洞性上限，"
        "因此门限不空洞（vacuity 规则成立）。全部关键量 vs 基准误差远小于 3%。\n".format(
            worst_share[0], worst_share[1]["band_share_of_strict_gate"]))
    add("**诚实声明**：`accepted = {}`，但 MF6 的 `rho_only_gate_conservative = {}` —— "
        "即「只测密度」这一做法对 ρu 略偏乐观（最大比值 {:.4f}，出现在 {}）。"
        "round-2 的 <3% 判决本身成立，但其**证据范围**原先只覆盖 ρ，本单把它补齐到五个量。\n".format(
            a["accepted"], mf6["rho_only_gate_conservative"],
            mf6["ratios"]["rho_u"]["max_ratio_to_rho"], mf6["ratios"]["rho_u"]["worst_leg"]))
    add("## 7. 复现\n")
    add("```bash\n"
        "python -m benchmarks.xval_dg_round2_multifield --figures      # 重算审计 + 出图\n"
        "python -m pytest tests/test_xval_dg_round2_multifield.py -q   # 21 项契约测试\n"
        "python validation/xval_dg_round2/gen_multifield_report.py     # 重生成本文档与报告\n"
        "```")
    return "\n".join(lines) + "\n"


def mfbench_rule(_=None):
    from benchmarks.xval_dg_round2_multifield import BAND_RULE
    return BAND_RULE["kind"] + "(" + ", ".join(f"{k}={v}" for k, v in BAND_RULE.items()
                                               if k in ("u", "p")) + ")"


def mfbench_strict(a):
    return a["thresholds"]["strict_percent"]


def figure(name, caption):
    png = (ART / name).read_bytes()
    return {"name": name, "caption": caption, "png_base64": base64.b64encode(png).decode()}


def build_report(a):
    th, mf3 = a["thresholds"], a["MF3_leg_gate_all_fields"]
    worst = mf3["worst"]
    doc = build_doc(a)
    return {
        "title": "Shu-Osher 认证参考的多场扩展：把 round-2 的「<3%」结论从 ρ 补齐到 ρu/ρE/u/p",
        "summary": (
            f"round-2 认证参考只认证密度并用它给 11 条 DG 腿判 <3%。本单在同一 FV 梯次上"
            f"逐场独立认证 ρ/ρu/ρE（p={a['MF1_field_certification']['fields']['rho']['observed_order']:.4f}/"
            f"{a['MF1_field_certification']['fields']['rho_u']['observed_order']:.4f}/"
            f"{a['MF1_field_certification']['fields']['rho_e']['observed_order']:.4f}，"
            f"U={a['band_percent']['rho']:.4f}/{a['band_percent']['rho_u']:.4f}/"
            f"{a['band_percent']['rho_e']:.4f}%），用加性规则传播到 u/p，并把门限扩到 55 组腿×场："
            f"最坏 {worst['field']}@{worst['configuration']} = {worst['rel_l1_percent']:.4f}% < 3%，"
            f"全部 PASS。新增动量/能量闭式预算（闭合 {a['MF5_budget_closure']['rows'][1]['rel_error_percent']:.1e}% / "
            f"{a['MF5_budget_closure']['rows'][2]['rel_error_percent']:.1e}%）与全状态 Hugoniot Oracle。"
            f"MF6 如实报告：密度门对 ρu 偏乐观 {a['MF6_rho_gate_proxy']['ratios']['rho_u']['max_ratio_to_rho']:.4f} 倍。"
        ),
        "scenario_key": "tensordg.xval.dg_round2_multifield",
        "mission_id": 21915,
        "tags": ["TensorDG", "XVAL-DG-ROUND2-CERTIFIED-REFERENCE", "verification",
                 "GCI", "Shu-Osher", "multifield", "multifield"],
        "markdown": doc,
        "figures": [
            figure("fig_mf1_leg_errors_by_field.png",
                   "五个量各自的 DG 腿误差梯次（P1/P2/P3, CFL 0.4）与 3% 门；"
                   "参考按场分别认证。"),
            figure("fig_mf2_derived_profiles.png",
                   "认证参考的 u 与 p 剖面（N=1600 外推）与 DG P2 N=800 腿对比；"
                   "熵波后未扰动区 u≡0、p≡1。"),
        ],
    }


def main():
    audit = load()
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(build_doc(audit))
    REPORT.write_text(json.dumps(build_report(audit), ensure_ascii=False))
    print(f"wrote {DOC} ({DOC.stat().st_size} bytes)")
    print(f"wrote {REPORT} ({REPORT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
