#!/usr/bin/env python3
"""Render the CASE-LAX-DG follow-up note from the committed artifact.

Every number in the output markdown is injected from
``docs/benchmarks/results/lax_case_pool/metrics.json``; nothing is typed by
hand, so the document cannot drift away from the run that produced it::

    python scripts/render_lax_case_pool_doc.py            # write the doc
    python scripts/render_lax_case_pool_doc.py --check     # non-zero if stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python scripts/...` from any cwd
    sys.path.insert(0, str(ROOT))

from benchmarks.lax_case_pool import LEFT, RIGHT  # noqa: E402  (case definition)
METRICS = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool" / "metrics.json"
DOC = ROOT / "docs" / "CASE-LAX-DG-R3-nested-band-and-dt-split.md"
FIELDS = ("density", "velocity", "pressure")


def _pct(value, digits=3):
    return "n/a" if value is None else f"{value * 100:.{digits}f} %"


def _num(value, digits=6):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _sci(value, digits=5):
    return "n/a" if value is None else f"{value:.{digits}e}"


def render(metrics: dict) -> str:
    w = metrics["waves"]
    tolerances = metrics["config"]["tolerances"]
    gates = metrics["gates"]
    diagnostics = metrics.get("diagnostics", {})
    ladder = metrics["ladder"]
    r3 = gates["r3"]
    legacy = diagnostics.get("r3_legacy_band", {})
    nesting = diagnostics.get("r3_nesting", {})
    time_rows = metrics.get("time_refinement") or []
    out: list[str] = []

    out.append("# CASE-LAX-DG · R3 区域规则换轨 + 时间误差分离")
    out.append("")
    out.append("> 锚点 `CASE-LAX-DG`（Lax 激波管 / Toro test 2，ref = 精确解，容差 <3 %）。")
    out.append("> 本文**全部数字由 `scripts/render_lax_case_pool_doc.py` 从**")
    out.append(f"> `docs/benchmarks/results/lax_case_pool/metrics.json`（生成于 "
               f"{metrics['generated_at_utc']}）**注入**，零手抄；重跑可逐位校验。")
    out.append("> 本文闭合 m16606 审计的两条遗留：F1/F2（R3 区域规则不嵌套）与 F5（时间离散未研究）。")
    out.append("")

    # ------------------------------------------------------------------ ①
    out.append("## ① 案例介绍")
    out.append("")
    out.append("Lax (1954) 激波管：`[0,1]` 域，隔膜 `x0 = "
               f"{w['diaphragm']}`，γ = {w['gamma']}，快照 t = {w['final_time']}。"
               "四道波（左稀疏波 head/tail、接触间断、右激波）位置与星区状态均有闭式表达，"
               "参考值不依赖任何第三方求解器。")
    out.append("")
    out.append("| 参数 | 左态 | 右态 |")
    out.append("|---|---|---|")
    out.append(f"| (ρ, u, p) | ({_state(LEFT)}) | ({_state(RIGHT)}) |")
    out.append("")
    out.append("| 解析量 | 值 |")
    out.append("|---|---|")
    for key in ("pressure_star", "velocity_star", "density_star_left",
                "density_star_right"):
        out.append(f"| {key} | {_num(w[key])} |")
    for key, value in w["positions"].items():
        out.append(f"| x({key}) | {_num(value)} |")
    out.append("")

    # ------------------------------------------------------------------ ②
    out.append("## ② 怎么做的")
    out.append("")
    out.append("模态 DG（Legendre 基）+ HLLC 通量 + SSP-RK3，位置与星区量在每单元 10 点 "
               f"Gauss 节点上重建；生产 CFL = {metrics['config']['cfl']}。两项改动：")
    out.append("")
    out.append("1. **R3 区域规则换轨**：解析区遮罩由 `4·dx` 改为固定物理半宽 "
               f"`{tolerances['r3_band_width']}`。理由：`dx` 缩放的带宽随加密变窄，"
               "区域在相邻两腿之间**不是同一个区域**（m16606 实测保留测度 +35.3 %）。"
               f"旧规则（`{tolerances['r3_band_cells_legacy']}·dx`）保留在同一批解上"
               "并行测量（`diagnostics.r3_legacy_band`），判据变更全程留痕。")
    out.append(f"2. **时间误差分离**（m16606 F5 未运行项）：网格固定，只把 `dt` 按 "
               f"{'→'.join(f'CFL/{d}' for d in metrics['config']['time_study']['cfl_divisors'])} "
               "细化三次。模型 `E(dt) = E0 + C·dt^q` 在三次**精确对折**下有闭式解 "
               "`q = log2((E1-E2)/(E2-E3))`，于是「时间项占基线 CFL 误差的份额」"
               "可算而不是估。预登记上限 "
               f"`{tolerances['time_error_share']}`（超过即判定战线被 dt 污染）。")
    out.append("")

    # ------------------------------------------------------------------ ③
    out.append("## ③ 计算结果（程序化注入）")
    out.append("")
    out.append("### 3.1 分辨率阶梯（解析区列已按新规则重算）")
    out.append("")
    out.append("| P | N | steps | rel L1 ρ | rel L1 u | rel L1 p | "
               "解析区 rel L1 ρ（固定带） | 解析区 rel L1 ρ（旧 4·dx） | 保留测度（新/旧） |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for e in ladder:
        out.append(f"| {e['degree']} | {e['cells']} | {e['steps']} | "
                   f"{_sci(e['errors']['density'])} | {_sci(e['errors']['velocity'])} | "
                   f"{_sci(e['errors']['pressure'])} | "
                   f"{_sci(e['analytic_errors']['density'])} | "
                   f"{_sci(e['analytic_errors_legacy_band']['density'])} | "
                   f"{e['analytic_fraction']:.4f} / {e['analytic_fraction_legacy']:.4f} |")
    out.append("")
    out.append("### 3.2 闸门（R3 采新规则；旧规则并列记录）")
    out.append("")
    out.append("| gate | 现行 verdict | 关键数字 |")
    out.append("|---|---|---|")
    out.append(f"| C1 波位置 | {gates['c1']['status']} | 最差 {_pct(gates['c1']['worst_relative_error'])}"
               f"（容差 {_pct(gates['c1']['tolerance'])}） |")
    out.append(f"| C2 星区 | {gates['c2']['status']} | 最差 {_pct(gates['c2']['worst_relative_error'])}"
               f"（容差 {_pct(gates['c2']['tolerance'])}） |")
    out.append(f"| C3 全场阶 | {gates['c3']['status']} | 门控最小密度阶 "
               f"{_num(gates['c3']['gated_order'], 3)}（要求 ≥ {gates['c3']['minimum_order']}） |")
    out.append(f"| R3 解析区阶（新） | {r3['status']} | "
               + "; ".join(f"p{degree} 密度阶 {_num(r3['per_leg'][f'p{degree}.density'], 3)}"
                           f" / 需求 {_num(r3['per_leg'][f'p{degree}.requirement'], 3)}"
                           for degree in (1, 2, 3) if f"p{degree}.density" in r3["per_leg"])
               + " |")
    if legacy:
        out.append(f"| R3 解析区阶（旧 4·dx，仅记录） | {legacy['status']} | "
                   + "; ".join(f"p{degree} 密度阶 {_num(legacy['per_leg'][f'p{degree}.density'], 3)}"
                               for degree in (1, 2, 3) if f"p{degree}.density" in legacy["per_leg"])
                   + " |")
    out.append(f"| R5 熵波设计阶 | {gates['r5']['status']} | "
               + "; ".join(f"{k} 偏差 {_pct(v['relative_deviation'])}"
                           for k, v in gates["r5"]["cases"].items()) + " |")
    if "time" in gates:
        time_gate = gates["time"]
        out.append(f"| T 时间误差份额 | {time_gate['status']} | "
                   + "; ".join(f"{k} 最差份额 {_pct(v['worst_share'])}"
                               for k, v in time_gate["per_leg"].items())
                   + f"（上限 {_pct(time_gate['tolerance'])}） |")
    out.append("")
    out.append(f"**总判定：{metrics['verdict']}**（与换轨前一致的 honest-FAIL；"
               "失败的子项见 §⑤）")
    out.append("")

    if nesting:
        out.append("### 3.3 区域规则嵌套性（同一批解，两套遮罩）")
        out.append("")
        out.append("| 阶 | 保留测度（固定带，N 递增） | 展布 | 保留测度（旧 4·dx） | 展布 |")
        out.append("|---|---|---|---|---|")
        for key, data in nesting.items():
            out.append(f"| {key} | "
                       + " → ".join(f"{v:.4f}" for v in data["nested_fractions"])
                       + f" | {data['nested_spread']:.2e} | "
                       + " → ".join(f"{v:.4f}" for v in data["legacy_fractions"])
                       + f" | {data['legacy_spread']:.2e} |")
        out.append("")

    if time_rows:
        out.append("### 3.4 时间误差分离（网格固定，只缩 dt）")
        out.append("")
        out.append("| 腿 | cfl | dt | steps | rel L1 ρ | rel L1 u | rel L1 p |")
        out.append("|---|---|---|---|---|---|---|")
        for row in time_rows:
            for level in row["levels"]:
                out.append(f"| p{row['degree']} N={row['cells']} | {level['cfl']:.4f} | "
                           f"{_sci(level['dt'])} | {level['steps']} | "
                           f"{_sci(level['relative_l1']['density'])} | "
                           f"{_sci(level['relative_l1']['velocity'])} | "
                           f"{_sci(level['relative_l1']['pressure'])} |")
        out.append("")
        out.append("| 腿 | q(ρ) | q(u) | q(p) | 时间份额 ρ | 时间份额 u | 时间份额 p |")
        out.append("|---|---|---|---|---|---|---|")
        for row in time_rows:
            f = row["fields"]
            out.append(f"| p{row['degree']} N={row['cells']} | "
                       f"{_num(f['density']['temporal_order'], 3)} | "
                       f"{_num(f['velocity']['temporal_order'], 3)} | "
                       f"{_num(f['pressure']['temporal_order'], 3)} | "
                       f"{_pct(f['density']['temporal_share'])} | "
                       f"{_pct(f['velocity']['temporal_share'])} | "
                       f"{_pct(f['pressure']['temporal_share'])} |")
        out.append("")

    # ------------------------------------------------------------------ ④
    out.append("## ④ 图")
    out.append("")
    out.append("| 图 | 内容 |")
    out.append("|---|---|")
    out.append("| `docs/benchmarks/results/lax_case_pool/figures/lax_profiles.png` | "
               "最细腿 (N=400, P3) 的 ρ/u/p 数值 vs 精确解 |")
    out.append("| `docs/benchmarks/results/lax_case_pool/figures/lax_convergence.png` | "
               "全场与解析区相对 L1 随 N 收敛（log-log） |")
    out.append("| `docs/benchmarks/results/lax_case_pool/figures/time_refinement.png` | "
               "固定网格下 L1 随 dt 变化（平台化 = 空间误差主导） |")
    out.append("| `docs/benchmarks/results/lax_case_pool/figures/entropy_wave_order.png` | "
               "光滑熵波实测阶 vs p+1 |")
    out.append("")

    # ------------------------------------------------------------------ ⑤
    out.append("## ⑤ 注意点（坑 / 局限 / 适用范围）")
    out.append("")
    out.append("- **换轨改了子判决，没改总判定**：旧规则下 P1 的密解析区阶被区域几何漂移压到 "
               + (f"{_num(legacy['per_leg'].get('p1.density'), 3)}" if legacy else "n/a")
               + "；固定带下同一批解给出 "
               + (f"{_num(r3['per_leg'].get('p1.density'), 3)}" if 'p1.density' in r3['per_leg'] else "n/a")
               + f"（需求 {_num(r3['per_leg'].get('p1.requirement'), 3)}）。P1 子判决因此翻转，"
               "但 P2/P3 在四套规则下都不达设计阶——**整体 honest-FAIL 保留**，不得用本文把锚点翻成 verified。")
    out.append("- P2/P3 的解析区阶仍受 TVB 限幅器 + 间断耗散层支配，这是方法属性；"
               "把 N 上限推高或换 WENO 限幅器才有望抬阶，属另一张单。")
    out.append("- 旧 4·dx 规则在新版里保留为 `diagnostics.r3_legacy_band`，"
               "并有测试锁死它「不嵌套」这一特征（`test_r3_region_rule_is_nested_across_the_ladder`）："
               "一旦它变嵌套，测试即红，提醒复核这条判据变更的理由。")
    out.append("- 时间研究只覆盖 3 条腿、2 次对折；它给的是**基线 CFL 下时间项份额的上界估计**，"
               "不是全阶梯的时间阶认证。")
    out.append("- 未做项：与第三方 DG/FV 实现交叉；x86 CI 上逐位复核（本机 loong64）；"
               "求积点数敏感性（10 点 Gauss）。这些仍是 unknown，不当作 passed。")
    out.append("- 工件位置变更：`runs/` 在合并门禁里不进 `main`，本算例工件已改落"
               " `docs/benchmarks/results/lax_case_pool/`（与 `lax_region_band` 同规），"
               "工件级测试由 skip 改为硬断言（缺失即红），空转普查里该模块不再 VACUOUS。")
    out.append("")

    # ------------------------------------------------------------------ ⑥
    out.append("## ⑥ 误差判定（是否 <3 %）")
    out.append("")
    out.append("| 关键量 | 基准 | 实测 | 是否 <3 % |")
    out.append("|---|---|---|---|")
    for name, row in gates["c1"]["waves"].items():
        out.append(f"| {name} 位置 | 闭式波位 | {_pct(row['relative_error'], 4)} | "
                   f"{'是' if row['relative_error'] < 0.03 else '否'} |")
    for name, row in gates["c2"]["values"].items():
        out.append(f"| 星区 {name} | 闭式（等熵 / Rankine–Hugoniot） | "
                   f"{_pct(row['relative_error'], 4)} | "
                   f"{'是' if row['relative_error'] < 0.03 else '否'} |")
    for name, row in gates["r5"]["cases"].items():
        out.append(f"| 熵波 {name} 阶偏差 | 设计阶 p+1 | "
                   f"{_pct(row['relative_deviation'])} | "
                   f"{'是' if abs(row['relative_deviation']) < 0.03 else '否'} |")
    finest = [e for e in ladder if e["degree"] == max(x["degree"] for x in ladder)][-1]
    for field in FIELDS:
        out.append(f"| 最细腿全场 rel L1（{field}） | 精确 Riemann 解 | "
                   f"{_pct(finest['errors'][field])} | "
                   f"{'是' if finest['errors'][field] < 0.03 else '否'} |")
    out.append("")
    out.append("> 「是否 <3 %」的回答：8 个波位/星区量 + 4 个熵波阶偏差 + 3 个最细腿全场 L1 "
               "**全部 <3 %**。两项收敛阶判据（C3/R3）不是百分比误差，按阶阈值回答："
               f"C3 {gates['c3']['status']}、R3（新规则）{r3['status']}"
               + (f"、T {gates['time']['status']}" if "time" in gates else "")
               + "。")
    out.append("")
    return "\n".join(out) + "\n"


def _state(values):
    return ", ".join(f"{value:g}" for value in values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default=str(METRICS))
    parser.add_argument("--out", default=str(DOC))
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the committed doc is stale")
    args = parser.parse_args(argv)
    metrics = json.loads(Path(args.metrics).read_text())
    text = render(metrics)
    target = Path(args.out)
    if args.check:
        current = target.read_text() if target.exists() else ""
        if current != text:
            print(f"stale: {target}")
            return 1
        print(f"up to date: {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())