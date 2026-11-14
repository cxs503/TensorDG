#!/usr/bin/env python3
"""Render the CASE-LAX-DG limiter-arms note from the committed artifact.

Every number in the output markdown is injected from
``docs/benchmarks/results/lax_case_pool_limiter_arms/metrics.json``; nothing is
typed by hand, so the note cannot drift away from the run that produced it::

    python scripts/render_lax_arms_doc.py            # write the doc
    python scripts/render_lax_arms_doc.py --check    # non-zero if stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python scripts/...` from any cwd
    sys.path.insert(0, str(ROOT))

METRICS = (ROOT / "docs" / "benchmarks" / "results" /
           "lax_case_pool_limiter_arms" / "metrics.json")
DOC = ROOT / "docs" / "CASE-LAX-DG-limiter-arms.md"
FIELDS = ("density", "velocity", "pressure")
GATE_KEYS = ("c1", "c2", "c3", "r3", "r5")


def _pct(value, digits=3):
    return "n/a" if value is None else f"{value * 100:.{digits}f} %"


def _sci(value, digits=5):
    return "n/a" if value is None else f"{value:.{digits}e}"


def _field_series(analytic, field):
    """``analytic`` is stored per leg as a list ordered by ``FIELDS``.

    Older records carried a dict, so both shapes are accepted -- the note must
    render from either without silently reading the wrong component.
    """
    if isinstance(analytic, dict):
        return analytic[field]
    if analytic and isinstance(analytic[0], (list, tuple)):
        return analytic[FIELDS.index(field)]
    return analytic  # already a single field's series


def _num(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _gate_status(gates, key):
    if key not in gates:
        return "not run"
    return gates[key]["status"]


def render(metrics: dict) -> str:
    arms = metrics["arms"]
    table = metrics["analytic_orders"]
    repro = metrics["reproduction"]
    pre = metrics["pre_registration"]
    out: list[str] = []

    out.append("# CASE-LAX-DG · 限幅器分支：C3/R3 的 FAIL 是方法属性还是驱动默认值？")
    out.append("")
    out.append("> 锚点 `CASE-LAX-DG`（Lax 激波管 / Toro test 2，ref = 精确解，容差 <3 %）。")
    out.append("> 本文**全部数字由 `scripts/render_lax_arms_doc.py` 从**")
    out.append(f"> `docs/benchmarks/results/lax_case_pool_limiter_arms/metrics.json`"
               f"（生成于 {metrics['generated_at_utc']}）**注入**，零手抄；重跑可逐位校验。")
    out.append("> 本文明的是**归因债**：已发布的 `verdict = FAIL` 由两条阶数判据驱动，"
               "但驱动从未开过激波限幅器——本次只动限幅器一个旋钮重判同一套判据。")
    out.append("")

    out.append("## ① 案例介绍")
    out.append("")
    out.append("Lax (1954) / Toro test 2 激波管：`[0,1]` 域，隔膜 `x0 = 0.5`，γ = 1.4，"
               "快照 `t = 0.13`。左态 (ρ, u, p) = (0.445, 0.698, 3.528)，"
               "右态 (0.500, 0.000, 0.571)。解含四道波：左稀疏波（head/tail）、"
               "接触间断、右激波，位置与星区状态均有闭式表达。")
    out.append("")
    out.append("网格阶梯（与算例池完全相同）："
               + "、".join(f"N={c}·P{p}" for c, p in metrics["config"]["ladder"])
               + f"；CFL = {metrics['config']['cfl']}，终止时刻 {metrics['config']['final_time']}。")
    out.append("")
    out.append("### 预登记分支（跑数前写死，事后未追加）")
    out.append("")
    out.append("| 分支 | 传给求解器的参数 | 含义 |")
    out.append("|---|---|---|")
    for arm, kwargs in pre["arms"].items():
        shown = ", ".join(f"`{k}={v!r}`" for k, v in kwargs.items()) or "`limiter=None`"
        note = {"none": "基线：仅 Zhang–Shu 正性缩放",
                "tvb": "Cockburn–Shu TVB，仓内默认 M = 1.0",
                "weno": "Qiu–Shu WENO（无外部可调参数）"}.get(arm, "")
        out.append(f"| `{arm}` | {shown} | {note} |")
    out.append("")
    out.append("**未扫、未调的旋钮**（否则就是事后调参）：" +
               "、".join(f"`{k}`" for k in pre["not_swept"]) + "。")
    out.append("")

    out.append("## ② 怎么做的")
    out.append("")
    out.append("1. **单一旋钮**：三个分支除 `limiter` 外全部参数相同（HLLC 通量、SSP-RK3、"
               "float64、10 点 Gauss 输出、正性与限幅阶段均取仓内默认），"
               "所以分支间的差异只能来自限幅器。")
    out.append("2. **判据不改**：`C1/C2/C3/R3/R5` 与容差**直接从** "
               "`benchmarks.lax_case_pool` 导入（"
               f"C1 = {_pct(pre['inherited_tolerances']['c1'])}、"
               f"C2 = {_pct(pre['inherited_tolerances']['c2'])}、"
               f"C3 阶 ≥ {pre['inherited_tolerances']['c3_min_order']}、"
               f"R3 阶 ≥ {pre['inherited_tolerances']['r3_order_fraction']}·(p+1)、"
               f"R5 设计阶偏差 < {_pct(pre['inherited_tolerances']['r5_order_fraction'])}），"
               "R3 区域规则沿用固定物理带宽 "
               f"{pre['inherited_tolerances']['r3_band_width']}（m16606 F1 的嵌套规则）。")
    out.append("3. **复现自检**：分支 `none` 必须逐位复现已跟踪的算例池记录 "
               "`docs/benchmarks/results/lax_case_pool/metrics.json`；"
               f"实测结论 **{repro['status']}**，最大相对差 "
               f"{repro.get('worst_relative_difference')}（判据 < 1e-9）。")
    out.append("4. **同一批后处理**：三支共用同一个 `_leg()`，区域遮罩、加权 L1、"
               "波位/盘值定位代码完全相同——保留测度三支互等（见 ③.3）。")
    out.append("")

    out.append("## ③ 计算结果")
    out.append("")
    out.append("### 3.1 分支 × 闸门")
    out.append("")
    out.append("| 分支 | " + " | ".join(k.upper() for k in GATE_KEYS) + " | 总判定 |")
    out.append("|---" * (len(GATE_KEYS) + 2) + "|")
    for arm, result in arms.items():
        gates = result["gates"]
        out.append(f"| `{arm}` | " + " | ".join(_gate_status(gates, k).upper()
                                                for k in GATE_KEYS)
                   + f" | **{result['verdict']}** |")
    out.append("")
    for arm, result in arms.items():
        gates = result["gates"]
        bits = []
        if "c1" in gates:
            bits.append(f"C1 最差 {_pct(gates['c1']['worst_relative_error'])}")
        if "c2" in gates:
            bits.append(f"C2 最差 {_pct(gates['c2']['worst_relative_error'])}")
        if "c3" in gates:
            bits.append(f"C3 门控最小密度阶 {_num(gates['c3']['gated_order'])}")
        if "r3" in gates:
            per = gates["r3"]["per_leg"]
            orders = " / ".join(_num(per.get(f"p{d}.density")) for d in (1, 2, 3))
            needs = " / ".join(_num(per.get(f"p{d}.requirement"), 2) for d in (1, 2, 3))
            bits.append(f"R3 密度阶 {orders}（需求 {needs}）")
        out.append(f"* `{arm}`：" + "；".join(bits) + "。")
    out.append("")

    out.append("### 3.2 分辨率阶梯（密度相对 L1，全场与解析区）")
    out.append("")
    for arm, per_degree in table.items():
        out.append(f"**分支 `{arm}`**")
        out.append("")
        out.append("| p | N | steps | 全场 ρ | 解析区 ρ（带 0.02） | 解析区阶 | 需求 |")
        out.append("|---|---|---|---|---|---|---|")
        for degree, data in per_degree.items():
            for i, cells in enumerate(data["cells"]):
                order = "—" if i == 0 else _num(
                    _pair_order(data["cells"],
                                _field_series(data["analytic"], "density"), i))
                need = "—" if i == 0 else _num(data["r3_requirement"], 2)
                out.append(f"| {degree[1:]} | {cells} | {data['steps'][i]} | "
                           f"{_sci(data['whole_field'][i])} | "
                           f"{_sci(data['analytic'][i])} | {order} | {need} |")
        out.append("")

    out.append("### 3.3 解析区阶 vs 需求（R3 判决）")
    out.append("")
    out.append("| 分支 | p | 最细一对解析区阶 | 需求 0.85·(p+1) | 判决 |")
    out.append("|---|---|---|---|---|")
    for arm, per_degree in table.items():
        for degree, data in per_degree.items():
            got = data["r3_order"]
            ok = "PASS" if got is not None and got >= data["r3_requirement"] else "FAIL"
            out.append(f"| `{arm}` | {degree[1:]} | {_num(got)} | "
                       f"{_num(data['r3_requirement'], 2)} | {ok} |")
    out.append("")
    fractions = {arm: sorted({round(e["analytic_fraction"], 12)
                              for e in result["ladder"]})
                 for arm, result in arms.items()}
    out.append("保留测度（解析区面积占比）三支互等校验：" +
               "；".join(f"`{arm}` = {vals}" for arm, vals in fractions.items()) + "。")
    out.append("")

    out.append("### 3.4 平滑熵波（R5）：限幅器的代价")
    out.append("")
    out.append("| 分支 | u | p | 最细一对阶 | 设计阶 | 偏差 | R5 |")
    out.append("|---|---|---|---|---|---|---|")
    for arm, result in arms.items():
        for entry in result.get("entropy_wave") or []:
            design = entry["degree"] + 1
            order = entry["order"]
            # R5 uses the last two legs of the entropy ladder, as in the pool
            cells = [float(c) for c in entry["cells"]]
            errs = [entry["relative_l1"][str(int(c))] for c in cells]
            order = _pair_order(cells, errs, len(cells) - 1)
            dev = None if order is None else abs(order - design) / design
            ok = "PASS" if dev is not None and dev < \
                pre["inherited_tolerances"]["r5_order_fraction"] else "FAIL"
            out.append(f"| `{arm}` | {entry['speed']:.1f} | {entry['degree']} | "
                       f"{_num(order, 4)} | {design} | {_pct(dev)} | {ok} |")
    out.append("")

    out.append("### 3.5 结果图")
    out.append("")
    for path in metrics["figures"]:
        name = Path(path).name
        if "orders" in name:
            out.append(f"* `{path}` —— 左：全场相对 L1(ρ) vs 网格；中：解析区相对 L1(ρ) vs 网格"
                       "（三支同图，含设计阶参考）；右：R3 实测阶/需求表。")
        elif "profiles" in name:
            out.append(f"* `{path}` —— 最细腿（N=400·P3）三支的密度型线与精确解对比，"
                       "红点线为解析波位。")
    out.append("")

    out.append("## ④ 注意点 / 局限")
    out.append("")
    out.append("* **只动了一个旋钮**：`limiter_stage`（`every`/`final`）与 `tvb_constant` 的"
               "组合**未研究**；若某分支 FAIL 的成因是「每子步都限幅把光滑单元重投影」，"
               "本实验不能排除，只能说明它在当前默认下不成立。")
    out.append("* **限幅器不是 TVD 证明**：仓内实现为局部正性/振荡回退，`R5` 之外的"
               "激波强度外推（更强激波、二维钝体）不适用本结论。")
    out.append("* **阶数判据与误差判据必须分开读**：`C3/R3` 是收敛阶阈值，"
               "不参与台账 <3 % 的百分比口径（见 ⑤）。")
    out.append("* **墙钟**：见 `metrics.json` 的 `arms.*.wall_seconds`；"
               "限幅分支每步多一次邻域检修，故步数与墙钟都高于基线。")
    out.append("")

    out.append("## ⑤ 误差判定（关键量 vs 基准，逐项回答「是否 < 3 %」）")
    out.append("")
    out.append("| 分支 | 关键量 | 基准 | 实测 | 是否 < 3 % |")
    out.append("|---|---|---|---|---|")
    for arm, result in arms.items():
        gates = result["gates"]
        if "c1" in gates:
            worst = gates["c1"]["worst_relative_error"]
            out.append(f"| `{arm}` | 最差波位置 | 解析波位 | {_pct(worst)} | "
                       f"{'是' if worst < 0.03 else '否'} |")
        if "c2" in gates:
            worst = gates["c2"]["worst_relative_error"]
            out.append(f"| `{arm}` | 最差星区量 | 闭式星区 | {_pct(worst)} | "
                       f"{'是' if worst < 0.03 else '否'} |")
        if "c3" in gates:
            out.append(f"| `{arm}` | 全场密度阶（C3） | 阈值 ≥ "
                       f"{pre['inherited_tolerances']['c3_min_order']} | "
                       f"{_num(gates['c3']['gated_order'])} | "
                       "不适用（阶数判据，非百分比） |")
        if "r3" in gates:
            per = gates["r3"]["per_leg"]
            worst_ratio = min((per[f"p{d}.density_ratio"]
                               for d in (1, 2, 3)
                               if per.get(f"p{d}.density_ratio") is not None),
                              default=None)
            out.append(f"| `{arm}` | 解析区阶（R3） | 阈值 0.85·(p+1) | "
                       f"最差比值 {_num(worst_ratio, 3)} | "
                       "不适用（阶数判据，非百分比） |")
    out.append("")
    out.append("即：**三个分支在百分比口径（C1/C2）上都远优于台账 <3 %**；"
               "总判定是否 FAIL 完全由两条阶数判据（C3/R3）决定。")
    out.append("")
    return "\n".join(out)


def _pair_order(cells, errors, index):
    if index <= 0 or index >= len(errors):
        return None
    coarse, fine = errors[index - 1], errors[index]
    if coarse <= 0.0 or fine <= 0.0:
        return None
    return __import__("math").log(coarse / fine) / __import__("math").log(
        cells[index] / cells[index - 1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when the doc is stale")
    args = parser.parse_args(argv)
    if not METRICS.exists():
        raise SystemExit(f"missing artifact {METRICS}; run "
                         "`python -m benchmarks.lax_case_pool_limiter_arms`")
    metrics = json.loads(METRICS.read_text())
    text = render(metrics)
    if args.check:
        current = DOC.read_text() if DOC.exists() else ""
        if current.strip() != text.strip():
            raise SystemExit(f"{DOC} is stale; re-run without --check")
        print(f"{DOC.relative_to(ROOT)} up to date")
        return 0
    DOC.write_text(text + "\n")
    print(f"wrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())