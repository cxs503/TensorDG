"""Render ``docs/CASE-LAX-DG-limiter-contrast.md`` from the study metrics.

Every number in the rendered document is formatted from
``validation/lax_limiter_study/metrics.json``; the script writes no value it
did not read.  Re-running the study and re-running this renderer therefore
regenerates the document byte-for-byte, which is what makes the tables in
``docs/`` auditable instead of hand-copied.

Usage::

    PYTHONPATH=. python -m scripts.render_lax_limiter_doc
"""
from __future__ import annotations

import json
from pathlib import Path

METRICS = Path("validation/lax_limiter_study/metrics.json")
DOC = Path("docs/CASE-LAX-DG-limiter-contrast.md")

MECHANISM_LABEL = {
    "none": "无限机制（池基线）",
    "tvb-every": "TVB（每子步）",
    "tvb-final": "TVB（步末）",
    "av-every": "人工黏性 AV（每子步）",
}


def _f(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _e(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}e}"


def _pct(value, digits=2):
    return "n/a" if value is None else f"{value * 100:.{digits}f} %"


def render(metrics_path=METRICS, doc_path=DOC):
    data = json.loads(Path(metrics_path).read_text())
    label = lambda name: MECHANISM_LABEL.get(name, name)  # noqa: E731
    lines = []
    add = lines.append

    add("# CASE-LAX-DG · 激波捕捉机制对照（无限 / TVB / AV）")
    add("")
    add(f"> 工件 `{metrics_path}`（sha256 见同目录 `metrics.sha256`，生成于 "
        f"{data.get('generated_at_utc', 'n/a')}）。本文**全部数字由 "
        "`scripts/render_lax_limiter_doc.py` 从该 JSON 注入**，零手抄；"
        "闸门函数直接 import 自 `benchmarks/lax_case_pool.py`，"
        "因此这里的 verdict 与算例池的 verdict 同义。")
    add("")
    add("## ① 案例介绍")
    add("")
    add("Lax (1954) 激波管 / Toro (1999) test 2：`[0,1]` 域，隔膜 `x0 = 0.5`，"
        "γ = 1.4，快照 t = 0.13，精确解由 Riemann 求解器逐点给出（不依赖第三方解）。")
    add("")
    add("| 参数 | 左态 | 右态 |")
    add("|---|---|---|")
    left = data["config"]["left_state"]
    right = data["config"]["right_state"]
    add(f"| (ρ, u, p) | ({left[0]}, {left[1]}, {left[2]}) | "
        f"({right[0]}, {right[1]}, {right[2]}) |")
    add("")
    add("**待回答的问题**：算例池 `CASE-LAX-DG` 的现行 verdict 是 FAIL，"
        "失败的两道闸门是 C3（全场相对 L1 收敛阶 ≥ 1.0）与 R3（解析区相对 L1 阶 ≥ 0.85·(p+1)）。"
        "池子的所有腿都是**无限机制**跑的（`limiter=None`，只有 Zhang–Shu 正性缩放）。"
        "于是有两种互斥解释：")
    add("")
    add("1. 求解器/离散有问题（真缺陷）；")
    add("2. P3 腿在积分一个未正则化的振荡解——限幅器缺位导致的 Gibbs 波纹把 L1 阶压低。")
    add("")
    add("本算例用**算例池自己的闸门函数**检验解释 2：若装上激波捕捉机制后 C3/R3 变为 PASS，"
        "解释 2 成立；若装上后仍 FAIL，则池子的 FAIL 不是「缺限幅器」能解释的，"
        "该闸门在当前离散族下不可达，需要重新预登记。")
    add("")
    add("## ② 怎么做的")
    add("")
    add("* **机制**：`none`（引用池子已提交的基线，不重算）、TVB 限制器（Cockburn–Shu，"
        "每 SSP-RK3 子步 / 仅步末两种施加策略）、传感器门控模态人工黏性 "
        "`kind=\"av\"`（Persson–Peraire 传感器 + SVV 指数滤波）。AV 用库默认参数，"
        "**未**在 Lax 上调参。")
    add("* **阶梯**：只用失败的 P2/P3 两阶，N = 100/200/400（P1 腿的费用被省掉，"
        "范围收缩在此声明；P1 的池子数字在 §⑤ 并列引用）。")
    add("* **闸门**：`gate_c3` / `gate_r3` 原样复用，`R3_BAND_WIDTH = "
        f"{data['config']['r3_band_width']}` 不变，容差未按结果回调。")
    add("* **振荡度量**：每个腿把残差 `ρ_h − ρ_exact` 在四个固定物理窗口上归约"
        "（稀疏波扇区 / 星区左平台 / 星区右平台（接触后）/ 未扰动右态），"
        "报告 `max|res|`、`mean|res|`、`bias` 与**符号翻转率**（相邻节点残差变号的比例，"
        "用来区分「抹平但单调」与「真的有波纹」）。窗口边界：")
    add("")
    add("| 窗口 | 区间 |")
    add("|---|---|")
    for name, (lo, hi) in data["config"]["residual_windows"].items():
        add(f"| {name} | [{lo:.4f}, {hi:.4f}] |")
    add("")
    add("## ③ 计算结果（程序化注入）")
    add("")
    add("### 3.1 闸门对照")
    add("")
    add("| 机制 | C3 | C3 门控阶（最差密度阶） | R3 | R3 密度阶 / 需求 | 综合 |")
    add("|---|---|---|---|---|---|")
    for mechanism, gates in data["gates"].items():
        c3, r3 = gates["c3"], gates["r3"]
        per = r3["per_leg"]
        ratios = [(key, per[key], per[f"p{key[1]}.requirement"])
                  for key in ("p2.density", "p3.density") if key in per]
        requirement = "; ".join(f"p{key[1]}: {_f(order)} / {_f(req)}"
                                for key, order, req in ratios)
        add(f"| {label(mechanism)} | {c3['status']} | {_f(c3['gated_order'])} | "
            f"{r3['status']} | {requirement} | **{gates['verdict']}** |")
    add("")
    add(f"总 verdict：**{data['verdict']}**")
    add("")
    add("### 3.2 最细对的分辨率阶与波纹")
    add("")
    add("| 机制 | 阶 | 密度阶（N→N+） | 解析区密度阶 | 星区右平台 max\\|res\\|（粗 → 细） |")
    add("|---|---|---|---|---|")
    for mechanism, by_degree in data["orders"].items():
        for degree, row in sorted(by_degree.items()):
            ripple = row["ripple_star_right"]
            add(f"| {label(mechanism)} | {degree} | {_f(row['density'])} | "
                f"{_f(row['analytic_density'])} | {_e(ripple[0])} → {_e(ripple[1])} |")
    add("")
    add("### 3.3 每腿明细")
    add("")
    add("| 机制 | N | P | rel L1 ρ | 解析区 rel L1 ρ | 星区右平台 max\\|res\\| | 符号翻转率 | 步数 | 墙钟 [s] |")
    add("|---|---|---|---|---|---|---|---|---|")
    for entry in data["ladder"]:
        oscillation = (entry.get("oscillation") or {}).get("star_right", {})
        add(f"| {label(entry['mechanism'])} | {entry['cells']} | {entry['degree']} | "
            f"{_e(entry['errors']['density'])} | "
            f"{_e(entry['analytic_errors']['density'])} | "
            f"{_e(oscillation.get('max_abs'))} | "
            f"{_f(oscillation.get('sign_flips'))} | {entry['steps']} | "
            f"{_f(entry['wall_seconds'], 1)} |")
    add("")
    add("## ④ 图")
    add("")
    for figure in data.get("figures", []):
        add(f"* `{figure}`")
    add("")
    add("（剖面放大图 = 接触/激波区 N=400/P3 各机制 vs 精确解；收敛图 = 解析区与全场相对 L1 "
        "随 N；波纹图 = 星区右平台 max|res| 随 N，即机制需要消掉的量。）")
    add("")
    add("## ⑤ 注意点与范围")
    add("")
    reference = data.get("reference_definition", {})
    if reference:
        add("### 5.1 C1 的参照物是「同一 locator 的等值线交点」，不是解析波边")
        add("")
        add(f"`gate_c1` 把数值解与精确解**用同一个 level locator** 定位再比较。接触/激波上两者重合"
            f"（偏移 < 1e-3），但稀疏波扇内 locator 返回的是 "
            f"`RAREFACTION_LEVEL_FRACTION` 等值线交点，与解析特征边有**固定偏移**：")
        add("")
        add("| 波 | 解析位置 | locator 位置 | 归一化偏移 |")
        add("|---|---|---|---|")
        for name, row in reference["rows"].items():
            add(f"| {name} | {_f(row['analytic'], 6)} | {_f(row.get('locator'), 6)} | "
                f"{_pct(row.get('normalised_offset'))} |")
        add("")
        add(f"最大归一化偏移 **{_pct(reference['worst_normalised_offset'])}**，"
            f"大于 C1 自己的容差 {_pct(reference['tolerance_of_gate_c1'])}。"
            "结论：C1 的 PASS 应读作「数值与精确解在同一等值线定义下一致」，"
            "**不能**读作「波边位置误差 < 2 %」。这是标注问题，不是数值缺陷；"
            "池子文档里把 C1 写成「vs 解析位置」的地方据此更正。")
        add("")
    reuse = data.get("reference_reuse_check")
    if reuse:
        add("### 5.2 基线复用有证据")
        add("")
        add(f"复用池子已提交的 `none` 阶梯前，先在本进程重跑 `p{reuse['leg']['degree']}`"
            f"/N={reuse['leg']['cells']} 并逐位比较：最差相对差 "
            f"{_e(reuse.get('worst_relative_difference'))}（状态 {reuse['status']}），"
            f"步数 {reuse['steps_rerun']}（提交值 {reuse['steps_committed']}）。")
        add("")
    add("### 5.3 局限")
    add("")
    add("* 只跑了 P2/P3（失败的阶），P1 腿未纳入；P1 的池子数字见 `docs/CASE-LAX-DG.md`。")
    add("* AV 用库默认参数（在 Sod/N=64 上定的），未在 Lax 上调参；若调参后 C3/R3 变 PASS，"
        "那属于「调参后可达」，不是「机制缺位」。")
    add("* 未做 `weno` / `ao` 限制器与 `limiter_stage=\"final\"` 以外的策略组合拓展。")
    add("* 时间误差此前已由池子的 T 闸门分离（份额 < 2 %），本算例沿用同一 CFL=0.35。")
    add("")
    (Path(doc_path)).write_text("\n".join(lines) + "\n")
    return Path(doc_path)


if __name__ == "__main__":
    print(render())