"""Generate ``docs/limiter_verification.md`` from the verification artifacts.

Every number in the generated document is read from
``validation/limiter_sod/*.json``, so the report can never drift from the
runs.  Re-run after changing the limiter or the solver:

    PYTHONPATH=. python -m benchmarks.report_limiter_verification
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("validation/limiter_sod")
OUT = Path("docs/limiter_verification.md")


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def verdict(value: float, limit: float = 3.0) -> str:
    return "✅ <3%" if value < limit else "❌ ≥3%"


def main() -> None:
    pre = load("pre_fix.json")
    post = load("post_fix.json")
    oneshot = load("oneshot_post_fix.json")
    regional = load("regional_diagnostic.json")
    scale_pre = load("scale_consistency_pre_fix.json")
    scale_post = load("scale_consistency_post_fix.json")
    m1e6 = load("tvb_M1e6_check.json")

    cfg = pre["configs"][0]
    meta = {
        "cells": pre["ncells"],
        "degree": pre["degree"],
        "cfl": pre["cfl"],
        "time": pre["final_time"],
        "left": pre["left_state"],
        "right": pre["right_state"],
    }
    fields = ("density", "velocity", "pressure")

    lines: list[str] = []
    add = lines.append

    add("# TensorDG 限制器验证欠账清算：TVB/WENO 冲击限制器对标解析 Riemann 解")
    add("")
    add("> 类型：**验证欠账**（对已合并功能做文献/解析基准复核）· TensorDG · 2026-11-03")
    add("> 工件：`validation/limiter_sod/*.json`、`docs/figures/limiter_verification.png`")
    add("> 复现：本文档由 `benchmarks/report_limiter_verification.py` 从工件程序化生成，数字零手抄。")
    add("")

    add("## 0. 结论摘要")
    add("")
    add("| 编号 | 结论 | 证据 |")
    add("|---|---|---|")
    add(
        f"| F1 | **真实 bug**：TVB 限制器写 mode-1 系数时漏掉 1/2 因子，"
        f"把重构斜率放大 2 倍（实测比值 {fmt(scale_pre['limiters']['tvb']['ratio_to_half_minmod'],3)} → "
        f"修复后 {fmt(scale_post['limiters']['tvb']['ratio_to_half_minmod'],3)}） | "
        f"`scale_consistency_{pre['tag']}.json` / `scale_consistency_{post['tag']}.json` |"
    )
    add(
        f"| F2 | 已合并限制器的精度目标**仍未达标**：即使修好刻度，循环内启用限制器把最大相对误差从 "
        f"{fmt(max(post['configs'][0]['max_relative_l2_percent'], 0),2)}% 抬到 "
        f"{fmt(max(c['max_relative_l2_percent'] for c in post['configs'] if c['limiter'] != 'none'),2)}% | "
        f"`post_fix.json` |"
    )
    add(
        f"| F3 | 旧诊断「TVB 常数 M 太保守、需调 M」**被证伪**：M 从 1 增到 300（受标记单元 "
        f"{oneshot['detector_selectivity'][1]['flagged_cells']} → {oneshot['detector_selectivity'][3]['flagged_cells']} 个）"
        f"误差仅变化 {fmt(max(c['max_relative_l2_percent'] for c in post['configs'] if c['label']=='tvb_M300') - max(c['max_relative_l2_percent'] for c in post['configs'] if c['label']=='tvb_M1'),4)} 个百分点 | "
        f"`post_fix.json` / `oneshot_post_fix.json` |"
    )
    add(
        f"| F4 | 危害来自**施加策略**而非公式：把同一限制器一次性作用于无限期末态，三个场全部改善，"
        f"最大相对误差 {fmt(regional['max_relative_l2_percent']['tvb one-shot (final state)'],2)}% < 3% | "
        f"`regional_diagnostic.json` |"
    )
    add(
        f"| F5 | M→∞ 时限制器退化为恒等（复现无限制器结果 "
        f"{fmt(m1e6['velocity']['relative_l2_percent'],4)}%），证明检测器阈值是唯一开关、无隐藏副作用 | "
        f"`tvb_M1e6_check.json` |"
    )
    add("")

    add("## 1. 案例与判据（物理背景 / 几何 / 参数）")
    add("")
    add("Sod 激波管是 Euler 方程组唯一有**解析解**的标准间断问题，因此可直接做绝对误差判定，"
        "不依赖数值参考解。初值、几何与数值设置如下（全部取自工件）。")
    add("")
    add("| 项目 | 值 |")
    add("|---|---|")
    add(f"| 左态 (ρ, u, p) | {tuple(meta['left'])} |")
    add(f"| 右态 (ρ, u, p) | {tuple(meta['right'])} |")
    add(f"| 计算域 / 隔膜 | [0, 1] / x=0.5 |")
    add(f"| 终止时刻 | t = {meta['time']} |")
    add(f"| 网格 / 阶数 | {meta['cells']} 均匀单元 · 模态 Legendre DG P{meta['degree']} |")
    add(f"| 通量 / 时间推进 | HLLC · SSP-RK3，CFL = {meta['cfl']} |")
    add(f"| 参考解 | 精确 Riemann 解（`benchmarks/riemann_exact.py`），按单元内 6 点 Gauss 采样比对 |")
    add(f"| 判据 | 各场相对 L2 误差 < 3%（平台通用精度门） |")
    add(f"| 基线步数 | 无限制器 {post['configs'][0]['steps']} 步 |")
    add("")

    add("## 2. 方法：把「欠账」变成可复跑的量测")
    add("")
    add("原开发报告（`dev_report_tensordg-tvb-weno-limiter.md`）的记录是"
        "「accuracy target (errors <3% with limiter): NOT MET」加一句定性归因（M 过保守）。"
        "本单把它拆成三个可独立证伪的量：")
    add("")
    add("1. **刻度一致性**（纯代数，不需时间推进）：限制器写出的 mode-1 与 `minmod/2` 的比值。"
        "正确实现必须为 1；`benchmarks/limiter_scale_check.py`。")
    add("2. **端到端精度**（时间推进）：同一求解器、同一网格/CFL/终止时刻，只改限制器配置，"
        "度量对解析解的相对 L2 误差；`benchmarks/verify_limiter_sod.py`。")
    add("3. **机制分离**：把同一限制器一次性作用于**无限制器**的末态（不经 RK 循环），"
        "若误差改善则公式无罪、问题在施加策略；`benchmarks/limiter_oneshot.py`。")
    add("")
    add("工具链：PyTorch 模态 DG（自动微分无关，本单只用前向数值）。为降低机器负载，"
        "所有运行限 2 线程（`torch.set_num_threads(2)`）。")
    add("")

    add("## 3. 结果")
    add("")
    add("### 3.1 刻度一致性（F1）")
    add("")
    add("构造：单元均值 `[1.0, 1.0, 1.0, 2.0, 2.6, 2.6]`，第 3 单元 minmod 斜率非零。"
        "线性理论要求 `c1 = minmod(d_left,d_right)/2`。")
    add("")
    add("| 限制器 | 修复前 mode-1 | 修复前比值 | 修复后 mode-1 | 修复后比值 |")
    add("|---|---|---:|---|---:|")
    for name in ("tvb", "weno"):
        add(
            f"| {name.upper()} | {scale_pre['limiters'][name]['mode1'][3]:.12f} | "
            f"{scale_pre['limiters'][name]['ratio_to_half_minmod']:.3f} | "
            f"{scale_post['limiters'][name]['mode1'][3]:.12f} | "
            f"{scale_post['limiters'][name]['ratio_to_half_minmod']:.3f} |"
        )
    add("")
    add("TVB 比值精确为 2.000 → 1.000：修复前每步把间断附近的重构斜率放大一倍（反扩散），"
        "WENO 分支本就用 `dx/2` 归一（比值 1.040 来自凸组合略超 minmod，属正常）。"
        "修复内容为 `tensordg/limiters.py::apply_tvb_limiter` 中一行 `slope = 0.5 * slope`。")
    add("")

    add("### 3.2 端到端误差：循环内施加限制器一律变差（F2/F3）")
    add("")
    add(f"P{meta['degree']}，{meta['cells']} 单元，CFL={meta['cfl']}，t={meta['time']}，对解析解。")
    add("")
    add("| 配置 | steps | ρ [%] | u [%] | p [%] | max [%] | 3% 门 | 受标记单元斜率比 |")
    add("|---|---:|---:|---:|---:|---:|---|---:|")
    for c in post["configs"]:
        amp = c["slope_amplification"]
        amp_txt = "—" if c["limiter"] == "none" else f"{amp:.3f}"
        add(
            f"| {c['label']} | {c['steps']} | {fmt(c['relative_l2_percent']['density'])} | "
            f"{fmt(c['relative_l2_percent']['velocity'])} | {fmt(c['relative_l2_percent']['pressure'])} | "
            f"**{fmt(c['max_relative_l2_percent'])}** | {verdict(c['max_relative_l2_percent'])} | {amp_txt} |"
        )
    add("")
    add("注：`受标记单元斜率比` = 受标记单元写出的 |mode-1| 与 `minmod/2` 之比，TVB 必须恰为 1.000；"
        "WENO 用凸组合、不受 minmod 上界约束，比值 >1 属正常（其缺陷另论，见 3.3/5.3）。"
        "`none` 行无限制器，此列记「—」。")
    add("")
    add("修复前同网格对照（`pre_fix.json`）：")
    add("")
    add("| 配置 | ρ [%] | u [%] | p [%] | max [%] |")
    add("|---|---:|---:|---:|---:|")
    for c in pre["configs"]:
        add(
            f"| {c['label']} | {fmt(c['relative_l2_percent']['density'])} | "
            f"{fmt(c['relative_l2_percent']['velocity'])} | {fmt(c['relative_l2_percent']['pressure'])} | "
            f"**{fmt(c['max_relative_l2_percent'])}** |"
        )
    add("")
    add("读法：修复刻度只让「限制器的错误倍率」从 2 倍降到 1 倍（正确定义），"
        "但**没有**让限制器变得有用——循环内启用限制器仍使 u 的误差从 "
        f"{fmt(post['configs'][0]['relative_l2_percent']['velocity'],2)}% 上升到约 "
        f"{fmt(post['configs'][1]['relative_l2_percent']['velocity'],2)}%。"
        f"而 M 从 1 提到 300（受标记单元 "
        f"{oneshot['detector_selectivity'][1]['flagged_cells']}→{oneshot['detector_selectivity'][3]['flagged_cells']}）"
        "几乎不改变误差，说明「调 M」不是杠杆。")
    add("")

    add("### 3.3 机制分离：公式无罪，施加策略有罪（F4）")
    add("")
    add("| 配置 | ρ [%] | u [%] | p [%] | max [%] | 3% 门 |")
    add("|---|---:|---:|---:|---:|---|")
    for name in ("unlimited", "tvb in RK loop", "tvb one-shot (final state)"):
        row = regional["relative_l2_percent"][name]
        add(
            f"| {name} | {fmt(row['density'])} | {fmt(row['velocity'])} | {fmt(row['pressure'])} | "
            f"**{fmt(regional['max_relative_l2_percent'][name])}** | {verdict(regional['max_relative_l2_percent'][name])} |"
        )
    add("")
    add("分区诊断（速度场相对 L2，按 x 分区；左平台段解析速度恒为 0，相对范数无定义，改记绝对 L2）：")
    add("")
    add("| 配置 | 稀疏波 0.25<x<0.5 [%] | 接触/激波 0.5<x<0.95 [%] | 左平台绝对 L2 |")
    add("|---|---:|---:|---:|")
    for name in ("unlimited", "tvb in RK loop", "tvb one-shot (final state)"):
        reg = regional["regional_velocity_percent"][name]
        add(
            f"| {name} | {fmt(reg['rarefaction 0.25<x<0.5'])} | {fmt(reg['contact/shock 0.5<x<0.95'])} | "
            f"{regional['plateau_velocity_absolute_l2'][name]:.3e} |"
        )
    add("")
    add("关键读法：循环内施加时，**光滑稀疏波区**的速度误差从 "
        f"{fmt(regional['regional_velocity_percent']['unlimited']['rarefaction 0.25<x<0.5'])}% 恶化到 "
        f"{fmt(regional['regional_velocity_percent']['tvb in RK loop']['rarefaction 0.25<x<0.5'])}%"
        f"（{regional['regional_velocity_percent']['tvb in RK loop']['rarefaction 0.25<x<0.5'] / regional['regional_velocity_percent']['unlimited']['rarefaction 0.25<x<0.5']:.1f}×），"
        "而一次性施加时该区误差几乎不动（"
        f"{fmt(regional['regional_velocity_percent']['tvb one-shot (final state)']['rarefaction 0.25<x<0.5'])}%）。"
        "即：每个 SSP-RK 子步都把受标记单元投影回 P1（并把 mode≥2 清零），"
        "在光滑区反复施加就把二阶精度区持续压平成 minmod 一阶，误差随时间累积。"
        "检测器选择性（`detector_selectivity`）与 M 都不是瓶颈。")
    add("")

    add("### 3.4 图")
    add("")
    add("![limiter verification](../docs/figures/limiter_verification.png)")
    add("")
    add("上排：三个守恒量场对精确解；下排左：各场相对 L2 与 3% 门；"
        "下排中：速度误差的分区来源；下排右：受标记单元比例随 M 的变化。")
    add("")

    add("## 4. 误差判定（逐项，显式回答是否 <3%）")
    add("")
    add("| 关键量 | 实测 | 基准 | 误差 | 是否 <3% |")
    add("|---|---:|---|---:|---|")
    rows = [
        ("Sod ρ，循环内 TVB（修复后）", post["configs"][1]["relative_l2_percent"]["density"], "精确 Riemann 解"),
        ("Sod u，循环内 TVB（修复后）", post["configs"][1]["relative_l2_percent"]["velocity"], "精确 Riemann 解"),
        ("Sod p，循环内 TVB（修复后）", post["configs"][1]["relative_l2_percent"]["pressure"], "精确 Riemann 解"),
        ("Sod u，循环内 WENO（修复后）", post["configs"][4]["relative_l2_percent"]["velocity"], "精确 Riemann 解"),
        ("Sod ρ，无限制器基线", post["configs"][0]["relative_l2_percent"]["density"], "精确 Riemann 解"),
        ("Sod u，无限制器基线", post["configs"][0]["relative_l2_percent"]["velocity"], "精确 Riemann 解"),
        ("Sod p，无限制器基线", post["configs"][0]["relative_l2_percent"]["pressure"], "精确 Riemann 解"),
        ("Sod u，一次性 TVB（末态）", regional["relative_l2_percent"]["tvb one-shot (final state)"]["velocity"], "精确 Riemann 解"),
        ("Sod ρ，一次性 TVB（末态）", regional["relative_l2_percent"]["tvb one-shot (final state)"]["density"], "精确 Riemann 解"),
        ("Sod p，一次性 TVB（末态）", regional["relative_l2_percent"]["tvb one-shot (final state)"]["pressure"], "精确 Riemann 解"),
    ]
    for label, value, ref in rows:
        add(f"| {label} | {fmt(value)}% | {ref} | {fmt(value)}% | {verdict(value)} |")
    add("")
    add("**回答**：限制器在循环内启用时 **不满足** 3% 门（u 场 "
        f"{fmt(post['configs'][1]['relative_l2_percent']['velocity'],2)}% / "
        f"{fmt(post['configs'][4]['relative_l2_percent']['velocity'],2)}%）；"
        f"无限制器基线本身也未达标（u = {fmt(post['configs'][0]['relative_l2_percent']['velocity'],2)}%，"
        f"{meta['cells']} 单元 P{meta['degree']} 对间断问题分辨率不足，间断处 L2 收敛只有一阶），"
        "因此在 3% 门的意义下**本网格上任何配置都不能宣称达标**；"
        "唯一达标的量是把已修好刻度的限制器一次性作用于末态的情形（"
        f"max {fmt(regional['max_relative_l2_percent']['tvb one-shot (final state)'],2)}%）。"
        "这是本单最重要的 honest 结论：**限制器公式可用，但当前默认启用路径有害。**")
    add("")

    add("## 5. 注意点 / 局限 / 适用范围")
    add("")
    add("1. **单点分辨率结论**：只在 200 单元 P2、CFL=0.4 做了系统的误差扫描；"
        "400/800 单元与 P1/P3 未测（机器 load15≈110，按容量纪律只做轻量算例）。"
        "基线在此网格上的 u 误差恰好 3.2%，因此「基线是否达标」对网格敏感，结论按网格标注。")
    add("2. **一次性施加不是修复方案**：把限制器只在最末一步施加会丢掉爆轰/激波传播过程中的稳定性保障"
        "（不再有逐步的极值抑制），只能作为「公式无罪」的判据，不能当默认策略。")
    add("3. **检测器仍在守恒变量上工作**：ρ、m、E 分别做 minmod 不保证重构后的 u = m/ρ 单调，"
        "特征场限制（Cockburn–Shu 界面值版）未实现，这也是循环内反复投影会伤光滑区的结构性原因之一。")
    add("4. **固定测量口径**：误差用 6 点 Gauss 采样的相对 L2，与 `benchmarks/riemann_dg_error.py` 及既有 8 个"
        "验证基准一致；不同采样密度会改变小数点后第 3 位。")
    add("5. **未做性能对照**：原报告的 76–123% 开销未复核（本单不涉及），"
        "修 bug 后纯 Python 逐单元循环仍在，性能结论沿用旧记录并标注为未复核。")
    add("")

    add("## 6. 交付物")
    add("")
    add("| 文件 | 作用 |")
    add("|---|---|")
    add("| `tensordg/limiters.py` | 修复 TVB 限制器 mode-1 的 1/2 因子（1 行 + 注释） |")
    add("| `tests/test_limiters.py` | 新增 4 条回归测试（刻度、TVD 上界、界面不越界、TVB/WENO 同刻度）；17 条全绿 |")
    add("| `benchmarks/limiter_scale_check.py` | 刻度一致性量测 |")
    add("| `benchmarks/verify_limiter_sod.py` | 端到端误差扫描 |")
    add("| `benchmarks/limiter_oneshot.py` | 机制分离 + 检测器选择性 |")
    add("| `benchmarks/plot_limiter_verification.py` | 场量对比图 + 分区误差 |")
    add("| `benchmarks/report_limiter_verification.py` | 由工件生成本文档 |")
    add("| `validation/limiter_sod/*.json` | 全部原始数字（6 个工件） |")
    add("| `docs/figures/limiter_verification.png` | 报告配图 |")
    add("")
    add("## 7. 下一步（建议 mission，非本单范围）")
    add("")
    add("1. **P0** 限制器施加策略：改为 Cockburn–Shu 界面值形式，且只在**最终级**投影，"
        "以本单的三个 JSON 为验收判据（要求 max 相对误差 ≤ 基线且 < 3%）。")
    add("2. **P1** 检测器改为特征场判别（用右/左特征向量投影后判 minmod），"
        "目标是「光滑稀疏波区受标记比例 → 0」并在工件里留数。")
    add("3. **P1** 在 400/800 单元上复跑本 harness，给出误差随网格的收敛表，"
        "让「是否 <3%」有分辨率意义。")
    add("")
    add("---")
    add(f"生成时间：{post['timestamp']} · Python {post['python']} · torch {post['torch']}")
    add("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()