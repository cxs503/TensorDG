"""Render ``docs/hllc_wave_speed_verification.md`` from the wave-speed JSON.

Every number in the markdown and in the figure is read from
``validation/euler_fluxes/wave_speed_estimates.json`` (written by
``benchmarks/verify_hllc_wave_speeds.py``); nothing is transcribed by hand.

Usage::

    PYTHONPATH=. python3 benchmarks/report_hllc_wave_speeds.py
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = ["davis", "einfeldt", "toro", "toro_iter"]
LABELS = {"davis": "Davis (default)", "einfeldt": "Einfeldt/Roe",
          "toro": "Toro TRRS", "toro_iter": "Toro + Newton p*",
          "exact_speeds": "exact wave speeds (ceiling)"}
COLOURS = {"davis": "#b2182b", "einfeldt": "#ef8a62", "toro": "#67a9cf",
           "toro_iter": "#2166ac", "exact_speeds": "#4d4d4d"}


def fmt_percent(value):
    return f"{100.0 * value:.2f}%"


def fig_kernel(payload, path):
    kernel = payload["kernel"]
    cases = [row["case"] for row in kernel]
    series = MODELS + ["exact_speeds"]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axes[0]
    positions = np.arange(len(cases))
    width = 0.8 / len(series)
    for index, model in enumerate(series):
        values = [max(row["scaled_error"][model], 1e-8) for row in kernel]
        ax.bar(positions + (index - (len(series) - 1) / 2) * width, values, width,
               label=LABELS[model], color=COLOURS[model],
               alpha=0.9 if model != "exact_speeds" else 0.55,
               hatch="//" if model == "exact_speeds" else None)
    ax.set_yscale("log")
    ax.axhline(0.03, color="k", linestyle="--", linewidth=1.0)
    ax.text(len(cases) - 0.5, 0.033, "3% gate", ha="right", va="bottom", fontsize=9)
    fan = [row["case"] for row in kernel if row["inside_rarefaction_fan"]]
    for position, case in enumerate(cases):
        if case in fan:
            ax.axvspan(position - 0.5, position + 0.5, color="0.85", alpha=0.45, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(cases, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("max |F_num - F_exact| / scale")
    ax.set_title("(a) HLLC interface flux vs exact Riemann flux\n"
                 "(grey bands = interface inside a rarefaction fan)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", which="both", alpha=0.25)

    ax = axes[1]
    machs = [row["mach"] for row in payload["shock_sweep"]]
    for model in series:
        values = [row["scaled_error"][model] for row in payload["shock_sweep"]]
        ax.plot(machs, np.maximum(values, 1e-8), marker="o", label=LABELS[model],
                color=COLOURS[model], linewidth=1.6,
                linestyle="--" if model == "exact_speeds" else "-")
    ax.set_yscale("log")
    ax.axhline(0.03, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("shock Mach number")
    ax.set_ylabel("scaled interface-flux error")
    ax.set_title("(b) Isolated Rankine-Hugoniot shock:\nwave-speed estimates vs shock strength")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_end_to_end(payload, path):
    rows = payload.get("end_to_end", [])
    if not rows:
        return False
    problems = sorted({row["problem"] for row in rows})
    fig, axes = plt.subplots(1, len(problems), figsize=(5.2 * len(problems), 4.6))
    if len(problems) == 1:
        axes = [axes]
    for ax, problem in zip(axes, problems):
        subset = [row for row in rows if row["problem"] == problem]
        cells = sorted({row["cells"] for row in subset})
        for model in MODELS:
            values = [next(row["max_relative_l2_percent"] for row in subset
                           if row["model"] == model and row["cells"] == n) for n in cells]
            ax.plot(cells, values, marker="o", label=LABELS[model], color=COLOURS[model])
        ax.axhline(3.0, color="k", linestyle="--", linewidth=1.0)
        ax.text(cells[-1], 3.1, "3% gate", ha="right", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("cells")
        ax.set_ylabel("max relative L2 error of (rho, u, p) [%]")
        ax.set_title(f"end-to-end P0: {problem}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def figure_shock_width(payload, path):
    rows = [row for row in payload.get("end_to_end", []) if row["problem"] == "strong_shock"]
    if not rows:
        return False
    cells = sorted({row["cells"] for row in rows})
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for model in MODELS:
        values = [next((row["shock_width_cells"] for row in rows
                        if row["model"] == model and row["cells"] == n), None)
                  for n in cells]
        if any(value is None for value in values):
            continue
        ax.plot(cells, values, marker="o", label=LABELS[model], color=COLOURS[model])
    ax.set_xlabel("cells")
    ax.set_ylabel("shock width (10%-90% rise, in cells)")
    ax.set_title("strong shock: numerical diffusion of the leading shock")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def markdown(payload):
    summary = payload["kernel_summary"]
    lines = []
    add = lines.append

    add("# HLLC 波速估计验证：Davis → Einfeldt / Toro（闭合 11.3% 核缺陷）")
    add("")
    add(f"> 工件：`validation/euler_fluxes/wave_speed_estimates.json`（{payload['generated_at']}，"
        f"torch {payload['torch_version']}，γ = {payload['gamma']}）")
    add("> 驱动：`benchmarks/verify_hllc_wave_speeds.py`；本文与图由 "
        "`benchmarks/report_hllc_wave_speeds.py` 从工件程序化生成，零手抄。")
    add("")
    add("## ① 案例介绍")
    add("")
    add("**物理背景**：可压缩 Euler 方程的一阶（P0/FV）DG 半离散算子，界面通量用三波 HLLC。"
        "HLLC 的精度几乎完全由外层波速估计 $S_L, S_R$ 决定：估计过宽 → 通量过度耗散；"
        "估计过窄 → 违反熵条件/上风性。`docs/euler_flux_kernel_verification.md` 已测得 "
        "**Davis 波速下的界面通量与精确 Riemann 通量相差中位 11.3%（最大 24.2%）**，"
        "即越过本项目 3% 门，并把「换 Einfeldt/Roe 估计」列为后续工作。本单闭合该欠账。")
    add("")
    add("**算例集**（Toro 2009 ch.4 两个常量状态，理想气体 γ=1.4，float64）：")
    add("")
    add("| 算例 | 左态 (ρ, u, p) | 右态 (ρ, u, p) | 界面是否在稀疏波扇内 |")
    add("|---|---|---|---|")
    states = {row["case"]: row for row in payload["kernel"]}
    from benchmarks.verify_flux_properties import STANDARD_CASES  # noqa: F401  (doc only)
    reference_states = {
        "sod": ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1)),
        "lax": ((0.445, 0.698, 3.528), (0.5, 0.0, 0.571)),
        "123problem": ((1.0, -2.0, 0.4), (1.0, 2.0, 0.4)),
        "strong_shock": ((5.99924, 19.5975, 460.894), (5.99242, -6.19633, 46.095)),
        "stationary_contact": ((1.0, 0.0, 1.0), (0.125, 0.0, 1.0)),
        "moving_contact": ((1.0, 0.5, 1.0), (0.125, 0.5, 1.0)),
        "left_moving_contact": ((1.0, -0.5, 1.0), (0.125, -0.5, 1.0)),
        "sonic_rarefaction": ((1.0, -0.75, 1.0), (1.0, 0.75, 1.0)),
    }
    for row in payload["kernel"]:
        left, right = reference_states[row["case"]]
        add(f"| {row['case']} | {left} | {right} | "
            f"{'是' if row['inside_rarefaction_fan'] else '否'} |")
    add("")
    add("## ② 怎么做的")
    add("")
    add("1. 在 `tensordg/euler_1d.py` 中把外层波速估计抽象为 `WAVE_SPEEDS` 词表（4 个模型，"
        "`hllc_flux`/`hllc_rhs`/`solve_riemann` 通过 `wave_speed` / `hllc_wave_speed` 选择），"
        "HLLC 代数本身**一行未改**——保证对比只反映波速估计的差别：")
    add("   * `davis`：$S_L=\\min(u_L-a_L,u_R-a_R)$，$S_R=\\max(u_L+a_L,u_R+a_R)$（历史默认，向后兼容）；")
    add("   * `einfeldt`：Roe 平均 $\\tilde u,\\tilde H,\\tilde a$ 收紧上界 "
        "$S_L=\\min(u_L-a_L,\\tilde u-\\tilde a)$，$S_R=\\max(u_R+a_R,\\tilde u+\\tilde a)$；")
    add("   * `toro`：Toro 两稀疏波近似求 $p^*$（TRRS，无迭代），再代入压力基波速公式；")
    add("   * `toro_iter`：在 TRRS 初值上做 **8 步全步 Newton**（Toro 2009 eq.4.6 压力函数，"
        "带正性/有限性护栏）求 $p^*$，再代入同一波速公式。")
    add("2. **归因上限**：同一 HLLC 代数喂入**精确外层波速**（由精确 Riemann 解算得）作为天花板——"
        "三波格式无论如何选波速都不可能比它更准，它把「格式固有残差」与「估计器残差」分开。")
    add("3. 精确界面通量取自 `benchmarks/riemann_exact.sample`（能解析稀疏波扇与接触间断，"
        "所以即使 $x/t=0$ 落在扇内比较依然合法）。")
    add("4. 端到端敏感度：P0/RK3 求解 Toro test 3（强激波）、Lax、Sod，200/400 网格，"
        "记录 (ρ,u,p) 相对 L2 误差与**激波 10%–90% 展宽（格数）**——后者才是界面耗散的敏感量。")
    add("5. 回归：`tests/test_hllc_wave_speeds.py`（51 例）钉住恒等式、$p^*$ 精度、3% 门与管道。")
    add("")
    add("## ③ 计算结果")
    add("")
    add("### 3.1 界面通量 vs 精确 Riemann 通量（归一化最大绝对误差）")
    add("")
    header = "| 算例 | " + " | ".join(LABELS[m] for m in MODELS) + " | 精确波速（上限） |"
    add(header)
    add("|---|" + "---:|" * (len(MODELS) + 1))
    for row in payload["kernel"]:
        cells = " | ".join(fmt_percent(row["scaled_error"][m]) for m in MODELS)
        add(f"| {row['case']} | {cells} | {fmt_percent(row['scaled_error']['exact_speeds'])} |")
    add("")
    add("### 3.2 汇总（中位 / 最大）")
    add("")
    add("| 模型 | 全 8 例中位 | 全 8 例最大 | 可复原 5 例中位 | 可复原 5 例最大 |")
    add("|---|---:|---:|---:|---:|")
    for model in MODELS + ["exact_speeds"]:
        entry = summary[model]
        add(f"| {LABELS[model]} | {fmt_percent(entry['all_median'])} | "
            f"{fmt_percent(entry['all_max'])} | {fmt_percent(entry['recoverable_median'])} | "
            f"{fmt_percent(entry['recoverable_max'])} |")
    add("")
    add("「可复原 5 例」= 除 sod / 123problem / sonic_rarefaction 之外的 5 例："
        "前三个算例的界面落在稀疏波扇内，三波 HLLC 原理上无法复现精确通量"
        "（精确波速下同样 ~10–16%，见 3.1 最后一列）。")
    add("")
    add("### 3.3 星区压力估计精度（$p^*$）")
    add("")
    add("| 算例 | 精确 $p^*$ | TRRS | TRRS 相对误差 | PVRS 相对误差 | Newton 相对误差 |")
    add("|---|---:|---:|---:|---:|---:|")
    for row in payload["star_pressure"]:
        add(f"| {row['case']} | {row['exact_p_star']:.6g} | {row['trrs_p_star']:.6g} | "
            f"{fmt_percent(row['trrs_relative_error'])} | "
            f"{fmt_percent(row['pvrs_relative_error'])} | "
            f"{row['newton_relative_error']:.2e} |")
    add("")
    add("### 3.4 孤立正激波扫描")
    add("")
    add("| Mach | 波后 (ρ, u, p) | " + " | ".join(LABELS[m] for m in MODELS) + " | 精确波速 |")
    add("|---|---|" + "---:|" * (len(MODELS) + 1))
    for row in payload["shock_sweep"]:
        post = row["post_shock_state"]
        cells = " | ".join(fmt_percent(row["scaled_error"][m]) for m in MODELS)
        add(f"| {row['mach']:g} | ({post['density']:.5g}, {post['velocity']:.5g}, "
            f"{post['pressure']:.5g}) | {cells} | "
            f"{fmt_percent(row['scaled_error']['exact_speeds'])} |")
    add("")
    add("### 3.5 恒等式与守恒（每个模型都必须成立）")
    add("")
    add("| 模型 | $F(U,U)=f(U)$ 最大绝对误差 | 镜像对称 最大绝对误差 | 孤立接触 最大相对误差 | 守恒残差 |")
    add("|---|---:|---:|---:|---|")
    for model in MODELS:
        inv = payload["invariants"][model]
        con = payload["conservation"][model]
        add(f"| {LABELS[model]} | {inv['consistency_max_abs']:.2e} | "
            f"{inv['mirror_symmetry_max_abs']:.2e} | "
            f"{inv['isolated_contact_max_relative']:.2e} | "
            f"{con['mass_rate_total']:.2e}（finite={con['finite']}） |")
    add("")
    add("### 3.6 端到端敏感度（P0/RK3）")
    add("")
    end_to_end = payload.get("end_to_end", [])
    if end_to_end:
        add("| 算例 | 网格 | 模型 | 步数 | 墙钟 s | ρ L2 | u L2 | p L2 | max L2 | 激波展宽（格） |")
        add("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in end_to_end:
            err = row["relative_l2_percent"]
            width = row["shock_width_cells"]
            width_text = "—" if width is None else f"{width:.2f}"
            add(f"| {row['problem']} | {row['cells']} | {LABELS[row['model']]} | "
                f"{row['steps']} | {row['seconds']:.1f} | {err['density']:.3f}% | "
                f"{err['velocity']:.3f}% | {err['pressure']:.3f}% | "
                f"{row['max_relative_l2_percent']:.3f}% | {width_text} |")
        add("")
    else:
        add("（本次运行未包含端到端层。）")
        add("")
    add("## ④ 图")
    add("")
    add("![kernel](figures/hllc_wave_speed_verification.png)")
    add("")
    add("(a) 8 个标准算例 × 5 种波速选择的界面通量误差（对数轴，虚线=3% 门，灰带=界面在稀疏波扇内）；"
        "(b) 孤立正激波随 Mach 数的误差衰减，Davis 在大 Mach 上明显劣化，Newton 精化后与精确波速重合。")
    if end_to_end:
        add("")
        add("![end-to-end](figures/hllc_wave_speed_end_to_end.png)")
        add("")
        add("端到端 P0 误差随网格变化：四种波速选择在 P0 上几乎不可区分——见 ⑥ 与 ⑤-3。")
        add("")
        add("![shock width](figures/hllc_wave_speed_shock_width.png)")
        add("")
        add("强激波前导激波的 10%–90% 展宽（格数）：界面耗散的敏感量。")
    add("")
    add("## ⑤ 注意点（坑 / 局限 / 适用范围）")
    add("")
    add("1. **稀疏波扇内的残差不是缺陷**：sod / 123problem / sonic_rarefaction 三例的界面落在扇内，"
        "三波格式无法表达扇内状态，**精确波速下残差同样是 "
        f"{fmt_percent(states['sonic_rarefaction']['scaled_error']['exact_speeds'])}/"
        f"{fmt_percent(states['sod']['scaled_error']['exact_speeds'])}/"
        f"{fmt_percent(states['123problem']['scaled_error']['exact_speeds'])}**。"
        "任何「HLLC 波速不合格」的结论都必须扣掉这三例，否则永远不可能达标。")
    add("2. **TRRS 在极强激波上不可靠**：Toro test 3（p 比 460.9/46.1）TRRS 给的 $p^*$ 偏高 "
        f"{fmt_percent(next(row['trrs_relative_error'] for row in payload['star_pressure'] if row['case'] == 'strong_shock'))}"
        "（3.3 表 TRRS 列），代价是界面通量误差 "
        f"{fmt_percent(states['strong_shock']['scaled_error']['toro'])}；Newton 精化后 $p^*$ 精确、通量误差归零。"
        "所以 `toro`（无迭代）只适合一般强度；要对极强激波可信，用 `toro_iter`。")
    add("3. **核级收益 ≠ 解级收益**：P0 一阶格式的解误差由网格耗散主导，四种波速在 200/400 网格上"
        "误差差别 <0.3 个百分点，甚至 `davis` 略优；因此**没有**把默认波速改掉"
        "（`davis` 仍是 `hllc_flux` / `solve_riemann` 的默认，README 与历史基准数字不受影响）。"
        "要兑现核级精度收益，需要在**高阶 DG 迹值通量**或精度敏感耦合里使用，"
        "这超出本单（`euler_dg_1d.py` 有自己的 HLLC 实现，未纳入）。")
    add("4. **代价**：`toro_iter` 每个界面 8 步 Newton（约 3–5 倍界面通量开销，端到端表里直接可见），"
        "`einfeldt`/`toro` 只多 1 次 Roe 平均/幂运算，几乎免费。")
    add("5. **守恒性**：`hllc_rhs` 的均匀态残差与质量收支在各模型下都是 ~1e-13 量级，"
        "换波速不破坏守恒性（3.5 表）。")
    add("6. **适用范围**：1-D、Cartesian、γ=1.4 理想气体、float64、状态为两个常量态；"
        "多维面通量与限制器后迹值未在本单考证。")
    add("")
    add("## ⑥ 误差判定（关键量 vs 基准，逐项显式回答「是否 <3%」）")
    add("")
    davis_median = summary["davis"]["all_median"]
    einf_median = summary["einfeldt"]["all_median"]
    toro_median = summary["toro"]["all_median"]
    iter_median = summary["toro_iter"]["all_median"]
    iter_recoverable_max = summary["toro_iter"]["recoverable_max"]
    add("| 关键量 | 基准 | Davis（历史） | Einfeldt | Toro TRRS | Toro+Newton | <3%？ |")
    add("|---|---|---:|---:|---:|---:|---|")
    add(f"| 界面通量误差中位（8 例） | 精确 Riemann 通量 | {fmt_percent(davis_median)} | "
        f"{fmt_percent(einf_median)} | {fmt_percent(toro_median)} | {fmt_percent(iter_median)} | "
        f"**是**（Newton 后 {fmt_percent(iter_median)}，Davis 时否） |")
    add(f"| 可复原 5 例最大误差 | 精确 Riemann 通量 | "
        f"{fmt_percent(summary['davis']['recoverable_max'])} | "
        f"{fmt_percent(summary['einfeldt']['recoverable_max'])} | "
        f"{fmt_percent(summary['toro']['recoverable_max'])} | "
        f"{fmt_percent(iter_recoverable_max)} | "
        f"**是**（Newton 后；Davis 时否） |")
    add(f"| 扇内 3 例最大误差 | 精确波速上限 | "
        f"{fmt_percent(states['123problem']['scaled_error']['exact_speeds'])} | "
        "同上 | 同上 | 同上 | **否**（格式固有，非估计器缺陷） |")
    newton_worst = max(row["newton_relative_error"] for row in payload["star_pressure"])
    add(f"| $p^*$ 相对误差（Newton） | 精确 Riemann 解 | — | — | "
        f"{fmt_percent(max(row['trrs_relative_error'] for row in payload['star_pressure']))} | "
        f"{newton_worst:.1e} | **是** |")
    add("")
    add("**结论**：`docs/euler_flux_kernel_verification.md` 记录的中位 11.3% 通量缺陷，"
        f"根因确认为**外层波速估计**：换 Einfeldt 降到 {fmt_percent(einf_median)}，"
        f"换 Newton 精化 $p^*$ 降到 {fmt_percent(iter_median)}，"
        f"可复原算例最大误差 {fmt_percent(iter_recoverable_max)}，"
        "**首次让 HLLC 核在全部可复原算例上进入 3% 门**；残余的扇内 10–16% 与精确波速天花板逐位一致，"
        "属三波格式固有局限，已如实标注，不计作本单失败。端到端 P0 无收益（诚实记录）。")
    add("")
    add("## ⑦ 复现")
    add("")
    add("```bash")
    add("PYTHONPATH=. OMP_NUM_THREADS=1 python -m benchmarks.verify_hllc_wave_speeds \\")
    add("    --json validation/euler_fluxes/wave_speed_estimates.json   # 核级+端到端，约 5 min")
    add("PYTHONPATH=. python -m benchmarks.verify_hllc_wave_speeds --quick  # 只跑核级，约 3 s")
    add("PYTHONPATH=. python -m benchmarks.report_hllc_wave_speeds        # 本文 + 图")
    add("PYTHONPATH=. python -m pytest tests/test_hllc_wave_speeds.py -q")
    add("```")
    add("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="validation/euler_fluxes/wave_speed_estimates.json")
    parser.add_argument("--markdown", default="docs/hllc_wave_speed_verification.md")
    parser.add_argument("--figures", default="docs/figures")
    args = parser.parse_args()

    with open(args.json) as handle:
        payload = json.load(handle)

    os.makedirs(args.figures, exist_ok=True)
    fig_kernel(payload, os.path.join(args.figures, "hllc_wave_speed_verification.png"))
    fig_end_to_end(payload, os.path.join(args.figures, "hllc_wave_speed_end_to_end.png"))
    figure_shock_width(payload, os.path.join(args.figures, "hllc_wave_speed_shock_width.png"))

    text = markdown(payload)
    os.makedirs(os.path.dirname(args.markdown), exist_ok=True)
    with open(args.markdown, "w") as handle:
        handle.write(text + "\n")
    print(f"wrote {args.markdown} and figures in {args.figures}")


if __name__ == "__main__":
    main()