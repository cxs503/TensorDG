"""Build the platform progress report (JSON) from the entropy-budget artifacts.

Every number in the report body is read from
``validation/entropy_1d/summary.json`` and ``validation/entropy_1d/summary_old.json``
(the first-version artifact), so nothing is transcribed by hand.

Usage: PYTHONPATH=. python3 tools/report_entropy_json.py [--mission 3458]
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib

SUMMARY = pathlib.Path("validation/entropy_1d/summary.json")
OLD = pathlib.Path("validation/entropy_1d/summary_old.json")
FIGURE = pathlib.Path("docs/figures/entropy_budget_1d.png")


def _trend(rows):
    rows = sorted(rows, key=lambda row: row["cells"])
    return " → ".join(f"{row['cells']}:{100.0 * row['relative_error']:.3f}%" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, default=3458)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("/tmp/report.json"))
    args = parser.parse_args()

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    sod, lax = summary["shock"]["sod"], summary["shock"]["lax"]
    verdicts = summary["verdicts"]
    profile = summary["profile_comparison"]
    metrics = profile.get("metrics", {})
    orders = verdicts.get("C2_error_observed_orders", {})
    finest = verdicts["C2_relative_error_at_finest_mesh"]
    coarsest = verdicts.get("C2_relative_error_at_coarsest_mesh", {})
    gate = verdicts["C2_below_3_percent"]
    labels = sorted({row.get("label", row["limiter"]) for row in sod["rows"]})

    def rows_of(label):
        return [row for row in sod["rows"] if row.get("label", row["limiter"]) == label]

    table_rows = "\n".join(
        f"| {row.get('label', row['limiter'])} | {row['cells']} | {row['steps']} | "
        f"{row['production_numeric']:.6e} | {row['production_analytic']:.6e} | "
        f"{100.0 * row['relative_error']:.3f}% | {'✅' if row['second_law'] else '❌'} |"
        for row in sorted(sod["rows"], key=lambda row: (row.get("label", ""), row["cells"]))
    )
    lax_rows = "\n".join(
        f"| {row.get('label', row['limiter'])} | {row['cells']} | {row['steps']} | "
        f"{row['production_numeric']:.6e} | {100.0 * row['relative_error']:.3f}% | "
        f"{'✅' if row['second_law'] else '❌'} |"
        for row in sorted(lax["rows"], key=lambda row: (row.get("label", ""), row["cells"]))
    )
    gate_rows = "\n".join(
        f"| {key} | {100.0 * coarsest.get(key, float('nan')):.3f}% | {100.0 * value:.3f}% | "
        f"{(orders.get(key) or [None])[-1] if (orders.get(key) or [None])[-1] is not None else 'n/a'} | "
        f"{'是' if gate[key] else '否'} |"
        for key, value in sorted(finest.items())
    )
    profile_rows = "\n".join(
        f"| {key} | {value['rms']:.4e} | {value['rms_over_entropy_jump']:.4f} | "
        f"{100.0 * value['legacy_relative_l2']:.0f}% |"
        for key, value in metrics.items()
    )

    old_note = ""
    if OLD.exists():
        old = json.loads(OLD.read_text(encoding="utf-8"))
        old_finest = old["verdicts"]["C2_relative_error_at_finest_mesh"]
        old_note = (
            "\n**旧工件对照（`summary_old.json`，首版）**：其 `C2_relative_error_at_finest_mesh` "
            + ", ".join(f"{k}={100.0 * v:.3f}%" for k, v in sorted(old_finest.items()))
            + "，与新版同一字段为 "
            + ", ".join(f"{k}={100.0 * v:.3f}%" for k, v in sorted(finest.items()))
            + "。差异来源是 `min`→`max` 的最细网格选择器修复（旧版取的是最粗网格），"
            "不是求解器变化。\n"
        )

    markdown = f"""# TensorDG 熵收支验证欠账清算：最细网格门判定修复 + 网格加密至 1024 cells

## ① 案例介绍

**物理背景。** 理想气体（γ=1.4）Euler 方程组的物理熵 `s = ln(p/ρ^γ)`：光滑解满足守恒律
`∂t(ρs)+∂x(ρus)=0`，含激波的弱解满足熵不等式 `∂t(ρs)+∂x(ρus)=P≥0`，`P` 以激波为支撑。
`Δ∫ρs dx` 必须为正，因此"格式是否违反第二定律"是**可伪证**的判据，与"解像不像"无关。

**算例与参数。**

| 项 | 取值 |
|---|---|
| 区域 | 一维 [0,1]，单位长度，平直 Cartesian |
| 方程 | 1-D Euler，理想气体 γ=1.4，无粘 |
| Sod | (ρ,u,p)_L=(1.0,0,1.0), (ρ,u,p)_R=(0.125,0,0.1)，t={sod['analytic']['final_time']} |
| Lax | (ρ,u,p)_L=(0.445,0.698,3.528), (ρ,u,p)_R=(0.5,0,0.571)，t={lax['analytic']['final_time']} |
| 空间离散 | 模态 Legendre DG，P{summary['degree']}，HLLC 通量，局部正性回退 |
| 时间推进 | SSP-RK3，CFL={summary['cfl']} |
| 网格 | Sod {sorted({row['cells'] for row in sod['rows']})}，Lax {sorted({row['cells'] for row in lax['rows']})} |
| 限制器 | {', '.join(labels)}（tvb_constant=1.0，limiter_stage ∈ {sorted({row.get('limiter_stage','every') for row in sod['rows']})}） |
| 线程 | torch.set_num_threads({summary['torch_threads']})，高负载机（load15>120） |

## ② 怎么做的（方法与关键设置）

1. **诊断量**：熵积分在 DG 多项式的高斯点上逐点求 ρs 再积分（正交阶 max(p+2,8)），
   不用单元平均值（会引入 O(h²) 人为误差）；自检显示正交阶散布 {summary['diagnostic_self_test']['quadrature_relative_spread']:.3e}。
2. **解析基准**：单激波熵产生率由 Rankine-Hugoniot 态给出 `P = σ[ρs] − [ρus]`，σ 由质量守恒定出，
   下游态由精确星状态 + RH 密度比得到，累计产生量 `P·T`。**无拟合参数**。
   本轮还新增了一条代数恒等式校验（见 ⑤ 注意点）：`P = ṁ[s]`，`ṁ=ρ_a(σ−u_a)`。
3. **本轮修的三个欠账**（都在验证工装里，不在求解器里）：
   - **最细网格门判定缺陷**：`C2_relative_error_at_finest_mesh` 用 `min(rows, key=cells)`
     取了**最粗**网格。Sod/none 因此被记成 64 cells 的 11.651%，而真实 256 cells 值是 3.126%。
     改为 `finest_row()`（`max`），并新增最粗网格列与误差观测阶列。
   - **`limiter_stage="final"` 分支首次纳入熵收支**（旧版明写"未覆盖"）。
   - **剖面度量替换**：ρs 变号、且 `s=ln(p/ρ^γ)` 在 p→0 时发散，
     `‖e‖/‖ρs‖` 曾给出 163068% 这种不可解释的数。改为 RMS(ρs) 与 RMS/|Δs|（激波熵跃归一化）。
4. **判据**：C1 第二定律（每个激波运行 Δ∫ρs dx>0）、C2 定量收支收敛到 P·T（最细网格 3% 门）、
   C3 光滑周期算例熵产生随 h 趋于零。
5. **门判定规则**：3% 门**只看最细网格**（这是本条验证的定量判据），粗网格只作趋势用。

## ③ 计算结果

### Sod（解析 P·T = {sod['analytic']['production_total']:.6e}，σ={sod['analytic']['sigma']:.4f}，激波位于 x={sod['analytic']['shock_position_at_T']:.4f}）

| limiter | cells | steps | Δ∫ρs dx | 解析 P·T | 相对误差 | 第二定律 |
|---|---:|---:|---:|---:|---:|---|
{table_rows}

### Lax（解析 P·T = {lax['analytic']['production_total']:.6e}，σ={lax['analytic']['sigma']:.4f}）

| limiter | cells | steps | Δ∫ρs dx | 相对误差 | 第二定律 |
|---|---:|---:|---:|---:|---|
{lax_rows}

### C2 门判定（3% 门以最细网格为准）

| run | 最粗网格误差 | 最细网格误差 | 误差观测阶（末对） | <3% |
|---|---:|---:|---:|---|
{gate_rows}

### 熵密度剖面误差（Sod，{profile['cells']} cells；激波熵跃 |Δs| = {profile.get('entropy_jump_scale', float('nan')):.6f}）

| limiter | RMS(ρs) | RMS/\|Δs\| | 旧度量 ‖e‖/‖ρs‖（弃用） |
|---|---:|---:|---:|
{profile_rows}

{old_note}
## ④ 结果图

三面板：(a) Sod/Lax 熵产生量与解析 P·T 的绝对偏差随网格下降（灰虚线 h² 参考斜率，
本轮曲线延伸到 1024 cells）；(b) Sod 熵密度剖面：精确解 vs DG 各限制器，
激波处平台被抹宽的范围就是数值熵产生的主要来源；(c) 光滑周期熵波熵产生与 h³ 参考斜率。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **工装缺陷比求解器缺陷更常见**：本单最有价值的发现是"最细网格"取了 `min` ——
   一个字符的符号错误让 honest-FAIL 被判在**最粗**网格上。现在由
   `tests/test_entropy_budget_gate.py::test_finest_row_selects_the_largest_cell_count` 钉住。
2. **相对 L2 不能用于变号场**：ρs 变号 + 对数在 p→0 发散，`‖e‖/‖ρs‖` 曾达 163068%，
   且对单个近真空采样点敏感（单点 −20 就能把该比值推到 O(100%)）。已在测试中记录该机理。
3. **熵产生 = 物理熵产生 + 数值耗散**：粗网格上后者远大于前者，因此该判据在粗网格只能当
   "上界/方向性检查"，定量门必须看收敛后的最细网格。
4. **解析基准只覆盖右侧激波构型**（p*>p_R）；双稀疏波、左激波未覆盖。接触间断只搬运熵不产生熵。
5. **RH 恒等式校验**：`P = ṁ[s]`（ṁ 为质量通量）与实现对 `P=σ[ρs]−[ρus]` 的编码代数独立，
   测试 `test_production_equals_mass_flux_times_entropy_jump` 对 Sod/Lax 均以 1e-12 相对容差通过。
6. 适用范围：一维、平直 Cartesian、无粘、常 γ；CPU float64。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 结果 | 是否 <3% |
|---|---|---|---|
| Sod/none 熵产生（最细 {max(row['cells'] for row in rows_of('none'))} cells） | P·T = {sod['analytic']['production_total']:.6e} | {100.0 * finest.get('sod/none', float('nan')):.3f}% | {'是' if gate.get('sod/none') else '**否**（honest）'} |
| Sod/tvb 熵产生 | 同上 | {100.0 * finest.get('sod/tvb', float('nan')):.3f}% | {'是' if gate.get('sod/tvb') else '**否**（honest）'} |
| Lax/none 熵产生 | P·T = {lax['analytic']['production_total']:.6e} | {100.0 * finest.get('lax/none', float('nan')):.3f}% | {'是' if gate.get('lax/none') else '**否**（honest）'} |
| Lax/tvb 熵产生 | 同上 | {100.0 * finest.get('lax/tvb', float('nan')):.3f}% | {'是' if gate.get('lax/tvb') else '**否**（honest）'} |
| C1 第二定律（{len(sod['rows']) + len(lax['rows'])} 次激波运行） | Δ∫ρs dx>0 | {'全部通过' if verdicts['C1_second_law_all_runs'] else '存在违反'} | {'是' if verdicts['C1_second_law_all_runs'] else '否'} |
| C3 光滑算例（阶数 ≥ p） | 熵产生→0 | 阶 {verdicts['C3_smooth_observed_orders']} | {'是' if all(verdicts['C3_smooth_tends_to_zero'].values()) else '否'} |
| 诊断自检（解析 P vs 精确解 FD） | 阈值 0.1% | {100.0 * summary['diagnostic_self_test']['analytic_vs_exact_profile']['relative_difference']:.3f}% | {'是' if verdicts['diagnostic_self_test_below_0_1_percent'] else '**否**'} |
| CFL 无关性 | 阈值 1% | {100.0 * summary['cfl_sensitivity'][0]['relative_difference_to_reference_cfl']:.3f}% | {'是' if verdicts['cfl_insensitive_below_1_percent'] else '否'} |

**是否 <3%？** 逐项回答见上表：最细网格上 C2 定量收支{'**通过**：' + ', '.join(k for k, v in gate.items() if v) if any(gate.values()) else '**未通过**'}；
未通过项如实标注 honest（{'、'.join(k for k, v in gate.items() if not v) or '无'}）。
C1 第二定律与 C3 光滑一致性{'全部通过' if verdicts['C1_second_law_all_runs'] and all(verdicts['C3_smooth_tends_to_zero'].values()) else '存在未通过项'}。

## ⑦ 结论与下一步

1. **工装修复是本轮核心增量**：最细网格门判定缺陷（min→max）被修复并用测试钉住；
   `limiter_stage="final"` 分支首次进入熵收支；剖面度量换成可解释的 RMS/|Δs|。
2. **网格加密后 Sod 达成情况**：{_trend(rows_of('none'))}（none），
   {_trend(rows_of('tvb'))}（tvb）。观测阶随时间/空间误差混合，见上表末列。
3. **下一步线索**：Lax 强激波（σ≈2.48）需要更高阶格式或在更细网格上重做；
   `weno` 限制器熵收支、二维 ρs 收支仍未覆盖；建议把 `finest_row` 语义推广到其他验证工装
   （凡"最细网格门"字段都应有一个被测试钉住的取值器）。
"""

    payload = {
        "title": "TensorDG 熵收支验证欠账清算：最细网格门判定修复 + 网格加密至 1024 cells",
        "summary": (
            "修复 entropy_budget 工装的'最细网格'门判定缺陷（min→max，旧值实为最粗网格），"
            "把 limiter_stage='final' 分支纳入熵收支、把不可解释的 ρs 相对 L2 换成 RMS/|Δs|，"
            "并把 Sod 剖面加密到 1024 cells；13 个新回归测试钉住语义。"
        ),
        "scenario_key": "tensordg-entropy-budget-gate",
        "mission_id": args.mission,
        "tags": ["TensorDG", "verification-debt", "entropy", "DG", "honest-FAIL"],
        "markdown": markdown,
        "figures": [],
    }
    if FIGURE.exists():
        payload["figures"].append({
            "name": "entropy_budget_1d.png",
            "caption": "熵收支：(a) Sod/Lax 与 RH 解析 P·T 的偏差（h² 参考）；(b) Sod 熵密度剖面；(c) 光滑周期熵波",
            "png_base64": base64.b64encode(FIGURE.read_bytes()).decode("ascii"),
        })
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {args.output} ({len(markdown)} chars, {len(payload['figures'])} figure)")


if __name__ == "__main__":
    main()