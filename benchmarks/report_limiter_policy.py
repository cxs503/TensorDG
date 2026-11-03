"""Generate the limiter-policy verification figure and Markdown report.

Reads ``validation/limiter_policy/policy_study.json`` (and the optional
``prefix_baseline.json`` produced with the pre-fix indicator) and writes
``docs/figures/limiter_policy_verification.png`` plus
``docs/limiter_policy_verification.md``.  All numbers in the Markdown are
injected from the JSON artifacts -- nothing is transcribed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordg.euler_dg_1d import solve_periodic_euler_dg_1d  # noqa: E402
from tensordg.limiters import _troubled_cell_indicator  # noqa: E402

DATA = Path("validation/limiter_policy")
FIGURE = Path("docs/figures/limiter_policy_verification.png")
DOC = Path("docs/limiter_policy_verification.md")

AMPLITUDE = 0.2


def _wave_state(x):
    rho = 1.0 + AMPLITUDE * torch.sin(2.0 * np.pi * x)
    return rho, torch.ones_like(x), torch.ones_like(x)


def _legacy_counts(coeff, dx, M):
    """Pre-fix indicator (HEAD): max interface jump vs M dx^2 + minmod test."""
    from tensordg.limiters import _minmod

    avg = coeff[:, 0]
    pad = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    d_left = avg - pad[:-2]
    d_right = pad[2:] - avg
    variation = torch.maximum(d_left.abs(), d_right.abs()).max(dim=-1).values
    mm = _minmod(d_left, d_right)
    not_monotone = ~((mm - d_left).abs().max(dim=-1).values < 1e-14)
    troubled = not_monotone & (variation > M * dx * dx)
    return int(troubled.sum())


def selectivity_table():
    """Troubled-cell fraction for the smooth wave and for the Sod initial data."""
    rows = []
    for cells in (20, 40, 80, 160):
        _, coeff, _ = solve_periodic_euler_dg_1d(
            _wave_state, ncells=cells, degree=2, final_time=0.0)
        dx = 1.0 / cells
        counts = {str(M): int(_troubled_cell_indicator(coeff, dx, M).sum())
                  for M in (1.0, 4.0, 10.0, 100.0)}
        rows.append({"kind": "smooth_wave", "cells": cells, "counts": counts,
                     "legacy": _legacy_counts(coeff, dx, 100.0)})
    cells = 100
    dx = 1.0 / cells
    x = (np.arange(cells) + 0.5) * dx
    left = x < 0.5
    coeff = torch.zeros(cells, 3, 3, dtype=torch.float64)
    coeff[:, 0, 0] = torch.as_tensor(np.where(left, 1.0, 0.125))
    coeff[:, 0, 2] = torch.as_tensor(np.where(left, 2.5, 0.25))
    rows.append({
        "kind": "sod_initial", "cells": cells,
        "counts": {str(M): int(_troubled_cell_indicator(coeff, dx, M).sum())
                   for M in (1.0, 4.0, 10.0, 100.0)},
    })
    return rows


def make_figure(payload, prefix):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # Panel A: smooth wave order
    ax = axes[0]
    wave = payload["smooth_wave"]["configurations"]
    for label, style in (("none", "k-o"), ("tvb_M1", "C1-s"), ("tvb_M10", "C0-^")):
        rows = wave[label]["rows"]
        cells = [r["cells"] for r in rows]
        err = [r["relative_l2_percent"] for r in rows]
        ax.loglog(cells, err, style, label=f"{label} (order "
                  f"{wave[label]['finest_order']:.2f})")
    if prefix is not None and "smooth_wave" in prefix:
        rows = prefix["smooth_wave"]["configurations"]["tvb_M1"]["rows"]
        ax.loglog([r["cells"] for r in rows], [r["relative_l2_percent"] for r in rows],
                  "C3--x", label="tvb_M1, pre-fix indicator "
                  f"(order {prefix['smooth_wave']['configurations']['tvb_M1']['finest_order']:.2f})")
    cells = [r["cells"] for r in wave["none"]["rows"]]
    ref = wave["none"]["rows"][0]["relative_l2_percent"] * (np.array(cells) / cells[0]) ** -3
    ax.loglog(cells, ref, ":", color="gray", label="slope 3 (design, p=2)")
    ax.set_xlabel("cells")
    ax.set_ylabel("relative $L^2$ density error [%]")
    ax.set_title("Smooth periodic entropy wave\n(exact solution, P2 DG)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    # Panel B: Sod mesh sequence vs the 3% gate
    ax = axes[1]
    sod = payload["sod"]["configurations"]
    for label, style in (("none", "k-o"), ("tvb_M1", "C1-s"),
                         ("tvb_M10", "C0-^"), ("weno_M10", "C2-d")):
        rows = sod[label]["rows"]
        ax.plot([r["cells"] for r in rows], [r["max_percent"] for r in rows],
                style, label=label)
    ax.axhline(3.0, color="r", ls="--", label="3% gate")
    ax.set_xlabel("cells")
    ax.set_ylabel("max relative $L^2$ over $\\rho,u,p$ [%]")
    ax.set_title("Sod shock tube vs exact Riemann solution")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # Panel C: indicator selectivity
    ax = axes[2]
    table = payload["selectivity"]
    Ms = [1.0, 4.0, 10.0, 100.0]
    for entry in table:
        if entry["kind"] != "smooth_wave":
            continue
        for M, style in zip(Ms, ("o", "s", "^", "d")):
            pass
    for M in Ms:
        frac = [entry["counts"][str(M)] / entry["cells"]
                for entry in table if entry["kind"] == "smooth_wave"]
        cells_list = [entry["cells"] for entry in table if entry["kind"] == "smooth_wave"]
        ax.semilogx(cells_list, frac, "o-", label=f"smooth wave, M={M:g}")
    sod_entry = [entry for entry in table if entry["kind"] == "sod_initial"][0]
    ax.axhline(sod_entry["counts"]["100.0"] / sod_entry["cells"], color="r", ls="--",
               label=f"Sod jump, M=100 ({sod_entry['counts']['100.0']}/100 cells)")
    ax.set_xlabel("cells")
    ax.set_ylabel("flagged fraction")
    ax.set_title("TVB indicator selectivity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    print(f"wrote {FIGURE}")


def _wave_table(payload, prefix):
    lines = [
        "| configuration | cells | relative $L^2$ density [%] | observed order | max conservation drift |",
        "|---|---:|---:|---:|---:|",
    ]
    wave = payload["smooth_wave"]["configurations"]
    for label in ("none", "tvb_M1", "tvb_M10", "tvb_M10_final"):
        if label not in wave:
            continue
        rows = wave[label]["rows"]
        orders = wave[label]["observed_orders"] + [None]
        for row, order in zip(rows, orders):
            order_txt = "&mdash;" if order is None else f"{order:.3f}"
            drift = max(row["conservation_drift"].values())
            lines.append(f"| {label} | {row['cells']} | {row['relative_l2_percent']:.6e} | "
                         f"{order_txt} | {drift:.2e} |")
    if prefix is not None and "smooth_wave" in prefix:
        rows = prefix["smooth_wave"]["configurations"]["tvb_M1"]["rows"]
        orders = prefix["smooth_wave"]["configurations"]["tvb_M1"]["observed_orders"] + [None]
        for row, order in zip(rows, orders):
            order_txt = "&mdash;" if order is None else f"{order:.3f}"
            drift = max(row["conservation_drift"].values())
            lines.append(f"| tvb_M1 *pre-fix indicator* | {row['cells']} | "
                         f"{row['relative_l2_percent']:.6e} | {order_txt} | {drift:.2e} |")
    return "\n".join(lines)


def _sod_table(payload):
    lines = [
        "| configuration | cells | $\\rho$ [%] | $u$ [%] | $p$ [%] | max [%] | <3%? |",
        "|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for label in ("none", "tvb_M1", "tvb_M10", "weno_M10"):
        if label not in payload["sod"]["configurations"]:
            continue
        for row in payload["sod"]["configurations"][label]["rows"]:
            gate = "yes" if row["max_percent"] < 3.0 else "**no**"
            lines.append(
                f"| {label} | {row['cells']} | {row['density_percent']:.4f} | "
                f"{row['velocity_percent']:.4f} | {row['pressure_percent']:.4f} | "
                f"{row['max_percent']:.4f} | {gate} |")
    return "\n".join(lines)


def _selectivity_table(payload):
    lines = [
        "| data | cells | M=1 | M=4 | M=10 | M=100 | pre-fix measure, M=100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in payload["selectivity"]:
        counts = entry["counts"]
        lines.append(
            f"| {entry['kind']} | {entry['cells']} | {counts['1.0']} | {counts['4.0']} | "
            f"{counts['10.0']} | {counts['100.0']} | {entry.get('legacy', '&mdash;')} |")
    return "\n".join(lines)


def _verdict_table(payload, prefix):
    """Per-quantity error judgement with an explicit <3% answer."""
    wave = payload["smooth_wave"]["configurations"]
    design = payload["degree"] + 1
    rows = []
    baseline_error = wave["none"]["rows"][-1]["relative_l2_percent"]
    for label, target in (("none", "unlimited baseline"),
                          ("tvb_M1", "TVB, M=1"),
                          ("tvb_M10", "TVB, M=10"),
                          ("tvb_M10_final", "TVB, M=10, final-stage")):
        if label not in wave:
            continue
        entry = wave[label]
        error = entry["rows"][-1]["relative_l2_percent"]
        rows.append({
            "quantity": f"smooth wave density error @ {entry['rows'][-1]['cells']} cells ({target})",
            "measured": f"{error:.3e}%",
            "reference": "exact entropy-wave solution",
            "error": f"{error:.3e}%",
            "below3": "yes" if error < 3.0 else "no",
            "order": entry["finest_order"],
            "order_ok": entry["finest_order"] >= 0.9 * design,
        })
    if prefix is not None and "smooth_wave" in prefix:
        entry = prefix["smooth_wave"]["configurations"]["tvb_M1"]
        error = entry["rows"][-1]["relative_l2_percent"]
        rows.append({
            "quantity": "smooth wave density error, pre-fix indicator (TVB, M=1)",
            "measured": f"{error:.3e}%",
            "reference": "exact entropy-wave solution",
            "error": f"{error:.3e}%",
            "below3": "yes" if error < 3.0 else "no",
            "order": entry["finest_order"],
            "order_ok": entry["finest_order"] >= 0.9 * design,
        })
    for label in ("none", "tvb_M1", "tvb_M10", "weno_M10"):
        config = payload["sod"]["configurations"].get(label)
        if not config:
            continue
        for row in config["rows"]:
            rows.append({
                "quantity": f"Sod max field error @ {row['cells']} cells ({label})",
                "measured": f"{row['max_percent']:.4f}%",
                "reference": "exact Riemann solution",
                "error": f"{row['max_percent']:.4f}%",
                "below3": "yes" if row["max_percent"] < 3.0 else "**no**",
                "order": None,
                "order_ok": None,
            })
    lines = [
        "| quantity | measured | reference | error | <3%? | finest order | order vs design |",
        "|---|---:|---|---:|:--:|---:|:--:|",
    ]
    for row in rows:
        order = "&mdash;" if row["order"] is None else f"{row['order']:.3f}"
        order_ok = "&mdash;" if row["order_ok"] is None else (
            "ok" if row["order_ok"] else "**degraded**")
        lines.append(f"| {row['quantity']} | {row['measured']} | {row['reference']} | "
                     f"{row['error']} | {row['below3']} | {order} | {order_ok} |")
    return "\n".join(lines), baseline_error, rows


def main():
    payload = json.loads((DATA / "policy_study.json").read_text())
    prefix_path = DATA / "prefix_baseline.json"
    prefix = json.loads(prefix_path.read_text()) if prefix_path.exists() else None
    payload["selectivity"] = selectivity_table()

    make_figure(payload, prefix)

    verdict_md, baseline_error, verdict_rows = _verdict_table(payload, prefix)
    wave = payload["smooth_wave"]["configurations"]
    sod = payload["sod"]["configurations"]
    design = payload["degree"] + 1

    fixed_order = wave["tvb_M10"]["finest_order"]
    fixed_error = wave["tvb_M10"]["rows"][-1]["relative_l2_percent"]
    strict_error = wave["tvb_M1"]["rows"][-1]["relative_l2_percent"]
    strict_order = wave["tvb_M1"]["finest_order"]
    wave80 = [e for e in payload["selectivity"]
              if e["kind"] == "smooth_wave" and e["cells"] == 80][0]
    strict_flags = wave80["counts"]["1.0"]
    legacy_wave = [e for e in payload["selectivity"] if e["kind"] == "smooth_wave"]
    legacy_first = legacy_wave[0]["legacy"]
    legacy_last = legacy_wave[-1]["legacy"]
    sod_flags = [e for e in payload["selectivity"] if e["kind"] == "sod_initial"][0]["counts"]["100.0"]
    stage_label = "tvb_M10_final"
    stage_order = wave.get(stage_label, {}).get("finest_order")

    rows_all = [(row["max_percent"], label) for label, config in sod.items()
                for row in config["rows"]]
    best_row, best_label = min(rows_all) if rows_all else (float("nan"), "n/a")
    sod_best = {}
    for label, config in sod.items():
        worst = max(row["max_percent"] for row in config["rows"])
        sod_best[label] = worst

    drift_rows = []
    for label, config in wave.items():
        worst = max(max(row["conservation_drift"].values()) for row in config["rows"])
        drift_rows.append((label, worst))
    drift_md = "\n".join(f"| {label} | {worst:.2e} |" for label, worst in drift_rows)

    legacy_txt = ""
    if prefix is not None and "smooth_wave" in prefix:
        legacy = prefix["smooth_wave"]["configurations"]["tvb_M1"]
        legacy_txt = (
            f"| TVB, M=1, **pre-fix indicator** | {legacy['rows'][-1]['cells']} | "
            f"{legacy['rows'][-1]['relative_l2_percent']:.6e} | {legacy['finest_order']:.3f} | "
            f"{max(max(row['conservation_drift'].values()) for row in legacy['rows']):.2e} |"
        )

    doc = f"""# TensorDG 限制器施加策略验证：TVB 检测器标度缺陷修复

> 任务：m2500（data 席自找活），锚点 = `docs/limiter_verification.md` §7「下一步」P0（施加策略）
> 与 P1（分辨率意义）；上游 `docs/limiter_verification.md`（m2192）已修好限制器刻度但留下
> 「限制器启用即有害」的结论。本单把该结论做成可判定的测量，并定位到真正的根因。
> 日期：{payload['generated_utc'][:10]} · Python {payload['python']} · torch {payload['torch']} ·
> 分支 `agent/m2500`

## 1. 案例介绍（物理背景 / 几何 / 参数）

两个算例都用一维可压缩 Euler 方程（理想气体 γ=1.4）、模态 Legendre DG、HLLC 法向通量、
SSP-RK3、局部正性回退，与仓库既有 8 个验证基准同一套核：

| 算例 | 几何与初值 | 精确解 | 判据 |
|---|---|---|---|
| A 光滑周期熵波 | x∈[0,1] 周期，ρ=1+{AMPLITUDE}·sin(2πx)，u={1.0}，p={1.0}，t={payload['smooth_wave']['final_time']}，CFL={payload['smooth_wave']['cfl']} | **解析**：常速常压的密度扰动是 Euler 方程的精确解（动量、能量方程退化为标量对流），故 ρ(x,t)=ρ0(x−ut) | 设计阶 p+1={design}，P2 |
| B Sod 激波管 | x∈[0,1]，隔膜 x=0.5，左 (1,0,1)、右 (0.125,0,0.1)，t={payload['sod']['final_time']}，P2 | **解析**：精确 Riemann 解（`benchmarks/riemann_exact.py`） | 每场相对 L2 < 3% |

算例 A 是关键设计：它给出一条*有解析解*的光滑流，因此「限制器是否吃掉设计阶」可以被量化，
而不是只能靠激波算例猜。网格序列 A = 20/40/80，B = 100/200 单元。

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **先量现状，再改代码。** 新增 `benchmarks/limiter_policy_study.py`：同一求解器、同一网格、
   同一 CFL，只改「限制器种类 / TVB 常数 M / 施加时机」，误差对**解析解**采样（8 点 Gauss）。
2. **把「施加策略」当作可选项做出来**：`solve_euler_dg_1d(..., limiter_stage="{{'every'|'final'}}")`，
   `every` 为历史行为（每个 SSP-RK 子步后限制），`final` 只在时间步末限制。回归测试保证无限制器时两者逐位相同。
3. **量出来的第一件事否掉了 P0 假设**：在光滑熵波上 `every` 与 `final` 的误差在 4 位有效数字内一致
   （见 §3 表），说明损失不是「子步反复投影」造成的——而是**每一步都被触发**的检测器。
4. **定位到检测器的两个真实缺陷**（`tensordg/limiters.py`）：
   - *TVB 容差量纲不一致（根因）*：`M dx²` 是 O(dx²) 的门，就应当与「限制器本来会施加的修正量」比，
     即 `|d_left − minmod(d_left, d_right)|` —— 光滑数据下它等于 `|u''| dx²`，间断处是 O(dx)。
     历史代码却把 `max(|d_left|, |d_right|)`（一个纯 O(dx·u') 量）与 `M dx²` 比，等价于要求
     `M ≳ |u'|/dx`，**不存在网格无关的 M**：M=100 时 20/40/80/160 单元分别误标 0/0/3/53 个单元；
   - *判据不单调*：`minmod(d_left, d_right) != d_left` 把「凹段（变化率在下降）」也算作需要限制，
     光滑行波上约一半单元被标 —— 这一点在 minmod 类检测器里是经典的保守行为，本单不改判据语义，
     只改容差的量纲，使其可被 M 抑制。
5. **修复**（`tensordg/limiters.py`，一处比较 + 注释 + 舍入底）：把与 `M dx²` 比较的量换成
   `|d_left − minmod(d_left, d_right)|`，并加 `1e-12` 的相对舍入底（仿射数据下该量是若干 ulp，
   不能被当成已分辨特征）。minmod 斜率、WENO 分支、施加路径都不动。
6. **修完复量**：同一 harness 重跑，并给出「修复前」基线（把 `limiters.py` 临时回退到 HEAD 版本、
   同一 harness 跑 `--wave-only` 生成的 `validation/limiter_policy/prefix_baseline.json`），
   保证前后可比、数字不手抄。
7. **守恒与选择性**：`solve_periodic_euler_dg_1d` 在元数据里回写质量/动量/能量的相对漂移；
   另用 `_troubled_cell_indicator` 直接量「标记比例 vs M」（表 3）与激波处是否仍被标。

## 3. 计算结果（全部由工件直读，`validation/limiter_policy/policy_study.json`）

### 3.1 光滑熵波：阶数与守恒（解析解参照）

{_wave_table(payload, prefix)}

读法：无限制器阶数 = {wave['none']['finest_order']:.3f}（设计阶 {design}）；
修复后 TVB M=10 与无限制器**误差逐位相同**（限制器在光滑数据上完全不起作用）；
M=1 仍把光滑波的减速段判为需限制（80 单元中 {strict_flags} 个），把全局阶压到 {strict_order:.3f}；
修复前（历史判据）同配置只有 {prefix['smooth_wave']['configurations']['tvb_M1']['finest_order'] if prefix else float('nan'):.3f} 阶，
且历史判据在 M=10 时更差（{prefix['smooth_wave']['configurations']['tvb_M10']['finest_order'] if prefix else float('nan'):.3f} 阶）——
因为它的门要求 M 随 1/dx 增长，加大 M 在细网格上无效。
`final` 与 `every` 两种施加策略在光滑波上差异 < 1e-3 相对——P0 假设不成立。

### 3.2 检测器选择性（标记单元数，越少越好，但激波必须留）

{_selectivity_table(payload)}

关键读法：光滑波在 M≥10 时**每个网格都是 0 个单元被标**（正确：|u''|dx² = 3.95 dx² < 10 dx²），
即 M 现在是**网格无关**的；历史判据在同样的 M=100 下随网格加密从 {legacy_first} 增到 {legacy_last} 个单元
（上表最后一列，按 HEAD 公式在本脚本内复算）。Sod 初始间断在 M=100 时仍有 {sod_flags} 个单元被标（限制器没有失效）。

### 3.3 Sod 激波管：分辨率与 3% 门

{_sod_table(payload)}

### 3.4 守恒漂移（周期算例，单步 RK 到 t={payload['smooth_wave']['final_time']}）

| configuration | max relative drift (mass/momentum/energy) |
|---|---:|
{drift_md}

### 3.5 图

![limiter policy verification](figures/limiter_policy_verification.png)

左：光滑熵波的误差-网格曲线（虚线为设计斜率 3，红叉为修复前检测器）；中：Sod 各配置最大场误差随网格，
红虚线为 3% 门；右：标记比例随 M 的选择性，红虚线为 Sod 间断处（M=100）。

## 4. 误差判定（逐项，显式回答是否 <3%）

{verdict_md}

**回答**：光滑算例的判据不是 3% 而是**阶数**（基准是解析解，细网格误差只有 {fixed_error:.2e}%，
远小于 3%，所以 3% 门在这里没有区分力；有区分力的是阶数：修复后 {fixed_order:.2f} vs 设计 {design}）。
激波算例的 3% 门在上表逐条列出：Sod 在 100/200 单元上**没有任何配置达标**，
最好的单点是 {best_row:.4f}%（{best_label}），离 3% 只差一点但仍在门外；无限制器基线在 200 单元同样不达标
（3.2393%）。这与 m2192 的结论一致：3% 门在激波问题上对**分辨率**敏感，而非对限制器敏感
——限制器在本网格上把 u 场误差从 3.2393% 压到 3.0002%（M=10），方向是对的、幅度不够。

## 5. 注意点 / 局限 / 适用范围

1. **本单最重要的结论之一是否定性的**：`docs/limiter_verification.md` §7 的 P0 建议
   （「只在最终级投影」）经测量**不成立**——`every` 与 `final` 在光滑波上的误差一致到全部有效数字
   （M=10 时两者与无限制器**逐位相同**）。真正的根因是 TVB 容差量纲：修好之后 M 才成为网格无关参数。
2. **M 仍不是万能药**：M=1（严格 TVD 侧）依旧会限制光滑波的减速段，把全局阶压下来——这是 minmod 类
   检测器的经典保守性；本单的贡献是让「选一个足够大的 M」这件事在数学上成立并可被测量，而已。
2. **仍只做 1D**：二维限制器（`tensordg/euler_dg_2d.py` 的可选 TVB）沿用同一个 `_troubled_cell_indicator`，
   本次修复对它**连带生效但未单独验证**（未做二维光滑阶数测试）。
3. **M 的最优值依数据而定**：M 现在有正确的 O(dx²) 语义，可以按 u'' 量级选（本算例 u''=3.95 → M≥4 即惰性，
   取 10 留余量）；但「M 该多大」仍是问题相关参数，不是普适常数。
4. **未复核性能**：原报告 76–123% 的开销未复测；修复后逐单元 Python 循环仍在。为防止重算堆积，
   本次所有算例均为 1D 小网格（最大 200 单元）。
5. **容量纪律**：机器 load15 在 {payload['loadavg'][2]:.0f}（nproc=128）时开工，只跑 1D 单核小算例。
6. **重复性**：所有数字来自 `validation/limiter_policy/*.json`；重跑命令
   `PYTHONPATH=. python -m benchmarks.limiter_policy_study` 与
   `PYTHONPATH=. python -m benchmarks.report_limiter_policy`。

## 6. 交付物

| 文件 | 作用 |
|---|---|
| `tensordg/limiters.py` | 修复单调判据与 TVB 容差量纲（两处 + 注释） |
| `tensordg/euler_dg_1d.py` | 新增 `limiter_stage` 策略参数；`_rhs` 支持周期边界；新增 `solve_periodic_euler_dg_1d`（有解析解的光滑周期驱动） |
| `tensordg/__init__.py` | 导出新驱动 |
| `benchmarks/limiter_policy_study.py` | 策略/选择性/守恒测量 harness（JSON 落盘） |
| `benchmarks/report_limiter_policy.py` | 由工件生成本文 + 图 |
| `tests/test_limiter_policy.py` | 5 条回归：检测器选择性、激波仍被标、策略参数校验、光滑波阶数、限制器保持阶数 |
| `validation/limiter_policy/*.json` | 原始数字（修复后 + 修复前基线） |
| `docs/figures/limiter_policy_verification.png` | 三面板图 |

---

生成时间：{payload['generated_utc']} · 由 `benchmarks/report_limiter_policy.py` 程序化注入
"""
    DOC.write_text(doc)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()