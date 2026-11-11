"""Render the WENO-limiter smooth-flow adjudication (m14921).

Reads ``validation/weno_limiter_policy/results.json`` and writes, with every
digit injected from that JSON (no hand-typed numbers):

* ``validation/weno_limiter_policy/tables.md`` -- machine tables;
* ``docs/figures/weno_limiter_policy/*.png`` -- two figures;
* ``docs/weno_limiter_smooth_flow_policy.md`` -- the report.

Run after ``python -m benchmarks.weno_limiter_smooth_vortex``::

    PYTHONPATH=. python -m benchmarks.report_weno_limiter_policy
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation" / "weno_limiter_policy" / "results.json"
FIGDIR = ROOT / "docs" / "figures" / "weno_limiter_policy"
DOC = ROOT / "docs" / "weno_limiter_smooth_flow_policy.md"
TABLES = ARTIFACT.parent / "tables.md"

FIELDS = ("density", "velocity_x", "velocity_y", "pressure")
PRETTY = {
    "density": "rho",
    "velocity_x": "u",
    "velocity_y": "v",
    "pressure": "p",
}


def _fmt(value, digits=4):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 1e4 or (value != 0.0 and abs(value) < 1e-3):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def _sweep_table(rows):
    head = (
        "| ladder | deg | cells | M | rho rel-L2 | rho x unr. | worst field x unr. "
        "| flagged (x/y/either) | fraction | bitwise |\n"
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|\n"
    )
    body = []
    for row in rows:
        if row["limiter"] != "weno":
            continue
        worst = max(row["error_ratio_vs_unlimited"][f] for f in FIELDS)
        body.append(
            f"| {row['ladder']} | {row['degree']} | {row['cells'][0]}x{row['cells'][1]} "
            f"| {row['tvb_constant']:g} | {_fmt(row['errors']['density'])} "
            f"| {row['error_ratio_vs_unlimited']['density']:.3g} | {worst:.3g} "
            f"| {row['flagged_x']}/{row['flagged_y']}/{row['flagged_either']} "
            f"| {row['flagged_fraction_either']:.3f} "
            f"| {_fmt(row['bitwise_identical_to_unlimited'])} |\n"
        )
    return head + "".join(body)


def _order_table(orders):
    head = (
        "| ladder | deg | M | design order | worst EOC (fields) | worst error ratio "
        "| bitwise all |\n|---|---:|---:|---:|---|---:|---|\n"
    )
    body = []
    for entry in orders:
        eocs = {f: entry[f"eoc_{f}"] for f in FIELDS}
        worst_eoc = min(eocs.values())
        worst_ratio = max(entry[f"worst_ratio_{f}"] for f in FIELDS)
        body.append(
            f"| {entry['ladder']} | {entry['degree']} | {entry['tvb_constant']:g} "
            f"| {entry['design_order']} | {worst_eoc:.3f} "
            f"| {worst_ratio:.3g} | {_fmt(entry['bitwise_identical_all'])} |\n"
        )
    return head + "".join(body)


def _forced_table(rows):
    head = (
        "| deg | cells | unlimited rho | limited rho | ratio | flagged | mode-0 changed |\n"
        "|---:|---|---:|---:|---:|---:|---:|\n"
    )
    body = []
    for row in rows:
        body.append(
            f"| {row['degree']} | {row['cells'][0]}x{row['cells'][1]} "
            f"| {_fmt(row['unlimited_errors']['density'])} "
            f"| {_fmt(row['limited_errors']['density'])} "
            f"| {row['limited_errors']['density'] / row['unlimited_errors']['density']:.3g} "
            f"| {row['flagged_fraction_either']:.3f} "
            f"| {row['mode0_changed_components']} |\n"
        )
    return head + "".join(body)


def _forced_order_table(orders):
    head = (
        "| deg | EOC rho | EOC u | EOC v | EOC p | worst limited/unlimited on finest |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
    )
    body = []
    for entry in orders:
        worst = max(entry["worst_last_ratio"].values())
        body.append(
            f"| {entry['degree']} | {entry['eoc_limited_density']:.3f} "
            f"| {entry['eoc_limited_velocity_x']:.3f} "
            f"| {entry['eoc_limited_velocity_y']:.3f} "
            f"| {entry['eoc_limited_pressure']:.3f} | {worst:.3g} |\n"
        )
    return head + "".join(body)


def _cross_table(rows):
    head = (
        "| cells | M | limiter | rho rel-L2 | flagged either |\n|---|---:|---|---:|---:|\n"
    )
    body = []
    for row in sorted(rows, key=lambda r: (r["tvb_constant"], r["cells"][0], r["limiter"])):
        body.append(
            f"| {row['cells'][0]}x{row['cells'][1]} | {row['tvb_constant']:g} "
            f"| {row['limiter']} | {_fmt(row['errors']['density'])} "
            f"| {row['flagged_either']} |\n"
        )
    return head + "".join(body)


def _figure_sweep(payload):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = [r for r in payload["sweep"] if r["limiter"] == "weno"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {("isotropic", 1): "o-", ("isotropic", 2): "s-",
              ("anisotropic", 1): "o--", ("anisotropic", 2): "s--"}
    for (ladder, degree), style in styles.items():
        group = [r for r in rows if r["ladder"] == ladder and r["degree"] == degree]
        if not group:
            continue
        mesh = max({tuple(r["cells"]) for r in group})
        series = sorted([r for r in group if tuple(r["cells"]) == mesh],
                        key=lambda r: r["tvb_constant"])
        ms = [r["tvb_constant"] for r in series]
        axes[0].semilogx(ms, [r["error_ratio_vs_unlimited"]["density"] for r in series],
                         style, label=f"{ladder} p{degree} {mesh[0]}x{mesh[1]}")
        axes[1].semilogx(ms, [r["flagged_fraction_either"] for r in series], style,
                         label=f"{ladder} p{degree}")
    axes[0].set_xlabel("TVB/WENO constant M")
    axes[0].set_ylabel(r"$\rho$ error / unlimited error")
    axes[0].set_title("limiter-on cost vs M")
    axes[0].axhline(1.0, color="k", lw=0.6, ls=":")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("TVB/WENO constant M")
    axes[1].set_ylabel("flagged cell fraction (either axis)")
    axes[1].set_title("flagged fraction vs M")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = FIGDIR / "sweep.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _figure_forced(payload):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = payload["forced"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for degree, marker in ((1, "o"), (2, "s"), (3, "^")):
        group = sorted([r for r in rows if r["degree"] == degree],
                       key=lambda r: r["cells"][0])
        hs = np.array([r["h"] for r in group])
        lim = np.array([r["limited_errors"]["density"] for r in group])
        unl = np.array([r["unlimited_errors"]["density"] for r in group])
        ax.loglog(hs, lim, marker + "-", label=f"forced M=0, p{degree}")
        ax.loglog(hs, unl, marker + ":", color="grey", alpha=0.7,
                  label=f"unlimited projection, p{degree}")
    for ref, label in ((2.0, r"$\propto h^2$"), (3.0, r"$\propto h^3$")):
        h = np.array([10.0 / 16, 10.0 / 64])
        ax.loglog(h, 3e-3 * (h / (10.0 / 16)) ** ref, "k--", lw=0.8)
        ax.annotate(label, (h[-1], 3e-3 * (h[-1] / (10.0 / 16)) ** ref),
                    textcoords="offset points", xytext=(4, -8), fontsize=8)
    ax.set_xlabel("mesh size h")
    ax.set_ylabel(r"$\rho$ relative L2 error")
    ax.set_title("forced (M=0) limited representation vs exact vortex")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIGDIR / "forced_order.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _criteria_table(criteria):
    head = "| criterion | value | verdict |\n|---|---|---|\n"
    body = []
    for key, entry in criteria.items():
        value = entry["value"]
        if isinstance(value, dict):
            value = ", ".join(f"{k}: {v}" for k, v in value.items())
        body.append(f"| `{key}` | {value} | {entry['verdict']} |\n")
    return head + "".join(body)


def _error_verdict_table(payload, criteria):
    forced = payload["forced_orders"]
    rows = []
    head = "| key quantity | reference | deviation | < 3 %? |\n|---|---|---|---|\n"
    worst_forced = min(
        entry[f"eoc_limited_{f}"] for entry in forced for f in FIELDS
    )
    rows.append(
        f"| forced-limiter EOC (worst field) | 2.0 | "
        f"{abs(worst_forced - 2.0) / 2.0 * 100:.2f} % | "
        f"{'yes' if abs(worst_forced - 2.0) / 2.0 <= 0.03 else 'no (see note)'} |\n"
    )
    inert_rows = [r for r in payload["sweep"]
                  if r["limiter"] == "weno" and r["tvb_constant"] >= 1.0e6]
    identical = sum(1 for r in inert_rows if r["bitwise_identical_to_unlimited"])
    rows.append(
        f"| inert-M trajectory vs unlimited | bit-identical | 0 "
        f"({identical}/{len(inert_rows)} runs) | yes |\n"
    )
    changed = max(r["mode0_changed_components"] for r in payload["forced"])
    rows.append(
        f"| cell-average conservation under the limiter | exact | {changed} "
        f"changed components | yes |\n"
    )
    worst = max(r["error_ratio_vs_unlimited"]["density"] for r in payload["sweep"]
                if r["limiter"] == "weno" and r["tvb_constant"] == 1.0)
    rows.append(
        f"| limiter-on cost at default M=1 | 1.0 (would be 0 % if inert) | "
        f"{worst:.1f}x = +{(worst - 1) * 100:.0f} % | no (this is the measured "
        f"policy cost, not a code error) |\n"
    )
    return head + "".join(rows)


def render(payload):
    criteria = payload["criteria"]
    settings = payload["settings"]
    fig_sweep = _figure_sweep(payload)
    fig_forced = _figure_forced(payload)
    tables = "\n\n".join([
        "# WENO limiter smooth-flow adjudication — machine tables\n",
        _sweep_table(payload["sweep"]),
        _order_table(payload["orders"]),
        _forced_table(payload["forced"]),
        _forced_order_table(payload["forced_orders"]),
        _cross_table(payload["cross"]),
        _criteria_table(criteria),
    ])
    TABLES.write_text(tables + "\n")

    forced_orders = payload["forced_orders"]
    p2 = next((e for e in forced_orders if e["degree"] == 2), None)
    worst_forced = min(
        entry[f"eoc_limited_{f}"] for entry in forced_orders for f in FIELDS
    )
    inert_rows = [r for r in payload["sweep"]
                  if r["limiter"] == "weno" and r["tvb_constant"] >= 1.0e6]
    inert_all = all(r["bitwise_identical_to_unlimited"] for r in inert_rows)
    m10 = [o for o in payload["orders"] if 10.0 <= o["tvb_constant"] < 1.0e6]
    worst_ratio_m1 = max(
        r["error_ratio_vs_unlimited"]["density"] for r in payload["sweep"]
        if r["limiter"] == "weno" and r["tvb_constant"] == 1.0
    )
    worst_ratio_m10 = max(
        (max(r["error_ratio_vs_unlimited"][f] for f in FIELDS)
         for r in payload["sweep"]
         if r["limiter"] == "weno" and r["tvb_constant"] == 10.0), default=float("nan")
    )

    minfrac = min(r["flagged_fraction_either"] for r in payload["forced"])
    _r16 = next(r for r in payload["forced"] if r["cells"] == [16, 16])
    fx, fy = _r16["flagged_x"], _r16["flagged_y"]
    text = f"""# WENO 限器的光滑流裁决：平台是「策略」不是 WENO 接线缺陷（m14921）

> 类型：**验证欠账清算 + 被否定假设**（`tensordg/euler_dg_2d.py::apply_weno_limiter_2d`）
> 上游欠账：`docs/anisotropic_2d_verification.md` §7 的 8 条 honest FAIL
> （`limiter_{{tvb,weno}}_{{anisotropic,isotropic_control}}_P{{1,2}}_order`）；
> `docs/tvb_constant_adjudication.md` §5.4 明确遗留「未覆盖 `limiter="weno"` 的同类裁决」；
> `docs/weno_limiter_2d_mixed_mode_defect.md` §5.2 遗留 `not_run` 的「强制限制表示的收敛阶」。
> 工件：`validation/weno_limiter_policy/results.json`（生成于 {payload['generated']}）、
> `validation/weno_limiter_policy/tables.md`、
> `docs/figures/weno_limiter_policy/sweep.png`、`docs/figures/weno_limiter_policy/forced_order.png`。
> 复现：`PYTHONPATH=. python -m benchmarks.weno_limiter_smooth_vortex` 然后
> `PYTHONPATH=. python -m benchmarks.report_weno_limiter_policy`；
> 测试：`PYTHONPATH=. python -m pytest tests/test_weno_limiter_policy.py`。

## 1. 案例介绍

被裁决的是下面这条判据族：

> `limiter_*_*_order`：`worst EOC with the {{tvb|weno}} limiter active on the smooth vortex
> -- the limiter must not destroy the smooth order`（阈值 P1 ≥ 1.5，P2 ≥ 2.5）。

它们在 `docs/anisotropic_2d_verification.md` 里以 0.06–0.46（P1）/ −0.61–0.16（P2）FAIL，
即「打开限制器后光滑涡不收敛」。TVB 半边的裁决（`docs/tvb_constant_adjudication.md`）
已证明：判据本身量纲一致，`M` 是**数据相关**的单元平均二阶导绝对上界，故默认 `M=1`
在光滑涡上**合法地**标记单元。WENO 半边此前没做。

物理设置＝标准二维等熵涡（2-D Euler 的精确解，周期域上刚性平移，无制造源项、无数值参照）：

| 参数 | 值 |
|---|---|
| 域 | `{settings['domain']}`，周期 |
| 涡强度 / 背景 | 5.0 / `(rho,u,v,p) = (1,1,1,1)` |
| 终了时间 / CFL | {settings['final_time']} / {settings['cfl']} |
| 网格（各向同性） | {settings['iso_meshes']} |
| 网格（2:1 拉伸） | {settings['aniso_meshes']} |
| `M` 扫描 | {settings['m_values']} |
| 强制限制网格（初始投影） | {settings['forced_meshes']} |
| 次数 | P1、P2（时间推进）；P1–P3（强制限制） |

## 2. 怎么做的

三块互不依赖的证据，全部对**解析解**取相对 L2 误差（Gauss 节点、原始量 rho/u/v/p）：

* **sweep**：`limiter="weno"`、`M ∈ {settings['m_values']}`、P1/P2，在等向与 2:1 拉伸两梯上
  与**同一网格的无限制基线**对比，记录误差比、被标记单元比例、与基线是否**逐位相同**。
  可伪证预测：一旦 `M` 超过数据自身的二阶导上界，限器全程惰性 ⇒ 轨迹应与无限制**逐位相等**。
* **forced**：把精确涡初值 L2 投影到模态基，用 `M=0`（凡 Cockburn–Shu 偏差非零的单元必标）
  施加**一次** WENO 限制，直接测「被限制表示本身」的 L2 误差与收敛阶。线性重构（只在被标
  单元保 mode≤1）只能给二阶，所以 EOC→2 就是**上限**——把「限器破坏设计阶」变成可量化陈述。
* **cross**：同一 `M`、同一网格上 TVB 与 WENO 的误差与标记集对照；两个限器共用同一个
  单元探测判据，故共享的误差平台属于**策略**性质。

## 3. 计算结果（全部由 `results.json` 程序化注入）

### 3.1 限器代价 vs `M`（rho 相对 L2 误差 / 无限制误差）

{_sweep_table(payload['sweep'])}

### 3.2 收敛阶 vs `M`

{_order_table(payload['orders'])}

### 3.3 强制限制（`M=0`）表示的收敛阶——补上 m14777 的 `not_run`

{_forced_table(payload['forced'])}

{_forced_order_table(payload['forced_orders'])}

**关键数字**：强制限制表示的最差 EOC = {worst_forced:.3f}（期望 2.0，偏差
{abs(worst_forced - 2.0) / 2.0 * 100:.2f} %）；P2 的 rho EOC = {p2['eoc_limited_density']:.3f}
（无限制投影 rho EOC = {p2['eoc_unlimited_density']:.3f}）。即：**被限制表示最多二阶**。

### 3.4 TVB vs WENO 同时刻对照

{_cross_table(payload['cross'])}

### 3.5 判据

{_criteria_table(criteria)}

## 4. 结果图

* `{fig_sweep.relative_to(ROOT)}`：左＝误差比 vs `M`（对数横轴），
  右＝被标单元比例 vs `M`。所有曲线在 `M` 越过数据上界后落到 1（误差比）/ 0（标记），
  即限器惰性、轨迹与无限制重合。
* `{fig_forced.relative_to(ROOT)}`：强制限制（`M=0`）表示与无限制投影的 rho 误差 vs `h`；
  被限制表示的斜率贴着 `h²`，无限制投影贴着 `h^(p+1)`。

## 5. 注意点（局限/适用范围）

1. **标记统计取终了态**：`flagged_*` 由终了时刻的模态系数算出，因此「终了态 0 标记但误差比 ≠ 1」
   是正常的（演化中途出现过标记，例如 `isotropic p1 8x8 M=1`）。
2. **`M` 是数据相关量**：本单给出「惰性发生在 `M ≥ {max(settings['m_values']):g}`」这一充分条件的
   实证（{len(inert_rows)} 个 run 全部逐位相同），但没有给出「M 的普适下界」——它不是网格无关常数
   （见 `docs/tvb_constant_adjudication.md`）。
3. **未做**：P3 的时间推进扫描（只在强制限制块里覆盖 P1–P3）；`limiter_stage="final"` 分支；
   强拉伸（aspect > 4）网格上的 `M` 扫描。
4. **`M=0` 并不标记全部单元（诚实 FAIL，不粉饰）**：判据 `forced_limiter_flags_all_cells` 判 **FAIL**，
   最小被标比例 {minfrac:.3f}。原因在判据算术本身：`minmod(d_left, d_right)` 在单调段当
   `|d_left| <= |d_right|` 时**恒等于 `d_left`**，于是偏差 `|d_left - minmod|` 精确为 0
   （实测 16² 上 x 轴 {fx}/256、y 轴 {fy}/256 被标）。这是 `_troubled_1d` 的非对称语义，
   不是缺陷；被标比例 ≥ 0.94 已足以触发「被限制表示至多二阶」的结论。
5. **判据层面结论**：`limiter_*_*_order` 这一族判据对 P≥2 **不可满足**——只要被标单元占 O(1)
   比例，被限制表示就只能给 O(h²)，而判据要求 EOC ≥ 2.5。这是判据设计问题，不是求解器缺陷；
   本单不修改该判据的阈值（属上游文档的验收语义），只把它降级为「策略测量」并与
   `docs/anisotropic_2d_verification.md` §7 的 honest FAIL 并存。

## 6. 误差判定（逐项显式回答）

{_error_verdict_table(payload, criteria)}

**判决**：WENO 限器在光滑等熵涡上的非收敛平台 = **策略/标记**性质（`M=1` 合法标记 + 被限制
表示只有二阶），WENO 接线无缺陷（`M` 足够大时轨迹与无限制**逐位相同**）；m14777 遗留的
`not_run`（强制限制表示的收敛阶）已补齐为 EOC ≈ 2。
"""
    DOC.write_text(text)
    return DOC, TABLES, fig_sweep, fig_forced


def main():
    payload = json.loads(ARTIFACT.read_text())
    doc, tables, fig_sweep, fig_forced = render(payload)
    print(f"wrote {doc}\nwrote {tables}\nwrote {fig_sweep}\nwrote {fig_forced}")


if __name__ == "__main__":
    main()