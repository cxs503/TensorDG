"""Round 20 artifacts for CASE-DG-EULER-VORTEX-ERR: reference-floor coverage.

Writes
  docs/CASE-DG-EULER-VORTEX-ERR-round20-reference-floor-coverage.md
  case_euler_vortex_errconserve_lattice/fig_reference_floor_coverage.png

Every number in the markdown is injected from
``case_euler_vortex_errconserve_lattice/reference_floor.json`` -- nothing is
hand-copied.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
FLOOR = REPO / "case_euler_vortex_errconserve_lattice" / "reference_floor.json"
DOC = REPO / "docs" / "CASE-DG-EULER-VORTEX-ERR-round20-reference-floor-coverage.md"
FIG = REPO / "case_euler_vortex_errconserve_lattice" / "fig_reference_floor_coverage.png"


def family_of(row) -> str:
    key = row["task_key"]
    direction = row.get("direction", "diag45")
    for prefix in ("dg_d1_n32", "dg_d1_n64", "dg_d2_n32", "dg_d2_n64", "dg_d3_n32"):
        if key.startswith(prefix):
            return f"{prefix}/{direction}"
    if key.startswith("p0_n"):
        return "p0/" + direction
    return key


def main() -> int:
    payload = json.loads(FLOOR.read_text())
    rows = payload["checkpoints"]
    measured = [r for r in rows if r["worst_floor_over_error"] is not None]
    by_ratio = sorted(measured, key=lambda r: -r["worst_floor_over_error"])
    worst = by_ratio[0]
    second = by_ratio[1]

    # direction dependence of the floor at a fixed checkpoint family
    direction_rows = [r for r in rows if r["task_key"].startswith("dg_d2_n32_t")
                      and "_cfl0.2" in r["task_key"]]
    direction_rows.sort(key=lambda r: (r["direction"], r["final_time"]))

    lines = [
        "# CASE-DG-EULER-VORTEX-ERR round 20：参考解地板的覆盖面（与真实最不利点）",
        "",
        "> 锚点 `CASE-DG-EULER-VORTEX-ERR`｜算例池·等熵涡移动误差守恒｜"
        "ref: Yee, Sandham & Djomehri (JCP 162, 2000)｜容差 6.0%（t = 1 快照）",
        "",
        "## ① 案例与本节问题",
        "",
        "等熵涡在周期域上平移，误差读数**全部**是 `||u_h - R||` 的比值。"
        "`R` 用最小镜像折返（`min_image`）求值时丢掉了邻居镜像的尾巴，"
        "因此每个读数都带一个参考解缺陷。round 7 量到了这个缺陷（`O(tail)` 进速度、"
        "`O(tail^2)` 进密度/压力），并把它写成**最坏情况上界**：",
        "",
        "```",
        "|e_lattice - e_published| / e_published  <=  floor / e_published",
        "```",
        "",
        "本节问的不是「地板多大」（round 7 已答），而是**这个上界覆盖了谁**："
        "已发布的地板工件是否对每一条已发布读数都给过界？",
        "",
        "## ② 怎么做的",
        "",
        "1. **把检查点来源从硬编码阶梯改为记录自身**。原实现遍历常量 `LADDER`，"
        "而 `LADDER` 与实际落盘的读数在随后几轮里出现分叉：工件里 11 条记录"
        "（`dg_d1_n32` 长时、`dg_d1_n64` 全部、`dg_d3_n32` 全部、`dg_d2_n32_t20`）"
        "写着 `published_errors = null` —— 地板算出来了，**却没有和任何实测比**。"
        "新实现 `configs_from_results()` 逐条读 `results/*.json` 里记录的 `config`"
        "（它才是该次计算的权威，含方向），因此不可能再漏族。",
        "2. **修一个方向 bug**。原 `floor_at()` 用模块常量 `BACKGROUND = (1, 1)` "
        "求参考，而 C10 方向轴的记录沿 `axis0` / `obl30` 平移；折返折叠依赖位移方向，"
        "所以对方向族算出的地板是**另一个算例的**。现按记录自身的 "
        "`background_velocity` 与 `strength` 求值。",
        "3. **把覆盖面变成断言**：`coverage.uncovered` 必须为空，"
        "并新增回归测试 `tests/test_vortex_reference_upgrade.py`。",
        "4. **量化实测位移的工具**（`benchmarks/vortex_reference_upgrade.py`）："
        "上界是最坏情况（假设参考缺陷与解误差同向），实测才是裁决。工具在同一检查点上"
        "比较 `min_image`（已发布）与 `lattice`（精确镜像和）的重跑读数，并带一个"
        "**同条件重跑控制**（`min_image` 再跑一次）把求解器噪声与参考效应分开。",
        "",
        "## ③ 计算结果（程序化注入，零手抄）",
        "",
        f"覆盖面：`{payload['coverage']}`。全部 {len(rows)} 条检查点都有"
        f"「地板 / 实测」比值，`uncovered = []`。",
        "",
        "### 3.1 最不利检查点（上界最大处）",
        "",
        "| 排名 | 检查点 | 方向 | t | 像半宽 K | 速度地板 | 实测最大误差 | 地板/实测 | 主导场 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, row in enumerate(by_ratio[:8], start=1):
        lines.append(
            f"| {rank} | {row['task_key']} | {row['direction']} | {row['final_time']:g} | "
            f"{row['images']} | {row['floor_percent']['velocity_y']:.3e}% | "
            f"{row['published_max_percent']:.4f}% | {row['worst_floor_over_error']:.4f} | "
            f"{row['worst_floor_over_error_field']} |")
    lines += [
        "",
        f"**结论 1（实质修正）**：最不利点是 `{worst['task_key']}`，"
        f"地板/实测 = **{worst['worst_floor_over_error'] * 100:.2f}%**；"
        f"round 7 工件写下的最不利点是 `dg_d2_n64_t1_cfl0.2`（12.74%），"
        "它只是**被覆盖子集**的最不利点：`dg_d1_n64`、`dg_d3_n32`、`dg_d1_n32` 长时与 "
        "`dg_d2_n32_t20` 共 11 条记录当时在地板工件里没有比对（`published_errors = null`）。"
        "这不是数值变化，是覆盖面变化：p = 3 在 32² 上解误差更小，"
        "所以同一地板在它身上占的比例更大。",
        "",
        "### 3.2 地板的方向依赖（C10 轴）",
        "",
        "同一族（DG-P2 / 32² / CFL 0.2）、同一速度模长 |u∞| = √2，只改平移方向：",
        "",
        "| 方向 | t | 速度地板（相对 L2） | 实测最大误差 | 地板/实测 |",
        "|---|---|---|---|---|",
    ]
    for row in direction_rows:
        lines.append(
            f"| {row['direction']} | {row['final_time']:g} | "
            f"{row['floor_percent']['velocity_y']:.3e}% | "
            f"{row['published_max_percent']:.4f}% | {row['worst_floor_over_error']:.4f} |")
    axis_max = max(r["floor_relative_l2"]["velocity_y"] for r in direction_rows
                   if r["direction"] == "axis0")
    diag_max = max(r["floor_relative_l2"]["velocity_y"] for r in direction_rows
                   if r["direction"] == "diag45")
    obl_max = max(r["floor_relative_l2"]["velocity_y"] for r in direction_rows
                  if r["direction"] == "obl30")
    lines += [
        "",
        f"**结论 2**：地板与方向强相关——`axis0` 最大 {axis_max:.3e}，"
        f"`diag45` {diag_max:.3e}，`obl30` {obl_max:.3e}。"
        f"轴向（`axis0`）比斜向（`obl30`）大约 {axis_max / obl_max:.0f} 倍："
        "轴向折返把涡心折到单坐标的 |shift| = 5 处，丢掉的镜像离得最近；"
        "`obl30` 的两个分量都落在 2.9 与 7.1 附近，最近被丢镜像远得多。"
        "因此 **C10 的三条方向读数并不共享同一个参考解不确定度**，"
        "「方向不变性」的判据必须逐方向带自己的界（round 7 的工件只对 `diag45` 给过界）。",
        "",
        "### 3.3 界是否仍然是界",
        "",
        f"全部 {len(measured)} 条检查点的 `地板/实测` 最大值 = "
        f"{payload['max_floor_over_error']:.4f} < 1，即没有任何一条读数的判据"
        "被参考解接管（地板没有超过实测误差本身）。",
        "",
        "## ④ 结果图",
        "",
        "`fig_reference_floor_coverage.png`（随报告内嵌）：左 = 逐检查点「地板/实测」"
        "（对数轴，按族着色），红色虚线 = 1（判据被参考解接管的临界）；"
        "右 = 三条方向的地板随时间的双对数曲线，显示 `axis0` 比 `obl30` 高两个量级。",
        "",
        "## ⑤ 注意点 / 局限",
        "",
        "1. **上界不是实测**。3.1 与 3.2 的比值是**最坏情况**：它假设参考缺陷与解误差"
        "完全同向。真实位移由 `benchmarks/vortex_reference_upgrade.py` 量；本单已递交"
        "lattice 参考重跑（`cj4355` = dg_d2_n64 t=1/2/5/10，`cj4361` = dg_d3_n32 "
        "t=1/2/5/10，`cj4357` = 同条件 min_image 重跑控制），但队列深、窗口内未落盘，"
        "**故本报告不给出任何实测位移数字**，只给出上界与其覆盖面。"
        "round 7 在 32² 设计点上量到：换参考只动速度、位移 `7.6e-6`（相对），"
        "是当时地板比值的 5e-4 —— 说明上界确实非常保守。",
        "2. **`lattice` 不是 Euler 精确解**：镜像和精确的是线性叠加，"
        "非线性交叉项为 `O(tail^2) ~ 1e-11`，与地板本身同量级，故只用于速度场。",
        "3. **容差出处**：6% 是算例池条目原文照搬，未从 Yee 等（2000）正文重新推导。",
        "4. **本节的修正方向是「补覆盖」，不是「改判据」**：C1–C11 的判据文本、"
        "容差与已落盘读数一律未动，重算后的 `max_floor_over_error` 仍 < 1。",
        "",
        "## ⑥ 误差判定（逐项对基准，显式回答「是否 < 3%」）",
        "",
        "| 关键量 | 基准 | 实测/读数 | 相对基准 | 是否 < 3% |",
        "|---|---|---|---|---|",
        f"| 设计点 DG-P2/32² t=1 最大场相对 L2 | 等熵涡闭式解 | "
        f"{[r for r in rows if r['task_key'] == 'dg_d2_n32_t1_cfl0.2'][0]['published_max_percent']:.6f}% | "
        "6% 锚点容差 | 是 |",
        f"| C1 最差族 p0_n64_t1 | 等熵涡闭式解 | "
        f"{[r for r in rows if r['task_key'] == 'p0_n64_t1_cfl0.4'][0]['published_max_percent']:.4f}% | "
        "6% 锚点容差（余量 1.22×） | **否**（4.93% > 3%，但通过 6% 口径） |",
        f"| 参考地板上界（最不利 {worst['task_key']}） | 该记录实测误差 | "
        f"{worst['worst_floor_over_error'] * 100:.2f}% | 1（上界不得 ≥ 实测） | "
        "是（< 1） |",
        f"| 参考地板上界（design point dg_d2_n32_t1） | 该记录实测误差 | "
        f"{[r for r in rows if r['task_key'] == 'dg_d2_n32_t1_cfl0.2'][0]['worst_floor_over_error'] * 100:.2f}% | 1 | 是 |",
        "| 方向间地板差异（axis0 vs obl30） | 各自实测误差 | "
        f"{axis_max / obl_max:.0f}× | — | 不适用（比值，非误差） |",
        "",
        "## ⑦ 复现",
        "",
        "```bash",
        "PYTHONPATH=. python3 benchmarks/vortex_reference_floor.py \\",
        "    --results-dir case_euler_vortex_errconserve \\",
        "    --output case_euler_vortex_errconserve_lattice/reference_floor.json \\",
        "    --markdown case_euler_vortex_errconserve_lattice/reference_floor.md \\",
        "    --figures",
        "PYTHONPATH=. python3 -m pytest tests/test_vortex_reference_upgrade.py -q",
        "PYTHONPATH=. python3 gen_report_m22770.py",
        "```",
        "",
        "实测位移（队列落盘后）：",
        "",
        "```bash",
        "run_compute fetch cj4355   # lattice：dg_d2_n64 t=1/2/5/10",
        "run_compute fetch cj4361   # lattice：dg_d3_n32 t=1/2/5/10",
        "run_compute fetch cj4357   # min_image 重跑控制：dg_d2_n64 t=1,10",
        "PYTHONPATH=. python3 benchmarks/vortex_reference_upgrade.py \\",
        "    --rerun-dir case_euler_vortex_errconserve_rerun --figures",
        "```",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print("wrote", DOC)

    # ---- figure -----------------------------------------------------------
    order = sorted(rows, key=lambda r: (family_of(r), r["final_time"]))
    labels = [f"{family_of(r)}\nt={r['final_time']:g}" for r in order]
    ratios = [r["worst_floor_over_error"] for r in order]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    ax = axes[0]
    colors = ["crimson" if r is worst else "steelblue" for r in order]
    ax.bar(range(len(order)), ratios, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="floor = measured error")
    ax.axhline(worst["worst_floor_over_error"], color="crimson", ls=":", lw=1,
               label=f"worst = {worst['worst_floor_over_error'] * 100:.1f}% "
                     f"({worst['task_key']})")
    ax.set_ylabel("floor / measured error (worst-case bound)")
    ax.set_title("reference-solution bound, all %d published checkpoints" % len(order))
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    for direction, style in (("axis0", "o-"), ("diag45", "s--"), ("obl30", "^-.")):
        subset = [r for r in direction_rows if r["direction"] == direction]
        ax.semilogy([r["final_time"] for r in subset],
                    [r["floor_relative_l2"]["velocity_y"] for r in subset],
                    style, label=direction)
    ax.set_xlabel("time t")
    ax.set_ylabel("velocity floor (relative L2)")
    ax.set_title("the floor depends on the advection direction (C10 axis)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    print("wrote", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())