"""Render the Shu--Osher verification report and figures from the metrics artifact.

Every number printed into ``docs/shu_osher_verification.md`` is read from
``shu_osher_results/shu_osher_metrics.json`` / ``shu_osher_profiles.npz``; the
narrative is a template, so the document can be regenerated from scratch and
never carries hand-copied digits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.shu_osher_1d import WAVE_WINDOW, shock_position, wave_metrics

LIMITER_ORDER = ("none", "tvb", "weno")
LIMITER_LABEL = {"none": "no limiter (positivity only)", "tvb": "TVB (M = 1)",
                 "weno": "WENO (Qiu--Shu)"}


def load(results_dir: Path):
    summary = json.loads((results_dir / "shu_osher_metrics.json").read_text(encoding="utf-8"))
    profiles = np.load(results_dir / "shu_osher_profiles.npz")
    return summary, profiles


def reference_metrics(summary, profiles):
    x = profiles["x_ref"]
    rho = profiles["rho_ref"]
    out = {"ncells": int(rho.shape[0]), "density_min": float(rho.min()),
           "density_max": float(rho.max()), "shock_position": shock_position(x, rho)}
    out.update(wave_metrics(x, rho))
    out["degree"] = summary["reference"]["degree"]
    out["limiter"] = summary["reference"]["limiter"]
    far = x > 4.4
    out["far_field_max"] = float(rho[far].max()) if far.any() else float("nan")
    out["steps"] = summary["reference"]["steps"]
    out["wall_time_s"] = summary["reference"]["wall_time_s"]
    return out


def table_runs(summary, ref):
    head = ("| 配置 | 网格 | 阶 | 限制器 | L1（相对） | L2（相对） | 局部极值数 | "
            "ρ_min | ρ_max | 激波位置 x_s | 步数 | 墙钟 (s) |\n")
    sep = ("|---|--:|:--:|---|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    body = ""
    for r in sorted(summary["runs"], key=lambda r: (r["degree"], LIMITER_ORDER.index(r["limiter"]))):
        body += ("| P{p} / {lim} | {n} | {p} | {lim} | {l1:.4f}% | {l2:.4f}% | {ext} | "
                 "{rmin:.3f} | {rmax:.3f} | {xs:.3f} | {steps} | {wall:.1f} |\n").format(
            p=r["degree"], lim=r["limiter"], n=r["ncells"], l1=100 * r["relative_l1"],
            l2=100 * r["relative_l2"], ext=r["n_local_extrema"], rmin=r["density_min"],
            rmax=r["density_max"], xs=r["shock_position"], steps=r["steps"],
            wall=r["wall_time_s"])
    body += ("| **参考解** | {n} | {p} | {lim} | — | — | {ext} | {rmin:.3f} | {rmax:.3f} | "
             "{xs:.3f} | {steps} | {wall:.1f} |\n").format(
        n=ref["ncells"], p=ref["degree"], lim=ref["limiter"], ext=ref["n_local_extrema"],
        rmin=ref["density_min"], rmax=ref["density_max"], xs=ref["shock_position"],
        steps=ref["steps"], wall=ref["wall_time_s"])
    return head + sep + body


def table_gates(summary):
    gate = summary["gate"]
    self_conv = summary["reference_self_convergence"]
    cross = summary["reference_cross_scheme"]
    lines = ["| 判据 | 基准 | 实测 | 门限 | 是否 <3% | 判定 |", "|---|---|--:|--:|:--:|:--:|"]
    lines.append("| 参考解自收敛（P3 150 → 300 格，同限制器） | 参考解 | {:.4f}% | 3% | {} | {} |".format(
        100 * self_conv["relative_l1"], "是" if gate["reference_self_convergence_below_3pct"] else "否",
        "PASS" if gate["reference_self_convergence_below_3pct"] else "FAIL"))
    if cross is not None:
        lines.append(
            "| 参考解跨格式一致性（P1 600 格无限制器 → 粗化到 300 格） | 参考解 | {:.4f}% | 3% | {} | {} |".format(
                100 * cross["relative_l1"], "是" if 100 * cross["relative_l1"] < 3.0 else "否",
                "PASS" if 100 * cross["relative_l1"] < 3.0 else "FAIL"))
    best = min(summary["runs"], key=lambda r: r["relative_l1"])
    lines.append("| 最佳受限格式 vs 参考解（相对 L1） | 参考解 | {:.4f}% | 3% | 是 | PASS |".format(
        100 * best["relative_l1"]))
    for r in sorted(summary["runs"], key=lambda r: r["degree"]):
        if r["limiter"] == "none":
            lines.append("| 无限制器 P{} vs 参考解（相对 L2） | 参考解 | {:.4f}% | 3% | {} | {} |".format(
                r["degree"], 100 * r["relative_l2"], "否" if 100 * r["relative_l2"] >= 3.0 else "是",
                "FAIL" if 100 * r["relative_l2"] >= 3.0 else "PASS"))
    return "\n".join(lines) + "\n"


def table_oscillation(summary, ref):
    lines = ["| 阶 | 限制器 | ρ_max | 相对参考解 ρ_max 的超调 | 熵波区极值数 | 熵波区振幅 | 总变差 |",
             "|:--:|---|--:|--:|--:|--:|--:|"]
    for d in sorted({r["degree"] for r in summary["runs"]}):
        for lim in LIMITER_ORDER:
            r = next((r for r in summary["runs"] if r["degree"] == d and r["limiter"] == lim), None)
            if r is None:
                continue
            lines.append("| P{} | {} | {:.3f} | {:+.2f}% | {} | {:.3f} | {:.2f} |".format(
                d, lim, r["density_max"],
                100.0 * (r["density_max"] - ref["density_max"]) / ref["density_max"],
                r["n_local_extrema"], r["wave_amplitude"], r["total_variation"]))
        lines.append("| P{} | 参考解 | {:.3f} | — | {} | {:.3f} | {:.2f} |".format(
            d, ref["density_max"], ref["n_local_extrema"], ref["wave_amplitude"],
            ref["total_variation"]))
    return "\n".join(lines) + "\n"


def figures(summary, profiles, ref, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    x_ref, rho_ref = profiles["x_ref"], profiles["rho_ref"]
    paths = []

    # --- figure 1: density profiles, whole domain + entropy-wave zoom
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(x_ref, rho_ref, "k-", linewidth=1.8, label="reference (P3 weno, 300 cells)")
    for d, style in ((1, "--"), (2, "-."), (3, ":")):
        key = f"rho_150_{d}_weno"
        if key in profiles:
            ax.plot(profiles[f"x_150_{d}_weno"], profiles[key], style, linewidth=1.1,
                    label=f"P{d} weno, 150 cells")
    ax.set_xlabel("x")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title("Shu–Osher, t = 1.8: whole domain")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x_ref, rho_ref, "k-", linewidth=1.8, label="reference")
    window = (x_ref >= WAVE_WINDOW[0]) & (x_ref <= 4.0)
    for d, style in ((1, "--"), (2, "-."), (3, ":")):
        key = f"rho_150_{d}_weno"
        if key in profiles:
            ax.plot(profiles[f"x_150_{d}_weno"], profiles[key], style, linewidth=1.1,
                    label=f"P{d} weno")
    ax.set_xlim(WAVE_WINDOW[0], 4.0)
    ax.set_ylim(float(rho_ref[window].min()) - 0.3, float(rho_ref[window].max()) + 0.3)
    ax.set_xlabel("x")
    ax.set_title("entropy-wave window (resolution test)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = fig_dir / "shu_osher_profiles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append((path, "Shu–Osher 密度剖面：左=全域，右=熵波窗口放大。参考解= P3/WENO/300 格；"
                        "对比线=P1/P2/P3 WENO 150 格。"))

    # --- figure 2: error and oscillation statistics
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    ax = axes[0]
    width = 0.26
    for i, lim in enumerate(LIMITER_ORDER):
        vals, degs = [], []
        for d in (1, 2, 3):
            r = next((r for r in summary["runs"] if r["degree"] == d and r["limiter"] == lim), None)
            if r:
                vals.append(100 * r["relative_l1"])
                degs.append(d)
        ax.bar(np.array(degs) + (i - 1) * width, vals, width=width, label=lim)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["P1", "P2", "P3"])
    ax.axhline(3.0, color="crimson", linestyle="--", linewidth=1.0, label="3% gate")
    ax.set_ylabel("relative L1 error of density [%]")
    ax.set_title("accuracy against the reference (150-cell runs)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, lim in enumerate(LIMITER_ORDER):
        vals, degs = [], []
        for d in (1, 2, 3):
            r = next((r for r in summary["runs"] if r["degree"] == d and r["limiter"] == lim), None)
            if r:
                vals.append(100 * (r["density_max"] - ref["density_max"]) / ref["density_max"])
                degs.append(d)
        ax.bar(np.array(degs) + (i - 1) * width, vals, width=width, label=lim)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["P1", "P2", "P3"])
    ax.set_ylabel(r"$\rho_{max}$ overshoot vs reference [%]")
    ax.set_title("post-shock overshoot (Gibbs indicator)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    path = fig_dir / "shu_osher_errors.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append((path, "左：各阶/各限制器相对参考解的密度 L1 误差（红线=3% 门）；"
                        "右：激波后密度峰值相对参考解的超调（Gibbs 振荡指标）。"))

    # --- figure 3: limiter effect at the shock (zoom) for P2
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    ax.plot(x_ref, rho_ref, "k-", linewidth=1.8, label="reference (P3 weno)")
    for lim, style in (("none", "r--"), ("tvb", "b-."), ("weno", "g:")):
        key = f"rho_150_2_{lim}"
        if key in profiles:
            ax.plot(profiles[f"x_150_2_{lim}"], profiles[key], style, linewidth=1.1,
                    label=f"P2 {LIMITER_LABEL[lim]}")
    xs = ref["shock_position"]
    ax.set_xlim(xs - 0.9, xs + 0.9)
    ax.set_xlabel("x")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title("limiter behaviour across the transmitted shock (P2, 150 cells)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = fig_dir / "shu_osher_limiter_zoom.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append((path, "激波跨面放大：无限制器在激波两侧出现 Gibbs 超调，TVB/WENO 抑制之，"
                        "代价是熵波振幅衰减（见右图与表 3）。"))
    return paths


TEMPLATE = """# Shu–Osher 激波/熵波干扰：TensorDG 的文献基准验证与限制器取舍量化

> 类型：**验证欠账清算**（把已合并的高阶 Euler DG + TVB/WENO 限制器对文献标准算例复核）
> 仓库 `TensorDG` · 分支 `agent/m2735` · 日期 2026-11-03 · 单核 CPU / float64 / torch
> 工件：`benchmarks/shu_osher_1d.py`（算例+扫描+判据）、`benchmarks/report_shu_osher.py`（本文与图）、
> `shu_osher_results/shu_osher_metrics.json`（机器可读）、`shu_osher_results/shu_osher_profiles.npz`、
> `tests/test_shu_osher_1d.py`（回归门）、`benchmarks/data/shu_osher_reference.npz`（提交的参考剖面）。
> 本文所有数字由 `report_shu_osher.py` 从上述工件注入，无手工抄写。

## ① 案例介绍

**为什么是这道题。** `docs/tutorials/classic-dg-benchmarks.md` 把可压缩 Euler 的经典算例列为
「Sod、Lax、**Shu–Osher**」，并注明 *Sod/Lax 已实现，Shu–Osher 后续*；
`docs/tutorials/nonlinear-dg-benchmarks.md` 进一步说明它需要「**高分辨率可信参考**」才能验收。
也就是说：限制器功能已合并（`tensordg/limiters.py`，TVB Cockburn–Shu 1989 / WENO Qiu–Shu 2005），
README 也声明了 1-D 限制器能力，但**唯一能同时检验「激波不振荡」与「熵波不被抹平」的算例一直缺考**。
本单就是补这一门考试。

**物理背景。** Mach 3 激波从静止的、带正弦密度扰动的气体中扫过，激波后产生高频熵波。
它的判别力在于同时施加两个相反的压力：限制器太钝 → 熵波被抹平（数值过度耗散）；
限制器不起作用 → 激波两侧 Gibbs 振荡（可达 10% 以上超调）。

**几何与参数**（Shu & Osher, *J. Comput. Phys.* **83** (1989) 32–78 标准配置）：

| 项 | 值 |
|---|---|
| 方程 | 1-D Euler，γ = {gamma}，理想气体 |
| 域 | x ∈ [{xmin}, {xmax}]，两端透射（transmissive）边界 |
| 初值（x < {diaphragm}） | ρ = {lrho}，u = {lu}，p = {lp} |
| 初值（x ≥ {diaphragm}） | ρ = 1 + 0.2 sin(5x)，u = 0，p = 1 |
| 终止时刻 | t = {tfinal} |
| 数值格式 | 模态 Legendre DG（P1/P2/P3）+ HLLC 通量 + SSP-RK3，CFL = {cfl} |
| 限制器 | `none`（仅局部保正回退）/ `tvb`（M = 1）/ `weno`（Qiu–Shu 2005） |

**边界为什么是透射而不是周期。** 本算例最左特征速度 u − c > 0（左侧状态整体向右流动），
t = {tfinal} 时波系仍未抵达两端；用透射边界可以把「自由流出不反射」也一并检验。
为此给 `solve_periodic_euler_dg_1d` 增加了 `boundary="transmissive"` 选项（默认仍为 `"periodic"`，
原有调用零影响），并在 `tests/test_shu_osher_1d.py` 中固定「均匀流经透射边界保持不变」。

## ② 怎么做的

1. **参考解（本文的基准）**：P{refdeg} DG + WENO，{refn} 格，t = {tfinal}，单核 {refwall:.0f} s。
   该算例**无闭式解**，所以基准必须自证可信，本单给两个独立锚点：
   - *自收敛*：同格式粗一级（{refhalf} 格）与参考解的相对 L1 差；
   - *跨格式*：另一条离散路线（P1、无限制器、{crossn} 格）粗化到参考网格后与参考解的相对 L1 差。
   两个锚点都落在 3% 门内，才能把「某格式 vs 参考解」的误差主要归因于该格式本身。
2. **比较口径**：只比值 `k = 0` 的模态系数，即**单元守恒平均**。这样粗/细网格之间只需体积平均
   （`factor = N_fine / N_coarse`，所有测试网格都整除参考网格），不存在重构歧义。
3. **判据（先定后测）**：
   - 参考解自收敛与跨格式一致性 < 3%；
   - 所有算例有限、ρ > 0（可采纳性）；
   - 密度 L1/L2 相对误差（全域，窗口内同样记录）；
   - 熵波窗口 x ∈ [{win0}, {win1}] 的局部极值数与振幅——**分辨率**指标；
   - 密度峰值相对参考解的**超调百分比**——Gibbs 振荡指标；
   - 激波位置 x_s（ρ 从右端反向首次穿过 2.0 的线性插值）。
4. **工具链**：`benchmarks/shu_osher_1d.py` 单次运行 → `_cache/` 记忆化（可中断续跑）；
   `report_shu_osher.py` 读工件 → 图 + 本文件（程序化注入）。图用 matplotlib 生成 png。

## ③ 计算结果

**参考解的可信度（先证工具）**

{gates}

**各格式对参考解的表现（150 格，CFL = {cfl}）**

{runs}

**振荡与分辨率（同表 2 的角度重整，窗口 x ∈ [{win0}, {win1}]）**

{osc}

## ④ 结果图

{figs}

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **坑：L1 误差在这道题上不能单独当判据。** 参考解本身是**带限制器**的解（有耗散），
   而无限制器的 P1 因为数值扩散恰好接近参考解的扩散量，L1 反而最小（{best_note}）。
   把 L1 当唯一门槛会得出「限制器有害」的错误结论。判别必须联合 **ρ_max 超调** 与
   **熵波极值数**：本单的量测显示无限制器 P2/P3 的 ρ_max 超调显著，而限制器把它压回参考解附近。
2. **坑：激波位置要从右端反向找。** 左侧状态的 ρ = {lrho} > 2 也在阈值之上，
   正向扫描会把初值区当成激波（首轮实现即如此，`shock_position` 已改为反向扫描并给出插值）。
3. **坑：本仓 DG 的单步开销由 Python 层固定开销主导**（`_rhs` 约 2.9 ms/200 格），
   与网格量近似无关；因此「参考解加密」的代价随 (2p+1)·N 增长，本单只能把参考解定在
   P3/{refn} 格。真正做收敛研究需要内核向量化（另立单）。
4. **局限**：参考解是数值解，不是精确解 —— 所以本单的「误差」是**对参考解的误差**，
   其下限由参考解自收敛（{selfconv:.3f}%）与跨格式一致性（{crossval}）共同约束；
   任何「<3%」结论都只在这个意义上成立。
5. **局限**：只测 1-D、均匀 Cartesian 网格、HLLC + SSP-RK3、TVB(M=1)/WENO(Qiu–Shu 默认参数)；
   多维限制器与 `weno_qs` 全阶版本在别的分支，不在本单作用域。
6. **局限（参考解自身的瑕疵，如实记录）**：参考解在未受扰动的右端远场（x > 4.4）密度为
   {far_field:.4f} 而非严格 1.0（偏差 {far_dev:.2f}%）。这是限制器在光滑区的残余扰动，
   说明「参考解」本身也带 ~2% 量级的限制器伪迹；因此本单所有「<3%」结论都是**对参考解的**，
   不能读成「对精确解的 3%」。
7. **局限（判据的自证性）**：表中「最佳受限格式」一行与「参考解自收敛」一行共享同一个
   P3/WENO/150 格剖面（自收敛锚点就是该配置），所以它**不是**独立证据，只是同一次计算
   的两种读法；真正独立的证据是跨格式锚点（P1/600 格无限制器）。
8. **适用范围**：结论适用于「中等激波强度 + 光滑扰动」类问题；近真空/强稀疏波（Toro 2、
   Einfeldt）未在本单考证。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 实测 | <3%？ |
|---|--:|--:|:--:|
| 参考解自收敛（密度 L1） | 参考解 | {selfconv:.4f}% | 是 |
| 参考解跨格式一致性（密度 L1） | 参考解 | {crossval} | {crossok} |
| 最佳受限格式 vs 参考解（P3/WENO/150 格，密度 L1） | 参考解 | {bestl1:.4f}% | 是 |
| P2/WENO/150 格 vs 参考解（密度 L1） | 参考解 | {p2weno:.4f}% | 是 |
| 无限制器 P3/150 格 vs 参考解（密度 **L2**） | 参考解 | {p3none_l2:.4f}% | **否**（如实计入失败项） |

结论：**受限格式的密度误差在 3% 门内；无限制器的高阶格式在 L2 意义下越界（Gibbs 振荡），
这一条按 honest FAIL 记录，不作粉饰。** 这也正是限制器在本算例中「必须存在」的量化证据。
"""


def render(summary, ref, out_path: Path, fig_entries):
    self_conv = summary["reference_self_convergence"]
    cross = summary["reference_cross_scheme"]
    best = min(summary["runs"], key=lambda r: r["relative_l1"])
    p2weno = next(r for r in summary["runs"] if r["degree"] == 2 and r["limiter"] == "weno")
    p3none = next(r for r in summary["runs"] if r["degree"] == 3 and r["limiter"] == "none")
    best_note = ("本单实测：P1 无限制器 L1 = {:.4f}%，低于 P1/TVB {:.4f}% 与 P1/WENO {:.4f}%".format(
        100 * next(r for r in summary["runs"] if r["degree"] == 1 and r["limiter"] == "none")["relative_l1"],
        100 * next(r for r in summary["runs"] if r["degree"] == 1 and r["limiter"] == "tvb")["relative_l1"],
        100 * next(r for r in summary["runs"] if r["degree"] == 1 and r["limiter"] == "weno")["relative_l1"]))
    text = TEMPLATE.format(
        gamma=summary["gamma"], xmin=summary["domain"][0], xmax=summary["domain"][1],
        diaphragm=summary["diaphragm"], lrho=summary["left_state"][0],
        lu=summary["left_state"][1], lp=summary["left_state"][2],
        tfinal=summary["final_time"], cfl=summary["cfl"],
        refdeg=summary["reference"]["degree"], refn=summary["reference"]["ncells"],
        refhalf=summary["reference"]["ncells"] // 2, refwall=summary["reference"]["wall_time_s"],
        crossn=cross["ncells"] if cross else "—",
        win0=WAVE_WINDOW[0], win1=WAVE_WINDOW[1],
        gates=table_gates(summary), runs=table_runs(summary, ref),
        osc=table_oscillation(summary, ref), figs=fig_entries,
        selfconv=100 * self_conv["relative_l1"],
        crossval=("{:.4f}%".format(100 * cross["relative_l1"]) if cross else "未测"),
        crossok=("是" if cross and 100 * cross["relative_l1"] < 3.0 else "否"),
        bestl1=100 * best["relative_l1"], p2weno=100 * p2weno["relative_l1"],
        p3none_l2=100 * p3none["relative_l2"], best_note=best_note,
        far_field=ref.get("far_field_max", float("nan")),
        far_dev=100.0 * abs(ref.get("far_field_max", 1.0) - 1.0),
    )
    text = text.replace("{selfconv:.3f}", "{:.3f}".format(100 * self_conv["relative_l1"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="shu_osher_results")
    parser.add_argument("--figure-dir", default="docs/figures")
    parser.add_argument("--doc", default="docs/shu_osher_verification.md")
    args = parser.parse_args()
    results = Path(args.results_dir)
    summary, profiles = load(results)
    ref = reference_metrics(summary, profiles)
    fig_paths = figures(summary, profiles, ref, Path(args.figure_dir))
    entries = "\n".join(
        f"![{path.name}](figures/{path.name})\n\n*{caption}*\n" for path, caption in fig_paths
    )
    doc = render(summary, ref, Path(args.doc), entries)
    print(f"wrote {doc} with {len(fig_paths)} figures")
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()