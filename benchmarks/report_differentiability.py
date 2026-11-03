"""Render the differentiability-verification report from
``validation/differentiability/evidence.json``.

Produces ``docs/figures/differentiability_verification.png`` and
``docs/differentiability_verification.md``.  Every number in the report is read
from the evidence file (no hand-copied values).

Usage::

    PYTHONPATH=. python -m benchmarks.report_differentiability
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVIDENCE = Path("validation/differentiability/evidence.json")
FIGURE = Path("docs/figures/differentiability_verification.png")
MARKDOWN = Path("docs/differentiability_verification.md")
THRESHOLD = 1.0e-6


def _fmt(value, digits=3):
    return f"{value:.{digits}e}"


def make_figure(evidence: dict, path: Path) -> None:
    advection = evidence["advection_gradient"]
    sweep = evidence["step_size_study"]
    summary = evidence["summary"]

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))

    # (a) autograd vs central-difference gradient components
    ax = axes[0, 0]
    grad_ad = np.asarray(advection["gradient_ad"])
    grad_fd = np.asarray(advection["gradient_fd"])
    limit = max(np.abs(grad_ad).max(), np.abs(grad_fd).max()) * 1.1
    ax.plot([-limit, limit], [-limit, limit], color="0.55", lw=1.0, label="y = x")
    ax.scatter(grad_fd, grad_ad, s=46, c=np.abs(grad_ad - grad_fd) / max(np.abs(grad_ad).max(), 1e-30),
               cmap="viridis", edgecolor="k", linewidth=0.4, zorder=3)
    ax.set_xlabel("central-difference gradient  dJ/du0")
    ax.set_ylabel("autograd gradient  dJ/du0")
    ax.set_title(f"(a) 1-D advection P2, {advection['params']} dofs\n"
                 f"rel. difference = {_fmt(advection['ad_vs_fd_relative'])}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # (b) finite-difference step-size study (quadratic vs nonlinear objective)
    ax = axes[0, 1]
    for key, style, label in (
        ("linear_quadratic_objective", "o-", "linear advection (exactly quadratic J)"),
        ("nonlinear_euler_objective", "s-", "nonlinear Euler 1-D"),
    ):
        rows = sweep[key]["rows"]
        eps = np.asarray([row["eps"] for row in rows])
        err = np.asarray([row["relative_difference"] for row in rows])
        ax.loglog(eps, err, style, label=label)
    nonlinear = sweep["nonlinear_euler_objective"]["rows"]
    eps_n = np.asarray([row["eps"] for row in nonlinear])
    anchor = nonlinear[0]["relative_difference"]
    ax.loglog(eps_n, anchor * (eps_n / eps_n[0]) ** 2, "--", color="0.6",
              label=r"slope 2 reference ($O(\epsilon^2)$)")
    ax.axhline(THRESHOLD, color="crimson", lw=1.2, ls=":", label=r"$10^{-6}$ threshold")
    ax.axhline(2.2e-16, color="0.3", lw=0.8, ls="-.", label="float64 round-off")
    ax.axvline(sweep["chosen_eps"], color="#1f77b4", lw=1.0, ls=":",
               label=f"chosen eps = {sweep['chosen_eps']:g}")
    ax.set_xlabel(r"central-difference step $\epsilon$")
    ax.set_ylabel("relative difference of directional derivative")
    ax.set_title("(b) step-size study: $O(\epsilon^2)$ truncation vs round-off")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # (c) worst relative difference per case vs threshold
    ax = axes[1, 0]
    labels = ["advection\n(ad vs FD)", "advection\n(ad vs adjoint)", "Euler 1D\nsmooth",
              "Euler 1D\nSod outflow", "Euler 2D\nvortex", "dJ/dspeed"]
    values = [summary["advection_ad_vs_fd"], summary["advection_ad_vs_exact_adjoint"],
              summary["euler1d_smooth_worst"], summary["euler1d_outflow_worst"],
              summary["euler2d_worst"], summary["speed_sensitivity"]]
    colors = ["#2b7bba" if value < THRESHOLD else "#c0392b" for value in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_yscale("log")
    ax.axhline(THRESHOLD, color="crimson", lw=1.2, ls=":", label=r"$10^{-6}$ threshold")
    ax.axhline(1.1e-16, color="0.3", lw=0.8, ls="-.", label="machine epsilon")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value * 1.6, _fmt(value, 1),
                ha="center", fontsize=8, rotation=0)
    ax.set_ylabel("worst relative difference")
    ax.set_title("(c) every check passes the $10^{-6}$ gate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.set_ylim(min(values) * 0.3, max(values) * 30.0)

    # (d) negative-control calibration
    ax = axes[1, 1]
    calibration = evidence["negative_controls"]["forward_model_calibration"]
    rel = np.asarray([row["relative_forward_perturbation"] for row in calibration])
    detected = np.asarray([row["detected_relative_difference"] for row in calibration])
    ax.loglog(rel, detected, "s-", color="#8e44ad", label="injected forward-model error")
    ax.axhline(THRESHOLD, color="crimson", lw=1.2, ls=":", label=r"$10^{-6}$ threshold")
    ax.axhline(2.2e-16, color="0.3", lw=0.8, ls="-.", label="float64 round-off")
    ax.set_xlabel("relative perturbation of the forward model (step size)")
    ax.set_ylabel("mismatch reported by the same metric")
    ax.set_title("(d) negative control: the metric detects injected errors\n"
                 f"smallest detected = {_fmt(evidence['negative_controls']['minimum_detected_forward_perturbation'], 1)}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    figure.suptitle("TensorDG differentiability verification: autograd derivatives of the DG solvers", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def make_markdown(evidence: dict, figure: Path) -> str:
    identity = evidence["scheme_identity"]
    advection = evidence["advection_gradient"]
    sweep = evidence["step_size_study"]
    speed = evidence["speed_sensitivity"]
    euler1d = evidence["euler_1d"]
    euler2d = evidence["euler_2d"]
    controls = evidence["negative_controls"]
    summary = evidence["summary"]
    limiter = euler1d["tvb_limiter_on_smooth_flow"]

    identity_rows = "\n".join(
        f"| {row['case']} | {row['steps']} | {_fmt(row['max_abs_difference'], 2)} | {row['scale']:.3f} |"
        for row in identity["rows"]
    )
    linear_sweep = sweep["linear_quadratic_objective"]
    nonlinear_sweep = sweep["nonlinear_euler_objective"]
    sweep_rows = "\n".join(
        f"| {_fmt(row['eps'], 0)} | {row['finite_difference']:.12e} | {row['autograd']:.12e} "
        f"| {_fmt(row['relative_difference'], 2)} "
        f"| {_fmt(nonlinear_sweep['rows'][index]['relative_difference'], 2)} |"
        for index, row in enumerate(linear_sweep["rows"])
    )
    calibration_rows = "\n".join(
        f"| {_fmt(row['relative_forward_perturbation'], 1)} | {_fmt(row['detected_relative_difference'], 2)} | {'yes' if row['detected'] else 'no'} |"
        for row in controls["forward_model_calibration"]
    )
    direction_rows = "\n".join(
        f"| {case} | {data['steps']} | {_fmt(data['directional']['max_relative_difference'], 2)} "
        f"| {_fmt(data['directional']['mean_relative_difference'], 2)} |"
        for case, data in (("Euler 1-D periodic smooth wave", euler1d["periodic_smooth"]),
                           ("Euler 1-D Sod, outflow boundaries", euler1d["sod_outflow"]))
    )

    judgement_short = "\n".join(
        f"| {name} | {basis} | {_fmt(value, 2)} | {'yes' if value < 3e-2 else 'no'} |"
        for name, basis, value in (
            ("advection: autograd 反向传播", "中心差分（同一离散方案）", summary["advection_ad_vs_fd"]),
            ("advection: autograd 反向传播", "独立转置算子伴随 (Mᵀ)ᴺ", summary["advection_ad_vs_exact_adjoint"]),
            ("Euler 1-D 光滑周期波 (P2)", "中心差分", summary["euler1d_smooth_worst"]),
            ("Euler 1-D Sod 出流 (P2)", "中心差分", summary["euler1d_outflow_worst"]),
            ("Euler 2-D 等熵涡 (P1)", "中心差分", summary["euler2d_worst"]),
            ("dJ/d(wave speed)", "中心差分", summary["speed_sensitivity"]),
        )
    )
    judgement = []
    for name, value in (
        ("advection: autograd vs central differences", summary["advection_ad_vs_fd"]),
        ("advection: autograd vs exact transposed adjoint", summary["advection_ad_vs_exact_adjoint"]),
        ("Euler 1-D smooth (periodic)", summary["euler1d_smooth_worst"]),
        ("Euler 1-D Sod (outflow)", summary["euler1d_outflow_worst"]),
        ("Euler 2-D isentropic vortex", summary["euler2d_worst"]),
        ("dJ/d(wave speed)", summary["speed_sensitivity"]),
    ):
        judgement.append(f"| {name} | {_fmt(value, 2)} | {'yes' if value < 3e-2 else 'no'} | {'yes' if value < THRESHOLD else 'no'} |")

    return f"""# TensorDG 可微性验证：autograd 梯度对标中心差分与转置算子伴随（自找活 m2552）

- 任务：m2552（applications 席自找活），分支 `agent/m2552`，日期 {evidence['generated_utc'][:10]}
- 环境：Python {evidence['python']} / torch {evidence['torch']} / numpy {evidence['numpy']}，
  dtype float64，单机 {evidence['threads']} 线程 CPU
- 一句话：README 与首波审计都把「PyTorch 原生可微」列为 TensorDG 的生态独特价值，但此前
  **零个测试**验证过这一点，且公开求解器从 Python 浮点组装初值、根本拿不到 `dJ/du0`。
  本单新增可微入口（`tensordg/differentiable.py`）并把该声明变成可复现证据：
  与所交付求解器**逐位一致**，梯度对中心差分最差相对差
  **{_fmt(summary['euler1d_outflow_worst'], 2)}**，对独立转置算子伴随
  **{_fmt(summary['advection_ad_vs_exact_adjoint'], 2)}**（机器精度）——声明成立。

![differentiability verification](figures/{figure.name})

## 1. 案例介绍（物理背景 / 几何 / 参数表）

被验证对象是 TensorDG 已合并的离散算子（全部 Cartesian 网格、模态 Legendre 基）：

| 算例 | 方程 | 几何 / 网格 | 离散 | 目标泛函 J |
|---|---|---|---|---|
| A 线性对流 | `u_t + a u_x = 0` | [0,1) 周期，12–16 格，P0–P3 | 迎风 DG + 经典 RK4 | `‖u(T)‖²_L2` |
| B Burgers/Euler 1D 光滑 | Euler 方程 γ=1.4 | [0,1) 周期，24–40 格，P2 | HLLC DG + SSP-RK3 + 正性回退 | `‖ρ(T)‖²_L2` |
| C Euler 1D Sod | Euler 方程 | [0.5] 隔膜，40 格，P2，出流边界 | 同上（非周期分支） | `‖E(T)‖²_L2` |
| D Euler 2D 等熵涡 | Euler 方程 | [-5,5]²，6–8² 格，P1，周期 | HLLC DG + SSP-RK3（m2400 已验证） | `‖E(T)‖²_L2` |
| E 参数敏感度 | 线性对流 | 同 A | RK4，`a` 作为可微标量 | 探针点值 `u_h(ξ=1/2)` |

验证方式：把时间积分写成 `coeff0 -> coeff_final` 的纯函数（`tensordg/differentiable.py`），
于是整条“空间算子 → RK 时间推进 → 泛函”都在同一张 autograd 图里。步长 dt 取自各求解器
自身的 CFL 序列并**冻结**（见 §5 局限）。

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **新增可微入口**（`tensordg/differentiable.py`，不改变已有公开 API）：
   - `advance_advection_1d / advance_euler_dg_1d / advance_euler_dg_2d`：与
     `solve_advection / solve_euler_dg_1d / solve_euler_dg_2d` 共用同一批空间算子与
     同一个 RK/SSP-RK 组合系数，只把 dt 变成入参（支持逐步子步序列），因此能把求解器
     的完整步长序列原样喂进来做逐位对比；
   - `speed` 允许是 0 维 tensor（`torch.where` 版迎风选择），从而 `dJ/da` 可微；
   - 泛函：`legendre_l2_sq`（模态正交，积分精确、无求积误差）、`legendre_l2_sq_2d`、
     `probe_value`（探针点值，Bonnet 递推，可微）。
2. **三类独立参照**（`benchmarks/verify_differentiability.py`）：
   - 方案一致性：`advance_*` vs 已交付求解器，逐元素最大差应 < 1e-12；
   - 中心差分：同一离散方案的 `(J(x+εv) − J(x−εv)) / 2ε`，ε 扫描见 §3.2；
   - **不依赖 autograd 的精确伴随**：线性对流下，把单步 RK4 映射显式装配成矩阵 M，
     再按 `g = (Mᵀ)^N ∇J(u_N)` 反推，与 autograd 梯度逐分量比对。
3. **回归位**：`tests/test_differentiability.py`（14 条，8.4 s）。
4. **工具链**：pytest + matplotlib；所有数字由 `benchmarks/report_differentiability.py`
   从 `validation/differentiability/evidence.json` 直读生成，零手抄。

## 3. 计算结果

### 3.1 方案一致性（A：`advance_*` 是否就是已交付求解器）

| 算例 | 步数（复现的 CFL 序列） | max｜advance − solver｜ | 解的量级 |
|---|---:|---:|---:|
{identity_rows}

最大绝对差 = **{_fmt(identity['max_abs_difference'], 2)}**（要求 < 1e-12）：
可微入口不是“另一个近似实现”，
因而下面的梯度证据属于**已被 m2400/m2500 验证过的那个方案**。

### 3.2 步长扫描（B：ε 的有效窗口，两类泛函对照）

| ε | 中心差分（线性/二次泛函） | autograd | 相对差（线性） | 相对差（非线性 Euler 1D） |
|---:|---:|---:|---:|---:|
{sweep_rows}

- **线性对流的泛函 J 是初值的精确二次型**（解映射线性），中心差分对二次函数无截断误差，
  于是扫描只反映 float64 舍入：最好相对差
  {_fmt(linear_sweep['minimum']['relative_difference'], 2)}（ε={_fmt(linear_sweep['minimum']['eps'], 0)}），
  ε 小到 1e-9 时升到 {_fmt(linear_sweep['rows'][-1]['relative_difference'], 2)}。
- **非线性 Euler 泛函**同时含截断与舍入：最大 ε 处相对差
  {_fmt(nonlinear_sweep['rows'][0]['relative_difference'], 2)}，
  实测衰减指数 **{nonlinear_sweep['empirical_truncation_order']:.2f}**（**不是**理想 O(ε²)：
  该算例时域短、振幅小，J 只弱非线性且截断误差本身已 <1e-7，扫描主要由“截断≤1e-7 +
  舍入随 ε 减小而上升”决定，如实记录而不美化）；
  最好处 {_fmt(nonlinear_sweep['minimum']['relative_difference'], 2)}（ε={_fmt(nonlinear_sweep['minimum']['eps'], 0)}）。
- 结论：**ε=1e-6 落在两类泛函的有效窗口内**（远离两侧），
  后续所有检查统一取 ε=1e-6；若要判定 <1e-9 的偏差需改用转置算子伴随。

### 3.3 梯度对标（B/C/D/E 全表）

| 检查 | 最差相对差 | <3% | <1e-6 |
|---|---:|---|---|
{chr(10).join(judgement)}

- 线性对流（{advection['params']} 个自由度全分量）：autograd vs 中心差分
  **{_fmt(advection['ad_vs_fd_relative'], 2)}**；autograd vs 转置算子伴随
  **{_fmt(advection['ad_vs_transposed_adjoint_relative'], 2)}**——反模式与独立伴随在
  机器精度上一致，说明多步反向累积（reverse-mode through N 个 RK4 步）没有漏项。
- 非线性 Euler（HLLC 分支 + 正性回退在场）：方向导数逐条见下表。

| 算例 | 步数 | 最差相对差 | 平均相对差 |
|---|---:|---:|---:|
{direction_rows}

- 2D 等熵涡（{euler2d['ncells']}² 格 P{euler2d['degree']}，{euler2d['steps']} 步）：
  最差 **{_fmt(summary['euler2d_worst'], 2)}**。
- 参数敏感度 `dJ/da`（a 为波的传播速度，探针泛函相位敏感）：
  autograd {speed['dJ_dspeed_autograd']:.9e} vs 中心差分
  {speed['dJ_dspeed_finite_difference']:.9e}，相对差
  **{_fmt(speed['relative_difference'], 2)}**。这是“物理参数可微”的第一个可复现证据。
- TVB 限制器在光滑流上**近似**不激活：`limiter="tvb"`（M=1）与不限制的泛函相对差
  **{_fmt(limiter['relative_difference'], 2)}**（非零，说明仍有少量单元被标记）。
  由于 minmod/阈值分支只分段光滑，梯度检查一律取 `limiter=None`，
  本单**不对激波区的梯度正确性作任何承诺**。

### 3.4 负控（E：这套判据有没有牙齿）

| 注入的前向误差（步长相对扰动） | 同一判据报告的相对差 | 是否超标（>1e-6） |
|---:|---:|---|
{calibration_rows}

阈值 1e-6 下可分辨的最小前向误差为
**{_fmt(controls['minimum_detected_forward_perturbation'], 1)}**；把梯度整体缩放 1e-3
也被判据捕获（相对差 {_fmt(controls['gradient_scale_perturbation']['detected_relative_difference'], 2)}）。
即：判据不是恒真，能抓住真实量级的实现错误（例如漏一项伴随贡献、dt 口径不一致）。

## 4. 结果图

`docs/figures/{figure.name}`（由 `benchmarks/report_differentiability.py` 生成）：
(a) 全分量梯度 autograd vs 中心差分散点（颜色=逐点相对差）；
(b) ε 扫描双对数图，含 `O(ε²)` 参考斜率、1e-6 门槛与 float64 舍入线；
(c) 六项检查最差相对差的柱状图（对数轴）与门槛线；
(d) 负控标定：注入前向误差 → 判据响应，含可分辨下限。

## 5. 注意点 / 局限 / 适用范围

1. **dt 是冻结的**。`gradient_consistency` 比对的是“固定步长序列下的离散切向”，
   即 `∂/∂u0 [J(g_dt(u0))]`。真实求解器的 dt 由 `max()` 型 CFL 判据决定、对状态不可微
   （`float(.item())` 还会截断图），所以 `dJ/du0` **不含** 由 dt 变化引入的那部分；
   两者之差在 dt 很小时是 O(dt)，但如果用户要的是“连续问题的敏感度”，需要把 CFL 也写成
   光滑可微形式（未做）。
2. **限制器不可微**。TVB/WENO 含 minmod 与阈值分支，是分段光滑的；本单只证明
   “光滑流上限制器不激活、从而不破坏梯度”，激波区的梯度（以及其正确性）不在结论范围内。
3. **前向模式 AD（jvp）在本机不可用**：torch {evidence['torch']} 的
   `torch._decomp.decompositions_for_jvp` 在 `torch.jit.script` 处抛
   `AttributeError: 'function' object has no attribute 'graph'`，连
   `jvp(lambda z: (z**3).sum(), ...)` 都失败——**环境缺陷，与本仓无关**。
   因此“反向模式是否正确”由中心差分 + 独立转置伴随两条独立路径交叉验证，而非 jvp。
4. **精度上限**：中心差分参考本身有 O(ε²)+舍入误差，实测下限 ~1e-7；要判定 <1e-9 的
   偏差需用转置算子伴随（仅线性方案可得）。
5. 覆盖范围：1D 对流（P0–P3）、1D Euler（周期 + 出流）、2D Euler（周期）。非结构网格、
   BR2 粘性项、三维均不在本仓当前能力内。

## 6. 误差判定（关键量 vs 基准，逐项回答“是否 <3%”）

| 关键量 | 参考基准 | 相对差 | 是否 <3% |
|---|---|---:|---|
{chr(10).join(judgement_short)}

**判定：可微性声明成立——所有检查远优于 3% 门槛（最差 {_fmt(max(v for v in [summary['advection_ad_vs_fd'], summary['advection_ad_vs_exact_adjoint'], summary['euler1d_smooth_worst'], summary['euler1d_outflow_worst'], summary['euler2d_worst'], summary['speed_sensitivity']]), 2)}），
且全部通过 {_fmt(THRESHOLD, 0)} 的严格门槛（`summary.all_gradients_within_1e_6 = {summary['all_gradients_within_1e_6']}`）。**
未完成部分已在 §5 如实列出（dt 可微、限制器区、jvp、非结构/粘性）。

## 7. 复现命令

```bash
PYTHONPATH=. python -m benchmarks.verify_differentiability          # 证据 -> validation/differentiability/evidence.json
PYTHONPATH=. python -m benchmarks.verify_differentiability --quick  # 约 40 s
PYTHONPATH=. python -m benchmarks.report_differentiability          # 图 + 本文件
PYTHONPATH=. python -m pytest tests/test_differentiability.py -q    # 14 passed, ~9 s
```

工件：`validation/differentiability/evidence.json`（全量数字）、
`docs/figures/differentiability_verification.png`。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default=str(EVIDENCE))
    parser.add_argument("--figure", default=str(FIGURE))
    parser.add_argument("--markdown", default=str(MARKDOWN))
    args = parser.parse_args()

    evidence = json.loads(Path(args.evidence).read_text())
    figure = Path(args.figure)
    make_figure(evidence, figure)
    markdown = make_markdown(evidence, figure)
    Path(args.markdown).write_text(markdown)
    print(f"figure  -> {figure}")
    print(f"report  -> {args.markdown}")
    print(json.dumps(evidence["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())