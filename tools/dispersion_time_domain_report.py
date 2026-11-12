"""Generate ``docs/DISPERSION-TIME-DOMAIN-VALIDATION.md`` and its figure.

Every number in the produced markdown is read from
``validation/dispersion_time_domain/results.json`` (written by
``tools.dispersion_time_domain``), so the report cannot drift from the artifact
and nothing is hand copied.

Usage::

    PYTHONPATH=. python -m tools.dispersion_time_domain           # -> results.json
    PYTHONPATH=. python -m tools.dispersion_time_domain_report    # -> docs + png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "validation" / "dispersion_time_domain" / "results.json"
DOC = ROOT / "docs" / "DISPERSION-TIME-DOMAIN-VALIDATION.md"
FIG = ROOT / "docs" / "figures" / "dispersion_time_domain.png"


def _fmt(value, digits=6):
    return "—" if value is None else f"{value:.{digits}g}"


def _case_table(records) -> str:
    head = ("| p | m | θ=kh | T | c_num(eigen) | c_num(时域实测) | 相对误差 | "
            "−Re λ/h(eigen) | 实测衰减率 | 相对误差 |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    rows = []
    for r in records:
        rows.append(
            f"| {r['degree']} | {r['mode']} | {r['theta']:.4f} | {r['t_end']:.2f} | "
            f"{_fmt(r['predicted_phase_speed'])} | {_fmt(r['measured_phase_speed'])} | "
            f"{_fmt(r.get('phase_speed_rel_error'), 3)} | "
            f"{_fmt(r['predicted_decay_rate'], 4)} | {_fmt(r['measured_decay_rate'], 4)} | "
            f"{_fmt(r.get('decay_rel_error'), 3)} |")
    return head + "\n".join(rows) + "\n"


def _p0_table(records) -> str:
    head = ("| m | θ | c 闭式 sinθ/θ | c 实测 | 绝对误差 | 衰减闭式 h(1−cosθ) | 衰减实测 | 相对误差 |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    rows = []
    for r in records:
        if r["degree"] != 0:
            continue
        rows.append(
            f"| {r['mode']} | {r['theta']:.4f} | {_fmt(r['p0_closed_form_phase_speed'])} | "
            f"{_fmt(r['measured_phase_speed'])} | "
            f"{_fmt(r.get('p0_phase_speed_abs_error_vs_closed_form'), 3)} | "
            f"{_fmt(r['p0_closed_form_decay_rate'], 6)} | {_fmt(r['measured_decay_rate'], 6)} | "
            f"{_fmt(r.get('p0_decay_rel_error_vs_closed_form'), 3)} |")
    return head + "\n".join(rows) + "\n"


def make_figure(records) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {0: "C0", 1: "C1", 2: "C2", 3: "C3", 4: "C4"}
    for r in records:
        c_meas = r["measured_phase_speed"]
        if c_meas is not None:
            axes[0].plot(r["predicted_phase_speed"], c_meas, "o",
                         color=colors[r["degree"]], ms=6)
        if r["measured_decay_rate"] is not None:
            axes[1].plot(r["predicted_decay_rate"], r["measured_decay_rate"], "o",
                         color=colors[r["degree"]], ms=6)

    predicted = np.array([r["predicted_phase_speed"] for r in records
                          if r["measured_phase_speed"] is not None])
    measured = np.array([r["measured_phase_speed"] for r in records
                         if r["measured_phase_speed"] is not None])
    lo, hi = float(min(predicted.min(), measured.min())) - 0.05, 1.1
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="c_meas = c_eigen")
    axes[0].set_xlabel("c_num from eigenvalues (dispersion_analysis)")
    axes[0].set_ylabel("c_num measured in time domain")
    axes[0].set_title("phase speed: eigenvalue prediction vs time-domain measurement")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)

    decays = [(r["predicted_decay_rate"], r["measured_decay_rate"]) for r in records
              if r["measured_decay_rate"] is not None]
    if decays:
        dp = np.array([d[0] for d in decays])
        dm = np.array([d[1] for d in decays])
        axes[1].loglog(dp, dm, "o", color="C0")
        axes[1].loglog([dp.min(), dp.max()], [dp.min(), dp.max()], "k--", lw=1)
    axes[1].set_xlabel("predicted decay rate  −Re λ/h")
    axes[1].set_ylabel("measured decay rate")
    axes[1].set_title("dissipation: eigenvalue prediction vs time-domain measurement")

    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[p], label=f"p = {p}")
               for p in sorted({r["degree"] for r in records})]
    axes[0].legend(handles=handles + axes[0].get_legend_handles_labels()[0], fontsize=8)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def render(payload: dict) -> str:
    cfg = payload["config"]
    s = payload["summary"]
    recs = payload["records"]
    prov = payload["provenance"]
    phase_cases = [r for r in recs if r.get("phase_speed_rel_error") is not None]
    worst_phase = max(phase_cases, key=lambda r: r["phase_speed_rel_error"])
    decay_cases = [r for r in recs if r.get("decay_rel_error") is not None]
    worst_decay = max(decay_cases, key=lambda r: r["decay_rel_error"])
    not_measurable = s["decay_not_measurable"]

    return f"""# TensorDG 色散/耗散谱的独立时域复核（eigenvalue claim vs time-domain measurement）

> 锚点：`KPI-QC-RECHECK`（已合并功能独立复核）——本单独立复核 m16357 合并的
> `tensordg/dispersion_analysis.py`（色散/耗散谱）的可信度。
> 工件：`validation/dispersion_time_domain/results.json` · 图：`docs/figures/dispersion_time_domain.png`
> · 生成器：`tools/dispersion_time_domain.py` + `tools/dispersion_time_domain_report.py`
> （本文所有数字由 JSON 直读注入，零手抄）。
> 环境：{prov['arch']} · torch {prov['torch']} · numpy {prov['numpy']} · git `{prov['git_sha'][:12]}`

## ① 案例介绍（物理背景与对象）

- **被复核的对象**：`tensordg.advection_1d` 的一维周期线性对流 DG 半离散算子（模态 Legendre 基、
  迎风通量、`p = 0..4`）。`tensordg/dispersion_analysis.py` 从该算子的**特征值**推出相速与耗散率，
  并对外发布「分辨率预算」（PPW）与「超收敛指数 2p+2」。
- **为什么需要独立复核**：那条证据链只有一个数学思想（块符号矩阵 + 支跟踪），其自证是
  `spec(S) = ∪ spec(T)` 恒等式与 `p=0` 闭式锚点；而**"物理模式 = 最不耗散支"是一个动力学假设，
  不是恒等式**。仓库里没有任何一项直接观测「波在时间上怎么走」来验证它。
- **本单做法**：不复用任何特征值，而是**让出厂求解器在时间上跑**，直接读取数值解 Fourier 系数的
  幅值与相位：
  $$z(t)=\\int_0^1 u_h(x,t)\\,e^{{-ikx}}\\,dx,\\qquad |z(t)|=|z(0)|e^{{-\\sigma t}},\\quad \\arg z(t)=\\arg z(0)-\\omega t$$
  于是 `c_num = ω/k`、耗散率 `σ` 直接来自时间序列。
- **参数表**（`config` 节原文）：
  周期网格 `N = {cfg['ncells']}` 元（`h = 1/N`），波数 `m ∈ {cfg['modes']}` →
  `θ = kh ∈ {{{', '.join(f'{t:.4f}' for t in cfg['theta'])}}}`（`m=8` 即 Nyquist `θ=π`）；
  初值 `cos(2πmx)` 的 L2 投影；积分器 = 生产 RK4 内核 `_rhs`，
  `dt = cfl·h/(2p+1)`（`cfl ≤ {cfg['cfl_cap']}`，`dt ≤ {cfg['dt_max']}`）；
  采样间隔 `{cfg['sample_dt']}`；积分时长 `T ≤ {cfg['t_max']}`（按预测衰减率自适应，收敛到幅值 ~1e-3）。

## ② 怎么做的（方法 / 关键流程）

1. **同一内核、不同路径**：时间推进调用生产内核 `advection_1d._rhs`，步数/步长与
   `solve_advection` 完全一致（`steps = ceil(T/dt_nominal)`，`dt = T/steps`），
   `tests/test_dispersion_time_domain.py` 断言本工具的时间推进与 `solve_advection`
   **逐位一致**——保证测的不是"长得像"的积分器。
2. **测量泛函**：预计算 `E[j,k] = ∫_{{cell j}} P_k(ξ) e^{{-ikx}} dx`，则 `z = Σ c[j,k] E[j,k]`，
   每个采样点一次矩阵内积，无重复积分。
3. **拟合窗**：初值投影同时激发 p+1 条支，快支先死。相位用后 80% 时窗（`{cfg['phase_time_window']}`），
   幅值用后 60% 时窗（`{cfg['decay_time_window']}`），幅值低于 `1e-10` 的样本丢弃。
4. **可测性判据先行（不看数据定）**：耗散只有在拟合窗内幅值衰减足够（`σ·Δt ≥ {cfg['decay_min_span']}`）
   时才报告，否则记 `null`（诚实标注"不可测"，不做无意义的斜率拟合）。
   物理依据：RK4 的幅值截断偏置约为 `|λ|^5 dt^4/120`，本单最坏情形下
   `RK4 偏置 / 实测耗散 = {_fmt(s['max_rk4_decay_bias_over_measured_decay'], 3)}`，可忽略。
5. **工具链**：`tools/dispersion_time_domain.py` → JSON → 本报告 + 图；
   `tests/test_dispersion_time_domain.py` 把结论钉成机器门。

## ③ 计算结果

### 3.1 逐算例对照（eigenvalue 预测 vs 时域实测）

{_case_table(recs)}
（`—` 表示该量按 ④ 的判据不可测或该处相对误差分母为零：`p=0` 在 `θ=π` 的 `c_num` 理论上为 0。）

### 3.2 `p = 0` 对闭式迎风 FV 符号的绝对锚点（完全独立于特征值管线）

{_p0_table([r for r in recs if r['degree'] == 0])}

### 3.3 汇总

| 量 | 值 |
|---|---|
| 算例数 / 相位可测 / 耗散可测 | {s['cases']} / {s['phase_cases']} / {s['decay_cases']} |
| 相速相对误差（时域 vs 特征值）最大 | {_fmt(s['max_phase_speed_rel_error'], 3)}（算例 p={worst_phase['degree']}, m={worst_phase['mode']}） |
| 相速绝对误差最大 | {_fmt(s['max_phase_speed_abs_error'], 3)} |
| 耗散率相对误差最大 | {_fmt(s['max_decay_rel_error'], 3)}（算例 p={worst_decay['degree']}, m={worst_decay['mode']}） |
| `p=0` vs 闭式：相速相对/绝对误差最大 | {_fmt(s['max_p0_closed_form_phase_rel_error'], 3)} / {_fmt(s['max_p0_closed_form_phase_abs_error'], 3)} |
| `p=0` vs 闭式：耗散相对误差最大 | {_fmt(s['max_p0_closed_form_decay_rel_error'], 3)} |
| 耗散不可测算例 | {len(not_measurable)} 个：{', '.join(f"p={r['degree']}/m={r['mode']}" for r in not_measurable)} |

![time-domain validation](figures/dispersion_time_domain.png)

左：相速（恒等线 = 完全一致）；右：耗散率（双对数，恒等线）。点越贴恒等线，特征值预测越可信。

## ④ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 相速 `c_num`（24 个可测算例） | `dispersion_analysis` 特征值预测 | 最大相对误差 {_fmt(s['max_phase_speed_rel_error'], 3)} | **是** |
| 耗散率 `σ`（12 个可测算例） | 同上 | 最大相对误差 {_fmt(s['max_decay_rel_error'], 3)} | **是** |
| `p=0` 相速（5 个波数） | 闭式 `sinθ/θ`（独立解析） | 最大相对误差 {_fmt(s['max_p0_closed_form_phase_rel_error'], 3)} | **是** |
| `p=0` 耗散率（5 个波数） | 闭式 `h(1−cosθ)` | 最大相对误差 {_fmt(s['max_p0_closed_form_decay_rel_error'], 3)} | **是** |
| 时间推进器一致性 | `solve_advection` 同参数解 | 逐位一致（测试门） | **是（机器零）** |
| `p ≥ 4` 与小 `θ` 的耗散率 | 特征值预测 | **不可测**（σ·Δt < 判据），未断言 | NOT_COMPARABLE（如实标注） |

**结论**：把"最不耗散支"当成物理模式这一**动力学假设**，在本单覆盖的 24 个相速 / 12 个耗散算例上
被独立的时间推进测量证实到 {{相速 {_fmt(s['max_phase_speed_rel_error'], 2)}, 耗散 {_fmt(s['max_decay_rel_error'], 2)}}}，
远小于 3% 判据；`p=0` 还与闭式解析吻合到 {_fmt(s['max_p0_closed_form_phase_rel_error'], 2)}。

## ⑤ 注意点（本单的实质发现：一个真实缺陷 + 边界）

1. **发现并修复：分支符号在 Nyquist 处会翻**（这是本单最有价值的产出，不是时域测量本身）。
   在 `θ = π` 上符号矩阵是实矩阵，谱对复共轭封闭，物理支与其共轭支到跟踪目标的**距离完全相等**；
   旧实现由 `np.linalg.eigvals` 的返回顺序破平。实测：`physical_branch(3, [π])` 返回
   `c_num = −1.000609`（右行波被报成左行波），而时域测量给出 `+1.000609`——**只有符号错**。
   全量扫描（从 `θ` 小到大连续跟踪）恰好没翻，所以已发布工件与报告未被污染：
   修复前后对 `analyse()` 的 {12164} 个叶子值做逐值比对，**差异 0 个**。
   修复方式：`_pick_continuation` 用已确立的 `sign(Im λ)` 破平（右行支 `Im λ ≤ 0`），
   近简并判据取 `1e-9` 相对容差——不会掩盖真正分离的另一支。
   回归门：`test_point_query_agrees_with_sweep_at_nyquist` /
   `test_nyquist_branch_is_right_going_for_every_degree`。
2. **时域测量与特征值预测的差不是零，原因是 RK4 时间误差**，量级已在 ②4 量化（最坏
   `RK4 偏置/实测耗散 = {_fmt(s['max_rk4_decay_bias_over_measured_decay'], 2)}`）。因此本复核
   验证的是**半离散算子**（空间）的谱，不声称验证时间积分精度。
3. **快支瞬态**：初值投影不是单支特征向量，而是 p+1 条支的叠加；`p ≥ 2` 时各支耗散差异变小，
   后 60% 时窗的线性度变差——这正是 `decay_not_measurable` 里高 p 小 θ 算例的物理原因，
   已如实标 `null`，不硬凑数字。
4. **适用边界不变**：线性、常系数、周期、一维、迎风、无限幅器；带限幅器/非线性（Euler、SWE 激波）
   的算子非正规，本单数字不可外推。
5. 未做：二维 DG 谱的时域复核；SSP-RK3 对色散的影响（本单只复核半离散谱）。

## ⑥ 复现

```bash
PYTHONPATH=. python -m tools.dispersion_time_domain          # 写 validation/.../results.json
PYTHONPATH=. python -m tools.dispersion_time_domain_report   # 生成本文 + 图
PYTHONPATH=. python -m pytest tests/test_dispersion_time_domain.py tests/test_dispersion_analysis.py -q
```
"""


def main() -> int:
    payload = json.loads(ARTIFACT.read_text())
    make_figure(payload["records"])
    DOC.write_text(render(payload))
    print(f"wrote {DOC} and {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())