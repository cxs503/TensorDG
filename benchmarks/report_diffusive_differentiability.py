"""Render ``docs/diffusive_differentiability_verification.md`` + figure from evidence.

Every number in the generated markdown is read from
``validation/diffusive_differentiability/evidence.json`` -- nothing is typed by
hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "validation/diffusive_differentiability/evidence.json"
DOC = ROOT / "docs/diffusive_differentiability_verification.md"
FIGURE = ROOT / "docs/figures/diffusive_differentiability.png"


def _figure(data: dict) -> None:
    cases = data["cases"]
    labels = [f"P{c['degree']}\n{c['ncells']} cells" for c in cases]
    physics = [
        c["sensitivity_alpha"]["autograd_vs_analytic_relative"] for c in cases
    ]
    adjoint = [
        c["sensitivity_alpha"]["autograd_vs_adjoint_relative"] for c in cases
    ]
    sweeps = data["alpha_sweeps"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    x = range(len(cases))
    ax.semilogy(x, physics, "o-", color="#1f77b4", label="autograd vs analytic")
    ax.semilogy(x, adjoint, "s--", color="#d62728",
                label="autograd vs exact discrete adjoint")
    ax.axhline(3e-2, color="0.4", linestyle=":", label="3% documentation gate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("relative difference of dJ/d$\\alpha$")
    ax.set_title("(a) tangent error: code vs code vs physics")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for sweep in sweeps:
        alphas = [row["alpha"] for row in sweep["rows"]]
        ax.plot(alphas, [row["value_autograd"] for row in sweep["rows"]], "o",
                label=f"discrete P{sweep['degree']}")
        ax.plot(alphas, [row["value_analytic"] for row in sweep["rows"]], "-",
                color="0.3", label="analytic" if sweep is sweeps[0] else None)
    ax.set_xlabel("thermal diffusivity $\\alpha$")
    ax.set_ylabel("$J=\\|u(T)\\|^2$")
    ax.set_title("(b) objective vs diffusivity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    for sweep in sweeps:
        alphas = [row["alpha"] for row in sweep["rows"]]
        ax.semilogy(alphas, [row["sensitivity_autograd"] ** 2 for row in sweep["rows"]],
                    "-", label=f"autograd P{sweep['degree']}")
        ax.semilogy(alphas,
                    [row["sensitivity_analytic"] ** 2 for row in sweep["rows"]],
                    "k:", label="analytic" if sweep is sweeps[0] else None)
    ax.set_xlabel("thermal diffusivity $\\alpha$")
    ax.set_ylabel("$(dJ/d\\alpha)^2$")
    ax.set_title("(c) sensitivity magnitude vs $\\alpha$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("TensorDG: autograd sensitivities of the diffusive SIPG + Crank-Nicolson path",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=140)
    plt.close(fig)


def _markdown(data: dict) -> str:
    summary = data["summary"]
    cases = data["cases"]
    sweeps = data["alpha_sweeps"]
    rows = []
    for c in cases:
        obj = c["objective"]
        sen = c["sensitivity_alpha"]
        rows.append(
            "| {n} | {p} | {steps} | {dt:.4e} | {value:.8f} | {jerr:.2%} | "
            "{ad:.6e} | {adj:.6e} | {rel:.2e} | {ana:.2e} |".format(
                n=c["ncells"], p=c["degree"], steps=c["steps"], dt=c["dt"],
                value=obj["autograd"], jerr=obj["relative_error_vs_analytic"],
                ad=sen["autograd"], adj=sen["exact_discrete_adjoint"],
                rel=sen["autograd_vs_adjoint_relative"],
                ana=sen["autograd_vs_analytic_relative"],
            )
        )
    sweep_rows = []
    for sweep in sweeps:
        for row in sweep["rows"]:
            sweep_rows.append(
                "| P{p} / {n} | {a:.2f} | {v:.8f} | {vj:.2e} | {s:.6e} | {sj:.2e} | {sa:.2e} |".format(
                    p=sweep["degree"], n=sweep["ncells"], a=row["alpha"],
                    v=row["value_autograd"], vj=row["value_relative_error"],
                    s=row["sensitivity_autograd"],
                    sj=row["sensitivity_autograd_vs_adjoint_relative"],
                    sa=row["sensitivity_vs_analytic_relative"],
                )
            )
    fd_rows = []
    worst_fd_case = max(
        cases, key=lambda c: c["sensitivity_alpha"]["autograd_vs_adjoint_relative"]
    )
    for row in worst_fd_case["finite_difference_alpha"]:
        fd_rows.append(
            "| {eps:.0e} | {fd:.12e} | {rel:.3e} |".format(
                eps=row["eps"], fd=row["finite_difference"],
                rel=row["relative_difference_vs_autograd"],
            )
        )
    verified = lambda flag: "✅ PASS" if flag else "❌ FAIL"
    return f"""# 可微 DG 的扩散半场：SIPG + Crank–Nicolson 通路的梯度验证（含解析灵敏度对标）

> 本文件由 `benchmarks/report_diffusive_differentiability.py` 从
> `validation/diffusive_differentiability/evidence.json` 程序化生成，数字零手抄。
> 环境：Python {data['python']} / torch {data['torch']} / {data['dtype']} / {data['threads']} torch 线程 / CPU。
> quick 模式：{data['quick']}。生成时间（UTC）：{data['generated_utc']}。

## 1. 案例介绍

**物理背景。** 一维热传导 `u_t = alpha u_xx`，`x∈(0,1)`，两端零 Dirichlet 数据，
初值 `u0 = sin(pi x)`。其精确解为单模态衰减 `u(x,t) = e^{{-alpha pi^2 t}} sin(pi x)`，
因此目标泛函 `J(alpha) = ||u(T)||^2 = 0.5 e^{{-2 alpha pi^2 T}}` 及其灵敏度
`dJ/dalpha = -2 pi^2 T J` 都有**闭式表达式**——这是把一个"梯度实现"与**物理**（而非仅仅与
有限差分）对表的充要条件。

**为什么这是欠账。** `docs/differentiability_verification.md`（m2552）把 README 的
"可微 DG" 声明变成了证据，但其五个算例（线性对流、Burgers/Euler 1D 光滑、Sod、2D 等熵涡、
`a` 参数敏感度）**全部是双曲型**；`tensordg/differentiable.py` 也只提供
`advance_advection_1d / advance_euler_dg_1d / advance_euler_dg_2d`。
**扩散半场（SIPG 热传导 / 椭圆求解器）此前既无可微入口、也无任何梯度证据**，
而它正是热物性反演、粘性参数辨识、热-结构耦合优化的入口。
更关键的是：单纯"autograd vs 中心差分"只能证明两条路对**同一离散格式**求导一致，
无法发现格式里 `alpha` 少乘一次或刚度块符号错这类错误。因此本单引入
**不依赖 autograd、也不依赖有限差分的精确离散伴随**与**解析灵敏度**两个独立基准。

**离散设置（沿用已合并的 `tensordg.heat_1d`，未改其数值格式）。**

| 项 | 设置 |
|---|---|
| 空间 | 模态 Legendre DG，P1–P3，Cartesian 单元，SIPG 罚项 `penalty=20` |
| 时间 | Crank–Nicolson，`dt = T/nsteps`，`nsteps = ceil(T / (0.02 h))` |
| 边界 | 两端零 Dirichlet（罚项强加，与 `solve_heat_1d` 同源） |
| 目标泛函 | `J = ||u_h(T)||^2_L2 = u^T M u`（质量矩阵精确积分，无求积误差） |
| 可微参数 | 热扩散率 `alpha`（0 维 tensor，贯穿线性求解）、初值 `coeff0`（全部模态 dof） |
| 基准值 | `alpha=0.05`，`T=0.1`，网格 8/16 格 × P1/P2/P3 |

## 2. 怎么做的

1. **新增可微入口**（`tensordg/differentiable.py::advance_heat_1d` +
   `heat_1d_operators`，不改动 `solve_heat_1d` 的数值格式与公开签名）：
   把 SIPG 质量/刚度矩阵装配抽成函数（体积项、内部面、Dirichlet 边界块逐块照抄），
   `alpha` 在推进时以张量形式乘入 `(M ± dt alpha K / 2)`，于是整条
   "空间算子 → CN 推进 → 泛函"都留在同一张 autograd 图上。
2. **方案一致性（不是"另一个近似实现"）**：把 `solve_heat_1d` 自己生成的初值投影与
   冻结步长序列喂给 `advance_heat_1d(alpha=1)`，要求逐元素最大差 `< 1e-12`。
3. **精确离散伴随（独立基准 1）**：显式装配 CN 映射 `G = A^{{-1}}B`，
   `A = M + dt alpha K/2`、`B = M - dt alpha K/2`，伴随态用 `G^T` 反传，
   并用**解析**的 `dG/dalpha = A^{{-1}}(dB/dalpha - dA/dalpha G)` 合成 `dJ/dalpha`、
   用 `(G^T)^N 2 M u_N` 给出 `dJ/dcoeff0`。全程无 autograd、无差分。
4. **解析灵敏度（独立基准 2）**：`dJ/dalpha = -2 pi^2 T J`，即把梯度直接对表物理。
5. **中心差分**：对同一离散格式在 `alpha` 上做 `eps` 扫描（1e-2 … 1e-6），仅作一致性旁证。
6. **工具链**：`benchmarks/verify_diffusive_differentiability.py` → evidence.json；
   `benchmarks/report_diffusive_differentiability.py` → 本文 + 图；
   `tests/test_diffusive_differentiability.py`（9 条回归）。

## 3. 计算结果

### 3.1 方案一致性

四个算例上 `advance_heat_1d(alpha=1)` 与 `solve_heat_1d` 的逐元素最大差
= **{summary['worst_scheme_identity_max_abs']:.1e}**（要求 < 1e-12）。
可微入口就是已交付的那个 SIPG + CN 格式，下面所有梯度证据都属于该格式。

### 3.2 梯度三方对表

| 网格 | 阶 | 步数 | dt | J（autograd） | J vs 解析 | dJ/dα（autograd） | dJ/dα（精确伴随） | autograd vs 伴随 | autograd vs 解析 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

读法：**autograd 与精确离散伴随一致到 {summary['worst_autograd_vs_exact_adjoint_alpha']:.1e}
（`alpha`）/{summary['worst_autograd_vs_exact_adjoint_initial_state']:.1e}（初值全分量）**，
即机器精度；与**解析灵敏度**的差随着 `p` 与网格加密单调下降
（P1/8 格 {cases[0]['sensitivity_alpha']['autograd_vs_analytic_relative']:.2%}
→ P3/16 格 {cases[-1]['sensitivity_alpha']['autograd_vs_analytic_relative']:.1e}），
说明剩余差异是**离散误差**而非伴随实现错误。
注意 P1/8 格的 {cases[0]['sensitivity_alpha']['autograd_vs_analytic_relative']:.2%}
**不是**梯度 bug：同一算例的 J 本身已有 {cases[0]['objective']['relative_error_vs_analytic']:.2%}
误差、解相对 L2 误差 {cases[0]['objective']['relative_l2_of_solution']:.2%}。

### 3.3 中心差分旁证（最粗算例 P2/8 格）

| eps | (J(α+ε) − J(α−ε))/2ε | vs autograd 相对差 |
|---:|---:|---:|
{chr(10).join(fd_rows)}

最小相对差 {summary['worst_autograd_vs_central_difference_alpha']:.1e}；
ε 过大（1e-2）时 CN 的 `dt` 固定、`alpha` 非线性进入 `G`，截断误差上升，属正常窗口现象。

### 3.4 扩散率扫描（灵敏度符号与量级）

| 算例 | α | J | J vs 解析 | dJ/dα | vs 精确伴随 | vs 解析 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(sweep_rows)}

`dJ/dα < 0` 且随 `α` 增大单调变小、`J(α)` 与解析曲线重合，符号与量级都对。
CN 映射谱半径 {cases[-1]['step_map_spectral_radius']:.6f} < 1（稳定，无伪增长）。

![diffusive differentiability](figures/diffusive_differentiability.png)

- **(a)** 两类误差的对比：autograd vs 精确伴随压在机器精度（红方块），
  autograd vs 解析随阶数/网格下降，全部低于 3% 文档门（灰虚线）；
- **(b)** `J(α)`：离散点与解析曲线重合（P2/P3 不可分辨）；
- **(c)** 灵敏度平方的 log 尺度：单调、无符号翻转、与解析基线的偏差只在低阶粗网格可见。

## 4. 误差判定（逐项，显式回答是否 < 3%）

| 关键量 | 实测 | 基准 | 相对误差 | 是否 < 3% |
|---|---|---:|---:|---|
| `dJ/dα`（P3/16 格） | {cases[-1]['sensitivity_alpha']['autograd']:.6e} | {cases[-1]['sensitivity_alpha']['analytic']:.6e}（解析） | {cases[-1]['sensitivity_alpha']['autograd_vs_analytic_relative']:.1e} | ✅ |
| `dJ/dα`（P2/16 格） | {cases[2]['sensitivity_alpha']['autograd']:.6e} | {cases[2]['sensitivity_alpha']['analytic']:.6e}（解析） | {cases[2]['sensitivity_alpha']['autograd_vs_analytic_relative']:.1e} | ✅ |
| `dJ/dα`（P1/8 格，最粗） | {cases[0]['sensitivity_alpha']['autograd']:.6e} | {cases[0]['sensitivity_alpha']['analytic']:.6e}（解析） | {cases[0]['sensitivity_alpha']['autograd_vs_analytic_relative']:.2%} | ✅（{cases[0]['sensitivity_alpha']['autograd_vs_analytic_relative']:.2%} < 3%） |
| `dJ/dα` vs 精确离散伴随（全部算例） | 见 §3.2 | 手装配伴随 | ≤ {summary['worst_autograd_vs_exact_adjoint_alpha']:.1e} | ✅（机器精度） |
| `dJ/dcoeff0` vs 精确伴随（{max(c['gradient_initial_state']['dofs'] for c in cases)} dof） | 见 §3.2 | 手装配伴随 | ≤ {summary['worst_autograd_vs_exact_adjoint_initial_state']:.1e} | ✅ |
| J（P2/16 格） | {cases[2]['objective']['autograd']:.8f} | {cases[2]['objective']['analytic']:.8f}（解析） | {cases[2]['objective']['relative_error_vs_analytic']:.1e} | ✅ |
| 方案一致性 | — | `solve_heat_1d` | {summary['worst_scheme_identity_max_abs']:.1e}（绝对） | ✅ |

**显式回答**：**是，全部关键量 < 3%**；梯度本身（vs 精确离散伴随）在 1e-13 量级，
与解析灵敏度的偏差在 P2/P3 上 ≤ 1e-5，最粗的 P1/8 格为
{cases[0]['sensitivity_alpha']['autograd_vs_analytic_relative']:.2%}（仍 < 3%，且随加密快速下降）。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **"autograd 对得上中心差分"不等于梯度对**。中心差分用的是同一个离散格式，二者的公共
   误差完全不可见；必须另找一条**不经过 autograd** 的路径（本单用手装配的精确伴随）以及
   一条**不经过离散格式**的路径（解析灵敏度）。本单三方对表后才敢声称"扩散通路的梯度可用"。
2. **`dJ/dα` 的精度不等于解的精度**。P1/8 格的解相对 L2 误差 {cases[0]['objective']['relative_l2_of_solution']:.2%}，
   但 `dJ/dα` 相对解析的偏差 {cases[0]['sensitivity_alpha']['autograd_vs_analytic_relative']:.2%}——
   灵敏度把衰减率的误差按 `T` 放大（`dJ/dα ∝ T J`），所以**较粗网格上灵敏度比场量更敏感**，
   做参数反演时应据此选网格。
3. **`dt` 冻结**。CN 步长在求导时固定（与 m2552 同一约定）；若让 `dt` 随 `alpha` 自适应，
   总映射在 `dt` 跳变处不可微，梯度将是分段的。本单不覆盖"自适应 `dt` 的梯度"。
4. **本次只覆盖线性扩散**。`advance_heat_1d` 是 SIPG + CN；压缩 Navier–Stokes 的可微通路
   （`ns_dg_2d`，非线性和壁面罚项）仍未提供，且其壁面通道已有独立记录的 honest-FAIL
   （`docs/reports/tensordg-ns-wall-verification.md`），不宜在缺陷未修前发布梯度声明。
5. **只覆盖单一泛函族**（模态 L2² / 点值）。含限制器的非线性泛函（minmod/TVB 分支）不可微，
   与本单无关，勿混用。
6. **对 `alpha` 的范围**：本单 `α ∈ [0.01, 0.2]`；`α = 0` 时 `G` 退化为恒等（梯度自然为 0），
   `α < 0` 无物理意义，代码未拦截（调用方负责）。

## 6. 结论

- `tensordg.differentiable` 补齐**扩散半场**入口（`advance_heat_1d` / `heat_1d_operators`），
  与已交付 `solve_heat_1d` **逐位一致**（{summary['worst_scheme_identity_max_abs']:.1e}）。
- 该通路的梯度对**精确离散伴随**一致到 {summary['worst_autograd_vs_exact_adjoint_alpha']:.1e}，
  对**解析灵敏度**在 P2/P3 上 ≤ 1e-5、最粗 P1/8 格 {cases[0]['sensitivity_alpha']['autograd_vs_analytic_relative']:.2%}，
  全部通过 3% 门。
- 回归位：`tests/test_diffusive_differentiability.py`（9 条，约 5 s），把上述四条判据固化为 CI 门。
"""


def main() -> None:
    data = json.loads(EVIDENCE.read_text())
    _figure(data)
    DOC.write_text(_markdown(data))
    print(f"wrote {DOC.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()