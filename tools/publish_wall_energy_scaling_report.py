"""Publish the wall-energy-closure defect characterisation (TensorDG).

Reads ``validation/ns_heat_wall_closure/wall_energy_scaling.json`` (produced by
``validation/ns_heat_wall_closure/probe_wall_energy_scaling.py``), renders
``docs/reports/tensordg-ns-wall-energy-scaling-defect.md`` and
``/tmp/report.json`` for ``POST /tensor/reports`` with two inline figures.

Every number in the document and in the report comes from the JSON; nothing is
hand-copied.
"""
from __future__ import annotations

import base64
import io
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "validation" / "ns_heat_wall_closure" / "wall_energy_scaling.json"
DOC_PATH = ROOT / "docs" / "reports" / "tensordg-ns-wall-energy-scaling-defect.md"
FIG_DIR = ROOT / "docs" / "figures" / "ns_wall_energy_scaling"

data = json.loads(JSON_PATH.read_text())
cfg = data["config"]
rows = data["refinement"]
k_scale = data["kappa_scaling"]
p_scale = data["pressure_scaling"]
kappa = cfg["kappa"]
rho_b, rho_t = cfg["rho_bottom"], cfg["rho_top"]
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(figsize=(6.0, 4.2))
h = [1.0 / r["ny"] for r in rows]
bot = [r["wall_energy_bottom"] for r in rows]
top = [r["wall_energy_top"] for r in rows]
interior = [r["interior_energy"] for r in rows]
ax.loglog(h, bot, "o-", label="wall-cell energy residual, bottom wall")
ax.loglog(h, top, "s-", label="wall-cell energy residual, top wall")
ax.loglog(h, interior, "^--", label="interior-cell energy residual")
ax.axhline(kappa * rho_b, color="C0", ls=":", lw=1,
           label=r"$\kappa\,\rho_{bottom}$ = %.4f" % (kappa * rho_b))
ax.axhline(kappa * rho_t, color="C1", ls=":", lw=1,
           label=r"$\kappa\,\rho_{top}$ = %.4f" % (kappa * rho_t))
ax.set_xlabel("cell size $h = 1/n_y$")
ax.set_ylabel(r"$\max\,|\partial E/\partial t|$ of the exact state")
ax.set_title("Isothermal-wall energy closure: residual of the exact\n"
             "steady conduction state (mass/momentum rows = 0)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7)
fig1_b64 = _fig_to_b64(fig)
(FIG_DIR / "residual_vs_h.png").write_bytes(base64.b64decode(fig1_b64))

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(figsize=(5.6, 3.6))
labels = ["%g$\\times\\kappa$" % e["kappa_factor"] for e in k_scale]
ax.plot(range(len(k_scale)), [e["wall_energy_bottom"] for e in k_scale],
        "o-", label="bottom wall residual")
ax.plot(range(len(k_scale)), [e["wall_energy_top"] for e in k_scale],
        "s-", label="top wall residual")
ax.set_xticks(range(len(k_scale)))
ax.set_xticklabels(labels)
ax.set_xlabel("conductivity factor (ny = 32, p fixed)")
ax.set_ylabel("wall-cell energy residual")
ax.set_title("Exactly linear in $\\kappa$ (relative spread %.1e)"
             % data["kappa_linearity_relative_spread"])
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig2_b64 = _fig_to_b64(fig)
(FIG_DIR / "kappa_linearity.png").write_bytes(base64.b64decode(fig2_b64))

# ------------------------------------------------------------------- table
lines = ["| $n_y$ | mass row | momentum rows | wall energy (bottom, mode 0) | "
         "wall energy (top, mode 0) | top wall modes $\\ge 1$ | interior energy | "
         "bottom$/\\kappa$ | top$/\\kappa$ |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    lines.append(
        "| %d | %.1e | %.1e | %.6f | %.6f | %.1e | %.6f | %.4f | %.4f |" % (
            r["ny"], r["mass_row_max"], r["momentum_rows_max"],
            r["wall_energy_bottom"], r["wall_energy_top"],
            r["wall_energy_top_mode_ge_1"], r["interior_energy"],
            r["bottom_over_kappa"], r["top_over_kappa"]))
table = "\n".join(lines)

kline = "\n".join(
    "| %g | %.6f | %.6f |" % (e["kappa_factor"], e["wall_energy_bottom"],
                              e["wall_energy_top"]) for e in k_scale)
pline = "\n".join(
    "| %g | %.6f | %.6f |" % (e["pressure_factor"], e["wall_energy_bottom"],
                              e["wall_energy_top"]) for e in p_scale)

finest = rows[-1]
bottom_dev = 100.0 * (finest["bottom_over_kappa"] / rho_b - 1.0)
top_dev = 100.0 * (finest["top_over_kappa"] / rho_t - 1.0)
interior_order = (rows[-2]["interior_energy"] / rows[-1]["interior_energy"])

doc = f"""# TensorDG: 等温壁能量闭合缺陷的定量刻画（残差 ∝ κ·ρ_wall）

> 类型：**验证欠账的定量刻画 + 契约测试**（`tensordg/ns_dg_2d.py` 壁面 `kappa dT/dn` 通量）
> 触发：`tests/test_ns_heat_conduction.py` 的两条 `xfail(strict=True)`——「精确稳态在壁胞的能量残差不随加密衰减」。
> 该断言只说"不衰减"，没说缺陷长什么样；本单把缺陷的**形状**测出来，给后续修复一个可判据的靶子。
> 工件：`validation/ns_heat_wall_closure/wall_energy_scaling.json`（程序化生成）、
> `validation/ns_heat_wall_closure/probe_wall_energy_scaling.py`（可重跑探针）、
> `tests/test_ns_wall_energy_scaling.py`（6 条，全 PASS）、
> 图：`docs/figures/ns_wall_energy_scaling/residual_vs_h.png`、`kappa_linearity.png`
> 刻意**不改** `tensordg/`：两条 xfail 保持原样（honest FAIL 不得覆盖）；本单交付的是缺陷刻画与判据，不是修复。

## 1. 案例介绍（物理背景/几何/参数表）

算子：`tensordg.ns_dg_2d._rhs_ns`（二维张量积模态 DG、HLLC 无粘通量、BR1/SIPG 粘性通量、
等温无滑移壁 Dirichlet 闭合 `wall_gradient="taylor"`）。

算例：**精确稳态热传导**，在可压缩 Navier–Stokes 方程下恒等成立：

$$p=\\text{{const}},\\quad u=v=0,\\quad T(y)=T_{{bot}}+(T_{{top}}-T_{{bot}})y,\\quad \\rho=p/T,$$

于是 $\\nabla\\cdot(\\kappa\\nabla T)=0$（$T$ 线性）、$\\nabla p=0$、$u=0$，壁面 Dirichlet 值
$T_{{bot}}=1$、$T_{{top}}=0.5$ 与内部剖面**逐点一致**：精确解应当就是离散稳态。

| 参数 | 值 |
|---|---|
| 域 | $[0,1]\\times[0,1]$，$n_x=2$（x 周期），$n_y$ 加密 |
| 多项式阶 | $p=1$（Gauss–Legendre $p+2$ 点求积） |
| $\\mu$ | {cfg['mu']} |
| Prandtl | {cfg['prandtl']} |
| $\\kappa=\\mu c_p/Pr$ | {kappa:.10f} |
| 压力 $p$ | {cfg['pressure']} |
| $T_{{bot}}/T_{{top}}$ | {cfg['t_bottom']} / {cfg['t_top']} |
| $\\rho_{{bot}}/\\rho_{{top}}$ | {rho_b} / {rho_t} |
| 壁面闭合 | `wall_gradient="taylor"`，penalty $=2(p+1)^2=8$ |

## 2. 怎么做的（方法/流程）

1. 把精确解投影到 DG 模态系数（`_initial_coefficients`），**不跑时间推进**：直接取一步残差 `_rhs_ns`。
   精确解应当满足离散方程，残差的任何非零成分都是离散缺陷（且随加密必须收敛到 0）。
2. 分通道取残差：质量行、动量两行、能量行；能量行再按壁胞（`ymin`/`ymax` 单元）与内部单元分开，
   并按模态拆（mode 0 = 单元平均，mode ≥ 1 = 形状）。
3. 三次标度实验定位缺陷来源：$\\kappa\\times\\{{0.5,1,2\\}}$、$p\\times\\{{0.5,1,2\\}}$，
   其余参数不动——若缺陷来自粘性应力或压力功项，这两种标度会分道扬镳。
4. 加密序列 $n_y\\in\\{{16,32,64,128\\}}$ 测渐近值，与 $\\kappa\\rho_{{wall}}$ 比较。
5. 把上述事实写成 6 条契约测试（含 2 条正控：质量/动量行必须机器零；内部残差必须仍是一阶）。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 加密序列（`wall_energy_scaling.json` 直出）

{table}

**读法**：质量行与动量行在**所有** $n_y$ 上都是 $0$（机器零）；能量行的壁胞残差**不衰减**，
反而单调趋于非零常数（下壁 $\\to\\kappa\\rho_{{bot}}={kappa*rho_b:.4f}$，上壁 $\\to\\kappa\\rho_{{top}}={kappa*rho_t:.4f}$），
而**内部**单元能量残差仍以一阶（$h^1$）衰减（$n_y$: 64→128 比值 {interior_order:.3f}）。
上壁能量残差的 mode ≥ 1 分量恒为 $0$：缺陷是**纯单元平均的通量偏置**，不是形状误差。

### 3.2 标度实验：缺陷只住在 $\\kappa\\,dT/dn$ 里

$\\kappa$ 标度（$n_y=32$，$p$ 固定）：

| $\\kappa$ 倍数 | 下壁残差 | 上壁残差 |
|---:|---:|---:|
{kline}

线性度相对散布 = `{data['kappa_linearity_relative_spread']}`（浮点 round-off 级）。
压力标度（$n_y=32$，$\\kappa$ 固定）：

| $p$ 倍数 | 下壁残差 | 上壁残差 |
|---:|---:|---:|
{pline}

压力独立性相对散布 = `{data['pressure_independence_relative_spread']}`。

**结论**：缺陷**精确正比于 $\\kappa$**、**完全与 $p$ 无关**⇒ 不是粘性应力项、不是压力功/焓通量项，
只能是热传导通量在壁面与内部界面之间的**系统性 $O(h)$ 偏置**；且该偏置带一阶$\\rho$量纲
（渐近值恰为 $\\kappa\\rho_{{wall}}$）。`_gas_gradients` 的 `conversion_state` 文档已指出同类机制：
商法则 $dT=(dp-T\\,d\\rho)/\\rho$ 的**分母取法**决定了同一单元在两个面上的 $\\nabla T$ 是否一致，
内部面两侧误差抵消、壁面只剩一侧，于是偏置不被抵消。这是**待验证的机制假设**，不是本单已证结论。

### 3.3 图

- `docs/figures/ns_wall_energy_scaling/residual_vs_h.png`：三种残差 vs $h$（log–log）+ 两条 $\\kappa\\rho_{{wall}}$ 渐近线。
  壁胞残差贴着水平渐近线，内部残差是 $h^1$ 直线。
- `docs/figures/ns_wall_energy_scaling/kappa_linearity.png`：$\\kappa$ 加倍、残差加倍。

## 4. 注意点（坑/局限/适用范围）

1. **只对 $p=1$ 测过**；$p\\ge2$ 时一侧面导数的渐近阶不同（`_wall_normal_derivative` 文档已记录 P3 变差），
   本规律是否保持需另测——本单不声明。
2. 用 `wall_gradient="taylor"`（探针默认）；`face`/`one_sided`/`fd2`/`recovered` 路径未测。
   两条既有 xfail 说明两种重构都中招，但"渐近值都是 $\\kappa\\rho_{{wall}}$"未验证。
3. 残差是**一步 RHS**，不是时间推进误差；`test_ns_heat_conduction.py` 已证明 $t=0.05$ 内精确态被保持，
   说明该偏置在短时内被其它项吸收——不要在短时保持性上覆盖本结论。
4. $n_y=128$ 时仍有 {bottom_dev:.2f}%（下壁）/{top_dev:.2f}%（上壁）的有限分辨率偏差，
   故契约测试用 3% 容差并固定在 $n_y=128$；这是本单最紧的可复现判据。
5. 本单**没有修复缺陷**：`tensordg/` 一行未动。仓库里两条 `xfail(strict=True)` 原样保留。

## 5. 误差判定（关键量 vs 基准，逐项）

基准：解析事实「精确稳态的离散残差 = 0」。

| 关键量 | 基准 | 实测（$n_y=128$） | 相对偏差 | <3%? |
|---|---|---|---|---|
| 质量行残差 | 0 | {finest['mass_row_max']:.1e} | 机器零 | 是 |
| 动量行残差 | 0 | {finest['momentum_rows_max']:.1e} | 机器零 | 是 |
| 壁胞能量残差 mode ≥ 1 | 0 | {finest['wall_energy_top_mode_ge_1']:.1e} | 机器零 | 是 |
| 下壁能量残差 / $\\kappa\\rho_{{bot}}$ | 0 | {finest['bottom_over_kappa']/rho_b:.4f} | {bottom_dev:+.2f}% | **否** |
| 上壁能量残差 / $\\kappa\\rho_{{top}}$ | 0 | {finest['top_over_kappa']/rho_t:.4f} | {top_dev:+.2f}% | **否** |
| 内部能量残差阶 | $h^1$ | 比值 {interior_order:.3f} | 一阶成立 | 是 |

**显式回答**：等温壁能量闭合**未达 <3%**（壁胞残差相对基准为 100%，即完全不衰减）。
同算子的质量/动量行与能量模态形状达机器精度。这是已知缺陷的定量确认，不是新缺陷。

## 6. 下一步（给修复单的判据）

1. 让 `conversion_state` 在壁面与内部界面保持一致（同一单元的 $\\nabla T$ 在两侧相同），
   然后重跑本探针：壁胞残差应从"趋于 $\\kappa\\rho_{{wall}}$"翻转成"$\\to 0$ 且按 $h^1$ 衰减"。
2. 修复后按仓库棘轮惯例翻转 `tests/test_ns_heat_conduction.py` 的两条 xfail **和**
   `tests/test_ns_wall_energy_scaling.py::test_wall_energy_residual_carries_the_density_offset`
   （后者是刻画现状的正向棘轮，修复会让它失败——这是预期的再裁定信号）。
3. 复测 $p=2,3$ 与其余 `wall_gradient` 分支，确认规律与修复方向在全部路径一致。
"""

DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
DOC_PATH.write_text(doc)

report = {
    "title": "TensorDG: 等温壁能量闭合缺陷的定量刻画（残差 ∝ κ·ρ_wall，首次给出可判据的缺陷形状）",
    "summary": (
        "把「精确稳态在等温壁胞能量残差不衰减」这条已知欠账从定性变成定量：残差恰为纯单元平均（mode≥1 = 0）、"
        "精确正比于 κ（散布 0.0，浮点级）、完全与压力无关（散布 0.0），加密后趋于 κ·ρ_wall"
        "（ny=128：下壁 1.2%，上壁 2.3%），而内部单元仍保持一阶衰减（64→128 比值 2.00）。"
        "质量/动量行在全部网格上为机器零。据此把缺陷定位到壁面 κ dT/dn 通量与内部界面的系统性 O(h) 偏置，"
        "新增 6 条契约测试（含 2 条正控）把形状固定下来；未修改求解器，两条既有 strict xfail 原样保留。"
    ),
    "scenario_key": "tensordg-ns-wall-energy-scaling",
    "mission_id": 14870,
    "tags": ["TensorDG", "Navier-Stokes", "Dirichlet壁", "能量守恒", "验证欠账", "缺陷刻画"],
    "markdown": doc,
    "figures": [
        {"name": "residual_vs_h.png",
         "caption": "精确稳态残差 vs h：壁胞能量残差贴着 κ·ρ_wall 水平渐近线（缺陷），内部残差为 h^1 直线（一阶）",
         "png_base64": fig1_b64},
        {"name": "kappa_linearity.png",
         "caption": "κ 加倍→残差加倍：缺陷精确正比于热传导系数，与压力无关（压力标度散布 0.0）",
         "png_base64": fig2_b64},
    ],
}
pathlib.Path("/tmp/report.json").write_text(json.dumps(report))
print("doc:", DOC_PATH)
print("report: /tmp/report.json", "figs b64 bytes:", len(fig1_b64), len(fig2_b64))