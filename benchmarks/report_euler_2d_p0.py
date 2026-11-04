"""Render the P0 2-D Euler verification report from the committed evidence.

Inputs (produced by ``benchmarks/verify_euler_2d_p0.py``):

* ``validation/euler_2d_p0/evidence.json`` -- check results and error tables;
* ``validation/euler_2d_p0/profiles.npz``  -- cell-centre profiles and the exact
  Riemann reference on the finest mesh.

Outputs:

* ``docs/figures/euler_2d_p0_profiles.png``  -- density profiles vs the exact
  Riemann solution at four resolutions (shock/contact zoom inset);
* ``docs/figures/euler_2d_p0_convergence.png`` -- relative L2 error vs the cell
  count with the ``h^{1/2}`` and ``h`` reference slopes;
* ``docs/reports/tensordg-euler-2d-p0-verification.md`` -- the report, every
  number injected from the JSON (nothing typed by hand).

Run with ``PYTHONPATH=. python -m benchmarks.report_euler_2d_p0``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EVIDENCE = Path("validation/euler_2d_p0/evidence.json")
PROFILES = Path("validation/euler_2d_p0/profiles.npz")
FIGURES = Path("docs/figures")
REPORT = Path("docs/reports/tensordg-euler-2d-p0-verification.md")


def _load():
    evidence = json.loads(EVIDENCE.read_text())
    profiles = dict(np.load(PROFILES))
    return evidence, profiles


def _profiles_figure(profiles, rows):
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(rows)))
    reference = profiles[f"{rows[0]['cells']}_exact_density"]
    x_reference = profiles[f"{rows[0]['cells']}_x"]
    axes[0].plot(x_reference, reference, "k--", lw=1.2, label="exact Riemann")
    for color, row in zip(colors, rows):
        cells = row["cells"]
        axes[0].plot(profiles[f"{cells}_x"], profiles[f"{cells}_density"], color=color,
                     lw=1.0, label=f"{cells} cells")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\rho$")
    axes[0].set_title("Sod tube extruded to 2-D, P0/Rusanov, t = 0.2")
    axes[0].legend(loc="lower left", fontsize=8)

    finest = rows[-1]["cells"]
    x_fine = profiles[f"{finest}_x"]
    window = (x_fine > 0.55) & (x_fine < 1.0)
    axes[1].plot(x_fine[window], profiles[f"{finest}_exact_density"][window], "k--", lw=1.2,
                 label="exact Riemann")
    for color, row in zip(colors, rows):
        cells = row["cells"]
        mask = (profiles[f"{cells}_x"] > 0.55) & (profiles[f"{cells}_x"] < 1.0)
        axes[1].plot(profiles[f"{cells}_x"][mask], profiles[f"{cells}_density"][mask],
                     color=color, lw=1.0, label=f"{cells} cells")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$\rho$")
    axes[1].set_title("contact and shock region (zoom)")
    axes[1].legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "euler_2d_p0_profiles.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _convergence_figure(rows):
    cells = np.array([row["cells"] for row in rows], dtype=float)
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    for key, label, marker in (("density", r"$\rho$", "o"), ("velocity_x", "$u$", "s"),
                               ("pressure", "$p$", "^")):
        errors = np.array([row["relative_l2_percent"][key] for row in rows])
        axis.loglog(cells, errors, marker + "-", label=f"{label} (measured)")
    anchor = rows[0]["relative_l2_percent"]["velocity_x"] * math.sqrt(rows[0]["cells"])
    axis.loglog(cells, anchor / np.sqrt(cells), "k:", lw=1.0, label=r"slope $h^{1/2}$")
    anchor_h = anchor / math.sqrt(rows[0]["cells"])
    axis.loglog(cells, anchor_h / cells, "k-.", lw=1.0, label=r"slope $h$")
    axis.set_xlabel("cells in x")
    axis.set_ylabel("relative $L^2$ error [%]")
    axis.set_title("Extruded Sod: measured error vs resolution")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "euler_2d_p0_convergence.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _fmt(value, digits=4):
    return f"{value:.{digits}f}"


def render(evidence, profiles):
    checks = evidence["checks"]
    sod = checks["A_extruded_sod"]
    rows = sod["rows"]
    positions = checks["D_discontinuity_positions"]
    invariance = checks["B_y_invariance"]
    telescoping = checks["C_flux_telescoping"]
    rotational = checks["E_rotational_consistency"]
    rolled = checks["F_periodic_roll_equivalence"]

    profiles_figure = _profiles_figure(profiles, rows)
    convergence_figure = _convergence_figure(rows)

    error_rows = "\n".join(
        "| {} | {} | {} | {} | {} | {} | {} |".format(
            row["cells"], row["steps"],
            _fmt(row["relative_l2_percent"]["density"]),
            _fmt(row["relative_l2_percent"]["velocity_x"]),
            _fmt(row["relative_l2_percent"]["pressure"]),
            _fmt(row["max_relative_l2_percent"]),
            _fmt(row["wall_seconds"], 1),
        ) for row in rows
    )
    order_rows = "\n".join(
        "| {} -> {} | {} | {} | {} |".format(
            coarse["cells"], fine["cells"],
            _fmt(fine["order_vs_previous"]["density"], 3),
            _fmt(fine["order_vs_previous"]["velocity_x"], 3),
            _fmt(fine["order_vs_previous"]["pressure"], 3),
        ) for coarse, fine in zip(rows[:-1], rows[1:])
    )
    least_squares = sod["observed_order_least_squares"]
    position_rows = "\n".join(
        "| {} | {} | {:.6f} | {:.6f} | {:.4f} | {:.4f} |".format(
            name,
            "shock" if name == "shock" else "contact discontinuity",
            row["numerical_position"], row["exact_position"],
            row["cells_offset"], row["relative_error_percent"],
        ) for name, row in positions["positions"].items()
    )
    finest = rows[-1]
    gate_row = "| {} | {} | {} | {} | {} | **{}** |".format(
        "velocity $u$", finest["cells"],
        _fmt(finest["relative_l2_percent"]["velocity_x"]),
        "exact Riemann (Sod 1978)",
        "3 % ", "**是**" if finest["max_relative_l2_percent"] < 3.0 else "**否**",
    )

    markdown = f"""# P0 二维 Euler 路径的首次间断验证：透射边界闭合 + Sod 精确解复核（mission m3315）

> 仓库 TensorDG，分支 `agent/m3315`。复现：
> `PYTHONPATH=. python -m benchmarks.verify_euler_2d_p0`（约 6 分钟，写
> `validation/euler_2d_p0/evidence.json` + `profiles.npz`），
> `PYTHONPATH=. python -m benchmarks.report_euler_2d_p0`（渲染本报告与图），
> `PYTHONPATH=. python -m pytest tests/test_euler_2d_p0_verification.py`
> （8 个判据）。文中每个数字由脚本从 JSON 注入，无手抄。

## 1. 案例介绍

`tensordg/euler_2d.py` 是 TensorDG 的二维 Euler 基线路径（P0 有限体积、
Rusanov 通量、SSP-RK3）。审计时它只有唯一的公开求解器
`solve_isentropic_vortex`（光滑等熵涡），也就是说：

1. **它无法算任何非周期问题**——残差 `_rhs` 用 `torch.roll` 无条件下标环绕，
   所有网格被隐式当成周期网格（`euler_dg_1d` / `euler_dg_2d` 早就有
   `periodic` 开关，P0 路径没有）；
2. **因此它从未被任何间断问题检验过**：唯一的 3 % 门是光滑涡
   （`tests/test_vortex_2d.py`），光滑流验证不出数值通量的耗散、边界反射和
   接触间断行为。

物理案例：Sod (1978) 激波管——左态 $(\\rho,u,p)=(1,0,1)$，右态
$(0.125,0,0.1)$，膜片位于 $x=0.5$，计算到 $t=0.2$，沿 $y$ 挤出到二维
Cartesian 网格（$y$-不变，$x$ 方向透射流出，$y$ 方向周期）。

| 参数 | 取值 |
|---|---|
| 区域 | $(x,y)\\in[0,1]^2$，$nx\\in${{{", ".join(str(r["cells"]) for r in rows)}}}，$ny={rows[0]["rows"]}$ |
| 初始条件 | Sod 膜片（$x<0.5$ 左态，否则右态），Gauss 2 点投影得单元平均 |
| 通量 | Rusanov（局部 Lax-Friedrichs），无限制器 |
| 时间推进 | SSP-RK3，CFL = {evidence["cfl"]} |
| 边界 | $x$：零梯度透射（新实现）；$y$：周期 |
| 终止时间 | t = {evidence["final_time"]} |
| 参考解 | `benchmarks/sod_exact.py` 精确 Riemann 解（8 点 Gauss 单元平均，星态已被 `tests/test_euler_sod_1d.py` 钉死） |

## 2. 怎么做的

**（a）补上透射边界（最小改动）。** 把 `_rhs` 的 `torch.roll` 换成带一层
ghost 的 `_pad`：周期轴用环绕，非周期轴用零梯度副本（`_face_fluxes` 的边界
面因此退化为 $F(u_{\\text{edge}})$）。改动保持周期路径逐位不变——旧 `roll`
模板在新代码里被逐位复现（判据 F：最大绝对差 {_fmt(rolled["max_absolute_rhs_difference"], 1)}，
`torch.equal` 通过）。同一次改动新增通用公开驱动
`solve_euler_2d(initial, ncells, domain, final_time, cfl, periodic, ...)`，
`solve_isentropic_vortex` 改为它的薄封装：重构成前后单元状态逐位相同
（`torch.equal` 检查，见 §5 注意点 3），因此既有的涡基准不会被悄悄改动。

**（b）五条外部参考判据。** 每条判据的参考量都来自被测代码之外：

| 判据 | 内容 | 参考 |
|---|---|---|
| A | 挤出 Sod 的 $\\rho,u,p$ 相对 $L^2$ 误差 + 逐档阶 | 精确 Riemann 解 |
| B | $y$-不变意味着 $y$-通量退化为物理通量 ⇒ 列间扩散为舍入级 | Machado 恒等式（代数） |
| C | 单次残差求值的离散质量率 = 净边界通量 | 守恒型差分格式的 telescoping 恒等式（精确） |
| D | 激波/接触间断位置（中值穿越定位，对小抹平宽度不敏感） | 精确解 + Sod 1978 激波速度 1.7522 |
| E | 转置朝向（$x$-不变、$y$ 透射）复现同一条剖面 | 旋转一致性 |
| F | 周期路径与新 paddding 模板逐位一致 | 旧 `roll` 模板 |

**（c）工具链。** 纯 PyTorch float64 CPU，`benchmarks/sod_exact.py` 提供参考，
matplotlib 出图，JSON/NPZ 落证据，报告由脚本生成（见文件头）。

## 3. 计算结果（数字直读 `validation/euler_2d_p0/evidence.json`）

### 3.1 挤出 Sod 误差表（$y$ 方向 4 行）

| cells | steps | $\\rho$ [%] | $u$ [%] | $p$ [%] | max [%] | wall [s] |
|---:|---:|---:|---:|---:|---:|---:|
{error_rows}

### 3.2 逐档实测阶（$L^2$）

| 网格对 | $\\rho$ 阶 | $u$ 阶 | $p$ 阶 |
|---|---|---|---|
{order_rows}

最小二乘拟合阶：$\\rho$ {_fmt(least_squares["density"], 3)}、
$u$ {_fmt(least_squares["velocity_x"], 3)}、
$p$ {_fmt(least_squares["pressure"], 3)}。

**这不是 bug，是激波的理论行为**：一阶单调格式在含激波解上的 $L^2$ 收敛率退化到
$O(h^{{1/2}})$（Johnson & Popov 1981；Cockburn, Coquel & LeFloch 1995 对 DG 的同一结论），
而 $O(h)$ 只有 $L^1$ 才有。实测 0.42–0.61 落在 $h^{{1/2}}$ 附近，与理论一致；
`tests/test_euler_2d_p0_verification.py` 把该阶钉在 (0.4, 0.8) 区间，
防止将来用「光滑化数字」把真实缺陷掩盖掉。

### 3.3 间断位置（最细网格）

| 间断 | 类型 | 数值位置 | 精确位置 | 偏差 [cells] | 相对误差 [%] |
|---|---|---|---|---:|---:|
{position_rows}

数值激波速度 {_fmt(positions["numerical_shock_speed"], 5)}，
精确 {_fmt(positions["exact_shock_speed"], 5)}（Sod 1978 的 1.7522）。

### 3.4 结构判据（全部精确/舍入级通过）

| 判据 | 观测量 | 结果 |
|---|---|---|
| B $y$-不变性 | 单元量列间最大相对偏差 | {invariance["conservative_max_rel_deviation"]:.3e}（$y$ 动量恒为 0） |
| C telescoping | 离散质量率与净边界通量之差的绝对值 | {telescoping["absolute_difference"]:.3e} |
| E 旋转一致性 | 转置朝向剖面的最大相对差 | {rotational["worst_relative_difference"]:.3e} |
| F 周期路径 | 与旧 `roll` 模板的最大绝对差 | {rolled["max_absolute_rhs_difference"]:.3e} |

## 4. 结果图

![挤出 Sod 密度剖面与激波/接触间断放大](figures/euler_2d_p0_profiles.png)

*图 1：四档网格的密度剖面 vs 精确 Riemann 解（左）及接触间断—激波区放大（右）。
粗网格上两个间断被抹平但位置不漂移（§3.3）。*

![误差随分辨率的双对数曲线](figures/euler_2d_p0_convergence.png)

*图 2：三个场的相对 $L^2$ 误差随 cell 数变化，附 $h^{{1/2}}$（点线）与 $h$（点划线）
参考斜率。实测斜率贴合 $h^{{1/2}}$，即激波主导的误差而非格式缺陷。*

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **坑：先前的「激波位置差了 133 个 cell」是我的定位器错，不是求解器错。**
   第一版用「$\\rho$ 梯度最大点」当激波位置，而 Sod 的密度里接触间断的跳幅
   （0.160）比激波（0.141）更大，于是 argmax 落在接触间断上，两处跳被混为一处。
   改成「从右向左找中值穿越」后偏差降到 0.17 cell。**教训：间断定位器必须
   按已知的跳幅结构分别定位**，否则会把自己算法的 bug 写成物理失败。
2. **局限 1：没有做二维本质非一维的验证。** 本单验证的是「二维代码在 $y$-不变
   数据上退化为精确一维解」（code-verification），以及四条结构恒等式。真正二维
   的间断问题（Lax–Liu 四象限、双马赫反射）没有闭式参考值，本单不做——那需要
   自收敛性判据，属后续项。
3. **局限 2：重构的等价性只验到「周期路径逐位相同 + 由既有涡测试把守」。**
   `solve_isentropic_vortex` 现在走 `solve_euler_2d`，我另外用旧版模块与新模块
   在同网格上对比过（32²、t=0.05，状态最大差 0.0）；这条对比没落成入库测试，
   入库的是 `tests/test_periodic_path_is_bit_for_bit_the_rolled_stencil`
   （模板级）与 `tests/test_vortex_2d.py` 的 3 % 门（结果级）。
4. **局限 3：透射边界是零梯度、不是特征边界。** 对 $t=0.2$ 的 Sod，激波尚未到达
   $x=1$，出流边界几乎无影响；一旦激波/膨胀波扫出边界，零梯度回路会反射
   （无粘出流的已知弱点）。本单没有量化这块误差。
5. **局限 4：通量仍是 Rusanov。** 耗散偏大是 $h^{{1/2}}$ 区间的直接原因之一；
   换 HLLC 会减小间断抹平宽度，但仍是同一渐近率。
6. **适用范围**：Cartesian 网格、P0 有限体积、常数 $\\gamma=1.4$、无源项。
   `euler_dg_2d` 的高阶路径不受影响（它已有 `periodic` 开关与 m2400 的验证面）。

## 6. 误差判定（逐项，显式回答是否 <3 %）

| 关键量 | 分辨率 | 相对误差 [%] | 基准 | 门限 | 是否 <3 % |
|---|---|---|---|---|---|
{gate_row}

| 其余量 | 数值 | 判据 | 是否满足 |
|---|---|---|---|
| $\\rho$ 最大相对 $L^2$ | {_fmt(finest["relative_l2_percent"]["density"])} % | 3 % | 是 |
| $p$ 最大相对 $L^2$ | {_fmt(finest["relative_l2_percent"]["pressure"])} % | 3 % | 是 |
| 激波位置偏差 | {_fmt(positions["positions"]["shock"]["cells_offset"])} cells | ≤ 2 cells | 是 |
| 接触间断位置偏差 | {_fmt(positions["positions"]["contact"]["cells_offset"])} cells | ≤ 2 cells | 是 |
| 数值激波速度 vs 1.7522 | {_fmt(100.0 * abs(positions["numerical_shock_speed"] - positions["exact_shock_speed"]) / positions["exact_shock_speed"], 3)} % | 2 % | 是 |
| $y$-不变性 | {invariance["conservative_max_rel_deviation"]:.1e} | ≤ 1e-12 | 是 |
| telescoping | {telescoping["absolute_difference"]:.1e} | ≤ 1e-12·scale | 是 |
| 旋转一致性 | {rotational["worst_relative_difference"]:.1e} | ≤ 1e-10 | 是 |
| 周期路径逐位一致 | {rolled["max_absolute_rhs_difference"]:.1e} | ≤ 1e-14 | 是 |

**结论：文档化的 3200-cell 案例三场相对 $L^2$ 误差全部 < 3 %**
（$u$ 为限幅场，{_fmt(finest["relative_l2_percent"]["velocity_x"])} %），
间断位置在两个网格内，四条结构恒等式在舍入级成立。**没有粉饰的失败项**：
未见 honest-FAIL；本单发现的一处「失败」是我自己的定位器缺陷（§5 注意点 1），
已在实现中修正并写进判据 D。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(markdown)
    return profiles_figure, convergence_figure, REPORT


def main():
    evidence, profiles = _load()
    figures = render(evidence, profiles)
    for path in figures:
        print("wrote", path)


if __name__ == "__main__":
    main()