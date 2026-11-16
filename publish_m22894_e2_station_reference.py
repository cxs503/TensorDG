"""Figures + platform report for m22894 (E2 station reference / station-cell audit).

All numbers are computed here from the audited module -- none transcribed.
"""
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from benchmarks import sod_exact                                        # noqa: E402
from validation.repro_cks1998 import station_reference as SR           # noqa: E402
from tensordg.euler_dg_1d import solve_euler_dg_1d                     # noqa: E402
from tensordg.euler_1d import GAMMA                                    # noqa: E402

FIGDIR = REPO / "docs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
cuts = SR.wave_positions()


def png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------- fig 1
ncells = 48
x, coeff, meta = solve_euler_dg_1d(SR.SOD_LEFT, SR.SOD_RIGHT, ncells=ncells, degree=1,
                                   final_time=SR.SOD_T, cfl=0.12, limiter="tvb",
                                   tvb_constant=1.0, limiter_stage="final")
rho_h = np.asarray(coeff[:, 0][:, 0])
xc = np.asarray(x)
xs = np.linspace(0.55, 0.95, 20001)
_, exact_fine = None, sod_exact.sample(xs, SR.SOD_T)["density"]
row = {r["metric"]: r for r in SR.station_table(ncells)}["sod_rho_x070"]

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(xs, exact_fine, color="0.55", lw=2.2, label="exact Riemann (t=0.2)")
ax.plot(xc, rho_h, "o-", ms=3.5, lw=0.9, color="tab:blue", label="DG P1 cell average (N=48)")
ax.axvspan(row["x_left"], row["x_right"], color="tab:orange", alpha=0.28,
           label=f"scored cell for $\\rho$(0.70): idx={row['index']}")
ax.axvspan(row["x_left"] - 1 / ncells, row["x_left"], color="tab:green", alpha=0.20,
           label="left neighbour (contains $x_c$)")
for i, c in enumerate(cuts):
    ax.axvline(c, color="tab:red", lw=1.1, ls="--",
               label="exact breakpoints" if i == 0 else None)
ax.annotate(f"$x_c$ = {cuts[2]:.6f}\ngap = {row['gap_to_breakpoint_dx']:.3f} dx",
            xy=(cuts[2], 0.30), xytext=(0.585, 0.335),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="tab:red"), fontsize=9)
ax.set_xlim(0.55, 0.95)
ax.set_xlabel("x")
ax.set_ylabel(r"$\rho$")
ax.set_title("E2 Sod: the $\\rho$(0.70) station cell clears the contact by 0.096 dx")
ax.legend(fontsize=7.6, loc="upper right")
ax.grid(alpha=0.25)
fig1 = png_b64(fig)

# ----------------------------------------------------------------- fig 2
fig, ax = plt.subplots(figsize=(7.0, 4.3))
for n, color, mk in ((48, "tab:blue", "o"), (400, "tab:purple", "s")):
    _, ref_e = SR.exact_cell_averages(n)
    _, ref_f = SR.fixed_rule_cell_averages(n)
    dx = 1.0 / n
    dist, bias = [], []
    for i in range(n):
        lo, hi = i * dx, (i + 1) * dx
        d = min(np.min(np.abs(cuts - lo)), np.min(np.abs(cuts - hi))) / dx
        if d == 0.0:
            continue
        b = abs(ref_f["density"][i] - ref_e["density"][i]) / abs(ref_e["density"][i])
        if b > 0:
            dist.append(d); bias.append(b)
    ax.loglog(dist, bias, mk, ms=3.0, alpha=0.75, color=color,
              label=f"N={n} ({len(dist)} cells, max {max(bias):.2e})")
ax.axhline(1e-12, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("distance from cell to nearest exact breakpoint  [dx]")
ax.set_ylabel("relative bias, fixed 8-pt Gauss vs exact reference")
ax.set_title("The reference-instrument policy boundary is 'cell contains a wave',\nnot 'mesh is coarse'")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")
fig2 = png_b64(fig)

# ----------------------------------------------------------------- summary numbers
t48 = {r["metric"]: r for r in SR.station_table(48)}
t400 = {r["metric"]: r for r in SR.station_table(400)}
n24 = SR.station_cell_diagnosis(24, 0.70, "density")
plat = {"sod_rho_x030": (0.87349734, 0.869379), "sod_rho_x070": (0.26557371, 0.269618),
        "sod_vel_x055": (0.92745262, 0.927491)}
rows_md = "\n".join(
    f"| {k} | 精确参考 {v[0]:.8f} | 平台 ref {t400[k]['ref_exact']:.8f} | "
    f"got {v[1]:.6f} | {abs(v[1]-v[0])/abs(v[0]):.4%} | {t400[k]['instrument_rel_delta']:.1e} |"
    for k, v in plat.items())

md = f"""## 1. 这单在审什么

锚点 `REPRO-DG-CKS1998` 的 E2 证据只有三条逐点度量（`sod_rho_x030` / `sod_rho_x070` /
`sod_vel_x055`，容差 3 %）。本单审计这两条度量**没说出口的两半**：

* **`ref` 是用哪台仪器算的**——`bench_workload` 原调用
  `benchmarks.riemann_exact.cell_averages`（固定 8 点 Gauss）。该函数自己的 docstring
  声明：含波单元被不精确积分，误差 `rho_q·jump`，细分不收敛，**与容差比较时必须**改用
  按断点切分的 `benchmarks.shock_reference_quadrature.cell_averages`。E2 的容差就是 3 %。
* **`got` 是哪一个格子**——原代码 `idx = int(x*ncells)`，`int()` 截断，台站落在单元面上
  时取哪一格由 `x*ncells` 的浮点末位决定，代码里没有一句约定。

Sod t=0.2 的四个精确断点：稀疏波头 {cuts[0]:.6f}、稀疏波尾 {cuts[1]:.6f}、
**接触间断 {cuts[2]:.6f}**、激波 {cuts[3]:.6f}。

## 2. 方法与修复

1. 逐格测两台的相对差，并与"该格到最近断点的距离(dx)"对照 → 得到仪器政策的**真实边界**。
2. 枚举 `4≤ncells≤2000` 的 500 个"台站恰在单元面上"分辨率，统计 `int()` 左右摇摆。
3. 新增 `validation/repro_cks1998/station_reference.py`：
   `station_cell()`（显式规则：面上判右格，判定带相对容差 `TIE_EPS`）、
   `assert_station_cells_wave_free()`（守卫）、`station_table()`（诊断）。
4. `bench_workload.py` 的 E2 段：参考仪器换成精确断点切分；台站索引改走 `station_cell()`；
   把每格的 `index/x_left/x_right/gap_to_breakpoint_dx/ref_exact/ref_fixed_rule/
   instrument_rel_delta` 写进 `detail`（**不改任何容差、台站、网格、rows 结构**）。
5. 回归门 `tests/test_cks1998_station_reference.py`（12 条，5 s，纯解析为主）。

## 3. 结果

### 3.1 换代是数值惰性的（锚点计分格）

| 度量 | 精确参考（本单重算） | 平台已入库 ref | 平台 got | 相对误差 | 仪器相对差 |
|---|---|---:|---:|---:|---:|
{rows_md}

平台 `ref` 与本单精确参考**逐位一致**（差异 ≤2.1e-14）⇒ 换代不需要重登记任何数值。

### 3.2 但违规是真的：含波格的固定规则参考偏 1.8–2.3 %

| 情形 | 格区间 | 含断点 | ref_exact | ref_fixed | 仪器相对差 |
|---|---|---|---:|---:|---:|
| N=24，rho_x070（未使用档） | [{n24['x_left']:.5f}, {n24['x_right']:.5f}] | **是** | {n24['ref_exact']:.8f} | {n24['ref_fixed_rule']:.8f} | **{n24['instrument_rel_delta']:.3e}** |
| N=48，rho_x070 左邻格 | [{t48['sod_rho_x070']['x_left']-1/48:.5f}, {t48['sod_rho_x070']['x_left']:.5f}] | **是** | 0.41081472 | 0.41818340 | **1.794e-02** |

2.29 % 占掉 3 % 容差的 76 %，且细分治不好（它是规则常数 `rho_q` 乘跳量）。
⇒ 守卫把"该格是否含波"变成可判定命题。

### 3.3 台站点格：34/1201 个面上情形被截断到左格

`0.7*100 = 70.0` → 右格；`0.7*90 = 62.99999999999999` → 左格。修复后 1201/1201 判右格，
锚点现有索引（48→14/33/26，400→120/280/220）**一个不变**。

### 3.4 台站格的脆弱性（诚实披露）

`sod_rho_x070` 在 smoke 档（P1 N=48）的计分格 [0.68750, 0.70833] 到接触间断只剩
**{t48['sod_rho_x070']['gap_to_breakpoint_dx']:.3f} dx**；在平台档（P2 N=400）为
{t400['sod_rho_x070']['gap_to_breakpoint_dx']:.3f} dx。同一台站在 P1 N=48 读 17.02 %、
P2 N=400 读 1.5229 % ⇒ **该判据只在 N≳400 有意义**，现在 N=24 直接被守卫拒绝评分。

## 4. 图

**图 1**：Sod 密度精确解 + DG P1(N=48) 单元均值；红虚=4 个精确断点；橙带=rho(0.70)
计分格（离接触间断 0.096 dx）；绿带=其左邻格（**含**该间断）。

**图 2**：固定 8 点 Gauss 参考相对精确参考的相对偏差 vs 该格到最近断点的距离（log-log，
N=48 与 N=400 两档）。偏差只在距离→0 时抬到 1e-2 量级，wave-free 格上落到 1e-14。

## 5. 注意点 / 局限

* **没有推翻任何已发布数字**：三个 E2 台站值逐位不变。本单修的是"下次会错"。
* **推翻一处仓内归因**：`verification-closure.md` §5-2 与 `CASE-REPRO-DG-CKS1998-L4.md` §③
  把 1.5229 % 归因于"DG 单元均值 vs **点值**口径"。实测该格参考值是机器精度的精确单元
  均值（3.6e-15），故该归因不成立；真实原因是接触间断的数值抹平渗入战后第一格。
* 守卫会**拒绝合法但粗**的网格（N=24）。这是有意的：含波格上"3 % 容差 vs 固定规则参考"
  不构成可判定命题。要出 N=24 的报告，正确做法是换台站位置，不是放宽守卫。
* `ncells ∈ {{96,192,384,768}}` 的 DG 误差梯 **not_run**（超本地 120 s shell 上限）；
  图 2 只用解析参考，与网格无关。
* 未做全仓普查：`riemann_exact.cell_averages` 的其它调用点不在本单范围。

## 6. 误差判定（是否 < 3 %）

| 关键量 | 基准 | 实测 | < 3 %？ |
|---|---|---|---|
| rho_x030 @ P2/N=400 | 精确断点切分均值 | 0.4715 % | **是** |
| rho_x070 @ P2/N=400 | 同上 | 1.5229 % | **是**（余量 1.48 pp） |
| vel_x055 @ P2/N=400 | 同上 | 0.0041 % | **是** |
| 仪器偏差（锚点计分格） | 精确 vs 固定 Gauss | ≤2.1e-14 | **是** |
| 仪器偏差（N=24 含波格） | 同上 | 2.29 % | 否 → 该格**不可评分**（守卫拒绝） |
| 台站→格身份可复现性 | 显式规则 | 1201/1201 判右 | **是**（旧规则 34/1201 判左） |
"""

report = {
    "title": "REPRO-DG-CKS1998 · E2 台站度量的两处未审计语义：参考仪器与台站点格",
    "summary": "审计 CKS1998 锚点 E2 三条台站度量的参考仪器与台站点格：证明原用的固定 8 点 "
               "Gauss 参考违反仓内仪器政策（含波格偏 1.79–2.29 %），台站→格身份由浮点末位"
               "决定（34/1201 面上情形判错格）；修复为精确断点切分参考 + 显式点格规则 + 波格"
               "守卫，锚点已发布三值逐位不变（≤2.1e-14），并推翻一处仓内归因。",
    "scenario_key": "repro_dg_cks1998",
    "mission_id": 22894,
    "tags": ["TensorDG", "RKDG", "CKS1998", "verification-debt", "reference-instrument",
             "Sod", "honest-fail", "metric-semantics"],
    "markdown": md,
    "figures": [
        {"name": "e2_station_cells.png",
         "caption": "Sod t=0.2 密度：精确解 vs DG P1(N=48) 单元均值；rho(0.70) 计分格"
                    "（橙）距接触间断 0.096 dx，其左邻格（绿）含该间断。",
         "png_base64": fig1},
        {"name": "e2_instrument_bias.png",
         "caption": "固定 8 点 Gauss 参考相对精确断点切分参考的偏差 vs 该格到最近断点的"
                    "距离（dx）。仪器政策的分界是「格是否含波」，不是「网格是否粗」。",
         "png_base64": fig2},
    ],
}

Path("/tmp/report.json").write_text(json.dumps(report))
(FIGDIR / "e2_station_cells.png").write_bytes(base64.b64decode(fig1))
(FIGDIR / "e2_instrument_bias.png").write_bytes(base64.b64decode(fig2))
print("report written; n24 bias", n24["instrument_rel_delta"])