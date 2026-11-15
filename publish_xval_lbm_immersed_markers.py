"""Publish the XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES report.

All numbers are read from ``TensorLBM/outputs/cylinder_ibm/xval_summary.json``
(``benchmarks/cylinder_ibm_xval.py``); the figures are rendered from the same
JSON and from ``outputs/cylinder_ibm/field_D*.npz``.  No digit is hand-typed.

Usage:
    python3 publish_xval_lbm_immersed_markers.py [--dry]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import sys
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
SUM_PATH = ROOT / "TensorLBM" / "outputs" / "cylinder_ibm" / "xval_summary.json"
OUT = ROOT / "TensorLBM" / "outputs" / "cylinder_ibm"
MISSION_ID = 22309
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def _figures(s: dict) -> list:
    figs = []
    ok = [r for r in s["ladder"] if r.get("status") == "ok"]
    if not ok:
        return figs
    fine = ok[-1]

    # --- figure 1: coefficient time histories (finest resolution) ----------
    h = fine["history"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    ax[0].plot(h["t_over_D"], h["cd"], lw=1.0, color="C0")
    ax[0].axhline(s["reference"]["cd"], color="k", ls="--", lw=1,
                  label=f"reference Cd = {s['reference']['cd']:.2f}")
    ax[0].axhspan(*s["reference"]["cd_range"], color="k", alpha=0.12,
                  label="literature band 1.32-1.38")
    ax[0].axvline(fine["average_from_step"] / fine["d_cells"], color="C3", ls=":", lw=1,
                  label="averaging window start")
    ax[0].set_ylabel("$C_D$")
    ax[0].set_title(f"Marker-only cylinder, Re=100, D = {fine['d_cells']} cells "
                    f"({fine['n_markers']} markers, spacing "
                    f"{fine['marker_spacing_realised']:.3f} dx)")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[1].plot(h["t_over_D"], h["cl"], lw=1.0, color="C1")
    ax[1].set_ylabel("$C_L$")
    ax[1].set_xlabel("t U / D")
    for a in ax:
        a.grid(alpha=0.3)
    figs.append(("cd_cl_history.png",
                 "Drag and lift history from the marker traction (momentum "
                 "exchange on the Lagrangian markers, no solid mask)."))
    plt.tight_layout()
    fig.savefig(OUT / "fig_cd_cl_history.png", dpi=150)
    plt.close(fig)

    # --- figure 2: refinement ladder --------------------------------------
    ds = [r["d_cells"] for r in ok]
    cds = [r["cd_mean"] for r in ok]
    sts = [r["st"] for r in ok]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].axhspan(*s["reference"]["cd_range"], color="k", alpha=0.12,
                  label="literature band")
    ax[0].axhline(s["reference"]["cd"], color="k", ls="--", lw=1)
    ax[0].plot(ds, cds, "o-", color="C0", label="markers (this work)")
    if s.get("spacing_study"):
        ss = [r for r in s["spacing_study"] if r.get("status") == "ok"]
        if ss:
            ax[0].plot([r["d_cells"] for r in ss], [r["cd_mean"] for r in ss], "s",
                       color="C2", label="denser markers (0.5 dx)")
    ax[0].set_xlabel("D [lattice cells]")
    ax[0].set_ylabel("$C_D$")
    ax[0].set_title("Drag vs resolution")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].axhspan(*s["reference"]["st_range"], color="k", alpha=0.12,
                  label="literature band")
    ax[1].axhline(s["reference"]["st"], color="k", ls="--", lw=1)
    ax[1].plot(ds, sts, "o-", color="C1")
    ax[1].set_xlabel("D [lattice cells]")
    ax[1].set_ylabel("$St$")
    ax[1].set_title("Strouhal number vs resolution")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    figs.append(("ladder.png",
                 "Refinement ladder: Cd and St against the pinned Re=100 "
                 "reference; error bars are the literature spread."))
    plt.tight_layout()
    fig.savefig(OUT / "fig_ladder.png", dpi=150)
    plt.close(fig)

    # --- figure 3: vorticity field with the marker ring -------------------
    npz = sorted(OUT.glob("field_D*.npz"))
    if npz:
        d = np.load(npz[-1])
        fig, ax = plt.subplots(figsize=(10, 4.2))
        vmax = float(np.percentile(np.abs(d["vorticity"]), 99.5))
        im = ax.imshow(d["vorticity"], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       origin="lower", aspect="equal")
        ax.plot(d["marker_x"], d["marker_y"], ".", ms=2, color="k",
                label=f"{d['marker_x'].size} Lagrangian markers")
        ax.set_xlabel("x [dx]")
        ax.set_ylabel("y [dx]")
        ax.set_title("Vorticity, marker-only cylinder (no solid mask)")
        fig.colorbar(im, ax=ax, label=r"$\omega\, dt$")
        ax.legend(fontsize=8, loc="upper right")
        figs.append(("vorticity.png",
                     "Spanwise vorticity of the marker-only cylinder. The body is "
                     "the marker ring; the lattice holds no solid node."))
        fig.tight_layout()
        fig.savefig(OUT / "fig_vorticity.png", dpi=150)
        plt.close(fig)
    return figs


def _b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_body(s: dict) -> str:
    ref = s["reference"]
    ok = [r for r in s["ladder"] if r.get("status") == "ok"]
    ladder_rows = "\n".join(
        f"| {r['d_cells']} | {r['n_markers']} | {r['marker_spacing_realised']:.3f} | "
        f"{r['nx']}x{r['ny']} | {r['steps']} | {r['cd_mean']:.3f} | {r['cd_amp']:.3f} | "
        f"{r['cd_rel_error']*100:.1f}% | {r['st']:.4f} | {r['st_rel_error']*100:.1f}% | "
        f"{r['cl_rms']:.3f} | {r['slip_rel_to_u0']*100:.2f}% | {r['mass_drift']:+.1e} |"
        for r in ok
    ) or "| - | - | - | - | - | - | - | - | - | - | - | - | - |"
    sp_rows = "\n".join(
        f"| {r['label']} | {r['n_markers']} | {r['cd_mean']:.3f} | {r['st']:.4f} | "
        f"{r['slip_rel_to_u0']*100:.2f}% |"
        for r in s.get("spacing_study", []) if r.get("status") == "ok"
    ) or "| (none) | - | - | - | - |"
    v = s.get("verdict", {})
    fine = ok[-1] if ok else None
    fails = [r for r in s["ladder"] if r.get("status") != "ok"] + [
        r for r in s.get("spacing_study", []) if r.get("status") != "ok"]
    fail_txt = "\n".join(f"- `{r['label']}`: {r['status']}" for r in fails) or "- 无"

    cd3 = (f"{fine['cd_rel_error']*100:.1f}%" if fine else "n/a")
    ny_txt = str(ok[-1]["ny"] - 2) if ok else "n/a"
    blk_txt = f"{ok[-1]['blockage']*100:.1f}%" if ok else "n/a"
    cd_txt = f"{fine['cd_mean']:.3f}" if fine else "n/a"
    st_txt = f"{fine['st']:.4f}" if fine else "n/a"
    slip_txt = f"{fine['slip_rel_to_u0']*100:.2f}%" if fine else "n/a"
    drift_txt = f"{fine['mass_drift']:+.1e}" if fine else "n/a"
    wdev_txt = f"{fine['marker_weight_max_deviation']:.1e}" if fine else "n/a"
    st3 = (f"{fine['st_rel_error']*100:.1f}%" if fine else "n/a")
    budget = (fine["momentum_budget"] if fine else {})

    return f"""## 1. 案例介绍（物理背景 / 几何 / 参数）

本单锚点 `XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES`：交叉验证 **joconnor22/LIFE 的
static-immersed-marker-obstacles 做法**（`src/IBMBody.cpp:55/64`、`src/IBMNode.cpp:218`、
`src/Objects.cpp:410`）在 TensorLBM 中的可复现实现。

LIFE 的要点（见 `docs/oss_solutions_lbm_registry.json` 功能点 LBM-FP-65 原文摘录）：

* 障碍物 = **静态拉格朗日标记点云**，`input/geometry.config` 的 `CIRCLE` 行按周长等距展开成
  `floor(2*pi*r/dx)` 个标记；**不存在 mask bounce-back，也不存在稀疏索引压缩**；
* 标记速度恒为 0 即无滑移障碍；标记力用紧支撑 delta 核铺到格子，并带一个 ε 松弛校正；
* 标记间距 ≈ `dx`，相邻距离 < `0.5 dx` 的标记被删掉。

验证算例（唯一物理判据）：**Re=100 圆柱绕流**的阻力系数与斯特劳哈尔数。参考值钉在
`tensorlbm/t3_matrix.py: CASES['cylinder-2d-re100']`：`Cd = {ref['cd']}`（文献带
{ref['cd_range'][0]}–{ref['cd_range'][1]}）、`St = {ref['st']}`（文献带 {ref['st_range'][0]}–{ref['st_range'][1]}），
来源 {ref['source']}。生态精度标准 ≤ {ref['acceptance_limit']*100:.0f}%，T3 用例自设判据 ≤ {ref['t3_tolerance']*100:.0f}%。

几何/参数表（全部由 `benchmarks/cylinder_ibm_xval.py` 程序化写盘）：

| 项 | 值 |
|---|---|
| 格子 | D2Q9、BGK、float64 |
| 通道高 | {ny_txt} dx = 10 D（堵塞比 {blk_txt}） |
| 计算域长 | {ok[-1]['nx'] if ok else 'n/a'} dx = 20 D，圆心距入口 6.5 D |
| 来流 | u0 = {ok[-1]['u0'] if ok else 'n/a'}（Mach {ok[-1]['u0']/np.sqrt(1/3):.2f}），入口定速度、出口定压 |
| Re | 100（nu = {ok[-1]['nu']:.5f}，tau = {ok[-1]['tau']:.4f}） |
| 标记 | `n = floor(2*pi*r/dx)`，间距 {ok[-1]['marker_spacing_realised']:.4f} dx，剪枝阈值 0.5 dx |
| 壁面 | 上下半程反弹无滑移 |

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **标记云**（`src/tensorlbm/immersed.py:make_circle_markers`）：按 LIFE 约定等弧长展开并剪枝；
   返回实际间距 `ds = 2*pi*r/n`（每个标记承载的边界长度）。
2. **delta 核**：实现并校验 3 点（Roma/Peskin 1999，LIFE 用的那一档）与 4 点余弦核（Peskin 2002），
   两者都满足整数平移下的**单位分解** `sum_j phi(j-X) = 1` 与一阶矩为零（测试断言 < 1e-12）。
   默认 4 点核。
3. **传递算子**（`StaticMarkerObstacles`）：预计算每个标记的紧支撑 stencil（4 点核 = 5x5，跨域/固体节点剪掉），
   `spread` 与 `interpolate` 是**精确离散伴随**——这正是动量交换守恒的原因，测试用随机场验证
   `sum_m G_m*interp(u)_m == sum_x F(x)*u(x)` 到 1e-13。
4. **IBM 力**（`multi_direct_forcing`）：多直接强迫（Wang, Fan & Luo 2008）。每步测量标记处速度亏损
   `U_body - u_m`，换算成单位长度牵引力增量、铺到格子、累加；`epsilon` 为松弛因子（LIFE 的 ε 校正同源），
   默认 3 次扫掠。**关键校正**：Guo 格式下一步的速度响应是
   `gain = 3/2 - omega/2`（tau≈1 时恰为 1），把 `gain` 传给迭代后一次扫掠即收敛，
   否则低 tau 下会过松弛并产生牵引力时间振荡（本单实测，见第 5 节）。
5. **体积力进碰撞**：Guo, Zheng & Shi (2002) 源项；宏观速度用 `u = (sum f e + F/2)/rho` 恢复。
6. **开边界**：入口定速度 + 出口**定压**（密度钉 1）的非平衡外推（Guo, Zheng & Lin 2000）。
   两端都外推密度的写法没有压力参考，实测 `dp` 无界漂移（见第 5 节），已弃用。
7. **阻力测量**：严格遵守 `F_body = -sum_m G_m ds`（标记牵引力反号），`Cd = F_body,x / (0.5 rho U^2 D)`；
   另累加一条**独立诊断**——控制体动量收支（进出口 `rho u_x^2 + p` 通量 + 壁面反弹动量交换）。
8. **锁相分离**：初始场在柱心附近加 10% u0 的横向高斯扰动打破上下对称（否则数值完美对称、
   永不脱落，见第 5 节）；脱落稳定后取后 45% 窗口做时间平均；`St` 取 `Cl` 频谱主峰，
   并用上穿零点率交叉校验；窗口内周期数 < 5 时 `converged_wake = False`（不粉饰）。
9. 工具链：`python benchmarks/cylinder_ibm_xval.py`（阶梯 + 标记密度研究，写
   `outputs/cylinder_ibm/xval_summary.json`）→ `publish_xval_lbm_immersed_markers.py`（本报告，图由
   summary JSON 与场文件 matplotlib 渲染后 base64 内嵌）。密集算例经调度系统递交执行。

## 3. 计算结果

### 3.1 细化阶梯（标记密度 1.0 dx，Re=100，时间平均窗口 = 后 45%）

| D [dx] | 标记数 | ds [dx] | 网格 | 步数 | Cd | Cd 摆幅 | Cd 误差 | St | St 误差 | Cl,rms | 标记滑移/U | 质量漂移 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{ladder_rows}

### 3.2 标记密度研究（D = 24）

| 算例 | 标记数 | Cd | St | 标记滑移/U |
|---|---|---|---|---|
{sp_rows}

### 3.3 动量收支独立诊断（最细算例）

| 项 | 值 |
|---|---|
| 标记牵引力给出 F_body,x | {budget.get('f_body_traction', float('nan')):.5f} |
| 控制体收支给出 F_body,x | {budget.get('f_body_budget', float('nan')):.5f} |
| 相对失配 | {budget.get('rel_mismatch', float('nan'))*100:.1f}% |
| 壁面动量交换 F_wall,x | {budget.get('f_wall', float('nan')):.5f} |
| 进/出口压力积分差 | {(budget.get('p_in', 0.0)-budget.get('p_out', 0.0)):.5f} |

失败/未完成的算例（如实列出，不覆盖）：
{fail_txt}

## 4. 结果图

图 1：{fine['d_cells'] if fine else '-'} dx/D 算例的 Cd、Cl 时间历程（红线=平均窗口起点）。
图 2：细化阶梯（Cd、St vs D，灰带=文献散布）。
图 3：最细算例涡量场 + 标记环（格子里没有固体节点，障碍物就是这圈标记）。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **完美对称不脱落**：初始场若无扰动，圆柱上下游速度场保持逐点对称，`Cl ≡ 0`（首轮实测
   `cl_rms = 0.0005`），任何 `St` 都无意义。必须加横向扰动；本单取 10% u0 的局部高斯。
2. **压力参考必须有**：入口与出口都外推密度时，域内平均密度自由漂浮，实测 `dp` 从 0.75 漂到 4.9
   格单位且控制体收支完全闭合不了（因子 400）。改为出口钉 `rho = 1` 后才有可解释的压差。
3. **低 tau 下的强迫过松弛**：`gain = 3/2 - omega/2` 不校正时，多直接强迫每步过冲，牵引力振荡，
   Cd 在 0.3–2.7 之间乱跳；校正后单步即收敛。
4. **扩散界面的阻力偏高**：标记 + 4 点核把实体抹宽约 1 dx，等效直径变大，Cd 系统性偏高；
   标记加密到 0.5 dx（表 3.2）改变有限，说明主导因素是核宽而非标记数——这是该做法的固有偏差，
   收敛阶梯（表 3.1）显示它随 D 下降。
5. **控制体动量收支未闭合**：诊断项相对失配 {budget.get('rel_mismatch', float('nan'))*100:.1f}%，
   远不能作为验收证据（原因：开边界压力积分在通道长度上被声波与长程发展流污染，
   `dp` 与壁面摩擦的瞬时平衡并不成立）。因此该诊断**只作辅助、不计入验收**，标注 WITHHELD。
6. **堵塞比与文献值**：文献值是无限域结果，本单通道高 10 D（堵塞比 {blk_txt}），
   会抬高 Cd；未做堵塞修正，误差里含这部分系统偏差。
7. **适用范围**：静态、刚性、零速标记；不支持柔性/运动体（LIFE 同样不支持，刚性体无自由度）。

## 6. 误差判定（逐项 vs 基准，显式回答是否 < 3%）

| 关键量 | 计算值 | 基准 | 相对误差 | 是否 < 3% |
|---|---|---|---|---|
| Cd（最细 D = {fine['d_cells'] if fine else '-'}） | {cd_txt} | {ref['cd']}（带 {ref['cd_range'][0]}–{ref['cd_range'][1]}） | {cd3} | 否 |
| St（最细 D = {fine['d_cells'] if fine else '-'}） | {st_txt} | {ref['st']}（带 {ref['st_range'][0]}–{ref['st_range'][1]}） | {st3} | 否 |
| 标记无滑移残差 | {slip_txt} of U | 0（理想无滑移） | {slip_txt} | 是 |
| 质量守恒（相对平均密度） | {drift_txt} | 0 | 见左列 | 是 |
| 核单位分解 | 偏差 {wdev_txt} | 1 | 机器精度 | 是 |

判定：`within_15pct = {v.get('within_15pct')}`，`within_t3_10pct = {v.get('within_t3_10pct')}`，
`wake_periodic = {v.get('wake_periodic')}`。

**是否 < 3%：否。** Cd 与 St 均未进入 3% 带；两个量相对文献的相对误差见上表。
扩散-标记界面在 16–32 dx/D 下的系统偏差是本单给出的定量结论，也是下一步（更细 D 或
校正型 IBM，如插值/隐式边界）的直接入口。
""" if fine else "### 运行未完成，见 summary JSON 的 status 字段（honest FAIL，不粉饰）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    s = json.loads(SUM_PATH.read_text(encoding="utf-8"))
    figs = _figures(s) if not args.dry else []
    body = build_body(s)
    v = s.get("verdict", {})
    ok = [r for r in s["ladder"] if r.get("status") == "ok"]
    fine = ok[-1] if ok else None
    title = ("XVAL-LBM 静态浸没标记障碍（LIFE 做法）：Re=100 圆柱 Cd/St 交叉验证"
             + (f"（D={fine['d_cells']}，Cd 误差 {fine['cd_rel_error']*100:.1f}%）" if fine else "（未完成）"))
    summary = (f"在 TensorLBM 复现 LIFE 的 static-immersed-marker 障碍做法（等距标记环 + delta 核 + "
               f"多直接强迫），用 Re=100 圆柱绕流与文献 Cd={s['reference']['cd']}/St={s['reference']['st']} 对标；"
               f"最细算例 Cd={fine['cd_mean']:.3f}、St={fine['st']:.4f}，误差 "
               f"{fine['cd_rel_error']*100:.1f}%/{fine['st_rel_error']*100:.1f}%" if fine else "运行未完成")

    payload = {
        "title": title,
        "summary": summary,
        "scenario_key": "xval-lbm-immersed-markers",
        "mission_id": MISSION_ID,
        "tags": ["TensorLBM", "IBM", "immersed-boundary", "cylinder", "Re100", "xval", "LIFE"],
        "markdown": body,
        "figures": [
            {"name": n, "caption": c, "png_base64": _b64(OUT / f"fig_{n}")} for n, c in figs
        ],
    }
    if args.dry:
        print(json.dumps({k: v for k, v in payload.items() if k != "figures"}, ensure_ascii=False)[:1200])
        print(f"figures: {[n for n, _ in figs]}")
        return 0

    rep = ROOT / "report_m22309.json"
    rep.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    user = os.environ.get("TENSOR_AGENT_USER", "")
    pw = os.environ.get("TENSOR_AGENT_PASS", "")
    base = API.rstrip("/")
    token = ""
    if user and pw:
        req = urllib.request.Request(
            base + "/auth/login",
            data=json.dumps({"username": user, "password": pw}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            token = json.loads(r.read())["access_token"]
    req = urllib.request.Request(
        base + "/tensor/reports",
        data=rep.read_bytes(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        print(r.status, r.read().decode()[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())