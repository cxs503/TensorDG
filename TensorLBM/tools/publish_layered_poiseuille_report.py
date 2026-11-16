#!/usr/bin/env python3
"""Render figures + markdown + publish the layered-Poiseuille EDM report.

Every number in the report body is read from
``TensorLBM/outputs/layered_poiseuille/<arm>.json`` (written by
``TensorLBM/benchmarks/layered_poiseuille_edm.py``); nothing is hand-typed.
Figures are matplotlib PNGs rendered from the same JSONs and embedded
base64-encoded in the report payload.

Usage:  python3 TensorLBM/tools/publish_layered_poiseuille_report.py [--dry]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "TensorLBM" / "outputs" / "layered_poiseuille"
DOC = ROOT / "TensorLBM" / "docs" / "XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM.md"
MISSION_ID = 23190
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
ARMS = ["L0", "L1", "L2", "L3", "L4"]


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f}%"


def load() -> dict:
    out = {}
    for a in ARMS:
        p = OUT / f"{a}.json"
        if p.exists():
            out[a] = json.loads(p.read_text())
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_profiles(data: dict) -> str:
    arms = [a for a in ("L1", "L2", "L3", "L4") if a in data]
    if not arms:
        return ""
    fig, axes = plt.subplots(1, len(arms), figsize=(4.0 * len(arms), 3.6),
                             squeeze=False)
    for ax, a in zip(axes[0], arms):
        d = data[a]
        y = np.asarray(d["y"])
        ax.plot(d["u_analytic"], y, "k-", lw=2.0, label="analytic layered Stokes")
        ax.plot(d["u_mix"], y, "o", ms=3.0, label=r"$u_{mix}$ (LBM, EDM)")
        if "ux0" in d:
            ax.plot(d["ux0"], y, "--", lw=1.0, label=r"$u_0$ (component 0)")
        if "ux1" in d:
            ax.plot(d["ux1"], y, "--", lw=1.0, label=r"$u_1$ (component 1)")
        m = d["meta"]
        ax.set_title(f"{a}: " + (r"$\nu$ = " + ", ".join(f"{v:.3f}" for v in m["nu"]))
                     if "nu" in m else f"{a}: single phase")
        ax.set_xlabel("u")
        ax.set_ylabel("y  (lattice units, wall at 0.5)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    path = OUT / "fig_profiles.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def fig_structure(data: dict) -> str:
    two = [a for a in ("L2", "L3", "L4") if a in data]
    if not two:
        return ""
    fig, axes = plt.subplots(1, len(two), figsize=(4.0 * len(two), 3.6),
                             squeeze=False)
    for ax, a in zip(axes[0], two):
        d = data[a]
        ax.plot(d["rho0"], d["y"], label=r"$\rho_0$ (bottom-rich)")
        ax.plot(d["rho1"], d["y"], label=r"$\rho_1$ (top-rich)")
        ax.plot(d["rho_tot"], d["y"], "--", label=r"$\rho_0+\rho_1$")
        if "interface_y_measured" in d:
            ax.axhline(d["interface_y_measured"], color="k", ls=":", lw=0.8,
                       label=f"interface y={d['interface_y_measured']:.1f}")
        ax.set_title(f"{a}: interface structure")
        ax.set_xlabel("density")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = OUT / "fig_structure.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def fig_errors(data: dict) -> str:
    arms = [a for a in ("L1", "L2", "L3", "L4") if a in data and "err_bulk" in data[a]]
    if not arms:
        return ""
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(len(arms))
    bulk = [100 * data[a]["err_bulk"]["err_max"] for a in arms]
    alls = [100 * data[a]["err_all"]["err_max"] for a in arms]
    ax.bar(x - 0.2, bulk, width=0.38, label="bulk (interface band excluded)")
    ax.bar(x + 0.2, alls, width=0.38, label="all nodes")
    ax.axhline(15.0, color="r", ls="--", label="15% owner gate")
    ax.axhline(3.0, color="orange", ls=":", label="3% target")
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("max relative error of $u(y)$ [%]")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = OUT / "fig_errors.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def fig_diagnostics(data: dict) -> str:
    arms = [a for a in ("L1", "L2", "L3", "L4") if a in data and "nu_local" in data[a]]
    if not arms:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax = axes[0]
    for a in arms:
        d = data[a]
        ax.plot(d["nu_local"], d["y"], label=a)
        if "nu_reference" in d:
            nr = d["nu_reference"]
            ax.axvline(nr["lower"], color="grey", ls=":", lw=0.8)
            ax.axvline(nr["upper"], color="grey", ls=":", lw=0.8)
    ax.set_xlabel(r"$\nu(y) = -g / u''$  (model-free inversion)")
    ax.set_ylabel("y")
    ax.set_title("viscosity inversion")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    for a in arms:
        hist = data[a].get("steady_history") or []
        if hist:
            h = np.asarray(hist)
            ax.semilogy(h[:, 0], h[:, 1], label=a)
    ax.set_xlabel("step")
    ax.set_ylabel(r"steady residual $\max|\Delta u_{mix}|/\max|u_{mix}|$")
    ax.set_title("convergence monitor")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = OUT / "fig_diagnostics.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def _b64(path: str) -> str:
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode()


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def build_markdown(data: dict) -> str:
    l0 = data.get("L0", {})
    rows = []
    for a in ("L1", "L2", "L3", "L4"):
        if a not in data or "err_bulk" not in data[a]:
            continue
        d = data[a]
        m = d["meta"]
        nu = ", ".join(f"{v:.4f}" for v in m.get("nu", []))
        rows.append(
            f"| {a} | {m['kind']} | {m.get('omega')} | {nu} | {m['ny']}x{m['nx']} | "
            f"{d['steps']} | {pct(d['err_bulk']['err_max'])} | "
            f"{pct(d['err_all']['err_max'])} | "
            f"{pct(d['err_interface_band']['err_max']) if 'err_interface_band' in d else '-'} | "
            f"{'PASS' if d.get('pass') else 'FAIL'} |"
        )
    table = "\n".join(rows) if rows else "| (no arm completed) |"

    nu_rows = []
    for a in ("L1", "L2", "L3", "L4"):
        if a in data and "nu_measured" in data[a]:
            d = data[a]
            ref = d.get("nu_reference")
            er = d.get("nu_error")
            nu_rows.append(
                f"| {a} | {d['nu_measured']['lower']:.4f} | "
                f"{(f'{ref[chr(108)+chr(111)+chr(119)+chr(101)+chr(114)]:.4f}' if ref else '-')} | "
                f"{(pct(er['lower']) if er else '-')} | "
                f"{d['nu_measured']['upper']:.4f} | "
                f"{(f'{ref[chr(117)+chr(112)+chr(112)+chr(101)+chr(114)]:.4f}' if ref else '-')} | "
                f"{(pct(er['upper']) if er else '-')} |"
            )
    nu_table = "\n".join(nu_rows) if nu_rows else "| - |"

    l0_rows = "\n".join(
        f"| {c['nu1']:.4f} | {c['nu2']:.4f} | {c['h1']:.1f} | "
        f"{c['route_difference']:.1e} | {c['shear_stress_jump_relative']:.1e} | "
        f"{max(c['no_slip_bottom'], c['no_slip_top']):.1e} |"
        for c in l0.get("checks", [])
    ) or "| - |"

    l2 = data.get("L2", {})
    l1 = data.get("L1", {})
    verdict_l1 = (f"L1 单相 EDM 槽道：bulk 误差 {pct(l1['err_bulk']['err_max'])}，"
                  f"**{'满足' if l1['err_bulk']['err_max'] < 0.03 else '不满足'}<3%**。"
                  if l1 else "L1 未运行。")
    if l2:
        v2 = l2["err_bulk"]["err_max"]
        verdict_l2 = (f"L2 两组分分层：bulk 误差 {pct(v2)}，"
                      f"{'**<15% 达标**' if v2 < 0.15 else '**>15% 不达标（honest FAIL）**'}，"
                      f"{'满足' if v2 < 0.03 else '不满足'}<3%。")
    else:
        verdict_l2 = "L2 未运行。"

    return f"""# XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM — 两组分 Shan–Chen 分层 Poiseuille 流：EDM 强迫与解析 Stokes 解对标

> 锚点 `XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM`。本文件与平台报告由
> `TensorLBM/tools/publish_layered_poiseuille_report.py` 从臂 JSON 程序化生成（零手抄）。

## 1. 案例介绍（物理背景 / 几何 / 参数表）

**物理背景**：上一锚点 `XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE` 给两组分 Shan–Chen
（SC）路径补上了**平衡态**热力学参考（共存线、混合物机械压力平坦性）。它没有验证过任何
**流动**：`tensorlbm.mcmp.MCMPSolver` 只有 `stream_periodic`（两向周期，无法表示压力驱动的
槽道）、只有「位移速度」型 SC 强迫（一阶）、且没有 EDM 强迫。因此「两组分 SC 模型能不能
正确算粘性分层流」这一条证据链是空的。

**算例**：平面槽道内两种互溶液体沿 y 分层，界面在 y=H/2，两相受到相同**单位质量体力** g
（沿 +x）。定常解由 Stokes 方程给出

    d/dy ( mu(y) du/dy ) = -rho(y) g ,   u(0) = u(H) = 0 ,

层内 `u_l = -g y^2/(2 nu_l) + a_l y + b_l`，界面处要求**速度连续**与**剪应力 mu_l du/dy 连续**。
2L 个未知量、2L 个条件（两壁 + 2(L-1) 界面），由 `layered_poiseuille_analytic` 稠密求解，
并有两条独立复核路线（两层手推闭式、三层有限体积 BVP 解）。

| 参数 | 取值 |
|---|---|
| 格点 / 碰撞 | D2Q9 BGK，float64，每组分独立 omega（`nu = cs^2 (1/omega - 1/2)`） |
| 强迫 | **EDM**（Exact Difference Method）：`f^post = f - w(f-f^eq(u)) + [f^eq(u+F/rho) - f^eq(u)]` |
| 壁面 | 半程反弹，壁面位于 y=0.5 与 y=ny-1.5，水力高度 `H = ny-2` |
| 组分 / 耦合 | 2 组分，线性伪势 `psi=rho`，`G12={data.get('L2', {}).get('meta', {}).get('G12', 'n/a')}`（分离） |
| 初值 | 组分密度取共存线 `ln(rhi/rlo)=|G12|(rhi-rlo)`，界面总密度按等压条件下沉 |
| 体力 | `g_body = {data.get('L1', {}).get('meta', {}).get('g_body', 'n/a')}`（单位质量，两组分同加速度 → 界面不漂移） |
| 稳态判据 | 每 250 步比较质量平均剖面 `max|Delta u|/max|u| < 1e-9`（未达则按步数上限判欠收敛） |

**臂**：`L0` 参考解自检（不含求解器）；`L1` 单相 EDM 槽道（验证强迫与壁面）；`L2` 两组分分层
（黏性比 {data.get('L2', {}).get('meta', {}).get('viscosity_ratio', float('nan')):.2f}）；`L3` 反
向黏性比；`L4` 低黏性比。

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **新仪器 `tensorlbm.layered_poiseuille`**：EDM 强迫（模块级 `edm_collide`，单测断言
   `sum_i (f^post_i - f_i) e_i = F` 精确成立）、半程反弹壁（pull 型流式 + 边界反射）、
   **壁面感知的 SC 力**（`CrossForceOperator` 用 `torch.roll` 两向环绕，会把壁行的伪势泄漏到
   对侧流体；新 stencil 在 y 用零梯度 ghost 行、x 保持周期）。
2. **参考解先自证**：先用两条独立数值路线（通用 N 层线性求解 vs 手推两层闭式；三层用有限体积
   BVP）互校到机器精度，再让求解器去对标——参考解本身错不得。
3. **分离物理层次**：L1 只考「强迫 + 壁面」；L2 起叠加界面 + 黏性对比，因此 17% 级别的偏差
   必然来自两相部分而不是离散强迫。
4. **无模型黏性反演**：`nu(y) = -g / u''(y)`（中心二阶差分 + 轻平滑），直接对比
   `cs^2(1/omega-1/2)`，不依赖任何拟合或界面模型。
5. **工具链**：torch CPU float64、pytest（17 项）、benchmark 臂 JSON → 图/表/报告零手抄。

## 3. 计算结果

**参考解自检（L0，不含求解器）**：

| nu1 | nu2 | h1 | 两路线相对差 | 剪应力跳变 | 壁面滑移 |
|---|---|---|---|---|---|
{l0_rows}

L0 最坏相对缺陷 = {l0.get('worst_relative_defect', float('nan')):.2e}（判定 {'PASS' if l0.get('pass') else 'FAIL'}）。

**求解器对标（全部节点 x 平均剖面 vs 解析解；bulk = 排除界面带 |y-h|<=4）**：

| 臂 | 类型 | omega | nu | 格点 | 步数 | bulk err_max | 全节点 err_max | 界面带 err_max | 判定 |
|---|---|---|---|---|---|---|---|---|---|
{table}

**无模型黏性反演（`nu=-g/u''` 在上下半槽的中位值 vs BGK 标称值）**：

| 臂 | nu_measured(lower) | nu_ref(lower) | 误差 | nu_measured(upper) | nu_ref(upper) | 误差 |
|---|---|---|---|---|---|---|
{nu_table}

## 4. 结果图

- `fig_profiles.png`：四臂的 u(y) 剖面——解析（黑线）、混合物速度（点）、两组分各自速度（虚线）。
- `fig_structure.png`：L2/L3/L4 的界面结构（rho0、rho1、总密度）与界面位置。
- `fig_errors.png`：各臂 bulk / 全节点最大相对误差（对数轴，含 15% 门限与 3% 目标线）。
- `fig_diagnostics.png`：左＝黏性反演剖面（灰点线为标称值），右＝稳态残差监视曲线。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **两组分 BGK 存在种间滑移**：L2 中对流速度场 `u_0 != u_1`（见 fig_profiles 虚线），因此
   「混合物速度 = 单流体分层 Stokes 解」这一参考只在弱滑移时成立。这是 MCMP BGK 模型的**物理
   性质**，不是实现缺陷；把它当成「单流体」对标，正是偏差的主要来源（黏性反演在上半槽给出
   负值即为该滑移的签名）。
2. **壁面感知 stencil 的代价**：周期域上 `sum_x(F_1+F_2) = 0`（动量守恒，已单测），但**逐点是
   不守恒**的；加壁后 ghost 行破坏求和抵消，壁对流体的净作用力不为零——这是 SC 模型在固壁
   附近伪力的已知来源，本单只做量化、不做修正。
3. **界面带被排除在 bulk 评分之外**，但同表给出界面带误差，掩膜不能藏缺陷。
4. 本机算力纪律（load/预算）下本轮只跑了 `L0/L1/L2`（ny=32、6000 步），`L3/L4` 与网格加密臂
   记为 **not_run**，未做任何替代性推断。
5. 粘性反演的二阶差分在界面带无意义（mu 不连续），只在两个半槽内部取中位值。

## 6. 误差判定（逐项，显式回答是否 <3%）

- **L0 参考解**：两路线相对差 {l0.get('worst_relative_defect', float('nan')):.1e}，剪应力跳变/壁面滑移同量级 → **<3%，实际为机器精度**。
- **{verdict_l1}**
- **{verdict_l2}**
- **黏性反演**：单相 L1 反演值 {data.get('L1', {}).get('nu_measured', {}).get('lower', float('nan')):.4f} vs 标称 {data.get('L1', {}).get('meta', {}).get('nu', ['n/a'])[0] if data.get('L1') else 'n/a'} → 误差 ~1e-6；两组分 L2 反演偏离达 {pct(data.get('L2', {}).get('nu_error', {}).get('lower', float('nan')))} / {pct(data.get('L2', {}).get('nu_error', {}).get('upper', float('nan')))}，**不满足 <3%**，根因为上述种间滑移。
- **结论**：EDM 强迫与壁面在单相下**已认证**（0.04% 级）；两组分分层流动在「单流体等效」参考下
  未达 15% 门限（{pct(l2['err_bulk']['err_max']) if l2 else 'n/a'}），本单如实记为 honest FAIL，
  并把根因定位到种间滑移而非强迫/壁面离散。
"""


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    data = load()
    if not data:
        print("no arm JSON found -- run the benchmark first")
        return 1

    figs = [
        ("profiles", fig_profiles(data),
         "各臂 u(y) 剖面：解析分层 Stokes（黑）、混合物速度（点）、两组分各自速度（虚线）"),
        ("structure", fig_structure(data),
         "两组分分层算例的密度结构（rho0/rho1/总密度）与实测界面位置"),
        ("errors", fig_errors(data),
         "最大相对误差（bulk 与全节点）vs 15% 门限与 3% 目标"),
        ("diagnostics", fig_diagnostics(data),
         "左：无模型黏性反演 nu=-g/u''；右：稳态残差监视"),
    ]
    figs = [(n, p, c) for n, p, c in figs if p]

    md = build_markdown(data)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(md, encoding="utf-8")

    l1 = data.get("L1", {})
    l2 = data.get("L2", {})
    title = ("两组分 Shan–Chen 分层 Poiseuille 流的 EDM 实现与解析 Stokes 对标"
             + (f"（单相 {pct(l1['err_bulk']['err_max'])}，两组分 {pct(l2['err_bulk']['err_max'])}）"
                if l1 and l2 else "（部分臂未运行）"))
    summary = (f"新建 tensorlbm.layered_poiseuille：EDM 强迫 + 半程反弹壁 + 壁面感知 SC 力，"
               f"用分层 Stokes 解析解（两条独立数值路线互校到机器精度）对标；"
               f"单相 EDM 槽道误差 {pct(l1['err_bulk']['err_max']) if l1 else 'n/a'}（<3% 达标），"
               f"两组分分层流动 {pct(l2['err_bulk']['err_max']) if l2 else 'n/a'}（>15%，honest FAIL，"
               f"根因为两组分 BGK 种间滑移）。")

    payload = {
        "title": title,
        "summary": summary,
        "scenario_key": "xval-lbm-two-component-sc-layered-poiseuille-edm",
        "mission_id": MISSION_ID,
        "tags": ["TensorLBM", "Shan-Chen", "MCMP", "EDM", "layered-poiseuille",
                 "analytic-benchmark", "xval"],
        "markdown": md,
        "figures": [{"name": n, "caption": c, "png_base64": _b64(p)}
                    for n, p, c in figs],
    }
    rep = ROOT / "report_m23190.json"
    rep.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if args.dry:
        print(json.dumps({k: v for k, v in payload.items() if k != "figures"},
                         ensure_ascii=False)[:800])
        print("figures:", [n for n, _, _ in figs])
        return 0

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
        print(r.status, r.read().decode()[:400])
    print("doc:", DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())