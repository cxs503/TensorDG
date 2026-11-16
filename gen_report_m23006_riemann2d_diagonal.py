"""发布 m23006 报告：CKS1998 二维 Riemann 算例的对角线精确解参考（图从 artifacts JSON 生成）。"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

ART = Path("validation/repro_cks1998/artifacts/riemann2d_diagonal")
FIG = Path("docs/figures")
FIG.mkdir(parents=True, exist_ok=True)
sym = json.loads((ART / "measure_n32_p1_t0.1.json").read_text())
asym = json.loads((ART / "measure_n32_p1_t0.1_asymmetric.json").read_text())


def profile_figure() -> Path:
    xi = np.asarray(sym["profile"]["xi"])
    rho = np.asarray(sym["profile"]["density_dg"])
    ref = np.asarray(sym["profile"]["density_exact"])
    lo, hi = sym["wave_window"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(xi, ref, "k-", lw=2.2, label="exact 1-D Riemann along y=x (Toro)")
    ax.plot(xi, rho, "o", ms=3.2, color="#c0392b", label="RKDG P1 N=32, t=0.1")
    ax.axvspan(lo, hi, color="#3498db", alpha=0.08, label="wave window")
    for st in sym["stations"]:
        ax.axvline(st["xi_request"], color="#7f8c8d", ls=":", lw=0.8)
    ax.set_xlim(-4.0, 4.0)
    ax.set_xlabel(r"similarity coordinate  $\xi = X/t$,  $X=(x+y-1)/\sqrt{2}$")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title("CKS1998 2-D Riemann (Lax-Liu config 3): diagonal vs exact solution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIG / "cks1998_riemann2d_diagonal_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def falsification_figure() -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for doc, style, label in (
        (sym, dict(color="#c0392b", ls="-"), "symmetric data (reduction holds)"),
        (asym, dict(color="#8e44ad", ls="--"), "asymmetric variant (reduction broken)"),
    ):
        xi = np.asarray(doc["profile"]["xi"])
        ax.plot(xi, np.asarray(doc["profile"]["density_dg"]), lw=1.6,
                label=label + " — DG", **style)
        if doc is sym:
            ax.plot(xi, np.asarray(doc["profile"]["density_exact"]), "k:", lw=1.4,
                    label="exact 1-D Riemann (symmetric data)")
    lo, hi = sym["wave_window"]
    ax.axvspan(lo, hi, color="#3498db", alpha=0.08)
    ax.set_xlim(-2.6, 2.4)
    ax.set_xlabel(r"$\xi = X/t$")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title("Falsification: same criterion, broken symmetry premise "
                 f"(wave L1 {100*sym['diag_rho_l1_wave']:.1f}% -> "
                 f"{100*asym['diag_rho_l1_wave']:.1f}%)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIG / "cks1998_riemann2d_diagonal_falsification.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    p1 = profile_figure()
    p2 = falsification_figure()
    ws = sym["wave_structure"]
    st_rows = "\n".join(
        f"| `{s['name']}` | {s['xi_request']:.4f} | {s['density_got']:.4f} | "
        f"{s['density_ref']:.4f} | {100*s['density_rel_err']:.2f} % |"
        for s in sym["stations"])
    markdown = f"""# CKS1998 二维 Riemann 算例：对角对称线上的精确解参考（验证欠账闭合）

锚点 `REPRO-DG-CKS1998`；单号 m23006；仓 TensorDG；代码
`validation/repro_cks1998/riemann2d_diagonal.py`；回归门
`tests/test_riemann2d_diagonal_reference.py`（6 条全绿）。详细版见
`docs/CASE-REPRO-DG-CKS1998-2D-RIEMANN-DIAGONAL.md`。

## ① 案例介绍
Lax & Liu (1998) config 3 四象限二维 Riemann 问题（CKS1998 多维系统的代表算例），γ=1.4，
域 [0,1]²，透射边界，P1/DG-HLLC/SSP-RK3/TVB，N=32，t=0.1，cfl=0.2。该初值关于 `y=x` 对称
（交换 x↔y 且交换 u↔v 后不变），LL=(0.138,1.206,1.206,0.3)，UR=(1.5,0,0,1.5)。

## ② 怎么做的
1. 旋转到 `X=(x+y-1)/√2, Y=(y-x)/√2`：ρ,p,u_X 为 Y 的偶函数、u_Y 为奇函数 ⇒ 在 Y=0 上
   二维 Euler 方程逐项退化为一维方程（奇函数的 ∂_Y 项与偶函数的 ∂_Y 项均为 0）⇒
   **对角线上的二维解精确等于一维 Riemann 问题的解**，左状态 (0.138, u_n=1.70554, 0.3)、
   右状态 (1.5, 0, 1.5)。
2. 在对角单元的 Gauss 点 `(ξ_j, ξ_j)`（严格落在 y=x 上）逐点求 DG 模态多项式值，
   同一点取 `benchmarks.riemann_exact`（Toro 精确 Riemann，独立于 DG）的参考值。
3. 度量：波区相对 L1（窗口 [最左波−0.5, 最右波+0.5]，先定后跑）、光滑区 L1（剔波面 ±2h_ξ）、
   台站相对误差、对称残差。台站由精确解波系结构定义。
4. 否证实验：把 LL 的 v 取反号破坏对称（横向剪切 u_Y=−1.706），用它自己的法向状态作参考
   跑同一判据。

精确解波系：s_shock={ws['left_wave']:.5f}，u*={ws['contact']:.5f}，稀疏扇尾
{ws['right_inner']:.5f}，扇头 {ws['right_outer']:.5f}，p*={ws['star_pressure']:.5f}。

## ③ 计算结果（N=32, P1, t=0.1，程序化取自 artifacts JSON）
| 量 | 数值 |
|---|---:|
| 对称残差 | {sym['symmetry_residual']:.2e} |
| 全线相对 L1 | {100*sym['diag_rho_l1_rel']:.3f} % |
| 波区相对 L1 | {100*sym['diag_rho_l1_wave']:.3f} %（{sym['wave_samples']}/{sym['diagonal_samples']} 点）|
| 光滑区相对 L1 | {100*sym['diag_rho_l1_smooth']:.3f} % |
| 否证：对称残差 | {asym['symmetry_residual']:.2e} |
| 否证：波区相对 L1 | {100*asym['diag_rho_l1_wave']:.2f} %（对称档 {100*sym['diag_rho_l1_wave']:.2f} %，比值 {asym['diag_rho_l1_wave']/sym['diag_rho_l1_wave']:.2f}）|

台站（DG vs 精确解）：
| 台站 | ξ | ρ_DG | ρ_exact | 相对误差 |
|---|---:|---:|---:|---:|
{st_rows}

## ④ 结果图
图 1：对角线 ρ(ξ)（DG vs 精确一维 Riemann 解），阴影为波区 L1 窗口。
图 2：否证实验——同一判据在破坏 y=x 对称的初值上波区误差由
{100*sym['diag_rho_l1_wave']:.1f}% 升到 {100*asym['diag_rho_l1_wave']:.1f}%，且对角线剖面本身移动
max|Δρ|={np.max(np.abs(np.asarray(asym['profile']['density_dg'])-np.asarray(sym['profile']['density_dg']))):.3f}。

## ⑤ 注意点 / 局限
1. 该组状态下右波是**极薄稀疏扇**（ξ∈[{ws['right_inner']:.3f},{ws['right_outer']:.3f}]，宽 0.257），
   N=32/t=0.1 时 h_ξ=√2h/t≈0.44，一个单元跨整个扇 ⇒ `fan_mid` 的 9.3 % 主要是分辨率。
2. 左侧星区平台宽 1.32 只容 ~3 个采样点，`star_right` 的 24.2 % 是波面抹平侵蚀平台。
3. 沿一条线采样**无法**还原二维单元均值（对角 Gauss 点集不是二维求积规则），故口径是
   "点值沿对角线 vs 精确点值"——恰好无口径差。
4. 本单只跑到 N=32（本机 <120 s 限制）；细化档 N=40/64（t=0.2）与 N=88 已递交集群
   `cj4528`/`cj4529`，递交时不回，**本文不写未回的细分数**。
5. 参考的正确性依赖初值关于 y=x 对称 + 边界处理保持该对称；对不对称初值无效（否证实验量化）。

## ⑥ 误差判定（是否 < 3 %）
| 关键量 | 基准 | 误差 | < 3 % |
|---|---|---:|---|
| 对称残差 | 0（精确） | {sym['symmetry_residual']:.2e} | 是（机器精度） |
| 光滑区 ρ | 一维精确 Riemann | {100*sym['diag_rho_l1_smooth']:.3f} % | **是** |
| 波区 ρ | 一维精确 Riemann | {100*sym['diag_rho_l1_wave']:.2f} % | **否** |
| 最差台站 | 一维精确 Riemann | {100*max(s['density_rel_err'] for s in sym['stations']):.2f} % | **否** |

**honest FAIL**：作为"精度门"，该判据在 N=32/t=0.1 上不达标（波区 18.3 %，最差台站 24.2 %），
原因是激波/接触只有 2–3 个单元的厚度。它成立的是**参考正确性**（光滑区 0.021 %、
对称残差 1e−15、否证实验有效：破坏前提后误差升 1.43 倍且剖面移动 0.35），
不是"DG 在二维 Riemann 上已达 3 % 精度"。收敛门（波区 L1 随 N 至少 N^-0.5 下降）待细化作业回执后另行闭合。
"""
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    token = requests.post(base + "/auth/login",
                          json={"username": user, "password": password}, timeout=30).json()
    token = token.get("access_token") or token.get("token")
    body = {
        "title": "CKS1998 二维 Riemann 算例：对角对称线精确解参考（波区 18.3% honest-FAIL，光滑区 0.02%）",
        "summary": ("Lax-Liu config 3 的对角线对称性把二维解精确降维成一维 Riemann 问题，"
                    "用 Toro 精确解作参考把该算例从结构性门升级为可被否证的定量判据："
                    "N=32 上光滑区 0.021%、对称残差 3.2e-15，波区 18.28%（honest 未达 3%）；"
                    "否证实验（破坏 y=x 对称）波区误差升到 26.13%。"),
        "scenario_key": "tensordg.cks1998.riemann2d_diagonal",
        "mission_id": 23006,
        "tags": ["TensorDG", "CKS1998", "RKDG", "2D-Riemann", "exact-reference",
                 "verification-debt", "falsification"],
        "markdown": markdown,
        "figures": [
            {"name": "cks1998_riemann2d_diagonal_profile.png",
             "caption": "对角线 ρ(ξ)：RKDG P1 N=32 vs 精确一维 Riemann 解（阴影=波区 L1 窗口）",
             "png_base64": b64(p1)},
            {"name": "cks1998_riemann2d_diagonal_falsification.png",
             "caption": "否证实验：破坏 y=x 对称后同一判据的波区误差由 18.3% 升到 26.1%",
             "png_base64": b64(p2)},
        ],
    }
    resp = requests.post(base + "/tensor/reports", json=body, timeout=120,
                         headers={"Authorization": f"Bearer {token}"})
    print(resp.status_code, resp.text[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
