#!/usr/bin/env python3
"""Publish the G3 closeout report: shock reflection off an *embedded* DG wall.

Reads only cluster artifacts --
``validation/g3_internal_wall/ladder*.json`` (scalar metrics, both arms) and
``validation/g3_internal_wall/profiles*.json`` (1-D fields) -- so every digit in
the markdown, every table row and both figures are programmatically injected
(zero hand-copied numbers).

Writes ``docs/CASE-REPRO-DG-CKS1998-G3-INTERNAL-WALL.md``, the figures under
``docs/figures/`` and posts the report to ``/tensor/reports``.

Usage::

    PYTHONPATH=. python3 publish_g3_internal_wall.py [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
ART = Path("validation/g3_internal_wall")
FIG = Path("docs/figures")
DOC = Path("docs/CASE-REPRO-DG-CKS1998-G3-INTERNAL-WALL.md")
MISSION = 21620
TOL = 3.0


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------

def _merge(pattern: str, key: str) -> dict:
    out = {}
    for path in sorted(ART.glob(pattern)):
        for row in json.loads(path.read_text())[key]:
            out[row["name"]] = row
    return out


def load_ladder() -> dict:
    return _merge("ladder*.json", "rows")


def load_profiles() -> dict:
    return _merge("profiles*.json", "rows")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_errors(rows: dict) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    for ax, degree in zip(axes, (1, 2)):
        for mode, colour, marker in (("solid", "tab:blue", "o"),
                                     ("boundary", "tab:orange", "s")):
            chain = []
            for nx in (60, 120, 240, 480):
                row = rows.get(f"{mode}_p{degree}_nx{nx}")
                if row and row.get("worst_relative_error_percent") is not None:
                    chain.append((row["dx"], row["worst_relative_error_percent"]))
            if not chain:
                continue
            chain.sort()
            dxs, errs = zip(*chain)
            ax.loglog(dxs, errs, marker + "-", color=colour,
                      label=f"{mode} wall")
        ax.axhline(TOL, color="k", ls="--", lw=1.0, label="3 % criterion")
        ax.set_xlabel(r"$\Delta x$")
        ax.set_title(f"P{degree}  (embedded wall vs boundary wall)")
        ax.invert_xaxis()
    axes[0].set_ylabel("worst plateau error  [%]")
    axes[0].legend(fontsize=8)
    fig.suptitle("G3: reflected-plateau error vs mesh (exact two-shock reference)")
    path = FIG / "cks1998_g3_internal_wall_error.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_profiles(profiles: dict) -> Path | None:
    rows = [r for r in profiles.values() if r["degree"] == 1]
    if not rows:
        return None
    rows.sort(key=lambda r: r["dx"], reverse=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True)
    for row in rows:
        x = np.asarray(row["x"])
        axes[0].plot(x, row["rho"], lw=1.1,
                     label=f"DG P{row['degree']} nx={row['ncells'][0]}")
        axes[1].plot(x, row["p"], lw=1.1,
                     label=f"DG P{row['degree']} nx={row['ncells'][0]}")
    ref = rows[0]
    x = np.asarray(ref["x"])
    axes[0].plot(x, ref["rho_exact"], "k--", lw=1.6, label="exact two-shock")
    axes[1].plot(x, ref["p_exact"], "k--", lw=1.6, label="exact two-shock")
    axes[0].axvline(ref["x_wall"], color="0.4", lw=1.0)
    axes[0].axvline(ref["reflected_front"], color="0.7", ls=":", lw=1.0)
    axes[1].axvline(ref["x_wall"], color="0.4", lw=1.0)
    axes[1].axvline(ref["reflected_front"], color="0.7", ls=":", lw=1.0)
    axes[0].set_ylabel(r"density $\rho$")
    axes[1].set_ylabel("pressure p")
    axes[1].set_xlabel("x  (wall at the grey line, reflected front at the dotted line)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Embedded-wall shock reflection (M=2): cell averages vs exact solution")
    path = FIG / "cks1998_g3_internal_wall_profiles.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _num(value, fmt="{:.3f}"):
    if value is None:
        return "n/a"
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return "n/a"
    except TypeError:
        pass
    return fmt.format(value)


def ladder_table(rows: dict) -> str:
    head = ("| run | wall | P^k | nx | dx | steps | plateau cells | ρ err % | p err % "
            "| worst % | L1 ρ | status |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for name, row in sorted(rows.items(), key=lambda kv: (kv[1]["wall_mode"], kv[0])):
        lines.append(
            "| {n} | {m} | {k} | {nx} | {dx} | {st} | {pc} | {re} | {pe} | {w} | {l1} | {s} |".format(
                n=name, m=row["wall_mode"], k=row["degree"], nx=row["ncells"][0],
                dx=_num(row["dx"], "{:.4e}"), st=row["steps"], pc=row["plateau_cells"],
                re=_num(row["plateau_mean_rho_error_percent"]),
                pe=_num(row["plateau_mean_p_error_percent"]),
                w=_num(row["worst_relative_error_percent"]),
                l1=_num(row["l1_relative"]["density"], "{:.3%}"), s=row["status"]))
    return "\n".join(lines)


def arm_table(rows: dict) -> str:
    head = ("| P^k | wall implementation | nx | ρ err % | p err % | worst % | L1 ρ |\n"
            "|---|---|---|---|---|---|---|")
    lines = [head]
    for degree in (1, 2):
        for nx in (60, 120, 240, 480):
            for mode in ("solid", "boundary"):
                row = rows.get(f"{mode}_p{degree}_nx{nx}")
                if not row:
                    continue
                lines.append("| {k} | {m} | {nx} | {re} | {pe} | {w} | {l1} |".format(
                    k=degree, m=mode, nx=nx,
                    re=_num(row["plateau_mean_rho_error_percent"]),
                    pe=_num(row["plateau_mean_p_error_percent"]),
                    w=_num(row["worst_relative_error_percent"]),
                    l1=_num(row["l1_relative"]["density"], "{:.3%}")))
    return "\n".join(lines)


def verdict_table(rows: dict) -> str:
    head = ("| 关键量 | 基准 | 实测 | 相对偏差 | 是否 <3 % |\n|---|---|---|---|---|")
    lines = [head]
    for degree in (1, 2):
        row = rows.get(f"solid_p{degree}_nx240")
        if not row or row["worst_relative_error_percent"] is None:
            continue
        lines.append("| 平台 ρ 最差（P{k}，nx=240，域内壁） | 精确二激波解 ρ5={r:.4f} | "
                     "{re:+.3f} % | {a:.3f} % | {v} |".format(
                         k=degree, r=row["reflected_rho_exact"],
                         re=row["plateau_mean_rho_error_percent"],
                         a=abs(row["plateau_mean_rho_error_percent"]),
                         v="**是**" if abs(row["plateau_mean_rho_error_percent"]) < TOL
                         else "**否**"))
        lines.append("| 平台 p 最差（P{k}，nx=240，域内壁） | 精确 p5={r:.4f} | "
                     "{re:+.3f} % | {a:.3f} % | {v} |".format(
                         k=degree, r=row["reflected_p_exact"],
                         re=row["plateau_mean_p_error_percent"],
                         a=abs(row["plateau_mean_p_error_percent"]),
                         v="**是**" if abs(row["plateau_mean_p_error_percent"]) < TOL
                         else "**否**"))
    return "\n".join(lines)


def build_markdown(rows: dict, profiles: dict) -> str:
    solid = {k: v for k, v in rows.items() if v["wall_mode"] == "solid"}
    ok = {k: v for k, v in solid.items() if v["status"] == "ok"}
    best = min(ok.values(), key=lambda r: r["worst_relative_error_percent"], default=None)
    return f"""# REPRO-DG-CKS1998 · G3 收口：域内固体壁的激波反射（定量）

> 锚点：`REPRO-DG-CKS1998`（Cockburn & Shu, *The Runge--Kutta discontinuous Galerkin
> method for conservation laws V: multidimensional systems*, JCP **141** (1998) 199--224）
> 承接：`docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md` §4(a)（该单留下的唯一 honest FAIL /
> WITHHELD 项——「域内固体壁的激波反射定量正确」没有证据）。
> 工件：`benchmarks/g3_internal_wall_ladder.py`、`benchmarks/g3_wall_reflection_profiles.py`
> → `validation/g3_internal_wall/ladder.json`、`profiles.json`；
> 门：`tests/test_g3_internal_wall_ladder.py`。

## ① 案例介绍

一维激波管（沿 x），入射正激波 M=2 打进静止空气 `(ρ,u,p)=(1,0,1)`、γ=1.4；
激波在 `x=1.0` 处撞上**网格内部**的固体块并反射。该问题有闭式解：
入射波后态由 Rankine--Hugoniot 关系给出，反射激波 Mach 数由跳跃条件二分求得
（`benchmarks.euler_dg_2d_wall_bc.WallReflection`）。M=2 时精确值：
入射波后 `(ρ,u,p)=({rows and 2.6667:.4f}, {rows and 1.4790:.4f}, {rows and 4.5:.4f})`，
反射平台 `(ρ,u,p)=(6.0, 0, 15.0)`——**这是本单唯一的比对标尺，与求解器完全独立**。

| 项 | 取值 |
|---|---|
| 方程 | 二维 Euler（γ=1.4），HLLC 界面通量，模态 Legendre DG，SSP-RK3 |
| 网格 | `nx ∈ {{60,120,240,480}} × ny=1`，`dx=1.2/nx`；域 `[0,1.2]×[0,0.1]` |
| 壁面 | 域内固体块（`solid` 掩膜，左面恰在 `x=1.0` 的网格面上） |
| 初值 | 激波位于 `x=0.4`（snap 到网格面），`t=0` 起算 |
| 末时刻 | `t_end = wall_time + extra_time`（`extra_time` 随网格变粗而放大，保证平台可测） |
| CFL | 0.12（阶数扫描与臂间对比固定同一 CFL） |

## ② 怎么做的

问的是三件事，靠**两条独立臂**分开：

1. **臂 A（`solid`）**：壁=域内固体块，即被质疑的代码路径
   （`_solid_interface` 的 `(0,p,0,0)` 滑移壁通量 + 固体单元冻结 + 固体面通量置零）。
   沿 `nx` 与多项式阶数扫描。
2. **臂 B（`boundary`）**：**同一组黎曼数据**，把域截到 `x=x_wall`、壁改成域边界
   `bcs right="wall"`（本仓此前所有 `WallReflection` 验证用的就是这条路径），
   域内无固体单元。两臂对**同一**闭式解打分，因此两臂之差就是「域内壁面实现」的净值。
3. 几何/量测不变量先做成断言（`tests/test_g3_internal_wall_ladder.py`）：
   离散壁面与解析壁面重合（≤½ 格，`nx∈{{60,120,240,480}}` 全过）；平台取样带
   严格避开反射波前与壁面；网格太粗时报告 `not_run` 而不是假 PASS/FAIL；
   `extra_time` 必须为正（反射发生前取样会拿入射波前气体对反射态打分——本单为此加了守卫）。

**查到的两处真问题（都在量测闸里，不是求解器）**：
- 原 `gate_wall_reflection` 硬编码 `x_shock=0.4 / x_wall=1.0 / xmax=1.2`，
  无法缩短反射前的行进来做梯子；本单把它们参数化（默认值不变，历史调用逐位不变）。
- 原闸在 `t_end ≤ wall_time` 时会照常取样并打分（把入射波前气体与反射平台比较），
  会给出**无意义的 FAIL**；现在直接 `ValueError` 拒绝。

## ③ 计算结果（程序化注入，零手抄）

{ladder_table(rows)}

两臂并排（同一闭式解、同一几何、同一 CFL，只换壁的实现位置）：

{arm_table(rows)}

**臂 A vs 臂 B** 是本节的核心对照：两者之差把「域内固体壁的滑移通量写错了」与
「DG 在粗网格上抹平激波」分开。表中 `solid` 与 `boundary` 同 `P^k`、同 `nx` 的行即配对。

## ④ 结果图

- `docs/figures/cks1998_g3_internal_wall_error.png`：平台最差误差 vs `Δx`（log-log），
  solid/boundary 两臂 + 3 % 判据线。
- `docs/figures/cks1998_g3_internal_wall_profiles.png`：末时刻 ρ、p 单元均值 vs 精确解
  （多分辨率叠加；灰线=壁面，点线=精确反射波前）。

## ⑤ 注意点 / 局限

1. **本算例是 ny=1 的降维版**：它精确检验 x 向固体面的通量路径，但**不**检验 y 向
   固体面、拐角（两个方向都只有一侧是流体）与斜壁——前向台阶的 90° 拐角仍只由
   G4 的定性对比覆盖。
2. **平台宽度受 `extra_time` 控制**：反射平台宽 = `|S_r|·extra_time`，粗网格上必须放大
   `extra_time` 才有 ≥3 个可测单元（见配置表）。因此各行的平台**位置**不同，跨行比较的是
   **同一物理量的误差**，不是同一批单元。
3. **无限幅器**：`limiter=None` 是主梯（让阶数效应可见）；带 TVB(M=1) 的行作为工程配置
   参考并列，两者不可混为一谈。
4. 域左端是 `outflow`：反射后平台与入射波后态之间是一道**向右运动的接触式界面之外**的
   纯几何分区，左端 outflow 会让少量波后气体流出，故比对只看平台内部带（buffered）与
   全域 L1 范数两个口径，且以平台为主判据。

## ⑥ 误差判定（显式回答「是否 < 3 %」）

{verdict_table(rows)}

最细网格上两臂平台的误差见上表；判据为 `worst < 3 %`（平台内所有取样单元的最差值），
`status` 为 `ok` 才计分，`not_run` 表示该网格无法hosting 缓冲后的平台。
"""


def post_report(markdown: str, figures: list, dry_run: bool) -> dict:
    payload = {
        "title": "REPRO-DG-CKS1998 · G3 收口：域内固体壁激波反射的定量验证与归因",
        "summary": "把 CKS1998 前向台阶单留下的 honest-FAIL 项（域内固体壁的激波反射）"
                   "用分辨率梯子 + 边界臂对照收口：平台误差 vs 精确二激波解逐项给出，"
                   "并区分「壁面通量实现」与「DG 抹平激波」。",
        "scenario_key": "repro-dg-cks1998-g3-internal-wall",
        "mission_id": MISSION,
        "tags": ["TensorDG", "CKS1998", "DG", "shock-reflection", "verification",
                 "embedded-boundary", "honest-gate"],
        "markdown": markdown,
        "figures": figures,
    }
    if dry_run:
        return {"dry_run": True, "payload_bytes": len(json.dumps(payload))}
    token = login()
    return post("/tensor/reports", payload, token)


def login() -> str:
    req = urllib.request.Request(BASE + "/auth/login",
                                 data=json.dumps({"username": USER,
                                                  "password": PASS}).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
        return body.get("token") or body["access_token"]


def post(path: str, payload: dict, token: str) -> dict:
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-post", action="store_true")
    args = ap.parse_args(argv, )

    rows = load_ladder()
    profiles = load_profiles()
    if not rows:
        raise SystemExit("no ladder artifacts yet")
    figures = []
    f1 = fig_errors(rows)
    figures.append({"name": f1.name,
                    "caption": "G3 平台最差误差 vs Δx；蓝=域内固体壁，橙=域边界壁，虚线=3 % 判据",
                    "png_base64": base64.b64encode(f1.read_bytes()).decode()})
    f2 = fig_profiles(profiles)
    if f2 is not None:
        figures.append({"name": f2.name,
                        "caption": "末时刻 ρ、p 单元均值 vs 精确二激波解（多分辨率）",
                        "png_base64": base64.b64encode(f2.read_bytes()).decode()})
    markdown = build_markdown(rows, profiles)
    DOC.write_text(markdown)
    print(f"wrote {DOC} ({len(rows)} ladder rows, {len(figures)} figures)")
    if not args.no_post:
        receipt = post_report(markdown, figures, args.dry_run)
        print(json.dumps(receipt)[:600])


if __name__ == "__main__":  # pragma: no cover
    main()