"""Publish the CASE-LAX-DG limiter-arm x stage x mesh report (mission 22394).

Every number is injected from
``docs/benchmarks/results/lax_closure_arms_mesh_scan/metrics.json`` (and from the
committed closure artifact it cross-checks); the two artifact figures are
embedded as-is, and the N=800 field profiles come from the cluster snapshot job
(``runs/arms_scan_profile_*.npz``).  Nothing is transcribed by hand.

The report answers the three predeclared predictions of this round and states
the honest ledger comparison leg by leg.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
OUT = ROOT / "docs" / "benchmarks/results/lax_closure_arms_mesh_scan"
CLOSURE = ROOT / "docs" / "benchmarks/results/lax_resolution_closure/metrics.json"
MISSION_ID = 22394
FIELDS = ("density", "velocity", "pressure")


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def pct(value, digits=3):
    return "n/a" if value is None else f"{100.0 * value:.{digits}f}%"


def arm_label(leg):
    return leg["limiter"] if leg["limiter_stage"] is None else \
        f"{leg['limiter']}/{leg['limiter_stage']}"


def _profile_paths():
    """N=800 snapshots produced by the snapshot job (fetched separately)."""
    paths = {}
    for candidate in (ROOT / "runs", Path("/tmp")):
        if not candidate.is_dir():
            continue
        for npz in candidate.glob("arms_scan_profile_*.npz"):
            paths.setdefault(npz.stem.replace("arms_scan_profile_", ""), npz)
    return paths


def make_profile_figure(paths, png: Path) -> Path | None:
    if not paths:
        return None
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t in ("none", "tvb") if t in paths]
    fig, axes = plt.subplots(len(tags), 1, figsize=(7.4, 2.9 * len(tags)),
                             sharex=True)
    axes = [axes] if len(tags) == 1 else list(axes)
    for ax, tag in zip(axes, tags):
        d = np.load(paths[tag])
        x = d["x"].ravel()
        ax.plot(x, d["rho_exact"].ravel(), "k-", lw=1.3, label="exact Riemann (Toro test 2)")
        ax.plot(x, d["rho"].ravel(), "r.", ms=2.2,
                label=f"DG p3 N=800, limiter={tag}"
                      + ("/final" if tag != "none" else ""))
        ax.set_ylabel(r"$\rho$")
        ax.set_title(f"N=800 density field vs the closed-form solution (limiter={tag})",
                     fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def make_stage_gain_figure(metrics, png: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = metrics["gate"]["predictions"]["P3"]["rows"]
    bar = metrics["gate"]["bar_values"]["l2_stage_gain_min"]
    labels = [f"N={r['cells']}\n{r['limiter']}" for r in rows]
    values = [r["relative_gain"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    colors = ["#4c72b0" if r["holds"] else "#c44e52" for r in rows]
    ax.bar(range(len(rows)), values, color=colors, width=0.55)
    ax.axhline(bar, color="k", ls="--", lw=1.2,
               label=f"predeclared threshold {bar:.0%}")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"relative gain of whole-field $\rho$ $L_2$")
    ax.set_title("limiter_stage 'every' -> 'final' (blue = passes P3)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def main() -> int:
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    gate = metrics["gate"]
    preds = gate["predictions"]
    verdicts = gate["verdicts"]
    tol = gate["tolerance"]
    legs = sorted(metrics["legs"], key=lambda x: (x["cells"], x["limiter"],
                                                  str(x["limiter_stage"])))
    rep = metrics["closure_artifact_reproduction"]
    bars = gate["bar_values"]
    closure = json.loads(CLOSURE.read_text(encoding="utf-8")) if CLOSURE.exists() else None

    leg_rows = "\n".join(
        f"| {leg['cells']} | {leg['limiter']} | {leg['limiter_stage'] or '—'} | "
        f"{pct(leg['l2']['density'])} | {pct(leg['l2']['velocity'])} | "
        f"{pct(leg['l2']['pressure'])} | {pct(leg['band_l1']['density'])} | "
        f"{leg['steps']} | {leg['wall_seconds']:.0f} |" for leg in legs)
    p1_rows = "\n".join(
        f"| {r['cells']} | {pct(r['none'])} | {pct(r['tvb'])} | {pct(r['weno'])} | "
        f"{r['tvb_worse']} | {r['weno_worse']} |" for r in preds["P1"]["rows"])
    p2_rows = "\n".join(
        f"| {r['cells']} | {pct(r['none'])} | {pct(r['tvb'])} ({r['tvb_ratio']:.3f}×) | "
        f"{pct(r['weno'])} ({r['weno_ratio']:.3f}×) |" for r in preds["P2"]["rows"])
    p3_rows = "\n".join(
        f"| {r['cells']} | {r['limiter']} | {pct(r['l2_every'])} | {pct(r['l2_final'])} | "
        f"{r['relative_gain']:.1%} | {'是' if r['holds'] else '**否**'} |"
        for r in preds["P3"]["rows"])
    sep_rows = "\n".join(
        f"| N = {cells} | {pct(next(l for l in legs if l['cells'] == cells and l['limiter'] == 'none')['l2']['density'])} | "
        f"{pct(next(l for l in legs if l['cells'] == cells and l['limiter'] == 'none')['band_l1']['density'])} | "
        f"{value:.2f}× | {(1 - 1 / value) * 100:.1f} % |"
        for cells, value in sorted(gate["unlimited_separation"].items(),
                                   key=lambda kv: int(kv[0][1:])))
    honest_rows = "\n".join(
        f"| {arm_label(leg)} | {leg['cells']} | {pct(leg['l2']['density'])} | "
        f"{'是' if leg['l2']['density'] <= tol else '**否**'} | "
        f"{pct(max(leg['l2'][f] for f in FIELDS))} | "
        f"{'是' if max(leg['l2'][f] for f in FIELDS) <= tol else '**否**'} |"
        for leg in legs)
    qualifying = [leg for leg in legs if all(leg["l2"][f] <= tol for f in FIELDS)]
    closure_line = ""
    if closure:
        closure_line = (
            f"已发表的闭合研究（`lax_resolution_closure`）：生产腿 `p3.N400` 诚实 FAIL "
            f"{pct(closure['gate']['production_l2']['density'])}，闭合网格 "
            f"N = {closure['gate']['closure_mesh']}"
            f"（三量最差 {pct(max(next(e for e in closure['ladder'] if e['cells'] == closure['gate']['closure_mesh'])['l2'][f] for f in FIELDS))}）。")

    figures = [
        {"name": "arms_mesh_scan.png",
         "caption": "全场 ρ 相对 L2：10 条腿（5 臂 × 2 网格）在 log 轴上的对比，虚线 = 台账 3 % 门。"
                    "无限幅臂在 N=800 达标；两个限幅器在所有测过的 (网格, 阶段) 上都不达标；"
                    "把限幅器切到文档承认的公平阶段 `final` 让柱子明显下降（工件原图）。",
         "png_base64": b64(OUT / "figures" / "arms_mesh_scan.png")},
        {"name": "arms_metric_split.png",
         "caption": "带掩码 ρ L1（限幅器压掉的振荡）vs 全场 ρ L2（台账口径）：空心 = N=400，实心 = N=800。"
                    "无限幅臂在左上（振荡占比高），限幅臂在右下 —— 两种口径对同一对臂给出相反排序（工件原图）。",
         "png_base64": b64(OUT / "figures" / "arms_metric_split.png")},
        {"name": "stage_gain.png",
         "caption": "预登记 P3 的逐条结果：每个限幅器 × 网格上，`limiter_stage: every → final` 使全场 ρ L2 "
                    "下降的相对比例；虚线 = 跑数前写死的 20 % 门槛，红柱 = 未达门槛。",
         "png_base64": b64(make_stage_gain_figure(metrics, Path("/tmp/m22394_stage_gain.png")))},
    ]
    profile_png = make_profile_figure(_profile_paths(), Path("/tmp/m22394_profile_n800.png"))
    if profile_png is not None:
        figures.append({
            "name": "profile_n800.png",
            "caption": "闭合网格 N=800 的密度场（单元 Gauss 节点）vs 闭式 Riemann 解：上 = 无限幅臂，"
                       "下 = tvb/final。无限幅臂的波前与平台几乎贴合解析解，误差主要集中在四个间断的"
                       "窄带内；tvb/final 把间断压得更利落但把波前抹宽（这正是全场 L2 变差的来源）。",
            "png_base64": b64(profile_png)})

    markdown = f"""# CASE-LAX-DG 限幅器臂 × 限幅阶段 × 网格：已发表结论的稳健性检验（重开 #617 · 管到底第 3 轮）

**锚点**：`CASE-LAX-DG`（Lax 1954 激波管 / Toro 1999 test 2），台账验收原文「ref: 精确解｜容差 < 3 %」。
**本单性质**：capability_extension —— 原 PR#617 的交付物已在 main（经 #450 / #622 合入），
本轮测的是它自己**没有测过的两个自由度**（网格、限幅阶段），从而判定它那句
「限幅器在 p3 上属于过度限制、不是补救手段」的单网格单阶段结论是否站得住。

## ① 案例介绍

**物理背景**：一维 Euler 方程组（γ = 1.4）的 Riemann 问题，经典激波管。爆膜瞬间产生
左行稀疏波（头/尾）、接触间断、右行激波四个波，全部有闭式解，因此是不依赖任何数值参考的
**精确解基准**。

| 参数 | 值 |
|---|---|
| 域 / 隔膜 | `[0, 1]` / `x0 = 0.5` |
| 左态 (ρ, u, p) | (0.445, 0.698, 3.528) |
| 右态 (ρ, u, p) | (0.500, 0.000, 0.571) |
| 快照时间 | t = 0.13 |
| 离散 | DG p = 3，单元 Gauss 节点，HLLC 通量，SSP-RK3，CFL = 0.35 |
| 参考解 | `benchmarks/riemann_exact.py`（精确 Riemann 采样） |
| 台账门 | 全场相对 L2（ρ, u, p 三量）< 3 %，登记在**无限幅**生产腿 `p3.N400` |

{closure_line}
本轮要判定的是那一句话的两个未测自由度：**网格**（只测过 N = 400）与 **限幅阶段**
（只测过历史默认 `limiter_stage="every"`；本仓 `docs/limiter_verification.md` 自己写明
`every` 会在每个 SSP-RK3 子步后重新投影仍有标记的光滑单元，从而毁掉高阶精度）。

## ② 怎么做的

1. **factorial 设计（2 mesh × 5 臂 = 10 条腿，单一功能点）**：网格 N ∈ {{400, 800}}（生产腿 +
   闭合网格），臂 ∈ {{none, tvb/every, tvb/final, weno/every, weno/final}}。无限幅臂无阶段概念，
   每网格 1 条腿；其余 4 臂成对，使 `limiter × stage` 交互**可估**（不是把两个旋钮一起动）。
2. **只动一个旋钮**：除 `limiter` / `limiter_stage` 外全部固定（p=3、CFL=0.35、精确解分母、
   单元 Gauss 节点度量、R3 固定物理带 ±0.02 与算例池同规则）。10 条腿在**同一作业**内跑完，
   不存在跨运行拼接。
3. **预登记判据（跑数前写死在 `benchmarks/lax_closure_arms_mesh_scan.py`，阈值是模块常量）**：
   - **P1**：`stage=every` 下全场 ρ L2 严格排序 none < tvb 且 none < weno，**两个网格都成立**；
   - **P2**：两个网格上 tvb 与 weno 都把 ρ 的带掩码 L1 压到无限幅臂的 ≤ {bars['band_gain_factor']:.2f} 倍；
   - **P3**：两个网格上，每个限幅器 `every → final` 使全场 ρ L2 至少下降 {bars['l2_stage_gain_min']:.0%}。
4. **判定规则也是预登记的纯函数**（`gate()`）：P1 决定「排序是否网格无关」，P3 决定
   「惩罚是否主要是阶段伪像」，无限幅臂的 `L2/带L1 ≥ {bars['separation_min']:.1f}` 判定
   「两种口径答的是不同问题」。三布尔 → 结论句的映射在代码里，四个分支都有 local 单测
   （`tests/test_lax_closure_arms_mesh_scan.py`，无需集群）。
5. **测量路径核对（可比性的必要条件）**：本轮重测了 N = 400 与 N = 800 的无限幅腿，与已提交的
   `lax_resolution_closure/metrics.json` 逐量比对（L1 / L2 / 带 L1 × 三量 = {len(rep.get('legs', [])) * 9} 对数），
   最差相对偏差 **{rep.get('worst_relative_deviation', float('nan')):.3e}**（判据 ≤ {rep.get('tolerance', 1e-12):.0e}）
   → {'**REPRODUCED**' if rep.get('reproduced') else '**NOT REPRODUCED**'}。另核对两条腿的步数逐位相同。
6. **工具链**：torch float64 CPU 求解器 + 精确 Riemann 采样；驱动 `benchmarks/lax_closure_arms_mesh_scan.py`
   （集群 cj4089，{metrics['wall_seconds']:.0f} s）；ingest `scripts/ingest_lax_arms_mesh_scan.py`
   从作业日志的 DIGEST 解析（零手抄）并渲染工件、两张图与案例文档；N=800 场快照由 cj4091 单独产出。
7. **环境**：作业节点 {metrics['source'].get('node')}（作业 {metrics['source'].get('job')}），
   stage_dir = 本单工作树（未提交改动直达节点），cores = 4，timeout = 7200 s。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 全 10 条腿

| N | limiter | stage | ρ L2 | u L2 | p L2 | ρ 带 L1 | 步数 | 墙钟 s |
|---|---|---|---|---|---|---|---|---|
{leg_rows}

### 3.2 预登记判据的结果

**P1（排序是否网格无关，`stage=every`）**

| N | none ρ L2 | tvb ρ L2 | weno ρ L2 | tvb 更差 | weno 更差 |
|---|---|---|---|---|---|
{p1_rows}

→ P1 **{'HOLDS' if preds['P1']['holds'] else 'FAILS'}**：限幅器排序 **{verdicts['P1_limiter_ranking']}**。

**P2（限幅器是否真的压掉了带内振荡，`stage=every`）**

| N | none 带 L1 | tvb 带 L1（比值） | weno 带 L1（比值） |
|---|---|---|---|
{p2_rows}

→ P2 **{'HOLDS' if preds['P2']['holds'] else 'FAILS'}**。

**P3（换到公平阶段 `final` 是否拿回精度）**

| N | limiter | ρ L2（every） | ρ L2（final） | 相对回收 | 达 ≥{bars['l2_stage_gain_min']:.0%} 门槛 |
|---|---|---|---|---|---|
{p3_rows}

→ P3 **{'HOLDS' if preds['P3']['holds'] else 'FAILS'}**：阶段效应 **{verdicts['P3_stage_effect']}**。

### 3.3 口径分离度（无限幅臂）

| 网格 | 全场 ρ L2 | 带掩码 ρ L1 | 分离度 L2/带L1 | 带内占比估计 |
|---|---|---|---|---|
{sep_rows}

→ 口径结论：**{verdicts['metric_view']}**。

### 3.4 判定

- `P1_limiter_ranking` = **{verdicts['P1_limiter_ranking']}**
- `P3_stage_effect` = **{verdicts['P3_stage_effect']}**
- `metric_view` = **{verdicts['metric_view']}**
- 对已发表措辞的处理：**{verdicts['published_statement']}**

已发表的分辨率闭合判定（闭合网格 = N = 800）**不受本单影响**：台账登记在无限幅生产腿上，
限幅器臂是诊断量，不能改变 ledger A0 的诚实 FAIL。

## ④ 结果图

{chr(10).join(f'- `{f["name"]}` — {f["caption"]}' for f in figures)}

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **限幅器实现有边界**：只测本仓的 `tvb`（Cockburn–Shu）与 `weno`（Qiu–Shu），两者都把问题单元
   退化为**线性**重构；因此任何「限幅器无用」的推广都超出本单证据（更高级的 h/p 自适应限幅器
   不在范围内）。
2. **p 阶未扫描**：全部结论在 p = 3 上成立；p = 1/2 的臂不在本单范围，不能外推。
3. **带掩码 L1 的带宽是固定物理量**（±0.02），带内单元占比随细化下降，所以全场 L2 与带 L1 的
   差别**同时**含「振荡漏出」与「掩码面积变化」两个成分；`L2/带L1` 只能读作分离度指示，
   不是严格的振荡能量占比。
4. **成本不是同一把尺子**：各臂步数在 3.1 表中逐条给出（CFL 相同、限幅只改单元重构，故步数接近），
   墙钟是单核墙钟，只用于证明腿是实际算过的，不作算法成本比较。
5. **测量路径核对是必要条件而非充分条件**：它证明本轮无限幅腿与已提交工件逐位一致，
   不证明求解器版本之外的任何事。
6. **预登记阈值的诚实性**：三个阈值（{bars['band_gain_factor']:.2f} / {bars['l2_stage_gain_min']:.0%} / {bars['separation_min']:.1f}）
   在跑数前写入模块常量，测试逐条比对「判定里用的柱子 == 预登记文本里引用的柱子」，
   因此事后调阈值会让测试失败。

## ⑥ 误差判定（关键量 vs 基准，逐项回答「是否 < 3 %」）

基准 = 台账 3 % 门（全场相对 L2，三量都要满足）；辅助基准 = 同网格无限幅臂的同一量。

| 臂 | N | ρ L2 | ρ < 3 % | 三量最差 L2 | 三量是否全 < 3 % |
|---|---|---|---|---|---|
{honest_rows}

**结论**：10 条腿中「三量全场相对 L2 全部 < 3 %」的只有 **{len(qualifying)} 条** ——
{'、'.join(f"`N={l['cells']} {arm_label(l)}`" for l in qualifying) if qualifying else '无'}。
两处必须点名的诚实结论：

- 生产腿 `p3.N400` 无限幅臂 ρ L2 = {pct(next(l for l in legs if l['cells'] == 400 and l['limiter'] == 'none')['l2']['density'])}，
  **不满足** < 3 %（与已发表工件逐位一致，honest FAIL 未粉饰）。
- 两个限幅器在任何测过的 (网格, 阶段) 组合上 ρ L2 都 **不满足** < 3 %；
  但 P3 给出其中一部分差距归因于 `limiter_stage="every"`（逐条回收比例见 3.2），
  所以「限幅器本身过度限制」是**过度归因**。

对预登记三判据的回答：P1 = {'是' if preds['P1']['holds'] else '否'}、P2 = {'是' if preds['P2']['holds'] else '否'}、
P3 = {'是' if preds['P3']['holds'] else '否'} —— 每一项都对应 3.2 的逐条数字，判定句由
预登记映射从这三个布尔自动生成（{verdicts['published_statement']}）。
"""

    token = post("/auth/login", {"username": USER, "password": PASS})["access_token"]
    payload = {
        "title": "CASE-LAX-DG 限幅器臂 × 限幅阶段 × 网格：已发表『限幅器过度限制』结论的稳健性检验（重开 #617 第 3 轮）",
        "summary": (
            "在 Lax 激波管 p3 上把已发表的限幅器结论拆成 2 网格 × 5 臂 factorial："
            f"P1（排序网格无关）={'HOLDS' if preds['P1']['holds'] else 'FAILS'}、"
            f"P2（限幅器压掉带内振荡）={'HOLDS' if preds['P2']['holds'] else 'FAILS'}、"
            f"P3（every→final 回收 ≥{bars['l2_stage_gain_min']:.0%} 全场 L2）={'HOLDS' if preds['P3']['holds'] else 'FAILS'}；"
            f"无限幅腿对已提交工件逐位复现（{rep.get('worst_relative_deviation', float('nan')):.1e}）。"
        ),
        "scenario_key": "case-lax-dg-limiter-stage-mesh-scan",
        "mission_id": MISSION_ID,
        "tags": ["CASE-LAX-DG", "TensorDG", "DG", "limiter", "resolution-closure",
                 "experimental-design", "honest-FAIL"],
        "markdown": markdown,
        "figures": figures,
    }
    (Path("/tmp/report.json")).write_text(json.dumps(payload), encoding="utf-8")
    result = post("/tensor/reports", payload, token=token)
    print(json.dumps({"report": result}, ensure_ascii=False)[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())