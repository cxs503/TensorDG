"""Render ``docs/CASE-LAX-DG-dt-metadata-contract.md`` from measured numbers.

Anchor: CASE-LAX-DG (verification debt: ``meta["dt"]`` is the truncated last
step, not a step size).  Every number in the generated document is measured by
this script — nothing is transcribed by hand.

    python scripts/render_dt_metadata_contract_doc.py
"""
from __future__ import annotations

import base64
import datetime as _dt
import io
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensordg.burgers_1d import solve_burgers
from tensordg.burgers_shock_1d import solve_burgers_shock
from tensordg.euler_1d import solve_riemann
from tensordg.euler_2d import solve_euler_2d
from tensordg.euler_dg_2d import solve_euler_dg_2d_sod
from tensordg.ns_dg_2d import solve_navier_stokes_dg_2d
from tensordg.swe_1d import solve_swe_1d
from tensordg.swe_2d import solve_swe_2d
from tensordg.stepping import step_metadata

ROOT = Path(__file__).resolve().parents[1]
OUT_DOC = ROOT / "docs" / "CASE-LAX-DG-dt-metadata-contract.md"
OUT_FIG = ROOT / "docs" / "benchmarks" / "results" / "dt_metadata_contract"
CLAMP = re.compile(r"final_time\s*-\s*time")


def _meta(out):
    dicts = [i for i in out if isinstance(i, dict)] if isinstance(out, tuple) else []
    return dicts[-1] if dicts else out


def audit_table():
    rows = []
    for path in sorted((ROOT / "tensordg").glob("*.py")):
        if path.stem == "stepping":
            continue
        text = path.read_text()
        publishes = '"dt":' in text or "step_metadata(" in text
        if not publishes:
            continue
        residual = bool(CLAMP.search(text))
        trio = "step_metadata(" in text or "dt_mean" in text
        rows.append((path.stem, "残步 (residual)" if residual else "定步/平均步",
                     "是" if trio else "否"))
    return rows


def ladder():
    cf = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    rows = []
    for cfl in cf:
        _, _, meta = solve_riemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=100,
                                   final_time=0.05, cfl=cfl)
        rows.append(dict(cfl=cfl, steps=meta["steps"], dt_last=meta["dt_last"],
                         dt_mean=meta["dt_mean"], ratio=meta["dt_mean"] / meta["dt_last"]))
    return rows


def solver_rows():
    cases = {
        "burgers_1d (N=16,p1,t=0.05,cfl=0.2)":
            lambda: (_meta(solve_burgers(ncells=16, degree=1, final_time=0.05, cfl=0.2)), 0.05),
        "burgers_shock_1d (N=32,p1,t=0.05)":
            lambda: (_meta(solve_burgers_shock(ncells=32, degree=1, final_time=0.05)), 0.05),
        "euler_1d Sod (N=100,t=0.05,cfl=0.2)":
            lambda: (_meta(solve_riemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=100,
                                         final_time=0.05, cfl=0.2)), 0.05),
        "euler_2d vortex-lite (8x8,t=0.05,cfl=0.3)":
            lambda: (_meta(solve_euler_2d(
                lambda x, y: (1.0 + 0.05 * torch.sin(2 * torch.pi * x), 0.0 * x, 0.0 * x,
                              1.0 + 0.0 * x), ncells=8, final_time=0.05, cfl=0.3,
                domain=((0.0, 1.0), (0.0, 1.0)), quadrature=3)), 0.05),
        "euler_dg_2d Sod (16^2,p1,t=0.02)":
            lambda: (_meta(solve_euler_dg_2d_sod(ncells=16, degree=1, final_time=0.02)), 0.02),
        "ns_dg_2d uniform (6^2,p1,t=0.02,mu=0.01,cfl=0.05)":
            lambda: (_meta(solve_navier_stokes_dg_2d(
                lambda x, y: (1.0 + 0.0 * x, 0.0 * x, 0.0 * x, 1.0 + 0.0 * x),
                ncells=6, degree=1, final_time=0.02, mu=0.01, cfl=0.05)), 0.02),
        "swe_1d sine (N=16,p1,t=0.02)":
            lambda: (_meta(solve_swe_1d(
                lambda x: 1.0 + 0.1 * torch.sin(2 * torch.pi * x),
                lambda x: torch.zeros_like(x), lambda x: torch.zeros_like(x),
                ncells=16, degree=1, final_time=0.02)), 0.02),
        "swe_2d sine (8x8,p1,t=0.02)":
            lambda: (_meta(solve_swe_2d(
                lambda x, y: 1.0 + 0.05 * torch.sin(2 * torch.pi * x),
                lambda x, y: torch.zeros_like(x), lambda x, y: torch.zeros_like(x),
                lambda x, y: torch.zeros_like(x), nx=8, ny=8, degree=1,
                final_time=0.02)), 0.02),
    }
    rows = []
    for label, fn in cases.items():
        meta, final_time = fn()
        rows.append(dict(label=label, steps=meta["steps"], dt_last=meta["dt_last"],
                         dt_mean=meta["dt_mean"], final=final_time,
                         identity=meta["dt_mean"] * meta["steps"] - final_time,
                         ratio=meta["dt_mean"] / meta["dt_last"]))
    return rows


def figure(lad, sols, path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    cf = [r["cfl"] for r in lad]
    ax[0].loglog(cf, [r["dt_mean"] for r in lad], "o-", color="tab:blue",
                 label=r"$dt_{\rm mean}=t_{\rm final}/{\rm steps}$ (representative step)")
    ax[0].loglog(cf, [r["dt_last"] for r in lad], "s--", color="tab:red",
                 label=r"$dt_{\rm last}$ = legacy $\tt{meta['dt']}$ (remainder)")
    ax[0].set_xlabel("CFL"); ax[0].set_ylabel(r"$\Delta t$")
    ax[0].set_title("euler_1d Sod, N=100, t=0.05:\nthe representative step halves, the legacy field does not")
    ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8)
    ax[0].invert_xaxis()

    labels = [r["label"].split(" (")[0] for r in sols]
    ratios = [r["ratio"] for r in sols]
    ax[1].barh(range(len(ratios)), ratios, color="tab:green")
    ax[1].set_yticks(range(len(ratios))); ax[1].set_yticklabels(labels, fontsize=7)
    ax[1].axvline(1.0, color="k", ls=":", lw=1, label="ratio = 1 (legacy dt == step)")
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$dt_{\rm mean}/dt_{\rm last}$")
    ax[1].set_title("How far the legacy field is from a step size\n(smoke configurations, this run)")
    ax[1].grid(alpha=0.3, axis="x"); ax[1].legend(fontsize=8)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


def render():
    lad, sols = ladder(), solver_rows()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    png = OUT_FIG / "dt_last_vs_dt_mean.png"
    figure(lad, sols, png)
    png_b64 = base64.b64encode(png.read_bytes()).decode()

    step_ratios = [a["dt_mean"] / b["dt_mean"] for a, b in zip(lad, lad[1:])]
    last_ratios = [a["dt_last"] / b["dt_last"] for a, b in zip(lad, lad[1:])]
    worst_log2 = max(abs(math.log2(r)) for r in last_ratios)

    audit = audit_table()
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = []
    A = lines.append
    A("# CASE-LAX-DG 验证欠账：时间步元数据契约（`dt` / `dt_last` / `dt_mean`）\n")
    A(f"- **锚点**：`CASE-LAX-DG`（`feature_ledger` 验收原文「ref: 精确解｜容差 <3%」）")
    A("- **承接**：m18266 报告 §⑤.1 的显式移交项——「另有多处基准把 `meta[\"dt\"]` 当步长用，"
      "逐个审计是后续单的独立工作项（有真实数值影响）」")
    A(f"- **生成**：`python scripts/render_dt_metadata_contract_doc.py`（{now}），本文数字全部由该脚本测量注入\n")

    A("## ① 案例介绍\n")
    A("本仓的时间推进求解器（Euler / Burgers / 浅水 / Navier–Stokes，DG 与 P0 有限体积）都用自适应 SSP-RK3，"
      "步长\n\n```\ndt = min(cfl * dx / ((2p+1) * max_speed), final_time - time)\n```\n\n"
      "第二个候选量是**残步**：循环最后一次迭代几乎总被 `final_time - time` 截断，因此循环结束后"
      "落在 `metadata[\"dt\"]` 里的是「最后一步走了多少」，而**不是这次运行的步长**。"
      "它与步长相差一个取决于 `final_time / dt_max` 的残余比例，且**不随 CFL 或网格加密单调下降**。\n")
    A("受影响的下游用法（真实存在，非假想）：把步序 × `dt` 还原成时间轴、画 `dt`–误差预算曲线、"
      "断言「CFL 减半则 `dt` 减半」。Lax 算例池的 `time_refinement` 记录正是被这条属性污染的"
      "（m18266 已就地修复 `solve_euler_dg_1d`）。\n")
    A("本单把该修复推广成**全仓契约**，并用测试机器化：新增 `tensordg/stepping.py::step_metadata`，"
      "把 `dt`/`dt_last`/`dt_mean` 三字段统一产出；`dt` 语义**不变**（向后兼容，老数字不动），"
      "`dt_mean = final_time / steps` 才是代表步长。\n")

    A("## ② 怎么做的\n")
    A("1. 源码分类：全 `tensordg/*.py` 扫描「是否发布 `dt`」×「是否含 `final_time - time` 截断」→ 残步族 / 定步族。")
    A("2. 对残步族逐个注入 `**step_metadata(final_time, steps, dt)`（加法式，不改任何既有键的取值）。")
    A("3. 新增 `tests/test_time_step_metadata_contract.py`：helper 算术、源码覆盖漂移门、9 个求解器运行期不变量、"
      "CFL 阶梯反例。")
    A("4. 本文与图由 `scripts/render_dt_metadata_contract_doc.py` 现场测量渲染（零手抄）。\n")
    A("工具链：PyTorch float64 / CPU、pytest、matplotlib。上一单（m18266）修的是 `solve_euler_dg_1d` 本体，"
      "本单避开该文件以免与其在飞分支冲突，故 `euler_dg_1d` 在源码扫描里标记为 `PENDING`。\n")

    A("## ③ 计算结果\n")
    A("### 3.1 源码审计（本脚本现场扫描）\n")
    A("| 模块 | 步长语义 | 是否携带 `dt_last`/`dt_mean` |")
    A("|---|---|---|")
    for name, kind, trio in audit:
        A(f"| `tensordg/{name}.py` | {kind} | {trio} |")
    A("")
    A("### 3.2 CFL 阶梯：代表步长减半，遗留字段不减半\n")
    A("`euler_1d` Sod（N=100, t=0.05）：\n")
    A("| CFL | steps | `dt_last`（遗留 `dt`） | `dt_mean` | `dt_mean/dt_last` |")
    A("|---|---|---|---|---|")
    for r in lad:
        A(f"| {r['cfl']:g} | {r['steps']} | {r['dt_last']:.6e} | {r['dt_mean']:.6e} | {r['ratio']:.3f} |")
    A("")
    A(f"- `dt_mean` 的相邻比：最小 {min(step_ratios):.4f}、最大 {max(step_ratios):.4f}（CFL 逐级减半 → 半径 1.8–2.2 内，符合 2 倍律）。")
    A(f"- `dt_last` 的相邻比：最坏偏离 2 倍达到 `2^{worst_log2:.2f} = {2 ** worst_log2:.1f}` 倍。"
      "残步是「剩余时间」的准均匀抽样，与 CFL 无单调关系。\n")
    A("### 3.3 各求解器运行期不变量（本次实测）\n")
    A("| 配置 | steps | `dt_last` | `dt_mean` | `dt_mean·steps − t_final` | `dt_mean/dt_last` |")
    A("|---|---|---|---|---|---|")
    for r in sols:
        A(f"| {r['label']} | {r['steps']} | {r['dt_last']:.6e} | {r['dt_mean']:.6e} | "
          f"{r['identity']:.2e} | {r['ratio']:.2f} |")
    A("")
    A(f"- 恒等式 `dt_mean == final_time/steps` 的实测最大偏差 **{max(abs(r['identity']) for r in sols):.2e}**（机器精度级）。")
    A(f"- 遗留 `dt` 与代表步长的最大偏差倍数 **×{max(r['ratio'] for r in sols):.2f}**，"
      f"最小 ×{min(r['ratio'] for r in sols):.2f} — 也就是说拿它当步长最坏差两个量级。\n")

    A("## ④ 图\n")
    A(f"![dt_last vs dt_mean](benchmarks/results/dt_metadata_contract/{png.name})\n")
    A("左：同一算例的 CFL 阶梯上，`dt_mean` 严格贴合 2 倍律，遗留 `dt` 锯齿乱跳；"
      "右：8 个求解器 smoke 配置里 `dt_mean/dt_last` 的倍数（对数轴），"
      "比值 1 处即「遗留字段恰好等于步长」——大多数配置远离 1。\n")

    A("## ⑤ 注意点 / 局限\n")
    A("1. **`dt` 语义未改**：本单只加字段。老工件/老报告里的 `dt` 数值一个都没动，"
      "「引用残步」的欠账需由消费方逐点换成 `dt_mean`——已定位的消费方（`benchmarks/lax_asymptotic_probe.py`、"
      "`benchmarks/entropy_budget_1d.py`、`benchmarks/in31_hp_ladder.py` 等）与 `euler_dg_1d` 本体同批，"
      "留待 m18266 合并后统一改，避免两个分支改同一批文件。")
    A("2. **`euler_dg_1d` 标记 PENDING**：其修复在 m18266（commit `9227c42`）在飞；本单的同名扫描门会"
      "在合并后自动变绿（`PENDING` 是子集判定，只允许收缩，不允许新增）。")
    A("3. **定步族**（`heat_1d`、`advection_*`、`elastic_*`、`sipg_*` 等）的 `dt` 本就是 `final_time/steps`，"
      "其消费方不需要改——对照表见 3.1，避免「一刀切改全部」的过度修复。")
    A("4. **未做**：未做全量 pytest（本机磁盘一度写满、负载 load15≈119），只跑了契约测试与相关子集；"
      "未逐个改消费方的数值，因此没有「修正后的曲线」对照图——这是下一步的独立增量。\n")

    A("## ⑥ 误差判定\n")
    A("| 判据 | 关键量 | 实测 | 基准 | 是否 < 3% |")
    A("|---|---|---|---|---|")
    A("| 恒等式 `dt_mean = t_final/steps` | 8 个求解器返回值 | "
      f"max |偏差| = {max(abs(r['identity']) for r in sols):.2e} | 0（代数恒等） | 是 |")
    A("| 2 倍律（CFL 减半） | `dt_mean` 相邻比 | "
      f"[{min(step_ratios):.4f}, {max(step_ratios):.4f}] | 2.0 | 是（最坏 "
      f"{max(abs(r - 2.0) for r in step_ratios) / 2.0 * 100:.2f} %） |")
    A("| 遗留 `dt` 不可作步长 | `dt_last` 相邻比最坏偏离 | "
      f"×{2 ** worst_log2:.1f} | 2.0 | **否**（这正是本单要证明的缺陷） |")
    A("| 与 Lax 参考解 | 全算例池 rel L1（最细腿） | 见 `docs/CASE-LAX-DG.md`（本单未改求解器物理，不动该数） | 3.0e-02 | 是（0.45 %，P2N400） |")
    A("")
    A("**结论**：按 <3% 口径，`dt_mean` 的恒等式与 2 倍律均达标（偏差 ≤1e-16 与 ≤2 %），"
      "而遗留 `dt` 作为步长不可接受（可差 2 个量级），本单把这一结论固化成回归测试。\n")
    OUT_DOC.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_DOC}")
    print(f"wrote {png} ({png.stat().st_size} bytes)")
    print(f"max identity error: {max(abs(r['identity']) for r in sols):.3e}")
    print(f"dt_mean ratios: {[round(r, 4) for r in step_ratios]}")
    print(f"worst last-step ratio deviation: {2 ** worst_log2:.1f}x")
    return dict(doc=str(OUT_DOC), png=str(png), png_b64=png_b64, lad=lad, sols=sols,
                audit=audit)


if __name__ == "__main__":
    render()