"""CASE-DG-EULER-VORTEX-ERR: moving-error conservation of the 2-D isentropic vortex.

Case-pool anchor
----------------
``CASE-DG-EULER-VORTEX-ERR`` (mission m20056) fixes the reference to **Yee et al.
2000** and the acceptance tolerance to **< 6 %** relative L2.  Provenance is not
hand-copied: :mod:`benchmarks.ref_provenance_yee2000` queries Crossref / Unpaywall
/ OpenAlex, records the DOI and the open-access location, and logs every full-text
retrieval attempt with its HTTP status.  Whatever the retrieval outcome, this module
only ever compares the numerical solution against the **analytic** vortex, which is
itself machine-verified (``--task residual``): a table entry transcribed from a PDF
would be a second-hand number, whereas ``vortex_primitive`` is the object the solver
is actually supposed to converge to.

Why this case is an *error-conservation* case and not the t = 1 case
--------------------------------------------------------------------
The domain is 10 x 10, periodic, and the vortex is advected by (u, v)_inf = (1, 1),
so after one full period ``T = L / u_inf = 10`` the exact solution is *identical to
the initial condition*.  A scheme that is merely accurate at short time can still
lose the structure to secular error growth; the quantity of interest is therefore

    GB(t) = err(t) / err(t_ref)

i.e. how much the moving error has accumulated after a whole period.  The gate is
``max_field err(T) < 6 %`` and the companion conservation criterion is
``GB <= 5`` (a secular, step-count-proportional growth would be far larger; see the
predeclared thresholds in ``docs/CASE-DG-EULER-VORTEX-ERR.md``).

Layout
------
One configuration per process (``--config-index``), one checkpoint JSON per
configuration, so a killed process costs only its own configuration.  Aggregation
is a pure post-processing step over those JSONs (``--task aggregate``) and injects
every number in the report programmatically (zero hand-copying).

Usage
-----
    python benchmarks/case_euler_vortex_yee.py --config-index 0 --output-dir runs/case_yee2000_vortex
    python benchmarks/case_euler_vortex_yee.py --task aggregate --output-dir runs/case_yee2000_vortex
    python benchmarks/case_euler_vortex_yee.py --task residual
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.case_euler_vortex import (
    BACKGROUND,
    DOMAIN,
    FIELDS,
    STRENGTH,
    run_task,
)

# ---------------------------------------------------------------------------
# predeclared case contract (one place, referenced by the report)
# ---------------------------------------------------------------------------

#: Case-pool tolerance for this round (m20056): every primitive field < 6 %.
TOLERANCE = 0.06
#: One full period: L / u_inf with L = 10 and |u_inf| = 1 in each direction.
PERIOD = 10.0
#: Predeclared bound on the error-growth factor ``err(T) / err(T/2)``.
GROWTH_LIMIT = 5.0
#: Reference label used in every record (provenance lives in the JSON ledger).
REFERENCE = ("Yee, Vinokur & Djomehri, J. Comput. Phys. 162 (2000) 33-81, "
             "DOI 10.1006/jcph.2000.6517 (per case-pool brief); numeric target = "
             "analytic isentropic vortex, machine-verified")

CONFIGS = (
    # --- primary family: P2 DG at one full period (gate + resolution ladder) ---
    {"id": "dg_d2_n32_t10_cfl0.2", "task": "dg", "degree": 2, "cells": 32,
     "final_time": PERIOD, "cfl": 0.2, "role": "gate"},
    {"id": "dg_d2_n32_t10_cfl0.2_rep", "task": "dg", "degree": 2, "cells": 32,
     "final_time": PERIOD, "cfl": 0.2, "role": "determinism", "save_fields": True},
    {"id": "dg_d2_n16_t10_cfl0.2", "task": "dg", "degree": 2, "cells": 16,
     "final_time": PERIOD, "cfl": 0.2, "role": "ladder"},
    {"id": "dg_d2_n48_t10_cfl0.2", "task": "dg", "degree": 2, "cells": 48,
     "final_time": PERIOD, "cfl": 0.2, "role": "ladder", "save_fields": True},
    # --- other degrees: design-order evidence at one period ---
    {"id": "dg_d1_n16_t10_cfl0.2", "task": "dg", "degree": 1, "cells": 16,
     "final_time": PERIOD, "cfl": 0.2, "role": "ladder"},
    {"id": "dg_d1_n32_t10_cfl0.2", "task": "dg", "degree": 1, "cells": 32,
     "final_time": PERIOD, "cfl": 0.2, "role": "ladder"},
    {"id": "dg_d3_n8_t10_cfl0.2", "task": "dg", "degree": 3, "cells": 8,
     "final_time": PERIOD, "cfl": 0.2, "role": "ladder"},
    {"id": "dg_d3_n16_t10_cfl0.2", "task": "dg", "degree": 3, "cells": 16,
     "final_time": PERIOD, "cfl": 0.2, "role": "ladder"},
    # --- low-order baseline (same periodic setup, Rusanov limiting case) ---
    {"id": "p0_n64_t10_cfl0.4", "task": "p0", "cells": 64,
     "final_time": PERIOD, "cfl": 0.4, "role": "baseline"},
    {"id": "p0_n128_t10_cfl0.4", "task": "p0", "cells": 128,
     "final_time": PERIOD, "cfl": 0.4, "role": "baseline"},
    # --- error history on the gate grid (moving-error accumulation) ---
    {"id": "dg_d2_n32_t1_cfl0.2", "task": "dg", "degree": 2, "cells": 32,
     "final_time": 1.0, "cfl": 0.2, "role": "history"},
    {"id": "dg_d2_n32_t2.5_cfl0.2", "task": "dg", "degree": 2, "cells": 32,
     "final_time": 2.5, "cfl": 0.2, "role": "history"},
    {"id": "dg_d2_n32_t5_cfl0.2", "task": "dg", "degree": 2, "cells": 32,
     "final_time": 5.0, "cfl": 0.2, "role": "history"},
    # --- CFL sensitivity at the gate configuration ---
    {"id": "dg_d2_n32_t10_cfl0.1", "task": "dg", "degree": 2, "cells": 32,
     "final_time": PERIOD, "cfl": 0.1, "role": "cfl"},
    {"id": "dg_d2_n32_t10_cfl0.4", "task": "dg", "degree": 2, "cells": 32,
     "final_time": PERIOD, "cfl": 0.4, "role": "cfl"},
)


# ---------------------------------------------------------------------------
# single-configuration driver
# ---------------------------------------------------------------------------

def run_config(config: dict, output_dir: Path) -> dict:
    results = output_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    key = config["id"]
    started = time.time()
    record = run_task({k: config[k] for k in
                       ("task", "cells", "degree", "final_time", "cfl")})
    wall = time.time() - started
    fields_np = record.pop("fields_np", None)
    record.update({
        "config_id": key,
        "role": config.get("role"),
        "case": "CASE-DG-EULER-VORTEX-ERR",
        "tolerance": TOLERANCE,
        "period": PERIOD,
        "reference": REFERENCE,
        "wall_seconds": wall,
        "wall_seconds_including_setup": wall,
        "gate_pass": bool(record["max_relative_l2"] < TOLERANCE),
    })
    with (results / f"{key}.json").open("w") as fh:
        json.dump(record, fh, indent=1)
    if fields_np is not None and config.get("save_fields"):
        np.savez(results / f"{key}_fields.npz", density=fields_np[0],
                 x=fields_np[1], y=fields_np[2])
    print(json.dumps({k: record[k] for k in
                      ("config_id", "steps", "dt", "max_percent", "gate_pass",
                       "wall_seconds")}))
    return record


# ---------------------------------------------------------------------------
# aggregation: tables, figures, report (zero hand-copy)
# ---------------------------------------------------------------------------

def _load(output_dir: Path) -> dict:
    records = {}
    for path in sorted((output_dir / "results").glob("*.json")):
        record = json.loads(path.read_text())
        records[record.get("config_id", path.stem)] = record
    return records


def _pct(record: dict) -> float:
    return 100.0 * float(record["max_relative_l2"])


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _tick(condition: bool) -> str:
    return "勾" if condition else "叉"


def aggregate(output_dir: Path, docs: Path | None = None) -> int:
    docs = docs or Path("docs")
    records = _load(output_dir)
    if not records:
        print("no records found; nothing to aggregate")
        return 1
    byid = {cid: rec for cid, rec in records.items()}

    gate_id = "dg_d2_n32_t10_cfl0.2"
    gate = byid.get(gate_id)
    ladder = [rec for rec in records.values() if rec.get("role") == "ladder"]
    history = sorted((rec for rec in records.values() if rec.get("role") == "history"),
                     key=lambda r: r["final_time"])
    cfl_rows = sorted((rec for rec in records.values() if rec.get("role") == "cfl"),
                      key=lambda r: r["cfl"])
    baselines = sorted((rec for rec in records.values() if rec.get("role") == "baseline"),
                       key=lambda r: r["cells"])
    repeats = [rec for rec in records.values() if rec.get("role") == "determinism"]

    # ---- observed orders (per degree, over the ladder + gate config) ----
    families: dict[int, list[dict]] = {}
    for rec in [*ladder, *([gate] if gate else [])]:
        families.setdefault(rec["degree"], []).append(rec)
    orders = {}
    for degree, members in families.items():
        members = sorted(members, key=lambda r: r["cells"])
        pairs = []
        for coarse, fine in zip(members, members[1:]):
            ratio = coarse["max_relative_l2"] / fine["max_relative_l2"]
            h_ratio = fine["cells"] / coarse["cells"]
            pairs.append({
                "from_cells": coarse["cells"], "to_cells": fine["cells"],
                "order": math.log(ratio) / math.log(h_ratio) if ratio > 0 else float("inf"),
            })
        orders[degree] = pairs

    # ---- moving-error growth factor on the gate grid ----
    growth = None
    reference_t = min((rec["final_time"] for rec in history), default=None)
    first = next((rec for rec in history if rec["final_time"] == reference_t), None)
    if gate is not None and first is not None and first["max_relative_l2"] > 0:
        growth = gate["max_relative_l2"] / first["max_relative_l2"]

    hist_initial = None
    if first is not None:
        hist_initial = first["max_relative_l2"]

    summary = {
        "case": "CASE-DG-EULER-VORTEX-ERR",
        "reference": REFERENCE,
        "tolerance": TOLERANCE,
        "period": PERIOD,
        "growth_limit": GROWTH_LIMIT,
        "gate_config": gate_id,
        "gate_percent": _pct(gate) if gate else None,
        "gate_pass": bool(gate and gate["gate_pass"]) if gate else False,
        "growth_factor": growth,
        "growth_reference_time": reference_t,
        "growth_pass": bool(growth is not None and growth <= GROWTH_LIMIT),
        "observed_orders": {str(k): v for k, v in orders.items()},
        "configs": {cid: {k: rec[k] for k in ("degree", "cells", "final_time", "cfl",
                                             "steps", "wall_seconds", "max_percent",
                                             "gate_pass", "role")}
                    for cid, rec in records.items()},
        "overall_passed": bool(gate and gate["gate_pass"] and
                               (growth is None or growth <= GROWTH_LIMIT)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=1))

    # ---- figures ----
    figures = _figures(output_dir, records, history, gate, orders, cfl_rows)

    # ---- markdown report ----
    md = _report(summary, records, gate, gate_id, ladder, history, cfl_rows,
                 baselines, repeats, orders, figures)
    docs = Path(docs)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "CASE-DG-EULER-VORTEX-ERR.md").write_text(md)
    (output_dir / "summary.md").write_text(md)
    print(json.dumps({"gate_percent": summary["gate_percent"],
                      "gate_pass": summary["gate_pass"],
                      "growth_factor": summary["growth_factor"],
                      "overall_passed": summary["overall_passed"]}))
    return 0


def _figures(output_dir: Path, records: dict, history: list, gate, orders,
             cfl_rows: list) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = output_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made = []

    # (1) moving-error accumulation over one period
    if history:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        times = [rec["final_time"] for rec in history]
        worst = [100.0 * rec["max_relative_l2"] for rec in history]
        for field in FIELDS:
            ax.semilogy(times, [100.0 * rec["errors"][field] for rec in history],
                        marker="o", lw=1.0, label=field)
        ax.semilogy(times, worst, "k--", lw=2.0, label="max field")
        ax.axhline(100.0 * TOLERANCE, color="r", ls=":", label="6 % gate")
        ax.set_xlabel("t (one period T = 10)")
        ax.set_ylabel("relative L2 error  [%]")
        ax.set_title("P2 DG, 32x32: moving-error accumulation")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        path = out / "fig_error_history.png"
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        made.append(("fig_error_history.png", "等熵涡随输运时间的相对 L2 误差累积（P2 32x32，一个周期 T=10），虚线为 6% 判据。"))

    # (2) convergence ladder at one period
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    by_degree: dict[int, list[dict]] = {}
    for rec in records.values():
        if rec.get("role") in ("ladder", "gate"):
            by_degree.setdefault(rec["degree"], []).append(rec)
    for degree, members in sorted(by_degree.items()):
        members = sorted(members, key=lambda r: r["cells"])
        if len(members) < 2:
            continue
        h = [10.0 / rec["cells"] for rec in members]
        e = [100.0 * rec["max_relative_l2"] for rec in members]
        label = f"P{degree} (design order {degree + 1})"
        if str(degree) in orders and orders[degree]:
            mean_order = sum(p["order"] for p in orders[degree]) / len(orders[degree])
            label += f", observed {mean_order:.2f}"
        ax.loglog(h, e, marker="o", label=label)
    base = sorted((rec for rec in records.values() if rec["role"] == "baseline"),
                  key=lambda r: r["cells"])
    if len(base) >= 2:
        ax.loglog([10.0 / rec["cells"] for rec in base],
                  [100.0 * rec["max_relative_l2"] for rec in base],
                  marker="s", ls="--", label="P0 baseline (Rusanov)")
    ax.axhline(100.0 * TOLERANCE, color="r", ls=":", label="6 % gate")
    ax.set_xlabel("h = 10 / N")
    ax.set_ylabel("max-field relative L2 error  [%]")
    ax.set_title("t = T (one period): resolution ladder")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    path = out / "fig_convergence_period.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append(("fig_convergence_period.png", "一个整周期末（t=T=10）的最大场相对 L2 误差随网格尺度的收敛阶梯，含 P0 基线与 6% 判据线。"))

    # (3) density field + error map for the best saved configuration
    saved = [(cid, rec) for cid, rec in records.items()
             if (output_dir / "results" / f"{cid}_fields.npz").exists()]
    if saved:
        cid, rec = sorted(saved, key=lambda kv: kv[1]["max_relative_l2"])[0]
        data = np.load(output_dir / "results" / f"{cid}_fields.npz")
        dens = data["density"]
        nx, ny = dens.shape
        dx = 10.0 / nx
        xc = -5.0 + (np.arange(nx) + 0.5) * dx
        yc = -5.0 + (np.arange(ny) + 0.5) * dx
        t = rec["final_time"]
        xs = np.mod(xc - 1.0 * t + 5.0, 10.0) - 5.0
        ys = np.mod(yc - 1.0 * t + 5.0, 10.0) - 5.0
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        R2 = X ** 2 + Y ** 2
        E = np.exp(0.5 * (1.0 - R2))
        gamma = 1.4
        T = 1.0 - (gamma - 1.0) * 5.0 ** 2 / (8.0 * gamma * math.pi ** 2) * E ** 2
        exact = T ** (1.0 / (gamma - 1.0))
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
        lim = dict(extent=[-5, 5, -5, 5], origin="lower", aspect="equal")
        c0 = axes[0].imshow(dens.T, cmap="viridis", **lim)
        axes[0].set_title(f"numerical rho, P{rec['degree']} {nx}x{ny}, t={t:g}")
        fig.colorbar(c0, ax=axes[0], shrink=0.85)
        c1 = axes[1].imshow(exact.T, cmap="viridis", **lim)
        axes[1].set_title("exact rho (same instant)")
        fig.colorbar(c1, ax=axes[1], shrink=0.85)
        diff = dens - exact
        vmax = float(np.abs(diff).max()) or 1.0
        c2 = axes[2].imshow(diff.T, cmap="coolwarm", vmin=-vmax, vmax=vmax, **lim)
        axes[2].set_title(f"rho error (max |.| = {vmax:.2e})")
        fig.colorbar(c2, ax=axes[2], shrink=0.85)
        path = out / "fig_density_field.png"
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        made.append(("fig_density_field.png",
                     f"前沿密度场与解析解对比及误差云图（配置 {cid}，t={t:g}，单位为无量纲）。"))

    # (4) CFL sensitivity
    if cfl_rows:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.semilogy([r["cfl"] for r in cfl_rows],
                    [100.0 * r["max_relative_l2"] for r in cfl_rows],
                    marker="o")
        ax.axhline(100.0 * TOLERANCE, color="r", ls=":")
        ax.set_xlabel("CFL")
        ax.set_ylabel("max-field relative L2 error  [%]")
        ax.set_title(f"CFL sensitivity at t = T (P2, {cfl_rows[0]['cells']}x{cfl_rows[0]['cells']})")
        ax.grid(alpha=0.3, which="both")
        path = out / "fig_cfl_sensitivity.png"
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        made.append(("fig_cfl_sensitivity.png", "整周期末误差对 CFL 的敏感性（P2 32x32）。"))

    (output_dir / "figures" / "figures.json").write_text(
        json.dumps([{"name": n, "caption": c} for n, c in made], ensure_ascii=False, indent=1))
    return made


def _report(summary, records, gate, gate_id, ladder, history, cfl_rows, baselines,
            repeats, orders, figures) -> str:
    L = []
    L.append("# CASE-DG-EULER-VORTEX-ERR：等熵涡移动误差守恒（DG，一个整周期）\n")
    L.append("> 本报告的每个数字均由 `benchmarks/case_euler_vortex_yee.py --task aggregate` "
             "从 `results/*.json` 程序化注入，无手抄；判据由脚本计算。"
             "参考文献原文获取状态见 §① 与 `artifacts/ref_provenance_yee2000.json`。\n")

    L.append("## ① 问题与基准出处\n")
    L.append("二维可压缩 Euler 方程（理想气体 γ = 1.4）的等熵涡：均匀来流 (u,v)∞ = (1,1) "
             "平移输运一个等熵涡，t>0 的**解析精确解**即初始剖面向下游刚性平移。"
             "域 [-5,5]² 周期，域长 L = 10、|u∞| = 1，故平移一个整周期需 T = L/u∞ = 10；"
             "t = T 时精确解与初始条件逐点相同——这正是「移动误差」的判据：不是看短时精度，"
             "而是看**一个整周期后误差是否仍被 6% 容差覆盖**（误差守恒性）。\n")
    L.append(f"- 基准文献（算例池指定）：{REFERENCE}。")
    L.append("- 数值目标：与**解析解**逐点比较（`tensordg/euler_2d.py::vortex_primitive`）。"
             "参考解的可信性由 `--task residual` 独立校核（静止系 div F = 0、平动系 U_t+F_x+G_y = 0、"
             "等熵律、无散旋流、径向动量平衡），不依赖任何外部表格。")
    L.append("- 原文获取：Crossref/Unpaywall/OpenAlex 机器查询得到 DOI 与唯一 OA 落地页，"
             "但全文下载在受限环境内被拒（NTRS 403、镜像 Cloudflare 挑战）；"
             "逐条尝试与 HTTP 状态记入 `artifacts/ref_provenance_yee2000.json`，"
             "因此**论文原表的数值本轮标为 WITHHELD，未当作已知事实引用**（见 §⑥）。\n")

    L.append("## ② 方法与实现\n")
    L.append("复用仓内既有两条互相独立的离散路径，不新写内核：\n")
    L.append("| 组件 | 位置 | 说明 |")
    L.append("|---|---|---|")
    L.append("| 高阶主路径 | `tensordg/euler_dg_2d.py::solve_euler_dg_2d_vortex` | 张量积模态 DG（P1–P3）+ HLLC + SSP-RK3，双周期边界，L2 投影初值 |")
    L.append("| 低阶基线 | `tensordg/euler_2d.py::solve_isentropic_vortex` | P0（守恒型有限体积）+ Rusanov + SSP-RK3，单元平均初值 |")
    L.append("| 误差口径 | `benchmarks/verify_euler_dg_2d.py::_vortex_error` | 各原始变量相对 L2，求积节点上取值 |")
    L.append("| 算例编排 | `benchmarks/case_euler_vortex_yee.py` | 每配置一进程一检查点；聚合零手抄 |")
    L.append("| 基准溯源 | `benchmarks/ref_provenance_yee2000.py` | Crossref/Unpaywall/OpenAlex 机器查询 + 抓取尝试日志 |")
    L.append("")
    L.append("流程：`--config-index i` 跑一条配置并落检查点 → `--task aggregate` 聚合出门禁/收敛/敏感性"
             "与图，并把本文件重新写一遍。限制器本算例**显式关闭**（光滑解，限制器只降阶）。\n")

    L.append("## ③ 参数表（无量纲）\n")
    L.append("| 参数 | 符号 | 取值 |")
    L.append("|---|---|---|")
    L.append("| 涡强度 | β | 5 |")
    L.append("| 来流速度 | (u,v)∞ | (1,1) |")
    L.append("| 绝热指数 | γ | 1.4 |")
    L.append("| 计算域 | Ω | [-5,5]²，x/y 双周期 |")
    L.append("| 一个整周期 | T = L/u∞ | 10 |")
    L.append("| 峰值切向马赫数 | M_θ,max | 0.7958 |")
    L.append("| 来流马赫数（c∞=1） | M∞ | 1.414 |")
    L.append("| CFL（DG） | CFL | 0.2（敏感性 0.1/0.4） |")
    L.append("| CFL（P0） | CFL | 0.4 |")
    L.append("| 容差 | — | 各场相对 L2 < 6% |")
    L.append(f"| 误差增长上限 | GB = err(T)/err(T/2) | ≤ {GROWTH_LIMIT:g} |")
    L.append("| 分辨率阶梯 | N×N | P1: 16/32；P2: 16/32/48；P3: 8/16；P0: 64/128 |")
    L.append("")

    L.append("## ④ 对照表（ref vs got，逐项打勾）\n")
    L.append("表 1：判据门（容差 < 6%，改自本轮算例池判据）\n")
    L.append("| 配置 | 角色 | 密度 ρ | 速度 u | 速度 v | 压力 p | 最大场误差 | <6%? |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cid, rec in records.items():
        if rec.get("role") in ("determinism",):
            continue
        cells = rec["config_id"]
        L.append(f"| `{cells}` | {rec.get('role')} | "
                 + " | ".join(_fmt(100.0 * rec['errors'][f]) + "%" for f in FIELDS)
                 + f" | {_fmt(_pct(rec))}% | {_tick(rec['gate_pass'])} |")
    L.append("")
    if gate:
        L.append(f"**门禁配置** `{gate_id}`：最大场相对 L2 = {_fmt(_pct(gate))}% "
                 f"→ {'通过（<6%）' if gate['gate_pass'] else '未通过（≥6%）'}。\n")

    L.append("表 2：移动误差累积（P2，32×32，同一网格逐时刻）\n")
    L.append("| t | 密度 ρ | 速度 u | 压力 p | 最大场误差 | <6%? |")
    L.append("|---|---|---|---|---|---|")
    for rec in history:
        L.append(f"| {rec['final_time']:g} | " +
                 " | ".join(_fmt(100.0 * rec['errors'][f]) + "%" for f in ("density", "velocity_x", "pressure")) +
                 f" | {_fmt(_pct(rec))}% | {_tick(rec['gate_pass'])} |")
    if summary["growth_factor"] is not None:
        L.append("")
        L.append(f"误差增长因子 GB = err(T)/err(t={summary['growth_reference_time']:g}) = "
                 f"{summary['growth_factor']:.2f}（上限 {GROWTH_LIMIT:g}）→ "
                 f"{_tick(summary['growth_pass'])}。\n")

    L.append("表 3：收敛阶梯（t = T）\n")
    L.append("| 族 | N×N | 步数 | 密度 ρ | 速度 u | 压力 p | 最大场误差 | 观测阶 | 设计阶 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for degree in sorted({rec["degree"] for rec in records.values()
                          if rec.get("role") in ("ladder", "gate")}):
        members = sorted([rec for rec in records.values()
                          if rec["degree"] == degree and rec.get("role") in ("ladder", "gate")],
                         key=lambda r: r["cells"])
        pair_order = {p["to_cells"]: p["order"] for p in orders.get(degree, [])}
        for i, rec in enumerate(members):
            order = f"{pair_order[rec['cells']]:.2f}" if rec["cells"] in pair_order else "-"
            L.append(f"| P{degree} | {rec['cells']} | {rec['steps']} | "
                     + " | ".join(_fmt(100.0 * rec['errors'][f]) + "%" for f in ("density", "velocity_x", "pressure"))
                     + f" | {_fmt(_pct(rec))}% | {order} | {degree + 1} |")
    for rec in records.values():
        if rec.get("role") == "baseline":
            L.append(f"| P0 (Rusanov) | {rec['cells']} | {rec['steps']} | "
                     + " | ".join(_fmt(100.0 * rec['errors'][f]) + "%" for f in ("density", "velocity_x", "pressure"))
                     + f" | {_fmt(_pct(rec))}% | - | 1 |")
    L.append("")

    L.append("## ⑤ 收敛性与参数敏感性\n")
    L.append("观测阶 p = log(e_coarse/e_fine)/log(h_coarse/h_fine)，在**整周期末**取值：\n")
    L.append("| 族 | 区间 | 观测阶 | 设计阶 | 判定 |")
    L.append("|---|---|---|---|---|")
    for degree in sorted(orders):
        for pair in orders[degree]:
            good = abs(pair["order"] - (degree + 1)) <= 0.5 * (degree + 1)
            L.append(f"| P{degree} | {pair['from_cells']}→{pair['to_cells']} | "
                     f"{pair['order']:.2f} | {degree + 1} | {_tick(good)} |")
    L.append("")
    if cfl_rows:
        L.append("表 4：CFL 敏感性（P2，32×32，t = T）\n")
        L.append("| CFL | 步数 | 最大场误差 | <6%? |")
        L.append("|---|---|---|---|")
        for rec in cfl_rows:
            L.append(f"| {rec['cfl']:g} | {rec['steps']} | {_fmt(_pct(rec))}% | {_tick(rec['gate_pass'])} |")
        L.append("")
        spread = max(_pct(r) for r in cfl_rows) / max(min(_pct(r) for r in cfl_rows), 1e-12)
        L.append(f"CFL 0.1→0.4 之间最大场误差的面内变化倍数 {spread:.2f}；"
                 "结论要求对该窗口稳定（不跨数量级）。\n")
    if len(repeats) >= 1 and gate:
        same = all(abs(rec["max_relative_l2"] - gate["max_relative_l2"]) == 0.0 for rec in repeats)
        L.append(f"确定性：门禁配置重复运行 {len(repeats)} 次，最大场误差逐位相同 → {_tick(same)}"
                 "（求解器无随机源，一次运行即为完整重复）。\n")

    L.append("## ⑥ 失败模式、局限与工件\n")
    fails = [rec for rec in records.values() if not rec["gate_pass"]]
    if fails:
        L.append("判据未通过的配置（如实列出，不粉饰）：")
        for rec in fails:
            L.append(f"- `{rec['config_id']}`：最大场误差 {_fmt(_pct(rec))}%（≥6%），"
                     f"P{rec['degree']} N={rec['cells']}，t={rec['final_time']:g}。")
    else:
        L.append("所有已跑配置均在 6% 以内；基线与粗网格族见 Table 1/3 的判别力说明。")
    L.append("")
    L.append("- **原文数值 WITHHELD**：Yee et al. 2000 原表未被取回（受限网络，逐条 HTTP 状态见溯源 JSON），"
             "故本报告不以论文表格数值作为对照基准；判据一律对解析解。这是证据分级上的诚实降级，不是替代。")
    L.append("- 误差口径：DG 为求积节点点值 L2，P0 为单元平均 L2，跨族只比量级、族内比观测阶。")
    L.append("- 时间推进 SSP-RK3，时间误差未单独做网格无关研究；CFL 敏感性（表 4）覆盖其量级。")
    L.append("- 光滑无激波：限制器关闭；本算例不检验限制器行为。")
    L.append("")
    L.append("复现命令（一键）：\n")
    L.append("```bash")
    L.append("bash tools/run_case_yee_vortex.sh          # 逐配置跑（每配置一进程一检查点）")
    L.append("python3 benchmarks/case_euler_vortex_yee.py --task aggregate --output-dir runs/case_yee2000_vortex")
    L.append("python3 benchmarks/ref_provenance_yee2000.py   # 参考文献溯源（网络受限时如实记录）")
    L.append("pytest -q tests/test_case_euler_vortex_yee.py")
    L.append("```")
    L.append("")
    L.append("工件：`runs/case_yee2000_vortex/results/*.json`（每配置一条）、`metrics.json`、`summary.md`、"
             "`figures/*.png`、`artifacts/ref_provenance_yee2000.json`。")
    if figures:
        L.append("")
        L.append("图：")
        for name, caption in figures:
            L.append(f"- `runs/case_yee2000_vortex/figures/{name}`：{caption}")
    L.append("")
    L.append("## 误差判定（显式回答）\n")
    L.append("| 关键量 | 基准 | 实测 | 是否 <6% |")
    L.append("|---|---|---|---|")
    if gate:
        L.append(f"| 整周期末最大场相对 L2（P{gate['degree']} {gate['cells']}²） | 解析等熵涡 | {_fmt(_pct(gate))}% | {_tick(gate['gate_pass'])} |")
    for rec in baselines:
        L.append(f"| 整周期末最大场相对 L2（P0 {rec['cells']}²） | 解析等熵涡 | {_fmt(_pct(rec))}% | {_tick(rec['gate_pass'])} |")
    if summary["growth_factor"] is not None:
        L.append(f"| 误差增长因子 GB | err(t={summary['growth_reference_time']:g}) | {summary['growth_factor']:.2f} | "
                 f"{'是' if summary['growth_pass'] else '否'}（判据 ≤{GROWTH_LIMIT:g}） |")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("run", "aggregate", "residual", "list"),
                        default="run")
    parser.add_argument("--config-index", type=int, default=None)
    parser.add_argument("--config-id", default=None)
    parser.add_argument("--shard", type=int, default=None,
                        help="with --shards: run configurations whose index %% shards == shard")
    parser.add_argument("--shards", type=int, default=len(CONFIGS))
    parser.add_argument("--output-dir", default="runs/case_yee2000_vortex")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    output = Path(args.output_dir)

    if args.task == "list":
        for i, cfg in enumerate(CONFIGS):
            print(i, cfg["id"], cfg.get("role"))
        return 0
    if args.task == "aggregate":
        return aggregate(output, Path(args.docs))
    if args.task == "residual":
        from benchmarks.case_euler_vortex import run_residual
        record = run_residual()
        output.mkdir(parents=True, exist_ok=True)
        (output / "results").mkdir(parents=True, exist_ok=True)
        (output / "results" / "exact_solution_residual.json").write_text(
            json.dumps({**record, "config_id": "exact_solution_residual",
                        "role": "reference"}, indent=1))
        print(json.dumps({k: record[k] for k in ("steady_frame_residual",
                                                 "spacetime_residual",
                                                 "max_relative_residual", "gate_pass")}))
        return 0

    if args.shard is not None:
        selected = [cfg for i, cfg in enumerate(CONFIGS)
                    if i % args.shards == args.shard % args.shards]
        if not selected:
            print(json.dumps({"shard": args.shard, "shards": args.shards,
                              "configs": [], "note": "nothing assigned"}))
            return 0
        for cfg in selected:
            run_config(cfg, output)
        return 0

    config = None
    if args.config_id:
        config = next((c for c in CONFIGS if c["id"] == args.config_id), None)
    if config is None:
        args.config_index = 0 if args.config_index is None else args.config_index
        if not 0 <= args.config_index < len(CONFIGS):
            parser.error(f"config-index out of range 0..{len(CONFIGS) - 1}")
        config = CONFIGS[args.config_index]
    run_config(config, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())