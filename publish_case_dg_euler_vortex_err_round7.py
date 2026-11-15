#!/usr/bin/env python3
"""Publish CASE-DG-EULER-VORTEX-ERR round-7 report (degree axis + design order).

Round 7 closes the scope debt the round-6 report declared explicitly: every
conservation statement of rounds 5/6 was certified at one polynomial degree
(p = 2) with the mesh and the CFL as the only design axes, and the diagnosis of
"the error is a grid property, not a time property" was never tested against a
*measured* convergence order at long time.  This round adds

* C8 -- degree invariance: p = 1 and p = 3 families get the same alpha <= 1 bound
  as the design degree, and at least two degrees must be decidable;
* C9 -- design order: over n = 32/64 the observed spatial order must reach
  0.8 * (p + 1) for p = 1 and p = 2 at both t = 1 and t = 10;
* a double box crossing (t = 20) on the design family, so C2's fit window is not
  pinned to a single crossing.

Every number is read from ``case_euler_vortex_errconserve/metrics.json`` --
no hand-copied digits.  Figures are generated from the same records.

Usage::

    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_round7.py [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")
OUT = Path("case_euler_vortex_errconserve")
DOC = Path("docs/CASE-DG-EULER-VORTEX-ERR-round7.md")
MISSION = 21419


def login() -> str:
    req = urllib.request.Request(BASE + "/auth/login",
                                 data=json.dumps({"username": USER,
                                                  "password": PASS}).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
        return body.get("token") or body["access_token"]


def post(path, payload, token):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def _png(name: str) -> dict:
    return {"name": name, "png_base64": base64.b64encode(
        (OUT / name).read_bytes()).decode()}


def figure_degree_axis(metrics: dict) -> None:
    """e(t) per degree family (left) and the measured order (right)."""
    records = metrics["records"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    groups = {}
    for rec in records:
        cfg = rec["config"]
        if cfg["task"] != "dg" or abs(cfg["cfl"] - 0.2) > 1e-12:
            continue
        groups.setdefault((cfg["degree"], cfg["cells"]), []).append(rec)
    for (degree, cells), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["final_time"])
        t = [max(r["final_time"], 1e-3) for r in rows]
        e = [r["max_percent"] for r in rows]
        axes[0].loglog(t, e, "o-", label=f"p={degree}, n={cells}")
    axes[0].axhline(6.0, color="crimson", ls="--", lw=1)
    axes[0].text(1.05, 6.6, "case-pool tolerance 6%", color="crimson", fontsize=8)
    axes[0].set_xlabel("transport time t (box crossing = 10)")
    axes[0].set_ylabel("max field relative L2 [%]")
    axes[0].set_title("(a) error history per degree (CFL 0.2)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, which="both", alpha=0.3)

    per_case = metrics["gates"]["C9_design_order"]["per_case"]
    labels, orders, targets = [], [], []
    for key, row in per_case.items():
        labels.append(key.replace("dg_", ""))
        orders.append(row["order"] if row["order"] == row["order"] else 0.0)
        targets.append(row["target"])
    x = range(len(labels))
    axes[1].bar([i - 0.2 for i in x], orders, 0.4, label="observed order")
    axes[1].bar([i + 0.2 for i in x], targets, 0.4, label="gate 0.8*(p+1)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].axhline(1.0, color="grey", lw=0.8)
    axes[1].set_title("(b) measured spatial order (C9)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_degree_axis.png", dpi=150)
    plt.close(fig)


def build_markdown(metrics: dict) -> tuple[str, list]:
    gates = metrics["gates"]
    ok = lambda name: "通过" if gates[name]["passed"] else "未通过"
    c8 = gates["C8_degree_invariance"]
    c9 = gates["C9_design_order"]
    c2 = gates["C2_error_conservation"]
    families = {}
    for rec in metrics["records"]:
        cfg = rec["config"]
        key = (cfg["task"], cfg.get("degree"), cfg["cells"], cfg["cfl"])
        families.setdefault(key, []).append(rec)

    lines = [
        "# CASE-DG-EULER-VORTEX-ERR 第 7 轮：阶数轴与设计阶（误差守恒的范围闭合）",
        "",
        f"判定：**{gates['verdict']}**（不可判定门禁：{gates['not_evaluated'] or '无'}；"
        f"失败门禁：{gates['failed'] or '无'}）",
        "",
        "## ① 案例介绍",
        "",
        "二维等熵涡在周期箱 [-5,5]² 内以 u_inf=(1,1) 匀速平移（β=5, γ=1.4）。精确解在"
        "平移下形状不变，因此**误差应当是网格属性而非时间属性**——这正是算例池条目 "
        "`CASE-DG-EULER-VORTEX-ERR`（ref: Yee et al. 2000，容差 <6%）要判定的东西。"
        "一轮过箱时间 L/|u_inf| = 10。",
        "",
        "| 参数 | 值 |",
        "|---|---|",
        f"| 域 | {metrics['domain']} |",
        f"| 涡强 β | {metrics['strength']} |",
        f"| 背景态 | {metrics['background']} |",
        f"| 容差（快照） | {metrics['tolerance_relative_l2']:.0%} |",
        "| 离散 | 模态 DG（HLLC + SSP-RK3，L2 投影初值）与 P0 有限体积（Rusanov） |",
        "",
        "## ② 怎么做的（本轮增量）",
        "",
        "前三轮的设计空间只动**网格轴**与 **CFL 轴**，多项式阶数固定 p=2，所以"
        "「误差守恒」严格说只是关于 p=2 的陈述；且「误差是网格属性」从未用**实测收敛阶**"
        "验证过。本轮：",
        "",
        "1. **C8 阶数不变性**：新增 p=1、p=3 两个族（n=32, CFL 0.2，t=1/2/5/10），"
        "与设计阶 p=2 用同一条上界 alpha ≤ 1；并要求至少 2 个阶数可判。",
        "2. **C9 设计阶**：在 t=1 与 t=10（一次过箱）两个检查点上，用 n=32/64 的观测空间阶"
        "检验是否达到 0.8×(p+1)（p=1 → ≥1.6，p=2 → ≥2.4）。0.8 是 C5 已在用的同一约定，"
        "不是对着结果调出来的数。",
        "3. **双过箱**：设计族新增 t=20 检查点，C2 的拟合窗口不再被钉在单次过箱上。",
        "4. p=1 的 n=64 族给满 4 个检查点，使其同时是 C7 的设计点（两点族会让 C7 在细网格上失明）。",
        "",
        "全部检查点经计算递交门（作业 cj3432，13 分片）落盘；本文数字由 "
        "`benchmarks/case_euler_vortex_errconserve.py --aggregate` 从 `results/*.json` "
        "程序化注入，零手抄。",
        "",
        "## ③ 计算结果",
        "",
        "| 门禁 | 内容 | 实测 | 判定 |",
        "|---|---|---|---|",
        f"| C1 快照容差 | 各族 t=1 最大场相对 L2 < {metrics['tolerance_relative_l2']:.0%} | "
        f"{gates['C1_tolerance']['measured']:.4%}（{gates['C1_tolerance']['worst_at']}） | {ok('C1_tolerance')} |",
        f"| C2 误差守恒 | 设计族 alpha ≤ 1（{c2['checkpoint_count']} 个检查点，t≤{c2['growth_ratio_window'][1]:g}） | "
        f"alpha {c2['exponent']:.3f}，e(t_last)/e(1) = {c2['growth_ratio']:.3f} | {ok('C2_error_conservation')} |",
        f"| C3 离散守恒 | 质量/动量/能量漂移 < 1e-10 | {gates['C3_discrete_conservation']['worst_drift']:.3e} | {ok('C3_discrete_conservation')} |",
        f"| C4 熵预算 | \\|dS\\|/S(0) < 1e-3 | {gates['C4_entropy_budget']['measured']:.3e} | {ok('C4_entropy_budget')} |",
        f"| C5 交叉校核阶 | P0 观测阶 ≥ 0.8 | {gates['C5_crosscheck_order']['order']:.3f} | {ok('C5_crosscheck_order')} |",
        f"| C6 阶梯完整 | 声明的 {gates['C6_ladder_completeness']['declared']} 个检查点全部落盘 | "
        f"已有 {gates['C6_ladder_completeness']['present']}，缺 {gates['C6_ladder_completeness']['missing'] or '无'} | {ok('C6_ladder_completeness')} |",
        f"| C7 设计空间不变 | 每个 ≥3 检查点的 DG 设计点 alpha ≤ 1 | 可判 {len(gates['C7_design_space_invariance']['evaluated_families'])} 族，"
        f"最大 alpha {gates['C7_design_space_invariance']['measured']:.3f} | {ok('C7_design_space_invariance')} |",
        f"| C8 阶数不变 | {list(metrics['degree_axis'])} 中每个 ≥3 检查点的阶数 alpha ≤ 1，≥2 阶数可判 | "
        f"可判 {c8['degrees_evaluated']}，最大 alpha {c8['measured']:.3f}（p={c8['worst_at']}） | {ok('C8_degree_invariance')} |",
        f"| C9 设计阶 | n=32/64 观测阶 ≥ 0.8(p+1)（p=1,2；t=1,10） | 最差 {c9['measured']:.3f}（{c9['worst_at']}） | {ok('C9_design_order')} |",
        "",
        "**阶数轴（C8）**",
        "",
        "| 阶数 | 族 | 检查点数 | t_max | alpha | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for degree, row in c8["per_degree"].items():
        lines.append(f"| p={degree} | {row['family']} | {row['checkpoints']} | "
                     f"{row['final_time_max'] if row['final_time_max'] is not None else '-'} | "
                     f"{row['exponent']:.3f} | {'通过' if row['passed'] else '未通过'} |")
    lines += ["", "**设计阶（C9）**", "",
              "| 判据点 | 阶数 | 网格 | 观测阶 | 目标 | 判定 |", "|---|---|---|---|---|---|"]
    for key, row in c9["per_case"].items():
        lines.append(f"| {key} | p={row['degree']} | {row['cells']} | {row['order']:.3f} | "
                     f"≥ {row['target']:.1f} | {'通过' if row['passed'] else '未通过'} |")
    lines += ["", "**逐族误差（程序化）**", "",
              "| 族 | t | 步数 | rho % | u % | p % | 最大场 % |", "|---|---|---|---|---|---|---|"]
    for (task, degree, cells, cfl), rows in sorted(
            families.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2], kv[0][3])):
        for rec in sorted(rows, key=lambda r: r["final_time"]):
            e = rec["errors"]
            name = (f"dg_d{degree}_n{cells}_cfl{cfl:g}" if task == "dg"
                    else f"p0_n{cells}_cfl{cfl:g}")
            lines.append(f"| {name} | {rec['final_time']:g} | {rec['steps']} | "
                         f"{100*e['density']:.4f} | {100*e['velocity_x']:.4f} | "
                         f"{100*e['pressure']:.4f} | {rec['max_percent']:.4f} |")
    return "\n".join(lines), [("fig_degree_axis.png", "阶数轴误差史与实测空间阶（C9）")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    metrics = json.loads((OUT / "metrics.json").read_text())
    metrics.setdefault("domain", "[-5,5]^2")
    metrics.setdefault("strength", 5.0)
    metrics.setdefault("background", [1.0, 1.0])
    import benchmarks.case_euler_vortex_errconserve as ec
    metrics["degree_axis"] = list(ec.DEGREE_AXIS)
    figure_degree_axis(metrics)
    markdown, figures = build_markdown(metrics)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(markdown)
    print(markdown[:1500])
    if args.dry_run:
        return 0
    token = login()
    payload = {
        "title": "CASE-DG-EULER-VORTEX-ERR 第 7 轮：阶数轴（p=1/3）与设计阶实测闭合",
        "summary": (f"等熵涡移动误差守恒：新增阶数轴（p=1,3）与双过箱检查点，"
                    f"C8 阶数不变 {'通过' if metrics['gates']['C8_degree_invariance']['passed'] else '未通过'}，"
                    f"C9 设计阶 {'通过' if metrics['gates']['C9_design_order']['passed'] else '未通过'}，"
                    f"总判定 {metrics['gates']['verdict']}"),
        "scenario_key": "dg-euler-vortex-err",
        "mission_id": MISSION,
        "tags": ["TensorDG", "Euler-vortex", "error-conservation", "degree-axis",
                 "design-order", "verification"],
        "markdown": markdown,
        "figures": [{"name": name, "caption": cap,
                     "png_base64": base64.b64encode(
                         (OUT / name).read_bytes()).decode()}
                    for name, cap in figures],
    }
    response = post("/tensor/reports", payload, token)
    print("report:", json.dumps(response)[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())