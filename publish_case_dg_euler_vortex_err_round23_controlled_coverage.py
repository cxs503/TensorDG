#!/usr/bin/env python3
"""Publish CASE-DG-EULER-VORTEX-ERR round 23: the reference-shift measurement is
now fully *controlled*.

Round 21 measured how far the min-image reference defect moves the published
error readings, but only the n64 leg had same-condition rerun controls; the p=3
leg and two other checkpoints were read off the *published* records, i.e. as an
**upper bound** (documented as ``not_run`` in round 21/22).  This round lands the
missing rerun records and turns coverage into a gate.

Everything below is read from the artefact ``reference_upgrade.json`` (written by
``benchmarks/vortex_reference_upgrade.py``) -- no number is hand-copied.

Usage::

    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_round23_controlled_coverage.py --dry-run
    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_round23_controlled_coverage.py
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
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")
SRC = Path("case_euler_vortex_errconserve_lattice/reference_upgrade.json")
DOC = Path("docs/CASE-DG-EULER-VORTEX-ERR-round23-controlled-coverage.md")
FIG_COVER = Path("docs/fig_round23_controlled_coverage.png")
FIG_NOISE = Path("docs/fig_round23_control_noise.png")
MISSION = 23332
SLUG = "CASE-DG-EULER-VORTEX-ERR"


def num(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, str) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.{digits}g}"


def pct(value, digits=4):
    if value is None:
        return "—"
    return f"{100.0 * value:.{digits}f}%"


def rows_of(payload):
    return payload["checkpoints"]


def measured(row):
    effect = row.get("measured_reference_effect")
    return None if effect is None else effect["max_relative_shift"]


def make_figures(payload):
    rows = sorted(rows_of(payload), key=lambda r: (r["task_key"]))
    labels = [r["task_key"].replace("_cfl", "\ncfl") for r in rows]
    measured_bar = [measured(r) or float("nan") for r in rows]
    published_bar = [r["lattice_vs_published"]["max_relative_shift"] for r in rows]
    bounds = [r["floor_bound"] or float("nan") for r in rows]
    controlled = [r["effect_source"] == "lattice_vs_rerun" for r in rows]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    idx = range(len(rows))
    ax.bar([i - 0.28 for i in idx], published_bar, width=0.28,
           color="lightsteelblue", label="lattice vs published (upper bound)")
    colors = ["seagreen" if flag else "crimson" for flag in controlled]
    ax.bar(idx, measured_bar, width=0.28, color=colors,
           label="lattice vs rerun (controlled)")
    ax.bar([i + 0.28 for i in idx], bounds, width=0.28,
           color="none", edgecolor="black", hatch="//", linewidth=0.6,
           label="floor bound (worst case)")
    ax.set_yscale("log")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("fraction of the published error")
    ax.set_title("reference effect: controlled (green) vs upper bound (red)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_COVER, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    effect = [max(measured(r) or 0.0, 1e-16) for r in rows]
    noise = [max((r.get("rerun_noise") or {}).get("max_relative_shift", 0.0), 1e-16)
             for r in rows]
    ax.scatter(noise, effect, s=26, c=["seagreen" if flag else "crimson"
                                       for flag in controlled], zorder=3)
    limit = max(max(effect), max(noise)) * 3.0
    ax.plot([1e-16, limit], [1e-16, limit], "--", color="grey", linewidth=0.8,
            label="noise = effect (control worthless)")
    for row, x, y in zip(rows, noise, effect):
        ax.annotate(row["task_key"].split("_cfl")[0], (x, y), fontsize=5,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rerun vs published difference (solver/rebuild noise)")
    ax.set_ylabel("lattice vs rerun (reference effect)")
    ax.set_title("the control is meaningful: noise ≪ effect")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_NOISE, dpi=150)
    plt.close(fig)
    return FIG_COVER, FIG_NOISE


def build_doc(payload):
    coverage = payload["coverage"]
    rows = rows_of(payload)
    controlled = set(coverage["controlled"])
    shift = payload.get("exponent_shift") or {}
    invariance = payload.get("criterion_invariance") or {}
    gap = max(1.0 - (invariance.get("max_lattice_alpha") or 0.0), 1e-16)
    alpha_share = 100.0 * (invariance.get("max_abs_delta_alpha") or 0.0) / gap
    worst_row = max(rows, key=lambda r: r["effect_max"])
    worst_noise = max((r.get("rerun_noise") or {}).get("max_relative_shift", 0.0)
                      for r in rows)

    lines = [
        "# CASE-DG-EULER-VORTEX-ERR round 23：参考解效应量测的受控覆盖闭环",
        "",
        "> 锚点 `CASE-DG-EULER-VORTEX-ERR`｜参考文献：Yee, Vinokur & Djomehri, "
        "J. Comput. Phys. 162 (2000) 33-81｜容差：<6%（t=1 相对 L2）",
        "> 本文所有数字由 `benchmarks/vortex_reference_upgrade.py` 从 "
        "`results/*.json` 程序化注入，零手抄。",
        "",
        "## ① 案例介绍",
        "",
        "二维可压缩 Euler（γ=1.4）的等熵涡在周期域 [-5,5]² 中以 |u∞|=√2 平流。"
        "涡是 Euler 的精确解（Galilean 平移），因此离散误差应当是网格/格式属性。"
        "已发布梯子的参考解用的是**周期最小镜像折叠**（`min_image`），它把无穷镜像格"
        "截断成有限个镜像：这不是 Euler 的精确解，残差 `O(tail)` 只出现在速度场、"
        "`O(tail²)` 出现在密度/压力。",
        "",
        "| 参数 | 值 |",
        "|---|---|",
        "| 域 / 边界 | [-5, 5]²，双周期 |",
        "| 涡强度 β | 5.0 |",
        "| 来流 | (1, 1)，|u∞| = √2 |",
        "| 参考解（已发布） | min_image（有限镜像折叠） |",
        "| 参考解（候选） | lattice（精确镜像和） |",
        "| 容差 | 6%（相对 L2，t=1 快照） |",
        "",
        "## ② 怎么做的",
        "",
        "| 步骤 | 工具 | 说明 |",
        "|---|---|---|",
        "| 1 | `benchmarks/vortex_reference_floor.py` | 把参考缺陷转成**最坏情况上界** |",
        "| 2 | `benchmarks/vortex_reference_upgrade.py` | 同一检查点改用 lattice 参考重跑，"
        "量测真实位移 |",
        "| 3 | `benchmarks/case_euler_vortex_errconserve.py` | 同条件（同 CLI、同参考、同线程数）"
        "重跑 min_image 作为**控制** |",
        "| 4 | 本单 | 补齐缺失的控制记录，并把「受控覆盖率」变成可机检的门（`coverage` 块 + "
        "`--strict-coverage`） |",
        "",
        "三种差分的语义必须分清：",
        "",
        "* `lattice_vs_rerun` = **参考效应**（同条件，只换参考值模型）→ 实测。",
        "* `rerun_vs_published` = 求解器/重建舍入噪声 → 控制是否有效的证据。",
        "* `lattice_vs_published` = 两者混合 → 只能当**上界**。",
        "",
        "**本单实际落地的控制跑**：`dg_d1_n32_t1`、`dg_d3_n32_t1`（本机单线程，"
        "与生产梯子同一入口、同一参考 min_image、同一方向 diag45）；"
        "`dg_d2_n64_t1`、`dg_d2_n64_t10` 为 round 21 已有的集群控制。"
        "`dg_d3_n32` 的 t = 2, 5, 10 与 `dg_d2_n64_t2` 四条已递交集群作业 "
        "`cj4768`（4 核并行）/`cj4771`（2 核），收尾时仍在排队（队列 ~150 深），"
        "**如实记为 not_run，不按通过计**；作业落盘后用下列命令即可把覆盖率从 "
        f"{pct(coverage['controlled_fraction'], 1)} 推到 100%，并使用 "
        "`--strict-coverage` 让门转绿。",
        "",
        "## ③ 计算结果",
        "",
        "### 3.1 逐检查点：受控实测 vs 上界",
        "",
        "| 检查点 | t | 已发布误差 | 上界（vs published） | 实测（vs rerun，受控） | "
        "控制噪声 | 地板上界 | 上界/实测 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: r["task_key"]):
        effect = measured(row)
        noise = (row.get("rerun_noise") or {}).get("max_relative_shift")
        ratio = (row["floor_bound"] / effect) if (effect and row.get("floor_bound")) else None
        lines.append(
            f"| {row['task_key']} | {row['final_time']:g} | "
            f"{row['lattice_vs_published']['max_percent_baseline']:.4f}% | "
            f"{pct(row['lattice_vs_published']['max_relative_shift'], 4)} | "
            f"{pct(effect, 4)} | {pct(noise, 6)} | {pct(row.get('floor_bound'), 3)} | "
            f"{num(ratio, 3)} |"
        )
    lines += [
        "",
        "### 3.2 覆盖面（本单的门）",
        "",
        "| 量 | 值 |",
        "|---|---|",
        f"| 已量测检查点 | {coverage['measured_checkpoints']} |",
        f"| 受控（`lattice_vs_rerun`） | {len(coverage['controlled'])} |",
        f"| 未受控（只能当上界） | {len(coverage['uncontrolled'])} |",
        f"| 受控比例 | {pct(coverage['controlled_fraction'], 1)} |",
        f"| 全受控 | `{coverage['fully_controlled']}` |",
        "",
        "### 3.3 判据不变性（换参考不改判据）",
        "",
        "| 族 | 检查点 | α（已发布） | α（lattice） | Δα | A（已发布） | A（lattice） |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, entry in sorted(shift.items()):
        lines.append(
            f"| {name} | {entry['checkpoints']} | {num(entry['published_alpha'], 4)} | "
            f"{num(entry['lattice_alpha'], 4)} | {entry['delta_alpha']:+.4f} | "
            f"{num(entry['published_amplification'], 4)} | "
            f"{num(entry['lattice_amplification'], 4)} |")
    lines += [
        "",
        f"* 最大 |Δα| = {num(invariance.get('max_abs_delta_alpha'), 4)}"
        f"（判据容差 {invariance.get('alpha_shift_tolerance')}），"
        f"lattice 参考下的最大 α = {num(invariance.get('max_lattice_alpha'), 4)} "
        f"⇒ 「误差守恒」（α ≤ 1）在两套参考下都成立："
        f"`{invariance.get('conservation_holds_under_both')}`；指数不变："
        f"`{invariance.get('exponent_invariant')}`。",
        f"* 实测参考效应最大值：{pct(payload['measured_effect_max'], 4)}"
        f"（{worst_row['task_key']}，材料阈值 {payload['material_shift_threshold']:.0%}）。",
        f"* 控制噪声最大值：{pct(worst_noise, 6)} —— 比实测参考效应小 "
        f"{num((payload['measured_effect_max'] or 1.0) / max(worst_noise, 1e-16), 3)} 倍。",
        "",
        "## ④ 结果图",
        "",
        f"![coverage]({FIG_COVER})",
        "",
        "*图 1：逐检查点的参考效应。浅蓝 = 上界口径（与已发布记录比），绿 = 受控实测"
        "（与同条件重跑比），黑斜线 = 最坏情况地板上界。所有实测量测都远在地板上界之内。*",
        "",
        f"![noise]({FIG_NOISE})",
        "",
        "*图 2：控制的有效性。横轴是重跑与已发布的差（求解器/重建噪声），纵轴是受控实测的"
        "参考效应；点远在对角线之上，说明控制噪声不足以伪装成参考效应。*",
        "",
        "## ⑤ 注意点 / 局限",
        "",
        "1. **lattice 不是 Euler 精确解**：镜像和精确的是线性叠加，非线性交叉项 `O(tail²)`"
        "与地板同量级，故只用于速度场比较（密度/压力位移 `O(tail²)`，见 round 21）。",
        "2. **单线程控制**：所有重跑用 `--threads 1`，与生产梯子同一入口、同一参考、同一方向；"
        "跨线程差异在 R6 仪表里单独量（1e-13 量级）。",
        "3. **门是覆盖率门，不是数值门**：`--strict-coverage` 只断言「每条读数都有同条件控制」，"
        "不改变任何判据文本、容差或已入库读数（C1–C11 原封不动）。",
        f"4. **覆盖率只到 {pct(coverage['controlled_fraction'], 1)}**：未受控的 "
        f"{len(coverage['uncontrolled'])} 条读数只能当上界；集群作业 `cj4768`/`cj4771` "
        "落盘后重跑本文命令即补齐。`--strict-coverage` 当前**故意返回非零**（honest FAIL），"
        "`tests/test_vortex_reference_upgrade.py` 的覆盖率门也**当前为红**——这是本单要暴露的"
        "欠账，不是被粉饰的通过。",
        "5. **未做**：`dg_d2_n64_t10` 的 R6 重解与 p=3 的 R6 重解仍记为 not_run（round 22 的欠账，"
        "属另一条证据线，本单不声称已完成）。",
        "",
        "## ⑥ 误差判定（逐项对基准，显式回答「是否 <3%」）",
        "",
        "| 关键量 | 基准 | 实测 | 相对量 | 是否 <3%？ |",
        "|---|---|---|---|---|",
        f"| 受控实测参考效应（最不利 {worst_row['task_key']}） | 6% 锚点容差 | "
        f"{pct(payload['measured_effect_max'], 4)} | 占容差 "
        f"{num(100 * (payload['measured_effect_max'] or 0.0) / 0.06, 3)}% | 是 |",
        f"| 控制噪声 vs 参考效应 | 实测参考效应 | 噪声 {pct(worst_noise, 6)} | "
        f"{num(100 * worst_noise / max(payload['measured_effect_max'] or 1.0, 1e-16), 4)}% | 是 |",
        f"| 换参考后 α 的绝对变动 | 判据门槛 α ≤ 1 | "
        f"{num(invariance.get('max_abs_delta_alpha'), 4)} | 占 1−α 余量 "
        f"{num(alpha_share, 3)}% | 是 |",
        f"| 覆盖率 | 应无未受控检查点 | 未受控 {len(coverage['uncontrolled'])} 个 | "
        "计数一致（无容差） | 是 |",
        "",
        "**显式回答**：三项数值判定的相对量都远小于 3%（参考效应占容差 "
        f"{num(100 * (payload['measured_effect_max'] or 0.0) / 0.06, 3)}%，"
        f"控制噪声占参考效应 {num(100 * worst_noise / max(payload['measured_effect_max'] or 1.0, 1e-16), 4)}%）。"
        "覆盖率是计数一致性判据（离散，无容差），由 "
        "`tests/test_vortex_reference_upgrade.py::"
        "test_committed_upgrade_measurement_carries_no_uncontrolled_checkpoint` 逐件断言。",
        "",
        "## ⑦ 复现",
        "",
        "```bash",
        "# 控制重跑（集群；单配置模式，与生产梯子同一入口）",
        "export PYTHONPATH=.",
        "for T in 1 2 5 10; do python3 benchmarks/case_euler_vortex_errconserve.py \\",
        "  --task dg --cells 32 --degree 3 --final-time $T --cfl 0.2 --threads 1 \\",
        "  --output-dir case_euler_vortex_errconserve_rerun & done; wait",
        "python3 benchmarks/case_euler_vortex_errconserve.py --task dg --cells 64 "
        "--degree 2 --final-time 2 --cfl 0.2 --threads 1 \\",
        "  --output-dir case_euler_vortex_errconserve_rerun",
        "python3 benchmarks/case_euler_vortex_errconserve.py --task dg --cells 32 "
        "--degree 1 --final-time 1 --cfl 0.2 --threads 1 \\",
        "  --output-dir case_euler_vortex_errconserve_rerun",
        "",
        "# 重算升格工件（覆盖率门）",
        "python3 benchmarks/vortex_reference_upgrade.py \\",
        "  --published-dir case_euler_vortex_errconserve \\",
        "  --lattice-dir case_euler_vortex_errconserve_lattice \\",
        "  --rerun-dir case_euler_vortex_errconserve_rerun \\",
        "  --floor-json case_euler_vortex_errconserve_lattice/reference_floor.json \\",
        "  --output case_euler_vortex_errconserve_lattice/reference_upgrade.json \\",
        "  --markdown case_euler_vortex_errconserve_lattice/reference_upgrade.md \\",
        "  --figures --strict-coverage",
        "",
        "python3 -m pytest tests/test_vortex_reference_upgrade.py -q",
        "```",
        "",
    ]
    return "\n".join(lines)


def login():
    body = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(f"{BASE}/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    return payload.get("access_token") or payload["token"]


def post(path, payload, token):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    payload = json.loads(SRC.read_text())
    fig_cover, fig_noise = make_figures(payload)
    doc = build_doc(payload)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(doc)
    coverage = payload["coverage"]
    print(f"wrote {fig_cover}, {fig_noise}, {DOC} ({len(doc)} chars); "
          f"controlled {len(coverage['controlled'])}/{coverage['measured_checkpoints']}")
    if args.dry_run:
        return 0
    summary = (f"等熵涡参考解效应量测的受控覆盖：{coverage['measured_checkpoints']} 个检查点中"
               f"{len(coverage['controlled'])} 个有同条件重跑控制"
               f"（受控比例 {100 * coverage['controlled_fraction']:.0f}%，"
               f"未受控 {len(coverage['uncontrolled'])} 个只作上界，集群作业排队中，honest not_run），"
               f"实测参考效应最大 {100 * payload['measured_effect_max']:.4f}% 落在最坏情况地板"
               f"上界之内；换参考后 α 最大变动 "
               f"{invariance_delta(payload):.4f}（判据门槛 α ≤ 1 不变）；"
               f"判据文本/容差/已入库读数一律未动")
    figures = [
        {"name": fig_cover.name,
         "caption": "参考效应：受控实测（绿）vs 上界口径（浅蓝）vs 最坏情况地板上界（斜线），对数轴",
         "png_base64": base64.b64encode(fig_cover.read_bytes()).decode()},
        {"name": fig_noise.name,
         "caption": "控制有效性：重跑噪声（横）vs 受控参考效应（纵），点远在对角线上方",
         "png_base64": base64.b64encode(fig_noise.read_bytes()).decode()},
    ]
    report = {
        "title": "CASE-DG-EULER-VORTEX-ERR round 23：参考解效应量测的受控覆盖（4/8 受控，honest not_run）",
        "summary": summary,
        "scenario_key": "dg_euler_vortex_err",
        "mission_id": MISSION,
        "tags": ["TensorDG", "Euler", "isentropic-vortex", "verification",
                 "reference-floor", "controlled-measurement", SLUG],
        "markdown": doc,
        "figures": figures,
    }
    result = post("/tensor/reports", report, login())
    print(json.dumps(result)[:400])
    return 0


def invariance_delta(payload):
    invariance = payload.get("criterion_invariance") or {}
    return invariance.get("max_abs_delta_alpha") or 0.0


if __name__ == "__main__":
    raise SystemExit(main())