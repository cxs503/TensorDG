#!/usr/bin/env python3
"""Publish the gate-power audit of CASE-DG-EULER-VORTEX-ERR (m22854).

Produces, from the machine artefacts only (``gate_power_matrix.json``,
``gate_threshold_calibration.json``, ``metrics.json`` and the pre-fix
``metrics.json`` read back out of git):

* the figure ``case_euler_vortex_errconserve/fig_gate_power_audit.png``;
* the narrative document ``docs/AUDIT-CASE-DG-EULER-VORTEX-ERR-gate-power.md``;
* the platform report (POST ``/tensor/reports``).
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "case_euler_vortex_errconserve"
DOC = ROOT / "docs" / "AUDIT-CASE-DG-EULER-VORTEX-ERR-gate-power.md"
FIG = BUNDLE / "fig_gate_power_audit.png"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def load():
    power = json.loads((BUNDLE / "gate_power_matrix.json").read_text())
    cal = json.loads((BUNDLE / "gate_threshold_calibration.json").read_text())
    metrics = json.loads((BUNDLE / "metrics.json").read_text())
    old = json.loads(subprocess.run(
        ["git", "show", "HEAD:case_euler_vortex_errconserve/metrics.json"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    return power, cal, metrics, old


def figure(power, metrics, old):
    names = list(power["acceptance"])
    ids = [row["id"] for row in power["rows"]]
    grid = np.zeros((len(names), len(ids)))
    for j, row in enumerate(power["rows"]):
        for i, name in enumerate(names):
            status = row["statuses"].get(name, "—")
            grid[i, j] = {"passed": 2.0, "failed": 0.0, "not_evaluated": 1.0}.get(status, 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    ax.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=2.0, aspect="auto")
    ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)), [n.split("_")[0] + " " + n.split("_", 1)[1].replace("_", " ")
                                      for n in names], fontsize=8)
    for j, row in enumerate(power["rows"]):
        for i, name in enumerate(names):
            mark = {"passed": "·", "failed": "F", "not_evaluated": "N"}.get(
                row["statuses"].get(name, "—"), "?")
            ax.text(j, i, mark, ha="center", va="center", fontsize=7)
    ax.set_title("gate status per injected defect\nF=failed N=not_evaluated ·=passed", fontsize=10)
    for j, row in enumerate(power["rows"]):
        if row["target"]:
            i = names.index(row["target"])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="black", lw=1.6))

    ax = axes[1]
    cases = [f"dg_d{k.split('_')[1]}_t{k.split('_')[2][1:]}" for k in old["gates"]["C9_design_order"]["per_case"]]
    before = [old["gates"]["C9_design_order"]["per_case"][k]["order"]
              for k in old["gates"]["C9_design_order"]["per_case"]]
    after = [metrics["gates"]["C9_design_order"]["per_case"][k]["order"]
             for k in old["gates"]["C9_design_order"]["per_case"]]
    x = np.arange(len(cases))
    ax.bar(x - 0.2, before, 0.4, label="published before the fix (mixed directions)", color="#c0504d")
    ax.bar(x + 0.2, after, 0.4, label="after the fix (baseline direction only)", color="#4f81bd")
    asymptote = [int(k.split("_")[1][1:]) + 1 for k in old["gates"]["C9_design_order"]["per_case"]]
    for xi, value in zip(x, asymptote):
        ax.hlines(value, xi - 0.45, xi + 0.45, color="black", ls="--", lw=1)
    ax.axhline(2.4, color="grey", ls=":", lw=1)
    ax.text(len(cases) - 0.5, 2.45, "C9 target 0.8(p+1)=2.4", fontsize=7, ha="right")
    ax.text(len(cases) - 0.5, 3.05, "asymptote p+1 (dashed)", fontsize=7, ha="right")
    ax.set_xticks(x, cases, fontsize=9)
    ax.set_ylabel("observed spatial order")
    ax.set_title("C9: the published DG-P2 reading sat above its own asymptote", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return FIG


def document(power, cal, metrics, old):
    c1 = metrics["gates"]["C1_tolerance"]
    c9_old = old["gates"]["C9_design_order"]["per_case"]
    c9_new = metrics["gates"]["C9_design_order"]["per_case"]
    lines = [
        "# AUDIT — CASE-DG-EULER-VORTEX-ERR gate power and threshold calibration",
        "",
        "> 锚点：`CASE-DG-EULER-VORTEX-ERR`（feature_ledger status=open）",
        "> 分支：`agent/m22854`　证据：`case_euler_vortex_errconserve/gate_power_matrix.json`、",
        "> `gate_threshold_calibration.json`、`metrics.json`、`fig_gate_power_audit.png`",
        "",
        "## 1. 这份审计要回答什么",
        "",
        "前 19 轮把 C1–C11 判成 PASS。**PASS 只说明门禁接受了真证据，不说明它会拒绝缺陷证据。**",
        "十一个判据里有十个带手写反例测试，C4（熵预算）没有——"
        "也就是说 C4 的鉴别力一直是假设。本单用机器做三件事：",
        "",
        "1. **杀伤力**：对每个判据注入一个真实缺陷形态，看它是否翻成 `failed`；",
        "2. **惰性对照**：改与判据无关的字段（墙钟、迭代计数），门禁不许有任何反应；",
        "3. **门限标定**：每个具名阈值两侧各打一针（内侧必过 / 外侧必败），"
        "证明「判据*声称*的数」就是「代码*用的*数」。",
        "",
        f"被审计对象是**已提交**的证据包（{power['record_count']} 条 result 记录 + "
        f"{power['certificate_count']} 张重放证书），不是合成夹具。",
        "",
        "## 2. 结果",
        "",
        f"* 基线裁决复现：**{power['baseline']['verdict']}**；",
        f"* 杀伤力：**{power['power']}**，`survivors = {power['survivors']}`；",
        f"* 惰性对照：`control_inert = {power['control_inert']}`；",
        f"* 门限标定：**{len(cal['calibrated'])}/{len(cal['rows'])}** 咬合，"
        f"`miscalibrated = {cal['miscalibrated']}`。",
        "",
        "## 3. 查出的阻断级缺陷（C9 设计阶混入方向轴）",
        "",
        "**现象**：已发布 `metrics.json` 里 `dg_d2_t1` 的观测空间阶读 **4.09**，"
        f"`dg_d2_t10` 读 **4.64**（本单修正后为 {c9_new['dg_d2_t1']['order']:.4f} / "
        f"{c9_new['dg_d2_t10']['order']:.4f}）。DG-P2 在渐近区不可能超过 p+1=3——"
        "这个数本身就是报警。",
        "",
        "**根因**：`_dg_order` 用最小二乘拟合 `log e` 对 `log h`，但筛选条件只钉了 CFL "
        "（`ORDER_CFL`），没有钉**对流方向**。C10 方向轴落地后，p=2 在 t=1/10 参与拟合的不再是"
        "「两个网格」，而是 `cells = [32,32,32,64]`——`diag45`/`axis0`/`obl30` 三条 32 网格腿"
        "加一条 64 网格腿。三个方向的误差在 32 网格上相差 6 倍"
        f"（{metrics['records'] and ''}`diraxis0` 0.0597% vs 设计点 0.0094%），"
        "回归斜率量到的主要是**方向差**，不是网格。",
        "",
        "**为什么这是缺陷而不是口味问题**：`_dg_order` 自己的 docstring 写明钉 CFL 的理由是"
        "「两个 CFL 在同一网格上是两个离散，拟合过这两点量到的是时间步而不是网格」——"
        "同一句话对「两个方向在同一网格上」逐字成立。方向由 C10 判，不该从 C9 后门进来。",
        "",
        "**修复**（`benchmarks/case_euler_vortex_errconserve.py`，最小改动）：",
        "新增 `record_direction(record)`（缺 `background_velocity` 的历史记录判为基线 `diag45`，"
        "与 LADDER 的默认一致），并把 `_dg_order` 的筛选条件加上"
        "`record_direction(r) == DIRECTION_BASELINE`，与已有的 CFL 钉子同构。",
        "",
        "**修复前后读数**（程序化读取，零手抄）：",
        "",
        "| 判据腿 | 修正前 cells | 修正前 order | 修正后 cells | 修正后 order | 目标 0.8(p+1) |",
        "|---|---|---|---|---|---|",
    ]
    for key in c9_new:
        lines.append(
            f"| `{key}` | {c9_old[key]['cells']} | {c9_old[key]['order']:.4f} | "
            f"{c9_new[key]['cells']} | {c9_new[key]['order']:.4f} | "
            f"{c9_new[key]['target']:.2f} |")
    lines += [
        "",
        "裁决**不变**（PASS，C9 四条腿在修正后仍全部达标），所以这不是把 FAIL 粉饰成 PASS 的故事，"
        "而是「数读错了、门禁也对它没反应」：修正前，把真正的设计网格细化对（`dg_d2_n32` → "
        "`dg_d2_n64`）人为压到 1 阶，C9 照样读 `passed`——它存在的那条失效路径它抓不到。"
        "修正后 `k09` 标定在 2.4 两侧咬合，`test_c9_detects_a_genuine_mesh_order_collapse` 锁死。",
        "",
        "## 4. 门禁杀伤力矩阵与标定表",
        "",
        (BUNDLE / "docs_placeholder").name and "",
        "见下（由 `tools/audit_vortex_gate_power.py --markdown` 生成，本文件末尾附全矩阵）。",
        "",
        "## 5. 关于「算例池 6% 容差覆盖到哪些读数」的诚实说明",
        "",
        f"C1 只在**设计检查点 t=1** 对全部族施加 {power['tolerance_relative_l2']:.0%} 容差，"
        f"最坏读数是 `{c1['worst_at']}` 的 {c1['measured']:.4%}"
        f"（容差余量仅 {(1 - c1['measured'] / power['tolerance_relative_l2']):.1%}）。"
        "长时检查点由 C2（累积指数）判，原始最坏值逐条留在 C1x（informational，不参与裁决）："
        "P0 一阶交叉验证腿 t=10 读到 14.3%，这是刻意的一阶路径，不是 DG 的精度声明。"
        "本单不改这条口径，只把它写清楚：**「<6%」在本算例指的是 t=1 快照容差，"
        "不是所有检查点**。",
        "",
        "## 6. 注意点与适用边界",
        "",
        "* 变异都是「一个缺陷一种形态」，不能证明门禁对*任意*缺陷敏感，只能证明这 14 种形态被照见；",
        "* 标定用 1e-3 的相对余量把探针放在阈值两侧，避免浮点噪声决定结论；",
        "* 附带翻转（如 m02 同时翻 C7/C8/C10）是**判据耦合**，不是缺陷：设计族同时是 C2 的拟合族和 "
        "C10 的基线方向，报告照实列出；",
        "* 本审计不重跑任何求解器（纯后处理），因此不涉及数值本身；它审的是「门禁有没有牙」。",
        "",
        "## 7. 复现",
        "",
        "```",
        "PYTHONPATH=. python3 tools/audit_vortex_gate_power.py --markdown docs/AUDIT-CASE-DG-EULER-VORTEX-ERR-gate-power.md",
        "PYTHONPATH=. python3 -m pytest -q tests/test_case_euler_vortex_errconserve_gate_power.py",
        "```",
    ]
    return "\n".join([line for line in lines if line is not None])


def post_report(title, summary, markdown, tags, figures):
    user = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
    password = os.environ.get("TENSOR_AGENT_PASS", "eco_admin")
    login = urllib.request.Request(
        f"{BASE}/auth/login",
        data=json.dumps({"username": user, "password": password}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(login, timeout=60) as response:
        token = json.loads(response.read())["access_token"]
    payload = {"title": title, "summary": summary, "scenario_key": "case_dg",
               "mission_id": 22854, "tags": tags, "markdown": markdown,
               "figures": figures}
    request = urllib.request.Request(
        f"{BASE}/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> int:
    power, cal, metrics, old = load()
    figure(power, metrics, old)
    markdown = document(power, cal, metrics, old)
    DOC.write_text(markdown + "\n\n" + (BUNDLE / "gate_power_matrix.md").read_text()
                   if (BUNDLE / "gate_power_matrix.md").exists() else markdown + "\n")
    table = json.loads((BUNDLE / "gate_power_matrix.json").read_text())
    from tools.audit_vortex_gate_power import render_markdown  # noqa: E402
    DOC.write_text(markdown + "\n\n" + render_markdown(table, cal) + "\n")
    png = base64.b64encode(FIG.read_bytes()).decode()
    result = post_report(
        "CASE-DG-EULER-VORTEX-ERR：门禁杀伤力与门限标定审计（C9 方向轴污染修复）",
        ("对已 PASS 的等熵涡算例做门禁对抗性审计：11/11 判据可被缺陷杀死、"
         "2 个惰性对照无反应、7/7 具名阈值两侧咬合；并查出阻断级缺陷——C9 的网格阶拟合混入 C10 方向轴，"
         f"使 DG-P2 读出 4.09 阶（超过自身 p+1=3 渐近极限），已最小修复，裁决仍 PASS。"),
        markdown + "\n\n" + render_markdown(table, cal),
        ["CASE-DG-EULER-VORTEX-ERR", "gate-power", "mutation-audit",
         "threshold-calibration", "C9", "C10", "verification-debt"],
        [{"name": "fig_gate_power_audit.png",
          "caption": ("左：11 判据 × 14 注入缺陷的状态矩阵（黑框=该变异的目标判据）；"
                      "右：C9 设计阶修正前后读数，虚线为 p+1 渐近线，点线为 0.8(p+1) 门限。"),
          "png_base64": png}])
    print("report:", json.dumps(result)[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())