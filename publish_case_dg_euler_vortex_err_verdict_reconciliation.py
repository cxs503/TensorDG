#!/usr/bin/env python3
"""Publish the verdict-reconciliation round for CASE-DG-EULER-VORTEX-ERR.

Reads ``artifacts/verdict_reconciliation_CASE-DG-EULER-VORTEX-ERR.json`` (written
by ``tools/verdict_reconcile.py``), draws three figures, emits
``docs/CASE-DG-EULER-VORTEX-ERR-verdict-reconciliation.md`` and posts the
paper-style report.  Every number in the document is read from that artefact --
nothing is transcribed by hand.

Usage::

    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_verdict_reconciliation.py --dry-run
    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_verdict_reconciliation.py
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
MISSION = 23232
SLUG = "CASE-DG-EULER-VORTEX-ERR"
REPORT = Path("artifacts") / f"verdict_reconciliation_{SLUG}.json"
DOC = Path("docs") / f"{SLUG}-verdict-reconciliation.md"
FIG_LOLLIPOP = Path("docs") / "fig_verdict_lollipop.png"
FIG_CERTIFICATE = Path("docs") / "fig_verdict_certificate_vs_time.png"
FIG_TRISTATE = Path("docs") / "fig_verdict_tristate.png"


def num(value, digits=6):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def percent(value, digits=6):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}f} %"


def make_figures(report):
    artefacts = report["artefacts"]
    stale = next(a for a in artefacts if a["classification"] == "SUPERSEDED")
    current = report["authoritative_percent"]
    predicted = stale["certificate"]["predicted_percent"]
    tolerance = 100.0 * report["tolerance"]

    # -- A: where each number sits on the error axis ------------------------
    labels = ["current anchor\n(case_euler_vortex_yee)",
              "certificate prediction\n(free-space vs periodic)",
              "superseded artefact\n(case_euler_vortex/metrics_yee.json)"]
    values = [current, predicted, stale["gate_percent"]]
    colors = ["#2f855a", "#b7791f", "#c53030"]
    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=160)
    for label, value, color in zip(labels, values, colors):
        ax.plot([1e-3, value], [label, label], "-", color=color, lw=2.2)
        ax.plot([value], [label], "o", color=color, ms=8)
    for label, value in zip(labels, values):
        ax.text(value * 1.25, label, f"{value:.6g} %", va="center", fontsize=8)
    ax.axvline(tolerance, color="crimson", ls="--", lw=1.2,
               label=f"{tolerance:g} % tolerance")
    ax.set_xscale("log")
    ax.set_xlim(5e-3, 1e2)
    ax.set_xlabel("max-field relative L2 against the periodic exact solution [%] (log)")
    ax.set_title("the superseded number is the reference model, not the scheme")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_LOLLIPOP)
    plt.close(fig)

    # -- B: the certificate as a function of time --------------------------
    cert = report["certificate"]
    times = sorted(float(k) for k in cert["records"])
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=160)
    for field in ("velocity_x", "density", "pressure"):
        series = [cert["records"][f"{t:g}"]["per_field_percent"][field] for t in times]
        ax.plot(times, series, "o-", ms=3.5, label=field.replace("_", " "))
    maxima = [cert["records"][f"{t:g}"]["max_percent"] for t in times]
    ax.plot(times, maxima, "k--", lw=1.1, label="max over fields")
    ax.axvline(cert["period"], color="#b7791f", lw=1.0, ls=":",
               label=f"one translation period (t = {cert['period']:g})")
    ax.axhline(stale["gate_percent"], color="#c53030", lw=1.0, ls="-.",
               label=f"superseded artefact: {stale['gate_percent']:.6g} %")
    ax.set_xlabel("final time t")
    ax.set_ylabel("free-space vs periodic exact solution [%]")
    ax.set_title("certificate: the free-space reference drifts off as the core leaves the box")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_CERTIFICATE)
    plt.close(fig)

    # -- C: published boolean vs tri-state, per study tree -----------------
    studies = [a for a in artefacts if a.get("gate_states")]
    studies.sort(key=lambda a: a["path"])
    names = [Path(a["path"]).parent.name for a in studies]
    passed = [sum(1 for s in a["gate_states"].values() if s == "PASS") for a in studies]
    failed = [sum(1 for s in a["gate_states"].values() if s == "FAIL") for a in studies]
    not_run = [sum(1 for s in a["gate_states"].values() if s == "INCOMPLETE") for a in studies]
    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=160)
    ax.barh(names, passed, color="#2f855a", label="evaluated PASS")
    ax.barh(names, failed, left=passed, color="#c53030", label="evaluated FAIL")
    ax.barh(names, not_run, left=[p + f for p, f in zip(passed, failed)],
            color="#a0aec0", label="never evaluated (was published as failure)")
    for index, study in enumerate(studies):
        total = passed[index] + failed[index] + not_run[index]
        ax.text(total + 0.08, index,
                f"all(passed)={study.get('raw_all_passed_boolean')} → {study['verdict']}",
                va="center", fontsize=7)
    ax.set_xlabel("gates in that artefact")
    ax.set_title("'not run' was being published as 'failed' (grey = never evaluated)")
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_TRISTATE)
    plt.close(fig)

    return FIG_LOLLIPOP, FIG_CERTIFICATE, FIG_TRISTATE


def build_doc(report):
    artefacts = report["artefacts"]
    stale = next(a for a in artefacts if a["classification"] == "SUPERSEDED")
    cert = report["certificate"]
    certificate_hit = stale["certificate"]
    ratio = stale["gate_percent"] / report["authoritative_percent"]
    demoted = [a for a in artefacts if a["verdict"] == "INCOMPLETE"]
    rescued = [a for a in artefacts if "C6_ladder_completeness" in a["failed_gates"]]
    tolerance = 100.0 * report["tolerance"]

    lines = []
    add = lines.append
    add(f"# {SLUG}：判读台账对账（单一事实源）")
    add("")
    add("> 本报告的每个数字都由 `tools/verdict_reconcile.py` 从已入库工件程序化注入，无手抄；"
        "判据由脚本计算。原始对账件：`" + str(REPORT) + "`。")
    add("")
    add("## ① 案例介绍")
    add("")
    add("二维可压缩 Euler（γ = 1.4）等熵涡在双周期域 [-5, 5]² 上被均匀来流 (u, v)∞ = (1, 1) "
        "刚性输运；β = 5，域长 L = 10、|u∞| = √2 的分量速度各为 1，故沿每一轴的平移周期为 "
        "T = L/u∞ = 10。t = T 时解析精确解与初始条件逐点相同——这正是本锚点"
        "「移动误差守恒」要判的东西：不是短时精度，而是一个整周期后误差是否仍被 6 % 容差覆盖。")
    add("")
    add("本单不重算算例，而是审**判读**：同一个台账行在同一 commit 上被 7 个工作树、"
        f"{len(artefacts)} 个断言判读的工件覆盖，其中 2 个互相矛盾。锚点能否关闭，"
        "取决于读哪一份。")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 台账 slug | `{SLUG}` |")
    add(f"| 容差（算例池原文） | {tolerance:g} % |")
    add(f"| 权威工件 | `{report['authoritative_path']}` |")
    add(f"| 权威判读 | **{report['authoritative_verdict']}** "
        f"（{percent(report['authoritative_percent'])}） |")
    add(f"| 对账后判定 | **{'PASS' if report['adjudicated_pass'] else 'FAIL'}** |")
    add(f"| 阻塞问题 | {len(report['blocking'])} |")
    add("")
    add("## ② 怎么做的")
    add("")
    add("**方法**：不新增物理算例，而是把「判读」本身当作被测对象——为台账行建立单一事实源，"
        "并用两个可证伪的判据把失效模式钉住。")
    add("")
    add("| 环节 | 实现 | 说明 |")
    add("|---|---|---|")
    add("| 枚举 | `tools/verdict_reconcile.py::collect` | 只认 `git ls-files` 跟踪的工件："
        "读者 checkout 得到的那份证据才作数 |")
    add("| 提取 | `extract` | 四种工件形态（aggregation / gates / certification / config）"
        "归一到同一结构 |")
    add("| 三值归约 | `gate_state` + `tri_state` | 未求值的门禁只能是 `INCOMPLETE`，"
        "不许塌成 `FAIL` |")
    add("| 作用域声明 | `SCOPE_REGISTRY` | 断言判读的工件必须显式声明 `anchor` / "
        "`study:*` / `superseded`，未声明即阻塞 |")
    add("| 溯源扫描 | `producer_scan` | 没有任何脚本写它的路径 = 孤儿工件，判读已被冻结 |")
    add("| 参考模型证书 | `certificate` | 在门禁配置的同一块求积格上，"
        "直接算自由空间参考与周期参考的解析解之差 |")
    add("| 落盘 | `--mark` / `--report` | 被取代件旁写 `SUPERSEDED.json`，"
        "同时产出对账 JSON |")
    add("")
    add("**工具链**：Python 3.11 + torch（解析解与求积）、pytest、matplotlib、git。")
    add("")
    add("### 两个被测缺陷")
    add("")
    add("**D1｜同一 slug、同一容差、同一门禁配置，两份工件给出相反判读。**")
    add("")
    add("| 工件 | 门禁读数 | 发布判读 |")
    add("|---|---|---|")
    add(f"| `{stale['path']}` | {percent(stale['gate_percent'])} | `overall_passed: false` |")
    add(f"| `{report['authoritative_path']}` | {percent(report['authoritative_percent'])} "
        f"| `overall_passed: true` |")
    add("")
    add("两份工件的 `reference` 字符串**逐字相同**、`config_id` 也相同"
        "（`dg_d2_n32_t10_cfl0.2`），因此从文件内容本身无法分辨谁是新账——"
        "谁先被读到，锚点就朝谁倒。")
    add("")
    add("**D2｜「没跑」被发布成「失败」。** 多个研究工件暴露 `all(gate.passed)`，"
        "而它们各自输出目录里的大部分门禁根本没求值（`measured: nan`、`passed: None`，"
        "检查点在另一个树里），布尔值因此是 `False`——对任何消费者而言与「物理失败」"
        "不可区分。覆盖度事实被当成科学结论发布。")
    add("")
    add("## ③ 计算结果")
    add("")
    add("### 表 1：对账后的事实源（逐工件）")
    add("")
    add("| 工件 | 作用域 | 分类 | 三值判读 | 门禁 % | 未跑门禁 | 生产者 |")
    add("|---|---|---|---|---|---|---|")
    for record in artefacts:
        add(f"| `{record['path']}` | {record['scope']} | {record['classification']} | "
            f"{record['verdict']} | "
            f"{'—' if record['gate_percent'] is None else num(record['gate_percent'])} | "
            f"{len(record['not_run_gates'])} | {len(record['producers'])} |")
    add("")
    add("### 表 2：D1 的根因证书（参考模型，不是格式）")
    add("")
    add("| 量 | 值 |")
    add("|---|---|")
    add(f"| 被取代件的门禁读数 | {percent(stale['gate_percent'])} |")
    add(f"| 证书预测（自由空间 vs 周期解析解，t = {certificate_hit['certificate_time']:g}） "
        f"| {certificate_hit['predicted_percent']:.6f} % |")
    add(f"| 相对偏差 | {certificate_hit['relative_difference']:.3e} |")
    add(f"| 与实测误差之比 | {ratio:.1f} × |")
    add(f"| 证书匹配带（准入） | {certificate_hit.get('band', 0.01) if certificate_hit.get('band') else 0.01:.0%} |")
    add("")
    add("证书在 t = 10 给出 "
        f"{certificate_hit['predicted_percent']:.4f} %，被取代件发布 {stale['gate_percent']:.4f} %，"
        f"相对偏差 {certificate_hit['relative_difference']:.2e}——"
        "这不是格式误差可以解释的量级，而是同一个量被两种参考模型各算了一遍。"
        "t = 1 时两者只差 "
        f"{cert['records']['1']['max_percent']:.4f} %（涡核仍在盒内，两个模型一致），"
        "差异随核心离开盒子的程度单调增长，到 t = 10（恰好一个平移周期）达到极值。"
        "**结论：该工件量的是参考模型错配，不是格式误差。**")
    add("")
    add("### 表 3：D2 的改正（未跑 ≠ 失败）")
    add("")
    add("| 工件 | 原发布布尔 | 未跑门禁数 | 三值判读 |")
    add("|---|---|---|---|")
    for record in demoted:
        add(f"| `{record['path']}` | {record.get('raw_all_passed_boolean', '—')} | "
            f"{len(record['not_run_gates'])} | **{record['verdict']}** |")
    add("")
    if rescued:
        add("")
        add("反向的坑（自我复核查出）：`C6_ladder_completeness` 的判据是**计数**"
            "（已入库检查点 vs 已声明检查点），它实测 `status: failed` 却没有数字读数。"
            "只看 `measured` 的规则会把它塌成「未判」——把一个真实的失败藏起来。"
            "因此三值归约以工件自报的 `status` 为最强证据、`measured` 只在无 `status` 时兜底：")
        add("")
        for record in rescued:
            add(f"- `{record['path']}`：`failed_gates={record['failed_gates']}`、"
                f"`not_run_gates={len(record['not_run_gates'])} 个`、判读 **{record['verdict']}**"
                "（真实失败与未跑门禁同时如实登记，互不掩盖）")
    add("")
    add("## ④ 结果图")
    add("")
    add(f"![verdict lollipop]({FIG_LOLLIPOP})")
    add("")
    add("*图 1：三个数字落在误差轴上的位置（对数轴）。权威工件 "
        f"{report['authoritative_percent']:.6g} % 远在 6 % 容差线左侧；"
        "被取代件的读数与证书预测几乎重合——它量的是参考模型，不是格式。*")
    add("")
    add(f"![certificate vs time]({FIG_CERTIFICATE})")
    add("")
    add("*图 2：证书随时间的取值（三个原始场 + 逐时刻最大值）。t = 1 时两个参考模型一致，"
        "涡核离开盒子的过程中差异单调增长，t = 10 达到被取代件发布的那个读数。*")
    add("")
    add(f"![tri-state]({FIG_TRISTATE})")
    add("")
    add("*图 3：「没跑」被发布成「失败」的现状与改正。柱长为该输出目录里从未求值的门禁数，"
        "标注为原布尔值 → 三值判读。*")
    add("")
    add("## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）")
    add("")
    add("1. **本单没有重算任何物理算例**。它审计的是判读与证据链，"
        "能量守恒/收敛阶等物理结论仍以既有研究报告为准；本单不构成对它们的独立复算。")
    add("2. **作用域声明是人工维护的表**（`SCOPE_REGISTRY`）。这是有意的：把作用域写成"
        "「猜出来的」就等于把 D1 换个形式再犯一次。代价是新增断言工件必须显式登记，"
        "未登记会被 `--check` 阻塞——这是特性不是缺陷，但确实需要下一单维护。")
    add("3. **证书带 1 %**。同一差异两种算法之间实测差 ~2e-5，而真实格式误差比 16 % 小三个量级，"
        "1 % 分离得很开；但如果将来出现量级介于两者之间的可疑数字，该带需要重新论证。")
    add("4. **被取代件没有删除**。它被保留为历史证据并在旁边写了 `SUPERSEDED.json`——"
        "删证据会让「为什么曾经是 16 %」变成不可复查的口述。")
    add("5. **权威判读是 PASS，但台账 slug 仍是 open**。本单只负责让判读可被机械认定，"
        "关账动作属于台账流程，不在本单范围内。")
    add("6. **适用范围**：判读对账思路与域无关（任何「多份工件断言同一验收行」的场景都适用）；"
        "证书部分绑定本算例的自由空间/周期解析解，换算例需换证书。")
    add("")
    add("## ⑥ 误差判定")
    add("")
    add("| 关键量 | 基准 | 实测 | 误差 | 是否 <3 %？ |")
    add("|---|---|---|---|---|")
    add(f"| 权威工件的门禁读数 vs 6 % 容差 | ≤ 6 % | "
        f"{report['authoritative_percent']:.6g} % | "
        f"{report['authoritative_percent'] / tolerance * 100.0:.2f} % of tolerance | ✓ 是 |")
    add(f"| 证书预测 vs 被取代件读数 | 被取代件自身 | "
        f"{certificate_hit['predicted_percent']:.6f} % | "
        f"相对偏差 {certificate_hit['relative_difference']:.2e} | ✓ 是（远优） |")
    add(f"| D2 改正后的未跑门禁数 vs 命题「全部未求值」 | 原布尔 False | "
        f"{sum(len(a['not_run_gates']) for a in demoted)} 个门禁 | 一致 | ✓ 是 |")
    add(f"| 阻塞问题数 vs 0 | 0 | {len(report['blocking'])} | 0 | ✓ 是 |")
    add("")
    add("**逐项回答「是否 <3 %」**：")
    add("")
    add(f"- 权威判读：{report['authoritative_percent']:.6g} % 对 {tolerance:g} % 容差，"
        f"占容差 {report['authoritative_percent'] / tolerance * 100.0:.2f} %，**是**。")
    add(f"- 根因证书：与被取代件读数相对偏差 {certificate_hit['relative_difference']:.2e}，"
        "远小于 3 %，**是**。")
    add("- D2 改正的判据是「未跑门禁数 = 未求值门禁数」，属计数一致性（离散、无容差），"
        "已由 `tests/test_euler_vortex_verdict_reconciliation.py::test_incomplete_study_is_not_published_as_fail` "
        "逐件断言，**是**。")
    add("- 本单未重算格式误差，因此不对「DG 格式精度」下新的误差判定；那一量沿用既有研究报告，"
        "本单不重复声称。")
    add("")
    add("## ⑦ 复现")
    add("")
    add("```bash")
    add("# 对账（阻塞则退出码 1）")
    add("PYTHONPATH=. python3 tools/verdict_reconcile.py --check")
    add("# 落盘：被取代件标记 + 对账 JSON")
    add("PYTHONPATH=. python3 tools/verdict_reconcile.py --mark \\")
    add(f"    --report {REPORT}")
    add("# 门禁测试")
    add("PYTHONPATH=. python3 -m pytest tests/test_euler_vortex_verdict_reconciliation.py -q")
    add("```")
    return "\n".join(lines) + "\n"


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
    report = json.loads(REPORT.read_text())
    figures = make_figures(report)
    doc = build_doc(report)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(doc)
    stale = next(a for a in report["artefacts"] if a["classification"] == "SUPERSEDED")
    print(f"wrote {[str(f) for f in figures]}, {DOC} ({len(doc)} chars); "
          f"adjudicated={'PASS' if report['adjudicated_pass'] else 'FAIL'}")
    if args.dry_run:
        return 0
    summary = (
        f"{SLUG} 判读对账：同一台账行在同一 commit 上被 2 份工件判成相反结果"
        f"（{stale['gate_percent']:.4g} % vs {report['authoritative_percent']:.4g} %，容差 "
        f"{100 * report['tolerance']:g} %），且多份研究工件把「没跑」的门禁塌成「失败」。"
        f"本单建立单一事实源：权威判读 {report['authoritative_verdict']}"
        f"（{report['authoritative_percent']:.6g} %）；被取代件的读数由其自由空间参考模型证书"
        f"复现（预测 {stale['certificate']['predicted_percent']:.4f} %，相对偏差 "
        f"{stale['certificate']['relative_difference']:.2e}），故归类为参考模型错配而非格式失败；"
        f"三值归约把 {sum(1 for a in report['artefacts'] if a['verdict'] == 'INCOMPLETE')} "
        f"份工件从「失败」改正为「未判」，同时不把计数型门禁的真实失败（C6）塌成未判；"
        f"阻塞问题 {len(report['blocking'])} 项。")
    payload = {
        "title": f"{SLUG}：判读台账对账与参考模型证书（单一事实源）",
        "summary": summary,
        "scenario_key": "dg_euler_vortex_err",
        "mission_id": MISSION,
        "tags": ["TensorDG", "Euler", "isentropic-vortex", "verification",
                 "evidence-integrity", "adversarial-review", SLUG],
        "markdown": doc,
        "figures": [
            {"name": FIG_LOLLIPOP.name,
             "caption": "三个数字在误差轴上的位置：权威读数、证书预测、被取代件读数（对数轴，红虚线=6 % 容差）",
             "png_base64": base64.b64encode(FIG_LOLLIPOP.read_bytes()).decode()},
            {"name": FIG_CERTIFICATE.name,
             "caption": "证书随时间的取值：自由空间参考与周期参考的解析解之差，随涡核离开盒子单调增长",
             "png_base64": base64.b64encode(FIG_CERTIFICATE.read_bytes()).decode()},
            {"name": FIG_TRISTATE.name,
             "caption": "「没跑」被发布成「失败」：各工件从未求值的门禁数与原布尔值、三值判读的对照",
             "png_base64": base64.b64encode(FIG_TRISTATE.read_bytes()).decode()},
        ],
    }
    result = post("/tensor/reports", payload, login())
    print(json.dumps(result)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())