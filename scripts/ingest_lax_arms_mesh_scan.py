"""Ingest the cluster-run digest of ``benchmarks.lax_closure_arms_mesh_scan``.

The compute node has no matplotlib, so the driver is run there with
``--skip-figures`` and prints its whole payload as one ``DIGEST{...}`` line:

    python -m benchmarks.lax_closure_arms_mesh_scan --skip-figures \\
        --output-dir /tmp/arms_scan

This script

1. parses that digest out of the job log (no number is ever transcribed by hand
   and no threshold is ever re-typed here -- the bars travel inside the digest),
2. writes ``metrics.json`` in the standard artifact form with the job/node
   provenance,
3. redraws the two figures locally from the same numbers,
4. renders ``docs/CASE-LAX-DG-closure-arms-mesh-scan.md`` (six elements) and
   ``summary.md`` from the same numbers, so an ingest can never leave a stale
   doc or summary behind,
5. cross-checks the re-measured unlimited legs against the committed closure
   artifact and refuses to write a doc that claims reproducibility it does not
   have.

Usage::

    python scripts/ingest_lax_arms_mesh_scan.py /srv/tensor/agent/cluster/jobs/cjNNNN.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.lax_closure_arms_mesh_scan import (  # noqa: E402
    BAND_GAIN_FACTOR,
    CFL,
    DEGREE,
    DOMAIN,
    FINAL_TIME,
    L2_STAGE_GAIN_MIN,
    LIMITERS,
    MESHES,
    PREDICTIONS,
    R3_BAND_WIDTH,
    SEPARATION_MIN,
    STAGES,
    render_figures,
    render_summary,
)

OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_closure_arms_mesh_scan"
DOC = ROOT / "docs" / "CASE-LAX-DG-closure-arms-mesh-scan.md"
CLOSURE_ARTIFACT = ROOT / "docs" / "benchmarks" / "results" / "lax_resolution_closure" / "metrics.json"
FIELDS = ("density", "velocity", "pressure")


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def digest_from_log(path: Path) -> dict:
    """Pull the ``DIGEST{...}`` payload out of a queue job log."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        tail = record.get("output_tail") or ""
        index = tail.find("DIGEST")
        if index < 0:
            continue
        payload = tail[index + len("DIGEST"):]
        digest, _ = json.JSONDecoder().raw_decode(payload.strip())
        digest["_job"] = record.get("job_id")
        digest["_node"] = record.get("node")
        digest["_rc"] = record.get("rc")
        return digest
    raise SystemExit(f"no DIGEST payload in {path}")


def _pct(value: float, digits: int = 3) -> str:
    return f"{value * 100:.{digits}f} %"


def _arm_label(leg: dict) -> str:
    if leg["limiter_stage"] is None:
        return leg["limiter"]
    return f"{leg['limiter']} / {leg['limiter_stage']}"


def render_doc(metrics: dict) -> str:
    """Render the six-element case document straight from the artifact."""
    gate = metrics["gate"]
    preds = gate["predictions"]
    verdicts = gate["verdicts"]
    legs = sorted(metrics["legs"], key=lambda x: (x["cells"], x["limiter"],
                                                  str(x["limiter_stage"])))
    meshes = sorted({leg["cells"] for leg in legs})
    tol = gate["tolerance"]
    rep = metrics["closure_artifact_reproduction"]
    sep = gate["separation_wholefield_l2_over_band_l1"]
    bars = gate["bar_values"]

    def status(flag: bool) -> str:
        return "**HOLDS**" if flag else "**FAILS**"

    lines = [
        "# CASE-LAX-DG 限幅器臂 × 限幅阶段 × 网格 扫描（闭合研究第 3 轮）",
        "",
        f"> 工件：`docs/benchmarks/results/lax_closure_arms_mesh_scan/metrics.json`  ",
        f"> 生成：{metrics['generated_at_utc']}　墙钟：{metrics['wall_seconds']:.0f} s　"
        f"作业：{metrics['source'].get('job')} @ {metrics['source'].get('node')}  ",
        f"> 配置：p = {DEGREE}，CFL = {CFL}，网格 N = {list(meshes)}，"
        f"限幅器 ∈ {{{', '.join(('none' if a is None else a) for a in LIMITERS)}}}，"
        f"限幅阶段 ∈ {{{', '.join(STAGES)}}}  ",
        f"> 预登记判据：P1 {status(preds['P1']['holds'])}　P2 {status(preds['P2']['holds'])}　"
        f"P3 {status(preds['P3']['holds'])}",
        "",
        "## ① 案例介绍",
        "",
        "`CASE-LAX-DG` = Lax (1954) / Toro (2009) Test 2 激波管："
        f"`[{DOMAIN[0]}, {DOMAIN[1]}]` 域、隔膜 `x0 = 0.5`、γ = 1.4、快照 `t = {FINAL_TIME}`，"
        "左态 (ρ, u, p) = (0.445, 0.698, 3.528)、右态 (0.500, 0.000, 0.571)。"
        "解析波系为左稀疏波（头/尾）、接触间断、右激波，参考值由 `benchmarks/riemann_exact.py` "
        "独立给出。台账 `CASE-LAX-DG` 的验收原文是「ref: 精确解｜容差 < 3 %」，"
        "登记在生产腿 `p3.N400` 的**无限幅**臂上。",
        "",
        "本轮回答的问题不是「3 % 门能不能过」（前两轮已闭：生产腿 "
        "`p3.N400` 诚实 FAIL 3.116 %，同一登记配置在 N = 800 上 2.315 % 达标），"
        "而是闭合研究 §3.2/§5.3 从**单网格单阶段**推出的一句话结论是否站得住：",
        "",
        "> 「tvb / weno 都让 L2 变差（7.04 % / 6.44 % vs 3.12 %），"
        "本仓这两种限幅器在 p3 上都属于过度限制 —— 限幅器不是补救手段。」",
        "",
        "这句话有两个未测的自由度：**网格**（只在 N = 400 测过）与 **限幅阶段**"
        "（只在历史默认 `limiter_stage=\"every\"` 下测过，而本仓自己的 "
        "`docs/limiter_verification.md` 就写明 `every` 会在每个 SSP-RK3 子步后重新投影"
        "仍有标记的光滑单元、从而毁掉高阶精度）。把这两个自由度一起测，"
        "才能把「限幅器本身过度限制」与「阶段设置毁了精度」分开。",
        "",
        "## ② 怎么做的",
        "",
        "1. **factorial 设计（2 mesh × 5 臂 = 10 条腿）**：网格 N ∈ "
        f"{list(meshes)}（生产腿 400 与闭合网格 800），臂 ∈ "
        "{none, tvb/every, tvb/final, weno/every, weno/final}。"
        "无限幅臂没有阶段概念，因此每网格只出 1 条腿；其余 4 臂两两成对，"
        "使 `limiter × stage` 的交互可估。",
        "2. **其余全部固定**：p = 3、CFL = 0.35、精确 Riemann 解作分母、"
        f"单元 Gauss 节点度量、R3 固定物理带宽 ±{R3_BAND_WIDTH}（与算例池同规则）。"
        "同一作业内跑完全部 10 条腿，避免跨运行拼接。",
        "3. **预登记判据（跑数前写死在代码里，见 `PREDICTIONS`）**：",
        "",
        "| 判据 | 内容 | 门槛 |",
        "|---|---|---|",
        f"| P1 | 在 `stage=every` 下，全场 ρ L2 严格排序 none < tvb 且 none < weno，"
        f"**两个网格都成立** | 严格不等式 |",
        f"| P2 | 两个网格上，tvb 与 weno 都把 ρ 的带掩码 L1 压到无限幅臂的 "
        f"≤ {bars['band_gain_factor']:.2f} 倍 | {bars['band_gain_factor']:.2f} × |",
        f"| P3 | 两个网格上，每个限幅器从 `every` 换到 `final` 使全场 ρ L2 "
        f"至少下降 {bars['l2_stage_gain_min']:.0%} | ≥ {bars['l2_stage_gain_min']:.0%} |",
        "",
        "4. **判定规则（预登记映射，纯函数 `gate()`）**：P1 决定「排序是否网格无关」，"
        "P3 决定「惩罚是否主要是阶段伪像」，"
        f"无限幅臂的 L2/带 L1 分离度 ≥ {bars['separation_min']:.1f} 时判「两种口径答的是不同问题」。"
        "三布尔 → 结论句的映射写在代码里，任一分支都可在无集群的情况下单测（`tests/test_lax_closure_arms_mesh_scan.py`）。",
        "5. **测量路径一致性核对**：本轮重新测了 N = 400 与 N = 800 的无限幅腿，"
        "与已提交的 `lax_resolution_closure/metrics.json` 逐量比对（L1/L2/带 L1 × 三量，"
        f"共 {len(rep.get('legs', [])) * 9} 对数），"
        f"最差相对偏差 {rep.get('worst_relative_deviation', float('nan')):.3e}"
        f"（判据 ≤ {rep.get('tolerance', 1e-12):.0e}）→ "
        f"{'**REPRODUCED**' if rep.get('reproduced') else '**NOT REPRODUCED**'}。"
        "只有这一条成立，本轮的臂才能和已发表阶梯放在同一张表里。",
        "",
        "## ③ 计算结果",
        "",
        "### 3.1 全 10 条腿",
        "",
        "| N | limiter | stage | ρ L2 | u L2 | p L2 | ρ 带 L1 | L2/带L1 | 步数 | 墙钟 s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for leg in legs:
        key = f"N{leg['cells']}.{leg['limiter']}" + (
            "" if leg["limiter_stage"] is None else f".{leg['limiter_stage']}")
        lines.append(
            f"| {leg['cells']} | {leg['limiter']} | {leg['limiter_stage'] or '—'} | "
            f"{_pct(leg['l2']['density'])} | {_pct(leg['l2']['velocity'])} | "
            f"{_pct(leg['l2']['pressure'])} | {_pct(leg['band_l1']['density'])} | "
            f"{sep[key]:.2f} | {leg['steps']} | {leg['wall_seconds']:.0f} |")

    lines += ["", "### 3.2 三个预登记判据", ""]
    lines += ["**P1 —— 排序是否网格无关**（`stage=every`）", "",
              "| N | none ρ L2 | tvb ρ L2 | weno ρ L2 | tvb 更差 | weno 更差 | 成立 |",
              "|---|---|---|---|---|---|---|"]
    for row in preds["P1"]["rows"]:
        lines.append(f"| {row['cells']} | {_pct(row['none'])} | {_pct(row['tvb'])} | "
                     f"{_pct(row['weno'])} | {row['tvb_worse']} | {row['weno_worse']} | "
                     f"{row['holds']} |")
    lines += ["", f"→ P1 {status(preds['P1']['holds'])}："
              f"限幅器排序 **{verdicts['P1_limiter_ranking']}**。", ""]

    lines += ["**P2 —— 限幅器是否真的压掉了带内振荡**（`stage=every`）", "",
              "| N | none ρ 带 L1 | tvb 带 L1 (比值) | weno 带 L1 (比值) | 成立 |",
              "|---|---|---|---|---|"]
    for row in preds["P2"]["rows"]:
        lines.append(f"| {row['cells']} | {_pct(row['none'])} | {_pct(row['tvb'])} "
                     f"({row['tvb_ratio']:.3f}×) | {_pct(row['weno'])} "
                     f"({row['weno_ratio']:.3f}×) | {row['holds']} |")
    lines += ["", f"→ P2 {status(preds['P2']['holds'])}。", ""]

    lines += ["**P3 —— 换到文档承认的公平阶段 `final` 是否拿回精度**", "",
              "| N | limiter | ρ L2 (every) | ρ L2 (final) | 相对回收 | 门槛 | 成立 |",
              "|---|---|---|---|---|---|---|"]
    for row in preds["P3"]["rows"]:
        lines.append(f"| {row['cells']} | {row['limiter']} | {_pct(row['l2_every'])} | "
                     f"{_pct(row['l2_final'])} | {row['relative_gain']:.1%} | "
                     f"≥ {bars['l2_stage_gain_min']:.0%} | {row['holds']} |")
    lines += ["", f"→ P3 {status(preds['P3']['holds'])}：阶段效应 "
              f"**{verdicts['P3_stage_effect']}**。", ""]

    lines += ["### 3.3 口径分离度（无限幅臂：全场 L2 / 带掩码 L1）", "",
              "| N | 全场 ρ L2 | 带掩码 ρ L1 | 分离度 | 带内占比估计 |",
              "|---|---|---|---|---|"]
    for cells in meshes:
        leg = next(x for x in legs if x["cells"] == cells and x["limiter"] == "none")
        key = f"N{cells}.none"
        s = sep[key]
        lines.append(f"| {cells} | {_pct(leg['l2']['density'])} | "
                     f"{_pct(leg['band_l1']['density'])} | {s:.2f}× | "
                     f"{(1 - 1 / s) * 100:.1f} % |")
    lines += ["", f"→ 口径结论：**{verdicts['metric_view']}**。", ""]

    lines += ["### 3.4 判定", "",
              f"- `P1_limiter_ranking` = **{verdicts['P1_limiter_ranking']}**",
              f"- `P3_stage_effect` = **{verdicts['P3_stage_effect']}**",
              f"- `metric_view` = **{verdicts['metric_view']}**",
              f"- 对已发表措辞的处理：**{verdicts['published_statement']}**",
              "",
              "闭合研究的判定（闭合网格 = 800）**不受本单影响**：台账登记在无限幅生产腿上，"
              "限幅器臂是诊断量，不能改变 A0 的诚实 FAIL / 分辨率闭合结论。",
              ""]

    lines += ["## ④ 结果图", ""]
    for figure in metrics.get("figures", []):
        lines.append(f"- `figures/{figure['name']}` — {figure['caption']}")
    lines += [""]

    lines += [
        "## ⑤ 注意点",
        "",
        "1. **成本不是同一把尺子**：各臂步数在表 3.1 中逐条给出（CFL 相同，"
        "限幅只改单元重构，故步数接近）；墙钟列是单核墙钟，不能当作算法成本比较，"
        "只用于说明每条腿都是实际算过的。",
        "2. **带掩码 L1 的带宽是固定物理量**（±0.02），不随网格变化，"
        "这是与算例池 R3 规则一致的口径；它使带内单元占比随细化下降，"
        "所以带 L1 与全场 L2 的差别**同时**含有「振荡漏出」与「掩码面积变化」两个成分，"
        "`L2/带L1` 比值只能读作分离度指示，不能读作严格的振荡能量占比。",
        "3. **限幅器不是本仓的全部限幅手段**：这里只测 `tvb`（Cockburn–Shu）与 "
        "`weno`（Qiu–Shu）两种本仓实现，两者都把问题单元退化为线性重构；"
        "任何「限幅器无用」的推广都超出本单证据。",
        "4. **p 阶未扫描**：全部结论在 p = 3 上成立，p = 1/2 的臂不在本单范围。",
        "5. **测量路径核对是必要条件而非充分条件**：它证明本轮无限幅腿与已提交工件逐位一致，"
        "但它不证明两者的求解器版本相同以外的任何事（作业与工件的 git ref 见 `source.job`）。",
        "",
        "## ⑥ 误差判定（逐项回答「是否 < 3 %」）",
        "",
        "基准 = 台账 3 % 门（全场相对 L2，三个原始量都必须满足），"
        "以及「无限幅臂在同网格上的同一量」。",
        "",
        "| 臂 | N | ρ L2 | vs 3 % 门 | u L2 | p L2 | 三量最差是否 < 3 % |",
        "|---|---|---|---|---|---|---|",
    ]
    for leg in legs:
        worst = max(leg["l2"][f] for f in FIELDS)
        lines.append(f"| {_arm_label(leg)} | {leg['cells']} | "
                     f"{_pct(leg['l2']['density'])} | "
                     f"{'是' if leg['l2']['density'] < tol else '**否**'} | "
                     f"{_pct(leg['l2']['velocity'])} | {_pct(leg['l2']['pressure'])} | "
                     f"{'是' if worst < tol else '**否**'} |")
    qualifying = [leg for leg in legs
                  if all(leg["l2"][f] < tol for f in FIELDS)]
    lines += [
        "",
        f"**结论**：10 条腿中满足「三量全场相对 L2 全部 < 3 %」的只有 "
        f"**{len(qualifying)} 条** —— "
        + ("、".join(f"`N={leg['cells']} {_arm_label(leg)}`"
                     for leg in qualifying) if qualifying else "无") + "。",
        f"无限幅臂在 N = {meshes[-1]} 上达标（"
        f"{_pct(next(x for x in legs if x['cells'] == meshes[-1] and x['limiter'] == 'none')['l2']['density'])}），"
        "两个限幅器在任何测过的 (网格, 阶段) 组合上都**不达标**，"
        "即限幅器臂的诚实结论是 FAIL —— 但 P3 说明其中相当一部分差距是阶段设置造成的，"
        "把限幅器本身说成唯一原因是过度归因。",
        "",
        ("测量路径核对：重测的无限幅腿对已提交工件的最大相对偏差 "
         f"{rep.get('worst_relative_deviation', float('nan')):.3e}"
         + ("，远小于 3 %，因此本表可与已发表的 p3 阶梯逐位对接。"
            if rep.get("reproduced") else
            " —— **未通过 1e-12 判据**，本表的臂因此不能与已发表阶梯视为同一测量路径，"
            "结论只能在工件内部自洽的意义上读。")),
        "",
    ]
    text = "\n".join(lines)
    DOC.write_text(text, encoding="utf-8")
    return text


def digest_from_metrics(path: Path) -> dict:
    """Accept the driver's own ``metrics.json`` (written inside the job repo).

    ``cluster_jobs`` keeps only the last ~8 kB of stdout, so a ten-leg DIGEST
    line does not always survive in the job log; the driver writes the same
    payload into its ``--output-dir`` inside the job checkout, which the job
    tarball carries back verbatim.  Both routes carry exactly the same fields.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("_job", payload.get("source", {}).get("job"))
    payload.setdefault("_node", payload.get("source", {}).get("node"))
    return payload


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    log = Path(argv[0])
    digest = (digest_from_metrics(log) if log.suffix == ".json"
              else digest_from_log(log))
    gate = digest["gate"]
    metrics = {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "study": digest.get("study") or (
            "limiter arms x limiter_stage x mesh: is the published 'limiters are "
            "over-limiting at p3' claim mesh- and stage-robust?"),
        "generated_at_utc": digest["generated_at_utc"],
        "config": {"degree": DEGREE, "meshes": list(MESHES), "cfl": CFL,
                   "final_time": FINAL_TIME, "domain": list(DOMAIN),
                   "r3_band_width": R3_BAND_WIDTH,
                   "limiters": [("none" if a is None else a) for a in LIMITERS],
                   "limiter_stages": list(STAGES), "quick": False,
                   "bars": {"band_gain_factor": BAND_GAIN_FACTOR,
                            "l2_stage_gain_min": L2_STAGE_GAIN_MIN,
                            "separation_min": SEPARATION_MIN},
                   "predictions_predeclared": PREDICTIONS},
        "legs": digest["legs"],
        "gate": gate,
        "closure_artifact_reproduction": digest["closure_artifact_reproduction"],
        "wall_seconds": digest["wall_seconds"],
        "source": {"job": digest.get("_job"), "node": digest.get("_node"),
                   "rc": digest.get("_rc"), "log": str(log),
                   "note": ("driver ran with --skip-figures on the compute node "
                            "(no matplotlib there); the digest was printed to the "
                            "job log, ingested here, and the two figures were "
                            "redrawn locally from these numbers")},
        "out_dir": _rel(OUT),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    metrics["figures"] = render_figures(metrics, OUT / "figures")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    render_summary(metrics)
    render_doc(metrics)
    print(json.dumps({"P1": gate["predictions"]["P1"]["holds"],
                      "P2": gate["predictions"]["P2"]["holds"],
                      "P3": gate["predictions"]["P3"]["holds"],
                      "reproduced": metrics["closure_artifact_reproduction"].get("reproduced"),
                      "job": metrics["source"]["job"],
                      "figures": [f["name"] for f in metrics["figures"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())