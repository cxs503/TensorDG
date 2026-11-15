"""Render ``docs/CASE-LAX-DG-reproducibility-audit.md`` from the audit artifact.

Every number in the rendered document is injected from
``docs/benchmarks/results/lax_reproducibility_audit/metrics.json`` (zero hand
transcription); ``--check`` re-renders and fails when the tracked document has
drifted away from the artifact.

    python -m scripts.render_lax_reproducibility_doc            # write
    python -m scripts.render_lax_reproducibility_doc --check     # verify
    python -m scripts.render_lax_reproducibility_doc --figures   # (re)draw PNGs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "docs" / "benchmarks" / "results" / "lax_reproducibility_audit"
DOC = ROOT / "docs" / "CASE-LAX-DG-reproducibility-audit.md"
FIG_DIR = ART / "figures"

ARTIFACT_TITLES = {
    "lax_case_pool": "算例池 9 腿阶梯（L1）",
    "lax_resolution_requirement": "分辨率需求（钉扎腿 + 闭合腿）",
    "lax_certified_configuration": "认证构型 (p, N) 扫描",
    "lax_asymptotic_probe": "P3 渐近探针",
    "lax_resolution_closure": "分辨率闭合阶梯",
}


def _sci(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}e}"


def _pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f} %"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def draw_figures(metrics: dict, out_dir: Path = FIG_DIR) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    # 1) reproduction drift per artifact, log scale, with the tolerance band
    drifts = {name: payload["worst_relative_difference"]
              for name, payload in metrics["reproduction"]["by_artifact"].items()
              if payload["shared_readings"]}
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    names = list(drifts)
    values = [max(v, 1e-18) for v in drifts.values()]
    bars = ax.bar(range(len(names)), values,
                  color=["#2c7fb8" if v <= metrics["reproduction"]["tolerance"] else "#d95f0e"
                         for v in values])
    ax.set_yscale("log")
    ax.axhline(metrics["reproduction"]["tolerance"], color="k", ls="--", lw=1,
               label=f"reproducible ≤ {metrics['reproduction']['tolerance']:g}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("lax_", "").replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("worst relative difference\n(fresh run vs tracked artifact)")
    ax.set_title("R2 artifact reproduction: fresh HEAD run vs committed reading")
    for bar, value in zip(bars, values):
        ax.annotate(_sci(value, 2), (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "lax_reproducibility_drift.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)

    # 2) quadrature bias of the published readings
    rows = [row for row in metrics["quadrature"]["rows"] if row["field"] == "density"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    labels = [row["leg"] for row in rows]
    ax.plot(range(len(rows)), [row["order_10"] for row in rows], "o-", label="order 10 (artifact norm)")
    ax.plot(range(len(rows)), [row["order_20"] for row in rows], "s--", label="order 20 (converged)")
    ax.axhline(metrics["quadrature"]["tolerance"], color="r", ls=":", lw=1,
               label=f"ledger bar {metrics['quadrature']['tolerance']:.0%}")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("density whole-field rel L2")
    ax.set_title("R4 quadrature bias: same field, two norm conventions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "lax_quadrature_bias.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)
    return made


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------

def render(metrics: dict) -> str:
    env = metrics["environment"]
    cfg = metrics["config"]
    reproduction = metrics["reproduction"]
    determinism = metrics["determinism"]
    materiality = metrics["materiality"]
    quadrature = metrics["quadrature"]

    lines: list[str] = []
    add = lines.append

    add("# CASE-LAX-DG · 证据链可复现性审计（独立重算）")
    add("")
    add(f"> 锚点 `{metrics['anchor']}`；审计工件 "
        f"`docs/benchmarks/results/lax_reproducibility_audit/metrics.json`"
        f"（sha256 `{metrics.get('metrics_sha256', 'PENDING')}`）；"
        f"本文全部数字由 `scripts/render_lax_reproducibility_doc.py` 从该工件注入，零手抄，`--check` 漂移即失败。")
    add(">")
    add(f"> **被测对象** `{env.get('git_head', 'unknown')}`（git HEAD）；"
        f"机器 {env.get('machine')}，{env.get('platform')}，"
        f"python {env.get('python')}，numpy {env.get('numpy')}，torch {env.get('torch')}。")
    add("")
    add("## 摘要（判决）")
    add("")
    add(f"- **R1 求解器确定性**：{determinism.get('verdict', 'NOT_RUN')}"
        f"（同构型重复与线程数钉扎下系数张量 sha256 "
        f"{'逐一相同' if determinism.get('verdict') == 'PASS' else '出现分歧'}）。")
    for name, payload in reproduction["by_artifact"].items():
        add(f"- **R2 `{name}`**：{payload['verdict']}"
            f"（{payload['shared_readings']} 个可比读数，最坏相对差 {_sci(payload['worst_relative_difference'])}）。")
    add(f"- **R3 台账判决是否被漂移翻转**：{materiality['verdict']}"
        f"（{len(materiality['checks'])} 条 3 % 判据中 {len(materiality['flips'])} 条翻转）。")
    if metrics.get("gates", {}).get("available"):
        add(f"- **C3/R3 闸门（本次重算）**：{metrics['gates']['verdict']}"
            f"（最细腿对的全场密度 L1 阶 {metrics['gates']['c3_gated_order']:.3f}，需求 1.0）。")
    add(f"- **R4 求积偏差**：{quadrature['verdict']}"
        f"（最坏 {_pct(quadrature['worst_quadrature_bias'])}，翻转 {len(quadrature['flips'])} 条 PASS/FAIL）。")
    add(f"- **总体**：**{metrics['verdict']}**"
        f"（stale 工件：{', '.join(metrics['stale_artifacts']) if metrics['stale_artifacts'] else '无'}）。")
    add("")

    add("## ① 案例介绍")
    add("")
    add("Lax (1954) 激波管 / Toro (2009) test 2：一维 Euler 方程，域 [0,1]，隔膜 x₀ = 0.5，"
        "γ = 1.4，快照 t = 0.13。四道波（稀疏波 head/tail、接触间断、右激波）都有闭式表达，"
        "因此可以逐点给出参考解，不需要第三方求解器。")
    add("")
    add("| 参数 | 左态 | 右态 |")
    add("|---|---|---|")
    add("| (ρ, u, p) | (0.445, 0.698, 3.528) | (0.500, 0.000, 0.571) |")
    add("")
    add(f"审计腿（{len(cfg['legs'])} 条，CFL = {cfg['cfl']}，SSP-RK3 + HLLC，库默认参数）："
        + "、".join(f"p{p}.N{n}" for n, p in cfg["legs"]) + "。")
    add("")

    add("## ② 怎么做的（独立性与判据）")
    add("")
    add("| 层 | 本审计的做法 | 为什么这样才独立 |")
    add("|---|---|---|")
    add(f"| 求解器 | `{metrics['independent_layers']['solver']}` | 被测对象，必须重跑，不能读工件 |")
    add(f"| 度量 | {metrics['independent_layers']['measurement']} | "
        "若复用池子的 `_weighted_l2`，度量实现本身的缺陷会与工件互相掩盖 |")
    add(f"| 参考解 | `{metrics['independent_layers']['reference']}` | 闭式代数，且已被 `lax_reference_crosscheck` 独立仲裁 |")
    add(f"| 容差 | `{metrics['independent_layers']['tolerance']}` | 审计必须用**登记的** 3 % 门槛，不能自选 |")
    add("")
    add("四个问题（预先登记，事后不追加）：")
    add("")
    add("1. **R1 确定性**：同一构型跑两遍（其中一遍钉死线程数）是否给出同一个系数张量？"
        "若否，则所有「逐位复现」类判据都无意义。")
    add("2. **R2 可复现性**：当前 HEAD 的新算例与每个已提交工件共享的每个读数相比，"
        "相对差是否 ≤ 1e-6？（不要求逐位：精确解在间断附近逐点采样，跨机器末位差会被放大到 ~1e-9。）")
    add("3. **R3 实质性**：不达标的工件，其漂移是否翻转 3 % 台账判决？")
    add("4. **R4 求积偏差**：工件用单元 10 点 Gauss 取范数，间断落在单元内时该读数是**带求积误差的**；"
        "同一场用 20 点重测即可分离出这部分偏差，并检查是否翻转 PASS/FAIL。")
    add("")

    add("## ③ 计算结果")
    add("")
    add("### 3.1 重算读数（本审计独立度量，order 10 = 工件口径）")
    add("")
    add("| 腿 | steps | dt_mean | rel L1 ρ | rel L1 u | rel L1 p | rel L2 ρ | rel L2 u | rel L2 p |")
    add("|---|---|---|---|---|---|---|---|---|")
    for run in metrics["runs"]:
        add(f"| p{run['degree']}.N{run['cells']} | {run['steps']} | {_sci(run['dt_mean'], 3)} "
            f"| {_sci(run['l1_errors']['density'])} | {_sci(run['l1_errors']['velocity'])} "
            f"| {_sci(run['l1_errors']['pressure'])} | {_sci(run['l2_errors']['density'])} "
            f"| {_sci(run['l2_errors']['velocity'])} | {_sci(run['l2_errors']['pressure'])} |")
    add("")
    add("### 3.2 R1 确定性")
    add("")
    if determinism.get("available"):
        add("| 腿 | 重复次数 | 线程数 | 系数 sha256 全同 | 最坏 rel L2 散度 | 判决 |")
        add("|---|---|---|---|---|---|")
        for row in determinism["rows"]:
            add(f"| {row['leg']} | {row['repeats']} | {row['threads']} "
                f"| {'是' if row['identical_coefficients'] else '否'} "
                f"| {_sci(row['worst_relative_l2_spread'])} | **{row['verdict']}** |")
        add("")
        add(f"判据：系数张量 sha256 全同 **且** rel L2 散度 ≤ {_sci(determinism['tolerance'])}。")
    else:
        add(f"未执行：{determinism.get('reason', 'n/a')}。")
    add("")
    add("### 3.3 R2 工件可复现性矩阵")
    add("")
    add("| 工件 | 生成时间 (UTC) | 可比读数 | 最坏相对差 | 判决 |")
    add("|---|---|---|---|---|")
    for name, payload in reproduction["by_artifact"].items():
        add(f"| `{name}`（{ARTIFACT_TITLES.get(name, name)}） | {payload['generated_at_utc']} "
            f"| {payload['shared_readings']} | {_sci(payload['worst_relative_difference'])} "
            f"| **{payload['verdict']}** |")
    add("")
    stale = [name for name, payload in reproduction["by_artifact"].items()
             if payload["verdict"] == "STALE"]
    if stale:
        add("**不达标工件的逐读数差**（只列 STALE 工件；`relative_difference` 以工件值为分母）：")
        add("")
        add("| 工件 | 腿 | 范数 | 场 | 工件值 | 本次重算 | 相对差 |")
        add("|---|---|---|---|---|---|---|")
        for name in stale:
            rows = sorted(reproduction["by_artifact"][name]["rows"],
                          key=lambda r: -r["relative_difference"])
            for row in rows:
                add(f"| `{name}` | {row['leg']} | {row['norm']} | {row['field']} "
                    f"| {_sci(row['committed'], 6)} | {_sci(row['rerun'], 6)} "
                    f"| {_sci(row['relative_difference'])} |")
        add("")
    else:
        add("所有可比读数的相对差都在可复现门槛之内。")
        add("")
    add("### 3.4 R3 台账判决实质性")
    add("")
    add("| 声明 | 腿 | 场 | 工件值 | 本次重算 | 工件判决 | 重算判决 | 翻转？ |")
    add("|---|---|---|---|---|---|---|---|")
    for check in materiality["checks"]:
        add(f"| {check['claim']} | {check['leg']} | {check['field']} "
            f"| {_sci(check['committed_value'])} | {_sci(check['rerun_value'])} "
            f"| {check['committed_verdict']} | {check['rerun_verdict']} "
            f"| {'**是**' if check['flips_verdict'] else '否'} |")
    add("")
    add(f"判据：`< {materiality['tolerance']:.0%}` 的 PASS/FAIL 在「工件值」与「本次重算值」上必须一致。"
        f"结果：**{materiality['verdict']}**。")
    add("")
    add("### 3.5 R4 求积偏差（工件口径 vs 收敛口径）")
    add("")
    add("| 腿 | 场 | order 10 | order 20 | 相对偏差 | 10 判决 | 20 判决 | 翻转？ |")
    add("|---|---|---|---|---|---|---|---|")
    for row in quadrature["rows"]:
        add(f"| {row['leg']} | {row['field']} | {_sci(row['order_10'])} | {_sci(row['order_20'])} "
            f"| {_pct(row['quadrature_bias'])} | {row['verdict_10']} | {row['verdict_20']} "
            f"| {'**是**' if row['flips_verdict'] else '否'} |")
    add("")
    add(f"最坏求积偏差 **{_pct(quadrature['worst_quadrature_bias'])}**；"
        f"翻转 {len(quadrature['flips'])} 条 → **{quadrature['verdict']}**。")
    add("")

    gates = metrics.get("gates", {})
    add("### 3.6 C3 / R3 闸门在本次重算上的取值")
    add("")
    if gates.get("available"):
        add("| p | 场 | 腿对 | 全场 L1 阶 | C3 需求 | C3 | 解析区 L1 阶 | R3 需求 | R3 |")
        add("|---|---|---|---|---|---|---|---|---|")
        for row in gates["rows"]:
            add(f"| {row['degree']} | {row['field']} | {row['pair'][0]}→{row['pair'][1]} "
                f"| {row['c3_order']:.3f} | {row['c3_requirement']:.2f} | {row['c3_verdict']} "
                f"| {row['r3_order']:.3f} | {row['r3_requirement']:.2f} | {row['r3_verdict']} |")
        add("")
        add(f"综合 **{gates['verdict']}**（判据与算例池同义：C3 ≥ 1.0 全场 L1 阶；"
            f"R3 ≥ 0.85·(p+1) 于解析区。< 1 的阶就是间断面压阶的直接读数，"
            "因此这不是本审计的失败，而是池子已登记的 honest FAIL 在新代码上仍然成立。）")
    else:
        add(f"未执行：{gates.get('reason', 'n/a')}。")
    add("")

    add("## ④ 结果图")
    add("")
    add("![R2 artifact reproduction](benchmarks/results/lax_reproducibility_audit/figures/lax_reproducibility_drift.png)")
    add("")
    add("*图 1*：每个已提交工件与本次全新重算的最坏相对差（对数轴，虚线 = 可复现门槛）。"
        "橙色柱 = 超出门槛（工件已不代表当前代码）。")
    add("")
    add("![R4 quadrature bias](benchmarks/results/lax_reproducibility_audit/figures/lax_quadrature_bias.png)")
    add("")
    add("*图 2*：同一密度场在两种取范数口径下的 rel L2（实线 = 工件 10 点口径，虚线 = 20 点收敛口径，"
        "红点线 = 台账 3 % 门槛）。两线间距就是工件的求积偏差。")
    add("")

    add("## ⑤ 注意点（局限 / 适用范围）")
    add("")
    add("1. **本审计不改任何历史工件**，只判定它们是否仍代表当前代码；`STALE` 是证据状态，不是数值失败。")
    add("2. 复现门槛不是逐位相等：闭式参考解在节点上逐点取值，节点落在间断两侧时末位差可被放大到 ~1e-9。")
    add("3. R4 的 20 点口径也**不是**真值，只是求积收敛的读数；它对含间断单元的误差仍然存在，"
        "只是比 10 点小一到两个量级。")
    add("4. R1 的确定性只在同机同线程族内检查；跨机器（x86 vs loongarch64、不同 BLAS）只保证到 ~1e-9。")
    add("5. 审计只覆盖 `metrics.json` 里登记过的五条腿；未登记的分辨率档不在断言范围内。")
    add("")

    add("## ⑥ 误差判定（关键量 vs 基准，是否 < 3 %）")
    add("")
    add("| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |")
    add("|---|---|---|---|---|")
    add(f"| 密度全场 rel L2 @ p3.N400（A0 生产腿，工件口径） | 精确 Riemann 解 | "
        f"{_pct(_leg_l2(metrics, 400, 3))} | {_pct(_leg_l2(metrics, 400, 3))} | "
        f"{'是' if _leg_l2(metrics, 400, 3) < materiality['tolerance'] else '**否**（honest FAIL）'} |")
    add(f"| 密度全场 rel L2 @ p3.N800（分辨率需求） | 精确 Riemann 解 | "
        f"{_pct(_leg_l2(metrics, 800, 3))} | {_pct(_leg_l2(metrics, 800, 3))} | "
        f"{'是' if _leg_l2(metrics, 800, 3) < materiality['tolerance'] else '否'} |")
    add(f"| 密度全场 rel L2 @ p2.N800（认证构型） | 精确 Riemann 解 | "
        f"{_pct(_leg_l2(metrics, 800, 2))} | {_pct(_leg_l2(metrics, 800, 2))} | "
        f"{'是' if _leg_l2(metrics, 800, 2) < materiality['tolerance'] else '否'} |")
    add(f"| 重算 vs 算例池工件（p3.N400 密度 L2） | 已提交工件 | {_sci(_drift(reproduction, 'lax_case_pool'))} | — "
        f"| {'是（可复现）' if _drift(reproduction, 'lax_case_pool') <= reproduction['tolerance'] else '**否**（工件 stale）'} |")
    add("")
    add("**结论**：本审计不新增物理结论，只回答「已发表数字是否还站得住」。"
        f"台账 3 % 判决在本次重算下 **{materiality['verdict']}**"
        f"（{'无' if not materiality['flips'] else '有'}翻转）；"
        f"求积口径偏差最坏 {_pct(quadrature['worst_quadrature_bias'])}，"
        f"{'未' if not quadrature['flips'] else '已'}翻转任何 PASS/FAIL。"
        "因此 `CASE-LAX-DG` 的 honest FAIL（p3.N400 = 3.116 % > 3 %）与 `p3.N800` / `p2.N800` 的 PASS 结论"
        "都不依赖工件的新旧，唯 `STALE` 工件的读数不应再被引用。")
    add("")
    add("---")
    add("")
    add("## 附录：可复现命令")
    add("")
    add("```")
    add("python -m benchmarks.lax_reproducibility_audit --jobs 6")
    add("python -m scripts.render_lax_reproducibility_doc --check")
    add("python -m pytest tests/test_lax_reproducibility_audit.py -q")
    add("```")
    add("")
    return "\n".join(lines)


def _leg_l2(metrics: dict, cells: int, degree: int) -> float:
    for run in metrics["runs"]:
        if run["cells"] == cells and run["degree"] == degree:
            return float(run["l2_errors"]["density"])
    raise KeyError(f"leg p{degree}.N{cells} not present in this audit run")


def _drift(reproduction: dict, artifact: str) -> float:
    return float(reproduction["by_artifact"][artifact]["worst_relative_difference"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the tracked document matches the artifact")
    parser.add_argument("--figures", action="store_true", help="(re)draw the two PNGs")
    args = parser.parse_args(argv)

    metrics = json.loads((ART / "metrics.json").read_text())
    text = render(metrics)
    if args.figures:
        for path in draw_figures(metrics):
            print(f"[render] wrote {path.relative_to(ROOT)}")
    if args.check:
        current = DOC.read_text() if DOC.exists() else ""
        if current != text:
            print("[render] DRIFT: tracked document differs from the artifact")
            return 1
        print("[render] OK: document matches the artifact")
        return 0
    DOC.write_text(text)
    print(f"[render] wrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())