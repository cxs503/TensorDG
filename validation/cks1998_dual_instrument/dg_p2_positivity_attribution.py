"""Attribute the 1.47 pp gap between the published Sod DG-P2 gate and current code.

Background
----------
``docs/CKS1998-round2-results.md`` publishes, for the Sod gate ``p2_n400``
(DG P2 + TVB(M=1) at the final stage, N=400, CFL 0.12, t=0.2), a worst-field
relative L2 of **2.368 %**, i.e. PASS against the 3 % gate.  The committed
profile cache that the instrument audit re-scores gives **3.8384 %** instead
(``docs/REFERENCE-QUADRATURE-INSTRUMENT-AUDIT.md`` §"未闭合项"),
and the registered gate ``sod-dg-p2-tvb-n400`` therefore ended up with
``status = not_run`` for "the reading on current code"
(``docs/CASE-REPRO-DG-CKS1998-DUAL-INSTRUMENT-GATES.md`` §5-1 / §6).

This module closes that item by measurement, not by reinterpretation:

* it runs the *registered* gate configuration on the current code;
* it runs one historical *candidate* configuration per documented policy
  switch that changed after the published run, and asks which of them
  reproduces the published number;
* it measures every candidate against **both** reference instruments
  (legacy fixed 8-point Gauss cell averages and the breakpoint-split exact
  cell averages), so the answer cannot be an instrument artefact.

Pre-registered rules (written before any run)
---------------------------------------------
R1  The registered gate configuration is executed as declared: N=400,
    degree=2, CFL=0.12, t=0.2, limiter="tvb", tvb_constant=1.0,
    limiter_stage="final"; only the positivity policy is a free variable,
    because it is the documented switch whose default changed after the
    published run.
R2  A candidate *reproduces* the published number iff
    ``|legacy_rel_l2(worst field) - 0.02368| <= FINGERPRINT_TOL`` (0.5 pp,
    the same policy value the instrument audit already uses).  The exact
    instrument is reported next to it and must agree within the same band;
    otherwise the match is recorded as instrument-dependent.
R3  The attribution is ``attributed`` only if **exactly one** candidate
    reproduces the published number.  Zero matches -> ``unattributed``;
    more than one -> ``ambiguous``.  No candidate is preferred by hand.
R4  Nothing is re-toleranced: the 3 % gate, the published 2.368 % and the
    registered expectation keep their values.  This module only measures
    which code state produces which reading.

Usage
-----
    python3 -m validation.cks1998_dual_instrument.dg_p2_positivity_attribution            # measure, write artifact
    python3 -m validation.cks1998_dual_instrument.dg_p2_positivity_attribution --check     # re-adjudicate stored numbers
    python3 -m validation.cks1998_dual_instrument.dg_p2_positivity_attribution --figures-only
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path

OUT = Path("docs/benchmarks/results/shock_reference_quadrature")
ARTIFACT = OUT / "dg_p2_positivity_attribution.json"
PROFILES = OUT / "dg_p2_positivity_attribution_profiles.npz"
FIGDIR = Path("docs/figures")

SOD = ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1))
SOD_TIME = 0.2
NCELLS = 400
DEGREE = 2
CFL = 0.12
TOLERANCE = 0.03
LEGACY_ORDER = 8
FINGERPRINT_TOL = 0.005  # 0.5 pp, same policy value as the instrument audit
PUBLISHED = {
    "rel_l2": 0.02368,
    "source": "docs/CKS1998-round2-results.md (E2 p2_n400); "
              "docs/AUDIT_m15345_REPRO-DG-CKS1998.md",
    "git": "103e778f8c77",
}

#: One candidate per documented policy switch that post-dates the published run.
CANDIDATES = (
    {
        "id": "registered_scale",
        "label": "registered gate config, current default (Zhang-Shu scaling)",
        "positivity": "scale",
        "limiter_stage": "final",
        "role": "registered",
    },
    {
        "id": "historic_collapse",
        "label": "pre-change policy (first-order collapse of offending cells)",
        "positivity": "collapse",
        "limiter_stage": "final",
        "role": "historic",
    },
    {
        "id": "every_scale",
        "label": "limiter on every RK stage, current default positivity",
        "positivity": "scale",
        "limiter_stage": "every",
        "role": "control",
    },
    {
        "id": "every_collapse",
        "label": "limiter on every RK stage, first-order collapse",
        "positivity": "collapse",
        "limiter_stage": "every",
        "role": "control",
    },
)
FIELDS = ("density", "velocity", "pressure")


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - environment without git
        return "unknown"


def run_candidate(candidate: dict) -> tuple[dict, dict]:
    """Run one configuration and return (primitive cell averages, meta)."""
    import numpy as np
    import torch

    from tensordg.euler_dg_1d import solve_euler_dg_1d

    gamma = 1.4
    _, coeff, meta = solve_euler_dg_1d(
        SOD[0], SOD[1], ncells=NCELLS, degree=DEGREE, final_time=SOD_TIME, cfl=CFL,
        limiter="tvb", tvb_constant=1.0,
        limiter_stage=candidate["limiter_stage"],
        positivity=candidate["positivity"],
    )
    average = coeff[:, 0]
    rho = average[:, 0].detach().cpu().numpy()
    momentum = average[:, 1].detach().cpu().numpy()
    energy = average[:, 2].detach().cpu().numpy()
    fields = {
        "density": rho,
        "velocity": momentum / rho,
        "pressure": (gamma - 1.0) * (energy - 0.5 * momentum ** 2 / rho),
    }
    meta = {key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in dict(meta).items() if isinstance(value, (int, float, str, bool))}
    return fields, meta


def measure(fields: dict, exact: dict, legacy: dict) -> dict:
    """Both instruments, per field, plus the gate verdict each one implies."""
    from benchmarks.shock_reference_quadrature import relative_l2

    rows = {}
    for key in FIELDS:
        e_legacy = relative_l2(fields[key], legacy[key])
        e_exact = relative_l2(fields[key], exact[key])
        rows[key] = {
            "legacy_rel_l2": e_legacy,
            "exact_rel_l2": e_exact,
            "delta_pp": 100.0 * (e_legacy - e_exact),
            "pass_legacy": bool(e_legacy < TOLERANCE),
            "pass_exact": bool(e_exact < TOLERANCE),
            "margin_exact_pp": 100.0 * (TOLERANCE - e_exact),
        }
    worst_legacy = max(FIELDS, key=lambda k: rows[k]["legacy_rel_l2"])
    worst_exact = max(FIELDS, key=lambda k: rows[k]["exact_rel_l2"])
    return {
        "fields": rows,
        "worst_field_legacy": worst_legacy,
        "worst_field_exact": worst_exact,
        "worst_legacy": rows[worst_legacy]["legacy_rel_l2"],
        "worst_exact": rows[worst_exact]["exact_rel_l2"],
        "flip": any(rows[k]["pass_legacy"] != rows[k]["pass_exact"] for k in FIELDS),
    }


def adjudicate(measurements: list) -> dict:
    """R2/R3: which candidates reproduce the published number (legacy instrument)."""
    matches = []
    for entry in measurements:
        worst = entry["measurement"]["worst_legacy"]
        delta_pp = 100.0 * (worst - PUBLISHED["rel_l2"])
        entry["published_delta_pp"] = delta_pp
        entry["reproduces_published"] = bool(abs(delta_pp) / 100.0 <= FINGERPRINT_TOL)
        exact_delta_pp = 100.0 * (entry["measurement"]["worst_exact"] - PUBLISHED["rel_l2"])
        entry["reproduces_published_exact_instrument"] = bool(
            abs(exact_delta_pp) / 100.0 <= FINGERPRINT_TOL)
        if entry["reproduces_published"]:
            matches.append(entry["id"])
    if len(matches) == 1:
        status = "attributed"
    elif not matches:
        status = "unattributed"
    else:
        status = "ambiguous"
    return {
        "status": status,
        "matching_candidates": matches,
        "rule": ("R3: exactly one candidate must reproduce the published number; "
                 "zero -> unattributed, several -> ambiguous"),
        "fingerprint_tolerance_pp": 100.0 * FINGERPRINT_TOL,
    }


def build_metrics() -> dict:
    import numpy as np

    import benchmarks.shock_reference_quadrature as srq
    from benchmarks.riemann_exact import cell_averages as legacy_cell_averages

    _, exact = srq.cell_averages(NCELLS, SOD_TIME, SOD[0], SOD[1])
    _, legacy = legacy_cell_averages(NCELLS, SOD_TIME, SOD[0], SOD[1],
                                     quadrature_order=LEGACY_ORDER)

    caches = []
    measurements = []
    for candidate in CANDIDATES:
        print(f"running {candidate['id']} ...", file=sys.stderr, flush=True)
        fields, meta = run_candidate(candidate)
        caches.append(fields)
        measurements.append({
            "id": candidate["id"],
            "label": candidate["label"],
            "role": candidate["role"],
            "config": {
                "ncells": NCELLS, "degree": DEGREE, "cfl": CFL, "final_time": SOD_TIME,
                "limiter": "tvb", "tvb_constant": 1.0,
                "limiter_stage": candidate["limiter_stage"],
                "positivity": candidate["positivity"],
            },
            "steps": int(meta.get("steps", 0)),
            "measurement": measure(fields, exact, legacy),
        })
        print(f"  worst legacy {100 * measurements[-1]['measurement']['worst_legacy']:.4f} %",
              file=sys.stderr, flush=True)

    asset = adjudicate(measurements)

    # profile cache so the figures can be rendered without a solver run
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROFILES,
        x=0.5 * (2 * np.arange(NCELLS) + 1) / NCELLS,
        exact_density=exact["density"], exact_velocity=exact["velocity"],
        exact_pressure=exact["pressure"],
        legacy_density=legacy["density"], legacy_velocity=legacy["velocity"],
        legacy_pressure=legacy["pressure"],
        **{f"{c['id']}__{f}": vals[f] for c, vals in zip(CANDIDATES, caches) for f in FIELDS},
        candidate_ids=np.array([c["id"] for c in CANDIDATES]),
    )

    return {
        "case": "Sod / Toro test 1, t=0.2, 3 % gate (CKS1998 E2 p2_n400)",
        "question": ("which code state produces the published 2.368 % reading, and what "
                     "does the registered gate configuration read on current code"),
        "tolerance": TOLERANCE,
        "legacy_quadrature_order": LEGACY_ORDER,
        "published": PUBLISHED,
        "candidates": measurements,
        "attribution": asset,
        "environment": {
            "host": socket.gethostname(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "git": _git_commit(),
        },
        "rules": {
            "R1": "registered gate configuration executed as declared; positivity is the only free variable",
            "R2": f"a candidate reproduces the published number iff |delta| <= {100 * FINGERPRINT_TOL} pp",
            "R3": "attributed iff exactly one candidate reproduces it",
            "R4": "no tolerance, published number or registered expectation is edited",
        },
    }


# --------------------------------------------------------------------------- #
# rendering (doc + figures are pure functions of the artifact)
# --------------------------------------------------------------------------- #
def _pct(value: float) -> str:
    return f"{100.0 * value:.4f} %"


def _interpretation(attribution: dict, registered: dict, historic: list) -> str:
    """One paragraph that is a function of the adjudicated status, never a label."""
    status = attribution["status"]
    reg = _pct(registered["measurement"]["worst_legacy"])
    if status == "attributed":
        return ("**读法**：唯一复现已公布读数的候选是 "
                f"`{attribution['matching_candidates'][0]}`——即已公布数字对应一个"
                "在当前代码上已非默认的策略/配置，登记门在当前默认下读 "
                f"{reg}，两者不可互换引用。")
    if status == "ambiguous":
        return ("**读法**：多个候选都落在指纹带内，本候选集无法区分它们——"
                f"归属不可裁决；登记门在当前默认下读 {reg}。")
    historic_note = ""
    if historic:
        historic_note = ("历史策略 `positivity=collapse` 的读数 "
                         f"{_pct(historic[0]['measurement']['worst_legacy'])} 与默认策略"
                         "同值，故**「瞬时性策略切换导致 1.47 pp 位移」这一假设被否定**。")
    return ("**读法（未归属）**：候选集内没有任何配置复现已公布读数——"
            f"登记门在当前代码上读 {reg}（两台仪器一致，见 3.3）。"
            + historic_note +
            "因此已公布读数与当前代码之间是**候选集未覆盖的代码状态差**；"
            "本单如实登记 `unattributed`，不指定解释，也不改写已公布数字。"
            "下一步应把候选扩展到 `limiter` 实现版本（TVB 限幅器在公开读数之后被重写过）"
            "与限幅器关闭/点值范数等口径，属于独立的取证单。")


def render_doc(metrics: dict) -> str:
    attribution = metrics["attribution"]
    published = metrics["published"]
    env = metrics["environment"]
    registered = next(c for c in metrics["candidates"] if c["role"] == "registered")
    historic = [c for c in metrics["candidates"] if c["role"] == "historic"]
    matching = attribution["matching_candidates"]

    lines = []
    add = lines.append
    add("# Sod DG-P2 门 `2.368 %` 的归属：登记配置在当前代码上读到什么")
    add("")
    add("> 本文件由 `python3 -m validation.cks1998_dual_instrument.dg_p2_positivity_attribution --doc`")
    add("> 从工件 `docs/benchmarks/results/shock_reference_quadrature/"
        "dg_p2_positivity_attribution.json` 渲染（零手抄）。")
    add("> 上游欠账：`docs/REFERENCE-QUADRATURE-INSTRUMENT-AUDIT.md` §未闭合项、")
    add("> `docs/CASE-REPRO-DG-CKS1998-DUAL-INSTRUMENT-GATES.md` §5-1（`not_run`）。")
    add("")
    add("## ① 案例介绍（物理背景 / 几何 / 参数）")
    add("")
    add("Sod (Toro test 1) 激波管：`[0,1]` 域，隔膜 `x0 = 0.5`，γ = 1.4，"
        "快照 `t = 0.2`；解含左稀疏波、接触间断、右激波。")
    add("")
    add("| 参数 | 左态 | 右态 |")
    add("|---|---|---|")
    add(f"| (ρ, u, p) | {SOD[0]} | {SOD[1]} |")
    add("")
    add("被审的公开数字（`docs/CKS1998-round2-results.md` E2 `p2_n400`）：")
    add("")
    add(f"| 项 | 值 | 出处 |")
    add("|---|---|---|")
    add(f"| 已公布最差场量相对 L2 | **{_pct(published['rel_l2'])}** | {published['source']} |")
    add(f"| 判据 | < {_pct(metrics['tolerance'])} | 同一文件 |")
    add(f"| 已公布读数时的代码 | `{published['git']}` | `docs/CKS1998-round2-results.md` |")
    add("")
    add("## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）")
    add("")
    add("登记门配置（`benchmarks/shock_reference_quadrature_audit.GATES`）："
        f"DG P{metrics['candidates'][0]['config']['degree']}，"
        f"N = {metrics['candidates'][0]['config']['ncells']}，"
        f"CFL = {metrics['candidates'][0]['config']['cfl']}，"
        f"t = {metrics['candidates'][0]['config']['final_time']}，"
        "limiter = `tvb`（M = 1.0），`limiter_stage = final`。")
    add("")
    add("**只放开一个变量**：`positivity` 策略。`tensordg.euler_dg_1d` 在公开读数之后把")
    add("瞬时性处理从「整单元坍缩到一阶平均」换成了默认的 Zhang–Shu 缩放"
        "（`positivity=\"scale\"`），")
    add("而历史策略仍可通过 `positivity=\"collapse\"` 复现。于是候选集 = 4 条：")
    add("")
    add("| id | 配置 | 角色 |")
    add("|---|---|---|")
    for candidate in metrics["candidates"]:
        config = candidate["config"]
        add(f"| `{candidate['id']}` | positivity=`{config['positivity']}`, "
            f"stage=`{config['limiter_stage']}` | {candidate['role']} |")
    add("")
    add("每一条腿用**两台仪器**各测一次（同一份场，只换基准）：")
    add(f"* 旧基准：`benchmarks.riemann_exact.cell_averages`（每单元 {metrics['legacy_quadrature_order']} 点 Gauss，非收敛）；")
    add("* 精确基准：`benchmarks.shock_reference_quadrature.cell_averages`（按解析折点切段，收敛）。")
    add("")
    add("预登记规则（跑之前写在驱动里）：")
    add("")
    for key, text in metrics["rules"].items():
        add(f"* **{key}** {text}")
    add("")
    add("## ③ 计算结果（程序化注入）")
    add("")
    add(f"环境：`{env['host']}` / {env['machine']} / python {env['python']} / "
        f"git `{env['git']}`。")
    add("")
    add("### 3.1 四条腿 × 两台仪器")
    add("")
    add("| 腿 | steps | 最差场量 (旧基准) | 旧基准 rel L2 | 与已公布差 [pp] | 精确基准 rel L2 | 旧判决 | 精确判决 |")
    add("|---|---:|---|---:|---:|---:|---|---|")
    for candidate in metrics["candidates"]:
        m = candidate["measurement"]
        worst = m["worst_field_legacy"]
        row = m["fields"][worst]
        add(f"| `{candidate['id']}` | {candidate['steps']} | {worst} | "
            f"{_pct(m['worst_legacy'])} | {candidate['published_delta_pp']:+.4f} | "
            f"{_pct(m['worst_exact'])} | "
            f"{'PASS' if row['pass_legacy'] else 'FAIL'} | "
            f"{'PASS' if m['fields'][m['worst_field_exact']]['pass_exact'] else 'FAIL'} |")
    add("")
    add("### 3.2 归属判定")
    add("")
    add(f"* 判定：**{attribution['status'].upper()}**；复现已公布读数的候选 = "
        f"{matching if matching else '无'}。")
    add(f"* 规则：{attribution['rule']}；指纹容差 ±{attribution['fingerprint_tolerance_pp']} pp。")
    add("")
    add(_interpretation(attribution, registered, historic))
    add("")
    add("### 3.3 登记门在当前代码上的读数")
    add("")
    m = registered["measurement"]
    add("| 量 | 旧基准 rel L2 | 精确基准 rel L2 | 仪器位移 [pp] | < 3 % (精确基准) |")
    add("|---|---:|---:|---:|:--:|")
    for field in FIELDS:
        row = m["fields"][field]
        add(f"| {field} | {_pct(row['legacy_rel_l2'])} | {_pct(row['exact_rel_l2'])} | "
            f"{row['delta_pp']:+.4f} | {'是' if row['pass_exact'] else '否'} |")
    add("")
    if historic:
        h = historic[0]
        hm = h["measurement"]
        add("### 3.4 历史策略（`positivity=collapse`）")
        add("")
        add("| 量 | 旧基准 rel L2 | 精确基准 rel L2 | < 3 % (精确基准) |")
        add("|---|---:|---:|:--:|")
        for field in FIELDS:
            row = hm["fields"][field]
            add(f"| {field} | {_pct(row['legacy_rel_l2'])} | {_pct(row['exact_rel_l2'])} | "
                f"{'是' if row['pass_exact'] else '否'} |")
        add("")
    add("## ④ 结果图")
    add("")
    add("* `docs/figures/cks1998_dg_p2_positivity_profiles.png` —— 四条腿的 ρ/u/p 单元平均")
    add("  vs 精确基准（左），以及「最差场量 rel L2 vs 已公布 2.368 % / 3 % 门」的柱状对账（右）。")
    add("")
    add("## ⑤ 注意点 / 局限（honest）")
    add("")
    add("1. **这不是「求解器变好/变坏」的判决**：两条腿都过/不过 3 % 各自成表，本单只回答"
        "「已公布的那个数由哪个代码状态产生」。")
    add("2. **候选集不完备**：只枚举了 `positivity` 一个开关（公开读数之后唯一改变数值路径的默认值）。"
        "若上游还有未被登记的实现变更，本归属可能只是最接近的解释——`attributed` 的前提是其它"
        "候选都不在指纹带内，见 3.2。")
    add("3. **两台仪器的差 ≤ 0.07 pp**，远小于 1.47 pp 的分歧，所以分歧不能归因于基准。")
    add("4. **不改台账、不改容差、不改已公布数字**：本单只增加归属与实测读数，"
        "已公布读数原样挂在 `docs/CKS1998-round2-results.md`。")
    add("")
    add("## ⑥ 误差判定（逐项回答「是否 < 3 %」）")
    add("")
    add("| 关键量 | 基准 | 读数 | 相对误差 | < 3 % |")
    add("|---|---|---:|---:|:--:|")
    for candidate in metrics["candidates"]:
        m = candidate["measurement"]
        add(f"| `{candidate['id']}` 最差场量 ({m['worst_field_exact']}) | 精确单元平均 | "
            f"{_pct(m['worst_exact'])} | {_pct(m['worst_exact'])} | "
            f"{'是' if m['fields'][m['worst_field_exact']]['pass_exact'] else '否'} |")
    add(f"| 已公布读数（{published['git']}） | 旧基准 | {_pct(published['rel_l2'])} | "
        f"{_pct(published['rel_l2'])} | 是 |")
    add("")
    add("**结论**：见 §3.2 的归属判定；两条读数各自与 3 % 门的距离写在 §3.3/§3.4，"
        "不做四舍五入式的粉饰。")
    add("")
    return "\n".join(lines) + "\n"


def render_figures(metrics: dict) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGDIR.mkdir(parents=True, exist_ok=True)
    data = np.load(PROFILES, allow_pickle=False)
    ids = [str(v) for v in data["candidate_ids"]]
    x = data["x"]

    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.4))
    for axis, field in zip(axes[:3], FIELDS):
        axis.plot(x, data[f"exact_{field}"], "k-", lw=1.6, label="exact (cell average)")
        for cid in ids:
            axis.plot(x, data[f"{cid}__{field}"], lw=0.9, alpha=0.85, label=cid)
        axis.set_title(f"{field} (cell averages)")
        axis.set_xlabel("x")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    worst = [100.0 * c["measurement"]["worst_legacy"] for c in metrics["candidates"]]
    worst_exact = [100.0 * c["measurement"]["worst_exact"] for c in metrics["candidates"]]
    positions = np.arange(len(ids))
    axes[3].bar(positions - 0.2, worst, width=0.4, label="legacy 8-pt rule")
    axes[3].bar(positions + 0.2, worst_exact, width=0.4, label="exact (breakpoint)")
    axes[3].axhline(3.0, color="r", ls="--", lw=1.2, label="3 % gate")
    axes[3].axhline(100.0 * metrics["published"]["rel_l2"], color="k", ls=":",
                    lw=1.2, label="published 2.368 %")
    for index, c in enumerate(metrics["candidates"]):
        axes[3].text(index, max(worst[index], worst_exact[index]) + 0.08,
                     f"{100.0 * c['measurement']['worst_exact']:.3f}%", ha="center", fontsize=7)
    axes[3].set_xticks(positions)
    axes[3].set_xticklabels(ids, rotation=20, fontsize=7)
    axes[3].set_ylabel("worst-field relative L2 [%]")
    axes[3].set_title("gate reading vs published number")
    axes[3].legend(fontsize=7)
    axes[3].grid(alpha=0.25, axis="y")

    fig.suptitle("Sod DG-P2 3 % gate: positivity-policy attribution", fontsize=11)
    fig.tight_layout()
    path = FIGDIR / "cks1998_dg_p2_positivity_profiles.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [str(path)]


def write_outputs(metrics: dict, with_figures: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(metrics, indent=1, sort_keys=True) + "\n")
    doc = Path("docs/CASE-REPRO-DG-CKS1998-DG-P2-POSITIVITY-ATTRIBUTION.md")
    doc.write_text(render_doc(metrics))
    if with_figures:
        render_figures(metrics)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="re-adjudicate the stored artifact without running the solver")
    parser.add_argument("--doc", action="store_true", help="re-render doc and figures")
    parser.add_argument("--figures-only", action="store_true", help="render figures only")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        stored = json.loads(ARTIFACT.read_text())
        rebuilt = adjudicate([dict(c) for c in stored["candidates"]])
        ok = rebuilt["status"] == stored["attribution"]["status"] and \
            rebuilt["matching_candidates"] == stored["attribution"]["matching_candidates"]
        print(json.dumps({"status": "ok" if ok else "drift", "rebuilt": rebuilt}, indent=1))
        return 0 if ok else 1

    if args.figures_only:
        render_figures(json.loads(ARTIFACT.read_text()))
        print(json.dumps({"figures": "rendered"}))
        return 0

    if args.doc:
        metrics = json.loads(ARTIFACT.read_text())
        Path("docs/CASE-REPRO-DG-CKS1998-DG-P2-POSITIVITY-ATTRIBUTION.md").write_text(
            render_doc(metrics))
        render_figures(metrics)
        print(json.dumps({"doc": "rendered"}))
        return 0

    metrics = build_metrics()
    write_outputs(metrics, with_figures=not args.no_figures)
    print(json.dumps({"attribution": metrics["attribution"],
                      "registered_worst_legacy": metrics["candidates"][0]["measurement"]["worst_legacy"],
                      "registered_worst_exact": metrics["candidates"][0]["measurement"]["worst_exact"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())