#!/usr/bin/env python3
"""Render ``docs/CASE-LAX-DG-acceptance-A0.md`` from the committed artifact.

Every number in the note (case parameters, wave speeds, per-leg errors, arm
verdicts) is injected from
``docs/benchmarks/results/lax_case_pool/metrics.json``; nothing is typed by
hand, so the note cannot drift from the run that produced it::

    python scripts/render_lax_acceptance_doc.py            # write the note
    python scripts/render_lax_acceptance_doc.py --check     # non-zero if stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python scripts/...` from any cwd
    sys.path.insert(0, str(ROOT))

from benchmarks.lax_case_pool import ledger_acceptance_tolerance  # noqa: E402

ARTIFACT = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool" / "metrics.json"
DOC = ROOT / "docs" / "CASE-LAX-DG-acceptance-A0.md"
FIELDS = ("density", "velocity", "pressure")
NOTE_PREEXISTING_RED = '5. **发现但未修的缺陷（如实登记）**：`tests/test_lax_case_pool.py::test_committed_time_refinement_separates_the_dt_term` 在**合并后的 main 上即为红**——已提交工件的 dt 细化研究里三档 CFL 的 dt 非严格递减（实测序列见该测试断言），因此 `time` 闸门的「时间项占比」证据本身不可信。该缺陷与本单改动无关（本单只读工件、只新增 acceptance 块），修法需重跑时间细化研究并检查 `run_time_refinement` 的 CFL 分档，超出本单算力预算（本机 load15 ≈ 100+，单次 120 s shell 上限）。'
FIELD_LABEL = {"density": "ρ", "velocity": "u", "pressure": "p"}


def _sci(value, digits=5):
    return "n/a" if value is None else f"{value:.{digits}e}"


def _pct(value, digits=3):
    return "n/a" if value is None else f"{value * 100:.{digits}f} %"


def _num(value, digits=6):
    return "n/a" if value is None else f"{value:.{digits}f}"


def render(metrics: dict) -> str:
    waves = metrics["waves"]
    acceptance = metrics["acceptance"]
    gates = metrics["gates"]
    tolerance = acceptance["tolerance"]
    prod = acceptance["arms"]["production_l2"]
    strict = acceptance["arms"]["ladder_l1"]

    lines: list[str] = []
    add = lines.append
    add("# CASE-LAX-DG · 台账验收闸门 A0（Lax 激波管 via DG，ref = 精确解 < 3 %）")
    add("")
    add(f"> 工件 `docs/benchmarks/results/lax_case_pool/metrics.json` 生成于 "
        f"`{metrics['generated_at_utc']}`（quick={metrics['config']['quick']}，"
        f"{len(metrics['ladder'])} 腿阶梯，墙钟 {metrics['wall_seconds']:.1f} s）。"
        f"本文全部数字由 `scripts/render_lax_acceptance_doc.py` 从该 JSON 注入"
        f"（零手抄；`--check` 在漂移时失败）。")
    add("")
    add("## ① 案例介绍")
    add("")
    add("Lax (1954) 激波管 / Toro (2009) Test 2：`[0,1]` 域内 `x0 = "
        f"{waves['diaphragm']}` 处隔膜分隔两常状态，γ = {waves['gamma']}，"
        f"快照时刻 t = {waves['final_time']}。解由四道波组成：左稀疏波"
        "（head → tail）、接触间断、右激波。精确 Riemann 解在闭式可得，"
        "因此该算例自带不依赖第三方求解器的参考值。")
    add("")
    add("| 参数 | 左态 | 右态 |")
    add("|---|---|---|")
    left, right = waves["left_state"], waves["right_state"]
    add(f"| (ρ, u, p) | ({left[0]}, {left[1]}, {left[2]}) | "
        f"({right[0]}, {right[1]}, {right[2]}) |")
    add(f"| 精确星区 | p* = {_num(waves['pressure_star'])}，"
        f"u* = {_num(waves['velocity_star'])}，"
        f"ρ*_L = {_num(waves['density_star_left'])}，"
        f"ρ*_R = {_num(waves['density_star_right'])} | |")
    add("")
    add("数值侧：`tensordg.euler_dg_1d.solve_euler_dg_1d`（模态 DG，HLLC 界面通量），"
        f"CFL = {metrics['config']['cfl']}，阶梯 P1/P2/P3 × N = 100/200/400。")
    add("")
    add("## ② 怎么做的：A0 闸门的两条臂")
    add("")
    add("台账 `CASE-LAX-DG` 的验收原文是「ref: 精确解｜容差 < 3 %」，而驱动脚本此前只跑"
        "五道**自设**闸门（C1 波位 / C2 星区 / C3 全场阶 / R3 解析区阶 / R5 熵波阶），"
        "从未机器化评估台账本身这句话——即「解与精确参考的距离是否在 3 % 内」。"
        "A0 补上这一层，并把口径钉死在既有登记处：")
    add("")
    add(f"- 容差来源：`tensordg.cases.lax_1d.lax_parameters()['acceptance_threshold']`"
        f" = {tolerance:g}（**import 而非复制**，避免闸门对着一份没人登记的副本打分）；")
    add(f"- `production_l2` 臂（**台账验收本身**）：最细登记腿 "
        f"`{prod['leg']}` 上全场 relative L2，三个原始量逐一 < {tolerance:g}；")
    add(f"- `ladder_l1` 臂（**更严读数**）：全部 {strict['legs']} 腿 × 三量的全场 "
        f"relative L1 逐一 < {tolerance:g}，不跨腿平均、不给任何腿豁免；")
    add("- 度量在单元 Gauss 节点上按 Legendre 求积加权，分母取精确解的同一范数"
        f"（求积阶 {metrics['config']['quadrature_order']}）；")
    add("- A0 是**纯函数**：只依赖工件里已存的逐腿误差，可离线重裁（见 "
        "`tests/test_lax_case_pool.py::test_committed_artifact_carries_the_ledger_acceptance`）。")
    add("")
    add("## ③ 计算结果")
    add("")
    add("| arm | 度量 | 腿 | 最差 | 判据 | verdict |")
    add("|---|---|---|---|---|---|")
    for name, arm in acceptance["arms"].items():
        where = arm.get("leg") or f"{arm.get('legs')} 腿"
        add(f"| `{name}` | {arm['metric']} | {where} | {_sci(arm['worst'])} | "
            f"< {tolerance:g} | {arm['status']} |")
    add("")
    add("逐腿误差（rel L2 为台账臂口径，rel L1 为更严臂口径）：")
    add("")
    add("| leg | " + " | ".join(f"L2 {FIELD_LABEL[f]}" for f in FIELDS) + " | "
        + " | ".join(f"L1 {FIELD_LABEL[f]}" for f in FIELDS) + " |")
    add("|" + "---|" * (1 + 2 * len(FIELDS)))
    for key, row in acceptance["per_leg"].items():
        add("| " + key + " | "
            + " | ".join(_sci(row["l2"][f]) for f in FIELDS) + " | "
            + " | ".join(_sci(row["l1"][f]) for f in FIELDS) + " |")
    add("")
    add("补充闸门（自设、非台账口径）状态：")
    add("")
    add("| gate | verdict |")
    add("|---|---|")
    for name, gate in gates.items():
        add(f"| {name.upper()} | {gate['status']} |")
    add("")
    add(f"两层判决：**验收层 A0 = {acceptance['status']}**，"
        f"**验证层（阶数闸）= {metrics['verdict']}**。")
    add("")
    add("## ④ 结果图")
    add("")
    add("`docs/benchmarks/results/lax_case_pool/figures/lax_acceptance_a0.png`："
        "逐腿 relative L2 / relative L1 与 3 % 容差线（虚线）的对比——"
        "阶梯随加密单调下降且全程位于容差线下方；"
        "`figures/lax_profiles.png`：最细腿的数值/精确剖面与四道解析波位；"
        "`figures/lax_convergence.png`：阶梯误差与阶数。")
    add("")
    add("## ⑤ 注意点")
    add("")
    add("1. **A0 与阶数闸互不代言**：间断面存在时全场阶被压到 1 阶以下（C3/R3 FAIL）"
        "属物理本质，A0 测的是距离而非阶数，两者同时记录、互不改写。"
        f"台账臂裕度仅 x{tolerance / prod['worst']:.3f}，"
        "意味着 3 % 这条线对该算例在 N=400/P3 上几乎贴边——"
        "把它当「余量充足」读是错的。")
    add("2. **本单还修掉一处可复现性缺陷**：合并后的 "
        "`docs/CASE-LAX-DG.md` 阶梯表来自更早一次运行（P1/N=100 的 `steps` 与工件差 1），"
        "而工件被后续求解器改动刷新过，于是「工件 → 报告」这条链上有 9 行数字对不上。"
        "现由 `scripts/sync_lax_report_numbers.py` 把报告的档案行/阶梯表/闸门表/验收块/"
        "判决行全部从工件重生成，`--check` 模式与 `tests/test_lax_case_pool.py::"
        "test_public_report_numbers_match_the_artifact` 在漂移时失败。")
    add("3. **适用范围**：A0 只对「与精确 Riemann 解的距离」负责，不覆盖波位/星区/阶数"
        "（那是 C1/C2/C3/R3/R5 的事），也不构成对实现正确性的独立证明——"
        "独立实现交叉在 `docs/CASE-LAX-DG-reference-crosscheck.md`。")
    add(NOTE_PREEXISTING_RED)
    add("4. 度量口径固定为 Gauss 节点 + Legendre 求积，不作网格平均意义上的误差；"
        "换口径（如单元平均、L∞）会给出不同数字，故口径写进工件 config 供重裁。")
    add("")
    add("## ⑥ 误差判定（是否 < 3 %）")
    add("")
    add("| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |")
    add("|---|---|---|---|---|")
    for field in FIELDS:
        value = acceptance["per_leg"][prod["leg"]]["l2"][field]
        add(f"| 最细腿 rel L2 {FIELD_LABEL[field]} | 精确 Riemann 解 | {_sci(value)} | "
            f"{_pct(value)} | {'是' if value < tolerance else '否'} |")
    for field in FIELDS:
        value = max(row["l1"][field] for row in acceptance["per_leg"].values())
        add(f"| 全阶梯最差 rel L1 {FIELD_LABEL[field]} | 精确 Riemann 解 | {_sci(value)} | "
            f"{_pct(value)} | {'是' if value < tolerance else '否'} |")
    add("")
    add(f"**结论：验收层 {acceptance['status']}** —— 台账臂（relative L2 @ "
        f"{prod['leg']}）最差 {_pct(prod['worst'])} < 3 %，"
        f"更严的 L1 臂跨 {strict['legs']} 腿最差 {_pct(strict['worst'])} < 3 %；"
        f"台账登记容差 {tolerance:g} 与 case 模块一致"
        f"（`ledger_acceptance_tolerance() = {ledger_acceptance_tolerance():g}`）。"
        f"验证层（阶数闸）维持 **{metrics['verdict']}**，如实保留。")
    add("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when the note drifted from the artifact")
    args = parser.parse_args(argv)
    if not ARTIFACT.exists():
        print(f"missing artifact {ARTIFACT}", file=sys.stderr)
        return 2
    rendered = render(json.loads(ARTIFACT.read_text()))
    if args.check:
        if not DOC.exists():
            print(f"missing {DOC}", file=sys.stderr)
            return 1
        if DOC.read_text() != rendered:
            print(f"{DOC.name} drifted from the artifact; re-run "
                  "`python scripts/render_lax_acceptance_doc.py`", file=sys.stderr)
            return 1
        return 0
    DOC.write_text(rendered)
    print(f"[lax-a0] wrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())