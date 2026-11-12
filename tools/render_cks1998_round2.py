"""Render the REPRO-DG-CKS1998 round-2 results section from artifacts only.

Zero hand-copied digits: every number in the emitted markdown comes from

* ``validation/repro_cks1998/artifacts/full/metrics.json``      (full profile)
* ``validation/repro_cks1998/artifacts/quick/metrics.json``     (round-1 profile, cross-check)
* ``validation/repro_cks1998/artifacts/sod_l1_ladder/metrics.json``    (P2 Sod L1 ladder)
* ``validation/repro_cks1998/artifacts/sod_l1_ladder_p1/metrics.json`` (P1 Sod L1 ladder)

Usage::

    OMP_NUM_THREADS=1 python -m tools.render_cks1998_round2
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "validation" / "repro_cks1998" / "artifacts"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(x, nd=6):
    return f"{x:.{nd}e}" if isinstance(x, float) else str(x)


def gate_table(profile: str, blob: dict) -> list[str]:
    sec = blob["sections"]
    rows = [f"### {profile} 档（墙钟 {blob.get('wall_seconds')} s）", "",
            "| 门 | 关键量 | 实测 | 判据 | 结果 |", "|---|---|---:|---|---|"]
    for key, item in sec["A1_smooth"]["by_degree"].items():
        rows.append(f"| A1 | {key} 光滑对流 LSQ 阶 | {item['observed_order']:.4f} | "
                    f"≥ {item.get('gate', float('nan')):.2f} | "
                    f"{'PASS' if item['pass'] else 'FAIL'} |")
    for key, item in sec["A2_discontinuous"]["by_degree"].items():
        rows.append(f"| A2 | {key} 方波 L1 阶 | {item['l1_order']:.4f} | "
                    f"≥ {blob.get('gates', {}).get('discont_l1_floor', float('nan')):.2f} | "
                    f"{'PASS' if item['pass'] else 'FAIL'} |")
    for tag, case in sorted(sec["E2_sod"]["cases"].items()):
        rows.append(f"| E2 | {tag} 最差分量相对 L2 | {case['worst_rel_l2']*100:.3f} % | "
                    f"< 3 % | {'PASS' if case['pass'] else 'FAIL'}"
                    f"{'（门）' if case['gate_case'] else '（趋势）'} |")
    for tag, case in sorted(sec["E3_tvd"]["cases"].items()):
        rows.append(f"| E3 | {tag} 点值超调 | {case['overshoot_pointwise_pct']:.3f} % | "
                    f"≤ 1 %（严格策略门） | "
                    f"{'PASS' if case['overshoot_pointwise_pct'] <= 1.0 else 'FAIL'} |")
    rows += ["", f"**{profile} 档 verdict：{'PASS' if blob['verdict'] else 'FAIL'}**", ""]
    return rows


def ladder_table(blob: dict, title: str) -> list[str]:
    rows = [f"### {title}", "",
            f"- degree P{blob['config']['degree']} · cfl={blob['config']['cfl']} · "
            f"tvb M={blob['config']['tvb_constant']} · 求积 {blob['config']['quadrature']} 点 · "
            f"墙钟 {blob['wall_seconds']} s · git `{blob['git_commit'][:12]}`",
            "",
            "| N | steps | L1 ρ | L1 u | L1 p | 单元平均相对 L2 最差 | 密度超调 |",
            "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in blob["rungs"]:
        rows.append(f"| {r['ncells']} | {r['steps']} | {r['l1_rel_density']:.4e} | "
                    f"{r['l1_rel_velocity']:.4e} | {r['l1_rel_pressure']:.4e} | "
                    f"{r['l2_cellavg_worst']:.4e} | {r['overshoot_density']:.2e} |")
    c = blob["control"]
    rows.append(f"| {c['ncells']}⁰ | {c['steps']} | {c['l1_rel_density']:.4e} | "
                f"{c['l1_rel_velocity']:.4e} | {c['l1_rel_pressure']:.4e} | "
                f"{c['l2_cellavg_worst']:.4e} | {c['overshoot_density']:.2e} |")
    rows += ["", "⁰ = unlimited control run（同网格，关闭限幅）", ""]
    cells = [r["ncells"] for r in blob["rungs"]]
    header = " | ".join(f"{a}→{b}" for a, b in zip(cells[:-1], cells[1:]))
    rows += ["| 观测阶 | " + header + " |", "|---|" + "---:|" * (len(cells) - 1)]
    for key in ("l1_rel_density", "l1_rel_velocity", "l1_rel_pressure", "l1_rel_worst",
                "l2_cellavg_worst"):
        rows.append(f"| {key} | " + " | ".join(f"{o['order']:.3f}"
                                               for o in blob["orders"][key]) + " |")
    rows += ["", "| 判据 | 结果 |", "|---|---|"]
    for key in blob["checks"]:
        rows.append(f"| {key} | **{'PASS' if blob['checks'][key] else 'FAIL'}** |")
    rows += ["", f"**{title} verdict：{'PASS' if blob['verdict'] else 'FAIL'}**", ""]
    if blob.get("cross_check_round1"):
        rows += ["| round-1 交叉核对 | 用例 | N | 最大相对差 |", "|---|---|---:|---:|"]
        for item in blob["cross_check_round1"]:
            rows.append(f"| {item['profile']} | {item['tag']} | {item['ncells']} | "
                        f"{item['max_rel_diff']:.2e} |")
        rows.append("")
    return rows


def main() -> int:
    quick = load(ART / "quick" / "metrics.json")
    full = load(ART / "full" / "metrics.json")
    p2 = load(ART / "sod_l1_ladder" / "metrics.json")
    p1 = load(ART / "sod_l1_ladder_p1" / "metrics.json")

    out = ["# REPRO-DG-CKS1998 round 2 —— 结果段（程序生成，零手抄）", "",
           "> 本文件由 `python -m tools.render_cks1998_round2` 从工件 JSON 生成；",
           "> 任何数字都不允许在此文件里手工改写。判据预登记见",
           "> `docs/REPRO-DG-CKS1998-round2.md` §2。", ""]
    if quick:
        out += gate_table("quick", quick)
    if full:
        out += gate_table("full", full)
    if quick and full:
        q_sod = quick["sections"]["E2_sod"]["cases"]
        f_sod = full["sections"]["E2_sod"]["cases"]
        out += ["### quick → full 关键量对照", "",
                "| 用例 | quick 最差分量相对 L2 | full 最差分量相对 L2 |", "|---|---:|---:|"]
        for tag in sorted(set(q_sod) & set(f_sod)):
            out.append(f"| {tag} | {q_sod[tag]['worst_rel_l2']*100:.3f} % | "
                       f"{f_sod[tag]['worst_rel_l2']*100:.3f} % |")
        for tag in sorted(set(f_sod) - set(q_sod)):
            out.append(f"| {tag} | — | {f_sod[tag]['worst_rel_l2']*100:.3f} % |")
        out += ["", f"- quick verdict：{'PASS' if quick['verdict'] else 'FAIL'} · "
                f"full verdict：{'PASS' if full['verdict'] else 'FAIL'}", ""]
    if p2:
        out += ladder_table(p2, "Sod L1 阶梯 · P2")
    if p1:
        out += ladder_table(p1, "Sod L1 阶梯 · P1")
    target = REPO / "docs" / "CKS1998-round2-results.md"
    target.write_text("\n".join(out))
    print(f"wrote {target} ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())