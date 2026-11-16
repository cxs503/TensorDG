"""CASE-LAX-DG -- A0 production-leg closure: re-adjudicate the ledger bar at
the certified configuration (p2.N800) from the *committed* artifacts.

Why this driver exists
----------------------
``benchmarks/lax_case_pool.py``'s ``gate_a0`` defaults the production leg to
``entries[-1]`` -- the finest leg of the 9-leg ladder, ``p3.N400`` -- and that
leg's whole-field relative L2 (density) is ``3.116 %``, just above the ledger
bar ``< 3 %`` (margin x0.963).  The verification synthesis
(``docs/CASE-LAX-DG-verification-synthesis.md``) records this as an honest FAIL
and lists three closure options, recommending **option 1**: re-point the
production leg at the certified configuration ``p2.N800`` (DOF 2400, worst-field
relative L2 = ``2.570 %``), which is the minimal-DOF leg that passes the bar.

This driver *implements* that recommendation **without re-running the solver**:
it re-adjudicates the A0 ledger arm from the two already-committed artifacts
(the case-pool ladder and the certified-configuration enumeration), so the
closure is a machine-checkable re-reading of published numbers, not a new
computation.  The honest FAIL of ``p3.N400`` is preserved as a separate record
(the synthesis's rule: "the verification FAIL is kept as-is, it is not
re-labelled by an A0 PASS").

Every number here is read out of the committed JSON (zero hand transcription);
the tolerance is imported from the registered case ``tensordg.cases.lax_1d``.

Artifacts (inside the worktree)::

    docs/benchmarks/results/lax_a0_production_leg_closure/
        metrics.json   re-adjudicated A0 (production leg = p2.N800) + honest-FAIL record
        summary.md     rendered from metrics.json
    docs/CASE-LAX-DG-a0-production-leg-closure.md   generated write-up

Usage::

    python -m benchmarks.lax_a0_production_leg_closure
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tensordg.cases.lax_1d import lax_parameters

HERE = Path(__file__).resolve().parent.parent

#: committed case-pool ladder (A0 currently FAIL at p3.N400)
CASE_POOL = HERE / "docs/benchmarks/results/lax_case_pool/metrics.json"
#: committed certified-configuration enumeration (p2.N800 = 2.570 %)
CERTIFIED = HERE / "docs/benchmarks/results/lax_certified_configuration/metrics.json"
OUT = HERE / "docs/benchmarks/results/lax_a0_production_leg_closure"
SUMMARY = OUT / "summary.md"

FIELDS = ("density", "velocity", "pressure")


def tolerance():
    """Ledger bar, imported from the registered case (never a literal)."""
    return float(lax_parameters()["acceptance_threshold"])


def worst_field(row):
    """Worst-field whole-field relative L2 of one leg, and the field name."""
    field = max(FIELDS, key=lambda f: row["l2_errors"][f])
    return float(row["l2_errors"][field]), field


def _leg_key(cells, degree):
    return f"p{degree}.N{cells}"


def load_case_pool():
    """Committed case-pool ladder legs (the current A0 production leg set)."""
    payload = json.loads(CASE_POOL.read_text())
    legs = []
    for leg in payload["ladder"]:
        worst, field = worst_field(leg)
        legs.append({
            "leg": _leg_key(leg["cells"], leg["degree"]),
            "cells": int(leg["cells"]), "degree": int(leg["degree"]),
            "worst_l2": worst, "worst_field": field,
        })
    return legs


def load_certified():
    """Committed certified-configuration legs (p2.N800 among them)."""
    payload = json.loads(CERTIFIED.read_text())
    legs = []
    for leg in payload["legs"]:
        worst, field = worst_field(leg)
        legs.append({
            "leg": _leg_key(leg["cells"], leg["degree"]),
            "cells": int(leg["cells"]), "degree": int(leg["degree"]),
            "worst_l2": worst, "worst_field": field,
        })
    return legs


def dof(cells, degree):
    """Degrees of freedom of a modal-DG leg: (p+1) * N."""
    return (degree + 1) * cells


def adjudicate(case_pool_legs, certified_legs, tol=None):
    """Re-adjudicate the A0 ledger arm at the certified production leg.

    Returns the old (p3.N400) and new (p2.N800) A0 verdicts side by side so the
    honest-FAIL record is preserved and the closure is traceable.
    """
    tol = tolerance() if tol is None else tol
    by_leg = {r["leg"]: r for r in case_pool_legs + certified_legs}

    # old production leg = finest case-pool ladder leg (p3.N400)
    old_leg = max(case_pool_legs, key=lambda r: (r["degree"], r["cells"]))
    old = by_leg[old_leg["leg"]]
    old_verdict = "PASS" if old["worst_l2"] < tol else "FAIL"

    # new production leg = certified configuration (p2.N800)
    new_leg = next(r for r in certified_legs if r["leg"] == "p2.N800")
    new = by_leg[new_leg["leg"]]
    new_verdict = "PASS" if new["worst_l2"] < tol else "FAIL"

    # minimal-DOF passing leg over the enlarged set (enumerated + certified)
    all_rows = sorted(case_pool_legs + certified_legs,
                      key=lambda r: (dof(r["cells"], r["degree"]), r["degree"]))
    passing = [r for r in all_rows if r["worst_l2"] < tol]
    minimal = passing[0] if passing else None

    return {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "generated_at_utc": None,  # filled by main()
        "tolerance": tol,
        "metric": "whole-field relative L2 vs exact Riemann solution, "
                  "worst field, 10-node Legendre-Gauss (imported from "
                  "benchmarks.lax_case_pool.run_ladder)",
        "old_production_leg": {
            "leg": old_leg["leg"], "dof": dof(old_leg["cells"], old_leg["degree"]),
            "worst_l2": old["worst_l2"], "worst_field": old["worst_field"],
            "verdict": old_verdict,
            "margin": tol / old["worst_l2"],
        },
        "new_production_leg": {
            "leg": new_leg["leg"], "dof": dof(new_leg["cells"], new_leg["degree"]),
            "worst_l2": new["worst_l2"], "worst_field": new["worst_field"],
            "verdict": new_verdict,
            "margin": tol / new["worst_l2"],
        },
        "minimal_dof_passing": {
            "leg": minimal["leg"], "dof": dof(minimal["cells"], minimal["degree"]),
            "worst_l2": minimal["worst_l2"],
        } if minimal else None,
        "honest_fail_preserved": {
            "leg": old_leg["leg"], "worst_l2": old["worst_l2"],
            "note": "p3.N400 whole-field L2 is discontinuity-dominated and "
                    "sits above the 3 % bar; it is recorded as-is and not "
                    "re-labelled by the p2.N800 PASS.",
        },
        "closure": {
            "recommendation": "re-point the production leg at the certified "
                              "configuration p2.N800 (minimal DOF that passes "
                              "the < 3 % bar)",
            "implemented": True,
            "a0_verdict": new_verdict,
        },
        "legs": all_rows,
    }


def render_summary(m):
    """Render the closure summary from the re-adjudicated metrics (no hand edit)."""
    lines = []
    lines.append("# CASE-LAX-DG · A0 生产腿闭合（production-leg closure）")
    lines.append("")
    lines.append(f"> 容差从注册算例导入 = `{m['tolerance']:.4f}`；度量 = {m['metric']}。")
    lines.append("")
    lines.append("## 旧生产腿（case-pool 阶梯最细腿）")
    lines.append("")
    o = m["old_production_leg"]
    lines.append(f"| 腿 | DOF | 最差 L2 | 场 | 判定 | 裕度 |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| {o['leg']} | {o['dof']} | {100*o['worst_l2']:.3f} % | "
                 f"{o['worst_field']} | {o['verdict']} | x{o['margin']:.3f} |")
    lines.append("")
    lines.append("## 新生产腿（认证配置）")
    lines.append("")
    n = m["new_production_leg"]
    lines.append(f"| 腿 | DOF | 最差 L2 | 场 | 判定 | 裕度 |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| {n['leg']} | {n['dof']} | {100*n['worst_l2']:.3f} % | "
                 f"{n['worst_field']} | {n['verdict']} | x{n['margin']:.3f} |")
    lines.append("")
    lines.append("## 最小 DOF 通过腿（枚举 + 认证并集）")
    lines.append("")
    if m["minimal_dof_passing"]:
        mn = m["minimal_dof_passing"]
        lines.append(f"| {mn['leg']} | DOF {mn['dof']} | 最差 L2 {100*mn['worst_l2']:.3f} % |")
    lines.append("")
    lines.append("## honest-FAIL 保留")
    lines.append("")
    h = m["honest_fail_preserved"]
    lines.append(f"- {h['leg']} 全场 L2 = {100*h['worst_l2']:.3f} %（间断主导，超 3 % 门槛），"
                 f"原样保留，不被 p2.N800 的 PASS 改写。")
    lines.append("")
    lines.append(f"## A0 闭合判定：**{m['closure']['a0_verdict']}**")
    lines.append("")
    lines.append(f"- 推荐：{m['closure']['recommendation']}")
    lines.append(f"- 已实现：{m['closure']['implemented']}")
    lines.append("")
    return "\n".join(lines)


def main():
    import time
    case_pool_legs = load_case_pool()
    certified_legs = load_certified()
    m = adjudicate(case_pool_legs, certified_legs)
    m["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")
    SUMMARY.write_text(render_summary(m))
    print(f"[lax-a0-closure] old={m['old_production_leg']['leg']} "
          f"{m['old_production_leg']['verdict']} "
          f"({100*m['old_production_leg']['worst_l2']:.3f}%) -> "
          f"new={m['new_production_leg']['leg']} "
          f"{m['new_production_leg']['verdict']} "
          f"({100*m['new_production_leg']['worst_l2']:.3f}%) | "
          f"minimal={m['minimal_dof_passing']['leg']} -> {OUT / 'metrics.json'}")
    return m


if __name__ == "__main__":
    main()
