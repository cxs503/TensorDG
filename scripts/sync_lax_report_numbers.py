#!/usr/bin/env python3
"""Bind the numeric blocks of ``docs/CASE-LAX-DG.md`` to the committed artifact.

Why this exists
---------------
The published report claims "zero hand-copy: every number is injected from the
artifact".  That claim is only checkable if a *machine* can compare the two.
It was not: the merged report carried the ladder table of an earlier run
(``steps = 554`` for the P1/N=100 leg) while the committed artifact
``docs/benchmarks/results/lax_case_pool/metrics.json`` held the numbers of the
current solver (``steps = 555``, same leg).  A reader re-running the artifact
would have believed the report was wrong, and a reader trusting the report
would have believed the artifact was wrong.

This script makes the numeric part of the report a pure function of the
artifact: it rewrites the provenance line, the ladder table, the gate table,
the acceptance (A0) block and the verdict lines from
``metrics.json`` and leaves every prose paragraph untouched.

Usage::

    python scripts/sync_lax_report_numbers.py           # rewrite the report
    python scripts/sync_lax_report_numbers.py --check   # non-zero if it drifted
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool" / "metrics.json"
REPORT = ROOT / "docs" / "CASE-LAX-DG.md"

FIELDS = ("density", "velocity", "pressure")


def _sci(value, digits=5):
    return "n/a" if value is None else f"{value:.{digits}e}"


def _num(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value, digits=3):
    return "n/a" if value is None else f"{value * 100:.{digits}f} %"


_BOUNDARY = r"(?=^## |^Verification verdict |\Z)"


def _replace_section(text: str, title: str, body: str) -> str:
    """Replace the body of ``## title`` up to the next heading / verdict paragraph."""
    pattern = re.compile(rf"^## {re.escape(title)}\n.*?{_BOUNDARY}", re.S | re.M)
    replacement = f"## {title}\n\n{body.strip()}\n\n"
    if pattern.search(text):
        return pattern.sub(lambda _m: replacement, text, count=1)
    # new section: keep it above the verdict paragraph when there is one
    anchor = re.search(r"^Verification verdict ", text, re.M)
    if anchor is not None:
        return text[:anchor.start()] + replacement + text[anchor.start():]
    return text.rstrip("\n") + "\n\n" + replacement


def _section_body(text: str, title: str) -> str | None:
    pattern = re.compile(rf"^## {re.escape(title)}\n(.*?){_BOUNDARY}", re.S | re.M)
    match = pattern.search(text)
    return None if match is None else match.group(1)


def provenance(metrics: dict) -> str:
    return (
        f"> 工件 `docs/benchmarks/results/lax_case_pool/metrics.json` 生成于 "
        f"{metrics['generated_at_utc']}（quick={metrics['config']['quick']}，"
        f"{len(metrics['ladder'])} 腿阶梯，墙钟 {metrics['wall_seconds']:.1f} s）。"
        f"本文档案行/阶梯表/闸门表/验收块/判决行全部由 "
        f"`scripts/sync_lax_report_numbers.py` 从该 JSON 注入（零手抄），"
        f"`--check` 模式在文档与工件漂移时失败。"
    )


def ladder_table(metrics: dict) -> str:
    lines = ["| P | N | steps | rel L1 rho | rel L1 u | rel L1 p | "
             "analytic-region rel L1 rho | wall [s] |",
             "|---|---|---|---|---|---|---|---|"]
    for e in metrics["ladder"]:
        lines.append(
            f"| {e['degree']} | {e['cells']} | {e['steps']} | "
            f"{_sci(e['errors']['density'])} | {_sci(e['errors']['velocity'])} | "
            f"{_sci(e['errors']['pressure'])} | "
            f"{_sci(e['analytic_errors']['density'])} | "
            f"{_num(e['wall_seconds'], 1)} |")
    return "\n".join(lines)


def gate_table(metrics: dict) -> str:
    lines = ["| gate | verdict | detail |", "|---|---|---|"]
    for name, gate in metrics["gates"].items():
        if name == "c1":
            detail = "; ".join(f"{k}: {_pct(v['relative_error'])}"
                               for k, v in gate["waves"].items())
        elif name == "c2":
            detail = "; ".join(f"{k}: {_pct(v['relative_error'])}"
                               for k, v in gate["values"].items())
        elif name in ("c3", "r3"):
            detail = "; ".join(f"{k}: {_num(v)}" for k, v in gate["per_leg"].items())
        elif name == "time":
            detail = "; ".join(f"{k}: worst share {_pct(v['worst_share'])}"
                               for k, v in gate["per_leg"].items())
        elif name == "r5":
            detail = "; ".join(
                f"{k}: order {_num(v['observed_order'])} (dev "
                f"{_pct(v['relative_deviation'], 2)})"
                for k, v in gate["cases"].items())
        else:  # pragma: no cover - unknown gate name
            detail = ""
        lines.append(f"| {name.upper()} | {gate['status']} | {detail} |")
    return "\n".join(lines)


def acceptance_block(metrics: dict) -> str:
    acceptance = metrics.get("acceptance")
    if not acceptance:
        return ("A0 not recorded in this artifact: regenerate with "
                "`python -m benchmarks.lax_case_pool`.")
    lines = [
        f"ledger 原文：{acceptance['wording']}（`{acceptance['source']}`）。"
        f"度量：{acceptance['metric']}；范围：{acceptance['regions']}。",
        "",
        "| arm | metric | leg / legs | worst | verdict |",
        "|---|---|---|---|---|",
    ]
    for name, arm in acceptance["arms"].items():
        where = arm.get("leg") or f"{arm.get('legs')} legs"
        lines.append(f"| {name} | {arm['metric']} | {where} | "
                     f"{_sci(arm['worst'])} | {arm['status']} |")
    lines.append("")
    lines.append("| leg | rel L2 rho | rel L2 u | rel L2 p | "
                 "rel L1 rho | rel L1 u | rel L1 p |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, row in acceptance["per_leg"].items():
        lines.append("| " + key + " | "
                     + " | ".join(_sci(row["l2"][f]) for f in FIELDS) + " | "
                     + " | ".join(_sci(row["l1"][f]) for f in FIELDS) + " |")
    lines.append("")
    prod = acceptance["arms"]["production_l2"]
    lines.append(
        f"A0 verdict: **{acceptance['status']}** -- ledger arm (relative L2, "
        f"{prod['leg']}) worst {_sci(prod['worst'])} vs tolerance "
        f"{acceptance['tolerance']:.2e} "
        f"(margin x{acceptance['tolerance'] / prod['worst']:.3f}); 更严的 L1 臂跨 "
        f"{acceptance['arms']['ladder_l1']['legs']} 腿最差 "
        f"{_sci(acceptance['arms']['ladder_l1']['worst'])}。")
    return "\n".join(lines)


def verdict_block(metrics: dict) -> str:
    verdicts = metrics.get("verdicts") or {}
    acceptance = verdicts.get("acceptance", (metrics.get("acceptance") or {}).get("status"))
    return (
        f"Verification verdict (supplementary order gates C1/C2/C3/R3/R5/TIME): "
        f"**{metrics['verdict']}**  \n"
        f"Acceptance verdict (ledger A0, relative L2 vs exact < 3 %): "
        f"**{acceptance}**\n\n"
        "两层判据回答不同问题，分开记录：A0 问「解与精确参考的距离是否在登记容差内」，"
        "阶数闸问「间断存在时族是否达设计阶」。验证层 FAIL 原样保留（honest FAIL），"
        "不被 A0 的 PASS 改写。"
    )


def apply(text: str, metrics: dict) -> str:
    # 1. provenance line (the blockquote paragraph right under the title)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("> 工件 `docs/benchmarks/results/lax_case_pool/"):
            end = index
            while end < len(lines) and lines[end].startswith(">"):
                end += 1
            lines[index:end] = provenance(metrics).splitlines()
            break
    else:  # pragma: no cover - report without the expected provenance line
        lines.insert(2, provenance(metrics))
        lines.insert(3, "")
    text = "\n".join(lines)

    pattern = re.compile(r"^Verification verdict .*?(?=\n## |\Z)", re.S | re.M)
    replacement = verdict_block(metrics) + "\n"
    if pattern.search(text):
        text = pattern.sub(lambda _m: replacement, text, count=1)
    else:  # pragma: no cover
        text = text.rstrip("\n") + "\n\n" + replacement
    text = _replace_section(text, "resolution ladder", ladder_table(metrics))
    text = _replace_section(text, "gates", gate_table(metrics))
    text = _replace_section(text, "acceptance (A0 ledger)", acceptance_block(metrics))
    return text if text.endswith("\n") else text + "\n"


def discrepancies(text: str, metrics: dict) -> list[str]:
    """Everything the report says about numbers that the artifact disagrees with."""
    problems = []
    expected_sections = {
        "resolution ladder": ladder_table(metrics),
        "gates": gate_table(metrics),
        "acceptance (A0 ledger)": acceptance_block(metrics),
    }
    for title, expected in expected_sections.items():
        found = _section_body(text, title)
        if found is None:
            problems.append(f"missing section: {title}")
        elif found.strip() != expected.strip():
            problems.append(f"stale section: {title}")
    if provenance(metrics) not in text:
        problems.append("stale provenance line")
    if verdict_block(metrics).splitlines()[0] not in text:
        problems.append("stale verification verdict line")
    verdicts = metrics.get("verdicts") or {}
    if verdicts.get("acceptance") and (
            f"Acceptance verdict (ledger A0, relative L2 vs exact < 3 %): "
            f"**{verdicts['acceptance']}**" not in text):
        problems.append("stale acceptance verdict line")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when the report drifted from the artifact")
    args = parser.parse_args(argv)
    if not ARTIFACT.exists():
        print(f"missing artifact {ARTIFACT}", file=sys.stderr)
        return 2
    metrics = json.loads(ARTIFACT.read_text())
    text = REPORT.read_text() if REPORT.exists() else "# CASE-LAX-DG\n"
    if args.check:
        problems = discrepancies(text, metrics)
        for problem in problems:
            print(f"[lax-report] {problem}", file=sys.stderr)
        print("[lax-report] artifact: "
              f"{metrics['generated_at_utc']}, verdicts: "
              f"{metrics.get('verdicts')}")
        return 1 if problems else 0
    REPORT.write_text(apply(text, metrics))
    print(f"[lax-report] wrote {REPORT.relative_to(ROOT)} from "
          f"{ARTIFACT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())