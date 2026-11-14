#!/usr/bin/env python3
"""Survey open-source / literature sources for CASE-DG-ADVECT-ORDER (round 2).

The mission asked for an ``oss_inventory``-style query
(``feature_slug ILIKE '%ORDER%'``) against the cluster OSS cache, followed by a
literature/method-genealogy survey.  No ``oss_inventory`` relation exists in this
environment (see ``PROBES`` below, recorded so the negative result is
reproducible rather than asserted), so the same filter is applied
programmatically to the two sources that *are* available:

1. the repository's own DG OSS registry
   (``docs/oss_inventory_dg_registry.json``, 24 repos, built by
   ``tools/build_oss_inventory_dg.py`` from a raw GitHub API probe), and
2. the on-disk OSS cache (``~/oss-cache``), scanned for source files that mix
   advection with explicit Runge-Kutta machinery -- i.e. the code that would be
   a "same feature" implementation to learn from.

Everything numeric (repo counts, file counts, star counts) is computed here, so
the report can quote this JSON instead of hand-typed numbers.

Usage::

    python3 tools/survey_advorder_oss.py            # writes json + md fragment
    python3 tools/survey_advorder_oss.py --check    # exit 1 if stale
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "oss_inventory_dg_registry.json"
OUT_JSON = ROOT / "docs" / "advorder_oss_survey.json"
OUT_MD = ROOT / "docs" / "advorder_oss_survey.md"
CACHE = pathlib.Path.home() / "oss-cache"

#: Keywords that make a repo/file "the same feature" as CASE-DG-ADVECT-ORDER:
#: a high-order advection accuracy study needs (a) an explicit RK family with a
#: tuned stage count and (b) a convergence/verification harness.
FEATURE_KEYWORDS = ("order", "accuracy", "convergence", "verification",
                    "time-integration", "ssp", "strong-stability")

#: Source patterns that identify "advection + explicit RK" code in the cache.
ADVECTION = re.compile(r"\badvection\b", re.I)
RK = re.compile(r"(ssp|strong[ _-]?stability|runge|rk4|rk45|tableau|"
                r"butcher|time[_ ]?integrator)", re.I)
DG = re.compile(r"(\bdg\b|discontinuous[ _-]?galerkin)", re.I)
SOURCE_SUFFIXES = {".py", ".cpp", ".cc", ".hpp", ".h", ".f90", ".f", ".jl", ".lua", ".c"}

#: Probes that establish *why* the SQL route was not used.  Each entry is a
#: command and its recorded outcome; they are re-run by this script so the
#: claim "no oss_inventory table here" is reproducible.
PROBES = [
    ("info_schema_scan",
     "for db in $(psql -lAt | cut -d'|' -f1); do psql $db -Atc "
     "\"select count(*) from information_schema.tables where table_name in "
     "('oss_inventory','oss_solutions')\"; done"),
    ("oss_cache_listing", "ls ~/oss-cache"),
]


def _run(cmd: str) -> str:
    try:
        out = subprocess.run(["bash", "-lc", cmd], capture_output=True,
                             text=True, timeout=120)
        return (out.stdout or "").strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"<error: {exc}>"


def probe_environment() -> dict:
    """Re-run the environment probes (no invoking psql if it is absent)."""
    have_psql = _run("command -v psql || true") != ""
    relations = 0
    if have_psql:
        counts = _run(
            "for db in $(psql -lAt 2>/dev/null | cut -d'|' -f1); do "
            "psql \"$db\" -Atc \"select count(*) from information_schema.tables "
            "where table_name in ('oss_inventory','oss_solutions')\" 2>/dev/null; done")
        relations = sum(int(x) for x in counts.split() if x.isdigit())
    return {
        "psql_present": have_psql,
        "oss_inventory_or_solutions_relations_found": relations,
        "sql_route_available": relations > 0,
        "note": ("no oss_inventory/oss_solutions relation exists in the reachable "
                 "databases -> the mission's SQL filter is emulated over the "
                 "in-repo DG registry + the on-disk OSS cache below"),
    }


def survey_registry() -> dict:
    payload = json.loads(REGISTRY.read_text())
    repos = payload["repos"]
    hits = []
    for repo in repos:
        haystack = " ".join([
            str(repo.get("repo", "")), str(repo.get("dg_role", "")),
            str(repo.get("focus", "")), " ".join(repo.get("capabilities", [])),
        ]).lower()
        matched = [k for k in FEATURE_KEYWORDS if k in haystack]
        if matched:
            hits.append({"repo": repo["repo"], "tier": repo.get("tier"),
                         "matched_keywords": matched,
                         "capabilities": repo.get("capabilities", []),
                         "stars": repo.get("stars"),
                         "tensordg_gaps": repo.get("tensordg_gaps", [])})
    return {"registry_path": str(REGISTRY.relative_to(ROOT)),
            "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
            "repos_total": len(repos), "feature_matches": len(hits),
            "matches": hits}


def survey_cache() -> dict:
    repos = []
    if not CACHE.is_dir():
        return {"cache_path": str(CACHE), "present": False, "repos": []}
    for repo in sorted(p for p in CACHE.iterdir() if p.is_dir()):
        adv, rk, dg, dg_adv = [], [], [], []
        for path in sorted(repo.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:  # pragma: no cover - unreadable file
                continue
            rel = str(path.relative_to(CACHE))
            a = bool(ADVECTION.search(text))
            r = bool(RK.search(text))
            d = bool(DG.search(text))
            if a:
                adv.append(rel)
            if r:
                rk.append(rel)
            if d:
                dg.append(rel)
            if a and r and d:
                dg_adv.append(rel)
        repos.append({
            "repo": repo.name,
            "source_files_scanned": sum(
                1 for p in repo.rglob("*")
                if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES),
            "advection_files": len(adv),
            "explicit_rk_files": len(rk),
            "dg_files": len(dg),
            "dg_advection_rk_files": len(dg_adv),
            "dg_advection_rk_examples": dg_adv[:5],
            "advection_examples": adv[:5],
        })
    return {"cache_path": str(CACHE), "present": True, "repos": repos,
            "same_feature_implementations": sum(r["dg_advection_rk_files"]
                                                for r in repos)}


def literature() -> list[dict]:
    """Method genealogy (hand-curated *citations*, no numeric claims)."""
    return [
        {"item": "RKDG original stability/accuracy theory",
         "source": "B. Cockburn, C.-W. Shu, Math. Comp. 52 (1989) 411-435",
         "role": "smooth-solution L2 order p+1 for DG(p) with an order-(p+1) RK"},
        {"item": "RKDG with TVD/SSP time stepping, limiter theory",
         "source": "B. Cockburn, C.-W. Shu, J. Comput. Phys. 141 (1998) 199-224",
         "role": "the (p+1)-stage convention used by every RKDG accuracy table"},
        {"item": "Stage-reduced inner operators (the reference table used here)",
         "source": ("Z. Sun, Math. Comp. (2024), DOI 10.1090/mcom/4037; "
                    "arXiv:2404.15453v2, table tab:acc-sin-1d-uni"),
         "role": "external numeric reference: 1D sin(2*pi*x), t=1, N=20..320"},
        {"item": "Order barrier for explicit RK",
         "source": "J. C. Butcher (1965); see Hairer-Wanner, Solving ODEs II, "
                   "Sec. II.5 (order 5 needs at least 6 stages)",
         "role": "explains why column RK5DG4 is 5-stage/4th-order"},
        {"item": "5-stage 4th-order SSP scheme (SSP-RK(5,4))",
         "source": ("R. J. Spiteri, S. J. Ruuth, SIAM J. Numer. Anal. 40 (2002) "
                    "469-491, eq. (3.5); D. Ketcheson, SIAM J. Sci. Comput. 30 "
                    "(2008) 2113-2135"),
         "role": "the tableau implemented in tensordg/runge_kutta.py::SSP_RK54"},
        {"item": "Method of manufactured solutions / exact-solution reference",
         "source": "P. J. Roache, Verification and Validation in Computational "
                   "Science and Engineering (2002), Ch. 5",
         "role": "the acceptance criterion: observed order vs reference column"},
    ]


def build() -> dict:
    payload = {
        "feature_slug": "CASE-DG-ADVECT-ORDER",
        "generated_by": "tools/survey_advorder_oss.py",
        "environment": probe_environment(),
        "registry": survey_registry(),
        "cache": survey_cache(),
        "literature": literature(),
        "selection": {
            "chosen": ("stage-matched RKDG: (p+1)-stage explicit RK per degree; "
                       "SSP-RK(5,4) at p=4; observed order compared with the "
                       "reference column using one shared estimator"),
            "rejected": [
                {"option": "RK4 for every degree (round 1)",
                 "reason": "stage count does not match the reference column; at "
                           "p=4 the time integrator caps the observed order"},
                {"option": "6-stage order-5 Butcher RK at p=4 (round 2 default)",
                 "reason": "order matched but the stage count did not; the "
                           "reference column has 5 stages"},
                {"option": "claim order = stages = p+1 at p=4",
                 "reason": "impossible: explicit 5-stage methods cannot reach "
                           "order 5 (Butcher barrier)"},
            ],
            "migration": [
                "tensordg/runge_kutta.py: add SSP_RK54 (derived from the Shu-Osher "
                "form) + MAX_ORDER_FOR_STAGES; tableau_for_degree selects by stage count",
                "tensordg/advection_1d.py: expose time_integrator='ssp_rk54'",
                "benchmarks/case_dg_advect_order_matched_rk.py: stage-matching M2 "
                "criterion + fixed-sign time-order probe (M5) instead of not_run",
                "tests: tableau order conditions, stage-matched selection, probe sign",
            ],
        },
    }
    return payload


def render_md(payload: dict) -> str:
    env, reg, cache = payload["environment"], payload["registry"], payload["cache"]
    lines = ["# CASE-DG-ADVECT-ORDER — OSS / literature survey (round 2)", "",
             "> Generated by `tools/survey_advorder_oss.py`; every number below is",
             "> computed from the registry JSON / the on-disk cache, none is typed by hand.", "",
             "## 1. Environment probe (why the SQL route was emulated)", "",
             f"- `oss_inventory`/`oss_solutions` relations found: **{env['oss_inventory_or_solutions_relations_found']}**",
             f"- SQL route available: **{env['sql_route_available']}**",
             f"- note: {env['note']}", "",
             "## 2. DG OSS registry filter (`feature ~ order|accuracy|convergence|verification|time-integration|ssp`)", "",
             f"- repos in registry: **{reg['repos_total']}**, matching: **{reg['feature_matches']}**", "",
             "| repo | tier | stars | matched keywords | caps |", "|---|---:|---:|---|---|"]
    for m in reg["matches"]:
        lines.append(f"| `{m['repo']}` | {m['tier']} | {m['stars']} | "
                     f"{', '.join(m['matched_keywords'])} | {', '.join(m['capabilities'])} |")
    lines += ["", "## 3. On-disk OSS cache scan (advection + explicit RK + DG)", ""]
    if not cache["present"]:
        lines.append(f"- cache `{cache['cache_path']}` not present")
    else:
        lines += [f"- cache: `{cache['cache_path']}`", "",
                  "| repo | files scanned | advection | explicit RK | DG | DG+advection+RK |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in cache["repos"]:
            lines.append(f"| `{r['repo']}` | {r['source_files_scanned']} | "
                         f"{r['advection_files']} | {r['explicit_rk_files']} | "
                         f"{r['dg_files']} | {r['dg_advection_rk_files']} |")
        lines += ["", f"- same-feature implementations found: "
                      f"**{cache['same_feature_implementations']}**"]
        if cache["same_feature_implementations"] == 0:
            lines += ["", "**Honest negative result**: the local OSS cache holds no "
                          "DG + advection + explicit-RK order study to copy from, so "
                          "the method came from the literature genealogy below and from "
                          "the repository's own solver."]
    lines += ["", "## 4. Method genealogy (citations only)", "",
              "| item | source | role in this case |", "|---|---|---|"]
    for lit in payload["literature"]:
        lines.append(f"| {lit['item']} | {lit['source']} | {lit['role']} |")
    sel = payload["selection"]
    lines += ["", "## 5. Selection (sedimented to `oss_solutions`)", "",
              f"**Chosen**: {sel['chosen']}", "", "**Rejected alternatives**:", ""]
    for rej in sel["rejected"]:
        lines.append(f"- `{rej['option']}` — {rej['reason']}")
    lines += ["", "**Migration plan**:", ""]
    for step in sel["migration"]:
        lines.append(f"- {step}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = build()
    text_json = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    text_md = render_md(payload)
    if args.check:
        ok = (OUT_JSON.exists() and OUT_JSON.read_text() == text_json
              and OUT_MD.exists() and OUT_MD.read_text() == text_md)
        print("stale" if not ok else "up to date")
        return 0 if ok else 1
    OUT_JSON.write_text(text_json, encoding="utf-8")
    OUT_MD.write_text(text_md, encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())