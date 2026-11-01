#!/usr/bin/env python3
"""TensorDG upstream-delta & commit-theme auditor.

Read-only git introspection.  Produces a machine-readable record of

  1. the commit delta between a local HEAD and the remote upstream
     (``origin/main``), plus every remote branch that is *not* reachable
     from HEAD (unmerged upstream work -- the highest-value audit target);
  2. the theme evolution of the last ``--recent`` commits, classified by
     touched paths, together with per-commit risk tags.

All numbers in the emitted JSON come straight from ``git log``/``git diff``;
nothing is hand-copied, and no timestamp of "now" is written, so the artifact
is reproducible for a fixed repository state (``git log --pretty`` fields and
``--numstat`` are stable).

Usage
-----
    python3 tools/upstream_delta.py --repo . --out artifacts/upstream_delta.json
    python3 tools/upstream_delta.py --repo . --check artifacts/upstream_delta.json

``--check`` recomputes and compares against an existing artifact, ignoring no
fields: a non-zero exit means the artifact is stale w.r.t. the repository and
must be regenerated.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence

SCHEMA = "tensordg.upstream_delta/1"

# --------------------------------------------------------------------------
# path -> theme rules (ordered: first match wins)
# --------------------------------------------------------------------------
THEME_RULES: Sequence = (
    # Top-level directory of the path decides first: a tutorial markdown under
    # docs/benchmarks/ is documentation, not a benchmark.
    ("ci", (".github/",)),
    ("tests", ("tests/",)),
    ("docs", ("docs/",)),
    ("artifacts", ("artifacts/", "limiter_comparison/", "sod_with_limiter/",
                   "validation_report", "follow_validation_report")),
    # then filename-based heuristics
    ("verification", ("benchmarks/riemann_dg_error", "benchmarks/riemann_exact",
                      "benchmarks/sod_exact", "benchmarks/compare_",
                      "tensordg/converge", "tensordg/run_all_convergence",
                      "tensordg/gen_plots")),
    ("api", ("tensordg/__init__.py",)),
    ("physics_core", ("tensordg/limiters.py", "tensordg/euler_dg_1d.py",
                      "tensordg/euler_1d.py", "tensordg/euler_2d.py",
                      "tensordg/advection_1d.py", "tensordg/advection_2d.py",
                      "tensordg/burgers_1d.py", "tensordg/heat_1d.py",
                      "tensordg/poisson_2d.py", "tensordg/sipg_matrixfree.py",
                      "tensordg/sipg_sparse.py")),
    ("benchmarks", ("benchmarks/",)),
    ("docs", (".md",)),
    ("packaging", ("pyproject.toml", "LICENSE", ".gitignore", "README.md")),
)

# Risk rules: id -> (level, note)
RISK_RULES: Dict[str, tuple] = {
    "R1_DOC_ONLY": ("INFO", "docs/artifacts only: no executable code touched, "
                            "claims must be traced back to an artifact or test"),
    "R2_TESTS_TOUCHED": ("MEDIUM", "test suite touched: must be re-run before "
                                   "trusting the affected capability"),
    "R3_NUMERIC_ARTIFACT": ("HIGH", "machine-readable numeric report rewritten: "
                                    "hand-edited numbers are indistinguishable "
                                    "from computed ones -> regenerate to verify"),
    "R4_API_SURFACE": ("MEDIUM", "tensordg/ public module touched: downstream "
                                 "importers/compat need a check"),
    "R5_PHYSICS_CORE": ("HIGH", "scheme/limiter core touched: accuracy verdict "
                                "requires re-running the exact-solution benchmarks"),
    "R6_MERGE_COMMIT": ("INFO", "merge commit: excluded from churn accounting to "
                                "avoid double counting"),
    "R7_LARGE_CHURN": ("MEDIUM", "large churn (>=400 changed lines): review "
                                 "attention / split into reviewable units"),
    "R8_CI_CONFIG": ("MEDIUM", "CI or packaging changed: green badge no longer "
                               "implies the previous gate set"),
}

LARGE_CHURN = 400


# --------------------------------------------------------------------------
# git helpers
# --------------------------------------------------------------------------
class GitError(RuntimeError):
    pass


def git(repo: str, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", repo, *args],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    if check and proc.returncode != 0:
        raise GitError("git %s failed (%d): %s"
                       % (" ".join(args), proc.returncode, proc.stderr.strip()))
    return proc.stdout


def rev(repo: str, ref: str) -> Optional[str]:
    out = git(repo, "rev-parse", "--verify", "--quiet", ref + "^{commit}", check=False)
    return out.strip() or None


def is_ancestor(repo: str, a: str, b: str) -> bool:
    proc = subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", a, b])
    return proc.returncode == 0


def count(repo: str, rng: str) -> int:
    out = git(repo, "rev-list", "--count", rng, check=False).strip()
    return int(out) if out.isdigit() else 0


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
def theme_of(path: str) -> str:
    for theme, needles in THEME_RULES:
        if any(n in path for n in needles):
            return theme
    return "other"


def is_binary_path(path: str) -> bool:
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".pt", ".npz", ".gif"))


def parse_numstat(repo: str, sha: str) -> List[Dict]:
    """Per-file change record of a (non-merge) commit."""
    files = []
    out = git(repo, "show", "--numstat", "--no-renames", "--format=", sha, check=False)
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        add, dele, path = parts
        files.append({
            "path": path,
            "insertions": None if add == "-" else int(add),
            "deletions": None if dele == "-" else int(dele),
            "binary": add == "-" or is_binary_path(path),
            "theme": theme_of(path),
        })
    return files


def commit_record(repo: str, sha: str, subject: Optional[str] = None,
                  date: Optional[str] = None, parents: int = 0) -> Dict:
    if subject is None or date is None or not parents:
        raw = git(repo, "log", "-1", "--pretty=%h%x1f%ad%x1f%p%x1f%s",
                  "--date=short", sha, check=False).strip()
        short, date, plist, subject = (raw.split("\x1f") + ["", "", "", ""])[:4]
        parents = len(plist.split())
    else:
        short = git(repo, "rev-parse", "--short", sha).strip()
    is_merge = parents > 1
    files = [] if is_merge else parse_numstat(repo, sha)
    churn = sum((f["insertions"] or 0) + (f["deletions"] or 0) for f in files)
    return {
        "sha": sha,
        "short": short,
        "date": date,
        "subject": subject,
        "merge": is_merge,
        "files_changed": len(files),
        "insertions": sum(f["insertions"] or 0 for f in files),
        "deletions": sum(f["deletions"] or 0 for f in files),
        "churn": churn,
        "files": files,
    }


def risk_tags(rec: Dict) -> List[Dict]:
    tags: List[Dict] = []
    paths = [f["path"] for f in rec["files"]]
    themes = {f["theme"] for f in rec["files"]}

    def add(rule: str, detail: str = "") -> None:
        level, note = RISK_RULES[rule]
        tags.append({"rule": rule, "level": level,
                     "note": note + ((" | " + detail) if detail else "")})

    if rec["merge"]:
        add("R6_MERGE_COMMIT")
        return tags
    non_doc = [p for p in paths if theme_of(p) not in ("docs", "artifacts")]
    if paths and not non_doc:
        add("R1_DOC_ONLY")
    if any(p.startswith("tests/") for p in paths):
        add("R2_TESTS_TOUCHED", "%d test file(s)" % sum(p.startswith("tests/") for p in paths))
    numeric = [p for p in paths
               if p.endswith(".json")
               and ("report" in p or theme_of(p) == "artifacts")]
    if numeric:
        add("R3_NUMERIC_ARTIFACT", ", ".join(numeric[:4]))
    if any(p.startswith("tensordg/") for p in paths):
        add("R4_API_SURFACE", "%d module(s)" % sum(p.startswith("tensordg/") for p in paths))
    if "physics_core" in themes:
        add("R5_PHYSICS_CORE", ", ".join(p for p in paths
                                         if theme_of(p) == "physics_core")[:120])
    if rec["churn"] >= LARGE_CHURN:
        add("R7_LARGE_CHURN", "%d lines" % rec["churn"])
    if any(p.startswith(".github/") or p in ("pyproject.toml",) for p in paths):
        add("R8_CI_CONFIG")
    return tags


def dominant_theme(files: Iterable[Dict]) -> str:
    counts: Dict[str, int] = {}
    for f in files:
        if f["theme"] == "docs" or f["theme"] == "artifacts":
            continue
        counts[f["theme"]] = counts.get(f["theme"], 0) + 1
    if not counts:
        for f in files:
            counts[f["theme"]] = counts.get(f["theme"], 0) + 1
    if not counts:
        return "none"
    return max(sorted(counts), key=lambda k: counts[k])


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------
def branch_delta(repo: str, base: str, head: str) -> Dict:
    base_sha, head_sha = rev(repo, base), rev(repo, head)
    if base_sha is None:
        raise GitError("upstream base %r not found (no remote?)" % base)
    mb = git(repo, "merge-base", base_sha, head_sha, check=False).strip() or None
    ahead = git(repo, "log", "--pretty=%H%x1f%h%x1f%ad%x1f%p%x1f%s", "--date=short",
                "%s..%s" % (base_sha, head_sha), check=False).splitlines()
    behind = git(repo, "log", "--pretty=%H%x1f%h%x1f%ad%x1f%p%x1f%s", "--date=short",
                 "%s..%s" % (head_sha, base_sha), check=False).splitlines()

    def expand(lines):
        recs = []
        for line in lines:
            sha, short, date, plist, subject = line.split("\x1f")
            recs.append(commit_record(repo, sha, subject, date, len(plist.split())))
        return recs

    return {
        "base_ref": base,
        "base_sha": base_sha,
        "head_ref": head,
        "head_sha": head_sha,
        "merge_base": mb,
        "identical": base_sha == head_sha,
        "ahead": expand(ahead),
        "behind": expand(behind),
    }


def unmerged_branches(repo: str, head: str, exclude: Sequence[str]) -> List[Dict]:
    """Remote branches whose tip is not reachable from ``head``."""
    out = git(repo, "for-each-ref", "--format=%(refname)", "refs/remotes/origin/",
              check=False)
    head_sha = rev(repo, head)
    result = []
    for ref in out.splitlines():
        if ref.endswith("/HEAD") or ref in exclude:
            continue
        sha = rev(repo, ref)
        if sha is None or is_ancestor(repo, sha, head_sha):
            continue
        commits = [commit_record(repo, l.split("\x1f")[0], l.split("\x1f")[4],
                                 l.split("\x1f")[2], len(l.split("\x1f")[3].split()))
                   for l in git(repo, "log", "--pretty=%H%x1f%h%x1f%ad%x1f%p%x1f%s",
                                "--date=short", "%s..%s" % (head_sha, sha),
                                check=False).splitlines()]
        new_files = [l.split("\t")[-1]
                     for l in git(repo, "diff", "--name-status",
                                  "%s...%s" % (head_sha, sha), check=False).splitlines()]
        touched = [p for p in new_files if p and p != "A" and not p.startswith("A\t")]
        result.append({
            "ref": ref.replace("refs/remotes/", ""),
            "sha": sha,
            "commits_ahead": len(commits),
            "commits": commits,
            "files_touched": len(touched),
            "insertions": sum(c["insertions"] for c in commits),
            "deletions": sum(c["deletions"] for c in commits),
            "themes": sorted({theme_of(p) for p in touched}),
            "risks": [{"rule": "R9_UNMERGED_BRANCH", "level": "HIGH",
                       "note": ("upstream branch not reachable from HEAD: work is "
                                "invisible to main CI/release and will drift")}],
        })
    return sorted(result, key=lambda b: -b["commits_ahead"])


def theme_evolution(records: Sequence[Dict]) -> Dict:
    per_commit = [{"short": r["short"], "date": r["date"],
                   "theme": dominant_theme(r["files"]),
                   "themes": sorted({f["theme"] for f in r["files"]}),
                   "churn": r["churn"], "merge": r["merge"],
                   "subject": r["subject"]} for r in records]
    def tally_of(seq: Sequence[Dict]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in seq:
            out[c["theme"]] = out.get(c["theme"], 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    ordered = [c for c in per_commit if not c["merge"]]
    dates = sorted({c["date"] for c in ordered})
    return {
        "per_commit": per_commit,
        "theme_counts": tally_of(ordered),
        "theme_counts_incl_merge": tally_of(per_commit),
        "merge_ratio": (round(1.0 - len(ordered) / len(per_commit), 3)
                        if per_commit else 0.0),
        "churn_by_theme": _churn_by_theme(records),
        "span_days": len(dates),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
    }


def _churn_by_theme(records: Sequence[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in records:
        if r["merge"]:
            continue
        for f in r["files"]:
            out[f["theme"]] = out.get(f["theme"], 0) + (f["insertions"] or 0) + (f["deletions"] or 0)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _upstream_url(repo: str, remote: str) -> Optional[str]:
    url = git(repo, "config", "--get", "remote.%s.url" % remote, check=False).strip()
    if not url:
        return None
    if url.endswith(".git"):
        url = url[:-4]
    if url.startswith("git@") or url.startswith("ssh://"):
        url = url.split(":", 1)[-1]
    else:
        for host in ("github.com/", "gitlab.com/"):
            if host in url:
                url = url.split(host, 1)[1]
    return url


def build(repo: str, base: str, head: str, recent: int, remote: str) -> Dict:
    delta = branch_delta(repo, base, head)
    log = git(repo, "log", "--pretty=%H%x1f%h%x1f%ad%x1f%p%x1f%s", "--date=short",
              "-n", str(recent), head, check=False).splitlines()
    records = []
    for line in log:
        sha, short, date, plist, subject = line.split("\x1f")
        rec = commit_record(repo, sha, subject, date, len(plist.split()))
        rec["risks"] = risk_tags(rec)
        records.append(rec)

    branches = unmerged_branches(repo, head, exclude=(base.replace("origin/", "refs/remotes/origin/"),))
    themes = theme_evolution(records)

    risk_tally: Dict[str, int] = {}
    for r in records:
        for t in r["risks"]:
            risk_tally[t["level"]] = risk_tally.get(t["level"], 0) + 1
    risk_tally_all = dict(risk_tally)
    for b in branches:
        for t in b["risks"]:
            risk_tally_all[t["level"]] = risk_tally_all.get(t["level"], 0) + 1

    return {
        "schema": SCHEMA,
        "repo": os.path.basename(os.path.abspath(repo)),
        "upstream_url": _upstream_url(repo, remote),
        "remote": remote,
        "head": {"ref": head, "sha": rev(repo, head)},
        "delta": delta,
        "unmerged_upstream_branches": branches,
        "recent_commits": records,
        "theme_evolution": themes,
        "risk_tally_by_level": dict(sorted(risk_tally.items())),
        "risk_tally_by_level_incl_upstream": dict(sorted(risk_tally_all.items())),
        "acceptance": {
            "main_pinned_to_upstream": delta["identical"],
            "unmerged_branch_count": len(branches),
            "unmerged_insertions": sum(b["insertions"] for b in branches),
            "recent_commit_count": len(records),
            "analysis_complete": True,
        },
    }


def _strip(obj):
    return json.loads(json.dumps(obj, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--recent", type=int, default=15)
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", default=None,
                    help="compare freshly computed payload with an existing artifact")
    ns = ap.parse_args(argv)

    payload = build(ns.repo, ns.base, ns.head, ns.recent, ns.remote)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    if ns.check:
        with open(ns.check, encoding="utf-8") as fh:
            old = json.load(fh)
        if _strip(old) != _strip(payload):
            print("STALE: %s differs from current repository state" % ns.check)
            return 1
        print("OK: %s is up to date (head=%s)" % (ns.check, payload["head"]["sha"][:7]))
        return 0

    if ns.out:
        os.makedirs(os.path.dirname(ns.out) or ".", exist_ok=True)
        with open(ns.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        acc = payload["acceptance"]
        print("wrote %s | head=%s | identical_to_upstream=%s | unmerged=%d (%d ins) "
              "| recent=%d commits"
              % (ns.out, payload["head"]["sha"][:7], acc["main_pinned_to_upstream"],
                 acc["unmerged_branch_count"], acc["unmerged_insertions"],
                 acc["recent_commit_count"]))
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())