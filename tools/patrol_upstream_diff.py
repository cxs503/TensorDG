#!/usr/bin/env python3
"""Upstream-diff patrol for TensorDG: what actually landed, and what is at risk.

Motivation (ecosystem patrol mission m17042)
-------------------------------------------
The repository advances through *squash merges of agent branches*
(``Merge #NNN (mXXXXX) via merge-master``).  Reading ``git log --stat`` on a
merge commit is misleading for two independent reasons that this tool removes:

1. **Combined-diff blindness.**  ``git show <merge>`` prints a *combined* diff
   for a true merge, which is empty whenever the merge is clean.  A mission
   therefore looks like "0 files changed" even though it delivered thousands of
   lines.  The tool always diffs against the **first parent**
   (``git diff <sha>^1 <sha>``), which is what "what did this mission add to
   main" actually means.
2. **Genuinely empty landings.**  The opposite case is real as well: the merge
   gate filters files out of agent branches (commit messages read
   ``N 个文件被门禁滤除``), and some landings end up **content-empty** (tree of
   the landing commit == tree of its parent).  Those must be separated from (1)
   instead of being silently counted as "merged".

Second question: the remote keeps only ``origin/main`` for most of the
ecosystem's ``agent/*`` branches (the rest are deleted after landing).  The
local checkout still carries hundreds of ``agent/*`` refs; some of them hold
files that are **absent from upstream** (typically binaries under
``validation/**/figures/`` that the gate drops).  The tool classifies every
branch delta file as

``ABSENT_ON_UPSTREAM``  the path does not exist upstream at all
``DIFFERS``             the path exists upstream with a different blob
``LANDED``              upstream carries the exact same blob

so that "already merged" and "work that never arrived" are distinguishable
without trusting patch-ids (a squash-merged/re-formatted commit changes its
patch-id, so ``git cherry`` reports false "unlanded" for correctly merged work).

Usage::

    python tools/patrol_upstream_diff.py                    # census (JSON on stdout)
    python tools/patrol_upstream_diff.py --write            # + JSON/Markdown/CSV in out dir
    python tools/patrol_upstream_diff.py --limit 30         # wider commit window
    python tools/patrol_upstream_diff.py --no-branches      # skip the (slow) branch audit
    python tools/patrol_upstream_diff.py --check            # non-zero exit if empty landings exist

The tool never touches the network and never mutates refs: it only reads what
is already fetched (``--fetch`` is deliberately not offered; run
``git fetch --prune`` yourself if you want fresh upstream state).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MERGE_RE = re.compile(r"^Merge #(\d+) \((m\d+)\) via ([\w-]+)$")
FILTER_HINT = ("门禁滤除", "filtered")

#: top-level path -> theme bucket.  Anything unmatched falls back to ``other``.
THEME_BUCKETS = (
    ("tensordg/", "core-solver"),
    ("tests/", "tests"),
    ("benchmarks/", "benchmarks"),
    ("validation/", "validation-artifacts"),
    ("runs/", "run-artifacts"),
    ("artifacts/", "run-artifacts"),
    ("docs/", "docs"),
    ("tools/", "tooling"),
    ("scripts/", "tooling"),
    ("scratch/", "scratch"),
)

BINARY_ASSET_SUFFIXES = (".png", ".pdf", ".jpg", ".jpeg", ".gif", ".svg", ".npz", ".h5", ".vtk")

#: single-file churn (insertions+deletions) above which an artifact is "big"
BIG_ARTIFACT_LINES = 2000


def git(repo: Path, *args: str, check: bool = True) -> str:
    """Run git and return stdout (text).  ``check=False`` returns '' on failure."""
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True
    )
    if proc.returncode != 0:
        if check:
            raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return ""
    return proc.stdout


def resolve_upstream(repo: Path, requested: str | None) -> tuple[str, str]:
    """Return (upstream_ref, how_it_was_resolved)."""
    if requested:
        if not git(repo, "rev-parse", "--verify", "--quiet", f"{requested}^{{commit}}", check=False).strip():
            raise SystemExit(f"upstream ref {requested!r} not found")
        return requested, "explicit"
    # 1) the current branch's upstream, if any
    up = git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", check=False).strip()
    if up and git(repo, "rev-parse", "--verify", "--quiet", f"{up}^{{commit}}", check=False).strip():
        return up, "branch_upstream"
    # 2) origin/HEAD
    head = git(repo, "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD", check=False).strip()
    if head:
        return head.replace("refs/remotes/", ""), "origin_HEAD"
    # 3) exclusions
    for cand in ("origin/main", "origin/master"):
        if git(repo, "rev-parse", "--verify", "--quiet", f"{cand}^{{commit}}", check=False).strip():
            return cand, "fallback_candidate"
    return "", "absent"


def diff_set(repo: Path, head: str, upstream: str) -> dict:
    """Commits that exist on only one side, plus the empty-upstream verdict."""
    raw = git(repo, "rev-list", "--left-right", "--count", f"{head}...{upstream}")
    ahead, behind = (int(x) for x in raw.split()[:2])
    only_local = [
        line for line in git(repo, "rev-list", f"{upstream}..{head}").splitlines() if line
    ]
    only_upstream = [
        line for line in git(repo, "rev-list", f"{head}..{upstream}").splitlines() if line
    ]
    verdict = "EMPTY" if ahead == 0 and behind == 0 else ("DIVERGED" if ahead and behind else ("AHEAD" if ahead else "BEHIND"))
    return {
        "ahead": ahead,
        "behind": behind,
        "verdict": verdict,
        "only_local": only_local,
        "only_upstream": only_upstream,
    }


def parse_subject(subject: str) -> dict:
    m = MERGE_RE.match(subject.strip())
    if not m:
        return {"is_landing": False, "pr": None, "mission": None, "channel": None}
    return {
        "is_landing": True,
        "pr": int(m.group(1)),
        "mission": m.group(2),
        "channel": m.group(3),
    }


def first_parent_numstat(repo: Path, sha: str) -> tuple[list[tuple[int, int, str]], bool]:
    """(rows, has_parent) for sha vs its first parent.

    ``has_parent=False`` marks a root commit: its empty numstat is *not* an
    empty landing (nothing to land on), so the two cases must stay distinct.
    """
    parents = git(repo, "rev-list", "--parents", "-n", "1", sha).split()[1:]
    if not parents:
        return [], False
    out = git(repo, "diff", "--numstat", "--no-renames", parents[0], sha, check=False)
    rows: list[tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        ins, dele, path = parts
        rows.append((-1 if ins == "-" else int(ins), -1 if dele == "-" else int(dele), path))
    return rows, True


def plumbing_numstat(repo: Path, sha: str) -> list[tuple[int, int, str]]:
    """What a *plumbing* reading (``git diff-tree`` without ``-m``) reports.

    Kept only as a baseline.  ``git diff-tree`` prints **no diff at all** for a
    merge commit unless ``-m``/``-c``/``--cc`` is given, while ``git show``
    prints the first-parent diff.  Any census built on the plumbing form
    therefore loses every true merge silently -- measured below as
    ``plumbing_read_blind_commits``.
    """
    out = git(repo, "diff-tree", "--numstat", "--no-commit-id", "-r", sha, check=False)
    rows: list[tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        ins, dele, path = parts
        rows.append((-1 if ins == "-" else int(ins), -1 if dele == "-" else int(dele), path))
    return rows


def theme_of(path: str) -> str:
    for prefix, theme in THEME_BUCKETS:
        if path.startswith(prefix):
            return theme
    if "/" not in path and path.endswith((".py", ".sh")):
        return "loose-scripts"
    return "other"


def commit_risks(subject: str, parsed: dict, numstat: list[tuple[int, int, str]], has_parent: bool = True) -> list[str]:
    """Deterministic risk codes for one landing (documented in the module docstring)."""
    risks: list[str] = []
    paths = [p for _, _, p in numstat]
    themes = {theme_of(p) for p in paths}

    if any(h in subject for h in FILTER_HINT):
        risks.append("R1-GATE-FILTER-DECLARED")
    if parsed["is_landing"] and parsed["channel"] and parsed["channel"] != "merge-master":
        risks.append("R2-MERGE-CHANNEL-" + re.sub(r"[^A-Z0-9]+", "_", parsed["channel"].upper()))
    if not numstat and has_parent:
        risks.append("R3-CONTENT-EMPTY-LANDING")
    if "scratch" in themes or "loose-scripts" in themes:
        risks.append("R4-SCRATCH-IN-HISTORY")
    if numstat and "core-solver" not in themes:
        risks.append("R5-NO-CORE-CHANGE")
    if len(themes) >= 4:
        risks.append("R6-SCOPE-CREEP")
    for ins, dele, path in numstat:
        if ins + dele >= BIG_ARTIFACT_LINES and theme_of(path) in {"validation-artifacts", "run-artifacts"}:
            risks.append("R7-BIG-ARTIFACT-CHURN")
            break
    for _, _, path in numstat:
        if theme_of(path) in {"validation-artifacts", "docs"} and path.lower().endswith(BINARY_ASSET_SUFFIXES) and "/figures/" in path:
            risks.append("R8-BINARY-FIGURE-ASSET")
            break
    if numstat and not any(p.endswith(".sha256") for p in paths) and any(
        p.startswith("validation/") and p.endswith((".json", ".csv")) for p in paths
    ):
        risks.append("R9-ARTIFACT-NO-SHA256")
    return risks


def commit_inventory(repo: Path, upstream: str, limit: int) -> list[dict]:
    raw = git(repo, "log", "-n", str(limit), "--first-parent", "--format=%H|%ad|%s", "--date=iso-strict", upstream)
    commits: list[dict] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        sha, date, subject = line.split("|", 2)
        parsed = parse_subject(subject)
        numstat, has_parent = first_parent_numstat(repo, sha)
        themes = Counter(theme_of(p) for _, _, p in numstat)
        commits.append(
            {
                "sha": sha,
                "short": sha[:7],
                "date": date,
                "subject": subject,
                **parsed,
                "has_parent": has_parent,
                "files": len(numstat),
                "insertions": sum(max(i, 0) for i, _, _ in numstat),
                "deletions": sum(max(d, 0) for _, d, _ in numstat),
                "themes": dict(themes),
                "paths": [p for _, _, p in numstat],
                "files_plumbing": len(plumbing_numstat(repo, sha)),
                "parents": len(git(repo, "rev-list", "--parents", "-n", "1", sha).split()) - 1,
                "risks": commit_risks(subject, parsed, numstat, has_parent),
            }
        )
    return commits


def tree_blobs(repo: Path, rev: str) -> dict[str, str]:
    out = git(repo, "ls-tree", "-r", "-z", "--full-tree", rev, check=False)
    blobs: dict[str, str] = {}
    for item in out.split("\0"):
        if not item:
            continue
        meta, _, path = item.partition("\t")
        parts = meta.split()
        if len(parts) >= 3 and parts[1] == "blob":
            blobs[path] = parts[2]
    return blobs


def branch_audit(repo: Path, upstream: str, pattern: str = "agent/", remote_names: set[str] | None = None) -> dict:
    """Classify the delta of every local branch (not reachable from upstream)."""
    upstream_blobs = tree_blobs(repo, upstream)
    refs = [
        line.strip()
        for line in git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()
        if line.strip() and pattern in line
    ]
    rows = []
    for ref in refs:
        already = git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False).strip()
        is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", already, upstream],
            cwd=str(repo), capture_output=True, text=True,
        ).returncode == 0
        if is_ancestor:
            rows.append({"branch": ref, "tip": already[:7], "state": "MERGED_REF", "absent": 0, "differs": 0,
                         "landed": 0, "absent_paths": [],
                         "upstream_ref_exists": (ref in remote_names) if remote_names is not None else None})
            continue
        base = git(repo, "merge-base", upstream, ref, check=False).strip()
        if not base:
            rows.append({"branch": ref, "tip": already[:7], "state": "NO_MERGE_BASE", "absent": 0, "differs": 0,
                         "landed": 0, "absent_paths": [],
                         "upstream_ref_exists": (ref in remote_names) if remote_names is not None else None})
            continue
        changed = [
            line.split("\t", 1)[1]
            for line in git(repo, "diff", "--name-status", "--no-renames", base, ref, check=False).splitlines()
            if "\t" in line and not line.startswith("D\t")
        ]
        branch_blobs = tree_blobs(repo, ref)
        absent, differs, landed, absent_paths = 0, 0, 0, []
        for path in changed:
            b_blob = branch_blobs.get(path)
            if b_blob is None:
                continue
            u_blob = upstream_blobs.get(path)
            if u_blob is None:
                absent += 1
                absent_paths.append(path)
            elif u_blob != b_blob:
                differs += 1
            else:
                landed += 1
        if absent or differs:
            state = "LOST_FILES" if absent else "DRIFTED"
        else:
            state = "CONTENT_LANDED" if landed else "NO_DELTA"
        rows.append(
            {
                "branch": ref,
                "tip": already[:7],
                "state": state,
                "absent": absent,
                "differs": differs,
                "landed": landed,
                "absent_paths": absent_paths,
                "upstream_ref_exists": (ref in remote_names) if remote_names is not None else None,
            }
        )
    states = Counter(r["state"] for r in rows)
    missing = Counter()
    for r in rows:
        for p in r["absent_paths"]:
            missing[p] += 1
    stale = [r for r in rows if r.get("upstream_ref_exists") is False]
    return {
        "branch_pattern": pattern,
        "remote_names_supplied": remote_names is not None,
        "stale_local_refs": len(stale),
        "stale_with_lost_files": sum(1 for r in stale if r["state"] == "LOST_FILES"),
        "stale_only_absent_files": sum(r["absent"] for r in stale),
        "branches_scanned": len(rows),
        "states": dict(states),
        "absent_file_hits": sum(r["absent"] for r in rows),
        "top_absent_paths": [{"path": p, "branches": n} for p, n in missing.most_common(15)],
        "rows": sorted(rows, key=lambda r: (-r["absent"], -r["differs"], r["branch"])),
    }


def referenced_absent(repo: Path, upstream: str, absent_paths: list[str]) -> dict:
    """Which upstream files reference a path that only exists on a local branch?

    A path missing from upstream is not automatically a loss: build logs,
    ``artifacts/*.png`` and driver scripts are filtered by the merge gate on
    purpose.  The loss that matters is a path upstream **depends on** -- a
    document that embeds the figure, or a test/gate that reads the artifact
    (missing artifact + ``pytest.skip`` = a gate that asserts nothing, see
    ``tools/audit_gate_vacuity.py``).  This function finds those references with
    one ``git grep`` over the upstream tree, so the split is evidence, not
    opinion.
    """
    if not absent_paths:
        return {"scanned": 0, "referenced": {}, "referenced_count": 0, "unreferenced_count": 0}
    tmp = Path(tempfile.mkstemp(prefix="patrol_absent_", suffix=".txt")[1])
    try:
        tmp.write_text("\n".join(sorted(set(absent_paths))) + "\n", encoding="utf-8")
        out = git(repo, "grep", "-F", "-I", "-f", str(tmp), upstream, "--", "*.md", "*.py", check=False)
    finally:
        tmp.unlink(missing_ok=True)
    referenced: dict[str, list[str]] = defaultdict(list)
    for line in out.splitlines():
        # "<rev>:<file>:<line text>"
        _rev, _, rest = line.partition(":")
        src, _, text = rest.partition(":")
        for path in absent_paths:
            if path in text and src != path:
                referenced[path].append(src)
    referenced = {p: sorted(set(v)) for p, v in referenced.items()}
    by_category: Counter = Counter()
    detail = {}
    consumers = 0
    for path, srcs in referenced.items():
        cats = sorted({src.split("/")[0] if "/" in src else "root" for src in srcs})
        # A *consumer* names the path because it must already exist (doc figure,
        # test fixture); a *producer* names it because it writes it.  Only the
        # consumer case is a loss when the file never lands.
        is_consumer = bool({"docs", "tests"} & set(cats))
        consumers += int(is_consumer)
        detail[path] = {"sources": srcs, "categories": cats, "consumer": is_consumer}
        for c in cats:
            by_category[c] += 1
    return {
        "scanned": len(set(absent_paths)),
        "referenced": referenced,
        "detail": detail,
        "by_category": dict(by_category.most_common()),
        "consumer_referenced_count": consumers,
        "referenced_count": len(referenced),
        "unreferenced_count": len(set(absent_paths)) - len(referenced),
    }


def gate_filter_declared(repo: Path, upstream: str, depth: int = 200) -> dict:
    """Commits *anywhere* in the last ``depth`` upstream commits that admit a gate filter.

    The first-parent window shows landings only; the admission usually sits on
    the mission's own work commit (a second parent), e.g. the merge message
    ``N 个文件被门禁滤除``.  Counting it over the full topology keeps the claim
    "the gate really does drop files" evidence-backed instead of anecdotal.
    """
    out = git(repo, "log", "-n", str(depth), "--format=%H|%s", upstream)
    hits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        sha, subject = line.split("|", 1)
        if any(h in subject for h in FILTER_HINT):
            hits.append({"sha": sha[:7], "subject": subject})
    return {"depth": depth, "count": len(hits), "hits": hits[:10]}


def build_report(repo: Path, upstream: str, head: str, limit: int, do_branches: bool,
                 remote_names: set[str] | None = None) -> dict:
    inv = commit_inventory(repo, upstream, limit)
    themes: Counter = Counter()
    for c in inv:
        for theme, n in c["themes"].items():
            themes[theme] += n
    risk_counter: Counter = Counter()
    for c in inv:
        for r in c["risks"]:
            risk_counter[r] += 1
    landings = [c for c in inv if c["is_landing"]]
    naive_blind = [c for c in inv if c["files"] > 0 and c["files_plumbing"] == 0]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": str(repo),
        "head": head,
        "upstream": upstream,
        "diff": diff_set(repo, head, upstream),
        "window": limit,
        "commits": inv,
        "summary": {
            "commits": len(inv),
            "landings": len(landings),
            "non_landing": len(inv) - len(landings),
            "content_empty_landings": sum(1 for c in inv if "R3-CONTENT-EMPTY-LANDING" in c["risks"]),
            "files_touched": sum(c["files"] for c in inv),
            "insertions": sum(c["insertions"] for c in inv),
            "deletions": sum(c["deletions"] for c in inv),
            "theme_lines": dict(themes.most_common()),
            "risk_counts": dict(risk_counter.most_common()),
            "landing_channels": dict(Counter(c["channel"] for c in landings)),
            "plumbing_read_blind_commits": len(naive_blind),
            "plumbing_read_blind_files_hidden": sum(c["files"] for c in naive_blind),
            "plumbing_read_blind_shas": [c["short"] for c in naive_blind],
        },
    }
    report["gate_filter_declared"] = gate_filter_declared(repo, upstream)
    if do_branches:
        report["branches"] = branch_audit(repo, upstream, remote_names=remote_names)
        absent_paths = sorted({p for row in report["branches"]["rows"] for p in row["absent_paths"]})
        report["lost_referenced"] = referenced_absent(repo, upstream, absent_paths)
    return report


def to_markdown(report: dict) -> str:
    d = report["diff"]
    s = report["summary"]
    lines = [
        f"# TensorDG upstream-diff patrol — {report['generated_at']}",
        "",
        f"- repo: `{report['repo']}`",
        f"- HEAD: `{report['head'][:7]}`  upstream: `{report['upstream']}`",
        f"-差集 verdict **{d['verdict']}** (ahead={d['ahead']}, behind={d['behind']})",
        f"- window: last {report['window']} first-parent commits of upstream",
        "",
        "## Landing window",
        "",
        "| # | sha | date | subject | files | +/− | themes | risk |",
        "|---:|---|---|---|---:|---|---|---|",
    ]
    for i, c in enumerate(report["commits"], 1):
        themes = ", ".join(f"{k}:{v}" for k, v in sorted(c["themes"].items(), key=lambda kv: -kv[1])) or "—"
        lines.append(
            f"| {i} | `{c['short']}` | {c['date'][:10]} | {c['subject'][:70]} | {c['files']} | "
            f"+{c['insertions']}/-{c['deletions']} | {themes} | {', '.join(c['risks']) or '—'} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        "```json",
        json.dumps(s, indent=2, ensure_ascii=False),
        "```",
    ]
    if "branches" in report:
        b = report["branches"]
        lines += [
            "",
            "## Branch audit",
            "",
            f"scanned {b['branches_scanned']} local branches; states {b['states']}; "
            f"files absent on upstream: {b['absent_file_hits']}",
            "",
            "| path | branches |",
            "|---|---:|",
        ]
        for row in b["top_absent_paths"]:
            lines.append(f"| `{row['path']}` | {row['branches']} |")
    lr = report.get("lost_referenced")
    if lr:
        lines += [
            "",
            "## Absent-on-upstream paths that upstream still references",
            "",
            f"{lr['referenced_count']} of {lr['scanned']} paths absent from upstream are referenced by "
            f"upstream markdown/scripts ({lr.get('consumer_referenced_count', 0)} of them by a *consumer* -- "
            f"docs/tests -- which makes the absence a defect, not a runtime artifact): "
            f"{json.dumps(lr['by_category'], ensure_ascii=False)}",
            "",
            "| path | referenced by |",
            "|---|---|",
        ]
        for path, meta in list(lr["detail"].items())[:25]:
            lines.append(f"| `{path}` | {', '.join('`'+x+'`' for x in meta['sources'][:3])} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(REPO_ROOT))
    ap.add_argument("--upstream", default=None)
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--out", default="artifacts/patrol_upstream_diff")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-branches", action="store_true")
    ap.add_argument(
        "--remote-branches",
        default=None,
        help="file with one branch name per line as reported by "
             "`git ls-remote --heads origin 'refs/heads/agent/*'`; annotates which local refs "
             "still exist upstream (offline: the tool never runs ls-remote itself)",
    )
    ap.add_argument("--check", action="store_true", help="non-zero exit if a content-empty landing or a lost-file branch is present")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    upstream, how = resolve_upstream(repo, args.upstream)
    if not upstream:
        print("no upstream ref found (no branch upstream, no origin/HEAD, no origin/main)", file=sys.stderr)
        return 3
    head = git(repo, "rev-parse", args.head).strip()
    remote_names = None
    if args.remote_branches:
        remote_names = {
            line.strip().replace("refs/heads/", "")
            for line in Path(args.remote_branches).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    report = build_report(repo, upstream, head, args.limit, not args.no_branches, remote_names)
    report["upstream_resolution"] = how

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if not args.write:
        print(payload)
    else:
        out = (repo / args.out) if not Path(args.out).is_absolute() else Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "patrol.json").write_text(payload + "\n", encoding="utf-8")
        (out / "patrol.md").write_text(to_markdown(report), encoding="utf-8")
        print(f"wrote {out/'patrol.json'} and {out/'patrol.md'}")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    if args.check:
        empty = report["summary"]["content_empty_landings"]
        lost = report.get("branches", {}).get("states", {}).get("LOST_FILES", 0)
        if empty or lost:
            print(f"CHECK FAILED: content-empty landings={empty}, branches with files absent upstream={lost}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())