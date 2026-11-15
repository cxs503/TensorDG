#!/usr/bin/env python3
"""Probe upstream ``path:line`` pointers of the LBM method set (FEAT-OSS-INVENTORY-LBM).

Runs **on a compute node with network access** (the agent host has no outbound
network).  For every repository in ``artifacts/oss_solutions_lbm/pointer_probe_input.json``
it does a shallow, blob-filtered clone of the default branch, resolves ``HEAD``
and reads the cited line, so a pointer is either backed by real source text or
reported as failed.  Nothing here is inferred.

Status values per pointer:

``ok``            file exists, cited line in range -> ``line_text`` is the real line
``line_oob``      file exists but the cited line is past EOF (upstream drift)
``missing_file``  path not present at HEAD
``clone_error``   repository could not be cloned (recorded with stderr tail)
``file_only_ok``  source row cited a file without a line (no line fabricated)

Usage::

    python3 tools/probe_oss_pointers.py [--out artifacts/oss_solutions_lbm/pointer_probe.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "artifacts" / "oss_solutions_lbm" / "pointer_probe_input.json"
DEFAULT_OUT = ROOT / "artifacts" / "oss_solutions_lbm" / "pointer_probe.json"
CLONE_TIMEOUT = 420
MAX_LINE_CHARS = 300


def run(cmd, cwd=None, timeout=CLONE_TIMEOUT):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def probe_repo(repo: str, pointers: list[dict], workdir: pathlib.Path) -> dict:
    slug = repo.replace("/", "__")
    dest = workdir / slug
    rec: dict = {"head": None, "clone": "ok", "pointers": {}}
    url = f"https://github.com/{repo}.git"
    try:
        cl = run(["git", "clone", "--filter=blob:none", "--depth", "1", "--no-checkout",
                  url, str(dest)], timeout=CLONE_TIMEOUT)
    except subprocess.TimeoutExpired:
        rec["clone"] = "timeout"
        rec["pointers"] = {p["ref"]: {"status": "clone_error", "detail": "clone timeout"}
                           for p in pointers}
        return rec
    if cl.returncode != 0:
        rec["clone"] = "error"
        tail = (cl.stderr or "").strip().splitlines()[-3:]
        rec["clone_stderr"] = " | ".join(tail)[:400]
        rec["pointers"] = {p["ref"]: {"status": "clone_error",
                                      "detail": rec["clone_stderr"]}
                           for p in pointers}
        return rec
    head = run(["git", "rev-parse", "HEAD"], cwd=dest, timeout=120)
    rec["head"] = head.stdout.strip() if head.returncode == 0 else None

    for p in pointers:
        path = p["path"]
        line = p.get("line")
        ref = p["ref"]
        blob = run(["git", "show", f"HEAD:{path}"], cwd=dest, timeout=300)
        if blob.returncode != 0:
            rec["pointers"][ref] = {
                "status": "missing_file",
                "detail": (blob.stderr or "").strip().splitlines()[-1][:300] if blob.stderr else "",
            }
            continue
        lines = blob.stdout.splitlines()
        if line is None:
            rec["pointers"][ref] = {
                "status": "file_only_ok",
                "n_lines": len(lines),
                "line_text": (lines[0].strip()[:MAX_LINE_CHARS] if lines else ""),
            }
            continue
        if line < 1 or line > len(lines):
            rec["pointers"][ref] = {"status": "line_oob", "n_lines": len(lines),
                                    "detail": f"cited line {line} > {len(lines)} lines"}
            continue
        text = lines[line - 1].strip()
        rec["pointers"][ref] = {
            "status": "ok",
            "n_lines": len(lines),
            "line_text": text[:MAX_LINE_CHARS],
            "line_nonempty": bool(text),
        }
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out", default=str(DEFAULT_OUT))
    ap.add_argument("--keep", action="store_true", help="keep clones (debugging)")
    args = ap.parse_args(argv)

    plan = json.loads(pathlib.Path(args.inp).read_text())
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="ossprobe_"))
    result = {}
    for repo in sorted(plan):
        print(f"[probe] {repo} ({len(plan[repo])} pointers)", flush=True)
        result[repo] = probe_repo(repo, plan[repo], workdir)
        st = result[repo]["pointers"]
        ok = sum(1 for v in st.values() if v["status"] in ("ok", "file_only_ok"))
        print(f"[probe] {repo}: head={result[repo]['head']} clone={result[repo]['clone']} "
              f"ok={ok}/{len(st)}", flush=True)
    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                   encoding="utf-8")
    n = sum(len(v["pointers"]) for v in result.values())
    ok = sum(1 for v in result.values() for p in v["pointers"].values()
             if p["status"] in ("ok", "file_only_ok"))
    print(f"[probe] total {ok}/{n} pointers backed by source, wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())