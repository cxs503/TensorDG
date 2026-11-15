#!/usr/bin/env python3
"""Wave-2 (deep-dive) upstream pointer harvest for the FVM open-source inventory.

Context
-------
``FEAT-OSS-INVENTORY-FVM`` wave-1 (TensorFVM, mission 14887) delivered a 30-repo
registry plus 8 ``GAP-FVM-*`` anchors, and stated in
``docs/oss-inventory-fvm/registry.json`` (``evidence_policy``) that the capability
tags were *expert curation*, explicitly **not** a source-reading pass:

    curation.capability_evidence == "curated_knowledge" ... NOT verified by
    reading upstream sources in this run; adoption requires a source-reading pass

The LBM wave-2 protocol (``docs/OSS-SOLUTIONS-LBM.md`` in this repo) requires
every harvested entry to carry a machine-verifiable ``repo:path:line`` pointer.
This script is the FVM relay of that protocol: for each of the eight
``GAP-FVM-*`` anchors it resolves the upstream implementation file **by pattern
over the live GitHub tree** (so no path is hand-typed), fetches the blob through
the contents API, and greps the mechanism needle to obtain the exact line.

Two channels are kept separate, as in wave-1:

``raw``     every GitHub response is cached under
            ``artifacts/oss_inventory_fvm_wave2/raw/`` (tree JSON + base64 blob
            JSON + sha256), so the published registry can be re-derived offline.
``curated`` only the (gap, repo, path-regex, needle-regex) *search intent* lives
            in ``SPECS`` below; every pointer, line number and snippet in the
            registry is produced by the probe.

Usage::

    python3 tools/probe_fvm_wave2_pointers.py --harvest   # network, writes registry
    python3 tools/probe_fvm_wave2_pointers.py --check     # offline, re-derive + verify

``--harvest`` is rate-limited by the unauthenticated GitHub API (60 req/h); the
raw cache makes reruns incremental (a cached tree/blob is never re-fetched).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "artifacts" / "oss_inventory_fvm_wave2" / "raw"
OUT = ROOT / "docs" / "oss_inventory_fvm_wave2_pointers.json"
WAVE1 = pathlib.Path("/root/tp-agent/work/m21837/docs/oss-inventory-fvm/registry.json")
API = "https://api.github.com"

# ---------------------------------------------------------------------------
# Search intent (curated layer).  One entry = one (anchor, upstream mechanism)
# claim to be located in the live tree.  ``path_rx`` selects candidate files
# from the repository tree; ``needle_rx`` is the mechanism evidence that must
# occur inside the selected file.
# ---------------------------------------------------------------------------
SPECS = [
    # --- GAP-FVM-AMR-QUADTREE: conservative coarse-fine flux refluxing -------
    dict(gap="GAP-FVM-AMR-QUADTREE", repo="AMReX-Codes/amrex",
         path_rx=r"AmrCore/AMReX_FluxRegister\.(H|cpp)$", needle_rx=r"FluxRegister",
         claim="coarse-fine flux register (refluxing) data structure"),
    dict(gap="GAP-FVM-AMR-QUADTREE", repo="OpenFOAM/OpenFOAM-dev",
         path_rx=r"dynamicRefineFvMesh\.(H|C)$", needle_rx=r"refine",
         claim="run-time h-refinement of an FV mesh (hanging-node aware)"),
    # --- GAP-FVM-IMPLICIT-DUAL-TIME: LU-SGS linearisation -------------------
    dict(gap="GAP-FVM-IMPLICIT-DUAL-TIME", repo="su2code/SU2",
         path_rx=r"solvers/CEulerSolver\.(cpp|hpp)$", needle_rx=r"LU_SGS|LUSGS|Precond",
         claim="LU-SGS / preconditioned implicit Euler linearisation"),
    dict(gap="GAP-FVM-IMPLICIT-DUAL-TIME", repo="OpenFOAM/OpenFOAM-dev",
         path_rx=r"lduMatrix/solvers/.*/(GaussSeidel|smoothSolver)\.(H|C)$",
         needle_rx=r"class\s+\w*Smoother|GaussSeidel",
         claim="iterative implicit smoother used by the steady-state loop"),
    # --- GAP-FVM-LOWMACH-PRECOND: Weiss-Smith / Turkel preconditioning ------
    dict(gap="GAP-FVM-LOWMACH-PRECOND", repo="su2code/SU2",
         path_rx=r"numerics/CNumerics\.(cpp|hpp)$", needle_rx=r"Precond|precond",
         claim="low-Mach preconditioning matrix inside the numerics kernel"),
    # --- GAP-FVM-MRF-SLIDING-AMI: arbitrary mesh interface ------------------
    dict(gap="GAP-FVM-MRF-SLIDING-AMI", repo="OpenFOAM/OpenFOAM-dev",
         path_rx=r"AMIInterpolation/AMIInterpolation\.(H|C)$", needle_rx=r"weight",
         claim="AMI weight generation for non-conformal sliding interfaces"),
    dict(gap="GAP-FVM-MRF-SLIDING-AMI", repo="OpenFOAM/OpenFOAM-dev",
         path_rx=r"MRF/MRFZone\.(H|C)$", needle_rx=r"MRF|rotating",
         claim="multiple reference frame source term"),
    # --- GAP-FVM-MPI-DECOMP: domain decomposition ---------------------------
    dict(gap="GAP-FVM-MPI-DECOMP", repo="code-saturne/code_saturne",
         path_rx=r"cs_parall\.(c|h)$", needle_rx=r"MPI_Comm_split|MPI_Comm_create",
         claim="MPI communicator split for the parallel domain decomposition"),
    dict(gap="GAP-FVM-MPI-DECOMP", repo="AMReX-Codes/amrex",
         path_rx=r"Base/AMReX_ParallelDescriptor\.(H|cpp)$", needle_rx=r"MPI_Comm_split|MyComm|MPI_Comm",
         claim="MPI communicator management for the AMR hierarchy"),
    # --- GAP-FVM-AMG-KRYLOV: algebraic multigrid ----------------------------
    dict(gap="GAP-FVM-AMG-KRYLOV", repo="petsc/petsc",
         path_rx=r"src/ksp/pc/impls/gamg/gamg\.c$", needle_rx=r"PCGAMG",
         claim="PETSc GAMG (smoothed-aggregation AMG) preconditioner entry"),
    dict(gap="GAP-FVM-AMG-KRYLOV", repo="hypre-space/hypre",
         path_rx=r"parcsr_ls/.*boomeramg\.c$", needle_rx=r"BoomerAMG",
         claim="BoomerAMG setup/solve driver"),
    # --- GAP-FVM-COMBUSTION-SPECIES: finite-rate chemistry ------------------
    dict(gap="GAP-FVM-COMBUSTION-SPECIES", repo="OpenFOAM/OpenFOAM-dev",
         path_rx=r"chemistryModel/.*/chemistryModel\.(H|C)$", needle_rx=r"chemistry",
         claim="finite-rate chemistry integration contract"),
    # --- GAP-FVM-RADIATION-P1: P1 / grey radiation --------------------------
    dict(gap="GAP-FVM-RADIATION-P1", repo="truchas/truchas",
         path_rx=r"[Rr]adiation.*\.(F90|f90)$", needle_rx=r"P1|radiation",
         claim="P1 (spherical-harmonics) radiation diffusion solve"),
]

MAX_FILES_PER_SPEC = 2          # tree candidates fetched before giving up
SLEEP = 0.6                     # be polite to the unauthenticated API


# ---------------------------------------------------------------------------
# raw cache helpers
# ---------------------------------------------------------------------------
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get(url: str, cache_path: pathlib.Path, budget: dict) -> dict | None:
    """Fetch ``url`` through the raw cache; ``None`` when unavailable."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if budget["remaining"] <= 1:
        return None
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "Accept-Encoding": "identity",
        "User-Agent": "tensordg-fvm-inventory-wave2",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as fh:
            chunks = []
            while True:                     # chunked transfer: read to EOF, never trust
                part = fh.read(65536)       # Content-Length (the proxy truncates it)
                if not part:
                    break
                chunks.append(part)
            payload = json.loads(b"".join(chunks).decode("utf-8"))
            remaining = int(fh.headers.get("X-RateLimit-Remaining", "0"))
            budget["remaining"] = remaining
    except urllib.error.HTTPError as exc:               # 403/404/451 ...
        remaining = int(exc.headers.get("X-RateLimit-Remaining") or 0)
        budget["remaining"] = remaining
        payload = {"_error": f"HTTP {exc.code}", "_url": url}
    except Exception as exc:                            # network down
        payload = {"_error": repr(exc), "_url": url}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    time.sleep(SLEEP)
    return payload


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug_dir(repo: str) -> pathlib.Path:
    return RAW_DIR / repo.replace("/", "__")


def tree_of(repo: str, budget: dict) -> dict | None:
    """Recursive tree of the default branch (cached)."""
    d = _slug_dir(repo)
    meta = _get(f"{API}/repos/{repo}", d / "repo.json", budget)
    if not meta or "_error" in meta:
        return None
    branch = meta.get("default_branch", "main")
    ref = meta.get("pushed_at")  # informational only
    tree = _get(f"{API}/repos/{repo}/git/trees/{branch}?recursive=1",
                d / f"tree__{branch}.json", budget)
    if not tree or "_error" in tree or "tree" not in tree:
        return None
    tree["_branch"] = branch
    tree["_pushed_at"] = ref
    tree["_repo_sha"] = meta.get("sha") or meta.get("id")
    return tree


def blob_of(repo: str, path: str, budget: dict) -> dict | None:
    """File content through the contents API (cached, base64 decoded)."""
    d = _slug_dir(repo)
    cache = d / "blobs" / (re.sub(r"[^A-Za-z0-9._-]", "_", path) + ".json")
    raw = _get(f"{API}/repos/{repo}/contents/{path}", cache, budget)
    if not raw or "_error" in raw or "content" not in raw:
        return None
    try:
        text = base64.b64decode(raw["content"]).decode("utf-8", "replace")
    except Exception:
        return None
    return {"path": raw.get("path", path), "sha": raw.get("sha"),
            "size": raw.get("size"), "text": text}


# ---------------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------------
def harvest(budget: dict, only: list[str] | None = None) -> dict:
    trees: dict[str, dict] = {}
    pointers: list[dict] = []
    for spec in SPECS:
        if only and not any(o.lower() in spec["repo"].lower() for o in only):
            continue
        repo, gap = spec["repo"], spec["gap"]
        if repo not in trees:
            t = tree_of(repo, budget)
            trees[repo] = t or {}
        tree = trees[repo]
        base = {"gap": gap, "repo": repo, "claim": spec["claim"],
                "path_rx": spec["path_rx"], "needle_rx": spec["needle_rx"]}
        if not tree:
            pointers.append({**base, "status": "unavailable",
                             "reason": "tree/API not retrievable"})
            continue
        paths = sorted(p["path"] for p in tree["tree"]
                       if p.get("type") == "blob" and re.search(spec["path_rx"], p["path"]))
        if not paths:
            pointers.append({**base, "status": "path_not_found",
                             "reason": f"no tree path matches {spec['path_rx']}"})
            continue
        hit = None
        tried = []
        for path in paths[:MAX_FILES_PER_SPEC]:
            blob = blob_of(repo, path, budget)
            if not blob:
                tried.append(path + " (fetch failed)")
                continue
            for lineno, line in enumerate(blob["text"].splitlines(), start=1):
                if re.search(spec["needle_rx"], line):
                    hit = dict(base, status="verified", path=path, line=lineno,
                               snippet=line.strip()[:200],
                               blob_sha=blob["sha"], file_bytes=blob["size"],
                               needle=_first_match(spec["needle_rx"], line),
                               tried=tried + [path])
                    break
            if hit:
                break
            tried.append(path)
        if hit is None:
            pointers.append({**base, "status": "needle_not_found",
                             "reason": f"needle {spec['needle_rx']} absent in first "
                                       f"{len(tried)} candidate file(s)", "tried": tried})
        else:
            pointers.append(hit)
    return {"pointers": pointers, "trees": trees}


def _first_match(rx: str, line: str) -> str:
    m = re.search(rx, line)
    return m.group(0) if m else ""


def build_registry(harvested: dict) -> dict:
    pointers = harvested["pointers"]
    trees = harvested["trees"]
    gap_ids = sorted({p["gap"] for p in pointers})
    by_gap: dict[str, dict] = {}
    for gap in gap_ids:
        sub = [p for p in pointers if p["gap"] == gap]
        by_gap[gap] = {
            "n_specs": len(sub),
            "n_verified": sum(1 for p in sub if p["status"] == "verified"),
            "repos_verified": sorted({p["repo"] for p in sub if p["status"] == "verified"}),
            "pointers": sub,
        }
    raw_files = sorted(str(p.relative_to(ROOT)) for p in RAW_DIR.rglob("*.json"))
    return {
        "schema_version": "1.0",
        "feature_anchor": "FEAT-OSS-INVENTORY-FVM",
        "wave": 2,
        "protocol": "LBM wave-2 relay: every entry carries a machine-verified "
                    "upstream path:line pointer re-derivable from the raw cache",
        "generated_at": _now(),
        "target_wave1_registry": str(WAVE1),
        "wave1_debt_addressed": "wave-1 evidence_policy: capability tags were curated, "
                                "not source-verified; this wave binds the eight "
                                "GAP-FVM-* anchors to located upstream code",
        "provenance": {
            "api": "api.github.com (unauthenticated, read-only)",
            "raw_dir": str(RAW_DIR.relative_to(ROOT)),
            "n_raw_files": len(raw_files),
            "trees": {repo: {"branch": t.get("_branch"),
                             "tree_sha": (t.get("sha") if t else None)}
                      for repo, t in trees.items()},
            "note": "path/line/snippet/licence-irrelevant metadata are produced by the "
                    "probe; only the search intent (gap, path regex, needle regex) is curated",
        },
        "coverage": {
            "gaps_targeted": len(gap_ids),
            "gaps_with_verified_pointer": sum(1 for g in by_gap.values()
                                              if g["n_verified"] > 0),
            "specs": len(pointers),
            "verified": sum(1 for p in pointers if p["status"] == "verified"),
            "status_counts": {s: sum(1 for p in pointers if p["status"] == s)
                              for s in sorted({p["status"] for p in pointers})},
            "repos_touched": sorted({p["repo"] for p in pointers}),
        },
        "gaps": by_gap,
        "limitations": [
            "One pointer per (gap, mechanism claim) — this is a source-reading *start*, "
            "not a full per-functional-point solution set (LBM wave-2 style 96-point "
            "taxonomy is still outstanding for FVM).",
            "Pointers prove that the mechanism is implemented at that location; they do "
            "not prove algorithm equivalence with TensorFVM's formulation.",
            "Licence flags are inherited from the wave-1 registry; GPL upstreams are "
            "read-only references (re-implement from the papers, do not copy code).",
            "GitHub default-branch trees move: the registry pins the tree sha and the "
            "blob sha of every quoted file, so drift is detectable but not prevented.",
        ],
    }


# ---------------------------------------------------------------------------
# offline re-derivation (used by the gate test)
# ---------------------------------------------------------------------------
def rederive(reg: dict) -> list[str]:
    """Re-derive every pointer from the raw cache; return human-readable errors."""
    errs: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for gap, block in reg["gaps"].items():
        for p in block["pointers"]:
            if p["status"] != "verified":
                continue
            key = (gap, p["repo"], p["path"])
            if key in seen:
                errs.append(f"duplicate pointer {key}")
            seen.add(key)
            if not re.search(p["path_rx"], p["path"]):
                errs.append(f"{key}: path does not match its own path_rx")
            cache = _slug_dir(p["repo"]) / "blobs" / (
                re.sub(r"[^A-Za-z0-9._-]", "_", p["path"]) + ".json")
            if not cache.exists():
                errs.append(f"{key}: missing raw blob cache {cache}")
                continue
            blob = json.loads(cache.read_text(encoding="utf-8"))
            if blob.get("sha") != p["blob_sha"]:
                errs.append(f"{key}: blob sha drift ({blob.get('sha')} != {p['blob_sha']})")
            text = base64.b64decode(blob["content"]).decode("utf-8", "replace")
            lines = text.splitlines()
            if not (1 <= p["line"] <= len(lines)):
                errs.append(f"{key}: line {p['line']} out of range {len(lines)}")
                continue
            actual = lines[p["line"] - 1].strip()[:200]
            if actual != p["snippet"]:
                errs.append(f"{key}: snippet mismatch ... {actual!r} != {p['snippet']!r}")
            if not re.search(p["needle_rx"], lines[p["line"] - 1]):
                errs.append(f"{key}: needle {p['needle_rx']} not on recorded line")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", action="store_true", help="fetch from GitHub and write registry")
    ap.add_argument("--check", action="store_true", help="offline re-derivation of the registry")
    ap.add_argument("--only", default="", help="comma-separated repo substrings (chunked runs)")
    ap.add_argument("--merge", action="store_true", help="merge this run into the existing registry")
    args = ap.parse_args()
    if args.harvest:
        budget = {"remaining": 60}
        only = [x for x in args.only.split(",") if x]
        harvested = harvest(budget, only)
        if args.merge and OUT.exists():
            prev = json.loads(OUT.read_text(encoding="utf-8"))
            done = {(p["gap"], p["repo"], p["claim"])
                    for blk in prev["gaps"].values() for p in blk["pointers"]}
            fresh = [p for p in harvested["pointers"]
                     if (p["gap"], p["repo"], p["claim"]) not in done]
            ptrs = [p for blk in prev["gaps"].values() for p in blk["pointers"]] + fresh
            trees = {r: t for r, t in harvested["trees"].items()}
        else:
            ptrs = harvested["pointers"]
            trees = harvested["trees"]
        harvested = {"pointers": ptrs, "trees": trees}
        reg = build_registry(harvested)
        OUT.write_text(json.dumps(reg, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(reg["coverage"], indent=1, ensure_ascii=False))
        for gap, blk in reg["gaps"].items():
            for p in blk["pointers"]:
                print(f"  {gap:30s} {p['repo']:28s} {p['status']:16s} "
                      f"{p.get('path','')}:{p.get('line','')}")
        errs = rederive(reg)
        print(f"offline re-derivation errors: {len(errs)}")
        for e in errs:
            print("  !", e)
        return 0 if not errs else 1
    if args.check:
        reg = json.loads(OUT.read_text(encoding="utf-8"))
        errs = rederive(reg)
        print(f"re-derivation errors: {len(errs)}")
        for e in errs:
            print("  !", e)
        return 0 if not errs else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())