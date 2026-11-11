"""Evidence-integrity audit for the machine-readable verification artifacts.

Why this exists
---------------
A verification artifact is only evidence while it is (a) *read* by something and
(b) *traceable* to the code that produced it.  Two failure modes silently
destroy that property:

* **orphan evidence** -- a JSON/PNG under ``validation/`` or ``artifacts/`` that
  no test, tool, benchmark or document ever reads.  Nothing fails when the
  solver moves on, so the file keeps asserting an old number forever.
* **stale evidence** -- the file *is* read (or worse: the document calls it the
  "single source of truth") but was produced by code that no longer exists.
  The audit cannot detect numeric staleness by itself; it detects the
  *precondition*: an artifact without a provenance block cannot be proven
  current, so it is flagged as ``unprovable``.

This tool implements the cheap, machine-checkable half of that audit and is the
engine behind ``tests/test_evidence_integrity.py`` (the guard) and
``validation/evidence_registry.json`` (the ratchet: known orphans are recorded
with a reason, new ones fail the guard).

Usage
-----
``python tools/audit_evidence_integrity.py``            # summary to stdout
``python tools/audit_evidence_integrity.py --write``    # + refresh registry/report
``python tools/audit_evidence_integrity.py --check``    # exit 1 on a new orphan

Exit codes: ``0`` clean, ``1`` new (unregistered) orphan evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories whose ``*.json`` files are treated as evidence artifacts.
ARTIFACT_ROOTS = ("validation", "artifacts", "docs/benchmarks/results")
# Text corpora that count as "something reads this artifact".
CONSUMER_ROOTS = ("tests", "tools", "benchmarks", "tensordg", "validation", "docs")
CONSUMER_EXTRA = ("README.md", "pyproject.toml")
TEXT_EXT = (".py", ".md", ".toml", ".yml", ".yaml", ".cfg", ".txt", ".json")

# File names too generic to prove a reference on their own: for these the parent
# directory name must also appear in the consumer corpus.
GENERIC_NAMES = frozenset({
    "metrics.json", "summary.json", "evidence.json", "results.json",
    "verify.json", "study.json", "report.json", "data.json", "output.json",
    "summary_old.json",
})

# The audit must not launder its own output into evidence: if the registry (or
# the audit report) were scanned as consumers, every registered orphan would
# "reference" itself and the orphan count would collapse to zero.
SELF_OUTPUTS = frozenset({
    "validation/evidence_registry.json",
    "artifacts/evidence_integrity_audit.json",
    "docs/evidence_integrity_audit.md",  # this page *describes* the orphans
})

PROVENANCE_KEYS = ("provenance", "_provenance")
REGISTRY_PATH = os.path.join(REPO_ROOT, "validation", "evidence_registry.json")
REPORT_PATH = os.path.join(REPO_ROOT, "artifacts", "evidence_integrity_audit.json")


def _iter_artifacts():
    out = []
    for root in ARTIFACT_ROOTS:
        base = os.path.join(REPO_ROOT, root)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".ipynb_checkpoints", "figs", "figures")]
            for name in sorted(filenames):
                if name.endswith(".json"):
                    out.append(os.path.relpath(os.path.join(dirpath, name), REPO_ROOT))
    return sorted(out)


def load_consumer_corpus() -> str:
    """Concatenated text of every file that could reference an artifact."""
    chunks = []
    for root in CONSUMER_ROOTS:
        base = os.path.join(REPO_ROOT, root)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
            for name in filenames:
                if not name.endswith(TEXT_EXT):
                    continue
                path = os.path.join(dirpath, name)
                if os.path.relpath(path, REPO_ROOT) in SELF_OUTPUTS:
                    continue
                try:
                    if os.path.getsize(path) > 4_000_000:
                        continue
                    with open(path, "r", errors="ignore") as fh:
                        chunks.append(fh.read())
                except OSError:
                    continue
    for name in CONSUMER_EXTRA:
        path = os.path.join(REPO_ROOT, name)
        if os.path.isfile(path):
            with open(path, "r", errors="ignore") as fh:
                chunks.append(fh.read())
    return "\n".join(chunks)


def is_referenced(relpath: str, corpus: str) -> bool:
    """True when some test/tool/benchmark/doc mentions this artifact."""
    if relpath in corpus:
        return True
    name = os.path.basename(relpath)
    parent = os.path.basename(os.path.dirname(relpath))
    if name in GENERIC_NAMES:
        # "metrics.json" appears everywhere: require the directory to be named too.
        return bool(parent) and parent in corpus
    return name in corpus


def provenance_status(payload) -> str:
    """'block' | 'per_block' | 'none' -- can this file be proven current?"""
    if not isinstance(payload, dict):
        return "none"
    if any(key in payload for key in PROVENANCE_KEYS):
        return "block"
    for value in payload.values():
        if isinstance(value, dict) and any(key in value for key in PROVENANCE_KEYS):
            return "per_block"
    if isinstance(payload.get("env"), dict):
        return "env_only"
    return "none"


def audit() -> dict:
    corpus = load_consumer_corpus()
    artifacts = []
    for relpath in _iter_artifacts():
        entry = {"path": relpath, "referenced": is_referenced(relpath, corpus)}
        try:
            with open(os.path.join(REPO_ROOT, relpath)) as fh:
                payload = json.load(fh)
            entry["readable"] = True
            entry["provenance"] = provenance_status(payload)
        except (OSError, json.JSONDecodeError) as exc:
            entry["readable"] = False
            entry["provenance"] = "none"
            entry["error"] = str(exc)[:160]
        entry["orphan"] = entry["referenced"] is False
        artifacts.append(entry)

    orphans = [e["path"] for e in artifacts if e["orphan"]]
    counts = {
        "artifacts": len(artifacts),
        "referenced": sum(1 for e in artifacts if e["referenced"]),
        "orphans": len(orphans),
        "provenance_block": sum(1 for e in artifacts if e["provenance"] == "block"),
        "provenance_per_block": sum(1 for e in artifacts if e["provenance"] == "per_block"),
        "provenance_env_only": sum(1 for e in artifacts if e["provenance"] == "env_only"),
        "provenance_none": sum(1 for e in artifacts if e["provenance"] == "none"),
        "unreadable": sum(1 for e in artifacts if not e["readable"]),
    }
    return {"counts": counts, "orphans": orphans, "artifacts": artifacts}


def load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {"orphans": {}, "provenance_exempt": [], "superseded": {}}
    with open(REGISTRY_PATH) as fh:
        return json.load(fh)


def new_orphans(result: dict, registry: dict) -> list:
    known = set((registry.get("orphans") or {}).keys())
    return [path for path in result["orphans"] if path not in known]


def stale_registry_entries(registry: dict) -> list:
    """Registry rows whose artifact disappeared (a ratchet must not rot)."""
    missing = []
    for path in list((registry.get("orphans") or {}).keys()) + \
            list((registry.get("superseded") or {}).keys()):
        artifact = path.split("#", 1)[0]  # "file.json#block" keys name a block
        if not os.path.exists(os.path.join(REPO_ROOT, artifact)):
            missing.append(path)
    return missing


def write_registry(result: dict, registry: dict) -> dict:
    """Refresh *known* orphans; never silently drop a registered reason."""
    known = dict(registry.get("orphans") or {})
    added = []
    for path in result["orphans"]:
        if path in known:
            continue
        known[path] = {
            "reason": "auto-registered by tools/audit_evidence_integrity.py; "
                      "needs a consumer or an explicit retirement note",
            "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "needs_review",
        }
        added.append(path)
    registry["orphans"] = known
    registry["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    registry["generator"] = "tools/audit_evidence_integrity.py --write"
    registry["artifact_roots"] = list(ARTIFACT_ROOTS)
    return {"registry": registry, "added": added}


def _print_summary(result: dict, registry: dict) -> None:
    counts = result["counts"]
    print(json.dumps(counts, indent=2))
    unknown = new_orphans(result, registry)
    print(f"registered orphans: {len(registry.get('orphans') or {})}, "
          f"new: {len(unknown)}")
    for path in unknown:
        print(f"  NEW ORPHAN {path}")
    for path in stale_registry_entries(registry):
        print(f"  STALE REGISTRY ROW {path} (artifact gone)")
    unprovable = [e["path"] for e in result["artifacts"]
                  if e["provenance"] == "none" and not e["orphan"]]
    print(f"referenced but without a provenance block: {len(unprovable)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="refresh validation/evidence_registry.json and artifacts/evidence_integrity_audit.json")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when an unregistered orphan exists")
    args = parser.parse_args(argv)

    result = audit()
    registry = load_registry()
    if args.write:
        written = write_registry(result, registry)
        registry = written["registry"]
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        result["registry"] = {"known_orphans": sorted((registry.get("orphans") or {})),
                              "new_orphans": new_orphans(result, registry)}
        with open(REPORT_PATH, "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        with open(REGISTRY_PATH, "w") as fh:
            json.dump(registry, fh, indent=2, sort_keys=True)
        print(f"wrote {os.path.relpath(REPORT_PATH, REPO_ROOT)} and "
              f"{os.path.relpath(REGISTRY_PATH, REPO_ROOT)} "
              f"(+{len(written['added'])} new registry rows)")
    _print_summary(result, registry)
    if args.check and new_orphans(result, registry):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())