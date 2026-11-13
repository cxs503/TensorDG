"""One place that knows where the Shu--Osher run artifacts live.

Two layers hold the same numbers:

``runs/shu_osher_case_pool/``
    Output of the full study (``run_all.sh``).  Regenerable, git-ignored, so it
    never reaches ``main``: on a clean checkout it simply does not exist.

``validation/repro_shu_osher/artifacts/``
    The curated, git-tracked copy of exactly those files (``metrics.json``,
    ``profiles.npz``, ``raw/*.npz``) plus the smoke record.

Every consumer in this directory therefore asks :func:`resolve_run_dir` instead
of hard-coding ``runs/...``: the documented commands keep working on a fresh
checkout against the curated copy, and the returned ``source`` tag says which
layer was read.  Override the search with ``TENSORDG_RUN_DIR=<dir>``.

``python validation/repro_shu_osher/bundle.py --verify`` re-hashes the curated
bundle and exits non-zero on any drift, so a third party can check the evidence
without trusting it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

#: ``<repo>/validation/repro_shu_osher``
BUNDLE = Path(__file__).resolve().parent
#: ``<repo>``
REPO = BUNDLE.parents[1]
#: git-tracked curated copy of the run artifacts
CURATED = BUNDLE / "artifacts"
#: output of the full study (absent on a clean checkout)
FRESH_RUN = REPO / "runs" / "shu_osher_case_pool"

ENV_RUN_DIR = "TENSORDG_RUN_DIR"
REQUIRED = ("metrics.json",)


def _complete(directory: Path) -> bool:
    return all((directory / name).is_file() for name in REQUIRED)


def candidate_run_dirs() -> list[tuple[Path, str]]:
    """Search order: ``$TENSORDG_RUN_DIR``, the fresh study, the curated copy."""
    out: list[tuple[Path, str]] = []
    env = os.environ.get(ENV_RUN_DIR)
    if env:
        out.append((Path(env).expanduser(), "env"))
    out.append((FRESH_RUN, "fresh"))
    out.append((CURATED, "curated"))
    return out


def resolve_run_dir() -> tuple[Path, str]:
    """Return ``(directory, source)`` for the run artifacts that are present.

    ``source`` is ``"env"`` (``$TENSORDG_RUN_DIR``), ``"fresh"`` (full study in
    ``runs/``) or ``"curated"`` (tracked bundle).  Raises ``FileNotFoundError``
    listing every directory tried when none carries ``metrics.json``.
    """
    tried = []
    for directory, source in candidate_run_dirs():
        tried.append(f"{directory} ({source})")
        if _complete(directory):
            return directory, source
    raise FileNotFoundError(
        "no Shu-Osher run artifacts found; run "
        "'bash validation/repro_shu_osher/run_all.sh' first. Tried: "
        + ", ".join(tried))


def load_metrics() -> tuple[dict, Path, str]:
    """``(metrics, path, source)`` read from :func:`resolve_run_dir`."""
    directory, source = resolve_run_dir()
    path = directory / "metrics.json"
    return json.loads(path.read_text()), path, source


def repo_relative(path: Path) -> str:
    """Manifest keys are repo-relative so every entry resolves from the root."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:  # pragma: no cover - artifact copied in from outside the tree
        return str(path)


def manifest_path() -> Path:
    return BUNDLE / "artifacts.sha256.json"


def resolve_manifest_entry(key: str, root: Path = REPO) -> Path | None:
    """Resolve one manifest key to a file, or ``None``.

    Keys written by ``make_evidence.py`` are repo-relative.  Bare names from
    older manifests are still accepted and looked up in the curated bundle, its
    ``raw/`` sub-directory and ``docs/figures/``.
    """
    candidate = Path(key)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    first = root / candidate
    if first.is_file():
        return first
    if candidate.parent == Path("."):  # legacy bare file name
        for fallback in (CURATED / key, CURATED / "raw" / key,
                         REPO / "docs" / "figures" / key):
            if fallback.is_file():
                return fallback
    return None


def verify_manifest(manifest: dict | None = None, root: Path = REPO) -> dict:
    """Re-hash every manifest entry; report per-entry status (no exceptions)."""
    if manifest is None:
        manifest = json.loads(manifest_path().read_text())
    entries = []
    for key, expected in manifest.items():
        path = resolve_manifest_entry(key, root=root)
        if path is None:
            entries.append({"key": key, "status": "missing", "expected": expected})
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({
            "key": key,
            "status": "ok" if got == expected else "mismatch",
            "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            "expected": expected,
            "got": got,
        })
    bad = [e["key"] for e in entries if e["status"] != "ok"]
    return {"ok": not bad, "checked": len(entries), "failed": bad, "entries": entries,
            "manifest": str(manifest_path())}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true",
                    help="re-hash the curated bundle and exit non-zero on drift")
    ap.add_argument("--quiet", action="store_true", help="one line per manifest entry only")
    args = ap.parse_args(argv)

    if not args.verify:
        ap.print_help()
        return 0
    report = verify_manifest()
    if args.quiet:
        for e in report["entries"]:
            print(f"{e['status']:9s} {e['key']}")
    else:
        print(json.dumps({k: report[k] for k in ("ok", "checked", "failed", "manifest")},
                         indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())