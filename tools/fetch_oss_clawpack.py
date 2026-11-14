"""Fetch + pin the OSS reference implementation (Clawpack) used by the XVAL study.

Why this tool exists
--------------------
``docs/XVAL-DG-ROUND2-CERTIFIED-REFERENCE.md`` rests on a *third-party* solver
(Clawpack 5.14.0 / pyclaw SharpClaw).  Three things about that dependency were
not reproducible from the repository:

1. **Provenance.** The first version of the study recorded ``clawpack unknown``
   for the version and ``missing`` for the sha256 of
   ``clawpack/riemann/euler_1D_py.py`` -- i.e. the evidence bundle could not say
   *which bytes* of the Riemann solver produced the certified reference.
2. **Assembly.** The released sdist is not importable as-is: it is a monorepo
   layout (``pyclaw/src/pyclaw``, ``riemann/riemann``) and it ships *without*
   the f2py-generated Riemann modules.  Getting ``import clawpack.riemann`` to
   work therefore involved undocumented local edits.
3. **Repeatability.** The cached OSS profiles lived outside the repository, so
   every number in the document was unreachable from a clean checkout.

This tool closes (1) and (2): it downloads the pinned sdist, verifies its sha256,
unpacks it, records the sha256 of every file the study actually leans on (at its
*sdist-relative* path, so it can be found in the unpacked tree), and creates the
missing f2py modules as **empty, explicitly logged placeholders** -- only the
pure-Python kernels are used by this study, so the placeholders are never
executed.  Nothing else is touched: the released sources stay byte-identical.

(3) is closed by ``tools/verify_xval_doc.py`` + the committed fixture
``artifacts/xval_round2/``.

Placeholder policy (honest bookkeeping)
---------------------------------------
An empty ``.py`` file is *not* a re-implementation, it is a hole.  It is created
only because ``clawpack/riemann/__init__.py`` imports every solver eagerly, and
each placeholder created is listed in the returned manifest
(``placeholders``) so the evidence bundle states exactly what was hollowed out.
``verify_oss_tree`` asserts the study's own files are unmodified, and the
placeholder list is committed next to the certified reference.

Usage::

    python -m tools.fetch_oss_clawpack --dest scratch/oss \\
        --manifest artifacts/xval_round2/oss_provenance.json

    # offline / air-gapped: reuse an already downloaded tarball
    python -m tools.fetch_oss_clawpack --tarball /tmp/clawpack-5.14.0.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

#: pinned release: (version, url, sha256, size) -- zero transcription, PyPI JSON
CLAWPACK = {
    "version": "5.14.0",
    "url": ("https://files.pythonhosted.org/packages/6f/58/"
            "69deae7b519e581937d7431af0ae72cc6ab9432caf838601147fdb4cacb0/"
            "clawpack-5.14.0.tar.gz"),
    "sha256": "b9f3208fe9e1d834122ae86e2af536a3010435d0d65e6629f360416ea9cffe89",
    "bytes": 9748479,
}
TARBALL_NAME = "clawpack-{}.tar.gz".format(CLAWPACK["version"])
#: top-level directory inside the sdist
SDIST_ROOT = "clawpack-{}".format(CLAWPACK["version"])

#: files the XVAL study leans on, as *sdist-relative* paths
#: (``clawpack/riemann/...`` is the post-install path of the same file; both are
#: resolved by :func:`resolve_study_files`)
STUDY_FILES = (
    "riemann/riemann/euler_1D_py.py",
    "pyclaw/src/pyclaw/sharpclaw/solver.py",
    "pyclaw/src/pyclaw/limiters/recon.py",
    "pyclaw/src/pyclaw/solver.py",
    "pyclaw/src/pyclaw/state.py",
    "clawpack/__init__.py",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(dest_tarball: Path, *, timeout: float = 300.0) -> Path:
    """Download the pinned sdist and refuse anything but the pinned bytes."""
    dest_tarball.parent.mkdir(parents=True, exist_ok=True)
    if dest_tarball.exists() and sha256_of(dest_tarball) == CLAWPACK["sha256"]:
        return dest_tarball
    tmp = dest_tarball.with_suffix(".part")
    with urllib.request.urlopen(CLAWPACK["url"], timeout=timeout) as resp:
        tmp.write_bytes(resp.read())
    got = sha256_of(tmp)
    if got != CLAWPACK["sha256"]:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"clawpack sdist sha256 mismatch: got {got}, pinned {CLAWPACK['sha256']}"
            f" -- refusing to build evidence on unpinned bytes")
    tmp.replace(dest_tarball)
    return dest_tarball


def unpack(tarball: Path, dest: Path) -> Path:
    """Unpack into ``dest`` and return the sdist root directory."""
    root = dest / SDIST_ROOT
    if (root / "clawpack" / "__init__.py").exists():
        return root
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    if not (root / "clawpack" / "__init__.py").exists():
        raise SystemExit(f"{tarball} did not unpack to the expected layout ({root})")
    return root


def make_importable(root: Path, riemann_pkg: Path | None = None) -> list[str]:
    """Create empty placeholders for the f2py modules the sdist does not ship.

    ``clawpack/riemann/__init__.py`` imports every solver eagerly; the compiled
    ones are build artefacts and are absent from the sdist.  Import until it
    succeeds (or until no progress is made), creating one empty module per
    missing name.  Returns the list of created placeholders.
    """
    riemann_pkg = riemann_pkg or (root / "riemann" / "riemann")
    placeholders: list[str] = []
    for _ in range(200):
        sys.path.insert(0, str(root))
        try:
            for name in [m for m in list(sys.modules) if m.startswith("clawpack")]:
                del sys.modules[name]
            importlib.import_module("clawpack.riemann")
            return placeholders
        except ImportError as exc:
            missing = getattr(exc, "name", None)
            short = (missing or "").split(".")[-1]
            target = riemann_pkg / f"{short}.py"
            if not short or target.exists() or not riemann_pkg.is_dir():
                raise SystemExit(
                    f"clawpack.riemann cannot be assembled offline: {exc}") from exc
            target.write_text(
                f'"""Placeholder created by tools/fetch_oss_clawpack.py.\n\n'
                f'The compiled (f2py) solver ``{short}`` is not shipped in the\n'
                f'clawpack sdist. This study only uses the pure-Python kernels\n'
                f'(``euler_1D_py.euler_hllc_1D``), so this module is never called.\n"""\n')
            placeholders.append(str(target.relative_to(root)))
        finally:
            if sys.path and sys.path[0] == str(root):
                sys.path.pop(0)
    raise SystemExit("clawpack.riemann still not importable after 200 placeholders")


def resolve_study_files(root: Path) -> dict:
    """sha256 of every file the study leans on, keyed by sdist-relative path."""
    out = {}
    for rel in STUDY_FILES:
        path = root / rel
        if not path.exists():
            out[rel] = {"missing": True}
            continue
        out[rel] = {"sha256": sha256_of(path), "bytes": path.stat().st_size}
    return out


def provenance(root: Path, tarball: Path, placeholders: list[str]) -> dict:
    return {
        "package": "clawpack",
        "version": CLAWPACK["version"],
        "url": CLAWPACK["url"],
        "tarball": {"name": TARBALL_NAME, "sha256": sha256_of(tarball),
                    "bytes": tarball.stat().st_size},
        "sdist_root": str(root),
        "study_files": resolve_study_files(root),
        "placeholders": placeholders,
        "python_only": True,
        "note": ("pure-Python kernels only (pyclaw SharpClaw Python solver + "
                 "riemann.euler_1D_py); compiled Riemann modules are absent from "
                 "the sdist and replaced by empty placeholders that are never called"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="scratch/oss", help="unpack directory")
    ap.add_argument("--tarball", help="reuse this local sdist instead of downloading")
    ap.add_argument("--manifest", help="write the provenance manifest (json) here")
    args = ap.parse_args(argv)

    dest = Path(args.dest)
    tarball = Path(args.tarball) if args.tarball else dest / TARBALL_NAME
    if args.tarball:
        got = sha256_of(tarball)
        if got != CLAWPACK["sha256"]:
            raise SystemExit(f"local tarball sha256 {got} != pinned {CLAWPACK['sha256']}")
    else:
        tarball = download(tarball)
    root = unpack(tarball, dest)
    placeholders = make_importable(root)
    prov = provenance(root, tarball, placeholders)
    if args.manifest:
        out = Path(args.manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in prov.items() if k != "study_files"}, indent=2))
    print(json.dumps(prov["study_files"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())