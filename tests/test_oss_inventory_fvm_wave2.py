"""Gate for the FEAT-OSS-INVENTORY-FVM wave-2 pointer ledger.

The deliverable is *evidence about third-party code*: the realistic failure mode
is a fabricated or drifted ``repo:path:line`` pointer, or a registry that claims
more coverage than the raw GitHub cache supports.  The strongest check therefore
re-derives every published pointer **offline** from
``artifacts/oss_inventory_fvm_wave2/raw/`` (base64 blob -> needle regex -> line
number -> verbatim snippet) and fails on any divergence.

Run: python3 -m pytest tests/test_oss_inventory_fvm_wave2.py -q
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import probe_fvm_wave2_pointers as P  # noqa: E402

REG_PATH = ROOT / "docs" / "oss_inventory_fvm_wave2_pointers.json"
RAW_DIR = ROOT / "artifacts" / "oss_inventory_fvm_wave2" / "raw"
GAP_RE = re.compile(r"^GAP-FVM-[A-Z0-9]+(-[A-Z0-9]+)*$")
STATUSES = {"verified", "path_not_found", "needle_not_found", "unavailable"}
WAVE2_GAPS = {
    "GAP-FVM-AMR-QUADTREE",
    "GAP-FVM-IMPLICIT-DUAL-TIME",
    "GAP-FVM-LOWMACH-PRECOND",
    "GAP-FVM-MRF-SLIDING-AMI",
    "GAP-FVM-MPI-DECOMP",
    "GAP-FVM-AMG-KRYLOV",
    "GAP-FVM-COMBUSTION-SPECIES",
    "GAP-FVM-RADIATION-P1",
}


@pytest.fixture(scope="module")
def reg():
    assert REG_PATH.exists(), f"missing registry {REG_PATH}"
    return json.loads(REG_PATH.read_text(encoding="utf-8"))


def _pointers(reg):
    return [p for blk in reg["gaps"].values() for p in blk["pointers"]]


def test_schema_and_identity(reg):
    for key in ("schema_version", "feature_anchor", "wave", "protocol",
                "provenance", "coverage", "gaps", "limitations"):
        assert key in reg, f"registry missing {key}"
    assert reg["feature_anchor"] == "FEAT-OSS-INVENTORY-FVM"
    assert reg["wave"] == 2
    assert reg["schema_version"] == "1.0"


def test_gap_ids_are_fvm_anchors_and_cover_wave1(reg):
    ids = set(reg["gaps"])
    for gid in ids:
        assert GAP_RE.match(gid), f"bad gap slug {gid}"
    # wave-2 must speak about every wave-1 gap, not a convenient subset
    assert ids == WAVE2_GAPS, f"gap set drift: {ids ^ WAVE2_GAPS}"


def test_every_gap_has_a_verified_pointer(reg):
    for gid, blk in reg["gaps"].items():
        assert blk["n_verified"] >= 1, f"{gid} has no machine-verified upstream pointer"
        assert blk["n_verified"] <= blk["n_specs"]
        assert blk["n_specs"] == len(blk["pointers"])


def test_status_enum_and_pointer_fields(reg):
    for p in _pointers(reg):
        assert p["status"] in STATUSES, p["status"]
        assert p["gap"] in WAVE2_GAPS
        re.compile(p["path_rx"])
        re.compile(p["needle_rx"])
        if p["status"] == "verified":
            for field in ("path", "line", "snippet", "blob_sha", "needle"):
                assert p.get(field) not in (None, ""), f"{p['gap']}/{p['repo']} missing {field}"
            assert isinstance(p["line"], int) and p["line"] >= 1
            assert re.search(p["path_rx"], p["path"]), "path does not satisfy its own path_rx"
            assert re.search(p["needle_rx"], p["snippet"]), "snippet lacks the needle"


def test_no_duplicate_pointers(reg):
    seen = set()
    for p in _pointers(reg):
        if p["status"] != "verified":
            continue
        key = (p["gap"], p["repo"], p["path"])
        assert key not in seen, f"duplicate pointer {key}"
        seen.add(key)


def test_coverage_stats_are_self_consistent(reg):
    ptrs = _pointers(reg)
    cov = reg["coverage"]
    assert cov["specs"] == len(ptrs)
    assert cov["verified"] == sum(1 for p in ptrs if p["status"] == "verified")
    assert cov["gaps_targeted"] == len(reg["gaps"])
    assert cov["gaps_with_verified_pointer"] == sum(
        1 for blk in reg["gaps"].values() if blk["n_verified"] > 0)
    assert cov["repos_touched"] == sorted({p["repo"] for p in ptrs})
    for status, n in cov["status_counts"].items():
        assert n == sum(1 for p in ptrs if p["status"] == status)


def test_raw_cache_present_and_trees_pinned(reg):
    assert RAW_DIR.is_dir(), "raw GitHub cache missing: pointers would be unfalsifiable"
    for repo, meta in reg["provenance"]["trees"].items():
        assert meta.get("tree_sha"), f"tree sha not pinned for {repo}"
        d = RAW_DIR / repo.replace("/", "__")
        assert d.is_dir(), f"raw dir missing for {repo}"
        trees = list(d.glob("tree__*.json"))
        assert trees, f"no cached tree for {repo}"
        payload = json.loads(trees[0].read_text())
        assert payload.get("sha") == meta["tree_sha"], f"tree sha drift for {repo}"


def test_pointers_rederive_offline_from_raw_cache(reg):
    errs = P.rederive(reg)
    assert not errs, "pointer re-derivation failed:\n  " + "\n  ".join(errs)


def test_no_pointer_points_outside_its_repo_raw_dir(reg):
    for p in _pointers(reg):
        if p["status"] != "verified":
            continue
        expected = RAW_DIR / p["repo"].replace("/", "__") / "blobs"
        assert expected.is_dir(), f"raw blob dir missing for {p['repo']}"