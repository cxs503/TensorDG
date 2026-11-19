"""Machine checks for the FEAT-OSS-INVENTORY-SPH wave-1 registry.

The deliverable is *evidence about third-party code*: the failure mode is silent
fabrication (hand-typed stars/licence/activity) or dead cross-references.  The
strongest check re-derives every numeric field from
``artifacts/oss_inventory_sph/github_probe_raw.json`` and fails on any divergence.

Run: python3 -m pytest tests/test_oss_inventory_sph_registry.py -q
"""
from __future__ import annotations

import datetime
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REG_PATH = ROOT / "docs" / "oss_inventory_sph_registry.json"
RAW_PATH = ROOT / "artifacts" / "oss_inventory_sph" / "github_probe_raw.json"
SLUG_RE = re.compile(r"^GAP-SPH-[A-Z0-9]+(-[A-Z0-9]+)*$")
STAT_FIELDS = ("stars", "forks", "language", "license", "pushed_at", "archived", "size_kb")


@pytest.fixture(scope="module")
def reg():
    assert REG_PATH.exists(), f"missing registry {REG_PATH}"
    return json.loads(REG_PATH.read_text())


@pytest.fixture(scope="module")
def raw():
    assert RAW_PATH.exists(), f"missing raw probe {RAW_PATH}"
    d = json.loads(RAW_PATH.read_text())
    items = {}
    for q, v in d["search"].items():
        for it in v.get("items", []):
            fn = it.get("full_name")
            if fn and fn not in items:
                items[fn] = it
    for fn, v in d.get("detail", {}).items():
        if v.get("ok") and fn not in items:
            items[fn] = v["raw"]
    return d, items


def test_schema_and_identity(reg):
    for key in ("schema_version", "feature_anchor", "wave", "protocol", "provenance",
                "tier_rubric", "capability_taxonomy", "coverage", "repos", "reserve",
                "gap_proposals", "limitations"):
        assert key in reg, f"registry missing key {key}"
    assert reg["feature_anchor"] == "FEAT-OSS-INVENTORY-SPH"
    assert reg["schema_version"] == "1.0"
    assert reg["wave"] == 1


def test_census_size_and_tiering(reg):
    repos = reg["repos"]
    assert len(repos) >= 20, f"census needs >=20 repos, got {len(repos)}"
    tiers = {r["tier"] for r in repos}
    assert tiers == {1, 2, 3}, f"all three tiers must be populated, got {tiers}"
    counts = {t: sum(1 for r in repos if r["tier"] == t) for t in (1, 2, 3)}
    assert counts[1] >= 5, f"tier-1 (core SPH solvers) too thin: {counts[1]}"
    assert reg["coverage"]["tier_counts"] == {str(k): v for k, v in counts.items()}
    assert reg["coverage"]["n_repos"] == len(repos)


def test_no_duplicate_repos(reg):
    names = [r["repo"] for r in reg["repos"]]
    assert len(names) == len(set(names)), "duplicate repos in census"


def test_numeric_fields_match_raw_probe(reg, raw):
    _, items = raw
    for r in reg["repos"]:
        it = items.get(r["repo"])
        assert it is not None, f"repo {r['repo']} has no raw evidence in probe"
        for f in STAT_FIELDS:
            if f == "license":
                expect = (it.get("license") or {}).get("spdx_id")
            elif f == "size_kb":
                expect = it.get("size")
            else:
                expect = it.get(f)
            if expect is None:
                assert r[f] is None, f"{r['repo']}.{f}: raw null but registry {r[f]!r}"
                continue
            assert r[f] == expect, f"{r['repo']}.{f}: registry {r[f]!r} != raw {expect!r}"


def test_activity_and_archives(reg):
    now = datetime.datetime.now(datetime.timezone.utc)
    recent = 0
    for r in reg["repos"]:
        assert r["archived"] in (True, False)
        pushed = datetime.datetime.fromisoformat(r["pushed_at"].replace("Z", "+00:00"))
        if (now - pushed).days <= 365:
            recent += 1
    assert recent / len(reg["repos"]) >= 0.3, "activity: <30% of census pushed within a year"


def test_gap_proposals(reg):
    gaps = reg["gap_proposals"]
    assert len(gaps) >= 4, f"need >=4 GAP-SPH-* proposals, got {len(gaps)}"
    repo_names = {r["repo"] for r in reg["repos"]}
    seen = set()
    for g in gaps:
        assert SLUG_RE.match(g["slug"]), f"bad slug {g['slug']}"
        assert g["slug"] not in seen, f"duplicate slug {g['slug']}"
        seen.add(g["slug"])
        assert len(g["evidence_repos"]) >= 2, f"{g['slug']} needs >=2 evidence repos"
        for e in g["evidence_repos"]:
            assert e in repo_names, f"{g['slug']}: evidence repo {e} not in census"
        assert g["acceptance"], f"{g['slug']} missing quantitative acceptance"
        assert any(ch.isdigit() for ch in g["acceptance"]), f"{g['slug']} acceptance not quantitative"
        assert g["priority"] in ("P1", "P2", "P3")


def test_focus_and_role_fields(reg):
    for r in reg["repos"]:
        assert r["focus"], f"{r['repo']} missing focus"
        assert r["sph_role"] in ("core-sph-solver", "sph-capable-framework", "ecosystem-infrastructure")
        assert r["url"] == f"https://github.com/{r['repo']}"
        assert r["evidence_source"] in ("search", "detail")


def test_dropped_honesty(reg, raw):
    _, items = raw
    dropped = set(reg["coverage"]["dropped_no_evidence"])
    for fn in dropped:
        assert fn not in items, f"{fn} listed as dropped but has raw evidence"
    assert reg["coverage"]["n_census_dropped_no_evidence"] == len(dropped)


def test_language_counts(reg):
    counts = {}
    for r in reg["repos"]:
        k = r["language"] or "unknown"
        counts[k] = counts.get(k, 0) + 1
    assert reg["coverage"]["language_counts"] == counts
