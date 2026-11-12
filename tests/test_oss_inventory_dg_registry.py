"""Machine checks for the FEAT-OSS-INVENTORY-DG wave-1 registry.

These tests exist because the deliverable is *evidence about third-party code*:
the failure mode is silent fabrication (hand-typed stars/licence/activity) or
dead cross-references.  The strongest check here re-derives every numeric field
from ``artifacts/oss_inventory_dg/github_probe_raw.json`` and fails on any
divergence.

Run: python3 -m pytest tests/test_oss_inventory_dg_registry.py -q
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REG_PATH = ROOT / "docs" / "oss_inventory_dg_registry.json"
RAW_PATH = ROOT / "artifacts" / "oss_inventory_dg" / "github_probe_raw.json"
SLUG_RE = re.compile(r"^GAP-DG-[A-Z0-9]+(-[A-Z0-9]+)*$")
STAT_FIELDS = ("stars", "forks", "language", "license", "pushed_at", "archived", "size_kb")


@pytest.fixture(scope="module")
def reg():
    assert REG_PATH.exists(), f"missing registry {REG_PATH}"
    return json.loads(REG_PATH.read_text())


@pytest.fixture(scope="module")
def raw():
    assert RAW_PATH.exists(), f"missing raw probe {RAW_PATH}"
    d = json.loads(RAW_PATH.read_text())
    ok = {v["full_name"]: v for v in d["candidates"].values() if v.get("ok")}
    return d, ok


def test_schema_and_identity(reg):
    for key in ("schema_version", "feature_anchor", "protocol", "tier_rubric",
                "capability_taxonomy", "coverage", "repos", "reserve",
                "gap_proposals", "limitations", "provenance"):
        assert key in reg, f"registry missing key {key}"
    assert reg["feature_anchor"] == "FEAT-OSS-INVENTORY-DG"
    assert reg["schema_version"] == "1.0"


def test_census_size_and_tiering(reg):
    repos = reg["repos"]
    assert len(repos) >= 20, f"census needs >=20 repos, got {len(repos)}"
    tiers = {r["tier"] for r in repos}
    assert tiers <= {1, 2, 3}, f"illegal tiers {tiers}"
    assert tiers == {1, 2, 3}, "all three tiers must be populated"
    counts = {t: sum(1 for r in repos if r["tier"] == t) for t in (1, 2, 3)}
    assert counts[1] >= 5, f"tier-1 (core DG solvers) too thin: {counts[1]}"
    assert reg["coverage"]["tier_counts"] == {str(k): v for k, v in counts.items()}


def test_no_duplicate_repos(reg):
    names = [r["repo"] for r in reg["repos"]]
    assert len(names) == len(set(names)), "duplicate repo entries"
    assert len({n.lower() for n in names}) == len(names), "duplicate repo (case-insensitive)"


def test_every_repo_is_fully_populated(reg):
    taxonomy = set(reg["capability_taxonomy"])
    gap_slugs = {g["slug"] for g in reg["gap_proposals"]}
    for r in reg["repos"]:
        for field in ("repo", "url", "focus", "dg_role", "capabilities",
                      "tensordg_gaps", "provenance"):
            assert r.get(field), f"{r.get('repo')}: empty {field}"
        assert r["url"] == "https://github.com/" + r["repo"]
        assert r["capabilities"], f"{r['repo']} has no capability tags"
        assert set(r["capabilities"]) <= taxonomy, f"{r['repo']} unknown capability tag"
        assert set(r["tensordg_gaps"]) <= gap_slugs, f"{r['repo']} anchors unknown gap"
        assert isinstance(r["focus"], str) and len(r["focus"]) >= 20, \
            f"{r['repo']}: focus too thin to be informative"
        for f in STAT_FIELDS:
            assert f in r, f"{r['repo']}: missing stat {f}"


def test_all_numeric_metadata_matches_raw_probe(reg, raw):
    """Zero-hand-copy invariant: stats must be byte-identical to the probe."""
    _, ok = raw
    for r in reg["repos"] + reg["reserve"]:
        name = r["repo"]
        assert name in ok, f"{name} not present as a successful probe"
        meta = ok[name]
        for f in STAT_FIELDS:
            if f in r:
                assert r[f] == meta[f], f"{name}.{f}: registry {r[f]!r} != probe {meta[f]!r}"


def test_probe_provenance_hash(reg, raw):
    d, _ = raw
    assert reg["provenance"]["raw_sha256"] == hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(), \
        "raw probe changed after registry build - rerun tools/build_oss_inventory_dg.py"
    assert d.get("probed_at"), "probe lacks timestamp"


def test_pushed_at_parsable_and_recent(reg):
    newest = datetime.datetime.now(datetime.timezone.utc)
    for r in reg["repos"]:
        ts = datetime.datetime.strptime(r["pushed_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc)
        assert ts <= newest + datetime.timedelta(hours=6), f"{r['repo']}: future pushed_at"
        # at least 60% of the census must have moved within 12 months = live ecosystem
    live = sum(1 for r in reg["repos"]
               if datetime.datetime.strptime(r["pushed_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
                   tzinfo=datetime.timezone.utc) > newest - datetime.timedelta(days=365))
    assert live >= 0.6 * len(reg["repos"]), f"only {live}/{len(reg['repos'])} repos active in 12m"


def test_gap_proposals_are_quantified_and_evidenced(reg):
    gaps = reg["gap_proposals"]
    assert len(gaps) >= 4, f"acceptance requires >=4 GAP-DG-* anchors, got {len(gaps)}"
    known = {r["repo"] for r in reg["repos"]}
    seen = set()
    for g in gaps:
        assert SLUG_RE.match(g["slug"]), f"bad slug {g['slug']}"
        assert g["slug"] not in seen, f"duplicate slug {g['slug']}"
        seen.add(g["slug"])
        for f in ("title", "rationale", "acceptance", "domain", "priority", "repos_evidence"):
            assert g.get(f), f"{g['slug']}: empty {f}"
        acc = g["acceptance"]
        # quantified acceptance: must carry a numeric threshold, not prose
        assert re.search(r"\d", acc) and re.search(r"(<|>=|<=|减少|x)", acc), \
            f"{g['slug']}: acceptance not quantified: {acc!r}"
        assert g["priority"] in {"P1", "P2", "P3"}
        assert len(g["repos_evidence"]) >= 1, f"{g['slug']}: no evidence repos"
        assert set(g["repos_evidence"]) <= known, \
            f"{g['slug']}: evidence {set(g['repos_evidence']) - known} not in registry"


def test_every_gap_is_anchored_by_at_least_two_repos(reg):
    """A gap nobody's code demonstrates is a wish, not an anchor."""
    back = {g["slug"]: set(g["repos_evidence"]) for g in reg["gap_proposals"]}
    for g in reg["gap_proposals"]:
        referenced_by = {r["repo"] for r in reg["repos"] if g["slug"] in r["tensordg_gaps"]}
        assert len(referenced_by | back[g["slug"]]) >= 2, \
            f"{g['slug']}: fewer than 2 repos anchor it"


def test_coverage_block_is_consistent(reg):
    repos = reg["repos"]
    tax = reg["capability_taxonomy"]
    assert reg["coverage"]["repos_in_registry"] == len(repos)
    for cap, n in reg["coverage"]["capability_coverage"].items():
        assert cap in tax
        assert n == sum(1 for r in repos if cap in r["capabilities"]), f"{cap} coverage miscounted"


def test_registry_matches_regeneration(reg):
    """Committed file must be exactly what the builder produces (no drift)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import build_oss_inventory_dg as builder  # noqa: E402

    fresh = builder.build()
    a = dict(reg)
    a.pop("generated_at", None)
    fresh.pop("generated_at", None)
    assert a == fresh, "docs/oss_inventory_dg_registry.json is stale: rerun the builder"


def test_limitations_are_declared(reg):
    assert len(reg["limitations"]) >= 3, "wave-1 limitations must be disclosed explicitly"
    blob = " ".join(reg["limitations"])
    assert "wave-2" in blob or "波2" in blob, "must state that code-level deep-dive is wave 2"


def test_reserve_entries_carry_reasons(reg):
    for r in reg["reserve"]:
        assert r.get("reason_excluded"), f"{r['repo']}: exclusion reason missing"
        assert r["repo"] not in {x["repo"] for x in reg["repos"]}, \
            f"{r['repo']} both registered and excluded"

def test_doc_numbers_match_registry(reg):
    """The write-up must not drift from the machine-readable registry."""
    doc = (ROOT / "docs" / "OSS-INVENTORY-DG.md")
    assert doc.exists(), "report markdown missing"
    text = doc.read_text()
    n = len(reg["repos"])
    assert f"本册给出 {n} 仓" in text, "doc census count disagrees with registry"
    assert f"{n} 条" in text or f"锚点提议（{len(reg['gap_proposals'])}" in text
    assert f"GAP 提议 {len(reg['gap_proposals'])} 条" in text
    for g in reg["gap_proposals"]:
        assert g["slug"] in text, f"doc omits gap {g['slug']}"
    for fig in ("fig1_tier_activity.png", "fig2_capability_matrix.png", "fig3_gap_evidence.png"):
        assert fig in text, f"doc does not reference {fig}"
        assert (ROOT / "docs" / "figs" / fig).exists(), f"missing figure file {fig}"
