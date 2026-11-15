"""Validation for the LBM open-source method set (FEAT-OSS-INVENTORY-LBM wave-2).

The registry must stay a *machine-checked* aggregation: every entry carries a
verbatim-copied source row, a file pointer extracted from that row and a
migration assessment, and the published artefacts must regenerate bit-for-bit.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_oss_solutions_lbm as B  # noqa: E402
import oss_solutions_lbm_taxonomy as T  # noqa: E402

REG_PATH = ROOT / "docs" / "oss_solutions_lbm_registry.json"
MD_PATH = ROOT / "docs" / "OSS-SOLUTIONS-LBM.md"
SQL_PATH = ROOT / "artifacts" / "oss_solutions_lbm" / "insert.sql"
ROWS_PATH = ROOT / "artifacts" / "oss_solutions_lbm" / "source_rows.json"


def _reg():
    return json.loads(REG_PATH.read_text(encoding="utf-8"))


def test_registry_exists_and_is_valid_json():
    reg = _reg()
    assert reg["anchor"]["slug"] == T.ANCHOR_SLUG
    assert reg["schema_version"] == T.SCHEMA_VERSION


def test_scale_meets_acceptance():
    st = _reg()["stats"]
    assert st["n_func_points"] >= T.MIN_FUNC_POINTS, st
    assert st["n_entries"] >= T.MIN_ENTRIES, st
    assert st["n_repos_cited"] >= 24, st  # acceptance: 24+ projects


def test_every_func_point_has_unique_id_and_category():
    fps = _reg()["functional_points"]
    ids = [fp["id"] for fp in fps]
    assert len(ids) == len(set(ids))
    cat_keys = {c["key"] for c in _reg()["categories"]}
    for fp in fps:
        assert fp["category"] in cat_keys
        assert fp["name"].strip()
        assert fp["why_for_tensorlbm"].strip()
        assert fp["n_solutions"] == len(fp["solutions"]) >= 1


def test_every_entry_has_file_pointer_and_migration_assessment():
    fps = _reg()["functional_points"]
    for fp in fps:
        for s in fp["solutions"]:
            assert s["pointer_kind"] in ("file:line", "file"), (fp["id"], s["repo"])
            assert s["upstream_path"], (fp["id"], s["repo"], s["tag"])
            if s["pointer_kind"] == "file:line":
                assert B.POINTER_RE.fullmatch(s["upstream_ref"]), s["upstream_ref"]
                assert s["upstream_line"] >= 1
            assert s["migration_level"] in T.MIGRATION_NAME
            assert s["assessment_verbatim"].strip(), (fp["id"], s["repo"])
            assert len(s["method_verbatim"].strip()) >= 20, (fp["id"], s["repo"])


def test_entries_are_verbatim_copies_of_source_rows():
    rows = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    by_key = {(r["repo"], r["feature_slug"], r["tag"]): r for r in rows}
    n = 0
    for fp in _reg()["functional_points"]:
        for s in fp["solutions"]:
            row = by_key[(s["repo"], s["source_feature_slug"], s["tag"])]
            assert s["method_verbatim"] == (row["method"] or "")
            assert s["files_verbatim"] == (row["files"] or "").strip()
            assert s["assessment_verbatim"] == (row["transfer"] or "")
            assert s["source_row_id"] == row["id"]
            n += 1
    assert n == _reg()["stats"]["n_entries"]


def test_no_first_party_repo_used_as_oss_evidence():
    for fp in _reg()["functional_points"]:
        for s in fp["solutions"]:
            assert s["repo"] not in T.FIRST_PARTY, (fp["id"], s["repo"])


def test_migration_levels_are_classified_from_transfer_prefix():
    for fp in _reg()["functional_points"]:
        for s in fp["solutions"]:
            assert s["migration_level"] == B.classify_migration(s["assessment_verbatim"])


def test_assessments_are_substantive_for_most_entries():
    entries = [s for fp in _reg()["functional_points"] for s in fp["solutions"]]
    substantive = sum(1 for s in entries if len(s["assessment_verbatim"].strip()) >= 20)
    assert substantive / len(entries) >= 0.8, (substantive, len(entries))


def test_stats_and_category_stats_are_self_consistent():
    reg = _reg()
    st = reg["stats"]
    entries = [s for fp in reg["functional_points"] for s in fp["solutions"]]
    assert st["n_entries"] == len(entries)
    assert st["n_solution_entries"] + st["n_gap_entries"] == st["n_entries"]
    assert st["n_repos_cited"] == len({s["repo"] for s in entries})
    assert st["n_upstream_pointers"] == sum(1 for s in entries if s["pointer_kind"] == "file:line")
    mig = {}
    for s in entries:
        mig[s["migration_level"]] = mig.get(s["migration_level"], 0) + 1
    for k, v in mig.items():
        assert st["migration_level_counts"][k] == v
    assert sum(c["n_entries"] for c in reg["category_stats"]) == st["n_entries"]
    assert sum(c["n_func_points"] for c in reg["category_stats"]) == st["n_func_points"]


def test_selectors_still_resolve_to_exactly_one_row():
    reg = _reg()
    assert len(reg["functional_points"]) == len(T.FP)


def test_sql_injection_artifact_matches_entry_count():
    sql = SQL_PATH.read_text(encoding="utf-8")
    n_insert = len(re.findall(r"^INSERT INTO oss_solutions", sql, flags=re.M))
    assert n_insert == _reg()["stats"]["n_entries"]
    assert T.ANCHOR_SLUG in sql
    assert sql.count("BEGIN;") == 1 and sql.count("COMMIT;") == 1
    # no unescaped single quote breakage: every VALUES tuple has exactly 8 fields
    for line in [l for l in sql.splitlines() if l.startswith("INSERT")]:
        assert line.count("'") % 2 == 0, line[:120]


def test_markdown_report_covers_required_sections():
    md = MD_PATH.read_text(encoding="utf-8")
    for token in ("FEAT-OSS-INVENTORY-LBM", "怎么做的", "误差判定", "注意点", "功能点清单"):
        assert token in md, token


def test_regeneration_is_deterministic():
    reg = B.build()
    on_disk = json.loads(REG_PATH.read_text(encoding="utf-8"))
    assert json.dumps(reg, sort_keys=True) == json.dumps(on_disk, sort_keys=True)
    assert B.render_md(on_disk) == MD_PATH.read_text(encoding="utf-8")
    assert B.render_sql(on_disk) == SQL_PATH.read_text(encoding="utf-8")