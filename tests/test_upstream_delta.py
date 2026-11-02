"""Contract tests for tools/upstream_delta.py.

Every case runs on a throw-away synthetic git repository, so the tests are
deterministic (fixed author/committer dates) and independent of the state of
the TensorDG history.  They pin the two things downstream reports rely on:

  * theme classification / risk tagging of a commit (path -> rule mapping);
  * the delta arithmetic (ahead/behind, identical, unmerged remote branch).
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import upstream_delta as ud  # noqa: E402

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
}


def _g(repo, *args):
    env = dict(os.environ, **GIT_ENV)
    proc = subprocess.run(["git", "-C", repo, *args], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _init(path):
    os.makedirs(path)
    _g(path, "init", "-q", "-b", "main")
    return path


def _commit(repo, files, msg):
    for rel, content in files.items():
        full = os.path.join(repo, rel)
        os.makedirs(os.path.dirname(full) or repo, exist_ok=True)
        with open(full, "w") as fh:
            fh.write(content)
    _g(repo, "add", "-A")
    _g(repo, "commit", "-q", "-m", msg)
    return _g(repo, "rev-parse", "HEAD")


# --------------------------------------------------------------------------
# theme / risk rules
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path,theme", [
    ("tensordg/limiters.py", "physics_core"),
    ("tensordg/sipg_matrixfree.py", "physics_core"),
    ("tensordg/__init__.py", "api"),
    ("tests/test_limiters.py", "tests"),
    (".github/workflows/ci.yml", "ci"),
    ("benchmarks/riemann_dg_error.py", "verification"),
    ("benchmarks/euler_sod_1d.py", "benchmarks"),
    ("docs/benchmarks/euler_sod.md", "docs"),
    ("validation_report.json", "artifacts"),
    ("limiter_comparison/sod_limiter_metrics.json", "artifacts"),
    ("pyproject.toml", "packaging"),
    ("something/unknown.xyz", "other"),
])
def test_theme_of(path, theme):
    assert ud.theme_of(path) == theme


def _rec(paths, merge=False):
    files = [{"path": p, "insertions": 1, "deletions": 0, "binary": False,
              "theme": ud.theme_of(p)} for p in paths]
    return {"sha": "0" * 40, "short": "0000000", "date": "2026-01-01",
            "subject": "s", "merge": merge, "files_changed": len(files),
            "insertions": len(files), "deletions": 0, "churn": len(files),
            "files": files}


def test_doc_only_commit_flags_r1_not_api():
    rules = [t["rule"] for t in ud.risk_tags(_rec(["docs/a.md", "artifacts/b.png"]))]
    assert "R1_DOC_ONLY" in rules
    assert "R4_API_SURFACE" not in rules
    assert "R5_PHYSICS_CORE" not in rules


def test_numeric_artifact_flags_r3_regeneration_risk():
    rules = [t["rule"] for t in ud.risk_tags(_rec(["validation_report.json"]))]
    assert "R3_NUMERIC_ARTIFACT" in rules
    assert [t["level"] for t in ud.risk_tags(_rec(["validation_report.json"]))
            if t["rule"] == "R3_NUMERIC_ARTIFACT"] == ["HIGH"]


def test_merge_commit_only_carries_r6():
    assert [t["rule"] for t in ud.risk_tags(_rec(["tensordg/limiters.py"], merge=True))] \
        == ["R6_MERGE_COMMIT"]


def test_physics_core_commit_flags_r2_r4_r5():
    rules = {t["rule"] for t in ud.risk_tags(
        _rec(["tensordg/euler_dg_1d.py", "tests/test_euler_sod_1d.py"]))}
    assert {"R2_TESTS_TOUCHED", "R4_API_SURFACE", "R5_PHYSICS_CORE"} <= rules


def test_large_churn_threshold():
    rec = _rec(["tensordg/heat_1d.py"])
    rec["churn"] = ud.LARGE_CHURN
    assert "R7_LARGE_CHURN" in [t["rule"] for t in ud.risk_tags(rec)]
    rec["churn"] = ud.LARGE_CHURN - 1
    assert "R7_LARGE_CHURN" not in [t["rule"] for t in ud.risk_tags(rec)]


def test_dominant_theme_ignores_docs():
    files = [{"theme": "docs"}, {"theme": "docs"}, {"theme": "physics_core"}]
    assert ud.dominant_theme(files) == "physics_core"


# --------------------------------------------------------------------------
# delta arithmetic on a synthetic repository
# --------------------------------------------------------------------------
@pytest.fixture()
def divergent_repo(tmp_path):
    """main: c1(docs) -> c2(physics) ; side branch from c1: c3(verification)."""
    repo = _init(str(tmp_path / "r"))
    c1 = _commit(repo, {"docs/a.md": "a\n"}, "docs only")
    c2 = _commit(repo, {"tensordg/limiters.py": "x = 1\n", "tests/test_x.py": "def t(): pass\n"},
                 "limiter core")
    _g(repo, "branch", "upstream/main", c2)
    _g(repo, "checkout", "-q", "-b", "side", c1)
    c3 = _commit(repo, {"tensordg/converge_poisson.py": "y = 2\n",
                        "tests/test_converge.py": "def t(): pass\n"}, "unmerged convergence")
    _g(repo, "update-ref", "refs/remotes/origin/side", c3)
    _g(repo, "checkout", "-q", "main")
    return repo, c1, c2, c3


def test_delta_identical_when_refs_match(divergent_repo):
    repo, _, c2, _ = divergent_repo
    d = ud.branch_delta(repo, "upstream/main", "main")
    assert d["identical"] is True
    assert d["ahead"] == [] and d["behind"] == []
    assert d["base_sha"] == d["head_sha"] == c2


def test_delta_reports_ahead_and_behind(divergent_repo):
    repo, c1, c2, c3 = divergent_repo
    d = ud.branch_delta(repo, "side", "main")       # main is 1 ahead of side
    assert d["identical"] is False
    assert [c["sha"] for c in d["ahead"]] == [c2]
    assert [c["sha"] for c in d["behind"]] == [c3]   # side-only commit
    assert d["merge_base"] == c1


def test_unmerged_branch_detected_with_theme_and_risk(divergent_repo):
    repo, _, _, c3 = divergent_repo
    br = ud.unmerged_branches(repo, "main", exclude=("refs/remotes/origin/main",))
    assert len(br) == 1
    b = br[0]
    assert b["sha"] == c3 and b["commits_ahead"] == 1
    assert "verification" in b["themes"] and "tests" in b["themes"]
    assert b["risks"][0]["rule"] == "R9_UNMERGED_BRANCH"
    assert b["insertions"] == 2


def test_build_acceptance_block_and_artifact_roundtrip(divergent_repo, tmp_path):
    repo, _, _, _ = divergent_repo
    payload = ud.build(repo, "upstream/main", "main", recent=2, remote="origin")
    acc = payload["acceptance"]
    assert payload["schema"] == ud.SCHEMA
    assert acc["main_pinned_to_upstream"] is True
    assert acc["unmerged_branch_count"] == 1
    assert acc["recent_commit_count"] == 2
    # every recent commit carries a risk list (possibly empty, never None)
    assert all(isinstance(r["risks"], list) for r in payload["recent_commits"])

    out = tmp_path / "d.json"
    assert ud.main(["--repo", repo, "--base", "upstream/main", "--head", "main",
                    "--recent", "2", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["acceptance"]["analysis_complete"] is True
    # --check must accept the artifact it just wrote ...
    assert ud.main(["--repo", repo, "--base", "upstream/main", "--head", "main",
                    "--recent", "2", "--check", str(out)]) == 0
    # ... and reject it once the repository moves on.
    _commit(repo, {"docs/b.md": "b\n"}, "later doc")
    assert ud.main(["--repo", repo, "--base", "upstream/main", "--head", "main",
                    "--recent", "2", "--check", str(out)]) == 1


def test_real_repository_artifact_is_fresh():
    """The committed artifact must match the repository it was generated from."""
    art = os.path.join(ROOT, "artifacts", "upstream_delta_m1634.json")
    if not os.path.exists(art):
        pytest.skip("artifact not generated")
    payload = json.loads(open(art, encoding="utf-8").read())
    assert payload["schema"] == ud.SCHEMA
    if subprocess.run(["git", "-C", ROOT, "rev-parse", "--verify", "--quiet",
                       "origin/main"], stdout=subprocess.DEVNULL).returncode != 0:
        pytest.skip("no origin/main in this checkout")
    # the audit is pinned to the sha it audited (see docs/ ⑤): HEAD moves with
    # every commit, including this mission's own, and must not rot the artifact
    assert ud.main(["--repo", ROOT, "--base", "origin/main",
                    "--head", payload["head"]["sha"],
                    "--recent", "15", "--check", art]) == 0

def test_risk_tally_incl_upstream_superset():
    """The upstream-aware tally must never lose a recent-commit risk tag."""
    art = os.path.join(ROOT, "artifacts", "upstream_delta_m1634.json")
    if not os.path.exists(art):
        pytest.skip("artifact not generated")
    d = json.loads(open(art, encoding="utf-8").read())
    recent, allv = d["risk_tally_by_level"], d["risk_tally_by_level_incl_upstream"]
    for level, n in recent.items():
        assert allv.get(level, 0) >= n


def test_rendered_report_matches_committed_doc():
    """docs/*.md must be byte-identical to what the generator produces now."""
    art = os.path.join(ROOT, "artifacts", "upstream_delta_m1634.json")
    doc = os.path.join(ROOT, "docs", "upstream_delta_audit_m1634.md")
    if not (os.path.exists(art) and os.path.exists(doc)):
        pytest.skip("report artifacts not generated")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import render_upstream_delta_md as rmd
    expected = rmd.render(ROOT, json.loads(open(art, encoding="utf-8").read()))
    assert open(doc, encoding="utf-8").read() == expected


def test_crosschecks_agree_with_git_within_three_percent():
    """The report's error table must come out exact (regression: HEAD~N range
    baselines silently returned 0 on this merge-heavy history)."""
    art = os.path.join(ROOT, "artifacts", "upstream_delta_m1634.json")
    if not os.path.exists(art):
        pytest.skip("artifact not generated")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import render_upstream_delta_md as rmd
    d = json.loads(open(art, encoding="utf-8").read())
    rows = rmd.crosschecks(ROOT, d)
    assert len(rows) >= 8
    for r in rows:
        assert r["rel_err_pct"] is not None, "MISMATCH on %r" % r["item"]
        assert r["rel_err_pct"] < 3.0, "%r off by %.3f%%" % (r["item"], r["rel_err_pct"])
