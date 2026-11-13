"""Unit tests for ``tools/patrol_upstream_diff.py`` (m17042 patrol tooling).

The tool answers two questions about the landing history and the branch zoo::

1. what did each landing actually add to main (first-parent diff, not the
   combined diff that makes clean merges look empty)? and
2. which files live only on a local ``agent/*`` branch and never reached
   upstream?

Both are exercised on a **synthetic repository** built in ``tmp_path`` so the
test is independent of this checkout's ever-changing history: a base commit,
an agent branch that adds a figure + a docs page that references it, a squash
landing that keeps only the docs page (the "merge gate filtered the binary"
failure mode), and a second landing that is *content-empty*.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import patrol_upstream_diff as patrol  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _commit(repo: Path, message: str, **files: str) -> str:
    for rel, content in files.items():
        path = repo / rel.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message, "--allow-empty")
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture()
def synthetic(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "patrol@test.invalid")
    _git(repo, "config", "user.name", "patrol test")
    _commit(repo, "base", **{"README.md": "base\n", "tensordg__solver.py": "x = 1\n"})

    # a mission branch: solver change + a binary figure + a doc that embeds it
    _git(repo, "checkout", "-q", "-b", "agent/m90001")
    _commit(
        repo,
        "m90001: add figure",
        **{
            "docs__CASE-figure.md": "![fig](validation/case/figures/err.png)\n",
            "validation__case__figures__err.png": "PNGDATA\n",
            "validation__case__metrics.json": '{"err": 0.01}\n',
        },
    )
    # squash landing that keeps only the docs page (gate filtered the PNG/json)
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "Merge #1 (m90001) via merge-master", **{"docs__CASE-figure.md": "![fig](validation/case/figures/err.png)\n"})
    # a content-empty landing: same tree, different commit
    _commit(repo, "Merge #2 (m90002) via push-merge")
    # make the agent branch a *local-only* ref that is not reachable from main,
    # exactly like the pruned post-merge agent refs in the real checkout
    _git(repo, "branch", "-f", "agent/m90001-unmerged", "agent/m90001")
    _git(repo, "branch", "-D", "agent/m90001")
    return repo


def test_parse_subject_and_channels():
    parsed = patrol.parse_subject("Merge #375 (m16938) via merge-master")
    assert parsed == {"is_landing": True, "pr": 375, "mission": "m16938", "channel": "merge-master"}
    assert patrol.parse_subject("m16938: matured mission deliverables")["is_landing"] is False


def test_theme_buckets():
    assert patrol.theme_of("tensordg/flux.py") == "core-solver"
    assert patrol.theme_of("validation/case/metrics.json") == "validation-artifacts"
    assert patrol.theme_of("scratch/probe.py") == "scratch"
    assert patrol.theme_of("loose_tool.py") == "loose-scripts"
    assert patrol.theme_of("docs/x.md") == "docs"


def test_risk_codes():
    empty = patrol.commit_risks("Merge #2 (m90002) via push-merge",
                                patrol.parse_subject("Merge #2 (m90002) via push-merge"), [])
    assert "R3-CONTENT-EMPTY-LANDING" in empty
    assert "R2-MERGE-CHANNEL-PUSH_MERGE" in empty

    docs_only = patrol.commit_risks("Merge #1 (m90001) via merge-master",
                                    patrol.parse_subject("Merge #1 (m90001) via merge-master"),
                                    [(3, 0, "docs/CASE.md")])
    assert "R5-NO-CORE-CHANGE" in docs_only

    fig = patrol.commit_risks("landing", patrol.parse_subject("landing"),
                              [(1, 0, "validation/case/figures/err.png")])
    assert "R8-BINARY-FIGURE-ASSET" in fig

    filtered = patrol.commit_risks("Merge #9 (m9) via merge-master（3 个文件被门禁滤除）",
                                  patrol.parse_subject("Merge #9 (m9) via merge-master"), [])
    assert "R1-GATE-FILTER-DECLARED" in filtered


def test_empty_landing_detected_against_first_parent(synthetic: Path):
    """A content-empty landing must be separable from "no merge happened"."""
    inv = patrol.commit_inventory(synthetic, "main", 3)
    empty = [c for c in inv if "R3-CONTENT-EMPTY-LANDING" in c["risks"]]
    assert len(empty) == 1, [c["subject"] for c in inv]
    assert empty[0]["files"] == 0
    # the docs-only landing is NOT empty: first-parent diff sees its file
    docs_landing = [c for c in inv if c["mission"] == "m90001"][0]
    assert docs_landing["files"] == 1 and docs_landing["themes"] == {"docs": 1}


def test_branch_audit_finds_file_lost_to_the_gate(synthetic: Path):
    audit = patrol.branch_audit(synthetic, "main")
    assert audit["branches_scanned"] == 1
    row = audit["rows"][0]
    assert row["state"] == "LOST_FILES"
    assert "validation/case/figures/err.png" in row["absent_paths"]
    assert "validation/case/metrics.json" in row["absent_paths"]


def test_referenced_absent_flags_the_broken_doc_link(synthetic: Path):
    audit = patrol.branch_audit(synthetic, "main")
    absent = sorted({p for row in audit["rows"] for p in row["absent_paths"]})
    ref = patrol.referenced_absent(synthetic, "main", absent)
    assert "validation/case/figures/err.png" in ref["referenced"]
    assert ref["detail"]["validation/case/figures/err.png"]["categories"] == ["docs"]


def test_diff_set_reports_empty_upstream_delta(synthetic: Path):
    up, how = patrol.resolve_upstream(synthetic, "main")
    assert up == "main" and how == "explicit"
    d = patrol.diff_set(synthetic, _git(synthetic, "rev-parse", "main").strip(), "main")
    assert d["verdict"] == "EMPTY" and d["ahead"] == 0 and d["behind"] == 0


def test_report_written_and_check_flags_debt(synthetic: Path, tmp_path: Path):
    out = tmp_path / "out"
    code = patrol.main(["--repo", str(synthetic), "--upstream", "main", "--limit", "3",
                        "--write", "--out", str(out), "--check"])
    assert code == 1, "a content-empty landing and a lost-file branch must fail --check"
    payload = json.loads((out / "patrol.json").read_text())
    assert payload["summary"]["content_empty_landings"] == 1
    assert payload["branches"]["states"].get("LOST_FILES") == 1
    md = (out / "patrol.md").read_text()
    assert "Branch audit" in md and "Absent-on-upstream paths" in md

def test_branch_audit_annotates_stale_refs(synthetic: Path):
    """A local ref absent from the remote `ls-remote` listing is flagged stale."""
    stale = patrol.branch_audit(synthetic, "main", remote_names={"agent/something-else"})
    assert stale["remote_names_supplied"] is True
    assert stale["stale_local_refs"] == 1
    assert stale["stale_with_lost_files"] == 1
    assert stale["rows"][0]["upstream_ref_exists"] is False

    alive = patrol.branch_audit(synthetic, "main", remote_names={"agent/m90001-unmerged"})
    assert alive["stale_local_refs"] == 0
    assert alive["rows"][0]["upstream_ref_exists"] is True

    unannotated = patrol.branch_audit(synthetic, "main")
    assert unannotated["remote_names_supplied"] is False
    assert unannotated["rows"][0]["upstream_ref_exists"] is None
