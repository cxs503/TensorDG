"""Artifact-consistency guards for the REPRO-DG-CKS1998 evidence bundle.

The CKS1998 anchor is only worth its verdict if the committed evidence can be
*checked* by someone who clones the repository:

* every sha256 in ``validation/repro_cks1998/evidence.md`` must describe bytes
  that travel with the repository (audit m15345 defect M1: three
  ``artifacts/full/*.png`` rows whose sha256 could never be verified);
* the committed ``benchmark_ref_vs_got.json`` must be the table the current
  builder produces from the committed gate-profile metrics -- not a stale table
  from an older/softer profile (the "口径混用" defect the audit logged twice);
* a row's ``verdict`` must *be* its tolerance rule, recomputed here from
  (ref, got, tolerance) without trusting the builder;
* the figures must be derivable from the recorded numbers, not from a module
  global that happens to hold a different profile (the ``figure_smooth`` crash
  that made the full-profile figures impossible to rebuild from artifacts).

Run: ``OMP_NUM_THREADS=1 python -m pytest tests/test_cks1998_artifact_consistency.py -q``
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "validation" / "repro_cks1998"
ART = BUNDLE / "artifacts"
EVIDENCE = BUNDLE / "evidence.md"


# ---------------------------------------------------------------------------
# 1. the evidence manifest describes bytes that are actually in the repository
# ---------------------------------------------------------------------------
def _git_tracked() -> set[str] | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"],
                             capture_output=True, text=True, check=True).stdout
    except Exception:                                     # pragma: no cover
        return None
    return {p for p in out.split("\0") if p}


def _manifest_rows() -> list[tuple[str, int, str]]:
    if not EVIDENCE.exists():
        pytest.skip("evidence bundle not present")
    txt = EVIDENCE.read_text()
    return [(f, int(n), h) for f, n, h in
            re.findall(r"^\| `([^`]+)` \| (\d+) \| `([0-9a-f]{64})` \|$", txt, re.M)]


def test_every_manifest_row_matches_the_bytes_on_disk():
    rows = _manifest_rows()
    assert rows, "manifest is empty"
    broken = []
    for rel, size, digest in rows:
        path = REPO / rel
        if not path.exists():
            broken.append((rel, "missing"))
            continue
        data = path.read_bytes()
        if len(data) != size:
            broken.append((rel, f"size {len(data)} != {size}"))
        if hashlib.sha256(data).hexdigest() != digest:
            broken.append((rel, "sha256 mismatch"))
    assert not broken, broken


def test_manifest_only_lists_files_that_travel_with_the_repository():
    """A sha256 of an untracked file is unverifiable in a fresh clone."""
    tracked = _git_tracked()
    if tracked is None:                                   # pragma: no cover
        pytest.skip("git unavailable")
    untracked = [rel for rel, _, _ in _manifest_rows() if rel not in tracked]
    assert not untracked, f"manifest claims unverifiable bytes: {untracked}"


def test_generated_but_untracked_artifacts_are_declared_as_such():
    """Anything under artifacts/ that is not in git must not masquerade as evidence."""
    text = EVIDENCE.read_text()
    section = text.split("Generated locally, **not** committed", 1)
    declared = set(re.findall(r"^- `([^`]+)`", section[1], re.M)) if len(section) > 1 else set()
    tracked = _git_tracked() or set()
    generated = {str(p.relative_to(REPO)) for p in ART.rglob("*")
                 if p.is_file() and str(p.relative_to(REPO)) not in tracked}
    # every locally generated artifact is either declared in the bundle or is a
    # sweep output that the bundle does not claim (sod_gate_margin is its own unit)
    undeclared = sorted(g for g in generated
                        if g not in declared and "/sod_gate_margin/" not in g)
    assert not undeclared, undeclared


# ---------------------------------------------------------------------------
# 2. the committed ref-vs-got table is not stale
# ---------------------------------------------------------------------------
def test_committed_ref_vs_got_table_is_rebuilt_from_the_gate_profile():
    from benchmarks.cks1998_benchmark import build_ref_vs_got

    committed = json.loads((BUNDLE / "benchmark_ref_vs_got.json").read_text())
    metrics = json.loads((ART / "full" / "metrics.json").read_text())
    fresh = build_ref_vs_got(metrics, "full")
    assert committed["profile"] == "full", (
        "the anchor gate is declared on the full profile; a table built from the "
        f"soft profile ({committed['profile']}) is a stale/soft claim")
    assert committed["counts"] == fresh["counts"]
    assert committed["gate_verdict"] == fresh["gate_verdict"]
    assert len(committed["rows"]) == len(fresh["rows"])
    for got, ref in zip(committed["rows"], fresh["rows"]):
        assert got == ref, (got["item"], got["quantity"])


# ---------------------------------------------------------------------------
# 3. verdict == tolerance rule (independent recomputation, no trust in builder)
# ---------------------------------------------------------------------------
_TOL_REL = re.compile(r"<\s*([0-9.]+)\s*%")
_TOL_OVER = re.compile(r"<=\s*([0-9.]+)\s*%\s*overshoot")


def test_scored_rows_are_exactly_their_tolerance_rule():
    table = json.loads((BUNDLE / "benchmark_ref_vs_got.json").read_text())
    scored = 0
    for row in table["rows"]:
        tol = row["tolerance"]
        ref, got = row["ref"], row["got"]
        if ref is None or got is None:
            continue
        if _TOL_OVER.search(tol):          # E3: got is the pointwise maximum
            limit = float(_TOL_OVER.search(tol).group(1)) / 100.0
            over = (row["got"] - 1.0)   # ref is 1.0: the initial maximum
            expect = "PASS" if over <= limit else "FAIL"
        elif _TOL_REL.search(tol) and row["ref_source"] == "analytic" and ref != 0:
            limit = float(_TOL_REL.search(tol).group(1)) / 100.0
            expect = "PASS" if abs(got - ref) <= limit * abs(ref) else "FAIL"
        else:
            continue                    # paper-theorem rows use a >= order rule
        assert row["verdict"] == expect, (row["item"], row["quantity"], row["verdict"], expect)
        scored += 1
    assert scored >= 20, scored         # the invariant must cover a real subset


# ---------------------------------------------------------------------------
# 4. figures are functions of the payload, not of a module-global profile
# ---------------------------------------------------------------------------
def test_smooth_figure_rebuilds_from_committed_full_metrics(tmp_path):
    from validation.repro_cks1998.repro_cks1998 import figure_smooth

    metrics = json.loads((ART / "full" / "metrics.json").read_text())
    out = tmp_path / "a1.png"
    figure_smooth(metrics["sections"]["A1_smooth"], out)   # crashed before the fix
    assert out.stat().st_size > 1000


def test_sod_figure_reads_its_case_from_the_payload(monkeypatch, tmp_path):
    import torch

    from validation.repro_cks1998 import repro_cks1998 as R

    seen = {}

    def fake_solve(left, right, *, ncells, degree, **kw):
        seen.update(ncells=ncells, degree=degree)
        x = torch.linspace(0.0, 1.0, ncells, dtype=torch.float64)
        return x, torch.zeros(ncells, degree + 1, 3, dtype=torch.float64), {}

    monkeypatch.setattr(R, "solve_euler_dg_1d", fake_solve)
    payload = {"cases": {"p1_n40": {"degree": 1, "ncells": 40, "gate_case": True},
                         "p1_n20": {"degree": 1, "ncells": 20, "gate_case": False}}}
    out = tmp_path / "sod.png"
    R.figure_sod(payload, out)
    assert (seen["ncells"], seen["degree"]) == (40, 1), seen
    assert out.stat().st_size > 1000


def test_tvd_figure_reads_degrees_and_cells_from_the_payload(monkeypatch, tmp_path):
    import torch

    from validation.repro_cks1998 import repro_cks1998 as R

    calls = []

    def fake_solve(*, ncells, degree, **kw):
        calls.append((ncells, degree))
        x = torch.linspace(0.0, 1.0, ncells, dtype=torch.float64)
        return x, torch.zeros(ncells, degree + 1, dtype=torch.float64), {}

    monkeypatch.setattr(R, "solve_advection", fake_solve)
    payload = {"cases": {"p3_tvb": {"ncells": 12, "limiter": "tvb"},
                         "p3_strict_all": {"ncells": 12, "limiter": "strict_all"}}}
    out = tmp_path / "tvd.png"
    R.figure_tvd(payload, out)
    assert {nc for nc, _ in calls} == {12}, calls
    assert {d for _, d in calls} == {3}, calls
    assert out.stat().st_size > 1000