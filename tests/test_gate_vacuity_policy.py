"""Política gate for the gate-vacuity census (see ``tools/audit_gate_vacuity.py``).

Why this module exists: a TensorDG acceptance gate that reads a committed
artifact usually ships a convenience escape hatch

    if not ARTIFACT.exists():
        pytest.skip("run benchmarks.foo to generate it")

That hatch is fine while the artifact is committed and fatal when it is not:
the ecosystem merge gate filters files out of agent branches, the artifact
disappears from ``main``, and the gate silently stops asserting anything while
CI stays green.  ``test_dg_advection_bench`` and ``test_shu_osher_case_pool``
were in exactly that state (see ``artifacts/gate_vacuity/gate_vacuity.json``).

The gates here defend the *census*, not the physics: they (1) keep the
committed census in sync with the tree, (2) forbid a new vacuous gate from
landing unregistered, (3) keep the allowance list self-cleaning, and (4) prove
the classifier itself has teeth on synthetic modules.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_gate_vacuity as gv  # noqa: E402

CENSUS = ROOT / "artifacts" / "gate_vacuity" / "gate_vacuity.json"
REPORT = ROOT / "docs" / "GATE-VACUITY-AUDIT.md"


def _census() -> dict:
    """Committed census, or a fresh in-memory scan if the file did not survive.

    The merge gate filters files out of agent branches, so ``artifacts/``
    content cannot be relied on to reach ``main`` -- and a census that
    disappears turns this module red without telling anyone anything.  The
    scan behind the census is a 0.15 s static pass, so the honest repair is to
    redo it in memory and keep asserting on its content.
    """
    if CENSUS.exists():
        return json.loads(CENSUS.read_text())
    return gv.census()


# ---------------------------------------------------------------------------
# 1. the committed census is reproducible from the tree (no hand edits)
# ---------------------------------------------------------------------------
def test_committed_census_matches_a_fresh_scan():
    committed = _census()
    fresh = gv.census()
    assert committed["modules"] == fresh["modules"], (
        "gate vacuity census drifted; re-run "
        "`python tools/audit_gate_vacuity.py --write`")
    assert committed["verdicts"] == fresh["verdicts"]
    assert committed["n_test_modules"] == len(fresh["modules"])


def test_report_is_rendered_from_the_census():
    data = _census()
    text = REPORT.read_text()
    v = data["verdicts"]
    for verdict in gv.VERDICTS:
        assert f"{verdict} **{v[verdict]}**" in text, verdict


# ---------------------------------------------------------------------------
# 2. no new vacuous gate without an explicit, reasoned registration
# ---------------------------------------------------------------------------
def test_no_unregistered_vacuous_gate():
    data = _census()
    vacuous = set(data["vacuous_modules"])
    registered = set(gv.ALLOWED_VACUOUS)
    unknown = sorted(vacuous - registered)
    assert not unknown, (
        "new gate(s) assert nothing because their artifact is missing: "
        f"{unknown}.  Either commit the artifact (and keep the hard assert) "
        "or register the module in tools/audit_gate_vacuity.py::ALLOWED_VACUOUS "
        "with a reason.")


def test_allowance_list_is_self_cleaning():
    data = _census()
    vacuous = set(data["vacuous_modules"])
    stale = sorted(set(gv.ALLOWED_VACUOUS) - vacuous)
    assert not stale, f"ALLOWED_VACUOUS entries no longer vacuous: {stale}"
    for module, reason in gv.ALLOWED_VACUOUS.items():
        assert reason.strip(), f"allowance for {module} needs a reason"


# ---------------------------------------------------------------------------
# 3. the classifier has teeth (fault injection on synthetic modules)
# ---------------------------------------------------------------------------
_VACUOUS_FIXTURE = '''
import pytest
from pathlib import Path
ARTIFACT = Path(__file__).resolve().parents[1] / "does_not_exist_xyz.json"

def test_gate():
    if not ARTIFACT.exists():
        pytest.skip("run benchmarks.foo to generate it")
    assert True
'''

_LATENT_FIXTURE = '''
import pytest
from pathlib import Path
ARTIFACT = Path(__file__).resolve().parents[1] / "does_not_exist_xyz.json"

def test_gate():
    assert True
'''

_HARDENED_FIXTURE = '''
from pathlib import Path
ARTIFACT = Path(__file__).resolve().parents[1] / "does_not_exist_xyz.json"

def test_gate():
    assert ARTIFACT.exists(), "run benchmarks.foo"
'''

_ENV_FIXTURE = '''
import pytest
from pathlib import Path

def test_gate():
    pytest.skip("no origin/main in this checkout")
'''

_CWD_FIXTURE = '''
import pytest
from pathlib import Path
LAW = Path("validation/somewhere/law.json")

def test_gate():
    if not LAW.exists():
        pytest.skip("run scratch/diag.py")
'''


def _scan(tmp_path, name, source):
    f = tmp_path / name
    f.write_text(source)
    return gv.scan_module(f)


def test_classifier_flags_a_missing_artifact_skip(tmp_path):
    rep = _scan(tmp_path, "test_vac.py", _VACUOUS_FIXTURE)
    assert rep.verdict == gv.VERDICT_VACUOUS
    assert rep.vacuous_artifacts and "does_not_exist_xyz.json" in rep.vacuous_artifacts[0]


def test_classifier_separates_latent_from_vacuous(tmp_path):
    rep = _scan(tmp_path, "test_lat.py", _LATENT_FIXTURE)
    assert rep.verdict == gv.VERDICT_PLAIN   # no skip at all, nothing to vacuous-out


def test_classifier_recognises_a_hard_assert(tmp_path):
    rep = _scan(tmp_path, "test_hard.py", _HARDENED_FIXTURE)
    assert rep.verdict == gv.VERDICT_HARDENED
    assert any(".exists()" in a for a in rep.hard_asserts)


def test_environment_skips_are_not_counted_as_vacuity(tmp_path):
    rep = _scan(tmp_path, "test_env.py", _ENV_FIXTURE)
    assert rep.verdict == gv.VERDICT_PLAIN
    assert [s.kind for s in rep.skip_sites] == [gv.SKIP_ENV]


def test_cwd_dependent_artifact_paths_are_flagged(tmp_path):
    rep = _scan(tmp_path, "test_cwd.py", _CWD_FIXTURE)
    assert rep.cwd_dependent == ["LAW"]


# ---------------------------------------------------------------------------
# 4. the audit covers the whole test tree, not a hand-picked subset
# ---------------------------------------------------------------------------
def test_census_covers_every_test_module():
    data = _census()
    on_disk = sorted(str(p.relative_to(ROOT))
                     for p in (ROOT / "tests").glob("test_*.py"))
    assert [m["file"] for m in data["modules"]] == on_disk
    assert data["n_test_modules"] == len(on_disk)
    assert sum(data["verdicts"].values()) == len(on_disk)

# ---------------------------------------------------------------------------
# 4. the gate must survive its own artifact being filtered out of the branch
# ---------------------------------------------------------------------------
def test_census_survives_a_filtered_artifact(monkeypatch, tmp_path):
    """Simulate the merge gate dropping ``artifacts/gate_vacuity/*``.

    Regression of a live defect: the census JSON was filtered out of ``main``,
    so five of the ten gates in this module failed for *absence of the file*
    rather than for any property of the test suite.  With the fallback in
    ``_census()`` the module keeps adjudicating the tree.
    """
    monkeypatch.setattr(sys.modules[__name__], "CENSUS", tmp_path / "missing.json")
    assert not (tmp_path / "missing.json").exists()
    filtered = _census()  # must not raise
    fresh = gv.census()
    assert filtered["modules"] == fresh["modules"]
    assert filtered["verdicts"] == fresh["verdicts"]
    assert filtered["vacuous_modules"] == fresh["vacuous_modules"]
