"""Política gate for the gate-vacuity census (see ``tools/audit_gate_vacuity.py``).

Why this module exists: a TensorDG acceptance gate that reads a committed
artifact usually ships a convenience escape hatch

    if not ARTIFACT.exists():
        pytest.skip("run benchmarks.foo to generate it")

That hatch is fine while the artifact is committed and fatal when it is not:
the ecosystem merge gate filters files out of agent branches, the artifact
disappears from ``main``, and the gate silently stops asserting anything while
CI stays green.  ``test_dg_advection_bench`` and ``test_shu_osher_case_pool``
were in exactly that state (see ``docs/gate_vacuity/gate_vacuity.json``).

The gates here defend the *census*, not the physics: they (1) keep the
committed census in sync with the tree, (2) forbid a new vacuous gate from
landing unregistered, (3) keep the allowance list self-cleaning, and (4) prove
the classifier itself has teeth on synthetic modules.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_gate_vacuity as gv  # noqa: E402

# tracked census path (``artifacts/`` was filtered out of ``main``; see m16815/m16875)
CENSUS = ROOT / "docs" / "gate_vacuity" / "gate_vacuity.json"
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


def _absolute_strings(obj, where: str = "") -> list[str]:
    """Every string in the payload that is an absolute filesystem path."""
    out: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            out += _absolute_strings(value, f"{where}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            out += _absolute_strings(value, f"{where}[{i}]")
    elif isinstance(obj, str) and obj.startswith("/"):
        out.append(f"{where}={obj}")
    return out


def test_census_carries_no_checkout_specific_path():
    """Regression: schema 1 stored the *generator's* absolute worktree root.

    ``--check`` then compared absolute paths, which can only agree on the machine
    that wrote the file: on ``main`` and in CI the drift guard was red (and
    ``test_committed_census_matches_a_fresh_scan`` failed) while the tree was
    consistent -- a drift guard that cannot pass outside one worktree, i.e. the
    very vacuity this census exists to count.  Every path must be repo-relative.
    """
    data = json.loads(CENSUS.read_text())
    offenders = _absolute_strings(data)
    assert not offenders, (
        "census records absolute paths, so --check can only pass on the machine "
        f"that generated it: {offenders[:5]}")
    assert data.get("schema", 0) >= 2, "census schema 1 embedded an absolute repo_root"


def test_drift_guard_passes_in_a_foreign_checkout(tmp_path):
    """The census must reproduce from a *clean checkout at another path*.

    ``--check`` is the guard that the committed census is not hand-edited; it is
    only worth having if it can pass somewhere other than the author's worktree.
    This builds a faithful stand-in for another checkout -- the files git carries
    that the census reads (``tests/``, the tool, the committed census, and every
    *tracked* artifact it resolves) at a different absolute path -- and requires
    ``--check`` to be green there.
    """
    data = json.loads(CENSUS.read_text())
    tracked = set(gv._tracked_files() or ())
    wanted = [Path("tools/audit_gate_vacuity.py"), Path("docs/gate_vacuity/gate_vacuity.json"),
              Path("docs/GATE-VACUITY-AUDIT.md")]
    wanted += [Path("tests") / p.name for p in (ROOT / "tests").glob("*.py")]
    for module in data["modules"]:
        for record in module["artifacts"] + module["skip_sites"]:
            rel = record.get("path") or record.get("artifact")
            if rel and rel in tracked:  # untracked artifacts are absent from a clean checkout
                wanted.append(Path(rel))
    checkout = tmp_path / "another_checkout"
    for rel in wanted:
        src = ROOT / rel
        if not src.is_file():
            continue
        dst = checkout / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.hardlink_to(src)
        except OSError:  # different filesystem
            dst.write_bytes(src.read_bytes())

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=checkout, check=True, capture_output=True)

    git("init", "-q")
    git("add", "-A")
    assert str(checkout) != str(ROOT)
    run = subprocess.run([sys.executable, "tools/audit_gate_vacuity.py", "--check"],
                         cwd=checkout, capture_output=True, text=True)
    assert run.returncode == 0, (
        "the drift guard is red in a checkout other than the author's:\n"
        f"{run.stdout}{run.stderr}")


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


def test_no_gate_hard_asserts_an_unfetchable_artifact():
    """A hard assert on a git-untracked artifact is RED on every clean checkout.

    Regression of a live defect (m16875): ``tests/test_positivity_scaling_limiter.py``
    hard-asserted ``artifacts/positivity_limiter/metrics.json``, a path the merge
    gate never carries, so three of its gates failed on ``main`` while the author's
    worktree was green (and the old census blessed it as ``HARDENED``).
    """
    data = _census()
    red = set(data["red_on_clean_modules"])
    declared = set(gv.ALLOWED_RED_CLEAN)
    assert not red - declared, (
        "gate(s) hard-assert an artifact git does not carry (red on a clean "
        f"checkout): {sorted(red - declared)}")
    stale = sorted(declared - red)
    assert not stale, f"ALLOWED_RED_CLEAN entries no longer red: {stale}"
    for module, reason in gv.ALLOWED_RED_CLEAN.items():
        assert reason.strip(), f"declaration for {module} needs a reason"


def test_no_skip_gate_pins_a_git_untracked_artifact():
    """A skip-guard on an untracked artifact is vacuous forever, not just today."""
    data = _census()
    assert data["untracked_artifacts"] == [], (
        f"skip gate(s) pinned on untracked artifacts: {data['untracked_artifacts']}")


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

# A genuinely HARDENED gate pins a *git-tracked* artifact: since m16875 the
# classifier separates that from "hard assert on a path git never carries"
# (RED-ON-CLEAN).  The fixture therefore points at a record that really is tracked.
_HARDENED_TARGET = ROOT / "validation" / "advection_1d" / "highorder_summary.json"
_HARDENED_FIXTURE = f'''
from pathlib import Path
ARTIFACT = Path(r"{_HARDENED_TARGET}")

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


_RED_CLEAN_FIXTURE = '''
from pathlib import Path
ARTIFACT = Path(__file__).resolve().parents[1] / "runs" / "never_committed" / "metrics.json"

def test_gate():
    assert ARTIFACT.exists(), "missing committed artifact"
'''


def test_classifier_flags_a_hard_assert_on_an_untrackable_artifact(tmp_path):
    rep = _scan(tmp_path, "test_red.py", _RED_CLEAN_FIXTURE)
    assert rep.verdict == gv.VERDICT_RED_CLEAN
    assert rep.red_artifacts and "missing" in rep.red_artifacts[0]


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
