"""The published Shu--Osher evidence bundle must be checkable on a clean checkout.

The case itself (``tensordg.cases.shock_entropy``) and its numbers are already
pinned by ``tests/test_shock_entropy_case_api.py`` and
``tests/test_shu_osher_case_pool.py``.  What is pinned *here* is the delivery
plumbing of the evidence bundle, which had three defects:

1. ``smoke_repro.py`` / ``make_evidence.py`` / ``platform_bench.py`` read the run
   directory only as ``runs/shu_osher_case_pool`` -- a git-ignored path that
   never reaches ``main`` -- so on a clean checkout every documented command
   died with ``FileNotFoundError`` even though the curated copies are tracked;
2. ``artifacts.sha256.json`` listed bare file names for the three figures that
   live in ``docs/figures/``, so a verifier could not resolve them (3/19
   entries unverifiable);
3. nothing re-hashed the bundle at all, so drift was invisible.

The tests below are the fixed contract: path resolution, manifest integrity,
the documented commands running without a fresh study, and the acceptance line.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "validation" / "repro_shu_osher"
sys.path.insert(0, str(BUNDLE_DIR))
import bundle  # noqa: E402  (sibling helper, imported by the three scripts)

CURATED = BUNDLE_DIR / "artifacts"
TOLERANCE_PERCENT = 5.0
#: the curated bundle is what a clean checkout has; the fresh study is optional
MISSING_RUN_DIR = Path("/nonexistent/shu_osher_case_pool")


def _clean_env(run_dir: Path) -> dict:
    env = dict(os.environ)
    env[bundle.ENV_RUN_DIR] = str(run_dir)
    return env


# --------------------------------------------------------------- path resolution
def test_run_dir_falls_back_to_the_tracked_bundle(tmp_path):
    """No fresh study -> the curated, git-tracked copy is used (was: crash)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(bundle.ENV_RUN_DIR, str(MISSING_RUN_DIR))
        directory, source = bundle.resolve_run_dir()
    assert source == "curated", source
    assert directory == CURATED
    assert (directory / "metrics.json").is_file()


def test_env_override_selects_a_fresh_study(tmp_path):
    (tmp_path / "metrics.json").write_text('{"ladder": [], "marker": "fresh"}')
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(bundle.ENV_RUN_DIR, str(tmp_path))
        directory, source = bundle.resolve_run_dir()
        metrics, path, loaded_source = bundle.load_metrics()
    assert source == loaded_source == "env"
    assert metrics["marker"] == "fresh"
    assert path.parent == tmp_path


def test_incomplete_directory_is_not_accepted(tmp_path):
    """A directory without metrics.json must not shadow the curated bundle."""
    (tmp_path / "profiles.npz").write_bytes(b"")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(bundle.ENV_RUN_DIR, str(tmp_path))
        directory, source = bundle.resolve_run_dir()
    assert source == "curated" and directory == CURATED


# ------------------------------------------------------------------- provenance
def test_manifest_resolves_and_rehashes_every_entry():
    report = bundle.verify_manifest()
    assert report["checked"] >= 19, report
    assert report["failed"] == [], report
    assert report["ok"] is True


def test_manifest_keys_are_repo_relative():
    """Bare names were the old bug: ``docs/figures`` entries were unresolvable."""
    manifest = json.loads((BUNDLE_DIR / "artifacts.sha256.json").read_text())
    for key in manifest:
        assert "/" in key, f"manifest key {key!r} is not repo-relative"
        assert (ROOT / key).is_file(), f"manifest key {key!r} does not exist"


def test_manifest_pins_the_tracked_deliverables():
    """The numbers actually shipped must be under hash, not just any files."""
    manifest = json.loads((BUNDLE_DIR / "artifacts.sha256.json").read_text())
    for required in ("validation/repro_shu_osher/artifacts/metrics.json",
                     "validation/repro_shu_osher/artifacts/profiles.npz"):
        assert required in manifest, required
    assert sum(1 for k in manifest if k.startswith("validation/repro_shu_osher/"
                                                   "artifacts/raw/")) >= 10


def test_verify_manifest_detects_tampering(tmp_path):
    """A wrong hash must be reported, not swallowed."""
    target = tmp_path / "f.txt"
    target.write_text("payload")
    manifest = {"f.txt": hashlib.sha256(b"something else").hexdigest()}
    report = bundle.verify_manifest(manifest, root=tmp_path)
    assert report["ok"] is False
    assert report["failed"] == ["f.txt"]
    assert report["entries"][0]["status"] == "mismatch"


# ------------------------------------------------- the documented commands work
@pytest.mark.parametrize("cmd,source_key", [
    (["smoke_repro.py", "--dry-run"], "run_dir"),
    (["platform_bench.py", "--dry-run"], "artifact_source"),
])
def test_documented_dry_runs_work_without_a_fresh_study(cmd, source_key, tmp_path):
    """README commands, run from an unrelated cwd with no ``runs/`` present."""
    script = BUNDLE_DIR / cmd[0]
    proc = subprocess.run([sys.executable, str(script), *cmd[1:]],
                          cwd=tmp_path, env=_clean_env(MISSING_RUN_DIR),
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    assert payload, cmd
    if source_key == "run_dir":
        assert Path(payload[source_key]) == CURATED
    else:
        assert payload[source_key] == "curated"
    assert "runs/" not in json.dumps(payload[source_key])


def test_smoke_record_on_disk_states_its_source():
    record = json.loads((CURATED / "repro_smoke.json").read_text())
    assert record["all_checks_pass"] is True
    assert record["artifact_source"] in {"curated", "fresh", "env"}
    assert record["checks"]["rel_l1_reproduces_tabulated"] is True
    assert record["got"]["rel_l1_percent"] == pytest.approx(
        record["tabulated"]["rel_l1_percent"], abs=record["tolerance_percent"])


# ------------------------------------------------------------ acceptance line
def test_bundled_ladder_meets_the_predeclared_acceptance_line():
    metrics, path, source = bundle.load_metrics()
    assert source in {"curated", "fresh", "env"}
    rows = metrics["ladder"]
    assert len(rows) == 9, f"expected the 9-config ladder, got {len(rows)}"
    for row in rows:
        assert row["rel_l1_percent"] < TOLERANCE_PERCENT, row["configuration"]
        assert row["rho_min"] > 0.0, row["configuration"]
        assert row["passes_5pct"] is (row["rel_l1_percent"] < TOLERANCE_PERCENT)
    best = min(rows, key=lambda r: r["rel_l1_percent"])
    assert best["rel_l1_percent"] < 3.0, best  # tighter than the gate, stated in README
    # the reference itself must be converged well inside the gate it enforces
    assert metrics["reference_self_convergence"]["rel_l1_percent"] < 1.0


def test_evidence_json_points_at_files_that_exist():
    evidence = json.loads((BUNDLE_DIR / "evidence.json").read_text())
    for key in ("impl_files", "entry"):
        for rel in evidence[key]:
            assert (ROOT / rel).is_file(), f"evidence.json {key}: missing {rel}"
    for rel in evidence["run_artifacts"]:
        if rel.startswith("runs/"):
            continue  # regenerable study output, absent on a clean checkout
        assert (ROOT / rel).is_file(), f"evidence.json run_artifacts: missing {rel}"
    gate = evidence["benchmark"]["tolerance_percent"]
    assert gate == TOLERANCE_PERCENT
    for cfg in evidence["benchmark"]["configurations"]:
        assert cfg["got_rel_l1_percent"] < gate, cfg
        assert cfg["passes_5pct"] is True, cfg