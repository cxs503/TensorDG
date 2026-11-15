"""Tests for the certificate landing filter (``tools/land_provenance_certificates.py``).

Landing is where "a job succeeded" turns into "the gate may cite it", so each
rejection rule is pinned individually: a certificate for the wrong mesh, the
wrong reference model, a non-reproducing delta, or an orphan key must not reach
the bundle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import land_provenance_certificates as lp  # noqa: E402

KEY = "dg_d2_n32_t5_cfl0.2"
CONFIG = {"task": "dg", "degree": 2, "cells": 32, "cfl": 0.2,
          "final_time": 5.0, "strength": 5.0, "reference": "min_image"}


def make_bundle(tmp_path, reference_model="min_image"):
    results = tmp_path / "case" / "results"
    results.mkdir(parents=True, exist_ok=True)
    record = {"task_key": KEY, "reference_model": None, "config": CONFIG,
              "errors": {}, "max_percent": 0.0111341}
    (results / f"{KEY}.json").write_text(json.dumps(record))
    if reference_model == "orphan":
        (results / f"{KEY}.json").unlink()
    return tmp_path / "case"


def certificate(**overrides):
    cert = {"task_key": KEY, "reference_model": "min_image", "config": dict(CONFIG),
            "stored_max_percent": 0.0111341, "replayed_max_percent": 0.0111341,
            "absolute_delta": {"density": 0.0}, "replay_delta": 1e-12,
            "wall_seconds": 1.0, "steps": 100, "certified_at": "now",
            "tolerance": 1e-9}
    cert.update(overrides)
    return cert


def test_good_certificate_lands(tmp_path):
    case_dir = make_bundle(tmp_path)
    report = lp.land({KEY: certificate()}, case_dir)
    assert [row[0] for row in report["landed"]] == [KEY]
    bundle = json.loads((case_dir / "provenance_certificates.json").read_text())
    assert bundle["reference_model"] == "min_image"
    assert bundle["tolerance"] == lp.ec.PROVENANCE_REPLAY_TOLERANCE
    assert KEY in bundle["certificates"]


@pytest.mark.parametrize("overrides,reason_fragment", [
    ({"config": dict(CONFIG, cells=64)}, "config mismatch on cells"),
    ({"config": dict(CONFIG, final_time=2.0)}, "config mismatch on final_time"),
    ({"reference_model": "free_space"}, "aggregate is"),
    ({"replay_delta": 1e-6}, "replay delta"),
    ({"replay_delta": float("nan")}, "replay delta"),
    ({"replayed_max_percent": 0.02}, "exceeds tolerance"),
    ({"stored_max_percent": None}, "both stored and replayed"),
])
def test_bad_certificates_are_rejected(tmp_path, overrides, reason_fragment):
    case_dir = make_bundle(tmp_path)
    report = lp.land({KEY: certificate(**overrides)}, case_dir)
    assert not report["landed"]
    assert reason_fragment in report["rejected"][0][1]
    assert not (case_dir / "provenance_certificates.json").exists()


def test_orphan_certificate_is_rejected(tmp_path):
    case_dir = make_bundle(tmp_path)
    report = lp.land({"some_other_key": certificate()}, case_dir)
    assert "no such record" in report["rejected"][0][1]


def test_existing_better_certificate_is_kept(tmp_path):
    case_dir = make_bundle(tmp_path)
    lp.land({KEY: certificate(replay_delta=0.0)}, case_dir)
    report = lp.land({KEY: certificate(replay_delta=5e-10)}, case_dir)
    assert not report["landed"]
    assert "at least as good" in report["kept_existing"][0][1]
    bundle = json.loads((case_dir / "provenance_certificates.json").read_text())
    assert bundle["certificates"][KEY]["replay_delta"] == 0.0


def test_dry_run_writes_nothing(tmp_path):
    case_dir = make_bundle(tmp_path)
    report = lp.land({KEY: certificate()}, case_dir, dry_run=True)
    assert report["landed"]
    assert not (case_dir / "provenance_certificates.json").exists()


def test_landed_certificate_satisfies_the_gate(tmp_path):
    """End-to-end: after landing, the provenance audit must grade it ``replayed``."""
    case_dir = make_bundle(tmp_path)
    lp.land({KEY: certificate()}, case_dir)
    records = lp.load_records(case_dir / "results")
    audit = lp.ec.provenance_audit(records, lp.ec.load_provenance_certificates(case_dir))
    assert audit["rows"][KEY]["status"] == "replayed"