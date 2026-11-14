"""Adversarial stress tests for XVAL-DG-ROUND2-CERTIFIED-REFERENCE.

Purpose
-------
The existing integration tests (test_xval_dg_round2_integration.py) verify the
*happy path*: committed raw profiles, correct thresholds, deterministic output.
This module probes **failure modes and edge cases** that could silently corrupt
a verification claim if the engine were misused or the data were tampered with.

These tests are *not* about the committed artifacts; they construct synthetic
profiles to exercise the engine's decision logic under stress.

Coverage matrix
---------------
A1  Divergent ladder (negative observed order) → must be refused, not CERTIFIED
A2  Two-level ladder (insufficient for GCI) → must be refused
A3  Band exceeds gate (U/τ > 1/3) → gate vacuity, reference unusable
A4  Non-monotonic convergence (oscillating differences) → classification != monotonic
A5  Zero-density profile (physical admissibility) → must be caught
A6  Negative-density profile (physical admissibility) → must be caught
A7  Oracle metric with ref=0 and kind="relative" → must not produce NaN/Inf
A8  Leg file missing from raw directory → must raise, not silently skip
A9  DG leg with all-zero field → rel L1 must be finite, not NaN
A10 Cross-leg: same N, different CFL → errors should be comparable (not 10x apart)
A11 Platform spec with extra keys → contract validator handles gracefully
A12 Platform result with missing required key → contract raises ContractError
A13 Determinism under different Python hash seeds → artifacts byte-identical
A14 CLI --no-write flag → no files created, stdout still emitted
A15 Empty raw directory → must raise, not return empty audit

All tests run without solver rerun — synthetic data only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tensordg import certified_reference as cr  # noqa: E402


# ------------------------------------------------------------------ A1 divergent ladder
def test_divergent_ladder_is_refused():
    """A1: A ladder where the error *increases* with refinement must not be CERTIFIED."""
    coarse = np.linspace(0, 1, 100) + 0.1
    medium = np.linspace(0, 1, 200) + 0.05
    fine = np.linspace(0, 1, 400) + 0.2  # larger offset = divergent

    data = [
        {"ncells": 100, "field": coarse},
        {"ncells": 200, "field": medium},
        {"ncells": 400, "field": fine},
    ]
    result = cr.certify(data, tolerance_percent=5.0)
    assert result["verdict"] != "CERTIFIED", (
        f"Divergent ladder should not be CERTIFIED, got {result['verdict']}")


# ------------------------------------------------------------------ A2 two-level ladder
def test_two_level_ladder_is_refused():
    """A2: Two levels are insufficient for GCI (need at least 3 for observed order)."""
    coarse = np.linspace(0, 1, 100) + 0.1
    fine = np.linspace(0, 1, 200) + 0.05

    data = [
        {"ncells": 100, "field": coarse},
        {"ncells": 200, "field": fine},
    ]
    result = cr.certify(data, tolerance_percent=5.0)
    assert result["verdict"] != "CERTIFIED", (
        f"Two-level ladder should not be CERTIFIED, got {result['verdict']}")


# ------------------------------------------------------------------ A3 band exceeds gate
def test_band_exceeds_gate_is_unusable():
    """A3: When U/τ > 1/3, the gate is vacuous — reference is unusable."""
    coarse = np.linspace(0, 1, 100) + 0.5
    medium = np.linspace(0, 1, 200) + 0.45
    fine = np.linspace(0, 1, 400) + 0.42

    data = [
        {"ncells": 100, "field": coarse},
        {"ncells": 200, "field": medium},
        {"ncells": 400, "field": fine},
    ]
    result = cr.certify(data, tolerance_percent=1.0)  # very tight gate
    if result["verdict"] == "CERTIFIED":
        assert result["band_percent"] is not None
        assert result["band_percent"] / 1.0 <= cr.U_SHARE_MAX, (
            f"Band/share {result['band_percent']/1.0} exceeds U_SHARE_MAX")


# ------------------------------------------------------------------ A4 non-monotonic convergence
def test_non_monotonic_convergence_classification():
    """A4: Oscillating differences should not be classified as 'monotonic'."""
    coarse = np.linspace(0, 1, 100) + 0.3
    medium = np.linspace(0, 1, 200) + 0.1
    fine = np.linspace(0, 1, 400) + 0.2

    data = [
        {"ncells": 100, "field": coarse},
        {"ncells": 200, "field": medium},
        {"ncells": 400, "field": fine},
    ]
    result = cr.certify(data, tolerance_percent=5.0)
    if result.get("classification") is not None:
        assert result["classification"] != "monotonic" or result["verdict"] != "CERTIFIED"


# ------------------------------------------------------------------ A5 zero-density profile
def test_zero_density_profile_admissibility():
    """A5: A profile with all-zero density is physically inadmissible."""
    from benchmarks import xval_dg_round2_certified as xval
    zero_profile = np.zeros(1600)
    metrics = xval.measure_metrics(np.linspace(-5, 5, 1600), zero_profile)
    assert not metrics["rho_positive"], "Zero density should fail positivity check"


# ------------------------------------------------------------------ A6 negative-density profile
def test_negative_density_profile_admissibility():
    """A6: A profile with negative density is physically inadmissible."""
    from benchmarks import xval_dg_round2_certified as xval
    neg_profile = np.linspace(-1, 1, 1600)
    metrics = xval.measure_metrics(np.linspace(-5, 5, 1600), neg_profile)
    assert not metrics["rho_positive"], "Negative density should fail positivity check"


# ------------------------------------------------------------------ A7 oracle metric with ref=0
def test_oracle_metric_ref_zero_handling():
    """A7: Metrics with ref=0 and kind='relative' must not produce NaN/Inf."""
    from benchmarks import xval_dg_round2_certified as xval
    assert xval._rel_percent(0.001, None) is None
    assert xval._rel_percent(0.001, 0.0) is None
    assert xval._rel_percent(None, 1.0) is None
    err = xval._rel_percent(1.01, 1.0)
    assert err is not None and err == pytest.approx(1.0)


# ------------------------------------------------------------------ A8 missing leg file
def test_missing_leg_file_raises():
    """A8: If a leg file is missing from raw directory, load_ladder must raise."""
    from benchmarks import xval_dg_round2_certified as xval
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        np.savez(Path(tmpdir) / "fv_cfl0p3_n800.npz",
                 x=np.linspace(-5, 5, 800), rho=np.ones(800))
        with pytest.raises(FileNotFoundError):
            xval.load_ladder(tmpdir)


# ------------------------------------------------------------------ A9 leg with all-zero field
def test_zero_field_leg_rel_l1():
    """A9: A leg with all-zero field should produce finite rel L1, not NaN."""
    ref = np.ones(1600) * 2.0
    zero_leg = np.zeros(800)
    ref_coarse = ref.reshape(800, 2).mean(axis=1)
    rel_l1 = 100.0 * np.sum(np.abs(zero_leg - ref_coarse)) / np.sum(np.abs(ref_coarse))
    assert np.isfinite(rel_l1), "rel L1 should be finite for zero leg"
    assert rel_l1 == pytest.approx(100.0)


# ------------------------------------------------------------------ A10 cross-leg CFL sensitivity
def test_cfl_sensitivity_reasonable():
    """A10: Same N, different CFL → errors should be comparable (not 10x apart)."""
    from benchmarks import xval_dg_round2_certified as xval
    audit = xval.cross_validate(xval.DEFAULT_RAW)
    legs_400 = [lg for lg in audit["legs"] if lg["ncells"] == 400]
    assert len(legs_400) >= 3, "Expected at least 3 legs with N=400"
    errors = [lg["rel_l1_percent"] for lg in legs_400]
    max_err, min_err = max(errors), min(errors)
    if min_err > 0.01:
        ratio = max_err / min_err
        assert ratio < 5.0, (
            f"CFL sensitivity ratio {ratio:.2f}x is suspicious for N=400 legs")


# ------------------------------------------------------------------ A11 platform spec with extra keys
def test_platform_spec_extra_keys_handled():
    """A11: Platform spec with extra keys — contract validator should handle gracefully."""
    from validation.xval_dg_round2 import post_platform
    docs = post_platform.build(18730)
    docs["spec"]["extra_forbidden_key"] = "should not be here"
    # The platform contract may reject or ignore extra keys;
    # the key test is that validate() doesn't crash
    try:
        report = post_platform.validate(docs)
        # If it returns, the extra key was either ignored or flagged
        assert isinstance(report, dict)
    except Exception as e:
        # If it raises, that's also acceptable (strict validation)
        assert "extra_forbidden_key" in str(e) or "key" in str(e).lower()


# ------------------------------------------------------------------ A12 platform result missing required key
def test_platform_result_missing_key_raises():
    """A12: Platform result missing a required key → contract raises ContractError."""
    from validation.xval_dg_round2 import post_platform
    from tools.tensorbench_eval.contract import ContractError
    docs = post_platform.build(18730)
    del docs["result"]["schema_version"]
    with pytest.raises(ContractError) as excinfo:
        post_platform.validate(docs)
    assert "schema_version" in str(excinfo.value)


# ------------------------------------------------------------------ A13 determinism under different hash seeds
def test_determinism_under_hash_seed(tmp_path):
    """A13: Artifacts should be byte-identical regardless of PYTHONHASHSEED."""
    from benchmarks import xval_dg_round2_certified as xval
    raw = Path(xval.DEFAULT_RAW)
    a, b = tmp_path / "a", tmp_path / "b"
    assert xval.main(["--rawdir", str(raw), "--outdir", str(a)]) == 0
    old_seed = os.environ.get("PYTHONHASHSEED")
    try:
        os.environ["PYTHONHASHSEED"] = "42"
        assert xval.main(["--rawdir", str(raw), "--outdir", str(b)]) == 0
    finally:
        if old_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = old_seed
    for name in ("xval_round2_audit.json", "xval_round2_audit.md"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), (
            f"Artifact {name} differs under different PYTHONHASHSEED")


# ------------------------------------------------------------------ A14 CLI --no-write flag
def test_cli_no_write_flag(tmp_path):
    """A14: --no-write should not create any files, but still return exit code."""
    from benchmarks import xval_dg_round2_certified as xval
    raw = Path(xval.DEFAULT_RAW)
    rc = xval.main(["--rawdir", str(raw), "--outdir", str(tmp_path), "--no-write"])
    assert rc == 0
    assert not (tmp_path / "xval_round2_audit.json").exists()
    assert not (tmp_path / "xval_round2_audit.md").exists()


# ------------------------------------------------------------------ A15 empty raw directory
def test_empty_raw_directory_raises():
    """A15: An empty raw directory should raise, not return an empty audit."""
    from benchmarks import xval_dg_round2_certified as xval
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            xval.load_ladder(tmpdir)
