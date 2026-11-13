"""Tests for the OSS cross-validation of the round-2 certified reference.

The unit tests need no OSS tree; the end-to-end test runs a *miniature* ladder
(N = 100/200/400, CFL = 0.9) with Clawpack and is skipped when the OSS tree is
absent, so CI stays green without the third-party download.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks import xval_oss_shu_osher as xval
from benchmarks.shu_osher_certified_ref import coarsen, convergence_study, rel_l1

OSS_ROOT = Path(os.environ.get("CLAWPACK_ROOT", "/tmp/oss/clawpack-5.14.0"))
HAS_OSS = (OSS_ROOT / "clawpack" / "__init__.py").exists()


def test_shu_osher_initial_state_matches_case_definition():
    x = np.array([-4.5, -4.0, -2.0, 0.0, 1.0])
    q = xval._shu_osher_state(x)
    assert q.shape == (3, 5)
    # post-shock Mach-3 state left of x = -4
    assert q[0, 0] == pytest.approx(3.857143)
    assert q[1, 0] / q[0, 0] == pytest.approx(2.629369)
    assert q[2, 0] == pytest.approx(10.33333 / 0.4 + 0.5 * 3.857143 * 2.629369 ** 2)
    # pre-shock sine, at rest, unit pressure
    assert q[0, 2] == pytest.approx(1.0 + 0.2 * math.sin(-10.0))
    assert q[1, 2] == pytest.approx(0.0)
    assert q[2, 2] == pytest.approx(1.0 / 0.4)


def test_verify_oss_tree_reports_missing_files_without_raising(tmp_path):
    prov = xval.verify_oss_tree(tmp_path)
    assert prov["files"], "provenance table must enumerate the files it leans on"
    assert all(meta.get("missing") is True for meta in prov["files"].values())


def test_compare_certified_detects_grid_mismatch():
    ours = {"reference_ncells": 800, "rho_certified": np.ones(800), "x": np.zeros(800)}
    oss = {"reference_ncells": 400, "rho_certified": np.ones(400), "x": np.zeros(400)}
    with pytest.raises(ValueError):
        xval.compare_certified(ours, oss)


def test_compare_certified_scores_identical_profiles_as_zero():
    x = np.linspace(-5, 5, 800, endpoint=False)
    rho = 1.0 + 0.1 * np.sin(x)
    study = {"reference_ncells": 800, "rho_certified": rho, "x": x,
             "gci_finest_percent": 0.1, "observed_order_used": 1.2,
             "extrapolation_displacement_percent": 0.2}
    same = dict(study, rho_certified=rho.copy())
    out = xval.compare_certified(study, same)
    assert out["difference_percent"] == pytest.approx(0.0)
    assert out["passes_tolerance"] is True
    assert out["inside_combined_band"] is True


def test_predeclared_gates_are_fixed():
    # pre-registered before the first OSS run; changing them invalidates the study
    assert xval.TOL_XVAL_PERCENT == 1.0
    assert xval.BAND_FACTOR == 3.0
    assert xval.FV_LEVELS == (800, 1600, 3200)


@pytest.mark.skipif(not HAS_OSS, reason=f"clawpack tree not present at {OSS_ROOT}")
def test_oss_leg_is_independent_and_converges():
    """Miniature ladder: the OSS leg must be finite, admissible and converging."""
    x, rho_small, meta = xval.run_oss_level(100, cfl=0.9, t_end=0.2, root=OSS_ROOT)
    assert np.all(np.isfinite(rho_small))
    assert rho_small.min() > 0.0
    assert meta["implementation"].startswith("clawpack")
    assert meta["riemann"].endswith("euler_hllc_1D")
    fine = xval.run_oss_level(200, cfl=0.9, t_end=0.2, root=OSS_ROOT)
    assert fine["rho"].shape == (200,)
    # coarse/fine difference at a crude grid pair must still be a small fraction
    d = rel_l1(coarsen(fine["rho"], 2), rho_small)
    assert 0.0 < d < 0.3, f"coarse/fine rel L1 {d:.4f} outside the sane band"


@pytest.mark.skipif(not HAS_OSS, reason=f"clawpack tree not present at {OSS_ROOT}")
def test_oss_certification_runs_end_to_end(tmp_path):
    levels = xval.build_oss_levels(tmp_path, levels=(100, 200, 400), cfl=0.9, root=OSS_ROOT,
                                   t_end=0.2)
    study = convergence_study(levels)
    assert study["levels"] == [100, 200, 400]
    assert study["reference_ncells"] == 200
    assert study["rho_certified"].shape == (200,)
    assert np.isfinite(study["observed_order_used"])
    assert study["gci_finest_percent"] >= 0.0
    # cache reuse: a second call must not re-run the solver
    again = xval.build_oss_levels(tmp_path, levels=(100, 200, 400), cfl=0.9, root=OSS_ROOT,
                                 t_end=0.2)
    assert all(lv["reused"] for lv in again)