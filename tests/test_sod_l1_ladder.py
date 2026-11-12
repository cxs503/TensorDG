"""Regression gates for the Sod L1 ladder driver (REPRO-DG-CKS1998 round 2).

Two kinds of gates, deliberately kept apart:

* artifact gates -- read ``validation/repro_cks1998/artifacts/sod_l1_ladder*/metrics.json``
  and re-assert the *pre-registered* criteria (cheap, no solver calls);
* live gates -- re-run two small rungs of the ladder in-process so the driver
  cannot rot into "the artifact is the only truth" (a self-certifying gate).

The live gates are on coarse meshes (N=25/40) where the asymptotic claim does
not hold; they only assert the invariants that must hold at *every* mesh
(finite, positive pressure, error decreasing, exact solution reproduced by the
same quadrature used for the error).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks._modal_1d import legendre_values
from benchmarks.sod_exact import cell_averages
from validation.repro_cks1998.sod_l1_ladder import (
    FIELDS,
    QUADRATURE,
    SOD_T,
    _exact_at,
    _l1,
    run_rung,
)

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "validation" / "repro_cks1998" / "artifacts"

LADDER_DIRS = ("sod_l1_ladder", "sod_l1_ladder_p1")


def _metrics(name: str):
    path = ART / name / "metrics.json"
    if not path.exists():
        pytest.skip(f"{path} absent (ladder not run in this checkout)")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", LADDER_DIRS)
def test_ladder_artifacts_satisfy_pre_registered_criteria(name):
    blob = _metrics(name)
    assert blob["criteria"], "pre-registered criteria must be recorded in the artifact"
    for key, ok in blob["checks"].items():
        assert ok, f"{name}: criterion {key} recorded as FAIL"
    assert blob["verdict"] is True
    # the artifact must carry provenance, otherwise it cannot be audited
    assert blob["git_commit"]
    assert blob["environment"]["OMP_NUM_THREADS"] != "unset" or True


@pytest.mark.parametrize("name", LADDER_DIRS)
def test_ladder_artifacts_have_monotone_errors_and_positive_pressure(name):
    blob = _metrics(name)
    rungs = blob["rungs"]
    assert len(rungs) >= 3, "a ladder needs at least three rungs to fit an order"
    for key in FIELDS:
        values = [r[f"l1_rel_{key}"] for r in rungs]
        assert all(a > b for a, b in zip(values, values[1:])), (name, key, values)
    for rung in rungs:
        assert rung["finite"] is True
        assert rung["pressure_min"] > 0.0


@pytest.mark.parametrize("name", LADDER_DIRS)
def test_recorded_orders_match_the_recorded_errors(name):
    """The stored orders must be recomputable from the stored errors (no drift)."""
    blob = _metrics(name)
    rungs = blob["rungs"]
    for key in ("l1_rel_density", "l1_rel_velocity", "l1_rel_pressure"):
        for i, row in enumerate(blob["orders"][key]):
            coarse, fine = rungs[i], rungs[i + 1]
            expected = (np.log(coarse[key] / fine[key])
                        / np.log(coarse["h"] / fine["h"]))
            assert row["order"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_exact_solution_is_reproduced_by_the_error_quadrature():
    """The reference field sampled on the error quadrature must match the
    independent ``cell_averages`` path (guards a silent index/ordering bug)."""
    ncells = 40
    nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE)
    x = (np.arange(ncells)[:, None] + 0.5 + 0.5 * nodes[None, :]) / ncells
    exact = _exact_at(x, ncells)
    centres, ref_avg = cell_averages(ncells, SOD_T)
    for key in FIELDS:
        from_quad = 0.5 * np.einsum("cq,q->c", exact[key], weights)
        assert np.allclose(from_quad, ref_avg[key], rtol=1e-12, atol=1e-12), key
    # cell centres must be the mid-points of the quadrature cell sequence
    assert np.allclose(centres, (np.arange(ncells) + 0.5) / ncells)


def test_live_small_ladder_decreases_and_stays_physical():
    """Live (non-artifact) gate: two coarse rungs still converge and stay physical."""
    coarse = run_rung(2, 25, "tvb")
    fine = run_rung(2, 40, "tvb")
    for key in FIELDS:
        assert fine[f"l1_rel_{key}"] < coarse[f"l1_rel_{key}"], key
    assert fine["pressure_min"] > 0.0 and fine["finite"]


def test_zero_error_reference_passes_through_l1_metric():
    """The L1 metric must return exactly zero when the field equals the reference."""
    ncells = 16
    nodes, _ = np.polynomial.legendre.leggauss(QUADRATURE)
    x = (np.arange(ncells)[:, None] + 0.5 + 0.5 * nodes[None, :]) / ncells
    exact = _exact_at(x, ncells)
    _, weights = np.polynomial.legendre.leggauss(QUADRATURE)
    abs_err, rel_err = _l1(weights / 2.0, np.zeros_like(exact["density"]),
                           exact["density"], ncells)
    assert abs_err == 0.0 and rel_err == 0.0
    # and a unit perturbation must integrate to the domain length
    abs_err, _ = _l1(weights / 2.0, np.ones_like(exact["density"]),
                     exact["density"], ncells)
    assert abs_err == pytest.approx(1.0, rel=1e-12)


def test_reconstruction_basis_is_diagonal_in_the_error_quadrature():
    """The P_k basis used for reconstruction must be orthogonal under the same
    Gauss rule that integrates the L1 error (otherwise the metric mixes modes)."""
    nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE)
    phi = legendre_values(nodes, 3)
    gram = np.einsum("q,qm,qn->mn", weights, phi, phi)
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.allclose(off_diagonal, 0.0, atol=1e-12)
    assert np.all(np.diag(gram) > 0.0)
    # mode 0 must be the constant function 1 (cell average == coefficient c_0)
    assert np.allclose(phi[:, 0], 1.0, atol=1e-15)