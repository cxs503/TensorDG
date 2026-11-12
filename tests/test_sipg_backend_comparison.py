"""Tests for SIPG backend comparison artifact.

Validates that all three backends (dense, sparse, matrix-free) produce
equivalent solutions for the Poisson 2D manufactured-solution problem,
and that the benchmark artifact is self-consistent.

Audit gap closure: m14490 §1.5 identified that sipg_matrixfree.py and
sipg_sparse.py had no benchmark artifacts showing performance comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensordg.poisson_2d import solve_poisson_2d, poisson_error_2d

ARTIFACT = Path(__file__).parent.parent / "validation" / "sipg_backend_comparison" / "backend_comparison.json"


@pytest.fixture(scope="module")
def artifact():
    with open(ARTIFACT) as f:
        return json.load(f)


def test_artifact_exists():
    """The comparison artifact must exist."""
    assert ARTIFACT.exists(), f"Artifact not found: {ARTIFACT}"


def test_artifact_has_all_backends(artifact):
    """All three backends must be represented."""
    backends = {r["backend"] for r in artifact["results"]}
    assert backends == {"dense", "sparse", "matrix-free"}, f"Missing backends: {backends}"


def test_accuracy_equivalence(artifact):
    """All backends must produce the same relative L2 error (within 1e-12)."""
    # Group by (ncells, degree)
    by_case = {}
    for r in artifact["results"]:
        key = (r["ncells"], r["degree"])
        by_case.setdefault(key, []).append(r)

    for key, results in by_case.items():
        errors = [r["relative_l2"] for r in results]
        max_diff = max(errors) - min(errors)
        assert max_diff < 1e-12, (
            f"Case {key}: relative L2 errors differ by {max_diff:.2e} across backends: "
            f"{[(r['backend'], r['relative_l2']) for r in results]}"
        )


def test_matrixfree_cg_converged(artifact):
    """Matrix-free CG must have converged for all recorded cases."""
    for r in artifact["results"]:
        if r["backend"] == "matrix-free" and r.get("cg_info"):
            assert r["cg_info"]["cg_converged"] is True, (
                f"Matrix-free CG did not converge for ncells={r['ncells']}: "
                f"residual={r['cg_info']['cg_relative_residual']:.2e}"
            )


def test_matrixfree_cg_residual_within_tolerance(artifact):
    """Matrix-free CG residual must be within the requested tolerance."""
    for r in artifact["results"]:
        if r["backend"] == "matrix-free" and r.get("cg_info"):
            rtol = 1e-12  # default in solve_poisson_2d
            assert r["cg_info"]["cg_relative_residual"] <= rtol * 1.05, (
                f"Matrix-free CG residual {r['cg_info']['cg_relative_residual']:.2e} "
                f"exceeds tolerance {rtol} for ncells={r['ncells']}"
            )


def test_live_reproduction_p1_8cells():
    """Live reproduction: all three backends agree for P1, 8x8 cells."""
    results = {}
    for backend in ["dense", "sparse", "matrix-free"]:
        x, y, coeff, meta = solve_poisson_2d(
            ncells=8, degree=1, backend=backend,
        )
        err = poisson_error_2d(x, y, coeff)
        results[backend] = err["relative_l2"]

    max_diff = max(results.values()) - min(results.values())
    assert max_diff < 1e-12, (
        f"Live reproduction: backends disagree by {max_diff:.2e}: {results}"
    )
    # Also verify the error is reasonable (P1 should be ~1.5% for 8x8)
    for backend, err in results.items():
        assert err < 0.02, f"{backend} error {err:.4e} exceeds 2%"
