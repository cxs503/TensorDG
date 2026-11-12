"""Regression guard for the documented LDG boundary-closure blocker.

``CASE-DG-LDG-POISSON`` is still open: the alternating-flux LDG closure with
Dirichlet data is rank deficient by exactly one, with the null vector supported
on the highest-order modal coefficient of the LAST element's u field.  These
tests pin that evidence down so that (a) the limitation cannot be silently
forgotten and (b) any future fix shows up as a deliberate change here.

Evidence source: ``artifacts/ldg1d_probe.py`` -> ``artifacts/ldg1d_probe.json``.
"""
from __future__ import annotations

import numpy as np
import pytest

from artifacts.ldg1d_probe import assemble


@pytest.mark.parametrize("ncells", [2, 4, 8])
@pytest.mark.parametrize("degree", [1, 2])
def test_alternating_closure_is_rank_deficient_by_one(ncells: int, degree: int):
    """The Dirichlet (data) closure leaves exactly one mode unconstrained."""
    matrix, _ = assemble(ncells, degree, "A", "data")
    ndofs = matrix.shape[0]
    rank = int(np.linalg.matrix_rank(matrix))
    assert ndofs == 2 * ncells * (degree + 1)
    assert ndofs - rank == 1


@pytest.mark.parametrize("degree", [1, 2])
def test_null_vector_lives_on_last_element_top_mode(degree: int):
    """The null vector is the top u mode of the last element and nothing else."""
    ncells = 4
    matrix, _ = assemble(ncells, degree, "A", "data")
    _, singular_values, vt = np.linalg.svd(matrix)
    assert singular_values[-1] < 1e-12
    null = vt[-1].reshape(ncells, 2, degree + 1)
    top = np.abs(null[ncells - 1, 0, :])
    # only the highest mode of the last element's u field is populated
    assert top[-1] == pytest.approx(1.0)
    assert np.allclose(top[:-1], 0.0, atol=1e-12)
    # no support anywhere else in the mesh
    elsewhere = np.delete(null.reshape(ncells, -1), ncells - 1, axis=0)
    assert np.linalg.norm(elsewhere) < 1e-12


def test_ldg_module_is_not_exported():
    """No overstatement: the unvalidated LDG prototype is not a public API."""
    import tensordg

    for name in ("solve_poisson_2d_ldg", "ldg_poisson_convergence"):
        assert not hasattr(tensordg, name)
