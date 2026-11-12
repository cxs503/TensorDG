"""Regression guard for the documented LDG boundary-closure blocker.

``CASE-DG-LDG-POISSON`` is still open: the alternating-flux LDG closure with
Dirichlet data is rank deficient by exactly one mode, and the null vector is
supported on the ``u`` block of a single element.  These tests pin that
evidence down so that (a) the limitation cannot be silently forgotten and
(b) any future fix shows up as a deliberate change here.

History: this file originally imported ``artifacts.ldg1d_probe`` (m16089), but
that artifact never entered the repository (merge gate filtered it), so the
module failed at *collection* time on ``main`` and the guard was inert
(m16266).  The probe now lives in the repository as
``validation/ldg_closure/probe_ldg1d.py`` and is the single evidence source.

Note on the free mode: which modal coefficient is left free depends on the
orientation of the alternating traces (see
``docs/ldg_closure_probe_2026-11-12.md`` §3.3); with this probe's ``u^- / q^+``
orientation it is the *constant* u mode of the last element, with the opposite
orientation it is the *highest* mode.  What is orientation independent -- and
what these tests therefore assert -- is the deficiency of exactly one and the
confinement of the null vector to one element.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / "validation" / "ldg_closure" / "probe_ldg1d.py"


def _probe():
    spec = importlib.util.spec_from_file_location("ldg_closure_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROBE_MOD = _probe()


@pytest.mark.parametrize("ncells", [2, 4, 8])
@pytest.mark.parametrize("degree", [1, 2])
def test_alternating_closure_is_rank_deficient_by_one(ncells: int, degree: int):
    """The Dirichlet (data) closure leaves exactly one mode unconstrained."""
    matrix, _ = PROBE_MOD.assemble(ncells, degree, "pure")
    ndofs = matrix.shape[0]
    rank = int(np.linalg.matrix_rank(matrix, tol=1e-10))
    assert ndofs == 2 * ncells * (degree + 1)
    assert ndofs - rank == 1


@pytest.mark.parametrize("degree", [1, 2])
def test_null_vector_is_supported_on_a_single_element_u_block(degree: int):
    """The null vector is machine-zero singular and confined to one element."""
    ncells = 4
    matrix, _ = PROBE_MOD.assemble(ncells, degree, "pure")
    _, singular_values, _ = np.linalg.svd(matrix)
    assert singular_values[-1] < 1e-12
    record = PROBE_MOD.null_mode(ncells, degree)
    assert record["sigma_min"] < 1e-12
    assert record["energy_fraction_in_argmax_element_u"] > 0.999


def test_prototype_solver_is_not_exported_as_a_validated_capability():
    """The 2-D prototype must stay out of the public namespace until it converges."""
    import tensordg

    assert not hasattr(tensordg, "solve_poisson_2d_ldg")
    module = (ROOT / "tensordg" / "poisson_2d_ldg.py").read_text()
    assert "STATUS: PROTOTYPE" in module