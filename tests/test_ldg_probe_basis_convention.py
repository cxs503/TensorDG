"""Guards for the LDG evidence reassessment (m16388).

Pins three facts so they cannot silently drift:

1. the two modal derivative-matrix conventions are *not* interchangeable
   (``D_coef`` is strictly upper, ``D_test`` strictly lower, and their sum is the
   endpoint boundary term);
2. the m16266 assembly (``validation/ldg_closure/probe_ldg1d.py`` reproduction in
   ``probe_basis_transpose.assemble``) is **inconsistent**: it leaves an O(1)
   residual on a solution its space represents exactly, under *both* matrices,
   so its verdict on Dirichlet closures cannot be attributed to the closures;
3. a consistent assembly of the same mixed form exists (``assemble_oriented``,
   machine-zero residual), but the Dirichlet closure question stays *open*:
   ``pure`` is rank deficient by one and the penalised closure, while full rank,
   does not reach order ``k+1``.

Test 5 is deliberately a *negative* guard: if someone later claims an optimal
LDG closure, this test starts failing and the claim has to be re-derived.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / "validation" / "ldg_closure" / "probe_basis_transpose.py"


def _probe():
    spec = importlib.util.spec_from_file_location("basis_transpose_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _probe()


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_derivative_conventions_are_not_interchangeable(degree: int):
    _, _, _, _, D_coef, D_test, _, _ = P.basis(degree)
    boundary = np.array([[1.0 - (-1.0) ** (i + j) for j in range(degree + 1)]
                         for i in range(degree + 1)])
    assert np.allclose(D_coef + D_test, boundary, atol=1e-12)
    assert np.allclose(np.tril(D_coef, -1), 0.0)      # strictly upper
    assert np.allclose(np.triu(D_test, 1), 0.0)       # strictly lower
    assert not np.allclose(D_coef, D_test)


@pytest.mark.parametrize("matrix", ["coef", "test"])
@pytest.mark.parametrize("degree", [2, 3])
def test_m16266_assembly_is_inconsistent_for_an_exact_solution(matrix: str, degree: int):
    """u = x^2, q = 2x is *in* the k>=2 space, yet the residual is O(1)."""
    residual = P.polynomial_residual(4, degree, matrix)
    assert residual > 1.0, (
        f"{matrix} assembly residual {residual:.3e}: if this drops to zero the "
        "m16266 verdict would have to be re-read as a closure result"
    )


@pytest.mark.parametrize("degree", [2, 3])
def test_oriented_assembly_is_consistent(degree: int):
    assert P.poly_residual_oriented(4, degree) < 1e-8


def _rank(n: int, k: int, c: float) -> tuple[int, int]:
    A, _ = P.assemble_oriented(n, k, c=c)
    return int(np.linalg.matrix_rank(A, tol=1e-10)), A.shape[0]


@pytest.mark.parametrize("degree", [1, 2])
def test_pure_dirichlet_closure_is_rank_deficient(degree: int):
    """q_hat = q_int on a Dirichlet face degenerates the last mass row (0 = h f)."""
    rank, ndofs = _rank(4, degree, c=0.0)
    assert ndofs - rank == 1


@pytest.mark.parametrize("degree", [1, 2])
def test_penalised_closure_is_full_rank(degree: int):
    rank, ndofs = _rank(4, degree, c=10.0)
    assert rank == ndofs


def test_penalised_closure_does_not_reach_optimal_order():
    """Negative guard: the open problem is real, so do not claim k+1 order."""
    degree = 2
    errs = []
    for n in (8, 16):
        A, b = P.assemble_oriented(n, degree, c=10.0)
        xi, w, phi, *_ = P.basis(degree)
        m = degree + 1
        h = 1.0 / n
        sol = np.linalg.solve(A, b)
        num = den = 0.0
        for e in range(n):
            xq = (e + 0.5) * h + 0.5 * h * xi
            uh = phi.T @ sol[e * 2 * m + m:(e + 1) * 2 * m]
            ex = np.sin(np.pi * xq)
            num += 0.5 * h * np.sum(w * (uh - ex) ** 2)
            den += 0.5 * h * np.sum(w * ex ** 2)
        errs.append(np.sqrt(num / den))
    order = np.log(errs[0] / errs[1]) / np.log(2)
    assert order < degree + 0.5, (
        f"observed order {order:.2f} for k={degree}: if this ever exceeds k+0.5 the "
        "closure has been fixed and this guard plus docs/ldg_closure_reassessment_m16388.md "
        "must be updated"
    )