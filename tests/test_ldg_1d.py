"""Verification gates for the mixed-form LDG solver (``tensordg.ldg_1d``).

Two layers, in the repository's convention:

* **independent references.**  The reference-element operators are checked against
  closed forms (``int L_i L_j = 2/(2i+1) delta_ij``, the ``D_coef + D_test``
  boundary identity, strict triangularity), the ``k = 0`` scheme against the
  classical staggered stencil, and exactness against a solution the space
  contains exactly.  A broken assembly therefore cannot fake a passing gate.
* **gates on measured behaviour.**  Observed convergence orders are compared with
  the theoretical ``k + 1``, the conditioning is bounded, and the two failure
  modes documented in ``docs/CASE-DG-LDG-1D-MIXED.md`` (missing boundary flux,
  wrong trace orientation) are asserted to be *detectable* by this test suite.
"""
import math

import numpy as np
import pytest

from tensordg.convergence import observed_order
from tensordg.ldg_1d import (
    TRACE_POLICIES,
    assemble_1d,
    reference_operators,
    solve_ldg_1d,
)

RHS = lambda x: np.pi ** 2 * np.sin(np.pi * x)
EXACT_U = lambda x: np.sin(np.pi * x)
EXACT_Q = lambda x: np.pi * np.cos(np.pi * x)
RESOLUTIONS = (8, 16, 32)


# --------------------------------------------------------------------------
# independent references
# --------------------------------------------------------------------------
def test_reference_mass_matrix_matches_closed_form():
    for degree in (0, 1, 2, 3, 4):
        ops = reference_operators(degree)
        expected = np.diag([2.0 / (2 * j + 1) for j in range(degree + 1)])
        assert np.allclose(ops.mass, expected, atol=1e-12)


def test_derivative_matrices_obey_the_boundary_identity():
    for degree in (1, 2, 3, 4):
        ops = reference_operators(degree)
        m = degree + 1
        boundary = np.array([[1.0 - (-1.0) ** (i + j) for j in range(m)] for i in range(m)])
        assert np.allclose(ops.derivative_coef + ops.derivative_test, boundary, atol=1e-12)
        # D_test = int L_i' L_j is strictly lower triangular, D_coef strictly upper
        assert np.allclose(np.triu(ops.derivative_test), 0.0, atol=1e-12)
        assert np.allclose(np.tril(ops.derivative_coef), 0.0, atol=1e-12)


def test_k0_scheme_reduces_to_the_staggered_stencil():
    """degree 0:  h q_e - u_e + u_{e-1} = 0  and  q_e - q_{e+1} = h f_e."""
    n = 4
    h = 1.0 / n
    matrix, force, _ = assemble_1d(n, 0, RHS, beta=2.0)
    dense = matrix.toarray()
    for e in range(1, n - 1):  # interior q-rows (both faces interior)
        row = dense[2 * e]
        assert row[2 * e] == pytest.approx(h, abs=1e-12)          # q_e
        assert row[2 * e + 1] == pytest.approx(-1.0, abs=1e-12)   # u_e
        assert row[2 * (e - 1) + 1] == pytest.approx(1.0, abs=1e-12)  # u_{e-1}
    for e in range(n - 1):  # interior flux rows
        row = dense[2 * e + 1]
        assert row[2 * e] == pytest.approx(1.0, abs=1e-12)        # q_e
        assert row[2 * (e + 1)] == pytest.approx(-1.0, abs=1e-12)  # q_{e+1}
        # the load is the *element integral* of f (not h times the midpoint value):
        # int_{x_l}^{x_r} pi^2 sin(pi x) dx = pi (cos(pi x_l) - cos(pi x_r))
        exact_load = np.pi * (np.cos(np.pi * e * h) - np.cos(np.pi * (e + 1) * h))
        assert force[2 * e + 1] == pytest.approx(exact_load, rel=1e-10)


def test_quadratic_solution_is_reproduced_exactly():
    exact = lambda x: x * (1.0 - x)
    result = solve_ldg_1d(4, 2, lambda x: 2.0 + 0.0 * x)
    assert result.l2_error_u(exact) < 1e-12


def test_dirichlet_data_is_interpolated_at_the_endpoints():
    exact = lambda x: x ** 2
    # -u'' = -2 for u = x^2
    result = solve_ldg_1d(6, 2, lambda x: -2.0 + 0.0 * x, g0=0.0, g1=1.0)
    assert result.l2_error_u(exact) < 1e-12
    assert result.u(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)
    assert result.u(np.array([1.0]))[0] == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# gates on measured behaviour
# --------------------------------------------------------------------------
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_ldg_is_optimal_order_in_u(degree):
    errors = [solve_ldg_1d(n, degree, RHS).l2_error_u(EXACT_U) for n in RESOLUTIONS]
    order = observed_order(list(RESOLUTIONS), errors)
    assert order == pytest.approx(degree + 1, abs=0.15), (degree, errors, order)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_ldg_auxiliary_field_converges(degree):
    """q_h -> u' at least at order k (measured 2.06 / 2.51 / 4.10 for k = 1/2/3)."""
    errors = [solve_ldg_1d(n, degree, RHS).l2_error_q(EXACT_Q) for n in RESOLUTIONS]
    assert observed_order(list(RESOLUTIONS), errors) >= degree


def test_level_zero_is_first_order():
    errors = [solve_ldg_1d(n, 0, RHS).l2_error_u(EXACT_U) for n in RESOLUTIONS]
    assert observed_order(list(RESOLUTIONS), errors) == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize("degree", [1, 2])
def test_periodic_closure_is_optimal_order(degree):
    rhs = lambda x: 4.0 * np.pi ** 2 * np.sin(2.0 * np.pi * x)
    exact = lambda x: np.sin(2.0 * np.pi * x)
    errors = [solve_ldg_1d(n, degree, rhs, boundary="periodic").l2_error_u(exact)
              for n in RESOLUTIONS]
    assert observed_order(list(RESOLUTIONS), errors) == pytest.approx(degree + 1, abs=0.2)


def test_conditioning_stays_polynomial():
    for degree in (1, 2, 3):
        result = solve_ldg_1d(64, degree, RHS)
        assert result.condition_number < 1e6, (degree, result.condition_number)


# --------------------------------------------------------------------------
# the two documented failure modes must be detectable, not merely documented
# --------------------------------------------------------------------------
def test_missing_boundary_discrete_flux_makes_the_system_singular():
    n, degree = 4, 1
    dense = lambda beta: assemble_1d(n, degree, RHS, beta=beta)[0].toarray()
    sv_bare = np.linalg.svd(dense(0.0), compute_uv=False)
    sv_fixed = np.linalg.svd(dense(2.0), compute_uv=False)
    assert np.sum(sv_bare < 1e-10 * sv_bare[0]) >= 1          # rank deficient
    assert np.sum(sv_fixed < 1e-10 * sv_fixed[0]) == 0        # invertible
    assert sv_bare[0] / sv_bare[-1] > 1e10
    assert sv_fixed[0] / sv_fixed[-1] < 1e6


def test_polynomial_exactness_cannot_rank_trace_sides_but_catches_bases():
    """The trap, precisely scoped.

    For an exactly representable solution every *trace-side* choice reproduces it
    (the traces coincide), so a polynomial residual cannot rank ``ldg`` against
    ``per_element`` or ``far_both`` -- the discriminating evidence has to be the
    order measured on a solution the space does *not* contain.  An *endpoint-basis*
    error (``self_endpoint``) is a different matter: it does show up here.
    """
    poly_rhs = lambda x: 2.0 + 0.0 * x
    poly_u = lambda x: x * (1.0 - x)
    poly_q = lambda x: 1.0 - 2.0 * x
    n, degree = 4, 2
    m = degree + 1
    h = 1.0 / n
    residuals = {}
    for policy in TRACE_POLICIES:
        matrix, force, ops = assemble_1d(n, degree, poly_rhs, trace_policy=policy)
        minv = np.linalg.inv(ops.mass)
        x_exact = np.zeros(2 * n * m)
        for e in range(n):
            xq = (e + 0.5) * h + 0.5 * h * ops.xi
            x_exact[e * 2 * m:e * 2 * m + m] = minv @ (ops.phi @ (ops.weights * poly_q(xq)))
            x_exact[e * 2 * m + m:(e + 1) * 2 * m] = minv @ (ops.phi @ (ops.weights * poly_u(xq)))
        residuals[policy] = float(np.linalg.norm(matrix @ x_exact - force))
    for policy in ("ldg", "per_element", "far_both"):
        assert residuals[policy] < 1e-12, (policy, residuals)
    assert residuals["self_endpoint"] > 1e-3, residuals


def test_wrong_trace_orientation_is_caught_by_the_order_gate():
    """The verified orientation reaches k + 1; the historical ones do not."""
    ladders = (8, 16, 32)
    good = [solve_ldg_1d(n, 2, RHS).l2_error_u(EXACT_U) for n in ladders]
    assert observed_order(list(ladders), good) > 2.8
    failing = []
    for policy in ("per_element", "far_both"):
        errors = []
        for n in ladders:
            try:
                errors.append(solve_ldg_1d(n, 2, RHS, trace_policy=policy).l2_error_u(EXACT_U))
            except FloatingPointError:
                errors.append(float("inf"))
        try:
            failing.append(observed_order(list(ladders), errors))
        except ValueError:      # singular ladder: no finite order at all
            failing.append(0.0)
    # a singular/solver-failed run yields nan or inf, which must count as a failure
    assert not any(f >= 2.0 for f in failing), failing


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"n_elements": 0, "degree": 1},
    {"n_elements": 4, "degree": -1},
])
def test_invalid_sizes_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_ldg_1d(rhs=RHS, **kwargs)


def test_unknown_policy_and_boundary_are_rejected():
    with pytest.raises(ValueError):
        solve_ldg_1d(4, 1, RHS, trace_policy="nope")
    with pytest.raises(ValueError):
        solve_ldg_1d(4, 1, RHS, boundary="neumann")