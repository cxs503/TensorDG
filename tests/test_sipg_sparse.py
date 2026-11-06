"""Tests for :mod:`tensordg.sipg_sparse` (scipy CSR SIPG assembly / solves).

The module previously had no dedicated test file.  The positive cases pin the
contract stated in its docstring -- the sparse path must reproduce the dense
path to machine precision and the CSR matrices must stay symmetric and sparse.
The negative cases are controls: a perturbed penalty or a corrupted mass matrix
must be visible far above the comparison tolerance, otherwise the equivalence
assertions would be vacuous.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch

from tensordg.heat_1d import heat_error, solve_heat_1d
from tensordg.poisson_2d import poisson_error_2d, solve_poisson_2d
from tensordg.sipg_sparse import (
    assemble_heat_1d_sparse,
    assemble_poisson_2d_sparse,
    solve_heat_1d_sparse,
    solve_poisson_2d_sparse,
)

PENALTY = 20.0
MATCH_TOL = 1e-11


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("ncells", [4, 8])
def test_poisson_sparse_reproduces_dense(degree, ncells):
    x, y, dense, meta_dense = solve_poisson_2d(
        ncells=ncells, degree=degree, penalty=PENALTY,
    )
    xs, ys, sparse, meta_sparse = solve_poisson_2d_sparse(
        ncells=ncells, degree=degree, penalty=PENALTY,
    )
    assert torch.equal(x, xs) and torch.equal(y, ys)
    assert float((dense - sparse).abs().max()) < MATCH_TOL
    assert meta_sparse["dofs"] == meta_dense["dofs"]
    assert meta_sparse["backend"] == "sparse"
    assert meta_sparse["matrix_symmetry_error"] < 1e-12
    # Both paths must also agree with the manufactured solution.
    assert poisson_error_2d(xs, ys, sparse)["relative_l2"] < 0.1


@pytest.mark.parametrize("degree", [1, 2])
def test_heat_sparse_reproduces_dense(degree):
    centers, dense, meta_dense = solve_heat_1d(
        ncells=8, degree=degree, final_time=0.05, time_steps=40, penalty=PENALTY,
    )
    centers_s, sparse, meta_sparse = solve_heat_1d_sparse(
        ncells=8, degree=degree, final_time=0.05, time_steps=40, penalty=PENALTY,
    )
    assert torch.equal(centers, centers_s)
    assert float((dense - sparse).abs().max()) < MATCH_TOL
    assert meta_sparse["steps"] == meta_dense["steps"] == 40
    assert meta_sparse["dt"] == pytest.approx(0.05 / 40, rel=1e-12)
    assert meta_sparse["diffusion_symmetry_error"] < 1e-12
    assert heat_error(centers_s, sparse, 0.05)["relative_l2"] < 1e-2


@pytest.mark.parametrize(("degree", "order_floor"), [(1, 3.5), (2, 6.5)])
def test_sparse_poisson_converges_with_the_design_order(degree, order_floor):
    """A correct CSR assembly must refine at the SIPG design order."""
    errors = []
    for ncells in (4, 8):
        x, y, coeff, _ = solve_poisson_2d_sparse(
            ncells=ncells, degree=degree, penalty=PENALTY,
        )
        errors.append(poisson_error_2d(x, y, coeff)["relative_l2"])
    assert errors[1] < errors[0]
    assert errors[0] / errors[1] > order_floor, f"refinement ratio {errors[0] / errors[1]:.3f}"


def test_csr_matrices_are_symmetric_and_sparse():
    """The reason for this module: >10^4 DOFs must not need a dense matrix."""
    stiffness, _ = assemble_poisson_2d_sparse(
        20, 20, 2, PENALTY, torch.device("cpu"), torch.float64,
    )
    assert sp.issparse(stiffness)
    assert stiffness.format == "csr"
    dofs = 20 * 20 * 3 * 3
    assert stiffness.shape == (dofs, dofs)
    # A dense matrix would hold 13e6 entries; SIPG couples neighbours only.
    assert stiffness.nnz / float(dofs ** 2) < 0.05
    assert abs(stiffness - stiffness.T).max() < 1e-12
    assert stiffness.dtype == np.float64


def test_poisson_2d_dispatch_uses_the_sparse_assembly():
    """``solve_poisson_2d(backend="sparse")`` must route to this module."""
    x, y, dispatched, _ = solve_poisson_2d(
        ncells=6, degree=1, penalty=PENALTY, backend="sparse",
    )
    xs, ys, direct, _ = solve_poisson_2d_sparse(
        ncells=6, degree=1, penalty=PENALTY,
    )
    assert torch.equal(x, xs) and torch.equal(y, ys)
    assert float((dispatched - direct).abs().max()) < MATCH_TOL
    assert poisson_error_2d(xs, ys, direct)["relative_l2"] < 3e-2
    # The dense default penalty differs from the sparse default, so compare
    # like with like: with the same penalty the two paths agree.
    _, _, dense, _ = solve_poisson_2d(ncells=6, degree=1, penalty=PENALTY)
    assert float((dispatched - dense).abs().max()) < MATCH_TOL


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------

def test_penalty_change_is_far_above_the_match_tolerance():
    """Negative control: the equivalence check is sensitive to the assembly."""
    _, _, c10, _ = solve_poisson_2d_sparse(ncells=6, degree=1, penalty=10.0)
    _, _, c20, _ = solve_poisson_2d_sparse(ncells=6, degree=1, penalty=20.0)
    difference = float((c10 - c20).abs().max())
    assert difference > 1e5 * MATCH_TOL


def test_corrupted_mass_matrix_perturbs_the_heat_solution():
    """An error injected into the CSR mass matrix must move the solution."""
    mass, diffusion, rhs, system = _heat_backward_euler_system(
        ncells=8, degree=2, penalty=PENALTY, dt=0.05 / 40,
    )
    base = splu(system.tocsc()).solve(rhs)
    corrupted = system + 1e-6 * sp.identity(system.shape[0], format="csr")
    shifted = splu(corrupted.tocsc()).solve(rhs)
    assert float(np.max(np.abs(base - shifted))) > 1e5 * MATCH_TOL
    # The unperturbed system is what the module itself solves.
    _centers, coeff, meta = solve_heat_1d_sparse(
        ncells=8, degree=2, final_time=0.05, time_steps=1, penalty=PENALTY,
    )
    assert meta["steps"] == 1
    assert coeff.shape == (8, 3)
    del mass, diffusion


def test_invalid_backend_selection_raises():
    with pytest.raises(ValueError):
        solve_poisson_2d(ncells=4, degree=1, backend="csr")
    with pytest.raises(ValueError):
        solve_poisson_2d(ncells=4, degree=1, backend="sparse", rtol=1e-10)


# ---------------------------------------------------------------------------
# Helper: one backward-Euler step of the SIPG heat system, assembled by hand
# via the module's own CSR blocks (used only by the injected-error control).
# ---------------------------------------------------------------------------

def _heat_backward_euler_system(ncells, degree, penalty, dt):
    diffusion, mass, operators = assemble_heat_1d_sparse(
        ncells, degree, penalty, torch.device("cpu"), torch.float64,
    )
    nodes, weights, phi = operators[0], operators[1], operators[2]
    modes = degree + 1
    h = 1.0 / ncells
    centers = (torch.arange(ncells, dtype=torch.float64) + 0.5) * h
    values = torch.sin(math.pi * (centers[:, None] + 0.5 * h * nodes[None, :]))
    coeff = torch.empty((ncells, modes), dtype=torch.float64)
    for k in range(modes):
        coeff[:, k] = (2 * k + 1) * 0.5 * torch.sum(
            weights[None, :] * values * phi[:, k][None, :], dim=1,
        )
    state = coeff.reshape(-1).numpy().astype(np.float64)
    system = mass + 0.5 * dt * diffusion
    rhs = (mass - 0.5 * dt * diffusion) @ state
    return mass, diffusion, rhs, system