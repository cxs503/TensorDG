"""Live gates for the 2-D LDG Poisson solver (docs/CASE-DG-LDG-2D.md).

These tests exercise the shipped solver -- they do not read a stored JSON.  The
companion artifact ``validation/ldg2d/ldg_poisson_2d_study.json`` is regenerated
by ``python validation/ldg2d/ldg_poisson_2d_study.py``; the numbers asserted
here are the pre-registered gates of that study (order >= k+1-0.3, <3% at the
finest mesh, beta-insensitivity, closure ablation, polynomial exactness).
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

import tensordg
from tensordg.poisson_2d_ldg import (
    ldg_poisson_convergence,
    manufactured_solution,
    solve_poisson_2d_ldg,
    superposition_solution,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUDY = ROOT / "validation" / "ldg2d" / "ldg_poisson_2d_study.py"


def _study():
    spec = importlib.util.spec_from_file_location("ldg2d_study", STUDY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _errors(solution, degree, grids, **kwargs):
    return np.array([
        solve_poisson_2d_ldg(
            n, degree, rhs=solution["f"], dirichlet=solution["dirichlet"],
            exact=solution["u"],
            exact_grad=(solution["grad_x"], solution["grad_y"]), **kwargs,
        )["l2_relative_error"]
        for n in grids
    ])


def _order(grids, errors):
    return float(np.polyfit(np.log(grids), np.log(errors), 1)[0] * -1.0)


def test_degree_one_converges_at_second_order():
    grids = [4, 8, 16]
    errs = _errors(manufactured_solution(), 1, grids)
    assert _order(grids, errs) >= 2.0 - 0.3
    assert errs[-1] < 3.0e-2


def test_degree_two_converges_at_third_order():
    grids = [4, 8, 16]
    errs = _errors(manufactured_solution(), 2, grids)
    assert _order(grids, errs) >= 3.0 - 0.3
    assert errs[-1] < 3.0e-2


def test_non_homogeneous_dirichlet_data_is_actually_used():
    """u = sin sin + x: a solver that dropped g_D on dOmega cannot pass."""
    solution = superposition_solution()
    grids = [4, 8, 16]
    errs = _errors(solution, 1, grids)
    assert _order(grids, errs) >= 1.7
    assert errs[-1] < 3.0e-2
    # a solver that dropped the Dirichlet data (g_D = 0) would be far worse:
    # the missing x-term is O(1) in the L2 norm
    zero_data = solve_poisson_2d_ldg(
        16, 1, rhs=solution["f"],
        dirichlet=lambda x, y: np.zeros(np.shape(x)),
        exact=solution["u"],
        exact_grad=(solution["grad_x"], solution["grad_y"]),
    )
    assert zero_data["l2_relative_error"] > 10.0 * errs[-1]


def test_dirichlet_flux_beta_does_not_change_the_solution():
    ref = solve_poisson_2d_ldg(8, 1)["l2_relative_error"]
    for beta in (2.0, 8.0):
        err = solve_poisson_2d_ldg(8, 1, dirichlet_flux_beta=beta)["l2_relative_error"]
        assert abs(err - ref) / ref < 0.10


def test_bare_closure_is_rank_deficient():
    """The bare Dirichlet closure (q_hat = q_int) must stay singular."""
    study = _study()
    record = study.gate_rank_deficiency(1, 4)
    assert record["deficiency"] >= 1
    assert record["sigma_min"] < 1e-10 * record["sigma_max"]


def test_legacy_mirror_closure_is_not_convergent():
    """Ablation: u_hat = 2 g_D - u_int was the prototype's catastrophic closure."""
    errs = [
        solve_poisson_2d_ldg(n, 1, boundary_closure="mirror")["l2_relative_error"]
        for n in (4, 8)
    ]
    assert min(errs) > 1.0  # errors are O(1e12), not a discretisation constant


def test_polynomial_exactness_gate_discriminates():
    """u = x^2 is in the k=2 space: relative residual machine zero; k=1: not."""
    study = _study()
    exact = study.gate_polynomial_exactness(2)
    coarse = study.gate_polynomial_exactness(1)
    assert exact["residual_relative"] < 1e-12
    assert coarse["residual_relative"] > 1e-6


def test_conditioning_stays_below_the_gate():
    study = _study()
    for degree in (1, 2):
        record = study.gate_condition(degree, 4)
        assert record["sigma_min"] > 0.0
        assert record["condition"] < 1e8


def test_convergence_study_api_reports_orders():
    table = ldg_poisson_convergence(degrees=(1,), ncells=(4, 8, 16))
    assert table["orders"][1][0] >= 1.7
    assert table["orders_least_squares"][1] >= 1.7
    assert len(table["rows"]) == 3


def test_validated_solver_is_exported():
    """The module left PROTOTYPE status: it is reachable from the package root."""
    assert hasattr(tensordg, "solve_poisson_2d_ldg")
    assert tensordg.solve_poisson_2d_ldg is solve_poisson_2d_ldg
    module = (ROOT / "tensordg" / "poisson_2d_ldg.py").read_text()
    assert "STATUS: PROTOTYPE" not in module
    assert "STATUS: VALIDATED" in module