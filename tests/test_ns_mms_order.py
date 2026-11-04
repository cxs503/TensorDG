"""Regression tests for the manufactured-solution order verification of the NS DG solver.

Two independent properties are checked:

* the sympy-derived manufactured source agrees with a central-difference
  evaluation of the same PDE (this is what protects the verification tool
  itself -- if the source were wrong, the convergence study below would silently
  measure nothing);
* the coupled convective + viscous operator converges at the design order
  ``degree + 1`` in the L2 norm for every conserved variable.
"""
import numpy as np
import pytest

from benchmarks.verify_ns_mms_order import (
    FIELDS,
    order_case,
    source_selfcheck,
)


def test_manufactured_source_matches_finite_differences():
    """The generated source must match an independent finite-difference PDE residual."""
    check = source_selfcheck()
    assert check["pass"], (
        "manufactured source disagrees with the finite-difference PDE residual: "
        f"relative difference {check['relative_difference']:.3e}"
    )
    assert check["relative_difference"] < 1.0e-5


def test_source_hook_is_recorded_in_metadata():
    case = order_case(1, 8, final_time=0.01)
    assert case["errors"]["density"]["l2"] > 0.0


@pytest.mark.parametrize("degree,meshes,minimum_order", [(1, (8, 16, 24), 1.7),
                                                        (2, (8, 16), 2.5)])
def test_design_order_is_recovered(degree, meshes, minimum_order):
    """Observed order of accuracy must approach ``degree + 1``."""
    cases = [order_case(degree, ncells, final_time=0.03) for ncells in meshes]
    for field in FIELDS:
        errors = [case["errors"][field]["l2"] for case in cases]
        assert all(later < earlier for earlier, later in zip(errors, errors[1:])), (
            f"degree {degree} {field}: L2 error is not decreasing with mesh refinement: {errors}"
        )
        orders = [
            float(np.log(errors[i] / errors[i + 1]) / np.log(meshes[i + 1] / meshes[i]))
            for i in range(len(meshes) - 1)
        ]
        mean_order = float(np.mean(orders))
        assert mean_order >= minimum_order, (
            f"degree {degree} {field}: mean observed order {mean_order:.3f} "
            f"< required {minimum_order} (per-mesh {orders})"
        )