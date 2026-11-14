"""p-Order convergence verification for linear advection (CASE-DG-ADVECT-ORDER).

Tests that the DG advection solver achieves the theoretical design order
``p + 1`` in the L2 norm for smooth solutions, for both 1-D and 2-D
periodic linear advection.

Theoretical background
----------------------
For the linear advection equation ``u_t + a·∇u = 0`` with a smooth exact
solution, the DG method with polynomial degree ``p`` and upwind flux has
a design order of accuracy ``p + 1`` in the L2 norm (Cockburn & Shu 2001,
Hesthaven & Warburton 2008).  This holds provided:

1. The time integrator is of sufficiently high order (RK4 is 4th-order,
   so it does not bind for ``p <= 3``; for ``p = 4`` the temporal error
   may contribute but is negligible at the resolutions used here).
2. The solution is smooth (no shocks or discontinuities).
3. The mesh is regular (uniform Cartesian in our case).

Acceptance criterion
--------------------
The observed order of accuracy (least-squares fit of ``log(error)`` vs
``log(resolution)``) must be within ``±0.3`` of the theoretical value
``p + 1``.  This tolerance is generous enough to account for:
- Pre-asymptotic effects on coarse grids
- Temporal discretisation error (RK4, not SSP)
- Quadrature error in the error norm evaluation

References
----------
- Cockburn, B. & Shu, C.-W. (2001). "Runge–Kutta discontinuous Galerkin
  methods for convection-dominated problems." J. Sci. Comput. 16(3), 173–261.
- Hesthaven, J.S. & Warburton, T. (2008). "Nodal Discontinuous Galerkin
  Methods." Springer.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensordg.convergence import convergence_table

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "benchmarks" / "results" / "linear-advection-porder" / "porder_convergence.json"

ORDER_TOLERANCE = 0.3


@pytest.fixture(scope="module")
def artifact():
    """Load the pre-computed convergence study artifact."""
    assert ARTIFACT.exists(), (
        f"Missing artifact {ARTIFACT}; run `python tools/run_advection_porder_study.py` first"
    )
    return json.loads(ARTIFACT.read_text())


# ---------------------------------------------------------------------------
# 1-D convergence tests
# ---------------------------------------------------------------------------
class Test1DConvergence:
    """1-D periodic linear advection: observed order vs theoretical p+1."""

    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
    def test_order_within_tolerance(self, artifact, degree):
        """Each degree must achieve observed order within tolerance of p+1."""
        key = f"p{degree}"
        data = artifact["study_1d"]["results"][key]
        assert data["order_within_tolerance"], (
            f"P{degree}: observed={data['observed_order_lsq']:.4f}, "
            f"theoretical={degree+1}, deviation={data['order_deviation']:.4f}"
        )

    @pytest.mark.parametrize("degree,expected_min", [
        (0, 0.7),   # P0: ~0.85 observed (pre-asymptotic, but > 0.7)
        (1, 1.5),   # P1: ~2.1 observed
        (2, 2.5),   # P2: ~3.0 observed
        (3, 3.5),   # P3: ~4.0 observed
        (4, 4.5),   # P4: ~5.0 observed
    ])
    def test_order_exceeds_minimum(self, artifact, degree, expected_min):
        """Observed order must exceed a conservative minimum."""
        key = f"p{degree}"
        data = artifact["study_1d"]["results"][key]
        assert data["observed_order_lsq"] >= expected_min, (
            f"P{degree}: observed={data['observed_order_lsq']:.4f} < {expected_min}"
        )

    def test_error_monotonically_decreases(self, artifact):
        """L2 error must decrease with refinement for every degree."""
        for key, data in artifact["study_1d"]["results"].items():
            errs = data["errors_l2"]
            for i in range(1, len(errs)):
                assert errs[i] < errs[i - 1], (
                    f"{key}: error not decreasing at refinement level {i}: "
                    f"{errs[i-1]:.6e} -> {errs[i]:.6e}"
                )


# ---------------------------------------------------------------------------
# 2-D convergence tests
# ---------------------------------------------------------------------------
class Test2DConvergence:
    """2-D periodic linear advection: observed order vs theoretical p+1."""

    @pytest.mark.parametrize("degree", [0, 1, 2, 3])
    def test_order_within_tolerance(self, artifact, degree):
        """Each degree must achieve observed order within tolerance of p+1."""
        key = f"p{degree}"
        data = artifact["study_2d"]["results"][key]
        assert data["order_within_tolerance"], (
            f"P{degree}: observed={data['observed_order_lsq']:.4f}, "
            f"theoretical={degree+1}, deviation={data['order_deviation']:.4f}"
        )

    def test_error_monotonically_decreases(self, artifact):
        """L2 error must decrease with refinement for every degree."""
        for key, data in artifact["study_2d"]["results"].items():
            errs = data["errors_l2"]
            for i in range(1, len(errs)):
                assert errs[i] < errs[i - 1], (
                    f"{key}: error not decreasing at refinement level {i}: "
                    f"{errs[i-1]:.6e} -> {errs[i]:.6e}"
                )


# ---------------------------------------------------------------------------
# Artifact contract
# ---------------------------------------------------------------------------
def test_artifact_contract(artifact):
    """The artifact must have the expected structure and pass overall."""
    assert artifact["benchmark"] == "linear_advection_porder"
    assert artifact["anchor"] == "CASE-DG-ADVECT-ORDER"
    assert artifact["repo"] == "TensorDG"
    assert artifact["verdict"]["overall"] == "PASS", artifact["verdict"]
    assert artifact["verdict"]["all_1d_pass"]
    assert artifact["verdict"]["all_2d_pass"]
    # Figures must exist
    for fig in artifact["figures"]:
        fig_path = ROOT / "docs" / "benchmarks" / "results" / "linear-advection-porder" / fig["name"]
        assert fig_path.exists(), f"Missing figure: {fig_path}"
