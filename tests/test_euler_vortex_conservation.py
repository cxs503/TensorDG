"""CASE-DG-EULER-VORTEX-ERR: Conservation verification for the isentropic vortex.

For the Euler equations on a periodic domain, a conservative numerical flux
(HLLC for DG, Rusanov for P0) must preserve the domain-integral of the four
conserved quantities (mass, x-momentum, y-momentum, energy) to machine
precision.  This test verifies that both the high-order DG path and the P0
baseline satisfy this invariant.

Reference: the conservation property follows from the divergence theorem:
  d/dt int_Omega U dV = - int_dOmega F·n dS = 0  (periodic BCs)
so the discrete scheme must satisfy sum_i U_i |Omega_i| = const.
"""
from __future__ import annotations

import math

import pytest

from benchmarks.case_euler_vortex import (
    CONSERVATION_NAMES,
    run_dg,
    run_p0,
)

# Machine-precision tolerance for conservation drift (float64).
# The DG scheme uses HLLC (exactly conservative) and the P0 scheme uses
# Rusanov (also exactly conservative); with periodic BCs the net boundary
# flux is zero, so drift should be at the level of round-off accumulation.
CONSERVATION_TOL = 1e-12


def _conservation_pass(record: dict) -> tuple[bool, dict]:
    """Check whether all conservation drifts are below tolerance."""
    drift = record.get("conservation_drift", {})
    results = {}
    for name in CONSERVATION_NAMES:
        value = drift.get(name, float("nan"))
        results[name] = value < CONSERVATION_TOL if not math.isnan(value) else False
    all_pass = all(results.values())
    return all_pass, results


class TestDGConservation:
    """High-order DG (HLLC flux) must conserve to machine precision."""

    @pytest.mark.parametrize("degree,cells", [(1, 8), (2, 8), (3, 6)])
    def test_dg_conservation_periodic(self, degree, cells):
        config = {
            "task": "dg",
            "degree": degree,
            "cells": cells,
            "final_time": 1.0,
            "cfl": 0.2,
        }
        record = run_dg(config)
        passed, details = _conservation_pass(record)
        assert passed, (
            f"DG degree={degree} cells={cells} conservation drift exceeds "
            f"{CONSERVATION_TOL:.0e}: {details}"
        )

    def test_dg_conservation_with_limiter(self):
        """Even with a limiter (which preserves cell averages), conservation holds."""
        from tensordg.euler_dg_2d import solve_euler_dg_2d_vortex

        x, y, coeff, meta = solve_euler_dg_2d_vortex(
            ncells=8, degree=2, final_time=0.5, cfl=0.2, limiter="tvb", tvb_constant=10.0
        )
        drift = meta.get("conservation_drift", {})
        for name in CONSERVATION_NAMES:
            value = drift.get(name, float("nan"))
            assert not math.isnan(value), f"Missing conservation drift for {name}"
            assert value < CONSERVATION_TOL, (
                f"DG+limiter conservation drift for {name} = {value:.2e} > {CONSERVATION_TOL:.0e}"
            )


class TestP0Conservation:
    """P0 finite volume (Rusanov flux) must conserve to machine precision."""

    @pytest.mark.parametrize("cells", [16, 32])
    def test_p0_conservation_periodic(self, cells):
        config = {
            "task": "p0",
            "cells": cells,
            "final_time": 1.0,
            "cfl": 0.4,
        }
        record = run_p0(config)
        passed, details = _conservation_pass(record)
        assert passed, (
            f"P0 cells={cells} conservation drift exceeds "
            f"{CONSERVATION_TOL:.0e}: {details}"
        )


class TestConservationVsSolutionError:
    """Conservation drift is independent of solution error magnitude."""

    def test_coarse_mesh_conservation_still_exact(self):
        """Even on a very coarse mesh where solution error is large,
        conservation must still hold to machine precision."""
        # P0 at 16x16 has ~8% solution error but must conserve exactly
        record = run_p0({"task": "p0", "cells": 16, "final_time": 1.0, "cfl": 0.4})
        # Solution error is large (expected)
        assert record["max_relative_l2"] > 0.01, "Solution error should be significant on coarse mesh"
        # But conservation is exact
        passed, details = _conservation_pass(record)
        assert passed, f"Coarse mesh conservation failed: {details}"

    def test_fine_mesh_conservation_still_exact(self):
        """On a fine mesh where solution error is tiny, conservation is still exact."""
        record = run_dg({"task": "dg", "degree": 2, "cells": 32, "final_time": 1.0, "cfl": 0.2})
        # Solution error is tiny
        assert record["max_relative_l2"] < 0.001, "Solution error should be tiny on fine mesh"
        # Conservation is still exact
        passed, details = _conservation_pass(record)
        assert passed, f"Fine mesh conservation failed: {details}"
