"""Regression tests for the transient wall heat-flux verification (m3640).

The benchmark ``benchmarks/verify_ns_transient_wall_heat.py`` closes the
unsteady wall-heat-flux verification debt of ``tensordg/ns_dg_2d.py``.  These
tests keep three separate things honest:

* the analytic reference (``kappa dT / sqrt(pi alpha t)``) against a finite
  difference of the erfc profile -- if the closed form were wrong the whole
  benchmark would silently measure nothing;
* the new time-dependent Dirichlet wall data (``boundary`` entries may be
  callables) -- a constant callable must reproduce the constant-dict run
  exactly and the metadata must stay JSON-serialisable;
* the physical claim -- the DG boundary flux at the wall reproduces the erfc
  heat flux within 3 % and improves with mesh refinement, and the residual
  ``O(dT/T0)`` low-Mach model error vanishes as the amplitude is extrapolated
  to zero.
"""
import json
import math

import pytest
import torch

import tensordg.ns_dg_2d as ns
from benchmarks.verify_ns_transient_wall_heat import (
    ALPHA,
    FINAL_TIME,
    KAPPA,
    T0,
    exact_flux_selfcheck,
    exact_wall_heat_flux,
    hot_wall_spec,
    quasi_static_ratio,
    run_case,
    top_boundary_influence,
)


def test_exact_flux_matches_finite_difference():
    """The closed-form wall flux must match a finite difference of erfc."""
    check = exact_flux_selfcheck()
    assert check["pass"], check
    assert check["relative_difference"] < 1.0e-4


def test_flow_parameters_are_consistent():
    """alpha and kappa follow the solver's own conventions (R = 1, c_p)."""
    expected_kappa = 0.01 * (1.4 / 0.4) / 0.71
    assert KAPPA == pytest.approx(expected_kappa, rel=1e-12)
    assert ALPHA == pytest.approx(KAPPA / (1.0 * (1.4 / 0.4)), rel=1e-12)
    assert exact_wall_heat_flux(0.2, 1.0) == pytest.approx(
        KAPPA / math.sqrt(math.pi * ALPHA * 0.2), rel=1e-12
    )


def test_top_wall_is_invisible():
    """The semi-infinite reference is only valid if the far wall is negligible."""
    assert top_boundary_influence(FINAL_TIME) < 1.0e-6


def test_regime_is_quasi_static():
    """m3640 trap: the layer must be acoustically equilibrated, else q_w is biased low.

    Measured on this solver: ratio ``2 sqrt(alpha t)/(c_s t)`` = 1.00 gives a
    5.6 % wall-flux bias, 0.24 gives 0.4 %; the defaults sit at the safe end.
    """
    assert quasi_static_ratio(FINAL_TIME) < 0.3


def _tiny_run(boundary_scale):
    def initial(x, y):
        ones = torch.ones_like(x * y)
        zeros = torch.zeros_like(ones)
        return ones, zeros, zeros, torch.full_like(ones, T0)

    def wall(time_value):
        return {"velocity": (0.0, 0.0), "temperature": T0 + boundary_scale}

    fixed = {"velocity": (0.0, 0.0), "temperature": T0 + boundary_scale}
    return initial, wall, fixed


def test_callable_boundary_matches_constant_boundary():
    """A callable wall returning a fixed state must be a bit-identical run."""
    initial, wall, fixed = _tiny_run(0.01)
    common = dict(
        ncells=(1, 4), degree=1, domain=((0.0, 1.0), (0.0, 1.0)),
        final_time=2.0e-3, mu=0.05, prandtl=0.71, periodic=(True, False),
        boundary={"ymin": fixed, "ymax": fixed}, device="cpu",
        dtype=torch.float64,
    )
    _, _, coeff_fixed, _ = ns.solve_navier_stokes_dg_2d(initial, **common)
    common["boundary"] = {"ymin": wall, "ymax": fixed}
    _, _, coeff_callable, meta = ns.solve_navier_stokes_dg_2d(initial, **common)
    assert torch.allclose(coeff_fixed, coeff_callable, rtol=0.0, atol=0.0)
    # metadata must stay serialisable when a callable wall is used
    json.dumps(meta["boundary"])
    assert meta["boundary"]["ymin"] == {"callable": wall.__name__}


def test_transient_wall_heat_flux_is_below_three_percent():
    """P1 / 16 cells in y at t = 0.2 must reproduce the erfc wall flux within 3 %."""
    row = run_case(degree=1, ny=16, amplitude=0.005, final_time=FINAL_TIME)
    assert row["wall_heat_flux_relative_error"] < 0.03, row["wall_heat_flux_relative_error"]
    assert row["profile_relative_l2"] < 0.01, row["profile_relative_l2"]


def test_wall_heat_flux_error_decreases_with_mesh():
    coarse = run_case(degree=1, ny=8, amplitude=0.005, final_time=FINAL_TIME)
    fine = run_case(degree=1, ny=16, amplitude=0.005, final_time=FINAL_TIME)
    assert fine["wall_heat_flux_relative_error"] < coarse["wall_heat_flux_relative_error"]
    observed = math.log(coarse["wall_heat_flux_relative_error"]
                        / fine["wall_heat_flux_relative_error"]) / math.log(2.0)
    assert observed > 0.6, observed


def test_low_mach_model_error_vanishes_at_zero_amplitude():
    """The O(dT/T0) model error must extrapolate away."""
    coarse = run_case(degree=1, ny=8, amplitude=0.02, final_time=FINAL_TIME)
    fine = run_case(degree=1, ny=8, amplitude=0.005, final_time=FINAL_TIME)
    ratio_coarse = coarse["wall_heat_flux"] / coarse["amplitude"]
    ratio_fine = fine["wall_heat_flux"] / fine["amplitude"]
    slope = (ratio_coarse - ratio_fine) / (coarse["amplitude"] - fine["amplitude"])
    limit = ratio_fine - slope * fine["amplitude"]
    target = exact_wall_heat_flux(FINAL_TIME, 1.0)
    assert abs(limit - target) / target < 0.03, (limit, target)
    # the raw error at the largest amplitude is the model error dominated case
    assert coarse["wall_heat_flux_relative_error"] > fine["wall_heat_flux_relative_error"]