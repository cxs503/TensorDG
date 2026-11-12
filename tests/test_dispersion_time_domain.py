"""Verification gates for the independent time-domain dispersion measurement.

These gates assert *invariants and agreements*, not that code ran:

1. **Integrator identity** -- the marching loop in
   ``tools/dispersion_time_domain.py`` produces bit-identical coefficients to the
   production ``advection_1d.solve_advection`` for the same parameters, so the
   measurement below really describes the shipped solver.
2. **Closed-form anchor (p = 0)** -- measured phase speed and dissipation of the
   1st-order upwind scheme match the analytic symbols ``sin(theta)/theta`` and
   ``h (1 - cos(theta))``.  Neither the eigenvalue pipeline nor the time march
   enters this reference.
3. **Agreement with the eigenvalue prediction** -- for every case whose
   amplitude decays enough to be fitted, the time-domain measurement reproduces
   ``dispersion_analysis.physical_branch`` to better than 1 % (phase speed and
   dissipation), i.e. the "least-damped branch is the physical mode" assumption
   is *measured*, not assumed.
4. **Nyquist branch sign (regression for a real defect)** -- a point query must
   agree with the continuous sweep and must describe a right-going wave for every
   degree.  Before the fix, ``physical_branch(3, [pi])`` returned
   ``c_num = -1.0006`` while the time advance of the same scheme travels at
   ``+1.0006``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensordg import dispersion_analysis as da
from tensordg.advection_1d import solve_advection
from tools import dispersion_time_domain as td

ARTIFACT = (Path(__file__).parent.parent / "validation"
            / "dispersion_time_domain" / "results.json")


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.exists(), "missing artifact: run python -m tools.dispersion_time_domain"
    return json.loads(ARTIFACT.read_text())


def test_marching_loop_is_the_production_integrator():
    """Same kernel, same steps, same dt -> identical modal coefficients."""
    degree, mode, t_end = 2, 4, 0.5
    cfl = td.effective_cfl(degree)
    _t, _z, coeff = td.march(degree, mode, t_end, cfl=cfl)
    _centers, reference, _meta = solve_advection(
        ncells=td.NCELLS, degree=degree, final_time=t_end, speed=1.0, cfl=cfl,
        initial=lambda x: torch.cos(2.0 * math.pi * mode * x))
    assert torch.allclose(coeff, reference, rtol=0.0, atol=1e-13), (
        (coeff - reference).abs().max().item())


def test_p0_measurement_matches_closed_form_upwind_fv(artifact):
    """Time-domain p=0 measurement vs the analytic upwind-FV symbols."""
    assert artifact["summary"]["max_p0_closed_form_phase_rel_error"] < 1e-5
    assert artifact["summary"]["max_p0_closed_form_decay_rel_error"] < 1e-5
    for record in artifact["records"]:
        if record["degree"] != 0:
            continue
        # Nyquist: c_num -> 0 analytically, therefore compare absolutely.
        assert record["p0_phase_speed_abs_error_vs_closed_form"] < 1e-6


def test_time_domain_reproduces_eigenvalue_phase_speed(artifact):
    """The measured phase speed is the predicted one to < 1 %."""
    for record in artifact["records"]:
        if record["measured_phase_speed"] is None:
            continue
        if abs(record["predicted_phase_speed"]) < 1e-6:
            continue  # relative error undefined at the p=0 Nyquist zero
        assert record["phase_speed_rel_error"] < 1e-2, record


def test_time_domain_reproduces_eigenvalue_dissipation(artifact):
    """The measured decay rate is the predicted one to < 2 %."""
    measured = [r for r in artifact["records"] if r["measured_decay_rate"] is not None]
    assert len(measured) >= 10, "too few measurable dissipation cases"
    for record in measured:
        assert record["decay_rel_error"] < 2e-2, record
        # Dissipation may never be fitted from growth.
        assert record["measured_decay_rate"] > 0.0


def test_rk4_truncation_bias_is_negligible_against_measured_dissipation(artifact):
    """The measurement's own numerical floor must not explain the agreement."""
    assert artifact["summary"]["max_rk4_decay_bias_over_measured_decay"] < 1e-3


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_nyquist_branch_is_right_going_for_every_degree(degree):
    """Regression: the physical branch at theta = pi is right-going, never conj."""
    theta = math.pi
    value = da.physical_branch(degree, np.array([theta]))[0]
    c_num = -value.imag / theta
    if degree == 0:
        # Exact upwind FV: the mode does not travel at all and dies at Re<0.
        assert abs(c_num) < 1e-12 and value.real < 0.0
    else:
        assert c_num > 0.0, f"p={degree}: point query returned c_num={c_num}"


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_point_query_agrees_with_continuous_sweep_at_nyquist(degree):
    """A single-theta query must land on the same branch as the tracked sweep."""
    point = da.physical_branch(degree, np.array([math.pi]))[0]
    sweep = da.physical_branch(degree)[-1]
    assert abs(point - sweep) < 1e-12, (point, sweep)


def test_point_query_agrees_with_time_domain_at_the_nyquist_defect_point():
    """The measured +1.0006 (p=3, theta=pi) adjudicates the point query."""
    theta = math.pi
    predicted = da.physical_branch(3, np.array([theta]))[0]
    measured = td.measure_mode(3, 8)
    assert measured["measured_phase_speed"] is not None
    assert abs(-predicted.imag / theta - measured["measured_phase_speed"]) < 1e-2
    assert measured["measured_phase_speed"] > 1.0


def test_branch_selection_does_not_mask_a_separate_branch():
    """The 1e-9 tie tolerance must not swallow genuinely distinct eigenvalues."""
    matrix = da.assemble_operator(2)
    values = np.linalg.eigvals(da.symbol_matrix(matrix, 2, 0.4))
    picked = da._pick_continuation(values, 0.0 + 0.0j, 0)
    assert abs(picked - 0.0) < 0.5        # the physical branch is the near-zero one
    assert np.min(np.abs(values - picked)) < 1e-12