"""Regression tests for the differentiable-inverse-problem tutorial.

The tutorial ``docs/tutorials/differentiable-inverse-problems.md`` claims four
things; each claim is pinned down by a test that would fail if the machinery
behind it broke:

1. the autograd gradient of a DG misfit equals a central difference of the same
   code (``< 1e-7`` relative), so the descent direction is the right one;
2. a fit driven only by that gradient returns the data-generating parameter
   (round-off floor) for the hyperbolic *and* the parabolic model;
3. identified parameters estimated from **continuum** data converge to the
   analytic value under mesh refinement, i.e. identification is consistent and
   not merely self-consistent;
4. the workflow has teeth -- and it also has a documented failure mode: with a
   frozen CFL step, a perfect residual coexists with a 37.5 % parameter error,
   while the matched-step fit recovers the truth.

Numbers are asserted on coarse meshes so the file stays a few seconds long;
``python tests/inverse_study.py`` regenerates the full tables and figures.
"""
from __future__ import annotations

import math

import pytest
import torch

from tests._inverse_tools import (
    add_relative_noise,
    advection_continuum_data,
    advection_data,
    advection_objective,
    advection_setup,
    central_difference,
    heat_continuum_data,
    heat_data,
    heat_objective,
    heat_setup,
    relative_error,
    secant_fit,
    value_and_grad,
)
from tests.inverse_study import (
    study_continuum_consistency,
    study_discrete_identification,
    study_fd_step_sweep,
    study_frozen_dt_degeneracy,
    study_gradient_check,
    study_noise,
)

C_TRUE = 1.0
ALPHA_TRUE = 0.2
BOUNDS_C = (0.1, 3.0)
BOUNDS_ALPHA = (1.0e-3, 5.0)


# ---------------------------------------------------------------------------
# 1. gradient identity
# ---------------------------------------------------------------------------

def test_autograd_matches_central_difference():
    check = study_gradient_check()
    for name, entry in check.items():
        assert entry["relative_gap"] < 1.0e-7, (name, entry)
        assert entry["verdict"] == "PASS", (name, entry)


def test_fd_step_sweep_has_a_sweet_spot():
    """The gap must dip below 1e-7 for *some* eps on both models."""
    sweep = study_fd_step_sweep()
    assert sweep["advection_speed_best_gap"] < 1.0e-7
    assert sweep["heat_diffusivity_best_gap"] < 1.0e-7
    # large eps is dominated by truncation error: the sweep is not flat
    coarse = [row for row in sweep["heat_diffusivity"] if row["eps"] == 1.0e-3][0]
    assert coarse["relative_gap"] > 1.0e-7


# ---------------------------------------------------------------------------
# 2. identification against discrete data
# ---------------------------------------------------------------------------

def test_advection_speed_identified_from_discrete_data():
    setup = advection_setup()
    objective = advection_objective(setup, advection_data(setup, C_TRUE))
    fit = secant_fit(objective, 0.8, bounds=BOUNDS_C)
    assert relative_error(fit["theta"], C_TRUE) < 1.0e-9
    assert fit["objective"] < 1.0e-24
    assert fit["iterations"] <= 10


def test_diffusivity_identified_from_discrete_data():
    setup = heat_setup()
    objective = heat_objective(setup, heat_data(setup, ALPHA_TRUE))
    fit = secant_fit(objective, 1.0, bounds=BOUNDS_ALPHA)
    assert relative_error(fit["theta"], ALPHA_TRUE) < 1.0e-6
    assert fit["objective"] < 1.0e-20


def test_secant_history_reduces_the_gradient():
    """Every accepted step must reduce |dJ/dtheta| once inside the basin."""
    setup = advection_setup()
    objective = advection_objective(setup, advection_data(setup, C_TRUE))
    history = secant_fit(objective, 0.8, bounds=BOUNDS_C)["history"]
    assert len(history) >= 3
    gradients = [abs(entry["gradient"]) for entry in history]
    assert gradients[-1] < gradients[0]
    objectives = [entry["objective"] for entry in history]
    assert objectives[-1] <= objectives[0]


def test_discrete_identification_study_matches_tests():
    study = study_discrete_identification()
    assert study["advection_speed"]["relative_error"] < 1.0e-9
    assert study["heat_diffusivity"]["relative_error"] < 1.0e-6


# ---------------------------------------------------------------------------
# 3. consistency against continuum data (mesh refinement)
# ---------------------------------------------------------------------------

#: Measured order of the identified-parameter error on the meshes below is
#: 3.89 (advection, P2) and 4.71 (heat, P2).  The thresholds are not cosmetic:
#: a forward-physics defect (e.g. a flipped upwind interface flux, which is
#: *invisible* to every self-consistent test in this file) drops the advection
#: order to 2.08 and leaves a 1.3e-4 parameter error on the finest mesh, i.e.
#: both assertions below go red.  See the tutorial's fault-injection matrix.
REFINEMENT_MESHES = (8, 16, 32, 48)
EXPECTED_ORDER = {"advection_speed": 3.2, "heat_diffusivity": 4.0}
FINEST_ERROR_GATE = 1.0e-6


def test_continuum_identification_error_shrinks_with_refinement():
    study = study_continuum_consistency(meshes=REFINEMENT_MESHES)
    for name in ("advection_speed", "heat_diffusivity"):
        errors = [row["relative_error"] for row in study[name]["rows"]]
        assert errors[-1] < errors[0] / 10.0, (name, errors)
        assert errors[-1] < FINEST_ERROR_GATE, (name, errors)
        order = study[name]["observed_order"]
        assert order is not None and order > EXPECTED_ORDER[name], (name, order)


def test_continuum_estimate_error_is_below_three_percent():
    setup = advection_setup(ncells=16)
    adv_fit = secant_fit(
        advection_objective(setup, advection_continuum_data(setup, C_TRUE)), 0.8,
        bounds=BOUNDS_C,
    )
    heat = heat_setup(ncells=16)
    heat_fit = secant_fit(
        heat_objective(heat, heat_continuum_data(heat, ALPHA_TRUE)), 1.0,
        bounds=BOUNDS_ALPHA,
    )
    assert relative_error(adv_fit["theta"], C_TRUE) < 3.0e-2
    assert relative_error(heat_fit["theta"], ALPHA_TRUE) < 3.0e-2


# ---------------------------------------------------------------------------
# 4. teeth: noise response and the frozen-dt failure mode
# ---------------------------------------------------------------------------

def test_noise_shifts_the_estimate_but_stays_bounded():
    study = study_noise(level=1.0e-2)
    assert study["advection_speed"]["relative_error"] < 3.0e-2
    assert study["heat_diffusivity"]["relative_error"] < 3.0e-2
    # the perturbation must actually move the estimate, otherwise this control
    # would be vacuous
    assert study["advection_speed"]["relative_error"] > 0.0


def test_teeth_a_one_per_mille_gradient_perturbation_is_detected():
    """A 1e-3 relative error in the forward model must break the FD agreement."""
    setup = advection_setup()
    objective = advection_objective(setup, advection_data(setup, C_TRUE))
    _, grad = value_and_grad(objective, 0.8)
    reference = central_difference(objective, 0.8, eps=1.0e-6)
    assert relative_error(grad, reference) < 1.0e-7
    shifted = value_and_grad(objective, 0.8 * (1.0 + 1.0e-3))[1]
    assert relative_error(shifted, grad) > 1.0e-4


def test_teeth_ascending_the_gradient_increases_the_misfit():
    setup = advection_setup()
    objective = advection_objective(setup, advection_data(setup, C_TRUE))
    value, grad = value_and_grad(objective, 0.8)
    upward = value_and_grad(objective, 0.8 + 0.02 * math.copysign(1.0, grad))[0]
    assert upward > value


def test_frozen_dt_degeneracy_is_reproduced():
    """Perfect residual with a 37.5 % wrong speed -- the documented trap."""
    study = study_frozen_dt_degeneracy(c_true=1.6)
    frozen = study["frozen_dt_model"]
    assert frozen["relative_error"] > 0.3
    assert frozen["residual_looks_converged"]
    # the trap is not a numerical accident: c_hat = c_true * dt_true / dt_frozen
    assert frozen["estimate"] == pytest.approx(frozen["predicted_by_time_rescaling"], rel=1.0e-9)
    assert frozen["relative_error"] == pytest.approx(
        1.0 - study["dt_used_for_measurements"] / study["dt_frozen"], rel=1.0e-6,
    )
    matched = study["matched_dt_model"]
    assert matched["relative_error"] < 1.0e-9


def test_noise_helper_is_reproducible():
    setup = advection_setup()
    data = advection_data(setup, C_TRUE)
    first = add_relative_noise(data, 1.0e-3)
    second = add_relative_noise(data, 1.0e-3)
    assert torch.equal(first, second)
    assert not torch.equal(first, data)