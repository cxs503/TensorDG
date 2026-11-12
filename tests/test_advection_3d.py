"""Tests for the 3-D tensor-product DG advection module.

The reference is the closed-form translated separable field of
:mod:`tensordg.advection_3d`, evaluated by Gauss quadrature -- no stored
numeric profile is used anywhere, so a regression that only rescales the
solution cannot pass.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg.advection_3d import (
    error_3d, exact_solution_3d, mass_3d, solve_advection_3d,
)

VELOCITY = (1.0, 0.5, 0.25)


def _run(ncells: int, degree: int, final_time: float,
         velocity=VELOCITY) -> tuple[dict[str, float], float, dict]:
    x, y, z, coeff, meta = solve_advection_3d(
        ncells=ncells, degree=degree, final_time=final_time,
        velocity=velocity,
    )
    metrics = error_3d(x, y, z, coeff, final_time, velocity=velocity)
    drift = abs(mass_3d(coeff) - 1.0)
    assert torch.all(torch.isfinite(coeff))
    return metrics, drift, meta


def _eoc(coarse: float, fine: float, ratio: float) -> float:
    return math.log(coarse / fine) / math.log(ratio)


def test_mass_is_conserved_to_round_off():
    """Upwind fluxes telescope; the analytic integral of the field is 1."""
    for velocity in (VELOCITY, (-1.0, -0.5, 0.25), (1.0, 1.0, 1.0)):
        _, drift, meta = _run(4, 1, 0.1, velocity=velocity)
        assert meta["steps"] > 0
        assert drift < 1e-12, drift


def test_p1_converges_at_second_order():
    coarse, _, _ = _run(4, 1, 0.1)
    fine, _, _ = _run(8, 1, 0.1)
    order = _eoc(coarse["relative_l2"], fine["relative_l2"], 2.0)
    assert 1.7 < order < 2.3, order


def test_p2_converges_at_third_order():
    coarse, _, _ = _run(4, 2, 0.25)
    fine, _, _ = _run(8, 2, 0.25)
    order = _eoc(coarse["relative_l2"], fine["relative_l2"], 2.0)
    assert 2.55 < order < 3.45, order


def test_initial_projection_alone_has_design_order():
    """With ``final_time = 0`` the only error is the L2 projection."""
    coarse, _, meta = _run(4, 2, 0.0)
    fine, _, _ = _run(8, 2, 0.0)
    assert meta["steps"] == 0
    order = _eoc(coarse["relative_l2"], fine["relative_l2"], 2.0)
    assert 2.7 < order < 3.3, order


def test_documented_case_is_below_three_percent():
    p1, _, _ = _run(8, 1, 0.25)
    p2, _, _ = _run(8, 2, 0.25)
    assert 100.0 * p1["relative_l2"] < 3.0, p1
    assert 100.0 * p2["relative_l2"] < 3.0, p2
    # and the higher degree must actually be more accurate on the same mesh
    assert p2["relative_l2"] < 0.2 * p1["relative_l2"]


def test_negative_velocity_components_conserve_and_converge():
    coarse, drift_c, _ = _run(4, 1, 0.1, velocity=(-1.0, -0.5, -0.25))
    fine, drift_f, _ = _run(8, 1, 0.1, velocity=(-1.0, -0.5, -0.25))
    assert max(drift_c, drift_f) < 1e-12
    assert _eoc(coarse["relative_l2"], fine["relative_l2"], 2.0) > 1.7


def test_zero_velocity_leaves_the_projection_untouched():
    _, _, meta = _run(4, 2, 1.0, velocity=(0.0, 0.0, 0.0))
    assert meta["steps"] == 0 and meta["dt"] == 0.0


def test_exact_solution_has_unit_mean_and_is_a_translation():
    nodes = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)[:-1]
    x = nodes[:, None, None]
    y = nodes[None, :, None]
    z = nodes[None, None, :]
    value = exact_solution_3d(x, y, z, 0.3, VELOCITY)
    assert float(value.mean().item()) == pytest.approx(1.0, abs=2e-3)
    shifted = exact_solution_3d(
        x - VELOCITY[0] * 0.3, y - VELOCITY[1] * 0.3, z - VELOCITY[2] * 0.3,
        0.0, VELOCITY,
    )
    assert torch.allclose(value, shifted, atol=1e-12)


def test_anisotropic_mesh_and_degree_zero_are_supported():
    """P0 on a stretched hexahedral mesh: conservative and first order."""
    coarse, drift_c, meta = _run((6, 4, 3), 0, 0.05)
    fine, drift_f, _ = _run((12, 8, 6), 0, 0.05)
    assert meta["ncells"] == (6, 4, 3)
    assert max(drift_c, drift_f) < 1e-12
    order = _eoc(coarse["relative_l2"], fine["relative_l2"], 2.0)
    # P0 upwind on this coarse stretched ladder is still pre-asymptotic.
    assert 0.6 < order < 1.4, order


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ncells": 1, "degree": 1},
        {"ncells": 4, "degree": 4},
        {"ncells": 4, "degree": 1, "final_time": -1.0},
        {"ncells": 4, "degree": 1, "cfl": 0.0},
        {"ncells": 4, "degree": 1, "velocity": (1.0, 1.0)},
    ],
)
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_advection_3d(**kwargs)