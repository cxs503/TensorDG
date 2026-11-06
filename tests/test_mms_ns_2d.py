"""Tests for :mod:`tensordg.mms_ns_2d` (symbolically derived NS source term).

The module previously had no dedicated test file.  The positive cases verify the
generated source against an *independent* residual evaluation: automatic
differentiation of ``S = dU/dt + div F_inv - div F_visc`` with the
``ns_dg_2d`` sign convention (unit gas constant, ``gamma = 1.4``).  Autodiff is
exact up to round-off, so the agreement can be demanded at the 1e-12 level --
much tighter than the finite-difference self-check in
``benchmarks/verify_ns_mms_order.py`` (1e-5).

The negative cases are controls: the residual must react to a wrong ``gamma``,
wrong transport coefficients, or a mismatched evaluation time, otherwise the
agreement above would be vacuous.
"""
from __future__ import annotations

import pytest
import torch

from tensordg.mms_ns_2d import (
    CP,
    CV,
    GAMMA,
    mms_conserved,
    mms_primitives,
    mms_source,
)

MU = 0.01
PRANDTL = 0.71
KAPPA = MU * CP / PRANDTL
AD_TOL = 1e-12


def _dtensor(values):
    return torch.as_tensor(values, dtype=torch.float64)


def sample_points(count=12, seed=4708):
    """Reproducible interior points, with a per-point time (see _residual)."""
    generator = torch.Generator().manual_seed(seed)
    x = 0.05 + 0.9 * torch.rand(count, generator=generator, dtype=torch.float64)
    y = 0.05 + 0.9 * torch.rand(count, generator=generator, dtype=torch.float64)
    t = torch.full_like(x, 0.13)
    return x, y, t


def _residual(x, y, t, mu, kappa, gamma=GAMMA):
    """Autodiff evaluation of ``dU/dt + div F_inv - div F_visc``.

    ``t`` is a *per-point* tensor on purpose: a 0-dimensional time would make
    ``autograd`` return the sum of the time derivatives over the batch, which
    silently corrupts the comparison.
    """
    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    state = list(mms_conserved(x, y, t))
    rho, mx, my, energy = state
    u = mx / rho
    v = my / rho
    pressure = (gamma - 1.0) * (energy - 0.5 * (mx * mx + my * my) / rho)
    temperature = pressure / rho

    def grad(field, wrt):
        return torch.autograd.grad(field.sum(), wrt, create_graph=True)[0]

    u_x, u_y = grad(u, x), grad(u, y)
    v_x, v_y = grad(v, x), grad(v, y)
    divergence = u_x + v_y
    tau_xx = 2.0 * mu * u_x - (2.0 / 3.0) * mu * divergence
    tau_yy = 2.0 * mu * v_y - (2.0 / 3.0) * mu * divergence
    tau_xy = mu * (u_y + v_x)

    inviscid_x = [mx, mx * u + pressure, mx * v, u * (energy + pressure)]
    inviscid_y = [my, my * u, my * v + pressure, v * (energy + pressure)]
    viscous_x = [
        0.0 * u, tau_xx, tau_xy,
        u * tau_xx + v * tau_xy + kappa * grad(temperature, x),
    ]
    viscous_y = [
        0.0 * u, tau_xy, tau_yy,
        u * tau_xy + v * tau_yy + kappa * grad(temperature, y),
    ]

    components = []
    for i in range(4):
        components.append(
            grad(state[i], t)
            + grad(inviscid_x[i], x) + grad(inviscid_y[i], y)
            - grad(viscous_x[i], x) - grad(viscous_y[i], y)
        )
    return torch.stack(components, dim=-1)


def source_at(x, y, t, mu=MU, kappa=KAPPA):
    return torch.stack(list(mms_source(x, y, t, mu, kappa)), dim=-1)


def relative_gap(reference, candidate):
    scale = float(reference.abs().max())
    assert scale > 0.0
    return float((reference - candidate).abs().max()) / scale


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("mu", "kappa"),
    [(MU, KAPPA), (0.0, 0.0), (3.0 * MU, 5.0 * KAPPA)],
)
def test_generated_source_matches_the_autodiff_residual(mu, kappa):
    x, y, t = sample_points()
    analytic = source_at(x, y, t, mu, kappa)
    numeric = _residual(x, y, t, mu, kappa)
    assert relative_gap(analytic, numeric) < AD_TOL


def test_conserved_state_is_consistent_with_the_primitives():
    x, y, _ = sample_points()
    rho, u, v, p = mms_primitives(x, y, _dtensor(0.37))
    conserved = mms_conserved(x, y, _dtensor(0.37))
    assert float((conserved[0] - rho).abs().max()) == pytest.approx(0.0, abs=1e-15)
    assert float((conserved[1] - rho * u).abs().max()) < 1e-14
    assert float((conserved[2] - rho * v).abs().max()) < 1e-14
    energy = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    assert float((conserved[3] - energy).abs().max()) < 1e-13


def test_fields_are_unit_periodic_in_both_directions():
    x, y, t = sample_points()
    for shift_x, shift_y in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        primitives = mms_primitives(x, y, t)
        primitives_shifted = mms_primitives(x + shift_x, y + shift_y, t)
        for a, b in zip(primitives, primitives_shifted):
            assert float((a - b).abs().max()) < 1e-13
        assert relative_gap(source_at(x, y, t), source_at(x + shift_x, y + shift_y, t)) < 1e-12


def test_vectorised_evaluation_equals_pointwise_evaluation():
    """The generated expressions must be elementwise (no cross-point coupling)."""
    x, y, t = sample_points(count=5)
    batched = source_at(x, y, t)
    for index in range(x.numel()):
        single = source_at(x[index: index + 1], y[index: index + 1], t[index: index + 1])
        assert float((batched[index] - single[0]).abs().max()) < 1e-14


def test_gas_constants_match_the_unit_gas_constant_convention():
    assert GAMMA == pytest.approx(1.4)
    assert CV == pytest.approx(1.0 / (GAMMA - 1.0))
    assert CP == pytest.approx(GAMMA / (GAMMA - 1.0))
    # R = cp - cv = 1: the pressure law used by ns_dg_2d.
    assert CP - CV == pytest.approx(1.0)
    assert KAPPA == pytest.approx(MU * CP / PRANDTL)


# ---------------------------------------------------------------------------
# Negative cases (they are red if the source or the residual degrades)
# ---------------------------------------------------------------------------

def test_wrong_gamma_is_detected():
    x, y, t = sample_points()
    analytic = source_at(x, y, t)
    wrong = _residual(x, y, t, MU, KAPPA, gamma=1.3)
    assert relative_gap(analytic, wrong) > 1e-2


@pytest.mark.parametrize(
    ("mu", "kappa", "floor"),
    [(MU, 2.0 * KAPPA, 1e-2), (3.0 * MU, KAPPA, 1e-2), (-MU, KAPPA, 1e-3)],
)
def test_wrong_transport_coefficients_are_detected(mu, kappa, floor):
    x, y, t = sample_points()
    analytic = source_at(x, y, t)
    assert relative_gap(analytic, _residual(x, y, t, mu, kappa)) > floor


def test_mismatched_evaluation_time_is_detected():
    x, y, t = sample_points()
    analytic = source_at(x, y, t)
    late = _residual(x, y, t + 0.1, MU, KAPPA)
    assert relative_gap(analytic, late) > 1e-2


def test_source_depends_on_the_transport_coefficients():
    """A source free of mu/kappa would make the MMS case useless for NS."""
    x, y, t = sample_points()
    base = source_at(x, y, t)
    perturbed = source_at(x, y, t, 2.0 * MU, 2.0 * KAPPA)
    assert relative_gap(base, perturbed) > 1e-6