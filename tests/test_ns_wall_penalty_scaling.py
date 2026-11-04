"""Guards for the Dirichlet wall closure of ``tensordg/ns_dg_2d.py``.

These tests pin the two facts measured by
``benchmarks/verify_ns_wall_penalty_scaling.py`` (artifact
``validation/ns_dg_2d_wall/penalty_scaling.json``):

1. **consistency** — for a state the polynomial space represents exactly
   (``rho, u, v, p = 1, y, 0, 1`` at P2/P3) the Dirichlet wall closure leaves a
   round-off residual in mass and both momentum channels, and its energy
   residual equals the physical dissipation ``mu U^2``.  Any nonzero wall
   residual reported for *non-representable* data (the analytic Brinkman
   temperature, ``rho = p/T``) is therefore representation error, not an
   inconsistent boundary operator.
2. **penalty scaling** — that representation error enters the residual through
   the SIPG penalty only, so ``max|R|`` is linear in the penalty scale with
   ``R^2 > 0.999``.  This is what makes the earlier "O(1) wall residual" numbers
   reproducible instead of mysterious.
"""
from __future__ import annotations

import torch

import tensordg.ns_dg_2d as ns
from benchmarks.verify_ns_wall_penalty_scaling import (
    MU,
    KAPPA,
    WALL_T,
    WALLS,
    _channels,
    _residual,
    representable_state,
    section_scaling,
)
from tools.diag_ns_wall import build_state

DEVICE = torch.device("cpu")
DTYPE = torch.float64
ROUNDOFF = 1e-11


def test_representable_state_has_no_wall_momentum_residual():
    """P2/P3 represent ``(1, y, 0, 1)`` exactly: momentum residual is round-off."""
    for degree in (2, 3):
        state = representable_state((2, 8), degree)
        rhs = _residual(state, WALLS, MU, KAPPA, 2.0 * (degree + 1.0) ** 2)
        channels = _channels(rhs)
        assert channels["mass"] < ROUNDOFF, (degree, channels)
        assert channels["x_momentum"] < ROUNDOFF, (degree, channels)
        assert channels["y_momentum"] < ROUNDOFF, (degree, channels)


def test_representable_state_energy_residual_is_the_dissipation():
    """With uniform ``T`` the only energy source is viscous work ``mu U^2``."""
    for degree in (2, 3):
        state = representable_state((2, 8), degree)
        rhs = _residual(state, WALLS, MU, KAPPA, 2.0 * (degree + 1.0) ** 2)
        energy = _channels(rhs)["energy"]
        assert abs(energy - MU) < 1e-9, (degree, energy)


def test_p1_cannot_hold_the_state_so_the_wall_residual_is_visible():
    """P1 stores a quadratic energy as piecewise linear: the source is O(1)."""
    state = representable_state((2, 8), 1)
    rhs = _residual(state, WALLS, 0.0, 0.0, 2.0 * (1.0 + 1.0) ** 2)
    channels = _channels(rhs)
    assert channels["y_momentum"] > 1e-3, channels
    assert channels["mass"] < ROUNDOFF, channels


def test_exact_state_residual_is_linear_in_the_penalty():
    """``R(sigma) = R(0) + k sigma`` with ``R^2 > 0.999`` at every degree."""
    _, fits = section_scaling()
    for fit in fits:
        assert fit["r2"] > 0.999, fit
        assert fit["slope"] > 0.0, fit


def test_penalty_free_wall_residual_is_the_representation_floor():
    """With ``sigma = 0`` the residual is the projection floor and drops with degree."""
    floors = {}
    for degree in (1, 2, 3):
        state = build_state((2, 8), degree)
        rhs = _residual(state, WALLS, MU, KAPPA, 0.0)
        floors[degree] = _channels(rhs)["maximum"]
    assert floors[1] > floors[2] > floors[3], floors
    assert floors[3] < 1e-3, floors