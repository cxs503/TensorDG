"""Estimator-level tests for :mod:`benchmarks.vortex_phase_shape_split`.

These tests need no solver run: they build the exact vortex on a uniform grid and
mutate it in ways whose correct reading is known analytically.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.vortex_phase_shape_split import (
    reference_fields,
    relative_l2,
    shift_spectrum,
    shifted_reference,
    split_state,
    optimal_shift,
    uniform_grid,
)

PER_CELL = 8


@pytest.fixture(scope="module")
def base():
    x, y, _dx, _dy = uniform_grid(16, PER_CELL)  # 128^2 grid
    ref = reference_fields(x, y, 1.0)
    return ref, x, y


def test_relative_l2_is_zero_for_identical(base):
    ref, _x, _y = base
    assert np.allclose(relative_l2(ref, ref), 0.0, atol=1e-12)


def test_known_subcell_shift_is_recovered(base):
    ref, _x, _y = base
    for truth in ((0.13, 0.0), (0.0, -0.4), (-0.21, 0.31)):
        moved = shifted_reference(ref, (truth[0] * PER_CELL, truth[1] * PER_CELL))
        result = optimal_shift(moved, ref, PER_CELL)
        assert not result["ambiguous"]
        err = np.hypot(result["delta_cells"][0] - truth[0],
                       result["delta_cells"][1] - truth[1])
        assert err <= 1e-4, (truth, result, err)
        # a pure translation leaves no shape error behind
        split = split_state(moved, ref, PER_CELL, 1.0)
        assert split["max_shape"] <= 1e-6


def test_amplitude_mutation_reads_as_shape_not_lag(base):
    ref, _x, _y = base
    background = np.array([1.0, 1.0, 1.0, 1.0])[None, None, :]
    mutated = ref + 0.02 * (ref - background)
    split = split_state(mutated, ref, PER_CELL, 1.0)
    # the mutation is a shape change, not a translation: the lag must vanish
    assert split["lag"]["lag_cells"] <= 1e-3
    # and the shape error must reproduce its analytic value
    # 0.02 ||ref - background|| / ||ref|| (max over fields)
    prediction = 0.02 * np.sqrt(
        ((ref - background) ** 2).sum(axis=(0, 1)) / (ref ** 2).sum(axis=(0, 1))
    ).max()
    assert abs(split["max_shape"] - prediction) <= 0.02 * prediction, (
        split["max_shape"], prediction)


def test_out_of_window_shift_is_flagged(base):
    ref, _x, _y = base
    outside = shifted_reference(ref, (0.7 * PER_CELL, 0.0))
    result = optimal_shift(outside, ref, PER_CELL)
    assert result["ambiguous"] is True


def test_spectrum_peak_matches_optimal_shift(base):
    ref, _x, _y = base
    truth = (0.17, -0.23)
    moved = shifted_reference(ref, (truth[0] * PER_CELL, truth[1] * PER_CELL))
    best = optimal_shift(moved, ref, PER_CELL)
    F = shift_spectrum(moved, ref)
    n = F.shape[0]
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k, indexing="ij")
    phase = np.exp(1j * 2 * np.pi * (kx * best["delta_grid"][0]
                                     + ky * best["delta_grid"][1]) / n)
    peak_at_optimum = np.real(np.sum(F * phase))
    # perturbing the lag must lower the correlation
    phase_off = np.exp(1j * 2 * np.pi * (kx * (best["delta_grid"][0] + 0.5)
                                         + ky * best["delta_grid"][1]) / n)
    assert np.real(np.sum(F * phase_off)) < peak_at_optimum


def test_shift_estimator_is_independent_of_field_weighting(base):
    """A common translation of the four fields must be read by all of them."""
    ref, _x, _y = base
    truth = (0.09, 0.06)
    moved = np.empty_like(ref)
    for index in range(ref.shape[-1]):
        moved[..., index] = shifted_reference(ref[..., index:index + 1],
                                              (truth[0] * PER_CELL, truth[1] * PER_CELL))[..., 0]
    split = split_state(moved, ref, PER_CELL, 1.0)
    assert abs(split["lag"]["delta_cells"][0] - truth[0]) <= 1e-4
    assert abs(split["lag"]["delta_cells"][1] - truth[1]) <= 1e-4
    assert split["max_shape"] <= 1e-6