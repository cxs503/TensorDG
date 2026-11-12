"""Tests for the 2-D acoustic-pulse case (CASE-ACOUSTIC-PULSE).

Anchors, in increasing order of cost:

* the Hankel-integral exact solution reduces to the initial Gaussian at ``t=0``
  (identity, machine precision) and is converged in its quadrature;
* the large-time on-axis asymptote ``-A b^2/(2 ln2 c0^2 t^2)`` is reproduced by
  the quadrature without any solver in the loop;
* the initial data contain no entropy mode (``rho' = p'/c0^2``);
* the production 2-D DG solver reproduces the exact waveform within the
  ledger tolerance, keeps the entropy noise O(A) and the acoustic energy
  within a few percent at a documented resolution.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg.acoustics import (
    acoustic_pulse_exact_pressure,
    acoustic_pulse_initial_state,
    acoustic_pulse_on_axis_asymptote,
    acoustic_pulse_sound_speed,
    alpha_from_width,
)


WIDTH = 0.4
AMPLITUDE = 1.0e-3
C0 = acoustic_pulse_sound_speed()


def test_exact_solution_reduces_to_the_initial_gaussian():
    radius = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    values = acoustic_pulse_exact_pressure(radius, 0.0, width=WIDTH,
                                           amplitude=AMPLITUDE, sound_speed=C0)
    reference = AMPLITUDE * np.exp(-alpha_from_width(WIDTH) * radius ** 2)
    assert np.max(np.abs(values - reference)) / AMPLITUDE < 1e-12


def test_exact_solution_quadrature_is_converged():
    radius = np.array([0.0, 0.2, 0.7])
    coarse = acoustic_pulse_exact_pressure(radius, 1.5, width=WIDTH, amplitude=AMPLITUDE,
                                           sound_speed=C0, panels=90, nodes=12)
    fine = acoustic_pulse_exact_pressure(radius, 1.5, width=WIDTH, amplitude=AMPLITUDE,
                                         sound_speed=C0, panels=360, nodes=36)
    assert np.max(np.abs(coarse - fine)) / AMPLITUDE < 1e-9


def test_on_axis_asymptote_matches_the_quadrature_for_large_time():
    """``p'(0,t) -> -A b^2/(2 ln2 c0^2 t^2)``: checks sign and t^-2 decay."""
    times = np.array([10.0, 20.0])
    numeric = np.array([float(acoustic_pulse_exact_pressure(
        np.array([0.0]), t, width=WIDTH, amplitude=AMPLITUDE, sound_speed=C0)[0]) for t in times])
    asymptote = acoustic_pulse_on_axis_asymptote(times, width=WIDTH,
                                                 amplitude=AMPLITUDE, sound_speed=C0)
    assert np.all(numeric < 0.0) and np.all(asymptote < 0.0)
    relative = np.abs(numeric - asymptote) / np.abs(numeric)
    assert relative[0] < 0.03, relative        # c0 t / b ~ 30
    assert relative[1] < 0.01, relative        # c0 t / b ~ 59
    # exponent: a quartering of the amplitude when the time doubles
    ratio = numeric[1] / numeric[0]
    assert 0.22 < ratio < 0.28, ratio


def test_initial_state_has_no_entropy_mode():
    x = np.linspace(-1.0, 1.0, 21)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    rho, u, v, p = acoustic_pulse_initial_state(xx, yy, width=WIDTH, amplitude=AMPLITUDE)
    assert np.all(u == 0.0) and np.all(v == 0.0)
    assert np.max(np.abs((rho - 1.0) - (p - 1.0) / C0 ** 2)) / AMPLITUDE < 1e-12
    assert abs(float(np.max(p - 1.0)) - AMPLITUDE) / AMPLITUDE < 1e-12


def test_initial_state_accepts_torch_tensors():
    x = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, x, indexing="xy")
    rho, u, v, p = acoustic_pulse_initial_state(xx, yy, width=WIDTH, amplitude=AMPLITUDE)
    assert isinstance(rho, torch.Tensor)
    assert torch.all(u == 0) and torch.all(v == 0)


def test_dg_reproduces_the_exact_waveform_within_the_ledger_tolerance():
    """Documented resolution: 64 cells, p=2, t=0.4 -- the 8 % gate and more."""
    from benchmarks.acoustic_pulse_2d import run_leg

    leg = run_leg(32, 2, final_time=0.4, threads=1)
    assert np.isfinite(leg["rel_l2_pressure"])
    assert leg["peak_rel_error"] < 0.08, leg          # ledger tolerance
    assert leg["rel_l2_pressure"] < 0.08, leg
    assert leg["entropy_noise_relative"] < 1e-3, leg   # O(A) spurious entropy
    assert 0.75 < leg["energy_ratio"] <= 1.02, leg  # dissipative but not growing
    assert leg["phase_error_pct"] is not None and leg["phase_error_pct"] < 60.0, leg
    assert leg["zero_crossing_numeric"] == pytest.approx(leg["zero_crossing_exact"],
                                                        rel=0.5)
