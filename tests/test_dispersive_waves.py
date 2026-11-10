"""Verification tests for :mod:`tensordg.dispersive` (node P165W3-N05).

Gates asserted here:

* E3 dispersive operator reproduces ``omega^2 = g k tanh(k h)`` within 3 % for
  ``k in [0.1, 5]`` (measured from the actual numerical solution *and* from the
  model kernels);
* E4 the KdV solitary wave keeps its ``sech^2`` shape after propagating ten
  soliton widths (relative L2 <= 5 %).

The classical Boussinesq kernel is included as an *honest negative result*: it
leaves the 3 % band at ``kh ~ 1.56``, which is the documented validity range of
that model rather than a defect of the implementation.
"""
from __future__ import annotations

import math

import torch

from tensordg import dispersive as ds

G = ds.GRAVITY
PERIODS = 4.0


def test_airy_kernel_reproduces_closed_form_dispersion():
    kh = torch.linspace(0.0, 8.0, 257, dtype=torch.float64)
    omega_sq = ds.dispersion_omega_sq(kh, 1.0, G, "airy")
    reference = G * kh * torch.tanh(kh)
    assert float(torch.max(torch.abs(omega_sq - reference))) < 1e-12
    # and the omega-level identity omega^2 = g k tanh(kh)
    k = kh[1:]
    assert float(torch.max(torch.abs(ds.airy_omega(k, 1.0, G) ** 2
                                     - G * k * torch.tanh(k)))) < 1e-12


def test_pade66_model_meets_criterion_e3():
    """A higher-order Pade (Boussinesq-class) kernel stays inside 3 % on [0.1, 5]."""
    k = torch.linspace(0.1, 5.0, 400, dtype=torch.float64)
    error = ds.model_dispersion_error(k, 1.0, G, "pade66")
    assert error["max_relative_error_omega"] < 0.03, error["max_relative_error_omega"]
    assert error["max_relative_error_omega_sq"] < 0.03, error["max_relative_error_omega_sq"]
    assert ds.validity_limit(model="pade66", tolerance=0.03) > 5.0


def test_classical_boussinesq_leaves_the_band():
    """Honest negative result: the classical kernel is only valid for kh < ~1.6."""
    k = torch.linspace(0.1, 5.0, 400, dtype=torch.float64)
    error = ds.model_dispersion_error(k, 1.0, G, "boussinesq")
    assert error["max_relative_error_omega"] > 0.03
    limit = ds.validity_limit(model="boussinesq", tolerance=0.03)
    assert 1.3 < limit < 1.8, limit
    # below the limit it must be inside the band
    inside = torch.linspace(0.1, limit * 0.95, 50, dtype=torch.float64)
    assert ds.model_dispersion_error(inside, 1.0, G, "boussinesq")[
        "max_relative_error_omega"] < 0.03


def test_measured_dispersion_matches_airy_operator():
    """E3 (numerical): measured frequency of the simulated field vs ``gk tanh(kh)``."""
    results = ds.measure_dispersion([0.1, 1.0, 5.0], depth=1.0, model="airy",
                                    periods=PERIODS, ncells=256)
    for r in results:
        assert r["relative_error_vs_airy"] < 0.03, r
        assert r["relative_error_vs_model"] < 1e-5, r


def test_measured_dispersion_of_pade66_operator():
    """The higher-order Boussinesq operator also meets 3 % when actually integrated."""
    results = ds.measure_dispersion([0.1, 1.0, 3.0, 5.0], depth=1.0, model="pade66",
                                    periods=PERIODS, ncells=256)
    for r in results:
        assert r["relative_error_vs_airy"] < 0.03, r
        assert r["relative_error_vs_model"] < 1e-5, r


def test_long_wave_phase_speed_limit():
    """Non-dispersive limit: ``c -> sqrt(g h)`` as ``kh -> 0``."""
    c_small = float(ds.airy_phase_speed(torch.tensor(1e-4, dtype=torch.float64), 1.0, G))
    assert abs(c_small - math.sqrt(G)) / math.sqrt(G) < 1e-4
    c_five = float(ds.airy_phase_speed(torch.tensor(5.0, dtype=torch.float64), 1.0, G))
    # short waves slow down: c(kh=5) / c(kh=0) = sqrt(tanh(5)/5)
    assert abs(c_five / math.sqrt(G) - math.sqrt(math.tanh(5.0) / 5.0)) < 1e-12


def test_soliton_speed_relation():
    """``c = sqrt(gh)(1 + a/2h)`` vs the physical ``sqrt(g(h+a))`` agree to <3 %."""
    for amplitude in (0.02, 0.05, 0.1):
        soliton = ds.kdv_soliton(amplitude, 1.0, G)
        assert soliton["relative_speed_difference"] < 0.03
        assert abs(soliton["width"] - math.sqrt(4.0 / (3.0 * amplitude))) < 1e-12


def test_soliton_propagation_criterion_e4():
    """E4: the ``sech^2`` wave keeps its shape over ten soliton widths."""
    result = ds.run_soliton_benchmark(amplitude=0.05, depth=1.0, widths=10.0,
                                      ncells=128)
    assert result["relative_l2"] <= 0.05, result["relative_l2"]
    assert result["distance"] == 10.0 * result["width"]


def test_soliton_amplitude_mass_and_speed_are_preserved():
    result = ds.run_soliton_benchmark(amplitude=0.05, depth=1.0, widths=10.0,
                                      ncells=128)
    assert abs(result["peak_retention"] - 1.0) < 1e-3, result["peak_retention"]
    assert result["mass_drift"] < 1e-5, result["mass_drift"]
    assert result["relative_error_speed_measured_vs_kdv"] < 0.03
    assert result["relative_error_speed_measured_vs_sqrt"] < 0.03


def test_kdv_linear_dispersion_relation():
    """Small-amplitude wave in the KdV solver: ``c = c0 (1 - (kh)^2/6)``.

    This pins the sign and the coefficient of the dispersive operator against
    the closed-form linearisation of KdV; a wrong ``1/6`` or a flipped sign
    would show up immediately.
    """
    length, ncells, mode, amplitude, final_time = 20.0, 128, 2, 1e-4, 1.0
    solver = ds.KdVSolver(length=length, ncells=ncells, depth=1.0, g=G)
    x = torch.arange(ncells, dtype=torch.float64) * solver.dx
    k = 2.0 * math.pi * mode / length
    eta0 = amplitude * torch.cos(k * x)
    phase0 = float(torch.angle(torch.fft.fft(eta0)[mode]))
    result = solver.solve(eta0, final_time)
    phase1 = float(torch.angle(torch.fft.fft(result["eta"])[mode]))
    delta = phase1 - phase0
    if delta > math.pi:
        delta -= 2.0 * math.pi
    if delta < -math.pi:
        delta += 2.0 * math.pi
    speed = -delta / (k * final_time)
    predicted = solver.c0 * (1.0 - (k * solver.depth) ** 2 / 6.0)
    assert abs(speed - predicted) / predicted < 3e-3, (speed, predicted)