"""Gates for the physical-case calibration of the certified-reference band.

These tests pin the two things ``tools/certify_physical_band.py`` claims:

1. the exact-solution machinery is really exact (analytic cell averages), so the
   measured "true error" is a truth and not another discretisation;
2. the *shipped* engine, driven by this repository's own DG ladders on physical
   PDEs with closed-form solutions, produces a band that covers the certified
   profile's true error -- and the superconvergent P3 ladders are refused by the
   documented order ceiling (a negative result about the engine, pinned so that a
   later change to ``P_MAX`` cannot silently alter it).
"""
import math

import numpy as np
import pytest
import torch

from tensordg.certified_reference import P_MAX, WITHHELD_ORDER_TOO_HIGH
from tools import certify_physical_band as cal


def test_sine_cell_averages_match_the_analytic_mean():
    n = 16
    h = 1.0 / n
    got = cal.cell_averages(lambda x: cal.exact_advection_sine(x, 0.0), n)
    xl = np.arange(n) * h
    # (1/h) * integral of sin(2 pi x) over [xl, xl+h]
    want = (np.cos(2 * math.pi * xl) - np.cos(2 * math.pi * (xl + h))) / (2 * math.pi * h)
    assert np.allclose(got, want, atol=1e-14)


def test_burgers_exact_satisfies_the_characteristic_relation():
    x = np.linspace(0.0, 1.0, 41)[:-1]
    T = 0.12
    u = cal.exact_burgers(x, T)
    implied = cal.BURGERS_MEAN + cal.BURGERS_AMP * np.sin(2 * math.pi * (x - u * T))
    assert np.max(np.abs(u - implied)) < 1e-11


def test_square_wave_cell_averages_conserve_the_wave_area():
    n, x1, x2, shift = 40, 0.4, 0.6, 0.5
    a = cal.square_wave_cell_averages(n, x1, x2, shift)
    assert a.mean() == pytest.approx(x2 - x1, abs=1e-13)
    assert np.all((a >= -1e-15) & (a <= 1.0 + 1e-15))


@pytest.fixture(scope="module")
def advection_P1():
    case = {"name": "advection_sine_P1", "kind": "advection", "degree": 1,
            "final_time": 1.0, "ladder": [16, 32, 64]}
    return cal.measure_ladder(case)


def test_band_covers_the_true_error_on_a_physical_ladder(advection_P1):
    """The whole point: the band is measured against a *known* exact solution."""
    assert advection_P1["verdict"] == "CERTIFIED"
    assert advection_P1["err_certified_percent"] is not None
    assert advection_P1["band_covers_truth"] is True
    assert advection_P1["true_over_band"] < 1.0


def test_richardson_actually_beats_the_raw_finest_level(advection_P1):
    """If extrapolation did not help, quoting the certified profile would be theatre."""
    assert advection_P1["richardson_gain"] > 1.0
    assert advection_P1["err_certified_percent"] < advection_P1["err_raw_second_finest_percent"]


def test_superconvergent_P3_ladder_is_refused_by_the_order_ceiling():
    """Documented negative result: p_max = 6 refuses a *legitimate* P3 ladder."""
    case = {"name": "advection_sine_P3", "kind": "advection", "degree": 3,
            "final_time": 1.0, "ladder": [16, 32, 64]}
    row = cal.measure_ladder(case)
    assert row["verdict"] == WITHHELD_ORDER_TOO_HIGH
    assert row["observed_order"] > P_MAX
    # refusal is a statement about the engine, not about the solution: the
    # certified profile would have been accurate to ~1e-12 relative
    assert row["gate_usable"] is False
    assert row["err_certified_percent"] < 1e-6


def test_document_is_rendered_from_the_artifact(tmp_path):
    result = cal.run(tmp_path, cases=[{"name": "advection_sine_P1", "kind": "advection",
                                       "degree": 1, "final_time": 1.0,
                                       "ladder": [16, 32, 64]}], with_probe=False)
    doc = cal.render(result, tmp_path / "doc.md")
    assert "advection_sine_P1" in doc
    assert str(result["summary"]["n_band_covers_truth"]) in doc
    assert (tmp_path / "results.json").exists()