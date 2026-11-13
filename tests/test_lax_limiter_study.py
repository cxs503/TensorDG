"""Guards for the CASE-LAX-DG mechanism-contrast study.

The study itself costs CPU-minutes, so it is not run here.  What *is* pinned:

* the mechanism registry maps onto limiter kinds that ``apply_limiter``
  actually knows (a typo would silently run the unlimited path);
* the oscillation metric separates "smeared but monotone" from "rippling" on
  synthetic fields with a known answer;
* the residual windows sit inside the domain and are ordered;
* the C1 reference-definition diagnostic keeps reporting the offset between
  the analytic characteristic edge and the locator crossing (this is the
  finding that stops ``C1`` from being quoted as a 0.2 % wave-edge error);
* the gates used here are the pool's own functions, and they still return
  FAIL for a ladder whose analytic-region order misses the requirement.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from benchmarks import lax_case_pool
from benchmarks.lax_limiter_study import (
    MECHANISMS,
    LADDER,
    _order,
    mechanism_gates,
    oscillation_metrics,
    reference_definition_diagnostic,
    residual_windows,
    run_leg,
)
from benchmarks.lax_case_pool import exact_waves
from tensordg.limiters import _LIMITERS


def test_every_mechanism_names_a_known_limiter():
    """A typo in the registry would silently run the unlimited path."""
    for name, config in MECHANISMS.items():
        kind = config["limiter"]
        assert kind is None or kind in _LIMITERS, f"{name}: unknown limiter {kind!r}"
        assert config["limiter_stage"] in ("every", "final")


def test_ladder_covers_the_failing_degrees_only():
    degrees = {degree for _, degree in LADDER}
    assert degrees == {2, 3}
    for degree in degrees:
        cells = sorted(c for c, p in LADDER if p == degree)
        assert cells == [100, 200, 400]


def test_residual_windows_are_ordered_and_inside_the_domain():
    windows = residual_windows(exact_waves())
    order = ["rarefaction_fan", "star_left", "star_right", "undisturbed_right"]
    previous = 0.0
    for name in order:
        lo, hi = windows[name]
        assert 0.0 < lo < hi < 1.0, name
        assert lo > previous, f"{name} overlaps the previous window"
        previous = hi


def test_oscillation_metrics_are_clean_for_the_exact_field():
    waves = exact_waves()
    points = np.linspace(0.0, 1.0, 2001)
    from benchmarks.lax_case_pool import exact_reference

    field = exact_reference(points[None, :])["density"][0]
    metrics = oscillation_metrics(points[None, :], field[None, :],
                                  field[None, :], waves)
    for name, row in metrics.items():
        assert row["max_abs"] == pytest.approx(0.0, abs=1e-12), name
        assert row["sign_flips"] == pytest.approx(0.0, abs=1e-12), name


def test_oscillation_metrics_detect_ripple_and_bias():
    """A single-cell-scale ripple must show up as flips, a shift as bias."""
    waves = exact_waves()
    points = np.linspace(0.0, 1.0, 4001)
    from benchmarks.lax_case_pool import exact_reference

    field = exact_reference(points[None, :])["density"][0]
    ripple = field.copy()
    ripple += 1e-3 * np.where(np.arange(points.size) % 2 == 0, 1.0, -1.0)
    metrics = oscillation_metrics(points[None, :], ripple[None, :],
                                  field[None, :], waves)
    fan = metrics["rarefaction_fan"]
    assert fan["max_abs"] == pytest.approx(1e-3, rel=1e-9)
    assert fan["sign_flips"] > 0.9
    assert abs(fan["bias"]) < 1e-6

    shifted = field + 2e-3
    bias_metrics = oscillation_metrics(points[None, :], shifted[None, :],
                                       field[None, :], waves)
    assert bias_metrics["star_left"]["bias"] == pytest.approx(2e-3, rel=1e-9)
    assert bias_metrics["star_left"]["sign_flips"] == pytest.approx(0.0, abs=1e-12)


def test_reference_definition_keeps_the_fan_offset_finding():
    diagnostic = reference_definition_diagnostic()
    rows = diagnostic["rows"]
    fan = [name for name in ("rarefaction_head", "rarefaction_tail")]
    for name in fan:
        assert rows[name]["normalised_offset"] > diagnostic["tolerance_of_gate_c1"], (
            f"{name}: locator crossing sits inside the fan at "
            f"{rows[name]['normalised_offset']:.4%}, so C1 cannot be quoted as a "
            "wave-edge error")
    for name in ("contact", "shock"):
        assert rows[name]["normalised_offset"] < 1e-3


def test_gates_are_the_pool_gates():
    assert mechanism_gates.__defaults__ is None  # no hidden tolerance override
    assert lax_case_pool.gate_c3.__module__ == "benchmarks.lax_case_pool"
    # a synthetic ladder with p+1 order in the analytic region must PASS R3
    entries = []
    for cells in (100, 200, 400):
        error = 1e-3 * (cells / 100.0) ** -(3.0)
        entries.append({"mechanism": "synthetic", "cells": cells, "degree": 2,
                        "errors": {"density": error, "velocity": error,
                                   "pressure": error},
                        "analytic_errors": {"density": error, "velocity": error,
                                            "pressure": error}})
    gates = mechanism_gates(entries)
    assert gates["r3"]["status"] == "PASS"          # 3.0 >= 0.85 * (2 + 1) = 2.55
    # ... and a first-order analytic region must FAIL
    entries = []
    for cells in (100, 200, 400):
        error = 1e-3 * (cells / 100.0) ** -0.5
        entries.append({"mechanism": "synthetic", "cells": cells, "degree": 2,
                        "errors": {"density": error, "velocity": error,
                                   "pressure": error},
                        "analytic_errors": {"density": error, "velocity": error,
                                            "pressure": error}})
    assert mechanism_gates(entries)["r3"]["status"] == "FAIL"
    assert mechanism_gates(entries)["c3"]["status"] == "FAIL"


def test_run_leg_records_the_active_mechanism():
    """One tiny leg: the metadata must name the mechanism that actually ran."""
    entry, fields = run_leg(24, 1, "tvb-final")
    assert entry["mechanism"] == "tvb-final"
    assert entry["limiter_reported"] == "tvb"
    assert entry["limiter_stage"] == "final"
    assert entry["steps"] > 0
    assert math.isfinite(entry["errors"]["density"])
    assert np.isfinite(fields["points"]).all()
    assert torch.isfinite(torch.as_tensor(entry["density_min"]))
    unlimited, _ = run_leg(24, 1, "none")
    assert unlimited["limiter_reported"] != "tvb"


def test_order_helper_matches_hand_computation():
    assert _order(100, 1e-2, 200, 2.5e-3) == pytest.approx(2.0)
    assert _order(100, 0.0, 200, 1e-3) is None