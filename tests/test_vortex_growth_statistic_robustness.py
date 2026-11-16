"""C12: the growth statistic C2/C7/C8/C10 read has to survive the estimator.

The point of these tests is not that the estimators exist; it is that the gate
*grades the conservative reading* and that the conservative reading is a strict
strengthening of the published one.  Every test therefore has a counter-example:
a curve whose least-squares slope is inside the threshold while another estimator
is outside it.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import case_euler_vortex_errconserve as ec

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "case_euler_vortex_errconserve"


def _records():
    return [json.loads(path.read_text())
            for path in sorted((BUNDLE / "results").glob("*.json"))]


def _gates(records):
    exact = ec.exact_invariants()
    exact["core_density"] = ec.exact_core_density()
    return ec.evaluate_gates(records, exact)["gates"]


def test_ols_variant_is_the_published_statistic():
    """C12's ``ols`` entry must be the number C2 reports, not a second opinion."""
    times = [1.0, 2.0, 5.0, 10.0]
    errors = [1.0e-3, 1.2e-3, 1.9e-3, 2.8e-3]
    assert ec.exponent_variants(times, errors)["ols"] == pytest.approx(
        ec._fit_exponent(times, errors), rel=1e-12)


def test_every_estimator_agrees_on_a_true_power_law():
    """e = 0.5 * t**0.7 has to be read as 0.7 by every estimator."""
    times = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    errors = 0.5 * times ** 0.7
    row = ec.exponent_variants(list(times), list(errors))
    for key in ("ols", "theil_sen", "endpoint", "drop_projection_floor",
                "leave_one_out_worst", "bootstrap_quantile"):
        assert row[key] == pytest.approx(0.7, abs=1e-9), key
    assert row["robust"] == pytest.approx(0.7, abs=1e-9)
    assert row["power_law"] is True


def test_late_onset_blow_up_evades_the_least_squares_slope():
    """The counter-example C12 exists for: OLS inside 1, endpoint outside.

    Flat at the projection floor for two box crossings and then unstable: the
    whole-window regression averages the jump away while the endpoint slope reads
    the full late growth.  C12 has to take the larger of the two.
    """
    times = [1.0, 2.0, 5.0, 10.0, 20.0]
    base = 1.0e-4
    errors = [base, base, base, base, base * 20.0 ** 1.1]
    row = ec.exponent_variants(times, errors)
    assert row["ols"] <= ec.MAX_GROWTH_EXPONENT
    assert row["endpoint"] > ec.MAX_GROWTH_EXPONENT
    assert row["robust"] > ec.MAX_GROWTH_EXPONENT
    assert row["robust_at"] != "ols"


def test_three_point_family_reports_undecidable_variants_not_zeros():
    """A three-point family cannot be refitted without one point: say so."""
    row = ec.exponent_variants([1.0, 2.0, 10.0], [1e-3, 1.1e-3, 1.2e-3])
    assert row["decidable"] is True
    assert np.isnan(row["drop_projection_floor"])
    assert "drop_projection_floor" in row["undecidable_variants"]
    assert "leave_one_out_worst" in row["undecidable_variants"]
    assert np.isfinite(row["bootstrap_quantile"])


def test_bootstrap_is_deterministic_and_brackets_the_fit():
    """Same seed -> same reading, and the 99% bound is above the point estimate."""
    times = [1.0, 2.0, 5.0, 10.0]
    errors = [3.1e-4, 3.6e-4, 4.4e-4, 5.2e-4]
    first = ec.exponent_variants(times, errors)
    second = ec.exponent_variants(times, errors)
    assert first["bootstrap_quantile"] == second["bootstrap_quantile"]
    assert first["bootstrap_quantile"] > first["ols"]
    assert first["bootstrap_std"] > 0.0


def test_bundle_carries_a_robust_growth_statistic():
    """The committed bundle has to pass C12, and the design family with room."""
    gates = _gates(_records())
    c12 = gates["C12_growth_statistic_robustness"]
    assert c12["status"] == "passed", c12["offenders"]
    assert c12["measured"] <= ec.MAX_GROWTH_EXPONENT
    assert len(c12["families_evaluated"]) >= ec.MIN_ROBUST_FAMILIES
    assert c12["design_family"]["robust"] >= c12["design_family"]["ols"]
    assert c12["design_family"]["strict_passed"] is True


def test_c12_refuses_a_bundle_whose_robust_statistic_crosses():
    """Inject the late-onset defect into the design family: C12 must go red."""
    records = _records()
    rows = [r for r in records
            if r["task_key"] in ("dg_d2_n32_t1_cfl0.2", "dg_d2_n32_t2_cfl0.2",
                                 "dg_d2_n32_t5_cfl0.2", "dg_d2_n32_t10_cfl0.2",
                                 "dg_d2_n32_t20_cfl0.2")]
    rows.sort(key=lambda r: r["final_time"])
    base = float(rows[0]["max_relative_l2"])
    for row in rows[:-1]:
        ec_row_errors(row, base)
    ec_row_errors(rows[-1], base * 20.0 ** 1.3)
    gates = _gates(records)
    assert gates["C12_growth_statistic_robustness"]["status"] == "failed"
    assert gates["C2_error_conservation"]["status"] == "passed"


def ec_row_errors(record, value):
    record["errors"] = {name: value for name in ec.FIELDS}
    record["max_relative_l2"] = value
    record["max_percent"] = 100.0 * value


def test_single_checkpoint_cannot_be_dropped_silently():
    """The leave-one-out reading is reported per family, not merely promised."""
    gates = _gates(_records())
    per_family = gates["C12_growth_statistic_robustness"]["per_family"]
    for name, row in per_family.items():
        assert np.isfinite(row["robust"]), name
        assert row["robust"] >= row["ols"] - 1e-12, name
        if row["leave_one_out"]:
            assert row["leave_one_out_worst"] == max(row["leave_one_out"])