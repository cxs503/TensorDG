"""Tests for the ice pressure-area exponent parameter library (GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-CALIB)."""
import math

import pytest

from tensordg.ice_pressure_area_library import (
    AREA_DEF_NORMAL_PROJECTION,
    AREA_DEF_WETTED,
    FIRST_HAND_SOURCES,
    PressureAreaSource,
    area_conversion_factor,
    calibrate_ex,
    convert_pressure_area,
    ex_environment_table,
    stress_bandwidth,
)


def test_at_least_two_first_hand_sources():
    """The library must hold >= 2 first-hand sources, each with an external_anchor."""
    assert len(FIRST_HAND_SOURCES) >= 2
    for src in FIRST_HAND_SOURCES:
        assert src.external_anchor  # provenance present
        assert src.ex < 0           # pressure-area exponent is negative
        assert src.units            # units recorded


def test_ex_environment_table_has_three_env_levels():
    """The ex-vs-environment table must cover >= 3 environment levels (sources)."""
    table = ex_environment_table()
    assert len(table) >= 3
    for row in table:
        for k in ("strain_rate_s", "temp_C", "grain_mm"):
            assert k in row


def test_area_conversion_identity_and_obliquity():
    """Area-definition conversion: identity at same convention, cos-law at obliquity."""
    assert area_conversion_factor(AREA_DEF_NORMAL_PROJECTION,
                                  AREA_DEF_NORMAL_PROJECTION) == 1.0
    assert area_conversion_factor(AREA_DEF_WETTED, AREA_DEF_WETTED) == 1.0
    # wetted -> normal at 30 deg obliquity: A_norm = A_wetted * cos(30)
    assert math.isclose(area_conversion_factor(AREA_DEF_WETTED,
                                               AREA_DEF_NORMAL_PROJECTION, 30.0),
                        math.cos(math.radians(30.0)))
    # normal -> wetted is the reciprocal
    assert math.isclose(area_conversion_factor(AREA_DEF_NORMAL_PROJECTION,
                                               AREA_DEF_WETTED, 30.0),
                        1.0 / math.cos(math.radians(30.0)))


def test_convert_pressure_area_preserves_exponent():
    """Changing area convention must not change ex (the law's independent variable)."""
    src = FIRST_HAND_SOURCES[0]
    conv = convert_pressure_area(src, AREA_DEF_WETTED, obliquity_deg=30.0)
    assert conv["ex"] == src.ex
    # C_R rescaled by f^ex ; f<1 and ex<0 => C_R_to < C_R_from
    assert conv["C_R_MPa"] < src.C_R_MPa


def test_calibrate_ex_returns_interval():
    """Calibration must return ex as an interval, never a bare point."""
    cal = calibrate_ex()
    assert "ex_interval" in cal
    lo, hi = cal["ex_interval"]
    assert lo < hi
    assert cal["n_sources"] >= 2
    assert cal["external_anchors"]


def test_calibrate_ex_env_filter():
    """Environment filter must narrow the source set and stay consistent."""
    # high strain-rate regime (>= 1e-3) -> only masterson_frederking1993
    cal = calibrate_ex(env_filter={"strain_rate_s": (1e-3, 1e-2)})
    assert cal["n_sources"] == 1
    assert cal["source_ids"] == ["masterson_frederking1993"]


def test_stress_bandwidth_collapses():
    """The full-library ex interval must give a finite, citable bandwidth."""
    cal = calibrate_ex()
    bw = stress_bandwidth(tuple(cal["ex_interval"]), area_ratio=10.0)
    assert bw["bandwidth"] > 1.0
    # The declared scanning range 0.3-0.7 (as e=-ex in 0.3..0.7 => ex in -0.7..-0.3)
    # gives a much wider bandwidth than the first-hand library; the library must
    # be narrower than that declared range.
    declared = stress_bandwidth((-0.7, -0.3), area_ratio=10.0)
    assert bw["bandwidth"] < declared["bandwidth"]
