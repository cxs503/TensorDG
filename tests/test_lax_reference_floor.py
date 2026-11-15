"""CASE-LAX-DG · reference-projection floor: does the committed evidence hold up?

These tests assert *invariants* of ``benchmarks/lax_reference_floor.py`` against the
committed artifacts (``docs/benchmarks/results/lax_reference_floor/metrics.json``
and the profiles it is derived from).  They are the machine-checkable half of
``docs/CASE-LAX-DG-reference-floor.md``: without them the decomposition would be
one more number a reader has to trust.

What is asserted (and why each matters):

1. the committed profiles really are degree-``p`` polynomials in the *conserved*
   variables -- otherwise the reconstruction (and every error number) is void;
2. the error decomposition closes: ``total^2 = solver^2 + floor^2 + cross``;
3. the cross term is *identically zero for density* (the only primitive field that
   lives in the DG space) and *non-zero for velocity/pressure* -- the honest
   caveat, measured rather than assumed;
4. the pool's own reading of the pinned leg is reproduced bit-for-bit, so this
   study grades on the same ruler as the ledger;
5. the reference floor is solver-free and decreases with N, and it is a
   co-dominant term at the pinned mesh (the study's falsifiable hypothesis);
6. the metric's quadrature convention does not flip the ledger verdict;
7. the split quadrature is saturated and the wave-split positions are not a
   hidden knob (sensitivity is linear in the nudge).
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import pytest

from benchmarks import lax_case_pool as pool
from benchmarks import lax_reference_floor as rf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "docs", "benchmarks", "results", "lax_reference_floor", "metrics.json")


@pytest.fixture(scope="module")
def metrics():
    if not os.path.exists(METRICS):
        pytest.skip("lax_reference_floor artifacts not generated")
    with open(METRICS) as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def ctx():
    return rf.case_context()


@pytest.fixture(scope="module")
def legs(metrics):
    return {(leg["cells"], leg["degree"]): leg for leg in metrics["legs"]}


# --------------------------------------------------------------------------- #
# 1. the profile artifact is what it claims to be
# --------------------------------------------------------------------------- #
def test_committed_profiles_are_degree_p_polynomials_in_conserved_variables(metrics):
    for leg in metrics["legs"]:
        residual = leg["conserved_reconstruction_residual"]
        assert residual < 1e-12, (leg["cells"], leg["degree"], residual)


def test_primitive_fields_are_not_polynomials_hence_the_cross_term():
    """The pitfall this study had to fix: u_h, p_h are rational, not polynomial."""
    data = np.load(rf.PROFILES)
    points = data["N400_p3_points"]
    numerical = data["N400_p3_numerical"]
    # primitive velocity straight from the nodes: not degree-p
    _, primitive_residual = rf.reconstruct_from_nodes(points, numerical[1], 3)
    # conserved momentum from the same nodes: degree-p to machine precision
    conserved = rf.conserved_from_primitive(numerical, 1.4)
    _, conserved_residual = rf.reconstruct_from_nodes(points, conserved[1], 3)
    assert conserved_residual < 1e-12
    assert primitive_residual > 1e-3


# --------------------------------------------------------------------------- #
# 2./3. the decomposition closes, and its caveat is real
# --------------------------------------------------------------------------- #
def test_error_decomposition_closes_on_every_leg(metrics):
    worst = max(leg["fields"][f]["decomposition_residual"]
                for leg in metrics["legs"] for f in rf.FIELDS)
    assert worst < 1e-12, worst


def test_pythagoras_is_exact_for_the_conserved_field(legs):
    for (cells, degree), leg in legs.items():
        pinned = leg["fields"]["density"]
        assert pinned["pythagoras_residual"] < 1e-12, (cells, degree, pinned["pythagoras_residual"])
        assert abs(pinned["cross_term"]) < 1e-12, (cells, degree, pinned["cross_term"])


def test_cross_term_is_nonzero_for_primitive_fields_only(legs):
    for (cells, degree), leg in legs.items():
        for field in ("velocity", "pressure"):
            assert abs(leg["fields"][field]["cross_term"]) > 0.0
    # and it stays a minority correction to the squared error it corrects
    worst = max(abs(leg["fields"][f]["cross_term"]) / leg["fields"][f]["total_exact"] ** 2
                for leg in legs.values() for f in rf.FIELDS)
    # measured, not assumed: on the coarse / low-degree legs the primitive-field
    # split is off by up to ~15 % of the squared error, which is why every claim
    # about velocity/pressure in the doc is quoted with the cross term attached.
    assert worst < 0.25, worst


# --------------------------------------------------------------------------- #
# 4. same ruler as the ledger
# --------------------------------------------------------------------------- #
def test_pool_reading_of_the_pinned_leg_is_reproduced(metrics, legs):
    with open(os.path.join(ROOT, "docs", "benchmarks", "results", "lax_case_pool", "metrics.json")) as h:
        committed = json.load(h)["acceptance"]["per_leg"]["p3.N400"]["l2"]
    leg = legs[(400, 3)]
    for field in rf.FIELDS:
        ours = leg["fields"][field]["total_quad"]
        assert ours == pytest.approx(committed[field], rel=1e-12), field
    assert metrics["verdict"]["reuse_worst_relative"] < 1e-12


# --------------------------------------------------------------------------- #
# 5. the floor leg: solver-free, monotone, co-dominant
# --------------------------------------------------------------------------- #
def test_reference_floor_is_monotone_in_n(metrics):
    cells = [int(n) for n in metrics["config"]["floor_cells"]]
    for degree in metrics["config"]["floor_degrees"]:
        table = metrics["floor_table"][str(degree)]
        values = [table[str(n)]["density"] for n in cells]
        assert all(a > b for a, b in zip(values, values[1:])), (degree, values)


def test_reference_floor_is_recomputed_without_a_solver(ctx):
    """The floor is a property of (mesh, exact solution): recompute it independently."""
    committed = rf.floor_only(400, 3, ctx)["density"]
    manual, floor_sq, norm_sq = rf.project_exact(400, 3, ctx)
    assert math.sqrt(floor_sq["density"] / norm_sq["density"]) == pytest.approx(committed, rel=1e-15)
    # and it matches the artifact
    with open(METRICS) as handle:
        artifact = json.load(handle)["floor_table"]["3"]["400"]["density"]
    assert committed == pytest.approx(artifact, rel=1e-12)


def test_hypothesis_floor_is_codominant_at_the_pinned_leg(metrics):
    hyp = metrics["hypothesis"]
    pinned = next(leg for leg in metrics["legs"] if (leg["cells"], leg["degree"]) == (400, 3))
    for field in rf.FIELDS:
        entry = pinned["fields"][field]
        ratio = entry["reference_floor"] / entry["solver_leg"]
        assert ratio == pytest.approx(hyp["ratios"][field], rel=1e-12)
    assert hyp["supported"] is True
    assert min(hyp["ratios"].values()) >= 0.25


def test_solver_leg_still_dominates_the_squared_error(metrics):
    """Honest reading: the floor is large but the solver leg is still the bigger half."""
    pinned = next(leg for leg in metrics["legs"] if (leg["cells"], leg["degree"]) == (400, 3))
    for field in rf.FIELDS:
        entry = pinned["fields"][field]
        assert entry["solver_share_of_squared_error"] > entry["floor_share_of_squared_error"]
        # the two shares do not add to one for the primitive fields: the measured
        # cross term is exactly the difference (see the module docstring)
        expected = 1.0 - entry["cross_term"] / entry["total_exact"] ** 2
        assert (entry["solver_share_of_squared_error"] + entry["floor_share_of_squared_error"]
                == pytest.approx(expected, rel=1e-12))


# --------------------------------------------------------------------------- #
# 6. the verdict is not an artefact of the quadrature convention
# --------------------------------------------------------------------------- #
def test_quadrature_convention_does_not_flip_the_ledger_verdict(metrics, legs):
    tolerance = metrics["config"]["tolerance"]
    for key in ((400, 3), (800, 3)):
        entry = legs[key]["fields"]["density"]
        assert (entry["total_quad"] >= tolerance) == (entry["total_exact"] >= tolerance), (key, entry)


def test_n800_passes_under_both_conventions(legs):
    entry = legs[(800, 3)]["fields"]["density"]
    assert entry["total_quad"] < 0.03
    assert entry["total_exact"] < 0.03


def test_quadrature_bias_is_measured_and_bounded(metrics):
    bias = metrics["verdict"]["quadrature_bias_worst"]
    assert 0.0 < bias < 0.25, bias


# --------------------------------------------------------------------------- #
# 7. quadrature / split robustness
# --------------------------------------------------------------------------- #
def test_split_quadrature_is_saturated(ctx):
    check = rf.saturation_check(200, 1, ctx)
    assert check["max_spread"] < 1e-12


def test_split_position_sensitivity_is_linear(ctx):
    readings = rf.split_sensitivity(200, 1, ctx, deltas=(1e-6, 1e-9))
    ratio = readings["1e-06"] / readings["1e-09"]
    assert 5e2 < ratio < 2e3, ratio          # linear: a 1e3 smaller nudge, a 1e3 smaller change


# --------------------------------------------------------------------------- #
# artifact hygiene
# --------------------------------------------------------------------------- #
def test_committed_digest_matches_metrics_json():
    digest_file = os.path.join(ROOT, "docs", "benchmarks", "results", "lax_reference_floor", "metrics.sha256")
    with open(digest_file) as handle:
        expected = handle.read().split()[0]
    with open(METRICS, "rb") as handle:
        actual = hashlib.sha256(handle.read()).hexdigest()
    assert actual == expected


def test_study_does_not_touch_the_ledger_verdict(metrics):
    """This study adds no numerical solution: the pinned leg keeps its FAIL."""
    tolerance = metrics["config"]["tolerance"]
    pinned = next(leg for leg in metrics["legs"] if (leg["cells"], leg["degree"]) == (400, 3))
    assert pinned["fields"]["density"]["total_quad"] >= tolerance
    assert metrics["reference_definition"]["floor_is_solver_free"] is True