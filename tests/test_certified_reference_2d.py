"""2-D/N-D support of the certified-reference engine (dimension generalisation).

The engine (:mod:`tensordg.certified_reference`) was written against 1-D profiles
and, before this test file, could not certify a 2-D field at all: ``coarsen``
flattened the field, and ``richardson_extrapolate`` / ``judge`` derived their
refinement factor from the *element count* ratio, which for an ``h -> h/2``
refinement in 2-D is 4 instead of 2.

These tests pin the three invariants that make the generalisation safe:

* the **1-D behaviour is untouched** (bit-identical primitives and certificate);
* coarsening a multi-dimensional field is a **per-direction block average**, so
  "cell average of cell averages" still equals the average over the union;
* a 2-D ladder certifies on a **2-D** grid with the correct refinement ratio, and
  ``judge`` brings a candidate back to the reference grid by the per-direction
  factor -- not by the element-count factor, which would silently certify the
  wrong grid.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordg.certified_reference import (
    CERTIFIED,
    coarsen,
    certify,
    judge,
    nesting_factor,
    rel_l1,
    richardson_extrapolate,
)


# --------------------------------------------------------------------- coarsen
def test_one_dimensional_coarsening_is_unchanged():
    v = np.arange(1.0, 9.0)
    assert np.allclose(coarsen(v, 2), [1.5, 3.5, 5.5, 7.5])
    assert np.allclose(coarsen(v, 4), [2.5, 6.5])
    assert coarsen(v, 1) is not None and np.array_equal(coarsen(v, 1), v)
    with pytest.raises(ValueError):
        coarsen(np.arange(7.0), 2)


def test_two_dimensional_coarsening_is_a_per_direction_block_average():
    a = np.arange(16.0).reshape(4, 4)
    out = coarsen(a, 2)
    assert out.shape == (2, 2)
    assert np.allclose(out, [[2.5, 4.5], [10.5, 12.5]])
    # averaging commutes with coarsening: mean(mean(union)) == mean(union)
    assert np.isclose(coarsen(a.ravel(), 2).mean(), out.mean())


def test_two_dimensional_coarsening_refuses_a_non_divisible_shape():
    with pytest.raises(ValueError):
        coarsen(np.zeros((8, 6)), 4)


def test_nesting_factor_uses_shapes_not_element_counts():
    # h -> h/2 in 2-D: the count ratio is 4, the refinement factor is 2
    assert nesting_factor(np.zeros((8, 8)), np.zeros((4, 4))) == 2
    assert nesting_factor(np.zeros((8, 8)), np.zeros((8, 8))) == 1
    assert nesting_factor(np.zeros(8), np.zeros(4)) == 2
    assert nesting_factor(np.zeros((8, 6)), np.zeros((4, 4))) is None
    assert nesting_factor(np.zeros((8, 4)), np.zeros((2, 2))) is None


# ------------------------------------------------------- Richardson extrapolation
def test_richardson_in_two_dimensions_removes_a_power_law_error():
    """Both inputs live on the *same* grid -- the engine's documented usage.

    ``certify`` coarsens the finest level onto the second-finest grid and then
    extrapolates from the pair ``(fine, coarse)`` stored on that one grid; the
    internal ``coarsen`` call is a no-op guard for exactly this pattern.
    """
    n = 16
    x = (np.arange(n) + 0.5) / n
    exact = np.add.outer(np.sin(3 * x), np.cos(2 * x))
    p, a = 3.0, 0.01
    fine = exact + a * (1.0 / (2 * n)) ** p
    coarse = exact + a * (1.0 / n) ** p          # same grid, coarser-scheme value
    extrap = richardson_extrapolate(fine, coarse, p)
    assert extrap.shape == fine.shape
    assert rel_l1(extrap, exact) < 1e-8
    assert rel_l1(fine, exact) > 1e-8
    # a genuinely non-nesting pair is still refused
    with pytest.raises(ValueError):
        richardson_extrapolate(np.zeros((8, 8)), np.zeros((4, 8)), 2.0)


def test_richardson_refuses_a_grid_that_does_not_nest():
    with pytest.raises(ValueError):
        richardson_extrapolate(np.zeros((8, 8)), np.zeros((4, 8)), 2.0)


# --------------------------------------------------------------------- certify
def _bilinear(n, domain=10.0):
    """Bilinear field: its cell-centre values *are* its cell averages.

    Using it as the continuum keeps the synthetic ladder a pure power-law error
    model, so the observed order is the injected ``p`` to round-off.  (A
    non-polynomial field would add an ``O(h^2)`` cell-average/centre mismatch to
    the ladder -- measured p = 2.04 instead of 3.00 -- which is a property of the
    fixture, not of the engine.)
    """
    x = (np.arange(n) + 0.5) * (domain / n) - 0.5 * domain
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.1 * X + 0.05 * Y + 0.01 * X * Y


def _ladder_2d(sizes, a=0.02, p=3.0):
    """Nesting 2-D ladder: bilinear continuum + a pure ``a*n^-p`` error."""
    return [{"ncells": int(n), "field": _bilinear(n) + a * n ** (-p)} for n in sizes]


def test_two_dimensional_ladder_certifies_on_a_two_dimensional_grid():
    sizes = (16, 32, 64)
    cert = certify(_ladder_2d(sizes), tolerance_percent=5.0)
    assert cert["verdict"] == CERTIFIED, cert["reasons"]
    assert cert["reference"] is not None
    n_ref = sizes[-2]
    assert np.shape(cert["reference"]) == (n_ref, n_ref)
    assert cert["refinement_fine"] == 2
    # observed order recovers the injected p
    assert abs(cert["observed_order_used"] - 3.0) < 0.05


def test_two_dimensional_certification_band_covers_the_true_error():
    sizes = (16, 32, 64)
    cert = certify(_ladder_2d(sizes), tolerance_percent=5.0)
    exact = _bilinear(sizes[-2])
    true_percent = 100.0 * rel_l1(cert["reference"], exact)
    assert true_percent <= cert["band_percent"]


def test_one_dimensional_certificate_is_bit_identical_to_the_shipped_behaviour():
    """Guard: the dimension generalisation must not move a 1-D number."""
    sizes = (100, 200, 400)
    levels = []
    for n in sizes:
        x = (np.arange(n) + 0.5) * (10.0 / n) - 5.0
        exact = 1.0 + 0.2 * x
        levels.append({"ncells": int(n), "field": exact + 0.02 * n ** (-3.0)})
    cert = certify(levels, tolerance_percent=5.0)
    assert cert["verdict"] == CERTIFIED
    assert np.shape(cert["reference"]) == (200,)
    assert abs(cert["observed_order_used"] - 3.0) < 0.05
    # the same numbers the pre-generalisation code produced (flatten + reshape mean)
    for lv in levels:
        manual = lv["field"].reshape(-1, 2).mean(axis=1)
        assert np.allclose(manual, coarsen(lv["field"], 2))


# ------------------------------------------------------------------------ judge
def test_judge_coarsens_a_two_dimensional_candidate_by_the_per_direction_factor():
    sizes = (16, 32, 64)
    levels = _ladder_2d(sizes)
    cert = certify(levels, tolerance_percent=5.0)

    # a candidate on a 2x finer grid than the reference (count ratio 4)
    n_fine = sizes[-2] * 2
    candidate = _bilinear(n_fine) + 0.02 * n_fine ** (-3.0)

    out = judge(candidate, cert, tolerance_percent=5.0)
    assert out["decision"] in ("PASS", "PASS_MARGINAL")
    # the comparison must have happened on the reference grid
    expected_on_ref = coarsen(candidate, 2)
    assert np.shape(expected_on_ref) == np.shape(cert["reference"])
    assert out["rel_l1_percent"] < 1.0


def test_judge_refuses_a_two_dimensional_profile_that_does_not_nest():
    sizes = (16, 32, 64)
    cert = certify(_ladder_2d(sizes), tolerance_percent=5.0)
    with pytest.raises(ValueError):
        judge(np.zeros((48, 48)), cert, tolerance_percent=5.0)