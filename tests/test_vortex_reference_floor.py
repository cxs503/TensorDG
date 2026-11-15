"""Truncation floor of the min-image periodic reference (round 7).

Round 4 replaced the free-space reference of CASE-EULER-VORTEX-ERR by the
min-image *periodic* one, which flipped a spurious 14 % "reference defect" into
sub-percent discretisation error.  The min-image fold is itself an approximation:
it keeps only the nearest lattice image and drops the neighbouring tails.  These
tests pin the size of that floor and the invariants of the lattice-sum reference
that removes it, so that "the reference is not what we are measuring" is an
assertion with a number behind it rather than a claim.

Why the floor is field-dependent: for the isentropic vortex the density
perturbation is *second* order in the Gaussian envelope (``T = 1 - c e**2``) while
the velocity perturbation is *first* order and displacement-weighted
(``u = u_inf - (theta/2pi) e * y_shift``).  A dropped tail of envelope ``e ~ 6e-6``
therefore enters density at ``O(e**2) ~ 1e-11`` and velocity at ``O(e) ~ 1e-6``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from benchmarks import case_euler_vortex_errconserve as ec
from benchmarks import vortex_reference_floor as rf
from tensordg.euler_2d import (
    minimum_images,
    solve_isentropic_vortex,
    vortex_error,
    vortex_primitive,
)

L = ec.PERIOD[0]
STRENGTH = ec.STRENGTH
BACKGROUND = ec.BACKGROUND
REPO = Path(__file__).resolve().parents[1]


def _grid(n=21, extent=5.0, dtype=torch.float64):
    xs = torch.linspace(-extent, extent, n, dtype=dtype)
    return torch.meshgrid(xs, xs, indexing="xy")


def _max_abs_diff(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# lattice-sum reference invariants
# ---------------------------------------------------------------------------

def test_images_require_a_period():
    x, y = _grid(5)
    with pytest.raises(ValueError, match="period"):
        vortex_primitive(x, y, 1.0, STRENGTH, BACKGROUND, images=2)


def test_lattice_sum_is_periodic_across_a_box_crossing():
    """t = L/|u_inf| advances the vortex by exactly one lattice vector.

    The periodic solution at t = 10 must therefore be *the same field* as at
    t = 0 -- for the lattice sum and for the min-image fold alike.  A fold or a
    lattice range that is off by one image breaks this to O(1).
    """
    x, y = _grid(41)
    for images in (None, 3):
        start = vortex_primitive(x, y, 0.0, STRENGTH, BACKGROUND, period=L,
                                 images=images)
        end = vortex_primitive(x, y, 10.0, STRENGTH, BACKGROUND, period=L,
                               images=images)
        assert _max_abs_diff(start, end) < 1e-12


def test_lattice_sum_converges_with_one_extra_ring():
    """One ring beyond :func:`minimum_images` is already machine-converged."""
    x, y = _grid(41)
    need = minimum_images(L, x, y, 10.0, BACKGROUND, margin=0)
    fine = vortex_primitive(x, y, 10.0, STRENGTH, BACKGROUND, period=L,
                            images=need + 1)
    coarse = vortex_primitive(x, y, 10.0, STRENGTH, BACKGROUND, period=L,
                              images=need)
    assert _max_abs_diff(coarse, fine) < 1e-12


def test_undersized_lattice_is_rejected_instead_of_silently_wrong():
    """At t = 10 the nearest image is two boxes away; K = 1 must not be accepted."""
    x, y = _grid(9)
    assert minimum_images(L, x, y, 10.0, BACKGROUND, margin=0) >= 2
    with pytest.raises(ValueError, match="nearest image"):
        vortex_primitive(x, y, 10.0, STRENGTH, BACKGROUND, period=L, images=1)


def test_min_image_fold_and_lattice_sum_agree_before_the_wrap():
    """Inside the first crossing the dropped tails are the only difference."""
    x, y = _grid(41)
    fold = vortex_primitive(x, y, 1.0, STRENGTH, BACKGROUND, period=L)
    lattice = vortex_primitive(x, y, 1.0, STRENGTH, BACKGROUND, period=L, images=2)
    assert _max_abs_diff(fold, lattice) < 1e-4   # measured 1.5e-5 on u, 6e-6 on rho


# ---------------------------------------------------------------------------
# the floor itself, in the case gate's metric
# ---------------------------------------------------------------------------

def test_floor_is_machine_level_in_density_and_first_order_in_velocity():
    config = {"task": "dg", "degree": 2, "cells": 32, "final_time": 1.0, "cfl": 0.2}
    floor = rf.floor_at(dict(config))["floor_relative_l2"]
    assert floor["density"] < 1e-9, floor
    assert floor["pressure"] < 1e-9, floor
    assert 1e-7 < floor["velocity_x"] < 1e-5, floor
    # the envelope argument of the module docstring: density floor ~ (u floor)^2
    assert floor["density"] < (floor["velocity_x"]) ** 2 * 10.0, floor


def test_floor_is_time_independent_but_the_discretisation_error_is_not():
    """The floor is a property of the *reference*, not of the transport time."""
    keys = (("dg", 2, 32, 1.0, 0.2), ("dg", 2, 32, 10.0, 0.2))
    floors = [rf.floor_at({"task": t, "degree": d, "cells": c, "final_time": ft,
                           "cfl": cfl})["floor_relative_l2"]["velocity_x"]
              for t, d, c, ft, cfl in keys]
    assert max(floors) / min(floors) < 1.1, floors


@pytest.mark.parametrize("task_key, upper", [
    ("dg_d2_n32_t1_cfl0.2", 0.05),      # design checkpoint: measurement-dominated
    ("dg_d2_n64_t10_cfl0.2", 0.20),     # finest checkpoint: floor is material
])
def test_floor_ratio_against_the_published_errors(task_key, upper):
    """Floor / published error, read from the committed round-6 evidence.

    The design checkpoint's error is dominated by discretisation (ratio ~1.5 %),
    the finest DG checkpoint's is not (ratio ~13 %) -- which is exactly why the
    ladder is re-measured against the lattice reference instead of trusting the
    min-image numbers at the refined end.
    """
    results = REPO / "case_euler_vortex_errconserve" / "results" / f"{task_key}.json"
    if not results.exists():  # pragma: no cover - evidence is committed
        pytest.skip(f"missing committed evidence {results}")
    config = json.loads(results.read_text())["config"]
    floor = rf.floor_at(dict(config))
    published = json.loads(results.read_text())
    assert published.get("reference_model", ec.REFERENCE_MIN_IMAGE) == ec.REFERENCE_MIN_IMAGE
    ratio = max(floor["floor_relative_l2"][name] / published["errors"][name]
                for name in rf.FIELDS)
    assert ratio < upper, (task_key, ratio)
    if task_key.startswith("dg_d2_n64"):
        # the floor is not negligible here: it must still be visible
        assert ratio > 0.05, (task_key, ratio)


# ---------------------------------------------------------------------------
# plumbing: the case gate can actually select the model
# ---------------------------------------------------------------------------

def test_reference_kwargs_select_the_lattice_model():
    lattice = ec._reference_kwargs({"reference": ec.REFERENCE_LATTICE}, 10.0)
    assert lattice["period"] == ec.PERIOD
    assert lattice["images"] == ec.reference_images(10.0)
    x, y = _grid(9)
    need = minimum_images(ec.PERIOD, x, y, 10.0, BACKGROUND, margin=0)
    assert lattice["images"] >= need
    assert ec._reference_kwargs({"reference": ec.REFERENCE_MIN_IMAGE}, 10.0) == {
        "period": ec.PERIOD}


def test_lattice_reference_moves_only_the_velocity_reading_end_to_end():
    """A solved P0 state read against both references: density/pressure unmoved."""
    x, y, fields, _ = solve_isentropic_vortex(
        ncells=24, final_time=1.0, cfl=0.4, domain=ec.DOMAIN,
        strength=STRENGTH, background_velocity=BACKGROUND)
    conservative = fields["conservative"] if isinstance(fields, dict) else fields
    fold = vortex_error(x, y, conservative, 1.0, STRENGTH, BACKGROUND,
                        period=ec.PERIOD)
    lattice = vortex_error(x, y, conservative, 1.0, STRENGTH, BACKGROUND,
                           period=ec.PERIOD, images=2)
    for name in ("density", "pressure"):
        assert abs(lattice[name]["relative_l2"] / fold[name]["relative_l2"] - 1.0) < 1e-6
    for name in ("velocity_x", "velocity_y"):
        shift = abs(lattice[name]["relative_l2"] / fold[name]["relative_l2"] - 1.0)
        assert 0.0 < shift < 1e-3


def test_aggregate_refuses_a_mixed_reference_bundle():
    records = [
        {"reference_model": ec.REFERENCE_MIN_IMAGE, "family": "f", "final_time": 1.0,
         "max_relative_l2": 1e-4, "errors": {}, "config": {}, "final": {}, "initial": {},
         "invariant_drift": {}, "entropy_drift": 0.0, "entropy_production": 0.0,
         "density_min_relative_error": 0.0, "task": "dg", "task_key": "a"},
        {"reference_model": ec.REFERENCE_LATTICE, "family": "f", "final_time": 1.0,
         "max_relative_l2": 1e-4, "errors": {}, "config": {}, "final": {}, "initial": {},
         "invariant_drift": {}, "entropy_drift": 0.0, "entropy_production": 0.0,
         "density_min_relative_error": 0.0, "task": "dg", "task_key": "b"},
    ]
    with pytest.raises(SystemExit, match="mixes reference models"):
        ec.aggregate(records, {}, Path("/tmp"))


def test_reference_images_covers_the_travel_distance():
    for final_time in (1.0, 5.0, 10.0, 20.0):
        images = ec.reference_images(final_time)
        assert images * L >= 5.0 + final_time * max(BACKGROUND)
        assert math.isfinite(images)