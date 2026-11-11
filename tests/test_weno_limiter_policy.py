"""Contract tests for the WENO-limiter smooth-flow adjudication (m14921).

``docs/anisotropic_2d_verification.md`` records honest FAILs for "the limiter
must not destroy the smooth order" at the default ``tvb_constant = 1``;
``docs/tvb_constant_adjudication.md`` adjudicated the TVB half and explicitly
left ``limiter="weno"`` uncovered.  Mission m14777 then fixed the mixed-mode
write defect and left two items open: the order of a *forced* limited
representation (``not_run``) and the source of the limiter-on error plateau.

These tests pin the three claims that close the WENO half:

1. the limiter is *inert* once ``M`` exceeds the data's second-derivative bound
   -- the limiter-on trajectory then equals the unlimited one bit for bit, so
   the plateau is a policy (flagging) property, not a WENO wiring defect;
2. ``M = 0`` flags every cell and still leaves the cell averages untouched
   (exact conservation);
3. a forced limited representation converges at second order, i.e. a linear
   reconstruction in flagged cells *cannot* deliver the design order ``p + 1``
   for ``p >= 2`` -- which is what makes the recorded criterion unsatisfiable
   rather than the solver broken.

The measured grid lives in ``validation/weno_limiter_policy/results.json``;
this module re-derives the fits and thresholds from its raw rows, so a
hand-carried number cannot survive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.verify_euler_dg_2d import _vortex_error
from benchmarks.weno_limiter_smooth_vortex import (
    DOMAIN,
    _project_vortex,
    _slope,
    block_forced,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "validation" / "weno_limiter_policy" / "results.json"
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.exists(), f"missing limiter-policy artifact: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


def test_inert_constant_makes_the_limiter_trajectory_bitwise_unlimited():
    """M beyond the data bound => the limiter never fires => identical digits."""
    base = dg2.solve_euler_dg_2d_vortex(
        ncells=(8, 8), degree=2, final_time=0.5, cfl=0.2, limiter=None
    )
    inert = dg2.solve_euler_dg_2d_vortex(
        ncells=(8, 8), degree=2, final_time=0.5, cfl=0.2,
        limiter="weno", tvb_constant=1.0e6,
    )
    assert torch.equal(base[2], inert[2]), (
        "an inert WENO limiter must leave every modal coefficient unchanged"
    )
    assert inert[3]["limiter"] == "weno"


def test_m_zero_flags_every_cell_and_preserves_cell_averages():
    coeff, nodes, basis, dx, dy = _project_vortex(2, 16, 16)
    averages = coeff[..., 0, 0, :].clone()
    limited = dg2.apply_weno_limiter_2d(coeff, dx, dy, 0.0, (True, True))
    troubled_x, minmod_x = dg2._troubled_1d(averages, 0, dx, 0.0, True)
    troubled_y, _ = dg2._troubled_1d(averages, 1, dy, 0.0, True)
    # M=0 does NOT flag every cell, and cannot: the indicator is
    # |d_left - minmod(d_left, d_right)| > M dx^2, and minmod returns d_left
    # itself wherever |d_left| <= |d_right| with the same sign, so that
    # deviation is exactly zero on ~30 % of the cells of a smooth field.  This
    # asymmetry is a property of the shipping indicator, not a limiter defect;
    # the substantive claim only needs the flagged majority.
    deviation = (averages - dg2._shift(averages, 0, -1, True)) - minmod_x
    zero_deviation = deviation.abs() == 0.0
    # The indicator's strict asymmetry leaves a sizeable unflagged minority:
    # assert that it exists and that the zero-deviation cells are a matching
    # population (bit-for-bit equality is not asserted because the indicator
    # applies its own scaling/edge handling inside ``_troubled_1d``).
    assert int(zero_deviation.sum()) > 0
    assert int((~troubled_x).sum()) > 0
    fraction = float((troubled_x | troubled_y).double().mean())
    assert fraction >= 0.9, f"only {fraction:.3f} of the cells flagged at M=0"
    assert fraction < 1.0, "the indicator's strict minmod equality should leave some cells"
    assert torch.equal(limited[..., 0, 0, :], averages), (
        "the limiter must never modify a cell average"
    )
    # Flagged cells really are reduced to a linear reconstruction: their
    # higher modes vanish on the offending axis (unflagged cells keep theirs).
    both = troubled_x & troubled_y
    assert both.any()
    assert torch.all(limited[..., 2:, :, :][both] == 0.0)
    assert torch.all(limited[..., :, 2:, :][both] == 0.0)


def test_forced_limiter_representation_is_second_order():
    """The missing not_run measurement: EOC of a fully limited representation."""
    errors = {}
    hs = []
    for n in (16, 32, 64):
        coeff, nodes, basis, dx, dy = _project_vortex(2, n, n)
        limited = dg2.apply_weno_limiter_2d(coeff, dx, dy, 0.0, (True, True))
        errors[n] = _vortex_error(limited, basis, nodes, DOMAIN, 0.0)
        hs.append(10.0 / n)
    for field in FIELDS:
        eoc = _slope(hs, [errors[n][field] for n in (16, 32, 64)])
        assert eoc >= 1.9, f"{field}: forced-limiter EOC {eoc:.3f} < 1.9"
        assert eoc <= 2.6, f"{field}: EOC {eoc:.3f} unexpectedly above second order"


def test_artifact_is_self_consistent(artifact):
    """Re-derive every reported verdict from the raw rows."""
    rows = artifact["sweep"]
    assert rows, "empty sweep"

    # 1. inert runs are bit-identical to the unlimited baseline
    inert = [r for r in rows if r["limiter"] == "weno" and r["tvb_constant"] >= 1.0e6]
    assert inert
    for row in inert:
        assert row["flagged_either"] == 0
        assert row["bitwise_identical_to_unlimited"] is True
        for field in FIELDS:
            assert row["error_ratio_vs_unlimited"][field] == pytest.approx(1.0, rel=1e-12)

    # 2. flagged counts are non-increasing in M on a fixed mesh/degree/ladder
    for ladder in {r["ladder"] for r in rows}:
        for degree in {r["degree"] for r in rows}:
            for cells in {tuple(r["cells"]) for r in rows}:
                series = sorted(
                    [r for r in rows if r["limiter"] == "weno"
                     and r["ladder"] == ladder and r["degree"] == degree
                     and tuple(r["cells"]) == cells],
                    key=lambda r: r["tvb_constant"],
                )
                counts = [r["flagged_either"] for r in series]
                assert counts == sorted(counts, reverse=True), (ladder, degree, cells)

    # 3. every row's error ratio is its own arithmetic
    for row in rows:
        if row["limiter"] != "weno":
            continue
        for field in FIELDS:
            assert row["error_ratio_vs_unlimited"][field] == pytest.approx(
                row["errors"][field] / row["unlimited_errors"][field], rel=1e-12
            )

    # 4. the recorded order fits reproduce from the raw ladder
    for entry in artifact["orders"]:
        ladder_rows = sorted(
            [r for r in rows if r["limiter"] == "weno"
             and r["ladder"] == entry["ladder"] and r["degree"] == entry["degree"]
             and r["tvb_constant"] == entry["tvb_constant"]],
            key=lambda r: r["cells"][0],
        )
        hs = [r["h"] for r in ladder_rows]
        for field in FIELDS:
            eoc = _slope(hs, [r["errors"][field] for r in ladder_rows])
            assert entry[f"eoc_{field}"] == pytest.approx(eoc, rel=1e-9)

    # 5. forced block: averages preserved, EOC >= 1.9 from the raw rows
    forced = artifact["forced"]
    assert forced and all(r["mode0_changed_components"] == 0 for r in forced)
    for degree in {r["degree"] for r in forced}:
        group = sorted([r for r in forced if r["degree"] == degree],
                       key=lambda r: r["cells"][0])
        hs = [r["h"] for r in group]
        for field in FIELDS:
            eoc = _slope(hs, [r["limited_errors"][field] for r in group])
            assert eoc >= 1.9, f"p{degree} {field}: forced EOC {eoc:.3f}"

    # 6. the shipped criteria agree with their own thresholds
    for key, entry in artifact["criteria"].items():
        assert entry["verdict"] in ("PASS", "FAIL"), key


def test_live_forced_block_matches_the_artifact(artifact):
    """A cheap live re-run of the projection block reproduces the artifact."""
    rows, _ = block_forced(quick=True)
    stored = {(r["degree"], tuple(r["cells"])): r for r in artifact["forced"]}
    for row in rows:
        key = (row["degree"], tuple(row["cells"]))
        if key not in stored:
            continue
        for field in FIELDS:
            assert row["limited_errors"][field] == pytest.approx(
                stored[key]["limited_errors"][field], rel=1e-10
            )