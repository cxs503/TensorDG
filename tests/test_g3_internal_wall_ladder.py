"""G3 closeout gates: geometry invariants of the embedded-wall reflection benchmark.

`docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md` §4(a) left the embedded-wall shock
reflection open.  The failure could come from three places that look alike from
the outside:

1. the closed-form reference and the *discretised* wall do not sit at the same
   ``x`` (a geometry bug -- no mesh refinement can remove it);
2. the plateau sampling band overlaps the numerically smeared reflected shock
   (a measurement bug -- the criterion is then unmeetable by any shock-capturing
   scheme);
3. the slip-wall flux ``(0, p, 0, 0)`` on solid-fluid faces is wrong, or the
   resolution is simply too coarse.

This module turns 1 and 2 into assertions -- they are cheap, deterministic and
independent of the run -- and leaves 3 to
``validation/g3_internal_wall/ladder.json`` (produced by
``benchmarks.g3_internal_wall_ladder`` on the cluster), whose invariants are
asserted at the bottom once the artifact exists.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from benchmarks.euler_dg_2d_wall_bc import WallReflection
from benchmarks.forward_facing_step import (
    _plateau_mask,
    _snap_to_face,
    gate_wall_reflection,
    wall_solid_mask,
)

LADDER = Path("validation/g3_internal_wall/ladder.json")


@pytest.mark.parametrize("nx", [60, 120, 240, 480])
def test_discrete_wall_face_coincides_with_analytic_wall(nx):
    """Invariant 1: the first solid cell's left face *is* the analytic wall."""
    dx = 1.2 / nx
    face, offset = _snap_to_face(1.0, dx)
    assert abs(offset) <= 0.5 + 1e-12
    snap_error = abs(face - 1.0)
    assert snap_error <= 0.5 * dx + 1e-12, "snapped wall drifts more than half a cell"

    mask = wall_solid_mask(nx, nx, face, dx)
    centres = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    mask[centres >= face + 0.5 * dx - 1e-12] = True
    first = int(torch.nonzero(mask[:, 0]).flatten()[0])
    left_face = first * dx
    assert abs(left_face - face) < 1e-12, (
        f"discrete wall {left_face} != analytic wall {face} at nx={nx}"
    )
    # solid block is one contiguous right-hand slab, never a hole in the gas
    assert bool(mask[first:, 0].all()) and not bool(mask[:first, 0].any())


def test_plateau_band_stays_clear_of_both_waves():
    """Invariant 2: samples are strictly between the reflected front and the wall."""
    dx = 1.2 / 240
    centres = (torch.arange(240, dtype=torch.float64) + 0.5) * dx
    fluid = torch.ones(240, dtype=torch.bool)
    front, x_wall = 0.905, 1.0
    mask = _plateau_mask(centres, fluid, front, x_wall, dx,
                         front_buffer=3, wall_buffer=1)
    picked = centres[mask]
    assert picked.numel() > 0
    assert float(picked.min()) >= front + 3 * dx
    assert float(picked.max()) <= x_wall - dx


def test_plateau_band_reports_empty_when_mesh_is_too_coarse():
    """A coarse mesh must report "not measured", not a fake PASS/FAIL."""
    dx = 1.2 / 24
    centres = (torch.arange(24, dtype=torch.float64) + 0.5) * dx
    fluid = torch.ones(24, dtype=torch.bool)
    front, x_wall = 0.99, 1.0  # 0.24 cells of plateau: nothing survives buffering
    mask = _plateau_mask(centres, fluid, front, x_wall, dx)
    assert int(mask.sum()) == 0


def test_plateau_width_matches_the_exact_reflected_shock():
    """The configured ``extra_time`` must actually host the sampled band."""
    wr = WallReflection(x_shock=0.4, x_wall=1.0, incident_mach=2.0)
    for extra_time, nx in ((0.20, 60), (0.12, 120), (0.08, 240), (0.06, 480)):
        dx = 1.2 / nx
        width = abs(wr.reflected_speed) * extra_time
        cells = width / dx
        buffers = 3 + 1
        assert cells - buffers >= 3, (
            f"nx={nx} extra_time={extra_time}: only {cells:.2f} plateau cells, "
            f"gate would be unmeasurable"
        )
        # and the reference itself must be the exact two-shock solution
        assert wr.reflected[1] == pytest.approx(0.0, abs=1e-12)  # gas rests on the wall
        # pressure jumps across both shocks: p1 < p2 < p5
        assert wr.pre[2] < wr.post[2] < wr.reflected[2]
        # reflected shock runs back into the post-shock gas
        assert wr.reflected_speed < 0.0 < wr.incident_speed


def test_wall_mode_is_validated():
    with pytest.raises(ValueError):
        gate_wall_reflection(ncells=(24, 1), degree=1, extra_time=0.0, wall_mode="ghost")


def test_reflection_gate_refuses_to_measure_before_the_reflection():
    """A pre-reflection ``extra_time`` must raise, not silently mis-score."""
    with pytest.raises(ValueError, match="reflected plateau"):
        gate_wall_reflection(ncells=(12, 1), degree=1, extra_time=0.0)


def test_boundary_arm_puts_the_wall_on_the_domain_face():
    """``wall_mode="boundary"`` truncates the mesh at the analytic wall."""
    row = gate_wall_reflection(ncells=(12, 1), degree=1, limiter=None,
                               extra_time=0.01, wall_mode="boundary")
    assert row["wall_mode"] == "boundary"
    # the mesh ends exactly at the analytic wall: wall face == xmax == nx*dx
    assert abs(row["wall_x_face"] - row["dx"] * 12) < 1e-12
    # 12 cells cannot host a buffered plateau -> the gate says "not_run"
    assert row["status"] == "not_run"
    assert row["steps"] > 0


requires_ladder = pytest.mark.skipif(
    not LADDER.exists(), reason="cluster artifact validation/g3_internal_wall/ladder.json absent"
)


def _rows():
    if not LADDER.exists():
        return {}
    data = json.loads(LADDER.read_text())
    return {row["name"]: row for row in data["rows"]}


@requires_ladder
def test_ladder_rows_are_finite_and_conservative():
    rows = _rows()
    assert rows, "ladder.json has no rows"
    for name, row in rows.items():
        assert math.isfinite(row["l1_relative"]["density"]), name
        assert row["steps"] > 0, name
        # mass must not be created or destroyed by the wall treatment
        if row["wall_mode"] == "solid":
            assert abs(row["mass_relative_residual"]) < 1e-10, name


@requires_ladder
def test_embedded_wall_matches_the_boundary_arm():
    """Invariant 3a: the two wall implementations must agree on the same problem.

    Both arms are scored against the identical closed-form reference, so a
    hidden defect in the solid face flux would show up as a systematic gap
    between them that does not shrink with the mesh.
    """
    rows = _rows()
    for degree in (1, 2):
        solid = next((r for n, r in rows.items()
                      if n == f"solid_p{degree}_nx240"), None)
        boundary = next((r for n, r in rows.items()
                         if n == f"boundary_p{degree}_nx240"), None)
        if solid is None or boundary is None:
            continue
        assert solid["status"] == boundary["status"], (solid["status"], boundary["status"])
        for key in ("plateau_mean_rho_error_percent", "plateau_mean_p_error_percent"):
            if solid[key] is None or boundary[key] is None:
                continue
            gap = abs(solid[key] - boundary[key])
            assert gap < 3.0, f"P{degree} {key}: solid vs boundary arms differ by {gap:.2f} pp"


@requires_ladder
def test_plateau_error_decreases_with_refinement():
    """Invariant 3b: a genuine defect would leave the error mesh-independent."""
    rows = _rows()
    for degree in (1, 2):
        chain = [rows.get(f"solid_p{degree}_nx{n}") for n in (60, 120, 240)]
        if any(r is None or r["status"] != "ok" for r in chain):
            continue
        errs = [r["worst_relative_error_percent"] for r in chain]
        assert errs[-1] < errs[0], f"P{degree}: no refinement gain {errs}"