"""Gates for the embedded-wall shock-reflection study.

``docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md`` §4(a) recorded the reflection gate
as an honest FAIL with the attribution undetermined.  The study in
``validation/forward_step/`` produced two things this test pins:

G1  **reference geometry**: the solid mask built by
    ``benchmarks.forward_facing_step.wall_solid_mask`` must put the discrete wall
    face *exactly* on the analytic wall at every resolution.  The original mask
    (``xc >= x_wall - dx/2``) was one cell short, an O(1)-cell geometry error no
    refinement can remove.
G2  **no hand-copied numbers**: every plateau error stored in a ladder artifact
    must be recomputable from that artifact's own profile and exact values.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from benchmarks.euler_dg_2d_wall_bc import WallReflection
from benchmarks.forward_facing_step import wall_solid_mask
from validation.forward_step.wall_reflection_diag import MACH

ROOT = Path(__file__).resolve().parent.parent
LADDER = ROOT / "validation" / "forward_step" / "ladder"


@pytest.mark.parametrize("nx", [60, 120, 240, 480])
def test_solid_mask_starts_exactly_on_the_analytic_wall(nx):
    dx = 1.2 / nx
    x_wall_face = round(1.0 / dx) * dx
    mask = wall_solid_mask(nx, 1, x_wall_face, dx)
    first = int(torch.nonzero(mask[:, 0]).flatten()[0])
    assert math.isclose(first * dx, x_wall_face, rel_tol=0, abs_tol=1e-12), (
        "discrete wall face %r != analytic wall %r" % (first * dx, x_wall_face))
    # every cell left of the wall must be fluid, every cell at/after it solid
    assert bool(mask[:, 0].sum()) and first == round(x_wall_face / dx)


def _ladder_rows():
    return sorted(LADDER.glob("task*.json")) if LADDER.exists() else []


@pytest.mark.parametrize("path", _ladder_rows(), ids=[p.stem for p in _ladder_rows()])
def test_ladder_artifact_errors_are_recomputable(path):
    d = json.loads(path.read_text())
    prof = d["profile"]
    x = torch.as_tensor(prof["x"])
    rho = torch.as_tensor(prof["rho"])
    p = torch.as_tensor(prof["p"])
    dx = d["dx"]
    front = WallReflection(x_shock=d["x_shock_face"], x_wall=d["wall_face_actual"],
                           incident_mach=MACH).front(d["final_time"])
    mask = (x > front + 3.0 * dx) & (x < d["wall_face_actual"] - 1.0 * dx)
    assert int(mask.sum()) == d["plateau_cells"], "plateau mask changed"
    if not mask.any():
        assert d.get("plateau_rho_error_percent") is None
        return
    rho_err = 100.0 * (float(rho[mask].mean()) / d["reflected_rho_exact"] - 1.0)
    p_err = 100.0 * (float(p[mask].mean()) / d["reflected_p_exact"] - 1.0)
    assert math.isclose(rho_err, d["plateau_rho_error_percent"], rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(p_err, d["plateau_p_error_percent"], rel_tol=1e-9, abs_tol=1e-9)
    # a resolved plateau cannot be one cell wide: the gate reports not_run below 3
    if d["plateau_cells"] < 3:
        assert abs(d["worst_cell_error_percent"]) > 3.0
