"""Regression gate: the production-mesh leg of the CKS1998 time-power closure.

m22930 (``CASE-REPRO-DG-CKS1998-TIME-POWER-CLOSURE``) certified
``GATE_VACUOUS_CERTIFIED`` on the **local n=10 mesh only** and declared the
production-mesh (n=32) measurement an open item, with the completion command
pre-registered as ``--meshes 10 32``.  m23825 executed that command (cluster
job cj4804) and committed the two-mesh artifact.

This gate fails if the artifact regresses to a single mesh, if the two meshes
disagree, if the verdict string is edited without re-running the sweep, or if
the vacuity (crossing above the admissible window) stops being *measured*.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "validation" / "cks1998_vortex_2d" / "time_power_closure.json"

#: Pre-registered constants (m22930, unchanged by m23825).
ETA_MAX = 0.05
ADMISSIBLE_CFL_MAX = 0.5
BETA = 4
ORDER_TOL = 0.10
LOCAL_MESH = 10
PRODUCTION_MESH = 32


def _payload() -> dict:
    assert ARTIFACT.exists(), (
        "time_power_closure.json missing: the production-mesh leg is "
        "unmeasured again (re-run the pre-registered command "
        "`python -m benchmarks.cks1998_vortex_time_power_closure "
        "--degree 1 --meshes 10 32`)")
    return json.loads(ARTIFACT.read_text())


def test_artifact_covers_both_meshes():
    p1 = _payload()["degrees"]["p1"]
    assert p1["meshes"] == [LOCAL_MESH, PRODUCTION_MESH], (
        f"artifact covers {p1['meshes']}: the production mesh {PRODUCTION_MESH} "
        "leg is missing -- the m22930 debt is open again")
    assert set(p1["per_mesh"]) == {str(LOCAL_MESH), str(PRODUCTION_MESH)}


def test_primary_mesh_is_the_production_mesh():
    p1 = _payload()["degrees"]["p1"]
    assert p1["primary_mesh"] == PRODUCTION_MESH, (
        "the certified configuration is the production mesh; the verdict must "
        f"be taken from it, got primary={p1['primary_mesh']}")


def test_mesh_independence_is_measured():
    p1 = _payload()["degrees"]["p1"]
    assert p1["mesh_independent"] is True, (
        f"mesh verdicts {p1['mesh_verdicts']} disagree: the vacuity claim is "
        "mesh-dependent and must not be stated as certified")
    assert len(p1["mesh_verdicts"]) == 2


def test_vacuity_is_measured_on_both_meshes():
    """The crossing must be *measured* above the admissible window on both."""
    p1 = _payload()["degrees"]["p1"]
    for mesh in (str(LOCAL_MESH), str(PRODUCTION_MESH)):
        m = p1["per_mesh"][mesh]
        full = m["power"]["full"]
        assert full is not None, f"mesh {mesh}: no full power fit"
        assert full["n_points"] == 4, (
            f"mesh {mesh}: the fit must use the four pre-registered CFLs, "
            f"got n={full['n_points']}")
        crossing = full["cfl_at_eta_max"]
        assert crossing is not None and math.isfinite(crossing)
        assert crossing > ADMISSIBLE_CFL_MAX, (
            f"mesh {mesh}: crossing CFL {crossing} is inside the admissible "
            "window -- the eta leg would be informative, re-derive the verdict")
        # every leave-one-out fit must also cross above the window
        for loo in m["power"].get("jackknife", []):
            c = loo.get("cfl_at_eta_max")
            assert c is not None and c > ADMISSIBLE_CFL_MAX, (
                f"mesh {mesh}: LOO fit {loo} crosses inside the window")


def test_eta_samples_are_admissible_and_complete():
    p1 = _payload()["degrees"]["p1"]
    for mesh in (str(LOCAL_MESH), str(PRODUCTION_MESH)):
        m = p1["per_mesh"][mesh]
        cfls = sorted(s["cfl"] for s in m["eta_samples"])
        assert all(c <= ADMISSIBLE_CFL_MAX + 1e-12 for c in cfls), (
            f"mesh {mesh}: sample outside the admissible window: {cfls}")
        # the production CFL and its beta partner must both be present
        assert any(abs(c - 0.2) < 1e-12 for c in cfls)
        assert any(abs(c - 0.2 / BETA) < 1e-12 for c in cfls)


def test_order_cross_check_is_consistent_on_both_meshes():
    p1 = _payload()["degrees"]["p1"]
    for mesh in (str(LOCAL_MESH), str(PRODUCTION_MESH)):
        m = p1["per_mesh"][mesh]
        a = m["order_agreement"]
        assert a["status"] in ("CONSISTENT", "UNRESOLVED"), (
            f"mesh {mesh}: fitted exponent and three-level order disagree "
            f"({a}) -- the instrument is not self-consistent on this mesh")
        if a["status"] == "CONSISTENT":
            assert a["relative_gap"] <= ORDER_TOL


def test_closure_verdict_matches_the_primary_mesh():
    p1 = _payload()["degrees"]["p1"]
    primary = p1["per_mesh"][str(p1["primary_mesh"])]
    base = primary["power"]["verdict"]
    assert p1["closure_verdict"].startswith(base), (
        f"closure verdict {p1['closure_verdict']!r} does not start from the "
        f"primary-mesh power verdict {base!r}")
    assert _payload()["verdicts"]["p1"] == p1["closure_verdict"]


def test_pre_registered_constants_untouched():
    payload = _payload()
    assert payload["eta_max"] == ETA_MAX
    assert payload["admissible_cfl_max"] == ADMISSIBLE_CFL_MAX
    assert payload["beta"] == BETA
    assert payload["order_tol"] == ORDER_TOL
    assert payload["production_cfl"] == 0.2
