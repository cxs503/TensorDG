"""Guard the hydrostatic well-balanced round-off drift-law artifact.

The drift law is a *derived* result: its constants and exponents must be
fitted from the measured per-resolution errors, never typed in by hand.  This
module re-derives every fit from ``validation/swe_thacker/hydrostatic_wb_drift.json``
with an independent implementation and checks that

* the run grid is complete and every run is a lake-at-rest well-balanced run,
* the residuals stay at float64 round-off level and grow with resolution,
* the recorded ``drift_fits`` agree with the refit (this is what catches a
  hand-carried constant),
* the markdown table and the figure shipped next to the JSON agree with it,
* one solver run reproduces the round-off level quoted in the artifact.

The expensive full study lives in
``validation/swe_thacker/run_hydrostatic_wb_drift.py``; here we only re-read
its artifact plus a single N=25/P1 spot check.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "validation" / "swe_thacker" / "hydrostatic_wb_drift.json"
TABLE = ARTIFACT.parent / "hydrostatic_wb_drift_table.md"
FIGURE = REPO_ROOT / "docs" / "figures" / "swe_wb_drift_law.png"

NCELLS = [25, 50, 100, 200]
DEGREES = [1, 2]
REFERENCE_NCELLS = 100.0


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.exists(), f"missing drift-law artifact: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


def _refit(runs: list[dict]) -> dict:
    """Independent OLS fit of eps = C * (N/N_ref)**exponent * eps_machine."""
    eps = float(np.finfo(np.float64).eps)
    n = np.array([r["ncells"] for r in runs], dtype=float)
    out = {}
    for name, field in (("eta", "eta_error_max"), ("u", "u_max")):
        values = np.array([r[field] for r in runs], dtype=float)
        exponent, log_c = np.polyfit(np.log(n), np.log(values / eps), 1)
        c = float(np.exp(log_c + exponent * np.log(REFERENCE_NCELLS)))
        out[f"C_{name}"] = c
        out[f"slope_{name}"] = float(exponent)
    return out


def test_provenance_is_recorded(artifact):
    prov = artifact["provenance"]
    for key in (
        "generated_at_utc",
        "command",
        "python",
        "torch",
        "numpy",
        "platform",
        "machine",
        "generator",
    ):
        assert prov.get(key), f"provenance key {key!r} is empty"
    # The timestamp must be parseable ISO-8601 so the artifact is traceable.
    from datetime import datetime

    datetime.fromisoformat(prov["generated_at_utc"])
    assert prov["generator"].endswith("run_hydrostatic_wb_drift.py")


def test_run_grid_is_complete(artifact):
    assert artifact["degrees"] == DEGREES
    assert artifact["ncells_list"] == NCELLS
    assert len(artifact["runs"]) == len(DEGREES) * len(NCELLS)
    seen = {(r["degree"], r["ncells"]) for r in artifact["runs"]}
    assert seen == {(d, n) for d in DEGREES for n in NCELLS}

    for run in artifact["runs"]:
        assert run["well_balanced"] is True
        assert run["reached_final_time"] is True
        assert run["bed_representation"] == "nodal_lgl_polynomial"
        assert run["steps"] > 0
        assert run["final_time"] == artifact["final_time"]


def test_well_balanced_state_holds_at_roundoff(artifact):
    """Lake at rest: η and u must stay at float64 round-off, not physical size."""
    for run in artifact["runs"]:
        label = f"P{run['degree']} N={run['ncells']}"
        assert 0.0 < run["eta_error_max"] < 1e-8, f"{label}: η drift {run['eta_error_max']:.3e}"
        assert 0.0 < run["u_max"] < 1e-7, f"{label}: spurious u {run['u_max']:.3e}"
        assert run["eta_error_l2"] <= run["eta_error_max"]
        assert run["u_l2"] <= run["u_max"]


def test_residual_grows_with_resolution(artifact):
    """Refining the mesh adds steps *and* per-step operations: drift must grow."""
    for degree in DEGREES:
        runs = sorted(
            (r for r in artifact["runs"] if r["degree"] == degree),
            key=lambda r: r["ncells"],
        )
        steps = [r["steps"] for r in runs]
        eta = [r["eta_error_max"] for r in runs]
        u = [r["u_max"] for r in runs]
        assert steps == sorted(steps) and len(set(steps)) == len(steps)
        assert all(b > a for a, b in zip(eta, eta[1:])), f"P{degree} η drift not monotone: {eta}"
        assert all(b > a for a, b in zip(u, u[1:])), f"P{degree} u drift not monotone: {u}"


def test_recorded_fits_are_derived_from_runs(artifact):
    """Every recorded constant must be reproducible from the committed runs."""
    assert set(artifact["drift_fits"]) == {f"P{d}" for d in DEGREES}
    for degree in DEGREES:
        key = f"P{degree}"
        runs = [r for r in artifact["runs"] if r["degree"] == degree]
        refit = _refit(runs)
        recorded = artifact["drift_fits"][key]
        for name in ("eta", "u"):
            assert recorded[f"slope_{name}"] == pytest.approx(
                refit[f"slope_{name}"], rel=1e-9, abs=1e-12
            ), f"{key} {name}: recorded exponent {recorded[f'slope_{name}']:.4f} " \
               f"!= refit {refit[f'slope_{name}']:.4f}"
            assert recorded[f"C_{name}"] == pytest.approx(
                refit[f"C_{name}"], rel=1e-9
            ), f"{key} {name}: recorded C {recorded[f'C_{name}']:.4g} " \
               f"!= refit {refit[f'C_{name}']:.4g}"
        assert recorded["n_points"] == len(runs)
        assert recorded["reference_ncells"] == REFERENCE_NCELLS
        assert recorded["eps_machine"] == pytest.approx(np.finfo(np.float64).eps, rel=1e-12)


def test_power_law_fit_quality_is_reported(artifact):
    """A single exponent cannot describe every series perfectly: say by how much."""
    for key, fit in artifact["drift_fits"].items():
        for name in ("eta", "u"):
            residual = fit[f"max_rel_residual_{name}"]
            assert 0.0 <= residual < 0.6, f"{key} {name}: power-law residual {residual:.2f}"
        # The drift must be super-linear in the naive "one op per cell" sense:
        # more cells also means more steps.
        assert fit["slope_eta"] > 1.0 and fit["slope_u"] > 1.0
        assert fit["eta_per_step_mean"] > 0.0 and fit["u_per_step_mean"] > 0.0


def test_markdown_table_matches_json(artifact):
    assert TABLE.exists(), f"missing drift-law table: {TABLE}"
    rows = {}
    for line in TABLE.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0] in ("P1", "P2"):
            rows[cells[0]] = cells
    assert set(rows) == {"P1", "P2"}
    for key, cells in rows.items():
        fit = artifact["drift_fits"][key]
        assert float(cells[1]) == pytest.approx(fit["C_eta"], rel=1e-2)
        assert float(cells[2]) == pytest.approx(fit["C_u"], rel=1e-2)
        assert float(cells[3]) == pytest.approx(fit["slope_eta"], abs=0.01)
        assert float(cells[4]) == pytest.approx(fit["slope_u"], abs=0.01)
        assert math.isfinite(float(cells[5])) and float(cells[5]) > 0.0


def test_summary_is_generated_from_the_data(artifact):
    """The prose summary must quote the measured numbers, not stale ones."""
    summary = artifact["summary"]
    for key in ("description", "key_finding", "practical_implication"):
        assert summary.get(key), f"summary key {key!r} is empty"
    for name, fit in artifact["drift_fits"].items():
        assert f"α={fit['slope_eta']:.2f}" in summary["key_finding"], (
            f"{name} exponent missing/stale in summary: {summary['key_finding']!r}"
        )
        assert f"β={fit['slope_u']:.2f}" in summary["key_finding"]
    worst = max(artifact["runs"], key=lambda r: r["eta_error_max"])
    assert f"{worst['eta_error_max']:.2e}" in summary["practical_implication"]


def test_figure_is_shipped(artifact):
    assert FIGURE.exists(), f"missing drift-law figure: {FIGURE}"
    # PNG magic + non-trivial payload: a stale/empty placeholder must not pass.
    payload = FIGURE.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(payload) > 10_000


def test_single_point_reproduces_roundoff_level(artifact):
    """Spot-check one artifact point against a fresh solve (N=25, P1)."""
    import torch

    from tensordg.swe_1d import reconstruct_fields, solve_swe_1d

    expected = next(
        r for r in artifact["runs"] if r["degree"] == 1 and r["ncells"] == 25
    )
    H0, L, eta0 = artifact["H0"], artifact["L"], artifact["eta0"]

    def zb(xx):
        return -H0 + (H0 / L**2) * xx**2

    _, coeff, zb_faces, meta = solve_swe_1d(
        lambda xx: eta0 - zb(xx),
        lambda xx: torch.zeros_like(xx),
        zb,
        ncells=25,
        degree=1,
        final_time=artifact["final_time"],
        domain=(-L, L),
        cfl=artifact["cfl"],
        flux_kind="hll",
        well_balanced=True,
    )
    h, u, eta = reconstruct_fields(coeff, zb_faces, zb_center=meta["zb_modes"][:, 0])
    eta_error = float((eta - eta0).abs().max())
    u_max = float(u.abs().max())

    assert meta["steps"] == expected["steps"]
    assert eta_error < 1e-12, f"fresh η drift {eta_error:.3e} is above round-off"
    assert u_max < 1e-10, f"fresh spurious u {u_max:.3e} is above round-off"
    # Same solver, same configuration: the measured level must be in the same
    # round-off decade (thread-dependent reduction order forbids bit equality).
    assert 0.1 * expected["eta_error_max"] < eta_error < 10.0 * expected["eta_error_max"]