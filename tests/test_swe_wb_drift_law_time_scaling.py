"""Guard the hydrostatic well-balanced round-off drift *time*-scaling artifact.

``docs/swe-hydrostatic-roundoff-drift-law.md`` claims that the round-off
residual of a lake-at-rest run grows linearly with the integration time
(``eps(T) ~ eps(1 s) * T``).  That claim was, until this artifact, a model
rather than a measurement: the mesh study only ever integrates to
``T = 1 s``.  ``validation/swe_thacker/run_hydrostatic_wb_drift_time.json``
scans ``T`` at fixed ``(degree, ncells)`` and fits

    eps(T) = C * (T / 1 s) ** gamma

with ``gamma`` *measured*, and evaluates the two competing hypotheses
(``gamma = 1`` coherent accumulation, ``gamma = 0.5`` random walk) on the same
points.

This module re-derives every exponent from the committed runs with an
independent implementation (so a hand-carried constant is caught), checks the
markdown table and the figure against the JSON, checks that the drift really
grows with integration time, and reproduces one artifact point with a fresh
solve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "validation" / "swe_thacker" / "hydrostatic_wb_drift_time.json"
TABLE = ARTIFACT.parent / "hydrostatic_wb_drift_time_table.md"
FIGURE = REPO_ROOT / "docs" / "figures" / "swe_wb_drift_time_law.png"
MESH_ARTIFACT = ARTIFACT.parent / "hydrostatic_wb_drift.json"

TIMES = [0.5, 1.0, 2.0, 4.0]
SERIES = [(1, 50), (1, 100), (2, 50)]


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.exists(), f"missing drift time-scaling artifact: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


def _refit(runs: list[dict]) -> dict:
    """Independent log-log fit of eps = C * (T/1s)**gamma."""
    times = np.array([r["final_time"] for r in runs], dtype=float)
    out = {}
    for name, field in (("eta", "eta_error_max"), ("u", "u_max")):
        values = np.array([r[field] for r in runs], dtype=float)
        gamma, log_c = np.polyfit(np.log(times), np.log(values), 1)
        out[f"gamma_{name}"] = float(gamma)
        out[f"C_{name}"] = float(np.exp(log_c))
    return out


def test_provenance_is_recorded(artifact):
    prov = artifact["provenance"]
    for key in ("generated_at_utc", "command", "python", "torch", "numpy", "machine"):
        assert prov.get(key), f"provenance key {key!r} is empty"
    from datetime import datetime

    datetime.fromisoformat(prov["generated_at_utc"])


def test_series_grid_is_complete(artifact):
    assert (artifact["H0"], artifact["L"], artifact["eta0"]) == (2.0, 1.0, 1.0)
    assert artifact["cfl"] == pytest.approx(0.3)
    assert [(s["degree"], s["ncells"]) for s in artifact["series"]] == SERIES
    for series in artifact["series"]:
        runs = sorted(series["runs"], key=lambda r: r["final_time"])
        assert [r["final_time"] for r in runs] == TIMES
        for run in runs:
            assert run["well_balanced"] is True
            assert run["reached_final_time"] is True
            assert run["steps"] > 0
            # Lake at rest: every residual must stay at round-off level.
            assert run["eta_error_max"] < 1e-9
            assert run["u_max"] < 1e-8
            assert run["eta_error_l2"] <= run["eta_error_max"]
            assert run["u_l2"] <= run["u_max"]


def test_steps_scale_with_integration_time(artifact):
    """A longer run must actually do more work, otherwise the scan is vacuous."""
    for series in artifact["series"]:
        runs = sorted(series["runs"], key=lambda r: r["final_time"])
        steps = [r["steps"] for r in runs]
        assert steps == sorted(steps) and len(set(steps)) == len(steps)
        # Fixed CFL: steps are proportional to T, so the ratio tracks T_max/T_min.
        ratio = steps[-1] / steps[0]
        assert ratio == pytest.approx(TIMES[-1] / TIMES[0], rel=0.05)


def test_drift_grows_with_integration_time(artifact):
    """The drift must increase over an 8x longer integration window."""
    for series in artifact["series"]:
        fit = series["time_fit"]
        label = f"P{series['degree']}/N={series['ncells']}"
        assert fit["eta_growth_first_to_last"] > 1.5, (
            f"{label}: η drift grew only {fit['eta_growth_first_to_last']:.2f}x "
            f"over an 8x longer run"
        )
        assert fit["u_growth_first_to_last"] > 1.5, (
            f"{label}: u drift grew only {fit['u_growth_first_to_last']:.2f}x"
        )


def test_recorded_fits_are_derived_from_runs(artifact):
    """No hand-carried exponent: every recorded constant must be reproducible."""
    for series in artifact["series"]:
        runs = series["runs"]
        refit = _refit(runs)
        recorded = series["time_fit"]
        label = f"P{series['degree']}/N={series['ncells']}"
        for name in ("eta", "u"):
            assert recorded[f"gamma_{name}"] == pytest.approx(
                refit[f"gamma_{name}"], rel=1e-9, abs=1e-12
            ), f"{label} {name}: recorded γ {recorded[f'gamma_{name}']:.4f} " \
               f"!= refit {refit[f'gamma_{name}']:.4f}"
            assert recorded[f"C_{name}"] == pytest.approx(
                refit[f"C_{name}"], rel=1e-9
            ), f"{label} {name}: recorded C {recorded[f'C_{name}']:.4g} " \
               f"!= refit {refit[f'C_{name}']:.4g}"
        assert recorded["n_points"] == len(runs)
        assert recorded["reference_time"] == pytest.approx(1.0)
        assert recorded["eps_machine"] == pytest.approx(np.finfo(np.float64).eps, rel=1e-12)


def _forced_residual(times: np.ndarray, values: np.ndarray, gamma: float) -> tuple:
    """Best constant for a prescribed exponent, plus its max/rms relative error."""
    c = float(np.exp(np.mean(np.log(values) - gamma * np.log(times))))
    rel = np.abs(c * times**gamma - values) / values
    return c, float(np.max(rel)), float(np.sqrt(np.mean(rel**2)))


def test_both_hypotheses_are_evaluated(artifact):
    """gamma=1 and gamma=0.5 must be scored on the same points, reproducibly."""
    for series in artifact["series"]:
        fit = series["time_fit"]
        label = f"P{series['degree']}/N={series['ncells']}"
        times = np.array([r["final_time"] for r in series["runs"]], dtype=float)
        for name, field in (("eta", "eta_error_max"), ("u", "u_max")):
            values = np.array([r[field] for r in series["runs"]], dtype=float)
            for gamma in (1.0, 0.5):
                c, max_rel, rms_rel = _forced_residual(times, values, gamma)
                assert fit[f"C_{name}_gamma{gamma:g}"] == pytest.approx(c, rel=1e-9)
                assert fit[f"max_rel_residual_{name}_gamma{gamma:g}"] == pytest.approx(
                    max_rel, rel=1e-9
                ), f"{label} {name}/gamma={gamma}: recorded max residual is not derived"
                assert fit[f"rms_rel_residual_{name}_gamma{gamma:g}"] == pytest.approx(
                    rms_rel, rel=1e-9
                )
            # The measured-exponent fit is the one that must be good; the forced
            # exponents are allowed to be bad -- that is the point of the table.
            assert fit[f"max_rel_residual_{name}"] < 0.6, (
                f"{label} {name}: fitted power law departs by "
                f"{fit[f'max_rel_residual_{name}']:.2f}"
            )
        assert fit["preferred_hypothesis_eta"] in (
            "coherent (gamma=1)",
            "random_walk (gamma=0.5)",
        )
        # eta drift cannot grow faster than steps^2 by more than the fit noise
        assert 0.0 < fit["gamma_eta"] < 3.0


def test_linear_model_verdict_is_derived(artifact):
    """The 'eps ~ eps(1 s) * T' verdict must follow from the measured exponent."""
    for series in artifact["series"]:
        fit = series["time_fit"]
        label = f"P{series['degree']}/N={series['ncells']}"
        for name, field in (("eta", "eta_error_max"), ("u", "u_max")):
            expected = "consistent" if abs(fit[f"gamma_{name}"] - 1.0) <= 0.3 else "rejected"
            assert fit[f"linear_model_{name}"] == expected, (
                f"{label} {name}: verdict {fit[f'linear_model_{name}']!r} does not follow "
                f"from gamma={fit[f'gamma_{name}']:.2f}"
            )
            assert fit[f"linear_model_underestimate_factor_{name}"] > 0.0
        # The documented falsification is not allowed to silently flip back:
        # eta must not be described by a linear-in-T law in these series.
        assert fit["linear_model_eta"] == "rejected", (
            f"{label}: eta drift is now consistent with gamma=1 "
            f"({fit['gamma_eta']:.2f}); the doc's 'linear' section must be revisited"
        )
        assert "rejected" in artifact["summary"]["linear_in_T_model"], (
            "summary does not record the rejected linear-in-T model"
        )


def test_cross_check_against_mesh_artifact(artifact):
    """The T=1 s point must reproduce the mesh study's committed value."""
    assert MESH_ARTIFACT.exists()
    entries = artifact["cross_check"]
    assert len(entries) == len(SERIES), "every series must be cross-checked at T=1 s"
    mesh = json.loads(MESH_ARTIFACT.read_text())
    for entry in entries:
        twin = next(
            r
            for r in mesh["runs"]
            if r["degree"] == entry["degree"] and r["ncells"] == entry["ncells"]
        )
        assert entry["steps"] == twin["steps"], (
            f"P{entry['degree']}/N={entry['ncells']}: steps {entry['steps']} != "
            f"{twin['steps']}"
        )
        assert entry["eta_error_max_artifact"] == pytest.approx(
            twin["eta_error_max"], rel=1e-12
        )
        # Same solver and configuration: identical to round-off, at worst the
        # same round-off decade if a reduction order changes.
        assert entry["rel_diff_eta"] < 0.5, (
            f"P{entry['degree']}/N={entry['ncells']}: η {entry['eta_error_max']:.3e} "
            f"vs mesh artifact {entry['eta_error_max_artifact']:.3e}"
        )


def test_markdown_table_matches_json(artifact):
    assert TABLE.exists(), f"missing drift time-scaling table: {TABLE}"
    text = TABLE.read_text()
    for series in artifact["series"]:
        fit = series["time_fit"]
        assert f"{fit['gamma_eta']:.2f}" in text, "γ_η missing/stale in table"
        assert f"{fit['gamma_u']:.2f}" in text, "γ_u missing/stale in table"
        for run in series["runs"]:
            assert f"{run['eta_error_max']:.3e}" in text, (
                f"η error {run['eta_error_max']:.3e} for P{series['degree']}/"
                f"N={series['ncells']}/T={run['final_time']} missing from table"
            )
    # The T=1 cross-check table must carry the mesh artifact's values.
    for entry in artifact["cross_check"]:
        assert f"{entry['eta_error_max_artifact']:.3e}" in text


def test_summary_is_generated_from_the_data(artifact):
    summary = artifact["summary"]
    for key in ("description", "key_finding", "practical_implication"):
        assert summary.get(key), f"summary key {key!r} is empty"
    for series in artifact["series"]:
        fit = series["time_fit"]
        assert f"γ_η={fit['gamma_eta']:.2f}" in summary["key_finding"], (
            f"P{series['degree']}/N={series['ncells']} exponent missing/stale in "
            f"summary: {summary['key_finding']!r}"
        )


def test_figure_is_shipped(artifact):
    assert FIGURE.exists(), f"missing drift time-scaling figure: {FIGURE}"
    payload = FIGURE.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(payload) > 10_000


def test_single_point_reproduces_roundoff_level(artifact):
    """Spot-check one artifact point (P1, N=50, T=0.5 s) with a fresh solve."""
    import torch

    from tensordg.swe_1d import reconstruct_fields, solve_swe_1d

    series = next(s for s in artifact["series"] if s["degree"] == 1 and s["ncells"] == 50)
    expected = next(r for r in series["runs"] if r["final_time"] == pytest.approx(0.5))
    H0, L, eta0 = artifact["H0"], artifact["L"], artifact["eta0"]

    def zb(xx):
        return -H0 + (H0 / L**2) * xx**2

    _, coeff, zb_faces, meta = solve_swe_1d(
        lambda xx: eta0 - zb(xx),
        lambda xx: torch.zeros_like(xx),
        zb,
        ncells=50,
        degree=1,
        final_time=0.5,
        domain=(-L, L),
        cfl=artifact["cfl"],
        flux_kind="hll",
        well_balanced=True,
    )
    _, u, eta = reconstruct_fields(coeff, zb_faces, zb_center=meta["zb_modes"][:, 0])
    eta_error = float((eta - eta0).abs().max())
    u_max = float(u.abs().max())

    assert meta["steps"] == expected["steps"]
    assert eta_error < 1e-12, f"fresh η drift {eta_error:.3e} is above round-off"
    assert u_max < 1e-10, f"fresh spurious u {u_max:.3e} is above round-off"
    assert 0.1 * expected["eta_error_max"] < eta_error < 10.0 * expected["eta_error_max"]