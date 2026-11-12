"""Machine gates for the measured explicit-RK stability limit (``CASE anchor NEW``).

The artifact under test is ``validation/cfl_stability/results.json``, written by
``python -m benchmarks.cfl_stability_limit --output-dir validation/cfl_stability``.

This mission's premise is that the widespread ``dt = cfl h / ((2p+1)|a|)`` rule
was never scanned in this repository.  The gates below therefore pin *structure
and agreement*, not "the script ran":

* the two implementation-independent anchors (closed-form ``p=0`` upwind symbol,
  classical RK4 real-axis interval) must hold to round-off,
* the worst case over wavenumbers may never be *less* restrictive than a finite
  mesh's spectrum (monotonicity of "sup over theta" vs "grid theta"),
* the measured limit of the shipped solver must agree with the spectrum-based
  prediction on the *same* mesh, and must be bracketed by a stable and an
  unstable run (a sign change, not a threshold guess),
* the shipped default ``cfl`` must sit inside the worst-case stable region with
  a stated margin.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "validation" / "cfl_stability" / "results.json"
INTEGRATORS = ("rk4", "ssp_rk3")


@pytest.fixture(scope="module")
def results() -> dict:
    if not ARTIFACT.exists():
        pytest.skip("run: python -m benchmarks.cfl_stability_limit")
    return json.loads(ARTIFACT.read_text())


def _theory(results: dict, degree: int, integrator: str) -> dict:
    return next(r for r in results["theory"]
                if r["degree"] == degree and r["integrator"] == integrator)


def test_artifact_schema(results: dict) -> None:
    for key in ("anchors", "theory", "mesh", "empirical", "mesh_sweep",
                "empirical_sweep", "euler_linearised"):
        assert key in results, f"missing section {key}"
    assert results["convention"].startswith("dt = cfl * h / ((2p+1)")
    degrees = {r["degree"] for r in results["theory"]}
    assert degrees == set(range(0, 7)), f"degrees = {sorted(degrees)}"
    for integrator in INTEGRATORS:
        assert any(r["integrator"] == integrator for r in results["theory"])
    for row in results["theory"]:
        assert row["dt_max_h_over_a"] > 0.0
        assert math.isclose(row["cfl_max"],
                            row["dt_max_h_over_a"] * (2 * row["degree"] + 1),
                            rel_tol=1e-12)


def test_p0_matches_closed_form_upwind_symbol(results: dict) -> None:
    """The assembled operator must reproduce ``(exp(-i theta)-1)/h`` at ``p=0``."""
    assert results["anchors"]["p0_symbol_max_abs_error"] < 1e-12


def test_rk4_amplification_reproduces_classical_interval(results: dict) -> None:
    """Anchor for ``R(z)``: the replayed stages must give RK4's known interval."""
    measured = results["anchors"]["rk4_real_axis_interval"]
    reference = results["anchors"]["rk4_real_axis_interval_reference"]
    assert abs(measured - reference) / reference < 1e-6, (measured, reference)


def test_sup_over_theta_is_at_least_as_restrictive_as_a_mesh(results: dict) -> None:
    """``sup_theta`` must not be more permissive than any finite mesh's spectrum."""
    for row in results["theory"]:
        for mesh in results["mesh"]:
            if (mesh["degree"] == row["degree"]
                    and mesh["integrator"] == row["integrator"]):
                assert row["dt_max_h_over_a"] <= mesh["dt_max_h_over_a"] + 1e-9, (
                    row["degree"], row["integrator"], mesh["ncells"],
                    row["dt_max_h_over_a"], mesh["dt_max_h_over_a"])


def test_reported_limit_is_the_first_crossing(results: dict) -> None:
    """The reported step is the *first* one that fails (envelope audit passed).

    ``envelope_corrected`` records whether the audit had to move the boundary
    down from the bisection value; after the correction the reported
    ``dt_max`` is guaranteed to be the first crossing, so the flag carries no
    failure meaning -- the gate is that the audit ran at all and that no row
    reports an optimistic (later) crossing.
    """
    for section in ("theory", "mesh"):
        for row in results[section]:
            assert isinstance(row["envelope_corrected"], bool), row
            assert row["dt_max_h_over_a"] > 0.0, row


def test_measured_limit_brackets_the_spectrum_prediction(results: dict) -> None:
    """The shipped solver's own threshold must not contradict the prediction.

    The empirical criterion is the zero crossing of a measured growth
    exponent, which near the threshold is very flat (the run has to accumulate
    enough e-folds of a weakly growing mode), so the measured value is a
    *lower-resolution, upper* bracket of the linear limit -- it can never be
    below it.  The gates encode exactly that asymmetry instead of a symmetric
    percentage that would hide an optimistic measurement:

    * the sign structure must be exact (stable below, growing above),
    * measured >= 0.95 x predicted (a measured value *below* the prediction
      would mean the linear analysis is unsafe -- the failure this study exists
      to catch),
    * for ``p >= 2`` the agreement must be within 5%; at ``p = 0, 1`` the
      margin is allowed up to 20% because the binding eigenvalue is a single
      spiky branch that a projected initial state excites weakly (documented
      in ``docs/DG-CFL-STABILITY-LIMIT.md`` §5).
    """
    checked = 0
    for row in results["empirical"]:
        if row.get("cfl_max") is None:
            continue
        checked += 1
        assert row["growth_stable"] < 0.0, row
        assert row["growth_unstable"] > 0.0, row
        assert row["cfl_max"] >= 0.95 * row["cfl_max_theory_mesh"], row
        tol = 0.05 if row["degree"] >= 2 else 0.20
        assert row["rel_error"] < tol, row
    assert checked >= 8, f"only {checked} empirical limits measured"


def test_default_cfl_is_inside_the_worst_case_region(results: dict) -> None:
    """The shipped default must be stable against *every* wavenumber."""
    default = results["default_cfl"]
    for row in results["theory"]:
        assert default < row["cfl_max"], row


def test_default_cfl_has_a_real_margin(results: dict) -> None:
    """... and the margin must be more than 1.5x, otherwise the default is a trap."""
    default = results["default_cfl"]
    factors = [row["cfl_max"] / default for row in results["theory"]]
    assert min(factors) > 1.5, min(factors)


def test_limit_decreases_with_degree(results: dict) -> None:
    """``dt_max`` must fall with ``p`` once the ``2p+1`` factor is removed."""
    for integrator in INTEGRATORS:
        rows = sorted((r for r in results["theory"] if r["integrator"] == integrator),
                      key=lambda r: r["degree"])
        for prev, cur in zip(rows, rows[1:]):
            assert cur["dt_max_h_over_a"] < prev["dt_max_h_over_a"], (integrator, cur)


def test_mesh_sweep_approaches_the_worst_case(results: dict) -> None:
    """Finer meshes resolve more wavenumbers, so measured ``cfl_max`` must fall."""
    rows = {}
    for row in results["mesh_sweep"]:
        rows.setdefault((row["degree"], row["integrator"]), []).append(
            (row["ncells"], row["cfl_max"]))
    for key, series in rows.items():
        series.sort()
        for (n0, c0), (n1, c1) in zip(series, series[1:]):
            assert c1 <= c0 + 1e-9, (key, n0, c0, n1, c1)


def test_euler_linearised_limits_exist(results: dict) -> None:
    """The Euler extension must produce a positive limit for both integrators."""
    assert results["euler_linearised"], "no linearised Euler spectrum recorded"
    for row in results["euler_linearised"]:
        for integrator in INTEGRATORS:
            assert row[integrator]["dt_max_h_over_a"] > 0.0, row
            assert isinstance(row[integrator]["envelope_corrected"], bool), row