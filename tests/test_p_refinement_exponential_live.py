"""Live (recomputing) gates for CASE-DG-PREFINEMENT-EXPONENTIAL.

Motivation
----------
``tests/test_p_refinement_exponential.py`` asserts *only* over the stored
artifact ``p_refinement_results.json``: 12 gates, ~0.14 s, zero solver calls.
That makes the published PASS self-certifying — a stale or hand-edited JSON
would keep the gate green while the solver it is supposed to certify could have
regressed.  This module closes the loop: it re-runs the shipped
``tensordg.advection_1d.solve_advection`` at the published configuration
(N=16 cells, cfl=0.08, p=1..3 — the cheap window where the stored numbers are
reproduced to ~1e-9) and re-measures the error through an independent NumPy
quadrature path, then checks the stored artifact against it.

It also carries the *falsification control* of the metric
``tensordg.convergence.p_decay_diagnostics``: the predeclared criterion
``sigma > 0.5`` is deliberately weak — an algebraic p^-2 error law also clears
it (see ``docs/AUDIT-m16499-CASE-DG-PREFINEMENT-EXPONENTIAL.md``).  The
discriminating criterion used here is ``sigma > 1`` *and* non-decreasing
pairwise decay rates, which an algebraic law cannot satisfy.

Runtime: ~10 s (p=1..3, RK4, temporally-converged cfl; see the cluster evidence
in ``docs/benchmarks/results/p-refinement-exponential/independent_check.json``).

Run::
    pytest -q tests/test_p_refinement_exponential_live.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.p_refinement_independent_check import (
    falsification_control,
    norms_independent,
    sha256_of,
)
from tensordg.advection_1d import l2_error, solve_advection
from tensordg.convergence import p_decay_diagnostics

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "docs" / "benchmarks" / "results" / "p-refinement-exponential" / "p_refinement_results.json"
INDEPENDENT = ARTIFACT.parent / "independent_check.json"

NCELLS = 16
CFL = 0.08
FINAL_TIME = 1.0
LIVE_DEGREES = (1, 2, 3)


@pytest.fixture(scope="module")
def live_runs():
    """Recompute p=1..3 with the shipped solver (module-scoped: ~10 s)."""
    torch.set_num_threads(1)
    runs = {}
    for degree in LIVE_DEGREES:
        centers, coeff, meta = solve_advection(
            ncells=NCELLS, degree=degree, final_time=FINAL_TIME,
            speed=1.0, cfl=CFL, time_integrator="rk4",
        )
        runs[degree] = {
            "centers": centers,
            "coeff": coeff,
            "meta": meta,
            "library_l2": l2_error(centers, coeff, FINAL_TIME, speed=1.0),
            "independent_l2": norms_independent(centers, coeff, FINAL_TIME, nq=64)["l2"],
        }
    return runs


def test_solver_runs_are_finite_and_positive(live_runs):
    for degree, run in live_runs.items():
        assert run["meta"]["steps"] > 0
        assert math.isfinite(run["library_l2"]) and run["library_l2"] > 0.0, degree
        assert math.isfinite(run["independent_l2"]) and run["independent_l2"] > 0.0, degree


def test_live_errors_decrease_monotonically(live_runs):
    errors = [live_runs[p]["library_l2"] for p in LIVE_DEGREES]
    assert all(b < a for a, b in zip(errors, errors[1:])), errors


def test_live_decay_rate_meets_predeclared_threshold(live_runs):
    errors = [live_runs[p]["library_l2"] for p in LIVE_DEGREES]
    diag = p_decay_diagnostics(LIVE_DEGREES, errors)
    assert diag["lsq_decay_rate"] > 0.5, diag


def test_live_decay_rate_is_in_the_exponential_band(live_runs):
    """Discriminating criterion: sigma > 1 and pairwise decay rates not decreasing.

    An algebraic error law has *decreasing* pairwise decay rates (see
    ``test_algebraic_law_fails_the_discriminating_criterion``), whereas spectral
    convergence shows rates that grow with p.
    """
    errors = [live_runs[p]["library_l2"] for p in LIVE_DEGREES]
    diag = p_decay_diagnostics(LIVE_DEGREES, errors)
    assert diag["lsq_decay_rate"] > 1.0, diag
    rates = diag["decay_rates"]
    assert all(b >= a for a, b in zip(rates, rates[1:])), rates
    assert diag["monotone_decreasing"] is True


def test_live_error_is_quadrature_independent(live_runs):
    """Two independent quadrature paths must agree far below the discretisation error."""
    for degree, run in live_runs.items():
        lib = run["library_l2"]
        ind = norms_independent(run["centers"], run["coeff"], FINAL_TIME, nq=64)["l2"]
        assert abs(lib - ind) / lib < 1e-9, (degree, lib, ind)
        # same solution re-measured at a second independent quadrature order
        e32 = norms_independent(run["centers"], run["coeff"], FINAL_TIME, nq=32)["l2"]
        assert abs(e32 - ind) / ind < 1e-9, (degree, e32, ind)


def test_live_matches_stored_artifact(live_runs):
    """The stored JSON must be reproducible from the current solver."""
    if not ARTIFACT.exists():
        pytest.skip("p_refinement_results.json missing")
    art = json.loads(ARTIFACT.read_text())
    stored = dict(zip(art["results"]["degrees"], art["results"]["errors"]))
    for degree, run in live_runs.items():
        assert degree in stored, degree
        gap = abs(stored[degree] - run["library_l2"]) / run["library_l2"]
        assert gap < 1e-6, (degree, stored[degree], run["library_l2"], gap)


def test_stored_artifact_diagnostics_consistent(live_runs):
    """Stored diagnostics must be what the metric gives for the stored errors."""
    if not ARTIFACT.exists():
        pytest.skip("p_refinement_results.json missing")
    art = json.loads(ARTIFACT.read_text())
    diag = p_decay_diagnostics(tuple(art["results"]["degrees"]), art["results"]["errors"])
    for key in ("lsq_decay_rate", "monotone_decreasing", "superlinear"):
        assert diag[key] == art["results"]["diagnostics"][key], key
    assert art["checks"]["decay_rate"]["pass"] == (diag["lsq_decay_rate"] > 0.5)


def test_artifact_figure_paths_are_relocatable():
    """Provenance hygiene: artifact paths must not point into a foreign worktree."""
    if not ARTIFACT.exists():
        pytest.skip("p_refinement_results.json missing")
    art = json.loads(ARTIFACT.read_text())
    for name, path in art.get("figures", {}).items():
        assert not str(path).startswith("/root/tp-agent/work/"), (name, path)


def test_metric_rejects_non_decaying_sequence():
    control = falsification_control(LIVE_DEGREES)
    const = control["constant"]
    assert const["lsq_decay_rate"] <= 0.5
    assert const["monotone_decreasing"] is False
    assert const["passes_predeclared_sigma_gt_0.5"] is False


def test_algebraic_law_fails_the_discriminating_criterion():
    """Known weakness of the predeclared threshold, pinned as a regression guard.

    ``sigma > 0.5`` alone accepts an algebraic p^-1/p^-2 error law, so the
    published PASS must be read together with the grown pairwise decay rates
    (exponential signature), not as a bare sigma threshold.
    """
    control = falsification_control(LIVE_DEGREES)
    for name in ("algebraic_p^-1", "algebraic_p^-2"):
        entry = control[name]
        # passes the (weak) predeclared gate ...
        assert entry["passes_predeclared_sigma_gt_0.5"] is True, name
        # ... but is rejected by the discriminating criterion
        rates = p_decay_diagnostics(
            LIVE_DEGREES,
            [math.exp(-entry["lsq_decay_rate"] * k) for k in range(len(LIVE_DEGREES))],
        )["decay_rates"]  # sanity: a true exponential has flat rates
        assert all(abs(b - a) / a < 0.35 for a, b in zip(rates, rates[1:])), rates
    # the algebraic sequences themselves must not be smooth-less: recompute the
    # metric on the actual algebraic data and require decreasing pairwise rates.
    degrees = LIVE_DEGREES
    p = np.asarray(degrees, dtype=float)
    p = p - p[0]
    for power in (1, 2):
        seq = [float(1.0 / (k + 1.0) ** power) for k in p]
        diag = p_decay_diagnostics(degrees, seq)
        assert all(b <= a for a, b in zip(diag["decay_rates"], diag["decay_rates"][1:])), (
            power, diag["decay_rates"],
        )


def test_independent_check_artifact_provenance():
    """If the independent-check artifact is present it must hash the primary JSON."""
    if not INDEPENDENT.exists() or not ARTIFACT.exists():
        pytest.skip("independent_check.json not generated")
    ind = json.loads(INDEPENDENT.read_text())
    assert ind["case"] == "CASE-DG-PREFINEMENT-EXPONENTIAL"
    assert ind["artifact_provenance"]["primary_sha256"] == sha256_of(ARTIFACT)
    # p=6 sits ~1e-6 relative above the float64 difference floor (the L2 error is
    # 2.2e-11 while (u_h - u_exact) is a difference of O(1) numbers), so the
    # acceptance band is 1e-5 -- still 5 orders below the discretisation signal.
    worst_q = max(lv["quadrature_spread_rel"] for lv in ind["levels"])
    assert worst_q < 1e-5, worst_q