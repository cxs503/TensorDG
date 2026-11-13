"""Contract tests for ``CASE-LAX-DG``'s p3 asymptotic probe.

Four independent layers:

1. **geometry layer** -- the analytic-region mask is *nested* by construction
   (a fixed physical band), so its retained measure must not drift with cell
   count.  This is the property whose violation invalidated the retired
   ``4*dx`` rule (m16606 F1), so it is pinned here again.
2. **estimator layer** -- the order estimator and the ripple-wavelength
   estimator are applied to synthetic fields with known answers; a diagnostic
   that cannot recover an injected 8-cell ripple must not be reported as
   measuring one.
3. **configuration layer** -- the pool ladder runs *without* a shock limiter
   (only the Zhang-Shu positivity scaling).  Several documents attributed the
   p3 order failure to the TVB limiter, so the configuration is pinned by a
   live run: if the driver ever starts limiting, this test says so.
4. **artifact layer** -- the committed ``metrics.json`` is re-adjudicated from
   its stored numbers (a hand-edited verdict cannot pass) and the published
   document must be *byte-identical* to a re-render from that metrics file, so
   no number in the prose can be typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.lax_asymptotic_probe import (
    DOC,
    ESCALATION_FACTOR,
    LADDER,
    OUT,
    PLATEAU,
    analytic_mask,
    escalate,
    measure_leg,
    order_table,
    plateau_window,
    render_doc,
    render_summary,
    ripple_diagnostics,
)
from benchmarks.lax_case_pool import (
    R3_BAND_WIDTH,
    R3_ORDER_FRACTION,
    exact_waves,
    gate_c3,
    gate_r3,
    modal_fields,
)
from benchmarks.riemann_exact import sample
from tensordg.euler_dg_1d import solve_euler_dg_1d

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / OUT / "metrics.json"
DIGEST = ROOT / OUT / "metrics.sha256"


# --------------------------------------------------------------- geometry layer
def _quadrature_points(cells, degree):
    """Gauss points of a uniform mesh, without running the solver."""
    centres = (np.arange(cells) + 0.5) / cells
    nodes, _ = np.polynomial.legendre.leggauss(10)
    dx = 1.0 / cells
    return centres[:, None] + 0.5 * dx * nodes[None, :]


def test_analytic_region_measure_does_not_drift_with_resolution():
    waves = exact_waves()
    measures = [float(analytic_mask(_quadrature_points(cells, 1), waves).mean())
                for cells in (100, 200, 400, 800)]
    # the retired 4*dx rule drifted +35% over the same range (m16606 F1)
    assert max(measures) - min(measures) < 1e-9
    assert 0.5 < measures[0] < 0.95


def test_mask_excludes_every_analytic_wave_by_the_band_width():
    waves = exact_waves()
    points = _quadrature_points(400, 1)
    mask = analytic_mask(points, waves)
    for name, position in waves["positions"].items():
        close = np.abs(points - position) <= R3_BAND_WIDTH
        assert not mask[close].any(), name


def test_plateau_window_lies_between_the_contact_and_the_shock():
    waves = exact_waves()
    points = _quadrature_points(400, 1)
    index = plateau_window(points.ravel(), waves)
    x = points.ravel()[index]
    assert x.min() > waves["positions"][PLATEAU[0]] + R3_BAND_WIDTH - 1e-12
    assert x.max() < waves["positions"][PLATEAU[1]] - R3_BAND_WIDTH + 1e-12


# -------------------------------------------------------------- estimator layer
def test_order_estimator_recovers_a_known_rate():
    orders = order_table([
        {"cells": 100, "degree": 1,
         "errors": {"density": 1.0e-2, "velocity": 1.0e-2, "pressure": 1.0e-2}},
        {"cells": 200, "degree": 1,
         "errors": {"density": 2.5e-3, "velocity": 2.5e-3, "pressure": 2.5e-3}},
        {"cells": 400, "degree": 1,
         "errors": {"density": 6.25e-4, "velocity": 6.25e-4, "pressure": 6.25e-4}},
    ], "errors")
    for pair in orders[1]["density"]:
        assert pair["order"] == pytest.approx(2.0, rel=1e-12)


def test_ripple_estimator_recovers_an_injected_wavelength():
    """A synthetic residual with 8 cells per wavelength must read back as ~8."""
    waves = exact_waves()
    cells, samples, wavelength_cells = 400, 10, 8.0
    dx = 1.0 / cells
    centres = (np.arange(cells) + 0.5) * dx
    nodes, _ = np.polynomial.legendre.leggauss(samples)
    points = (centres[:, None] + 0.5 * dx * nodes[None, :]).ravel()
    residual = 1e-4 * np.sin(2.0 * np.pi * points / (wavelength_cells * dx))
    measured = ripple_diagnostics(points, residual, waves, dx)
    assert measured["band_cells"] == pytest.approx(R3_BAND_WIDTH / dx)
    assert measured["wavelength_cells_smoothed"] == pytest.approx(
        wavelength_cells, rel=0.25)
    assert measured["max_abs_residual"] == pytest.approx(1e-4, rel=0.05)


def test_escalation_contract_flags_both_readings():
    def pairs(previous, finest):
        return {3: {"density": [
            {"coarse": 200, "fine": 400, "order": previous},
            {"coarse": 400, "fine": 800, "order": finest},
        ]}}

    rising = escalate(pairs(0.2, 1.0), 3)
    assert rising["escalates"] is True
    assert rising["escalation_factor"] == pytest.approx(5.0)
    stalled = escalate(pairs(0.3, 0.3), 3)
    assert stalled["escalates"] is False
    assert stalled["meets_requirement"] is False
    assert stalled["requirement"] == pytest.approx(R3_ORDER_FRACTION * 4.0)
    assert escalate({}, 3)["available"] is False


# ---------------------------------------------------------- configuration layer
def test_pool_configuration_runs_without_a_shock_limiter():
    """Pin the measured configuration: no TVB/WENO limiter, positivity only."""
    waves = exact_waves()
    leg, _ = measure_leg(40, 1, waves)
    assert leg["limiter_reported"] == "local positivity fallback"
    assert leg["cfl"] == pytest.approx(0.35)
    assert leg["density_min"] > 0.0 and leg["pressure_min"] > 0.0


def test_probe_ladder_actually_extends_the_committed_pool_ladder():
    pool = json.loads((ROOT / "docs/benchmarks/results/lax_case_pool/metrics.json").read_text())
    committed = {(entry["cells"], entry["degree"]) for entry in pool["ladder"]}
    extension = {(cells, degree) for cells, degree in LADDER} - committed
    assert extension, "the probe must add at least one new leg"
    assert (800, 3) in extension, "the decisive p3 leg is N=800"


# --------------------------------------------------------------- artifact layer
@pytest.mark.skipif(not METRICS.exists(), reason="probe artifact not generated yet")
def test_committed_verdict_is_re_adjudicated_from_the_stored_numbers():
    payload = json.loads(METRICS.read_text())
    entries = payload["ladder"]
    c3 = gate_c3(entries)
    r3 = gate_r3(entries, band={"width": R3_BAND_WIDTH,
                                "rule": "fixed physical band (nested)"})
    assert payload["gates"]["c3"]["status"] == c3["status"]
    assert payload["gates"]["r3"]["status"] == r3["status"]
    expected = "PASS" if c3["status"] == "PASS" and r3["status"] == "PASS" else "FAIL"
    assert payload["verdict"] == expected
    for leg in entries:
        assert leg["density_min"] > 0.0
        assert leg["pressure_min"] > 0.0
        assert all(value > 0.0 for value in leg["errors"].values())
        assert all(value > 0.0 for value in leg["analytic_errors"].values())


@pytest.mark.skipif(not METRICS.exists(), reason="probe artifact not generated yet")
def test_stored_orders_are_reproducible_from_the_stored_errors():
    payload = json.loads(METRICS.read_text())
    recomputed = order_table(payload["ladder"], "analytic_errors")
    for degree, fields in payload["analytic_orders"].items():
        for pair, stored in zip(fields["density"],
                                recomputed[int(degree)]["density"]):
            if stored["order"] is None:
                assert pair["order"] is None
            else:
                assert pair["order"] == pytest.approx(stored["order"], rel=1e-12)


@pytest.mark.skipif(not METRICS.exists(), reason="probe artifact not generated yet")
def test_analytic_region_measure_is_constant_across_the_stored_ladder():
    payload = json.loads(METRICS.read_text())
    for degree in {leg["degree"] for leg in payload["ladder"]}:
        measured = [leg["analytic_fraction"] for leg in payload["ladder"]
                    if leg["degree"] == degree]
        assert max(measured) - min(measured) < 2e-2, (degree, measured)


@pytest.mark.skipif(not METRICS.exists(), reason="probe artifact not generated yet")
def test_published_document_is_a_re_render_of_the_metrics():
    """Anti-hand-copy gate: the prose must be exactly what the renderer emits."""
    payload = json.loads(METRICS.read_text())
    digest = DIGEST.read_text().split()[0] if DIGEST.exists() else None
    doc = ROOT / DOC
    assert doc.exists(), DOC
    assert doc.read_text() == render_doc(payload, digest)
    summary = ROOT / OUT / "summary.md"
    assert summary.read_text().strip() == render_summary(payload, digest).strip()


@pytest.mark.skipif(not METRICS.exists(), reason="probe artifact not generated yet")
def test_reuse_check_section_is_present_and_clean():
    payload = json.loads(METRICS.read_text())
    reuse = payload["reuse_check"]
    assert reuse["compared"] is True
    # the pool legs re-run in this process must match the committed numbers
    assert reuse["worst_relative_difference"] < 1e-10


@pytest.mark.skipif(not METRICS.exists(), reason="probe artifact not generated yet")
def test_hypothesis_reading_matches_the_stored_escalation():
    payload = json.loads(METRICS.read_text())
    block = payload["p3_resolution"]["r3"]
    assert block["available"] is True
    assert block["finest_pair"] == [400, 800]
    reading = payload["hypotheses"]["resolution"]
    if block["meets_requirement"]:
        assert "H1" in reading
    elif block["escalation_factor"] >= ESCALATION_FACTOR:
        assert "H1" in reading
    else:
        assert "H2" in reading