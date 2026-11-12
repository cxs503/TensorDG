"""Invariant tests for the certified-reference (Richardson + GCI) machinery.

The tests are deliberately of two kinds:

* **synthetic power-law controls** -- data with a *known* exact solution and a
  *known* order, so the estimator itself is verified (order recovery,
  extrapolation gain, GCI formula) before it is trusted on the real case;
* **committed-artifact gates** -- the acceptance of CASE-SHU-OSHER re-read from
  the committed raw profiles, so a later "improvement" cannot silently move the
  reference or the gate.

Run: ``pytest -q tests/test_shu_osher_certified_ref.py``
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import shu_osher_certified_ref as cr

RAW = Path(__file__).resolve().parents[1] / "validation" / "repro_shu_osher" / "artifacts" / "raw"


# --------------------------------------------------------------- estimator unit
def _power_law_ladder(order: float = 2.0, const: float = 0.3, levels=(400, 800, 1600, 3200)):
    """Exact solution + C*N^-order perturbation, sampled as cell averages.

    The exact part is *linear* in ``x`` and the perturbation is *constant*, so
    midpoint sampling equals the cell average exactly on every nesting grid.
    That keeps the synthetic data a pure power law, which is what makes the
    order/extrapolation assertions exact rather than approximate -- the same
    trick used for manufactured solutions in code verification.
    """
    out = []
    for n in levels:
        x = (np.arange(n) + 0.5) * (10.0 / n) - 5.0
        exact = 1.0 + 0.2 * x
        out.append({"ncells": n, "x": x,
                    "rho": exact + const * n ** (-order)})
    return out


def test_observed_order_recovers_the_synthetic_order():
    for order in (1.0, 1.5, 2.0, 3.0):
        levels = _power_law_ladder(order=order)
        study = cr.convergence_study(levels)
        # residual 1e-4 level is subtraction round-off in the difference of two
        # nearly identical profiles (not an estimator defect): the synthesised
        # amplitudes are 1e-3..1e-4 of the profile, so ~1e-12 cancellation error
        # appears amplified by 1/log(2) in the order.
        assert study["observed_order_used"] == pytest.approx(order, rel=2e-3)


def test_richardson_recovers_the_exact_profile():
    levels = _power_law_ladder(order=2.0)
    study = cr.convergence_study(levels)
    x = study["x"]
    exact = 1.0 + 0.2 * x
    err_raw = cr.rel_l1(study["rho_raw_finest"], exact)
    err_ext = cr.rel_l1(study["rho_certified"], exact)
    # extrapolation must reduce the error by at least a decade for p = 2, r = 2
    assert err_ext < 0.05 * err_raw
    assert err_ext < 1e-12


def test_gci_formula_and_positivity():
    d_coarse, d_fine, p = 1e-2, 5e-3, 2.0
    assert cr.gci(d_coarse, d_fine, p) == pytest.approx(1.25 * d_fine / (2.0 ** p - 1.0))
    assert cr.gci(d_coarse, d_fine, p) > 0.0
    with pytest.raises(ValueError):
        cr.observed_order(0.0, 1e-3)


def test_coarsen_is_the_volume_average_and_rel_l1_is_zero_for_identity():
    v = np.arange(16, dtype=float)
    assert np.allclose(cr.coarsen(v, 4), v.reshape(-1, 4).mean(axis=1))
    assert cr.rel_l1(v, v) == 0.0


def test_needs_three_levels_and_nesting_grids():
    with pytest.raises(ValueError):
        cr.convergence_study(_power_law_ladder()[:2])
    bad = _power_law_ladder(levels=(400, 800, 1200))
    with pytest.raises(ValueError):
        cr.convergence_study(bad)


# ------------------------------------------------------- committed-artifact gate
@pytest.fixture(scope="module")
def study():
    levels = cr.load_fv_levels(RAW)
    return cr.convergence_study(levels), levels


def test_committed_reference_ladder_has_at_least_three_levels(study):
    conv, levels = study
    assert [lv["ncells"] for lv in levels][:3] == [800, 1600, 3200]
    assert len(levels) >= 3


def test_committed_reference_uncertainty_is_well_below_the_gate(study):
    conv, _ = study
    # observed order must be measured, not assumed: shocks cap the L1 order near 1
    assert 0.6 < conv["observed_order_used"] < 3.0
    # the reference's own uncertainty must be a small fraction of the 5 % gate
    assert conv["gci_finest_percent"] < 0.5
    for d in conv["differences"]:
        assert d["rel_l1_percent"] > 0.0
    # differences must shrink under refinement in the asymptotic regime
    assert conv["differences"][-1]["rel_l1"] <= conv["differences"][0]["rel_l1"]


def test_every_committed_dg_leg_passes_the_5pct_gate():
    full = cr.build_study(RAW)
    assert full["missing_legs"] == []
    assert full["all_legs_pass"] is True
    for lg in full["dg_legs"]:
        assert lg["rel_l1_percent"] < cr.TOLERANCE_PERCENT
        assert lg["rho_min"] > 0.0            # admissibility of the DG solution
        assert lg["windows_percent"]["plateau"] < lg["windows_percent"]["whole"] or \
            lg["windows_percent"]["plateau"] < 1.0


def test_window_decomposition_separates_shock_from_entropy_wave():
    full = cr.build_study(RAW)
    best = min((lg for lg in full["dg_legs"] if lg["cfl"] == 0.4),
               key=lambda lg: lg["rel_l1_percent"])
    w = best["windows_percent"]
    # the untouched pre-shock sine must be resolved far better than the whole domain
    assert w["preshock_sine"] < 0.2 * w["whole"]
    assert w["plateau"] < 0.2 * w["whole"]


def test_committed_certified_bundle_matches_a_fresh_recomputation(tmp_path):
    """Structural anti-transcription check: the shipped JSON/npz are reproducible."""
    out = tmp_path / "bundle"
    rc = cr.main(["--rawdir", str(RAW), "--outdir", str(out)])
    assert rc == 0
    shipped_json = RAW.parents[3] / "artifacts" / "m15803" / "certified_reference.json"
    if not shipped_json.exists():
        pytest.skip("shipped bundle not present in this checkout")
    fresh = json.loads((out / "certified_reference.json").read_text())
    shipped = json.loads(shipped_json.read_text())
    assert fresh["observed_order_used"] == pytest.approx(shipped["observed_order_used"], rel=1e-12)
    assert fresh["gci_finest_percent"] == pytest.approx(shipped["gci_finest_percent"], rel=1e-12)
    with np.load(out / "certified_reference.npz") as a, \
            np.load(shipped_json.parent / "certified_reference.npz") as b:
        assert np.allclose(a["rho_certified"], b["rho_certified"], rtol=0.0, atol=1e-14)