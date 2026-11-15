"""Regression tests for the CASE-LAX-DG wave-structure / admissibility audit.

The audit artifact ``docs/benchmarks/results/lax_wave_structure/metrics.json`` is
computed **from the committed certified fields** (``lax_case_pool`` P3/N=400 and
``lax_resolution_requirement`` P3/N=800) plus the closed-form Riemann solution --
no solver run is needed to reproduce it.  The integral L2 numbers are *cited*
from ``lax_resolution_requirement/metrics.json`` (``banked_l2``), never recomputed
here.

The tests pin four things:

1. **artifact integrity** -- schema, figures, summary, provenance of ``banked_l2``;
2. **consistency with the source artifacts** -- every audit leg points at an
   existing profile file, and the cited L2 numbers equal the source artifact's;
3. **the documented finding** -- the certified configurations are *not* monotone:
   nodes leave the exact range, ``TV(num)/TV(exact)`` is far above 1 and grows
   with refinement.  Bounds are recorded so a regression to monotonicity shows up
   as a change that must be re-documented;
4. **analytic cross-checks** -- exact TV is available in closed form from the star
   state, and the exact wave positions / star state match Toro's values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ARTIFACT = Path("docs/benchmarks/results/lax_wave_structure/metrics.json")
PROFILES = Path("docs/benchmarks/results/lax_wave_structure/profiles.npz")
FIGURES = Path("docs/benchmarks/results/lax_wave_structure/figures")
SUMMARY = Path("docs/benchmarks/results/lax_wave_structure/summary.md")
BANKED_PATH = Path("docs/benchmarks/results/lax_resolution_requirement/metrics.json")


@pytest.fixture(scope="module")
def metrics():
    if not ARTIFACT.exists():
        pytest.skip(f"audit artifact missing: {ARTIFACT}")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def legs(metrics):
    return {leg["leg"]: leg for leg in metrics["legs"]}


def test_artifact_carries_the_audit_schema(metrics):
    for key in ("anchor", "case", "question", "pre_registered_definitions",
                "exact", "legs", "limiter_arms", "banked_l2", "reuse_check",
                "trends", "verdict", "metrics_sha256"):
        assert key in metrics, key
    assert metrics["anchor"] == "CASE-LAX-DG"
    assert set(metrics["pre_registered_definitions"]) == {"W1", "W2", "W3", "W4", "W5"}


def test_audit_covers_both_certified_legs(legs):
    assert set(legs) == {"p3.N400", "p3.N800"}
    for leg in legs.values():
        assert leg["degree"] == 3
        assert Path(leg["source"]).exists()
        assert leg["l2_errors"] is None            # integral norm is cited, not recomputed


def test_figures_and_summary_are_committed(metrics):
    assert SUMMARY.exists()
    assert metrics["figures"]
    for name in metrics["figures"]:
        path = FIGURES / name
        assert path.exists(), path
        assert path.stat().st_size > 5000, path


def test_profiles_npz_holds_every_leg(legs):
    data = np.load(PROFILES)
    for leg in legs:
        points = data[f"{leg}.points"]
        assert points.shape == data[f"{leg}.numerical_density"].shape
        assert points.ndim == 2 and points.shape[1] == 10


def test_cited_l2_matches_the_source_artifact(metrics):
    """``banked_l2`` must be a faithful copy of the source artifact's numbers."""
    assert metrics["banked_l2"]["status"] == "ok"
    source = json.loads(BANKED_PATH.read_text(encoding="utf-8"))
    src = {(row.get("leg") or f"p{row['degree']}.N{row['cells']}"):
           row["l2_errors"]["density"] for row in source["legs"]}
    for leg, row in metrics["banked_l2"]["values"].items():
        assert row["density"] == pytest.approx(src[leg], rel=0, abs=1e-15)


def test_ledger_gate_verdict_is_the_banked_one(metrics):
    verdict = metrics["verdict"]["l2_pass_at_3pct"]
    assert verdict["p3.N400"] is False            # 3.116 % > 3 %
    assert verdict["p3.N800"] is True             # 2.315 % < 3 %


def test_certified_configurations_leave_the_exact_range(metrics):
    for leg in metrics["legs"]:
        row = leg["extrema"]["density"]
        assert row["nodes_outside_exact_range"] > 0, leg["leg"]
        assert leg["extrema"]["pressure"]["num_min"] < \
            leg["extrema"]["pressure"]["exact_min"]


def test_total_variation_exceeds_the_monotone_limit(metrics):
    for leg in metrics["legs"]:
        assert leg["tv"]["density"]["ratio"] > 3.0, leg["leg"]
        assert leg["tv"]["pressure"]["ratio"] > 5.0, leg["leg"]


def test_spurious_variation_grows_with_refinement(metrics):
    trends = metrics["trends"]["tv_ratio_change"]
    for field in ("density", "pressure"):
        assert trends[field]["grows_with_refinement"], field
        assert trends[field]["last"] > trends[field]["first"]


def test_undershoot_amplitude_grows_with_refinement(metrics):
    first, last = metrics["legs"][0], metrics["legs"][-1]
    for field in ("density", "pressure"):
        assert (last["extrema"][field]["undershoot_rel_span"]
                > first["extrema"][field]["undershoot_rel_span"])


def test_admissibility_gate_fails(metrics):
    assert metrics["verdict"]["admissibility_pass"] is False
    assert "nodes_outside_exact_range == 0" in \
        metrics["verdict"]["admissibility_gate_proposed"]


def test_wave_positions_remain_within_the_c1_bar(metrics):
    for leg in metrics["legs"]:
        for name, row in leg["position_errors"].items():
            if row is None:
                continue
            assert row["rel_error"] < 0.02, (leg["leg"], name, row["rel_error"])


def test_captured_discontinuities_stay_thin_in_cells(metrics):
    for leg in metrics["legs"]:
        contact = leg["thickness"]["contact"]["width_cells"]
        shock = leg["thickness"]["shock"]["width_cells"]
        assert 0.0 < contact < 4.0, (leg["leg"], contact)
        assert 0.0 < shock < 4.0, (leg["leg"], shock)


def test_thickness_scaling_is_recorded_for_both_waves(metrics):
    for wave in ("contact", "shock"):
        row = metrics["trends"][f"thickness_scaling_exponent_{wave}"]
        widths = row["width_cells"]
        assert len(widths) == len(metrics["legs"])
        assert all(w is not None and w > 0 for w in widths)


def test_star_state_matches_the_closed_form(metrics):
    star = metrics["exact"]["star"]
    assert star["pressure"] == pytest.approx(2.466098, abs=2e-6)
    assert star["velocity"] == pytest.approx(1.528723, abs=2e-6)
    assert star["density_star_left"] == pytest.approx(0.344568, abs=2e-6)
    assert star["density_star_right"] == pytest.approx(1.304085, abs=2e-6)


def test_exact_wave_positions_match_toro(metrics):
    pos = metrics["exact"]["positions"]
    assert pos["rarefaction_head"] == pytest.approx(0.157637, abs=2e-6)
    assert pos["rarefaction_tail"] == pytest.approx(0.287229, abs=2e-6)
    assert pos["contact"] == pytest.approx(0.698734, abs=2e-6)
    assert pos["shock"] == pytest.approx(0.822312, abs=2e-6)


def test_exact_total_variation_is_analytic(metrics):
    """TV of the exact density profile is a closed-form combination of jumps."""
    star = metrics["exact"]["star"]
    l, r = metrics["case"]["left"], metrics["case"]["right"]
    analytic = (abs(l[0] - star["density_star_left"])
                + abs(star["density_star_right"] - star["density_star_left"])
                + abs(r[0] - star["density_star_right"]))
    for leg in metrics["legs"]:
        assert leg["tv"]["density"]["exact"] == pytest.approx(analytic, rel=1e-4)


def test_reuse_check_recomputes_the_same_field_metrics(metrics, legs):
    """The reuse block (built from the same npz) must agree with the audit legs."""
    for name, row in metrics["reuse_check"].items():
        assert row["status"] == "ok"
        assert row["tv"]["density"]["ratio"] == pytest.approx(
            legs[name]["tv"]["density"]["ratio"], abs=1e-12)
        assert row["thickness"]["contact"]["width"] == pytest.approx(
            legs[name]["thickness"]["contact"]["width"], abs=1e-12)