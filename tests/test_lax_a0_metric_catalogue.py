"""Gates for ``benchmarks/lax_a0_metric_catalogue`` (offline, zero solver runs).

The catalogue asks *which target* the ``CASE-LAX-DG`` ledger arm A0 measures its
relative L2 on.  These tests lock the three things that make its conclusions
checkable rather than asserted:

1. the catalogue re-reads the committed pool value bit-for-bit (no drift);
2. the cell-average metrics are **exactly** invariant under same-mesh P0
   truncation -- that invariance *is* the non-discrimination argument, so it must
   stay exact;
3. the four-term pointwise decomposition is an identity (residual ~ 1e-14) and
   its total equals the canonical, breakpoint-aligned A0 reading.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.lax_a0_metric_catalogue import (
    CONSERVED,
    DOC,
    OUT,
    PRIMITIVE,
    build_metrics,
    cell_average_norm,
    load_leg,
    pythagoras_split,
    pointwise_norm,
    render_doc,
    truncate_to_cell_means,
)
from benchmarks.lax_a0_quadrature_audit import POOL_PROFILES
from benchmarks.lax_case_pool import ledger_acceptance_tolerance
from benchmarks.lax_case_pool import production_errors_from_profiles


@pytest.fixture(scope="module")
def metrics():
    return build_metrics(with_figures=False)


@pytest.fixture(scope="module")
def fields():
    return load_leg(POOL_PROFILES)


def test_reproduces_committed_pool_reading(metrics):
    """The order-10 catalogue reading is the committed pool A0 value."""
    committed = production_errors_from_profiles(POOL_PROFILES, quadrature_order=10)["l2"]
    catalogue = metrics["catalogue"]["primitive/l2/pointwise"]["fields"]
    for field_name in PRIMITIVE:
        reference = committed[field_name]
        assert catalogue[field_name]["dg"] == pytest.approx(reference, rel=1e-12)


def test_cell_average_metrics_are_invariant_under_p0_truncation(fields, metrics):
    """A metric blind to sub-cell content cannot rank schemes on one mesh."""
    p0 = truncate_to_cell_means(fields)
    for variables, keys in (("primitive", PRIMITIVE), ("conserved", CONSERVED)):
        for norm in ("l2", "l1"):
            full = cell_average_norm(fields, variables, norm)
            truncated = cell_average_norm(p0, variables, norm)
            for key in keys:
                drift = abs(full[key] - truncated[key]) / full[key]
                assert drift < 1e-12, (variables, norm, key, drift)
    gate = metrics["gates"]["cell_average_invariance_worst_relative_drift"]
    assert gate["status"] == "PASS"
    assert gate["value"] < 1e-12


def test_pointwise_family_fails_and_cell_average_passes(metrics):
    """The verdict of A0 depends on the target, and only the pointwise one bites."""
    tolerance = ledger_acceptance_tolerance()
    for key in ("primitive/l2/pointwise", "primitive/l2/pointwise_aligned_converged",
                "conserved/l2/pointwise"):
        entry = metrics["catalogue"][key]
        assert entry["worst_dg"] > tolerance, (key, entry["worst_dg"])
        assert entry["verdict"] == "FAIL"
    for key in ("primitive/l2/cell_average", "conserved/l2/cell_average",
                "primitive/l1/cell_average", "conserved/l1/cell_average"):
        entry = metrics["catalogue"][key]
        assert entry["worst_dg"] < tolerance, (key, entry["worst_dg"])
        assert entry["verdict"] == "PASS"


def test_decomposition_is_an_identity_and_reproduces_canonical_a0(metrics):
    """sub-cell + cell-mean + exact-sub-cell + 2*alignment == pointwise error^2."""
    for field_name in PRIMITIVE:
        entry = metrics["decomposition"][field_name]
        assert entry["residual"] < 1e-10, (field_name, entry["residual"])
        rebuilt = (entry["dg_sub_cell"] + entry["cell_mean_error"]
                   + entry["exact_sub_cell"] + 2.0 * entry["alignment"])
        assert rebuilt == pytest.approx(entry["total"], rel=1e-12)
    gate = metrics["gates"]["decomposition_total_matches_canonical_a0"]
    assert gate["status"] == "PASS"
    assert gate["value"] <= 1e-12


def test_direct_split_matches_catalogue(fields, metrics):
    """Calling the decomposition directly gives the same numbers as the artifact."""
    for field_name in PRIMITIVE:
        direct = pythagoras_split(fields, field_name)
        stored = metrics["decomposition"][field_name]
        for key in ("dg_sub_cell", "cell_mean_error", "exact_sub_cell", "alignment", "total"):
            assert direct[key] == pytest.approx(stored[key], rel=1e-12), (field_name, key)


def test_metric_family_decides_whether_higher_order_helps(metrics):
    """Documented invariant: the P3/P0 ranking flips between metric families.

    * pointwise L2 of primitives: DG wins in density, loses in velocity/pressure;
    * pointwise L1 of primitives: DG loses in *all three*;
    * pointwise L2 of conserved variables: DG wins in all three;
    * cell average: exactly 1.000 (blind by construction).
    """
    ratios = {key: {f: v["discrimination_ratio"]
                    for f, v in metrics["catalogue"][key]["fields"].items()}
              for key in ("primitive/l2/pointwise_aligned_converged",
                          "primitive/l1/pointwise",
                          "conserved/l2/pointwise",
                          "conserved/l2/cell_average",
                          "primitive/l2/cell_average")}
    assert ratios["primitive/l2/pointwise_aligned_converged"]["density"] < 0.8
    assert ratios["primitive/l2/pointwise_aligned_converged"]["velocity"] > 1.0
    assert ratios["primitive/l2/pointwise_aligned_converged"]["pressure"] > 1.0
    for key in PRIMITIVE:
        assert ratios["primitive/l1/pointwise"][key] > 1.0, key
        assert ratios["primitive/l2/cell_average"][key] == pytest.approx(1.0, abs=1e-12)
    for key in CONSERVED:
        assert ratios["conserved/l2/pointwise"][key] < 1.0, key
        assert ratios["conserved/l2/cell_average"][key] == pytest.approx(1.0, abs=1e-12)


def test_committed_artifact_is_reproducible(metrics):
    """The committed metrics.json must be what the code produces now (no drift)."""
    committed = json.loads((OUT / "metrics.json").read_text())
    for key, entry in metrics["catalogue"].items():
        reference = committed["catalogue"][key]["worst_dg"]
        drift = abs(entry["worst_dg"] - reference) / abs(reference)
        assert drift < 1e-9, (key, drift)
    for field_name, entry in metrics["decomposition"].items():
        reference = committed["decomposition"][field_name]["total"]
        assert entry["total"] == pytest.approx(reference, rel=1e-9)


def test_doc_is_regenerated_from_the_artifact(metrics):
    """The checked-in doc must be the rendering of the current numbers."""
    assert DOC.exists()
    assert DOC.read_text() == render_doc(metrics)