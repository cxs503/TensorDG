"""Machine gates for the public API surface of :mod:`tensordg`.

These tests do not "check that the code ran": they pin the *contract* that
``tensordg/__init__.py`` claims, and they fail on the exact defect class this
mission found in the audit trail -- a merged, tested, documented capability that
cannot be reached from the documented entry point ``from tensordg import ...``
(``audit_report_m16246_dalembert.md`` finding #1: ``elastic_1d`` imported but
absent from ``__all__``; ``docs/ldg_poisson_status.md`` §1: a tutorial claiming
an LDG solver that did not exist).

The gates are computed by ``tools/api_contract.py`` (same code that writes
``validation/api_contract/contract.json``), so the test and the published
artifact cannot disagree.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "api_contract.py"
BASELINE = ROOT / "validation" / "api_contract" / "doc_claims_baseline.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location("api_contract", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def contract() -> dict:
    return _load_tool().audit()


@pytest.fixture(scope="module")
def tensordg():
    import tensordg as package

    return package


def test_every_public_symbol_is_reachable_from_the_package_root(contract):
    """No merged capability may be reachable only through its deep module path."""
    coverage = contract["coverage"]
    assert coverage["ratio"] == 1.0, (
        f"{coverage['public_symbols'] - coverage['exported_symbols']} public symbols "
        f"are not exported; first offenders: "
        f"{[(m['module'], m['missing'][:4]) for m in coverage['modules'] if m['missing']][:5]}"
    )
    assert coverage["modules_with_zero_exports"] == []


def test_exported_names_are_the_same_objects_as_their_sources(contract):
    """A root name must be its source symbol, not a same-named look-alike."""
    assert contract["identity"]["violations"] == [], contract["identity"]["violations"]
    assert contract["identity"]["checked"] == contract["dunder_all"]["count"]


def test_dunder_all_is_consistent_with_the_namespace(contract, tensordg):
    report = contract["dunder_all"]
    assert report["duplicates"] == []
    assert report["unresolved"] == []
    assert report["in_all_not_in_namespace"] == []
    assert report["in_namespace_not_in_all"] == [], (
        "public names in the package namespace but missing from __all__: "
        f"{report['in_namespace_not_in_all']}"
    )
    assert report["registry_names_not_in_all"] == []
    assert report["in_all_not_in_registry"] == []
    assert len(tensordg.__all__) == len(set(tensordg.__all__))


def test_documented_import_snippets_resolve(contract):
    """Every ``from tensordg[.x] import ...`` printed in docs/ must work."""
    assert contract["docs"]["snippets_checked"] > 0
    assert contract["docs"]["snippets_unresolved"] == [], (
        contract["docs"]["snippets_unresolved"]
    )


def test_no_unregistered_claims_about_missing_modules(contract):
    """A doc may not reference a ``tensordg/<path>.py`` that does not exist.

    Known, explained references live in ``doc_claims_baseline.json`` with a
    reason and a status; anything new is a red gate.
    """
    assert contract["docs"]["unregistered_claims"] == [], (
        contract["docs"]["unregistered_claims"]
    )


def test_claim_baseline_is_auditable():
    """Every registered stale reference must carry a reason and a status."""
    payload = json.loads(BASELINE.read_text())
    entries = payload["known_stale"]
    assert entries
    for entry in entries:
        assert entry["claim"].startswith("tensordg/")
        assert entry["reason"].strip()
        assert entry["status"] in {
            "OPEN_OVERCLAIM", "PLANNED_MOVE", "HISTORICAL_UPSTREAM",
        }


@pytest.mark.parametrize("name", [
    # the mission's regression set: capabilities merged before this fix but
    # unreachable from `from tensordg import ...`
    "solve_elastic_1d", "dalembert_pulse", "elastic_wave_speed",
    "ritter_exact", "thacker_radial_basin", "solve_ldg_dirichlet_1d",
    "solve_poisson_2d_sparse", "solve_poisson_2d_matrixfree",
    "solve_kovasznay_dg_2d", "solve_taylor_green_dg_2d",
    "solve_shear_wave_dg_2d", "wall_heat_flux",
    "DGAdvection2D", "CurvedAdvection", "physical_branch",
    "CASES", "SWE_GRAVITY", "DISPERSION_MODELS",
])
def test_named_capabilities_are_exported(tensordg, name):
    assert name in tensordg.__all__, f"{name} missing from tensordg.__all__"
    assert hasattr(tensordg, name), f"tensordg.{name} does not resolve"


def test_prototypes_are_excluded_from_the_contract_and_stay_unexported(contract, tensordg):
    """An unvalidated prototype is not a capability: it may not be exported.

    Regression for the red-suite defect found by m16505: the API-coverage sweep
    (m16429) exported ``tensordg.poisson_2d_ldg`` even though the module's own
    docstring says it is deliberately not re-exported and
    ``tests/test_ldg_closure_rank.py`` forbids it.  The gate is now machine
    checked from both sides: excluded from the coverage denominator, marker
    required, and no symbol may leak into the root namespace.
    """
    prototypes = contract["prototypes"]
    # The registry may legitimately be *empty*: the last prototype
    # (``tensordg.poisson_2d_ldg``) was validated and promoted, see
    # ``docs/CASE-DG-LDG-2D.md``.  What must never happen is a registered
    # prototype that lost its marker or leaked into the root namespace.
    assert contract["gates"]["prototypes_excluded_and_marked"]
    for record in prototypes["prototypes"]:
        assert record["marker_present"], (
            f"{record['module']} lost its '{record['marker']}' status marker")
        assert record["leaked_exports"] == [], (
            f"prototype {record['module']} leaked into the public API: "
            f"{record['leaked_exports']}")
        for name in record["leaked_exports"]:
            assert not hasattr(tensordg, name)


def test_colliding_names_are_disambiguated_by_alias(tensordg):
    """``sound_speed`` exists in both the Euler and the SWE module: both kept."""
    from tensordg import euler_1d, swe_1d

    assert tensordg.euler_sound_speed is euler_1d.sound_speed
    assert tensordg.swe_sound_speed is swe_1d.sound_speed
    assert tensordg.euler_sound_speed is not tensordg.swe_sound_speed
    assert tensordg.euler_physical_flux is euler_1d.physical_flux
    assert tensordg.swe_physical_flux is swe_1d.physical_flux
    assert tensordg.advection_diffusion_l2_error is not tensordg.l2_error
    assert tensordg.ice_edge_observed_order is not tensordg.observed_order


def test_case_registry_matches_its_modules(tensordg):
    """``CASES`` is the documented single source of truth for shipped cases."""
    from tensordg.cases import (
        ELASTIC_ROD_CASE_KEY, CASE_KEY,
        LAX_CASE_KEY, VISCOUS_BURGERS_CASE_KEY,
    )

    assert tensordg.CASES == {
        "shock_entropy": CASE_KEY,
        "elastic_rod_1d": ELASTIC_ROD_CASE_KEY,
        "lax_1d": LAX_CASE_KEY,
        "viscous_burgers_1d": VISCOUS_BURGERS_CASE_KEY,
    }
    assert tensordg.ELASTIC_ROD_CASE_KEY == ELASTIC_ROD_CASE_KEY
    assert tensordg.LAX_CASE_KEY == LAX_CASE_KEY
    assert tensordg.VISCOUS_BURGERS_CASE_KEY == VISCOUS_BURGERS_CASE_KEY