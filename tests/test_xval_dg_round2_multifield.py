"""Contract tests for the multifield extension of the round-2 certified reference.

The point of these tests is that the audit is an *invariant* of the committed
ladder, not a snapshot: every number the report quotes is recomputed here from
the raw ``npz`` files, and the negative controls below show that the five gates
actually bite (a tampered leg must turn MF3 red, a refused DG ladder must not
certify itself).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from benchmarks import xval_dg_round2_multifield as mfbench
from tensordg.cases.shock_entropy import GAMMA, LEFT_STATE, RIGHT_STATE, exact_mass_budget
from tensordg.certified_reference import CERTIFIED, certify


RAW = Path("validation/repro_shu_osher/artifacts/raw")
ARTIFACT = Path("validation/xval_dg_round2/artifacts/multifield_audit.json")

pytestmark = pytest.mark.skipif(
    not (RAW / "fv_cfl0p3_n800.npz").exists(),
    reason="committed round-2 raw ladder is not available",
)


@pytest.fixture(scope="module")
def audit():
    return mfbench.cross_validate(RAW)


# ---------------------------------------------------------------- MF1 / MF2
def test_every_conserved_field_certifies_on_the_same_ladder(audit):
    """All three fields use the SAME ladder, so all three must be CERTIFIED."""
    rows = audit["MF1_field_certification"]["fields"]
    assert set(rows) == set(mfbench.CONSERVED_FIELDS)
    for field, row in rows.items():
        assert row["verdict"] == CERTIFIED, field
        assert row["classification"] == "monotonic", field
        assert row["reference_ncells"] == 1600, field
        # gate-validity rule: U must be a third of the strict gate at most
        assert row["band_share_of_strict_gate"] <= 1.0 / 3.0, field
    assert audit["MF1_field_certification"]["pass"]


def test_observed_orders_differ_between_fields(audit):
    """A common-mode reconstruction bug would move the three ladders together.

    The orders are *close* (they share the flux and limiter) but not identical,
    which is the evidence that each field really got its own extrapolation
    rather than the density band being reused.
    """
    orders = [row["observed_order"] for row in audit["MF1_field_certification"]["fields"].values()]
    assert len(set(round(p, 6) for p in orders)) == 3
    assert max(orders) - min(orders) < 0.2


def test_derived_bands_follow_the_declared_additive_rule(audit):
    bands = audit["band_percent"]
    assert bands["u"] == pytest.approx(bands["rho_u"] + bands["rho"])
    assert bands["p"] == pytest.approx(bands["rho_e"] + bands["rho_u"] + bands["rho"])
    for derived in mfbench.DERIVED:
        sources = mfbench.BAND_RULE[derived]
        assert bands[derived] >= max(bands[s] for s in sources)
        assert audit["MF2_derived_usability"]["derived"][derived]["gate_usable_at_strict"]
    assert audit["MF2_derived_usability"]["pass"]


# ---------------------------------------------------------------- MF3
@pytest.mark.parametrize("field,gated", [
    ("rho", True), ("rho_u", True), ("rho_e", True), ("u", True), ("p", True),
])
def test_every_leg_passes_the_strict_gate_in_every_field(audit, field, gated):
    values = [leg[field]["rel_l1_percent"] for leg in audit["legs"]]
    assert len(values) == len(mfbench.DG_LEGS)
    assert all(v is not None and v < mfbench.STRICT_PERCENT for v in values), field
    assert all(leg[field]["decision_strict"] != "FAIL" for leg in audit["legs"]), field
    assert all(leg[field]["margin_percent"] > 0.0 for leg in audit["legs"]), field


def test_mf3_counts_cover_every_leg_and_field(audit):
    n = len(mfbench.DG_LEGS) * len(mfbench.FIELDS)
    assert audit["MF3_leg_gate_all_fields"]["n_comparisons"] == n
    rows = [leg[field] for leg in audit["legs"] for field in mfbench.FIELDS]
    assert all(row["rel_l1_percent"] is not None for row in rows)


def test_tampered_leg_turns_mf3_red(tmp_path, audit):
    """Negative control: the gate must bite.

    The committed momentum of one leg is scaled by 1.05 *and* the density by
    1.0; the momentum error must then exceed the 3 % gate.  Without this the
    "55/55 PASS" claim would be a statement about the test, not the solver.
    """
    raw = tmp_path / "raw"
    shutil.copytree(RAW, raw)
    victim = mfbench.leg_filename(2, 800, 0.4)
    with np.load(raw / victim) as data:
        arrays = {k: np.array(data[k]) for k in data.files}
    arrays["rho_u"] = arrays["rho_u"] * 1.05
    np.savez(raw / victim, **arrays)

    tampered = mfbench.cross_validate(raw)
    assert tampered["MF3_leg_gate_all_fields"]["pass"] is False
    assert tampered["accepted"] is False
    bad = tampered["MF3_leg_gate_all_fields"]["worst"]
    assert bad["field"] in mfbench.FIELDS
    assert bad["rel_l1_percent"] > mfbench.STRICT_PERCENT


def test_no_dg_family_can_certify_a_new_field_on_its_own(audit):
    """The reference for rho_u / rho_e must come from the FV ladder, not the DG one."""
    ladder = mfbench.load_field_ladders(RAW)
    for field in ("rho_u", "rho_e"):
        levels = [{"ncells": leg["ncells"], "field": leg[field]}
                  for leg in ladder["dg"] if leg["degree"] == 2 and leg["cfl"] == 0.4]
        verdict = certify(levels, tolerance_percent=mfbench.GATE_PERCENT)["verdict"]
        assert verdict != CERTIFIED, field


# ---------------------------------------------------------------- MF4
def test_state_oracles_hit_the_hugoniot_state(audit):
    rows = {row["name"]: row for row in audit["MF4_state_oracles"]["rows"]}
    rho_l, u_l, p_l = LEFT_STATE
    assert rows["plateau_rho"]["got"] == pytest.approx(rho_l, rel=1e-6)
    assert rows["plateau_u"]["got"] == pytest.approx(u_l, rel=1e-6)
    assert rows["plateau_p"]["got"] == pytest.approx(p_l, rel=1e-6)
    assert audit["MF4_state_oracles"]["pass"]


def test_quiet_zone_is_left_bit_untouched(audit):
    """Behind the entropy wave nothing is disturbed: u == 0 and p == 1 exactly."""
    certified = mfbench.certify_multifield(RAW)
    x = certified["x"]
    quiet = x >= mfbench.state_oracles()["undisturbed_window"][0]
    assert np.all(certified["derived"]["u"][quiet] == 0.0)
    assert np.allclose(certified["derived"]["p"][quiet], 1.0, rtol=0.0, atol=1e-12)
    # and the deviation rows of MF4 are exactly the readings above
    rows = {row["name"]: row for row in audit["MF4_state_oracles"]["rows"]}
    assert rows["entropy_wave_u_deviation"]["rel_error_percent"] == 0.0
    assert rows["entropy_wave_p_deviation"]["rel_error_percent"] < 1e-10


# ---------------------------------------------------------------- MF5
def test_closed_form_mass_matches_the_case_module():
    """Two independent closed forms of the same budget must agree to round-off."""
    mine = mfbench.closed_form_budgets()
    theirs = exact_mass_budget()["mass_final_predicted"]
    assert mine["mass"]["final_predicted"] == pytest.approx(theirs, rel=1e-13)


def test_momentum_and_energy_fluxes_are_the_physical_ones():
    budgets = mfbench.closed_form_budgets()
    rho_l, u_l, p_l = LEFT_STATE
    rho_r, u_r, p_r = RIGHT_STATE
    assert budgets["momentum"]["inflow_flux"] == pytest.approx(rho_l * u_l ** 2 + p_l)
    assert budgets["momentum"]["outflow_flux"] == pytest.approx(p_r)
    assert budgets["momentum"]["outflow_flux"] == pytest.approx(p_r * 1.0 + rho_r * u_r ** 2)
    e_l = p_l / (GAMMA - 1.0) + 0.5 * rho_l * u_l ** 2
    assert budgets["energy"]["rate"] == pytest.approx((e_l + p_l) * u_l)
    assert budgets["momentum"]["rate"] == pytest.approx(
        budgets["momentum"]["inflow_flux"] - budgets["momentum"]["outflow_flux"])


def test_reference_closes_all_three_budgets(audit):
    rows = {row["name"]: row for row in audit["MF5_budget_closure"]["rows"]}
    for name in ("mass", "momentum", "energy"):
        assert rows[name]["rel_error_percent"] < mfbench.BUDGET_TOL_PERCENT, name
    assert audit["MF5_budget_closure"]["pass"]
    # the richer budgets are the reason this is not redundant with round 2:
    # they carry the strong flux terms and are still two orders tighter
    assert rows["momentum"]["rel_error_percent"] < rows["mass"]["rel_error_percent"]


# ---------------------------------------------------------------- MF6 / misc
def test_density_only_gate_reported_against_the_other_quantities(audit):
    """MF6 never gates, but it must carry the number that motivated this audit."""
    ratios = audit["MF6_rho_gate_proxy"]["ratios"]
    assert set(ratios) == {"rho_u", "rho_e", "u", "p"}
    momentum = ratios["rho_u"]["max_ratio_to_rho"]
    # momentum is the worst of the five quantities on at least one leg ...
    assert momentum > 1.0
    # ... yet still inside the strict gate, so the round-2 verdict stands
    assert audit["MF1_field_certification"]["pass"]
    assert audit["MF3_leg_gate_all_fields"]["per_field"]["rho_u"]["max_percent"] < mfbench.STRICT_PERCENT


def test_derive_rejects_nonpositive_density():
    rho = np.array([1.0, -0.5, 1.0])
    with pytest.raises(ValueError):
        mfbench.derive(rho, np.zeros(3), np.ones(3))


def test_derive_matches_the_analytic_primitive_state():
    rho_l, u_l, p_l = LEFT_STATE
    rho = np.full(4, rho_l)
    rho_u = np.full(4, rho_l * u_l)
    rho_e = np.full(4, p_l / (GAMMA - 1.0) + 0.5 * rho_l * u_l ** 2)
    u, p = mfbench.derive(rho, rho_u, rho_e)
    assert np.allclose(u, u_l)
    assert np.allclose(p, p_l)


def test_artifact_is_not_stale(audit):
    """The committed artifact must equal a fresh run (report numbers are bound)."""
    assert ARTIFACT.exists(), "run python -m benchmarks.xval_dg_round2_multifield --figures"
    stored = json.loads(ARTIFACT.read_text())
    assert stored["accepted"] == audit["accepted"]
    assert stored["benchmark"] == audit["benchmark"]
    assert stored["band_percent"] == pytest.approx(audit["band_percent"])
    assert (stored["MF3_leg_gate_all_fields"]["worst"]["configuration"]
            == audit["MF3_leg_gate_all_fields"]["worst"]["configuration"])
    for name in ("MF4_state_oracles", "MF5_budget_closure"):
        stored_rows = {row["name"]: row for row in stored[name]["rows"]}
        fresh_rows = {row["name"]: row for row in audit[name]["rows"]}
        assert set(stored_rows) == set(fresh_rows)
        for key, row in fresh_rows.items():
            assert stored_rows[key]["rel_error_percent"] == pytest.approx(
                row["rel_error_percent"], rel=1e-9, abs=1e-14), (name, key)
    # the figures the report embeds must exist
    for name in ("fig_mf1_leg_errors_by_field.png", "fig_mf2_derived_profiles.png"):
        fig = ARTIFACT.parent / name
        assert fig.exists() and fig.stat().st_size > 10_000, name


def test_required_cli_gate_returns_nonzero_when_rejected(tmp_path):
    """``--require-accepted`` is the CI hook: prove it can fail."""
    raw = tmp_path / "raw"
    shutil.copytree(RAW, raw)
    victim = mfbench.leg_filename(1, 200, 0.4)
    with np.load(raw / victim) as data:
        arrays = {k: np.array(data[k]) for k in data.files}
    arrays["rho"] = arrays["rho"] * 1.10
    np.savez(raw / victim, **arrays)
    assert mfbench.main(["--rawdir", str(raw), "--json-stdout", "--require-accepted"]) == 1
