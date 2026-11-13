"""Tests for :mod:`tensordg.certified_reference` (round-2 practice as an engine).

Three kinds of assertions, deliberately kept apart:

* **estimator unit tests** -- synthetic ladders with an analytic exact solution
  and a known order, so order recovery, extrapolation and band arithmetic are
  verified before the machinery is trusted on real data;
* **refusal tests** -- stagnating / round-off / gate-consuming ladders must be
  *withheld*, because a reference built from them is a number without meaning;
* **equivalence and gate tests on committed data** -- the engine must reproduce
  the shipped CASE-SHU-OSHER round-2 numbers bit for bit (so the practice is
  codified, not re-invented), and the decision rule must place every shipped DG
  leg where the published report places it.

Run: ``pytest -q tests/test_certified_reference.py``
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks import shu_osher_certified_ref as legacy
from tensordg import certified_reference as cr

RAW = legacy.Path(__file__).resolve().parents[1] / "validation" / "repro_shu_osher" / "artifacts" / "raw"

#: published in docs/CASE-SHU-OSHER-round2.md (round-2 certified band, percent)
PUBLISHED_BAND_PERCENT = 0.08279718833081827
#: published observed order of the FV ladder
PUBLISHED_ORDER = 1.2825091761164593


# ------------------------------------------------------------------ fixtures
def _power_law_ladder(order: float = 2.0, const: float = 0.3, levels=(400, 800, 1600)):
    """Exact solution plus a ``C*N^-order`` perturbation, as cell averages.

    The exact part is linear in ``x`` and the perturbation constant, so midpoint
    sampling equals the cell average exactly on every nesting grid: the ladder is
    a pure power law and the order assertion is exact rather than approximate.
    """
    out = []
    for n in levels:
        x = (np.arange(n) + 0.5) * (10.0 / n) - 5.0
        out.append({"ncells": n, "field": 1.0 + 0.2 * x + const * n ** (-order)})
    return out


def _fv_levels():
    out = []
    for n in (800, 1600, 3200):
        prof = legacy.load_profile(RAW, f"fv_cfl0p3_n{n}.npz")
        out.append({"ncells": n, "field": prof["rho"]})
    return out


def _dg_levels(degree: int):
    out = []
    for n in (200, 400, 800):
        prof = legacy.load_profile(RAW, f"dg_cfl0p4_limweno_n{n}_p{degree}.npz")
        out.append({"ncells": n, "field": prof["rho"]})
    return out


requires_raw = pytest.mark.skipif(
    not (RAW / "fv_cfl0p3_n3200.npz").exists(),
    reason="committed raw profiles of CASE-SHU-OSHER not present",
)


# ------------------------------------------------------- estimator unit tests
@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_order_recovered_from_synthetic_ladder(order):
    cert = cr.certify(_power_law_ladder(order=order), tolerance_percent=5.0)
    # the profile L1 norm used by rel_l1 itself varies as O(N^-p), so a two-pair
    # estimate is exact only to ~1e-3 relative (same tolerance as the round-2
    # driver's own estimator test)
    assert cert["observed_order_used"] == pytest.approx(order, rel=2e-3)
    assert cert["verdict"] == cr.CERTIFIED
    assert cert["gate_usable"] is True


def test_richardson_extrapolation_is_exact_for_a_power_law():
    order = 2.0
    cert = cr.certify(_power_law_ladder(order=order, const=0.03), tolerance_percent=5.0)
    n_ref = cert["reference_ncells"]
    x = (np.arange(n_ref) + 0.5) * (10.0 / n_ref) - 5.0
    exact = 1.0 + 0.2 * x
    got = cert["reference"]
    # residual of the extrapolation is O(const * N^-(p+1)); the exact part is
    # reproduced to round-off, which is what this asserts
    assert np.max(np.abs(got - exact)) < 1e-10
    finest_on_ref = cr.coarsen(cert["field_raw_finest"], 2)
    assert cr.rel_l1(got, exact) < cr.rel_l1(finest_on_ref, exact)


def test_band_of_a_power_law_matches_the_analytic_error():
    """GCI/Fs of the finest level must be the true error of that level."""
    order, const = 2.0, 0.3
    cert = cr.certify(_power_law_ladder(order=order, const=const), tolerance_percent=5.0)
    n_ref = cert["reference_ncells"]
    x = (np.arange(n_ref) + 0.5) * (10.0 / n_ref) - 5.0
    exact = 1.0 + 0.2 * x
    true_error_percent = 100.0 * cr.rel_l1(cr.coarsen(cert["field_raw_finest"], 2), exact)
    assert cert["band_percent"] / cert["fs_gci"] == pytest.approx(true_error_percent, rel=1e-6)


def test_displacement_identity_and_the_wrong_reading():
    """Displacement from the replaced level is r^p/(r^p-1) x its own error.

    The certified profile lives on the second-finest grid, so its distance to the
    raw level it replaces is the error of *that* level -- i.e. a factor
    ``r^p/(r^p-1)`` (about 1.7 for the shipped case) larger than the band ``U``.
    Reading that factor as "the band is understated" is the classic misreading;
    this test pins the arithmetic so the misreading cannot reappear silently.
    """
    cert = cr.certify(_power_law_ladder(order=2.0), tolerance_percent=5.0)
    p, r = cert["observed_order_used"], cert["refinement_fine"]
    factor = r ** p / (r ** p - 1.0)
    assert cert["displacement_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert cert["extrapolation_displacement_percent"] == pytest.approx(
        factor * cert["extrapolation_displacement_expected_percent"] / factor, rel=1e-12)
    assert cert["extrapolation_displacement_percent"] > cert["band_percent"]


def test_band_covers_the_true_error_on_two_term_ladders():
    """Coverage check of ``U``: with a known exact solution, ``U`` must cover it.

    Falsifiable: a band that is too tight would show ``coverage_fraction < 1``.
    The ladders here are smooth (high observed order), so the extrapolation
    removes almost all of the error and the band comes out very conservative --
    which is *not* evidence about the shock case, where the true error cannot be
    measured at all.
    """
    cov = cr.band_coverage_study(n_cases=60, seed=0)
    assert cov["n_cases"] == 60
    assert cov["coverage_fraction"] == 1.0
    assert cov["worst_true_over_band"] < 0.05


# ------------------------------------------------------------ refusal tests
def test_stagnating_ladder_is_withheld_not_averaged_into_a_number():
    """A ladder whose differences stop shrinking must not certify anything."""
    levels = _power_law_ladder(order=2.0)
    # the finest two levels are the same data on the same grid: last difference = 0
    levels[-1] = {"ncells": levels[-1]["ncells"],
                  "field": np.repeat(levels[-2]["field"], 2)}
    cert = cr.certify(levels, tolerance_percent=5.0)
    assert cert["verdict"] == cr.WITHHELD_ROUNDOFF_FLOOR
    assert cert["gate_usable"] is False
    assert cert["classification"] == "roundoff_floor"


def test_roundoff_floor_is_withheld():
    """Three bit-identical levels: every difference is exactly zero."""
    base = _power_law_ladder(order=2.0)[0]["field"]
    levels = [{"ncells": 400, "field": base} for _ in range(3)]
    cert = cr.certify(levels, tolerance_percent=5.0)
    assert cert["verdict"] == cr.WITHHELD_ROUNDOFF_FLOOR
    assert cert["gate_usable"] is False


def test_too_few_levels_is_withheld():
    cert = cr.certify(_power_law_ladder()[:2], tolerance_percent=5.0)
    assert cert["verdict"] == cr.WITHHELD_TOO_FEW_LEVELS
    assert cert["gate_usable"] is False


def test_band_that_eats_the_gate_is_withheld():
    """A reference whose band is > 1/3 of the tolerance cannot decide that gate."""
    cert5 = cr.certify(_power_law_ladder(order=2.0, const=0.3), tolerance_percent=5.0)
    taus = cert5["band_percent"]
    # shrink the gate until the band exceeds one third of it
    tau = taus / 0.5
    cert = cr.certify(_power_law_ladder(order=2.0, const=0.3), tolerance_percent=tau)
    assert cert["verdict"] == cr.WITHHELD_BAND_CONSUMES_GATE
    assert cert["gate_usable"] is False
    assert cert["band_share_of_gate"] > cr.U_SHARE_MAX


# ---------------------------------------- equivalence with the shipped practice
@requires_raw
def test_engine_reproduces_the_shipped_round2_certification_bit_for_bit():
    legacy_study = legacy.convergence_study(legacy.load_fv_levels(RAW))
    cert = cr.certify(_fv_levels(), tolerance_percent=5.0)
    assert cert["observed_order_used"] == legacy_study["observed_order_used"]
    assert cert["band_percent"] == legacy_study["gci_finest_percent"]
    assert cert["extrapolation_displacement_percent"] == \
        legacy_study["extrapolation_displacement_percent"]
    assert np.array_equal(cert["reference"], legacy_study["rho_certified"])
    # and the published round-2 numbers are what the engine still produces
    assert cert["band_percent"] == pytest.approx(PUBLISHED_BAND_PERCENT, abs=0)
    assert cert["observed_order_used"] == pytest.approx(PUBLISHED_ORDER, abs=0)
    assert cert["verdict"] == cr.CERTIFIED


@requires_raw
def test_judge_reproduces_every_shipped_dg_leg_error():
    legacy_study = legacy.build_study(RAW)
    cert = cr.certify(_fv_levels(), tolerance_percent=5.0)
    for lg in legacy_study["dg_legs"]:
        assert "missing" not in lg, lg
        prof = legacy.load_profile(RAW, lg["file"])
        got = cr.judge(prof["rho"], cert)
        assert got["rel_l1_percent"] == pytest.approx(lg["rel_l1_percent"], rel=1e-12)
        assert got["rel_l2_percent"] == pytest.approx(lg["rel_l2_percent"], rel=1e-9)


@requires_raw
def test_every_shipped_leg_passes_the_3_percent_gate_robustly():
    """owner precision standard: >15 % unacceptable; the report claims < 3 %."""
    cert = cr.certify(_fv_levels(), tolerance_percent=3.0)
    assert cert["gate_usable"] is True
    for lg in [(1, 200, 0.4), (1, 400, 0.4), (1, 800, 0.4),
               (2, 200, 0.4), (2, 400, 0.4), (2, 800, 0.4),
               (3, 200, 0.4), (3, 400, 0.4), (3, 800, 0.4),
               (2, 400, 0.1), (2, 400, 0.2)]:
        name = f"dg_cfl{lg[2]:g}".replace(".", "p") + f"_limweno_n{lg[1]}_p{lg[0]}.npz"
        prof = legacy.load_profile(RAW, name)
        got = cr.judge(prof["rho"], cert)
        assert got["decision"] == cr.PASS, (name, got)
        assert got["rel_l1_percent"] + cert["band_percent"] < 3.0


@requires_raw
def test_dg_families_cannot_certify_themselves_quantitatively():
    """The DG ladder is not asymptotic: its band is > 100 % of the gate."""
    for degree in (1, 2, 3):
        cert = cr.certify(_dg_levels(degree), tolerance_percent=5.0)
        assert cert["verdict"] != cr.CERTIFIED
        assert cert["gate_usable"] is False
        assert cert["band_share_of_gate"] > cr.U_SHARE_MAX
        assert cert["observed_order_used"] < cr.P_MIN


# ---------------------------------------------------------------- judge rules
def test_judge_withheld_when_reference_is_not_certified():
    cert = cr.certify(_dg_levels(1), tolerance_percent=5.0)
    got = cr.judge(cert["reference"], cert)
    assert got["decision"] == cr.WITHHELD


def test_judge_pass_marginal_when_margin_is_inside_the_band():
    levels = _power_law_ladder(order=2.0, const=0.3)
    cert = cr.certify(levels, tolerance_percent=5.0)
    n_ref = cert["reference_ncells"]
    x = (np.arange(n_ref) + 0.5) * (10.0 / n_ref) - 5.0
    exact = 1.0 + 0.2 * x
    # error just below a tolerance barely larger than the band: PASS_MARGINAL
    probe = exact + 2.0e-4 * np.sin(3.0 * x)
    err = 100.0 * cr.rel_l1(probe, cert["reference"])
    tau = err + 0.5 * cert["band_percent"]          # margin smaller than the band
    got = cr.judge(probe, cert, tolerance_percent=tau)
    assert got["decision"] == cr.PASS_MARGINAL, got
    # widen the tolerance and the same candidate becomes a clean PASS
    assert cr.judge(probe, cert, tolerance_percent=10.0)["decision"] == cr.PASS
    # tighten it below the error and it fails
    assert cr.judge(probe, cert, tolerance_percent=err * 0.9)["decision"] == cr.FAIL


def test_judge_rejects_non_nesting_candidate():
    cert = cr.certify(_power_law_ladder(order=2.0), tolerance_percent=5.0)
    with pytest.raises(ValueError):
        cr.judge(np.ones(7), cert)

# --------------------------------------------------------------- root reachability
def test_the_engine_is_reachable_from_the_package_root():
    """``from tensordg import certify`` -- audit finding B1 (blocking L3).

    ``docs/AUDIT-m17539-XVAL-DG-ROUND2.md`` found the engine unreachable at the
    root (``ImportError: cannot import name 'certify'``) and the package
    contract gate RED with 11 of its 15 missing symbols coming from this module.
    The root export block must stay in place, and the five names shared with
    :mod:`tensordg.convergence` must stay disambiguated rather than shadowed.
    """
    import tensordg as package
    from tensordg import certified_reference as cr
    from tensordg import convergence

    for name in ("certify", "judge", "classify", "digest_levels", "audit_ladder",
                 "band_coverage_study"):
        assert getattr(package, name) is getattr(cr, name), name
    for name in ("coarsen", "rel_l1", "observed_order", "richardson_extrapolate", "gci"):
        assert getattr(package, "certified_" + name) is getattr(cr, name), name
        if hasattr(convergence, name):
            assert getattr(package, name) is getattr(convergence, name), name
    assert "certify" in package.__all__ and "certified_coarsen" in package.__all__
