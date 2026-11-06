"""Conservation audit of the **2-D Euler system** with per-step statistics.

The round-2 study (``tests/test_conservation.py``) audits the 1-D Euler budget
and the modal invariants of scalar DG paths.  This file extends the audit to one
new scenario: the four global invariants of the *system* of conservation laws in
two space dimensions,

    d/dt int_Omega U dOmega = 0   for U = (rho, rho*u, rho*v, E)

on a doubly periodic box.  A bilinear flux-difference scheme conserves all four
components exactly when its face fluxes telescope in *both* directions, so the
criterion needs no reference solution -- only float64 round-off.

Two things distinguish this audit from an end-state difference:

* **per-step statistics** -- the solver's ``observer`` hook samples the
  invariants once per accepted step, so the ledger is judged by its per-step
  increments (mean, rms, bias) and not only by ``U(t_final) - U(0)``.  Round-off
  increments are zero-mean noise; a one-sided leak has ``|mean|/rms ~ 1``.
* **an anisotropic mesh** -- ``dx != dy`` (40 x 20 cells on a square box), so a
  swapped or dropped cell width cannot hide.

Falsifiability is proved in-file: :func:`injected_mass_source` breaks the
telescoping stencil in one cell by a *known* amount, and the tests below then
require the ledger to (a) turn red, (b) localise the imbalance in the mass
account only, and (c) recover the injected ``leak*T`` to within a few percent.
"""
import numpy as np
import pytest

from tests._conservation_tools import (
    conserved_totals_2d,
    drift_statistics,
)
from tests.conservation_study import (
    CRITERIA,
    EULER2D_INVARIANTS,
    study_euler_2d_conservation,
)

BOXCASE = (("square_32", (32, 32), 1.0, ((-5.0, 5.0), (-5.0, 5.0))),)
LEAK = 1.0e-3  # units of mass per unit time, inside a single cell


@pytest.fixture(scope="module")
def clean_study():
    """The unconverted 2-D Euler audit: three runs, one sample per step."""
    return study_euler_2d_conservation()


# ---------------------------------------------------------------------------
# measurement self-tests (no solver involved)
# ---------------------------------------------------------------------------
def test_conserved_totals_2d_weighs_both_cell_widths():
    states = np.zeros((2, 3, 4))
    states[..., 0] = 1.0  # rho
    states[0, 0, 1] = 6.0  # rho*u in one cell only
    states[1, 2, 3] = 12.0  # E in one cell only
    totals = conserved_totals_2d(states, dx=0.5, dy=2.0)  # cell area = 1.0
    assert totals["mass"] == pytest.approx(6.0)
    assert totals["momentum_x"] == pytest.approx(6.0)
    assert totals["momentum_y"] == pytest.approx(0.0)
    assert totals["energy"] == pytest.approx(12.0)
    # Halving dx halves every integral: the weighing is dx*dy, not dx.
    half = conserved_totals_2d(states, dx=0.25, dy=2.0)
    assert half["mass"] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        conserved_totals_2d(states[..., :3], dx=1.0, dy=1.0)
    with pytest.raises(ValueError):
        conserved_totals_2d(states, dx=0.0, dy=1.0)


def test_drift_statistics_separate_roundoff_noise_from_a_systematic_leak():
    """The estimator must be able to fail; a one-sided leak is its red state."""
    rng = np.random.default_rng(20261105)
    noise = 100.0 + np.concatenate(([0.0], np.cumsum(rng.normal(0.0, 1e-15, 200))))
    stats = drift_statistics(noise, scale=100.0)
    assert stats["scaled_drift"] <= CRITERIA["euler2d_drift_max"]
    assert stats["bias_ratio"] <= CRITERIA["euler2d_step_bias_max"]
    assert stats["closure_residual"] <= CRITERIA["euler2d_closure_max"]
    assert stats["samples"] == 201

    leak = 100.0 + 1e-13 * np.arange(201)  # same magnitude, one-sided
    leaked = drift_statistics(leak, scale=100.0)
    assert leaked["scaled_drift"] == pytest.approx(2e-13, rel=1e-9)
    assert leaked["scaled_drift"] > CRITERIA["euler2d_drift_max"]
    assert leaked["bias_ratio"] > 0.99  # |mean|/rms -> 1 for a ramp
    assert leaked["increment_mean"] == pytest.approx(1e-15, rel=1e-9)

    with pytest.raises(ValueError):
        drift_statistics([1.0, 1.0])
    with pytest.raises(ValueError):
        drift_statistics([0.0, 1e-16, 2e-16])  # zero default scale


# ---------------------------------------------------------------------------
# the new scenario: 2-D Euler system invariants
# ---------------------------------------------------------------------------
def test_periodic_euler_2d_conserves_all_four_invariants(clean_study):
    """Mass, both momenta and energy stay at round-off over the whole run."""
    result = clean_study
    assert len(result["rows"]) == 3
    for row in result["rows"]:
        assert row["samples"] == row["steps"] + 1, row  # one sample per step
        assert set(row["invariants"]) == set(EULER2D_INVARIANTS)
        assert row["time_covered"] == pytest.approx(row["final_time"], rel=1e-12)
        for invariant in EULER2D_INVARIANTS:
            stats = row["invariants"][invariant]
            assert stats["scaled_drift"] <= CRITERIA["euler2d_drift_max"], (row["case"], invariant, stats)
            assert stats["bias_ratio"] <= CRITERIA["euler2d_step_bias_max"], (row["case"], invariant, stats)
            assert stats["closure_residual"] <= CRITERIA["euler2d_closure_max"], (row["case"], invariant, stats)
        # The imbalance is measured per step, so the residual never compounds:
        # the largest single-step relative increment is still round-off.
        assert max(
            row["invariants"][invariant]["max_step_increment"]
            for invariant in EULER2D_INVARIANTS
        ) <= 100.0 * CRITERIA["euler2d_drift_max"], row


def test_euler_2d_drift_rate_is_independent_of_the_step_count(clean_study):
    """Quadrupling the step count must not amplify the per-step leak rate."""
    rows = {row["case"]: row for row in clean_study["rows"]}
    short, long = rows["square_32"], rows["square_32_long"]
    assert short["ncells"] == long["ncells"]
    assert long["steps"] > 3 * short["steps"]
    rate_short = short["max_scaled_drift"] / short["steps"]
    rate_long = long["max_scaled_drift"] / long["steps"]
    assert long["max_scaled_drift"] <= CRITERIA["euler2d_drift_max"]
    assert rate_long <= 10.0 * max(rate_short, 1e-18), (rate_short, rate_long)
    # No account may carry a systematic component either.
    for row in clean_study["rows"]:
        for invariant in EULER2D_INVARIANTS:
            assert abs(row["step_mean_increments"][invariant]) <= (
                CRITERIA["euler2d_drift_max"] / row["samples"]
            ), (row["case"], invariant)


def test_anisotropic_mesh_conserves_both_momentum_components(clean_study):
    """dx != dy: a swapped or dropped cell width would break the telescope."""
    row = {item["case"]: item for item in clean_study["rows"]}["anisotropic_40x20"]
    dx, dy = row["cell_widths"]
    assert dx != pytest.approx(dy)
    for invariant in ("mass", "momentum_x", "momentum_y", "energy"):
        stats = row["invariants"][invariant]
        assert stats["scaled_drift"] <= CRITERIA["euler2d_drift_max"], (invariant, stats)
        assert stats["bias_ratio"] <= CRITERIA["euler2d_step_bias_max"], (invariant, stats)


# ---------------------------------------------------------------------------
# negative control: the audit must turn red under an injected error
# ---------------------------------------------------------------------------
def test_injected_cell_source_turns_the_ledger_red(clean_study):
    """Break the telescoping stencil in one cell -> the budget must not balance.

    The injection adds ``LEAK`` units of mass per unit time in cell (0, 0), so
    the analytic answer for the mass account is ``LEAK*T`` relative to the
    initial mass -- and nothing else may move.
    """
    dirty = study_euler_2d_conservation(cases=BOXCASE, leak=LEAK)
    row = dirty["rows"][0]
    mass = row["invariants"]["mass"]
    clean_mass = clean_study["rows"][0]["invariants"]["mass"]

    # (a) red: the imbalance is many orders above the clean ledger and above
    #     the criterion the clean run was judged by.
    assert mass["scaled_drift"] > 1.0e3 * CRITERIA["euler2d_drift_max"], mass
    assert mass["scaled_drift"] > 1.0e6 * clean_mass["scaled_drift"], (mass, clean_mass)
    # (b) recovered quantitatively: the measured drift equals LEAK*T/|M0|.
    assert dirty["leak_recovery_error"] <= CRITERIA["euler2d_leak_recovery_max"], row
    assert mass["scaled_drift"] == pytest.approx(row["expected_mass_drift"], rel=0.05)
    # (c) localised: a mass-only source leaves momenta and energy untouched,
    #     and the per-step increments become one-sided (bias_ratio -> 1).
    assert mass["bias_ratio"] > 0.9, mass
    for invariant in ("momentum_x", "momentum_y", "energy"):
        stats = row["invariants"][invariant]
        assert stats["scaled_drift"] <= CRITERIA["euler2d_drift_max"], (invariant, stats)
        assert stats["bias_ratio"] <= CRITERIA["euler2d_step_bias_max"], (invariant, stats)


def test_the_criterion_catches_a_leak_far_below_the_clean_drift():
    """Sensitivity: a 1e-9 leak is still red although it is 1e-8 of the mass."""
    result = study_euler_2d_conservation(cases=BOXCASE, leak=1.0e-9)
    mass = result["rows"][0]["invariants"]["mass"]
    assert mass["scaled_drift"] == pytest.approx(result["rows"][0]["expected_mass_drift"], rel=0.05)
    assert mass["scaled_drift"] > CRITERIA["euler2d_drift_max"]
    assert mass["bias_ratio"] > 0.5, mass
    assert result["leak_recovery_error"] <= CRITERIA["euler2d_leak_recovery_max"]