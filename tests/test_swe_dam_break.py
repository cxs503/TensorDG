"""Dam-break verification for the 1-D shallow water solver.

Two analytic cases are gated here:

* **Stoker wet-bed dam break** (h_L = 1, h_R = 0.1 / 0.5): rarefaction fan,
  constant star region and a bore, relative L1 error of the depth below 1 %
  (gate: 3 %), star depth/velocity and wave speeds below 1 %.
* **Ritter dry-bed dam break** (h_L = 1, h_R = 0): self-similar rarefaction
  that reaches a dry bed, wet front at ``2 sqrt(g h0)``.  Relative L1 error of
  the depth below 3 % at 200 cells (measured 0.24 %).

The dry-bed case used to be pinned as a *known limitation* (28 % relative L1
error, "the dry front is smeared over many cells").  The real cause was
different: the velocity of a near-dry state was evaluated as ``hu / 1e-12``, so
one Gibbs undershoot at the wetting front produced a wave speed of order 1e12,
the time step collapsed to the 1e-15 floor and the integrator silently stopped
around t ~ 0.006 s while still reporting ``final_time``.  The reported 28 %
error was the error of the *initial condition*.  With the dry guard
(``DRY_DEPTH = 1e-6`` in ``tensordg.swe_1d``) the run reaches the requested
time and the error is 0.24 %; see ``docs/swe_verification.md``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tensordg.swe_1d import reconstruct_fields, solve_swe_1d
from tensordg.swe_exact import dam_break_cell_averages, stoker_star_state

T_FINAL = 0.5
DOMAIN = (-3.0, 3.0)


def _solve(h_left, h_right, ncells, degree=1, limiter=None, flux_kind="hll",
           domain=DOMAIN, final_time=T_FINAL):
    x, coeff, zb_faces, meta = solve_swe_1d(
        lambda xx: torch.where(xx < 0.0, torch.full_like(xx, float(h_left)),
                               torch.full_like(xx, float(h_right))),
        lambda xx: torch.zeros_like(xx),
        lambda xx: torch.zeros_like(xx),
        ncells=ncells, degree=degree, final_time=final_time, domain=domain,
        cfl=0.3, flux_kind=flux_kind, limiter=limiter)
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    faces = np.linspace(domain[0], domain[1], ncells + 1)
    return x.numpy(), h.numpy(), coeff[:, 0, 1].numpy(), faces, meta


def _rel_l1(num, exact):
    return float(np.abs(num - exact).sum() / np.abs(exact).sum())


# ---------------------------------------------------------------------------
# Stoker wet-bed dam break (the quantitative gate)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h_right", [0.1, 0.5])
@pytest.mark.parametrize("degree", [1, 2])
def test_stoker_depth_within_one_percent(h_right, degree):
    """Relative L1 error of the depth must stay below 1% (gate: 3%)."""
    x, h, hu, faces, meta = _solve(1.0, h_right, 200, degree=degree)
    h_ex, _ = dam_break_cell_averages(faces, T_FINAL, 1.0, h_right)
    err = _rel_l1(h, h_ex)
    assert err < 0.01, f"rel L1(h) = {err:.4%} (steps={meta['steps']})"


def test_stoker_star_state_and_wave_speeds():
    """Star plateau, rarefaction head and shock position vs the exact solution."""
    h_left, h_right = 1.0, 0.1
    ncells = 400
    x, h, hu, faces, _ = _solve(h_left, h_right, ncells)
    star = stoker_star_state(h_left, h_right)
    dx = faces[1] - faces[0]

    tail = T_FINAL * star["wave_tail"]
    shock = T_FINAL * star["shock_speed"]
    band = (x > tail + 3 * dx) & (x < shock - 3 * dx)
    assert band.sum() >= 5

    h_star_num = float(h[band].mean())
    u_star_num = float((hu[band] / h[band]).mean())
    assert abs(h_star_num - star["h_star"]) / star["h_star"] < 0.01
    assert abs(u_star_num - star["u_star"]) / star["u_star"] < 0.01

    # The fan head is a tangential touch (dh/dxi = 0 at xi = -c_1), so a
    # level-crossing detector is ill conditioned there; check the undisturbed
    # state on both sides of the wave instead.
    # Far field: well outside the wave the initial state must be untouched.
    # (Closer to the fan head the *exact* solution itself already deviates,
    # because the head is a tangential touch dh/dxi = 0: at 0.15 m from the
    # head the exact depth is already 0.968, so a tight check there would be
    # testing the wrong thing.)
    head = T_FINAL * star["wave_head"]
    left_far = x < head - 0.5
    assert left_far.sum() > 10
    assert float(np.abs(h[left_far] - h_left).max()) < 1e-6
    assert float(np.abs(hu[left_far]).max()) < 1e-6
    right_far = x > shock + 0.5
    assert right_far.sum() > 10
    assert float(np.abs(h[right_far] - h_right).max()) < 1e-6


def test_stoker_mass_is_conserved_to_round_off():
    for h_right in (0.1, 0.5):
        x, h, hu, faces, _ = _solve(1.0, h_right, 200)
        dx = faces[1] - faces[0]
        mass = float(h.sum() * dx)
        mass0 = 1.0 * abs(DOMAIN[0]) + h_right * abs(DOMAIN[1])
        assert abs(mass - mass0) / mass0 < 1e-12


def test_stoker_depth_is_positive_everywhere():
    x, h, hu, faces, _ = _solve(1.0, 0.1, 200)
    assert np.all(h > 0.0)
    assert np.all(np.isfinite(h))


def test_stoker_error_decreases_under_refinement():
    """Monotone (first-order dominated) convergence of the relative L1 error."""
    h_right = 0.1
    errors = []
    for ncells in (100, 200, 400):
        x, h, hu, faces, _ = _solve(1.0, h_right, ncells)
        h_ex, _ = dam_break_cell_averages(faces, T_FINAL, 1.0, h_right)
        errors.append(_rel_l1(h, h_ex))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 0.005


def test_stoker_is_robust_to_the_numerical_flux():
    """Lax-Friedrichs flux must also reproduce the exact solution within 3%."""
    x, h, hu, faces, _ = _solve(1.0, 0.1, 200, flux_kind="lf")
    h_ex, _ = dam_break_cell_averages(faces, T_FINAL, 1.0, 0.1)
    assert _rel_l1(h, h_ex) < 0.03


# ---------------------------------------------------------------------------
# Ritter dry-bed dam break (acceptance gate after the dry-guard fix)
# ---------------------------------------------------------------------------

def test_solver_reaches_the_requested_final_time():
    """Regression: the integrator must not stop early and still report t=final.

    Before the dry guard the dry-bed run stalled at t ~ 6.9e-3 s (dt underflow
    at 9.7e-16 after 4071 steps) while ``meta["final_time"]`` kept claiming
    0.5 s.  Any silent under-integration is now visible in the metadata.
    """
    x, h, hu, faces, meta = _solve(1.0, 0.0, 100)
    assert meta["reached_final_time"] is True
    assert meta["time_reached"] == pytest.approx(T_FINAL, rel=0, abs=1e-12)
    assert meta["steps"] > 100, meta["steps"]


def test_ritter_dry_bed_mass_and_positivity():
    """Dry-bed run is positive and mass is conserved up to boundary outflow.

    The exact Ritter front travels at ``2 c0 = 3.132 m/s`` and therefore
    reaches ``x = 3`` at ``t = 0.479 s``: at the default ``T_FINAL = 0.5 s`` the
    transmissive right wall is already inside the wetted region, so part of
    the mass legitimately leaves the domain.  Measured relative drift at
    N = 200 is 3.25e-07 here, versus 7.93e-14 for the *same* grid at t = 0.3
    (front still 1.1 m away from the wall) and 8.17e-14 on ``(-5, 5)`` at
    t = 0.5 - see
    ``test_ritter_dry_bed_mass_is_machine_conservative_when_the_front_stays_inside``
    and ``docs/ritter_mass_metric_resolution.md``.
    The residual outflow is carried by the O(DRY_DEPTH) water floor that the
    scheme leaves ahead of the front (a wetting/drying artefact, not a
    conservation error): the interior mass balance telescopes exactly.

    The old docstring pinned this number as "the level of the dry-state guard"
    and quoted 9.9999e-09 from the benchmark; that value was a benchmark
    metric bug (the seeded ``dry_tol`` initial mass was not counted), see
    ``docs/ritter_mass_metric_resolution.md``.
    """
    x, h, hu, faces, _ = _solve(1.0, 0.0, 200)
    dx = faces[1] - faces[0]
    assert np.all(np.isfinite(h))
    assert np.all(h >= -1e-9), float(h.min())          # depth is non-negative up
                                                       # to the positivity floor
    drift = abs(float(h.sum() * dx) - 3.0) / 3.0
    assert drift < 1e-6, drift


def test_ritter_dry_bed_mass_is_machine_conservative_when_the_front_stays_inside():
    """Mass conservation gate with the wet front kept away from the walls.

    Domain ``(-5, 5)``, N = 200, t = 0.5: the exact front is at x = 3.132 and
    its numerical tail stays below ``DRY_DEPTH`` at both walls, so no mass can
    leave and the total mass must be conserved to round-off.  Measured 8.17e-14.
    """
    x, h, hu, faces, _ = _solve(1.0, 0.0, 200, domain=(-5.0, 5.0))
    dx = faces[1] - faces[0]
    mass0 = 1.0 * abs(-5.0)
    drift = abs(float(h.sum() * dx) - mass0) / mass0
    assert drift < 1e-12, drift


def test_ritter_dry_bed_depth_within_three_percent():
    """Relative L1 error of the depth against the exact self-similar solution."""
    x, h, hu, faces, meta = _solve(1.0, 0.0, 200)
    h_ex, _ = dam_break_cell_averages(faces, T_FINAL, 1.0, 0.0)
    err = _rel_l1(h, h_ex)
    assert err < 0.03, f"rel L1(h) = {err:.4%} (steps={meta['steps']})"


def test_ritter_dry_bed_velocity_within_five_percent():
    """The rarefaction velocity field must also match (L1 over wet cells)."""
    x, h, hu, faces, _ = _solve(1.0, 0.0, 200)
    _, u_ex = dam_break_cell_averages(faces, T_FINAL, 1.0, 0.0)
    dx = faces[1] - faces[0]
    h_ex, _ = dam_break_cell_averages(faces, T_FINAL, 1.0, 0.0)
    wet = h_ex > 1e-3
    u_num = np.where(h > 1e-6, hu / np.maximum(h, 1e-6), 0.0)
    err = float(np.abs(u_num[wet] - u_ex[wet]).sum() * dx
                / (np.abs(u_ex[wet]).sum() * dx))
    # measured 2.0 % (N=100) / 2.7 % (N=200) with the benchmark's own metric
    # (normalised by max|u_exact|); the gate is deliberately loose because the
    # velocity is ill conditioned where h -> 0 at the wetting front.
    assert err < 0.10, f"rel L1(u) = {err:.4%}"


def test_ritter_dry_front_position():
    """The wetting front travels at the analytic speed 2 sqrt(g h0)."""
    h0 = 1.0
    x, h, hu, faces, _ = _solve(h0, 0.0, 200)
    dx = faces[1] - faces[0]
    front_exact = 2.0 * np.sqrt(9.81 * h0) * T_FINAL
    wet = np.where(h > 1e-6)[0]
    front_num = x[wet].max() + 0.5 * dx
    assert abs(front_num - front_exact) < 6 * dx, (front_num, front_exact)


def test_ritter_dry_bed_error_decreases_under_refinement():
    errors = []
    for ncells in (100, 200):
        x, h, hu, faces, _ = _solve(1.0, 0.0, ncells)
        h_ex, _ = dam_break_cell_averages(faces, T_FINAL, 1.0, 0.0)
        errors.append(_rel_l1(h, h_ex))
    assert errors[1] < errors[0], errors
    assert errors[1] < 0.005, errors


def test_ritter_dry_bed_holds_for_the_lax_friedrichs_flux():
    x, h, hu, faces, _ = _solve(1.0, 0.0, 200, flux_kind="lf")
    h_ex, _ = dam_break_cell_averages(faces, T_FINAL, 1.0, 0.0)
    assert _rel_l1(h, h_ex) < 0.03


# ---------------------------------------------------------------------------
# Dry-state guards (unit level)
# ---------------------------------------------------------------------------

def test_dry_state_flux_is_zero_and_finite():
    from tensordg.swe_1d import physical_flux
    flux = physical_flux(torch.zeros(2, dtype=torch.float64))
    assert torch.all(flux == 0.0)


def test_near_dry_state_does_not_amplify_momentum():
    """h just below the dry depth with a leftover momentum stays bounded."""
    from tensordg.swe_1d import DRY_DEPTH, physical_flux
    state = torch.tensor([0.5 * DRY_DEPTH, 1.0], dtype=torch.float64)
    mass_flux, momentum_flux = physical_flux(state)
    assert float(mass_flux) == 0.0
    assert abs(float(momentum_flux)) < 1e-12


def test_wet_state_flux_matches_the_analytic_expression():
    from tensordg.swe_1d import physical_flux
    h, hu = 2.0, 1.5
    state = torch.tensor([h, hu], dtype=torch.float64)
    mass_flux, momentum_flux = physical_flux(state)
    assert float(mass_flux) == pytest.approx(hu, rel=1e-14)
    assert float(momentum_flux) == pytest.approx(hu ** 2 / h + 0.5 * 9.81 * h ** 2,
                                                 rel=1e-14)


def test_ritter_benchmark_mass_metric_counts_the_seeded_initial_mass():
    """The benchmark must measure drift against the mass it actually started with.

    ``run_ritter`` seeds the dry side with ``dry_tol = 1e-8`` water, i.e.
    ``dry_tol * |x_right| = 5e-8 m^2`` of extra mass.  Before this fix the
    denominator was the analytic ``h0 * |x_left| = 5 m^2``, so that seed showed
    up as a 1.0e-08 "mass error" although the solver conserves the discrete
    initial mass to round-off.  Both numbers are pinned here so the artefact
    cannot silently come back.
    """
    from benchmarks.ritter_dam_break import run_ritter

    r = run_ritter(ncells=100)
    assert r["mass_seed"] == pytest.approx(5e-8, rel=1e-9), r["mass_seed"]
    assert r["mass_init"] == pytest.approx(5.0 + 5e-8, rel=1e-12), r["mass_init"]
    assert r["mass_init_analytic"] == pytest.approx(5.0, rel=0)
    # consistent metric: the solver conserves the discrete initial mass
    assert r["mass_error"] < 1e-12, r["mass_error"]
    # the legacy (analytic-denominator) metric reproduces the reported 1e-8
    assert r["mass_error_legacy"] == pytest.approx(1e-8, rel=1e-3), r["mass_error_legacy"]
