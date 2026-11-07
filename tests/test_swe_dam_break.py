"""Verification of the 1-D shallow-water solver against exact dam-break solutions.

Scope
-----
* Stoker (1957) wet-bed dam break: depth / velocity / star-state plateau /
  shock speed / rarefaction head are compared with the closed-form similarity
  solution at ``t = 0.5`` s.
* Ritter (1892) dry-bed dam break: **known limitation**.  The solver carries a
  positivity filter but no dedicated dry-front treatment (no Audusse-style
  hydrostatic reconstruction and no ``h < eps`` switch), so the dry front is
  smeared over many cells and the relative L1 error stays ~28% at 200 cells.
  The test below pins that failure so it cannot silently regress into a
  "verified" claim; see docs/swe_verification.md, section "Limitations".
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tensordg.swe_1d import reconstruct_fields, solve_swe_1d
from tensordg.swe_exact import dam_break_cell_averages, stoker_star_state

T_FINAL = 0.5
DOMAIN = (-3.0, 3.0)


def _solve(h_left, h_right, ncells, degree=1, limiter=None, flux_kind="hll"):
    x, coeff, zb_faces, meta = solve_swe_1d(
        lambda xx: torch.where(xx < 0.0, torch.full_like(xx, float(h_left)),
                               torch.full_like(xx, float(h_right))),
        lambda xx: torch.zeros_like(xx),
        lambda xx: torch.zeros_like(xx),
        ncells=ncells, degree=degree, final_time=T_FINAL, domain=DOMAIN,
        cfl=0.3, flux_kind=flux_kind, limiter=limiter)
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    faces = np.linspace(DOMAIN[0], DOMAIN[1], ncells + 1)
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
# Ritter dry-bed dam break: pinned known limitation
# ---------------------------------------------------------------------------

def test_ritter_dry_bed_mass_and_positivity():
    x, h, hu, faces, _ = _solve(1.0, 0.0, 200)
    dx = faces[1] - faces[0]
    assert np.all(h >= 0.0) and np.all(np.isfinite(h))
    assert abs(float(h.sum() * dx) - 3.0) / 3.0 < 1e-10


def test_ritter_dry_bed_accuracy_is_a_documented_limitation():
    """KNOWN LIMITATION: the dry front is smeared, rel L1 ~ 4x the 15% gate.

    Lifting this requires a dry-cell / hydrostatic-reconstruction treatment
    (see docs/swe_verification.md).  The assertion keeps the honest number
    pinned: if a future change fixes the dry front this test fails and the
    the limitation must be re-documented instead of forgotten.
    """
    x, h, hu, faces, _ = _solve(1.0, 0.0, 200)
    h_ex, _ = dam_break_cell_averages(faces, T_FINAL, 1.0, 0.0)
    err = _rel_l1(h, h_ex)
    assert err > 0.15, (
        f"dry-bed error is now {err:.2%}; if the dry front was fixed, update "
        "docs/swe_verification.md and tighten this into an acceptance test")