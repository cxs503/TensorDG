"""Tests for the single-phase plane-Poiseuille baseline skeleton.

Anchor: ``XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETO``

The tests pin four separate claims, and they are written so that a regression in
any one of them is visible on its own:

1. the analytic reference is written in a *declared* wall convention, and the
   two conventions differ by half a cell (the defect the legacy benchmark had);
2. the Guo forcing term supplies exactly ``(1 - omega/2) F`` of momentum per
   step and the ``F/2`` half-step is in the momentum definition;
3. at convergence the numerical profile is an exact parabola: the *shape* error
   is at round-off, all remaining error is a uniform slip that follows the
   closed form ``(48 nu^2 - 1) / L^2``; at ``tau* = 1/2 + 3/sqrt(48)`` even the
   slip vanishes and the solution matches the analytic profile to round-off;
4. the gate is pre-registered: it refuses an out-of-range tolerance, refuses a
   non-finite value, and it fails a transient (unconverged) sample instead of
   passing it.

Runs are kept small on purpose (Ny = 8..16, analytic initial condition) so the
whole file stays inside the local-compute budget; the full ladder lives in
``benchmarks/poiseuille_baseline_run.py``.
"""

import math

import pytest
import torch

from tensorlbm.poiseuille_baseline import (
    BASELINE_SPEC,
    MAX_TOLERANCE,
    PoiseuilleConfig,
    ReL2Gate,
    analytic_profile,
    guo_forcing,
    run,
    simple_forcing,
)
from tensorlbm.lattice import D2Q9

TAU_STAR = 0.5 + 3.0 / math.sqrt(48.0)


def slip_law(tau: float, width: float) -> float:
    nu = (tau - 0.5) / 3.0
    return (48.0 * nu * nu - 1.0) / width ** 2


def rel_l2_vs(profile, cfg_ref, cfg_solid):
    """Relative L2 of a profile against another declared wall convention."""
    ref = analytic_profile(cfg_ref)
    fluid = torch.zeros(cfg_solid.Ny, dtype=torch.bool)
    fluid[1:-1] = True
    p = torch.as_tensor(profile)
    return float(torch.sqrt(torch.mean((p[fluid] - ref[fluid]) ** 2))) / cfg_solid.u_max_analytic


# --------------------------------------------------------------- reference
def test_analytic_profile_declares_its_wall_convention():
    cfg = PoiseuilleConfig(Ny=16, Nx=2, tau=1.0, wall_origin=0.5)
    prof = analytic_profile(cfg)
    width = cfg.width
    # half-way convention: no-slip surfaces at y = 0.5 and y = Ny - 1.5, so the
    # channel is Ny - 2 = 14 lattice units wide and node i sits i - 0.5 from the
    # wall.  The zero is therefore *between* nodes, not on one.
    assert width == 14.0
    fluid = prof[1:-1]
    assert torch.all(fluid > 0)                       # positive inside
    assert prof[0].item() < 0                         # extrapolated behind wall
    assert prof[0].item() == pytest.approx(
        -(cfg.force / (2 * cfg.nu)) * 0.5 * (width + 0.5), rel=1e-12)
    # symmetry about mid-channel (y = 7.5), peak value below the exact u_max
    assert prof[7].item() == pytest.approx(prof[8].item(), rel=1e-14)
    assert prof[8].item() < cfg.u_max_analytic
    assert cfg.u_max_analytic == pytest.approx(
        cfg.force * width ** 2 / (8 * cfg.nu), rel=1e-14)


def test_wall_conventions_differ_by_half_a_cell():
    cfg_half = PoiseuilleConfig(Ny=16, Nx=2, tau=1.0, wall_origin=0.5)
    cfg_node = PoiseuilleConfig(Ny=16, Nx=2, tau=1.0, wall_origin=0.0)
    assert cfg_half.width == cfg_node.width - 1.0
    # the node-centred parabola is written for a 1-lattice-unit wider channel,
    # so its peak is higher: using it against a half-way bounce-back solution
    # injects a systematic shape error, not a small offset
    assert cfg_node.u_max_analytic > cfg_half.u_max_analytic
    fluid = slice(1, -1)
    node_ref = analytic_profile(cfg_node)[fluid]
    assert torch.min(torch.abs(node_ref)).item() > 0.0


# --------------------------------------------------------------- forcing
def test_guo_forcing_supplies_first_order_moment():
    """sum_i e_i F_i must equal (1 - omega/2) F for u = 0 (Guo 1993)."""
    cfg = PoiseuilleConfig(Ny=4, Nx=3, tau=0.8)
    omega = cfg.omega
    ux = torch.zeros((4, 3), dtype=torch.float64)
    fy = torch.zeros((4, 3), dtype=torch.float64)
    src = guo_forcing(ux, fy, cfg.force, omega)
    assert src.shape == (D2Q9.Q, 4, 3)
    # first moment of the source term, per node (sum over the 9 directions)
    mx = (D2Q9.e[:, 0].view(-1, 1, 1) * src).sum(dim=0)
    my = (D2Q9.e[:, 1].view(-1, 1, 1) * src).sum(dim=0)
    assert torch.allclose(mx, torch.full_like(mx, (1.0 - 0.5 * omega) * cfg.force),
                          rtol=1e-12)
    assert torch.allclose(my, torch.zeros_like(my), atol=1e-18)


def test_simple_forcing_lacks_the_half_step_factor():
    """The legacy term supplies F of momentum, i.e. no (1 - omega/2) and no u."""
    cfg = PoiseuilleConfig(Ny=4, Nx=3, tau=0.8)
    src = simple_forcing(cfg.force)
    mx = float((D2Q9.e[:, 0].view(-1, 1, 1) * src).sum(dim=0).mean())
    assert mx == pytest.approx(cfg.force, rel=1e-12)
    assert mx != pytest.approx((1.0 - 0.5 * cfg.omega) * cfg.force, rel=1e-3)


# --------------------------------------------------------------- physics
def test_baseline_profile_is_an_exact_parabola():
    """At convergence the shape error is round-off; only a slip is left."""
    cfg = PoiseuilleConfig(Nx=2, Ny=8, tau=1.0, init="analytic",
                           residual_tol=1e-11, max_steps=20_000, check_every=50)
    res = run(cfg)
    assert res["convergence"]["converged"] is True
    assert res["errors"]["rel_l2_shape"] < 1e-10
    # and the remaining error is the uniform slip predicted by the closed form
    law = slip_law(cfg.tau, cfg.width)
    assert res["errors"]["rel_l2"] == pytest.approx(abs(law), rel=1e-3)


def test_slip_follows_closed_form_at_second_width():
    cfg = PoiseuilleConfig(Nx=2, Ny=16, tau=0.8, init="analytic",
                           residual_tol=1e-11, max_steps=40_000, check_every=100)
    res = run(cfg)
    assert res["convergence"]["converged"] is True
    law = slip_law(cfg.tau, cfg.width)
    assert res["errors"]["rel_l2"] == pytest.approx(abs(law), rel=1e-3)
    assert res["errors"]["offset_over_force"] == pytest.approx(
        6.0 * cfg.nu - 1.0 / (8.0 * cfg.nu), rel=1e-6)


def test_slip_free_tau_star_reaches_roundoff():
    """tau* = 1/2 + 3/sqrt(48) makes 48 nu^2 = 1, so the slip cancels exactly."""
    cfg = PoiseuilleConfig(Nx=2, Ny=8, tau=TAU_STAR, init="analytic",
                           residual_tol=1e-12, max_steps=20_000, check_every=50)
    res = run(cfg)
    assert res["convergence"]["converged"] is True
    assert res["errors"]["rel_l2"] < 1e-9
    assert res["errors"]["rel_l2_shape"] < 1e-9
    assert res["errors"]["rel_u_max"] == pytest.approx(0.0, abs=1e-9)


def test_relative_error_is_force_independent():
    """The residual error is O(F), so its *relative* size does not move with F."""
    errs = []
    for F in (1e-5, 1e-4):
        cfg = PoiseuilleConfig(Nx=2, Ny=8, tau=1.0, force=F, init="analytic",
                               residual_tol=1e-11, max_steps=20_000, check_every=50)
        errs.append(run(cfg)["errors"]["rel_l2"])
    assert errs[0] == pytest.approx(errs[1], rel=1e-6)


def test_initial_condition_does_not_change_the_steady_state():
    common = dict(Nx=2, Ny=8, tau=1.0, residual_tol=1e-11, max_steps=20_000,
                  check_every=50)
    a = run(PoiseuilleConfig(init="rest", **common))
    b = run(PoiseuilleConfig(init="analytic", **common))
    pa = torch.tensor(a["profile_u"])
    pb = torch.tensor(b["profile_u"])
    assert float((pa - pb).abs().max()) / a["errors"]["u_max_analytic"] < 1e-6
    assert a["convergence"]["steps"] != b["convergence"]["steps"]


def test_mass_is_conserved():
    cfg = PoiseuilleConfig(Nx=2, Ny=8, tau=1.0, init="rest",
                           residual_tol=1e-11, max_steps=20_000, check_every=50)
    res = run(cfg)
    assert abs(res["mass_drift"]) < 1e-8


# --------------------------------------------------------------- defect
def test_node_centred_reference_reproduces_the_delivered_defect():
    """Same converged solution, two references: the node-centred one fails."""
    cfg = PoiseuilleConfig(Nx=2, Ny=16, tau=1.0, init="analytic",
                           residual_tol=1e-11, max_steps=40_000, check_every=100)
    res = run(cfg)
    cfg_node = PoiseuilleConfig(Nx=2, Ny=16, tau=1.0, wall_origin=0.0)
    err_halfway = res["errors"]["rel_l2"]
    err_node = rel_l2_vs(res["profile_u"], cfg_node, cfg)
    # halfway reference: inside the 1 % gate; node-centred: far outside it
    assert err_halfway < 0.01
    assert err_node > 0.01
    assert err_node > 3.0 * err_halfway


def test_forcing_schemes_differ_only_by_the_half_force_momentum():
    """guo vs the legacy term: same shape, slip differs by exactly F/2.

    Measured, not assumed: with ``u = (sum e f + F/2)/rho`` the Guo steady state
    is the legacy steady state shifted by ``F/2`` in velocity, so the two
    schemes are *not* separated by a first-order shape error — the legacy
    benchmark's 11-14 % came from the reference convention and from running
    2000 steps, and this test pins that attribution.
    """
    common = dict(Nx=2, Ny=16, tau=1.0, init="analytic", residual_tol=1e-11,
                  max_steps=40_000, check_every=100)
    legacy = run(PoiseuilleConfig(forcing="simple", **common))
    guo = run(PoiseuilleConfig(forcing="guo", **common))
    assert legacy["errors"]["rel_l2_shape"] < 1e-9      # both shapes exact
    assert guo["errors"]["rel_l2_shape"] < 1e-9
    delta = guo["errors"]["offset_over_force"] - legacy["errors"]["offset_over_force"]
    assert delta == pytest.approx(0.5, rel=1e-6)


# --------------------------------------------------------------- gate
def test_gate_refuses_tolerance_above_ecosystem_cap():
    with pytest.raises(ValueError):
        ReL2Gate(tolerance=MAX_TOLERANCE + 1e-9)
    with pytest.raises(ValueError):
        ReL2Gate(tolerance=0.0)
    with pytest.raises(ValueError):
        ReL2Gate(tolerance=-0.01)
    assert ReL2Gate(tolerance=MAX_TOLERANCE).tolerance == MAX_TOLERANCE


def test_gate_cannot_pass_a_missing_or_nonfinite_value():
    gate = ReL2Gate(tolerance=0.01)
    assert gate.check(None).passed is False
    assert gate.check(float("nan")).passed is False
    assert gate.check(float("inf")).passed is False


def test_gate_fails_an_unconverged_transient():
    cfg = PoiseuilleConfig(Nx=2, Ny=16, tau=1.0, max_steps=100, check_every=100,
                           residual_tol=0.0)
    res = run(cfg)
    gate = ReL2Gate(tolerance=BASELINE_SPEC["gate"]["tolerance"])
    assert res["convergence"]["converged"] is False
    assert gate.check(res["errors"]["rel_l2"]).passed is False


def test_baseline_spec_is_preregistered_and_legal():
    gate = BASELINE_SPEC["gate"]
    assert 0.0 < gate["tolerance"] <= MAX_TOLERANCE
    assert gate["basis"]
    assert BASELINE_SPEC["canonical_case"]["forcing"] == "guo"
    assert BASELINE_SPEC["canonical_case"]["wall_origin"] == 0.5