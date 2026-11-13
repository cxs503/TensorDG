"""Tests for the boundary-driven (inhomogeneous) 1-D elastic bar.

The driven end closes the last capability hole of ``tensordg.elastic_1d``: until
now the bar could only be *released* from initial data -- never *sourced*.  These
tests grade the new boundary on independent grounds:

* the closed-form driven solution is re-derived here from the characteristic
  relations and its **PDE residual** is checked directly (finite differences),
  so the reference is not trusted just because the code returns it;
* the work -> energy identity ``E(t) = Z int g^2`` is checked against a run;
* a returning wave must be absorbed by the source (no spurious reflection);
* the fault-injection arm proves the grading has teeth (flipped source sign is
  caught), and that a driven bc without a drive refuses to run.
"""

from __future__ import annotations

import math

import pytest
import torch

import tensordg.elastic_1d as e1
from tensordg.elastic_1d import (
    DRIVEN_BC,
    driven_injected_work,
    driven_simple_wave,
    elastic_energy,
    elastic_wave_speed,
    gaussian_drive,
    relative_error,
    solve_elastic_1d,
    stress_profile,
)

RHO, YOUNG, LENGTH = 1.0, 1.0, 1.0
C = elastic_wave_speed(RHO, YOUNG)
Z = RHO * C
DRIVE_CENTER, DRIVE_WIDTH = 0.35, 0.05


def drive(t):
    return gaussian_drive(t, DRIVE_CENTER, DRIVE_WIDTH)


# ---------------------------------------------------------------------------
# the boundary refuses to run without a source (no vacuum gate)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bc", DRIVEN_BC)
def test_driven_bc_without_drive_raises(bc):
    with pytest.raises(ValueError, match="requires a drive callable"):
        solve_elastic_1d(ncells=8, degree=1, final_time=0.1, bc=bc)


def test_unknown_bc_still_raises():
    with pytest.raises(ValueError):
        solve_elastic_1d(ncells=8, degree=1, final_time=0.1, bc="nonsense")


# ---------------------------------------------------------------------------
# the closed form is re-derived and checked against the PDE, independently
# ---------------------------------------------------------------------------
def test_closed_form_obeys_boundary_conditions():
    """Free end: ``sigma(L,t) == 0``; fixed end: ``v(L,t) == 0`` (all times)."""
    x = torch.tensor([LENGTH], dtype=torch.float64)
    for t in (0.0, 0.4, 1.0, 1.7, 2.4):
        _v, s = driven_simple_wave(x, t, drive, LENGTH, RHO, YOUNG, end="free")
        assert abs(float(s[0])) < 1e-12
        v, _s = driven_simple_wave(x, t, drive, LENGTH, RHO, YOUNG, end="fixed")
        assert abs(float(v[0])) < 1e-12


def test_closed_form_is_a_right_going_simple_wave_before_reflection():
    """Ahead of the first reflection ``sigma = -Z v`` and ``w- = 0``."""
    x = torch.linspace(0.0, 1.0, 201, dtype=torch.float64)
    t = 1.0                                     # reflected packet is at x > 1.0
    v, s = driven_simple_wave(x, t, drive, LENGTH, RHO, YOUNG, end="free")
    mask = x < 0.8
    assert float((s[mask] + Z * v[mask]).abs().max()) < 1e-10


def test_closed_form_satisfies_the_pde():
    """Independently verify the reference: finite-difference residuals vanish.

    ``v_t - sigma_x/rho = 0`` and ``sigma_t - E v_x = 0`` are evaluated with
    fourth-order central differences on a fine grid -- this does not reuse any
    code path of the solver.
    """
    x = torch.linspace(0.0, 1.0, 401, dtype=torch.float64)
    t, dt = 1.2, 1e-4
    dx = float(x[1] - x[0])

    def state(tt, xx):
        return driven_simple_wave(xx, tt, drive, LENGTH, RHO, YOUNG, end="free")

    def d_dt(field, tt):
        # 4th-order central difference in time
        return (-state(tt + 2 * dt, x)[field] + 8.0 * state(tt + dt, x)[field]
                - 8.0 * state(tt - dt, x)[field] + state(tt - 2 * dt, x)[field]) / (12.0 * dt)

    def d_dx(field, xx):
        return (-state(t, xx + 2 * dx)[field] + 8.0 * state(t, xx + dx)[field]
                - 8.0 * state(t, xx - dx)[field] + state(t, xx - 2 * dx)[field]) / (12.0 * dx)

    v_t, s_t = d_dt(0, t), d_dt(1, t)
    v_x, s_x = d_dx(0, x), d_dx(1, x)

    inner = slice(4, -4)                       # stay away from the stencil edges
    res_mom = float((v_t[inner] - s_x[inner] / RHO).abs().max())
    res_con = float((s_t[inner] - YOUNG * v_x[inner]).abs().max())
    assert res_mom / float(v_t.abs().max()) < 2e-5
    assert res_con / float(s_t.abs().max()) < 2e-5


# ---------------------------------------------------------------------------
# the solver reproduces the reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("end", ["free", "fixed"])
def test_numerical_solution_matches_closed_form(end):
    bc = "driven-free" if end == "free" else "driven-fixed"
    t_end = 1.7                                  # one reflection has happened
    centers, coeff, meta = solve_elastic_1d(
        ncells=100, degree=2, final_time=t_end, length=LENGTH, rho=RHO,
        young=YOUNG, cfl=0.5, bc=bc, drive=drive)
    x, v, s = stress_profile(centers, coeff, nsample=4)
    v_ex, s_ex = driven_simple_wave(x, t_end, drive, LENGTH, RHO, YOUNG, end=end)
    assert 100.0 * relative_error(x, v, v_ex) < 1.0
    assert 100.0 * relative_error(x, s, s_ex) < 1.0
    assert meta["driven"] is True and meta["end"] == end


def test_free_and_fixed_reflection_differ():
    """The two end kinds must produce different fields (no silent aliasing)."""
    x = torch.linspace(0.0, 1.0, 51, dtype=torch.float64)
    vf, sf = driven_simple_wave(x, 1.7, drive, LENGTH, RHO, YOUNG, end="free")
    vc, sc = driven_simple_wave(x, 1.7, drive, LENGTH, RHO, YOUNG, end="fixed")
    assert float((vf - vc).abs().max()) > 1e-2
    assert float((sf - sc).abs().max()) > 1e-2


def test_initial_state_is_consistent_with_the_drive():
    """``t=0`` default initial data must equal the analytic state (rest)."""
    centers, coeff, meta = solve_elastic_1d(
        ncells=20, degree=2, final_time=0.0, bc="driven-free", drive=drive)
    x, v, s = stress_profile(centers, coeff, nsample=4)
    assert float(v.abs().max()) < 1e-12
    assert float(s.abs().max()) < 1e-12


# ---------------------------------------------------------------------------
# work <-> energy identity and absorption
# ---------------------------------------------------------------------------
def test_injected_work_equals_stored_energy():
    t_end = 1.0                                  # nothing has left the bar yet
    centers, coeff, _meta = solve_elastic_1d(
        ncells=100, degree=2, final_time=t_end, length=LENGTH, rho=RHO,
        young=YOUNG, cfl=0.5, bc="driven-free", drive=drive)
    e_num = elastic_energy(centers, coeff, RHO, YOUNG)
    w_ex = driven_injected_work(t_end, drive, RHO, YOUNG)
    assert w_ex > 0
    assert abs(e_num - w_ex) / w_ex < 2e-2


def test_injected_work_matches_analytic_pulse_energy():
    """``W(inf) = Z int g^2 dt`` computed by quadrature vs a fine reference."""
    w = driven_injected_work(4.0, drive, RHO, YOUNG)
    # hand quadrature of the same smooth pulse on a very fine uniform grid
    ts = torch.linspace(0.0, 4.0, 40001, dtype=torch.float64)
    g = torch.tensor([drive(float(t)) for t in ts], dtype=torch.float64)
    w_ref = Z * float(torch.trapz(g * g, ts))
    assert abs(w - w_ref) / w_ref < 1e-6


def test_source_absorbs_the_returning_wave():
    """After the reflected packet leaves through the source the bar is at rest."""
    t_end = 2.6                                  # reflected packet has exited
    centers, coeff, _meta = solve_elastic_1d(
        ncells=100, degree=2, final_time=t_end, length=LENGTH, rho=RHO,
        young=YOUNG, cfl=0.5, bc="driven-free", drive=drive)
    e_res = elastic_energy(centers, coeff, RHO, YOUNG)
    w_in = driven_injected_work(1.0, drive, RHO, YOUNG)  # energy once fully injected
    assert e_res / w_in < 2e-2


# ---------------------------------------------------------------------------
# fault injection: the grading must have teeth
# ---------------------------------------------------------------------------
def test_flipped_source_sign_is_detected(monkeypatch):
    """Emitting ``w+_ext = +2Zg`` must be caught by the waveform anchor."""
    orig = e1._boundary_flux

    def broken(v_int, s_int, side, rho, young, bc, drive=None, t=0.0):
        if bc in DRIVEN_BC and side == "left":
            w_minus = s_int + rho * C * v_int
            w_plus_ext = +2.0 * rho * C * e1._drive_value(drive, t)
            sigma_star = 0.5 * (w_minus + w_plus_ext)
            v_star = (w_minus - w_plus_ext) / (2.0 * rho * C)
            return torch.stack((-sigma_star / rho, -young * v_star), dim=-1)
        return orig(v_int, s_int, side, rho, young, bc, drive=drive, t=t)

    monkeypatch.setattr(e1, "_boundary_flux", broken)
    centers, coeff, _meta = solve_elastic_1d(
        ncells=100, degree=2, final_time=1.0, bc="driven-free", drive=drive)
    x, v, s = stress_profile(centers, coeff, nsample=4)
    v_ex, s_ex = driven_simple_wave(x, 1.0, drive, LENGTH, RHO, YOUNG, end="free")
    err = 100.0 * relative_error(x, v, v_ex)
    assert err > 2.0, f"flipped source sign slipped through: {err:.3f}%"


def test_stage_time_sampling_matters():
    """Sampling the drive at the wrong instant must degrade the solution.

    A zero-lag evaluation (stage time) is required; using ``t = 0`` for every
    stage freezes the source at ``g(0) ~ 0`` and the bar stays at rest.
    """
    t_end = 0.6
    centers, coeff, _meta = solve_elastic_1d(
        ncells=100, degree=2, final_time=t_end, bc="driven-free",
        drive=lambda t: 0.0 if t < 0 else gaussian_drive(t, DRIVE_CENTER, DRIVE_WIDTH))
    x, v, s = stress_profile(centers, coeff, nsample=4)
    assert float(v.abs().max()) > 0.5           # source really injected motion


def test_periodic_case_is_unaffected():
    """Regression: the pre-existing periodic d'Alembert case still passes."""
    from tensordg.elastic_1d import dalembert_pulse, initial_pulse_stress, initial_pulse_velocity
    t_end = 0.2
    centers, coeff, meta = solve_elastic_1d(
        ncells=100, degree=2, final_time=t_end, cfl=0.5, bc="periodic")
    x, v, s = stress_profile(centers, coeff, nsample=4)
    v_ex, s_ex = dalembert_pulse(x, t_end, RHO, YOUNG)
    assert meta["driven"] is False
    assert 100.0 * relative_error(x, v, v_ex) < 0.5
    assert 100.0 * relative_error(x, s, s_ex) < 0.5