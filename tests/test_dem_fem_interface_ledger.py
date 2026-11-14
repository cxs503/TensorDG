"""Tests for the DEM-FEM interface ledger (anchor ``GAP-ICE-LEDGER-PHYSICS``).

The assertions are on *invariants and analytic references*, not on execution:

* the FEM participant reproduces the analytic statics exactly and converges to
  the analytic clamped-free / free-free eigenfrequencies at O(h^2);
* the DEM contact law reproduces the analytic harmonic oscillator and the
  analytic two-body impact in the stiff-rod limit;
* the interface ledger closes on physical residuals <= 1e-3 with a FEM
  participant present;
* the two injected defects break the physical gate while the backend port
  parity number stays below the legacy 5e-3 threshold -- i.e. parity is
  demonstrably not conservation evidence.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg.dem_fem_interface_ledger import (
    RESIDUAL_GATE,
    CouplingConfig,
    FemRod,
    ModalReference,
    ParticleSpec,
    RodSpec,
    backend_port_parity,
    run_coupling,
    verify_contact_oscillator,
    verify_fem_eigenfrequencies,
    verify_fem_static,
)

LEGACY_PARITY_GATE = 5.0e-3


# ---------------------------------------------------------------- FEM side
def test_fem_static_elongation_is_exact():
    """P1 rod reproduces ``F L / (E A)`` exactly for a self-equilibrated pair."""
    rod = FemRod(RodSpec(), 16)
    res = verify_fem_static(rod, load=2.0e4)
    assert res["relative_error"] < 1e-12, res


def test_fem_eigenfrequencies_converge_to_analytic():
    """Relative error of the first three free-free frequencies falls like h^2."""
    sweep = verify_fem_eigenfrequencies(RodSpec(), n_elems=(8, 16, 32))
    errs = [max(r["relative_errors"]) for r in sweep]
    assert errs[0] > errs[1] > errs[2], errs
    assert errs[-1] < 1e-2, errs
    # second-order convergence
    ratio = errs[0] / errs[1]
    assert 2.5 < ratio < 6.0, ratio


def test_modal_reference_handles_rigid_mode():
    """The free-free system has a zero mode; the exact integrator must cope."""
    rod = FemRod(RodSpec(), 8)
    ref = ModalReference(rod)
    assert float(ref.omega.min()) < 1e-6 * float(ref.omega.max())
    u0 = torch.zeros(rod.nfree, dtype=torch.float64)
    v0 = torch.zeros(rod.nfree, dtype=torch.float64)
    v0[0] = 1.0
    load = torch.zeros(8, rod.nfree, dtype=torch.float64)
    out = ref.solve(u0, v0, load, 1e-5)
    assert torch.isfinite(out["u"]).all()
    assert torch.isfinite(out["v"]).all()


# ---------------------------------------------------------------- DEM side
def test_contact_oscillator_matches_analytic():
    res = verify_contact_oscillator(dt=1.0e-7)
    assert res["relative_error_peak_force"] < 1e-4, res
    assert res["relative_error_duration"] < 1e-3, res


# ------------------------------------------------------- coupled system
@pytest.fixture(scope="module")
def healthy():
    return run_coupling(CouplingConfig(dt=2.0e-6, t_end=2.0e-3))


def test_fem_participant_is_present_and_carries_load(healthy):
    led = healthy["ledger"]
    assert led["fem_participant_dofs"] > 0
    assert led["fem_participant_mass"] > 0.0
    # the FEM participant must actually absorb interface work / store energy
    assert led["rod_peak_strain_energy"] > 0.0
    assert abs(led["impulse_fem"]) > 0.0
    assert abs(led["work_fem"]) > 0.0
    # FEM and DEM impulses are of the same order (a real exchange happened)
    assert 0.5 < abs(led["impulse_fem"]) / abs(led["impulse_dem"]) < 2.0


def test_healthy_ledger_passes_physical_gate(healthy):
    led = healthy["ledger"]
    assert led["momentum_residual"] <= RESIDUAL_GATE
    assert led["energy_residual"] <= RESIDUAL_GATE
    assert led["interface_impulse_residual"] <= RESIDUAL_GATE
    assert led["interface_work_residual"] <= RESIDUAL_GATE
    assert led["passes_gate"] is True
    # momentum is conserved to round-off for a monolithic RK4 step
    assert led["momentum_residual"] < 1e-10


def test_parity_is_small_for_a_healthy_run(healthy):
    par = backend_port_parity(healthy)
    assert par["label"] == "backend port parity"
    assert par["value"] < LEGACY_PARITY_GATE


def test_force_split_defect_breaks_gate_but_not_parity():
    """A Newton's-third-law violation is invisible to backend parity."""
    res = run_coupling(CouplingConfig(dt=1.0e-6, t_end=2.0e-3, force_split=0.99))
    led = res["ledger"]
    assert led["interface_impulse_residual"] > RESIDUAL_GATE
    assert led["momentum_residual"] > RESIDUAL_GATE
    assert led["passes_gate"] is False
    # ... while the parity number would have passed the legacy gate
    assert backend_port_parity(res)["value"] < LEGACY_PARITY_GATE


def test_staggered_coupling_leaks_energy_but_parity_passes():
    stag = run_coupling(CouplingConfig(dt=1.0e-6, t_end=2.0e-3, frozen_coupling=True))
    led = stag["ledger"]
    assert led["energy_residual"] > RESIDUAL_GATE
    assert led["passes_gate"] is False
    assert backend_port_parity(stag)["value"] < LEGACY_PARITY_GATE


def test_staggered_leak_scales_with_dt():
    coarse = run_coupling(CouplingConfig(dt=5.0e-6, t_end=2.0e-3,
                                         frozen_coupling=True))["ledger"]
    fine = run_coupling(CouplingConfig(dt=1.0e-6, t_end=2.0e-3,
                                       frozen_coupling=True))["ledger"]
    assert coarse["energy_residual"] > fine["energy_residual"]
    # explicit partitioning is first order in dt -> factor ~5 for a 5x step
    ratio = coarse["energy_residual"] / max(fine["energy_residual"], 1e-300)
    assert 3.0 < ratio < 8.0, ratio


def test_stiff_rod_limit_matches_two_body_analytic():
    spec = RodSpec(young=9.0e11)
    cfg = CouplingConfig(rod=spec, n_elem=16, dt=5.0e-7, t_end=3.0e-3)
    out = run_coupling(cfg, keep_history=True)
    big_m = spec.total_mass
    mass, k, v0 = 5.0, cfg.contact.stiffness, cfg.particles.initial_velocity
    w_rel = math.sqrt(k * (mass + big_m) / (mass * big_m))
    peak = float(out["history"]["force"].max())
    analytic_peak = v0 * math.sqrt(k * mass * big_m / (mass + big_m))
    assert abs(peak - analytic_peak) / analytic_peak < 0.01
    assert out["ledger"]["energy_residual"] <= RESIDUAL_GATE
    assert abs(math.pi / w_rel - 1.78e-3) < 1.0e-4