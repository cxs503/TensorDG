"""Tests for the Taylor-Green physics audit (XVAL host-staged halo close-out).

The audit resolves the honest FAIL recorded by mission m18373
(``XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO``): the container solver was
blamed for a 15-85 % Taylor-Green decay error, but the defect is in the
*initial condition* of the cross-validation driver (wavenumber applied twice).
These tests pin both facts:

* the corrected Taylor-Green field is divergence-free, zero-mean, of the right
  energy, and the D3Q19 BGK solver decays it at the analytic rate (tight arm);
* the malformed field of the driver reproduces the reported artefact
  (regression fixture);
* the subject implementation (``tensorlbm.host_staged_halo``) reproduces the
  oracle field to machine precision when both are given the corrected field,
  which is what attributes the FAIL to the IC rather than to the solver or the
  host-staged transport.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.tg_physics_audit import (
    IndependentD3Q19BGK,
    analytic_decay_rate,
    audit_case,
    dominant_modes,
    mode_amplitude,
    taylor_green_velocity,
    taylor_green_velocity_double_scaled,
)

U0 = 0.05
PORT_DEFAULT = "/root/tensor-ecosystem/TensorLBM/src/tensorlbm/host_staged_halo.py"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _load_port():
    src = os.environ.get("TENSORLBM_PORT_SRC", PORT_DEFAULT)
    path = Path(src) / "tensorlbm" / "host_staged_halo.py" if Path(src).is_dir() else Path(src)
    if not path.exists():
        pytest.skip(f"subject implementation not available at {path}")
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("_port_host_staged_halo", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


port = pytest.fixture(scope="module")(_load_port)


def _spectral_divergence(u: np.ndarray, n: int) -> float:
    kx = 2.0 * np.pi * np.fft.fftfreq(n)
    ux_hat = np.fft.fftn(u[0])
    uy_hat = np.fft.fftn(u[1])
    div_hat = 1j * kx[None, :] * ux_hat + 1j * kx[:, None] * uy_hat
    return float(np.abs(div_hat).max())


# ---------------------------------------------------------------------------
# 1. the corrected initial condition is a Taylor-Green vortex
# ---------------------------------------------------------------------------


def test_corrected_ic_properties():
    n = 16
    u = taylor_green_velocity(n, n, n, U0)
    assert u.shape == (3, n, n, n)
    # zero net momentum (the malformed field has 1.08e-2 of drift)
    assert abs(u[0].mean()) < 1e-12 and abs(u[1].mean()) < 1e-12
    # kinetic energy of the IC is exactly u0^2/4
    assert abs(0.5 * (u**2).sum(axis=0).mean() - U0**2 / 4.0) < 1e-15
    # discretely divergence-free
    assert _spectral_divergence(u, n) / (U0 * n**2) < 1e-12
    # energy lives in the four (+-1, +-1) modes only
    modes = dict(dominant_modes(u, top=4))
    for m in ((0, 1, 1), (0, 1, n - 1), (0, n - 1, 1), (0, n - 1, n - 1)):
        assert modes[m] == pytest.approx(0.25, abs=1e-6), modes
    # u0/4 per component (u_x and u_y) -> sqrt(2)*u0/4 combined
    assert mode_amplitude(u, (0, 1, 1)) == pytest.approx(U0 * math.sqrt(2) / 4.0, rel=1e-12)


def test_malformed_ic_reproduces_driver_defect():
    """The driver's field is sin(k^2 i): not zero-mean, wrong energy, wrong modes."""
    n = 16
    u_bad = taylor_green_velocity_double_scaled(n, n, n, U0)
    # net momentum drift of 1.08e-2 in lattice units -> conserved forever
    assert u_bad[0].mean() == pytest.approx(0.0108, abs=1e-3)
    ke_bad = 0.5 * (u_bad**2).sum(axis=0).mean()
    assert abs(ke_bad - U0**2 / 4.0) / (U0**2 / 4.0) > 0.02
    modes = dict(dominant_modes(u_bad, top=4))
    assert (0, 0, 0) in modes  # net momentum drift already carries 19 % of the energy
    assert modes[(0, 0, 0)] > 0.1
    # ... and it takes over once the physical modes have decayed (see the
    # late-time spectrum in outputs/tg_physics_audit/figures/fig3_spectrum.png)
    sim = IndependentD3Q19BGK(n, nu=0.02)
    sim.set_velocity(u_bad)
    sim.step(300)
    _, u_late = sim.moments()
    assert dict(dominant_modes(u_late, top=1))[(0, 0, 0)] > 0.5


# ---------------------------------------------------------------------------
# 2. the solver decays the corrected field at the analytic rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nu", [0.02, 0.05])
def test_modal_decay_matches_analytic(nu):
    n = 16
    row = audit_case(n, nu, u0=U0, steps=300, fit_window=(40, 280), stride=20)
    theory = analytic_decay_rate(nu, 2.0 * math.pi / n, "velocity")
    assert row["gamma_modal"] == pytest.approx(theory, rel=0.03)
    assert row["gamma_modal_rel_err"] < 0.03


@pytest.mark.parametrize("nu", [0.02, 0.05])
def test_ke_decay_matches_analytic(nu):
    n = 16
    row = audit_case(n, nu, u0=U0, steps=300, fit_window=(40, 280), stride=20)
    theory = analytic_decay_rate(nu, 2.0 * math.pi / n, "kinetic_energy")
    assert row["gamma_ke"] == pytest.approx(theory, rel=0.03)
    assert row["gamma_ke_rel_err"] < 0.03


def test_decay_rate_scales_with_viscosity():
    """The flat-in-nu signature of the FAIL must be gone for the corrected field."""
    n = 16
    kw = dict(u0=U0, steps=300, fit_window=(40, 280), stride=20)
    g1 = audit_case(n, 0.02, **kw)["gamma_modal"]
    g2 = audit_case(n, 0.04, **kw)["gamma_modal"]
    assert g2 / g1 == pytest.approx(2.0, rel=0.03)


def test_rate_is_grid_independent():
    kw = dict(u0=U0, steps=240, fit_window=(40, 220), stride=20)
    g16 = audit_case(16, 0.02, **kw)["gamma_modal"]
    g24 = audit_case(24, 0.02, **kw)["gamma_modal"]
    t16 = analytic_decay_rate(0.02, 2.0 * math.pi / 16, "velocity")
    t24 = analytic_decay_rate(0.02, 2.0 * math.pi / 24, "velocity")
    # each matches its own analytic rate to <3 %
    assert abs(g16 - t16) / t16 < 0.03
    assert abs(g24 - t24) / t24 < 0.03


def test_malformed_ic_fails_the_physics_arm():
    """Regression: the vendor driver IC keeps the recorded honest FAIL reproducible."""
    row = audit_case(16, 0.02, u0=U0, steps=300, fit_window=(40, 280), stride=20,
                     malformed=True)
    assert row["gamma_modal_rel_err"] > 0.15     # exceeds acceptance
    assert row["gamma_ke_rel_err"] > 0.15        # what the driver reported
    assert row["mean_momentum0"] > 1e-3          # the conserved drift


def test_hardcoded_k0_would_bias_the_rate():
    """Secondary defect: the driver compares against u0^2/4 instead of measuring K0."""
    good = audit_case(16, 0.02, u0=U0, steps=120, fit_window=(40, 100), stride=20)
    bad = audit_case(16, 0.02, u0=U0, steps=120, fit_window=(40, 100), stride=20,
                     malformed=True)
    assert good["k0_rel_err"] < 1e-12
    assert bad["k0_rel_err"] > 0.02


# ---------------------------------------------------------------------------
# 3. the subject implementation == the oracle (attributes the FAIL to the IC)
# ---------------------------------------------------------------------------


def test_port_matches_oracle_on_corrected_field(port):
    """Host-staged, decomposed port vs independent NumPy oracle: machine precision."""
    import torch

    n, nu, steps = 16, 0.02, 20
    u = taylor_green_velocity(n, n, n, U0)
    sim = port.HostStagedD3Q19(
        port.HostStagedLayout(n, n, n, 2, 1, 2), nu=nu, dtype=torch.float64
    )
    sim.set_velocity(torch.tensor(u, dtype=torch.float64))
    sim.step(steps)
    oracle = IndependentD3Q19BGK(n, nu=nu)
    oracle.set_velocity(u)
    oracle.step(steps)
    rho_o, u_o = oracle.moments()
    assert np.abs(sim.project_velocity().numpy() - u_o).max() < 1e-12
    assert np.abs(sim.project_density().numpy() - rho_o).max() < 1e-12


def test_port_meets_acceptance_with_corrected_ic(port):
    """The container solver passes the physics arm once the IC is fixed."""
    import torch

    n, nu, steps = 16, 0.02, 200
    k = 2.0 * math.pi / n
    theory = analytic_decay_rate(nu, k, "kinetic_energy")
    u = torch.tensor(taylor_green_velocity(n, n, n, U0), dtype=torch.float64)
    for devices in ((1, 1, 1), (2, 1, 2)):
        sim = port.HostStagedD3Q19(
            port.HostStagedLayout(n, n, n, *devices), nu=nu, dtype=torch.float64
        )
        sim.set_velocity(u)
        ts, ke = [], []
        for step in range(1, steps + 1):
            sim.step(1)
            if step % 20 == 0:
                ts.append(float(step))
                ke.append(sim.kinetic_energy())
        slope = -np.polyfit(np.array(ts), np.log(np.array(ke)), 1)[0]
        assert abs(slope - theory) / theory < 0.03, (devices, slope, theory)


def test_port_transport_is_decomposition_invariant_with_corrected_ic(port):
    """Transport invariance on the corrected physics (complements the m18373 D1 arm)."""
    import torch

    n, nu, steps = 16, 0.02, 40
    u = torch.tensor(taylor_green_velocity(n, n, n, U0), dtype=torch.float64)
    fields = []
    for devices in ((1, 1, 1), (2, 2, 1)):
        sim = port.HostStagedD3Q19(
            port.HostStagedLayout(n, n, n, *devices), nu=nu, dtype=torch.float64
        )
        sim.set_velocity(u)
        sim.step(steps)
        fields.append(sim.project_velocity().numpy())
    assert np.abs(fields[0] - fields[1]).max() < 1e-12