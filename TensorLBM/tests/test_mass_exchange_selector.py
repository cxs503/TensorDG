"""Tests for mass-exchange selector: conservation, accuracy, and scheme selection."""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorlbm import D2Q9, MassExchangeSelector, Scheme


class TestMassExchangeSelector:
    """Test suite for MassExchangeSelector."""

    @pytest.fixture
    def small_field(self):
        """Small test field for quick tests."""
        Ny, Nx = 16, 16
        rho = torch.ones(Ny, Nx, dtype=torch.float64)
        ux = 0.01 * torch.sin(torch.linspace(0, 2 * 3.14159, Nx)).unsqueeze(0).repeat(Ny, 1)
        uy = torch.zeros(Ny, Nx, dtype=torch.float64)
        f = D2Q9.equilibrium(rho, ux, uy)
        return f, rho, ux, uy

    @pytest.fixture
    def non_equilibrium_field(self):
        """Non-equilibrium field to test scheme differences."""
        Ny, Nx = 16, 16
        rho = torch.ones(Ny, Nx, dtype=torch.float64)
        ux = 0.05 * torch.ones(Ny, Nx, dtype=torch.float64)
        uy = torch.zeros(Ny, Nx, dtype=torch.float64)
        feq = D2Q9.equilibrium(rho, ux, uy)
        # Add non-equilibrium perturbation
        f = feq + 0.01 * torch.randn(D2Q9.Q, Ny, Nx, dtype=torch.float64)
        return f, rho, ux, uy

    def test_bgk_mass_conservation(self, small_field):
        """BGK collision conserves mass to machine precision."""
        f, rho, ux, uy = small_field
        selector = MassExchangeSelector(scheme=Scheme.BGK, omega=1.0)

        mass_before = f.sum().item()
        f_out = selector.collide(f, rho, ux, uy)
        mass_after = f_out.sum().item()

        assert abs(mass_after - mass_before) < 1e-12, \
            f"BGK mass drift: {abs(mass_after - mass_before):.2e}"

    def test_mrt_mass_conservation(self, small_field):
        """MRT collision conserves mass to machine precision."""
        f, rho, ux, uy = small_field
        s_vec = torch.zeros(9, dtype=torch.float64)
        s_vec[1] = 1.6  # Non-conserved moments
        s_vec[2] = 1.8
        s_vec[4] = 1.6
        s_vec[6] = 1.6
        s_vec[7] = 1.0
        s_vec[8] = 1.0

        selector = MassExchangeSelector(scheme=Scheme.MRT, s_vec=s_vec)

        mass_before = f.sum().item()
        f_out = selector.collide(f, rho, ux, uy)
        mass_after = f_out.sum().item()

        assert abs(mass_after - mass_before) < 1e-12, \
            f"MRT mass drift: {abs(mass_after - mass_before):.2e}"

    def test_trt_mass_conservation(self, small_field):
        """TRT collision conserves mass to machine precision."""
        f, rho, ux, uy = small_field
        selector = MassExchangeSelector(scheme=Scheme.TRT, omega=1.0, magic_number=0.25)

        mass_before = f.sum().item()
        f_out = selector.collide(f, rho, ux, uy)
        mass_after = f_out.sum().item()

        assert abs(mass_after - mass_before) < 1e-12, \
            f"TRT mass drift: {abs(mass_after - mass_before):.2e}"

    def test_bgk_momentum_conservation(self, small_field):
        """BGK collision conserves momentum (no force)."""
        f, rho, ux, uy = small_field
        selector = MassExchangeSelector(scheme=Scheme.BGK, omega=1.0)

        # Compute momentum before
        px_before = (D2Q9.e[:, 0].view(-1, 1, 1) * f).sum().item()
        py_before = (D2Q9.e[:, 1].view(-1, 1, 1) * f).sum().item()

        f_out = selector.collide(f, rho, ux, uy)

        # Compute momentum after
        px_after = (D2Q9.e[:, 0].view(-1, 1, 1) * f_out).sum().item()
        py_after = (D2Q9.e[:, 1].view(-1, 1, 1) * f_out).sum().item()

        assert abs(px_after - px_before) < 1e-10, \
            f"BGK x-momentum drift: {abs(px_after - px_before):.2e}"
        assert abs(py_after - py_before) < 1e-10, \
            f"BGK y-momentum drift: {abs(py_after - py_before):.2e}"

    def test_scheme_selection(self, non_equilibrium_field):
        """All schemes produce different results on non-equilibrium fields."""
        f, rho, ux, uy = non_equilibrium_field

        # Use omega=0.5 (tau=2.0) to ensure BGK and TRT differ
        f_bgk = MassExchangeSelector(scheme=Scheme.BGK, omega=0.5).collide(f, rho, ux, uy)
        f_trt = MassExchangeSelector(scheme=Scheme.TRT, omega=0.5).collide(f, rho, ux, uy)

        # BGK and TRT should differ (different relaxation behavior)
        diff = (f_bgk - f_trt).abs().max().item()
        assert diff > 1e-10, f"BGK and TRT should produce different results, got diff={diff:.2e}"

    def test_verify_mass_conservation(self, small_field):
        """verify_mass_conservation returns correct verdict."""
        f, rho, ux, uy = small_field
        selector = MassExchangeSelector(scheme=Scheme.BGK, omega=1.0)

        result = selector.verify_mass_conservation(f, n_steps=100)

        assert result["pass"] is True, "BGK should pass mass conservation"
        assert result["mass_drift_relative"] < 1e-10
        assert result["scheme"] == "bgk"

    def test_equilibrium_reconstruction(self):
        """Equilibrium distribution reconstructs macroscopic fields exactly."""
        Ny, Nx = 16, 16
        rho = torch.ones(Ny, Nx, dtype=torch.float64) * 1.5
        ux = 0.05 * torch.randn(Ny, Nx, dtype=torch.float64)
        uy = 0.03 * torch.randn(Ny, Nx, dtype=torch.float64)

        feq = D2Q9.equilibrium(rho, ux, uy)
        rho_rec, ux_rec, uy_rec = D2Q9.macroscopic(feq)

        assert torch.allclose(rho_rec, rho, atol=1e-12), "Density reconstruction failed"
        assert torch.allclose(ux_rec, ux, atol=1e-12), "ux reconstruction failed"
        assert torch.allclose(uy_rec, uy, atol=1e-12), "uy reconstruction failed"

    def test_streaming_periodic(self):
        """Periodic streaming preserves mass."""
        Ny, Nx = 16, 16
        rho = torch.ones(Ny, Nx, dtype=torch.float64)
        ux = 0.01 * torch.ones(Ny, Nx, dtype=torch.float64)
        uy = torch.zeros(Ny, Nx, dtype=torch.float64)
        f = D2Q9.equilibrium(rho, ux, uy)

        from tensorlbm.streaming import stream_periodic

        mass_before = f.sum().item()
        f_streamed = stream_periodic(f)
        mass_after = f_streamed.sum().item()

        assert abs(mass_after - mass_before) < 1e-12, \
            f"Streaming mass drift: {abs(mass_after - mass_before):.2e}"
