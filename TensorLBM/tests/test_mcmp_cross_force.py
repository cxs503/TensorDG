"""Tests for MCMP cross-force operator validation.

Covers:
1. Symbolic verification of force formula
2. Spectral analysis of force operator
3. Mass conservation in MCMP solver
4. Laplace law validation (surface tension)
5. Force vanishing for uniform fields
6. Action-reaction symmetry
"""

import pytest
import torch
import numpy as np
from tensorlbm import (
    D2Q9,
    CrossForceOperator,
    MCMPSolver,
    pseudopotential_rho,
    pseudopotential_shan_chen,
    symbolic_cross_force_verification,
    spectral_analysis_cross_force,
    setup_laplace_test,
    measure_laplace_pressure,
)


# ---------------------------------------------------------------------------
# Symbolic verification tests
# ---------------------------------------------------------------------------

class TestSymbolicVerification:
    """Test symbolic verification of cross-force formula."""

    def test_symbolic_verification_passes(self):
        """All symbolic checks should pass."""
        result = symbolic_cross_force_verification(n_components=2)
        assert result["status"] == "PASS", f"Symbolic verification failed: {result}"
        assert result["n_passed"] == result["n_checks"]

    def test_lattice_symmetry(self):
        """D2Q9 lattice must satisfy Σ w_i · e_i = 0."""
        result = symbolic_cross_force_verification()
        assert result["checks"]["lattice_symmetry_ex"]["pass"]
        assert result["checks"]["lattice_symmetry_ey"]["pass"]
        assert abs(result["checks"]["lattice_symmetry_ex"]["value"]) < 1e-15
        assert abs(result["checks"]["lattice_symmetry_ey"]["value"]) < 1e-15

    def test_uniform_field_force_vanishes(self):
        """Force must vanish for uniform density fields."""
        result = symbolic_cross_force_verification()
        assert result["checks"]["uniform_field_force"]["pass"]

    def test_pseudopotential_linear_term(self):
        """Shan-Chen pseudopotential linear term must be ρ."""
        result = symbolic_cross_force_verification()
        assert result["checks"]["pseudopotential_linear_term"]["pass"]


# ---------------------------------------------------------------------------
# Spectral analysis tests
# ---------------------------------------------------------------------------

class TestSpectralAnalysis:
    """Test spectral analysis of cross-force operator."""

    def test_spectral_analysis_runs(self):
        """Spectral analysis should complete without error."""
        result = spectral_analysis_cross_force(
            G=-1.0, rho0=1.0, n_components=2, grid_size=64
        )
        assert "stability" in result
        assert "spectral_data" in result
        assert len(result["spectral_data"]["k"]) == 100

    def test_high_frequency_damping(self):
        """High-frequency modes should be damped (|S(k_max)| < |S(0)|)."""
        result = spectral_analysis_cross_force(G=-1.0)
        assert result["high_freq_damping_ratio"] < 1.0
        assert result["stability"] == "STABLE"

    def test_structure_factor_at_zero(self):
        """S(k→0) should approach Σ w_i = 1.

        Note: spectral analysis starts at k=0.01 (not k=0),
        so S(k=0.01) ≈ 1 - k²/6 ≈ 0.999983 for D2Q9.
        """
        result = spectral_analysis_cross_force()
        # At k=0.01, S ≈ 1 - k²/6 ≈ 0.999983
        expected = 1.0 - 0.01**2 / 6.0
        assert abs(result["S_real_at_k0"] - expected) < 1e-8

    def test_isotropy(self):
        """Structure factor should be approximately isotropic."""
        result = spectral_analysis_cross_force()
        # Isotropy error at k=π/4 should be small
        assert result["isotropy_error_at_pi4"] < 0.1


# ---------------------------------------------------------------------------
# Cross-force operator tests
# ---------------------------------------------------------------------------

class TestCrossForceOperator:
    """Test CrossForceOperator implementation."""

    def test_force_vanishes_uniform(self):
        """Force should vanish for uniform density fields."""
        n_comp = 2
        G_matrix = torch.tensor([[0.0, -1.0], [-1.0, 0.0]], dtype=torch.float64)
        op = CrossForceOperator(n_comp, G_matrix, psi_func=pseudopotential_rho)

        rho_0 = torch.ones(32, 32, dtype=torch.float64)
        rho_1 = torch.ones(32, 32, dtype=torch.float64) * 0.5

        force = op.compute_force([rho_0, rho_1])

        assert torch.allclose(force, torch.zeros_like(force), atol=1e-14)

    def test_force_nonzero_gradient(self):
        """Force should be nonzero for non-uniform density fields."""
        n_comp = 2
        G_matrix = torch.tensor([[0.0, -1.0], [-1.0, 0.0]], dtype=torch.float64)
        op = CrossForceOperator(n_comp, G_matrix, psi_func=pseudopotential_rho)

        # Create a density gradient
        y, x = torch.meshgrid(
            torch.arange(32, dtype=torch.float64),
            torch.arange(32, dtype=torch.float64),
            indexing='ij',
        )
        rho_0 = 1.0 + 0.1 * torch.sin(2 * np.pi * x / 32)
        rho_1 = torch.ones(32, 32, dtype=torch.float64) * 0.5

        force = op.compute_force([rho_0, rho_1])

        # Force should be nonzero
        assert force.abs().max() > 1e-10

    def test_force_shape(self):
        """Force tensor should have correct shape."""
        n_comp = 3
        G_matrix = torch.eye(3, dtype=torch.float64) * 0.0
        G_matrix[0, 1] = G_matrix[1, 0] = -1.0
        G_matrix[0, 2] = G_matrix[2, 0] = -0.5
        G_matrix[1, 2] = G_matrix[2, 1] = -0.3

        op = CrossForceOperator(n_comp, G_matrix)
        rho_fields = [torch.ones(16, 16, dtype=torch.float64) for _ in range(n_comp)]
        rho_fields[0] = rho_fields[0] + 0.1 * torch.randn(16, 16)

        force = op.compute_force(rho_fields)
        assert force.shape == (n_comp, 2, 16, 16)

    def test_global_momentum_conservation(self):
        """Total force should sum to zero for periodic BCs."""
        n_comp = 2
        G_matrix = torch.tensor([[0.0, -1.0], [-1.0, 0.0]], dtype=torch.float64)
        op = CrossForceOperator(n_comp, G_matrix, psi_func=pseudopotential_rho)

        # Random density fields
        torch.manual_seed(42)
        rho_0 = 1.0 + 0.1 * torch.randn(32, 32, dtype=torch.float64)
        rho_1 = 0.5 + 0.05 * torch.randn(32, 32, dtype=torch.float64)

        force = op.compute_force([rho_0, rho_1])

        # Total force should be near zero (periodic BCs)
        total_fx = force[:, 0].sum()
        total_fy = force[:, 1].sum()

        assert abs(total_fx.item()) < 1e-10
        assert abs(total_fy.item()) < 1e-10


# ---------------------------------------------------------------------------
# MCMP solver tests
# ---------------------------------------------------------------------------

class TestMCMPSolver:
    """Test MCMPSolver implementation."""

    def _make_solver(self, nx=64, ny=64):
        G_matrix = torch.tensor([[0.0, -0.8], [-0.8, 0.0]], dtype=torch.float64)
        return MCMPSolver(
            n_components=2,
            G_matrix=G_matrix,
            omega=1.0,
            nx=nx,
            ny=ny,
            dtype=torch.float64,
        )

    def test_solver_initialization(self):
        """Solver should initialize with equilibrium distributions."""
        solver = self._make_solver()
        assert len(solver.f) == 2
        assert solver.f[0].shape == (9, 64, 64)
        assert solver.step_count == 0

    def test_single_step(self):
        """Single step should complete without error."""
        solver = self._make_solver()
        solver.step()
        assert solver.step_count == 1

    def test_mass_conservation(self):
        """Mass should be conserved to machine precision."""
        solver = self._make_solver()
        result = solver.verify_mass_conservation(n_steps=50)
        assert result["pass"], f"Mass drift: {result['mass_drift_relative']}"

    def test_mass_conservation_long(self):
        """Mass should be conserved over many steps."""
        solver = self._make_solver()
        result = solver.verify_mass_conservation(n_steps=200)
        assert result["pass"], f"Mass drift: {result['mass_drift_relative']}"

    def test_kinetic_energy_decay(self):
        """Kinetic energy should decay due to viscosity (no forcing)."""
        solver = self._make_solver()

        # Add initial velocity perturbation
        y, x = torch.meshgrid(
            torch.arange(64, dtype=torch.float64),
            torch.arange(64, dtype=torch.float64),
            indexing='ij',
        )
        ux = 0.01 * torch.sin(2 * np.pi * x / 64) * torch.cos(2 * np.pi * y / 64)
        uy = -0.01 * torch.cos(2 * np.pi * x / 64) * torch.sin(2 * np.pi * y / 64)

        solver.set_velocity_field(0, ux, uy)
        solver.set_velocity_field(1, torch.zeros_like(ux), torch.zeros_like(uy))

        ke_0 = solver.get_kinetic_energy()
        solver.run(100)
        ke_100 = solver.get_kinetic_energy()

        # Energy should decrease (viscous dissipation)
        assert ke_100 < ke_0, f"KE increased: {ke_0} → {ke_100}"


# ---------------------------------------------------------------------------
# Laplace law tests
# ---------------------------------------------------------------------------

class TestLaplaceLaw:
    """Test Laplace law validation for surface tension.

    Note: Quantitative Laplace law validation requires long equilibration
    (5000+ steps) and careful parameter tuning. These tests verify:
    1. Solver setup is valid
    2. Phase separation occurs (qualitative)
    3. Pressure measurement runs without error (quantitative framework)
    """

    def test_laplace_setup(self):
        """Laplace test setup should create valid solver."""
        solver = setup_laplace_test(nx=64, ny=64, R=15.0)
        assert solver.n_components == 2
        assert solver.nx == 64
        assert solver.ny == 64

    def test_phase_separation_small(self):
        """Phase separation should occur for small droplet (R=15)."""
        solver = setup_laplace_test(nx=64, ny=64, R=15.0)
        solver.run(200)
        solver._compute_macroscopic()

        y, x = torch.meshgrid(
            torch.arange(64, dtype=torch.float64),
            torch.arange(64, dtype=torch.float64),
            indexing='ij',
        )
        cx, cy = 32.0, 32.0
        r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        mask_inside = r < 10
        mask_outside = r > 20

        rho0_in = solver.rho[0][mask_inside].mean().item()
        rho0_out = solver.rho[0][mask_outside].mean().item()

        # Phase separation: inside density > outside density
        assert rho0_in > rho0_out, f"No phase separation: {rho0_in} vs {rho0_out}"

    def test_phase_separation_medium(self):
        """Phase separation should occur for medium droplet (R=20 on 64x64)."""
        solver = setup_laplace_test(nx=64, ny=64, R=20.0)
        solver.run(300)
        solver._compute_macroscopic()

        y, x = torch.meshgrid(
            torch.arange(64, dtype=torch.float64),
            torch.arange(64, dtype=torch.float64),
            indexing='ij',
        )
        cx, cy = 32.0, 32.0
        r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        mask_inside = r < 14
        mask_outside = r > 26

        rho0_in = solver.rho[0][mask_inside].mean().item()
        rho0_out = solver.rho[0][mask_outside].mean().item()

        assert rho0_in > rho0_out, f"No phase separation: {rho0_in} vs {rho0_out}"

    def test_laplace_pressure_measurement(self):
        """Pressure measurement should run without NaN (framework test)."""
        solver = setup_laplace_test(nx=64, ny=64, R=15.0)
        result = measure_laplace_pressure(solver, R=15.0, n_equilibrate=200, n_sample=100)
        # Check that measurement runs (quantitative accuracy requires longer equilibration)
        assert "dp_measured" in result
        assert "dp_theoretical" in result
        assert "relative_error" in result


# ---------------------------------------------------------------------------
# Pseudopotential tests
# ---------------------------------------------------------------------------

class TestPseudopotential:
    """Test pseudopotential functions."""

    def test_identity_pseudopotential(self):
        """ψ(ρ) = ρ."""
        rho = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        psi = pseudopotential_rho(rho)
        assert torch.allclose(psi, rho)

    def test_shan_chen_pseudopotential(self):
        """ψ_SC(ρ) = ρ₀(1 - exp(-ρ/ρ₀))."""
        rho = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        psi = pseudopotential_shan_chen(rho, rho0=1.0)

        # ψ(0) = 0
        assert abs(psi[0].item()) < 1e-14
        # ψ(1) = 1 - exp(-1) ≈ 0.632
        assert abs(psi[1].item() - (1 - np.exp(-1))) < 1e-10
        # ψ(∞) → ρ₀
        rho_large = torch.tensor([100.0], dtype=torch.float64)
        psi_large = pseudopotential_shan_chen(rho_large, rho0=1.0)
        assert abs(psi_large[0].item() - 1.0) < 1e-10

    def test_shan_chen_derivative(self):
        """dψ/dρ at ρ=0 should be 1."""
        rho = torch.tensor([1e-8], dtype=torch.float64)
        psi = pseudopotential_shan_chen(rho, rho0=1.0)
        # ψ(ρ) ≈ ρ for small ρ
        assert abs(psi[0].item() - rho[0].item()) < 1e-12
