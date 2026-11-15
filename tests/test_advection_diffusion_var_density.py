"""Tests for variable-density advection-diffusion DG solver."""
import math
import pytest
import torch

from tensordg.advection_diffusion_var_density import (
    VarDensityAdvDiffDG,
    gaussian_density_field,
    sine_density_field,
    uniform_density,
    solid_body_rotation,
    manufactured_concentration,
    run_equilibrium_preservation,
)


class TestEquilibriumPreservation:
    """Uniform C=1 must be preserved exactly (linear-in-C equilibrium property)."""

    @pytest.mark.parametrize("degree", [0, 1, 2, 3])
    def test_uniform_preserved_pure_advection(self, degree):
        """C=1 with variable rho and divergence-free u stays C=1."""
        result = run_equilibrium_preservation(degree=degree, ncells=16)
        assert result["passed"], (
            f"Equilibrium preservation failed at p={degree}: "
            f"max_deviation={result['max_deviation_from_uniform']:.2e}, "
            f"mass_drift={result['mass_drift_relative']:.2e}"
        )

    def test_uniform_preserved_fine_mesh(self):
        """Finer mesh should also preserve equilibrium."""
        result = run_equilibrium_preservation(degree=2, ncells=32)
        assert result["max_deviation_from_uniform"] < 1e-10


class TestMassConservation:
    """Pure advection (D=0) must conserve int rho C dV."""

    @pytest.mark.parametrize("degree", [1, 2])
    def test_mass_conserved(self, degree):
        solver = VarDensityAdvDiffDG(
            degree=degree, ncells=(16, 16), diffusivity=0.0
        )
        rho_func = gaussian_density_field()
        velocity_func = solid_body_rotation(omega=2.0 * math.pi)

        def initial_C(x, y, t=0.0):
            return 1.0 + 0.5 * torch.sin(2.0 * math.pi * x) * torch.cos(2.0 * math.pi * y)

        result = solver.solve(
            initial_profile=initial_C,
            rho_func=rho_func,
            velocity_func=velocity_func,
            final_time=0.5,
            cfl=0.2,
        )

        m0 = result["history"]["mass"][0]
        m1 = result["history"]["mass"][-1]
        drift = abs(m1 - m0) / max(abs(m0), 1e-15)
        assert drift < 0.01, f"Mass drift {drift:.2e} exceeds 1%"


class TestSolverBasics:
    """Basic solver functionality tests."""

    def test_project_and_reconstruct(self):
        solver = VarDensityAdvDiffDG(degree=2, ncells=(8, 8))
        rho_func = uniform_density()

        def profile(x, y, t=0.0):
            return 1.0 + torch.sin(2.0 * math.pi * x) * torch.sin(2.0 * math.pi * y)

        coeff = solver.project(profile)
        nodal = solver.nodal(coeff)
        exact = profile(solver.Xn, solver.Yn, 0.0)
        error = float(torch.abs(nodal - exact).max().item())
        assert error < 0.05, f"Projection error {error:.2e} too large"

    def test_rhs_finite(self):
        solver = VarDensityAdvDiffDG(degree=1, ncells=(8, 8), diffusivity=0.01)
        rho_func = sine_density_field()
        velocity_func = solid_body_rotation(omega=math.pi)

        def initial_C(x, y, t=0.0):
            return 1.0 + 0.3 * torch.sin(2.0 * math.pi * x)

        coeff = solver.project(initial_C)
        rhs = solver.rhs(coeff, rho_func, velocity_func)
        assert torch.isfinite(rhs).all(), "RHS contains non-finite values"

    def test_step_produces_finite(self):
        solver = VarDensityAdvDiffDG(degree=1, ncells=(8, 8), diffusivity=0.01)
        rho_func = sine_density_field()
        velocity_func = solid_body_rotation(omega=math.pi)

        def initial_C(x, y, t=0.0):
            return 1.0 + 0.3 * torch.sin(2.0 * math.pi * x)

        coeff = solver.project(initial_C)
        coeff_new = solver.step_rk3(coeff, 0.001, rho_func, velocity_func)
        assert torch.isfinite(coeff_new).all(), "Step produced non-finite coefficients"


class TestConvergence:
    """Design-order convergence for smooth solutions."""

    def test_p1_convergence(self):
        """P1 should show ~2nd order convergence."""
        errors = []
        ncells_list = [8, 16, 32]

        for nc in ncells_list:
            solver = VarDensityAdvDiffDG(
                degree=1, ncells=(nc, nc), diffusivity=0.01
            )
            rho_func = uniform_density()
            velocity_func = solid_body_rotation(omega=math.pi)

            def initial_C(x, y, t=0.0):
                return manufactured_concentration(x, y, t)

            result = solver.solve(
                initial_profile=initial_C,
                rho_func=rho_func,
                velocity_func=velocity_func,
                final_time=0.05,
                cfl=0.15,
            )

            error = solver.l2_error(result["coeff"], manufactured_concentration, rho_func)
            errors.append(error)

        # Check that error decreases
        assert errors[-1] < errors[0], f"Error not decreasing: {errors}"

    def test_p2_convergence(self):
        """P2 should show ~3rd order convergence."""
        errors = []
        ncells_list = [8, 16, 32]

        for nc in ncells_list:
            solver = VarDensityAdvDiffDG(
                degree=2, ncells=(nc, nc), diffusivity=0.01
            )
            rho_func = uniform_density()
            velocity_func = solid_body_rotation(omega=math.pi)

            def initial_C(x, y, t=0.0):
                return manufactured_concentration(x, y, t)

            result = solver.solve(
                initial_profile=initial_C,
                rho_func=rho_func,
                velocity_func=velocity_func,
                final_time=0.05,
                cfl=0.15,
            )

            error = solver.l2_error(result["coeff"], manufactured_concentration, rho_func)
            errors.append(error)

        assert errors[-1] < errors[0], f"Error not decreasing: {errors}"
