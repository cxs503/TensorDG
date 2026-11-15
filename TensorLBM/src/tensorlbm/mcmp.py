"""MCMP (Multi-Component Multi-Phase) LBM solver.

Implements a multi-component lattice Boltzmann solver using the
Shan-Chen pseudopotential model for inter-component interactions.

Each component has its own distribution function f^α, and components
interact through the cross-force operator.

Usage:
    solver = MCMP Solver(n_components=2, G_matrix=G, ...)
    for step in range(n_steps):
        solver.step()
"""

from typing import Callable, List, Optional, Tuple
import torch
from .lattice import D2Q9
from .collision import bgk_collision
from .streaming import stream_periodic
from .cross_force import (
    CrossForceOperator,
    pseudopotential_rho,
    pseudopotential_shan_chen,
)


class MCMPSolver:
    """Multi-Component Multi-Phase LBM solver.

    Parameters
    ----------
    n_components : int
        Number of fluid components.
    G_matrix : torch.Tensor (n_components, n_components)
        Interaction strength matrix.
    omega : float or list of float
        Relaxation parameter(s) for each component.
    psi_func : callable, optional
        Pseudopotential function. Default: identity.
    nx, ny : int
        Grid dimensions.
    device : str, optional
        Computation device. Default: 'cpu'.
    dtype : torch.dtype, optional
        Computation dtype. Default: torch.float64.
    """

    def __init__(
        self,
        n_components: int,
        G_matrix: torch.Tensor,
        omega: float = 1.0,
        psi_func: Optional[Callable] = None,
        nx: int = 128,
        ny: int = 128,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.n_components = n_components
        self.nx = nx
        self.ny = ny
        self.device = device
        self.dtype = dtype

        # Relaxation parameters
        if isinstance(omega, (list, tuple)):
            self.omega = list(omega)
        else:
            self.omega = [omega] * n_components

        # Cross-force operator
        self.cross_force = CrossForceOperator(
            n_components=n_components,
            G_matrix=G_matrix.to(device=device, dtype=dtype),
            psi_func=psi_func or pseudopotential_rho,
            lattice=D2Q9,
        )

        # Initialize distribution functions
        self._init_distributions()

        # Tracking
        self.step_count = 0
        self.mass_history: List[float] = []
        self.energy_history: List[float] = []

    def _init_distributions(self):
        """Initialize distribution functions to equilibrium."""
        Q = D2Q9.Q
        self.f = []
        self.rho = []
        self.ux = []
        self.uy = []

        for alpha in range(self.n_components):
            # Default: uniform density, zero velocity
            rho_0 = torch.ones(self.ny, self.nx, dtype=self.dtype, device=self.device)
            ux_0 = torch.zeros(self.ny, self.nx, dtype=self.dtype, device=self.device)
            uy_0 = torch.zeros(self.ny, self.nx, dtype=self.dtype, device=self.device)

            feq = D2Q9.equilibrium(rho_0, ux_0, uy_0)

            self.f.append(feq)
            self.rho.append(rho_0)
            self.ux.append(ux_0)
            self.uy.append(uy_0)

    def set_density_field(self, component: int, rho: torch.Tensor):
        """Set density field for a component and re-equilibrate."""
        assert rho.shape == (self.ny, self.nx)
        self.rho[component] = rho.to(device=self.device, dtype=self.dtype)
        feq = D2Q9.equilibrium(
            self.rho[component], self.ux[component], self.uy[component]
        )
        self.f[component] = feq

    def set_velocity_field(self, component: int, ux: torch.Tensor, uy: torch.Tensor):
        """Set velocity field for a component and re-equilibrate."""
        self.ux[component] = ux.to(device=self.device, dtype=self.dtype)
        self.uy[component] = uy.to(device=self.device, dtype=self.dtype)
        feq = D2Q9.equilibrium(
            self.rho[component], self.ux[component], self.uy[component]
        )
        self.f[component] = feq

    def _compute_macroscopic(self):
        """Compute macroscopic fields from distributions."""
        for alpha in range(self.n_components):
            self.rho[alpha], self.ux[alpha], self.uy[alpha] = D2Q9.macroscopic(
                self.f[alpha]
            )

    def step(self):
        """Perform one MCMP LBM step: compute force → collide with shifted velocity → stream."""
        # Compute macroscopic fields
        self._compute_macroscopic()

        # Compute cross-force
        self.cross_force.compute_force(self.rho)

        # Collision + force application for each component
        f_post = []
        for alpha in range(self.n_components):
            # Standard Shan-Chen forcing: shift velocity by force
            # u^eq = u + tau * F / rho
            tau = 1.0 / self.omega[alpha]
            fx = self.cross_force.force[alpha, 0]
            fy = self.cross_force.force[alpha, 1]
            ux_shifted = self.ux[alpha] + tau * fx / self.rho[alpha]
            uy_shifted = self.uy[alpha] + tau * fy / self.rho[alpha]

            # BGK collision with shifted velocity
            f_coll = bgk_collision(
                self.f[alpha],
                self.rho[alpha],
                ux_shifted,
                uy_shifted,
                self.omega[alpha],
            )
            f_post.append(f_coll)

        # Streaming
        for alpha in range(self.n_components):
            self.f[alpha] = stream_periodic(f_post[alpha])

        self.step_count += 1

        # Track mass
        total_mass = sum(f.sum().item() for f in self.f)
        self.mass_history.append(total_mass)

    def run(self, n_steps: int):
        """Run simulation for n_steps."""
        for _ in range(n_steps):
            self.step()

    def get_total_mass(self) -> float:
        """Get total mass across all components."""
        return sum(f.sum().item() for f in self.f)

    def get_kinetic_energy(self) -> float:
        """Get total kinetic energy."""
        self._compute_macroscopic()
        ke = 0.0
        for alpha in range(self.n_components):
            ke += (0.5 * self.rho[alpha] * (self.ux[alpha]**2 + self.uy[alpha]**2)).sum().item()
        return ke

    def verify_mass_conservation(self, n_steps: int = 100) -> dict:
        """Verify mass conservation over n_steps."""
        mass_0 = self.get_total_mass()
        self.run(n_steps)
        mass_final = self.get_total_mass()

        mass_drift = abs(mass_final - mass_0) / max(abs(mass_0), 1e-15)
        max_drift = 0.0
        if len(self.mass_history) >= n_steps:
            recent = self.mass_history[-n_steps:]
            max_drift = max(abs(m - mass_0) / max(abs(mass_0), 1e-15) for m in recent)

        return {
            "mass_initial": mass_0,
            "mass_final": mass_final,
            "mass_drift_relative": mass_drift,
            "max_step_drift": max_drift,
            "n_steps": n_steps,
            "pass": mass_drift < 1e-8,
        }


# ---------------------------------------------------------------------------
# Laplace law test (surface tension validation)
# ---------------------------------------------------------------------------

def setup_laplace_test(
    nx: int = 128,
    ny: int = 128,
    R: float = 30.0,
    rho_high: float = 2.0,
    rho_low: float = 0.5,
    G: float = -1.5,
    omega: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> MCMPSolver:
    """Set up Laplace law test: circular droplet in periodic domain.

    The pressure difference across a curved interface should satisfy:
        Δp = σ / R

    where σ is surface tension and R is droplet radius.

    Parameters
    ----------
    nx, ny : int
        Grid size.
    R : float
        Droplet radius.
    rho_high : float
        Density inside droplet.
    rho_low : float
        Density outside droplet.
    G : float
        Interaction strength (negative = attraction).
    omega : float
        Relaxation parameter.
    device, dtype : str, torch.dtype
        Computation settings.

    Returns
    -------
    solver : MCMPSolver
        Initialized solver with droplet configuration.
    """
    n_components = 2
    G_matrix = torch.tensor([[0.0, G], [G, 0.0]], dtype=dtype)

    solver = MCMPSolver(
        n_components=n_components,
        G_matrix=G_matrix,
        omega=omega,
        psi_func=pseudopotential_shan_chen,
        nx=nx,
        ny=ny,
        device=device,
        dtype=dtype,
    )

    # Initialize: component 0 = droplet, component 1 = surrounding
    y, x = torch.meshgrid(
        torch.arange(ny, dtype=dtype, device=device),
        torch.arange(nx, dtype=dtype, device=device),
        indexing='ij',
    )
    cx, cy = nx / 2, ny / 2
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Smooth interface (tanh profile)
    interface_width = 3.0
    phi = torch.tanh((R - r) / interface_width)

    # Component 0: high density inside, low outside
    rho_0 = rho_low + 0.5 * (rho_high - rho_low) * (1 + phi)
    # Component 1: low density inside, high outside
    rho_1 = rho_high - 0.5 * (rho_high - rho_low) * (1 + phi)

    solver.set_density_field(0, rho_0)
    solver.set_density_field(1, rho_1)

    return solver


def measure_laplace_pressure(
    solver: MCMPSolver,
    R: float,
    n_equilibrate: int = 5000,
    n_sample: int = 1000,
) -> dict:
    """Measure pressure difference across interface and compare to Laplace law.

    Returns
    -------
    result : dict
        Contains measured Δp, theoretical Δp, relative error, and verdict.
    """
    # Equilibrate
    solver.run(n_equilibrate)

    # Compute macroscopic fields
    solver._compute_macroscopic()

    # Measure pressure in bulk regions
    # Component 0 density inside droplet
    ny, nx = solver.ny, solver.nx
    y, x = torch.meshgrid(
        torch.arange(ny, dtype=solver.dtype, device=solver.device),
        torch.arange(nx, dtype=solver.dtype, device=solver.device),
        indexing='ij',
    )
    cx, cy = nx / 2, ny / 2
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Inside region (r < R - 5)
    mask_inside = r < (R - 5)
    # Outside region (r > R + 5)
    mask_outside = r > (R + 5)

    # Pressure from equation of state: p = ρ·cs² + G·ψ₁·ψ₂·cs²
    cs2 = D2Q9.cs2.item()
    G = solver.cross_force.G_matrix[0, 1].item()

    rho0_in = solver.rho[0][mask_inside].mean().item()
    rho1_in = solver.rho[1][mask_inside].mean().item()
    rho0_out = solver.rho[0][mask_outside].mean().item()
    rho1_out = solver.rho[1][mask_outside].mean().item()

    # Pressure: p = cs² * (ρ₀ + ρ₁ + G·ψ₀·ψ₁)
    # For identity pseudopotential: ψ = ρ
    p_in = cs2 * (rho0_in + rho1_in + G * rho0_in * rho1_in)
    p_out = cs2 * (rho0_out + rho1_out + G * rho0_out * rho1_out)

    dp_measured = abs(p_in - p_out)

    # Theoretical Laplace pressure: Δp = σ / R
    # Surface tension for Shan-Chen: σ = -G·cs²·Σ w_i·|e_i|²·(Δψ)² / 2
    # Simplified estimate: σ ≈ -G·cs²·(ρ_high - ρ_low)²·C
    # where C is a lattice-dependent constant
    # For D2Q9 with identity ψ: σ ≈ -G·cs²·(Δρ)²·(1/6)
    delta_rho = abs(rho0_in - rho0_out)
    sigma_estimate = -G * cs2 * delta_rho ** 2 / 6.0
    dp_theoretical = sigma_estimate / R

    rel_error = abs(dp_measured - dp_theoretical) / max(dp_theoretical, 1e-15)

    return {
        "rho0_inside": rho0_in,
        "rho1_inside": rho1_in,
        "rho0_outside": rho0_out,
        "rho1_outside": rho1_out,
        "pressure_inside": p_in,
        "pressure_outside": p_out,
        "dp_measured": dp_measured,
        "dp_theoretical": dp_theoretical,
        "sigma_estimate": sigma_estimate,
        "relative_error": rel_error,
        "pass": rel_error < 0.15,  # <15% error acceptable
    }
