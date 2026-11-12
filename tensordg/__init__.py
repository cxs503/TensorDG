"""TensorDG: differentiable discontinuous Galerkin methods built with PyTorch."""

from .advection_1d import l2_error, solve_advection
from .convergence import (
    convergence_table,
    gci_study,
    observed_order,
    p_decay_diagnostics,
    pairwise_orders,
    richardson_extrapolate,
)
from .advection_2d import error_2d, solve_advection_2d
from .advection_3d import (
    error_3d,
    exact_solution_3d,
    mass_3d,
    solve_advection_3d,
)
from .burgers_1d import burgers_error, solve_burgers
from .euler_1d import solve_riemann
from .euler_2d import solve_euler_2d, solve_isentropic_vortex, vortex_error
from .euler_dg_1d import solve_euler_dg_1d, solve_periodic_euler_dg_1d, verify_shock_entropy
from .cases.shock_entropy import (
    solve_dg_case as solve_shock_entropy_dg,
    rankine_hugoniot_anchor as shock_entropy_anchor,
    gate_report as shock_entropy_gate_report,
)
from .euler_dg_2d import (
    apply_tvb_limiter_2d,
    apply_weno_limiter_2d,
    solve_euler_dg_2d,
)
from .heat_1d import heat_error, solve_heat_1d
from .limiters import apply_limiter, apply_tvb_limiter, apply_weno_limiter
from .ns_dg_2d import (
    solve_couette_dg_2d,
    solve_navier_stokes_dg_2d,
)
from .poisson_2d import poisson_error_2d, solve_poisson_2d
from .poisson_2d_bc import (
    assemble_poisson_2d_bc,
    l2_error_2d,
    solve_poisson_2d_bc,
)
from .differentiable import (
    advance_advection_1d,
    advance_euler_dg_1d,
    advance_euler_dg_2d,
    advance_heat_1d,
    heat_1d_operators,
    central_difference_directional,
    central_difference_gradient,
    directional_derivative_ad,
    gradient_consistency,
    legendre_l2_sq,
    legendre_l2_sq_2d,
    probe_value,
    relative_difference,
)
from .swe_1d import (
    solve_swe_1d,
    reconstruct_fields,
    physical_flux as swe_physical_flux,
    hll_flux as swe_hll_flux,
    sound_speed as swe_sound_speed,
    GRAVITY as SWE_GRAVITY,
)
from .advection_diffusion import (
    cosine_bell,
    deformational_flow_velocity,
    gaussian_hill,
    rotated_field,
    run_advection_diffusion_decay,
    run_deformational_flow,
    run_solid_body_rotation,
    solid_body_rotation_velocity,
    solve_advection_diffusion,
)
from .dispersive import (
    DISPERSION_MODELS,
    KdVSolver,
    LinearDispersiveWave,
    airy_omega,
    airy_phase_speed,
    dispersion_omega_sq,
    kdv_soliton,
    measure_dispersion,
    model_dispersion_error,
    run_soliton_benchmark,
    validity_limit,
)
from .swe_2d import (
    solve_swe_2d,
    physical_flux_2d as swe_physical_flux_2d,
    hll_flux_2d as swe_hll_flux_2d,
)
from .tvb import minmod, troubled_from_differences

__all__ = [
    "solve_advection_diffusion", "run_solid_body_rotation",
    "run_deformational_flow", "run_advection_diffusion_decay",
    "solid_body_rotation_velocity", "deformational_flow_velocity",
    "cosine_bell", "gaussian_hill", "rotated_field",
    "airy_omega", "airy_phase_speed", "dispersion_omega_sq",
    "model_dispersion_error", "validity_limit", "measure_dispersion",
    "LinearDispersiveWave", "DISPERSION_MODELS", "KdVSolver", "kdv_soliton",
    "run_soliton_benchmark",
    "solve_advection", "l2_error",
    "observed_order", "pairwise_orders", "richardson_extrapolate",
    "gci_study", "convergence_table", "p_decay_diagnostics",
    "solve_advection_2d", "error_2d",
    "solve_advection_3d", "error_3d", "mass_3d", "exact_solution_3d",
    "solve_burgers", "burgers_error",
    "solve_heat_1d", "heat_error",
    "solve_poisson_2d", "poisson_error_2d",
    "solve_poisson_2d_bc", "assemble_poisson_2d_bc", "l2_error_2d",
    "solve_riemann", "solve_euler_dg_1d", "solve_periodic_euler_dg_1d",
    "solve_euler_2d", "solve_isentropic_vortex", "vortex_error",
    "solve_euler_dg_2d", "apply_tvb_limiter_2d", "apply_weno_limiter_2d",
    # Shu--Osher (1988) shock/entropy-wave case, shipped with the library
    "solve_shock_entropy_dg", "shock_entropy_anchor", "shock_entropy_gate_report",
    "verify_shock_entropy",
    "solve_navier_stokes_dg_2d", "solve_couette_dg_2d",
    "apply_limiter", "apply_tvb_limiter", "apply_weno_limiter",
    "minmod", "troubled_from_differences",
    "advance_advection_1d", "advance_euler_dg_1d", "advance_euler_dg_2d",
    "advance_heat_1d", "heat_1d_operators",
    "legendre_l2_sq", "legendre_l2_sq_2d", "probe_value",
    "directional_derivative_ad", "central_difference_directional",
    "central_difference_gradient", "gradient_consistency",
    "relative_difference",
    # Shallow water equations (mission m6144)
    "solve_swe_1d", "reconstruct_fields",
    "swe_physical_flux", "swe_hll_flux", "swe_sound_speed", "SWE_GRAVITY",
    "solve_swe_2d", "swe_physical_flux_2d", "swe_hll_flux_2d",
]
