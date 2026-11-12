"""TensorDG: differentiable discontinuous Galerkin methods built with PyTorch.

Public API contract
-------------------
Every solver entry point, analytic anchor and measurement helper that a
``tensordg`` module defines as public (no leading underscore, not a ``main``
CLI hook) is re-exported from the package root, so that::

    from tensordg import solve_ldg_dirichlet_1d, ritter_exact, wall_heat_flux

works for *every* merged capability instead of only for the ones added before
the root ``__init__`` drifted.  The contract is machine-checked by
``tests/test_public_api_contract.py`` (module inventory vs. root namespace vs.
``__all__`` vs. the import snippets printed in ``docs/``) and audited by
``tools/api_contract.py`` -> ``validation/api_contract/contract.json``.

Name discipline
---------------
* A symbol keeps its own name at the root unless that name is owned by two
  different modules; the collision is then resolved by prefixing the *less
  general* owner with its module name (e.g. ``euler_sound_speed`` vs.
  ``swe_sound_speed``).  The alias table is this file; nothing is hidden.
* Names that would be meaningless out of context at the root
  (``solve``, ``rhs``, ``sample``, ``parameters``, ...) are prefixed with
  their module.  Their unprefixed form stays available via the module path,
  e.g. ``tensordg.advection_diffusion.solve``.
* Constants and case registries (``DISPERSION_MODELS``, ``SWE_GRAVITY``,
  ``CASES``) are exported at the bottom of this file.
"""

# ---- advection_1d
from .advection_1d import (
    solve_advection,
    l2_error,
    l1_error,
    linf_error,
)

# ---- advection_2d
from .advection_2d import (
    solve_advection_2d,
    error_2d,
)

# ---- advection_2d_var
from .advection_2d_var import (
    Basis1D,
    basis_1d,
    solid_body_rotation,
    reentrant_vortex,
    trig_profile,
    gaussian_bump,
    rotated_profile,
    cone_profile,
    slotted_cylinder,
    smooth_bump,
    DGAdvection2D,
    estimate_max_speed,
    estimate_max_components,
)

# ---- advection_3d
from .advection_3d import (
    exact_solution_3d,
    solve_advection_3d,
    mass_3d,
    error_3d,
)

# ---- advection_diffusion
from .advection_diffusion import (
    Operators,
    project as advection_diffusion_project,
    cell_quadrature,
    mass_total as advection_diffusion_mass_total,
    l2_error as advection_diffusion_l2_error,
    rhs as advection_diffusion_rhs,
    solve as advection_diffusion_solve,
    cosine_bell,
    gaussian_hill,
    sine_wave,
    solid_body_rotation_velocity,
    deformational_flow_velocity,
    rotated_field,
    vanishing_velocity,
    translation_velocity,
    run_solid_body_rotation,
    run_deformational_flow,
    run_advection_diffusion_decay,
)

# ---- advection_diffusion_1d
from .advection_diffusion_1d import (
    exact_solution as advection_diffusion_1d_exact_solution,
    exact_flux as advection_diffusion_1d_exact_flux,
    solve_advection_diffusion_1d,
    advection_diffusion_error,
    sample_solution as advection_diffusion_1d_sample_solution,
    face_fluxes as advection_diffusion_1d_face_fluxes,
)

# ---- burgers_1d
from .burgers_1d import (
    initial_profile,
    exact_profile,
    solve_burgers,
    burgers_error,
)

# ---- burgers_shock_1d
from .burgers_shock_1d import (
    RiemannData,
    riemann_initial,
    exact_solution as burgers_shock_exact_solution,
    shock_expected_position,
    lax_friedrichs_flux,
    roe_flux,
    positivity_filter,
    gibbs_overshoot,
    total_variation,
    shock_position,
    solve_burgers_shock,
    solve_burgers_tracer,
    shock_position_moment,
    shock_position_from_mass,
)

# ---- cases.elastic_rod_1d
from .cases.elastic_rod_1d import (
    exact_pulse,
    exact_standing,
    peak_stress_amplitude,
    front_positions,
    probe_arrival_time,
    clamped_free_frequency,
    reflection_pulse,
    reflection_velocity,
    initial_state as elastic_rod_initial_state,
    boundary_condition as elastic_rod_boundary_condition,
    solve_case as solve_elastic_rod_case,
    sample as elastic_rod_sample,
    boundary_values as elastic_rod_boundary_values,
    parameters as elastic_rod_parameters,
    anchor_table,
)

# ---- cases.shock_entropy
from .cases.shock_entropy import (
    initial_state as shock_entropy_initial_state,
    conserved_state,
    solve_dg_case as solve_shock_entropy_dg,
    rankine_hugoniot_anchor as shock_entropy_anchor,
    exact_mass_budget,
    coarsen,
    relative_l1_error,
    gate_report as shock_entropy_gate_report,
)

# ---- convergence
from .convergence import (
    pairwise_orders,
    observed_order,
    gci_study,
    richardson_extrapolate,
    convergence_table,
    p_decay_diagnostics,
    modal_mean_1d,
    modal_mean_2d,
    modal_point_value_1d,
    modal_point_value_2d,
    modal_l2_norm_1d,
)

# ---- curved_geometry
from .curved_geometry import (
    legendre_eval_matrix,
    legendre_derivative_matrix,
    gll_nodes_weights,
    gll_differentiation_matrix,
    gauss_nodes_weights,
    gauss_differentiation_matrix,
    IsoparametricMap,
    metric_identity_report,
    CurvedMesh,
    CurvedAdvection,
    warped_periodic_mesh,
    cartesian_periodic_mesh,
    annulus_mesh,
    bspline_basis_matrix,
    nurbs_curve_points,
    circle_contract,
    math_sqrt2,
    validate_geometry_contract,
    mesh_from_contract,
)

# ---- differentiable
from .differentiable import (
    advance_advection_1d,
    advance_euler_dg_1d,
    advance_euler_dg_2d,
    legendre_l2_sq,
    legendre_l2_sq_2d,
    probe_value,
    relative_difference,
    directional_derivative_ad,
    central_difference_directional,
    central_difference_gradient,
    gradient_consistency,
    heat_1d_operators,
    advance_heat_1d,
)

# ---- dispersion_analysis
from .dispersion_analysis import (
    assemble_operator,
    symbol_matrix,
    symbol_union_error,
    upwind_fv_symbol,
    physical_branch,
    resolution_table,
    analyse,
)

# ---- dispersive
from .dispersive import (
    kernel_airy,
    kernel_boussinesq,
    kernel_pade22,
    kernel_pade66,
    dispersion_omega_sq,
    airy_omega,
    airy_phase_speed,
    model_dispersion_error,
    validity_limit,
    LinearDispersiveWave,
    measure_dispersion,
    kdv_soliton,
    KdVSolver,
    run_soliton_benchmark,
)

# ---- elastic_1d
from .elastic_1d import (
    elastic_wave_speed,
    gaussian_pulse,
    initial_pulse_velocity,
    initial_pulse_stress,
    dalembert_pulse,
    standing_wave,
    initial_standing_velocity,
    initial_standing_stress,
    upwind_flux,
    solve_elastic_1d,
    stress_profile,
    relative_error,
    elastic_energy,
    peak_stress,
)

# ---- euler_1d
from .euler_1d import (
    primitive_to_conservative,
    primitive_fields,
    physical_flux as euler_physical_flux,
    sound_speed as euler_sound_speed,
    rusanov_rhs,
    davis_wave_speeds,
    einfeldt_wave_speeds,
    star_pressure,
    toro_wave_speeds,
    toro_iterative_wave_speeds,
    hllc_flux,
    hllc_rhs,
    solve_riemann,
)

# ---- euler_2d
from .euler_2d import (
    vortex_primitive,
    solve_euler_2d,
    solve_isentropic_vortex,
    vortex_error,
)

# ---- euler_dg_1d
from .euler_dg_1d import (
    solve_euler_dg_1d,
    solve_periodic_euler_dg_1d,
    verify_shock_entropy,
)

# ---- euler_dg_2d
from .euler_dg_2d import (
    legendre_operators,
    apply_tvb_limiter_2d,
    apply_weno_limiter_2d,
    solve_euler_dg_2d,
    cell_fields,
    solve_euler_dg_2d_vortex,
    solve_euler_dg_2d_sod,
    solve_euler_dg_2d_riemann,
)

# ---- heat_1d
from .heat_1d import (
    exact_temperature,
    assemble_sipg_1d,
    solve_heat_1d,
    solve_heat_1d_exact_time,
    heat_error,
)

# ---- ice_edge_bench
from .ice_edge_bench import (
    observed_order as ice_edge_observed_order,
    OrderStudy,
    CaseSpec,
    run_order_study,
    run_revolution_check,
    run_vortex_peak_check,
    mass_balance_trace,
)

# ---- kovasznay
from .kovasznay import (
    default_domain,
    decay_rate,
    kovasznay_exact,
    kovasznay_residual,
    kovasznay_thermodynamic_state,
    modal_evaluator,
    velocity_l2_error,
    solve_kovasznay_dg_2d,
)

# ---- limiters
from .limiters import (
    apply_tvb_limiter,
    apply_weno_limiter,
    apply_limiter,
)

# ---- mms_ns_2d
from .mms_ns_2d import (
    mms_primitives,
    mms_conserved,
    mms_source,
)

# ---- ns_dg_2d
from .ns_dg_2d import (
    gas_constants,
    wall_heat_flux,
    wall_heat_flux_summary,
    solve_navier_stokes_dg_2d,
    couette_exact,
    heat_conduction_exact,
    shear_wave_exact,
    solve_couette_dg_2d,
    solve_heat_conduction_dg_2d,
    solve_shear_wave_dg_2d,
    taylor_green_exact,
    solve_taylor_green_dg_2d,
)

# ---- ns_poiseuille
from .ns_poiseuille import (
    BodyForcePoiseuille,
)

# ---- poisson_2d
from .poisson_2d import (
    interior_face_operator,
    dirichlet_face_operator,
    solve_poisson_2d,
    poisson_error_2d,
)

# ---- poisson_2d_bc
from .poisson_2d_bc import (
    assemble_poisson_2d_bc,
    solve_poisson_2d_bc,
    l2_error_2d,
)

# ---- ldg_dirichlet_1d (validated 1-D LDG; see docs/LDG-DIRICHLET-CLOSURE-1D.md)
from .ldg_dirichlet_1d import (
    assemble_ldg_dirichlet_1d,
    ldg_l2_error_1d,
    solve_ldg_dirichlet_1d,
)

# ---- poisson_2d_ldg
# STATUS: PROTOTYPE -- NOT VALIDATED (rank-deficient Dirichlet closure, see
# docs/ldg_poisson_status.md).  Deliberately NOT re-exported: coverage of the
# public contract excludes prototype modules (tools/api_contract.py
# PROTOTYPE_MODULES) and tests/test_ldg_closure_rank.py enforces the guard.

# ---- sipg_matrixfree
from .sipg_matrixfree import (
    matvec_poisson_2d,
    solve_poisson_2d_matrixfree,
    matvec_heat_diffusion_1d,
    matvec_heat_mass_1d,
    solve_heat_1d_matrixfree,
)

# ---- sipg_sparse
from .sipg_sparse import (
    assemble_heat_1d_sparse,
    assemble_poisson_2d_sparse,
    solve_poisson_2d_sparse,
    solve_heat_1d_sparse,
)

# ---- swe_1d
from .swe_1d import (
    physical_flux as swe_physical_flux,
    sound_speed as swe_sound_speed,
    manning_friction_slope,
    manning_normal_depth,
    hll_flux as swe_hll_flux,
    solve_swe_1d,
    reconstruct_fields,
    check_hydrostatic_equilibrium,
    create_hydrostatic_initial_conditions,
    compute_hydrostatic_pressure_force,
    estimate_well_balanced_error_bound,
    validate_well_balanced_setup,
)

# ---- swe_2d
from .swe_2d import (
    physical_flux_2d as swe_physical_flux_2d,
    sound_speed_2d,
    hll_flux_2d as swe_hll_flux_2d,
    legendre_operators_2d,
    solve_swe_2d,
    reconstruct_fields_2d,
    check_hydrostatic_equilibrium_2d,
)

# ---- swe_exact
from .swe_exact import (
    stoker_star_state,
    stoker_exact,
    ritter_exact,
    dam_break_cell_averages,
    thacker_parabolic_bed,
    thacker_parabolic_bed_slope,
    thacker_planar_basin,
    thacker_planar_basin_initial,
    thacker_planar_basin_period,
    thacker_planar_basin_boundary_flux,
    thacker_radial_params,
    thacker_radial_period,
    thacker_radial_bed,
    thacker_radial_bed_slope,
    thacker_radial_shoreline,
    thacker_radial_basin,
    thacker_radial_basin_initial,
    thacker_radial_velocity,
)

# ---- swe_steady_bump
from .swe_steady_bump import (
    bump_bed,
    bump_bed_slope,
    critical_depth,
    bernoulli_head,
    bump_steady_state,
    bump_boundary_flux,
    steady_state_defect,
)

# ---- tvb
from .tvb import (
    minmod,
    troubled_from_differences,
)

# ---- wall_flux_extrapolation
from .wall_flux_extrapolation import (
    extrapolation_weights,
    wall_heat_flux_extrapolated,
    wall_heat_flux_extrapolated_summary,
)

# ---- shipped constants and case registry
from .dispersive import DISPERSION_MODELS
from .swe_1d import GRAVITY as SWE_GRAVITY
from .cases import CASES, ELASTIC_ROD_CASE_KEY

__all__ = [
    "assemble_ldg_dirichlet_1d",
    "ldg_l2_error_1d",
    "solve_ldg_dirichlet_1d",
    "advance_advection_1d",
    "advance_euler_dg_1d",
    "advance_euler_dg_2d",
    "advance_heat_1d",
    "advection_diffusion_1d_exact_flux",
    "advection_diffusion_1d_exact_solution",
    "advection_diffusion_1d_face_fluxes",
    "advection_diffusion_1d_sample_solution",
    "advection_diffusion_error",
    "advection_diffusion_l2_error",
    "advection_diffusion_mass_total",
    "advection_diffusion_project",
    "advection_diffusion_rhs",
    "advection_diffusion_solve",
    "airy_omega",
    "airy_phase_speed",
    "analyse",
    "anchor_table",
    "annulus_mesh",
    "apply_limiter",
    "apply_tvb_limiter",
    "apply_tvb_limiter_2d",
    "apply_weno_limiter",
    "apply_weno_limiter_2d",
    "assemble_heat_1d_sparse",
    "assemble_operator",
    "assemble_poisson_2d_bc",
    "assemble_poisson_2d_sparse",
    "assemble_sipg_1d",
    "Basis1D",
    "basis_1d",
    "bernoulli_head",
    "BodyForcePoiseuille",
    "bspline_basis_matrix",
    "bump_bed",
    "bump_bed_slope",
    "bump_boundary_flux",
    "bump_steady_state",
    "burgers_error",
    "burgers_shock_exact_solution",
    "cartesian_periodic_mesh",
    "CASES",
    "CaseSpec",
    "cell_fields",
    "cell_quadrature",
    "central_difference_directional",
    "central_difference_gradient",
    "check_hydrostatic_equilibrium",
    "check_hydrostatic_equilibrium_2d",
    "circle_contract",
    "clamped_free_frequency",
    "coarsen",
    "compute_hydrostatic_pressure_force",
    "cone_profile",
    "conserved_state",
    "convergence_table",
    "cosine_bell",
    "couette_exact",
    "create_hydrostatic_initial_conditions",
    "critical_depth",
    "CurvedAdvection",
    "CurvedMesh",
    "dalembert_pulse",
    "dam_break_cell_averages",
    "davis_wave_speeds",
    "decay_rate",
    "default_domain",
    "deformational_flow_velocity",
    "DGAdvection2D",
    "directional_derivative_ad",
    "dirichlet_face_operator",
    "DISPERSION_MODELS",
    "dispersion_omega_sq",
    "einfeldt_wave_speeds",
    "elastic_energy",
    "elastic_rod_boundary_condition",
    "elastic_rod_boundary_values",
    "ELASTIC_ROD_CASE_KEY",
    "elastic_rod_initial_state",
    "elastic_rod_parameters",
    "elastic_rod_sample",
    "elastic_wave_speed",
    "error_2d",
    "error_3d",
    "estimate_max_components",
    "estimate_max_speed",
    "estimate_well_balanced_error_bound",
    "euler_physical_flux",
    "euler_sound_speed",
    "exact_mass_budget",
    "exact_profile",
    "exact_pulse",
    "exact_solution_3d",
    "exact_standing",
    "exact_temperature",
    "extrapolation_weights",
    "front_positions",
    "gas_constants",
    "gauss_differentiation_matrix",
    "gauss_nodes_weights",
    "gaussian_bump",
    "gaussian_hill",
    "gaussian_pulse",
    "gci_study",
    "gibbs_overshoot",
    "gll_differentiation_matrix",
    "gll_nodes_weights",
    "gradient_consistency",
    "heat_1d_operators",
    "heat_conduction_exact",
    "heat_error",
    "hllc_flux",
    "hllc_rhs",
    "ice_edge_observed_order",
    "initial_profile",
    "initial_pulse_stress",
    "initial_pulse_velocity",
    "initial_standing_stress",
    "initial_standing_velocity",
    "interior_face_operator",
    "IsoparametricMap",
    "kdv_soliton",
    "KdVSolver",
    "kernel_airy",
    "kernel_boussinesq",
    "kernel_pade22",
    "kernel_pade66",
    "kovasznay_exact",
    "kovasznay_residual",
    "kovasznay_thermodynamic_state",
    "l1_error",
    "l2_error",
    "l2_error_2d",
    "lax_friedrichs_flux",
    "legendre_derivative_matrix",
    "legendre_eval_matrix",
    "legendre_l2_sq",
    "legendre_l2_sq_2d",
    "legendre_operators",
    "legendre_operators_2d",
    "LinearDispersiveWave",
    "linf_error",
    "manning_friction_slope",
    "manning_normal_depth",
    "mass_3d",
    "mass_balance_trace",
    "math_sqrt2",
    "matvec_heat_diffusion_1d",
    "matvec_heat_mass_1d",
    "matvec_poisson_2d",
    "measure_dispersion",
    "mesh_from_contract",
    "metric_identity_report",
    "minmod",
    "mms_conserved",
    "mms_primitives",
    "mms_source",
    "modal_evaluator",
    "modal_l2_norm_1d",
    "modal_mean_1d",
    "modal_mean_2d",
    "modal_point_value_1d",
    "modal_point_value_2d",
    "model_dispersion_error",
    "nurbs_curve_points",
    "observed_order",
    "Operators",
    "OrderStudy",
    "p_decay_diagnostics",
    "pairwise_orders",
    "peak_stress",
    "peak_stress_amplitude",
    "physical_branch",
    "poisson_error_2d",
    "positivity_filter",
    "primitive_fields",
    "primitive_to_conservative",
    "probe_arrival_time",
    "probe_value",
    "reconstruct_fields",
    "reconstruct_fields_2d",
    "reentrant_vortex",
    "reflection_pulse",
    "reflection_velocity",
    "relative_difference",
    "relative_error",
    "relative_l1_error",
    "resolution_table",
    "richardson_extrapolate",
    "riemann_initial",
    "RiemannData",
    "ritter_exact",
    "roe_flux",
    "rotated_field",
    "rotated_profile",
    "run_advection_diffusion_decay",
    "run_deformational_flow",
    "run_order_study",
    "run_revolution_check",
    "run_solid_body_rotation",
    "run_soliton_benchmark",
    "run_vortex_peak_check",
    "rusanov_rhs",
    "shear_wave_exact",
    "shock_entropy_anchor",
    "shock_entropy_gate_report",
    "shock_entropy_initial_state",
    "shock_expected_position",
    "shock_position",
    "shock_position_from_mass",
    "shock_position_moment",
    "sine_wave",
    "slotted_cylinder",
    "smooth_bump",
    "solid_body_rotation",
    "solid_body_rotation_velocity",
    "solve_advection",
    "solve_advection_2d",
    "solve_advection_3d",
    "solve_advection_diffusion_1d",
    "solve_burgers",
    "solve_burgers_shock",
    "solve_burgers_tracer",
    "solve_couette_dg_2d",
    "solve_elastic_1d",
    "solve_elastic_rod_case",
    "solve_euler_2d",
    "solve_euler_dg_1d",
    "solve_euler_dg_2d",
    "solve_euler_dg_2d_riemann",
    "solve_euler_dg_2d_sod",
    "solve_euler_dg_2d_vortex",
    "solve_heat_1d",
    "solve_heat_1d_exact_time",
    "solve_heat_1d_matrixfree",
    "solve_heat_1d_sparse",
    "solve_heat_conduction_dg_2d",
    "solve_isentropic_vortex",
    "solve_kovasznay_dg_2d",
    "solve_navier_stokes_dg_2d",
    "solve_periodic_euler_dg_1d",
    "solve_poisson_2d",
    "solve_poisson_2d_bc",
    "solve_poisson_2d_matrixfree",
    "solve_poisson_2d_sparse",
    "solve_riemann",
    "solve_shear_wave_dg_2d",
    "solve_shock_entropy_dg",
    "solve_swe_1d",
    "solve_swe_2d",
    "solve_taylor_green_dg_2d",
    "sound_speed_2d",
    "standing_wave",
    "star_pressure",
    "steady_state_defect",
    "stoker_exact",
    "stoker_star_state",
    "stress_profile",
    "SWE_GRAVITY",
    "swe_hll_flux",
    "swe_hll_flux_2d",
    "swe_physical_flux",
    "swe_physical_flux_2d",
    "swe_sound_speed",
    "symbol_matrix",
    "symbol_union_error",
    "taylor_green_exact",
    "thacker_parabolic_bed",
    "thacker_parabolic_bed_slope",
    "thacker_planar_basin",
    "thacker_planar_basin_boundary_flux",
    "thacker_planar_basin_initial",
    "thacker_planar_basin_period",
    "thacker_radial_basin",
    "thacker_radial_basin_initial",
    "thacker_radial_bed",
    "thacker_radial_bed_slope",
    "thacker_radial_params",
    "thacker_radial_period",
    "thacker_radial_shoreline",
    "thacker_radial_velocity",
    "toro_iterative_wave_speeds",
    "toro_wave_speeds",
    "total_variation",
    "translation_velocity",
    "trig_profile",
    "troubled_from_differences",
    "upwind_flux",
    "upwind_fv_symbol",
    "validate_geometry_contract",
    "validate_well_balanced_setup",
    "validity_limit",
    "vanishing_velocity",
    "velocity_l2_error",
    "verify_shock_entropy",
    "vortex_error",
    "vortex_primitive",
    "wall_heat_flux",
    "wall_heat_flux_extrapolated",
    "wall_heat_flux_extrapolated_summary",
    "wall_heat_flux_summary",
    "warped_periodic_mesh",
]
