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

# ---- acoustics
from .acoustics import (
    Acoustic2DResult,
    AcousticMedium,
    acoustic_energy,
    acoustic_flux_x,
    acoustic_flux_y,
    acoustic_plane_pulse_exact,
    acoustic_pulse_exact_pressure,
    acoustic_pulse_exact_state,
    acoustic_pulse_exact_velocity,
    acoustic_pulse_initial,
    acoustic_riemann_flux_x,
    acoustic_riemann_flux_y,
    solve_acoustic_2d,
)

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
    resample_cell_averages,
    shock_position,
    initial_state as shock_entropy_initial_state,
    conserved_state,
    solve_dg_case as solve_shock_entropy_dg,
    rankine_hugoniot_anchor as shock_entropy_anchor,
    exact_mass_budget,
    coarsen,
    relative_l1_error,
    gate_report as shock_entropy_gate_report,
)


# ---- cases.lax_1d
from .cases.lax_1d import (
    CASE_KEY as LAX_CASE_KEY,
    lax_parameters,
    star_state as lax_star_state,
    solve_lax_case,
)

# ---- cases.prefinement_exponential
from .cases.prefinement_exponential import (
    initial_condition as prefinement_initial_condition,
    exact_solution as prefinement_exact_solution,
    prefinement_parameters,
    solve_level as prefinement_solve_level,
    p_ladder as prefinement_p_ladder,
    anchor_table as prefinement_anchor_table,
    solve_prefinement_case,
)

# ---- cases.prefinement_exponential_2d
from .cases.prefinement_exponential_2d import (
    CASE_KEY as PREFINEMENT_2D_CASE_KEY,
    initial_condition as prefinement_2d_initial_condition,
    exact_solution as prefinement_2d_exact_solution,
    prefinement_parameters as prefinement_2d_parameters,
    solve_level as prefinement_2d_solve_level,
    p_ladder as prefinement_2d_p_ladder,
    anchor_table as prefinement_2d_anchor_table,
    solve_prefinement_2d_case,
)

# ---- cases.viscous_burgers_1d
from .cases.viscous_burgers_1d import (
    CASE_KEY as VISCOUS_BURGERS_CASE_KEY,
    viscous_burgers_parameters,
    initial_profile as viscous_burgers_initial_profile,
    theta_initial,
    exact_solution as viscous_burgers_exact_solution,
    exact_solution_fourier,
    periodic_sipg,
    stable_timestep,
    rhs_operator,
    solve_viscous_burgers_1d,
    l2_error as viscous_burgers_l2_error,
    total_mass,
    total_energy,
)
# ---- convergence
from .convergence import (
    exponential_decay_criterion,
    semidiscrete_operator,
    operator_linearity_residual,
    expm_scaled_squaring,
    exact_time_solution,
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
    roundoff_floor,
    signal_to_noise,
    resolved_levels,
)

# ---- certified_reference
# The round-2 certification engine.  Four of its names (``coarsen``,
# ``observed_order``, ``richardson_extrapolate``, and the Roache index) are also
# owned by :mod:`tensordg.convergence`; the certification ones are exported
# under a ``certified_`` prefix so both readings stay reachable and the
# identity gate can tell them apart.
from .certified_reference import (
    coarsen as certified_coarsen,
    rel_l1 as certified_rel_l1,
    observed_order as certified_observed_order,
    richardson_extrapolate as certified_richardson_extrapolate,
    gci as certified_gci,
    classify,
    digest_levels,
    certify,
    judge,
    audit_ladder,
    band_coverage_study,
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
    # driven / injected-work helpers (public in the module, missing at the root
    # until the contract gate counted them)
    gaussian_drive,
    driven_simple_wave,
    driven_injected_work,
)

# ---- elastic_1d_layered
from .elastic_1d_layered import (
    wave_speed as layered_wave_speed,
    impedance as layered_impedance,
    scattering_coefficients as layered_scattering_coefficients,
    interface_state as layered_interface_state,
    layered_interface_flux,
    solve_elastic_1d_layered,
    layered_energy,
    layered_stress_profile,
    rightgoing_pulse_initial_state as layered_rightgoing_pulse_initial_state,
    scattering_valid_time as layered_scattering_valid_time,
    exact_rightgoing_scattering as layered_exact_rightgoing_scattering,
)


# ---- elastic_2d
from .elastic_2d import (
    ElasticMedium,
    legendre_operators_2d as elastic_2d_legendre_operators,
    flux_x,
    flux_y,
    riemann_flux_x,
    riemann_flux_y,
    Elastic2DResult,
    quadrature_grid,
    modal_to_quadrature,
    elastic_energy as elastic_2d_energy,
    relative_l2_error as elastic_2d_relative_l2_error,
    plane_wave,
    fourier_mode_analysis,
    solve_elastic_2d,
)

# ---- elastic_dispersion
from .elastic_dispersion import (
    assemble_operator as elastic_dispersion_assemble_operator,
    system_symbol,
    symbol_union_error as elastic_dispersion_symbol_union_error,
    p0_closed_form,
    p0_symbol_error,
    physical_branch as elastic_dispersion_physical_branch,
    phase_and_dissipation,
    exponent_details,
    fit_exponent,
    dominant_theta,
    resolution_table as elastic_dispersion_resolution_table,
    analyse as elastic_dispersion_analyse,
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
    conservative_state,
    normalise_bcs,
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


# ---- limiter_ao
from .limiter_ao import (
    ao_window,
    ao_candidates,
    smoothness_indicators,
    apply_ao_limiter,
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

# ---- artificial_viscosity
from .artificial_viscosity import (
    apply_av_filter,
    av_sensor,
    av_strength,
    exponential_filter_factor,
    modal_smoothness,
    sensor_reference,
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


# ---- ldg_1d
from .ldg_1d import (
    ReferenceOperators,
    reference_operators,
    LDG1DResult,
    assemble_1d,
    solve_ldg_1d,
    convergence_study,
)
# ---- ldg_dirichlet_1d (validated 1-D LDG; see docs/LDG-DIRICHLET-CLOSURE-1D.md)
from .ldg_dirichlet_1d import (
    assemble_ldg_dirichlet_1d,
    ldg_l2_error_1d,
    solve_ldg_dirichlet_1d,
)

# ---- poisson_2d_ldg (validated 2-D LDG; see docs/CASE-DG-LDG-2D.md)
from .poisson_2d_ldg import (
    manufactured_solution as ldg_poisson_2d_manufactured_solution,
    superposition_solution as ldg_poisson_2d_superposition_solution,
    solve_poisson_2d_ldg,
    ldg_poisson_convergence,
)

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
    momentum_flux,
    shock_heads,
    shock_position as bump_shock_position,
    bump_transcritical_state,
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


# ---- wall_flux_stencil
from .wall_flux_stencil import (
    stencil_weights,
    wall_heat_flux_stencil,
    wall_heat_flux_stencil_summary,
)
# ---- shipped constants and case registry
# ---- stepping
from .stepping import step_metadata

from .dispersive import DISPERSION_MODELS
from .swe_1d import GRAVITY as SWE_GRAVITY
from .cases import CASES, ELASTIC_ROD_CASE_KEY, LAX_CASE_KEY, VISCOUS_BURGERS_CASE_KEY, PREFINEMENT_2D_CASE_KEY

__all__ = [
    "Acoustic2DResult",
    "AcousticMedium",
    "Basis1D",
    "BodyForcePoiseuille",
    "CASES",
    "CaseSpec",
    "CurvedAdvection",
    "CurvedMesh",
    "DGAdvection2D",
    "DISPERSION_MODELS",
    "ELASTIC_ROD_CASE_KEY",
    "Elastic2DResult",
    "ElasticMedium",
    "IsoparametricMap",
    "KdVSolver",
    "LAX_CASE_KEY",
    "LDG1DResult",
    "LinearDispersiveWave",
    "Operators",
    "OrderStudy",
    "ReferenceOperators",
    "RiemannData",
    "SWE_GRAVITY",
    "VISCOUS_BURGERS_CASE_KEY",
    "acoustic_energy",
    "acoustic_flux_x",
    "acoustic_flux_y",
    "acoustic_plane_pulse_exact",
    "acoustic_pulse_exact_pressure",
    "acoustic_pulse_exact_state",
    "acoustic_pulse_exact_velocity",
    "acoustic_pulse_initial",
    "acoustic_riemann_flux_x",
    "acoustic_riemann_flux_y",
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
    "ao_candidates",
    "ao_window",
    "apply_ao_limiter",
    "apply_av_filter",
    "apply_limiter",
    "apply_tvb_limiter",
    "apply_tvb_limiter_2d",
    "apply_weno_limiter",
    "apply_weno_limiter_2d",
    "assemble_1d",
    "assemble_heat_1d_sparse",
    "assemble_ldg_dirichlet_1d",
    "assemble_operator",
    "assemble_poisson_2d_bc",
    "assemble_poisson_2d_sparse",
    "assemble_sipg_1d",
    "audit_ladder",
    "av_sensor",
    "av_strength",
    "band_coverage_study",
    "basis_1d",
    "bernoulli_head",
    "bspline_basis_matrix",
    "bump_bed",
    "bump_bed_slope",
    "bump_boundary_flux",
    "bump_shock_position",
    "bump_steady_state",
    "bump_transcritical_state",
    "burgers_error",
    "burgers_shock_exact_solution",
    "cartesian_periodic_mesh",
    "cell_fields",
    "cell_quadrature",
    "central_difference_directional",
    "central_difference_gradient",
    "certified_coarsen",
    "certified_gci",
    "certified_observed_order",
    "certified_rel_l1",
    "certified_richardson_extrapolate",
    "certify",
    "check_hydrostatic_equilibrium",
    "check_hydrostatic_equilibrium_2d",
    "circle_contract",
    "clamped_free_frequency",
    "classify",
    "coarsen",
    "compute_hydrostatic_pressure_force",
    "cone_profile",
    "conservative_state",
    "conserved_state",
    "convergence_study",
    "convergence_table",
    "cosine_bell",
    "couette_exact",
    "create_hydrostatic_initial_conditions",
    "critical_depth",
    "dalembert_pulse",
    "dam_break_cell_averages",
    "davis_wave_speeds",
    "decay_rate",
    "default_domain",
    "deformational_flow_velocity",
    "digest_levels",
    "directional_derivative_ad",
    "dirichlet_face_operator",
    "dispersion_omega_sq",
    "dominant_theta",
    "driven_injected_work",
    "driven_simple_wave",
    "einfeldt_wave_speeds",
    "elastic_2d_energy",
    "elastic_2d_legendre_operators",
    "elastic_2d_relative_l2_error",
    "elastic_dispersion_analyse",
    "elastic_dispersion_assemble_operator",
    "elastic_dispersion_physical_branch",
    "elastic_dispersion_resolution_table",
    "elastic_dispersion_symbol_union_error",
    "elastic_energy",
    "elastic_rod_boundary_condition",
    "elastic_rod_boundary_values",
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
    "exact_solution_fourier",
    "exact_standing",
    "exact_temperature",
    "exact_time_solution",
    "expm_scaled_squaring",
    "exponent_details",
    "exponential_decay_criterion",
    "exponential_filter_factor",
    "extrapolation_weights",
    "fit_exponent",
    "flux_x",
    "flux_y",
    "fourier_mode_analysis",
    "front_positions",
    "gas_constants",
    "gauss_differentiation_matrix",
    "gauss_nodes_weights",
    "gaussian_bump",
    "gaussian_drive",
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
    "judge",
    "kdv_soliton",
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
    "lax_parameters",
    "lax_star_state",
    "layered_energy",
    "layered_exact_rightgoing_scattering",
    "layered_impedance",
    "layered_interface_flux",
    "layered_interface_state",
    "layered_rightgoing_pulse_initial_state",
    "layered_scattering_coefficients",
    "layered_scattering_valid_time",
    "layered_stress_profile",
    "layered_wave_speed",
    "ldg_l2_error_1d",
    "ldg_poisson_2d_manufactured_solution",
    "ldg_poisson_2d_superposition_solution",
    "ldg_poisson_convergence",
    "legendre_derivative_matrix",
    "legendre_eval_matrix",
    "legendre_l2_sq",
    "legendre_l2_sq_2d",
    "legendre_operators",
    "legendre_operators_2d",
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
    "modal_smoothness",
    "modal_to_quadrature",
    "model_dispersion_error",
    "momentum_flux",
    "normalise_bcs",
    "nurbs_curve_points",
    "observed_order",
    "operator_linearity_residual",
    "p0_closed_form",
    "p0_symbol_error",
    "p_decay_diagnostics",
    "pairwise_orders",
    "peak_stress",
    "peak_stress_amplitude",
    "periodic_sipg",
    "phase_and_dissipation",
    "physical_branch",
    "plane_wave",
    "poisson_error_2d",
    "positivity_filter",
    "prefinement_anchor_table",
    "prefinement_exact_solution",
    "prefinement_initial_condition",
    "prefinement_p_ladder",
    "prefinement_parameters",
    "prefinement_solve_level",
    "primitive_fields",
    "primitive_to_conservative",
    "probe_arrival_time",
    "probe_value",
    "quadrature_grid",
    "reconstruct_fields",
    "reconstruct_fields_2d",
    "reentrant_vortex",
    "reference_operators",
    "reflection_pulse",
    "reflection_velocity",
    "relative_difference",
    "relative_error",
    "relative_l1_error",
    "resolution_table",
    "resolved_levels",
    "rhs_operator",
    "richardson_extrapolate",
    "riemann_flux_x",
    "riemann_flux_y",
    "riemann_initial",
    "ritter_exact",
    "roe_flux",
    "rotated_field",
    "rotated_profile",
    "roundoff_floor",
    "run_advection_diffusion_decay",
    "run_deformational_flow",
    "run_order_study",
    "run_revolution_check",
    "run_solid_body_rotation",
    "run_soliton_benchmark",
    "run_vortex_peak_check",
    "rusanov_rhs",
    "semidiscrete_operator",
    "sensor_reference",
    "shear_wave_exact",
    "shock_entropy_anchor",
    "shock_entropy_gate_report",
    "shock_entropy_initial_state",
    "shock_expected_position",
    "shock_heads",
    "shock_position",
    "shock_position_from_mass",
    "shock_position_moment",
    "signal_to_noise",
    "sine_wave",
    "slotted_cylinder",
    "smooth_bump",
    "smoothness_indicators",
    "solid_body_rotation",
    "solid_body_rotation_velocity",
    "solve_acoustic_2d",
    "solve_advection",
    "solve_advection_2d",
    "solve_advection_3d",
    "solve_advection_diffusion_1d",
    "solve_burgers",
    "solve_burgers_shock",
    "solve_burgers_tracer",
    "solve_couette_dg_2d",
    "solve_elastic_1d",
    "solve_elastic_1d_layered",
    "solve_elastic_2d",
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
    "solve_lax_case",
    "solve_ldg_1d",
    "solve_ldg_dirichlet_1d",
    "solve_navier_stokes_dg_2d",
    "solve_periodic_euler_dg_1d",
    "solve_poisson_2d",
    "solve_poisson_2d_bc",
    "solve_poisson_2d_ldg",
    "solve_poisson_2d_matrixfree",
    "solve_poisson_2d_sparse",
    "solve_prefinement_case",
    "solve_riemann",
    "solve_shear_wave_dg_2d",
    "solve_shock_entropy_dg",
    "solve_swe_1d",
    "solve_swe_2d",
    "solve_taylor_green_dg_2d",
    "solve_viscous_burgers_1d",
    "sound_speed_2d",
    "stable_timestep",
    "standing_wave",
    "star_pressure",
    "steady_state_defect",
    "stencil_weights",
    "stoker_exact",
    "stoker_star_state",
    "stress_profile",
    "swe_hll_flux",
    "swe_hll_flux_2d",
    "swe_physical_flux",
    "swe_physical_flux_2d",
    "swe_sound_speed",
    "symbol_matrix",
    "symbol_union_error",
    "system_symbol",
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
    "theta_initial",
    "toro_iterative_wave_speeds",
    "toro_wave_speeds",
    "total_energy",
    "total_mass",
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
    "viscous_burgers_exact_solution",
    "viscous_burgers_initial_profile",
    "viscous_burgers_l2_error",
    "viscous_burgers_parameters",
    "vortex_error",
    "vortex_primitive",
    "wall_heat_flux",
    "wall_heat_flux_extrapolated",
    "wall_heat_flux_extrapolated_summary",
    "wall_heat_flux_stencil",
    "wall_heat_flux_stencil_summary",
    "wall_heat_flux_summary",
    "warped_periodic_mesh",
]

