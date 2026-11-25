"""Verification gates for the steady 1-D advection--diffusion DG solver.

Every gate is an *external* reference (the classical exponential-layer solution
or an analytically differentiated flux), never a self-consistency check.  The
negative controls (coarse layer, collapsed SIPG penalty) prove that the gates
can fail; their measured values are frozen here so that a regression that makes
the solver "always pass" trips a test.

Reference: the boundary-value problem is the textbook singular-perturbation
model problem (e.g. Roos, Stynes & Tobiska, *Robust Numerical Methods for
Singularly Perturbed Differential Equations*, Ch. 1), whose exact solution is
used here in closed form.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg.advection_diffusion_1d import (
    advection_diffusion_error,
    exact_flux,
    exact_solution,
    face_fluxes,
    sample_solution,
    solve_advection_diffusion_1d,
)

THREE_PERCENT = 0.03


def _relative_l2(ncells: int, degree: int, peclet: float, **kwargs) -> float:
    centers, coeff, _ = solve_advection_diffusion_1d(
        ncells=ncells, degree=degree, peclet=peclet, **kwargs,
    )
    return advection_diffusion_error(centers, coeff, peclet)["relative_l2"]


def _eoc(errors: list[float]) -> list[float]:
    return [
        math.log(errors[i] / errors[i + 1]) / math.log(2.0)
        for i in range(len(errors) - 1)
    ]


# --------------------------------------------------------------------------- #
# reference solution sanity
# --------------------------------------------------------------------------- #
def test_exact_solution_satisfies_boundary_data_and_is_monotone():
    x = torch.linspace(0.0, 1.0, 201, dtype=torch.float64)
    u = exact_solution(x, peclet=10.0)
    assert abs(float(u[0])) < 1e-12
    assert abs(float(u[-1]) - 1.0) < 1e-12
    assert bool(torch.all(torch.diff(u) > 0.0)), "exact layer must be monotone"


def test_exact_flux_matches_numerical_derivative_of_the_reference_solution():
    """The analytic flux (3) is re-derived from (2) by finite differences."""
    peclet, speed, step = 10.0, 1.0, 1e-6
    diffusivity = speed / peclet
    for x in (0.05, 0.25, 0.5, 0.75, 0.95):
        point = torch.tensor(x, dtype=torch.float64)
        value = exact_solution(point, peclet)
        derivative = (
            exact_solution(point + step, peclet) - exact_solution(point - step, peclet)
        ) / (2.0 * step)
        flux = speed * value - diffusivity * derivative
        assert abs(float(flux) - exact_flux(peclet)) < 1e-6 * max(1.0, abs(exact_flux(peclet)))


# --------------------------------------------------------------------------- #
# design-order gates
# --------------------------------------------------------------------------- #
def test_p1_is_second_order_at_peclet_one():
    errors = [_relative_l2(n, 1, 1.0) for n in (16, 32, 64, 128)]
    eocs = _eoc(errors)
    assert min(eocs) > 1.9, eocs


def test_p2_is_third_order_at_peclet_one():
    errors = [_relative_l2(n, 2, 1.0) for n in (16, 32, 64, 128)]
    assert min(_eoc(errors)) > 2.9, _eoc(errors)


def test_p3_is_fourth_order_at_peclet_one():
    errors = [_relative_l2(n, 3, 1.0) for n in (16, 32, 64)]
    assert min(_eoc(errors)) > 3.85, _eoc(errors)


def test_p2_keeps_third_order_up_to_peclet_ten():
    errors = [_relative_l2(n, 2, 10.0) for n in (32, 64, 128)]
    assert min(_eoc(errors)) > 2.9, _eoc(errors)


# --------------------------------------------------------------------------- #
# documented-case gates (3 % red line)
# --------------------------------------------------------------------------- #
def test_documented_layer_cases_meet_three_percent():
    # Pe=10 is resolved by every degree; Pe=100 requires p=3 at 64 cells.
    assert _relative_l2(64, 2, 10.0) < THREE_PERCENT
    assert _relative_l2(64, 3, 10.0) < THREE_PERCENT
    assert _relative_l2(64, 3, 100.0) < THREE_PERCENT


def test_diffusion_dominated_limit_recovers_the_linear_profile():
    """Pe -> 0: the exact solution tends to u = x, which DG represents exactly."""
    error = _relative_l2(32, 2, 0.01)
    assert error < 1e-2, error


# --------------------------------------------------------------------------- #
# face-flux consistency (independent of the L2 norm)
# --------------------------------------------------------------------------- #
def test_face_fluxes_converge_to_the_exact_constant_flux():
    deviations = []
    for ncells in (8, 32, 128):
        _, coeff, _ = solve_advection_diffusion_1d(ncells=ncells, degree=3, peclet=10.0)
        deviation = float((face_fluxes(coeff, 10.0) - exact_flux(10.0)).abs().max())
        deviations.append(deviation / 1.0)  # normalised by a*|g_R-g_L| = 1
    assert deviations[0] > deviations[1] > deviations[2]
    assert deviations[-1] < 1e-5, deviations
    assert _eoc(deviations[:2] + [deviations[-1]])[-1] > 3.0


# --------------------------------------------------------------------------- #
# resolution limit and negative controls
# --------------------------------------------------------------------------- #
def test_cell_peclet_resolution_limit_is_measured_not_assumed():
    """At 64 cells the 3 % gate is met up to cell Peclet ~0.3 (P1), 1.5 (P2), 3.1 (P3)."""
    h = 1.0 / 64
    passing = {
        degree: max(
            (peclet for peclet in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
             if _relative_l2(64, degree, peclet) < THREE_PERCENT),
            default=0.0,
        )
        for degree in (1, 2, 3)
    }
    assert passing[1] == pytest.approx(20.0)
    assert passing[2] == pytest.approx(100.0)
    assert passing[3] == pytest.approx(200.0)
    assert passing[1] * h < 0.4 and passing[3] * h > 3.0


def test_under_resolved_layer_is_a_real_failure_not_a_rounding_artefact():
    coarse = _relative_l2(16, 2, 100.0)
    fine = _relative_l2(128, 2, 100.0)
    assert coarse > 0.10, coarse
    assert fine < THREE_PERCENT, fine


def test_collapsed_sipg_penalty_is_detected_by_the_l2_gate():
    reference = _relative_l2(32, 2, 10.0)
    collapsed = _relative_l2(32, 2, 10.0, penalty=1e-2)
    assert reference < 1e-3
    assert collapsed > 10.0 * reference, (reference, collapsed)


def test_downwind_advective_flux_is_not_a_discriminating_control():
    """Documented finding: the SIPG penalty keeps the solution nearly continuous.

    Upwind and downwind advective fluxes therefore give the same solution to
    ~1e-6 relative accuracy at every cell Peclet tested, so this variant must
    *not* be used as a negative control (the penalty collapse above is the real
    implementation control).  If this test ever fails, the discretisation has
    changed enough that the earlier reports' claim no longer holds.
    """
    for peclet in (10.0, 100.0, 1000.0):
        upwind = _relative_l2(32, 2, peclet)
        downwind = _relative_l2(32, 2, peclet, advection_scheme="downwind")
        assert abs(upwind - downwind) < 1e-5 * max(upwind, 1.0), (peclet, upwind, downwind)


# --------------------------------------------------------------------------- #
# baseline comparison helper (used by the report)
# --------------------------------------------------------------------------- #
def test_first_order_finite_volume_baseline_converges_at_first_order():
    from benchmarks.verify_advection_diffusion_1d import finite_volume_upwind

    errors = [finite_volume_upwind(ncells, 10.0)[0] for ncells in (16, 32, 64, 128)]
    assert min(_eoc(errors)) > 0.9, _eoc(errors)


def test_sampled_solution_is_pointwise_consistent_with_the_modal_coefficients():
    """sample_solution is the plotting path used by the report; it must agree with
    the modal evaluation at the same points, otherwise the reported figures would
    not represent the coefficients under test."""
    centers, coeff, _ = solve_advection_diffusion_1d(ncells=64, degree=2, peclet=10.0)
    x, values = sample_solution(centers, coeff, samples_per_cell=16)
    exact = exact_solution(x, 10.0)
    pointwise = float((values - exact).abs().max())
    reference = advection_diffusion_error(centers, coeff, 10.0)["absolute_l2"]
    # pointwise error is a weaker norm than L2 up to the cell count
    assert pointwise < 20.0 * reference, (pointwise, reference)
    assert float(x.min()) < 0.01 and float(x.max()) > 0.99
    assert bool(torch.all(torch.diff(x) > 0.0)), "sample abscissae must be increasing"
    assert np.all(np.diff(x.numpy()) > 0.0)


def test_reverse_flow_resolves_inlet_layer_and_conserves_flux():
    errors = []
    for n in (16, 32, 64):
        x, coeff, meta = solve_advection_diffusion_1d(
            ncells=n, degree=2, peclet=4, speed=-2, left_value=2, right_value=5,
        )
        assert meta["diffusivity"] == 0.5
        errors.append(advection_diffusion_error(x, coeff, 4, -2, 2, 5)["absolute_l2"])
    assert min(_eoc(errors)) > 2.8
    # For u = A + B exp(-4x), the total flux is -2A.
    expected = -2 * (2 + 3 / (1 - math.exp(-4)))
    assert exact_flux(4, -2, 2, 5) == pytest.approx(expected)
    assert (face_fluxes(coeff, 4, -2) - expected).abs().max() < 0.005


@pytest.mark.parametrize("speed",[-1.25,1.25])
@pytest.mark.parametrize("scheme",["upwind","downwind"])
def test_quadratic_source_and_linear_reaction_close_every_face(speed,scheme):
    from tensordg.advection_diffusion_1d import numerical_face_fluxes
    n,pe = 7,4.
    d = abs(speed)/pe
    def u(x):
        return .2+x*(.5-x)
    def source(x):
        return speed*(.5-2*x)+2*d+(1+x)*u(x)
    centers,coeff,_ = solve_advection_diffusion_1d(n,2,pe,speed,left_value=u(0.),right_value=u(1.),
            source=source,reaction=lambda x:1+x,advection_scheme=scheme)
    x,values = sample_solution(centers,coeff)
    torch.testing.assert_close(values,u(x),atol=2e-12,rtol=0.)
    faces = torch.linspace(0.,1.,n+1,dtype=coeff.dtype)
    flux = numerical_face_fluxes(coeff,pe,speed,left_value=u(0.),right_value=u(1.),advection_scheme=scheme)
    torch.testing.assert_close(flux,speed*u(faces)-d*(.5-2*faces),atol=2e-12,rtol=0.)


@pytest.mark.parametrize("speed",[-1.3,1.3])
@pytest.mark.parametrize("ends",[(True,False),(False,True),(True,True)])
def test_rightward_boundary_flux_with_variable_reaction(speed,ends):
    from tensordg.advection_diffusion_1d import numerical_face_fluxes
    d = abs(speed)/4
    def u(x):
        return 1.2-.4*x+.7*x*x
    def flux(x):
        return speed*u(x)-d*(-.4+1.4*x)
    imposed = tuple(flux(i) if use else None for i,use in enumerate(ends))
    options = dict(total_flux=imposed,left_value=None if ends[0] else u(0.),
                   right_value=None if ends[1] else u(1.))
    x,coeff,_ = solve_advection_diffusion_1d(8,2,4.,speed,**options,reaction=lambda x:2+x,
                source=lambda x:speed*(-.4+1.4*x)-1.4*d+(2+x)*u(x))
    points,values = sample_solution(x,coeff)
    torch.testing.assert_close(values,u(points),atol=5e-12,rtol=0.)
    estimate = numerical_face_fluxes(coeff,4.,speed,**options)
    torch.testing.assert_close(estimate,flux(torch.linspace(0.,1.,9,dtype=x.dtype)),atol=2e-12,rtol=0.)


def test_two_flux_walls_need_reaction_anchor():
    with pytest.raises(ValueError,match="nonzero reaction"):
        solve_advection_diffusion_1d(total_flux=(.1,.1))


@pytest.mark.parametrize("speed",[-1.,1.])
def test_flux_pair_with_mean_selects_exponential_stationary_mode(speed):
    from tensordg.advection_diffusion_1d import numerical_face_fluxes
    pe,n = 3.,20
    x,coeff,_ = solve_advection_diffusion_1d(n,3,pe,speed,total_flux=(.4,.4),mean_value=1.7)
    pts,uh = sample_solution(x,coeff)
    exponent = speed*pe
    expected = .4/speed+(1.7-.4/speed)*exponent*torch.exp(exponent*pts)/math.expm1(exponent)
    torch.testing.assert_close(uh,expected,atol=2e-6,rtol=0.)
    torch.testing.assert_close(coeff[:,0].mean(),coeff.new_tensor(1.7),atol=5e-13,rtol=0.)
    flux = numerical_face_fluxes(coeff,pe,speed,total_flux=(.4,.4))
    torch.testing.assert_close(flux,torch.full_like(flux,.4),atol=2e-10,rtol=0.)
    with pytest.raises(ValueError,match="incompatible"):
        solve_advection_diffusion_1d(n,2,pe,speed,total_flux=(.4,.41),mean_value=1.7)


def test_nonuniform_flux_driven_solution_and_width_weighted_mean():
    from tensordg.advection_diffusion_1d import numerical_face_fluxes
    edges = torch.tensor([0.,.07,.2,.53,.8,1.],dtype=torch.float64)
    speed,pe = -1.3,4.
    d = abs(speed)/pe
    def field(x):
        return .8-.4*x+.6*x*x
    def flux(x):
        return speed*field(x)-d*(-.4+1.2*x)
    imposed = (flux(0.),flux(1.))
    centers,coeff,meta = solve_advection_diffusion_1d(degree=2,peclet=pe,speed=speed,
             cell_edges=edges,total_flux=imposed,mean_value=.8,source=lambda x:speed*(-.4+1.2*x)-1.2*d)
    x,uh = sample_solution(centers,coeff,cell_edges=meta["cell_edges"])
    torch.testing.assert_close(uh,field(x),atol=1e-11,rtol=0.)
    torch.testing.assert_close(face_fluxes(coeff,pe,speed,cell_edges=edges),flux(edges[1:]),atol=1e-11,rtol=0.)
    recovered = numerical_face_fluxes(coeff,pe,speed,total_flux=imposed,cell_edges=edges)
    torch.testing.assert_close(recovered,flux(edges),atol=1e-11,rtol=0.)


@pytest.mark.parametrize("edges",[[0.,.2,.2,1.],[0.,.2,.9],[0.,float("nan"),1.]])
def test_reject_invalid_transport_edges(edges):
    with pytest.raises(ValueError,match="cell_edges"):
        solve_advection_diffusion_1d(cell_edges=edges)


@pytest.mark.parametrize("speed",[-1.3,1.3])
def test_layered_quadratic_transport_in_resistance_coordinate(speed):
    from tensordg.advection_diffusion_1d import numerical_face_fluxes
    edges = torch.tensor([0.,.08,.3,.5,.7,1.],dtype=torch.float64)
    ratio = edges.new_tensor([.4,.4,3.,3.,3.])
    def y(x):
        return torch.minimum(x,x.new_tensor(.3))/.4+torch.clamp(x-.3,min=0.)/3.
    def u(x):
        z = y(x)
        return .7+z*(.25-z)
    d = abs(speed)/3.
    def flux(x):
        return speed*u(x)-d*(.25-2*y(x))
    ends = u(edges)[[0,-1]].tolist()
    opts = dict(left_value=ends[0],right_value=ends[1],cell_edges=edges,diffusivity_ratio=ratio)
    centers,coeff,_ = solve_advection_diffusion_1d(degree=2,peclet=3.,speed=speed,**opts,
                    source=lambda x:(speed*(.25-2*y(x))+2*d)/ratio[:,None])
    x,uh = sample_solution(centers,coeff,cell_edges=edges)
    torch.testing.assert_close(uh,u(x),atol=2e-11,rtol=0.)
    torch.testing.assert_close(numerical_face_fluxes(coeff,3.,speed,**opts),flux(edges),atol=2e-11,rtol=0.)
    torch.testing.assert_close(face_fluxes(coeff,3.,speed,cell_edges=edges,diffusivity_ratio=ratio),
                               flux(edges[1:]),atol=2e-11,rtol=0.)


@pytest.mark.parametrize("ratio",[0.,-.1,float("nan"),[1.,2.],1.+.1j])
def test_reject_invalid_transport_diffusivity_ratio(ratio):
    with pytest.raises(ValueError,match="diffusivity_ratio"):
        solve_advection_diffusion_1d(3,diffusivity_ratio=ratio)


@pytest.mark.parametrize("speed",[-1.,1.])
def test_face_heating_on_a_graded_material_interface(speed):
    from tensordg.advection_diffusion_1d import numerical_face_fluxes
    edges = torch.tensor([0.,.1,.3,.65,1.],dtype=torch.float64)
    ratio = edges.new_tensor([2.,2.,.5,.5])
    slopes = edges.new_tensor([.7,.7,-.8,-.8])
    def u(x):
        return 1.+.7*x-1.5*(x-.3).clamp_min(0.)
    sheet = edges.new_zeros(3)
    sheet[1] = (2*.7+.5*.8)/4
    opts = dict(cell_edges=edges,diffusivity_ratio=ratio,interface_source=sheet)
    total = (speed*u(edges[0])-ratio[0]*slopes[0]/4,speed*u(edges[-1])-ratio[-1]*slopes[-1]/4)
    mean = float(((u(edges[:-1])+u(edges[1:]))*edges.diff()/2).sum())
    x,c,_ = solve_advection_diffusion_1d(degree=1,peclet=4.,speed=speed,total_flux=total,
                 source=speed*slopes,mean_value=mean,**opts)
    pts,values = sample_solution(x,c,cell_edges=edges)
    torch.testing.assert_close(values,u(pts),atol=2e-11,rtol=0.)
    west,east = numerical_face_fluxes(c,4.,speed,total_flux=total,return_cell_fluxes=True,**opts)
    torch.testing.assert_close(west,speed*u(edges[:-1])-ratio*slopes/4,atol=2e-11,rtol=0.)
    torch.testing.assert_close(east,speed*u(edges[1:])-ratio*slopes/4,atol=2e-11,rtol=0.)


@pytest.mark.parametrize("speed",[-1.3,1.3])
def test_point_reaction_anchors_two_flux_boundaries(speed):
    from tensordg.advection_diffusion_1d import numerical_face_fluxes
    edges = torch.tensor([0.,.2,.4,.7,1.],dtype=torch.float64)
    # u is continuous with slope 1 to the left and 2 to the right of x=.4.
    # At that face J_right-J_left=-D, fixing the absolute value through k*u.
    x0,k,d = .4,1.7,abs(speed)/4
    def u(x):
        return .8+x+(x-x0).clamp_min(0.)
    surface_source = d*(-1)+k*(.8+x0)
    opts = dict(cell_edges=edges,interface_source=[0.,surface_source,0.],
                interface_reaction=[0.,k,0.],total_flux=(speed*.8-d,speed*2.4-2*d))
    centers,c,_ = solve_advection_diffusion_1d(degree=1,peclet=4.,speed=speed,
                        source=speed*edges.new_tensor([1.,1.,2.,2.]),**opts)
    x,uh = sample_solution(centers,c,cell_edges=edges)
    torch.testing.assert_close(uh,u(x),atol=2e-11,rtol=0.)
    west,east = numerical_face_fluxes(c,4.,speed,return_cell_fluxes=True,**opts)
    torch.testing.assert_close(west[1:]-east[:-1],edges.new_tensor([0.,-d,0.]),atol=2e-11,rtol=0.)


@pytest.mark.parametrize("options",[{"interface_source":[1.,2.]},{"interface_source":1j},
                                    {"interface_reaction":-.1},{"interface_reaction":float("nan")}])
def test_invalid_internal_face_law(options):
    with pytest.raises(ValueError,match="interface_"):
        solve_advection_diffusion_1d(ncells=4,**options)


@pytest.mark.parametrize("speed",[-1.25,1.25])
@pytest.mark.parametrize("kw,ke",[(0.,2.),(1.3,0.),(1.3,2.)])
def test_exchange_and_flux_offsets_with_no_volume_reaction(speed,kw,ke):
    from tensordg.advection_diffusion_1d import numerical_face_fluxes
    pe = 4.
    def field(x):
        return .4+.3*x-.8*x*x
    def flux(x):
        return speed*field(x)-abs(speed)/pe*(.3-1.6*x)
    ambient = (1.2,-.1)
    offsets = (flux(0.)+kw*(field(0.)-ambient[0]),flux(1.)-ke*(field(1.)-ambient[1]))
    opts = dict(total_flux=offsets,boundary_exchange=((kw,ambient[0]),(ke,ambient[1])))
    centers,c,_ = solve_advection_diffusion_1d(8,2,pe,speed,**opts,
                                source=lambda x:speed*(.3-1.6*x)+1.6*abs(speed)/pe)
    pts,uh = sample_solution(centers,c)
    torch.testing.assert_close(uh,field(pts),atol=1e-11,rtol=0.)
    actual = numerical_face_fluxes(c,pe,speed,**opts)
    torch.testing.assert_close(actual,flux(torch.linspace(0.,1.,9,dtype=c.dtype)),atol=1e-11,rtol=0.)


def test_insulating_exchange_requires_mean_and_valid_coefficients():
    with pytest.raises(ValueError,match="mean_value"):
        solve_advection_diffusion_1d(boundary_exchange=((0.,1.),(0.,2.)))
    with pytest.raises(ValueError,match="boundary exchange"):
        solve_advection_diffusion_1d(boundary_exchange=((-.2,1.),None))


@pytest.mark.parametrize("orientation",[-1.,1.])
def test_conservative_variable_velocity_with_interior_stagnation(orientation):
    from tensordg.advection_diffusion_1d import numerical_face_fluxes
    edges = torch.tensor([0.,.12,.25,.48,.7,1.],dtype=torch.float64)
    def velocity(x):
        return orientation*2*(x-.25)
    def u(x):
        return .8+.6*x-.3*x*x
    def flux(x):
        return velocity(x)*u(x)-.2*(.6-.6*x)
    opts = dict(cell_edges=edges,velocity=velocity,left_value=u(0.),right_value=u(1.))
    centers,c,_ = solve_advection_diffusion_1d(degree=2,peclet=5.,**opts,
                 source=lambda x:2*orientation*u(x)+velocity(x)*(.6-.6*x)+.12)
    x,uh = sample_solution(centers,c,cell_edges=edges)
    torch.testing.assert_close(uh,u(x),atol=5e-12,rtol=0.)
    torch.testing.assert_close(numerical_face_fluxes(c,5.,**opts),flux(edges),atol=5e-12,rtol=0.)
    torch.testing.assert_close(face_fluxes(c,5.,cell_edges=edges,velocity=velocity),flux(edges[1:]),atol=5e-12,rtol=0.)


def test_smooth_diffusivity_with_a_material_jump_and_nonzero_flux():
    from tensordg.advection_diffusion_1d import sample_transport_flux,numerical_face_fluxes
    edges = torch.tensor([0.,.12,.35,.6,1.],dtype=torch.float64)
    ratio = edges.new_tensor([.5,.5,2.,2.])
    def profile(x):
        return 1.+x+x*x
    def field(x):
        return .4+.2*x+.3*x*x
    def derivative(x):
        return .2+.6*x
    def flux(x,material):
        return field(x)-.25*material*profile(x)*derivative(x)
    def source(x):
        return derivative(x)-.25*ratio[:,None]*((1+2*x)*derivative(x)+.6*profile(x))
    sheet = edges.new_zeros(3)
    sheet[1] = flux(edges[2],2.)-flux(edges[2],.5)
    opts = dict(cell_edges=edges,diffusivity_ratio=ratio,diffusivity_profile=profile)
    x,c,_ = solve_advection_diffusion_1d(degree=2,peclet=4.,source=source,interface_source=sheet,
                                         left_value=field(0.),right_value=field(1.),**opts)
    pts,j = sample_transport_flux(x,c,4.,**opts)
    torch.testing.assert_close(j.reshape(4,-1),flux(pts.reshape(4,-1),ratio[:,None]),atol=2e-11,rtol=0.)
    west,east = numerical_face_fluxes(c,4.,interface_source=sheet,return_cell_fluxes=True,
                                      left_value=field(0.),right_value=field(1.),**opts)
    torch.testing.assert_close(west,flux(edges[:-1],ratio),atol=2e-11,rtol=0.)
    torch.testing.assert_close(east,flux(edges[1:],ratio),atol=2e-11,rtol=0.)
    torch.testing.assert_close(face_fluxes(c,4.,**opts),flux(edges[1:],ratio),atol=2e-11,rtol=0.)


@pytest.mark.parametrize("profile",[0.,-1.,float("nan"),1j,lambda x:x*(1-x)])
def test_diffusivity_must_be_positive_at_interior_and_boundary_points(profile):
    with pytest.raises(ValueError,match="diffusivity_profile"):
        solve_advection_diffusion_1d(4,diffusivity_profile=profile)


@pytest.mark.parametrize("speed",[-1.,1.])
def test_periodic_seam_point_reaction_anchors_ring(speed):
    from tensordg.advection_diffusion_1d import numerical_face_fluxes
    edges = torch.tensor([0.,.1,.5,.8,1.],dtype=torch.float64)
    ratio = edges.new_tensor([.5,.5,2.,2.])
    slope = edges.new_tensor([1.,1.,-1.,-1.])
    def u(x):
        return 1.+torch.minimum(x,1-x)
    # J_right-J_left = -(D_right*u'_right - D_left*u'_left)
    jump = .25*(ratio*slope-(ratio*slope).roll(-1))
    uptake = edges.new_tensor([0.,0.,0.,.7])
    opts = dict(cell_edges=edges,diffusivity_ratio=ratio,periodic=True,
                interface_source=jump+uptake*u(edges[1:]),interface_reaction=uptake)
    x,c,_ = solve_advection_diffusion_1d(degree=1,peclet=4.,speed=speed,source=speed*slope,**opts)
    pts,uh = sample_solution(x,c,cell_edges=edges)
    torch.testing.assert_close(uh,u(pts),atol=1e-11,rtol=0.)
    west,east = numerical_face_fluxes(c,4.,speed,return_cell_fluxes=True,**opts)
    torch.testing.assert_close(west,speed*u(edges[:-1])-.25*ratio*slope,atol=1e-11,rtol=0.)
    torch.testing.assert_close(east,speed*u(edges[1:])-.25*ratio*slope,atol=1e-11,rtol=0.)


def test_periodic_transport_rejects_incompatible_data():
    with pytest.raises(ValueError,match="incompatible"):
        solve_advection_diffusion_1d(4,periodic=True,mean_value=1.,source=1.)
    with pytest.raises(ValueError,match="periodic ends"):
        solve_advection_diffusion_1d(4,periodic=True,total_flux=(0.,None))
    with pytest.raises(ValueError,match="velocity must agree"):
        solve_advection_diffusion_1d(4,periodic=True,mean_value=1.,velocity=lambda x:x)


@pytest.mark.parametrize("theta",[.5,1.])
def test_transient_periodic_ring_with_unbalanced_load_and_modal_restart(theta):
    from tensordg.advection_diffusion_1d import solve_transient_advection_diffusion_1d,numerical_face_fluxes
    times = torch.tensor([0.,.03125,.085,.125],dtype=torch.float64)
    edges = times.new_tensor([0.,.15,.45,.8,1.])
    opts = dict(cell_edges=edges,degree=2,periodic=True,peclet=5.,interface_source=[0.,0.,0.,.7])
    initial = lambda x: 1.+.2*torch.cos(2*torch.pi*x)
    centers,history,_ = solve_transient_advection_diffusion_1d(times,initial,theta=theta,**opts)
    for i,dt in enumerate(times.diff()):
        stage = (1-theta)*history[i]+theta*history[i+1]
        west,east = numerical_face_fluxes(stage,5.,periodic=True,cell_edges=edges,
                                          interface_source=opts["interface_source"],return_cell_fluxes=True)
        delta = (history[i+1,:,0]-history[i,:,0])*edges.diff()
        torch.testing.assert_close(delta,dt*(west-east),atol=5e-13,rtol=0.)
        torch.testing.assert_close(delta.sum(),.7*dt,atol=5e-13,rtol=0.)
    _,restarted,_ = solve_transient_advection_diffusion_1d(times[1:],history[1],theta=theta,**opts)
    torch.testing.assert_close(restarted,history[1:])


@pytest.mark.parametrize("theta",[.5,1.])
def test_time_dependent_ring_source_hits_interior_jumps(theta):
    from tensordg.advection_diffusion_1d import solve_transient_advection_diffusion_1d
    # Pulse starts and ends strictly between requested output times.
    times = [0.,.04,.075,.12]
    def source(t,x):
        return (3.+torch.sin(2*torch.pi*x))*float(.018<=t<.083)
    _,u,_ = solve_transient_advection_diffusion_1d(times,0.,theta=theta,ncells=8,
                periodic=True,source=.5,source_history=source,source_jump_times=(.018,.083))
    t = u.new_tensor(times)
    exact_mass = .5*t+3*(t-.018).clamp(min=0,max=.083-.018)
    torch.testing.assert_close(u[:,:,0].mean(dim=1),exact_mass,atol=2e-12,rtol=0.)


@pytest.mark.parametrize("periodic",[False,True])
def test_jump_in_interface_source_matches_manual_substeps(periodic):
    from tensordg.advection_diffusion_1d import solve_transient_advection_diffusion_1d as solve
    edges = torch.tensor([0.,.2,.5,.8,1.],dtype=torch.float64)
    loads = edges.new_tensor([-.2,.7,.1,.3] if periodic else [-.2,.7,.1])
    opts = dict(cell_edges=edges,diffusivity_ratio=[.5,2.,1.,.7],degree=2,periodic=periodic,
                total_flux=(None,None) if periodic else (0.,0.))
    def pulse(t):
        return loads*float(.021<=t<.075)
    _,u,_ = solve([0.,.048,.1],0.,interface_source_history=pulse,source_jump_times=[.021,.075],**opts)
    state = 0.
    for a,b in zip([0.,.021,.048,.075],[.021,.048,.075,.1]):
        _,part,_ = solve([a,b],state,interface_source=pulse(a),**opts)
        state = part[-1]
    torch.testing.assert_close(u[-1],state,atol=1e-13,rtol=0.)
    torch.testing.assert_close(u[-1,:,0] @ edges.diff(),loads.sum()*.054,atol=1e-13,rtol=0.)


def test_transport_damping_restarts_after_internal_jump():
    from tensordg.advection_diffusion_1d import solve_transient_advection_diffusion_1d as solve
    opts = dict(periodic=True,ncells=6,degree=2)
    time = [0.,.03125,.0625,.125]
    jump = .09375
    distribution = torch.tensor([0.,1.,0.,0.,-.5,0.],dtype=torch.float64)
    _,auto,_ = solve(time,0.,rannacher=True,source_jump_times=[jump],
                     interface_source_history=lambda t:distribution*(1. if t<jump else -1.),**opts)
    _,early,_ = solve([0.,.015625,.03125],0.,theta=1.,interface_source=distribution,**opts)
    _,mid,_ = solve([.03125,.0625,jump],early[-1],interface_source=distribution,**opts)
    _,late,_ = solve([jump,.109375,.125],mid[-1],theta=1.,interface_source=-distribution,**opts)
    torch.testing.assert_close(auto,torch.stack((early[0],early[-1],mid[1],late[-1])),atol=2e-14,rtol=0.)


@pytest.mark.parametrize("rannacher",[False,True])
@pytest.mark.parametrize("periodic",[False,True])
def test_interval_face_transport_balances_pulses_and_distributed_load(rannacher,periodic):
    from tensordg.advection_diffusion_1d import solve_transient_advection_diffusion_1d as solve
    edges = torch.tensor([0.,.15,.45,.8,1.],dtype=torch.float64)
    times = [0.,.041,.11]
    pulse = edges.new_tensor([-.3,1.2,.5,-.8] if periodic else [-.3,1.2,.5])
    opts = {} if periodic else dict(boundary_exchange=((1.,.8),(.7,-.2)),total_flux=(.3,-.1))
    _,u,_,(west,east) = solve(times,.35,degree=2,cell_edges=edges,periodic=periodic,**opts,
                  diffusivity_ratio=[1.,3.,.6,2.],rannacher=rannacher,interface_source=.1,
                  source_history=lambda t,x:.7*float(.018<=t<.067),
                  interface_source_history=lambda t:pulse*float(.018<=t<.067),
                  source_jump_times=(.018,.067),return_face_transport=True)
    on = edges.new_tensor([.023,.026])
    delta = u[:,:,0].diff(dim=0)*edges.diff()
    torch.testing.assert_close(delta,west-east+.7*on[:,None]*edges.diff(),atol=2e-12,rtol=0.)
    jump = west.roll(-1,dims=1)-east if periodic else west[:,1:]-east[:,:-1]
    torch.testing.assert_close(jump,on[:,None]*pulse+.1*edges.new_tensor(times).diff()[:,None],atol=2e-12,rtol=0.)


@pytest.mark.parametrize("rannacher",[False,True])
def test_pulsed_end_fluxes_and_constant_offsets(rannacher):
    from tensordg.advection_diffusion_1d import solve_transient_advection_diffusion_1d
    t = [0.,.041,.11]
    _,u,_,(west,east) = solve_transient_advection_diffusion_1d(t,.2,ncells=6,total_flux=(.2,-.1),
              total_flux_history=lambda t:(1.,-.4) if .018<=t<.067 else (0.,0.),source_jump_times=(.018,.067),
              rannacher=rannacher,return_face_transport=True)
    dt,on = u.new_tensor(t).diff(),u.new_tensor([.023,.026])
    torch.testing.assert_close(west[:,0],.2*dt+on,atol=1e-14,rtol=0.)
    torch.testing.assert_close(east[:,-1],-.1*dt-.4*on,atol=1e-14,rtol=0.)
    torch.testing.assert_close(u[:,:,0].diff(dim=0)/6,west-east,atol=1e-12,rtol=0.)


def test_end_flux_history_rejects_dirichlet_load():
    from tensordg.advection_diffusion_1d import solve_transient_advection_diffusion_1d
    with pytest.raises(ValueError,match="Dirichlet"):
        solve_transient_advection_diffusion_1d([0.,.1],0.,ncells=4,total_flux=(None,0.),
                                               total_flux_history=lambda t:(1.,0.))


@pytest.mark.parametrize("speed",[-1.,1.])
@pytest.mark.parametrize("rannacher",[False,True])
def test_switched_dirichlet_offsets_match_piecewise_static_ends(speed,rannacher):
    from tensordg.advection_diffusion_1d import solve_transient_advection_diffusion_1d as solve
    opts = dict(ncells=5,degree=2,speed=speed,left_value=.2,right_value=-.3,return_face_transport=True)
    jump,end = .03125,.09375
    def values(t):
        return (.8,-.4) if t<jump else (-.3,.6)
    _,history,_,(west,east) = solve([0.,end],.1,**opts,rannacher=rannacher,
                        boundary_value_history=values,source_jump_times=[jump])
    previous = .1
    expected_w,expected_e = 0.,0.
    for a,b in ((0.,jump),(jump,end)):
        l,r = values(a)
        static = dict(opts,left_value=.2+l,right_value=-.3+r)
        _,part,_,(w,e) = solve([a,b],previous,**static,rannacher=rannacher)
        previous = part[-1]
        expected_w,expected_e = expected_w+w,expected_e+e
    torch.testing.assert_close(history[-1],previous,atol=1e-12,rtol=0.)
    torch.testing.assert_close(west,expected_w,atol=1e-12,rtol=0.)
    torch.testing.assert_close(east,expected_e,atol=1e-12,rtol=0.)


def test_simultaneous_mixed_boundary_histories_reproduce_affine_transient():
    from tensordg.advection_diffusion_1d import solve_transient_advection_diffusion_1d
    # u=(1+x)*t gives u_t+J_x=1+x+t and J(1,t)=(2-D)*t.
    times = torch.tensor([0.,.04,.1],dtype=torch.float64)
    x,u,_,(west,east) = solve_transient_advection_diffusion_1d(times,0.,ncells=5,degree=1,peclet=4.,
                total_flux=(None,0.),boundary_value_history=lambda t:(t,0.),
                total_flux_history=lambda t:(0.,1.75*t),source_history=lambda t,x:1+x+t,
                return_face_transport=True)
    torch.testing.assert_close(u[:,:,0],times[:,None]*(1+x),atol=1e-12,rtol=0.)
    integral = .5*times.square().diff()
    torch.testing.assert_close(west[:,0],.75*integral,atol=1e-12,rtol=0.)
    torch.testing.assert_close(east[:,-1],1.75*integral,atol=1e-12,rtol=0.)
