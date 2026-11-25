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
