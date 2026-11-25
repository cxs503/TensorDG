"""Verification of the per-side Neumann/Dirichlet SIPG Poisson solver.

Three manufactured solutions on the unit square (see
``docs/reports/tensordg-poisson-neumann-verification.md``):

* ``A``  pure Neumann, ``u = cos(pi x) cos(pi y)`` -- exercises the singular
  operator, the constant nullspace and the rank-one gauge;
* ``B``  mixed, ``u = cos(pi x/2) cos(pi y/2)`` -- Neumann on the left/bottom
  edges, Dirichlet on the right/top edges (no nullspace);
* ``C``  mixed with an *inhomogeneous* flux, ``u = sin(pi x) sin(pi y)`` with
  ``du/dn = -pi sin(pi y)`` on the left edge -- exercises the Neumann load.

The all-Dirichlet assembly is also checked to be bit-for-bit identical to the
shipped :func:`tensordg.solve_poisson_2d`, which pins the shared face operators.
"""
import math

import pytest
import torch

from tensordg.poisson_2d import poisson_error_2d, solve_poisson_2d
from tensordg.poisson_2d_bc import (
    assemble_poisson_2d_bc,
    l2_error_2d,
    solve_poisson_2d_bc,
)

PI = math.pi
NEUMANN_ALL = {side: "neumann" for side in ("left", "right", "bottom", "top")}


def _forcing_sine(x, y):
    return 2.0 * PI ** 2 * torch.sin(PI * x) * torch.sin(PI * y)


def _exact_sine(x, y):
    return torch.sin(PI * x) * torch.sin(PI * y)


def _forcing_cosine(x, y):
    return 2.0 * PI ** 2 * torch.cos(PI * x) * torch.cos(PI * y)


def _exact_cosine(x, y):
    return torch.cos(PI * x) * torch.cos(PI * y)


def _forcing_quarter(x, y):
    return 0.5 * PI ** 2 * torch.cos(0.5 * PI * x) * torch.cos(0.5 * PI * y)


def _exact_quarter(x, y):
    return torch.cos(0.5 * PI * x) * torch.cos(0.5 * PI * y)


def _flux_left_sine(x, y):
    """Outward normal derivative of sin(pi x) sin(pi y) on x = 0."""
    return -PI * torch.sin(PI * y)


def test_all_dirichlet_reproduces_reference_solver_exactly():
    """The shared face operators must give the shipped Dirichlet answer."""
    for degree, cells in ((1, 8), (2, 8)):
        x_ref, y_ref, coeff_ref, _ = solve_poisson_2d(ncells=cells, degree=degree)
        x, y, coeff, meta = solve_poisson_2d_bc(
            ncells=cells, degree=degree, forcing=_forcing_sine,
        )
        assert meta["bc"] == {side: "dirichlet" for side in NEUMANN_ALL}
        assert float(torch.max(torch.abs(coeff - coeff_ref)).item()) == 0.0
        assert poisson_error_2d(x_ref, y_ref, coeff_ref)["relative_l2"] == pytest.approx(
            l2_error_2d(x, y, coeff, _exact_sine)["relative_l2"], rel=1e-12,
        )


def test_pure_neumann_operator_annihilates_constants_and_gauge_is_zero_sum():
    matrix, _, _, meta = assemble_poisson_2d_bc(
        ncells=8, degree=1, bc=NEUMANN_ALL, forcing=_forcing_cosine,
    )
    nx, ny = meta["ncells"]
    local = meta["degree"] + 1
    ones = torch.zeros(matrix.shape[0], dtype=matrix.dtype)
    ones.view(nx, ny, local * local)[:, :, 0] = 1.0
    relative = float(torch.max(torch.abs(matrix @ ones)).item()) / float(
        torch.max(torch.abs(matrix)).item()
    )
    assert relative < 1e-12, relative

    x, y, coeff, meta = solve_poisson_2d_bc(
        ncells=8, degree=1, bc=NEUMANN_ALL, forcing=_forcing_cosine,
    )
    assert meta["nullspace_residual"] < 1e-12
    assert meta["gauge"].startswith("rank-one")
    # The gauge fixes the additive constant: the pointwise mean of the
    # difference to the (zero-mean) exact field must vanish.
    error = l2_error_2d(x, y, coeff, _exact_cosine, remove_constant=True)
    assert abs(error["constant_offset"]) < 1e-9, error["constant_offset"]


def test_pure_neumann_cosine_meets_order_and_three_percent_gate():
    errors = []
    for cells in (8, 16):
        x, y, coeff, _ = solve_poisson_2d_bc(
            ncells=cells, degree=1, bc=NEUMANN_ALL, forcing=_forcing_cosine,
        )
        errors.append(l2_error_2d(x, y, coeff, _exact_cosine, remove_constant=True)["relative_l2"])
    order = math.log2(errors[0] / errors[1])
    assert order > 1.8, (errors, order)
    assert errors[-1] < 0.03, (errors, order)


def test_mixed_dirichlet_neumann_quarter_cosine_converges_at_theory_order():
    errors_p1, errors_p2 = [], []
    for cells in (8, 16):
        x, y, coeff, meta = solve_poisson_2d_bc(
            ncells=cells, degree=1,
            bc={"left": "neumann", "bottom": "neumann"}, forcing=_forcing_quarter,
        )
        assert meta["matrix_symmetry_error"] < 1e-11
        errors_p1.append(l2_error_2d(x, y, coeff, _exact_quarter)["relative_l2"])
        x, y, coeff, _ = solve_poisson_2d_bc(
            ncells=cells, degree=2,
            bc={"left": "neumann", "bottom": "neumann"}, forcing=_forcing_quarter,
        )
        errors_p2.append(l2_error_2d(x, y, coeff, _exact_quarter)["relative_l2"])
    order_p1 = math.log2(errors_p1[0] / errors_p1[1])
    order_p2 = math.log2(errors_p2[0] / errors_p2[1])
    assert order_p1 > 1.8, (errors_p1, order_p1)
    assert order_p2 > 2.7, (errors_p2, order_p2)
    assert errors_p1[-1] < 0.03 and errors_p2[-1] < 0.03


def test_inhomogeneous_neumann_flux_is_required_and_correct():
    """The flux load must matter: dropping it is a large, detectable error."""
    x, y, coeff, _ = solve_poisson_2d_bc(
        ncells=16, degree=1, bc={"left": "neumann"}, forcing=_forcing_sine,
        neumann_flux={"left": _flux_left_sine},
    )
    used = l2_error_2d(x, y, coeff, _exact_sine)["relative_l2"]
    x0, y0, coeff0, _ = solve_poisson_2d_bc(
        ncells=16, degree=1, bc={"left": "neumann"}, forcing=_forcing_sine,
    )
    dropped = l2_error_2d(x0, y0, coeff0, _exact_sine)["relative_l2"]
    assert used < 0.03, used
    assert dropped > 5.0 * used, (dropped, used)
    assert dropped > 0.01, dropped


def test_boundary_kind_and_side_validation():
    with pytest.raises(ValueError):
        solve_poisson_2d_bc(ncells=4, degree=1, bc={"west": "neumann"}, forcing=_forcing_sine)
    with pytest.raises(ValueError):
        solve_poisson_2d_bc(ncells=4, degree=1, bc={"left": "robin"}, forcing=_forcing_sine)
    with pytest.raises(ValueError):
        solve_poisson_2d_bc(ncells=4, degree=1, penalty=0.0, forcing=_forcing_sine)
    with pytest.raises(TypeError):
        solve_poisson_2d_bc(ncells=4, degree=1)

@pytest.mark.parametrize("side,sign", [("top", 1.), ("bottom", -1.)])
def test_nonzero_dirichlet_and_tangentially_varying_neumann_on_rectangle(side, sign):
    exact = lambda x,y: .7+2*x-.3*y+x*y
    x, y, u, _ = solve_poisson_2d_bc((3, 5), 1, bc={side:"neumann"},
            forcing=lambda x,y: 0*x+0*y, neumann_flux={side:lambda x,y: sign*(x-.3)},
            dirichlet_values={s:exact for s in NEUMANN_ALL if s != side})
    assert l2_error_2d(x, y, u, exact)["absolute_l2"] < 2e-13


def test_numerical_face_power_local_source_and_boundary_signs():
    from tensordg import poisson_face_powers_2d

    # Quadratic exact solution lies in the P2 space.
    exact = lambda x,y: .4+x*x+.5*y*y+x*y
    bc = {"bottom":"neumann"}
    data = dict(bc=bc, neumann_flux={"bottom":lambda x,y: -x},
                dirichlet_values={s:exact for s in ("top","left","right")})
    x, y, u, _ = solve_poisson_2d_bc((3,5), 2, forcing=lambda x,y: -3+0*x+0*y, **data)
    qx, qy = poisson_face_powers_2d(u, **data)
    xf, yf = torch.arange(4,dtype=x.dtype)/3, torch.arange(6,dtype=x.dtype)/5
    torch.testing.assert_close(qx, -(2*xf[:,None]+y)/5, atol=3e-13, rtol=0)
    torch.testing.assert_close(qy, -(x[:,None]+yf)/3, atol=3e-13, rtol=0)
    torch.testing.assert_close(qx.diff(dim=0)+qy.diff(dim=1), torch.full_like(u[:,:,0,0], -.2),
                               atol=1e-12, rtol=0)


def test_pure_neumann_heat_balance_and_reference_offset():
    from tensordg import poisson_face_powers_2d

    # u=x²+xy, with independently prescribed outward derivatives on all walls.
    data = dict(bc=NEUMANN_ALL, neumann_flux={"right": lambda x,y: 2+y, "top": lambda x,y: x,
                "bottom": lambda x,y: -x, "left": lambda x,y: -y})
    x, y, u, meta = solve_poisson_2d_bc((3,5), 2, forcing=lambda x,y: -2+0*x+0*y, **data)
    assert abs(meta["net_neumann_load"]) < 2e-14
    qx, qy = poisson_face_powers_2d(u, **data)
    torch.testing.assert_close(qx.diff(dim=0)+qy.diff(dim=1), torch.full_like(u[:,:,0,0], -2/15),
                               rtol=0, atol=1e-12)
    for shift in (0., 8.):
        err = l2_error_2d(x,y,u, lambda x,y: x*x+x*y+shift, remove_constant=True)
        assert err["constant_offset"] == pytest.approx(-1/3-1/4-shift, abs=2e-12)
        assert err["absolute_l2"] < 2e-13
    with pytest.raises(ValueError, match="incompatible pure Neumann"):
        solve_poisson_2d_bc((3,5), 2, forcing=lambda x,y: 1+0*x+0*y, bc=NEUMANN_ALL)


def test_variable_robin_film_on_all_sides_and_zero_film_gauge():
    from tensordg import poisson_face_powers_2d

    exact = lambda x,y: 1.+x*x+x*y+.5*y*y
    h = lambda x,y: 1.+x+2*y
    deriv = dict(left=lambda x,y: -(2*x+y), right=lambda x,y: 2*x+y,
                 bottom=lambda x,y: -(x+y), top=lambda x,y: x+y)
    films = {s:(h, lambda x,y, dn=dn: exact(x,y)+dn(x,y)/h(x,y)) for s,dn in deriv.items()}
    bc = dict.fromkeys(deriv, "robin")
    x,y,u,meta = solve_poisson_2d_bc((3,5), 2, forcing=lambda x,y: 0*x+0*y-3, bc=bc, robin_films=films)
    assert meta["temperature_anchored"]
    assert l2_error_2d(x,y,u,exact)["absolute_l2"] < 5e-12
    qx,qy = poisson_face_powers_2d(u, bc=bc, robin_films=films)
    torch.testing.assert_close(qx.diff(dim=0)+qy.diff(dim=1), torch.full_like(u[:,:,0,0], -.2), atol=1e-12, rtol=0)
    zero_films = dict.fromkeys(bc, (0., 2.))
    *_, meta = solve_poisson_2d_bc(3, bc=bc, robin_films=zero_films, forcing=lambda x,y: 0*x+0*y)
    assert meta["gauge"].startswith("rank-one")
    with pytest.raises(ValueError, match="incompatible"):
        solve_poisson_2d_bc(3, bc=bc, robin_films=zero_films, forcing=lambda x,y: 1+0*x+0*y)
