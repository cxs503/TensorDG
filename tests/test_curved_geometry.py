"""Tests for the curved isoparametric geometry layer (module
``tensordg.curved_geometry``) and for the curved-geometry verification gates.

The gates implemented here are the cheap versions of the campaign study
(``benchmarks/curved_geometry_study.py``):

* J1 free-stream preservation on curved meshes,
* J2 metric (Piola) identity: algebraic, nodal (SBP) and coefficient-space,
* J3 curved-wall analytic order (smoke test on two coarse mesh levels),
* J4 negative control: an exactly affine mesh gives a *bitwise* zero defect,
* J5 CAD/NURBS contract: exact circle, invalid payloads rejected, and the
  CAD-derived DG geometry still preserves a free stream.

The tests also contain a mutation check (a deliberately inconsistent metric must
break free-stream preservation) so that the gate is known to have detection
power, and a regression test for the geometric-limiting root cause found in the
study (a polygonal wall at p=1 caps the order at one).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordg.curved_geometry import (
    CONTRACT_SCHEMA,
    CurvedAdvection,
    IsoparametricMap,
    annulus_mesh,
    cartesian_periodic_mesh,
    circle_contract,
    gauss_nodes_weights,
    gll_differentiation_matrix,
    gll_nodes_weights,
    legendre_derivative_matrix,
    legendre_eval_matrix,
    mesh_from_contract,
    metric_identity_report,
    nurbs_curve_points,
    validate_geometry_contract,
    warped_periodic_mesh,
)

VEL = (0.6, 0.8)


def const_vel(x, y):
    return np.full_like(x, VEL[0]), np.full_like(y, VEL[1])


def rotation_vel(x, y, omega=1.0):
    return -omega * y, omega * x


def wave(x, y, t):
    return np.sin(2 * np.pi * (x - VEL[0] * t)) * np.cos(2 * np.pi * (y - VEL[1] * t))


def rotation_exact(x, y, t, omega=1.0, mode=2, r_mid=1.5, width=0.35):
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    return np.exp(-((r - r_mid) / width) ** 2) * np.cos(mode * (th - omega * t))


# ---------------------------------------------------------------------------
# 1-D Legendre / quadrature building blocks
# ---------------------------------------------------------------------------
def test_legendre_derivative_matrix_matches_polynomial_derivative():
    rng = np.random.default_rng(7)
    for p in (1, 3, 6):
        c = rng.standard_normal(p + 1)
        exact = np.polynomial.legendre.legder(c)
        ours = legendre_derivative_matrix(p) @ c
        np.testing.assert_allclose(ours[: exact.size], exact, atol=1e-12)
        np.testing.assert_allclose(ours[exact.size:], 0.0, atol=1e-13)


def test_derivative_of_constant_is_bitwise_zero():
    for p in range(1, 8):
        c = np.zeros(p + 1)
        c[0] = 3.25
        assert np.all(legendre_derivative_matrix(p) @ c == 0.0)


def test_gll_nodes_weights():
    for p in range(1, 9):
        x, w = gll_nodes_weights(p)
        assert x.size == w.size == p + 1
        assert x[0] == -1.0 and x[-1] == 1.0
        assert np.all(np.diff(x) > 0.0)
        np.testing.assert_allclose(w.sum(), 2.0, atol=1e-14)
        np.testing.assert_allclose(x, -x[::-1], atol=1e-15)   # symmetry


def test_gll_differentiation_is_exact_and_annihilates_constants():
    for p in range(1, 7):
        x, _ = gll_nodes_weights(p)
        D = gll_differentiation_matrix(p)
        np.testing.assert_allclose(D @ np.ones(p + 1), 0.0, atol=1e-14)
        for k in range(p + 1):
            c = np.zeros(p + 1)
            c[k] = 1.0
            values = legendre_eval_matrix(p, x) @ c
            dv = legendre_eval_matrix(p, x) @ (legendre_derivative_matrix(p) @ c)
            np.testing.assert_allclose(D @ values, dv, atol=1e-12)


def test_gauss_rule_exactness():
    for nq in (2, 4, 6):
        x, w = gauss_nodes_weights(nq)
        for k in range(2 * nq):
            got = float(np.sum(w * x ** k))
            want = 0.0 if k % 2 else 2.0 / (k + 1)
            assert abs(got - want) < 1e-13


# ---------------------------------------------------------------------------
# J2: metric identities
# ---------------------------------------------------------------------------
def test_affine_map_metric_divergence_is_bitwise_zero():
    m = IsoparametricMap.affine(0.0, 0.25, 0.0, 0.125, degree=3)
    div = m.metric_divergence_coefficients(*VEL)
    assert np.all(div == 0.0)
    rep = metric_identity_report(m, *VEL)
    assert rep["piola_coeff"] == 0.0
    assert rep["piola_nodal"] < 1e-14
    assert abs(rep["inverse_metric"]) < 1e-14


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
def test_curved_map_metric_identities(p):
    mesh = warped_periodic_mesh(4, 4, p, amplitude=0.08)
    for m in mesh.flat_maps():
        rep = metric_identity_report(m, *VEL)
        assert rep["jacobian_min"] > 0.0
        assert rep["inverse_metric"] < 1e-12      # dxi_i/dx_j dx_j/dxi_k = delta
        assert rep["piola_nodal"] < 1e-12         # SBP-norm Piola identity
        assert rep["piola_coeff"] < 1e-12         # exact polynomial identity


# ---------------------------------------------------------------------------
# J1 / J4: free-stream preservation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [1, 2, 3])
def test_free_stream_preserved_on_curved_mesh(p):
    dg = CurvedAdvection(warped_periodic_mesh(5, 5, p, 0.08), p, const_vel, cfl=0.2)
    fs = dg.free_stream_defect(1.0, steps=40)
    assert fs["u_t_exact"] <= 1e-14
    assert fs["drift"] <= 1e-14


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_free_stream_bitwise_zero_on_straight_mesh(p):
    dg = CurvedAdvection(cartesian_periodic_mesh(5, 5, p), p, const_vel, cfl=0.2)
    fs = dg.free_stream_defect(1.0, steps=40)
    assert fs["u_t_exact"] == 0.0          # bitwise, not merely small
    assert fs["metric_div_coeff"] == 0.0
    assert fs["drift"] <= 1e-14


def test_free_stream_gate_detects_inconsistent_metrics():
    """Mutation check: perturbing the metric fluxes must break free-stream
    preservation, otherwise the J1 gate would be vacuous."""
    dg = CurvedAdvection(warped_periodic_mesh(5, 5, 2, 0.08), 2, const_vel, cfl=0.2)
    clean = dg.free_stream_defect(1.0, steps=10)["drift"]
    dg.a1g = dg.a1g * (1.0 + 1e-6)
    dirty = dg.free_stream_defect(1.0, steps=10)["drift"]
    assert clean <= 1e-14
    assert dirty > 1e-9


def test_linear_field_is_reproduced_exactly_on_affine_mesh():
    """A DG scheme with exact quadrature and consistent metrics is exact for a
    linear field on an affine mesh -- for the interior cells, which are the ones
    that do not see the (non-periodic) jump of ``x + y`` across the seam."""
    nx = ny = 6
    dg = CurvedAdvection(cartesian_periodic_mesh(nx, ny, 2), 2, const_vel)
    U = dg.project(lambda x, y: x + y)
    dU = dg.rhs(U).reshape(nx, ny, -1)[1:, 1:, :]      # drop the seam cells
    np.testing.assert_allclose(dU[:, :, 0], -(VEL[0] + VEL[1]), atol=1e-11)
    rest = np.abs(dU).copy()
    rest[:, :, 0] = 0.0
    assert rest.max() < 1e-11


def test_zero_velocity_gives_zero_operator():
    dg = CurvedAdvection(warped_periodic_mesh(4, 4, 2, 0.06), 2,
                         lambda x, y: (np.zeros_like(x), np.zeros_like(y)))
    rng = np.random.default_rng(3)
    U = rng.standard_normal((dg.mesh.ncells, dg.K, dg.K))
    assert np.max(np.abs(dg.rhs(U))) < 1e-12


# ---------------------------------------------------------------------------
# J5: CAD / NURBS geometry contract
# ---------------------------------------------------------------------------
def test_nurbs_circle_is_exact():
    c = validate_geometry_contract(circle_contract(2.0, (0.5, -0.25)))
    t = np.linspace(0.0, 1.0, 4001)
    pts = nurbs_curve_points(c["degree"], c["knots"], c["control_points"],
                             c["weights"], t)
    rad = np.hypot(pts[:, 0] - 0.5, pts[:, 1] + 0.25)
    assert np.max(np.abs(rad - 2.0)) < 1e-13
    np.testing.assert_allclose(pts[0], pts[-1], atol=1e-13)


def _mutations():
    base = circle_contract(1.0)

    def copy():
        return {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}

    cases = {}

    c = copy(); c["schema"] = "bogus"; cases["schema"] = c
    c = copy(); c["knots"] = list(reversed(c["knots"])); cases["knot_order"] = c
    c = copy(); c["knots"] = c["knots"][:-1]; cases["knot_count"] = c
    c = copy(); c["weights"] = list(c["weights"]); c["weights"][1] = -1.0
    cases["negative_weight"] = c
    c = copy(); c["knots"] = list(c["knots"]); c["knots"][0] = -1.0
    cases["not_clamped"] = c
    c = copy(); c["control_points"] = [list(p) for p in c["control_points"][:-1]]
    cases["open_curve"] = c
    c = copy(); c.pop("degree"); cases["missing_degree"] = c
    c = copy(); c["degree"] = 0; cases["zero_degree"] = c
    return cases


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_contract_validation_rejects_malformed_payloads(name):
    with pytest.raises(ValueError):
        validate_geometry_contract(_mutations()[name])


def test_contract_rejects_non_circular_wall():
    c = circle_contract(1.0)
    c["control_points"] = [list(p) for p in c["control_points"]]
    c["control_points"][2] = [0.0, 3.0]        # no longer on the circle
    with pytest.raises(ValueError):
        mesh_from_contract(c, 4, 8, 2.0, 2)


def test_cad_derived_mesh_is_valid_and_preserves_free_stream():
    contract = circle_contract(1.0)
    mesh = mesh_from_contract(contract, 6, 12, 2.0, 2)
    area = float(np.sum(mesh.areas()))
    assert abs(area - math.pi * 3.0) / (math.pi * 3.0) < 5e-3
    # every wall node lies on the circle (only the element interpolation and the
    # geometry interpolation of the wall remain as error)
    nodes, _ = gll_nodes_weights(2)
    th = (0.5 * (nodes + 1.0)) / 12 * 2 * np.pi
    x, y = mesh.maps[0][0].eval(np.array([-1.0]), th)
    assert np.max(np.abs(np.hypot(x, y) - 1.0)) < 1e-3
    # radially periodic CAD mesh: free stream must be preserved
    mesh_p = mesh_from_contract(contract, 6, 12, 2.0, 2, periodic_radial=True)
    dg = CurvedAdvection(mesh_p, 2, const_vel, cfl=0.2)
    assert dg.free_stream_defect(1.0, steps=40)["drift"] <= 1e-14


# ---------------------------------------------------------------------------
# geometry validation gate (the analogue of a Cb / L / B check: the geometric
# measure must match its analytic value before any physics is believed)
# ---------------------------------------------------------------------------
def test_geometry_area_converges_to_the_analytic_value():
    exact = math.pi * (2.0 ** 2 - 1.0 ** 2)
    errs = []
    for nr in (4, 8):
        mesh = annulus_mesh(nr, 2 * nr, 1.0, 2.0, 2)
        errs.append(abs(float(np.sum(mesh.areas())) - exact) / exact)
    assert errs[0] < 1e-2
    assert errs[1] < errs[0] / 4.0        # ~h^2 geometric convergence


def test_polygonal_wall_area_error_is_visible():
    exact = math.pi * 3.0
    g1 = abs(float(np.sum(annulus_mesh(4, 8, 1.0, 2.0, 1).areas())) - exact) / exact
    g2 = abs(float(np.sum(annulus_mesh(4, 8, 1.0, 2.0, 2).areas())) - exact) / exact
    assert g1 > 5e-2          # a chord-approximated circle loses ~10% of the area
    assert g2 < 1e-2


# ---------------------------------------------------------------------------
# J3: curved-wall analytic order (cheap two-level smoke gate) and the
# geometric-limiting regression
# ---------------------------------------------------------------------------
def _annulus_error(p, gdeg, nr, nt, T=0.15, cfl=0.25):
    mesh = annulus_mesh(nr, nt, 1.0, 2.0, gdeg)
    dg = CurvedAdvection(mesh, p, rotation_vel, cfl=0.25)
    U = dg.project(lambda x, y: rotation_exact(x, y, 0.0))
    U2, dt, _ = dg.evolve(U, T, dg.stable_dt())
    num, den, rel = dg.l2_error(U2, lambda x, y: rotation_exact(x, y, T))
    return rel, float(np.sqrt(np.sum(np.einsum("nqr,q,r->n", dg.Jg, dg.w, dg.w))
                              / mesh.ncells))


def test_curved_wall_order_smoke_p2():
    e1, h1 = _annulus_error(2, 2, 4, 8)
    e2, h2 = _annulus_error(2, 2, 8, 16)
    order = math.log(e1 / e2) / math.log(h1 / h2)
    assert order > 2.4, (order, e1, e2)


def test_polygonal_wall_caps_first_order_convergence():
    """Regression for the root cause of the geometry-limited convergence:
    a p=1 solution on a chord (g=1) wall loses an order, a quadratic wall
    (g=2) restores p+1 = 2."""
    e1, h1 = _annulus_error(1, 1, 4, 8)
    e2, h2 = _annulus_error(1, 1, 8, 16)
    poly_order = math.log(e1 / e2) / math.log(h1 / h2)
    f1, g1 = _annulus_error(1, 2, 4, 8)
    f2, g2 = _annulus_error(1, 2, 8, 16)
    quad_order = math.log(f1 / f2) / math.log(g1 / g2)
    assert poly_order < 1.4, poly_order
    assert quad_order > 1.7, quad_order