"""Regression pins for the isothermal-wall closure of ``ns_dg_2d``.

The pure-conduction channel is the only public entry of ``ns_dg_2d`` whose
verification was never closed (``validation/ns_heat_wall_closure/evidence.json``,
``docs/reports/tensordg-ns-heat-wall-closure.md``).  These tests pin what the
closure *provably* does today so a future fix has to show up as a red-to-green
transition instead of a silent drift:

* the closure is *consistent* -- on a state that the DG space represents
  exactly (``T = p/rho = 1``, no motion, isothermal walls) the residual is
  machine zero, walls included;
* the mass and momentum rows of the exact steady state are machine zero, so a
  wall defect can only live in the energy row;
* the isothermal-wall heat flux is first order, not second, at ``P1``

Run: ``PYTHONPATH=. python -m pytest -q tests/test_ns_heat_wall_closure.py``
"""
from __future__ import annotations

import torch

from tensordg import ns_dg_2d as ns
from tensordg.euler_dg_2d import _initial_coefficients, legendre_operators

MU = 0.1
PRANDTL = 0.71
KAPPA = MU * ns.CP / PRANDTL
PRESSURE = 1.0
T_BOTTOM, T_TOP = 1.0, 0.5


def _full(x, y):
    """Force the full quadrature grid: a y-only profile would collapse the x axis."""
    return x + torch.zeros_like(y), y + torch.zeros_like(x)


def _uniform(xq, yq):
    x, _ = _full(xq, yq)
    one = torch.ones_like(x)
    return one, torch.zeros_like(x), torch.zeros_like(x), one.clone()


def _conduction(xq, yq):
    x, y = _full(xq, yq)
    temperature = T_BOTTOM + (T_TOP - T_BOTTOM) * y
    return (PRESSURE / temperature, torch.zeros_like(x), torch.zeros_like(x),
            torch.full_like(temperature, PRESSURE))


def _boundary(t_bottom=T_BOTTOM, t_top=T_TOP):
    return {"xmin": None, "xmax": None,
            "ymin": {"velocity": (0.0, 0.0), "temperature": t_bottom},
            "ymax": {"velocity": (0.0, 0.0), "temperature": t_top}}


def _setup(ncells, degree=1, initial=_conduction, nx=2):
    dtype = torch.float64
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, max(degree + 2, 3), "cpu", dtype)
    dx, dy = 1.0 / nx, 1.0 / ncells
    xc = (torch.arange(nx, dtype=dtype) + 0.5) * dx
    yc = (torch.arange(ncells, dtype=dtype) + 0.5) * dy
    coeff = _initial_coefficients(initial, xc, yc, dx, dy, nodes, weights, basis,
                                  "cpu", dtype)
    penalty = 2.0 * (degree + 1.0) ** 2
    return {"coeff": coeff, "basis": basis, "derivative": derivative,
            "weights": weights, "signs": signs, "dx": dx, "dy": dy,
            "penalty": penalty}


def _residual(setup, wall_gradient="face", boundary=None):
    return ns._rhs_ns(setup["coeff"], setup["basis"], setup["derivative"], setup["weights"],
                      setup["signs"], setup["dx"], setup["dy"], (True, False),
                      boundary or _boundary(), MU, KAPPA, setup["penalty"], wall_gradient)


def _wall_flux_error(setup, wall_gradient="face"):
    """Max |closure heat flux - kappa dT/dy| on either wall."""
    coeff, basis = setup["coeff"], setup["basis"]
    right = ns._face_primitives(coeff, basis, setup["derivative"], setup["signs"], 1, "right")
    left = ns._face_primitives(coeff, basis, setup["derivative"], setup["signs"], 1, "left")
    exact = KAPPA * (T_TOP - T_BOTTOM)
    errors = []
    for side, wall, index in ((left, _boundary()["ymin"], 0), (right, _boundary()["ymax"], -1)):
        flux = ns._boundary_flux(side, wall, 1, setup["dy"], MU, KAPPA, setup["penalty"],
                                 2.0 / setup["dy"], 2.0 / setup["dx"])
        key = [slice(None)] * flux.dim()
        key[1] = index
        # the closure returns F_inv - F_visc, i.e. minus the physical kappa dT/dy
        errors.append(float((flux[tuple(key)][..., 3] + exact).abs().max()))
    return max(errors)


def test_exactly_representable_state_has_machine_zero_residual():
    """Consistency: a state the DG space holds exactly must not produce a residual."""
    for ncells in (8, 16):
        setup = _setup(ncells, initial=_uniform)
        for wall_gradient in ("face", "taylor"):
            residual = _residual(setup, wall_gradient, _boundary(1.0, 1.0))
            assert float(residual.abs().max()) < 1e-12, (ncells, wall_gradient)


def test_exact_steady_state_keeps_mass_and_momentum_rows_machine_zero():
    """The wall defect is confined to the energy row, so the fix is local."""
    setup = _setup(16)
    residual = _residual(setup).abs().amax(dim=(2, 3))  # (nx, ny, 4)
    assert float(residual[..., 0].max()) < 1e-12          # mass
    assert float(residual[..., 1:3].max()) < 1e-12        # both momentum rows
    # the energy row of the wall cells does *not* vanish -- that is the defect
    assert float(residual[:, 0, 3].max()) > 1e-2


def test_wall_heat_flux_is_first_order_for_both_gradients():
    """Pinned behaviour: halving h halves the wall heat-flux error (design order 2).

    ``wall_gradient="taylor"`` is pinned too -- the m3139 note measured that the
    three-point stencil does not lift the order at ``P1`` either.
    """
    for wall_gradient in ("face", "taylor"):
        coarse = _wall_flux_error(_setup(16), wall_gradient)
        fine = _wall_flux_error(_setup(32), wall_gradient)
        ratio = coarse / fine
        assert 1.7 < ratio < 2.4, (wall_gradient, coarse, fine, ratio)


def test_interior_energy_residual_decays_but_wall_rows_do_not():
    """Discriminates a boundary closure defect from a global discretisation defect.

    Measured on ``wall_gradient="face"`` (the default) at ``P1``::

        ny    bottom-wall |R_E|   interior max |R_E|
         8       1.9561e-01         2.9398e-01
        16       1.5739e-01         1.7093e-01
        32       1.3985e-01         9.2912e-02
        64       1.3143e-01         4.8554e-02

    The interior row halves under refinement (16/32 = 1.8397, i.e. O(h)) while
    the wall row only crawls towards its floor: 16/32 = 1.1254, 32/64 = 1.0640,
    and |R_E| stays O(1) (1.31e-1) instead of vanishing.  The bound here is set
    from those numbers (6% margin over the measured 16/32 ratio) instead of the
    earlier 1.1, which sat 2.3% *below* the measured 1.1254 and made the pin red
    on every runner.
    """
    coarse = _residual(_setup(16)).abs().amax(dim=(2, 3))    # ny = 16
    mid = _residual(_setup(32)).abs().amax(dim=(2, 3))       # ny = 32
    fine = _residual(_setup(64)).abs().amax(dim=(2, 3))      # ny = 64
    interior_ratio = float(coarse[:, 1:-1, 3].max() / mid[:, 1:-1, 3].max())
    wall_16_32 = float(coarse[:, 0, 3].max() / mid[:, 0, 3].max())
    wall_32_64 = float(mid[:, 0, 3].max() / fine[:, 0, 3].max())
    assert interior_ratio > 1.7                 # ~O(h) inside
    assert wall_16_32 < 1.2                     # pinned: does not decay at the wall
    assert wall_32_64 < 1.1                     # ... the ratio keeps falling to 1
    assert wall_32_64 < wall_16_32              # monotone stagnation, not noise
    assert float(fine[:, 0, 3].max()) > 1e-2    # O(1) floor, not machine zero


def test_conduction_entry_point_completes_and_stays_physical():
    """Ratchet flip (2026-11-04): the entry point no longer aborts.

    This replaces ``test_conduction_entry_point_still_crashes_on_the_documented_default``,
    whose ``pytest.raises(FloatingPointError)`` stopped firing once main's m3139
    wall repair landed (one-sided DG wall derivative as the default, plus the
    positivity filter).  The crash guard is still in place, so what is pinned
    here is completion *and* physicality.  Measured at ``ncells=(2, 8)``,
    ``final_time=0.5``: p in [7.3238e-01, 9.2602e-01], rho in
    [9.1246e-01, 1.3363e+00], theta = p/rho monotone in y from 0.9841 (hot
    wall) down to 0.5617 (cold wall), 1070 explicit steps.

    Accuracy is deliberately *not* asserted: the wall closure is still
    inconsistent (see ``test_interior_energy_residual_decays_but_wall_rows_do_not``),
    so this transient carries a 1.23e-1 offset against the exact conduction
    profile -- completion must not be read as verification.
    """
    x, y, coeff, metadata = ns.solve_heat_conduction_dg_2d(
        ncells=(2, 8), final_time=0.5, dtype=torch.float64)
    assert torch.isfinite(coeff).all()
    fields = ns.cell_fields(coeff)
    pressure, density = fields["pressure"], fields["density"]
    assert float(pressure.min()) > 0.0 and float(density.min()) > 0.0
    # R = 1 in this package, so theta = p / rho; y runs along the second axis
    temperature = pressure / density
    assert bool((temperature[0][1:] <= temperature[0][:-1]).all()), "profile must be monotone"
    assert float(temperature.min()) > T_TOP - 1e-2
    assert float(temperature.max()) < T_BOTTOM + 1e-2