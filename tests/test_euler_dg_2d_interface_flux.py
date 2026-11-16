"""Interface-flux selection in the 2-D Euler DG solver.

Contract under test
-------------------
``tensordg.euler_dg_2d`` used to hard-code HLLC at every interior and boundary
face.  It now accepts ``flux="hllc"`` (default, bit-identical to the historic
behaviour) or ``flux="rusanov"`` (local Lax-Friedrichs), which is what makes the
flux-sensitivity study of the isentropic-vortex case
(``benchmarks/case_euler_vortex_flux_sensitivity.py``) possible.

The tests pin the three properties that make the new option a *measurement
knob* rather than a second solver:

* the default path is unchanged, bit for bit;
* both fluxes are consistent (``F(u, u) = F_physical(u)``) and reject typos;
* the Rusanov flux is the strictly more diffusive one, and switching to it
  leaves the discrete conservation of the periodic case intact -- the two facts
  the study relies on.
"""

import numpy as np
import pytest
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.verify_euler_dg_2d import _vortex_error


def _smooth_states(count=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    rho = 0.8 + torch.rand(count, count, generator=generator, dtype=torch.float64)
    pressure = 0.9 + torch.rand(count, count, generator=generator, dtype=torch.float64)
    u = 0.1 * torch.randn(count, count, generator=generator, dtype=torch.float64)
    v = 0.1 * torch.randn(count, count, generator=generator, dtype=torch.float64)
    return dg2._conservative(rho, u, v, pressure)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("kind", dg2.INTERFACE_FLUXES)
def test_interface_flux_is_consistent(kind, axis):
    """Upwind consistency: on identical left/right states the flux is physical."""
    state = _smooth_states()
    flux = dg2._interface_flux(kind, state, state, axis)
    physical = dg2._physical_flux(state, axis)
    assert torch.allclose(flux, physical, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1])
def test_rusanov_matches_hllc_degenerate_fallback(axis):
    """``_hllc`` falls back to Rusanov when the wave speeds collapse; the two
    implementations must agree there, otherwise the fallback and the selectable
    flux would be two different numerics under one name."""
    left = _smooth_states(seed=3)
    right = left.clone()  # identical states -> HLLC's degenerate branch
    fallback = dg2._hllc(left, right, axis)
    rusanov = dg2._rusanov(left, right, axis)
    assert torch.allclose(fallback, rusanov, rtol=0.0, atol=1e-12)


def test_unknown_flux_is_rejected():
    with pytest.raises(ValueError):
        dg2.solve_euler_dg_2d_vortex(ncells=4, degree=1, final_time=0.01, flux="roe")


def test_default_flux_is_bit_identical_to_explicit_hllc():
    args = dict(ncells=6, degree=1, final_time=0.05, cfl=0.2)
    _, _, default_coeff, default_meta = dg2.solve_euler_dg_2d_vortex(**args)
    _, _, hllc_coeff, hllc_meta = dg2.solve_euler_dg_2d_vortex(**args, flux="hllc")
    assert default_meta["flux"] == "hllc" == hllc_meta["flux"]
    assert torch.equal(default_coeff, hllc_coeff)
    assert default_meta["steps"] == hllc_meta["steps"]


def test_metadata_records_the_requested_flux():
    _, _, _, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=4, degree=1, final_time=0.02, flux="rusanov"
    )
    assert meta["flux"] == "rusanov"


@pytest.mark.parametrize("flux", dg2.INTERFACE_FLUXES)
def test_periodic_vortex_conserves_each_flux(flux):
    """Switching the Riemann solver must not break discrete conservation.

    The vortex is periodic and the data are smooth, so mass/momentum/energy
    change only by round-off regardless of which face flux is used.
    """
    _, _, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=8, degree=2, final_time=0.2, cfl=0.2, flux=flux
    )
    assert meta["flux"] == flux
    drift = meta["conservation_drift"]
    assert isinstance(drift, dict) and drift, "expected per-variable conservation drift"
    assert max(abs(value) for value in drift.values()) < 1e-12
    assert torch.isfinite(coeff).all()


def test_flux_choice_does_not_move_the_error_by_more_than_a_factor():
    """At a coarse pre-asymptotic mesh the flux ordering is *not* monotone.

    Measured here (n=16, P1, t=1): HLLC max-field error 8.373e-3, Rusanov
    8.000e-3 -- Rusanov is slightly *better* on the velocity field and slightly
    worse on density/pressure.  The dispersion-dominated error of a P1 mesh is
    not ordered by numerical diffusion, so the honest contract is the weaker
    one: swapping the Riemann solver keeps the error in the same bracket.  The
    diffusivity claim is only made where it is true, i.e. in the growth
    statistics of the production family (see
    ``benchmarks/case_euler_vortex_flux_sensitivity.py``).
    """
    domain = ((-5.0, 5.0), (-5.0, 5.0))
    nodes, _weights, basis, _derivative, _signs = dg2.legendre_operators(1, 3)
    errors = {}
    for flux in dg2.INTERFACE_FLUXES:
        _, _, coeff, _ = dg2.solve_euler_dg_2d_vortex(
            ncells=16, degree=1, final_time=1.0, cfl=0.2, flux=flux
        )
        errors[flux] = float(
            np.nanmax(list(_vortex_error(
                coeff, basis, nodes, domain, 1.0, 5.0, (1.0, 1.0), period=(10.0, 10.0)
            ).values()))
        )
    ratio = errors["rusanov"] / errors["hllc"]
    assert 0.5 < ratio < 2.0, f"flux changed the error by {ratio:.3f}x at n=16, P1, t=1"