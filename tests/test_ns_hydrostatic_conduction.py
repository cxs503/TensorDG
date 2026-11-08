"""Pins for the representable hydrostatic-conduction equilibrium (m6528).

The isothermal-wall closure of ``tensordg.ns_dg_2d`` was previously measured only
on the L2 *projection* of ``rho = p0/T`` (a rational function, hence outside the
DG space).  The pins below use a state that is inside the space,

    u = v = 0,  T = 1 - y/2,  rho = 1 + a y^2,  p = rho T,  S = (0, 0, dp/dy, 0),

so a consistent closure must hold it to round-off while carrying the nonzero
wall heat flux ``kappa dT/dy = -mu c_p / (2 Pr)``.

Evidence: ``validation/ns_hydrostatic/evidence.json``,
``docs/reports/tensordg-ns-hydrostatic-representability.md``.
"""
from __future__ import annotations

import math

import torch

from tools import verify_ns_hydrostatic_conduction as probe

EXACT = probe.EXACT_FLUX  # kappa dT/dy < 0; the closure returns its negative


def _residual(degree, ny, amplitude):
    coeff, ops, source_quad = probe.build(
        probe.hydrostatic_state(amplitude), degree, ny,
        probe.hydrostatic_source(amplitude))
    return probe.residual(coeff, ops, degree, source_quad)


def test_representable_state_is_a_discrete_steady_state():
    """Round-off residual on a state that lies in P_k (no projection involved)."""
    for degree, amplitude in ((1, 0.0), (3, 0.0), (3, 0.5)):
        for ny in (8, 16):
            resid = _residual(degree, ny, amplitude)
            assert float(resid.abs().max()) < 1e-9, (degree, ny, amplitude)


def test_representable_wall_heat_flux_is_exact_on_both_walls():
    """A nonzero wall heat flux is reproduced to round-off, not at first order."""
    for degree, amplitude in ((1, 0.0), (3, 0.5)):
        coeff, ops, _ = probe.build(probe.hydrostatic_state(amplitude), degree, 16)
        bottom, top = probe.wall_energy_flux(coeff, ops, degree)
        assert float(bottom.mean()) > 0.0  # closure returns -kappa dT/dy
        assert float((bottom + EXACT).abs().max()) < 1e-12
        assert float((top + EXACT).abs().max()) < 1e-12
        assert abs(float(bottom.mean()) + EXACT) / abs(EXACT) < 1e-12


def test_representable_mass_and_momentum_rows_are_machine_zero():
    resid = _residual(3, 16, 0.5).abs()
    assert float(resid[..., 0].max()) < 1e-12
    assert float(resid[..., 1:3].max()) < 1e-9


def test_textbook_wall_trace_defect_is_second_order():
    """rho = p0/T is not representable: its trace defect is O(h^2)."""
    defects = []
    for ny in (8, 32):
        coeff, ops, _ = probe.build(probe.textbook_state, 1, ny)
        bottom, _ = probe.trace_defect(coeff, ops)
        defects.append(bottom)
    order = math.log(defects[0] / defects[1]) / math.log(4.0)
    assert 1.7 < order < 2.4, (defects, order)


def test_textbook_flux_error_is_the_penalty_on_the_projection_defect():
    """|q_h - q| / ((sigma/h) * |T^- - T_D|) is resolution independent (~0.86).

    This is what identifies the measured first-order wall heat flux as a
    *projection* artefact rather than a property of the closure: the same closure
    is exact (1e-13) for the representable state above.
    """
    ratios = []
    for ny in (8, 32):
        coeff, ops, _ = probe.build(probe.textbook_state, 1, ny)
        bottom, _ = probe.wall_energy_flux(coeff, ops, 1)
        defect_bottom, _ = probe.trace_defect(coeff, ops)
        sigma_over_h = 2.0 * (1 + 1.0) ** 2 * ny
        ratios.append(float((bottom + EXACT).abs().max())
                      / (sigma_over_h * defect_bottom))
    assert 0.6 < ratios[0] < 1.1 and 0.6 < ratios[1] < 1.1, ratios
    assert abs(ratios[0] / ratios[1] - 1.0) < 0.1, ratios


def test_textbook_residual_floor_does_not_decay():
    """Honest record: the strict xfail floor of test_ns_heat_conduction is real.

    The strong-form residual is O(1) because the O(h) penalty flux is divided by
    h; it is not a sign error and it does not decay under refinement.
    """
    floors = []
    for ny in (8, 64):
        coeff, ops, _ = probe.build(probe.textbook_state, 1, ny)
        r = probe.rows(probe.residual(coeff, ops, 1))
        floors.append(r["energy_bottom_row"])
    assert floors[0] > 0.05 and floors[1] > 0.05
    assert floors[1] > 0.9 * floors[0], floors


def test_source_free_textbook_state_is_not_a_fixed_point():
    """Contrast pin: dropping the source (textbook) leaves an O(1) residual."""
    coeff, ops, _ = probe.build(probe.hydrostatic_state(0.0), 1, 16)
    without = probe.residual(coeff, ops, 1, None)   # no body force -> not steady
    with_source = _residual(1, 16, 0.0)
    assert float(with_source.abs().max()) < 1e-9
    assert float(without.abs().max()) > 1e-2
