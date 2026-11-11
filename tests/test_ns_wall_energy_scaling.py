"""Contract pins for the *shape* of the isothermal-wall energy-closure defect.

``tests/test_ns_heat_conduction.py`` already pins *that* the wall-cell energy
residual of the exact steady conduction state does not decay.  These pins
record *what the defect is made of*, so a fix can be verified against them
instead of against the "it got smaller" hand-wave:

* the mass and momentum rows of the same state are at machine zero, so the
  defect is confined to the ``kappa dT/dn`` wall flux (positive control);
* it is a pure **cell-average** defect: every higher modal coefficient of the
  wall-cell energy residual is zero, so it is a flux *offset*, not a shape
  error;
* it is exactly **linear in ``kappa``** (relative spread 0 to round-off) and
  **independent of the pressure**, which rules out a viscous-stress or
  pressure-work origin;
* the interior energy residual still decays at first order, so the interior
  interface procedure is not what breaks;
* the measured asymptote is ``residual_wall -> kappa * rho_wall``
  (``rho_bottom = p / T_bottom``, ``rho_top = p / T_top``), i.e. the wall face
  and the interior face of the wall cell disagree by a systematic ``O(h)``
  flux term that carries one power of the density the quotient rule for
  ``T = p / rho`` divides out.  These last two pins are *positive* ratchets on
  the measured defect (1.2 % / 2.3 % on ``ny = 128``): a fix that removes the
  offset will fail them, which is the intended re-adjudication signal.

Probe that produced the numbers::

    PYTHONPATH=. python validation/ns_heat_wall_closure/probe_wall_energy_scaling.py
    -> validation/ns_heat_wall_closure/wall_energy_scaling.json

See ``docs/reports/tensordg-ns-wall-energy-scaling-defect.md``.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_PROBE = (pathlib.Path(__file__).resolve().parents[1]
          / "validation" / "ns_heat_wall_closure" / "probe_wall_energy_scaling.py")
_spec = importlib.util.spec_from_file_location("wall_energy_probe", _PROBE)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

KAPPA = probe.KAPPA
RHO_BOTTOM = probe.PRESSURE / probe.T_BOT
RHO_TOP = probe.PRESSURE / probe.T_TOP


def _wall(residual, channel=3):
    return residual[..., channel]


def test_mass_and_momentum_rows_vanish_on_the_exact_state():
    """Positive control: the same closure is exact for the other rows."""
    residual = probe.wall_residual(32)
    assert float(_wall(residual, 0).abs().max()) < 1e-12
    assert float(_wall(residual, 1).abs().max()) < 1e-12
    assert float(residual[..., 2].abs().max()) < 1e-12


def test_wall_energy_defect_is_a_cell_average_offset_only():
    """Every mode >= 1 of the wall-cell energy residual is zero."""
    residual = probe.wall_residual(32)
    for wall_index in (0, -1):
        higher = residual[..., 3][:, wall_index, 1:]
        assert float(higher.abs().max()) < 1e-10, (
            "the defect must be a face-flux offset, not a mode-shape error")
        # and the offset really is there (this is the documented defect)
        assert float(residual[..., 3][:, wall_index, 0].abs().max()) > 0.1


def test_wall_energy_defect_is_linear_in_conductivity():
    """Exact linearity in kappa: no viscous-stress / convective contribution."""
    base = probe.wall_residual(32)
    doubled = probe.wall_residual(32, kappa=2.0 * KAPPA)
    for wall_index in (0, -1):
        b = float(base[..., 3][:, wall_index, 0].abs().max())
        d = float(doubled[..., 3][:, wall_index, 0].abs().max())
        assert d == pytest.approx(2.0 * b, rel=1e-9), (
            "the wall energy residual must scale exactly with kappa")


def test_wall_energy_defect_is_pressure_independent():
    """Independent of the pressure: rules out a pressure-work boundary term."""
    base = probe.wall_residual(32)
    halved = probe.wall_residual(32, pressure=0.5 * probe.PRESSURE)
    doubled = probe.wall_residual(32, pressure=2.0 * probe.PRESSURE)
    for wall_index in (0, -1):
        b = float(base[..., 3][:, wall_index, 0].abs().max())
        for other in (halved, doubled):
            value = float(other[..., 3][:, wall_index, 0].abs().max())
            assert value == pytest.approx(b, rel=1e-12)


def test_interior_energy_residual_keeps_decaying():
    """Control: the interior interface procedure is not the broken part."""
    coarse = float(probe.wall_residual(32)[..., 3][:, 1:-1, :].abs().max())
    fine = float(probe.wall_residual(64)[..., 3][:, 1:-1, :].abs().max())
    assert fine < 0.6 * coarse, (coarse, fine)


def test_wall_energy_residual_carries_the_density_offset():
    """Ratchet (2026-11-11): the *shape* of the wall energy defect, measured.

    The residual converges to ``kappa * rho_wall`` (relative error 1.2 % at the
    bottom wall, 2.3 % at the top wall on ``ny = 128``), not to zero.  This is
    a positive pin on the defect because the law itself is verified -- a fix
    that removes the offset will make these two assertions fail and must
    re-adjudicate this pin (same convention as the ratchet flips in
    ``test_ns_heat_conduction.py``), which is exactly the signal we want.

    See ``docs/reports/tensordg-ns-wall-energy-scaling-defect.md``.
    """
    residual = probe.wall_residual(128)
    bottom = float(residual[..., 3][:, 0, 0].abs().max()) / KAPPA
    top = float(residual[..., 3][:, -1, 0].abs().max()) / KAPPA
    assert bottom == pytest.approx(RHO_BOTTOM, rel=0.03), (
        f"bottom-wall residual/kappa = {bottom:.4f} should be ~{RHO_BOTTOM}")
    assert top == pytest.approx(RHO_TOP, rel=0.03), (
        f"top-wall residual/kappa = {top:.4f} should be ~{RHO_TOP}")