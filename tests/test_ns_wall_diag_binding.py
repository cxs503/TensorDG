"""The wall *diagnostics* must speak the production calling convention.

Mission m3415 fixed the shipped solver (low-side wall penalty oriented along the
outward normal, ``outward_sign=-1``; fd2 stencil running inward,
``side_name="left"``) but left one item open: the diagnostic call sites were
never re-audited, so artifacts produced after the fix could still be measuring
the *retired* operator.  Two concrete drifts existed and are pinned here:

1. ``tools.diag_ns_wall.wall_fluxes`` labelled the low wall with the default
   ``side_name="right"`` and never passed ``centre=``; for the fd2/br1 modes that
   silently falls back to the one-sided face derivative, so its ``fd2`` and
   ``br1`` rows were **bit-identical to its ``face`` rows** (a mislabelled mode,
   not a measurement).
2. ``benchmarks.verify_ns_wall_gradient._row`` called the low wall without
   ``outward_sign=-1``, i.e. with the pre-m2782 anti-dissipative penalty.  Its
   published numbers (25.41 / 12.69 / 6.34 / 3.17 %) are therefore properties of
   the retired operator; with the shipped convention the same case is
   3.73 / 1.84 / 0.91 / 0.46 %.

Both defects are algebraically local, so the tests below are cheap and exact;
see ``docs/reports/tensordg-ns-wall-diag-convention.md``.
"""
import torch

import tensordg.ns_dg_2d as ns
from tools import diag_ns_wall as diag

MU = diag.MU
KAPPA = diag.KAPPA
ANALYTIC_E_BOTTOM = diag.ANALYTIC_E_FLUX_BOTTOM  # -kappa dT/dy at the cold wall


def _production_bottom_flux(ny, degree=1, mode="face", side_name="left", outward_sign=-1.0):
    """Bottom-wall energy flux with the *production* keyword set.

    Mirrors ``ns._axis_fluxes_ns``: the low wall is ``side_name="left"`` with
    ``outward_sign=-1``, fd2 passes the cell-centre primitives and retires the
    jump penalty (``WALL_PENALTY``).
    """
    d = diag.build_state((2, ny), degree)
    h = d["dy"]
    penalty = 2.0 * (degree + 1.0) ** 2
    left = ns._face_primitives(d["coeff"], d["basis"], d["derivative"], d["signs"], 1, "left")
    kwargs = {"side_name": side_name, "outward_sign": outward_sign}
    if mode == "taylor":
        recon = ns._wall_normal_derivative(d["coeff"], d["basis"], d["signs"], h, 1)
        kwargs["normal_derivative"] = recon["left"]
    if mode in ("fd2", "br1"):
        centre_u, centre_v, centre_T = ns._cell_centre_primitives(d["coeff"])
        kwargs["centre"] = {"u": centre_u, "v": centre_v, "T": centre_T}
        penalty = ns.WALL_PENALTY
    flux = ns._boundary_flux(left, d["boundary"]["ymin"], 1, h, MU, KAPPA, penalty,
                             2.0 / h, 2.0 / d["dx"], **kwargs)
    return float(flux[0, 0, 0, 3])


def test_tool_wall_fluxes_fd2_is_not_a_face_alias():
    """The shipped tool used to return the *face* numbers for fd2/br1."""
    face = diag.wall_fluxes((2, 8), 1, wall_gradient="face")["energy_bottom"]
    fd2 = diag.wall_fluxes((2, 8), 1, wall_gradient="fd2")["energy_bottom"]
    br1 = diag.wall_fluxes((2, 8), 1, wall_gradient="br1")["energy_bottom"]
    assert abs(fd2 - face) > 1e-6, (face, fd2)
    assert abs(br1 - fd2) < 1e-14, (br1, fd2)


def test_tool_wall_fluxes_low_wall_matches_the_production_convention():
    for mode in ("face", "taylor", "fd2"):
        tool = diag.wall_fluxes((2, 8), 1, wall_gradient=mode)["energy_bottom"]
        production = _production_bottom_flux(8, mode=mode)
        assert abs(tool - production) < 1e-12, (mode, tool, production)


def test_low_wall_side_label_flips_the_fd2_flux():
    """The m3415 root cause: the stencil direction is a *label*, not a mesh effect.

    With ``side_name="right"`` at the low wall the fd2 flux comes out with the
    opposite sign (202 % error vs 2.47 %); the ratio is exactly -1 on every mesh.
    """
    for ny in (8, 16, 32):
        wrong = _production_bottom_flux(ny, mode="fd2", side_name="right")
        right = _production_bottom_flux(ny, mode="fd2", side_name="left")
        assert right < 0.0 < wrong
        assert abs(wrong / right + 1.0) < 1e-12, (ny, wrong, right)


def test_fd2_heat_flux_meets_the_three_percent_gate():
    """Sanity gate on the now-correctly-wired fd2 mode."""
    for ny, gate in ((8, 0.025), (16, 0.013), (32, 0.007)):
        tool = diag.wall_fluxes((2, ny), 1, wall_gradient="fd2")["energy_bottom"]
        relative = abs(tool - ANALYTIC_E_BOTTOM) / abs(ANALYTIC_E_BOTTOM)
        assert relative < gate, (ny, relative)


def test_gradient_benchmark_reports_the_shipped_wall_sign():
    """``verify_ns_wall_gradient`` must measure the shipped operator, and the
    retired-operator numbers must stay reproducible as a negative control."""
    from benchmarks import verify_ns_wall_gradient as bench

    shipped = bench._row(8)["flux_total_error"] / abs(ANALYTIC_E_BOTTOM)
    legacy = bench._legacy_row(8)["flux_total_error"] / abs(ANALYTIC_E_BOTTOM)
    # ``_row`` carries the m2879 ``neighbour=`` prototype (extrapolated trace
    # derivative), which the production default does *not* enable, so the shipped
    # number here (8.478 %) is not the production face value (3.734 %,
    # ``test_tool_wall_fluxes_low_wall_matches_the_production_convention``).
    assert shipped < 0.09, shipped
    assert legacy > 0.30, legacy
    # the *published* 25.41 % was taken with the pre-m2782 sign **and** before the
    # m2879 ``neighbour=`` prototype existed; that exact configuration is pinned
    # here, while ``_legacy_row`` (which does use ``neighbour=``) is 37.62 %.
    retired = _production_bottom_flux(8, side_name="right", outward_sign=1.0)
    retired_relative = abs(retired - ANALYTIC_E_BOTTOM) / abs(ANALYTIC_E_BOTTOM) * 100.0
    assert abs(retired_relative - 25.40824642499248) < 1e-9, retired_relative
    assert abs(legacy * 100.0 - 37.62002889488339) < 1e-9, legacy * 100.0