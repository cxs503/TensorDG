"""Regression gates for the 2-D P-SV elastic DG face-term assembly (m17172).

Background
----------
``docs/CASE-DG-ELASTIC-2D-PSV.md`` registered an honest FAIL: the closed
periodic box gained elastic energy (+14.52 % at t = 0.02 -> 0.08), the
semi-discrete operator had eigenvalues with positive real part, and the
plane-wave ladder did not converge.

Two defects of the face-term (numerical flux) assembly are fixed here:

1. **Missing reference -> physical mapping factor ``2/h``.**  The volume term
   carries ``2/h``; the face term did not, so the numerical-flux correction was
   under-weighted by ``O(h)``.  This breaks consistency and skew-symmetry.
2. **Upwind bias applied to the wrong side.**  ``_rhs`` passes the interface
   states as ``(neighbour's left trace, own right trace)``; the characteristic
   jump of the Godunov flux must be taken accordingly.  The sign used by the
   original code picked the downwind characteristic, i.e. an anti-dissipative
   closure, which is what produced the energy *growth*.

The gates below are the ones the previous version of the case could not pass.
``test_pulse_cross_check_open_defect`` deliberately pins the *remaining*
honest FAIL (see the case doc) so that the defect cannot silently disappear
from the evidence trail: tighten it to ``< 3e-2`` only when it is really fixed.
"""
from __future__ import annotations

import math

import torch

from tensordg.elastic_2d import (ElasticMedium, flux_x, plane_wave,
                                 riemann_flux_x, solve_elastic_2d)

MEDIUM = ElasticMedium()          # rho = lam = mu = 1, c_p = sqrt(3)
K = 2.0 * math.pi


def _rel_l2(num, exact, weights):
    w2 = torch.outer(weights, weights).reshape(1, 1, weights.numel(),
                                               weights.numel(), 1)
    return float((((num - exact) ** 2 * w2).sum()
                  / ((exact ** 2 * w2).sum())).sqrt())


def test_plane_wave_ladder_p2_converges():
    """P wave, p=2: relative L2 must drop at least 2nd order (was EOC 0.30)."""
    exact = plane_wave("P", K, 0.0, 1.0, MEDIUM)
    errs = []
    for n in (8, 16, 32):
        res = solve_elastic_2d(lambda x, y: exact(x, y, 0.0), nx=n, ny=1,
                               degree=2, final_time=0.02, cfl=0.1)
        x, y = res.quadrature_grid()
        errs.append(_rel_l2(res.fields(), exact(x, y, 0.02),
                            res.ops["weights"]))
    eoc = [math.log(errs[i - 1] / errs[i]) / math.log(2.0)
           for i in range(1, len(errs))]
    assert errs[0] < 1e-2, errs
    assert min(eoc) > 1.8, (errs, eoc)
    assert errs[-1] < 1e-4, errs


def test_closed_periodic_box_conserves_elastic_energy():
    """Closed box energy must not grow (was +14.52 %)."""
    exact = plane_wave("P", K, 0.0, 1.0, MEDIUM)
    res = solve_elastic_2d(lambda x, y: exact(x, y, 0.0), nx=16, ny=1,
                           degree=2, final_time=0.08, cfl=0.1,
                           snapshot_times=(0.02, 0.04, 0.06))
    snaps = sorted(res.snapshots.items())
    energies = [res.energy(c).item() for _, c in snaps]
    ratios = [e / energies[0] for e in energies]
    assert max(ratios) < 1.0 + 1e-3, ratios           # no anti-dissipation
    assert min(ratios) > 1.0 - 1e-3, ratios           # and no runaway damping


def test_numerical_flux_takes_the_upwind_side():
    """Documents *why* the old plane-wave ladder was blind to the face term.

    A P plane wave satisfies ``sigma_nn = -Z_p v_n`` exactly, so both traces on
    an interface lie on the same characteristic set and the Godunov flux falls
    back on the physical flux -- the old verification never exercised the face
    term at all (a defect of the *test*).  Off that subspace the term is alive:
    a pure velocity jump must move the flux.
    """
    zp = MEDIUM.zp
    vac = torch.zeros(1, 1, 1, 5, dtype=torch.float64)
    wave = torch.zeros(1, 1, 1, 5, dtype=torch.float64)
    wave[..., 0] = 1.0
    wave[..., 2] = -zp                      # right-going P polarisation
    # one-way state on the upwind side: flux reproduced => face term inactive
    assert (riemann_flux_x(wave, vac, MEDIUM)
            - flux_x(wave, MEDIUM)).abs().max().item() < 1e-12
    # pure velocity jump: the face term must be O(1)
    jump = torch.zeros(1, 1, 1, 5, dtype=torch.float64)
    jump[..., 0] = 1.0
    g = riemann_flux_x(vac, jump, MEDIUM) - flux_x(jump, MEDIUM)
    assert g.abs().max().item() > 1e-2, g.abs().max().item()


def test_pulse_cross_check_open_defect():
    """Honest-FAIL pin: the broadband pulse is still wrong vs the 1-D library.

    TensorDG's audited 1-D DG (``tensordg.elastic_1d``, L4) solves the same
    y-invariant problem to 1.2e-2 at n=50.  The 2-D solver still returns an
    O(1) relative error, so this module remains NOT VERIFIED and must not be
    exported from ``tensordg``.  Tighten this bound to 3e-2 when it is fixed.
    """
    medium = ElasticMedium(rho=1.0, lam=0.5, mu=0.25)      # c_p = 1

    def init(x, y):
        out = torch.zeros(*x.shape, 5, dtype=torch.float64)
        out[..., 0] = torch.exp(-((x - 0.5) / 0.05) ** 2)
        return out

    res = solve_elastic_2d(init, nx=50, ny=1, degree=2, final_time=0.2,
                           cfl=0.3, medium=medium)
    x, _ = res.quadrature_grid()
    v = res.fields()[:, 0, :, 0, 0].reshape(-1)
    xq = x[:, 0, :, 0].reshape(-1)
    c = 1.0
    g_m = torch.exp(-((xq - c * 0.2 - 0.5) / 0.05) ** 2)
    g_p = torch.exp(-((xq + c * 0.2 - 0.5) / 0.05) ** 2)
    exact = 0.5 * (g_m + g_p)
    err = float((v - exact).norm() / exact.norm())
    assert err > 1.0, err                                  # still broken
    assert v.abs().max().item() < 20.0, v.abs().max().item()  # no blow-up