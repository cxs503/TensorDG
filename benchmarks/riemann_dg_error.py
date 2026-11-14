"""Quadrature-based primitive-variable errors for high-order Euler Riemann DG.

.. warning::

   ``quadrature_order`` is **not** a free knob for a discontinuous exact
   solution.  The Lax tube's kinks (rarefaction head/tail, contact, shock) fall
   inside cells, so a fixed Gauss rule is not exact there and the measured
   relative L2 drifts by several percent with the order chosen.  At ``p3.N400``
   the density value walks between 2.91e-2 (order 8, this module's default) and
   3.35e-2 (order 5), while the *converged* value -- refined quadrature that
   splits every cell at its analytic breakpoints -- is 3.02507e-2, i.e. **above**
   the 3 % ledger bar.  The default therefore reads low; use
   ``benchmarks.lax_a0_quadrature_audit`` (aligned rule) when the number is
   compared against a tolerance.  See ``docs/CASE-LAX-DG-a0-quadrature-audit.md``.
"""
from __future__ import annotations

import numpy as np

from benchmarks.riemann_exact import sample


def riemann_dg_errors(x, coeff, final_time, left, right, diaphragm=0.5, quadrature_order=8):
    values = coeff.detach().cpu().numpy()
    centers = x.detach().cpu().numpy()
    cells, modes, _ = values.shape
    dx = (centers[-1] - centers[0]) / (cells - 1)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    phi = []
    for k in range(modes):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        phi.append(np.polynomial.legendre.legval(nodes, polynomial))
    phi = np.stack(phi, axis=1)
    numerical = np.einsum("cmv,qm->cqv", values, phi)
    primitive = {
        "density": numerical[..., 0],
        "velocity": numerical[..., 1] / numerical[..., 0],
        "pressure": 0.4 * (numerical[..., 2] - 0.5 * numerical[..., 1] ** 2 / numerical[..., 0]),
    }
    points = centers[:, None] + 0.5 * dx * nodes[None, :]
    exact = sample(points, final_time, left, right, diaphragm)
    result = {}
    for key in ("density", "velocity", "pressure"):
        error2 = 0.5 * dx * np.sum(weights[None, :] * (primitive[key] - exact[key]) ** 2)
        norm2 = 0.5 * dx * np.sum(weights[None, :] * exact[key] ** 2)
        absolute = np.sqrt(error2)
        relative = np.sqrt(error2 / norm2)
        result[key] = {
            "absolute_l2": float(absolute), "relative_l2": float(relative),
            "relative_l2_percent": float(100.0 * relative),
        }
    return result
