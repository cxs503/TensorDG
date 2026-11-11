"""Higher-order post-processing of the isothermal-wall heat flux (m15258).

Why this module exists
---------------------
``tensordg.ns_dg_2d.wall_heat_flux`` reports *the flux the solver used*, so it
inherits the accuracy of the wall normal-derivative estimator: design order
``p`` (measured 1.03 at P1, 2.05 at P2,
``validation/ns_heat_conduction/wall_flux_order.json``).  On a P1 mesh that
leaves ~1 % error in the wall heat flux of an exactly representable conduction
state, while the *interior* faces of the very same operator are superconvergent
(measured 2.06 at both P1 and P2, because the two one-sided derivatives that are
averaged there cancel their leading error).

The wall face cannot use that cancellation, but the interior faces next to it
can be *extrapolated* to the wall plane: the normal heat-flux profile
``q(y) = -kappa dT/dn`` is smooth, so a Lagrange extrapolation of the first
``points`` interior face fluxes onto the wall returns the wall flux at the
accuracy of the interior data instead of the accuracy of the wall estimator.
That is what :func:`wall_heat_flux_extrapolated` does.  It is a **pure
post-processing** estimator: the operator, the residual and therefore the
solution are untouched.

Measured effect (``benchmarks/verify_ns_wall_flux_extrapolation.py``,
``validation/ns_heat_conduction/wall_flux_extrapolation.json``): the wall heat
flux of the exact steady conduction state goes from order 1.03 to **1.99** at
P1, and the finest-grid error from 1.05 % to **0.012 %** (``face``: 0.26 % to
0.012 %); at P2 the extrapolated value is order 2.01 with 0.007 % error, i.e.
slightly less accurate than the direct P2 flux (0.003 %) but order-matched.

Scope / limitations (deliberate, see the report)
------------------------------------------------
* Only defined for isothermal walls (Dirichlet temperature): the extrapolation
  uses the interior face fluxes of the *same* conserved state, so the wall
  temperature enters only through the state that produced them.
* The estimator assumes the normal heat-flux profile is smooth on the scale
  ``points * h``.  If the physical solution carries a boundary layer thinner
  than that, the extrapolated value is *not* the wall flux; use
  :func:`tensordg.ns_dg_2d.wall_heat_flux` (solver flux) there.
* ``points = 1`` degenerates to the nearest interior face; ``points = 2``
  (default, ``2 q(h) - q(2h)``) is the cheapest setting that reaches order 2 for
  P1 in the measured cases.  ``points = 3`` gives the same accuracy because the
  remaining error is the interior data error, not the extrapolation stencil.
"""
from __future__ import annotations

import tensordg.ns_dg_2d as _ns

WALL_NAMES = ("xmin", "xmax", "ymin", "ymax")


def extrapolation_weights(points):
    """Lagrange weights extrapolating face values at ``s = 1..points`` to ``s = 0``.

    The nodes are the first ``points`` interior faces measured in cells, so the
    weights depend only on ``points``; ``points = 2`` returns ``[2, -1]``.
    """
    count = int(points)
    if count < 1:
        raise ValueError(f"points must be >= 1, got {points!r}")
    nodes = [float(index) for index in range(1, count + 1)]
    weights = []
    for i, node in enumerate(nodes):
        weight = 1.0
        for j, other in enumerate(nodes):
            if i != j:
                weight *= (0.0 - other) / (node - other)
        weights.append(weight)
    return weights


def _face_values(projected, axis, index, norm0):
    """Physical face value of the energy component of one projected face."""
    face = projected.narrow(axis, index, 1).squeeze(axis)
    return face[..., 0, 3] / norm0


def wall_heat_flux_extrapolated(coeff, basis, derivative, signs, weights, dx, dy,
                                boundary, mu, kappa, penalty,
                                wall_gradient="face", wall_penalty=None,
                                periodic=(False, False), points=2):
    """Wall heat flux reconstructed by extrapolating interior face fluxes.

    Parameters
    ----------
    coeff, basis, derivative, signs, weights, dx, dy, mu, kappa, penalty,
    wall_gradient, wall_penalty, periodic :
        Exactly the arguments of :func:`tensordg.ns_dg_2d.wall_heat_flux`; the
        state, the discrete operators and the boundary description.
    points : int
        Number of interior faces used by the extrapolation (1, 2 or 3 are the
        meaningful settings; see the module docstring).

    Returns
    -------
    dict
        ``{wall_name: torch.Tensor}`` with one value per wall cell, same sign
        convention as :func:`tensordg.ns_dg_2d.wall_heat_flux` (the physical
        flux component along ``+axis``, hence the same sign on both walls of an
        axis).  Empty when the state has no isothermal wall on a non-periodic
        axis.

    Raises
    ------
    ValueError
        If an axis with a wall has ``ncells <= points`` (the stencil would leave
        the domain), or if the modal basis is degenerate.
    """
    if not boundary:
        return {}
    walls = {name: boundary.get(name) for name in WALL_NAMES}
    if all(value is None for value in walls.values()):
        return {}
    norm0 = float(weights @ basis[:, 0])
    if norm0 == 0.0:
        raise ValueError("degenerate modal basis: int phi_0 dxi == 0")
    weights_ext = extrapolation_weights(points)
    distance = range(len(weights_ext))
    flux = {}
    for axis, left_name, right_name in ((0, "xmin", "xmax"), (1, "ymin", "ymax")):
        if periodic[axis]:
            continue
        wall_left, wall_right = walls[left_name], walls[right_name]
        if wall_left is None and wall_right is None:
            continue
        ncells = int(coeff.shape[axis])
        if ncells <= len(weights_ext):
            raise ValueError(
                f"cannot extrapolate {len(weights_ext)} interior faces on an axis "
                f"with {ncells} cells; use points < {ncells}")
        right, left, _, _ = _ns._axis_fluxes_ns(
            coeff, basis, derivative, signs, weights, dx, dy, False,
            wall_left, wall_right, mu, kappa, penalty, axis,
            wall_gradient, wall_penalty)
        if wall_right is not None:
            # wall at the +axis end: the k-th interior face inside the domain is
            # the left face of cell (ncells - 1 - k)
            values = [_face_values(left, axis, ncells - 1 - k, norm0) for k in distance]
            flux[right_name] = sum(w * v for w, v in zip(weights_ext, values))
        if wall_left is not None:
            # wall at the -axis end: the k-th interior face is the right face of
            # cell k (both faces of one interior face hold the same flux)
            values = [_face_values(right, axis, k, norm0) for k in distance]
            flux[left_name] = sum(w * v for w, v in zip(weights_ext, values))
    return flux


def wall_heat_flux_extrapolated_summary(coeff, basis, derivative, signs, weights,
                                        dx, dy, boundary, mu, kappa, penalty,
                                        wall_gradient="face", wall_penalty=None,
                                        periodic=(False, False), points=2):
    """Tangential-cell average of :func:`wall_heat_flux_extrapolated` as floats."""
    flux = wall_heat_flux_extrapolated(
        coeff, basis, derivative, signs, weights, dx, dy, boundary, mu, kappa,
        penalty, wall_gradient, wall_penalty, periodic, points)
    return {name: float(value.mean()) for name, value in flux.items()}