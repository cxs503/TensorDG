"""One-sided cell-centre stencil wall heat flux (m16493).

Why this module exists
----------------------
Two estimators of the isothermal-wall heat flux already ship:

* :func:`tensordg.ns_dg_2d.wall_heat_flux` reports *the flux the solver used*, so
  it inherits the wall normal-derivative estimator's design order ``p``
  (measured 1.03 at P1, 2.05 at P2, 3.06 at P3 on the benchmark case);
* :func:`tensordg.wall_flux_extrapolation.wall_heat_flux_extrapolated` reads the
  first interior *face* fluxes and Lagrange-extrapolates them onto the wall;
  that reaches order 1.99 at P1 (its leading use case) but is capped at about
  order 2 for ``p <= 2`` because an interior face flux is itself only
  ``O(h**2)``-superconvergent.

The temperature *at the cell centres* is superconvergent to a much higher order
than a face flux: measured ``O(h**2)`` at P1 and ``O(h**3.9)`` at P2/P3
(``benchmark_case -> validation/ns_heat_conduction/wall_flux_stencil.json``,
``centre_data``).  A one-sided finite-difference stencil that combines the
imposed wall temperature (``s = 0``) with the first ``points`` cell centres
(``s = (i + 1/2) h``, measured inward) therefore reaches order
``min(points + 1, centre_data_order - 1)``.  Measured on the exact conduction
state (``ny = 64``, hot wall):

======  =========  ========  =============
degree  solver     extrap.   stencil (pts)
======  =========  ========  =============
P1      1.03       1.99      1.01 (any)
P2      2.05       1.99      3.04 (1), 3.03 (2)
P3      3.06       3.95      3.04 (1), 3.03 (2)
======  =========  ========  =============

so the honest recommendation is **P1 -> face extrapolation, P2 -> this stencil
with ``points = 1`` (error 3600x below the solver flux), P3 -> the solver flux
already has its design order and the stencil only matches it**
(``verdict.D5`` records that negative result instead of overselling the ladder).
``points = 2`` is kept as the default because it is the same stencil the
operator's own opt-in ``wall_gradient="fd2"`` uses, and the two independent code
paths agree to round-off (``max |diff| <= 9e-16``, ``verdict.D4``) -- that is the
sign/normal-convention validation of this module.

Like the extrapolation estimator this is pure **post-processing**: it reads the
conserved state (cell centres) and the imposed wall temperature, never the
operator's wall flux, so it is blind to ``wall_gradient`` / ``wall_penalty`` and
the discrete solution is unchanged.

Scope / limitations (deliberate)
--------------------------------
* Only isothermal (Dirichlet-temperature) walls: the wall temperature enters the
  stencil directly.  An adiabatic or outflow face has no such value and is
  skipped (``None`` in ``boundary``).
* The stencil assumes the wall-normal temperature is smooth on the scale
  ``points * h``.  Inside a boundary layer thinner than that the estimate is not
  the wall flux -- use :func:`tensordg.ns_dg_2d.wall_heat_flux` there.
* The tangential flux component is not reconstructed; this is the normal
  conductive flux ``q_n = -kappa dT/dn`` only.
"""
from __future__ import annotations

import torch

import tensordg.ns_dg_2d as _ns

WALL_NAMES = ("xmin", "xmax", "ymin", "ymax")

# Axis metadata: (axis, wall at the -axis end, wall at the +axis end, spacing key)
_AXES = ((0, "xmin", "xmax", "dx"), (1, "ymin", "ymax", "dy"))


def stencil_weights(points):
    """Weights of the wall-normal derivative from ``points`` cell centres.

    Nodes (in units of the cell size ``h``, measured *inward* from the wall):
    ``s_0 = 0`` (the wall) and ``s_i = i - 1/2`` for ``i = 1..points`` (the cell
    centres).  The returned weights ``w`` satisfy

        ``d f / d s |_{s=0} = (1 / h) * sum_i w_i f(s_i)``

    and are exact for polynomials up to degree ``points``.  ``points = 1``
    returns ``[-2.0, 2.0]`` (two-point one-sided difference), ``points = 2``
    returns ``[-8/3, 3, -1/3]`` (parabola through the wall and two centres).
    """
    count = int(points)
    if count < 1:
        raise ValueError(f"points must be >= 1, got {points!r}")
    nodes = [0.0] + [index - 0.5 for index in range(1, count + 1)]
    weights = []
    for i, node in enumerate(nodes):
        # derivative of the node's Lagrange cardinal polynomial at s = 0
        weight = 0.0
        for j, other in enumerate(nodes):
            if j == i:
                continue
            term = 1.0
            for k, third in enumerate(nodes):
                if k in (i, j):
                    continue
                term *= (0.0 - third) / (node - third)
            weight += term / (node - other)
        weights.append(weight)
    return weights


def _wall_temperature(wall, name):
    """Imposed wall temperature of a Dirichlet wall specification."""
    if not isinstance(wall, dict) or "temperature" not in wall:
        raise ValueError(
            f"{name}: the stencil estimator needs an isothermal wall "
            f"(a dict with a 'temperature' key), got {wall!r}")
    return float(wall["temperature"])


def wall_heat_flux_stencil(coeff, dx, dy, boundary, kappa, points=2,
                           periodic=(False, False)):
    """Wall-normal conductive heat flux from a one-sided cell-centre stencil.

    Parameters
    ----------
    coeff : torch.Tensor
        Conserved DG coefficients of shape ``(nx, ny, p+1, p+1, 4)`` -- the state
        returned by :func:`tensordg.ns_dg_2d.solve_navier_stokes_dg_2d` (or any
        other route that produces the same layout).
    dx, dy : float
        Cell sizes along x and y.  Only the spacing *normal* to the wall enters
        the stencil, so an anisotropic mesh is handled exactly.
    boundary : dict
        ``{"xmin", "xmax", "ymin", "ymax"}``; an isothermal wall is
        ``{"velocity": (u, v), "temperature": T}``, ``None`` marks an
        outflow/periodic face.
    kappa : float
        Thermal conductivity (``kappa = mu c_p / Pr``).
    points : int
        Number of interior cell centres used besides the wall value
        (default 2 -> parabola, exact for quadratic temperature).
    periodic : (bool, bool)
        Periodicity of the two axes; a periodic axis is skipped.

    Returns
    -------
    dict
        ``{wall_name: torch.Tensor}`` with one value per wall *tangential* cell.
        Sign convention matches :func:`tensordg.ns_dg_2d.wall_heat_flux`: the
        physical flux component along ``+axis``, ``q_axis = -kappa dT/dx_axis``,
        so steady conduction between ``T_hot`` (``-axis`` wall) and ``T_cold``
        (``+axis`` wall) reports ``q = kappa (T_hot - T_cold) / L`` with the
        same positive sign at both walls.  Empty when no isothermal wall sits on
        a non-periodic axis.

    Raises
    ------
    ValueError
        If ``points < 1``, if an axis carrying a wall has fewer than ``points``
        cells (the stencil would leave the domain), or if a wall entry carries
        no imposed temperature.
    """
    if not boundary:
        return {}
    walls = {name: boundary.get(name) for name in WALL_NAMES}
    if all(value is None for value in walls.values()):
        return {}
    count = int(points)
    weights = stencil_weights(count)
    state = _ns._cell_centre_primitives(coeff)[2]  # temperature at the centres
    spacing = {0: float(dx), 1: float(dy)}
    flux = {}
    for axis, left_name, right_name, _ in _AXES:
        if periodic[axis]:
            continue
        wall_left, wall_right = walls[left_name], walls[right_name]
        if wall_left is None and wall_right is None:
            continue
        ncells = int(coeff.shape[axis])
        if ncells < count:
            raise ValueError(
                f"cannot build a {count}-cell one-sided stencil on an axis with "
                f"{ncells} cells; use points <= {ncells}")
        scale = 1.0 / spacing[axis]
        for name, wall, at_left in ((left_name, wall_left, True),
                                    (right_name, wall_right, False)):
            if wall is None:
                continue
            imposed = _wall_temperature(wall, name)
            centres = (state.narrow(axis, 0, count) if at_left
                       else state.narrow(axis, ncells - count, count).flip(axis))
            inward = weights[0] * imposed * scale
            for index in range(count):
                slice_ = [slice(None)] * centres.dim()
                slice_[axis] = slice(index, index + 1)
                inward = inward + weights[index + 1] * centres[tuple(slice_)].squeeze(axis) * scale
            # the stencil differentiates inward; the operator's face flux is
            # stored with the normal along +axis, so the far wall changes sign
            derivative = inward if at_left else -inward
            flux[name] = -kappa * derivative
    return flux


def wall_heat_flux_stencil_summary(coeff, dx, dy, boundary, kappa, points=2,
                                   periodic=(False, False)):
    """Tangential-cell average of :func:`wall_heat_flux_stencil` as floats."""
    flux = wall_heat_flux_stencil(coeff, dx, dy, boundary, kappa, points, periodic)
    return {name: float(value.mean()) for name, value in flux.items()}