"""Steady flow over a bump: exact solution **and** discrete steady-state defect.

Why this module exists
----------------------
Every well-balanced (WB) verification in :mod:`tensordg` before this module was a
*rest* state: lake at rest, wet/dry shoreline rest, Thacker sloshing.  The WB
property, however, is normally claimed for the *moving* steady state too
("flux divergence balances the bed-slope source").  Up to now that claim was
untested: nothing in the repository measured what the discrete operator does
when it is applied to an exact steady solution with ``q = hu != 0`` over a
curved bed.

This module supplies the two ingredients needed for that test.

1. ``bump_steady_state`` -- the exact subcritical solution of the classical
   Goutal & Maurel (1997) / Garcia-Navarro & Vazquez-Cendon (2000) bump case::

        z_b(x) = 0.2 (1 - ((x - 10) / 2)^2)   for 8 <= x <= 12,  0 elsewhere

   with constant discharge ``q = 4.42 m^2/s`` and ``h(25) = 2 m``.  The flow is
   steady, so ``hu = q`` and the Bernoulli head is constant::

        H = z_b + h + q^2 / (2 g h^2) = const

   ``h(x)`` is the *subcritical* root of that cubic.  The solution is smooth
   (no shock) and subcritical everywhere: the margin above the critical head is
   ``min(H - z_b) - 1.5 h_c = 0.1617 m`` with ``h_c = (q^2/g)^(1/3)``.  Every
   identity used here is re-checked numerically in
   ``tests/test_swe_steady_bump.py`` (Bernoulli residual, discharge constancy,
   critical margin, and the residual of the steady momentum balance).

2. ``steady_state_defect`` -- the semi-discrete residual of the *production*
   operator :func:`tensordg.swe_1d._rhs` evaluated on the interpolant of that
   exact solution::

        R = dU_h/dt |_{U_h = interpolant(exact)}.

   For a well-balanced scheme ``||R|| = 0`` at rest to round-off.  For a moving
   steady state it is *not* zero, and the measured order of ``||R||`` versus the
   mesh size is the honest measure of how well the discrete operator represents
   the steady momentum balance.  ``R`` is also the drift rate of the solution:
   starting from the exact state, the numerical solution moves at ``R`` per unit
   time, which the validation driver checks end-to-end.

References
----------
Goutal, N. & Maurel, F. (1997) *Proceedings of the 2nd Workshop on Dam-Break
Wave Simulation*, EDF, HE-43/97/016/B.  Garcia-Navarro, P. & Vazquez-Cendon,
M.E. (2000) "On numerical treatment of the source terms in the shallow water
equations", *Computers & Fluids* 29, 951-979.
"""
from __future__ import annotations

import functools
import math
from typing import Callable, Optional

import torch

from tensordg import swe_1d

__all__ = [
    "BUMP_HEIGHT",
    "BUMP_CENTRE",
    "BUMP_HALF_WIDTH",
    "DOMAIN",
    "DISCHARGE",
    "DEPTH_OUT",
    "bump_bed",
    "bump_bed_slope",
    "critical_depth",
    "bernoulli_head",
    "bump_steady_state",
    "bump_boundary_flux",
    "BUMP_SHOCK_DISCHARGE",
    "BUMP_SHOCK_DEPTH_OUT",
    "momentum_flux",
    "shock_heads",
    "shock_position",
    "bump_transcritical_state",
    "steady_state_defect",
]

GRAVITY = 9.81
BUMP_HEIGHT = 0.2
BUMP_CENTRE = 10.0
BUMP_HALF_WIDTH = 2.0
DOMAIN = (0.0, 25.0)
DISCHARGE = 4.42          # m^2/s, Garcia-Navarro & Vazquez-Cendon (2000)
DEPTH_OUT = 2.0           # m at x = 25 (subcritical control)


def bump_bed(x, height: float = BUMP_HEIGHT, centre: float = BUMP_CENTRE,
             half_width: float = BUMP_HALF_WIDTH):
    """Bed elevation of the classical bump test case (parabola inside the bump)."""
    x = torch.as_tensor(x, dtype=torch.float64)
    inside = (x >= centre - half_width) & (x <= centre + half_width)
    z = height * (1.0 - ((x - centre) / half_width) ** 2)
    return torch.where(inside, z, torch.zeros_like(x))


def bump_bed_slope(x, height: float = BUMP_HEIGHT, centre: float = BUMP_CENTRE,
                   half_width: float = BUMP_HALF_WIDTH):
    """Analytic ``dz_b/dx`` of :func:`bump_bed` (piecewise linear, discontinuous)."""
    x = torch.as_tensor(x, dtype=torch.float64)
    inside = (x >= centre - half_width) & (x <= centre + half_width)
    dz = -2.0 * height * (x - centre) / half_width ** 2
    return torch.where(inside, dz, torch.zeros_like(x))


def critical_depth(discharge: float = DISCHARGE, gravity: float = GRAVITY) -> float:
    """Critical depth ``h_c = (q^2/g)^(1/3)``; the head there is ``1.5 h_c``."""
    return (discharge * discharge / gravity) ** (1.0 / 3.0)


def bernoulli_head(h, discharge: float = DISCHARGE, gravity: float = GRAVITY):
    """Specific-energy head ``h + q^2/(2 g h^2)`` (relative to the bed)."""
    return h + discharge * discharge / (2.0 * gravity * h * h)


def _depth_of_head(head, discharge: float, gravity: float, subcritical: bool = True) -> float:
    """Invert ``h + q^2/(2 g h^2) = head`` by bisection on the requested branch."""
    hc = (discharge * discharge / gravity) ** (1.0 / 3.0)
    if subcritical:
        lo, hi = hc * (1.0 + 1e-14), max(float(head), hc) * 4.0 + 1.0
    else:
        lo, hi = 1e-14, hc * (1.0 - 1e-14)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = mid + discharge * discharge / (2.0 * gravity * mid * mid) - head
        if (f < 0.0) == subcritical:      # f increasing on the subcritical branch
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bump_steady_state(x, discharge: float = DISCHARGE, depth_out: float = DEPTH_OUT,
                      gravity: float = GRAVITY, x_right: float = DOMAIN[1],
                      height: float = BUMP_HEIGHT, centre: float = BUMP_CENTRE,
                      half_width: float = BUMP_HALF_WIDTH) -> dict:
    """Exact subcritical steady solution of the bump case.

    Returns a dict with ``h``, ``u``, ``eta`` (tensors shaped like ``x``) plus the
    scalar diagnostics ``head``, ``critical_margin``, ``froude_max`` and
    ``regime``.  Raises ``ValueError`` when the parameters do not give a
    subcritical solution everywhere (then the flow is choked and this smooth
    solution does not exist).
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    h_out = float(depth_out)
    head = h_out + discharge * discharge / (2.0 * gravity * h_out * h_out)
    flat = x.reshape(-1)
    z = bump_bed(flat, height=height, centre=centre, half_width=half_width)
    hc = (discharge * discharge / gravity) ** (1.0 / 3.0)
    margin = float((head - z).min()) - 1.5 * hc
    if margin <= 0.0:
        raise ValueError(
            "no smooth subcritical solution: the bump chokes the flow "
            f"(head above bed minus critical head = {margin:.6g} m <= 0)")
    h = torch.empty_like(flat)
    for i, (hi, zi) in enumerate(zip((head - z).tolist(), z.tolist())):
        h[i] = _depth_of_head(hi, discharge, gravity, subcritical=True)
    h = h.reshape(x.shape)
    u = discharge / h
    froude = u / torch.sqrt(gravity * h)
    if float(froude.max()) >= 1.0:      # pragma: no cover - guarded by the margin
        raise ValueError("computed solution is not subcritical")
    return {
        "h": h,
        "u": u,
        "eta": h + bump_bed(x, height=height, centre=centre, half_width=half_width),
        "head": head,
        "critical_margin": margin,
        "froude_max": float(froude.max()),
        "regime": "subcritical",
        "discharge": discharge,
        "depth_out": h_out,
        "geometry": {"height": height, "centre": centre, "half_width": half_width},
    }


def bump_boundary_flux(side: str, **kwargs) -> Callable[[float], torch.Tensor]:
    """Constant-in-time exact boundary flux ``(hu, hu^2 + g h^2/2)`` for swe_1d."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    xb = DOMAIN[0] if side == "left" else DOMAIN[1]
    state = bump_steady_state(torch.tensor([xb], dtype=torch.float64), **kwargs)
    h = state["h"][0]
    u = state["u"][0]
    flux = torch.stack((h * u, h * u * u + 0.5 * GRAVITY * h * h))

    def boundary_flux(time: float) -> torch.Tensor:
        return flux

    boundary_flux.exact_state = state
    return boundary_flux


def steady_state_defect(h_fn, u_fn, bed_fn, ncells: int = 100, degree: int = 1,
                        domain: tuple = DOMAIN,
                        zb_slope: Optional[Callable] = None,
                        well_balanced: bool = True,
                        hydrostatic: bool = True) -> dict:
    """Semi-discrete residual of ``swe_1d`` evaluated on an exact steady state.

    ``dU/dt = R`` is assembled by the *production* operator
    (:func:`tensordg.swe_1d._rhs`), with the state discretised exactly the way
    :func:`tensordg.swe_1d.solve_swe_1d` does it:

    * ``well_balanced=True``  -- nodal (LGL) interpolation of ``h``, ``u`` and
      ``z_b``, with the Audusse hydrostatic reconstruction active and the
      bed-slope source evaluated from the *polynomial* bed (the shape this
      module's ``well_balanced`` name refers to);
    * ``well_balanced=False`` -- the legacy L2 projection of ``h`` and ``hu`` at
      the Gauss points; ``zb_slope`` is then used if supplied, else the
      piecewise-constant face-difference slope is left in place.

    Because the boundary data are the exact steady fluxes, a scheme that
    represents the steady balance exactly would return ``R = 0``.  The reported
    relative norms ``defect_h`` / ``defect_hu`` are therefore the discrete
    source-term/flux imbalance per unit time (dimensionless: ``||R||/||U||``).
    """
    lower, upper = domain
    dx = (upper - lower) / ncells
    dev, dtype = torch.device("cpu"), torch.float64
    x = lower + (torch.arange(ncells, dtype=dtype) + 0.5) * dx
    nodes, weights, phi, dphi, phi_left, phi_right = swe_1d._operators(degree, dev, dtype)
    vinv, lgl = swe_1d._lgl_transform(degree, dev, dtype)
    node_x = x[:, None] + 0.5 * dx * lgl[None, :]
    cz = swe_1d._nodal_values(bed_fn, node_x, "zb") @ vinv.T
    zb_l, zb_r = cz @ phi_left, cz @ phi_right
    zb_slope_quad = torch.einsum("cm,qm->cq", cz, dphi) * (2.0 / dx)

    coeff = torch.zeros((ncells, degree + 1, 2), dtype=dtype)
    if well_balanced:
        h_nodes = swe_1d._nodal_values(h_fn, node_x, "h")
        u_nodes = swe_1d._nodal_values(u_fn, node_x, "u")
        coeff[:, :, 0] = h_nodes @ vinv.T
        coeff[:, :, 1] = (h_nodes * u_nodes) @ vinv.T
    else:
        quad_x = x[:, None] + 0.5 * dx * nodes[None, :]
        h0 = torch.as_tensor(h_fn(quad_x), dtype=dtype)
        hu0 = h0 * torch.as_tensor(u_fn(quad_x), dtype=dtype)
        if zb_slope is not None:
            zb_slope_quad = torch.as_tensor(zb_slope(quad_x), dtype=dtype)
        else:
            zb_slope_quad = None
        for k in range(degree + 1):
            coeff[:, k, 0] = ((2 * k + 1) / 2.0) * torch.einsum(
                "cq,q,q->c", h0, weights, phi[:, k])
            coeff[:, k, 1] = ((2 * k + 1) / 2.0) * torch.einsum(
                "cq,q,q->c", hu0, weights, phi[:, k])

    residual = swe_1d._rhs(
        coeff, dx, weights, phi, dphi, phi_left, phi_right, zb_r - zb_l,
        "hll", False, 0.0, None, None, zb_slope_quad,
        cz if hydrostatic else None)

    # a lake at rest has zero momentum: guard the relative measure
    scale_h = max(float(torch.linalg.norm(coeff[:, :, 0])), 1e-12)
    scale_hu = max(float(torch.linalg.norm(coeff[:, :, 1])), 1e-12)
    return {
        "ncells": ncells,
        "degree": degree,
        "well_balanced": well_balanced,
        "hydrostatic": hydrostatic,
        "exact_slope": zb_slope_quad is not None,
        "defect_h": float(torch.linalg.norm(residual[:, :, 0])) / scale_h,
        "defect_hu": float(torch.linalg.norm(residual[:, :, 1])) / scale_hu,
        "defect_hu_linf": float(residual[:, :, 1].abs().max()),
        "dx": dx,
    }

# ===========================================================================
# Transcritical (choked) regime: smooth passage through critical at the crest
# followed by a stationary lee-side hydraulic jump.
# ===========================================================================
#
# The subcritical branch built by :func:`bump_steady_state` exists only while
# ``head - z_b >= 1.5 h_c`` everywhere.  When the bump is high enough for that
# condition to fail, the flow is *choked*: the smooth branch no longer exists
# and the standard solution of Goutal & Maurel (1997, test 3) is transcritical
# -- subcritical upstream, critical exactly at the crest, supercritical on the
# lee side, and a **stationary shock** that restores the subcritical downstream
# control.  That regime was previously unsupported (the module raised
# ``ValueError``); the helpers below supply it.
#
# Construction (all three ingredients are stated in the docstrings below and
# each one is re-checked numerically by ``tests/test_swe_steady_bump_shock.py``):
#
# 1. Energy head is conserved along the *smooth* parts of a steady solution,
#       H = z_b + h + q^2/(2 g h^2) = const,        (Bernoulli)
#    because the steady momentum balance  d/dx(q^2/h + g h^2/2) = -g h dz_b/dx
#    is exactly equivalent to dH/dx = 0 where the solution is differentiable.
# 2. The crest (maximum bed) is the section of minimum specific head, so a
#    solution that reaches it must pass through the critical depth, which pins
#    the upstream head: ``H_L = z_b(crest) + 1.5 h_c``.
# 3. Across the stationary shock both mass and momentum are conserved (there is
#    no bed-slope force over a zero-width jump, and the bed is continuous):
#       q = const,   q^2/h + g h^2/2 = const,
#    while the energy head *drops* by the (positive) shock loss.  The shock
#    location ``x_s`` is the running point where the momentum fluxes of the two
#    branches are equal; it is found by bisection of that scalar residual.
#
# Because the downstream head is fixed by the outlet control ``depth_out`` at
# ``x = x_right`` (flat bed there), ``H_R = depth_out + q^2/(2 g depth_out^2)``,
# and the shock loss is ``H_L - H_R > 0`` for the choked case.

BUMP_SHOCK_DISCHARGE = 0.18       # m^2/s  (Goutal & Maurel 1997, test 3)
BUMP_SHOCK_DEPTH_OUT = 0.33       # m      (downstream subcritical control)


def momentum_flux(h, discharge: float = BUMP_SHOCK_DISCHARGE,
                  gravity: float = GRAVITY):
    """Steady momentum flux ``q^2/h + g h^2/2`` (the quantity a jump conserves)."""
    h = torch.as_tensor(h, dtype=torch.float64)
    return discharge * discharge / h + 0.5 * gravity * h * h


def shock_heads(discharge: float = BUMP_SHOCK_DISCHARGE,
                depth_out: float = BUMP_SHOCK_DEPTH_OUT,
                gravity: float = GRAVITY,
                height: float = BUMP_HEIGHT,
                x_right: float = DOMAIN[1]) -> dict:
    """Upstream/downstream energy heads of the choked bump case.

    ``H_L = height + 1.5 h_c`` (critical passage at the crest) and
    ``H_R = depth_out + q^2/(2 g depth_out^2)`` (outlet control).  Raises
    ``ValueError`` when ``H_L <= H_R``: then no shock (and no energy loss) is
    needed -- the case is not the choked one.
    """
    hc = (discharge * discharge / gravity) ** (1.0 / 3.0)
    head_left = height + 1.5 * hc
    head_right = depth_out + discharge * discharge / (2.0 * gravity * depth_out * depth_out)
    if head_left <= head_right:
        raise ValueError(
            "not a choked transcritical case: upstream head %.6g <= downstream head %.6g"
            % (head_left, head_right))
    return {
        "head_left": head_left,
        "head_right": head_right,
        "head_loss": head_left - head_right,
        "critical_depth": hc,
        "critical_head": 1.5 * hc,
    }


def _depth_of_head_tensor(head, discharge: float, gravity: float, subcritical: bool = True,
                          iterations: int = 90):
    """Vectorised inversion of ``h + q^2/(2 g h^2) = head`` on one branch.

    Same bracket as :func:`_depth_of_head` (bisection, monotone on each branch)
    but operating on tensors, because the exact state is evaluated at every RK
    stage of the validation runs.
    """
    head = torch.as_tensor(head, dtype=torch.float64)
    hc = (discharge * discharge / gravity) ** (1.0 / 3.0)
    q2 = discharge * discharge
    if subcritical:
        lo = torch.full_like(head, hc * (1.0 + 1e-14))
        hi = torch.maximum(head, torch.full_like(head, hc)) * 4.0 + 1.0
    else:
        lo = torch.full_like(head, 1e-14)
        hi = torch.full_like(head, hc * (1.0 - 1e-14))
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f = mid + q2 / (2.0 * gravity * mid * mid) - head
        below = f < 0.0
        if subcritical:
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        else:
            hi = torch.where(below, mid, hi)
            lo = torch.where(below, lo, mid)
    return 0.5 * (lo + hi)


@functools.lru_cache(maxsize=32)
def shock_position(discharge: float = BUMP_SHOCK_DISCHARGE,
                   depth_out: float = BUMP_SHOCK_DEPTH_OUT,
                   gravity: float = GRAVITY, height: float = BUMP_HEIGHT,
                   centre: float = BUMP_CENTRE, half_width: float = BUMP_HALF_WIDTH,
                   iterations: int = 200) -> float:
    """Leeward location ``x_s`` of the stationary shock (momentum-flux balance).

    On the lee slope the left (supercritical, head ``H_L``) and right
    (subcritical, head ``H_R``) branches both exist once ``H_R - z_b >= 1.5 h_c``;
    at the first such section the two branches coincide at the critical depth and
    the residual is identically zero (a degenerate root).  The physical shock is
    the *second* root, where the residual crosses from positive to negative; it
    is bracketed by scanning and then bisected.
    """
    heads = shock_heads(discharge, depth_out, gravity, height)
    hc = heads["critical_depth"]
    head_left, head_right = heads["head_left"], heads["head_right"]
    # deepest section that still admits a subcritical root of the right family
    gap = head_right - 1.5 * hc
    if gap <= 0.0:
        x_window = centre + half_width
    else:
        frac = max(0.0, 1.0 - gap / height)
        x_window = centre + half_width * math.sqrt(frac)

    def residual(xv: float) -> float:
        z = float(bump_bed([xv], height=height, centre=centre, half_width=half_width)[0])
        h_left = _depth_of_head(head_left - z, discharge, gravity, subcritical=False)
        h_right = _depth_of_head(head_right - z, discharge, gravity, subcritical=True)
        return float(momentum_flux(h_left, discharge, gravity)
                     - momentum_flux(h_right, discharge, gravity))

    lo = x_window + 1e-9 * (half_width + 1.0)
    hi = centre + half_width
    xs_grid = [lo + (hi - lo) * k / 64.0 for k in range(65)]
    bracket = None
    for a, b in zip(xs_grid[:-1], xs_grid[1:]):
        fa, fb = residual(a), residual(b)
        if fa > 0.0 and fb <= 0.0:
            bracket = (a, b)
            break
    if bracket is None:
        raise ValueError("no stationary shock: the momentum-flux residual does not "
                         "change sign on the lee slope")
    a, b = bracket
    for _ in range(iterations):
        mid = 0.5 * (a + b)
        if residual(mid) > 0.0:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def bump_transcritical_state(x, discharge: float = BUMP_SHOCK_DISCHARGE,
                             depth_out: float = BUMP_SHOCK_DEPTH_OUT,
                             gravity: float = GRAVITY, x_right: float = DOMAIN[1],
                             height: float = BUMP_HEIGHT, centre: float = BUMP_CENTRE,
                             half_width: float = BUMP_HALF_WIDTH) -> dict:
    """Exact transcritical steady solution of the (choked) bump case.

    Returns the same keys as :func:`bump_steady_state` plus ``x_shock``,
    ``head_left`` / ``head_right`` / ``head_loss``, ``shock_depths``,
    ``froude_left`` / ``froude_right`` (Froude numbers immediately left/right of
    the jump) and ``regime="transcritical_shock"``.  The depth is piecewise:

    * ``x <= crest``: subcritical root of ``h + q^2/(2 g h^2) = H_L - z_b``;
    * ``crest < x < x_s``: supercritical root of the same equation;
    * ``x > x_s``: subcritical root of ``h + q^2/(2 g h^2) = H_R - z_b``.

    Sampling exactly at ``crest`` or ``x_s`` returns the left-hand limit; the
    right-hand limit is obtained by sampling ``crest + eps`` / ``x_s + eps``.
    The evaluation is vectorised (bisection on tensors) because the validation
    drivers call it at every RK stage.
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    heads = shock_heads(discharge, depth_out, gravity, height)
    head_left, head_right = heads["head_left"], heads["head_right"]
    x_s = shock_position(discharge, depth_out, gravity, height, centre, half_width)
    z = bump_bed(x, height=height, centre=centre, half_width=half_width)
    h_up = _depth_of_head_tensor(head_left - z, discharge, gravity, subcritical=True)
    h_lee = _depth_of_head_tensor(head_left - z, discharge, gravity, subcritical=False)
    h_down = _depth_of_head_tensor(head_right - z, discharge, gravity, subcritical=True)
    h = torch.where(x <= centre, h_up, torch.where(x <= x_s, h_lee, h_down))
    u = discharge / h
    froude = u / torch.sqrt(gravity * h)
    z_s = float(bump_bed([x_s], height=height, centre=centre, half_width=half_width)[0])
    h_left_lim = float(_depth_of_head_tensor(torch.tensor([head_left - z_s]), discharge,
                                             gravity, subcritical=False)[0])
    h_right_lim = float(_depth_of_head_tensor(torch.tensor([head_right - z_s]), discharge,
                                              gravity, subcritical=True)[0])
    return {
        "h": h,
        "u": u,
        "eta": h + z,
        "head": head_left,                     # upstream (pre-shock) energy head
        "head_left": head_left,
        "head_right": head_right,
        "head_loss": head_left - head_right,
        "x_shock": x_s,
        "shock_depths": (h_left_lim, h_right_lim),
        "shock_momentum_flux": float(momentum_flux(h_left_lim, discharge, gravity)),
        "froude_left": float(discharge / h_left_lim / math.sqrt(gravity * h_left_lim)),
        "froude_right": float(discharge / h_right_lim / math.sqrt(gravity * h_right_lim)),
        "froude_max": float(froude.max()),
        "critical_depth": heads["critical_depth"],
        "regime": "transcritical_shock",
        "discharge": discharge,
        "depth_out": depth_out,
        "geometry": {"height": height, "centre": centre, "half_width": half_width},
    }
