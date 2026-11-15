"""Wall boundary conditions for the D2Q9 solver.

Provides half-way bounce-back with an optional wall velocity (no-slip
:math:`\\mathbf{u}_w = 0`, moving lid, or Couette-type walls) on an arbitrary
static solid mask.

The scheme is the standard *pull* formulation: the population entering a
fluid node ``y`` along direction ``i`` is

* ``f_i(y, t+1) = f_i^post(y - e_i, t)``            if ``y - e_i`` is fluid,
* ``f_i(y, t+1) = f_opp(i)^post(y, t)
  + 6 w_i rho_w (e_i . u_w)``                        if ``y - e_i`` is solid.

The second line is the half-way (link-wise) bounce-back rule; the momentum
term injects the wall velocity with the correct sign and magnitude such that
the no-slip condition is satisfied half-way between the fluid node and the
solid node (Ladd 1994; Kruger et al., *The Lattice Boltzmann Method*, 2017,
ch. 8).  For ``u_w = 0`` it reduces to plain bounce-back.  The wall velocity
and wall density are read on the *solid* node, i.e. they are shifted along
the link.

Solid nodes are never written to; populations there stay at their initial
value and are only ever read through the reflection rule above.

:class:`HalfwayBounceBack` precomputes the per-direction masks and the wall
momentum source so that the per-step cost is one periodic shift plus one
``where`` per direction.
"""

from __future__ import annotations

import torch

from .lattice import D2Q9


def shift(arr, ey, ex):
    """Return ``arr`` such that ``out[y, x] = arr[y - ey, x - ex]`` (periodic)."""
    if ey == 0 and ex == 0:
        return arr
    return torch.roll(arr, shifts=(ey, ex), dims=(0, 1))


class HalfwayBounceBack:
    """Precomputed half-way bounce-back operator on a static solid mask.

    Parameters
    ----------
    solid : torch.Tensor (Ny, Nx), bool
        ``True`` marks solid (wall) nodes.
    u_wall_x, u_wall_y : torch.Tensor (Ny, Nx), optional
        Wall velocity field (read on solid nodes).  Default 0.
    rho_wall : torch.Tensor (Ny, Nx), optional
        Wall density for the momentum term.  Default 1.
    """

    def __init__(self, solid, u_wall_x=None, u_wall_y=None, rho_wall=None):
        self.solid = solid.bool()
        self.fluid = ~self.solid
        Ny, Nx = solid.shape
        dtype = torch.float64
        if u_wall_x is None:
            u_wall_x = torch.zeros((Ny, Nx), dtype=dtype)
        if u_wall_y is None:
            u_wall_y = torch.zeros((Ny, Nx), dtype=dtype)
        if rho_wall is None:
            rho_wall = torch.ones((Ny, Nx), dtype=dtype)

        self._terms = []
        solid_f = self.solid.to(dtype)
        for i in range(1, D2Q9.Q):
            ex = int(D2Q9.e[i, 0])
            ey = int(D2Q9.e[i, 1])
            up_solid = shift(solid_f, ey, ex) > 0.5
            mask = up_solid & self.fluid
            boost = 6.0 * float(D2Q9.w[i]) * shift(rho_wall, ey, ex) * (
                float(D2Q9.e[i, 0]) * shift(u_wall_x, ey, ex)
                + float(D2Q9.e[i, 1]) * shift(u_wall_y, ey, ex)
            )
            self._terms.append((i, int(D2Q9.opp[i]), ex, ey, mask, boost))

    def stream(self, f_post):
        """Stream ``f_post`` and apply bounce-back on wall links."""
        f_out = torch.empty_like(f_post)
        for i in range(D2Q9.Q):
            ex = int(D2Q9.e[i, 0])
            ey = int(D2Q9.e[i, 1])
            f_out[i] = shift(f_post[i], ey, ex)
        for i, opp, ex, ey, mask, boost in self._terms:
            if bool(mask.any()):
                f_out[i] = torch.where(mask, f_post[opp] + boost, f_out[i])
        return f_out


def stream_with_walls(
    f_post: torch.Tensor,
    solid: torch.Tensor,
    u_wall_x: torch.Tensor | None = None,
    u_wall_y: torch.Tensor | None = None,
    rho_wall: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convenience one-shot wrapper around :class:`HalfwayBounceBack`."""
    return HalfwayBounceBack(solid, u_wall_x, u_wall_y, rho_wall).stream(f_post)


def wall_adjacent_fluid(solid: torch.Tensor) -> torch.Tensor:
    """Return mask of fluid nodes that have at least one solid neighbour."""
    out = torch.zeros_like(solid)
    for i in range(1, D2Q9.Q):
        out |= shift(solid.to(solid.dtype), int(D2Q9.e[i, 1]), int(D2Q9.e[i, 0])) > 0.5
    return out & ~solid