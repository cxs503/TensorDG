"""Sparse list-based AOS LBM — cross-validation implementation.

Reimplements the storage/execution strategy used by ``lssfau/lbmpy``'s
``sparse-list-based-aos-lbm`` code generation inside TensorLBM:

1. **List-based (sparse) domain** — only *fluid* cells are simulated.  Solid
   cells are removed from the state and replaced by an explicit neighbour
   lookup table ``src[q, c]`` that maps every fluid cell ``c`` and streaming
   direction ``q`` to the list index of its upstream neighbour ``x - e_q``.
   ``src[q, c] == -1`` means "upstream cell is solid or outside the domain";
   that population is then supplied by the half-way bounce-back rule
   ``f_new[c, q] = f_post[c, opp(q)]``.

2. **AOS (array-of-structures) population layout** — per cell the ``Q``
   populations are stored contiguously, i.e. ``f`` has shape ``(n_fluid, Q)``
   with unit stride along the direction axis.  The SoA alternative
   (``(Q, n_fluid)``) is provided as ``layout="soa"`` so the two can be timed
   and cross-checked on exactly the same physics.

The pull (source-indexed) streaming form used here is the standard realisation
of list-based streaming: every target population is fetched from a precomputed
source index, so no per-step index arithmetic on the lattice is needed and the
kernel is branch-free.

Notes
-----
* Periodic boundaries wrap in the dense grid *before* the fluid-cell lookup, so
  a fluid cell adjacent to the domain edge still couples to the opposite edge
  (unless ``periodic=(False, ...)``, in which case the edge acts as a wall).
* The AOS state is bit-for-bit compatible with the dense
  ``(Q, Ny, Nx)`` state used by :mod:`tensorlbm.streaming`: rounding order of
  the moment sums differs only in the accumulation order of ``Q`` terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .lattice import D2Q9

__all__ = ["SparseDomain", "build_sparse_domain", "SparseAOSLBM"]


@dataclass
class SparseDomain:
    """A list-based (sparse) LBM domain.

    Attributes
    ----------
    shape : tuple[int, int]
        Dense ``(Ny, Nx)`` shape the domain was cut from.
    mask : torch.BoolTensor (Ny, Nx)
        ``True`` for fluid cells, ``False`` for solid cells.
    index : torch.LongTensor (n,)
        Row-major dense linear index of every fluid cell (the "list").
    coords : torch.LongTensor (n, 2)
        ``(y, x)`` coordinates of every fluid cell.
    src : torch.LongTensor (Q, n)
        Pull-streaming source table: ``src[q, c]`` is the list index of the cell
        at ``coords[c] - e_q``, or ``-1`` if that cell is not fluid.
    periodic : tuple[bool, bool]
        Periodic flags per axis.
    """

    shape: Tuple[int, int]
    mask: torch.Tensor
    index: torch.Tensor
    coords: torch.Tensor
    src: torch.Tensor
    periodic: Tuple[bool, bool]

    @property
    def n_fluid(self) -> int:
        return int(self.index.numel())

    @property
    def n_dense(self) -> int:
        ny, nx = self.shape
        return int(ny * nx)

    @property
    def fluid_fraction(self) -> float:
        return self.n_fluid / self.n_dense

    @property
    def wall_mask(self) -> torch.Tensor:
        """``(Q, n)`` boolean: ``True`` where direction ``q`` hits a wall."""
        return self.src < 0

    def gather(self, field: torch.Tensor) -> torch.Tensor:
        """Dense ``(Ny, Nx)`` or ``(Ny*Nx,)`` field -> ``(n,)`` fluid values."""
        return field.reshape(-1)[self.index]

    def scatter(self, values: torch.Tensor, fill: float = float("nan")) -> torch.Tensor:
        """``(n,)`` fluid values -> dense ``(Ny, Nx)`` field (``fill`` for solid)."""
        flat = torch.full(
            (self.n_dense,), fill, dtype=values.dtype, device=values.device
        )
        flat[self.index] = values
        ny, nx = self.shape
        return flat.reshape(ny, nx)


def build_sparse_domain(
    mask: torch.Tensor,
    periodic: Sequence[bool] = (True, True),
) -> SparseDomain:
    """Build the list-based domain (fluid-cell list + streaming source table).

    Parameters
    ----------
    mask : torch.BoolTensor (Ny, Nx)
        Fluid-cell mask (``True`` = fluid, simulated).
    periodic : sequence of 2 bools
        Periodicity per spatial axis.  Non-periodic axes are treated as walls.

    Returns
    -------
    SparseDomain

    Raises
    ------
    ValueError
        If the mask contains no fluid cell.
    """
    mask = mask.to(torch.bool)
    if mask.dim() != 2:
        raise ValueError(f"mask must be 2-D (Ny, Nx), got shape {tuple(mask.shape)}")
    ny, nx = int(mask.shape[0]), int(mask.shape[1])
    periodic = (bool(periodic[0]), bool(periodic[1]))

    flat_index = torch.arange(ny * nx, dtype=torch.long).reshape(ny, nx)
    index = flat_index[mask].contiguous()
    n = int(index.numel())
    if n == 0:
        raise ValueError("mask contains no fluid cell")

    # dense linear index -> list index ( -1 = solid / not simulated )
    lut = torch.full((ny * nx,), -1, dtype=torch.long)
    lut[index] = torch.arange(n, dtype=torch.long)

    ys = torch.div(index, nx, rounding_mode="floor")
    xs = index - ys * nx

    Q = D2Q9.Q
    src = torch.full((Q, n), -1, dtype=torch.long)
    for q in range(Q):
        # D2Q9.e columns are (e_x, e_y)
        ex = int(D2Q9.e[q, 0].item())
        ey = int(D2Q9.e[q, 1].item())
        sy = ys - ey
        sx = xs - ex

        if periodic[0]:
            sy = torch.remainder(sy, ny)
        if periodic[1]:
            sx = torch.remainder(sx, nx)
        nbr_lin = sy * nx + sx
        # after wrapping, in-range <=> neighbour is a simulated fluid cell
        valid = (sy >= 0) & (sy < ny) & (sx >= 0) & (sx < nx)
        src_q = torch.where(
            valid,
            lut[nbr_lin.clamp(0, ny * nx - 1)],
            torch.full_like(nbr_lin, -1),
        )
        src[q] = src_q

    coords = torch.stack([ys, xs], dim=1)
    return SparseDomain(
        shape=(ny, nx),
        mask=mask,
        index=index,
        coords=coords,
        src=src,
        periodic=periodic,
    )


def channel_mask(ny: int, nx: int, wall_thickness: int = 1) -> torch.Tensor:
    """Solid walls on the two ``y`` edges, periodic in ``x``."""
    mask = torch.zeros(ny, nx, dtype=torch.bool)
    mask[wall_thickness : ny - wall_thickness, :] = True
    return mask


class SparseAOSLBM:
    """List-based (sparse) BGK LBM with AOS or SoA population layout.

    Parameters
    ----------
    domain : SparseDomain
        List-based domain (see :func:`build_sparse_domain`).
    omega : float
        BGK relaxation rate ``1/tau``.
    layout : {"aos", "soa"}
        ``aos``  -> ``f`` has shape ``(n_fluid, Q)`` (lbmpy sparse AOS layout);
        ``soa``  -> ``f`` has shape ``(Q, n_fluid)``.
    dtype : torch.dtype
        Floating point dtype of the state.
    """

    def __init__(
        self,
        domain: SparseDomain,
        omega: float = 1.0,
        layout: str = "aos",
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ) -> None:
        if layout not in ("aos", "soa"):
            raise ValueError(f"layout must be 'aos' or 'soa', got {layout!r}")
        self.domain = domain
        self.omega = float(omega)
        self.layout = layout
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")

        n = domain.n_fluid
        Q = D2Q9.Q
        if layout == "aos":
            self.f = torch.zeros(n, Q, dtype=dtype, device=self.device)
        else:
            self.f = torch.zeros(Q, n, dtype=dtype, device=self.device)

        self._e = D2Q9.e.to(dtype=dtype, device=self.device)  # (Q,2)
        self._w = D2Q9.w.to(dtype=dtype, device=self.device)  # (Q,)
        self._opp = D2Q9.opp.to(self.device)                  # (Q,)
        self._src = domain.src.to(self.device)
        self._valid = (self._src >= 0)
        self._src_clamped = self._src.clamp(min=0)
        self.n_steps = 0

    # ------------------------------------------------------------------ state
    def from_dense(self, f_dense: torch.Tensor) -> None:
        """Load a dense ``(Q, Ny, Nx)`` population field into the sparse state."""
        if f_dense.dim() != 3:
            raise ValueError("f_dense must be (Q, Ny, Nx)")
        if tuple(f_dense.shape[1:]) != tuple(self.domain.shape):
            raise ValueError(
                f"f_dense spatial shape {tuple(f_dense.shape[1:])} != domain {self.domain.shape}"
            )
        flat = f_dense.reshape(D2Q9.Q, -1)[:, self.domain.index].to(
            dtype=self.dtype, device=self.device
        )
        self.f = flat.t().contiguous() if self.layout == "aos" else flat.contiguous()

    def to_dense(self, fill: float = float("nan")) -> torch.Tensor:
        """Sparse state -> dense ``(Q, Ny, Nx)`` field (``fill`` on solid cells)."""
        if self.layout == "aos":
            flat = self.f.t().contiguous()
        else:
            flat = self.f
        ny, nx = self.domain.shape
        out = torch.full(
            (D2Q9.Q, ny * nx), fill, dtype=self.dtype, device=self.device
        )
        out[:, self.domain.index] = flat
        return out.reshape(D2Q9.Q, ny, nx)

    def initialize(self, rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> None:
        """Initialise from dense macroscopic fields (shape ``(Ny, Nx)``)."""
        feq = self.equilibrium(
            self.domain.gather(rho).to(self.dtype),
            self.domain.gather(ux).to(self.dtype),
            self.domain.gather(uy).to(self.dtype),
        )
        self.f = feq

    # ------------------------------------------------------------- operators
    def macroscopic(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(rho, ux, uy)`` as ``(n,)`` fluid-cell vectors."""
        if self.layout == "aos":
            rho = self.f.sum(dim=1)
            mom = self.f @ self._e                      # (n, 2)
        else:
            rho = self.f.sum(dim=0)
            mom = self.f.t() @ self._e
        ux = mom[:, 0] / rho
        uy = mom[:, 1] / rho
        return rho, ux, uy

    def equilibrium(
        self, rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor
    ) -> torch.Tensor:
        """Second-order equilibrium in the active layout."""
        ex = self._e[:, 0]
        ey = self._e[:, 1]
        u_sq = ux * ux + uy * uy
        if self.layout == "aos":
            eu = ux[:, None] * ex[None, :] + uy[:, None] * ey[None, :]  # (n, Q)
            feq = (
                rho[:, None]
                * self._w[None, :]
                * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq[:, None])
            )
        else:
            eu = ux[None, :] * ex[:, None] + uy[None, :] * ey[:, None]  # (Q, n)
            feq = (
                rho[None, :]
                * self._w[:, None]
                * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq[None, :])
            )
        return feq

    def collide(
        self,
        f: Optional[torch.Tensor] = None,
        force: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        """BGK collision (optionally with a constant body force per unit mass)."""
        f = self.f if f is None else f
        rho, ux, uy = self._macro(f)
        feq = self.equilibrium(rho, ux, uy)
        f_post = f - self.omega * (f - feq)
        if force is not None:
            gx, gy = float(force[0]), float(force[1])
            if self.layout == "aos":
                src = 3.0 * self._w[None, :] * rho[:, None] * (
                    self._e[:, 0][None, :] * gx + self._e[:, 1][None, :] * gy
                )
            else:
                src = 3.0 * self._w[:, None] * rho[None, :] * (
                    self._e[:, 0][:, None] * gx + self._e[:, 1][:, None] * gy
                )
            f_post = f_post + src
        return f_post

    def _macro(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.layout == "aos":
            rho = f.sum(dim=1)
            mom = f @ self._e
        else:
            rho = f.sum(dim=0)
            mom = f.t() @ self._e
        return rho, mom[:, 0] / rho, mom[:, 1] / rho

    def stream(self, f_post: torch.Tensor) -> torch.Tensor:
        """List-based pull streaming with half-way bounce-back on walls."""
        Q = D2Q9.Q
        if self.layout == "aos":
            out = torch.empty_like(f_post)
            for q in range(Q):
                v = self._valid[q]
                out[:, q] = torch.where(
                    v,
                    f_post[self._src_clamped[q], q],
                    f_post[:, self._opp[q]],
                )
        else:
            out = torch.empty_like(f_post)
            for q in range(Q):
                v = self._valid[q]
                out[q] = torch.where(
                    v,
                    f_post[q].index_select(0, self._src_clamped[q]),
                    f_post[self._opp[q]],
                )
        return out

    def step(self, force: Optional[Tuple[float, float]] = None) -> None:
        """One full LBM step (collide -> stream)."""
        self.f = self.stream(self.collide(self.f, force=force))
        self.n_steps += 1

    # ----------------------------------------------------------- diagnostics
    def total_mass(self) -> torch.Tensor:
        if self.layout == "aos":
            return self.f.sum(dim=1).sum()
        return self.f.sum(dim=0).sum()

    def total_momentum(self) -> torch.Tensor:
        if self.layout == "aos":
            return (self.f @ self._e).sum(dim=0)
        return (self.f.t() @ self._e).sum(dim=0)

    @property
    def state_bytes(self) -> int:
        """Bytes of the population state alone (excludes index tables)."""
        return self.f.numel() * self.f.element_size()

    @property
    def table_bytes(self) -> int:
        """Bytes of the sparse lookup tables (``src``: ``Q*n`` int64)."""
        return self._src.numel() * self._src.element_size()