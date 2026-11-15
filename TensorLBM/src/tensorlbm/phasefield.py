"""Single-component two-phase (phase-field / Allen-Cahn) LBM -- the *no-MCMP* path.

Motivation
----------
The only two-phase capability in this package before this module was ``mcmp.py``
(multi-component Shan-Chen pseudopotential).  That path needs one distribution
function **per component**, its surface tension is not a model *input* (it has to
be calibrated from a Young-Laplace fit), and its cross-validation report
(``docs/mcmp_cross_force_xval_report.md``) records Laplace-pressure errors of
+100 % and a dp*R dispersion of 38 % -- i.e. the MCMP path is not quantitatively
usable for capillary physics.

``lssfau/walberla``'s "two-phase only, no MCMP" tree keeps a **phase-field**
model instead: one component, one density, an order parameter ``phi`` and an
explicit free energy whose surface tension is an analytic function of the model
parameters.  This module re-implements that approach (D2Q9, matched density,
Allen-Cahn order parameter), so that surface tension is *predicted*, not fitted.

Model
-----
Free energy (double well + gradient penalty)::

    F[phi] = int [ beta (phi^2 - 1)^2 + (kappa/2) |grad phi|^2 ] dV

    psi(phi)    = beta (phi^2 - 1)^2
    mu(phi)     = psi'(phi) - kappa lap(phi) = 4 beta (phi^3 - phi) - kappa lap(phi)

Planar equilibrium ``kappa d2phi/ds2 = psi'(phi)`` gives
``(dphi/ds)^2 = (2 beta/kappa)(1 - phi^2)^2``, i.e. the tanh profile

    phi(s) = tanh(s/ell),     ell = sqrt(kappa / (2 beta))

and the surface tension follows from the classical quadrature

    sigma = kappa * int (dphi/ds)^2 ds = (8/3) beta ell
          = (4 sqrt(2)/3) sqrt(beta kappa)                     <-- analytic

Both ``ell`` and ``sigma`` are therefore *predictions* of ``beta`` and ``kappa``;
they are not tuned.  That is the point of the no-MCMP path.

Hydrodynamics: D2Q9 BGK with a Guo body force carrying the Korteweg force

    F = mu grad(phi)     (``force_form="mu_grad_phi"``, default)
    F = phi grad(mu)     (``force_form="phi_grad_mu"``, divergence form)

The two differ by ``grad(phi mu)``, i.e. only by the convention used for the
"pressure" the LBM carries; the benchmark reports both so the size of that
convention shift is *measured* rather than assumed.

Order parameter: Allen-Cahn, advanced with second-order isotropic finite
differences on the same grid::

    dphi/dt + div(phi u) = M lap(mu)

Explicit stability of that update (dt = dx = 1) requires ``M*kappa <= 2/64``;
``stability_margin()`` reports the ratio and ``check_stability()`` raises before
a run that would blow up.

Boundaries: periodic (a droplet in a periodic box).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

CS2 = 1.0 / 3.0

# D2Q9 (same ordering as tensorlbm.lattice.D2Q9)
_W = torch.tensor(
    [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
     1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=torch.float64)
_EX = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0],
                   dtype=torch.float64)
_EY = torch.tensor([0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                   dtype=torch.float64)


def _shift(f: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    return torch.roll(f, shifts=(dy, dx), dims=(-2, -1))


def isotropic_grad(f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """8-point isotropic (Kumar 2004) central gradient, 2nd order.

    Weights 4/12 on the axis neighbours and 1/12 on the diagonals make the
    operator isotropic to leading lattice order.  This matters for a circular
    interface: with plain 2-point differences the droplet pressure carries a
    four-fold anisotropy of O(1).
    """
    # note: torch.roll with shift +1 brings index i-1 to i, hence the signs
    gx = (4.0 * (_shift(f, 0, -1) - _shift(f, 0, 1))
          + (_shift(f, 1, -1) - _shift(f, 1, 1))
          + (_shift(f, -1, -1) - _shift(f, -1, 1))) / 12.0
    gy = (4.0 * (_shift(f, -1, 0) - _shift(f, 1, 0))
          + (_shift(f, -1, -1) - _shift(f, 1, 1))
          + (_shift(f, -1, 1) - _shift(f, 1, -1))) / 12.0
    return gx, gy


def isotropic_laplacian(f: torch.Tensor) -> torch.Tensor:
    """Isotropic 9-point Laplacian (axis weight 4/6, diagonal 1/6)."""
    axis = (_shift(f, 0, 1) + _shift(f, 0, -1) + _shift(f, 1, 0) + _shift(f, -1, 0))
    diag = (_shift(f, 1, 1) + _shift(f, 1, -1) + _shift(f, -1, 1) + _shift(f, -1, -1))
    return (4.0 * axis + diag - 20.0 * f) / 6.0


def _central_div(flux_x: torch.Tensor, flux_y: torch.Tensor) -> torch.Tensor:
    """Second-order central div(flux) -- non-dissipative, hence it does not
    smear a thin interface (first-order upwinding adds |u|dx/2 of artificial
    diffusion per step, which is what dissolves a 4-cell interface in ~1e3
    steps at the spurious-current level of a static droplet)."""
    return (0.5 * (_shift(flux_x, 0, -1) - _shift(flux_x, 0, 1))
            + 0.5 * (_shift(flux_y, -1, 0) - _shift(flux_y, 1, 0)))


def _upwind_div(flux_x: torch.Tensor, flux_y: torch.Tensor,
                ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """div(flux) with first-order upwinding (stable when u != 0)."""
    d_x = torch.where(ux >= 0.0, flux_x - _shift(flux_x, 0, -1),
                      _shift(flux_x, 0, 1) - flux_x)
    d_y = torch.where(uy >= 0.0, flux_y - _shift(flux_y, -1, 0),
                      _shift(flux_y, 1, 0) - flux_y)
    return d_x + d_y


@dataclass
class PhaseFieldTwoPhase:
    """Two-phase LBM with an Allen-Cahn order parameter (no MCMP).

    Parameters
    ----------
    nx, ny : int
        Periodic box size (lattice units).
    beta : float
        Double-well coefficient, ``psi = beta (phi^2 - 1)^2``.
    kappa : float
        Gradient-penalty coefficient.
    mobility : float
        Allen-Cahn mobility ``M``.  Must satisfy ``M*kappa <= 2/64``.
    omega : float
        BGK relaxation frequency for the hydrodynamic populations.
    force_form : {"mu_grad_phi", "phi_grad_mu"}
        Korteweg force convention (see module docstring).
    """

    nx: int = 128
    ny: int = 128
    beta: float = 6.25e-3
    kappa: float = 1.125e-1
    mobility: float = 5.0e-2
    omega: float = 1.0
    force_form: str = "mu_grad_phi"
    advection: str = "central"
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    f: Optional[torch.Tensor] = None
    phi: Optional[torch.Tensor] = None
    ux: Optional[torch.Tensor] = None
    uy: Optional[torch.Tensor] = None
    rho: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.force_form not in ("mu_grad_phi", "phi_grad_mu"):
            raise ValueError(f"unknown force_form {self.force_form!r}")
        if self.advection not in ("central", "upwind"):
            raise ValueError(f"unknown advection {self.advection!r}")
        self.w = _W.to(self.dtype)
        self.ex = _EX.to(self.dtype)
        self.ey = _EY.to(self.dtype)
        z = torch.zeros(self.ny, self.nx, dtype=self.dtype, device=self.device)
        self.phi = -torch.ones_like(z)          # start in the "gas" phase phi=-1
        self.rho = torch.ones_like(z)
        self.ux = torch.zeros_like(z)
        self.uy = torch.zeros_like(z)
        self.f = self._equilibrium(self.rho, self.ux, self.uy)
        self.steps = 0

    # ------------------------------------------------------------------ model
    @property
    def interface_width(self) -> float:
        """Equilibrium tanh width ell = sqrt(kappa / (2 beta))."""
        return math.sqrt(self.kappa / (2.0 * self.beta))

    @property
    def sigma_analytic(self) -> float:
        """sigma = kappa int (dphi/ds)^2 ds = (4 sqrt(2)/3) sqrt(beta kappa)."""
        return (8.0 / 3.0) * self.beta * self.interface_width

    def stability_margin(self) -> float:
        """``mobility*kappa`` normalised by the explicit bound 2/64."""
        return self.mobility * self.kappa / (2.0 / 64.0)

    def check_stability(self):
        if self.stability_margin() >= 1.0:
            raise ValueError(
                "Allen-Cahn explicit update unstable: M*kappa="
                f"{self.mobility * self.kappa:.4g} exceeds bound 2/64="
                f"{2.0 / 64.0:.4g}")

    def chemical_potential(self, phi: Optional[torch.Tensor] = None) -> torch.Tensor:
        phi = self.phi if phi is None else phi
        return 4.0 * self.beta * (phi ** 3 - phi) - self.kappa * isotropic_laplacian(phi)

    # ------------------------------------------------------------------ fields
    def _equilibrium(self, rho: torch.Tensor, ux: torch.Tensor,
                     uy: torch.Tensor) -> torch.Tensor:
        w = self.w.view(-1, 1, 1)
        ex = self.ex.view(-1, 1, 1)
        ey = self.ey.view(-1, 1, 1)
        eu = ex * ux.unsqueeze(0) + ey * uy.unsqueeze(0)
        uu = ux * ux + uy * uy
        return w * rho.unsqueeze(0) * (
            1.0 + eu / CS2 + eu ** 2 / (2.0 * CS2 ** 2) - uu / (2.0 * CS2))

    def radial_coordinate(self) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(self.ny, dtype=self.dtype, device=self.device),
            torch.arange(self.nx, dtype=self.dtype, device=self.device),
            indexing="ij")
        # minimum image, so the periodic box behaves like free space
        dx = (xx - self.nx / 2.0)
        dy = (yy - self.ny / 2.0)
        dx = dx - self.nx * torch.round(dx / self.nx)
        dy = dy - self.ny * torch.round(dy / self.ny)
        return torch.sqrt(dx * dx + dy * dy)

    def set_phi(self, phi: torch.Tensor, rho0: float = 1.0):
        self.phi = phi.to(self.dtype).clone()
        self.rho = torch.full_like(self.phi, float(rho0))
        self.ux = torch.zeros_like(self.phi)
        self.uy = torch.zeros_like(self.phi)
        self.f = self._equilibrium(self.rho, self.ux, self.uy)
        self.steps = 0

    def setup_droplet(self, radius: float, rho0: float = 1.0,
                      phi_in: float = 1.0, phi_out: float = -1.0):
        """phi = tanh((R - r)/ell): the *exact* planar profile, so any residual
        dynamics is genuine relaxation / spurious current generation."""
        r = self.radial_coordinate()
        prof = torch.tanh((float(radius) - r) / self.interface_width)
        self.set_phi(0.5 * (phi_in + phi_out) + 0.5 * (phi_in - phi_out) * prof,
                     rho0=rho0)
        return self

    def setup_planar_interface(self, x0: Optional[float] = None,
                               orientation: str = "x", rho0: float = 1.0):
        if x0 is None:
            x0 = (self.nx / 2.0) if orientation == "x" else (self.ny / 2.0)
        if orientation == "x":
            s = (torch.arange(self.nx, dtype=self.dtype, device=self.device) - x0)
            prof = torch.tanh(s / self.interface_width).unsqueeze(0).repeat(self.ny, 1)
        else:
            s = (torch.arange(self.ny, dtype=self.dtype, device=self.device) - x0)
            prof = torch.tanh(s / self.interface_width).unsqueeze(1).repeat(1, self.nx)
        self.set_phi(prof, rho0=rho0)
        return self

    # ------------------------------------------------------------------- step
    def korteweg_force(self) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = self.phi
        if self.force_form == "mu_grad_phi":
            mu = self.chemical_potential(phi)
            gx, gy = isotropic_grad(phi)
            return mu * gx, mu * gy
        mu = self.chemical_potential(phi)
        mx, my = isotropic_grad(mu)
        return phi * mx, phi * my

    def _lbm_step(self, fx: torch.Tensor, fy: torch.Tensor):
        f, w, ex, ey = self.f, self.w, self.ex, self.ey
        rho = f.sum(dim=0)
        momx = (ex.view(-1, 1, 1) * f).sum(dim=0)
        momy = (ey.view(-1, 1, 1) * f).sum(dim=0)
        # Guo: velocity contains half the force
        ux = (momx + 0.5 * fx) / rho
        uy = (momy + 0.5 * fy) / rho
        feq = self._equilibrium(rho, ux, uy)
        e_f = ex.view(-1, 1, 1) * fx.unsqueeze(0) + ey.view(-1, 1, 1) * fy.unsqueeze(0)
        u_f = ux * fx + uy * fy
        eu = ex.view(-1, 1, 1) * ux.unsqueeze(0) + ey.view(-1, 1, 1) * uy.unsqueeze(0)
        src = (1.0 - 0.5 * self.omega) * w.view(-1, 1, 1) * (
            (e_f - u_f.unsqueeze(0)) / CS2
            + eu * e_f / (CS2 * CS2))
        f_post = f - self.omega * (f - feq) + src
        f_new = torch.empty_like(f_post)
        for i in range(9):
            f_new[i] = torch.roll(f_post[i],
                                  shifts=(int(self.ey[i].item()), int(self.ex[i].item())),
                                  dims=(0, 1))
        self.f = f_new
        self.rho = f_new.sum(dim=0)
        self.ux = ((ex.view(-1, 1, 1) * f_new).sum(dim=0)) / self.rho
        self.uy = ((ey.view(-1, 1, 1) * f_new).sum(dim=0)) / self.rho
        self.steps += 1

    def _phi_step(self, mu: Optional[torch.Tensor] = None):
        mu = self.chemical_potential(self.phi) if mu is None else mu
        diffusion = self.mobility * isotropic_laplacian(mu)
        if self.advection == "upwind":
            advection = _upwind_div(self.phi * self.ux, self.phi * self.uy,
                                    self.ux, self.uy)
        else:
            advection = _central_div(self.phi * self.ux, self.phi * self.uy)
        self.phi = self.phi + diffusion - advection

    def step(self, n: int = 1):
        for _ in range(n):
            if self.force_form == "mu_grad_phi":
                mu = self.chemical_potential(self.phi)
                gx, gy = isotropic_grad(self.phi)
                fx, fy = mu * gx, mu * gy
            else:
                mu = self.chemical_potential(self.phi)
                mx, my = isotropic_grad(mu)
                fx, fy = self.phi * mx, self.phi * my
            self._lbm_step(fx, fy)
            self._phi_step(mu)

    def run(self, n_steps: int):
        self.check_stability()
        self.step(int(n_steps))
        return self

    # -------------------------------------------------------------- diagnostics
    def pressure(self) -> torch.Tensor:
        """Pseudo-pressure carried by the LBM (``p = rho/3``)."""
        return self.rho * CS2

    def spurious_current(self) -> float:
        return float(torch.sqrt(self.ux ** 2 + self.uy ** 2).max())

    def kinetic_energy(self) -> float:
        return float(0.5 * (self.rho * (self.ux ** 2 + self.uy ** 2)).sum())

    def total_phi(self) -> float:
        return float(self.phi.sum())

    def interface_profile(self, axis: str = "x") -> Tuple[torch.Tensor, torch.Tensor]:
        """Centre-of-domain profile of phi along ``axis`` (averaged over the
        other direction) and the matching coordinate."""
        if axis == "x":
            prof = self.phi.mean(dim=0)
            coord = torch.arange(self.nx, dtype=self.dtype) - self.nx / 2.0
            coord = coord - self.nx * torch.round(coord / self.nx)
        else:
            prof = self.phi.mean(dim=1)
            coord = torch.arange(self.ny, dtype=self.dtype) - self.ny / 2.0
            coord = coord - self.ny * torch.round(coord / self.ny)
        return coord, prof

    def sigma_from_profile(self, axis: str = "x") -> float:
        """sigma = kappa * int (dphi/ds)^2 ds measured on the current field.

        Uses the trailing half of the interface (sorted by coordinate) so that
        the discrete derivative is taken on a monotone stencil.
        """
        s, prof = self.interface_profile(axis)
        order = torch.argsort(s)
        s = s[order]
        prof = prof[order]
        # keep only the monotone window around the interface; the periodic
        # wrap-around junction (phi jumps -1 -> +1 across one cell) would
        # otherwise dominate the quadrature with a fake derivative.
        win = min(6.0 * self.interface_width, 0.4 * min(self.nx, self.ny))
        keep = s.abs() <= win
        s = s[keep]
        prof = prof[keep]
        d = torch.zeros_like(prof)
        # central difference where possible
        d[1:-1] = (prof[2:] - prof[:-2]) / (s[2:] - s[:-2])
        return float(self.kappa * (d ** 2).sum() * (s[1] - s[0]))

    def pressure_drop(self, radius: float, band: Optional[float] = None) -> dict:
        """Laplace pressure drop of a droplet: p(centre) - p(far field)."""
        r = self.radial_coordinate()
        band = 1.5 * self.interface_width if band is None else band
        inside = r < (radius - band)
        outside = r > (radius + band)
        if int(inside.sum()) == 0 or int(outside.sum()) == 0:
            raise ValueError(
                f"pressure_drop(R={radius}) has an empty sampling region "
                f"(band={band}): need R > 2*band and R + 2*band < box/2")
        p = self.pressure()
        p_in = float(p[inside].mean())
        p_out = float(p[outside].mean())
        return {
            "p_in": p_in,
            "p_out": p_out,
            "dp": p_in - p_out,
            "n_inside": int(inside.sum()),
            "n_outside": int(outside.sum()),
            "dp_times_R": (p_in - p_out) * float(radius),
        }