"""Two-dimensional DG advection with divergence-free, space/time varying velocity.

Generalises :mod:`tensordg.advection_2d` (constant velocity only) to velocity
fields ``a(x, y, t)`` needed by the *ice-edge transport* reference problems:

* **solid-body rotation** -- rotating front/bump in a periodic square, exact
  solution = rigidly rotated initial field, reentrant after one revolution;
* **reentrant deformational vortex** -- LeVeque/Nair deformational flow with
  time modulation ``cos(pi t / T)`` whose exact flow map over ``[0, T]`` is the
  identity (forward-then-backward deformation), so the L2 error at ``t = T``
  measured against the initial condition is a genuine error norm.

Discretisation: modal tensor-product Legendre basis on a uniform periodic
Cartesian mesh of the unit square, weak conservative form, upwind numerical
flux chosen node-wise from the sign of the face-normal velocity, classical RK4.
The flux through each mesh face is single valued for both adjacent cells, so
the global mass ``sum_cells c_00 h_x h_y`` telescopes to zero and is conserved
to round-off on the periodic mesh, independent of the velocity field.

Normalisation follows :mod:`tensordg.advection_2d`: mode ``[0, 0]`` is the cell
average.  With ``x = x_c + (h_x/2) xi`` the semi-discrete operator is

    dc_kl/dt = (2k+1)(2l+1)/(h_x h_y) * ( int_K (a u).grad(phi_k phi_l) dV
                                          - int_dK (a.n) uh phi_k phi_l dS )

which, mapped to the reference element, yields the factors
``(2k+1)(2l+1)/(2 h_x)`` (x terms) and ``(2k+1)(2l+1)/(2 h_y)`` (y terms).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

Velocity = Callable[[torch.Tensor, torch.Tensor, float], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class Basis1D:
    """Modal Legendre basis sampled at Gauss-Legendre nodes."""

    degree: int
    nodes: torch.Tensor
    weights: torch.Tensor
    phi: torch.Tensor      # (nq, p+1)  P_k(xi_q)
    dphi: torch.Tensor     # (nq, p+1)  dP_k/dxi at xi_q
    phi_left: torch.Tensor
    phi_right: torch.Tensor

    @property
    def nq(self) -> int:
        return int(self.nodes.shape[0])


def basis_1d(degree: int, nq: int | None = None, device="cpu",
             dtype: torch.dtype = torch.float64) -> Basis1D:
    """Build the modal Legendre basis and its derivative on Gauss nodes."""
    if degree < 0:
        raise ValueError("degree must be >= 0")
    nq = max(2 * degree + 3, 4) if nq is None else int(nq)
    if nq < degree + 1:
        raise ValueError("nq must be >= degree + 1")
    dev = torch.device(device)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    phi_cols, dphi_cols = [], []
    for k in range(degree + 1):
        poly = np.zeros(k + 1)
        poly[-1] = 1.0
        phi_cols.append(np.polynomial.legendre.legval(nodes_np, poly))
        dpoly = np.polynomial.legendre.legder(poly)
        if dpoly.size == 0:
            dphi_cols.append(np.zeros_like(nodes_np))
        else:
            dphi_cols.append(np.polynomial.legendre.legval(nodes_np, dpoly))
    return Basis1D(
        degree=degree,
        nodes=torch.as_tensor(nodes_np, device=dev, dtype=dtype),
        weights=torch.as_tensor(weights_np, device=dev, dtype=dtype),
        phi=torch.as_tensor(np.stack(phi_cols, axis=1), device=dev, dtype=dtype),
        dphi=torch.as_tensor(np.stack(dphi_cols, axis=1), device=dev, dtype=dtype),
        phi_left=torch.tensor([(-1.0) ** k for k in range(degree + 1)],
                              device=dev, dtype=dtype),
        phi_right=torch.ones(degree + 1, device=dev, dtype=dtype),
    )


# --------------------------------------------------------------------------
# Reference velocity fields (divergence free, periodic unit square)
# --------------------------------------------------------------------------
def solid_body_rotation(omega: float = 2.0 * math.pi,
                        center: Sequence[float] = (0.5, 0.5)) -> Velocity:
    """Rigid rotation ``a = omega * (-(y-y0), (x-x0))``; ``omega=2pi`` -> T=1."""
    x0, y0 = float(center[0]), float(center[1])

    def velocity(x, y, t):
        del t
        return -omega * (y - y0), omega * (x - x0)

    return velocity


def reentrant_vortex(period: float = 1.0, amplitude: float = 1.0 / math.pi) -> Velocity:
    """Reentrant deformational vortex: ``psi = A sin^2(pi x) sin^2(pi y) cos(pi t/T)``.

    Gives ``u = -A pi sin^2(pi x) sin(2 pi y) cos(pi t/T)`` and
    ``v = +A pi sin(2 pi x) sin^2(pi y) cos(pi t/T)``.  The modulation is
    antisymmetric about ``T/2``, so the exact flow map over ``[0, T]`` is the
    identity and ``u(T) == u(0)``.
    """
    if period <= 0.0:
        raise ValueError("period must be > 0")

    def velocity(x, y, t):
        g = math.cos(math.pi * t / period)
        sx = torch.sin(math.pi * x)
        sy = torch.sin(math.pi * y)
        ax = -amplitude * math.pi * sx * sx * torch.sin(2.0 * math.pi * y) * g
        ay = amplitude * math.pi * torch.sin(2.0 * math.pi * x) * sy * sy * g
        return ax, ay

    return velocity


# --------------------------------------------------------------------------
# Initial / exact profiles
# --------------------------------------------------------------------------
def trig_profile(x, y, t=0.0):
    """Smooth periodic profile ``1 + sin(2 pi x) sin(2 pi y)``."""
    del t
    return 1.0 + torch.sin(2.0 * math.pi * x) * torch.sin(2.0 * math.pi * y)


def gaussian_bump(x, y, t=0.0, center: Sequence[float] = (0.5, 0.35),
                  sigma: float = 0.075, peak: float = 1.0):
    """Smooth Gaussian bump, used for peak-preservation tests."""
    del t
    x0, y0 = float(center[0]), float(center[1])
    return peak * torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))


def rotated_profile(profile: Callable[..., torch.Tensor], omega: float,
                    center: Sequence[float] = (0.5, 0.5)) -> Callable[..., torch.Tensor]:
    """Analytic exact solution under :func:`solid_body_rotation`."""
    x0, y0 = float(center[0]), float(center[1])

    def exact(x, y, t):
        theta = omega * t
        c, s = math.cos(theta), math.sin(theta)
        xs = x0 + c * (x - x0) + s * (y - y0)
        ys = y0 - s * (x - x0) + c * (y - y0)
        return profile(xs, ys, 0.0)

    return exact# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
class DGAdvection2D:
    """Periodic DG advection solver for ``u_t + div(a u) = 0``, ``a = a(x, y, t)``.

    Parameters
    ----------
    degree : polynomial degree ``p`` (modal Legendre, tensor product).
    ncells : int or ``(nx, ny)`` for the uniform periodic unit square.
    """

    def __init__(self, degree: int, ncells: int | tuple[int, int] = 16,
                 device="cpu", dtype: torch.dtype = torch.float64,
                 nq: int | None = None) -> None:
        if degree < 0 or degree > 6:
            raise ValueError("degree must be in [0, 6]")
        self.degree = int(degree)
        self.ncells = (int(ncells), int(ncells)) if isinstance(ncells, int) else (
            int(ncells[0]), int(ncells[1]))
        self.nx, self.ny = self.ncells
        if self.nx < 2 or self.ny < 2:
            raise ValueError("ncells must be >= 2 in each direction")
        self.hx, self.hy = 1.0 / self.nx, 1.0 / self.ny
        self.device = torch.device(device)
        self.dtype = dtype
        self.basis = basis_1d(self.degree, nq, self.device, dtype)

        idx = torch.arange(self.degree + 1, device=self.device, dtype=dtype)
        self._mode = 2.0 * idx + 1.0
        # (2k+1)(2l+1)/(2 h_x) and (2k+1)(2l+1)/(2 h_y)
        self._facx = torch.outer(self._mode, self._mode) / (2.0 * self.hx)
        self._facy = torch.outer(self._mode, self._mode) / (2.0 * self.hy)

        nodes = self.basis.nodes
        xc = (torch.arange(self.nx, device=self.device, dtype=dtype) + 0.5) * self.hx
        yc = (torch.arange(self.ny, device=self.device, dtype=dtype) + 0.5) * self.hy
        xq = xc[:, None] + 0.5 * self.hx * nodes[None, :]      # (nx, nq)
        yq = yc[:, None] + 0.5 * self.hy * nodes[None, :]      # (ny, nq)
        self.xq, self.yq = xq, yq
        self.Xn = xq[:, None, :, None]        # -> (nx, ny, nq, nq) when broadcast
        self.Yn = yq[None, :, None, :]
        # Right/top face of each cell: x = (i+1)hx, y = (j+1)hy.  The left face
        # flux of a cell is obtained by rolling the right-face flux (periodic).
        xf = (torch.arange(self.nx, device=self.device, dtype=dtype) + 1.0) * self.hx
        yf = (torch.arange(self.ny, device=self.device, dtype=dtype) + 1.0) * self.hy
        self.XFace, self.YFaceNodes = xf[:, None, None], yq[None, :, :]
        self.XFaceNodes, self.YFace = xq[:, None, :], yf[None, :, None]

    # -- projection / post-processing -------------------------------------
    def project(self, profile: Callable[..., torch.Tensor]) -> torch.Tensor:
        """L2 projection of a nodal profile onto the modal space."""
        b = self.basis
        vals = profile(self.Xn, self.Yn, 0.0)
        coeff = torch.einsum("xyqr,q,r,qk,rl->xykl", vals, b.weights, b.weights,
                            b.phi, b.phi)
        norm = torch.outer(self._mode, self._mode) / 4.0
        return coeff * norm[None, None]

    def nodal(self, coeff: torch.Tensor) -> torch.Tensor:
        """Evaluate the modal field at the quadrature nodes, ``(nx, ny, nq, nq)``."""
        b = self.basis
        return torch.einsum("xykl,qk,rl->xyqr", coeff, b.phi, b.phi)

    def mass(self, coeff: torch.Tensor) -> float:
        """Global integral of the solution (mode [0, 0] is the cell average)."""
        return float(coeff[:, :, 0, 0].sum().item() * self.hx * self.hy)

    def peak(self, coeff: torch.Tensor) -> tuple[float, float]:
        """Return ``(max cell average, max nodal value)``."""
        return (float(coeff[:, :, 0, 0].max().item()),
                float(self.nodal(coeff).max().item()))

    # -- spatial operator --------------------------------------------------
    def rhs(self, coeff: torch.Tensor, velocity: Velocity, t: float = 0.0) -> torch.Tensor:
        """Semi-discrete DG operator for ``u_t + div(a u) = 0``."""
        b = self.basis
        phi, dphi, w, pl, pr = b.phi, b.dphi, b.weights, b.phi_left, b.phi_right
        u = torch.einsum("xykl,qk,rl->xyqr", coeff, phi, phi)
        ax_n, ay_n = velocity(self.Xn, self.Yn, t)

        # volume terms (x then y)
        volx = torch.einsum(
            "xyql,q,qk->xykl",
            torch.einsum("xyqr,r,rl->xyql", ax_n * u, w, phi), w, dphi)
        voly = torch.einsum(
            "xykr,r,rl->xykl",
            torch.einsum("xyqr,q,qk->xykr", ay_n * u, w, phi), w, dphi)

        # x faces: traces, node-wise upwind selection, eta integration
        tr_r = torch.einsum("xykl,k->xyl", coeff, pr)
        tr_l = torch.einsum("xykl,k->xyl", coeff, pl)
        u_left = torch.einsum("xyl,rl->xyr", tr_r, phi)                    # from cell i
        u_right = torch.roll(torch.einsum("xyl,rl->xyr", tr_l, phi), -1, 0)  # from cell i+1
        ax_f = velocity(self.XFace, self.YFaceNodes, t)[0]
        flux_x = torch.where(ax_f >= 0.0, ax_f * u_left, ax_f * u_right)
        f_r = torch.einsum("xyr,r,rl->xyl", flux_x, w, phi)
        f_l = torch.roll(f_r, 1, 0)

        # y faces
        tr_t = torch.einsum("xykl,l->xyk", coeff, pr)
        tr_b = torch.einsum("xykl,l->xyk", coeff, pl)
        u_top = torch.einsum("xyk,qk->xyq", tr_t, phi)
        u_bot = torch.roll(torch.einsum("xyk,qk->xyq", tr_b, phi), -1, 1)
        ay_f = velocity(self.XFaceNodes, self.YFace, t)[1]
        flux_y = torch.where(ay_f >= 0.0, ay_f * u_top, ay_f * u_bot)
        f_t = torch.einsum("xyq,q,qk->xyk", flux_y, w, phi)
        f_b = torch.roll(f_t, 1, 1)

        dx = volx - torch.einsum("k,xyl->xykl", pr, f_r) + torch.einsum("k,xyl->xykl", pl, f_l)
        dy = voly - torch.einsum("l,xyk->xykl", pr, f_t) + torch.einsum("l,xyk->xykl", pl, f_b)
        return (self._facx[None, None] * dx) + (self._facy[None, None] * dy)

    # -- time integration --------------------------------------------------
    def solve(self, coeff: torch.Tensor, velocity: Velocity, final_time: float,
              cfl: float = 0.3, max_speed: float | None = None,
              record_mass: bool = False) -> dict:
        """Advance to ``final_time`` with RK4; returns ``(coeff, metadata)``."""
        if final_time < 0.0 or cfl <= 0.0:
            raise ValueError("final_time must be >= 0 and cfl must be > 0")
        if max_speed is None:
            max_speed = estimate_max_components(
                velocity, times=(0.0, 0.5 * final_time, final_time))
        if isinstance(max_speed, (tuple, list)):
            ax_max, ay_max = float(max_speed[0]), float(max_speed[1])
        else:
            ax_max = ay_max = float(max_speed)
        rate = (2 * self.degree + 1) * (ax_max / self.hx + ay_max / self.hy)
        steps = 0 if (final_time == 0.0 or rate == 0.0) else int(
            math.ceil(final_time * rate / cfl))
        dt = final_time / steps if steps else 0.0
        mass0 = self.mass(coeff)
        history = [mass0]
        for n in range(steps):
            t = n * dt
            k1 = self.rhs(coeff, velocity, t)
            k2 = self.rhs(coeff + 0.5 * dt * k1, velocity, t + 0.5 * dt)
            k3 = self.rhs(coeff + 0.5 * dt * k2, velocity, t + 0.5 * dt)
            k4 = self.rhs(coeff + dt * k3, velocity, t + dt)
            coeff = coeff + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if record_mass:
                history.append(self.mass(coeff))
        mass1 = self.mass(coeff)
        meta = {
            "ncells": list(self.ncells), "degree": self.degree,
            "final_time": float(final_time), "dt": float(dt), "steps": steps,
            "cfl": float(cfl), "max_speed": [ax_max, ay_max],
            "dofs": int(self.nx * self.ny * (self.degree + 1) ** 2),
            "mass_initial": mass0, "mass_final": mass1,
            "mass_drift_abs": abs(mass1 - mass0),
            "mass_drift_rel": abs(mass1 - mass0) / abs(mass0) if mass0 else 0.0,
            "mass_drift_max_rel_history": (
                max(abs(m - mass0) for m in history) / abs(mass0) if mass0 else 0.0),
        }
        return {"coeff": coeff, "meta": meta, "mass_history": history}

    # -- error norms -------------------------------------------------------
    def l2_error(self, coeff: torch.Tensor, exact: Callable[..., torch.Tensor],
                 t: float = 0.0, radius: float | None = None,
                 center: Sequence[float] = (0.5, 0.5)) -> dict:
        """L2 error against an analytic field (Gauss quadrature).

        ``radius`` restricts the norm to the disk of that radius about
        ``center``.  This is required for the solid-body rotation case: the
        rigid-rotation velocity ``(-(y-y0), x-x0)`` is *discontinuous* across the
        periodic seam, so a thin O(1) layer of error is generated there.  Since
        the characteristics of a rigid rotation are circles about the centre,
        that seam noise stays at its own radius and a disk interior to the
        bump trajectory gives a well-posed error norm.
        """
        b = basis_1d(self.degree, max(2 * self.degree + 6, 10), self.device, self.dtype)
        num = torch.einsum("xykl,qk,rl->xyqr", coeff, b.phi, b.phi)
        xq = torch.as_tensor(
            [(i + 0.5) * self.hx for i in range(self.nx)], device=self.device,
            dtype=self.dtype)[:, None] + 0.5 * self.hx * b.nodes[None, :]
        yq = torch.as_tensor(
            [(j + 0.5) * self.hy for j in range(self.ny)], device=self.device,
            dtype=self.dtype)[:, None] + 0.5 * self.hy * b.nodes[None, :]
        X = xq[:, None, :, None]
        Y = yq[None, :, None, :]
        ref = exact(X, Y, t)
        wq = b.weights[None, None, :, None] * b.weights[None, None, None, :]
        if radius is not None:
            x0, y0 = float(center[0]), float(center[1])
            mask = ((X - x0) ** 2 + (Y - y0) ** 2 <= radius ** 2).to(wq.dtype)
            wq = wq * mask
        jac = 0.25 * self.hx * self.hy
        err2 = jac * torch.sum(wq * (num - ref) ** 2)
        ref2 = jac * torch.sum(wq * ref ** 2)
        abs_l2 = float(torch.sqrt(err2).item())
        ref_l2 = float(torch.sqrt(ref2).item())
        return {"absolute_l2": abs_l2, "reference_l2": ref_l2,
                "relative_l2": abs_l2 / ref_l2 if ref_l2 else abs_l2}

    def l1_error(self, coeff: torch.Tensor, exact: Callable[..., torch.Tensor],
                 t: float = 0.0, radius: float | None = None,
                 center: Sequence[float] = (0.5, 0.5)) -> dict:
        """L1 (mass-norm) error against an analytic field, same quadrature as :meth:`l2_error`.

        Also returns ``coverage``: the fraction of the *global* reference L1 mass
        that the (optional) disk mask keeps.  It is reported so that a masked
        gate can never silently drop an unquantified part of the field -- the
        solid-body-rotation seam argument is only honest if the excluded share
        is on record.
        """
        b = basis_1d(self.degree, max(2 * self.degree + 6, 10), self.device, self.dtype)
        num = torch.einsum("xykl,qk,rl->xyqr", coeff, b.phi, b.phi)
        xq = torch.as_tensor(
            [(i + 0.5) * self.hx for i in range(self.nx)], device=self.device,
            dtype=self.dtype)[:, None] + 0.5 * self.hx * b.nodes[None, :]
        yq = torch.as_tensor(
            [(j + 0.5) * self.hy for j in range(self.ny)], device=self.device,
            dtype=self.dtype)[:, None] + 0.5 * self.hy * b.nodes[None, :]
        X = xq[:, None, :, None]
        Y = yq[None, :, None, :]
        ref = exact(X, Y, t)
        wq = b.weights[None, None, :, None] * b.weights[None, None, None, :]
        jac = 0.25 * self.hx * self.hy
        ref_l1_global = jac * torch.sum(wq * ref.abs())
        if radius is not None:
            x0, y0 = float(center[0]), float(center[1])
            mask = ((X - x0) ** 2 + (Y - y0) ** 2 <= radius ** 2).to(wq.dtype)
            wq = wq * mask
        abs_l1 = jac * torch.sum(wq * (num - ref).abs())
        ref_l1 = jac * torch.sum(wq * ref.abs())
        abs_l1_v = float(abs_l1.item())
        ref_l1_v = float(ref_l1.item())
        return {"absolute_l1": abs_l1_v, "reference_l1": ref_l1_v,
                "relative_l1": abs_l1_v / ref_l1_v if ref_l1_v else abs_l1_v,
                "coverage": (ref_l1_v / float(ref_l1_global.item())
                             if float(ref_l1_global.item()) else 0.0),
                "masked": radius is not None}

    def l2_difference(self, coeff_a: torch.Tensor, coeff_b: torch.Tensor) -> dict:
        """Exact L2 norm of the difference of two modal fields (polynomial)."""
        diff = coeff_a - coeff_b
        b = basis_1d(self.degree, max(2 * self.degree + 3, 6), self.device, self.dtype)
        num = torch.einsum("xykl,qk,rl->xyqr", diff, b.phi, b.phi)
        wq = b.weights[None, None, :, None] * b.weights[None, None, None, :]
        jac = 0.25 * self.hx * self.hy
        return {"absolute_l2": float(torch.sqrt(jac * torch.sum(wq * num ** 2)).item())}


def estimate_max_speed(velocity: Velocity, times: Sequence[float] = (0.0,),
                       n: int = 24, device="cpu",
                       dtype: torch.dtype = torch.float64) -> float:
    """Upper estimate of ``max |a|`` sampled on a coarse node grid and in time."""
    g = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    X = g[:, None]
    Y = g[None, :]
    best = 0.0
    for t in times:
        ax, ay = velocity(X, Y, float(t))
        speed = float(torch.sqrt(ax ** 2 + ay ** 2).max().item())
        best = max(best, speed)
    return max(best, 1e-14)

def estimate_max_components(velocity: Velocity, times: Sequence[float] = (0.0,),
                            n: int = 24, device="cpu",
                            dtype: torch.dtype = torch.float64) -> tuple[float, float]:
    """Directional maximum ``(max|a_x|, max|a_y|)`` sampled on a node grid / in time.

    Used for the RK4 stability bound ``dt <= cfl / ((2p+1)(|a_x|/h_x + |a_y|/h_y))``;
    taking the components separately is tighter than a single ``|a|`` bound.
    """
    g = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    X, Y = g[:, None], g[None, :]
    bx = by = 0.0
    for t in times:
        ax, ay = velocity(X, Y, float(t))
        bx = max(bx, float(ax.abs().max().item()))
        by = max(by, float(ay.abs().max().item()))
    return max(bx, 1e-14), max(by, 1e-14)
