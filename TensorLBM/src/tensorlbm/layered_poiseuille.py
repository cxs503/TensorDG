"""Two-component Shan-Chen *layered* Poiseuille flow with EDM forcing.

Anchor
------
``XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM`` — the two-component
Shan-Chen layered-Poiseuille practice (sailfish-team ``two-component-sc-
layered-poiseuille-edm``).  The previous anchor in this repository
(``XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE``) certified the *equilibrium*
thermodynamics of the two-component Shan-Chen (SC) path (tie-line, mechanical
pressure flatness).  It never certified a *flow*: no forcing scheme, no wall
boundary condition, and no viscosity contrast had ever been scored against an
analytic solution.  This module closes that gap.

What is new here (all three are absent from :mod:`tensorlbm.mcmp`):

1. **EDM forcing** (Exact Difference Method, Kupershtokh & Medvedev 2009):

       f_i^post = f_i - omega (f_i - f_i^eq(rho, u))
                      + [ f_i^eq(rho, u + F/rho) - f_i^eq(rho, u) ]

   The second bracket injects exactly ``F`` per unit volume per time step into
   the first moment (checked in ``tests/test_layered_poiseuille.py``), which is
   what makes the *steady* profile comparable with the Stokes solution; the
   classic "shifted velocity" forcing used by ``MCMPSolver.step`` injects
   ``omega * F`` instead and is therefore first-order in the force.

2. **Half-way bounce-back walls** on a straight channel (the existing
   ``stream_periodic`` is periodic in both directions, so it cannot represent a
   pressure-driven channel at all).

3. **Wall-aware cross-force stencil**.  ``CrossForceOperator`` wraps both
   directions with ``torch.roll``; inside a channel that leaks the pseudopo-
   tential of the lower wall row into the upper fluid row (and vice versa).  The
   local copy :func:`wall_aware_cross_force` uses a zero-gradient (wetting
   neutral) ghost value in ``y`` and keeps periodicity in ``x``.

The independent reference is the exact piecewise-parabolic Stokes solution of
two (or more) immiscible layers with different viscosities driven by a uniform
body force per unit mass, which is *derived here* (see
:func:`layered_poiseuille_analytic`) and cross-checked by a second, independent
route (:func:`layered_poiseuille_two_layer_closed_form`).

Reference: Kupershtokh A.L., Medvedev D.A., Karpov D.I. (2009),
"On equations of state in a lattice Boltzmann method", Comput. Math. Appl. 58,
965; Huang, Sukop & Lu (2015), *Lattice Boltzmann Method for Complex Flows*,
Springer, Ch. 5 (two-component layered Poiseuille example).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .lattice import D2Q9

CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Analytic reference: layered Stokes-Poiseuille solution
# ---------------------------------------------------------------------------

def viscosity_from_omega(omega: float) -> float:
    """Kinematic viscosity of BGK: nu = cs^2 (1/omega - 1/2)."""
    return CS2 * (1.0 / float(omega) - 0.5)


def omega_from_viscosity(nu: float) -> float:
    """Inverse of :func:`viscosity_from_omega`."""
    return 1.0 / (float(nu) / CS2 + 0.5)


def layered_poiseuille_analytic(
    nu: Sequence[float],
    interfaces: Sequence[float],
    H: float,
    g: float,
    y: Optional[Sequence[float]] = None,
    rho: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Exact steady velocity of N immiscible layers between two no-slip walls.

    Physics
    -------
    Fully developed, plane, incompressible flow in ``0 < y < H`` driven by a
    uniform body force per unit mass ``g`` (accelerating in ``+x``).  Each
    layer ``l`` has its own kinematic viscosity ``nu_l`` (and optionally its own
    density ``rho_l``, so that the shear-stress condition carries the dynamic
    viscosity ``mu = rho nu``).  The Stokes equations reduce to

        d/dy ( mu(y) du/dy ) = - rho(y) g ,      u(0) = u(H) = 0

    i.e. inside layer ``l``

        u_l(y) = - g y^2 / (2 nu_l) + a_l y + b_l ,

    with **continuity of velocity** and **continuity of shear stress**
    ``mu_l du_l/dy`` at every interface.  For ``L`` layers that is ``2L``
    unknowns (``a_l, b_l``) and ``2L`` equations (two walls + ``2(L-1)``
    interface conditions), solved here as a dense linear system in float64.
    Because the system is linear and well conditioned, the solution is exact to
    machine precision -- it is an *independent* reference (no LBM in it).

    Parameters
    ----------
    nu : sequence of float
        Kinematic viscosity of each layer, bottom to top.
    interfaces : sequence of float
        Interface heights, strictly increasing, all inside ``(0, H)``.
        For ``L`` layers there must be ``L - 1`` interfaces.
    H : float
        Channel height (walls at ``y = 0`` and ``y = H``).
    g : float
        Body force per unit mass (``+x`` direction driver).
    y : sequence of float, optional
        Sample locations.  Default: 400 points spanning ``(0, H)``.
    rho : sequence of float, optional
        Layer densities (mass per unit volume).  Default: all ones, in which
        case ``mu = nu`` and the stress condition is ``nu_l du/dy`` continuous.

    Returns
    -------
    dict with keys ``y``, ``u``, ``nu``, ``interfaces``, ``H``, ``g``,
    ``coeffs`` (per-layer ``(a_l, b_l)``) and ``layer_of`` (layer index of each
    sample).
    """
    nu = np.asarray(nu, dtype=np.float64)
    interfaces = np.asarray(interfaces, dtype=np.float64)
    n_layers = nu.size
    if interfaces.size != n_layers - 1:
        raise ValueError(
            f"need {n_layers - 1} interfaces for {n_layers} layers, "
            f"got {interfaces.size}"
        )
    if np.any(np.diff(interfaces) <= 0) or np.any(interfaces <= 0) or np.any(
        interfaces >= H
    ):
        raise ValueError("interfaces must be strictly increasing inside (0, H)")
    if np.any(nu <= 0):
        raise ValueError("viscosities must be positive")
    rho = np.ones(n_layers) if rho is None else np.asarray(rho, dtype=np.float64)
    if rho.size != n_layers or np.any(rho <= 0):
        raise ValueError("rho must be positive with one entry per layer")

    h = np.concatenate(([0.0], interfaces, [float(H)]))
    mu = rho * nu

    # Unknowns: a_1, b_1, a_2, b_2, ... (2 per layer)
    n_unk = 2 * n_layers
    A = np.zeros((n_unk, n_unk))
    rhs = np.zeros(n_unk)
    row = 0

    def _u_coeffs(layer: int, yv: float):
        """(coeff_a, coeff_b) such that u = a*y + b + particular(y)."""
        return yv, 1.0

    # Bottom wall: u_1(0) = 0  ->  b_1 = 0
    A[row, 2 * 0 + 1] = 1.0
    rhs[row] = 0.0
    row += 1

    # Top wall: u_L(H) = 0
    A[row, 2 * (n_layers - 1) + 0] = H
    A[row, 2 * (n_layers - 1) + 1] = 1.0
    rhs[row] = g * H * H / (2.0 * nu[-1])
    row += 1

    for l, h_l in enumerate(interfaces):
        # Velocity continuity: u_l(h) - u_{l+1}(h) = 0
        A[row, 2 * l + 0] = h_l
        A[row, 2 * l + 1] = 1.0
        A[row, 2 * (l + 1) + 0] = -h_l
        A[row, 2 * (l + 1) + 1] = -1.0
        rhs[row] = g * h_l * h_l / (2.0 * nu[l]) - g * h_l * h_l / (2.0 * nu[l + 1])
        row += 1

        # Shear-stress continuity: mu_l a_l - mu_{l+1} a_{l+1} = 0
        # (the -g y/nu * mu particular parts cancel identically: mu g y / nu = rho g y)
        A[row, 2 * l + 0] = mu[l]
        A[row, 2 * (l + 1) + 0] = -mu[l + 1]
        rhs[row] = 0.0
        row += 1

    coeffs = np.linalg.solve(A, rhs).reshape(n_layers, 2)

    if y is None:
        y_arr = np.linspace(0.0, float(H), 400)
    else:
        y_arr = np.asarray(y, dtype=np.float64)
    y_arr = np.clip(y_arr, 0.0, float(H))

    u = np.zeros_like(y_arr)
    layer_of = np.zeros_like(y_arr, dtype=int)
    for l in range(n_layers):
        mask = (y_arr >= h[l]) & (y_arr <= h[l + 1])
        # put samples exactly on an interface into the lower layer; both agree
        if l > 0:
            mask &= ~(y_arr == h[l])
        layer_of[mask] = l
        u[mask] = (
            -g * y_arr[mask] ** 2 / (2.0 * nu[l])
            + coeffs[l, 0] * y_arr[mask]
            + coeffs[l, 1]
        )

    return {
        "y": y_arr,
        "u": u,
        "nu": nu,
        "rho": rho,
        "interfaces": interfaces,
        "H": float(H),
        "g": float(g),
        "coeffs": coeffs,
        "layer_of": layer_of,
    }


def layered_poiseuille_two_layer_closed_form(
    nu1: float, nu2: float, h1: float, H: float, g: float,
    y: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Closed-form two-layer solution (independent route to the same result).

    Eliminating the interface conditions by hand (equal densities, equal body
    force per unit mass) gives

        a1 = (nu2 / nu1) a2,
        a2 = g/2 * (H^2/nu2 - h1^2/nu2 + h1^2/nu1) / (H + h1 (nu2/nu1 - 1)),
        u1 = -g y^2/(2 nu1) + a1 y,
        u2 = -g y^2/(2 nu2) + a2 y + g H^2/(2 nu2) - a2 H .

    Used only to verify :func:`layered_poiseuille_analytic` (two numerical
    routes must agree to machine precision before either scores a solver).
    """
    ratio = nu2 / nu1
    a2 = 0.5 * g * (H * H / nu2 - h1 * h1 / nu2 + h1 * h1 / nu1) / (
        H + h1 * (ratio - 1.0)
    )
    a1 = ratio * a2
    b2 = g * H * H / (2.0 * nu2) - a2 * H

    if y is None:
        y_arr = np.linspace(0.0, float(H), 400)
    else:
        y_arr = np.asarray(y, dtype=np.float64)
    u = np.where(
        y_arr <= h1,
        -g * y_arr ** 2 / (2.0 * nu1) + a1 * y_arr,
        -g * y_arr ** 2 / (2.0 * nu2) + a2 * y_arr + b2,
    )
    return {"y": y_arr, "u": u, "a1": a1, "a2": a2, "b2": b2, "H": float(H),
            "g": float(g), "h1": float(h1)}


# ---------------------------------------------------------------------------
# Local (wall-aware) cross force
# ---------------------------------------------------------------------------

def wall_aware_cross_force(
    psi: Sequence[torch.Tensor],
    G12: float,
    solid_rows: Sequence[int] = (),
) -> torch.Tensor:
    """Two-component Shan-Chen force with zero-gradient ghost rows in ``y``.

    Sign convention matches :mod:`tensorlbm.cross_force` (and the certified
    planar-interface anchor):

        F_a = +G_ab psi_a(x) * sum_i w_i psi_b(x + e_i) e_i ,   a != b

    with ``G_ab < 0`` demixing.  Neighbours in ``x`` are periodic; neighbours in
    ``y`` are clamped at the domain boundary and, when ``solid_rows`` is given,
    the ghost values in those rows are replaced by the adjacent fluid row
    (neutral wetting, zero normal gradient).  ``CrossForceOperator`` cannot do
    this -- it wraps in both directions.

    Parameters
    ----------
    psi : list of 2 tensors (ny, nx)
    G12 : float
        Off-diagonal Shan-Chen coupling (the matrix is ``[[0, G12], [G12, 0]]``).
    solid_rows : sequence of int, optional
        Row indices that are solid.  Their ``psi`` is overwritten by the nearest
        fluid row before the finite-difference stencil is applied.

    Returns
    -------
    force : torch.Tensor (2, 2, ny, nx)   (component, {fx, fy}, ny, nx)
    """
    if len(psi) != 2:
        raise ValueError("wall_aware_cross_force is the two-component stencil")
    ny, nx = psi[0].shape
    dtype, device = psi[0].dtype, psi[0].device

    solid = set(int(r) for r in solid_rows)
    psi_eff = []
    for p in psi:
        q = p.clone()
        # zero-gradient (neutral wetting) ghost rows: solid rows copy the
        # adjacent fluid row so that the stencil never sees a wall-row value
        # that is not a fluid state.
        for r in solid:
            src = r + 1 if r < ny // 2 else r - 1
            q[r, :] = q[src, :]
        psi_eff.append(q)

    e = D2Q9.e.to(device=device, dtype=dtype)
    w = D2Q9.w.to(device=device, dtype=dtype)
    force = torch.zeros(2, 2, ny, nx, dtype=dtype, device=device)

    # Pre-shift both pseudopotential fields once per direction.
    shifted = []
    for q in psi_eff:
        per_dir = []
        for i in range(D2Q9.Q):
            ex, ey = int(e[i, 0]), int(e[i, 1])
            t = torch.roll(q, shifts=ex, dims=1) if ex else q
            if ey > 0:
                t = torch.cat((t[ey:, :], t[-1:, :].expand(ey, nx)), dim=0)
            elif ey < 0:
                k = -ey
                t = torch.cat((t[:1, :].expand(k, nx), t[:-k, :]), dim=0)
            per_dir.append(t)
        shifted.append(per_dir)

    for a in range(2):
        b = 1 - a
        acc_x = torch.zeros(ny, nx, dtype=dtype, device=device)
        acc_y = torch.zeros(ny, nx, dtype=dtype, device=device)
        for i in range(D2Q9.Q):
            wt = w[i] * shifted[b][i]
            acc_x = acc_x + wt * e[i, 0]
            acc_y = acc_y + wt * e[i, 1]
        force[a, 0] = G12 * psi_eff[a] * acc_x
        force[a, 1] = G12 * psi_eff[a] * acc_y
    return force


# ---------------------------------------------------------------------------
# EDM collision
# ---------------------------------------------------------------------------

def edm_collide(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    omega: float,
) -> torch.Tensor:
    """Exact-Difference-Method forced BGK collision.

        f^post = f - omega (f - f^eq(rho, u)) + [f^eq(rho, u + F/rho) - f^eq(rho, u)]

    The second bracket adds exactly ``F`` to the first moment,
    ``sum_i (f^post_i - f_i) e_i = F``, which is the invariant that makes the
    steady state agree with the Stokes solution (asserted in the unit tests).
    """
    rho_safe = torch.clamp(rho, min=1e-12)
    dux = fx / rho_safe
    duy = fy / rho_safe
    feq_u = D2Q9.equilibrium(rho, ux, uy)
    feq_du = D2Q9.equilibrium(rho, ux + dux, uy + duy)
    return f - omega * (f - feq_u) + (feq_du - feq_u)


# ---------------------------------------------------------------------------
# Solver: two-component SC layered Poiseuille with EDM forcing
# ---------------------------------------------------------------------------

class ShanChenLayeredPoiseuille:
    """Two-component Shan-Chen flow in a straight channel with EDM forcing.

    Geometry: ``ny`` rows, ``nx`` columns; rows ``0`` and ``ny - 1`` are solid,
    rows ``1 .. ny - 2`` are fluid, ``x`` is periodic and homogeneous.  Walls are
    half-way bounce-back, i.e. the no-slip planes sit at ``y = 0.5`` and
    ``y = ny - 1.5``, so the hydrodynamic channel height is ``H = ny - 2`` and
    node ``j`` sits at ``y = j - 0.5`` (measured from the bottom wall).

    Forcing: EDM (see module docstring).  The physical velocity of component
    ``a`` reported by :meth:`macroscopic` is ``u_a + F_a / (2 rho_a)``; the
    ``1/2`` factor follows the standard forcing convention and is *verified*
    against the single-phase analytic parabola in
    ``benchmarks/layered_poiseuille_edm.py`` (arm ``L1``).

    Parameters
    ----------
    ny, nx : int
        Grid size (``ny >= 8``, ``nx >= 1``).
    omegas : float or sequence of float
        BGK relaxation rates per component (``nu = cs^2 (1/omega - 1/2)``).
    G12 : float
        Shan-Chen coupling between the two components (0 disables the
        interaction; ``G12 < 0`` demixes for ``psi = rho`` with
        ``|G12| > 1/rho_bar``).
    g_body : float or sequence of float
        Body force *per unit mass* along ``+x`` for each component.  A scalar
        applies the same acceleration to both components, which is the layered
        Poiseuille configuration: the fluids co-accelerate and the interface
        does not drift.
    psi_kind : {'linear', 'exp'}
        ``linear`` -> ``psi = rho`` (thermodynamically exact, has a tie-line);
        ``exp`` -> ``psi = 1 - exp(-rho)``.
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        omegas: Sequence[float],
        G12: float = 0.0,
        g_body: float = 0.0,
        psi_kind: str = "linear",
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        if ny < 8:
            raise ValueError("ny must be >= 8 to hold a channel + interface")
        self.ny, self.nx = int(ny), int(nx)
        self.dtype, self.device = dtype, device
        self.omega = [float(o) for o in omegas]
        self.n_components = len(self.omega)
        if self.n_components not in (1, 2):
            raise ValueError("1 (reference) or 2 (two-component) components")
        self.G12 = float(G12)
        self.psi_kind = psi_kind
        if psi_kind not in ("linear", "exp"):
            raise ValueError("psi_kind must be 'linear' or 'exp'")
        if self.n_components == 1 and self.G12 != 0.0:
            raise ValueError("a single component has no coupling")

        if np.isscalar(g_body):
            self.g_body = [float(g_body)] * self.n_components
        else:
            self.g_body = [float(v) for v in g_body]
            if len(self.g_body) != self.n_components:
                raise ValueError("g_body must be scalar or one value per component")

        self.solid_rows = (0, self.ny - 1)
        self.f: List[torch.Tensor] = []
        self.rho: List[torch.Tensor] = []
        self.ux: List[torch.Tensor] = []
        self.uy: List[torch.Tensor] = []
        self.step_count = 0
        self._last_force: Optional[torch.Tensor] = None
        self._last_du: Optional[List[torch.Tensor]] = None

        zeros = torch.zeros(self.ny, self.nx, dtype=dtype, device=device)
        ones = torch.ones(self.ny, self.nx, dtype=dtype, device=device)
        for _ in range(self.n_components):
            self.rho.append(ones.clone())
            self.ux.append(zeros.clone())
            self.uy.append(zeros.clone())
            f0 = D2Q9.equilibrium(ones, zeros, zeros).to(dtype=dtype)
            for r in self.solid_rows:      # solid nodes carry no populations
                f0[:, r, :] = 0.0
            self.f.append(f0)

    # -- helpers ---------------------------------------------------------

    def _psi(self, rho: torch.Tensor) -> torch.Tensor:
        if self.psi_kind == "linear":
            return rho
        return 1.0 - torch.exp(-rho)

    @property
    def channel_height(self) -> int:
        """Hydrodynamic channel height ``H = ny - 2`` (half-way walls)."""
        return self.ny - 2

    def node_y(self) -> np.ndarray:
        """Distance of each fluid node from the bottom wall (``j - 0.5``)."""
        j = np.arange(1, self.ny - 1, dtype=np.float64)
        return j - 0.5

    # -- initialisation --------------------------------------------------

    def set_fields(
        self,
        rho_fields: Sequence[torch.Tensor],
        ux: Optional[Sequence[torch.Tensor]] = None,
    ) -> None:
        """Set densities (and optionally ``ux``) and re-equilibrate."""
        if len(rho_fields) != self.n_components:
            raise ValueError("one density field per component")
        for a, rf in enumerate(rho_fields):
            assert tuple(rf.shape) == (self.ny, self.nx)
            self.rho[a] = rf.to(dtype=self.dtype, device=self.device).clone()
            if ux is None:
                self.ux[a] = torch.zeros_like(self.rho[a])
            else:
                self.ux[a] = ux[a].to(dtype=self.dtype, device=self.device).clone()
            self.uy[a] = torch.zeros_like(self.rho[a])
            self.f[a] = D2Q9.equilibrium(self.rho[a], self.ux[a], self.uy[a]).to(
                dtype=self.dtype
            )
            # solid rows carry no populations
            for r in self.solid_rows:
                self.f[a][:, r, :] = 0.0

    def set_layered_profile(
        self,
        rho_hi: float,
        rho_lo: float,
        interface_y: float,
        width: float = 2.0,
        rho_bar: Optional[float] = None,
        pressure_matched: bool = True,
        ux_analytic: Optional[np.ndarray] = None,
    ) -> None:
        """Component 0 rich below the interface, component 1 rich above it.

        With ``pressure_matched`` the total density is depressed at the
        interface so that the mixture mechanical pressure
        ``p = cs^2 (rho_1 + rho_2 - G12 psi_1 psi_2)`` is flat to leading order;
        the depression is obtained from the same constant-pressure condition
        used by the certified planar-interface reference (no fitted constant).

        ``interface_y`` is measured from the bottom wall, i.e. in the
        ``node_y`` coordinate (a node at row ``j`` sits at ``y = j - 0.5``).
        """
        y_nodes = torch.arange(self.ny, dtype=self.dtype).view(-1, 1) - 0.5
        width = float(width)
        amp = 0.5 * (rho_hi - rho_lo)
        mid = 0.5 * (rho_hi + rho_lo)
        s = torch.tanh((float(interface_y) - y_nodes) / width).expand(
            self.ny, self.nx
        ).clone()

        if pressure_matched:
            p_bulk = rho_hi + rho_lo - self.G12 * self._psi_scalar(rho_hi) * (
                self._psi_scalar(rho_lo)
            )
            # at the interface the two component densities are equal, x
            #   A x^2 + 2 x - p_bulk = 0,  A = -G12 * (psi/x)^2 -> linear psi: A = -G12
            x_mid = self._interface_density(p_bulk)
            dip = max(0.5 * (rho_hi + rho_lo) - x_mid, 0.0)
            d = dip * (1.0 - s * s)  # sech^2-like, vanishes in the bulk
        else:
            d = torch.zeros(self.ny, self.nx, dtype=self.dtype)

        r0 = mid + amp * s - d
        r1 = mid - amp * s - d
        ux = None
        if ux_analytic is not None:
            prof = torch.as_tensor(ux_analytic, dtype=self.dtype).view(-1, 1)
            ux = [prof.expand(self.ny, self.nx).clone() for _ in range(self.n_components)]
        self.set_fields([r0, r1], ux=ux)

    def _psi_scalar(self, rho: float) -> float:
        return rho if self.psi_kind == "linear" else 1.0 - np.exp(-rho)

    def _interface_density(self, p_bulk: float) -> float:
        """Equal component density at a flat interface giving ``p = p_bulk``."""
        if self.psi_kind == "linear":
            A = -self.G12  # p/cs2 = 2x - G x^2  ->  A x^2 + 2x - p = 0, A = -G
            if abs(A) < 1e-14:
                return 0.5 * p_bulk
            disc = 1.0 + A * p_bulk
            if disc < 0:
                return 0.5 * p_bulk
            return (np.sqrt(disc) - 1.0) / A
        # exp pseudopotential: solve numerically
        from scipy.optimize import brentq  # local import keeps base deps light

        def f(x: float) -> float:
            return 2.0 * x - self.G12 * self._psi_scalar(x) ** 2 - p_bulk

        return float(brentq(f, 1e-8, 10.0))

    # -- integration -----------------------------------------------------

    def _macroscopic(self) -> None:
        for a in range(self.n_components):
            r, ux, uy = D2Q9.macroscopic(self.f[a])
            self.rho[a], self.ux[a], self.uy[a] = r, ux, uy

    def _compute_force(self) -> torch.Tensor:
        """Total force per component: SC interaction + body force."""
        ny, nx = self.ny, self.nx
        F = torch.zeros(self.n_components, 2, ny, nx,
                        dtype=self.dtype, device=self.device)
        if self.n_components == 2 and self.G12 != 0.0:
            F = wall_aware_cross_force(
                [self._psi(self.rho[0]), self._psi(self.rho[1])],
                self.G12,
                solid_rows=self.solid_rows,
            )
        for a in range(self.n_components):
            F[a, 0] = F[a, 0] + self.rho[a] * self.g_body[a]
        # no force may act inside the solid
        for r in self.solid_rows:
            F[:, :, r, :] = 0.0
        return F

    def step(self) -> None:
        """One EDM-forced BGK step with half-way bounce-back walls."""
        self._macroscopic()
        F = self._compute_force()
        self._last_force = F

        rho_safe = [torch.clamp(self.rho[a], min=1e-12) for a in range(self.n_components)]
        du = [F[a] / rho_safe[a] for a in range(self.n_components)]
        self._last_du = du

        f_post = []
        for a in range(self.n_components):
            f = edm_collide(
                self.f[a], self.rho[a], self.ux[a], self.uy[a],
                F[a, 0], F[a, 1], self.omega[a],
            )
            for r in self.solid_rows:
                f[:, r, :] = 0.0
            f_post.append(f)

        for a in range(self.n_components):
            self.f[a] = self._stream_bounce_back(f_post[a])
        self.step_count += 1

    def _stream_bounce_back(self, f_post: torch.Tensor) -> torch.Tensor:
        """Stream (periodic in x) + half-way bounce-back at both walls."""
        Q, ny, nx = f_post.shape
        e = D2Q9.e.to(dtype=self.dtype)
        out = torch.zeros_like(f_post)
        for i in range(Q):
            ex, ey = int(e[i, 0]), int(e[i, 1])
            src = torch.roll(f_post[i], shifts=ex, dims=1) if ex else f_post[i]
            if ey == 0:
                out[i] = src
            elif ey > 0:
                out[i, ey:, :] = src[:-ey, :]
            else:
                k = -ey
                out[i, :-k, :] = src[k:, :]

        # Half-way bounce-back: the population that would have come *out of* a
        # solid row is replaced by the counter-moving post-collision population
        # of the adjacent fluid node (wall plane at n +/- 1/2).
        opp = D2Q9.opp
        for i in range(Q):
            ey = int(e[i, 1])
            if ey > 0:      # arriving at row 1 from the bottom solid row 0
                out[i, 1, :] = f_post[int(opp[i]), 1, :]
            elif ey < 0:    # arriving at row ny-2 from the top solid row ny-1
                out[i, ny - 2, :] = f_post[int(opp[i]), ny - 2, :]
        for r in self.solid_rows:
            out[:, r, :] = 0.0
        return out

    def run(self, n_steps: int, monitor: bool = False) -> Optional[np.ndarray]:
        """Advance ``n_steps``; optionally return the ``|du/dt|`` history.

        The monitored quantity is the ``L2`` change of the mass-averaged stream
        velocity profile between consecutive steps, i.e. a direct
        steady-state residual.
        """
        hist = [] if monitor else None
        prev = None
        for _ in range(n_steps):
            self.step()
            if monitor:
                cur = self.profile_u_mix()
                hist.append(
                    float(np.linalg.norm(cur - prev)) if prev is not None else np.nan
                )
                prev = cur
        return None if hist is None else np.asarray(hist)

    # -- measurement -----------------------------------------------------

    def macroscopic_fields(self) -> Dict[str, List[torch.Tensor]]:
        """Per-component ``rho`` and *physical* velocity (``u + F/2rho``)."""
        self._macroscopic()
        ux, uy = [], []
        for a in range(self.n_components):
            if self._last_du is not None:
                ux.append(self.ux[a] + 0.5 * self._last_du[a][0])
                uy.append(self.uy[a] + 0.5 * self._last_du[a][1])
            else:
                ux.append(self.ux[a].clone())
                uy.append(self.uy[a].clone())
        return {"rho": self.rho, "ux": ux, "uy": uy}

    def profile_u_mix(self) -> np.ndarray:
        """Mass-averaged stream velocity profile, one value per fluid row."""
        mf = self.macroscopic_fields()
        num = torch.zeros(self.ny, self.nx, dtype=self.dtype)
        den = torch.zeros(self.ny, self.nx, dtype=self.dtype)
        for a in range(self.n_components):
            num = num + mf["rho"][a] * mf["ux"][a]
            den = den + mf["rho"][a]
        u = (num / torch.clamp(den, min=1e-30)).mean(dim=1).cpu().numpy()
        return u[1:self.ny - 1]

    def profiles(self) -> Dict[str, np.ndarray]:
        """x-averaged profiles: ``y``, ``u_mix``, per-component rho/ux/uy."""
        mf = self.macroscopic_fields()
        out: Dict[str, np.ndarray] = {"y": self.node_y()}
        num = torch.zeros(self.ny, self.nx, dtype=self.dtype)
        den = torch.zeros(self.ny, self.nx, dtype=self.dtype)
        for a in range(self.n_components):
            num = num + mf["rho"][a] * mf["ux"][a]
            den = den + mf["rho"][a]
        out["u_mix"] = (num / torch.clamp(den, min=1e-30)).mean(dim=1).cpu().numpy()[1:-1]
        for a in range(self.n_components):
            out[f"rho{a}"] = mf["rho"][a].mean(dim=1).cpu().numpy()[1:-1]
            out[f"ux{a}"] = mf["ux"][a].mean(dim=1).cpu().numpy()[1:-1]
            out[f"uy{a}"] = mf["uy"][a].mean(dim=1).cpu().numpy()[1:-1]
        out["rho_tot"] = sum(out[f"rho{a}"] for a in range(self.n_components))
        return out

    def total_mass(self) -> List[float]:
        """Fluid mass per component (solid rows are excluded)."""
        return [float(f[:, 1:-1, :].sum().item()) for f in self.f]

    def max_spurious_velocity(self) -> float:
        """max |u_y| over fluid nodes (transverse spurious current)."""
        mf = self.macroscopic_fields()
        vmax = 0.0
        for a in range(self.n_components):
            vmax = max(vmax, float(mf["uy"][a][1:-1].abs().max().item()))
        return vmax


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def profile_error(
    y: np.ndarray,
    u_num: np.ndarray,
    u_ref_fn: Callable[[np.ndarray], np.ndarray],
    exclude: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, float]:
    """Relative error of a measured profile against a reference function.

    ``exclude`` is a list of ``(lo, hi)`` bands (e.g. the diffuse interface
    band) removed before scoring; the result also reports the unscored fraction
    so an over-generous mask cannot hide behind a small number.
    """
    y = np.asarray(y, dtype=np.float64)
    u_num = np.asarray(u_num, dtype=np.float64)
    u_ref = np.asarray(u_ref_fn(y), dtype=np.float64)
    keep = np.ones(y.shape, dtype=bool)
    for lo, hi in (exclude or []):
        keep &= ~((y >= lo) & (y <= hi))
    denom = np.max(np.abs(u_ref)) if np.max(np.abs(u_ref)) > 0 else 1.0
    err = np.abs(u_num[keep] - u_ref[keep]) / denom
    return {
        "err_max": float(err.max()),
        "err_rms": float(np.sqrt(np.mean(err ** 2))),
        "n_scored": int(keep.sum()),
        "n_total": int(y.size),
        "scored_fraction": float(keep.sum() / max(y.size, 1)),
        "u_ref_max": float(np.max(np.abs(u_ref))),
    }


def local_viscosity(
    y: np.ndarray, u: np.ndarray, g: float, half_width: int = 3
) -> np.ndarray:
    """Invert the Stokes balance ``nu = -g / u''`` node by node (central 2nd diff).

    The node spacing equals the lattice spacing (half-way walls do not enter a
    *second* derivative to leading order), so this is the direct, model-free
    viscosity measurement used to certify ``nu = cs^2 (1/omega - 1/2)``.
    """
    y = np.asarray(y, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    nu = np.full_like(u, np.nan)
    d2 = (u[2:] - 2.0 * u[1:-1] + u[:-2])
    with np.errstate(divide="ignore", invalid="ignore"):
        nu[1:-1] = -g / d2
    if half_width > 1:
        # light smoothing to fight lattice-scale even-odd noise
        k = np.ones(half_width) / half_width
        nu_s = np.convolve(np.nan_to_num(nu, nan=0.0), k, mode="same")
        nu = np.where(np.isnan(nu), np.nan, nu_s)
    return nu