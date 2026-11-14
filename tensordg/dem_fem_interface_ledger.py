"""DEM-FEM interface energy / momentum ledger with *physical* residuals.

Ledger anchor: ``GAP-ICE-LEDGER-PHYSICS``
    「冰-结构耦合能量/动量账本须给物理残差, 不得以后端 parity 充抵」
    (DEM-FEM interface flux conservation must be machine-graded on a
    *physical* residual <= 1e-3 and an FEM participant must be present; a
    backend-consistency number may only be labelled "backend port parity".)

What this module provides
-------------------------
A reduced (1-D axial) DEM-FEM impact benchmark with three independent
participants and a ledger closed from physics, not from implementation
agreement:

``FemRod``
    Consistent-mass P1 axial rod, clamped at ``x = L`` and free at ``x = 0``.
    This is the FEM participant: an assembled mass/stiffness pair whose
    free-end momentum change is measured independently of the DEM force
    record.
``DemParticles`` (see :class:`ParticleSpec` / :func:`_dem_dem_forces`)
    A train of rigid particles (ice-rubble analogue) with pairwise penalty
    contact, travelling towards the free end of the rod.  This is the DEM
    participant.
``ModalReference``
    Independent integrator: exact Duhamel solution (piecewise-linear load) of
    the *same* semi-discrete FEM system in its eigenbasis.  It is used both to
    verify the RK4 integrator and to produce the **backend port parity**
    number that the ledger anchor explicitly refuses as conservation
    evidence.

Ledger residuals (dimensionless, all physical)
----------------------------------------------
``momentum_residual``
    ``|p(T) - p(0)| / p_ref`` with ``p = sum_i m_i v_i + (M 1)^T v_fem``,
    the exact momentum functional of the P1 interpolant (``M 1`` integrates
    the shape functions, ``int rho A N dx``).  No external load acts on the
    system, so the physical residual is zero.
``energy_residual``
    ``|E(T) + D_diss - E(0)| / E_ref`` with
    ``E = K_dem + K_fem + U_fem + sum 1/2 k delta^2`` and ``D_diss`` the
    accumulated damping work.
``interface_impulse_residual``
    ``|J_dem - dp_fem| / |J_dem|`` where ``J_dem`` integrates the *DEM-side*
    contact-force record and ``dp_fem`` is the momentum change of the *FEM
    participant alone*.  The two numbers come from disjoint code paths, so
    this is a real Newton's-third-law / interface-flux check -- not a parity
    check between two discretizations of the same participant.
``interface_work_residual``
    ``|W_dem + W_fem + dE_spring + D_int| / E_ref``: interface work measured on
    the DEM side plus the work measured on the FEM side must close against the
    contact-spring energy and the interface damping work.

Deliberate defects (used to falsify "parity is enough")
------------------------------------------------------
``force_split``
    applies ``force_split * F`` to the FEM participant but ``-F`` to the DEM
    particle: a genuine interface-flux (Newton's third law) violation.
``frozen_coupling``
    explicit staggered partitioned coupling: the interface force is evaluated
    once per step and held constant for both participants -- the classic
    partitioned-coupling energy leak.

Both defects leave the backend port parity number essentially unchanged (the
FEM participant is still integrated to O(dt^4) against the modal reference),
which is precisely why parity cannot stand in for a physical residual;
``benchmarks/ice_contact_ledger.py`` quantifies it.

Scope / limitations
-------------------
* 1-D axial continuum.  The ledger algebra is dimension independent, but the
  bending / plate / 3-D DEM-pack case is *not* covered here.
* Penalty contact only (no CZM, no crushable ice), linear elasticity only.
* Small: rod wave transit ~1e-3 s, contact ~2e-3 s, ``dt`` ~1e-6 s -> seconds
  of CPU on one core.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

D = torch.float64

#: Machine gate demanded by the ledger anchor.
RESIDUAL_GATE = 1.0e-3


# --------------------------------------------------------------------------
# Specifications
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RodSpec:
    """P1 axial rod (FEM participant); clamped at ``x = L``, free at ``x = 0``."""

    length: float = 1.0
    density: float = 900.0
    young: float = 9.0e9
    area: float = 0.01

    @property
    def wave_speed(self) -> float:
        return math.sqrt(self.young / self.density)

    @property
    def total_mass(self) -> float:
        return self.density * self.area * self.length

    @property
    def axial_stiffness(self) -> float:
        return self.young * self.area


@dataclass(frozen=True)
class ParticleSpec:
    """Rigid particles in a train; index 0 is nearest the rod free end."""

    masses: tuple[float, ...] = (5.0,)
    radii: tuple[float, ...] = (0.05,)
    gaps: tuple[float, ...] = (0.0,)
    initial_velocity: float = 1.0

    def __post_init__(self) -> None:
        if not (len(self.masses) == len(self.radii) == len(self.gaps)):
            raise ValueError("masses / radii / gaps must have equal length")


@dataclass(frozen=True)
class ContactSpec:
    """Linear penalty contact (elastic part + optional linear damping)."""

    stiffness: float = 1.0e7
    damping: float = 0.0
    particle_stiffness: float | None = None

    @property
    def dem_stiffness(self) -> float:
        return self.stiffness if self.particle_stiffness is None else self.particle_stiffness


@dataclass(frozen=True)
class CouplingConfig:
    rod: RodSpec = field(default_factory=RodSpec)
    particles: ParticleSpec = field(default_factory=ParticleSpec)
    contact: ContactSpec = field(default_factory=ContactSpec)
    n_elem: int = 32
    dt: float = 1.0e-6
    t_end: float = 2.0e-3
    force_split: float = 1.0
    frozen_coupling: bool = False


# --------------------------------------------------------------------------
# FEM participant
# --------------------------------------------------------------------------
class FemRod:
    """Consistent-mass P1 axial rod: node 0 free (impact end), node n clamped."""

    def __init__(self, spec: RodSpec, n_elem: int = 32, clamped: bool = False):
        if n_elem < 2:
            raise ValueError("n_elem must be >= 2")
        self.spec = spec
        self.n_elem = int(n_elem)
        self.clamped = bool(clamped)
        self.h = spec.length / self.n_elem
        self.ndof_full = self.n_elem + 1
        self.nfree = self.ndof_full - 1 if self.clamped else self.ndof_full

        ea = spec.axial_stiffness
        rho_a = spec.density * spec.area
        h = self.h
        ke = ea / h * torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=D)
        me = rho_a * h / 6.0 * torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=D)

        k_full = torch.zeros(self.ndof_full, self.ndof_full, dtype=D)
        m_full = torch.zeros(self.ndof_full, self.ndof_full, dtype=D)
        for e in range(self.n_elem):
            for a in range(2):
                for b in range(2):
                    k_full[e + a, e + b] += ke[a, b]
                    m_full[e + a, e + b] += me[a, b]

        self.k_full = k_full
        self.m_full = m_full
        self.k = k_full[: self.nfree, : self.nfree].clone()
        self.m = m_full[: self.nfree, : self.nfree].clone()
        #: exact momentum functional ``(M_full @ 1)`` restricted to free dofs
        self.momentum_functional = (m_full @ torch.ones(self.ndof_full, dtype=D))[: self.nfree].clone()

    # -- energies / momentum ---------------------------------------------
    def strain_energy(self, u: torch.Tensor) -> torch.Tensor:
        return 0.5 * u @ (self.k @ u)

    def kinetic_energy(self, v: torch.Tensor) -> torch.Tensor:
        return 0.5 * v @ (self.m @ v)

    def momentum(self, v: torch.Tensor) -> torch.Tensor:
        return self.momentum_functional @ v

    def acceleration(self, u: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(self.m, load - self.k @ u)

    # -- exact analytic cross-checks (references, not parity) ------------
    def analytic_static_elongation(self, load: float) -> float:
        """``F L / (E A)``: the P1 FEM reproduces this *exactly* (node-wise)."""
        return load * self.spec.length / self.spec.axial_stiffness

    def fem_static_elongation(self, load: float) -> float:
        """Self-equilibrated pair ``+F`` at node 0, ``-F`` at the last free node."""
        k = self.k
        rhs = torch.zeros(self.nfree, dtype=D)
        rhs[0] += load
        rhs[-1] -= load
        # K is singular (rigid mode) -> solve on the mean-zero subspace
        u = torch.linalg.lstsq(k, rhs.unsqueeze(1)).solution.squeeze(1)
        return float(u[0] - u[-1])

    def analytic_eigenfrequencies(self, n_modes: int = 3) -> list[float]:
        """Free-free: ``n pi c / L`` (n = 0 rigid, 1, 2 ...); clamped-free: odd."""
        c = self.spec.wave_speed
        if self.clamped:
            return [(2 * n - 1) * math.pi * c / (2.0 * self.spec.length)
                    for n in range(1, n_modes + 1)]
        return [n * math.pi * c / self.spec.length for n in range(n_modes)]

    def fem_eigenfrequencies(self, n_modes: int = 3) -> list[float]:
        lam, _ = self.generalised_eigen()
        return [math.sqrt(max(float(x), 0.0)) for x in lam[:n_modes]]

    def generalised_eigen(self):
        """``K phi = omega^2 M phi`` by Cholesky whitening (M is SPD)."""
        chol = torch.linalg.cholesky(self.m)
        inv_chol = torch.linalg.inv(chol)
        a = inv_chol @ self.k @ inv_chol.T
        lam, vec = torch.linalg.eigh(a)
        return lam, inv_chol.T @ vec  # M-orthonormal eigenvectors


# --------------------------------------------------------------------------
# Independent integrator: exact modal (Duhamel) solution
# --------------------------------------------------------------------------
class ModalReference:
    """Exact solution of the semi-discrete FEM system under a recorded load.

    For a load that is piecewise linear on the uniform recording grid the
    Duhamel convolution is exact through a two-term recursion, so this
    reference differs from the RK4 trajectory only by the time-integration
    error: an independent integrator, not a re-run of the same code.
    """

    def __init__(self, rod: FemRod):
        self.rod = rod
        lam, phi = rod.generalised_eigen()
        self.omega = torch.sqrt(torch.clamp(lam, min=0.0))
        self.phi = phi

    def solve(self, u0: torch.Tensor, v0: torch.Tensor,
              load_history: torch.Tensor, dt: float) -> dict:
        """``load_history`` has shape ``(n_steps + 1, nfree)`` at ``t_k = k dt``."""
        omega = self.omega
        zero = omega <= 1e-9 * max(float(omega.max()), 1.0)
        w = torch.where(zero, torch.ones_like(omega), omega)
        modal_load = load_history @ self.phi
        n_steps = modal_load.shape[0] - 1
        q0 = self.phi.T @ (self.rod.m @ u0)
        qd0 = self.phi.T @ (self.rod.m @ v0)

        q = torch.zeros(n_steps + 1, w.numel(), dtype=D)
        qd = torch.zeros_like(q)
        q[0] = q0
        qd[0] = qd0

        sin_wd = torch.sin(w * dt)
        cos_wd = torch.cos(w * dt)
        a_i = (1.0 - cos_wd) / w
        b_i = sin_wd / w ** 2 - dt * cos_wd / w
        a_j = sin_wd / w
        b_j = (cos_wd - 1.0) / w ** 2 + dt * sin_wd / w

        # oscillatory modes: I = int sin(w(t-s)) f ds, J = int cos(w(t-s)) f ds
        i_prev = torch.zeros_like(w)
        j_prev = torch.zeros_like(w)
        # rigid modes: P = int (t-s) f ds, R = int f ds
        p_prev = torch.zeros_like(w)
        r_prev = torch.zeros_like(w)

        for k in range(n_steps):
            f0 = modal_load[k]
            df = modal_load[k + 1] - f0
            loc_i = f0 * a_i + (df / dt) * (dt * a_i - b_i)
            loc_j = f0 * a_j + (df / dt) * (dt * a_j - b_j)
            i_new = cos_wd * i_prev + sin_wd * j_prev + loc_i
            j_new = -sin_wd * i_prev + cos_wd * j_prev + loc_j
            loc_p = f0 * (dt * dt / 2.0) + df * (dt * dt / 6.0)
            p_new = p_prev + dt * r_prev + loc_p
            r_new = r_prev + 0.5 * (f0 + modal_load[k + 1]) * dt

            t = (k + 1) * dt
            q_osc = q0 * torch.cos(w * t) + (qd0 / w) * torch.sin(w * t) + i_new / w
            qd_osc = -q0 * w * torch.sin(w * t) + qd0 * torch.cos(w * t) + j_new
            q_rig = q0 + qd0 * t + p_new
            qd_rig = qd0 + r_new
            q[k + 1] = torch.where(zero, q_rig, q_osc)
            qd[k + 1] = torch.where(zero, qd_rig, qd_osc)
            i_prev, j_prev = i_new, j_new
            p_prev, r_prev = p_new, r_new
        return {"u": q @ self.phi.T, "v": qd @ self.phi.T, "q": q, "omega": omega}


# --------------------------------------------------------------------------
# DEM participant helpers
# --------------------------------------------------------------------------
def initial_particle_state(spec: ParticleSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Particle centres (``x``) and velocities; particle 0 abuts the rod end."""
    n = len(spec.masses)
    x = torch.zeros(n, dtype=D)
    x[0] = -(spec.radii[0] + spec.gaps[0])
    for i in range(1, n):
        x[i] = x[i - 1] - (spec.radii[i - 1] + spec.radii[i]) - spec.gaps[i]
    v = torch.full((n,), float(spec.initial_velocity), dtype=D)
    return x, v


def interface_force(x: torch.Tensor, v: torch.Tensor, u_rod_end: float,
                    v_rod_end: float, contact: ContactSpec,
                    lead_radius: float) -> torch.Tensor:
    """Penalty contact force particle 0 <-> rod free end (compression only)."""
    overlap = float(x[0]) + lead_radius - u_rod_end
    if overlap <= 0.0:
        return torch.zeros((), dtype=D)
    f = contact.stiffness * overlap + contact.damping * (float(v[0]) - v_rod_end)
    return torch.tensor(max(f, 0.0), dtype=D)


def dem_dem_forces(x: torch.Tensor, v: torch.Tensor, spec: ParticleSpec,
                   contact: ContactSpec) -> torch.Tensor:
    """Net inter-particle force (positive = +x) on every particle."""
    n = x.numel()
    net = torch.zeros(n, dtype=D)
    k = contact.dem_stiffness
    for i in range(n - 1):
        overlap = float(x[i] - spec.radii[i] - (x[i + 1] + spec.radii[i + 1]))
        if overlap > 0.0:
            f = k * overlap + contact.damping * (float(v[i]) - float(v[i + 1]))
            f = max(f, 0.0)
            net[i] -= f
            net[i + 1] += f
    return net


def dissipation_rate(u: torch.Tensor, v: torch.Tensor, x: torch.Tensor,
                      w: torch.Tensor, ps: ParticleSpec,
                      contact: ContactSpec, lead_radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(interface damping rate, DEM-DEM damping rate) in W."""
    rate_int = torch.zeros((), dtype=D)
    overlap = float(x[0]) + lead_radius - float(u[0])
    if overlap > 0.0:
        rel = float(w[0]) - float(v[0])
        rate_int = contact.damping * rel * rel
    rate_dem = torch.zeros((), dtype=D)
    for i in range(x.numel() - 1):
        ov = float(x[i] - ps.radii[i] - (x[i + 1] + ps.radii[i + 1]))
        if ov > 0.0:
            rel = float(w[i]) - float(w[i + 1])
            rate_dem = rate_dem + contact.damping * rel * rel
    return rate_int, rate_dem


# --------------------------------------------------------------------------
# Coupled integration + ledger
# --------------------------------------------------------------------------
def run_coupling(cfg: CouplingConfig, keep_history: bool = True) -> dict:
    """RK4 integration of the coupled DEM-FEM system plus the physical ledger."""
    rod = FemRod(cfg.rod, cfg.n_elem)
    ps = cfg.particles
    cs = cfg.contact
    nfree = rod.nfree
    npart = len(ps.masses)
    masses = torch.tensor(ps.masses, dtype=D)
    lead_radius = ps.radii[0]

    u = torch.zeros(nfree, dtype=D)
    v = torch.zeros(nfree, dtype=D)
    x, w = initial_particle_state(ps)

    n_steps = max(1, int(round(cfg.t_end / cfg.dt)))
    dt = cfg.t_end / n_steps

    def forces(uu, vv, xx, ww):
        f = interface_force(xx, ww, float(uu[0]), float(vv[0]), cs, lead_radius)
        return f, dem_dem_forces(xx, ww, ps, cs)

    def deriv(uu, xx, ww, f_int, net_dem):
        load = torch.zeros(nfree, dtype=D)
        load[0] = cfg.force_split * f_int
        a_rod = rod.acceleration(uu, load)
        net = net_dem.clone()
        net[0] = net[0] - f_int
        return a_rod, net / masses

    def rk4(uu, vv, xx, ww, frozen):
        """One RK4 step of (u, v, x, w); ``frozen`` holds the interface force."""
        f0, n0 = forces(uu, vv, xx, ww)
        af = (lambda *_: (f0, n0)) if frozen else forces   # noqa: E731

        # stage 1 -- dy/dt = (v, a_rod, w, a_part)
        a_u1, a_w1 = deriv(uu, xx, ww, *af(uu, vv, xx, ww))
        b_u1, b_w1 = vv, ww

        # stage 2
        su = uu + 0.5 * dt * b_u1
        sv = vv + 0.5 * dt * a_u1
        sx = xx + 0.5 * dt * b_w1
        sw = ww + 0.5 * dt * a_w1
        a_u2, a_w2 = deriv(su, sx, sw, *af(su, sv, sx, sw))
        b_u2, b_w2 = sv, sw

        # stage 3
        su = uu + 0.5 * dt * b_u2
        sv = vv + 0.5 * dt * a_u2
        sx = xx + 0.5 * dt * b_w2
        sw = ww + 0.5 * dt * a_w2
        a_u3, a_w3 = deriv(su, sx, sw, *af(su, sv, sx, sw))
        b_u3, b_w3 = sv, sw

        # stage 4
        su = uu + dt * b_u3
        sv = vv + dt * a_u3
        sx = xx + dt * b_w3
        sw = ww + dt * a_w3
        a_u4, a_w4 = deriv(su, sx, sw, *af(su, sv, sx, sw))
        b_u4, b_w4 = sv, sw

        u_new = uu + dt / 6.0 * (b_u1 + 2 * b_u2 + 2 * b_u3 + b_u4)
        v_new = vv + dt / 6.0 * (a_u1 + 2 * a_u2 + 2 * a_u3 + a_u4)
        x_new = xx + dt / 6.0 * (b_w1 + 2 * b_w2 + 2 * b_w3 + b_w4)
        w_new = ww + dt / 6.0 * (a_w1 + 2 * a_w2 + 2 * a_w3 + a_w4)
        return u_new, v_new, x_new, w_new

    # -- ledger state ----------------------------------------------------
    p_ref = float((masses * w.abs()).sum())
    e_ref = float(0.5 * (masses * w ** 2).sum())
    p0 = float(rod.momentum(v) + (masses * w).sum())
    e0 = float(rod.kinetic_energy(v) + rod.strain_energy(u)
               + 0.5 * (masses * w ** 2).sum())

    j_dem = 0.0
    w_dem = 0.0        # work of the interface force on the DEM particle
    w_fem = 0.0        # work of the interface force on the FEM participant
    d_diss = 0.0
    d_diss_int = 0.0

    hist = {"time": torch.zeros(n_steps + 1, dtype=D),
            "force": torch.zeros(n_steps + 1, dtype=D),
            "rod_end_velocity": torch.zeros(n_steps + 1, dtype=D),
            "rod_end_displacement": torch.zeros(n_steps + 1, dtype=D),
            "particle_velocity": torch.zeros(n_steps + 1, npart, dtype=D),
            "energy": torch.zeros(n_steps + 1, 4, dtype=D),
            "load": torch.zeros(n_steps + 1, nfree, dtype=D)}

    def record(k, uu, vv, xx, ww):
        f, _ = forces(uu, vv, xx, ww)
        hist["time"][k] = k * dt
        hist["force"][k] = f
        hist["rod_end_velocity"][k] = vv[0]
        hist["rod_end_displacement"][k] = uu[0]
        hist["particle_velocity"][k] = ww
        hist["load"][k, 0] = cfg.force_split * f
        e_spr = torch.zeros((), dtype=D)
        ov = float(xx[0]) + lead_radius - float(uu[0])
        if ov > 0.0:
            e_spr = 0.5 * cs.stiffness * ov * ov
        hist["energy"][k, 0] = rod.kinetic_energy(vv)
        hist["energy"][k, 1] = rod.strain_energy(uu)
        hist["energy"][k, 2] = 0.5 * (masses * ww ** 2).sum()
        hist["energy"][k, 3] = e_spr
        return f

    f_prev = record(0, u, v, x, w)
    v_rod_prev = float(v[0])
    v_part_prev = float(w[0])
    rate_int_prev, rate_dem_prev = dissipation_rate(u, v, x, w, ps, cs, lead_radius)

    for k in range(n_steps):
        u, v, x, w = rk4(u, v, x, w, cfg.frozen_coupling)
        f_now = record(k + 1, u, v, x, w)
        v_rod_now = float(v[0])
        v_part_now = float(w[0])
        j_dem += 0.5 * (f_prev + f_now) * dt
        w_dem += 0.5 * (-f_prev * v_part_prev - f_now * v_part_now) * dt
        w_fem += 0.5 * (cfg.force_split * f_prev * v_rod_prev
                        + cfg.force_split * f_now * v_rod_now) * dt
        rate_int_now, rate_dem_now = dissipation_rate(u, v, x, w, ps, cs, lead_radius)
        d_diss_int += 0.5 * (rate_int_prev + rate_int_now) * dt
        d_diss += 0.5 * (rate_int_prev + rate_dem_prev + rate_int_now + rate_dem_now) * dt
        f_prev, v_rod_prev, v_part_prev = f_now, v_rod_now, v_part_now
        rate_int_prev, rate_dem_prev = rate_int_now, rate_dem_now

    # -- physical residuals ---------------------------------------------
    p_end = float(rod.momentum(v) + (masses * w).sum())
    e_end = float(rod.kinetic_energy(v) + rod.strain_energy(u)
                  + 0.5 * (masses * w ** 2).sum())
    dp_fem = float(rod.momentum(v)) - float(rod.momentum(torch.zeros(nfree, dtype=D)))
    e_spr_end = float(hist["energy"][-1, 3])

    momentum_residual = abs(p_end - p0) / max(p_ref, 1e-300)
    energy_residual = abs((e_end + d_diss) - e0) / max(e_ref, 1e-300)
    impulse_residual = abs(j_dem - dp_fem) / max(abs(j_dem), 1e-300)
    work_residual = abs(w_dem + w_fem + e_spr_end + d_diss_int) / max(e_ref, 1e-300)

    ledger = {
        "gate": RESIDUAL_GATE,
        "momentum_residual": momentum_residual,
        "energy_residual": energy_residual,
        "interface_impulse_residual": impulse_residual,
        "interface_work_residual": work_residual,
        "passes_gate": bool(max(momentum_residual, energy_residual,
                                impulse_residual, work_residual) <= RESIDUAL_GATE),
        "p_ref": p_ref, "e_ref": e_ref,
        "impulse_dem": j_dem, "impulse_fem": dp_fem,
        "work_dem": w_dem, "work_fem": w_fem,
        "spring_energy_final": e_spr_end, "dissipated": d_diss,
        "fem_participant_dofs": rod.nfree,
        "fem_participant_mass": float(rod.momentum_functional.sum()),
        "rod_peak_strain_energy": float(hist["energy"][:, 1].max()),
        "peak_contact_force": float(hist["force"].max()),
    }

    out = {"ledger": ledger, "rod": rod, "cfg": cfg, "dt": dt, "n_steps": n_steps}
    if keep_history:
        out["history"] = hist
    return out


# --------------------------------------------------------------------------
# Backend port parity (the metric the anchor refuses)
# --------------------------------------------------------------------------
def backend_port_parity(result: dict) -> dict:
    """Compare the FEM participant against the modal integrator.

    Both are driven by the *identical recorded nodal load*, so this number
    measures agreement between two backends of the same participant.  It is
    reported as ``backend port parity`` only -- the ledger anchor states it
    must not be used as conservation evidence.
    """
    rod: FemRod = result["rod"]
    hist = result["history"]
    ref = ModalReference(rod).solve(torch.zeros(rod.nfree, dtype=D),
                                    torch.zeros(rod.nfree, dtype=D),
                                    hist["load"], result["dt"])
    num = hist["rod_end_velocity"]
    ref_v = ref["v"][:, 0]
    denom = float(torch.linalg.vector_norm(ref_v))
    err = float(torch.linalg.vector_norm(num - ref_v))
    return {
        "label": "backend port parity",
        "metric": "relative L2 of rod free-end velocity, FEM RK4 vs modal backend",
        "value": err / max(denom, 1e-300),
        "absolute": err,
    }


# --------------------------------------------------------------------------
# Reference (non-parity) verifications
# --------------------------------------------------------------------------
def verify_fem_static(rod: FemRod, load: float = 1.0e4) -> dict:
    """Self-equilibrated pair: ``u(0) - u(L) = F L / (E A)`` exactly for P1."""
    exact = rod.analytic_static_elongation(load)
    fem = rod.fem_static_elongation(load)
    return {"exact": exact, "fem": fem,
            "relative_error": abs(fem - exact) / abs(exact)}


def verify_fem_eigenfrequencies(spec: RodSpec, n_elems=(8, 16, 32, 64),
                                n_modes: int = 3, clamped: bool = False) -> list[dict]:
    out = []
    for n in n_elems:
        rod = FemRod(spec, n, clamped=clamped)
        exact = rod.analytic_eigenfrequencies(n_modes + 1 if not clamped else n_modes)
        fem = rod.fem_eigenfrequencies(len(exact))
        pairs = [(f, e) for f, e in zip(fem, exact) if e > 0.0]
        out.append({"n_elem": n,
                    "exact": [e for _, e in pairs], "fem": [f for f, _ in pairs],
                    "relative_errors": [abs(f - e) / e for f, e in pairs]})
    return out


def verify_contact_oscillator(mass: float = 5.0, stiffness: float = 1.0e7,
                              v0: float = 1.0, dt: float = 1.0e-8) -> dict:
    """Rigid-wall limit of the DEM participant.

    A particle of mass ``m`` hitting a fixed wall through the same linear
    penalty spring satisfies ``delta(t) = (v0/w) sin(w t)`` with
    ``w = sqrt(k/m)``; hence contact duration ``pi/w`` and peak force
    ``v0 sqrt(k m)``.  This is an analytic reference for the contact law, not a
    comparison between two of our own backends.
    """
    w = math.sqrt(stiffness / mass)
    analytic = {"omega": w, "contact_duration": math.pi / w,
                "peak_force": v0 * math.sqrt(stiffness * mass)}

    def acc(delta, ddelta):
        return -w * w * delta, stiffness * max(delta, 0.0)

    delta, ddelta = 0.0, v0
    t, peak, contact_time = 0.0, 0.0, 0.0
    n_max = int(round(4.0 * analytic["contact_duration"] / dt))
    for _ in range(n_max):
        a1, f1 = acc(delta, ddelta)
        a2, _ = acc(delta + 0.5 * dt * ddelta, ddelta + 0.5 * dt * a1)
        a3, _ = acc(delta + 0.5 * dt * (ddelta + 0.5 * dt * a1), ddelta + 0.5 * dt * a2)
        a4, _ = acc(delta + dt * (ddelta + 0.5 * dt * a2), ddelta + dt * a3)
        delta = delta + dt / 6.0 * (ddelta + 2 * (ddelta + 0.5 * dt * a1)
                                    + 2 * (ddelta + 0.5 * dt * a2) + (ddelta + dt * a3))
        ddelta = ddelta + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        t += dt
        peak = max(peak, f1)
        contact_time = t
        if delta < 0.0:
            break
    return {"analytic": analytic,
            "numerical": {"peak_force": peak, "contact_duration": contact_time},
            "relative_error_peak_force": abs(peak - analytic["peak_force"]) / analytic["peak_force"],
            "relative_error_duration": abs(contact_time - analytic["contact_duration"])
            / analytic["contact_duration"]}


def verify_stiff_rod_limit(mass: float = 5.0, stiffness: float = 1.0e7,
                           v0: float = 1.0, n_elem: int = 16,
                           young: float = 9.0e11, dt: float = 5.0e-7,
                           t_end: float = 3.0e-3) -> dict:
    """Analytic two-body impact limit of the *coupled* system.

    As ``E A -> inf`` the FEM participant becomes a rigid mass ``M``, so the
    contact is the two-body harmonic oscillator with reduced mass
    ``m M / (m + M)``::

        w_rel = sqrt(k (m + M) / (m M)),  F_peak = v0 sqrt(k m M / (m + M)),
        contact duration = pi / w_rel.

    This is the analytic reference for the *coupled* DEM-FEM run (the rod
    merely has to be stiff enough); it is a physical benchmark, not a
    comparison between two of our own backends.
    """
    spec = RodSpec(young=young)
    cfg = CouplingConfig(rod=spec, particles=ParticleSpec(
        masses=(mass,), radii=(0.05,), gaps=(0.0,), initial_velocity=v0),
        n_elem=n_elem, dt=dt, t_end=t_end)
    out = run_coupling(cfg, keep_history=True)
    big_m = spec.total_mass
    w_rel = math.sqrt(stiffness * (mass + big_m) / (mass * big_m))
    analytic = {"omega_rel": w_rel,
                "reduced_mass": mass * big_m / (mass + big_m),
                "peak_force": v0 * math.sqrt(stiffness * mass * big_m / (mass + big_m)),
                "contact_duration": math.pi / w_rel}
    f = out["history"]["force"]
    t = out["history"]["time"]
    active = f > 1.0
    if not bool(active.any()):
        raise RuntimeError("no contact detected in the stiff-rod limit run")
    idx = torch.nonzero(active).flatten()
    first, last = int(idx[0]), int(idx[0])
    for i in idx.tolist():
        if i <= last + 1:
            last = i
        else:
            break
    duration = float(t[last] - t[first])
    peak = float(f.max())
    return {"analytic": analytic,
            "numerical": {"peak_force": peak, "contact_duration": duration},
            "relative_error_peak_force": abs(peak - analytic["peak_force"]) / analytic["peak_force"],
            "relative_error_duration": abs(duration - analytic["contact_duration"])
            / analytic["contact_duration"],
            "ledger": out["ledger"]}
