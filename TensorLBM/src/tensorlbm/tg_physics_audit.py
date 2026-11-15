"""Taylor-Green physics audit for the host-staged halo cross-validation.

Context
-------
Mission ``XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO`` validated the
FluidX3D-style host-staged halo *transport* (decomposition invariance to
~1e-14) but recorded an **honest FAIL** on its physics arm: the container
solver ``HostStagedD3Q19`` missed the analytic Taylor-Green kinetic-energy
decay by 15-85 % and the measured rate did not scale with ``nu``.

This module is an *independent* audit of that FAIL.  It re-implements a
D3Q19 BGK periodic solver from scratch (single array, push streaming,
NumPy only -- no torch, no ghost layers, no shared code with
``tensorlbm.host_staged_halo``), reproduces the reported artefact, and
localises it.

Root cause (see ``docs/XVAL-LBM-HOST-STAGED-TG-MODAL-CLOSEOUT.md``)
------------------------------------------------------------------
The defect is **in the initial condition, not in the solver or the
transport**.  The XVAL driver builds the Taylor-Green field with

    x, y = meshgrid(arange(n) * k, arange(n) * k)      # already scaled
    ux   = sin(k * x) * cos(k * y)                     # k applied twice

so the field is ``sin(k^2 i)``, not ``sin(k i)``: not a Taylor-Green vortex,
not divergence-free, and -- because the double-scaled wavenumber is
irrational in lattice units -- the sampled field is neither periodic nor
zero-mean.  The residual net momentum is conserved by the lattice Boltzmann
collision, so the late-time kinetic energy is dominated by that mean drift,
whose decay is unrelated to ``2 nu k^2``.  That single defect explains all
three symptoms: the large error, the flat dependence on ``nu``, and the
non-convergence with grid.

This module provides
    * the corrected initial condition (:func:`taylor_green_velocity`),
    * a faithful reproduction of the malformed one
      (:func:`taylor_green_velocity_double_scaled`) for the regression test,
    * the physically meaningful observable -- the *modal* amplitude of the
      fundamental Fourier mode -- next to the integral kinetic energy,
    * :class:`IndependentD3Q19BGK`, a from-scratch reference solver used as
      the audit oracle,
    * :func:`run_audit`, which tabulates decay rates against the analytic
      ``2 nu k^2`` (velocity amplitude) and ``4 nu k^2`` (kinetic energy).

All quantities are in lattice units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "D3Q19_C",
    "D3Q19_W",
    "taylor_green_velocity",
    "taylor_green_velocity_double_scaled",
    "equilibrium",
    "IndependentD3Q19BGK",
    "mode_amplitude",
    "dominant_modes",
    "fit_exponential_rate",
    "analytic_decay_rate",
    "audit_case",
    "run_audit",
    "AUDIT_PLAN",
]

# ---------------------------------------------------------------------------
# D3Q19 lattice (independent definition; component order (x, y, z))
# ---------------------------------------------------------------------------

D3Q19_C: np.ndarray = np.array(
    [
        (0, 0, 0),
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
        (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
        (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
    ],
    dtype=np.float64,
)

D3Q19_W: np.ndarray = np.array(
    [1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12, dtype=np.float64
)

_CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def taylor_green_velocity(
    nx: int, ny: int | None = None, nz: int | None = None, u0: float = 0.05
) -> np.ndarray:
    """Correct 2-D Taylor-Green vortex embedded in a 3-D periodic box.

    ``u = u0 (sin kx cos ky, -cos kx sin ky, 0)`` with ``k = 2*pi/nx``.

    It is an exact solution of the incompressible Navier-Stokes equations on
    the torus: ``(u.grad)u = grad p`` pointwise, so the velocity amplitude
    decays as ``exp(-2 nu k^2 t)`` and the kinetic energy ``0.5<|u|^2>`` as
    ``exp(-4 nu k^2 t)``.

    The field is discretely divergence-free for ``k`` commensurate with the
    box (central differences) and carries only the ``(+-1, +-1)`` modes.
    """
    ny = nx if ny is None else ny
    nz = nx if nz is None else nz
    kx, ky = 2.0 * math.pi / nx, 2.0 * math.pi / ny
    # NOTE the single scaling: coordinates are lattice indices times k, and the
    # trigonometric functions are applied once.
    xx = np.arange(nx) * kx
    yy = np.arange(ny) * ky
    zz = np.arange(nz) * (2.0 * math.pi / nz)
    z, y, x = np.meshgrid(zz, yy, xx, indexing="ij")
    ux = u0 * np.sin(x) * np.cos(y)
    uy = -u0 * np.cos(x) * np.sin(y)
    uz = np.zeros_like(ux)
    return np.stack([ux, uy, uz])  # (3, nz, ny, nx)


def taylor_green_velocity_double_scaled(
    nx: int, ny: int | None = None, nz: int | None = None, u0: float = 0.05
) -> np.ndarray:
    """**Malformed** initial condition, as built by the XVAL driver.

    Reproduces the defect of ``tools/xval_host_staged_halo`` (see module
    docstring): the meshgrid is pre-scaled by ``k`` and the trigonometric
    argument is scaled by ``k`` a second time, so the sampled field is
    ``sin(k^2 i)``.  Kept only as a regression fixture -- do not use it as a
    physical initial condition.
    """
    ny = nx if ny is None else ny
    nz = nx if nz is None else nz
    kx, ky = 2.0 * math.pi / nx, 2.0 * math.pi / ny
    xx = np.arange(nx) * kx
    yy = np.arange(ny) * ky
    zz = np.arange(nz) * (2.0 * math.pi / nz)
    z, y, x = np.meshgrid(zz, yy, xx, indexing="ij")
    ux = u0 * np.sin(kx * x) * np.cos(ky * y)
    uy = -u0 * np.cos(kx * x) * np.sin(ky * y)
    return np.stack([ux, uy, np.zeros_like(ux)])


def equilibrium(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Second-order Maxwell-Boltzmann equilibrium, ``(19, nz, ny, nx)``."""
    cu = np.einsum("qa,azyx->qzyx", D3Q19_C, u)
    u2 = (u * u).sum(axis=0)
    return D3Q19_W[:, None, None, None] * rho[None] * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2[None]
    )


# ---------------------------------------------------------------------------
# Independent reference solver (audit oracle)
# ---------------------------------------------------------------------------


@dataclass
class IndependentD3Q19BGK:
    """Periodic D3Q19 BGK solver, single array, push streaming, NumPy only.

    Deliberately written in a different style from the subject
    implementation (``tensorlbm.host_staged_halo.HostStagedD3Q19``, which uses
    per-device slabs, ghost layers and *pull* streaming), so agreement between
    the two is evidence about the operator rather than about shared code.
    """

    nx: int
    ny: int | None = None
    nz: int | None = None
    nu: float = 0.02
    f: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ny = self.nx if self.ny is None else self.ny
        self.nz = self.nx if self.nz is None else self.nz
        self.omega = 1.0 / (3.0 * self.nu + 0.5)
        shape = (19, self.nz, self.ny, self.nx)
        self.f = np.zeros(shape)
        self.f[0] = 1.0  # rest state rho=1, u=0

    # -- setup ---------------------------------------------------------
    def set_velocity(self, u: np.ndarray, rho: float = 1.0) -> None:
        expected = (3, self.nz, self.ny, self.nx)
        if u.shape != expected:
            raise ValueError(f"expected {expected}, got {u.shape}")
        self.f = equilibrium(np.full(u.shape[1:], rho), u)

    # -- moments -------------------------------------------------------
    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        rho = self.f.sum(axis=0)
        u = np.einsum("qa,qzyx->azyx", D3Q19_C, self.f) / rho
        return rho, u

    def kinetic_energy(self, rho: np.ndarray | None = None, u: np.ndarray | None = None) -> float:
        if rho is None or u is None:
            rho, u = self.moments()
        return float((0.5 * rho * (u * u).sum(axis=0)).mean())

    def total_mass(self) -> float:
        return float(self.f.sum() / self.f[0].size)

    def total_momentum(self) -> np.ndarray:
        rho, u = self.moments()
        return (rho[None] * u).sum(axis=(1, 2, 3))

    # -- time step -----------------------------------------------------
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            rho, u = self.moments()
            feq = equilibrium(rho, u)
            self.f -= self.omega * (self.f - feq)
            out = np.empty_like(self.f)
            for q in range(19):
                cx, cy, cz = (int(v) for v in D3Q19_C[q])
                # push: out[i + c] = f[i]
                out[q] = np.roll(self.f[q], (cz, cy, cx), axis=(0, 1, 2))
            self.f = out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def mode_amplitude(u: np.ndarray, mode: tuple[int, int, int] = (0, 1, 1)) -> float:
    """Amplitude of the Fourier mode ``mode`` (``(kz, ky, kx)``) of ``u``.

    Returns ``sqrt(sum_comp |u_hat|^2)`` with ``u_hat = FFT(u)/N``.  A single
    travelling wave ``A*sin(kx)`` gives ``A/2``; the Taylor-Green field
    ``(u0 sin kx cos ky, -u0 cos kx sin ky)`` has ``u0/4`` per component and
    therefore ``u0*sqrt(2)/4`` in the mode amplitude of ``(+-1, +-1)``.
    """
    n = u[0].size
    hat = np.fft.fftn(u, axes=(1, 2, 3)) / n
    kz, ky, kx = mode
    return float(np.sqrt((np.abs(hat[:, kz, ky, kx]) ** 2).sum()))


def dominant_modes(u: np.ndarray, top: int = 6) -> list[tuple[tuple[int, int, int], float]]:
    """Fraction of ``<|u|^2>`` carried by the ``top`` Fourier modes."""
    n = u[0].size
    hat = np.fft.fftn(u, axes=(1, 2, 3)) / n
    power = (np.abs(hat) ** 2).sum(axis=0)  # (nz, ny, nx)
    total = float(power.sum())
    order = np.argsort(power.ravel())[::-1][:top]
    out = []
    for flat in order:
        idx = np.unravel_index(flat, power.shape)
        out.append(((int(idx[0]), int(idx[1]), int(idx[2])), float(power[idx] / total)))
    return out


def fit_exponential_rate(t: np.ndarray, y: np.ndarray) -> float:
    """Least-squares rate ``gamma`` of ``y ~ A exp(-gamma t)``."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.size < 2:
        raise ValueError("need at least two samples")
    slope = np.polyfit(t, np.log(y), 1)[0]
    return float(-slope)


def analytic_decay_rate(nu: float, k: float, observable: str = "velocity") -> float:
    """Analytic decay exponent of the Taylor-Green fundamental mode.

    ``velocity`` amplitude (and hence ``u_rms``) decays with ``2 nu k^2``;
    the kinetic energy ``0.5<|u|^2>`` with ``4 nu k^2``.
    """
    if observable == "velocity":
        return 2.0 * nu * k * k
    if observable == "kinetic_energy":
        return 4.0 * nu * k * k
    raise ValueError("observable must be 'velocity' or 'kinetic_energy'")


AUDIT_PLAN: dict[str, object] = {
    "grids": (16, 24, 32),
    "nus": (0.005, 0.01, 0.02, 0.05, 0.10),
    "steps": 600,
    "fit_window": (60, 480),
    "u0": 0.05,
    "observables": ("velocity", "kinetic_energy"),
}


def _fit_window_samples(fit_window: tuple[int, int], stride: int) -> np.ndarray:
    lo, hi = fit_window
    return np.arange(lo, hi + 1, stride)


def audit_case(
    n: int,
    nu: float,
    u0: float = 0.05,
    steps: int = 600,
    fit_window: tuple[int, int] = (60, 480),
    stride: int = 20,
    malformed: bool = False,
) -> dict[str, object]:
    """Run one Taylor-Green decay case and return the measured rates."""
    k = 2.0 * math.pi / n
    builder = taylor_green_velocity_double_scaled if malformed else taylor_green_velocity
    u_init = builder(n, n, n, u0)

    sim = IndependentD3Q19BGK(n, nu=nu)
    sim.set_velocity(u_init)
    rho0, uu0 = sim.moments()
    k0_measured = sim.kinetic_energy(rho0, uu0)
    k0_hardcoded = u0 * u0 / 4.0  # what the XVAL driver assumed
    momentum0 = sim.total_momentum()

    window = _fit_window_samples(fit_window, stride)
    ts, amps, kes = [], [], []
    for t in range(1, steps + 1):
        sim.step(1)
        if t in set(int(w) for w in window):
            rho, u = sim.moments()
            ts.append(float(t))
            amps.append(mode_amplitude(u, (0, 1, 1)))
            kes.append(sim.kinetic_energy(rho, u))

    ts_a = np.array(ts)
    g_amp = fit_exponential_rate(ts_a, np.array(amps))
    g_ke = fit_exponential_rate(ts_a, np.array(kes))
    g_ke_cum = math.log(k0_hardcoded / kes[-1]) / ts[-1] if kes[-1] > 0 else float("nan")

    rho, u = sim.moments()
    theo_amp = analytic_decay_rate(nu, k, "velocity")
    theo_ke = analytic_decay_rate(nu, k, "kinetic_energy")
    modes = dominant_modes(u, top=5)
    return {
        "grid": n,
        "nu": nu,
        "tau": 3.0 * nu + 0.5,
        "malformed_ic": bool(malformed),
        "k": k,
        "k0_measured": k0_measured,
        "k0_hardcoded": k0_hardcoded,
        "k0_rel_err": abs(k0_hardcoded - k0_measured) / k0_measured,
        "momentum0": [float(m) for m in momentum0],
        "mean_momentum0": float(np.abs(momentum0 / (n ** 3)).max()),
        "gamma_modal": g_amp,
        "gamma_modal_theory": theo_amp,
        "gamma_modal_rel_err": abs(g_amp - theo_amp) / theo_amp,
        "gamma_ke": g_ke,
        "gamma_ke_theory": theo_ke,
        "gamma_ke_rel_err": abs(g_ke - theo_ke) / theo_ke,
        "gamma_ke_cum_hardcoded_k0": g_ke_cum,
        "gamma_ke_cum_rel_err": abs(g_ke_cum - theo_ke) / theo_ke,
        "dominant_modes": [(m, round(f, 6)) for m, f in modes],
        "steps": steps,
        "fit_window": list(fit_window),
        "stride": stride,
    }


def run_audit(
    grids: tuple[int, ...] = (16, 24),
    nus: tuple[float, ...] = (0.02, 0.05),
    steps: int = 400,
    fit_window: tuple[int, int] = (40, 320),
    stride: int = 20,
    u0: float = 0.05,
    include_malformed: bool = True,
) -> list[dict[str, object]]:
    """Sweep grids x viscosities, corrected and (optionally) malformed IC."""
    rows: list[dict[str, object]] = []
    for n in grids:
        for nu in nus:
            rows.append(
                audit_case(n, nu, u0=u0, steps=steps, fit_window=fit_window, stride=stride)
            )
            if include_malformed:
                rows.append(
                    audit_case(
                        n, nu, u0=u0, steps=steps, fit_window=fit_window, stride=stride,
                        malformed=True,
                    )
                )
    return rows