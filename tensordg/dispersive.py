"""Linear dispersive water-wave operators and solitary-wave (KdV) verification.

Scope
-----
TensorDG previously had no dispersive-wave capability at all: every verified
case in :mod:`tensordg` is either hyperbolic conservation (Euler/SWE/Burgers) or
parabolic (heat/Poisson/SIPG).  Wave problems where the *dispersion relation*
sets the answer were untested.  This module supplies two things and one
verification surface for each.

1. A **linear dispersive operator family** for the surface-elevation equation

       eta_tt = -g K eta,     K^ = k tanh(k h)   (Airy / exact linear water wave)

   The caret denotes the Fourier symbol; ``h`` is the still-water depth.  The
   operator is applied pseudo-spectrally, so the eigenfunctions are the Fourier
   modes and the semi-discrete frequency of mode ``k`` is exactly the kernel
   used.  Boussinesq / Green--Naghdi class models are obtained by replacing
   ``tanh(x)/x`` with a rational (Pade) approximation, which is precisely what
   those models do:

   =======================  ===========================================
   ``model``                ``R(x) ~ tanh(x)/x``
   =======================  ===========================================
   ``"airy"``               ``tanh(x)/x`` (exact, reference)
   ``"boussinesq"``         ``1 / (1 + x^2/3)``   (classical Boussinesq)
   ``"pade22"``             ``(15 + x^2) / (15 + 6 x^2)``  ([2/2] Pade)
   ``"pade66"``             ``[6/6]`` Pade of ``tanh``
   =======================  ===========================================

   Given ``omega^2 = g k^2 h R(kh)`` the exact relation is ``omega^2 = g k tanh(kh)``.

2. A **solitary wave** solver for the KdV equation (the weakly non-linear,
   weakly dispersive member of the same family)

       eta_t + c0 eta_x + (3 c0 / 2h) eta eta_x + (c0 h^2 / 6) eta_xxx = 0,
       c0 = sqrt(g h),

   whose exact travelling solution is ``eta = a sech^2((x - c t)/w)`` with
   ``w = sqrt(4 h^3 / (3 a))`` and ``c = c0 (1 + a/(2h))``.  KdV is integrated
   pseudo-spectrally with classical RK4 and the 2/3-rule dealiasing.

Everything here is deterministic (no random numbers), which keeps the residuals
reproducible bit for bit.
"""
from __future__ import annotations

import math

import numpy as np
import torch

GRAVITY = 9.80665


# --------------------------------------------------------------------------- #
# dispersion kernels
# --------------------------------------------------------------------------- #
def _safe_sinc(x: torch.Tensor) -> torch.Tensor:
    """``sinh(x)/x``-free helper: ``tanh(x)/x`` with the value 1 at ``x = 0``."""
    return torch.where(torch.abs(x) < 1e-8, torch.ones_like(x), torch.tanh(x) / torch.where(
        torch.abs(x) < 1e-8, torch.ones_like(x), x))


def kernel_airy(x: torch.Tensor) -> torch.Tensor:
    """``tanh(x)/x`` -- the exact linear water-wave (Airy) dispersion kernel."""
    return _safe_sinc(x)


def kernel_boussinesq(x: torch.Tensor) -> torch.Tensor:
    """Classical Boussinesq / Green--Naghdi level-1 kernel ``1/(1 + x^2/3)``."""
    return 1.0 / (1.0 + x * x / 3.0)


def kernel_pade22(x: torch.Tensor) -> torch.Tensor:
    """``[2/2]`` Pade approximant of ``tanh(x)/x``."""
    return (15.0 + x * x) / (15.0 + 6.0 * x * x)


def kernel_pade66(x: torch.Tensor) -> torch.Tensor:
    """``[6/6]`` Pade approximant of ``tanh(x)/x`` (higher-order Boussinesq)."""
    x2 = x * x
    num = 135135.0 + 17325.0 * x2 + 378.0 * x2 * x2 + x2 * x2 * x2
    den = 135135.0 + 62370.0 * x2 + 3150.0 * x2 * x2 + 28.0 * x2 * x2 * x2
    return num / den


DISPERSION_MODELS = {
    "airy": kernel_airy,
    "boussinesq": kernel_boussinesq,
    "pade22": kernel_pade22,
    "pade66": kernel_pade66,
}


def dispersion_omega_sq(k, h: float = 1.0, g: float = GRAVITY,
                        model: str = "airy"):
    """Dispersive model ``omega^2 = g k^2 h R(kh)`` for wavenumber(s) ``k``."""
    if model not in DISPERSION_MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(DISPERSION_MODELS)}")
    k = torch.as_tensor(k, dtype=torch.float64)
    return g * k * k * h * DISPERSION_MODELS[model](k * h)


def airy_omega(k, h: float = 1.0, g: float = GRAVITY):
    """Exact linear water-wave frequency ``omega = sqrt(g k tanh(k h))``."""
    k = torch.as_tensor(k, dtype=torch.float64)
    return torch.sqrt(g * k * torch.tanh(k * h))


def airy_phase_speed(k, h: float = 1.0, g: float = GRAVITY):
    """Exact phase speed ``c = omega / k = sqrt(g tanh(kh)/k)``."""
    k = torch.as_tensor(k, dtype=torch.float64)
    return torch.sqrt(g * torch.tanh(k * h) / k)


def model_dispersion_error(k, h: float = 1.0, g: float = GRAVITY,
                           model: str = "airy") -> dict:
    """Relative dispersion error of ``model`` against ``omega^2 = g k tanh(kh)``."""
    k = torch.as_tensor(k, dtype=torch.float64)
    omega_exact = airy_omega(k, h, g)
    omega_model = torch.sqrt(dispersion_omega_sq(k, h, g, model))
    err_omega = torch.abs(omega_model - omega_exact) / omega_exact
    err_omega_sq = torch.abs(omega_model ** 2 - omega_exact ** 2) / omega_exact ** 2
    return {
        "model": model,
        "k": k.numpy(),
        "relative_error_omega": err_omega.numpy(),
        "relative_error_omega_sq": err_omega_sq.numpy(),
        "max_relative_error_omega": float(err_omega.max()),
        "max_relative_error_omega_sq": float(err_omega_sq.max()),
    }


def validity_limit(h: float = 1.0, g: float = GRAVITY, model: str = "boussinesq",
                   tolerance: float = 0.03, upper: float = 20.0) -> float:
    """Largest ``kh`` for which the model stays within ``tolerance`` (bisection)."""
    if model == "airy":
        return float("inf")
    grid = torch.linspace(1e-6, upper, 4001, dtype=torch.float64)
    err = model_dispersion_error(grid, h, g, model)["relative_error_omega"]
    bad = np.nonzero(err > tolerance)[0]
    if bad.size == 0:
        return float(upper)
    lo = float(grid[bad[0] - 1]) if bad[0] > 0 else 0.0
    hi = float(grid[bad[0]])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        value = float(model_dispersion_error(
            torch.tensor([mid], dtype=torch.float64), h, g, model,
        )["max_relative_error_omega"])
        if value > tolerance:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# pseudo-spectral linear wave solver
# --------------------------------------------------------------------------- #
class LinearDispersiveWave:
    """``eta_tt = -g K eta`` on a periodic domain, operator applied pseudo-spectrally.

    The (real) Fourier wavenumbers are ``k_n = 2 pi n / L`` for ``n = 0..N/2``;
    mode ``n`` is an eigenfunction of the discrete operator with the *kernel*
    frequency, so a per-mode measurement isolates (i) the time integrator if the
    measurement is compared with the kernel frequency and (ii) the kernel model
    if it is compared with the Airy frequency.
    """

    def __init__(self, length: float = 2.0 * math.pi * 5.0 / 0.1, ncells: int = 512,
                 depth: float = 1.0, g: float = GRAVITY, model: str = "airy",
                 dtype=torch.float64):
        self.length = float(length)
        self.ncells = int(ncells)
        self.depth = float(depth)
        self.g = float(g)
        self.model = model
        self.dtype = dtype
        self.dx = self.length / self.ncells
        self.x = torch.arange(self.ncells, dtype=dtype) * self.dx
        # rfft angular wavenumbers
        self.k = 2.0 * math.pi * torch.fft.rfftfreq(
            self.ncells, d=self.dx, dtype=dtype)
        self.omega_sq = dispersion_omega_sq(self.k, self.depth, self.g, model)

    # -- operator ---------------------------------------------------------- #
    def apply_operator(self, eta_hat: torch.Tensor) -> torch.Tensor:
        return -self.omega_sq * eta_hat

    def period(self, k: float) -> float:
        omega = float(torch.sqrt(dispersion_omega_sq(
            torch.tensor([k], dtype=self.dtype), self.depth, self.g, self.model)))
        return 2.0 * math.pi / omega

    # -- time integration -------------------------------------------------- #
    def simulate(self, eta0: torch.Tensor, t_end: float, dt: float,
                 sample_every: int = 1) -> dict:
        """RK4 on ``(eta, v)``; returns the sampled series at index 0 and samples."""
        eta_hat = torch.fft.rfft(eta0)
        v_hat = torch.zeros_like(eta_hat)
        steps = max(1, int(math.ceil(t_end / dt)))
        dt = t_end / steps

        def rhs(e_hat, v_hat_):
            return v_hat_, self.apply_operator(e_hat)

        series = [float(torch.fft.irfft(eta_hat, n=self.ncells)[0])]
        times = [0.0]
        for step in range(steps):
            k1e, k1v = rhs(eta_hat, v_hat)
            k2e, k2v = rhs(eta_hat + 0.5 * dt * k1e, v_hat + 0.5 * dt * k1v)
            k3e, k3v = rhs(eta_hat + 0.5 * dt * k2e, v_hat + 0.5 * dt * k2v)
            k4e, k4v = rhs(eta_hat + dt * k3e, v_hat + dt * k3v)
            eta_hat = eta_hat + (dt / 6.0) * (k1e + 2 * k2e + 2 * k3e + k4e)
            v_hat = v_hat + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
            if (step + 1) % sample_every == 0 or step + 1 == steps:
                series.append(float(torch.fft.irfft(eta_hat, n=self.ncells)[0]))
                times.append((step + 1) * dt)
        return {
            "times": np.asarray(times),
            "probe": np.asarray(series),
            "steps": steps,
            "dt": dt,
            "eta_hat": eta_hat,
            "v_hat": v_hat,
        }

    # -- measurement ------------------------------------------------------- #
    def measure_frequency(self, mode: int, periods: float = 10.0,
                          samples_per_period: int = 400,
                          cfl: float = 0.02) -> dict:
        """Simulate single mode ``mode`` and recover its frequency by LSQ fit.

        The probe signal ``eta(0, t)`` is fitted with ``A cos(w t) + B sin(w t)``;
        the frequency minimising the residual is found by golden-section search
        around the analytic value.  This measures the *scheme* (space + time),
        not the analytic formula.
        """
        k_mode = 2.0 * math.pi * mode / self.length
        omega_model = math.sqrt(float(dispersion_omega_sq(
            torch.tensor([k_mode], dtype=self.dtype), self.depth, self.g, self.model)))
        t_end = periods * 2.0 * math.pi / omega_model
        dt = cfl / omega_model
        sample_every = max(1, int(round(t_end / (periods * samples_per_period))))
        eta0 = torch.cos(k_mode * self.x)
        sim = self.simulate(eta0, t_end, dt, sample_every=sample_every)
        t = torch.as_tensor(sim["times"], dtype=self.dtype)
        y = torch.as_tensor(sim["probe"], dtype=self.dtype)

        def residual(omega: torch.Tensor) -> torch.Tensor:
            basis = torch.stack([torch.cos(omega * t), torch.sin(omega * t)], dim=1)
            coeff, *_ = torch.linalg.lstsq(basis, y[:, None])
            r = y[:, None] - basis @ coeff
            return torch.sum(r * r)

        lo = 0.98 * omega_model
        hi = 1.02 * omega_model
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, fd = float(residual(torch.tensor(c))), float(residual(torch.tensor(d)))
        for _ in range(120):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = float(residual(torch.tensor(c)))
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = float(residual(torch.tensor(d)))
        omega_hat = 0.5 * (a + b)
        omega_exact = float(airy_omega(torch.tensor([k_mode]), self.depth, self.g)[0])
        return {
            "model": self.model,
            "mode": mode,
            "kh": k_mode * self.depth,
            "k": k_mode,
            "omega_model": omega_model,
            "omega_measured": omega_hat,
            "omega_airy": omega_exact,
            "relative_error_vs_model": abs(omega_hat - omega_model) / omega_model,
            "relative_error_vs_airy": abs(omega_hat - omega_exact) / omega_exact,
            "steps": sim["steps"],
            "dt": sim["dt"],
        }


def measure_dispersion(kh_values, depth: float = 1.0, g: float = GRAVITY,
                       model: str = "airy", periods: float = 10.0,
                       length: float | None = None, ncells: int = 512) -> list[dict]:
    """Measure the numerical dispersion at a set of ``kh`` targets."""
    if length is None:
        # keep the longest target wave 1 mode above the fundamental
        k_min = min(kh_values) / depth
        length = 2.0 * math.pi / k_min
    wave = LinearDispersiveWave(length=length, ncells=ncells, depth=depth, g=g,
                                model=model)
    out = []
    for kh in kh_values:
        k_target = kh / depth
        mode = int(round(k_target * length / (2.0 * math.pi)))
        mode = max(1, min(mode, ncells // 2 - 1))
        out.append(wave.measure_frequency(mode, periods=periods))
    return out


# --------------------------------------------------------------------------- #
# KdV solitary wave
# --------------------------------------------------------------------------- #
def kdv_soliton(amplitude: float, depth: float = 1.0, g: float = GRAVITY,
                x0: float = 0.0) -> dict:
    """Exact KdV solitary wave: ``a sech^2((x - x0 - c t)/w)``.

    ``w = sqrt(4 h^3 / (3 a))``, ``c = sqrt(g h) (1 + a/(2h))``.  For reference
    the "nonlinear shallow-water" speed ``sqrt(g (h + a))`` is also returned;
    the two agree to ``O((a/h)^2)``.
    """
    c0 = math.sqrt(g * depth)
    width = math.sqrt(4.0 * depth ** 3 / (3.0 * amplitude))
    speed = c0 * (1.0 + amplitude / (2.0 * depth))
    speed_sqrt = math.sqrt(g * (depth + amplitude))

    def profile(x, t: float = 0.0):
        return amplitude / torch.cosh((x - x0 - speed * t) / width) ** 2

    return {
        "amplitude": amplitude,
        "width": width,
        "speed": speed,
        "speed_sqrt_g_h_plus_a": speed_sqrt,
        "relative_speed_difference": abs(speed - speed_sqrt) / speed_sqrt,
        "c0": c0,
        "profile": profile,
    }


class KdVSolver:
    """Periodic pseudo-spectral KdV with RK4 and 2/3-rule dealiasing."""

    def __init__(self, length: float = 60.0, ncells: int = 256,
                 depth: float = 1.0, g: float = GRAVITY,
                 dtype=torch.float64, dealias: bool = True):
        self.length = float(length)
        self.ncells = int(ncells)
        self.depth = float(depth)
        self.g = float(g)
        self.dtype = dtype
        self.c0 = math.sqrt(g * depth)
        self.dx = self.length / self.ncells
        self.k = 2.0 * math.pi * torch.fft.fftfreq(self.ncells, d=self.dx, dtype=dtype)
        self.dealias = dealias
        if dealias:  # 2/3 rule mask
            k_cut = (self.ncells // 3) * 2.0 * math.pi / self.length
            self.mask = (torch.abs(self.k) <= k_cut).to(dtype)
        else:
            self.mask = torch.ones_like(self.k)
        self.alpha = self.c0 * depth ** 2 / 6.0      # dispersion coefficient
        self.beta = 3.0 * self.c0 / (2.0 * depth)    # nonlinearity coefficient
        self.max_k = float(torch.max(torch.abs(self.k)))

    def rhs(self, eta: torch.Tensor) -> torch.Tensor:
        eta_hat = torch.fft.fft(eta)
        nonlinear = -1j * self.k * (0.5 * self.beta) * torch.fft.fft(eta * eta)
        linear = -1j * self.c0 * self.k * eta_hat + 1j * self.alpha * self.k ** 3 * eta_hat
        return torch.fft.ifft(linear + nonlinear * self.mask).real

    def stable_dt(self, safety: float = 0.4) -> float:
        """Explicit-RK4 bound from the stiffest linear mode (real-axis intercept 2.785)."""
        omega_max = self.alpha * self.max_k ** 3 + self.c0 * self.max_k
        return safety * 2.78 / omega_max

    def solve(self, eta0: torch.Tensor, t_end: float, dt: float | None = None,
              record_every: int | None = None) -> dict:
        dt = self.stable_dt() if dt is None else dt
        steps = max(1, int(math.ceil(t_end / dt)))
        dt = t_end / steps
        if record_every is None:
            record_every = max(1, steps // 20)
        eta = eta0.clone()
        times, snapshots = [0.0], [eta.clone()]
        for step in range(steps):
            k1 = self.rhs(eta)
            k2 = self.rhs(eta + 0.5 * dt * k1)
            k3 = self.rhs(eta + 0.5 * dt * k2)
            k4 = self.rhs(eta + dt * k3)
            eta = eta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if (step + 1) % record_every == 0 or step + 1 == steps:
                times.append((step + 1) * dt)
                snapshots.append(eta.clone())
        return {"eta": eta, "times": times, "snapshots": snapshots,
                "steps": steps, "dt": dt}


def _l2_error(numerical: torch.Tensor, reference: torch.Tensor, dx: float) -> dict:
    diff = numerical - reference
    absolute = float(torch.sqrt(torch.sum(diff ** 2) * dx))
    norm = float(torch.sqrt(torch.sum(reference ** 2) * dx))
    return {"absolute_l2": absolute, "reference_l2": norm,
            "relative_l2": absolute / norm if norm > 0 else float("nan"),
            "max_abs_error": float(torch.max(torch.abs(diff)))}


def run_soliton_benchmark(amplitude: float = 0.05, depth: float = 1.0,
                          g: float = GRAVITY, widths: float = 10.0,
                          length: float | None = None, ncells: int = 256,
                          x0: float = 0.0, dtype=torch.float64) -> dict:
    """Propagate the exact KdV soliton by ``widths`` soliton widths and re-measure it.

    The propagation distance is ``widths * w`` (``w`` = the ``sech^2`` width, the
    only intrinsic length of the solution), i.e. ten soliton lengths, and the
    final profile is compared with the exact solution translated by ``c T``.
    """
    soliton = kdv_soliton(amplitude, depth, g, x0=x0)
    width = soliton["width"]
    distance = widths * width
    t_end = distance / soliton["speed"]
    if length is None:
        # room for the wave to travel without meeting its own periodic image
        length = max(2.5 * distance, 12.0 * width)
    solver = KdVSolver(length=length, ncells=ncells, depth=depth, g=g, dtype=dtype)
    x = torch.arange(ncells, dtype=dtype) * solver.dx
    x0 = 0.5 * (length - distance)          # start centred so it ends mid-domain
    soliton = kdv_soliton(amplitude, depth, g, x0=x0)
    eta0 = soliton["profile"](x)
    result = solver.solve(eta0, t_end)

    def exact(xx, t: float = t_end):
        # the wave never wraps inside the reported window, so no image summation
        return soliton["profile"](xx, t)

    error = _l2_error(result["eta"], exact(x), solver.dx)
    # ---- propagation speed measured from the discrete peak (parabolic fit) --
    idx = int(torch.argmax(result["eta"]))
    lo = (idx - 1) % ncells
    hi = (idx + 1) % ncells
    y_lo, y_mid, y_hi = (float(result["eta"][lo]), float(result["eta"][idx]),
                         float(result["eta"][hi]))
    denom = y_lo - 2.0 * y_mid + y_hi
    shift = 0.5 * (y_lo - y_hi) / denom if abs(denom) > 1e-300 else 0.0
    x_peak = float(x[idx]) + shift * solver.dx
    speed_measured = (x_peak - x0) / t_end

    peak_numerical = float(result["eta"].max())
    peak_exact = float(exact(x).max())
    mass_num = float(result["eta"].sum() * solver.dx)
    mass_exact = float(exact(x).sum() * solver.dx)
    return {
        "amplitude": amplitude,
        "amplitude_ratio": amplitude / depth,
        "depth": depth,
        "g": g,
        "width": width,
        "speed": soliton["speed"],
        "speed_sqrt_g_h_plus_a": soliton["speed_sqrt_g_h_plus_a"],
        "relative_speed_difference": soliton["relative_speed_difference"],
        "speed_measured": speed_measured,
        "x_start": x0,
        "x_peak_end": x_peak,
        "relative_error_speed_measured_vs_kdv":
            abs(speed_measured - soliton["speed"]) / soliton["speed"],
        "relative_error_speed_measured_vs_sqrt":
            abs(speed_measured - soliton["speed_sqrt_g_h_plus_a"]) / soliton["speed_sqrt_g_h_plus_a"],
        "distance": distance,
        "end_time": t_end,
        "length": length,
        "ncells": ncells,
        "dx": solver.dx,
        "dt": result["dt"],
        "steps": result["steps"],
        "relative_l2": error["relative_l2"],
        "absolute_l2": error["absolute_l2"],
        "max_abs_error": error["max_abs_error"],
        "peak_numerical": peak_numerical,
        "peak_exact": peak_exact,
        "peak_retention": peak_numerical / peak_exact,
        "mass_numerical": mass_num,
        "mass_exact": mass_exact,
        "mass_drift": abs(mass_num - mass_exact) / max(abs(mass_exact), 1e-30),
        "x": x,
        "eta_final": result["eta"],
        "eta_exact": exact(x),
        "eta_initial": eta0,
    }