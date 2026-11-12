"""Independent, time-domain measurement of the DG advection dispersion relation.

Why this exists (verification debt)
-----------------------------------
``tensordg/dispersion_analysis.py`` derives every published phase speed and
dissipation rate from the *eigenvalues* of the assembled operator: one analytic
idea (block symbol + branch tracking) is checked only against itself (the
``spec(S) = union spec(T)`` identity) and against the closed form at ``p = 0``.
The risky step in that pipeline is the *branch selection* -- "the physical mode
is the least-damped branch" -- which is an assumption about the dynamics, not an
identity.

This module measures the same two quantities a completely different way: it
**marches the shipped solver in time** (the production RK4 kernel
``advection_1d._rhs``, exactly the marching produced by
``advection_1d.solve_advection``) and reads the amplitude and phase of the
Fourier coefficient of the numerical solution,

    z(t) = integral_0^1 u_h(x, t) exp(-i k x) dx,      k = 2 pi m,

so that ``|z(t)| = |z(0)| exp(-sigma t)`` and ``arg z(t) = arg z(0) - omega t``
with ``c_num = omega / k``.  Nothing here reuses an eigenvalue: the measurement
is of the time series itself.

Adjudication value
------------------
The first run of this harness exposed a real defect in the published branch
tracker: the point query ``physical_branch(3, [pi])`` returned
``c_num = -1.000609`` (a right-going wave reported as left-going) while the
time-domain measurement of the same scheme gives ``+1.000609``.  At
``theta = pi`` the symbol matrix is real, so the physical pair is exactly
degenerate in distance from the tracking target and ``eigvals`` ordering decided
the sign.  :func:`tensordg.dispersion_analysis._pick_continuation` now anchors
the sign; this harness is the evidence that the anchor is correct.

Usage::

    PYTHONPATH=. python -m tools.dispersion_time_domain          # writes JSON
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from tensordg import dispersion_analysis as da
from tensordg.advection_1d import _operators, _rhs

DTYPE = torch.float64

#: Periodic mesh used for the measurement.  ``N = 16`` makes every measured
#: wavenumber an exact eigen-wavenumber of the mesh (``theta = 2 pi m / N``), so
#: the initial data is a genuine Bloch mode of the discrete operator.
NCELLS = 16
#: Wavenumbers ``m`` -> ``theta = 2 pi m / 16`` in ``(0, pi]`` (``m = 8`` is the
#: Nyquist limit, the most demanding and most defect-prone point).
MODES = (1, 2, 4, 6, 8)
DEGREES = tuple(range(0, 5))
#: Time step is ``cfl * h / (2p+1)`` (the production formula) with ``cfl`` capped
#: so that ``dt <= DT_MAX``.  RK4's amplitude error per unit time behaves like
#: ``|lambda|^5 dt^4 / 120``; at ``|lambda| ~ 50`` and ``dt = 2e-3`` that is
#: ``~4e-5``, far below every dissipation rate reported as measurable.
DT_MAX = 2.0e-3
CFL_CAP = 0.144
#: Sampling interval of ``z(t)``: short enough that the phase step ``omega * dt``
#: stays far below ``pi`` for the highest wavenumber (``omega ~ 2 pi * 8``).
SAMPLE_DT = 0.02
#: Longest integration time; longer is pointless because the amplitude has
#: already reached the round-off floor for the cases that need it.
T_MAX = 6.0
#: Fit windows are *time* based, and the fast (non-physical) branches that the
#: initial projection also excites are left to die out first: the phase is fitted
#: on the last 80 % of the run, the amplitude on the last 60 %.  Amplitude
#: samples below ``AMP_FLOOR`` are discarded (round-off of the projection).
PHASE_T_WINDOW = (0.2, 1.0)
DECAY_T_WINDOW = (0.4, 1.0)
AMP_FLOOR = 1.0e-10
MIN_SAMPLES = 8
#: A dissipation rate is only *measurable* when the amplitude decays enough over
#: the fit window for the log slope to stand out of the RK4 truncation bias
#: (``|lambda|^5 dt^4 / 120 ~ 5e-5`` per unit time here).  Fixed before looking
#: at the data; cases below the threshold are reported as not measurable rather
#: than fitted.
DECAY_MIN_SPAN = 0.2           # sigma * (t1 - t0) needed for a trustworthy slope
PHASE_MIN_RAD = 0.5            # rad of accumulated phase needed for the fit


# --------------------------------------------------------------------------- #
# measurement primitives
# --------------------------------------------------------------------------- #
def _operators_for(degree: int):
    return _operators(degree, torch.device("cpu"), DTYPE)


def _mesh(ncells: int = NCELLS) -> np.ndarray:
    return (np.arange(ncells) + 0.5) * (1.0 / ncells)


def fourier_projection_matrix(degree: int, mode: int, ncells: int = NCELLS) -> np.ndarray:
    """``E[j, k] = integral_{cell j} P_k(xi) exp(-i k_phys x) dx``.

    The complex Fourier coefficient of a modal state ``c`` is then
    ``z = sum_{j,k} c[j, k] E[j, k]`` -- one mat-vec, no re-quadrature per call.
    """
    xq, wq, phi, _volume, _pl, _pr = _operators_for(degree)
    h = 1.0 / ncells
    k_phys = 2.0 * math.pi * mode
    x = _mesh(ncells)[:, None] + 0.5 * h * xq.numpy()[None, :]
    kernel = np.exp(-1j * k_phys * x)
    return 0.5 * h * np.einsum("q,qk,jq->jk", wq.numpy(), phi.numpy(), kernel)


def project(coeff: torch.Tensor, e_matrix: np.ndarray) -> complex:
    return complex(np.sum(coeff.numpy() * e_matrix))


def initial_coefficients(degree: int, mode: int, ncells: int = NCELLS) -> torch.Tensor:
    """L2 projection of ``cos(2 pi m x)`` -- identical to ``solve_advection``."""
    xq, wq, phi, _volume, _pl, _pr = _operators_for(degree)
    h = 1.0 / ncells
    centers = (torch.arange(ncells, dtype=DTYPE) + 0.5) * h
    x_phys = centers[:, None] + 0.5 * h * xq[None, :]
    u0 = torch.cos(2.0 * math.pi * mode * x_phys)
    coeff = torch.empty((ncells, degree + 1), dtype=DTYPE)
    for k in range(degree + 1):
        coeff[:, k] = (2 * k + 1) * 0.5 * torch.sum(
            wq[None, :] * u0 * phi[:, k][None, :], dim=1)
    return coeff


def effective_cfl(degree: int, ncells: int = NCELLS) -> float:
    """Largest of the production CFL and the ``DT_MAX`` cap that is admissible."""
    h = 1.0 / ncells
    return min(CFL_CAP, DT_MAX * (2 * degree + 1) / h)


def march(degree: int, mode: int, t_end: float, ncells: int = NCELLS,
          cfl: float | None = None, sample_dt: float = SAMPLE_DT):
    """Time-march the production RK4 kernel and sample ``z(t)``.

    The step count and the resulting ``dt`` follow ``solve_advection`` exactly
    (``steps = ceil(t_end / dt_nominal)``, ``dt = t_end / steps``), which lets
    ``tests/test_dispersion_time_domain.py`` assert that this loop is the
    shipped integrator and not a look-alike.
    """
    xq, _wq, phi, volume, phi_l, phi_r = _operators_for(degree)
    h = 1.0 / ncells
    coeff = initial_coefficients(degree, mode, ncells)
    cfl = effective_cfl(degree, ncells) if cfl is None else cfl
    dt_nominal = cfl * h / (2 * degree + 1)
    steps = max(1, math.ceil(t_end / dt_nominal))
    dt = t_end / steps
    every = max(1, int(round(sample_dt / dt)))
    e_matrix = fourier_projection_matrix(degree, mode, ncells)
    times = [0.0]
    values = [project(coeff, e_matrix)]
    for step in range(1, steps + 1):
        k1 = _rhs(coeff, h, 1.0, volume, phi_l, phi_r)
        k2 = _rhs(coeff + 0.5 * dt * k1, h, 1.0, volume, phi_l, phi_r)
        k3 = _rhs(coeff + 0.5 * dt * k2, h, 1.0, volume, phi_l, phi_r)
        k4 = _rhs(coeff + dt * k3, h, 1.0, volume, phi_l, phi_r)
        coeff = coeff + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if step % every == 0 or step == steps:
            times.append(step * dt)
            values.append(project(coeff, e_matrix))
    return np.asarray(times), np.asarray(values), coeff


def _window_fit(times, series, mask, min_span):
    """Least-squares fit of ``series`` against ``t`` inside a selected window."""
    inside = np.flatnonzero(mask)
    if inside.size < MIN_SAMPLES:
        return None
    t = times[inside]
    if t[-1] - t[0] < min_span:
        return None
    slope, _intercept = np.polyfit(t, series[inside], 1)
    return float(slope), float(t[0]), float(t[-1]), int(inside.size)


def _time_window(times: np.ndarray, t_end: float, window) -> np.ndarray:
    return (times >= window[0] * t_end) & (times <= window[1] * t_end)


def measure_mode(degree: int, mode: int, ncells: int = NCELLS,
                 t_max: float = T_MAX) -> dict:
    """Measure phase speed and dissipation of one (degree, wavenumber) pair."""
    theta = 2.0 * math.pi * mode / ncells
    lam = da.physical_branch(degree, np.array([theta]))[0]
    c_pred = -lam.imag / theta
    sigma_pred = -lam.real * ncells           # /h with h = 1/ncells, per unit time
    # Integrate until the amplitude has decayed to ~1e-3, capped at t_max.
    t_end = float(min(t_max, max(0.5, math.log(1e3) / sigma_pred)) if sigma_pred > 0 else t_max)

    times, values, coeff = march(degree, mode, t_end, ncells)
    amp = np.abs(values)
    amp = amp / amp[0]
    phase = np.unwrap(np.angle(values))

    k_phys = 2.0 * math.pi * mode
    above_floor = amp >= AMP_FLOOR
    phase_fit = _window_fit(times, phase,
                           _time_window(times, t_end, PHASE_T_WINDOW) & above_floor,
                           PHASE_MIN_RAD / max(k_phys, 1e-30))
    c_meas = None if phase_fit is None else -phase_fit[0] / k_phys

    decay_fit = _window_fit(times, np.log(amp),
                            _time_window(times, t_end, DECAY_T_WINDOW) & above_floor,
                            DECAY_MIN_SPAN / max(sigma_pred, 1e-30))
    sigma_meas = None
    if decay_fit is not None:
        span = sigma_pred * (decay_fit[2] - decay_fit[1])
        sigma_meas = -decay_fit[0] if span >= DECAY_MIN_SPAN and -decay_fit[0] > 0 else None

    record = {
        "degree": degree,
        "mode": mode,
        "theta": theta,
        "t_end": t_end,
        "dt": t_end / max(1, math.ceil(t_end / (effective_cfl(degree, ncells)
                                                * (1.0 / ncells) / (2 * degree + 1)))),
        "predicted_phase_speed": float(c_pred),
        "predicted_decay_rate": float(sigma_pred),
        "measured_phase_speed": None if c_meas is None else float(c_meas),
        "measured_decay_rate": None if sigma_meas is None else float(sigma_meas),
        "phase_fit_window": None if phase_fit is None else phase_fit[1:],
        "decay_fit_window": None if decay_fit is None else decay_fit[1:],
        "final_amplitude_ratio": float(amp[-1]),
    }
    # ``c_pred`` legitimately vanishes at the p = 0 Nyquist point (the classic
    # upwind scheme has c_num = sin(theta)/theta -> 0), so a *relative* error is
    # only defined away from that zero crossing; the absolute error is always
    # recorded.
    record["rk4_decay_bias_estimate"] = float(
        abs(lam) ** 5 * (1.0 / max(1, math.ceil(t_end / (effective_cfl(degree, ncells)
                                                           * (1.0 / ncells) / (2 * degree + 1))))) ** 4 / 120.0)
    if c_meas is not None:
        record["phase_speed_abs_error"] = abs(c_meas - c_pred)
        record["phase_speed_error_vs_exact"] = abs(c_meas - 1.0)
        if abs(c_pred) >= 1e-6:
            record["phase_speed_rel_error"] = abs(c_meas - c_pred) / abs(c_pred)
    if sigma_meas is not None:
        record["decay_rel_error"] = abs(sigma_meas - sigma_pred) / sigma_pred
    if degree == 0:
        # Closed-form upwind finite-volume symbols: completely independent of
        # both the eigenvalue pipeline and the time march.
        record["p0_closed_form_phase_speed"] = math.sin(theta) / theta
        record["p0_closed_form_decay_rate"] = ncells * (1.0 - math.cos(theta))
        if c_meas is not None:
            delta = abs(c_meas - record["p0_closed_form_phase_speed"])
            record["p0_phase_speed_abs_error_vs_closed_form"] = delta
            if abs(record["p0_closed_form_phase_speed"]) >= 1e-6:
                record["p0_phase_speed_rel_error_vs_closed_form"] = delta / abs(
                    record["p0_closed_form_phase_speed"])
        if sigma_meas is not None:
            record["p0_decay_rel_error_vs_closed_form"] = (
                abs(sigma_meas - record["p0_closed_form_decay_rate"])
                / record["p0_closed_form_decay_rate"])
    _ = coeff
    return record


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # pragma: no cover - not a git checkout
        return "unknown"


def run(degrees=DEGREES, modes=MODES, ncells: int = NCELLS, t_max: float = T_MAX) -> dict:
    records = [measure_mode(degree, mode, ncells, t_max)
               for degree in degrees for mode in modes]
    phase = [r for r in records if r.get("phase_speed_rel_error") is not None]
    decay = [r for r in records if r.get("decay_rel_error") is not None]
    summary = {
        "cases": len(records),
        "phase_cases": len(phase),
        "decay_cases": len(decay),
        "max_phase_speed_rel_error": max(
            (r["phase_speed_rel_error"] for r in phase), default=None),
        "max_decay_rel_error": max(
            (r["decay_rel_error"] for r in decay), default=None),
        "max_phase_speed_abs_error": max(
            (r["phase_speed_abs_error"] for r in phase), default=None),
        "max_p0_closed_form_phase_rel_error": max(
            (r["p0_phase_speed_rel_error_vs_closed_form"] for r in records
             if "p0_phase_speed_rel_error_vs_closed_form" in r), default=None),
        "max_p0_closed_form_phase_abs_error": max(
            (r["p0_phase_speed_abs_error_vs_closed_form"] for r in records
             if "p0_phase_speed_abs_error_vs_closed_form" in r), default=None),
        "max_rk4_decay_bias_over_measured_decay": max(
            (r["rk4_decay_bias_estimate"] / r["measured_decay_rate"] for r in decay),
            default=None),
        "max_p0_closed_form_decay_rel_error": max(
            (r["p0_decay_rel_error_vs_closed_form"] for r in records
             if "p0_decay_rel_error_vs_closed_form" in r), default=None),
        "decay_not_measurable": [
            {"degree": r["degree"], "mode": r["mode"]}
            for r in records if r.get("decay_rel_error") is None],
    }
    return {
        "config": {
            "ncells": ncells,
            "degrees": list(degrees),
            "modes": list(modes),
            "theta": [2.0 * math.pi * m / ncells for m in modes],
            "integrator": "production advection_1d RK4 kernel (_rhs), dt=cfl*h/(2p+1)",
            "cfl_cap": CFL_CAP,
            "dt_max": DT_MAX,
            "sample_dt": SAMPLE_DT,
            "t_max": t_max,
            "decay_time_window": list(DECAY_T_WINDOW),
            "phase_time_window": list(PHASE_T_WINDOW),
            "decay_min_span": DECAY_MIN_SPAN,
            "operator": "tensordg.advection_1d._rhs, upwind flux, speed=1",
        },
        "summary": summary,
        "records": records,
        "provenance": {
            "git_sha": _git_sha(),
            "arch": platform.machine(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "validation" / "dispersion_time_domain"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = run()
    path = out_dir / "results.json"
    path.write_text(json.dumps(payload, indent=1))
    s = payload["summary"]
    print(f"cases={s['cases']} phase_cases={s['phase_cases']} decay_cases={s['decay_cases']}")
    print(f"max phase-speed rel error vs eigenvalue prediction: {s['max_phase_speed_rel_error']:.3e}"
          f" (abs {s['max_phase_speed_abs_error']:.3e})")
    print(f"max decay rel error vs eigenvalue prediction:      {s['max_decay_rel_error']:.3e}")
    print(f"p=0 vs closed form: phase rel {s['max_p0_closed_form_phase_rel_error']:.3e} "
          f"abs {s['max_p0_closed_form_phase_abs_error']:.3e} "
          f"decay {s['max_p0_closed_form_decay_rel_error']:.3e}")
    print(f"RK4 decay bias / measured decay (worst): "
          f"{s['max_rk4_decay_bias_over_measured_decay']:.3e}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())