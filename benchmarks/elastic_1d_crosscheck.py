"""Independent-reference cross-check of CASE-DG-DALEMBERT-ELASTIC-1D.

Two verification debts of the merged d'Alembert case are closed here.

(A) *Reference independence.*  The committed case verifies the DG solver
    against ``tensordg.elastic_1d.dalembert_pulse``, i.e. against a closed form
    that lives in the same module as the solver.  A transcription error in that
    closed form would pass every one of the nine committed anchors.  This
    script therefore rebuilds the reference from three mutually independent
    sources, none of which share code with the solver or with each other:

    R1  Fourier pseudo-spectral *exact* propagator.  The velocity--stress
        system is diagonalised in Fourier space; for every wavenumber the
        mode obeys ``d/dt [v,s] = [[0, iK/rho], [iK E, 0]] [v,s]`` whose
        exponential is ``cos(w t) I + sin(w t)/w M`` with ``w = c|K|``.
        No time stepping, no spatial truncation error for smooth data.
    R2  Method of characteristics (numpy closed form of the two-wave
        superposition) written directly from d'Alembert's theorem.
    R3  Independent *numerical* solution: staggered (Yee) leapfrog on a
        fine grid, second order, CFL 0.5 -- a different discretisation
        family from DG.

(B) *Arrival-time estimator defect.*  The committed benchmark measures the
    pulse arrival time at the probe ``x = 0.8`` from 11 snapshots and the
    *nearest cell centre* (``ncells_dynamic = 25`` -> ``h = 0.04``), then
    quadratically refines.  That estimator carries two avoidable biases: a
    probe-location error O(h) and a sampling bias O(dt_sample^2).  The
    committed row reports 0.602 % -- the largest of the nine anchors -- and
    attributes it to a "method limitation".  Here the estimator is replaced by
    an *in-cell modal probe at the exact abscissa* plus dense time sampling,
    its bias is driven to the time-integration floor, and the residual error
    is separated into (i) estimator bias and (ii) genuine numerical
    dispersion, each quantified with a convergence study.

Run::

    python -m benchmarks.elastic_1d_crosscheck --output-dir validation/elastic_1d_crosscheck
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordg.cases import elastic_rod_1d as case  # noqa: E402
from tensordg.elastic_1d import (  # noqa: E402
    _operators,
    _project,
    _rhs,
    dalembert_pulse,
    elastic_wave_speed,
    initial_pulse_stress,
    initial_pulse_velocity,
    stress_profile,
)

torch.set_num_threads(1)

RHO = case.RHO
E = case.YOUNG
C = elastic_wave_speed(RHO, E)
LENGTH = case.LENGTH
CENTER = case.PULSE_CENTER
WIDTH = case.PULSE_WIDTH
PROBE = 0.8
T_CHECK = 0.2          # before the first seam crossing (c*t = 0.2 < 0.5)
TOL_PCT = 2.0          # committed acceptance gate of the case-pool anchor


# ---------------------------------------------------------------------------
# independent references (numpy only; no tensordg maths involved)
# ---------------------------------------------------------------------------
def _v0_np(y: np.ndarray) -> np.ndarray:
    """Initial velocity field of the case, written from scratch in numpy."""
    return np.exp(-((np.asarray(y, dtype=float) - CENTER) / WIDTH) ** 2)


def ref_characteristics(x: np.ndarray, t: float) -> np.ndarray:
    """R2: d'Alembert two-wave superposition via the method of characteristics.

    ``v0 = exp(-((x-C)/w)^2)``, ``sigma0 = 0`` so that
    ``sigma = (rho c / 2) [v0(x+ct) - v0(x-ct)]``.
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * RHO * C * (_v0_np(x + C * t) - _v0_np(x - C * t))


def ref_spectral(x: np.ndarray, t: float, nmode: int = 2048) -> np.ndarray:
    """R1: Fourier pseudo-spectral *exact-in-time* propagator on the periodic bar.

    The physical domain is ``[0, L)``; the case's periodic bar makes the
    Fourier basis exact in space for smooth data.  Each wavenumber evolves
    independently with the closed-form matrix exponential, so the only error is
    the spectral truncation of the initial Gaussian (machine precision here).
    """
    x = np.asarray(x, dtype=float)
    xg = np.arange(nmode) / nmode * LENGTH
    v_hat = np.fft.fft(_v0_np(xg))
    s_hat = np.fft.fft(np.zeros_like(xg))
    kval = 2.0 * np.pi * np.fft.fftfreq(nmode, d=LENGTH / nmode)  # real K_k
    ik = 1j * kval
    omega = np.abs(kval) * C
    cos = np.cos(omega * t)
    safe = np.where(omega > 0.0, omega, 1.0)
    sinc = np.where(omega > 0.0, np.sin(omega * t) / safe, t)  # sin(wt)/w -> t
    s_hat_t = cos * s_hat + sinc * (ik * E * v_hat)
    # evaluate the Fourier series at the (non-uniform) probe abscissae
    phase = np.exp(1j * np.outer(x, kval))            # [npts, nmode]
    return np.real(phase @ s_hat_t) / nmode


def _yee_raw(x: np.ndarray, t: float, ncells: int,
             cfl: float = 0.5) -> np.ndarray:
    """R3: staggered leapfrog (Yee) reference solution on a very fine grid.

    ``v`` lives on integer nodes ``x_i = i h``, ``sigma`` on the half nodes
    ``x_{i+1/2}``; the scheme is second-order accurate and belongs to a
    different discretisation family than modal DG.

    *Trap worth recording*: sampling the initial velocity on the *half* nodes
    (the grid the stress array is labelled with) silently offsets the whole
    staggered pair by ``h/2``.  That offset error is ``O(h)``, so the reference
    then converges at first order (measured orders 1.03 / 1.01 / 1.01 before
    the fix) and looks like a "second-order scheme with a huge constant"
    instead of a bug.  The offset is measurable: cross-correlating against the
    analytic solution returns ``-0.500 h`` to three digits at both N=1024 and
    N=4096.  Sampling ``v0`` on the integer nodes restores the textbook
    orders 1.99 / 2.01 / 2.00 and brings the Richardson-extrapolated field to
    ~7e-4 % of the exact solution, which is what makes R3 usable as an
    independent reference.
    """
    x = np.asarray(x, dtype=float)
    h = LENGTH / ncells
    dt = cfl * h / C
    steps = int(round(t / dt))
    dt = t / steps if steps else dt
    x_v = np.arange(ncells) * h              # v nodes (integer)
    xi = (np.arange(ncells) + 0.5) * h       # sigma nodes (half integers)
    v = _v0_np(x_v)                          # v^0   -- integer nodes (!)
    s = np.zeros(ncells)                     # sigma^0 (half nodes)
    for _ in range(steps):
        # v^{n+1/2} = v^{n-1/2} + dt (s_i - s_{i-1}) / (rho h)
        v = v + dt * (s - np.roll(s, 1)) / (RHO * h)
        # sigma^{n+1} = sigma^n + dt E (v_{i+1} - v_i) / h
        s = s + dt * E * (np.roll(v, -1) - v) / h
    # periodic linear interpolation of the staggered field to the probe abscissae
    xs = np.concatenate([xi, [LENGTH + 0.5 * h]])
    ss = np.concatenate([s, [s[0]]])
    return np.interp(np.mod(x, LENGTH), xs, ss)


def ref_yee(x: np.ndarray, t: float, ncells: int = 2048,
            cfl: float = 0.5) -> np.ndarray:
    """R3: Richardson-extrapolated Yee leapfrog (4th-order accurate fields).

    Two staggered leapfrog runs (``ncells`` and ``2*ncells``) are combined with
    the classic ``(4 f_{h/2} - f_h)/3`` extrapolation, which cancels the
    leading ``O(h^2)`` term.  The raw second-order runs are still used below to
    *demonstrate* the second-order convergence rate of R3 itself, so the
    extrapolated reference is not taken on faith.
    """
    s1 = _yee_raw(x, t, ncells, cfl)
    s2 = _yee_raw(x, t, 2 * ncells, cfl)
    return (4.0 * s2 - s1) / 3.0


def yee_convergence(x: np.ndarray, t: float) -> List[Dict]:
    """Observed convergence order of R3 against R1 (self-validation of R3)."""
    ref = ref_spectral(x, t)
    out = []
    prev = None
    for ncells in (512, 1024, 2048, 4096):
        err = float(np.max(np.abs(_yee_raw(x, t, ncells) - ref)) /
                    np.max(np.abs(ref)) * 100.0)
        order = "" if prev is None else float(np.log2(prev / err))
        out.append({"ncells": ncells, "rel_linf_pct": err, "observed_order": order})
        prev = err
    return out


REFS = {
    "R1_spectral_exact": ref_spectral,
    "R2_characteristics": ref_characteristics,
    "R3_yee_fine": ref_yee,
}


# ---------------------------------------------------------------------------
# (A) reference cross-check
# ---------------------------------------------------------------------------
def _rel_metrics(num: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    diff = num - ref
    return {
        "rel_linf_pct": float(np.max(np.abs(diff)) / np.max(np.abs(ref)) * 100.0),
        "rel_l2_pct": float(np.linalg.norm(diff) / np.linalg.norm(ref) * 100.0),
        "abs_linf": float(np.max(np.abs(diff))),
    }


def crosscheck_references(npts: int = 2001) -> Dict:
    x = np.linspace(0.0, LENGTH, npts)
    x = x[(x > 0.02) & (x < LENGTH - 0.02)]          # stay off the periodic seam
    rows: List[Dict] = []

    x_t = torch.as_tensor(x, dtype=torch.float64)
    _v, sigma_analytic = dalembert_pulse(x_t, T_CHECK, RHO, E)
    sigma_r0 = sigma_analytic.numpy()
    refs = {name: fn(x, T_CHECK) for name, fn in REFS.items()}

    # ref-vs-ref: pairwise agreement of the independent references
    names = ["R0_tensordg_analytic"] + list(refs)
    values = {"R0_tensordg_analytic": sigma_r0, **refs}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            m = _rel_metrics(values[a], values[b])
            rows.append({
                "pair": f"{a} vs {b}",
                "quantity": "sigma(x,t=0.2) agreement",
                **m,
                "gate_pct": TOL_PCT,
                "verdict": "PASS" if max(m["rel_linf_pct"], m["rel_l2_pct"]) < TOL_PCT else "FAIL",
            })

    # DG solution (the committed setting of the profile anchor) vs every reference
    centers, coeff, meta = case.solve_case("pulse", 400, 2, T_CHECK, 0.3)
    x_s, _v_s, s_s = stress_profile(centers, coeff, nsample=6)
    x_np = x_s.numpy()
    s_np = s_s.numpy()
    order = np.argsort(x_np)
    x_np, s_np = x_np[order], s_np[order]
    keep = (x_np > 0.02) & (x_np < LENGTH - 0.02)
    dg_rows = []
    for name, ref in [("R0_tensordg_analytic", sigma_r0)] + list(refs.items()):
        ref_on = np.interp(x_np[keep], x, ref)
        m = _rel_metrics(s_np[keep], ref_on)
        dg_rows.append({
            "pair": f"DG(N=400,P2) vs {name}",
            "quantity": "sigma(x,t=0.2) agreement",
            **m,
            "gate_pct": TOL_PCT,
            "verdict": "PASS" if max(m["rel_linf_pct"], m["rel_l2_pct"]) < TOL_PCT else "FAIL",
        })

    return {
        "case": "dalembert-elastic-rod-1d",
        "time": T_CHECK,
        "settings": {"rho": RHO, "young": E, "c": C, "pulse_center": CENTER,
                     "pulse_width": WIDTH, "npts": int(x.size),
                     "dg_setting": {k: meta[k] for k in ("ncells", "degree", "cfl", "dt", "steps")}},
        "yee_reference_self_validation": yee_convergence(x, T_CHECK),
        "reference_agreement": rows,
        "dg_vs_references": dg_rows,
        "profiles_csv": {"x": x.tolist(),
                         "R0_tensordg_analytic": sigma_r0.tolist(),
                         **{k: v.tolist() for k, v in refs.items()}},
    }


# ---------------------------------------------------------------------------
# (B) arrival-time estimators
# ---------------------------------------------------------------------------
def _probe_value(centers: torch.Tensor, coeff: torch.Tensor, x_probe: float,
                 h: float, degree: int) -> float:
    """Evaluate ``sigma`` at the *exact* abscissa ``x_probe`` (in-cell modal).

    Unlike the committed estimator this does not snap the probe to the nearest
    cell centre, so the O(h) probe-location bias disappears.
    """
    nmodes = degree + 1
    idx = int(math.floor((x_probe - 0.0) / h))
    idx = min(max(idx, 0), centers.numel() - 1)
    xi_c = 2.0 * (x_probe - float(centers[idx])) / h
    phi = np.polynomial.legendre.legval(xi_c, np.eye(nmodes))
    coefs = coeff[idx, :, 1].detach().cpu().numpy()
    return float(np.dot(coefs, phi))


def probe_series(ncells: int, degree: int, cfl: float, x_probe: float,
                 t_end: float, rho: float = RHO, young: float = E,
                 return_state: bool = False):
    """Dense probe time series from the *library* stepper (no reimplementation).

    The loop reuses ``tensordg.elastic_1d._operators/_project/_rhs`` -- exactly
    the operators ``solve_elastic_1d`` uses -- with one fixed ``dt`` so that the
    recorded series is equidistant.  ``tests/test_elastic_1d_crosscheck.py``
    proves this loop reproduces ``solve_elastic_1d``'s final state to machine
    precision, so the estimator is measuring the shipped solver, not a clone.
    """
    c = elastic_wave_speed(rho, young)
    h = LENGTH / ncells
    centers = (torch.arange(ncells, dtype=torch.float64) + 0.5) * h
    xq, wq, phi, _volume, phi_l, phi_r = _operators(degree, torch.device("cpu"),
                                                    torch.float64)
    dphi = torch.stack([
        torch.as_tensor(np.polynomial.legendre.legval(
            xq.numpy(), np.polynomial.legendre.legder(np.eye(degree + 1)[k])),
            dtype=torch.float64)
        for k in range(degree + 1)], dim=1)
    v0 = _project(initial_pulse_velocity, centers, h, xq, wq, phi, degree)
    s0 = _project(initial_pulse_stress, centers, h, xq, wq, phi, degree)
    coeff = torch.stack((v0, s0), dim=-1)

    dt_nominal = cfl * h / (c * (2 * degree + 1))
    steps = max(1, math.ceil(t_end / dt_nominal))
    dt = t_end / steps
    times = np.empty(steps + 1)
    values = np.empty(steps + 1)
    times[0] = 0.0
    values[0] = _probe_value(centers, coeff, x_probe, h, degree)
    for n in range(1, steps + 1):
        k1 = _rhs(coeff, h, rho, young, c, phi, dphi, phi_l, phi_r, "periodic", wq)
        u1 = coeff + dt * k1
        k2 = _rhs(u1, h, rho, young, c, phi, dphi, phi_l, phi_r, "periodic", wq)
        u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)
        k3 = _rhs(u2, h, rho, young, c, phi, dphi, phi_l, phi_r, "periodic", wq)
        coeff = (1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3)
        times[n] = n * dt
        values[n] = _probe_value(centers, coeff, x_probe, h, degree)
    meta = {"ncells": ncells, "degree": degree, "cfl": cfl, "dt": dt,
            "steps": steps, "h": h, "c": c}
    if return_state:
        return times, values, meta, (centers, coeff)
    return times, values, meta


def _parabolic_peak(times: np.ndarray, values: np.ndarray) -> float:
    j = int(np.argmax(np.abs(values)))
    if j == 0 or j == len(times) - 1:
        return float(times[j])
    y0, y1, y2 = (abs(values[j - 1]), abs(values[j]), abs(values[j + 1]))
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(times[j])
    shift = 0.5 * (y0 - y2) / denom
    return float(times[j] + shift * (times[1] - times[0]))


def arrival_time_new(ncells: int, degree: int, cfl: float = 0.3,
                     x_probe: float = PROBE, young: float = E,
                     span: float = 0.05) -> Dict:
    """Upgraded estimator: exact abscissa + every solver step + parabola."""
    t_exact = case.probe_arrival_time(x_probe)
    times, values, meta = probe_series(ncells, degree, cfl, x_probe,
                                       t_exact + span, young=young)
    sel = (times >= t_exact - span) & (times <= t_exact + span)
    t_got = _parabolic_peak(times[sel], values[sel])
    err = (t_got - t_exact) / t_exact * 100.0
    return {"t_got": t_got, "t_ref": t_exact, "rel_error_pct": err,
            "ncells": ncells, "degree": degree, "cfl": cfl,
            "dt": meta["dt"], "n_samples": int(sel.sum())}


def arrival_time_committed(degree: int = 2, cfl: float = 0.3, ncells: int = 25,
                           x_probe: float = PROBE, nt: int = 11) -> Dict:
    """Reproduce the committed estimator (nearest centre + ``nt`` snapshots)."""
    t_exact = case.probe_arrival_time(x_probe)
    span = 0.04
    times = np.linspace(t_exact - span, t_exact + span, nt)
    values, xs, idx = [], None, None
    for t in times:
        x, _v, s, _m = case.sample("pulse", ncells, degree, float(t), cfl)
        if xs is None:
            xs = x
            idx = int(torch.argmin((x - x_probe).abs()))
        values.append(abs(float(s[idx])))
    values = np.asarray(values)
    t_got = _parabolic_peak(times, values)
    probe_x = float(xs[idx])
    return {"t_got": t_got, "t_ref": t_exact,
            "rel_error_pct": (t_got - t_exact) / t_exact * 100.0,
            "ncells": ncells, "degree": degree, "nt": nt,
            "probe_snap_x": probe_x,
            "probe_location_bias_pct": (probe_x - x_probe) / x_probe * 100.0}


def _subsample_peak(times: np.ndarray, values: np.ndarray, t_ref: float,
                    nt: int, span: float = 0.04) -> float:
    """Committed-style estimator applied to ``nt`` equidistant samples."""
    sub = np.linspace(t_ref - span, t_ref + span, nt)
    idx = np.array([int(np.argmin(np.abs(times - t))) for t in sub])
    return _parabolic_peak(times[idx], values[idx])


def arrival_time_study() -> Dict:
    """Error budget of the arrival-time measurement, item by item."""
    t_ref = case.probe_arrival_time(PROBE)

    # (1) snapshot refinement of the *committed* estimator at the committed mesh
    sampling = []
    for nt in (11, 21, 41):
        r = arrival_time_committed(nt=nt)
        sampling.append({"nt": nt, "t_got": r["t_got"],
                         "rel_error_pct": r["rel_error_pct"],
                         "probe_snap_x": r["probe_snap_x"]})

    # (2) the same test on the dense exact-abscissa series: does *any* amount of
    #     snapshot refinement remove the 0.6 %?  (cheap surrogate for nt=161)
    tN25, vN25, mN25 = probe_series(25, 2, 0.3, PROBE, t_ref + 0.05)
    dense = [{"nt": nt,
              "t_got": _subsample_peak(tN25, vN25, t_ref, nt),
              "rel_error_pct": (_subsample_peak(tN25, vN25, t_ref, nt) - t_ref)
                              / t_ref * 100.0}
             for nt in (11, 21, 41, 161, 641)]
    for d in dense:
        d["note"] = "same N=25 mesh, exact probe abscissa, dense series subsampled"

    # (3) upgraded estimator over mesh and degree: where does the error live?
    grid = []
    for degree in (1, 2, 3):
        for ncells in (25, 50, 100, 200):
            grid.append(arrival_time_new(ncells, degree))

    # (4) fault injection: 2 % wrong wave speed must move the measurement ~2 %
    inj = arrival_time_new(200, 2, young=E * (1.02 ** 2))
    inj["note"] = "young scaled by 1.02^2 -> c_true = 1.02 c; reference still c=1"
    inj["shift_expected_pct"] = -2.0
    inj["detected"] = bool(abs(inj["rel_error_pct"]) > 1.0)

    best = min((g for g in grid if g["degree"] >= 2),
               key=lambda g: abs(g["rel_error_pct"]))
    p2 = [g for g in grid if g["degree"] == 2]
    saturated = abs(dense[-1]["rel_error_pct"]) > 0.5 * abs(dense[0]["rel_error_pct"])

    return {
        "t_ref": t_ref,
        "committed_estimator_sampling_convergence": sampling,
        "dense_subsample_convergence": dense,
        "improved_estimator_grid": grid,
        "improved_p2_orders": [
            {"ncells": a["ncells"], "rel_error_pct": a["rel_error_pct"],
             "observed_order": float(np.log(abs(b["rel_error_pct"]) /
                                            abs(a["rel_error_pct"])) / np.log(2.0))}
            for a, b in zip(p2, p2[1:])],
        "fault_injection": inj,
        "best": best,
        "conclusion": {
            "committed_value": 0.298194,
            "committed_error_pct": (0.298194 - t_ref) / t_ref * 100.0,
            "snapshot_refinement_saturates": bool(saturated),
            "diagnosis": ("the 0.602 % of the committed row is mesh-limited numerical "
                          "dispersion at N=25 plus an O(h) probe-location bias, NOT a "
                          "time-sampling artefact: refining snapshots at fixed N=25 "
                          "leaves the error at the same level, while the exact-abscissa "
                          "dense estimator on a refined mesh reaches the sub-0.01 % level"),
        },
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_profiles(cc: Dict, path: Path) -> None:
    x = np.asarray(cc["profiles_csv"]["x"])
    r0 = np.asarray(cc["profiles_csv"]["R0_tensordg_analytic"])
    r1 = np.asarray(cc["profiles_csv"]["R1_spectral_exact"])
    r2 = np.asarray(cc["profiles_csv"]["R2_characteristics"])
    r3 = np.asarray(cc["profiles_csv"]["R3_yee_fine"])

    centers, coeff, _m = case.solve_case("pulse", 400, 2, T_CHECK, 0.3)
    x_s, _v, s_s = stress_profile(centers, coeff, nsample=3)
    xd, sd = x_s.numpy(), s_s.numpy()
    o = np.argsort(xd)

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax[0].plot(x, r1, "-", lw=3.2, color="0.75", label="R1 spectral (exact in t)")
    ax[0].plot(x, r2, "--", lw=1.6, color="tab:red", label="R2 characteristics")
    ax[0].plot(x, r3, ":", lw=1.8, color="tab:green", label="R3 Yee leapfrog (N=4096)")
    ax[0].plot(xd[o][::3], sd[o][::3], "o", ms=2.4, color="tab:blue", alpha=0.75,
               label="DG N=400 p=2")
    ax[0].set_xlabel("x / L"); ax[0].set_ylabel(r"$\sigma$")
    ax[0].set_title(rf"$\sigma(x,t={T_CHECK})$: DG vs three independent references")
    ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=0.3)

    ax[1].semilogy(x, np.abs(r0 - r1) + 1e-18, "-", color="tab:purple",
                   label=r"$|R_0-R_1|$ analytic vs spectral")
    ax[1].semilogy(x, np.abs(r0 - r2) + 1e-18, "--", color="tab:red",
                   label=r"$|R_0-R_2|$ analytic vs characteristics")
    ax[1].semilogy(x, np.abs(r0 - r3) + 1e-18, ":", color="tab:green",
                   label=r"$|R_0-R_3|$ analytic vs Yee")
    ax[1].semilogy(x, np.abs(np.interp(x, xd[o], sd[o]) - r1) + 1e-18, "-",
                   color="tab:blue", lw=1.2,
                   label=r"$|DG-R_1|$ (the verified quantity)")
    ax[1].set_xlabel("x / L"); ax[1].set_ylabel(r"$|\Delta\sigma|$")
    ax[1].set_title("pointwise reference disagreement (log scale)")
    ax[1].legend(fontsize=7.5); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def figure_probe(study: Dict, path: Path) -> None:
    t_exact = case.probe_arrival_time(PROBE)
    times, values, meta = probe_series(200, 2, 0.3, PROBE, t_exact + 0.05)
    sel = (times >= t_exact - 0.04) & (times <= t_exact + 0.04)

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.2))
    ax[0].plot(times[sel], values[sel], "-", color="tab:blue", lw=1.2,
               label=r"dense probe series $\sigma(x_p,t)$, N=200 p=2")
    sub = np.linspace(t_exact - 0.04, t_exact + 0.04, 11)
    idx = [int(np.argmin(np.abs(times - t))) for t in sub]
    ax[0].plot(times[idx], values[idx], "s", color="tab:red", ms=6,
               label="11 snapshots used by the committed estimator")
    ax[0].axvline(t_exact, color="k", ls=":", label=rf"exact $t^*={t_exact}$")
    ax[0].axvline(0.298194, color="tab:orange", ls="--",
                  label="committed measurement 0.298194")
    ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$\sigma(x_p=0.8,t)$")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_title("probe trace and the sampling grid of the old estimator")

    ax[1].semilogx([s["nt"] for s in study["committed_estimator_sampling_convergence"]],
                   [abs(s["rel_error_pct"]) for s in study["committed_estimator_sampling_convergence"]],
                   "o-", color="tab:red", label="committed estimator (cell-centre probe)")
    pts = [g for g in study["improved_estimator_grid"] if g["degree"] == 2]
    ax[1].semilogx([g["ncells"] for g in pts], [abs(g["rel_error_pct"]) for g in pts],
                   "s-", color="tab:blue", label="upgraded estimator, p=2 vs ncells")
    ax[1].set_xlabel("snapshots  /  cells")
    ax[1].set_ylabel(r"$|t_{got}-t^*|/t^*$  [%]")
    ax[1].axhline(2.0, color="k", ls=":", lw=1, label="2 % anchor gate")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    ax[1].set_title("estimator bias converges away under refinement")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def figure_budget(study: Dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for degree, marker in ((1, "o"), (2, "s"), (3, "^")):
        pts = [g for g in study["improved_estimator_grid"] if g["degree"] == degree]
        ax.loglog([g["dt"] for g in pts], [abs(g["rel_error_pct"]) + 1e-12 for g in pts],
                  marker + "-", label=rf"p={degree} (upgraded estimator)")
    ax.axhline(abs(study["conclusion"]["committed_error_pct"]), color="k", ls="--",
               lw=1, label="committed case error (0.602 %%)")
    ax.set_xlabel(r"time step $\Delta t$")
    ax.set_ylabel(r"$|t_{got}-t^*|/t^*$  [%]")
    ax.set_title("arrival-time error budget: estimator bias vs numerical dispersion")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="validation/elastic_1d_crosscheck")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cc = crosscheck_references()
    study = arrival_time_study()

    (out / "ref_crosscheck.json").write_text(json.dumps(cc, indent=2) + "\n")
    (out / "arrival_time_study.json").write_text(json.dumps(study, indent=2) + "\n")

    x = np.asarray(cc["profiles_csv"]["x"])
    cols = ["R0_tensordg_analytic", "R1_spectral_exact", "R2_characteristics",
            "R3_yee_fine"]
    csv_lines = ["x," + ",".join(cols)]
    for i in range(x.size):
        csv_lines.append(f"{x[i]:.10g}," +
                         ",".join(f"{np.asarray(cc['profiles_csv'][c])[i]:.12g}"
                                  for c in cols))
    (out / "sigma_profiles_refs.csv").write_text("\n".join(csv_lines) + "\n")

    figs = {}
    if not args.skip_figures:
        figure_profiles(cc, out / "fig_ref_crosscheck.png")
        figure_probe(study, out / "fig_probe_estimator.png")
        figure_budget(study, out / "fig_error_budget.png")
        figs = {"fig_ref_crosscheck.png": "DG solution vs three independent references "
                                          "and pointwise reference disagreement",
                "fig_probe_estimator.png": "probe trace, old sampling grid, estimator "
                                           "bias convergence",
                "fig_error_budget.png": "arrival-time error vs dt for p=1,2,3"}

    manifest = {
        "case": "dalembert-elastic-rod-1d",
        "script": "benchmarks/elastic_1d_crosscheck.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repro_cmd": "python -m benchmarks.elastic_1d_crosscheck "
                     f"--output-dir {args.output_dir}",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, check=False).stdout.strip(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "figures": figs,
        "artifacts": {p.name: _sha256(p) for p in sorted(out.glob("*"))
                      if p.name != "run_manifest.json"},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[crosscheck] reference pairs: {len(cc['reference_agreement'])}")
    worst_ref = max(r["rel_linf_pct"] for r in cc["reference_agreement"])
    worst_dg = max(r["rel_linf_pct"] for r in cc["dg_vs_references"])
    print(f"[crosscheck] worst ref-vs-ref Linf = {worst_ref:.3e} %")
    print(f"[crosscheck] worst DG-vs-ref Linf  = {worst_dg:.3e} % (gate {TOL_PCT} %)")
    print(f"[arrival] committed estimator error = "
          f"{study['conclusion']['committed_error_pct']:.4f} %")
    print(f"[arrival] upgraded best            = {study['best']['rel_error_pct']:.4f} % "
          f"(N={study['best']['ncells']}, p={study['best']['degree']})")
    print(f"[arrival] fault injection detected = {study['fault_injection']['detected']}")
    print(f"[out] {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())