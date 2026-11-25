"""Boussinesq solitary-wave benchmark for the 1-D DG shallow-water solver.

Absorption benchmark (TensorDG#1001): verify that the DG(P1/P2) SWE solver
``tensordg/swe_1d.py`` propagates a small-amplitude Boussinesq solitary wave
with (i) peak-height preservation and (ii) the correct nonlinear wave speed,
both to < 3 % against the analytical Boussinesq solution.

Physics
-------
The classical Boussinesq solitary wave over a flat bed of depth ``h0``,

    eta(x, t) = h0 + a sech^2((x - x0 - c t) / w),
    w = sqrt(4 h0^3 / (3 a)),   c = sqrt(g (h0 + a)),

is an exact solution of the weakly-nonlinear, weakly-dispersive Boussinesq
equations.  The 1-D *nonlinear shallow water equations* (SWE) solved here are
the non-dispersive leading-order limit of that family, so for small amplitude
``a/h0 << 1`` and short propagation the SWE solution stays close to the
Boussinesq profile: the peak height is preserved to O((a/h0)^2) and the wave
front advances at the nonlinear speed ``c = sqrt(g(h0+a))`` (the linear speed
``c0 = sqrt(g h0)`` is recovered only in the ``a -> 0`` limit).

The benchmark therefore measures, against the analytical Boussinesq solution:

* **peak height**  ``eta_max`` vs ``h0 + a``            (shape preservation)
* **peak speed**   ``x_peak(t)/t`` vs ``c``             (nonlinear wave speed)
* **profile L2**   ``||eta_num - eta_ref||_2``          (combined shape+phase)
* **mass drift**   ``|M(t) - M(0)| / M(0)``             (conservation)

The peak position is measured with **sub-cell parabolic interpolation**
(vertex of the parabola through the argmax cell and its two neighbours), so
the reported wave speed is not quantised to the cell grid.

Initial condition (standard solitary-wave seeding for SWE solvers):

    h(x, 0) = h0 + a sech^2(x / w),
    u(x, 0) = c * (h(x,0) - h0) / h(x, 0)   (= c eta / (h0 + eta)).

Reference
---------
* Boussinesq, J. (1872). "Théorie des ondes et des remous qui se propagent
  le long d'un canal horizontal rectangulaire." *J. Math. Pures Appl.* 17, 55-108.
* Cockburn, B. & Shu, C.-W. (1998). "The Runge-Kutta Discontinuous Galerkin
  Method for Conservation Laws V." *J. Comput. Phys.* 141, 199-224.

Run
---
    python docs/benchmarks/swe_solitary_wave.py            # full benchmark
    python docs/benchmarks/swe_solitary_wave.py --quick    # main case only

Outputs (written by the script, no hand transcription)
-----------------------------------------------------
* ``artifacts/swe_solitary_wave_metrics.json``   machine-readable metrics
* ``docs/figs/swe_solitary_wave_profile.png``    final profile vs Boussinesq
* ``docs/figs/swe_solitary_wave_timeseries.png`` peak height / position vs t
* ``docs/figs/swe_solitary_wave_convergence.png`` L2 error vs resolution
* ``docs/figs/swe_solitary_wave_amplitude.png``   speed error vs a/h0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# repo root on sys.path (script lives in docs/benchmarks/)
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tensordg.swe_1d import GRAVITY, solve_swe_1d, reconstruct_fields

G = GRAVITY  # 9.81 m/s^2


# ---------------------------------------------------------------------------
# Analytical Boussinesq solitary wave
# ---------------------------------------------------------------------------

def boussinesq_params(h0: float, a: float, g: float = G) -> dict:
    """Width ``w``, nonlinear speed ``c`` and linear speed ``c0``."""
    if not h0 > 0.0 or not a > 0.0:
        raise ValueError("h0 and a must be positive")
    w = math.sqrt(4.0 * h0 ** 3 / (3.0 * a))
    c = math.sqrt(g * (h0 + a))
    c0 = math.sqrt(g * h0)
    return {"h0": h0, "a": a, "g": g, "w": w, "c": c, "c0": c0,
            "a_over_h": a / h0, "c_over_c0": c / c0}


def boussinesq_profile(x: np.ndarray, t: float, x0: float,
                       p: dict) -> np.ndarray:
    """Analytical free surface ``eta(x, t) = h0 + a sech^2((x-x0-ct)/w)``."""
    return p["h0"] + p["a"] / np.cosh((x - x0 - p["c"] * t) / p["w"]) ** 2


def _init_h(x: torch.Tensor, x0: float, p: dict) -> torch.Tensor:
    return p["h0"] + p["a"] / torch.cosh((x - x0) / p["w"]) ** 2


def _init_u(x: torch.Tensor, x0: float, p: dict) -> torch.Tensor:
    eta = p["a"] / torch.cosh((x - x0) / p["w"]) ** 2
    return p["c"] * eta / (p["h0"] + eta)


def _zb(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


# ---------------------------------------------------------------------------
# Solver wrapper
# ---------------------------------------------------------------------------

def run_swe(init_h, init_u, zb, *, ncells: int, degree: int,
            final_time: float, domain: tuple, cfl: float = 0.15,
            flux_kind: str = "hll", limiter=None, rk_order: int = 3,
            dtype=torch.float64) -> dict:
    """Run the DG SWE solver; return x, h, u, eta and metadata."""
    t0 = time.perf_counter()
    x, coeff, zb_faces, meta = solve_swe_1d(
        init_h, init_u, zb,
        ncells=ncells, degree=degree, final_time=final_time,
        domain=domain, cfl=cfl, flux_kind=flux_kind,
        limiter=limiter, rk_order=rk_order,
        device="cpu", dtype=dtype,
    )
    wall = time.perf_counter() - t0
    h_avg, u_avg, eta = reconstruct_fields(coeff, zb_faces)
    return {"x": x.numpy(), "h": h_avg.numpy(), "u": u_avg.numpy(),
            "eta": eta.numpy(), "meta": meta, "wall": wall}


def _subcell_peak(eta: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Parabolic sub-cell peak: vertex of the parabola through the argmax
    cell and its two neighbours.  Returns ``(x_peak, eta_peak)``."""
    i = int(np.argmax(eta))
    if i == 0 or i == len(eta) - 1:
        return float(x[i]), float(eta[i])
    dx = x[1] - x[0]
    y0, y1, y2 = eta[i - 1], eta[i], eta[i + 1]
    denom = y2 - 2.0 * y1 + y0
    if abs(denom) < 1e-30:
        return float(x[i]), float(y1)
    xv = x[i] - 0.5 * dx * (y2 - y0) / denom
    yv = y1 - 0.25 * (y2 - y0) ** 2 / denom
    return float(xv), float(yv)


def _peak_metrics(eta: np.ndarray, x: np.ndarray, t: float, p: dict) -> dict:
    """Peak height, sub-cell peak position and derived speed at time ``t``."""
    x_peak, eta_max = _subcell_peak(eta, x)
    ref_eta = p["h0"] + p["a"]
    return {
        "t": t,
        "eta_max": eta_max,
        "eta_max_ref": ref_eta,
        "peak_height_rel_err": abs(eta_max - ref_eta) / ref_eta,
        "x_peak": x_peak,
        "x_peak_ref": p["c"] * t,
        "speed_meas": x_peak / t if t > 0 else float("nan"),
        "speed_ref": p["c"],
        "speed_rel_err": abs(x_peak / t - p["c"]) / p["c"] if t > 0 else float("nan"),
    }


def _l2_profile(eta_n: np.ndarray, eta_r: np.ndarray, dx: float) -> dict:
    """Relative L2 error of the numerical profile vs the analytical one."""
    diff = eta_n - eta_r
    l2_abs = math.sqrt(float(np.sum(diff ** 2)) * dx)
    l2_ref = math.sqrt(float(np.sum(eta_r ** 2)) * dx)
    return {"l2_abs": l2_abs, "l2_ref": l2_ref,
            "l2_rel": l2_abs / l2_ref if l2_ref > 0 else float("nan"),
            "max_abs": float(np.max(np.abs(diff)))}


def _mass(h: np.ndarray, dx: float) -> float:
    return float(np.sum(h) * dx)


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------

def run_case(h0: float, a: float, widths: float, ncells: int, degree: int,
             domain: tuple = (-60.0, 100.0), cfl: float = 0.15,
             x0: float = 0.0) -> dict:
    """One solitary-wave propagation case; returns full metrics."""
    p = boussinesq_params(h0, a)
    t_end = widths * p["w"] / p["c"]
    dx = (domain[1] - domain[0]) / ncells

    res = run_swe(
        lambda x: _init_h(x, x0, p), lambda x: _init_u(x, x0, p), _zb,
        ncells=ncells, degree=degree, final_time=t_end, domain=domain,
        cfl=cfl, flux_kind="hll", limiter=None, rk_order=3,
    )
    eta_n = res["eta"]
    x_n = res["x"]
    # analytical reference at the same grid (translated by c*t)
    eta_r = boussinesq_profile(x_n, t_end, x0, p)

    peak = _peak_metrics(eta_n, x_n, t_end, p)
    l2 = _l2_profile(eta_n, eta_r, dx)
    mass0 = _mass(_init_h(torch.as_tensor(x_n, dtype=torch.float64), x0, p).numpy(), dx)
    mass1 = _mass(res["h"], dx)
    h_min = float(res["h"].min())

    return {
        "h0": h0, "a": a, "widths": widths, "ncells": ncells,
        "degree": degree, "domain": list(domain), "cfl": cfl,
        "dx": dx, "w": p["w"], "c": p["c"], "c0": p["c0"],
        "a_over_h": p["a_over_h"], "c_over_c0": p["c_over_c0"],
        "t_end": t_end, "steps": res["meta"]["steps"],
        "wall": round(res["wall"], 2),
        "reached_final": res["meta"]["reached_final_time"],
        "peak": peak, "l2": l2,
        "mass0": mass0, "mass1": mass1,
        "mass_rel_drift": abs(mass1 - mass0) / mass0,
        "h_min": h_min,
        "profile": {"x": x_n.tolist(), "eta_num": eta_n.tolist(),
                    "eta_ref": eta_r.tolist()},
    }


# ---------------------------------------------------------------------------
# Time series (peak tracking)
# ---------------------------------------------------------------------------

def run_timeseries(h0: float, a: float, ncells: int = 200,
                   domain: tuple = (-60.0, 100.0),
                   n_probe: int = 6) -> list[dict]:
    """Peak height / peak position vs time (``n_probe`` snapshots)."""
    p = boussinesq_params(h0, a)
    t_end = 2.0 * p["w"] / p["c"]          # 2 widths
    times = np.linspace(0.0, t_end, n_probe + 1)[1:]
    rows = []
    for t in times:
        res = run_swe(
            lambda x: _init_h(x, 0.0, p), lambda x: _init_u(x, 0.0, p), _zb,
            ncells=ncells, degree=1, final_time=float(t), domain=domain,
            cfl=0.15, flux_kind="hll", limiter=None, rk_order=3,
        )
        rows.append(_peak_metrics(res["eta"], res["x"], float(t), p))
    return rows


# ---------------------------------------------------------------------------
# Convergence / amplitude studies
# ---------------------------------------------------------------------------

def run_convergence(h0: float, a: float, widths: float = 2.0,
                    cells=(200, 400, 800), degree: int = 1) -> list[dict]:
    """L2 / peak error vs resolution at fixed propagation distance."""
    return [run_case(h0, a, widths, nc, degree) for nc in cells]


def run_amplitude_sweep(h0: float, widths: float = 2.0,
                        amps=(0.005, 0.01, 0.02, 0.05),
                        ncells: int = 400) -> list[dict]:
    """Speed / peak error vs amplitude (a/h0) at fixed resolution."""
    return [run_case(h0, a, widths, ncells, 1) for a in amps]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_profile(case: dict, path: Path) -> None:
    x = np.array(case["profile"]["x"])
    eta_n = np.array(case["profile"]["eta_num"])
    eta_r = np.array(case["profile"]["eta_ref"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, eta_n, "b-", lw=1.6, label=f"DG P{case['degree']} (N={case['ncells']})")
    ax.plot(x, eta_r, "r--", lw=1.4, label="Boussinesq exact")
    ax.axhline(case["h0"], color="k", lw=0.6, ls=":")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("free surface  $\\eta$ [m]")
    ax.set_title(f"Solitary wave profile at t={case['t_end']:.2f} s "
                 f"(a/h0={case['a_over_h']:.3f})")
    ax.legend(loc="upper right")
    ax.set_xlim(-20, 60)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_timeseries(rows: list[dict], p: dict, path: Path) -> None:
    t = np.array([r["t"] for r in rows])
    eta_max = np.array([r["eta_max"] for r in rows])
    x_peak = np.array([r["x_peak"] for r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(t, eta_max, "bo-", lw=1.4, label="DG measured")
    ax1.axhline(p["h0"] + p["a"], color="r", ls="--", lw=1.2,
                label=f"Boussinesq {p['h0']+p['a']:.4f}")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("peak $\\eta_{max}$ [m]")
    ax1.set_title("Peak height vs time")
    ax1.legend()
    ax2.plot(t, x_peak, "bo-", lw=1.4, label="DG measured")
    ax2.plot(t, p["c"] * t, "r--", lw=1.2, label=f"c·t (c={p['c']:.4f})")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("peak position $x_{peak}$ [m]")
    ax2.set_title("Peak position vs time (wave speed)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_convergence(rows: list[dict], path: Path) -> None:
    nc = np.array([r["ncells"] for r in rows])
    l2 = np.array([r["l2"]["l2_rel"] for r in rows])
    peak = np.array([r["peak"]["peak_height_rel_err"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(nc, l2, "bo-", lw=1.4, label="profile rel L2")
    ax.loglog(nc, peak, "gs-", lw=1.4, label="peak height rel err")
    ax.set_xlabel("N cells")
    ax.set_ylabel("relative error")
    ax.set_title("Solitary-wave error vs resolution (P1)")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_amplitude(rows: list[dict], path: Path) -> None:
    ah = np.array([r["a_over_h"] for r in rows])
    speed = np.array([r["peak"]["speed_rel_err"] for r in rows])
    peak = np.array([r["peak"]["peak_height_rel_err"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(ah, speed * 100, "bo-", lw=1.4, label="wave speed rel err")
    ax.semilogx(ah, peak * 100, "gs-", lw=1.4, label="peak height rel err")
    ax.axhline(3.0, color="r", ls="--", lw=1.0, label="3% gate")
    ax.set_xlabel("a / h0")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Solitary-wave error vs amplitude (P1, N=400)")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(quick: bool = False) -> dict:
    h0, a = 1.0, 0.02
    p = boussinesq_params(h0, a)
    print(f"== Boussinesq solitary wave on DG SWE ==")
    print(f"h0={h0} m, a={a} m (a/h0={p['a_over_h']:.4f}), "
          f"w={p['w']:.4f} m, c={p['c']:.4f} m/s, c0={p['c0']:.4f} m/s")

    # main case: 2 widths, P1, 400 cells
    main_case = run_case(h0, a, 2.0, 400, 1)
    print(f"[main] 2w, N=400, P1: peak_err={main_case['peak']['peak_height_rel_err']*100:.4f}%, "
          f"speed_err={main_case['peak']['speed_rel_err']*100:.4f}%, "
          f"L2={main_case['l2']['l2_rel']*100:.4f}%, "
          f"mass_drift={main_case['mass_rel_drift']:.2e}, "
          f"steps={main_case['steps']}, wall={main_case['wall']}s")

    # long-time case: 5 widths (for the report)
    long_case = run_case(h0, a, 5.0, 400, 1)
    print(f"[long] 5w, N=400, P1: peak_err={long_case['peak']['peak_height_rel_err']*100:.4f}%, "
          f"speed_err={long_case['peak']['speed_rel_err']*100:.4f}%, "
          f"L2={long_case['l2']['l2_rel']*100:.4f}%, "
          f"mass_drift={long_case['mass_rel_drift']:.2e}, "
          f"steps={long_case['steps']}, wall={long_case['wall']}s")

    # time series (peak tracking)
    ts = run_timeseries(h0, a, ncells=200, n_probe=6)
    print(f"[ts] {len(ts)} snapshots: "
          f"peak_err range {min(r['peak_height_rel_err'] for r in ts)*100:.3f}-"
          f"{max(r['peak_height_rel_err'] for r in ts)*100:.3f}%, "
          f"speed_err range {min(r['speed_rel_err'] for r in ts)*100:.3f}-"
          f"{max(r['speed_rel_err'] for r in ts)*100:.3f}%")

    # convergence
    conv = run_convergence(h0, a, 2.0, (200, 400, 800), 1)
    for r in conv:
        print(f"[conv] N={r['ncells']}: L2={r['l2']['l2_rel']*100:.4f}%, "
              f"peak_err={r['peak']['peak_height_rel_err']*100:.4f}%")

    # amplitude sweep
    amp = run_amplitude_sweep(h0, 2.0, (0.005, 0.01, 0.02, 0.05), 400)
    for r in amp:
        print(f"[amp] a/h0={r['a_over_h']:.3f}: "
              f"speed_err={r['peak']['speed_rel_err']*100:.4f}%, "
              f"peak_err={r['peak']['peak_height_rel_err']*100:.4f}%")

    # P2 comparison (main case)
    p2 = run_case(h0, a, 2.0, 400, 2, cfl=0.08)
    print(f"[p2] 2w, N=400, P2: peak_err={p2['peak']['peak_height_rel_err']*100:.4f}%, "
          f"speed_err={p2['peak']['speed_rel_err']*100:.4f}%, "
          f"L2={p2['l2']['l2_rel']*100:.4f}%, "
          f"steps={p2['steps']}, wall={p2['wall']}s")

    # figures
    figdir = _REPO / "docs" / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    _fig_profile(main_case, figdir / "swe_solitary_wave_profile.png")
    _fig_timeseries(ts, p, figdir / "swe_solitary_wave_timeseries.png")
    _fig_convergence(conv, figdir / "swe_solitary_wave_convergence.png")
    _fig_amplitude(amp, figdir / "swe_solitary_wave_amplitude.png")
    print(f"[figs] wrote {figdir}/swe_solitary_wave_*.png")

    # JSON
    artdir = _REPO / "artifacts"
    artdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "swe_solitary_wave",
        "mission": "TensorDG#1001",
        "physics": "Boussinesq solitary wave on 1-D nonlinear SWE (DG P1/P2)",
        "params": p,
        "main_case": main_case,
        "long_case": long_case,
        "timeseries": ts,
        "convergence": conv,
        "amplitude_sweep": amp,
        "p2_case": p2,
        "figures": {
            "profile": "docs/figs/swe_solitary_wave_profile.png",
            "timeseries": "docs/figs/swe_solitary_wave_timeseries.png",
            "convergence": "docs/figs/swe_solitary_wave_convergence.png",
            "amplitude": "docs/figs/swe_solitary_wave_amplitude.png",
        },
    }
    # strip bulky profile arrays from the JSON (keep them in the case dicts
    # for the report, but the JSON stays lean)
    for key in ("main_case", "long_case", "p2_case"):
        payload[key].pop("profile", None)
    for r in payload["convergence"] + payload["amplitude_sweep"]:
        r.pop("profile", None)
    jpath = artdir / "swe_solitary_wave_metrics.json"
    jpath.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[json] wrote {jpath}")

    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="main case only (skip long/ts/conv/p2)")
    args = ap.parse_args()
    if args.quick:
        h0, a = 1.0, 0.02
        p = boussinesq_params(h0, a)
        main_case = run_case(h0, a, 2.0, 400, 1)
        print(json.dumps({k: v for k, v in main_case.items()
                          if k != "profile"}, indent=2, default=float))
    else:
        main()
