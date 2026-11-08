"""Dynamical verification of the 2-D shallow-water DG solver (``tensordg.swe_2d``).

Why this exists
---------------
Until mission m6386 ``solve_swe_2d`` had **no unsteady verification at all**: the
only 2-D tests were static ones (lake at rest, constant-state residual, uniform
projection).  Those pass on a solver whose y-direction interface traces are
wrong, because a constant or y-invariant *steady* state has zero face-flux
difference.  In fact the unrestricted solver blew up in one time step on every
unsteady flow (see ``docs/swe_2d_dynamical_verification.md``).

References used here are independent of the code under test:

* **G1 Stoker wet-bed dam break** -- closed-form similarity solution of the 1-D
  shallow-water equations (``tensordg.swe_exact``).  A y-invariant 2-D run must
  reproduce it, so the 2-D mesh/operator/flux machinery is checked against an
  exact solution of the *same* PDE.
* **G2 y-invariance** -- for y-invariant data the y-face fluxes are the physical
  flux and the y-momentum must stay identically zero.  Not an accuracy claim,
  an algebraic invariant (machine zero).
* **G3 linear standing wave on a skewed wave vector** -- ``k = (2 pi m/Lx,
  2 pi n/Ly)`` with ``m=1, n=2``.  Exact solution of the *linearised* SWE:
  ``h = H + A cos(k.x) cos(w t)``, ``u = A sqrt(g/H) khat sin(k.x) sin(w t)``,
  ``w = sqrt(g H) |k|``.  For ``A/H = 1e-3`` the nonlinear solution differs from
  this reference by O(A/H) = 1e-3, two orders below the gate.  It exercises both
  axes with different wavelengths, so axis swaps and axis-local defects show up.
* **G4 limiter control** -- the TVB limiter (Cockburn & Shu 1989, tensor-product
  form from ``euler_dg_2d``) must keep the dam-break error inside the 3 % gate
  and must not create over/undershoot of its own.

Everything written to ``--out`` is derived from the measured arrays; no number
in this file or in the docs is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordg.swe_2d import solve_swe_2d
from tensordg.swe_exact import dam_break_cell_averages, stoker_star_state

GRAVITY = 9.81
DEFAULT_OUT = os.path.join("docs", "benchmarks", "results", "swe_2d_dynamics")

# ---------------------------------------------------------------------------
# thresholds (stored in the artifact next to the measured values)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "rel_l1_h": 0.03,            # 3 % gate on the dam-break depth profile
    "rel_l1_hu": 0.03,           # 3 % gate on the momentum profile
    "y_invariance": 1.0e-12,     # machine-zero invariant on y-invariant data
    "hv_coefficient": 1.0e-12,   # y-momentum must stay identically zero
    "wave_rel_l2": 0.03,         # 3 % gate against the linear standing wave
    "phase_closure": 0.03,       # 3 % gate after one exact linear period
    "order_min": 0.9,            # HLL adds diffusion: accept >= first order
    }


def _zeros(x, y):
    return torch.zeros_like(x + 0.0 * y)


# ---------------------------------------------------------------------------
# G1 / G2: y-invariant Stoker dam break in the 2-D solver
# ---------------------------------------------------------------------------
def stoker_2d(nx=100, ny=4, degree=1, final_time=0.3, limiter=None, hl=1.0,
              hr=0.1, cfl=0.3):
    x, y, coeff, zb_faces, meta = solve_swe_2d(
        lambda X, Y: torch.where(X + 0.0 * Y < 0.0,
                                 torch.full_like(X + 0.0 * Y, hl),
                                 torch.full_like(X + 0.0 * Y, hr)),
        _zeros, _zeros, _zeros,
        nx=nx, ny=ny, degree=degree, final_time=final_time,
        domain=((-5.0, 5.0), (-1.0, 1.0)), cfl=cfl, limiter=limiter)

    h = coeff[:, :, 0, 0, 0].numpy()   # (nx, ny) cell averages = mode (0,0)
    hu = coeff[:, :, 0, 0, 1].numpy()
    faces = np.linspace(-5.0, 5.0, nx + 1)
    h_ex, hu_ex = dam_break_cell_averages(faces, final_time, hl, hr,
                                          gravity=GRAVITY)
    denom_hu = np.abs(hu_ex).sum()
    star = stoker_star_state(hl, hr, GRAVITY)
    jump = hl - hr
    return {
        "nx": nx, "ny": ny, "degree": degree, "final_time": final_time,
        "limiter": limiter, "steps": meta["steps"], "dt": meta["dt"],
        "rel_l1_h": float(np.abs(h[:, 0] - h_ex).sum() / np.abs(h_ex).sum()),
        "rel_l1_hu": float(np.abs(hu[:, 0] - hu_ex).sum() / denom_hu),
        "y_spread": float(np.abs(h - h[:, :1]).max()),
        "hv_coefficient": float(np.abs(coeff[:, :, :, :, 2]).max()),
        "h_min": float(h.min()), "h_max": float(h.max()),
        "overshoot": float(max(0.0, h.max() - hl) / jump),
        "undershoot": float(max(0.0, hr - h.min()) / jump),
        "star_depth": float(star["h_star"]),
        "max_depth": float(h.max()),
        "star_depth_error": float(abs(h.max() - star["h_star"]) / star["h_star"]),
        "shock_speed": float(star["shock_speed"]),
        "h_numeric": h[:, 0], "h_exact": h_ex, "hu_numeric": hu[:, 0],
        "hu_exact": hu_ex, "x": x.numpy(),
    }


# ---------------------------------------------------------------------------
# G3: linear standing wave with a skewed wave vector, periodic domain
# ---------------------------------------------------------------------------
def standing_wave(nx=48, ny=48, degree=1, H=1.0, A=1.0e-3, m=1, n=2,
                  Lx=10.0, Ly=10.0, quarter_periods=2, cfl=0.2, limiter=None):
    k1 = 2.0 * math.pi * m / Lx
    k2 = 2.0 * math.pi * n / Ly
    kmod = math.hypot(k1, k2)
    omega = math.sqrt(GRAVITY * H) * kmod
    period = 2.0 * math.pi / omega
    final_time = quarter_periods * 0.25 * period

    def theta(X, Y):
        return k1 * X + k2 * Y

    def exact_h(X, Y, t):
        th = theta(X, Y)
        return H + A * np.cos(th) * math.cos(omega * t)

    def exact_u(X, Y, t):
        th = theta(X, Y)
        amp = A * math.sqrt(GRAVITY / H) * math.sin(omega * t)
        return amp * np.sin(th)

    kmask = A / kmod
    x, y, coeff, zb_faces, meta = solve_swe_2d(
        lambda X, Y: H + A * torch.cos(k1 * (X + 0.0 * Y) + k2 * (0.0 * X + Y)),
        lambda X, Y: _zeros(X, Y) + kmask * k1
        * torch.sin(k1 * (X + 0.0 * Y) + k2 * (0.0 * X + Y)) * 0.0,
        _zeros, _zeros,
        nx=nx, ny=ny, degree=degree, final_time=final_time,
        domain=((0.0, Lx), (0.0, Ly)), cfl=cfl, periodic=(True, True),
        limiter=limiter)

    h = coeff[:, :, 0, 0, 0].numpy()
    xc = x.numpy()
    yc = y.numpy()
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    h_ref = exact_h(X, Y, final_time)
    # the perturbation amplitude is the natural error scale
    scale = A * math.sqrt(Lx * Ly)
    err = float(np.sqrt(((h - h_ref) ** 2).sum() * (Lx / nx) * (Ly / ny)) / scale)
    h_ref0 = exact_h(X, Y, 0.0)
    closure = float(np.sqrt(((h - h_ref0) ** 2).sum() * (Lx / nx) * (Ly / ny))
                    / scale)
    return {
        "nx": nx, "ny": ny, "degree": degree, "cfl": cfl,
        "quarter_periods": quarter_periods, "final_time": final_time,
        "period": period, "omega": omega, "wavenumber": kmod,
        "steps": meta["steps"], "dt": meta["dt"], "limiter": limiter,
        "rel_l2_vs_exact": err, "phase_closure": closure,
        "h_min": float(h.min()), "h_max": float(h.max()),
        "h": h, "h_ref": h_ref, "h_ref0": h_ref0, "x": xc, "y": yc,
        "A": A, "H": H, "Lx": Lx, "Ly": Ly,
    }


# ---------------------------------------------------------------------------
def provenance():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:                                    # pragma: no cover
        commit = "unknown"
    return {
        "commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "command": " ".join([sys.executable, "-m", "benchmarks.swe_2d_verification"]),
    }


def _verdict(checks):
    return "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"


def _check(name, value, threshold, sense):
    ok = (value <= threshold) if sense == "<=" else (value >= threshold)
    return {"name": name, "value": float(value), "threshold": float(threshold),
            "sense": sense, "status": "PASS" if ok else "FAIL"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--quick", action="store_true",
                    help="cheaper grids (used for smoke runs)")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    checks, records = [], {}

    dam_cases = [(100, 1), (200, 1)] if not args.quick else [(60, 1)]
    if not args.quick:
        dam_cases.append((100, 2))
    dam = []
    for nx, degree in dam_cases:
        r = stoker_2d(nx=nx, ny=4, degree=degree)
        checks.append(_check(f"stoker rel L1(h) nx={nx} p={degree}",
                             r["rel_l1_h"], THRESHOLDS["rel_l1_h"], "<="))
        checks.append(_check(f"stoker rel L1(hu) nx={nx} p={degree}",
                             r["rel_l1_hu"], THRESHOLDS["rel_l1_hu"], "<="))
        checks.append(_check(f"y-invariance nx={nx} p={degree}",
                             r["y_spread"], THRESHOLDS["y_invariance"], "<="))
        checks.append(_check(f"hv == 0 nx={nx} p={degree}",
                             r["hv_coefficient"],
                             THRESHOLDS["hv_coefficient"], "<="))
        dam.append(r)

    lim = stoker_2d(nx=100, ny=4, degree=1, limiter="tvb")
    checks.append(_check("stoker + TVB limiter rel L1(h)",
                         lim["rel_l1_h"], THRESHOLDS["rel_l1_h"], "<="))
    checks.append(_check("stoker + TVB limiter overshoot",
                         max(lim["overshoot"], lim["undershoot"]), 1.0e-2, "<="))

    wave_cases = [(32, 32), (48, 48), (64, 64)] if not args.quick else [(32, 32)]
    waves = []
    for nx, ny in wave_cases:
        # t = T/2: the standing wave is exactly inverted -> wave-shape gate.
        # t = T  : the exact solution returns to the initial state -> the
        #          residual is the phase/amplitude error accumulated over one
        #          period (a discrete dispersion-relation check).
        r = standing_wave(nx=nx, ny=ny, quarter_periods=2)
        rp = standing_wave(nx=nx, ny=ny, quarter_periods=4)
        r["phase_closure"] = rp["phase_closure"]
        r["period_time"] = rp["final_time"]
        r["period_steps"] = rp["steps"]
        checks.append(_check(f"standing wave rel L2 nx={nx}",
                             r["rel_l2_vs_exact"], THRESHOLDS["wave_rel_l2"], "<="))
        checks.append(_check(f"period closure nx={nx}",
                             r["phase_closure"], THRESHOLDS["phase_closure"], "<="))
        waves.append(r)
    # observed order (only meaningful with >= 2 grids)
    orders = []
    if len(waves) >= 2:
        for a, b in zip(waves[:-1], waves[1:]):
            if a["rel_l2_vs_exact"] > 0 and b["rel_l2_vs_exact"] > 0:
                orders.append({
                    "nx_from": a["nx"], "nx_to": b["nx"],
                    "order": float(math.log(a["rel_l2_vs_exact"]
                                            / b["rel_l2_vs_exact"])
                                   / math.log(b["nx"] / a["nx"])),
                })
    if orders:
        checks.append(_check("observed order (coarsest pair)",
                             min(o["order"] for o in orders),
                             THRESHOLDS["order_min"], ">="))

    verdict = _verdict(checks)
    result = {
        "verdict": verdict,
        "provenance": provenance(),
        "thresholds": THRESHOLDS,
        "elapsed_seconds": time.time() - t_start,
        "dam_break": [{k: v for k, v in r.items()
                       if not isinstance(v, np.ndarray)} for r in dam],
        "dam_break_limiter": {k: v for k, v in lim.items()
                              if not isinstance(v, np.ndarray)},
        "standing_wave": [{k: v for k, v in r.items()
                           if not isinstance(v, np.ndarray)} for r in waves],
        "orders": orders,
        "checks": checks,
    }

    # ---- CSV artifacts -------------------------------------------------
    with open(os.path.join(args.out, "dam_break_profiles.csv"), "w") as fh:
        fh.write("case,cell_centre,h_numeric,h_exact,hu_numeric,hu_exact\n")
        for r in dam + [lim]:
            tag = f"nx{r['nx']}_p{r['degree']}_lim{r['limiter']}"
            centers = 0.5 * (np.linspace(-5, 5, r["nx"] + 1)[:-1]
                             + np.linspace(-5, 5, r["nx"] + 1)[1:])
            for xc, hn, he, mn, me in zip(centers, r["h_numeric"], r["h_exact"],
                                          r["hu_numeric"], r["hu_exact"]):
                fh.write(f"{tag},{xc:.6f},{hn:.12e},{he:.12e},"
                         f"{mn:.12e},{me:.12e}\n")
    with open(os.path.join(args.out, "standing_wave_convergence.csv"), "w") as fh:
        fh.write("nx,ny,rel_l2_vs_exact,period_closure,steps\n")
        for r in waves:
            fh.write(f"{r['nx']},{r['ny']},{r['rel_l2_vs_exact']:.12e},"
                     f"{r['phase_closure']:.12e},{r['steps']}\n")

    # ---- figures -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for r in dam:
        centers = 0.5 * (np.linspace(-5, 5, r["nx"] + 1)[:-1]
                         + np.linspace(-5, 5, r["nx"] + 1)[1:])
        ax.plot(centers, r["h_numeric"], "o-", ms=2.5, lw=0.9,
                label=f"DG P{r['degree']}, nx={r['nx']} "
                      f"(L1 {r['rel_l1_h']:.2%})")
    ax.plot(centers, dam[-1]["h_exact"], "k--", lw=1.4, label="exact Stoker")
    ax.set_xlabel("x [m]"); ax.set_ylabel("cell-average depth h [m]")
    ax.set_title(f"G1 y-invariant Stoker dam break, t={dam[0]['final_time']} s")
    ax.set_xlim(-3.0, 3.0); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax = axes[1]
    centers = 0.5 * (np.linspace(-5, 5, lim["nx"] + 1)[:-1]
                     + np.linspace(-5, 5, lim["nx"] + 1)[1:])
    ax.plot(centers, lim["h_numeric"], "-", lw=1.2, label="DG P1 + TVB limiter")
    ax.plot(centers, lim["h_exact"], "k--", lw=1.4, label="exact Stoker")
    ax.set_xlabel("x [m]"); ax.set_title("G4 limiter control (nx=100)")
    ax.set_xlim(-3.0, 3.0); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "g1_stoker_profiles.png"), dpi=140)
    plt.close(fig)

    ref = waves[-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    ext = [ref["x"][0], ref["x"][-1], ref["y"][0], ref["y"][-1]]
    vmax = float(np.abs(ref["h"] - ref["H"]).max())
    for ax, field, title in (
            (axes[0], ref["h"] - ref["H"], "numerical (h - H)"),
            (axes[1], ref["h_ref"] - ref["H"], "exact linear SWE (h - H)"),
            (axes[2], ref["h"] - ref["h_ref"], "difference")):
        im = ax.imshow(field.T, origin="lower", extent=ext, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{title}\nnx={ref['nx']}, t={ref['final_time']:.3f} s",
                     fontsize=8)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("G3 skew standing wave (k=(2pi/10, 4pi/10)), A/H = 1e-3",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "g3_standing_wave_fields.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ns = [w["nx"] for w in waves]
    ax.loglog(ns, [w["rel_l2_vs_exact"] for w in waves], "o-", label="rel L2 vs exact")
    ax.loglog(ns, [w["phase_closure"] for w in waves], "s--", label="after one period")
    if len(ns) > 1:
        ref_line = [waves[0]["rel_l2_vs_exact"] * (n / ns[0]) ** -1.0 for n in ns]
        ax.loglog(ns, ref_line, "k:", lw=1.0, label="slope -1 reference")
    ax.set_xlabel("cells per direction"); ax.set_ylabel("relative error")
    ax.set_title("G3 standing-wave convergence"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "g3_convergence.png"), dpi=140)
    plt.close(fig)

    result["figures"] = ["g1_stoker_profiles.png", "g3_standing_wave_fields.png",
                         "g3_convergence.png"]
    with open(os.path.join(args.out, "gates_verified.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    # ---- summary.md rendered from the JSON (no hand-typed numbers) ------
    lines = ["# 2-D SWE dynamical verification (auto-generated)", "",
             f"verdict: **{verdict}**  ",
             f"commit: `{result['provenance']['commit']}`  ",
             f"torch {result['provenance']['torch']}, "
             f"python {result['provenance']['python']}", "",
             "## G1/G2 Stoker dam break (2-D solver, y-invariant data)", "",
             "| nx | p | limiter | steps | rel L1(h) | rel L1(hu) | y-spread | "
             "h_star err |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in dam + [lim]:
        lines.append(f"| {r['nx']} | {r['degree']} | {r['limiter']} | "
                     f"{r['steps']} | {r['rel_l1_h']:.4%} | "
                     f"{r['rel_l1_hu']:.4%} | {r['y_spread']:.2e} | "
                     f"{r['star_depth_error']:.4%} |")
    lines += ["", "## G3 standing wave", "",
              "| nx | rel L2 vs exact | period closure | steps |",
              "|---:|---:|---:|---:|"]
    for r in waves:
        lines.append(f"| {r['nx']} | {r['rel_l2_vs_exact']:.4%} | "
                     f"{r['phase_closure']:.4%} | {r['steps']} |")
    if orders:
        lines += ["", "observed order: "
                  + ", ".join(f"{o['nx_from']}->{o['nx_to']}: {o['order']:.2f}"
                              for o in orders)]
    lines += ["", "## checks", "", "| check | value | threshold | status |",
              "|---|---:|---:|---|"]
    for c in checks:
        lines.append(f"| {c['name']} | {c['value']:.4e} | {c['threshold']:.4e} "
                     f"| {c['status']} |")
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"verdict={verdict} ({len(checks)} checks, "
          f"{result['elapsed_seconds']:.1f} s) -> {args.out}")
    for c in checks:
        if c["status"] != "PASS":
            print(f"  FAIL {c['name']}: {c['value']:.4e} vs {c['threshold']:.4e}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
