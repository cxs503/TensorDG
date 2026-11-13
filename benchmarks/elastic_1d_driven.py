"""Boundary-driven (inhomogeneous) 1-D elastic bar: exact d'Alembert benchmark.

Motivation
----------
Every elastic case shipped so far in this repository excites the bar through its
*initial data* and then lets it ring against homogeneous ends
(``periodic`` / ``fixed-free`` / ``free-fixed``): ``CASE-DG-DALEMBERT-ELASTIC-1D``
(m16246, L4), its reflection/image-series follow-up, the layered-impedance case
and the seam cross-check all share that assumption.  A wave code that can only
be released is half a wave code -- it can never be *sourced*.  This driver
grades the missing half: the left end prescribes the velocity

    v(0, t) = g(t)          (a smooth Gaussian pulse in time),

and the entering characteristic is fixed accordingly, ``w+_ext = -2 rho c g(t)``,
while the outgoing characteristic ``w-`` is taken from the interior trace, i.e.
the source is *absorbing*: a returning wave leaves the domain instead of being
reflected.  The right end is free (``sigma = 0``) or fixed (``v = 0``).

Exact reference (no physical dispersion, closed form)
-----------------------------------------------------
A right-going simple wave of the velocity--stress system has ``sigma = -Z v``
with ``Z = rho c``.  Adding the single image reflected at ``x = L`` gives, for
``g`` supported in ``t > 0`` and initial rest,

    v(x,t)     = g(t - x/c) + s g(t + (x-2L)/c)
    sigma(x,t) = -Z [ g(t - x/c) - s g(t + (x-2L)/c) ]

with ``s = +1`` (free end: ``sigma = 0``, velocity doubles) or ``s = -1``
(fixed end: ``v = 0``, stress doubles).  The formula stays exact for all times
because the reflected packet is absorbed at the source.

Energy / work identity (independent of the waveform comparison)
--------------------------------------------------------------
The energy density ``e = (rho v^2 + sigma^2/E)/2`` obeys ``e_t = (sigma v)_x``;
at the driven end the boundary power is ``-sigma(0,t) v(0,t) = Z g(t)^2``, so
while nothing has left the bar

    E(t) = Z \\int_0^t g(tau)^2 dtau      (``driven_injected_work``).

``E(t)/W(t) - 1`` is graded as a separate anchor: it does not depend on the
image formula at all, only on the discrete boundary flux and the interior
conservation.

Grading metric (energy norm)
----------------------------
``||v - v_ex||^2 + ||(sigma - sigma_ex)/(rho c)||^2`` normalised by its exact
counterpart -- the same norm as ``benchmarks/elastic_1d_reflections.py``, used
because the exact stress passes through zero at the free end.

Outputs (``--output-dir``)
--------------------------
``elastic_1d_driven.json`` (sweeps + anchors + fault injection),
``benchmark_ref_vs_got.json`` (row-wise verdicts),
``driven_profiles.csv`` (wall/field samples),
``driven_energy_history.json`` (E(t) vs W(t) curve),
``elastic_1d_driven.png``, ``run_manifest.json`` (sha256 of every artefact).

Usage::

    python -m benchmarks.elastic_1d_driven --output-dir validation/elastic_1d_driven
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.elastic_1d as e1
from tensordg.elastic_1d import (
    DRIVEN_BC,
    driven_injected_work,
    driven_simple_wave,
    elastic_energy,
    elastic_wave_speed,
    gaussian_drive,
    relative_error,
    solve_elastic_1d,
    stress_profile,
)

ROOT = Path(__file__).resolve().parent.parent

#: Anchor tolerance of the case under test (feature_ledger acceptance, <2%).
REL_TOL_PCT = 2.0

#: Physical parameters -- fully non-dimensional, ``c = sqrt(E/rho) = 1``.
RHO = 1.0
YOUNG = 1.0
LENGTH = 1.0
CFL = 0.5
DRIVE_CENTER = 0.35
DRIVE_WIDTH = 0.05
DRIVE_AMPLITUDE = 1.0

#: Scenario windows.  The source pulse spans t in [0.2, 0.5]; it reaches x = L
#: at t = 1.2, the reflected packet returns to x = 0 at t = 2.35.
SCENARIOS = {
    "single-wave": {"final_time": 1.00, "bc": "driven-free", "end": "free",
                    "note": "no reflection yet: pure right-going simple wave"},
    "free-end": {"final_time": 1.70, "bc": "driven-free", "end": "free",
                 "note": "one reflection at sigma=0: velocity doubles"},
    "fixed-end": {"final_time": 1.70, "bc": "driven-fixed", "end": "fixed",
                  "note": "one reflection at v=0: stress doubles"},
    "absorption": {"final_time": 2.60, "bc": "driven-free", "end": "free",
                   "note": "reflected packet leaves through the source: bar back at rest"},
}


def drive(t):
    """Boundary velocity history ``v(0,t) = g(t)`` (smooth Gaussian pulse)."""
    return gaussian_drive(t, DRIVE_CENTER, DRIVE_WIDTH, DRIVE_AMPLITUDE)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def energy_norm_error(x, v, s, v_ex, s_ex, rho=RHO, young=YOUNG):
    """Relative error in the energy norm (robust where the exact stress is 0)."""
    c = elastic_wave_speed(rho, young)
    num = torch.sum((v - v_ex) ** 2 + ((s - s_ex) / (rho * c)) ** 2)
    den = torch.sum(v_ex ** 2 + (s_ex / (rho * c)) ** 2)
    if float(den) <= 0.0:
        raise ValueError("exact field has zero energy norm")
    return math.sqrt(float(num) / float(den))


def run_scenario(name, ncells, degree, cfl=CFL, time_integrator="ssp_rk3"):
    """Solve one scenario and return ``(x, v, sigma, coeff, centers, meta)``."""
    sc = SCENARIOS[name]
    t0 = time.time()
    centers, coeff, meta = solve_elastic_1d(
        ncells=ncells, degree=degree, final_time=sc["final_time"],
        length=LENGTH, rho=RHO, young=YOUNG, cfl=cfl, bc=sc["bc"],
        drive=drive, time_integrator=time_integrator)
    x, v, s = stress_profile(centers, coeff, nsample=max(degree + 2, 4))
    return x, v, s, coeff, centers, dict(meta, wall_seconds=time.time() - t0)


def exact(name, x, final_time=None):
    sc = SCENARIOS[name]
    t = sc["final_time"] if final_time is None else final_time
    return driven_simple_wave(x, t, drive, length=LENGTH, rho=RHO, young=YOUNG,
                              end=sc["end"])


def wall_values(x, v, s, nsample_expected=None):
    """Trace values at x = 0 (left) and x = L (right)."""
    return {"x_first": float(x[0]), "x_last": float(x[-1]),
            "v_left": float(v[0]), "sigma_left": float(s[0]),
            "v_right": float(v[-1]), "sigma_right": float(s[-1])}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:  # pragma: no cover - not a git checkout on the cluster
        return "unknown"


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------
def convergence_sweep(cells, degrees):
    """Grid/degree sweep per scenario: energy-norm error + observed order."""
    out = {}
    for name in SCENARIOS:
        rows = []
        for degree in degrees:
            prev = None
            for ncells in cells:
                x, v, s, coeff, centers, meta = run_scenario(name, ncells, degree)
                v_ex, s_ex = exact(name, x)
                err = energy_norm_error(x, v, s, v_ex, s_ex)
                eoc = (None if prev is None
                       else math.log(prev[1] / err) / math.log(ncells / prev[0]))
                rows.append({
                    "ncells": ncells, "degree": degree,
                    "h": LENGTH / ncells, "steps": meta["steps"],
                    "dt": meta["dt"],
                    "err_energy_norm_pct": 100.0 * err,
                    "rel_l2_v_pct": 100.0 * relative_error(x, v, v_ex),
                    "rel_l2_sigma_pct": 100.0 * relative_error(x, s, s_ex),
                    "eoc": eoc,
                    "energy": elastic_energy(centers, coeff, RHO, YOUNG),
                    "wall_seconds": meta["wall_seconds"],
                })
                prev = (ncells, err)
        out[name] = rows
    return out


def energy_history(ncells=100, degree=2, nt=41):
    """``E(t)`` from independent runs vs the closed-form injected work ``W(t)``."""
    c = elastic_wave_speed(RHO, YOUNG)
    t_end = LENGTH / c + DRIVE_CENTER + 4.0 * DRIVE_WIDTH  # packet fully inside
    times = np.linspace(0.0, t_end, nt)
    rows = []
    for t in times:
        centers, coeff, meta = solve_elastic_1d(
            ncells=ncells, degree=degree, final_time=float(t), length=LENGTH,
            rho=RHO, young=YOUNG, cfl=CFL, bc="driven-free", drive=drive)
        e_num = elastic_energy(centers, coeff, RHO, YOUNG)
        w_ex = driven_injected_work(float(t), drive, RHO, YOUNG)
        rows.append({"t": float(t), "energy": e_num, "work_exact": w_ex,
                     "rel_error_pct": (100.0 * (e_num - w_ex) / w_ex) if w_ex > 0 else None})
    return {"ncells": ncells, "degree": degree, "rows": rows}


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------
def anchor(name, ref, got, tol_pct, kind="rel"):
    """Build one graded row.  ``ref == 0`` means an absolute (percent) bound."""
    if kind == "gt":                      # gate-must-have-teeth rows
        return {"anchor": name, "ref": ref, "got": got, "tol_pct": tol_pct,
                "pass": bool(abs(got) > tol_pct), "kind": "must exceed"}
    if ref == 0.0:
        err_pct = abs(got)
    else:
        err_pct = 100.0 * abs(got - ref) / abs(ref)
    return {"anchor": name, "ref": ref, "got": got, "err_pct": err_pct,
            "tol_pct": tol_pct, "pass": bool(err_pct <= tol_pct), "kind": "rel"}


def build_anchors(conv, energy_hist):
    """Turn the sweeps into the ref-vs-got verdict table."""
    rows = []
    c = elastic_wave_speed(RHO, YOUNG)
    Z = RHO * c

    # 1-2. waveform at T = 1.0, finest grid: the exact solution is a simple wave
    fine = [r for r in conv["single-wave"] if r["ncells"] == max(
        r2["ncells"] for r2 in conv["single-wave"]) and r["degree"] == max(
            r2["degree"] for r2 in conv["single-wave"])][0]
    rows.append(anchor("single-wave  energy-norm error (p=3, N=400)", 0.0,
                       fine["err_energy_norm_pct"], REL_TOL_PCT))
    rows.append(anchor("single-wave  rel L2 error of v", 0.0,
                       fine["rel_l2_v_pct"], REL_TOL_PCT))
    rows.append(anchor("single-wave  rel L2 error of sigma", 0.0,
                       fine["rel_l2_sigma_pct"], REL_TOL_PCT))

    # 3. work <-> energy identity (independent of the image formula)
    last = [r for r in energy_hist["rows"] if r["work_exact"] > 0][-1]
    rows.append(anchor("work/energy identity  E(t)/W(t)-1", 0.0,
                       abs(last["rel_error_pct"]), REL_TOL_PCT))

    # 4-7. wall values at the instant the pulse centre sits on x = L
    t_wall = LENGTH / c + DRIVE_CENTER
    g_wall = drive(DRIVE_CENTER)
    for name, keys in (("free-end", ("v_right", "sigma_right")),
                       ("fixed-end", ("v_right", "sigma_right"))):
        x, v, s, coeff, centers, meta = run_scenario(name, 400, 3)
        # re-run to the exact wall instant
        centers, coeff, meta = solve_elastic_1d(
            ncells=400, degree=3, final_time=t_wall, length=LENGTH, rho=RHO,
            young=YOUNG, cfl=CFL, bc=SCENARIOS[name]["bc"], drive=drive)
        x, v, s = stress_profile(centers, coeff, nsample=6)
        wall = wall_values(x, v, s)
        if name == "free-end":
            rows.append(anchor("free end  v(L,t)/(2 g)", 2.0 * g_wall,
                               wall["v_right"], REL_TOL_PCT))
            rows.append(anchor("free end  sigma(L,t)/(Z g)  (must vanish)", 0.0,
                               100.0 * wall["sigma_right"] / (Z * g_wall), REL_TOL_PCT))
        else:
            rows.append(anchor("fixed end  sigma(L,t)/(-2 Z g)", -2.0 * Z * g_wall,
                               wall["sigma_right"], REL_TOL_PCT))
            rows.append(anchor("fixed end  v(L,t)/(g)  (must vanish)", 0.0,
                               100.0 * wall["v_right"] / g_wall, REL_TOL_PCT))

    # 8. absorption: after the reflected packet leaves, the bar must be at rest
    x, v, s, coeff, centers, meta = run_scenario("absorption", 200, 3)
    e_res = elastic_energy(centers, coeff, RHO, YOUNG)
    w_in = driven_injected_work(SCENARIOS["absorption"]["final_time"], drive,
                               RHO, YOUNG)
    rows.append(anchor("absorbing source  residual energy / injected work", 0.0,
                       100.0 * e_res / w_in, REL_TOL_PCT))

    # 9. observed order of accuracy for the highest degree
    top = [r for r in conv["single-wave"] if r["degree"] == max(
        r2["degree"] for r2 in conv["single-wave"]) and r["eoc"] is not None]
    rows.append(anchor(f"observed order, degree {top[-1]['degree']} "
                       f"(theory p+1)", float(top[-1]["degree"] + 1),
                       top[-1]["eoc"], 5.0))
    return rows


# ---------------------------------------------------------------------------
# fault injection: the source sign must matter
# ---------------------------------------------------------------------------
def fault_injection_wrong_sign(ncells=200, degree=2):
    """Emit the source with the wrong characteristic sign; the gate must catch it.

    A right-going simple wave has ``sigma = -Z v``, hence ``w+ = -2 Z g``.  If a
    future edit dropped the sign (``w+_ext = +2 Z g``) the boundary would emit a
    wave of the opposite stress polarity; the waveform anchor must then blow up.
    This is a monkeypatched copy of the production boundary flux -- the solver
    itself is untouched.
    """
    orig = e1._boundary_flux
    c = elastic_wave_speed(RHO, YOUNG)
    Z = RHO * c

    def broken(v_int, s_int, side, rho, young, bc, drive=None, t=0.0):
        if bc in DRIVEN_BC and side == "left":
            w_minus = s_int + rho * c * v_int
            w_plus_ext = +2.0 * rho * c * e1._drive_value(drive, t)  # <- flipped
            sigma_star = 0.5 * (w_minus + w_plus_ext)
            v_star = (w_minus - w_plus_ext) / (2.0 * rho * c)
            return torch.stack((-sigma_star / rho, -young * v_star), dim=-1)
        return orig(v_int, s_int, side, rho, young, bc, drive=drive, t=t)

    e1._boundary_flux = broken
    try:
        x, v, s, coeff, centers, meta = run_scenario("single-wave", ncells, degree)
        v_ex, s_ex = exact("single-wave", x)
        err = 100.0 * energy_norm_error(x, v, s, v_ex, s_ex)
        e_num = elastic_energy(centers, coeff, RHO, YOUNG)
    finally:
        e1._boundary_flux = orig
    w_in = driven_injected_work(SCENARIOS["single-wave"]["final_time"], drive,
                               RHO, YOUNG)
    return {"ncells": ncells, "degree": degree,
            "err_energy_norm_pct": err,
            "energy": e_num, "work_exact": w_in,
            "energy_ratio": e_num / w_in,
            "note": "w+_ext sign flipped: error must be far above the 2% gate"}


def fault_injection_missing_drive():
    """A driven bc without a source must raise, not silently free-end."""
    try:
        solve_elastic_1d(ncells=8, degree=1, final_time=0.1, bc="driven-free")
    except ValueError as exc:
        return {"raised": True, "message": str(exc)}
    return {"raised": False, "message": "no error raised -- vacuum gate"}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figure(outdir: Path, conv, energy_hist, fault):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    # (a) single-wave profiles
    ax = axes[0][0]
    x, v, s, coeff, centers, meta = run_scenario("single-wave", 200, 3)
    v_ex, s_ex = exact("single-wave", x)
    ax.plot(x, v, lw=1.6, label=r"$v$ (DG p=3, N=200)")
    ax.plot(x, v_ex, "k--", lw=0.9, label=r"$v$ exact")
    ax.plot(x, s, lw=1.6, label=r"$\sigma$ (DG)")
    ax.plot(x, s_ex, "r:", lw=0.9, label=r"$\sigma$ exact")
    ax.set_xlabel("x"); ax.set_ylabel("field")
    ax.set_title(r"(a) driven simple wave, $t=1.0$ ($\sigma=-Zv$)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (b) reflections: free vs fixed
    ax = axes[0][1]
    for name, style in (("free-end", "-"), ("fixed-end", "--")):
        x, v, s, coeff, centers, meta = run_scenario(name, 200, 3)
        v_ex, _ = exact(name, x)
        ax.plot(x, v, style, lw=1.4, label=f"{name}: v")
        ax.plot(x, v_ex, "k:", lw=0.8)
    ax.set_xlabel("x"); ax.set_ylabel(r"$v$")
    ax.set_title(r"(b) one reflection, $t=1.7$ (dots = exact)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (c) convergence
    ax = axes[1][0]
    for degree in sorted({r["degree"] for r in conv["single-wave"]}):
        rows = [r for r in conv["single-wave"] if r["degree"] == degree]
        h = [r["h"] for r in rows]
        errs = [r["err_energy_norm_pct"] for r in rows]
        ax.loglog(h, errs, "o-", label=f"p={degree}")
        if len(h) >= 2:
            ref = errs[-1] * (np.array(h) / h[-1]) ** (degree + 1)
            ax.loglog(h, ref, "k:", lw=0.7)
    ax.set_xlabel("h"); ax.set_ylabel("energy-norm error [%]")
    ax.set_title(r"(c) convergence, $t=1.0$ (dotted: $h^{p+1}$)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    # (d) work vs energy, and absorption
    ax = axes[1][1]
    t = [r["t"] for r in energy_hist["rows"]]
    ax.plot(t, [r["work_exact"] for r in energy_hist["rows"]], "k--", lw=1.2,
            label=r"$Z\int_0^t g^2$ (exact work)")
    ax.plot(t, [r["energy"] for r in energy_hist["rows"]], "o", ms=3,
            label=r"$E_h(t)$ (DG)")
    ax.set_xlabel("t"); ax.set_ylabel("energy")
    ax.set_title("(d) injected work vs discrete energy")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.tight_layout()
    path = outdir / "elastic_1d_driven.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="validation/elastic_1d_driven")
    ap.add_argument("--cells", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    ap.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    outdir = ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    conv = convergence_sweep(args.cells, args.degrees)
    hist = energy_history()
    anchors = build_anchors(conv, hist)
    fault_sign = fault_injection_wrong_sign()
    fault_missing = fault_injection_missing_drive()

    # fault-injection verdict: wrong sign must be detected by the 2% gate
    fault_sign["detected"] = bool(fault_sign["err_energy_norm_pct"] > REL_TOL_PCT)
    anchors.append(anchor("fault: wrong source sign is rejected (err must exceed 2%)",
                          0.0, fault_sign["err_energy_norm_pct"], REL_TOL_PCT,
                          kind="gt"))
    anchors.append({"anchor": "fault: driven bc without drive raises ValueError",
                    "pass": bool(fault_missing["raised"]), "kind": "raise"})

    # profiles for the CSV / figure
    profiles = {}
    for name in SCENARIOS:
        x, v, s, coeff, centers, meta = run_scenario(name, 200, 3)
        v_ex, s_ex = exact(name, x)
        profiles[name] = {"x": x, "v": v, "sigma": s, "v_ex": v_ex,
                          "sigma_ex": s_ex, "wall": wall_values(x, v, s)}

    fig_path = None if args.no_figure else make_figure(outdir, conv, hist, fault_sign)

    payload = {
        "case": {
            "slug": "CASE-DG-DALEMBERT-DRIVEN-END",
            "parent_anchor": "CASE-DG-DALEMBERT-ELASTIC-1D",
            "repo": "TensorDG",
            "nondimensional": True,
            "length": LENGTH, "rho": RHO, "young": YOUNG,
            "c": elastic_wave_speed(RHO, YOUNG),
            "Z": RHO * elastic_wave_speed(RHO, YOUNG),
            "cfl": CFL,
            "time_integrator": "ssp_rk3",
            "drive": {"form": "Gaussian velocity pulse in time",
                      "amplitude": DRIVE_AMPLITUDE, "center": DRIVE_CENTER,
                      "width": DRIVE_WIDTH},
            "scenarios": SCENARIOS,
            "tolerance_pct": REL_TOL_PCT,
            "git_rev": git_rev(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "convergence": conv,
        "energy_history": hist,
        "anchors": anchors,
        "fault_injection": {"wrong_sign": fault_sign,
                            "missing_drive": fault_missing},
        "profiles": {name: {k: (v.tolist() if isinstance(v, torch.Tensor) else v)
                            for k, v in p.items()}
                     for name, p in profiles.items()},
        "summary": {
            "n_anchors": len(anchors),
            "n_pass": sum(1 for a in anchors if a["pass"]),
            "all_pass": all(a["pass"] for a in anchors),
            "max_err_pct": max((a.get("err_pct", a.get("got", 0.0))
                                for a in anchors if a["kind"] != "raise"), default=0.0),
            "wall_seconds": time.time() - t_start,
        },
    }
    json_path = outdir / "elastic_1d_driven.json"
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    (outdir / "benchmark_ref_vs_got.json").write_text(
        json.dumps({"tolerance_pct": REL_TOL_PCT, "rows": anchors}, indent=2,
                   default=float))
    (outdir / "driven_energy_history.json").write_text(
        json.dumps(hist, indent=2, default=float))

    csv_path = outdir / "driven_profiles.csv"
    with csv_path.open("w") as fh:
        fh.write("scenario,x,v,sigma,v_exact,sigma_exact\n")
        for name, p in profiles.items():
            for xi, vi, si, ve, se in zip(p["x"].tolist(), p["v"].tolist(),
                                          p["sigma"].tolist(),
                                          p["v_ex"].tolist(),
                                          p["sigma_ex"].tolist()):
                fh.write(f"{name},{xi:.10g},{vi:.10g},{si:.10g},"
                         f"{ve:.10g},{se:.10g}\n")

    manifest = {
        "git_rev": git_rev(),
        "command": "python -m benchmarks.elastic_1d_driven "
                   f"--output-dir {args.output_dir}",
        "wall_seconds": time.time() - t_start,
        "files": {},
    }
    for p in sorted(outdir.iterdir()):
        if p.is_file() and p.name != "run_manifest.json":
            manifest["files"][p.name] = {"sha256": sha256(p),
                                         "bytes": p.stat().st_size}
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    s = payload["summary"]
    print(f"anchors {s['n_pass']}/{s['n_anchors']} PASS  "
          f"max_err={s['max_err_pct']:.4g}%  wall={s['wall_seconds']:.1f}s")
    for a in anchors:
        flag = "PASS" if a["pass"] else "FAIL"
        got = a.get("got", "-")
        err = a.get("err_pct")
        print(f"  [{flag}] {a['anchor']}: got={got if isinstance(got,str) else f'{got:.6g}'}"
              + (f" err={err:.4g}%" if err is not None else ""))
    return 0 if s["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())