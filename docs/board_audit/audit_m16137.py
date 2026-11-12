"""Board audit of m16137 (CASE-DG-DALEMBERT-ELASTIC-1D): independent re-derivation.

This script is the evidence engine for the maturity verdict. It deliberately
does **not** call any of the delivered verification helpers
(``dalembert_exact``, ``evaluate``, ``relative_error``, ``case_error``,
``gate_report``): only the solver under test (``solve_elastic_rod``) and its
case initial data are imported. Everything else -- the analytic reference, the
modal reconstruction, the L2 norm, the Fourier (spectral) PDE solution used to
cross-check the reference, and the energy integral -- is re-implemented here.

Checks
------
C1  gate value: relative L2 stress error of the delivered run (p=3, N=200,
    cfl=0.3) recomputed with the auditor's own reference and norm.
C2  convergence: observed L2 orders for p=1..4 recomputed from scratch.
C3  reference discrimination: the closed-form d'Alembert reference is compared
    with a Fourier spectral solution of the *same* PDE (no d'Alembert formula
    involved). If the reference were degenerate/self-referential this check
    fails.
C4  energy conservation of the DG state, recomputed with the auditor's own
    Gauss-Legendre integral.
C5  reflection wall-stress ratio (fixed end = 2.0, free end = 0) recomputed.

Usage
-----
    python docs/board_audit/audit_m16137.py --repo /tmp/audit_m16137 \
        --outdir artifacts/audit_m16137
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# --- case definition (read from the delivered module: this is the problem
# --- statement, not the verification logic) ---------------------------------
SIGMA_AMPLITUDE = 1.0e6
LENGTH = 1.0
PACKET_CENTER = 0.5
PACKET_WIDTH = 0.08
PACKET_WAVELENGTH = 0.20
TRAVEL_DOUBLE_WAVE = 0.40
E_STEEL, RHO_STEEL = 210e9, 7850.0
C_STEEL = math.sqrt(E_STEEL / RHO_STEEL)

NQ_PER_CELL = 40  # auditor's own quadrature density


# ---------------------------------------------------------------- reference --
def packet_stress_np(x):
    """Initial stress of the ``double-wave`` scenario (auditor re-typed)."""
    return SIGMA_AMPLITUDE * np.exp(-((x - PACKET_CENTER) / PACKET_WIDTH) ** 2) * np.sin(
        2.0 * math.pi * (x - PACKET_CENTER) / PACKET_WAVELENGTH)


def dalembert_np(x, t, *, c=C_STEEL, length=LENGTH):
    """Auditor's own d'Alembert solution for the periodic double-wave case.

    ``v(x,0) = 0`` so the two characteristic components are identical:
    ``sigma(x,t) = 1/2 [ s0(x-ct) + s0(x+ct) ]`` with both arguments folded
    modulo ``length``.
    """
    x = np.asarray(x, dtype=float)
    xs = np.mod(x - c * t, length)
    xp = np.mod(x + c * t, length)
    return 0.5 * (packet_stress_np(xs) + packet_stress_np(xp))


def spectral_np(x, t, *, modes=4096, c=C_STEEL, length=LENGTH):
    """Independent reference: Fourier spectral solution of the same rod system.

    ``sigma_t = E v_x``, ``v_t = sigma_x/rho`` with periodic BC. In Fourier
    space the system is ``d/dt (s,v) = [[0, ikE],[ik/rho, 0]] (s,v)`` whose exact
    propagator is ``cos(ckt)`` for ``sigma`` and ``i sin(ckt)/(rho c)``-coupling
    for ``v``. No d'Alembert superposition is used anywhere.
    """
    grid = np.linspace(0.0, length, modes, endpoint=False)
    dx = grid[1] - grid[0]
    s0 = packet_stress_np(grid)
    k = 2.0 * np.pi * np.fft.fftfreq(modes, d=dx)
    s_hat = np.fft.fft(s0) * np.cos(c * k * t)
    sigma = np.real(np.fft.ifft(s_hat))
    # evaluate at requested points by Fourier interpolation of that field
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    chunk = 2000
    for i in range(0, x.size, chunk):
        sl = slice(i, i + chunk)
        e = np.exp(1j * np.outer(x[sl], k))
        out[sl] = np.real(e @ (np.fft.fft(s0) * np.cos(c * k * t))) / modes
    return out


# --------------------------------------------------------- DG reconstruction --
def legendre_basis(xi, degree):
    cols = []
    for k in range(degree + 1):
        coeff = np.zeros(k + 1)
        coeff[-1] = 1.0
        cols.append(np.polynomial.legendre.legval(xi, coeff))
    return np.stack(cols, axis=-1)


def dg_field(coeff, h, degree, samples_per_cell=NQ_PER_CELL):
    """Auditor's own modal reconstruction: returns ``(x, sigma_h, w)``.

    Composite Gauss-Legendre quadrature over every cell, so the L2 norms below
    are computed without any delivered helper.
    """
    nq = 8
    xi1, w1 = np.polynomial.legendre.leggauss(nq)
    coeff = np.asarray(coeff, dtype=float)
    ncells = coeff.shape[0]
    # split every cell into `panels` Gauss panels for a dense quadrature
    panels = max(1, samples_per_cell // nq)
    xs, ws, sig = [], [], []
    for p in range(panels):
        a = -1.0 + 2.0 * p / panels
        b = -1.0 + 2.0 * (p + 1) / panels
        xi = 0.5 * (a + b) + 0.5 * (b - a) * xi1
        w = 0.5 * (b - a) * w1
        phi = legendre_basis(xi, degree)                  # [nq, k]
        vals = coeff[:, :, 0] @ phi.T                     # [ncells, nq]
        xs.append(xi[None, :] * 0.5 * h + (np.arange(ncells)[:, None] + 0.5) * h)
        ws.append(np.broadcast_to(w[None, :] * 0.5 * h, vals.shape))
        sig.append(vals)
    return np.concatenate(xs, axis=1).ravel(), np.concatenate(sig, axis=1).ravel(), \
        np.concatenate(ws, axis=1).ravel()


def rel_l2(coeff, h, degree, t):
    x, sigma_h, w = dg_field(coeff, h, degree)
    ref = dalembert_np(x, t)
    num = math.sqrt(float(np.sum(w * (sigma_h - ref) ** 2)))
    den = math.sqrt(float(np.sum(w * ref ** 2)))
    return num / den


def energy_audit(centers_t, coeff, h, degree, E=E_STEEL, rho=RHO_STEEL):
    """Elastic energy with the auditor's own quadrature (sigma and v)."""
    coeff = np.asarray(coeff, dtype=float)
    xi, w = np.polynomial.legendre.leggauss(2 * degree + 4)
    phi = legendre_basis(xi, degree)
    s = coeff[:, :, 0] @ phi.T
    v = coeff[:, :, 1] @ phi.T
    dens = 0.5 * rho * v ** 2 + s ** 2 / (2.0 * E)
    return float(0.5 * h * np.sum(w[None, :] * dens))


def wall_stress(coeff, degree):
    coeff = np.asarray(coeff, dtype=float)
    phi_r = np.ones(degree + 1)
    return float(coeff[-1, :, 0] @ phi_r)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/tmp/audit_m16137")
    ap.add_argument("--outdir", default="artifacts/audit_m16137")
    ap.add_argument("--quick", action="store_true", help="skip the N=200 sweeps")
    args = ap.parse_args(argv)

    sys.path.insert(0, args.repo)
    import torch  # noqa: F401  (imported after sys.path for the repo build)
    from tensordg.elastic_1d import solve_elastic_rod
    from tensordg.cases.elastic_rod_1d import (
        gaussian_stress, initial_data, material, final_time,
    )

    out = {"repo": str(Path(args.repo).resolve()), "checks": {}}
    mat = material("steel")
    c = mat["c"]
    out["material"] = {"E": mat["E"], "rho": mat["rho"], "c": c,
                       "auditor_c": C_STEEL, "c_agrees": abs(c - C_STEEL) < 1e-9}
    out["material"]["handbook_band_5100_5200_m_s"] = 5100.0 < c < 5200.0

    # ---- C3: does the closed-form reference agree with an independent PDE solve
    t_gate = TRAVEL_DOUBLE_WAVE / c
    x_probe = np.array([0.02, 0.10, 0.50, 0.90, 0.98, 0.37])
    ref_closed = dalembert_np(x_probe, t_gate)
    ref_spectral = spectral_np(x_probe, t_gate)
    scale = float(np.max(np.abs(ref_closed)))
    out["checks"]["C3_reference_vs_spectral"] = {
        "points": x_probe.tolist(),
        "dalembert_Pa": ref_closed.tolist(),
        "fourier_spectral_Pa": ref_spectral.tolist(),
        "max_abs_diff_rel_to_amplitude": float(np.max(np.abs(ref_closed - ref_spectral)) / scale),
        "verdict": "PASS" if np.max(np.abs(ref_closed - ref_spectral)) / scale < 1e-8 else "FAIL",
        "note": "Fourier propagator exp(i c k t) solves the same PDE without any "
                "d'Alembert superposition -> validates the reference independently",
    }

    # ---- C1: headline gate, p=3 N=200 cfl=0.3 (auditor recompute)
    sigma0, v0 = initial_data("double-wave", mat)
    centers, coeff, meta = solve_elastic_rod(
        200, 3, t_gate, E=mat["E"], rho=mat["rho"], length=LENGTH, cfl=0.3,
        bc="periodic", initial_sigma=sigma0, initial_velocity=v0)
    h = LENGTH / 200
    err = rel_l2(coeff.numpy(), h, 3, t_gate)
    out["checks"]["C1_gate_p3_n200"] = {
        "relative_l2_sigma_auditor": err,
        "claimed_in_delivery_results_json": 6.429376841693551e-07,
        "tolerance_case_pool": 0.02,
        "margin_vs_tolerance": 0.02 / err,
        "steps": meta["steps"], "dt": meta["dt"],
        "verdict": "PASS" if err < 0.02 else "FAIL",
        "three_percent_criterion": "PASS" if err < 0.03 else "FAIL",
    }

    # ---- C4: energy conservation, auditor quadrature
    c0, coeff0, _ = solve_elastic_rod(
        200, 3, 0.0, E=mat["E"], rho=mat["rho"], length=LENGTH,
        initial_sigma=sigma0, initial_velocity=v0)
    e0 = energy_audit(centers, coeff0.numpy(), h, 3)
    e1 = energy_audit(centers, coeff.numpy(), h, 3)
    out["checks"]["C4_energy"] = {
        "energy_initial_auditor": e0, "energy_final_auditor": e1,
        "relative_drift_auditor": abs(e1 - e0) / e0,
        "claimed_relative_drift": 5.8084806813553234e-08,
        "verdict": "PASS" if abs(e1 - e0) / e0 < 1e-6 else "FAIL",
    }

    # ---- C2: convergence p=1..4 (auditor metric)
    grids = (25, 50, 100) if args.quick else (25, 50, 100, 200)
    conv = {}
    for degree in (1, 2, 3, 4):
        rows = []
        for n in grids:
            cc, cf, mm = solve_elastic_rod(
                n, degree, t_gate, E=mat["E"], rho=mat["rho"], length=LENGTH,
                cfl=0.3, bc="periodic", initial_sigma=sigma0, initial_velocity=v0)
            rows.append({"ncells": n, "relative_l2": rel_l2(cf.numpy(), LENGTH / n, degree, t_gate),
                         "steps": mm["steps"]})
        errs = [r["relative_l2"] for r in rows]
        orders = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        conv[f"p{degree}"] = {
            "rows": rows,
            "pairwise_orders": orders,
            "observed_order": -float(np.polyfit([math.log(r["ncells"]) for r in rows],
                                                [math.log(e) for e in errs], 1)[0]),
        }
        print(f"p={degree}: {[f'{e:.3e}' for e in errs]} orders={[f'{o:.3f}' for o in orders]}",
              flush=True)
    out["checks"]["C2_convergence"] = conv
    c2 = {f"p{p}": abs(conv[f"p{p}"]["observed_order"] - (p + 1)) < 0.55 for p in (1, 2, 3, 4)}
    out["checks"]["C2_convergence"]["verdict"] = all(c2.values())
    out["checks"]["C2_convergence"]["per_degree_within_0p55_of_p_plus_1"] = c2
    out["checks"]["C2_convergence"]["note"] = (
        "observed orders are the auditor's own least-squares slope of the auditor's "
        "own L2 norm; the sign is flipped to the conventional (positive) definition")

    # ---- C5: reflection wall stress (fixed 2.0 / free 0)
    refl = {}
    for end in ("fixed", "free"):
        sig0, vel0 = initial_data("reflection", mat)
        t_end = final_time("reflection", mat)
        cc, cf, mm = solve_elastic_rod(
            200, 3, t_end, E=mat["E"], rho=mat["rho"], length=LENGTH, cfl=0.3,
            bc=end, initial_sigma=sig0, initial_velocity=vel0)
        num = wall_stress(cf.numpy(), 3)
        exact = 2.0 * 1.0e6 if end == "fixed" else 0.0
        refl[end] = {"wall_sigma_num": num, "wall_sigma_exact": exact,
                     "ratio": num / SIGMA_AMPLITUDE,
                     "rel_error_vs_amplitude": abs(num - exact) / SIGMA_AMPLITUDE,
                     "claimed_ratio": 1.9999997835553325 if end == "fixed" else 2.1642228866023268e-07}
    out["checks"]["C5_reflection"] = refl
    out["checks"]["C5_reflection"]["verdict"] = (
        abs(refl["fixed"]["ratio"] - 2.0) < 2e-3 and abs(refl["free"]["ratio"]) < 2e-3)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "independent_recompute.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) else v)
                      for k, v in out["checks"].items()}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())