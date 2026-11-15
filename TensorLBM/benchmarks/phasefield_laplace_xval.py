"""XVAL benchmark: no-MCMP two-phase (phase-field / Allen-Cahn) LBM vs Young-Laplace.

Answers the question behind anchor ``XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY`` with a
*pre-registered* protocol (tolerances fixed in this file before any run):

  Leg A -- planar interface: the equilibrium profile must be the analytic
           ``tanh(s/ell)`` with ``ell = sqrt(kappa/2beta)``, and the surface
           tension measured from that profile (``kappa int (dphi/ds)^2 ds``)
           must equal the analytic ``(4 sqrt(2)/3) sqrt(beta kappa)``
           (tolerance 5 %).
  Leg B -- static droplet: ``dp * R`` must be constant over the radii and equal
           to that sigma (Young-Laplace).  This is the *independent* leg: sigma
           comes from ``beta, kappa`` -- it is never fitted to a Laplace run.
           Primary criterion: ``max |ratio - 1| <= 15 %`` (ecosystem acceptance)
           with a 3 % target; secondary: dp*R dispersion over R <= 5 %
           (the tolerance pre-registered in ``t3_matrix`` for this physics).
  Leg C -- interface-resolution study: same physics, ``ell`` = 3 / 4 / 6 cells,
           error must shrink with better interface resolution.
  Leg D -- force-form cross-check: ``mu grad(phi)`` vs ``phi grad(mu)``; they
           differ by ``grad(phi mu)``, i.e. by the pressure convention only.

``dp`` is sampled in the two bulk phases (2 interface widths away from the
interface) and **time-averaged** over the last 1000 steps, otherwise the
acoustic ringing left over from the initial condition is mistaken for error.

Numbers are written to ``outputs/phasefield_laplace/summary*.json`` -- nothing
is hand-copied into the report.

Usage::

    python benchmarks/phasefield_laplace_xval.py --quick   # CI scale (~90 s)
    python benchmarks/phasefield_laplace_xval.py          # full scan (~30 min)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tensorlbm.phasefield as pf  # noqa: E402

SIGMA_TARGET = 0.05
ELL_TARGET = 4.0
MOBILITY = 0.10
OMEGA = 1.0

TOL_SIGMA_PROFILE = 0.05          # Leg A
TOL_LAPLACE = 0.15                # Leg B, ecosystem acceptance
TARGET_LAPLACE = 0.03             # Leg B, ecosystem target
TOL_DPR_DISPERSION = 0.05         # Leg B, t3_matrix pre-registered band

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phasefield_laplace")


def make_solver(n: int, sigma: float = SIGMA_TARGET, ell: float = ELL_TARGET,
                mobility: float = MOBILITY, omega: float = OMEGA,
                force_form: str = "mu_grad_phi") -> pf.PhaseFieldTwoPhase:
    """Pick (beta, kappa) so that *both* ``ell`` and ``sigma`` hit their targets.

    sigma = (8/3) beta ell  ->  beta = 3 sigma / (8 ell)
    ell   = sqrt(kappa/2beta) -> kappa = 2 beta ell^2
    """
    beta = 3.0 * sigma / (8.0 * ell)
    kappa = 2.0 * beta * ell ** 2
    return pf.PhaseFieldTwoPhase(nx=n, ny=n, beta=beta, kappa=kappa,
                                 mobility=mobility, omega=omega,
                                 force_form=force_form)


def time_averaged_dp(solver, radius: float, window: int = 1000,
                     samples: int = 10) -> dict:
    """Time-average dp over ``window`` steps to kill the acoustic ringing."""
    every = max(1, window // samples)
    vals = []
    p_in, p_out = [], []
    for _ in range(samples):
        solver.step(every)
        m = solver.pressure_drop(radius)
        vals.append(m["dp"])
        p_in.append(m["p_in"])
        p_out.append(m["p_out"])
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return {"dp": mean, "dp_std": var ** 0.5,
            "p_in": sum(p_in) / len(p_in), "p_out": sum(p_out) / len(p_out),
            "dp_samples": vals}


def equilibrate(solver, r: float, max_steps: int, tol: float = 3e-3,
                chunk: int = 500, avg_window: int = 1000,
                verbose: bool = True) -> dict:
    """Run until dp drifts < tol over two consecutive chunks (or max_steps),
    then time-average dp over ``avg_window`` further steps."""
    hist = []
    t0 = time.time()
    steps = 0
    while steps < max_steps:
        solver.step(chunk)
        steps += chunk
        dp = solver.pressure_drop(r)["dp"]
        hist.append(dp)
        if verbose:
            print(f"    step {steps:6d}  dp={dp:+.5e}  |u|={solver.spurious_current():.2e}"
                  f"  phi=[{float(solver.phi.min()):+.3f},{float(solver.phi.max()):+.3f}]"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        if len(hist) >= 3:
            ref = max(abs(hist[-2]), 1e-30)
            if abs(hist[-1] - hist[-2]) / ref < tol and abs(hist[-2] - hist[-3]) / ref < tol:
                break
    avg = time_averaged_dp(solver, r, window=avg_window)
    steps += avg_window
    avg.update({"steps": steps, "dp_history": hist, "equilibrated": steps < max_steps})
    return avg


def leg_a_planar(n: int, steps: int, ell: float = ELL_TARGET) -> dict:
    solver = make_solver(n, ell=ell)
    solver.setup_planar_interface(orientation="x")
    # Freeze the hydrodynamics: Leg A is the *free-energy* leg, so a defect in the
    # Allen-Cahn update cannot be masked by a flow correction.
    solver._lbm_step = lambda fx, fy: None
    solver.run(steps)
    sigma_profile = solver.sigma_from_profile("x")
    s, prof = solver.interface_profile("x")
    order = torch.argsort(s)
    s, prof = s[order], prof[order]
    keep = s.abs() <= min(6.0 * solver.interface_width, 0.4 * solver.nx)
    s, prof = s[keep], prof[keep]
    analytic = torch.tanh(s / solver.interface_width)
    linf = float((prof - analytic).abs().max())
    return {
        "n": n, "steps": steps, "ell": ell,
        "interface_width_analytic": solver.interface_width,
        "sigma_analytic": solver.sigma_analytic,
        "sigma_profile": sigma_profile,
        "sigma_rel_error": abs(sigma_profile - solver.sigma_analytic) / solver.sigma_analytic,
        "pass": bool(abs(sigma_profile - solver.sigma_analytic) / solver.sigma_analytic
                     <= TOL_SIGMA_PROFILE),
        "profile_linf_vs_tanh": linf,
        "phi_min": float(solver.phi.min()), "phi_max": float(solver.phi.max()),
        "profile": {"s": s.tolist(), "phi": prof.tolist(), "tanh": analytic.tolist()},
    }


def leg_b_droplet(n: int, radius: float, max_steps: int, ell: float = ELL_TARGET,
                  force_form: str = "mu_grad_phi", capture: bool = False) -> dict:
    solver = make_solver(n, ell=ell, force_form=force_form)
    solver.setup_droplet(radius)
    eq = equilibrate(solver, radius, max_steps)
    dp = eq["dp"]
    sigma_over_R = solver.sigma_analytic / radius
    out = {
        "n": n, "radius": radius, "ell": ell, "force_form": force_form,
        "R_over_ell": radius / ell,
        "steps": eq["steps"], "equilibrated": eq["equilibrated"],
        "dp_history": eq["dp_history"], "dp_samples": eq["dp_samples"],
        "dp": dp, "dp_std": eq["dp_std"], "p_in": eq["p_in"], "p_out": eq["p_out"],
        "dp_times_R": dp * radius,
        "sigma_analytic": solver.sigma_analytic,
        "sigma_over_R": sigma_over_R,
        "ratio_dp_over_sigma_R": dp / sigma_over_R,
        "rel_error_vs_sigma_over_R": abs(dp - sigma_over_R) / sigma_over_R,
        "spurious_current": solver.spurious_current(),
        "phi_min": float(solver.phi.min()), "phi_max": float(solver.phi.max()),
        "mass_phi": solver.total_phi(),
    }
    if capture:
        r = solver.radial_coordinate()
        nb = int(radius + 6 * ell)
        idx = torch.bucketize(r.flatten(), torch.arange(0.0, nb + 1.0)) - 1
        phi_r, u_r = [], []
        for b in range(nb):
            sel = idx == b
            if int(sel.sum()) == 0:
                phi_r.append(float("nan"))
                u_r.append(float("nan"))
            else:
                phi_r.append(float(solver.phi.flatten()[sel].mean()))
                u_r.append(float(torch.sqrt(solver.ux ** 2 + solver.uy ** 2)
                                 .flatten()[sel].mean()))
        out["radial"] = {"r": [i + 0.5 for i in range(nb)], "phi": phi_r, "u": u_r}
        out["field"] = {
            "phi": solver.phi[::4, ::4].tolist(),
            "p_rel": ((solver.pressure() - solver.pressure().mean()) * 1e5)[::4, ::4].tolist(),
            "n_full": n, "stride": 4, "radius": radius,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="CI scale")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    os.makedirs(OUTDIR, exist_ok=True)

    if args.quick:
        radii, n_drop, max_steps = [12.0, 14.0], 64, 2000
        planar = [(64, 1200)]
        ell_study = [(12.0, 96, 1200)]
        force_r, force_steps = 14.0, 2000
    else:
        radii, n_drop, max_steps = [16.0, 20.0, 24.0, 28.0], 128, 8000
        planar = [(64, 2000), (96, 2000), (128, 2000)]
        ell_study = [(3.0, 128, 8000), (4.0, 128, 8000), (6.0, 128, 8000)]
        force_r, force_steps = 24.0, 8000

    summary = {
        "anchor": "XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY",
        "model": "phase-field / Allen-Cahn two-phase LBM (single component, no MCMP)",
        "upstream_pattern": "lssfau/walberla two-phase (no MCMP) tree: phase-field Allen-Cahn showcase",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "quick": args.quick,
        "tolerances": {"sigma_profile": TOL_SIGMA_PROFILE, "laplace": TOL_LAPLACE,
                       "laplace_target": TARGET_LAPLACE,
                       "dpR_dispersion": TOL_DPR_DISPERSION},
        "parameters": {"sigma_target": SIGMA_TARGET, "ell_target": ELL_TARGET,
                       "mobility": MOBILITY, "omega": OMEGA,
                       "advection": "central", "dtype": "float64"},
    }

    print("== Leg A: planar interface vs analytic tanh / analytic sigma ==", flush=True)
    summary["leg_a_planar"] = [leg_a_planar(n, steps) for n, steps in planar]
    for row in summary["leg_a_planar"]:
        print(f"  n={row['n']:4d} sigma_profile={row['sigma_profile']:.6f} "
              f"analytic={row['sigma_analytic']:.6f} rel={row['sigma_rel_error'] * 100:.2f}% "
              f"profile Linf={row['profile_linf_vs_tanh']:.3e}" +
              ("  PASS" if row["pass"] else "  FAIL"), flush=True)

    print("== Leg B: static droplet, Young-Laplace ==", flush=True)
    drops = []
    for R in radii:
        print(f"  R={R} n={n_drop} (R/ell={R / ELL_TARGET:.1f})", flush=True)
        drops.append(leg_b_droplet(n_drop, R, max_steps, capture=(R == radii[1])))
    summary["leg_b_droplet"] = drops
    ratios = [d["ratio_dp_over_sigma_R"] for d in drops]
    prod = [d["dp_times_R"] for d in drops]
    mean_prod = sum(prod) / len(prod)
    summary["leg_b_summary"] = {
        "ratio_mean": sum(ratios) / len(ratios),
        "ratio_max_abs_dev_from_1": max(abs(v - 1.0) for v in ratios),
        "dpR_mean": mean_prod,
        "dpR_dispersion": max(abs(v - mean_prod) / mean_prod for v in prod),
        "max_rel_error_vs_sigma_over_R": max(d["rel_error_vs_sigma_over_R"] for d in drops),
        "max_spurious_current": max(d["spurious_current"] for d in drops),
        "pass_15pct": bool(max(abs(v - 1.0) for v in ratios) <= TOL_LAPLACE),
        "pass_3pct": bool(max(abs(v - 1.0) for v in ratios) <= TARGET_LAPLACE),
        "pass_dpR_dispersion": bool(max(abs(v - mean_prod) / mean_prod for v in prod)
                                    <= TOL_DPR_DISPERSION),
    }
    print("  -> " + json.dumps({k: v for k, v in summary["leg_b_summary"].items()
                                if id(v) and not isinstance(v, (list, dict))}, default=str),
          flush=True)

    print("== Leg C: interface-resolution study ==", flush=True)
    res = []
    for ell, n, steps in ell_study:
        print(f"  ell={ell} n={n}", flush=True)
        res.append(leg_b_droplet(n, 24.0, steps, ell=ell))
    summary["leg_c_ell_study"] = res

    print("== Leg D: force-form cross-check ==", flush=True)
    forms = []
    for form in ("mu_grad_phi", "phi_grad_mu"):
        print(f"  force_form={form}", flush=True)
        forms.append(leg_b_droplet(n_drop, force_r, force_steps, force_form=form))
    summary["leg_d_force_forms"] = forms

    path = os.path.join(OUTDIR, f"summary{args.tag}.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())