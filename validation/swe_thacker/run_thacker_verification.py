"""Verification driver: Thacker (1981) planar sloshing in a parabolic basin.

Two questions are answered here, both with programmatically generated numbers
(the JSON next to this script is the single source of truth for the report):

1. How accurate is ``solve_swe_1d`` against the *correct* planar exact solution
   once the analytic boundary flux is imposed (verification mode)?
2. What does the default transmissive boundary closure cost?  The basin rim
   ``z_b(+-L) = 0`` carries water, so the exact solution has a non-zero mass
   flux through both ends; a transmissive closure reflects the outgoing wave
   and the reflected wave returns to the interior after one domain crossing
   ``t_ref ~ 2L / sqrt(g h)``.

Run:  PYTHONPATH=. python validation/swe_thacker/run_thacker_verification.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from tensordg.swe_1d import solve_swe_1d, reconstruct_fields
from tensordg import swe_exact

G = 9.81
H0 = 1.0
AMP = 0.5      # initial surface tilt eta(x, 0) = h0 + A x / L
L = 1.0
T_END = 0.5
OUT = os.path.dirname(os.path.abspath(__file__))


def exact_boundary_flux(t):
    fl, fr = swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)
    return (torch.as_tensor(fl, dtype=torch.float64),
            torch.as_tensor(fr, dtype=torch.float64))


def run(ncells, degree, t_end=T_END, exact_bc=True):
    fl, fr = exact_boundary_flux(0.0)

    def left(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)[0],
                               dtype=torch.float64)

    def right(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)[1],
                               dtype=torch.float64)

    def init_h(x):
        h, _ = swe_exact.thacker_planar_basin_initial(x.numpy(), H0, AMP, L)
        return torch.as_tensor(h, dtype=x.dtype)

    def zb(x):
        return torch.as_tensor(swe_exact.thacker_parabolic_bed(x.numpy(), H0, L), dtype=x.dtype)

    kwargs = dict(bc_left_flux=left, bc_right_flux=right) if exact_bc else {}
    x, coeff, zbf, meta = solve_swe_1d(
        init_h, lambda z: torch.zeros_like(z), zb, ncells=ncells, degree=degree,
        final_time=t_end, domain=(-L, L), cfl=0.3, flux_kind="hll",
        limiter=None, rk_order=3, dtype=torch.float64, **kwargs)
    h, u, eta = reconstruct_fields(coeff, zbf)
    xe = x.numpy()
    h_e, u_e, eta_e = swe_exact.thacker_planar_basin(xe, t_end, H0, AMP, L, G)
    h_e = torch.as_tensor(h_e, dtype=torch.float64)
    u_e = torch.as_tensor(u_e, dtype=torch.float64)
    eta_e = torch.as_tensor(eta_e, dtype=torch.float64)

    def rel(a, b):
        return float(torch.sqrt(torch.mean((a - b) ** 2)) / b.abs().max()) * 100.0

    dx = 2.0 * L / ncells
    vol_num = float(h.sum()) * dx
    # exact volume: V = int (eta - z_b) dx = 2 eta_c + 4/3 h0 L
    eta_c = H0 + (AMP ** 2 / (4.0 * H0)) * math.sin(swe_exact._thacker_params(H0, L, G) * t_end) ** 2
    vol_exact = 2.0 * eta_c + (4.0 / 3.0) * H0 * L
    return dict(n=ncells, degree=degree, t_end=t_end, exact_bc=exact_bc,
                steps=meta["steps"], eta_rel_pct=rel(eta, eta_e),
                u_rel_pct=rel(u, u_e), h_rel_pct=rel(h, h_e),
                vol_num=vol_num, vol_exact=vol_exact,
                vol_rel_pct=(vol_num - vol_exact) / vol_exact * 100.0,
                x=xe, eta=eta.numpy(), u=u.numpy(), eta_exact=eta_e.numpy(),
                u_exact=u_e.numpy())


def main():
    res = {"params": dict(h0=H0, amplitude=AMP, half_width=L, gravity=G, t_end=T_END,
                          period=swe_exact.thacker_planar_basin_period(H0, L, G),
                          reflection_time=2.0 * L / math.sqrt(G * 2.0)),
           "runs": []}
    print(f"Thacker planar basin | h0={H0} A={AMP} L={L} T={T_END} "
          f"(period {res['params']['period']:.4f} s, crossing {res['params']['reflection_time']:.4f} s)")

    print("\n[1] exact boundary flux (verification mode) -- design order")
    for deg in (1, 2):
        prev = None
        for n in (20, 40, 80):
            try:
                r = run(n, deg, exact_bc=True)
            except Exception as exc:
                res["runs"].append(dict(n=n, degree=deg, t_end=T_END, exact_bc=True,
                                        status="failed", error=f"{type(exc).__name__}: {exc}"))
                print(f"  P{deg} N={n:4d} FAILED: {type(exc).__name__}: {exc}")
                continue
            r["status"] = "ok"
            r["eoc_eta"] = (math.log2(prev / r["eta_rel_pct"])
                            if prev is not None else None)
            prev = r["eta_rel_pct"]
            res["runs"].append({k: v for k, v in r.items()
                                if k not in ("x", "eta", "u", "eta_exact", "u_exact")})
            eoc = r["eoc_eta"]
            print(f"  P{deg} N={n:4d} steps={r['steps']:5d} eta={r['eta_rel_pct']:8.4f}% "
                  f"u={r['u_rel_pct']:8.4f}% h={r['h_rel_pct']:8.4f}% "
                  f"V={r['vol_num']:.5f} (exact {r['vol_exact']:.5f}, {r['vol_rel_pct']:+.3f}%) "
                  f"EOC={'n/a' if eoc is None else round(eoc, 2)}")

    print("\n[2] default transmissive closure (same runs, no exact boundary data)")
    for deg in (1, 2):
        for n in (20, 80):
            try:
                r = run(n, deg, exact_bc=False)
            except Exception as exc:
                print(f"  P{deg} N={n:4d} FAILED: {type(exc).__name__}: {exc}")
                continue
            r["status"] = "ok"
            res["runs"].append({k: v for k, v in r.items()
                                if k not in ("x", "eta", "u", "eta_exact", "u_exact")})
            print(f"  P{deg} N={n:4d} steps={r['steps']:5d} eta={r['eta_rel_pct']:8.4f}% "
                  f"u={r['u_rel_pct']:8.4f}% V={r['vol_num']:.5f} "
                  f"({r['vol_rel_pct']:+.3f}%)")
            if deg == 2 and n == 80:
                ref = r

    # keep profile data of the P2 N=80 exact-BC run for the figure
    prof = run(80, 2, exact_bc=True)
    ref.setdefault("x", prof["x"]) if "x" not in ref else None
    with open(os.path.join(OUT, "thacker_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(prof["x"], prof["eta_exact"], "k-", lw=3, label="exact")
        ax[0].plot(prof["x"], prof["eta"], "o", ms=3, label="DG P2 N=80 (exact BC)")
        ax[0].plot(ref["x"], ref["eta"], "s", ms=3, alpha=.6,
                   label="DG P2 N=80 (transmissive)")
        ax[0].set_xlabel("x / m"); ax[0].set_ylabel(r"$\eta$ / m")
        ax[0].set_title(f"free surface at t={T_END} s"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].plot(prof["x"], prof["u_exact"], "k-", lw=3, label="exact")
        ax[1].plot(prof["x"], prof["u"], "o", ms=3, label="DG P2 N=80 (exact BC)")
        ax[1].plot(ref["x"], ref["u"], "s", ms=3, alpha=.6, label="DG P2 N=80 (transmissive)")
        ax[1].set_xlabel("x / m"); ax[1].set_ylabel("u / (m/s)")
        ax[1].set_title("velocity at t=0.5 s (exact: spatially uniform)")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "thacker_profiles.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.4, 4))
        for deg, style in ((1, "o-"), (2, "s-")):
            runs = [r for r in res["runs"] if r["degree"] == deg and r["exact_bc"]]
            ns = [r["n"] for r in runs]
            ax.loglog(ns, [r["eta_rel_pct"] for r in runs], style, label=f"P{deg} exact BC")
            ax.loglog(ns, [2e2 * np.array(ns, dtype=float) ** -(deg + 1)], "k--", lw=.8,
                      label=f"slope {deg + 1}" if deg == 1 else None)
        trans = [r for r in res["runs"] if not r["exact_bc"]]
        ns = sorted({r["n"] for r in trans}, key=int)
        ax.loglog(ns, [next(r["eta_rel_pct"] for r in trans if r["n"] == n and r["degree"] == 1)
                       for n in ns], "^-", alpha=.7, label="P1 transmissive")
        ax.set_xlabel("cells"); ax.set_ylabel(r"relative $L_2$ error of $\eta$ / %")
        ax.set_title("Thacker basin: convergence"); ax.grid(alpha=.3, which="both")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "thacker_convergence.png"), dpi=150)
        plt.close(fig)
        print("\nfigures written: thacker_profiles.png, thacker_convergence.png")
    except Exception as exc:  # pragma: no cover
        print("figure generation skipped:", exc)
    print("results written: thacker_results.json")


if __name__ == "__main__":
    main()