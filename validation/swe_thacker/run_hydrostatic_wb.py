"""Programmatic source of every number in docs/tensordg-swe-hydrostatic-wellbalanced.md.

Produces:
  * lake-at-rest stationarity on a curved bed, default vs ``well_balanced`` (P1/P2);
  * Thacker (1981) planar-basin accuracy at t = 0.5 s for ``zb_slope`` vs
    ``well_balanced`` (P1/P2, N = 20);
  * SSP-RK2 / SSP-RK3 volume conservation with the analytic boundary flux;
  * ``hydrostatic_wb_results.json`` + two figures (``hydrostatic_wb.png``,
    ``thacker_wb_profiles.png``).

Run:  PYTHONPATH=. python validation/swe_thacker/run_hydrostatic_wb.py [--heavy]
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from tensordg.swe_1d import reconstruct_fields, solve_swe_1d
from tensordg import swe_exact

G = 9.81
OUT = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# 1. lake at rest on a curved bed
# --------------------------------------------------------------------------- #
ETA0 = 2.0


def _bed(xx):
    return 0.4 * torch.sin(math.pi * xx / 2.0)


def _lake(degree, well_balanced, ncells=64, final_time=1.0):
    x, coeff, _, meta = solve_swe_1d(
        lambda xx: ETA0 - _bed(xx), lambda xx: torch.zeros_like(xx), _bed,
        ncells=ncells, degree=degree, final_time=final_time,
        domain=(-2.0, 2.0), cfl=0.3, well_balanced=well_balanced)
    return coeff, meta


def lake_table(ncells=64):
    rows = []
    for degree in (1, 2):
        c0, _ = _lake(degree, True, ncells, 0.0)
        for wb in (False, True):
            c1, meta = _lake(degree, wb, ncells, 1.0)
            rows.append(dict(
                degree=degree, well_balanced=wb, ncells=ncells,
                steps=meta["steps"],
                max_momentum=float(c1[:, :, 1].abs().max()),
                max_state_change=float((c1[:, :, 0] - c0[:, :, 0]).abs().max()),
            ))
            print(f"  lake at rest P{degree} wb={wb}: steps={meta['steps']:5d} "
                  f"max|hu|={rows[-1]['max_momentum']:.3e} "
                  f"max|dc|={rows[-1]['max_state_change']:.3e}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# 2. Thacker basin, analytic boundary flux
# --------------------------------------------------------------------------- #
H0, AMP, HALF, T_END = 1.0, 0.5, 1.0, 0.5


def _thacker(ncells, degree, mode, t_end=T_END):
    def bed(xx):
        return torch.as_tensor(swe_exact.thacker_parabolic_bed(xx.numpy(), H0, HALF),
                               dtype=xx.dtype)

    def slope(xx):
        return torch.as_tensor(swe_exact.thacker_parabolic_bed_slope(
            xx.numpy(), H0, HALF), dtype=xx.dtype)

    def left(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(
            t, H0, AMP, HALF, G)[0], dtype=torch.float64)

    def right(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(
            t, H0, AMP, HALF, G)[1], dtype=torch.float64)

    kwargs = dict(bc_left_flux=left, bc_right_flux=right)
    if mode == "zb_slope":
        kwargs["zb_slope"] = slope
    else:
        kwargs["well_balanced"] = True

    x, coeff, zb_faces, meta = solve_swe_1d(
        lambda xx: torch.as_tensor(
            swe_exact.thacker_planar_basin_initial(xx.numpy(), H0, AMP, HALF)[0],
            dtype=xx.dtype),
        lambda xx: torch.zeros_like(xx), bed,
        ncells=ncells, degree=degree, final_time=t_end, domain=(-HALF, HALF),
        cfl=0.3, **kwargs)

    h, u, eta = reconstruct_fields(coeff, zb_faces)
    xe = x.numpy()
    h_e, u_e, eta_e = swe_exact.thacker_planar_basin(xe, t_end, H0, AMP, HALF)

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b) * 100.0)

    dx = 2.0 * HALF / ncells
    omega = math.sqrt(2.0 * G * H0) / HALF
    eta_c = H0 + AMP ** 2 / (4.0 * H0) * math.sin(omega * t_end) ** 2
    vol_exact = 2.0 * eta_c + (4.0 / 3.0) * H0 * HALF
    vol = float(coeff[:, 0, 0].sum()) * dx

    return dict(ncells=ncells, degree=degree, mode=mode, steps=meta["steps"],
                eta_pct=rel(eta.numpy(), eta_e), h_pct=rel(h.numpy(), h_e),
                u_pct=rel(u.numpy(), u_e),
                volume_pct=(vol - vol_exact) / vol_exact * 100.0), dict(
                    x=xe, eta=eta.numpy(), eta_exact=eta_e, u=u.numpy(), u_exact=u_e)


def thacker_table(heavy=False):
    rows, profiles = [], {}
    cases = [(1, "zb_slope"), (2, "zb_slope"), (2, "well_balanced"), (1, "well_balanced")]
    ns = (20, 40) if heavy else (20,)
    for ncells in ns:
        for degree, mode in cases:
            try:
                row, prof = _thacker(ncells, degree, mode)
            except Exception as exc:                     # honest FAIL, not hidden
                row = dict(ncells=ncells, degree=degree, mode=mode,
                           status=f"failed: {type(exc).__name__}: {exc}")
                print(f"  P{degree} {mode} N={ncells}: FAILED {type(exc).__name__}", flush=True)
                rows.append(row)
                continue
            row["status"] = "ok"
            rows.append(row)
            profiles[(degree, mode, ncells)] = prof
            print(f"  P{degree} {mode:14s} N={ncells:3d} steps={row['steps']:5d} "
                  f"eta={row['eta_pct']:8.4f}% h={row['h_pct']:8.4f}% "
                  f"u={row['u_pct']:8.4f}% V={row['volume_pct']:+.2e}%", flush=True)
    return rows, profiles


# --------------------------------------------------------------------------- #
# 3. volume conservation: RK2 vs RK3 with the analytic boundary flux
# --------------------------------------------------------------------------- #
def rk_volume(rk_order, ncells=20, t_end=0.2):
    def bed(xx):
        return torch.as_tensor(swe_exact.thacker_parabolic_bed(xx.numpy(), H0, HALF),
                               dtype=xx.dtype)

    def left(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(
            t, H0, AMP, HALF, G)[0], dtype=torch.float64)

    def right(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(
            t, H0, AMP, HALF, G)[1], dtype=torch.float64)

    x, coeff, _, meta = solve_swe_1d(
        lambda xx: torch.as_tensor(
            swe_exact.thacker_planar_basin_initial(xx.numpy(), H0, AMP, HALF)[0],
            dtype=xx.dtype),
        lambda xx: torch.zeros_like(xx), bed,
        ncells=ncells, degree=1, final_time=t_end, domain=(-HALF, HALF),
        cfl=0.3, rk_order=rk_order, bc_left_flux=left, bc_right_flux=right)
    dx = 2.0 * HALF / ncells
    vol = float(coeff[:, 0, 0].sum()) * dx
    omega = math.sqrt(2.0 * G * H0) / HALF
    eta_c = H0 + AMP ** 2 / (4.0 * H0) * math.sin(omega * t_end) ** 2
    vol_exact = 2.0 * eta_c + (4.0 / 3.0) * H0 * HALF
    return dict(rk_order=rk_order, steps=meta["steps"],
                volume_rel_err=(vol - vol_exact) / vol_exact)


def main(heavy=False):
    print("Thacker planar basin + lake-at-rest verification (hydrostatic reconstruction)")
    print("[1] lake at rest, curved bed 0.4 sin(pi x / 2), N=64, t=1 s")
    lake = lake_table()
    print("[2] Thacker basin, t=0.5 s (period %.4f s), analytic boundary flux"
          % swe_exact.thacker_planar_basin_period(H0, HALF, G))
    thacker, profiles = thacker_table(heavy)
    print("[3] volume conservation with the analytic boundary flux (N=20, t=0.2 s)")
    rk = []
    for order in (2, 3):
        row = rk_volume(order)
        rk.append(row)
        print(f"  rk{order}: steps={row['steps']} volume rel err={row['volume_rel_err']:+.3e}")

    res = dict(params=dict(eta0=ETA0, h0=H0, amplitude=AMP, half_width=HALF,
                           t_end=T_END, seed=None),
               lake_at_rest=lake, thacker=thacker, rk_volume=rk)
    with open(os.path.join(OUT, "hydrostatic_wb_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("results written: hydrostatic_wb_results.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
        labels = [f"P{r['degree']} {'WB' if r['well_balanced'] else 'default'}"
                  for r in lake]
        vals = [max(r["max_momentum"], 1e-16) for r in lake]
        colors = ["tab:gray" if not r["well_balanced"] else "tab:blue" for r in lake]
        ax[0].bar(labels, vals, color=colors)
        ax[0].set_yscale("log")
        ax[0].axhline(1e-10, ls="--", c="k", lw=.8, label="round-off floor")
        ax[0].set_ylabel(r"$\max|hu|$ after t = 1 s")
        ax[0].set_title("lake at rest on a curved bed (N=64)")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=.3, axis="y")
        for i, v in enumerate(vals):
            ax[0].text(i, v * 1.3, f"{v:.1e}", ha="center", fontsize=7)

        keys = [k for k in profiles if k[0] == 2 and k[2] == 20]
        if keys:
            prof = profiles[keys[0]]
            ax[1].plot(prof["x"], prof["eta_exact"], "k-", lw=3, label="exact")
            ax[1].plot(prof["x"], prof["eta"], "o", ms=4, label="DG P2 N=20")
            ax[1].set_xlabel("x / m")
            ax[1].set_ylabel(r"$\eta$ / m")
            title = "Thacker basin free surface, t = 0.5 s"
            ax[1].set_title(title)
            ax[1].legend(fontsize=8)
            ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "hydrostatic_wb.png"), dpi=150)
        plt.close(fig)
        print("figure written: hydrostatic_wb.png")
    except Exception as exc:  # pragma: no cover
        print("figure generation skipped:", exc)


if __name__ == "__main__":
    import sys
    main(heavy="--heavy" in sys.argv)