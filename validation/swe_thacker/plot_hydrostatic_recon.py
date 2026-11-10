"""Figures for the Audusse reconstruction report (no hand-copied numbers)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1]
                                               / "tensordg").is_dir() else Path(".")
sys.path.insert(0, str(ROOT))

from tensordg.swe_1d import (_lgl_transform, reconstruct_fields,  # noqa: E402
                             solve_swe_1d)

ETA0 = 2.0
STEP_BED = 0.5
DOMAIN = (-2.0, 2.0)
OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)


def cell_step_bed(ncells, degree):
    lgl = _lgl_transform(degree, torch.device("cpu"), torch.float64)[1]
    dx = (DOMAIN[1] - DOMAIN[0]) / ncells
    xc = DOMAIN[0] + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    node_x = xc[:, None] + 0.5 * dx * lgl[None, :]
    left = (torch.arange(ncells) < ncells // 2)[:, None]
    zb = torch.where(left, torch.full_like(node_x, STEP_BED),
                     torch.zeros_like(node_x))
    return xc, node_x, zb


def run(ncells, degree, final_time=1.0, hydrostatic=True, wet_dry=False):
    xc, node_x, zb = cell_step_bed(ncells, degree)
    if wet_dry:
        left = (torch.arange(ncells) < ncells // 2)[:, None]
        zb = torch.where(left, torch.full_like(node_x, 0.5),
                         torch.full_like(node_x, 2.5))
        h = torch.where(left, torch.full_like(node_x, 1.5),
                        torch.zeros_like(node_x))
    else:
        h = (ETA0 - zb).clamp_min(0.0)
    x, coeff, zb_faces, meta = solve_swe_1d(
        h, torch.zeros_like(node_x), zb, ncells=ncells, degree=degree,
        final_time=final_time, domain=DOMAIN, cfl=0.3, rk_order=3,
        well_balanced=True, hydrostatic=hydrostatic)
    hh, u, eta = reconstruct_fields(coeff, zb_faces,
                                    zb_center=meta["zb_modes"][:, 0])
    return x, hh, u, eta, zb_faces, meta


def fig_profile():
    x, h, u, eta, zb, meta = run(64, 1, hydrostatic=False)
    xw, hw, uw, etaw, zbw, _ = run(64, 1, hydrostatic=True)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    ax = axes[0]
    ax.plot(xw.numpy(), (eta - ETA0).numpy(), "C3-", lw=1.6,
            label=r"$\eta-\eta_0$, hydrostatic=False")
    ax.plot(xw.numpy(), (etaw - ETA0).numpy(), "C0-", lw=1.6,
            label=r"$\eta-\eta_0$, Audusse reconstruction")
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_ylabel(r"free-surface deviation [m]")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Lake at rest across a 0.5 m bed step (P1, N=64, t = 1 s)")
    ax = axes[1]
    ax.plot(xw.numpy(), u.numpy(), "C3-", lw=1.6, label="u, hydrostatic=False")
    ax.plot(xw.numpy(), uw.numpy(), "C0-", lw=1.6, label="u, reconstruction")
    zb_mid = 0.5 * (zb[:, 0] + zb[:, 1])
    ax.plot(xw.numpy(), zb_mid.numpy(), "k--", lw=1.0, label=r"$z_b$")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("velocity [m/s]")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "swe_hydrostatic_recon_profiles.png", dpi=140)
    return fig


def fig_convergence(res: dict):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for rec in ("rest_state_across_step", "rest_state_without_reconstruction"):
        for degree in (1, 2):
            pts = sorted([r for r in res[rec] if r["degree"] == degree],
                         key=lambda r: r["ncells"])
            xs = [r["ncells"] for r in pts]
            ys = [max(r["u_max"], 1e-17) for r in pts]
            style = "o-" if "without" not in rec else "s--"
            col = "C0" if degree == 1 else "C1"
            lab = f"P{degree}, " + ("reconstruction" if "without" not in rec
                                    else "no reconstruction")
            ax.loglog(xs, ys, style, color=col, label=lab)
    ax.set_xlabel("cells N")
    ax.set_ylabel(r"max |u| [m/s] at t = 1 s")
    ax.set_title("Spurious current in a still lake across a bed step")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "swe_hydrostatic_recon_convergence.png", dpi=140)
    return fig


def fig_wet_dry():
    x, h, u, eta, zb, meta = run(64, 2, wet_dry=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x.numpy(), eta.numpy(), "C0-", lw=1.6, label=r"$\eta$ (surface)")
    ax.plot(x.numpy(), (eta - h).numpy(), "k--", lw=1.2, label=r"$z_b$ (bed)")
    ax.plot(x.numpy(), u.numpy(), "C2-", lw=1.2, label="u")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("[m], [m/s]")
    ax.set_title("Wet/dry wall: lake impounded against a 2.5 m bed (P2, N=64, t = 1 s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "swe_hydrostatic_recon_wetdry.png", dpi=140)
    return fig


def main():
    res = json.loads(Path("validation/swe_thacker/hydrostatic_recon_results.json").read_text())
    fig_profile()
    fig_convergence(res)
    fig_wet_dry()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
