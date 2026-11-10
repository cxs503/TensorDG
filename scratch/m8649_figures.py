"""Figures for m8649: SWE hydrostatic well-balanced validation artifact."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordg.swe_1d import reconstruct_fields, solve_swe_1d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MAIN = json.load(open(ROOT / "validation/swe_thacker/hydrostatic_wb_results.json"))
ALT = json.load(open(ROOT / "artifacts/m8649/hydrostatic_wb_results_loong64_cj30.json"))


# ---------------------------------------------------------------- figure 1
def fig_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, label in zip(
        axes, ("eta_error_max", "u_max"), (r"max$|\eta-\eta_0|$  [m]", r"max$|u|$  [m/s]")
    ):
        for tag, data, style in (
            ("host A (this repo tree)", MAIN, "-o"),
            ("host B (cluster cj30)", ALT, "--s"),
        ):
            for block in data["convergence_study"]["runs"]:
                deg = block["degree"]
                n = [r["ncells"] for r in block["results"]]
                v = [r[key] for r in block["results"]]
                ax.loglog(n, v, style, ms=5, lw=1.4,
                          label=f"P{deg}, {tag.split(' ')[0]}")
        ax.axhline(1e-8, color="k", ls=":", lw=1)
        ax.text(52, 1.4e-8, "pass criterion 1e-8", fontsize=7)
        ax.set_xlabel("cells N")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_title("Lake-at-rest drift on Thacker parabolic basin (t = 1.0 s)")
    fig.tight_layout()
    p = OUT / "swe_hydrostatic_wb_roundoff.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 2
def thacker_bed(x, H0=2.0, L=1.0):
    return -H0 + (H0 / L**2) * x**2


def fig_state(ncells=100, degree=1):
    eta0, H0, L = 1.0, 2.0, 1.0
    zb = lambda xx: thacker_bed(xx, H0, L)
    h0 = lambda xx: eta0 - zb(xx)
    u0 = lambda xx: torch.zeros_like(xx)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    d = MAIN["single_run"]
    x, coeff, zb_faces, meta = solve_swe_1d(
        h0, u0, zb, ncells=ncells, degree=degree, final_time=1.0,
        domain=(-L, L), cfl=0.3, flux_kind="hll", well_balanced=True,
    )
    h, u, eta = reconstruct_fields(coeff, zb_faces, zb_center=meta["zb_modes"][:, 0])
    x = x.detach().cpu().numpy()
    ax = axes[0]
    ax.plot(x, eta.detach().cpu().numpy(), "C0-", lw=2, label=r"$\eta_h$ (DG)")
    ax.plot(x, (h0(torch.tensor(x)) + zb(torch.tensor(x))).numpy(), "k--", lw=1,
            label=r"$\eta$ exact = 1.0 m")
    ax.plot(x, zb(torch.tensor(x)).numpy(), "C2-", lw=1.4, label=r"$z_b$")
    ax.plot(x, h.detach().cpu().numpy(), "C3-", lw=1.4, label="$h_h$")
    ax.set_xlabel("x [m]"); ax.set_ylabel("elevation / depth [m]")
    ax.set_title(f"Lake at rest, N={ncells}, P{degree}, t = 1.0 s")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # spurious velocity vs time
    times = [0.1, 0.25, 0.5, 0.75, 1.0]
    umax = []
    for t in times:
        _, c, zbf, m = solve_swe_1d(
            h0, u0, zb, ncells=ncells, degree=degree, final_time=t,
            domain=(-L, L), cfl=0.3, flux_kind="hll", well_balanced=True,
        )
        _, uu, _ = reconstruct_fields(c, zbf, zb_center=m["zb_modes"][:, 0])
        umax.append(float(uu.abs().max()))
    ax = axes[1]
    ax.semilogy(times, umax, "o-", label=f"well_balanced=True (P{degree})")
    ax.axhline(2.1e-2, color="C3", ls="--", lw=1.2)
    ax.text(0.12, 3.0e-2, "well_balanced=False, t=1 s\n(doc §Comparison): 2.1e-2 m/s",
            color="C3", fontsize=7)
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"max$|u|$ [m/s]")
    ax.set_title("Spurious current stays at round-off")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / "swe_hydrostatic_wb_state.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, times, umax


if __name__ == "__main__":
    p1 = fig_convergence()
    print("wrote", p1)
    p2, times, umax = fig_state()
    print("wrote", p2)
    print("times", times)
    print("spurious u_max", ["%.3e" % v for v in umax])
    for p in (p1, p2):
        print(p.name, "sha256", __import__("hashlib").sha256(p.read_bytes()).hexdigest())