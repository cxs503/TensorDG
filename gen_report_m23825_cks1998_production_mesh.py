"""Report figures for m23825: CKS1998 time-power closure on the production mesh.

Produces three figures for the mission report (all numbers read from the
committed artifact ``validation/cks1998_vortex_2d/time_power_closure.json``
or from one fresh production-configuration solve; nothing is hand-typed):

* ``fig1_density_field.png``  -- density field at T = 1.0 on the production
  mesh (n = 32, P1, CFL = 0.2): numerical vs exact, plus the L1 error map;
* ``fig2_eta_ladder.png``     -- the eta(CFL) ladder on both meshes (10 and
  32) with the 5 % admissibility edge and the fitted power law;
* ``fig3_density_profile.png``-- density profile along y = 0 at T = 1.0,
  numerical vs exact (the analytic reference of CKS1998 section 5.3).

Usage::

    python gen_report_m23825_cks1998_production_mesh.py --out-dir /tmp/m23825_figs
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

ARTIFACT = REPO / "validation" / "cks1998_vortex_2d" / "time_power_closure.json"


def load_payload() -> dict:
    return json.loads(ARTIFACT.read_text())


def solve_production():
    """One production-configuration solve (n=32, P1, CFL=0.2, T=1.0)."""
    import torch

    from benchmarks.cks1998_vortex_order_certification import (
        DOMAIN,
        FINAL_TIME,
        CFL0,
        gauss_fields,
        vortex_conservative,
    )
    from tensordg.euler_dg_2d import solve_euler_dg_2d
    from benchmarks.cks1998_vortex_order_certification import initial

    n = 32
    x, y, coeff, metadata = solve_euler_dg_2d(
        initial,
        ncells=(n, n),
        degree=1,
        domain=DOMAIN,
        final_time=FINAL_TIME,
        cfl=CFL0,
        periodic=(True, True),
        device="cpu",
    )
    values, nodes, weights = gauss_fields(coeff, 1)
    (xmin, xmax), (ymin, ymax) = DOMAIN
    hx, hy = (xmax - xmin) / n, (ymax - ymin) / n
    X = x[:, None, None, None] + nodes[None, None, :, None] * (hx / 2.0)
    Y = y[None, :, None, None] + nodes[None, None, None, :] * (hy / 2.0)
    exact = vortex_conservative(X, Y, FINAL_TIME)
    rho_h = values[..., 0].numpy()
    rho_e = exact[..., 0].numpy()
    return X.numpy(), Y.numpy(), rho_h, rho_e, metadata


def fig1_field(out: Path) -> None:
    X, Y, rho_h, rho_e, metadata = solve_production()
    # collapse the Gauss-node axis by averaging (display only)
    rho_h_c = rho_h.mean(axis=2)
    rho_e_c = rho_e.mean(axis=2)
    xc = (X[..., 0] + X[..., -1]) / 2.0
    yc = (Y[0, :, 0, 0] + Y[0, :, -1, 0]) / 2.0
    diff = np.abs(rho_h_c - rho_e_c)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    vmax = max(rho_h_c.max(), rho_e_c.max())
    vmin = min(rho_h_c.min(), rho_e_c.min())
    for ax, field, title in (
        (axes[0], rho_h_c, f"numerical $\\rho_h$ (n=32, P1, CFL=0.2, {metadata['steps']} steps)"),
        (axes[1], rho_e_c, "exact $\\rho$ (CKS1998 §5.3)"),
    ):
        im = ax.contourf(XC := xc, YC := yc, field, levels=24,
                         cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046)
    im = axes[2].contourf(xc, yc, diff, levels=24, cmap="magma")
    axes[2].set_title(f"$|\\rho_h - \\rho_{\\mathrm{{exact}}}|$  "
                     f"(max {diff.max():.3e})", fontsize=9)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle("CKS1998 §5.3 isentropic vortex, T = 1.0, periodic [-5,5]², "
                 "strength 5, background (1,1)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig2_eta_ladder(payload: dict, out: Path) -> None:
    bundle = payload["degrees"]["p1"]
    eta_max = payload["eta_max"]
    cfl_max = payload["admissible_cfl_max"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = {"10": "tab:blue", "32": "tab:red"}
    for mesh, m in bundle["per_mesh"].items():
        xs, ys = [], []
        for s in m["eta_samples"]:
            if s.get("eta") is not None and math.isfinite(s["eta"]) and s["eta"] > 0:
                xs.append(s["cfl"])
                ys.append(s["eta"])
        xs, ys = np.array(xs), np.array(ys)
        ax.plot(xs, ys, "o-", color=colors.get(mesh, "k"),
                label=f"mesh n={mesh} (measured)", ms=5)
        full = (m["power"].get("full") or {})
        s_exp = full.get("exponent")
        if s_exp is not None and len(xs) >= 2:
            c_ref = xs[0]
            y_ref = ys[0]
            cline = np.logspace(math.log10(0.01), math.log10(0.8), 100)
            ax.plot(cline, y_ref * (cline / c_ref) ** s_exp, "--",
                   color=colors.get(mesh, "k"), lw=1,
                   label=f"n={mesh} fit $\\eta = A\\,\\mathrm{{CFL}}^{{{s_exp:.2f}}}$")
    ax.axhline(eta_max, color="k", ls=":", lw=1.2,
                 label=f"admissibility edge $\\eta_{{max}}$ = {eta_max:.0%}")
    ax.axvline(cfl_max, color="gray", ls=":", lw=1.2,
                label=f"admissible window edge CFL = {cfl_max}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CFL")
    ax.set_ylabel("temporal error share $\\eta$ (log)")
    ax.set_title("Temporal error share vs CFL — local n=10 vs production n=32\n"
                 "(both meshes: every leave-one-out fit crosses above the window)",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig3_profile(out: Path) -> None:
    X, Y, rho_h, rho_e, metadata = solve_production()
    rho_h_c = rho_h.mean(axis=2)
    rho_e_c = rho_e.mean(axis=2)
    xc = (X[..., 0] + X[..., -1]) / 2.0
    yc = (Y[0, :, 0, 0] + Y[0, :, -1, 0]) / 2.0
    j = int(np.argmin(np.abs(yc - 0.0)))
    order = np.argsort(xc[j])
    xs = xc[j][order]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(xs, rho_h_c[j][order], "b-", lw=1.6,
             label="DG n=32 (production mesh)")
    xx = np.linspace(-5, 5, 800)
    from benchmarks.cks1998_vortex_order_certification import vortex_primitive
    rho_line, _, _, _ = vortex_primitive(xx, np.zeros_like(xx), 1.0)
    ax.plot(xx, rho_line, "k--", lw=1.4, label="exact")
    ax.set_xlabel("x (y = 0)")
    ax.set_ylabel("density")
    ax.set_title("Density profile along y = 0 at T = 1.0 — production mesh",
                fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/m23825_figs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = load_payload()
    fig1_field(args.out_dir / "fig1_density_field.png")
    fig2_eta_ladder(payload, args.out_dir / "fig2_eta_ladder.png")
    fig3_profile(args.out_dir / "fig3_density_profile.png")
    print("figures written to", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
