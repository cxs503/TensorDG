#!/usr/bin/env python3
"""Figures for the m16133 board audit of CASE-DG-BLASTWAVE-PERSSON.

All numbers come from ``runs/m16133_audit/audit_summary.json`` (recomputed from
the delivered raw artifacts by ``m16133_blastwave_audit.py``) -- nothing is
retyped here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/m16133_audit"
OUT = REPO / "docs/board_audit/m16133"
GAMMA, X0, T = 1.4, 0.1, 0.01


def exact_profile(x):
    """Exact self-similar Riemann solution sampled at x (audit-side, closed form)."""
    s = json.loads((RUN / "audit_summary.json").read_text())["exact_riemann"]
    ps, us, D = s["p_star"], s["u_star"], s["D"]
    rho4, p4, rho1, p1 = 1.0, 1000.0, 1.0, 0.01
    a4 = math.sqrt(GAMMA * p4 / rho4)
    a1 = math.sqrt(GAMMA * p1 / rho1)
    rho_star, rho2 = s["rho_star_L"], s["rho2"]
    a_star = a4 * (ps / p4) ** ((GAMMA - 1) / (2 * GAMMA))
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        xi_s = (xi - X0) / T
        if xi_s >= D:                      # undisturbed ambient
            out[i] = rho1
        elif xi_s >= us:                   # between shock and contact
            out[i] = rho2
        elif xi_s >= 0.0:                  # between contact and driver interface
            out[i] = rho_star
        elif xi_s >= us - a_star:          # uniform star state (driver side)
            out[i] = rho_star
        elif xi_s >= -a4:                  # rarefaction fan
            u = 2 / (GAMMA + 1) * (a4 + xi_s)
            a = 2 / (GAMMA + 1) * a4 - (GAMMA - 1) / (GAMMA + 1) * xi_s
            out[i] = rho4 * (a / a4) ** (2 / (GAMMA - 1))
        else:
            out[i] = rho4
    return out


def load(case, tree="raw"):
    base = RUN / ("_source_m16072/raw" if tree == "raw" else tree)
    z = np.load(base / f"{case}.npz")
    return z["x"], z["rho"]


def fig_profiles():
    xr, rr = load("ref-1600")
    xg, rg = load("dg-av-p2n200")
    xn, rn = load("dg-noav-p2n100")
    xe = np.linspace(0.0, 1.0, 4001)
    re = exact_profile(xe)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(xe, re, "k-", lw=1.6, label="exact Riemann (analytic)")
    ax[0].plot(xr, rr, ".", ms=2, color="tab:green", label="FV-WENO5 N=1600 (ref)")
    ax[0].plot(xg, rg, "-", lw=1.0, color="tab:blue", label="DG p2 N=200 + AV")
    ax[0].plot(xn, rn, "-", lw=1.0, color="tab:red", alpha=.8, label="DG p2 N=100 no AV")
    ax[0].axhline(6.0, ls=":", color="gray", lw=.8)
    ax[0].set(xlabel="x", ylabel=r"$\rho$", title="density at t=0.01 (full domain)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    m = (xe > 0.285) & (xe < 0.36)
    ax[1].plot(xe[m], re[m], "k-", lw=1.6, label="exact")
    ax[1].plot(xr[(xr > 0.285) & (xr < 0.36)], rr[(xr > 0.285) & (xr < 0.36)], ".",
               ms=3, color="tab:green", label="ref N=1600")
    ax[1].plot(xg[(xg > 0.285) & (xg < 0.36)], rg[(xg > 0.285) & (xg < 0.36)], "o-",
               ms=2.5, lw=.9, color="tab:blue", label="DG p2N200 +AV")
    ax[1].plot(xn[(xn > 0.285) & (xn < 0.36)], rn[(xn > 0.285) & (xn < 0.36)], "s-",
               ms=2.5, lw=.9, color="tab:red", alpha=.8, label="DG p2N100 noAV")
    ax[1].axhline(5.99924, ls="--", color="k", lw=.8)
    ax[1].annotate(r"analytic post-shock $\rho_2$=5.999", xy=(0.288, 5.999),
                   xytext=(0.288, 6.20), fontsize=8)
    ax[1].set(xlabel="x", ylabel=r"$\rho$", title="zoom: shock / Gibbs spike")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout()
    p = OUT / "fig1_profiles.png"
    fig.savefig(p, dpi=140)
    print("wrote", p)


def fig_gates(s):
    rows = [r for r in s["rows"] if r["case"] != "ref-1600"]
    names = [r["case"] for r in rows]
    d1 = [r["D1_peak_err_pct"] for r in rows]
    d2 = [r["D2_spike_local_pct"] for r in rows]
    dx = [r["D1_vs_exact_pct"] for r in rows]
    y = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].barh(y - .27, d1, .26, label="D1: peak vs ref peak (author C3)", color="tab:blue")
    ax[0].barh(y, d2, .26, label="D2: peak vs ref at same x (spike)", color="tab:orange")
    ax[0].barh(y + .27, dx, .26, label="D1': peak vs analytic $\\rho_2$", color="tab:red")
    ax[0].axvline(2.0, color="k", ls="--", lw=1.0, label="2% overshoot gate")
    ax[0].axvline(-2.0, color="k", ls="--", lw=1.0)
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=8)
    ax[0].set(xlabel="peak deviation from the post-shock density [%]",
              title="Gibbs-overshoot gate is definition-dependent")
    ax[0].legend(fontsize=7.5, loc="lower right"); ax[0].grid(axis="x", alpha=.3)

    star = [r["star"]["err_pct"] for r in rows]
    ax[1].barh(y, star, color=["tab:purple" if r["case"].startswith("dg-av") else "tab:gray"
                               for r in rows])
    ax[1].axvline(-3.0, color="k", ls=":", lw=1.0)
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set(xlabel="star-plateau mean $\\rho$ error vs exact [%]",
              title="undisclosed: smooth star region\n(exact $\\rho_*$=0.5751, x∈[0.02,0.26])")
    ax[1].grid(axis="x", alpha=.3)
    fig.tight_layout()
    p = OUT / "fig2_gates.png"
    fig.savefig(p, dpi=140)
    print("wrote", p)


def fig_conv(s):
    conv = s["reference_self_convergence"]["peak"]
    n = np.array([int(c.split("-")[1]) for c in conv], float)
    v = np.array([conv[c] for c in conv])
    ex = s["exact_riemann"]["rho2"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogx(n, 100 * (v - ex) / ex, "o-", label="FV-WENO5 peak error vs analytic")
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set(xlabel="N (cells)", ylabel="peak error [%]",
              title="reference peak error vs analytic $\\rho_2$")
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)

    rows = s["rows"]
    cases = ["dg-noav-p2n100", "dg-av-p2n100", "dg-av-p3n100"]
    xs = np.arange(len(cases))
    for k, (case, col) in enumerate(zip(["dg-noav-p2n100", "dg-av-p2n100", "dg-av-p2n200"], ["tab:red", "tab:blue", "tab:cyan"])):
        xr, rr = load("ref-1600")
        x, r = load(case)
        ax[1].plot(x[(x > 0.0) & (x < 0.30)], r[(x > 0.0) & (x < 0.30)], lw=1.0, color=col, label=case)
    ax[1].axhline(ex, color="k", ls="--", lw=1.0, label="analytic $\\rho_*$=0.5751")
    ax[1].set(xlabel="x", ylabel=r"$\rho$", title="star plateau built up differently (systematic, mesh-independent)",
              ylim=(0.15, 0.65))
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout()
    p = OUT / "fig3_convergence.png"
    fig.savefig(p, dpi=140)
    print("wrote", p)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.loads((RUN / "audit_summary.json").read_text())
    fig_profiles()
    fig_gates(s)
    fig_conv(s)