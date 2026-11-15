#!/usr/bin/env python3
"""Law check (trimmed) + report figures for the Poiseuille baseline skeleton.

One pass: cheap-members evidence for the closed-form error law, written to
``outputs/xval_poiseuille_baseline/law_check.json``, then the three report
figures rendered from ``summary.json`` + ``law_check.json``.

Budgets: residual tolerances are chosen so each run is a *converged* sample
(the discretisation error is O(1e-3), so 1e-7..1e-8 on the velocity residual is
two orders below what is being measured); the only 1e-12 run is the slip-free
tau* case, where round-off is the claim under test.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.poiseuille_baseline import (  # noqa: E402
    BASELINE_SPEC, PoiseuilleConfig, ReL2Gate, analytic_profile, run,
)

OUT = ROOT / "outputs" / "xval_poiseuille_baseline"
OUT.mkdir(parents=True, exist_ok=True)
TAU_STAR = 0.5 + 3.0 / math.sqrt(48.0)


def slip_law(tau: float, width: float) -> float:
    nu = (tau - 0.5) / 3.0
    return (48.0 * nu * nu - 1.0) / width ** 2


def law_check() -> dict:
    t0 = time.time()
    widths = []
    for Ny, tol in ((8, 1e-8), (16, 1e-8)):
        cfg = PoiseuilleConfig(Nx=4, Ny=Ny, tau=0.8, init="analytic",
                               residual_tol=tol, max_steps=60_000, check_every=100)
        res = run(cfg)
        law = slip_law(cfg.tau, cfg.width)
        widths.append({
            "Ny": Ny, "width": cfg.width, "nu": cfg.nu, "residual_tol": tol,
            "rel_l2": res["errors"]["rel_l2"],
            "rel_l2_shape": res["errors"]["rel_l2_shape"],
            "offset_over_force": res["errors"]["offset_over_force"],
            "law_signed": law, "law_abs": abs(law),
            "ratio_to_law": res["errors"]["rel_l2"] / abs(law),
            "steps": res["convergence"]["steps"],
        })
        print(f"  width Ny={Ny} relL2={widths[-1]['rel_l2']:.4e} law={abs(law):.4e}",
              flush=True)

    taus = []
    for tau, tol in ((0.6, 1e-7), (0.8, 1e-7), (1.0, 1e-8), (1.5, 1e-8)):
        cfg = PoiseuilleConfig(Nx=4, Ny=16, tau=tau, init="analytic",
                               residual_tol=tol, max_steps=60_000, check_every=100)
        res = run(cfg)
        law = slip_law(tau, cfg.width)
        taus.append({
            "tau": tau, "nu": cfg.nu, "residual_tol": tol,
            "rel_l2": res["errors"]["rel_l2"],
            "rel_l2_shape": res["errors"]["rel_l2_shape"],
            "offset_over_force": res["errors"]["offset_over_force"],
            "closed_form_offset_over_force": 6.0 * cfg.nu - 1.0 / (8.0 * cfg.nu),
            "law_signed": law, "law_abs": abs(law),
            "ratio_to_law": res["errors"]["rel_l2"] / abs(law),
            "steps": res["convergence"]["steps"],
        })
        print(f"  tau={tau} relL2={taus[-1]['rel_l2']:.4e} law={abs(law):.4e}",
              flush=True)

    star = []
    star_profile = None
    for Ny in (8, 16):
        cfg = PoiseuilleConfig(Nx=4, Ny=Ny, tau=TAU_STAR, init="analytic",
                               residual_tol=1e-12, max_steps=60_000, check_every=100)
        res = run(cfg)
        ana = analytic_profile(cfg)
        pointwise = (torch.tensor(res["profile_u"]) - ana).abs().tolist()
        star.append({"Ny": Ny, "tau": TAU_STAR, "nu": cfg.nu,
                     "rel_l2": res["errors"]["rel_l2"],
                     "rel_linf": res["errors"]["rel_linf"],
                     "rel_l2_shape": res["errors"]["rel_l2_shape"],
                     "rel_u_max": res["errors"]["rel_u_max"],
                     "steps": res["convergence"]["steps"],
                     "final_residual": res["convergence"]["final_residual"],
                     "pointwise_abs_diff": pointwise,
                     "profile_u": res["profile_u"],
                     "reference_u": res["reference_u"]})
        if Ny == 16:
            star_profile = star[-1]
        print(f"  tau* Ny={Ny} relL2={star[-1]['rel_l2']:.3e}", flush=True)

    common = dict(Nx=4, Ny=8, tau=1.0, residual_tol=1e-10, max_steps=60_000,
                  check_every=50)
    a = run(PoiseuilleConfig(init="rest", **common))
    b = run(PoiseuilleConfig(init="analytic", **common))
    pa = torch.tensor(a["profile_u"])
    pb = torch.tensor(b["profile_u"])
    invariants = {
        "ic_rest_steps": a["convergence"]["steps"],
        "ic_analytic_steps": b["convergence"]["steps"],
        "rel_max_profile_diff": float((pa - pb).abs().max())
        / a["errors"]["u_max_analytic"],
        "mass_drift_rest": a["mass_drift"],
        "mass_drift_analytic": b["mass_drift"],
    }

    gate = ReL2Gate(tolerance=BASELINE_SPEC["gate"]["tolerance"])
    checks = {f"tau{row['tau']}": gate.check(row["rel_l2"]).as_dict() for row in taus}
    checks.update({f"tau_star_Ny{row['Ny']}": gate.check(row["rel_l2"]).as_dict()
                   for row in star})
    checks.update({f"width_Ny{row['Ny']}": gate.check(row["rel_l2"]).as_dict()
                   for row in widths})
    return {
        "description": "rel L2 of the converged profile follows the signed closed "
                       "form (48 nu^2 - 1) / L^2; tau* = 1/2 + 3/sqrt(48) makes "
                       "the residual round-off only",
        "tau_star": TAU_STAR,
        "width_ladder": widths,
        "tau_ladder": taus,
        "slip_free": [{k: v for k, v in row.items()
                       if k not in ("profile_u", "reference_u")} for row in star],
        "slip_free_profile_Ny16": {"profile_u": star_profile["profile_u"],
                                   "reference_u": star_profile["reference_u"]},
        "invariants": invariants,
        "gate": {"tolerance": gate.tolerance, "basis": gate.basis,
                 "checks": checks,
                 "n_pass": sum(1 for v in checks.values() if v["passed"]),
                 "n_total": len(checks)},
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def figures(law: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = json.loads((OUT / "summary.json").read_text())
    made = []

    legacy = spec["A_delivery_replica"]["result"]
    if "profile_u" not in legacy:                 # summary keeps slim records
        legacy = run(PoiseuilleConfig(Nx=64, Ny=32, tau=0.8, forcing="simple",
                                      wall_origin=0.0, init="rest", max_steps=2000,
                                      check_every=2000, residual_tol=0.0))
    cfg_base = PoiseuilleConfig(**{**BASELINE_SPEC["canonical_case"], "Nx": 8})
    ana = analytic_profile(cfg_base).tolist()
    y = list(range(1, cfg_base.Ny - 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot([ana[i] for i in y], y, "k-", lw=2, label="analytic (half-way walls)")
    ax.plot([legacy["profile_u"][i] for i in y], y, "o-", ms=3, lw=1,
            color="tab:red", label="delivered config replica (2000 steps)")
    ax.set_xlabel("u_x (lattice units)")
    ax.set_ylabel("y (lattice node)")
    ax.set_title(f"delivered case replica: rel-L2 = "
                 f"{legacy['errors']['rel_l2']:.1%} at 2000 steps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    taus = law["tau_ladder"]
    t = [r["tau"] for r in taus]
    ax.semilogy(t, [r["rel_l2"] for r in taus], "o-", label="measured rel-L2")
    ax.semilogy(t, [r["law_abs"] for r in taus], "k--", label=r"$|48\nu^2-1|/L^2$")
    ax.axhline(BASELINE_SPEC["gate"]["tolerance"], color="tab:green", ls=":",
               label="pre-registered gate 1%")
    ax.axhline(0.15, color="tab:gray", ls=":", label="ecosystem cap 15%")
    ax.axvline(TAU_STAR, color="tab:purple", ls="-.", lw=1,
               label=r"$\tau^*$ slip-free")
    ax.set_xlabel(r"BGK relaxation time $\tau$  (Ny=16, L=14)")
    ax.set_ylabel("relative L2 error vs analytic")
    ax.set_title("error law reproduced across viscosity")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_poiseuille_baseline.png", dpi=140)
    plt.close(fig)
    made.append("fig_poiseuille_baseline.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ws = law["width_ladder"]
    ax.loglog([r["width"] for r in ws], [r["rel_l2"] for r in ws], "o-",
              label="measured rel-L2 (tau=0.8)")
    ax.loglog([r["width"] for r in ws], [r["law_abs"] for r in ws], "k--",
              label=r"$|48\nu^2-1|/L^2$")
    for r in ws:
        ax.annotate(f"Ny={r['Ny']}", (r["width"], r["rel_l2"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("channel width L = Ny - 2 (lattice units)")
    ax.set_ylabel("relative L2 error")
    ax.set_title(r"$L^{-2}$ decay with an $O(\nu^2)$ slip floor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    sp = law["slip_free_profile_Ny16"]
    prof = torch.tensor(sp["profile_u"])
    ref = torch.tensor(sp["reference_u"])
    diff = (prof - ref)[1:-1]
    ax.semilogy(list(range(1, len(prof) - 1)),
                torch.clamp(diff.abs(), min=1e-17).tolist(), "o-", ms=3,
                label="|u_num - u_analytic|, Ny=16")
    ax.axhline(1e-15, color="gray", ls=":", label="float64 round-off scale")
    ax.set_xlabel("y (lattice node)")
    ax.set_ylabel("pointwise |difference| (lattice units)")
    ax.set_title(r"$\tau^*=0.5+3/\sqrt{48}$: agreement at round-off")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_poiseuille_law.png", dpi=140)
    plt.close(fig)
    made.append("fig_poiseuille_law.png")
    return made


def main() -> int:
    law = law_check()
    (OUT / "law_check.json").write_text(json.dumps(law, indent=1))
    print("wrote", OUT / "law_check.json", flush=True)
    try:
        made = figures(law)
    except Exception as exc:                      # figures are not evidence-critical
        made = [f"figure generation failed: {exc!r}"]
    print("figures:", made, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())