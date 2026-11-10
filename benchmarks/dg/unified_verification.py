#!/usr/bin/env python3
"""Unified advection--diffusion + dispersive-wave verification for TensorDG.

Campaign node P165W3-N05.  Produces

* ``results/dg_unified_verification.json`` -- every number quoted in the report,
* ``figures/*.png`` -- the report figures.

Criteria (pre-declared, thresholds fixed before the runs):

============  ============================================================  ========
id            quantity                                                      limit
============  ============================================================  ========
E1            2-D solid-body rotation, one revolution, relative L2 error      <= 5 %
E2            deformational (cosine-hill) flow, T=1, relative L2 error        <= 5 %
E3            dispersive operator vs ``omega^2 = g k tanh(kh)``, k in [0.1,5] <= 3 %
E4            KdV solitary wave after 10 soliton widths, relative L2 error    <= 5 %
E5            total mass change of the advection(+diffusion) solver          <= 1 %
============  ============================================================  ========

Usage::

    python benchmarks/dg/unified_verification.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from tensordg import advection_diffusion as ad  # noqa: E402
from tensordg import dispersive as ds  # noqa: E402

torch.set_num_threads(min(16, os.cpu_count() or 1))

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(
    HERE, "..", "..", "docs", "benchmarks", "results", "dg-unified-verification")

THRESHOLDS = {"E1_rotation_l2": 0.05, "E2_deformation_l2": 0.05,
              "E3_dispersion": 0.03, "E4_soliton_l2": 0.05, "E5_mass_drift": 0.01}

# declared ("headline") configurations -- fixed before any run
ROT_DECLARED = dict(ncells=24, degree=2, cfl=0.7, revolutions=1.0)
DEF_DECLARED = dict(ncells=24, degree=2, cfl=0.7)
KH_TARGETS = [0.1, 0.5, 1.0, 2.0, 3.5, 5.0]


# --------------------------------------------------------------------------- #
def operator_checks() -> dict:
    """Point-wise consistency of the DG operator on exactly representable data."""
    out = {}
    for degree in (1, 2):
        nx = 6
        ops = ad.Operators(degree, torch.device("cpu"), torch.float64)
        hx, hy, xc, yc, xf, yf = ad._mesh_arrays(nx, nx, ad.UNIT_SQUARE,
                                                 torch.device("cpu"), torch.float64)
        geo = (xc, yc, xf, yf)
        # (a) linear field, constant velocity: residual = -a.grad(u) exactly
        coeff, _ = ad.project(lambda x, y: 1.0 + 2.0 * x + 3.0 * y, nx, nx, degree,
                              ad.UNIT_SQUARE, "cpu", torch.float64)
        r = ad.rhs(coeff, 0.0, ad.translation_velocity((1.0, 0.5)), 0.0, hx, hy, ops,
                   "extrapolate", 2.0, geo)
        rq = torch.einsum("xykl,ak,bl->xyab", r, ops.phi, ops.phi)
        inner = rq[1:-1, 1:-1]
        out[f"advection_linear_field_deg{degree}"] = float(
            torch.max(torch.abs(inner + 3.5)))
        # (b) quadratic field, pure SIPG diffusion: residual = nu*lap(u) exactly
        coeff, _ = ad.project(lambda x, y: 1.0 + 0.3 * x + 0.2 * y + 0.5 * x * x + 0.25 * y * y,
                              nx, nx, max(degree, 2), ad.UNIT_SQUARE, "cpu", torch.float64)
        r = ad.rhs(coeff, 0.0, ad.vanishing_velocity(), 0.1, hx, hy, ops, "extrapolate",
                   2.0, geo) if degree == 2 else None
        if r is not None:
            rq = torch.einsum("xykl,ak,bl->xyab", r, ops.phi, ops.phi)
            out[f"diffusion_quadratic_field_deg{degree}"] = float(
                torch.max(torch.abs(rq[1:-1, 1:-1] - 0.15)))
    # (c) both velocity fields are divergence free (physics precondition)
    xq = torch.linspace(0.0, 1.0, 101, dtype=torch.float64).view(-1, 1)
    yq = torch.linspace(0.0, 1.0, 101, dtype=torch.float64).view(1, -1)
    eps = 1e-5
    div = {}
    for name, vel in (("solid_body_rotation", ad.solid_body_rotation_velocity()),
                      ("deformational_flow", ad.deformational_flow_velocity())):
        for t in (0.0, 0.37):
            axp, _ = vel(t, xq + eps, yq)
            axm, _ = vel(t, xq - eps, yq)
            _, ayp = vel(t, xq, yq + eps)
            _, aym = vel(t, xq, yq - eps)
            d = (axp - axm) / (2 * eps) + (ayp - aym) / (2 * eps)
            div[f"{name}_t{t}"] = float(torch.max(torch.abs(d)))
    out["max_abs_divergence"] = max(div.values())
    out["divergence_detail"] = div
    return out


def rotation_study(quick: bool) -> dict:
    cases = [dict(ncells=12, degree=2), dict(ncells=16, degree=2),
             dict(ncells=24, degree=2)]
    if not quick:
        cases.append(dict(ncells=32, degree=2))
    runs = []
    for cfg in cases:
        t0 = time.time()
        m = ad.run_solid_body_rotation(cfl=0.7, **cfg)
        m["wall_seconds"] = time.time() - t0
        runs.append(m)
        print(f"  rotation {cfg} relL2={m['relative_l2']:.4e} drift={m['mass_drift']:.2e} "
              f"({m['wall_seconds']:.1f}s)", flush=True)
    declared = next(r for r in runs
                    if r["mesh"] == [ROT_DECLARED["ncells"]] * 2
                    and r["degree"] == ROT_DECLARED["degree"])
    quarter = ad.run_solid_body_rotation(ncells=24, degree=2, cfl=0.7, revolutions=0.25)
    return {"runs": runs, "declared": declared, "quarter_revolution": quarter}


def deformation_study(quick: bool) -> dict:
    cases = [dict(ncells=12, degree=2), dict(ncells=16, degree=2),
             dict(ncells=24, degree=2)]
    if not quick:
        cases.append(dict(ncells=32, degree=2))
    runs = []
    for cfg in cases:
        t0 = time.time()
        m = ad.run_deformational_flow(cfl=0.7, **cfg)
        m["wall_seconds"] = time.time() - t0
        runs.append(m)
        print(f"  deformation {cfg} relL2={m['relative_l2']:.4e} drift={m['mass_drift']:.2e} "
              f"({m['wall_seconds']:.1f}s)", flush=True)
    declared = next(r for r in runs
                    if r["mesh"] == [DEF_DECLARED["ncells"]] * 2
                    and r["degree"] == DEF_DECLARED["degree"])
    return {"runs": runs, "declared": declared}


def diffusion_anchor() -> dict:
    """Periodic advection--diffusion of one Fourier mode with an exact solution."""
    runs = []
    for cfg in (dict(ncells=8, degree=2), dict(ncells=16, degree=2), dict(ncells=8, degree=3)):
        m = ad.run_advection_diffusion_decay(nu=0.02, **cfg)
        runs.append(m)
        print(f"  advection-diffusion {cfg} relL2={m['relative_l2']:.3e}", flush=True)
    return {"runs": runs}


def dispersion_study() -> dict:
    ks = torch.linspace(0.1, 5.0, 200, dtype=torch.float64)
    table = {}
    for model in ("airy", "boussinesq", "pade22", "pade66"):
        err = ds.model_dispersion_error(ks, 1.0, ds.GRAVITY, model)
        table[model] = {
            "max_relative_error_omega": err["max_relative_error_omega"],
            "max_relative_error_omega_sq": err["max_relative_error_omega_sq"],
            "validity_kh_3pct": ds.validity_limit(model=model, tolerance=0.03),
            "curve_kh": ks.numpy().tolist(),
            "curve_relative_error_omega": err["relative_error_omega"].tolist(),
        }
    measured_airy = ds.measure_dispersion(KH_TARGETS, depth=1.0, model="airy",
                                          periods=10.0, ncells=256)
    for r in measured_airy:
        print(f"  airy kh={r['kh']:.2f} measured_err={r['relative_error_vs_airy']:.2e}", flush=True)
    measured_pade = ds.measure_dispersion([0.1, 1.0, 3.0, 5.0], depth=1.0,
                                          model="pade66", periods=10.0, ncells=256)
    for r in measured_pade:
        print(f"  pade66 kh={r['kh']:.2f} vs_airy={r['relative_error_vs_airy']:.4f} "
              f"vs_model={r['relative_error_vs_model']:.2e}", flush=True)
    return {"model_table": table, "measured_airy": measured_airy,
            "measured_pade66": measured_pade,
            "kh_targets": KH_TARGETS}


def soliton_study(quick: bool) -> dict:
    runs = []
    for ncells in ((128, 256) if not quick else (64, 128)):
        r = ds.run_soliton_benchmark(amplitude=0.05, depth=1.0, widths=10.0,
                                     ncells=ncells)
        runs.append(r)
        print(f"  soliton N={ncells} relL2={r['relative_l2']:.3e} "
              f"speed_err={r['relative_error_speed_measured_vs_sqrt']:.2e}", flush=True)
    return {"runs": runs, "declared": runs[-1]}


# --------------------------------------------------------------------------- #
def make_figures(outdir: str, data: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    names = []

    # ---- figure 1: solid body rotation ---------------------------------- #
    out, metrics = ad.run_solid_body_rotation(
        ncells=ROT_DECLARED["ncells"], degree=ROT_DECLARED["degree"],
        cfl=ROT_DECLARED["cfl"], return_state=True)
    meta = out["metadata"]
    field = ad.cosine_bell((0.5, 0.75), 0.15)
    xc = (np.arange(meta["nx"]) + 0.5) * meta["hx"]
    yc = (np.arange(meta["ny"]) + 0.5) * meta["hy"]
    initial = ad.project(field, meta["nx"], meta["ny"], meta["degree"], meta["domain"],
                         "cpu", torch.float64)[0][:, :, 0, 0].numpy() * 0.5
    values = out["coeff"][:, :, 0, 0].numpy() * 0.5
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, arr, title in zip(
            axes, (initial, values, np.abs(initial - values)),
            ("initial cosine bell", "t = 1 revolution (DG p=2, 24x24)",
             "|pointwise error| (cell averages)")):
        im = ax.pcolormesh(xc, yc, arr.T, shading="nearest", cmap="viridis")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("E1 solid-body rotation:  omega = 2*pi, T = 1", fontsize=10)
    fig.tight_layout()
    path = os.path.join(figdir, "e1_solid_body_rotation.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    names.append(path)

    # ---- figure 2: deformational flow ----------------------------------- #
    mid = ad.run_deformational_flow(ncells=DEF_DECLARED["ncells"],
                                    degree=DEF_DECLARED["degree"], cfl=0.7,
                                    snapshot_half=True)
    out2, metrics2 = ad.run_deformational_flow(
        ncells=DEF_DECLARED["ncells"], degree=DEF_DECLARED["degree"], cfl=0.7,
        return_state=True)
    meta2 = out2["metadata"]
    initial2 = ad.project(field, meta2["nx"], meta2["ny"], meta2["degree"],
                          meta2["domain"], "cpu", torch.float64)[0][:, :, 0, 0].numpy() * 0.5
    half_vals = mid["half_state_coeff"][:, :, 0, 0].numpy() * 0.5
    vals2 = out2["coeff"][:, :, 0, 0].numpy() * 0.5
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, arr, title in zip(
            axes, (initial2, half_vals, vals2),
            ("initial cosine hill", "t = T/2 (maximum stretching)",
             "t = T (flow reversed)")):
        im = ax.pcolormesh(xc, yc, arr.T, shading="nearest", cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("E2 deformational flow: reversible LeVeque field, T = 1", fontsize=10)
    fig.tight_layout()
    path = os.path.join(figdir, "e2_deformational_flow.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    names.append(path)

    # ---- figure 3: dispersion ------------------------------------------- #
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for model, style in (("airy", dict(color="k", lw=2.2)),
                         ("boussinesq", dict(color="tab:red", lw=1.4)),
                         ("pade22", dict(color="tab:orange", lw=1.4)),
                         ("pade66", dict(color="tab:blue", lw=1.4))):
        entry = data["dispersion"]["model_table"][model]
        ax.semilogy(entry["curve_kh"], np.maximum(entry["curve_relative_error_omega"], 1e-16),
                    label=model, **style)
    ax.axhline(0.03, color="tab:green", ls="--", lw=1.2, label="3 % criterion")
    measured = data["dispersion"]["measured_airy"]
    ax.plot([m["kh"] for m in measured],
            [max(m["relative_error_vs_airy"], 1e-16) for m in measured],
            "kx", ms=7, label="measured (Airy operator, RK4)")
    ax.set_xlabel("kh")
    ax.set_ylabel(r"$|\omega_{model}/\omega_{exact} - 1|$")
    ax.set_title("E3 dispersion:  $\\omega^2 = gk\\tanh(kh)$ vs models", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figdir, "e3_dispersion.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    names.append(path)

    # ---- figure 4: solitary wave ---------------------------------------- #
    sol = data["soliton"]["declared"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = sol["x"].numpy()
    axes[0].plot(x, sol["eta_initial"].numpy(), "--", color="tab:gray",
                 label="initial $a\\,\\mathrm{sech}^2$")
    axes[0].plot(x, sol["eta_final"].numpy(), color="tab:blue", lw=2.0,
                 label="numerical, 10 widths")
    axes[0].plot(x, sol["eta_exact"].numpy(), ":", color="tab:red", lw=2.0,
                 label="exact, translated")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("$\\eta$")
    axes[0].set_title("E4 KdV solitary wave propagation", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].plot(x, (sol["eta_final"] - sol["eta_exact"]).numpy(), color="tab:purple")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("$\\eta_{num} - \\eta_{exact}$")
    axes[1].set_title(f"pointwise error (rel. L2 = {sol['relative_l2']:.2e})", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figdir, "e4_solitary_wave.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    names.append(path)

    # ---- figure 5: mesh refinement -------------------------------------- #
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for key, label, marker in (("rotation", "solid-body rotation", "o"),
                               ("deformation", "deformational flow", "s")):
        runs = sorted(data[key]["runs"], key=lambda r: r["mesh"][0])
        h = [1.0 / r["mesh"][0] for r in runs]
        err = [r["relative_l2"] for r in runs]
        ax.loglog(h, err, marker=marker, label=label)
    ax.axhline(0.05, color="tab:green", ls="--", lw=1.2, label="5 % criterion")
    ax.set_xlabel("h = 1/N")
    ax.set_ylabel("relative $L_2$ error at T")
    ax.set_title("DG p=2 mesh refinement (T = 1)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(figdir, "mesh_refinement.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    names.append(path)

    return names


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    started = time.time()
    print("== operator consistency ==", flush=True)
    operators = operator_checks()
    print(json.dumps(operators, indent=2)[:800], flush=True)

    print("== E1 solid-body rotation ==", flush=True)
    rotation = rotation_study(args.quick)
    print("== E2 deformational flow ==", flush=True)
    deformation = deformation_study(args.quick)
    print("== advection-diffusion anchor ==", flush=True)
    diffusion = diffusion_anchor()
    print("== E3 dispersion ==", flush=True)
    dispersion = dispersion_study()
    print("== E4 solitary wave ==", flush=True)
    soliton = soliton_study(args.quick)

    gates = {
        "E1_rotation_l2": {"value": rotation["declared"]["relative_l2"],
                           "limit": THRESHOLDS["E1_rotation_l2"],
                           "pass": rotation["declared"]["relative_l2"] <= THRESHOLDS["E1_rotation_l2"]},
        "E2_deformation_l2": {"value": deformation["declared"]["relative_l2"],
                              "limit": THRESHOLDS["E2_deformation_l2"],
                              "pass": deformation["declared"]["relative_l2"] <= THRESHOLDS["E2_deformation_l2"]},
        "E3_dispersion": {
            "value": max(max(m["relative_error_vs_airy"] for m in dispersion["measured_airy"]),
                         dispersion["model_table"]["pade66"]["max_relative_error_omega"]),
            "value_airy_measured": max(m["relative_error_vs_airy"]
                                       for m in dispersion["measured_airy"]),
            "limit": THRESHOLDS["E3_dispersion"],
            "pass": (max(m["relative_error_vs_airy"] for m in dispersion["measured_airy"]) <= THRESHOLDS["E3_dispersion"]
                     and dispersion["model_table"]["pade66"]["max_relative_error_omega"] <= THRESHOLDS["E3_dispersion"]),
        },
        "E4_soliton_l2": {"value": soliton["declared"]["relative_l2"],
                          "limit": THRESHOLDS["E4_soliton_l2"],
                          "pass": soliton["declared"]["relative_l2"] <= THRESHOLDS["E4_soliton_l2"]},
        "E5_mass_drift": {"value": max(rotation["declared"]["mass_drift"],
                                       deformation["declared"]["mass_drift"]),
                          "limit": THRESHOLDS["E5_mass_drift"],
                          "pass": max(rotation["declared"]["mass_drift"],
                                      deformation["declared"]["mass_drift"]) <= THRESHOLDS["E5_mass_drift"]},
    }

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()
                    if k not in ("eta_final", "eta_exact", "eta_initial", "x",
                                 "half_state_coeff", "half_state_metadata")}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    data = {
        "node": "P165W3-N05",
        "repository": "TensorDG",
        "modules": ["tensordg/advection_diffusion.py", "tensordg/dispersive.py"],
        "thresholds": THRESHOLDS,
        "declared_configs": {"rotation": ROT_DECLARED, "deformation": DEF_DECLARED,
                             "kh_targets": KH_TARGETS},
        "operator_checks": operators,
        "rotation": rotation,
        "deformation": deformation,
        "advection_diffusion": diffusion,
        "dispersion": dispersion,
        "soliton": soliton,
        "gates": gates,
        "wall_seconds": time.time() - started,
        "seeds": "deterministic (no random numbers)",
        "environment": {"torch": torch.__version__, "numpy": np.__version__,
                        "threads": torch.get_num_threads()},
    }

    print("== figures ==", flush=True)
    figures = make_figures(outdir, data)
    data["figures"] = [os.path.relpath(p, outdir) for p in figures]

    payload = clean(data)
    payload["soliton"]["declared"].update({
        "peak_retention": soliton["declared"]["peak_retention"],
        "speed_measured": soliton["declared"]["speed_measured"],
    })
    path = os.path.join(outdir, "dg_unified_verification.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
    print(f"wrote {path}", flush=True)
    print("gates:", json.dumps({k: (v["value"], v["pass"]) for k, v in gates.items()},
                              indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()