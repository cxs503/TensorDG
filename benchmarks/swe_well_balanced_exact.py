"""Evidence generator: exact lake-at-rest on a curved bed (``well_balanced=True``).

Reproduces every number quoted in ``docs/swe_well_balanced_exact.md`` and the
evidence file ``artifacts/swe_well_balanced_exact.json`` plus one figure.

Run:  python -m benchmarks.swe_well_balanced_exact
"""
from __future__ import annotations

import json
import math
import os

import torch

from tensordg import swe_exact
from tensordg.swe_1d import reconstruct_fields, solve_swe_1d

ETA0 = 2.0
SLOPE = 0.4
OUT_JSON = "artifacts/swe_well_balanced_exact.json"
OUT_PNG = "docs/figures/swe_well_balanced_drift.png"


def bed(x):
    return SLOPE * torch.sin(0.5 * math.pi * x)


def lake_at_rest(ncells, degree, well_balanced, use_poly_bed=True):
    x, coeff, zb_faces, meta = solve_swe_1d(
        lambda xx: ETA0 - bed(xx), lambda xx: torch.zeros_like(xx), bed,
        ncells=ncells, degree=degree, final_time=1.0, domain=(-2.0, 2.0),
        cfl=0.3, limiter=None, rk_order=3, well_balanced=well_balanced)
    assert meta["reached_final_time"]
    zb_center = meta["zb_modes"][:, 0] if (meta["well_balanced"] and use_poly_bed) else None
    h, u, eta = reconstruct_fields(coeff, zb_faces, zb_center=zb_center)
    return {
        "ncells": ncells, "degree": degree, "well_balanced": bool(well_balanced),
        "post_processing_bed": "polynomial_average" if zb_center is not None else "trapezoidal",
        "eta_drift_max_abs_m": float((eta - ETA0).abs().max()),
        "eta_drift_relative_to_depth": float((eta - ETA0).abs().max()) / ETA0,
        "spurious_momentum_max_abs": float(coeff[:, :, 1].abs().max()),
        "steps": meta["steps"], "t_reached": meta["time_reached"],
    }


def thacker(degree, well_balanced, ncells=20, t_end=0.1):
    g, h0, amp, length = 9.81, 1.0, 0.5, 1.0

    def init_h(x):
        h, _ = swe_exact.thacker_planar_basin_initial(x.numpy(), h0, amp, length)
        return torch.as_tensor(h, dtype=x.dtype)

    def bedf(x):
        return torch.as_tensor(swe_exact.thacker_parabolic_bed(x.numpy(), h0, length),
                               dtype=x.dtype)

    x, coeff, zb_faces, meta = solve_swe_1d(
        init_h, lambda z: torch.zeros_like(z), bedf, ncells=ncells, degree=degree,
        final_time=t_end, domain=(-length, length), cfl=0.3, rk_order=3,
        well_balanced=well_balanced)
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    h_ex, u_ex, eta_ex = swe_exact.thacker_planar_basin(x.numpy(), t_end, h0, amp, length, g)
    h_ex = torch.as_tensor(h_ex); u_ex = torch.as_tensor(u_ex); eta_ex = torch.as_tensor(eta_ex)

    def rel(a, b):
        return float(torch.sqrt(torch.mean((a - b) ** 2)) / b.abs().max()) * 100.0

    return {"degree": degree, "well_balanced": bool(well_balanced), "ncells": ncells,
            "t_end": t_end, "eta_rel_l2_percent": rel(eta, eta_ex),
            "u_rel_l2_percent": rel(u, u_ex), "h_rel_l2_percent": rel(h, h_ex)}


def main():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)
    meshes = [int(v) for v in os.environ.get("WB_MESHES", "32,64,128").split(",")]
    rows = []
    for degree in (1, 2):
        for ncells in meshes:
            rows.append(lake_at_rest(ncells, degree, True))
            rows.append(lake_at_rest(ncells, degree, False))
    p2_trapz = [lake_at_rest(n, 2, True, use_poly_bed=False) for n in meshes]
    thacker_rows = [thacker(d, wb) for d in (1, 2) for wb in (False, True)]

    evidence = {
        "case": "lake at rest, eta0=2 m, z_b=0.4 sin(pi x/2), x in [-2,2], t=1 s, RK3, HLL, cfl=0.3",
        "lake_at_rest": rows,
        "p2_read_with_trapezoidal_bed_average": p2_trapz,
        "thacker_planar_basin_t0.1": thacker_rows,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(evidence, fh, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for degree, style in ((1, "o-"), (2, "s-")):
            for wb, colour in ((False, "tab:red"), (True, "tab:blue")):
                sel = [r for r in rows if r["degree"] == degree and r["well_balanced"] == wb]
                axes[0].loglog([r["ncells"] for r in sel],
                               [max(r["eta_drift_max_abs_m"], 1e-16) for r in sel],
                               style, color=colour,
                               label=f"P{degree} {'nodal (well_balanced)' if wb else 'L2 default'}")
        axes[0].axhline(1e-12, color="k", ls=":", lw=0.8, label="round-off")
        axes[0].set_xlabel("cells"); axes[0].set_ylabel(r"$\max|\eta-\eta_0|$ [m]")
        axes[0].set_title("Lake at rest, curved bed, t = 1 s")
        axes[0].legend(fontsize=7); axes[0].grid(True, which="both", alpha=0.3)

        sel = [r for r in rows if r["degree"] == 2 and r["well_balanced"]]
        axes[1].loglog([r["ncells"] for r in sel],
                       [r["spurious_momentum_max_abs"] for r in sel], "o-",
                       label="nodal: momentum")
        selw = [r for r in rows if r["degree"] == 2 and not r["well_balanced"]]
        axes[1].loglog([r["ncells"] for r in selw],
                       [r["spurious_momentum_max_abs"] for r in selw], "s-",
                       color="tab:red", label="L2 default: momentum")
        axes[1].set_xlabel("cells"); axes[1].set_ylabel(r"$\max|(hu)_k|$ [m$^2$/s]")
        axes[1].set_title("Spurious momentum, P2")
        axes[1].legend(fontsize=8); axes[1].grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=150)
        print("figure ->", OUT_PNG)
    except Exception as exc:  # pragma: no cover
        print("figure skipped:", exc)

    print(json.dumps({k: v for k, v in evidence.items() if k != "lake_at_rest"}, indent=2))
    for r in rows:
        print(f"p{r['degree']} N{r['ncells']} wb={r['well_balanced']}: "
              f"drift={r['eta_drift_max_abs_m']:.3e} m mom={r['spurious_momentum_max_abs']:.3e}")
    print("evidence ->", OUT_JSON)


if __name__ == "__main__":
    main()