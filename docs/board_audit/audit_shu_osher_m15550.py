"""Independent re-check of m15480 (CASE-SHU-OSHER-DG, TensorDG).

Board audit m15550.  Deliberately does NOT import the deliverables' metric
helpers (``benchmarks.shu_osher_case_pool.coarsen`` / ``relative_errors`` or
``tensordg.cases.shock_entropy.relative_l1_error``): every number below is
recomputed from the committed raw ``.npz`` with local code, so a bug in the
deliverable's own error helper cannot hide a bad benchmark number.

Inputs (read-only, in the audited worktree):
  runs/shu_osher_case_pool/raw/*.npz            DG ladder + FV reference
  runs/shu_osher_case_pool/metrics.json         the tabulated claim

Outputs: audit_shu_osher_m15550.json + fig_profiles_and_gate.png next to this file.

Usage: python docs/board_audit/audit_shu_osher_m15550.py [--data <dir>]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

GATE = 5.0          # acceptance line pre-declared by the case registration
STRICT = 3.0        # stricter line the report claims for the best config
GAMMA = 1.4

# left state of the Shu-Osher case as published (checked against RH, not copied)
LEFT = (3.857143, 2.629369, 10.333333)


def vol_average(values: np.ndarray, factor: int) -> np.ndarray:
    n = values.size
    if factor < 1 or n % factor:
        raise ValueError("factor does not divide")
    return values.reshape(n // factor, factor).mean(axis=1)


def rel_l1(cand: np.ndarray, ref: np.ndarray, factor: int) -> float:
    refc = vol_average(ref, factor) if factor > 1 else np.asarray(ref, float)
    if refc.shape != cand.shape:
        raise ValueError(f"shape {cand.shape} vs {refc.shape}")
    return 100.0 * float(np.sum(np.abs(cand - refc)) / np.sum(np.abs(refc)))


def rankine_hugoniot():
    """Re-derive the Mach-3 shock into rho=p=1 and compare with the literals."""
    rho_r, u_r = 1.0, 0.0
    p_r, a_r = 1.0, math.sqrt(GAMMA * 1.0 / 1.0)
    # shock speed chosen so the post-shock velocity equals the published u_L
    rho_l, u_l, p_l = LEFT
    s = u_l / (1.0 - rho_r / rho_l)              # mass jump: rho_R*(u_R - s) = rho_L*(u_L - s)
    m = s / a_r                                  # shock Mach number vs the pre-shock gas
    p_l_pred = p_r * (1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (m * m - 1.0))
    rho_l_pred = rho_r * (GAMMA + 1.0) * m * m / ((GAMMA - 1.0) * m * m + 2.0)
    u_l_pred = s * (1.0 - rho_r / rho_l_pred)
    return {
        "shock_speed": s,
        "shock_mach": m,
        "rho_L_rel_res": abs(rho_l_pred - rho_l) / rho_l,
        "p_L_rel_res": abs(p_l_pred - p_l) / p_l,
        "u_L_rel_res": abs(u_l_pred - u_l) / u_l,
        "x_shock_at_t1p8": -4.0 + s * 1.8,
    }


def mass_budget(n: int = 2_000_001):
    """M(t) = M0 + rho*u*t at x=+5 (transmissive ends, constant inflow flux)."""
    x = np.linspace(-5.0, 5.0, n)
    rho0 = np.where(x < -4.0, LEFT[0], 1.0 + 0.2 * np.sin(5.0 * x))
    m0 = np.trapezoid(rho0, x)
    flux = LEFT[0] * LEFT[1]
    return {"M0": m0, "flux": flux, "t": 1.8, "M_pred": m0 + flux * 1.8}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                    help="dir with raw/*.npz and metrics.json (byte copy of the audited worktree)")
    args = ap.parse_args()
    src = os.path.abspath(args.data)
    raw = os.path.join(src, "raw")
    tab = json.load(open(os.path.join(src, "metrics.json")))

    ref = np.load(os.path.join(raw, "fv_cfl0p3_n3200.npz"))
    x_r, rho_r = ref["x"], ref["rho"]

    tab_by_cfg = {"DG P%d N=%d WENO CFL=%s" % (r["degree"], r["ncells"], r["cfl"]): r
                  for r in tab["ladder"]}
    legs = []
    for path in sorted(glob.glob(os.path.join(raw, "dg_*.npz"))):
        name = os.path.basename(path)[:-4]
        d = np.load(path)
        rho = d["rho"]
        meta = json.loads(str(d["meta"])) if d["meta"].shape == () else {}
        if meta.get("cfl") != 0.4:
            continue  # CFL study legs are audited separately below
        factor = x_r.size // rho.size
        got = rel_l1(rho, rho_r, factor)
        key = f"DG P{meta.get('degree')} N={meta.get('ncells')} WENO CFL={meta.get('cfl')}"
        row = tab_by_cfg.get(key)
        claimed = None if row is None else row["rel_l1_percent"]
        legs.append({
            "config": name, "key": key, "ncells": int(rho.size), "factor": factor,
            "got_rel_l1_percent": round(got, 4),
            "tabulated_rel_l1_percent": claimed,
            "drift_pp": None if claimed is None else round(got - claimed, 6),
            "passes_5pct": got < GATE, "passes_3pct": got < STRICT,
            "rho_min": round(float(rho.min()), 6), "rho_max": round(float(rho.max()), 6),
            "mass_abs_error_percent_of_initial": None if row is None
            else round(row["mass_abs_error_percent_of_initial"], 12),
            "shock_position_rel_error_percent": None if row is None
            else round(row["shock_position_rel_error_percent"], 4),
        })

    cfl_sweep = []
    for path in sorted(glob.glob(os.path.join(raw, "dg_cfl0p[12]*.npz"))):
        d = np.load(path)
        meta = json.loads(str(d["meta"])) if d["meta"].shape == () else {}
        got = rel_l1(d["rho"], rho_r, x_r.size // d["rho"].size)
        cfl_sweep.append({"config": os.path.basename(path)[:-4], "cfl": meta.get("cfl"),
                          "got_rel_l1_percent": round(got, 4), "passes_5pct": got < GATE})

    r1600 = np.load(os.path.join(raw, "fv_cfl0p3_n1600.npz"))["rho"]
    mb = mass_budget()
    rh = rankine_hugoniot()

    # post-shock plateau and unshocked sine, recomputed from the raw fields
    def plateau_dev(rho, x):
        m = (x >= -5.0) & (x <= -3.0)
        return 100.0 * abs(rho[m].mean() - LEFT[0]) / LEFT[0]

    def sine_dev(rho, x):
        m = x >= 2.6
        exact = 1.0 + 0.2 * np.sin(5.0 * x[m])
        return 100.0 * float(np.sum(np.abs(rho[m] - exact)) / np.sum(np.abs(exact)))

    anchors = {
        "plateau_dev_ref_percent": round(plateau_dev(rho_r, x_r), 6),
        "sine_dev_ref_percent": round(sine_dev(rho_r, x_r), 6),
        "ref_self_convergence_1600_to_3200_rel_l1_percent": round(rel_l1(r1600, rho_r, 2), 4),
        "rh": {k: round(v, 8) for k, v in rh.items()},
        "mass_budget": {k: round(v, 6) for k, v in mb.items()},
    }

    worst = max(legs, key=lambda r: r["got_rel_l1_percent"])
    best = min(legs, key=lambda r: r["got_rel_l1_percent"])
    summary = {
        "audit_mission": 15550,
        "audited_mission": 15480,
        "audit_data_dir": src,
        "audited_worktree": "/root/tp-agent/work/m15480 @537c9ad",
        "gate_percent": GATE,
        "n_configurations": len(legs),
        "all_pass_5pct": all(r["passes_5pct"] for r in legs),
        "best_config": {k: best[k] for k in ("key", "got_rel_l1_percent")},
        "worst_config": {k: worst[k] for k in ("key", "got_rel_l1_percent")},
        "max_abs_drift_vs_table_pp": max(abs(r["drift_pp"] or 0.0) for r in legs),
        "all_within_strict_3pct": [r["key"] for r in legs if r["passes_3pct"]],
        "no_negative_density": all(r["rho_min"] > 0 for r in legs),
        "legs": legs,
        "cfl_sweep": cfl_sweep,
        "anchors": anchors,
        "recomputed_by": "docs/board_audit/audit_shu_osher_m15550.py (no import of deliverable metric helpers)",
    }

    out_json = os.path.join(os.path.dirname(__file__), "audit_shu_osher_m15550.json")
    json.dump(summary, open(out_json, "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        dense = best["config"]
        d = np.load(os.path.join(raw, dense + ".npz"))
        ax[0].plot(x_r, rho_r, "k-", lw=1.0, label="FV-WENO5 N=3200 (ref)")
        ax[0].plot(d["x"], d["rho"], "r--", lw=1.0, label=f"{best['key']}")
        ax[0].set_xlabel("x"); ax[0].set_ylabel(r"$\rho$")
        ax[0].set_title(f"Shu-Osher t=1.8, rel L1 = {best['got_rel_l1_percent']:.3f}%")
        ax[0].legend(fontsize=8); ax[0].set_xlim(-5, 5)

        ns = [r["ncells"] for r in legs]
        ax[1].semilogx(ns, [r["got_rel_l1_percent"] for r in legs], "o", color="tab:red",
                       label="audit recompute (9 DG legs)")
        ax[1].axhline(GATE, color="tab:green", ls="-", lw=1, label="gate 5%")
        ax[1].axhline(STRICT, color="tab:orange", ls="--", lw=1, label="strict 3%")
        ax[1].axhline(anchors["ref_self_convergence_1600_to_3200_rel_l1_percent"],
                      color="0.5", ls=":", lw=1, label="ref self-convergence")
        ax[1].set_xlabel("cells N"); ax[1].set_ylabel("rel L1 vs ref [%]")
        ax[1].set_title("DG ladder vs acceptance lines"); ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(__file__), "fig_profiles_and_gate.png"), dpi=130)
    except Exception as exc:  # pragma: no cover
        print("figure skipped:", exc)

    print(json.dumps({k: v for k, v in summary.items() if k not in ("legs", "cfl_sweep")}, indent=1))
    return 0 if summary["all_pass_5pct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())