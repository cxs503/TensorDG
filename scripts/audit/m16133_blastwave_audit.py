#!/usr/bin/env python3
"""Board audit of m16072 / CASE-DG-BLASTWAVE-PERSSON (TensorDG).

Recomputes every metric *from the delivered raw artifacts* (``validation/
blast_wave_av_1d/raw/<case>.npz``) with independently written code -- the
author's ``summary.json`` is only compared against, never trusted.

Two overshoot definitions are reported because the case-pool tolerance says
"no Gibbs overshoot > 2%" and the author's C3 measures peak-vs-reference-peak,
which conflates a sub-cell spike with a global peak error:

  D1 (author's C3): (peak_dg - peak_ref) / peak_ref
  D2 (spike-local): (peak_dg - rho_ref(peak_x_dg)) / rho_ref(peak_x_dg)
      -- how far the DG peak rises above what the fine reference has *at the
         same abscissa*; a pure sub-cell Gibbs spike shows up here and not in
         the L1 error.

Usage:  python scripts/audit/m16133_blastwave_audit.py [--src DIR] [--out JSON]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

GAMMA = 1.4
T_FINAL = 0.01
X0 = 0.1            # driver / ambient interface at t=0
AMBIENT_A = float(np.sqrt(GAMMA * 0.01 / 1.0))   # p=0.01, rho=1  -> 0.11832
TOL_PEAK_PCT = 5.0  # tensor_case_pool tol_policy (DG-BLASTWAVE-PERSSON)
TOL_FRONT_PCT = 5.0
TOL_OVER_PCT = 2.0


def load(raw: Path, case: str):
    z = np.load(raw / f"{case}.npz")
    meta = json.loads((raw / f"{case}.json").read_text())
    return z["x"], z["rho"], meta.get("metrics", {}), meta


def front_index(x: np.ndarray, rho: np.ndarray) -> int:
    return int(np.argmax(np.abs(np.gradient(rho, x))))


def metrics(x, rho):
    fi = front_index(x, rho)
    pi = int(np.argmax(rho))
    return {
        "peak": float(rho.max()),
        "peak_x": float(x[pi]),
        "front": float(x[fi]),
        "rho_min": float(rho.min()),
        "travel": float(x[fi] - X0),
    }


def rel_l1(x, rho, ref_x, ref_rho) -> float:
    r = np.interp(x, ref_x, ref_rho)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    denom = float(trapz(np.abs(r), x))
    return 100.0 * float(trapz(np.abs(rho - r), x)) / denom


def total_variation(rho: np.ndarray) -> float:
    return float(np.abs(np.diff(rho)).sum())


def interior_extrema(rho: np.ndarray) -> int:
    d = np.diff(rho)
    s = np.sign(d)
    s = s[s != 0]
    return int((np.diff(s) != 0).sum())


def richardson(v1, v2, v3, r=2.0):
    """Observed order and extrapolated limit from three nested grids."""
    d21, d32 = v2 - v1, v3 - v2
    if d32 == 0:
        return float("nan"), v3
    p = float(np.log(abs(d21 / d32)) / np.log(r))
    lim = v3 + d32 / (r ** p - 1.0)
    return p, float(lim)


def rh_check(rho1, p1, u1, rho2, p2, u2, gamma=GAMMA):
    """Rankine-Hugoniot flux residuals across a stationary shock (speed D)."""
    out = {}
    for name, (rl, pl, ul), (rr, pr, ur) in {
        "upstream/downstream": ((rho1, p1, u1), (rho2, p2, u2)),
    }.items():
        D = (rr * ur - rl * ul) / (rr - rl)
        jm = rl * (ul - D) - rr * (ur - D)
        jp = (pl + rl * (ul - D) ** 2) - (pr + rr * (ur - D) ** 2)
        je = (ul * (0.5 * rl * (ul - D) ** 2 + pl / (gamma - 1))
              - ur * (0.5 * rr * (ur - D) ** 2 + pr / (gamma - 1))) \
            - D * (0.5 * rl * (ul - D) ** 2 + pl / (gamma - 1)
                   - 0.5 * rr * (ur - D) ** 2 - pr / (gamma - 1))
        scale_m = max(abs(rl * ul), abs(rr * ur), 1e-30)
        scale_p = max(abs(pl), abs(pr), 1e-30)
        out[name] = {
            "D": float(D),
            "mass_resid_rel": float(abs(jm) / max(scale_m, abs(rr * ur), 1e-30)),
            "mom_resid_rel": float(abs(jp) / scale_p),
            "ener_resid_rel": float(abs(je) / scale_p),
        }
    return out



def exact_riemann(rho4=1.0, p4=1000.0, rho1=1.0, p1=0.01, gamma=GAMMA):
    """Exact self-similar Riemann solution (Toro, *Riemann Solvers and Numerical
    Methods for Fluid Dynamics*, ch. 4) for the blast-wave initial data.

    Driver (left) state is rarefied, ambient (right) state is shocked; the star
    pressure is found by bisection on f_L(p) + f_R(p) = 0.  Returned values are
    the analytic truth used to audit both the DG run and the FV-WENO5 reference:
    ``rho_star_L`` (density on the driver side of the contact -- the uniform
    "star" plateau), ``rho2`` (post-shock density, the analytic maximum) and
    ``D`` (shock speed, hence the analytic front position).
    """
    a4 = math.sqrt(gamma * p4 / rho4)
    a1 = math.sqrt(gamma * p1 / rho1)
    A4, B4 = 2.0 / ((gamma + 1) * rho4), (gamma - 1) / (gamma + 1)
    A1, B1 = 2.0 / ((gamma + 1) * rho1), (gamma - 1) / (gamma + 1)

    def fL(p):
        if p <= p4:
            return 2 * a4 / (gamma - 1) * ((p / p4) ** ((gamma - 1) / (2 * gamma)) - 1)
        return (p - p4) * math.sqrt(A4 / (p + B4 * p4))

    def fR(p):
        if p <= p1:
            return 2 * a1 / (gamma - 1) * ((p / p1) ** ((gamma - 1) / (2 * gamma)) - 1)
        return (p - p1) * math.sqrt(A1 / (p + B1 * p1))

    lo, hi = 1e-8, 1e8
    for _ in range(400):
        mid = math.sqrt(lo * hi)
        if fL(mid) + fR(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    ps = math.sqrt(lo * hi)
    us = 0.5 * (fR(ps) - fL(ps))
    rho_star_L = rho4 * (ps / p4) ** (1.0 / gamma)
    pr = ps / p1
    rho2 = rho1 * (pr + B1) / (pr * B1 + 1.0)
    D = rho2 * us / (rho2 - rho1)
    return {"p_star": ps, "u_star": us, "rho_star_L": rho_star_L, "rho2": rho2,
            "D": D, "a4": a4, "a1": a1, "mach": D / a1,
            "x_shock_t": X0 + D * T_FINAL, "x_contact_t": X0 + us * T_FINAL}


def star_region(x, rho, exact_rho, lo=0.02, hi=0.26):
    """Mean density / deviation in the uniform star plateau (driver side).

    ``lo``/``hi`` stay clear of both the outflow boundary cell and the contact
    surface, where the exact solution is *not* uniform.
    """
    m = (x >= lo) & (x <= hi)
    return {
        "mean_rho": float(rho[m].mean()),
        "exact_rho": exact_rho,
        "err_pct": 100.0 * float(rho[m].mean() - exact_rho) / exact_rho,
        "max_abs_dev": float(np.abs(rho[m] - exact_rho).max()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/tp-agent/work/m16072/validation/blast_wave_av_1d")
    ap.add_argument("--out", default="/root/tp-agent/work/m16133/runs/m16133_audit/audit_summary.json")
    ap.add_argument("--repro", default="/root/tp-agent/work/m16133/runs/m16133_audit/repro")
    args = ap.parse_args(argv)
    src = Path(args.src)
    raw = src / "raw"
    summary = json.loads((src / "summary.json").read_text())
    author = {r["case"]: r for r in summary["rows"]}

    cases = sorted(p.stem for p in raw.glob("*.npz"))
    ref_case = summary["fine_reference"]
    ref_x, ref_rho, ref_meta, _ = load(raw, ref_case)
    ref_m = metrics(ref_x, ref_rho)

    rows = []
    for c in cases:
        x, rho, meta_m, meta = load(raw, c)
        m = metrics(x, rho)
        d1 = 100.0 * (m["peak"] - ref_m["peak"]) / ref_m["peak"]
        d2 = 100.0 * (m["peak"] - float(np.interp(m["peak_x"], ref_x, ref_rho))) \
            / float(np.interp(m["peak_x"], ref_x, ref_rho))
        fe = 100.0 * (m["front"] - ref_m["front"]) / ref_m["travel"]
        win = (x >= 0.28) & (x <= 0.36)
        row = {
            "case": c,
            "kind": meta.get("config", {}).get("kind", "dg"),
            "degree": meta.get("config", {}).get("degree"),
            "ncells": meta.get("config", {}).get("ncells"),
            "av": meta.get("config", {}).get("av"),
            "limiter": meta.get("config", {}).get("limiter"),
            "peak": m["peak"], "peak_x": m["peak_x"], "front": m["front"],
            "front_excess_pct": fe,
            "undershoot": m["rho_min"],
            "D1_peak_err_pct": d1,
            "D2_spike_local_pct": d2,
            "L1_rel_pct": rel_l1(x, rho, ref_x, ref_rho),
            "max_pointwise_dev_pct": 100.0 * float(
                np.max(np.abs(rho - np.interp(x, ref_x, ref_rho))
                       / np.maximum(np.interp(x, ref_x, ref_rho), 1e-12))),
            "TV_shock_window": total_variation(rho[win]),
            "n_extrema_shock_window": interior_extrema(rho[win]),
            "steps": meta.get("steps"),
            "wall_time_s": meta.get("wall_time_s"),
            "n_samples": int(x.size),
        }
        if c != ref_case:
            row["author_peak_err_pct"] = author.get(c, {}).get("peak_err_pct")
            row["author_l1_err_pct"] = author.get(c, {}).get("l1_err_pct")
            row["author_overshoot_pct"] = author.get(c, {}).get("overshoot_pct")
        rows.append(row)

    ref_win = (ref_x >= 0.28) & (ref_x <= 0.36)
    gates = {}
    for row in rows:
        if row["case"] == ref_case:
            continue
        gates[row["case"]] = {
            "C1_peak_lt5": abs(row["D1_peak_err_pct"]) < TOL_PEAK_PCT,
            "C2_front_lt5": abs(row["front_excess_pct"]) < TOL_FRONT_PCT,
            "C3_overshoot_lt2_D1": abs(row["D1_peak_err_pct"]) < TOL_OVER_PCT,
            "C3p_overshoot_lt2_D2": abs(row["D2_spike_local_pct"]) < TOL_OVER_PCT,
        }

    # reference self-convergence / uncertainty
    refs = {}
    for c in cases:
        if not c.startswith("ref-"):
            continue
        x, rho, _, _ = load(raw, c)
        m = metrics(x, rho)
        refs[c] = m
    ref_conv = {"peak": {}, "front": {}}
    for c, m in refs.items():
        ref_conv["peak"][c] = m["peak"]
        ref_conv["front"][c] = m["front"]
    order = {}
    lim = {}
    for key in ("peak", "front"):
        vals = [refs[c][key] for c in sorted(refs, key=lambda s: int(s.split("-")[1]))]
        order[key], lim[key] = richardson(*vals)
    ref_unc_pct = {
        "peak": 100.0 * abs(refs["ref-800"]["peak"] - refs["ref-1600"]["peak"]) / refs["ref-1600"]["peak"],
        "front": 100.0 * abs(refs["ref-800"]["front"] - refs["ref-1600"]["front"]) / ref_m["travel"],
    }

    # ---- analytic truth: exact self-similar Riemann solution -----------------
    exact = exact_riemann()
    for row in rows:
        x, rho, _, _ = load(raw, row["case"])
        row["star"] = star_region(x, rho, exact["rho_star_L"])
        row["D1_vs_exact_pct"] = 100.0 * (row["peak"] - exact["rho2"]) / exact["rho2"]
        row["front_vs_exact_pct"] = 100.0 * (row["front"] - exact["x_shock_t"]) / (
            exact["x_shock_t"] - X0)
    exact_gates = {}
    for row in rows:
        exact_gates[row["case"]] = {
            "C1_peak_lt5_vs_exact": abs(row["D1_vs_exact_pct"]) < TOL_PEAK_PCT,
            "C2_front_lt5_vs_exact": abs(row["front_vs_exact_pct"]) < TOL_FRONT_PCT,
            "C3_overshoot_lt2_vs_exact": row["D1_vs_exact_pct"] < TOL_OVER_PCT,
        }

    # ---- audit-side rerun of one case (determinism / reproducibility) --------
    repro = None
    rp = Path(args.repro)
    if rp.exists():
        rj = json.loads((rp / "dg-av-p2n100.json").read_text())
        ra = np.load(rp / "dg-av-p2n100.npz")
        da = np.load(raw / "dg-av-p2n100.npz")
        ma = metrics(da["x"], da["rho"])
        mb = metrics(ra["x"], ra["rho"])
        repro = {
            "case": "dg-av-p2n100",
            "delivered": {k: ma[k] for k in ("peak", "front", "rho_min")},
            "audit_rerun": {k: mb[k] for k in ("peak", "front", "rho_min")},
            "peak_rel_diff_pct": 100.0 * (mb["peak"] - ma["peak"]) / ma["peak"],
            "max_abs_rho_diff": float(np.abs(ra["rho"] - da["rho"]).max()),
            "argmax_at_x": float(da["x"][int(np.argmax(np.abs(ra["rho"] - da["rho"])))]),
            "star_mean_delivered": float(da["rho"][(da["x"] >= 0.02) & (da["x"] <= 0.26)].mean()),
            "star_mean_rerun": float(ra["rho"][(ra["x"] >= 0.02) & (ra["x"] <= 0.26)].mean()),
            "bitwise_identical": bool(np.array_equal(ra["rho"], da["rho"])),
            "gate_margin_vs_rerun_scatter_pp": TOL_OVER_PCT - max(
                100.0 * (ma["peak"] - ref_m["peak"]) / ref_m["peak"],
                100.0 * (mb["peak"] - ref_m["peak"]) / ref_m["peak"]),
        }

    # analytic Rankine-Hugoniot anchor from the measured shock speed
    D_meas = ref_m["travel"] / T_FINAL
    M = D_meas / AMBIENT_A
    rh_ratio = (GAMMA + 1) * M * M / ((GAMMA - 1) * M * M + 2)
    rh_p_ratio = 1.0 + 2 * GAMMA * (M * M - 1) / (GAMMA + 1)
    analytic = {
        "D_measured": D_meas,
        "shock_mach_vs_ambient": M,
        "rh_density_ratio": rh_ratio,
        "rh_pressure_ratio": rh_p_ratio,
        "ref_peak_vs_rh_pct": 100.0 * (ref_m["peak"] - rh_ratio) / rh_ratio,
        "dg_best_peak_vs_rh_pct": 100.0 * (min(r["peak"] for r in rows if r["case"].startswith("dg-av"))
                                           - rh_ratio) / rh_ratio,
    }

    # reproduce RH residual check on the fine reference profile
    fi = front_index(ref_x, ref_rho)
    a, b = max(fi - 6, 0), min(fi + 6, ref_x.size - 1)
    rl, rr = float(ref_rho[a]), float(ref_rho[b])
    up, dn = (rl, 0.01, 0.0), (rr, float(np.interp(ref_x[b], ref_x, np.ones_like(ref_rho)) * 0.01), 0.0)

    out = {
        "audited_m16072_commit": "f2dc095c74fe6ac91e4f20382eb32436c4d4c6fb",
        "m16072_worktree": "/root/tp-agent/work/m16072",
        "raw_dir": str(raw),
        "cases": len(cases),
        "ref_case": ref_case,
        "reference_self_convergence": ref_conv,
        "observed_order": order,
        "richardson_limit": lim,
        "reference_uncertainty_pct": ref_unc_pct,
        "analytic_anchor": analytic,
        "tolerances": {"peak_pct": TOL_PEAK_PCT, "front_pct": TOL_FRONT_PCT,
                       "overshoot_pct": TOL_OVER_PCT},
        "gates": gates,
        "ref_window_TV": total_variation(ref_rho[ref_win]),
        "ref_window_extrema": interior_extrema(ref_rho[ref_win]),
        "exact_riemann": exact,
        "exact_gates": exact_gates,
        "repro": repro,
        "rows": rows,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=1))

    print(f"cases={len(cases)} ref={ref_case} ref_peak={ref_m['peak']:.5f} "
          f"ref_front={ref_m['front']:.6f} D={D_meas:.3f} M={M:.1f} RH_rho={rh_ratio:.5f}")
    print(f"ref self-conv: {ref_conv['peak']} order={order['peak']:.2f} "
          f"lim={lim['peak']:.5f} unc_peak={ref_unc_pct['peak']:.2f}%")
    print(f"author peak_err vs my D1 recompute:")
    for row in rows:
        print(f"  {row['case']:24s} peak={row['peak']:.4f}@{row['peak_x']:.4f} "
              f"front={row['front']:.5f} D1={row['D1_peak_err_pct']:+6.2f}% "
              f"D2={row['D2_spike_local_pct']:+6.2f}% L1={row['L1_rel_pct']:5.2f}% "
              f"maxdev={row['max_pointwise_dev_pct']:6.1f}% TV={row['TV_shock_window']:.3f} "
              f"extrema={row['n_extrema_shock_window']}")
    print(f"ref TV in window={out['ref_window_TV']:.3f} extrema={out['ref_window_extrema']}")
    print(f"exact Riemann: p*={exact['p_star']:.3f} u*={exact['u_star']:.3f} "
          f"rho*_L={exact['rho_star_L']:.5f} rho2={exact['rho2']:.5f} D={exact['D']:.5f} "
          f"x_shock={exact['x_shock_t']:.6f} x_contact={exact['x_contact_t']:.6f}")
    print("vs exact:  case                     peak%   front%   star_err%  C3(pass<2%)")
    for row in rows:
        g = exact_gates[row["case"]]
        print(f"  {row['case']:24s} {row['D1_vs_exact_pct']:+7.2f} {row['front_vs_exact_pct']:+7.2f} "
              f"{row['star']['err_pct']:+8.2f}   {g['C3_overshoot_lt2_vs_exact']}")
    print(f"repro: {json.dumps(repro, indent=1) if repro else 'not run'}")
    print(f"gates: {json.dumps(gates, indent=1)}")
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())