"""Verification study for the Manning bottom-friction source of ``swe_1d``.

Three independent reference problems, none of which can be satisfied by a
wrong friction term (see ``docs/swe_manning_friction.md``):

A. **Inertial decay on a flat bed.**  A spatially uniform flow over a flat bed
   obeys ``du/dt = -g n² u² / h^(4/3)``; the continuum solution is
   ``u(t) = u0 / (1 + (g n² u0 / h^(4/3)) t)``.  Because the state is uniform,
   the DG flux divergence vanishes identically and the scheme reduces to the
   ODE (checked below: the higher modes stay at round-off), so this case tests
   the *magnitude and sign* of the friction source against an exact solution.

B. **Normal-depth fixed point.**  On a constant slope the bed-slope source
   ``g h S0`` and the friction sink ``g n² u² / h^(1/3)`` cancel at Manning's
   normal depth ``h_n = (q n / √S0)^(3/5)``, so a uniform flow initialised
   exactly there must be a *discrete fixed point* to round-off.  Gate power is
   probed by perturbing ``n`` (1 %) and by switching friction off: both must
   break the gate by many orders of magnitude.

C. **Dam break onto a dry bed.**  Friction must keep the run physical (no
   NaN, non-negative depth), conserve mass, and shorten the runout monotonically
   in ``n``.  This is also the case that *diverges* without the drag limiter.

Usage::

    python validation/swe_manning_friction/study.py            # all cases
    python validation/swe_manning_friction/study.py --quick    # small grids

Writes ``validation/swe_manning/metrics.json``, ``docs/figs/swe_manning_friction.png``
and the generated results block of ``docs/swe_manning_friction.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone

import torch

from tensordg.swe_1d import (GRAVITY, manning_normal_depth, manning_friction_slope,
                             solve_swe_1d)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS = os.path.join(REPO_ROOT, "validation", "swe_manning", "metrics.json")
DOC = os.path.join(REPO_ROOT, "docs", "swe_manning_friction.md")
FIG = os.path.join(REPO_ROOT, "docs", "figs", "swe_manning_friction.png")

BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"

#: verification parameters (SI units, unit width)
N_MANNING = 0.03        # smooth concrete-ish channel
H0, U0 = 1.0, 1.0       # flat-bed decay initial state
SLOPE, DISCHARGE = 1.0e-3, 1.0
DECAY_TIMES = (5.0, 10.0, 15.0, 20.0)


# --------------------------------------------------------------------------- #
# reference cases
# --------------------------------------------------------------------------- #
def inertial_decay(n_mann=0.03, h0=H0, u0=U0, times=DECAY_TIMES, cfl=0.4,
                   ncells=8, degree=1):
    """Case A: uniform flow over a flat bed vs the exact drag ODE."""
    k = GRAVITY * n_mann ** 2 / h0 ** (4.0 / 3.0)   # du/dt = -k u²
    out = []
    for t_end in times:
        x, coeff, _zb, meta = solve_swe_1d(
            lambda xx: torch.full_like(xx, h0),
            lambda xx: torch.full_like(xx, u0),
            lambda xx: torch.zeros_like(xx),
            ncells=ncells, degree=degree, final_time=t_end, domain=(0.0, 8.0),
            cfl=cfl, periodic=True, manning_n=n_mann)
        h = coeff[:, 0, 0]
        hu = coeff[:, 0, 1]
        u_num = float((hu / h).mean())
        u_exact = u0 / (1.0 + k * u0 * t_end)
        out.append({
            "t": t_end,
            "u_numeric": u_num,
            "u_exact": u_exact,
            "rel_error": abs(u_num - u_exact) / u_exact,
            "depth_drift": float((h - h0).abs().max()),
            "higher_modes_max": float(coeff[:, 1:, :].abs().max()) if degree >= 1 else 0.0,
            "steps": meta["steps"],
            "friction_capped_steps": meta["friction_capped_steps"],
            "reached_final_time": meta["reached_final_time"],
        })
    return out


def normal_depth(n_mann=0.03, slope=SLOPE, discharge=DISCHARGE, t_end=20.0,
                 ncells=8, degree=1, cfl=0.4, well_balanced=False, n_run=None):
    """Case B: uniform flow initialised at Manning's normal depth.

    ``n_run`` decouples the roughness used by the solver from the one that
    defines the reference state: ``n_run=0`` initialises the *frictionless*
    steady-flow depth and then runs without drag, which is the gate-power probe
    (a fixed point that also survives with the sink switched off would not be a
    test of friction at all).
    """
    n_run = n_mann if n_run is None else n_run
    h_n = manning_normal_depth(discharge, slope, n_mann)
    u_n = discharge / h_n
    x, coeff, _zb, meta = solve_swe_1d(
        lambda xx: torch.full_like(xx, h_n),
        lambda xx: torch.full_like(xx, u_n),
        lambda xx: -slope * xx,
        ncells=ncells, degree=degree, final_time=t_end, domain=(0.0, 8.0),
        cfl=cfl, manning_n=n_run, well_balanced=well_balanced,
        zb_slope=lambda xx: torch.full_like(xx, -slope))
    h = coeff[:, 0, 0]
    hu = coeff[:, 0, 1]
    return {
        "n_mann": n_mann,
        "n_run": n_run,
        "slope": slope,
        "discharge": discharge,
        "normal_depth": h_n,
        "normal_velocity": u_n,
        "froude": u_n / math.sqrt(GRAVITY * h_n),
        "friction_slope_manning": float(manning_friction_slope(
            torch.tensor(u_n, dtype=torch.float64),
            torch.tensor(h_n, dtype=torch.float64), n_mann)),
        "rel_depth_drift": float(((h - h_n) / h_n).abs().max()),
        "rel_discharge_drift": float(((hu - discharge) / discharge).abs().max()),
        # A fixed point has to hold *both* fields: a frictionless run keeps the
        # depth nearly pinned while the discharge accelerates, so gating on the
        # depth alone would accept a solution with no friction at all.
        "rel_state_drift": max(float(((h - h_n) / h_n).abs().max()),
                               float(((hu - discharge) / discharge).abs().max())),
        "well_balanced": well_balanced,
        "steps": meta["steps"],
        "friction_capped_steps": meta["friction_capped_steps"],
        "reached_final_time": meta["reached_final_time"],
    }


def dam_break_dry_bed(n_mann, ncells=60, t_end=0.5, cfl=0.3):
    """Case C: Ritter (1892) dam break onto a dry bed, with and without drag."""
    x, coeff, _zb, meta = solve_swe_1d(
        lambda xx: torch.where(xx < 0.0, torch.ones_like(xx), torch.zeros_like(xx)),
        lambda xx: torch.zeros_like(xx),
        lambda xx: torch.zeros_like(xx),
        ncells=ncells, degree=1, final_time=t_end, domain=(-3.0, 3.0), cfl=cfl,
        manning_n=n_mann)
    h = coeff[:, 0, 0]
    dx = 6.0 / ncells
    mass = float(h.sum() * dx)
    wet = h > 1e-3
    return {
        "n_mann": n_mann,
        "steps": meta["steps"],
        "friction_capped_steps": meta["friction_capped_steps"],
        "time_reached": meta["time_reached"],
        "reached_final_time": meta["reached_final_time"],
        "h_max": float(h.max()),
        "h_min": float(h.min()),
        "finite": bool(torch.isfinite(coeff).all()),
        "mass": mass,
        "mass_error": abs(mass - 3.0) / 3.0,
        "front_position": float(x[wet].max()) if bool(wet.any()) else 0.0,
        "profile": {"x": [float(v) for v in x], "h": [float(v) for v in h]},
    }


def zero_friction_is_a_noop(cfl=0.3):
    """Manning n = 0 must leave the legacy scheme bit-for-bit unchanged."""
    ic_h = lambda xx: torch.where(xx < 0.0, torch.ones_like(xx), 0.1 * torch.ones_like(xx))
    ic_u = lambda xx: torch.zeros_like(xx)
    bed = lambda xx: torch.zeros_like(xx)
    kwargs = dict(ncells=24, degree=1, final_time=0.1, domain=(-3.0, 3.0), cfl=cfl)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, a, _, ma = solve_swe_1d(ic_h, ic_u, bed, **kwargs)
        _, b, _, mb = solve_swe_1d(ic_h, ic_u, bed, manning_n=0.0, **kwargs)
    return {
        "bitwise_identical": bool(torch.equal(a, b)),
        "max_abs_difference": float((a - b).abs().max()),
        "steps_without": ma["steps"], "steps_with": mb["steps"],
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def run_all(quick: bool = False) -> dict:
    nc = 4 if quick else 8
    times = DECAY_TIMES[:2] if quick else DECAY_TIMES
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": {
            "manning_n": N_MANNING, "flat_bed_h0": H0, "flat_bed_u0": U0,
            "slope": SLOPE, "discharge": DISCHARGE,
            "friction_limiter": True, "quick": quick,
        },
        "case_a_inertial_decay": inertial_decay(times=times, ncells=nc),
        "case_b_normal_depth": normal_depth(ncells=nc),
        "case_b_gate_power": {
            "friction_off": normal_depth(n_run=0.0, ncells=nc),
            "n_minus_1pct": normal_depth(n_run=0.99 * N_MANNING, ncells=nc),
            "n_plus_1pct": normal_depth(n_run=1.01 * N_MANNING, ncells=nc),
        },
        "case_b_well_balanced_path": normal_depth(ncells=nc, well_balanced=True),
        "case_c_dam_break_dry_bed": [dam_break_dry_bed(n, ncells=20 if quick else 60)
                                     for n in (0.0, N_MANNING, 2 * N_MANNING)],
        "case_d_zero_friction": zero_friction_is_a_noop(),
    }
    metrics["gates"] = evaluate_gates(metrics)
    metrics["provenance"] = provenance()
    return metrics


DECAY_GATE = 1.0e-9       # Case A: |u_num - u_exact| / u_exact
FIXPOINT_GATE = 1.0e-9    # Case B: max relative fixed-point drift
GATE_POWER_GATE = 1.0e-4  # Case B probe: a wrong n must break the gate
MASS_GATE = 1.0e-5        # Case C: relative mass error


def evaluate_gates(metrics):
    a = metrics["case_a_inertial_decay"]
    b = metrics["case_b_normal_depth"]
    probes = metrics["case_b_gate_power"]
    c = metrics["case_c_dam_break_dry_bed"]
    worst_a = max(row["rel_error"] for row in a)
    probes_min = min(v["rel_state_drift"] for v in probes.values())
    fronts = [row["front_position"] for row in c]
    return {
        "case_a_max_rel_error": worst_a,
        "case_a_gate": DECAY_GATE,
        "case_a_pass": worst_a < DECAY_GATE and all(r["reached_final_time"] for r in a),
        "case_b_rel_state_drift": b["rel_state_drift"],
        "case_b_rel_depth_drift": b["rel_depth_drift"],
        "case_b_gate": FIXPOINT_GATE,
        "case_b_pass": b["rel_state_drift"] < FIXPOINT_GATE,
        "case_b_gate_power_min_probe_drift": probes_min,
        "case_b_gate_power_gate": GATE_POWER_GATE,
        "case_b_gate_power_pass": probes_min > GATE_POWER_GATE,
        "case_c_front_positions": fronts,
        "case_c_runout_monotone": all(fronts[i] > fronts[i + 1]
                                      for i in range(len(fronts) - 1)),
        "case_c_max_mass_error": max(row["mass_error"] for row in c),
        "case_c_mass_gate": MASS_GATE,
        "case_c_pass": (max(row["mass_error"] for row in c) < MASS_GATE
                        and all(row["finite"] and row["reached_final_time"] for row in c)),
        "case_d_pass": metrics["case_d_zero_friction"]["bitwise_identical"],
    }


def provenance() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(("git",) + args, cwd=REPO_ROOT,
                                           text=True).strip()
        except Exception:  # noqa: BLE001 - provenance is best effort
            return "unknown"
    commit = git("rev-parse", "HEAD")
    return {
        "generator": "validation/swe_manning_friction/study.py",
        "command": "python validation/swe_manning_friction/study.py",
        "argv": sys.argv[1:],
        "git_commit": commit,
        "git_commit_short": commit[:7],
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "cpu_count": os.cpu_count(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def _f(value: float, digits: int = 3) -> str:
    """Canonical formatting shared by the document block and its test."""
    return f"{value:.{digits}e}" if value != 0.0 else "0.000e+00"


def results_block(metrics: dict) -> str:
    """The markdown table block injected into the document (single source)."""
    g = metrics["gates"]
    p = metrics["parameters"]
    lines = [
        f"*Generated by `{metrics['provenance']['generator']}` "
        f"(commit `{metrics['provenance']['git_commit_short']}`, "
        f"torch {metrics['provenance']['torch']}).  Do not edit by hand: "
        "`tests/test_swe_manning_friction.py` compares this block with "
        "`validation/swe_manning/metrics.json`.*",
        "",
        f"Parameters: n = {p['manning_n']}, flat-bed h0 = {p['flat_bed_h0']} m, "
        f"u0 = {p['flat_bed_u0']} m/s, S0 = {p['slope']}, q = {p['discharge']} m²/s.",
        "",
        "### A. Uniform flow over a flat bed vs the exact drag ODE",
        "",
        "| t (s) | u numeric (m/s) | u exact (m/s) | rel. error | depth drift | higher modes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["case_a_inertial_decay"]:
        lines.append(
            f"| {row['t']:g} | {row['u_numeric']:.10f} | {row['u_exact']:.10f} | "
            f"{_f(row['rel_error'])} | {_f(row['depth_drift'])} | "
            f"{_f(row['higher_modes_max'])} |")
    lines += [
        "",
        f"Gate |u_num − u_exact|/u_exact < {_f(g['case_a_gate'])}: "
        f"max {_f(g['case_a_max_rel_error'])} → "
        f"**{'PASS' if g['case_a_pass'] else 'FAIL'}**",
        "",
        "### B. Normal-depth fixed point on a constant slope",
        "",
        f"Manning normal depth h_n = {metrics['case_b_normal_depth']['normal_depth']:.10f} m, "
        f"u_n = {metrics['case_b_normal_depth']['normal_velocity']:.10f} m/s, "
        f"Fr = {metrics['case_b_normal_depth']['froude']:.4f}, "
        f"S_f = {_f(metrics['case_b_normal_depth']['friction_slope_manning'])} "
        f"(= S0 to round-off).",
        "",
        "| state | rel. depth drift | rel. discharge drift | combined drift | capped steps |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = [("n = n_nominal", metrics["case_b_normal_depth"])]
    rows += [(label, metrics["case_b_gate_power"][key]) for label, key in
             (("friction off (n = 0)", "friction_off"),
              ("n − 1 %", "n_minus_1pct"), ("n + 1 %", "n_plus_1pct"))]
    rows += [("n_nominal, well_balanced=True", metrics["case_b_well_balanced_path"])]
    for label, row in rows:
        lines.append(f"| {label} | {_f(row['rel_depth_drift'])} | "
                     f"{_f(row['rel_discharge_drift'])} | {_f(row['rel_state_drift'])} | "
                     f"{row['friction_capped_steps']} |")
    lines += [
        "",
        f"Gate (combined drift) ≤ {_f(g['case_b_gate'])}: "
        f"{_f(g['case_b_rel_state_drift'])} → "
        f"**{'PASS' if g['case_b_pass'] else 'FAIL'}**;  gate power (a wrong n must "
        f"break it, ≥ {_f(g['case_b_gate_power_gate'])}): "
        f"min probe drift {_f(g['case_b_gate_power_min_probe_drift'])} → "
        f"**{'PASS' if g['case_b_gate_power_pass'] else 'FAIL'}**",
        "",
        "### C. Dam break onto a dry bed",
        "",
        "| n | runout x_f (m) | h_max (m) | mass error | capped steps | steps |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["case_c_dam_break_dry_bed"]:
        lines.append(f"| {row['n_mann']:g} | {row['front_position']:.4f} | "
                     f"{row['h_max']:.4f} | {_f(row['mass_error'])} | "
                     f"{row['friction_capped_steps']} | {row['steps']} |")
    lines += [
        "",
        f"Runout monotone decreasing in n: **{g['case_c_runout_monotone']}**;  "
        f"mass error ≤ {_f(g['case_c_mass_gate'])}: {_f(g['case_c_max_mass_error'])} → "
        f"**{'PASS' if g['case_c_pass'] else 'FAIL'}**",
        "",
        f"Zero-friction no-op (bitwise identical to the legacy scheme): "
        f"**{'PASS' if g['case_d_pass'] else 'FAIL'}**",
    ]
    return "\n".join(lines)


def make_figure(metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    p = metrics["parameters"]

    # (a) inertial decay: numeric points vs the exact curve
    ax = axes[0]
    k = GRAVITY * p["manning_n"] ** 2 / p["flat_bed_h0"] ** (4.0 / 3.0)
    tt = [i * 0.02 for i in range(0, 1001)]
    ax.plot(tt, [p["flat_bed_u0"] / (1.0 + k * p["flat_bed_u0"] * t) for t in tt],
            "-", color="#1f77b4", lw=2,
            label=r"exact $u_0/(1+g n^2 u_0 t/h^{4/3})$")
    rows = metrics["case_a_inertial_decay"]
    ax.plot([r["t"] for r in rows], [r["u_numeric"] for r in rows], "o",
            color="#d62728", ms=7, label="DG + Manning (n=%.2f)" % p["manning_n"])
    ax.set_xlabel("t (s)"); ax.set_ylabel("depth-averaged u (m/s)")
    ax.set_title("A: flat-bed inertial decay\n(rel. error %.1e)" %
                 metrics["gates"]["case_a_max_rel_error"])
    ax.legend(); ax.grid(alpha=0.3)

    # (b) fixed-point sharpness: drift vs the roughness ratio
    ax = axes[1]
    ratios, drifts = [1.0], [metrics["case_b_normal_depth"]["rel_state_drift"]]
    for label, key in (("0.99", "n_minus_1pct"), ("1.01", "n_plus_1pct")):
        ratios.append(float(label))
        drifts.append(metrics["case_b_gate_power"][key]["rel_state_drift"])
    ratios.append(0.0); drifts.append(
        metrics["case_b_gate_power"]["friction_off"]["rel_state_drift"])
    order = sorted(range(len(ratios)), key=lambda i: ratios[i])
    ax.semilogy([ratios[i] for i in order], [max(drifts[i], 1e-17) for i in order],
                "o-", color="#2ca02c")
    ax.axhline(metrics["gates"]["case_b_gate"], ls="--", color="k", lw=1,
               label="gate %.0e" % metrics["gates"]["case_b_gate"])
    ax.set_xlabel(r"$n / n_{\rm nominal}$  (0 = friction off)")
    ax.set_ylabel("rel. state drift (max of h, q) at t = 20 s")
    ax.set_title("B: normal-depth fixed point\n(gate power: wrong n breaks it)")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    # (c) dam-break depth profiles
    ax = axes[2]
    for row, colour in zip(metrics["case_c_dam_break_dry_bed"],
                           ("#7f7f7f", "#1f77b4", "#d62728")):
        prof = row["profile"]
        ax.plot(prof["x"], prof["h"], "-", color=colour, lw=1.6,
                label="n = %.2f, x_f = %.2f m" % (row["n_mann"], row["front_position"]))
        ax.axvline(row["front_position"], color=colour, ls=":", lw=1)
    ax.set_xlabel("x (m)"); ax.set_ylabel("depth h (m)")
    ax.set_title("C: dry-bed dam break, t = 0.5 s\n(friction shortens the runout)")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=130)
    plt.close(fig)


def inject_doc_block(metrics: dict) -> None:
    if not os.path.exists(DOC):
        return
    text = open(DOC).read()
    if BEGIN not in text or END not in text:
        return
    head, rest = text.split(BEGIN, 1)
    _stale, tail = rest.split(END, 1)
    open(DOC, "w").write(head + BEGIN + "\n" + results_block(metrics) + "\n" + END + tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="coarse grids (smoke run)")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    metrics = run_all(quick=args.quick)
    os.makedirs(os.path.dirname(METRICS), exist_ok=True)
    with open(METRICS, "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not args.no_figure:
        make_figure(metrics)
    inject_doc_block(metrics)

    gates = metrics["gates"]
    for key in ("case_a_pass", "case_b_pass", "case_b_gate_power_pass",
                "case_c_pass", "case_d_pass"):
        print("%-26s %s" % (key, "PASS" if gates[key] else "FAIL"))
    print("metrics ->", os.path.relpath(METRICS, REPO_ROOT))
    return 0 if all(gates[k] for k in ("case_a_pass", "case_b_pass",
                                       "case_b_gate_power_pass", "case_c_pass",
                                       "case_d_pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())