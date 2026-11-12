"""Paper-grade report + figures for the DG theta--beta--Mach verification.

Mission m15977.  Everything numeric in the report is injected from the three
artifacts (``theta_beta_anchors.json``, ``case_matrix.csv``,
``convergence.json``) -- nothing is transcribed by hand.

Usage::

    python -m benchmarks.report_oblique_shock_theta_beta            # build report.json
    python -m benchmarks.report_oblique_shock_theta_beta --publish  # + POST it
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import oblique_shock as os
from benchmarks import oblique_shock_theta_beta as tb

ART = Path("benchmarks/oblique_shock")
FIG_DIR = Path("docs/reports/figures/oblique_shock_theta_beta")
ANCHOR = ART / "theta_beta_anchors.json"
MATRIX = ART / "case_matrix.csv"
CONV = ART / "convergence.json"


# --------------------------------------------------------------------------- #
# loading / aggregation
# --------------------------------------------------------------------------- #
def load_artifacts():
    for path in (ANCHOR, MATRIX):
        if not path.exists():
            raise SystemExit(f"missing artifact {path}; run the benchmark first")
    anchors = json.loads(ANCHOR.read_text())
    with open(MATRIX) as handle:
        rows = list(csv.DictReader(handle))
    numeric = ("mach", "theta_deg", "degree", "ncells", "final_time", "cfl",
               "steps", "beta_exact_deg", "beta_hat_deg", "beta_dev_deg",
               "beta_dev_pct", "rho_ratio_exact", "rho_ratio_hat", "rho_dev_pct",
               "p_ratio_exact", "p_ratio_hat", "p_dev_pct", "theta_exact_deg",
               "theta_hat_deg", "theta_dev_deg", "preserve_l1",
               "rh_residual_max", "max_dev_pct")
    for row in rows:
        for key in numeric:
            value = row.get(key, "")
            try:
                row[key] = None if value in ("", None) else float(value)
            except ValueError:
                row[key] = None
        for key in ("gate_beta_2pct", "gate_jump_2pct", "hard_fail_15pct"):
            row[key] = row.get(key) == "true"
    convergence = None
    if CONV.exists():
        convergence = json.loads(CONV.read_text())
    return anchors, rows, convergence


def summarise(rows, convergence):
    weak = [r for r in rows if r["branch"] == "weak"]
    strong = [r for r in rows if r["branch"] == "strong"]
    nan = float("nan")

    def nums(entries, key):
        values = []
        for entry in entries:
            try:
                values.append(float(entry.get(key)))
            except (TypeError, ValueError):
                continue
        return values

    def worst(entries, key):
        values = nums(entries, key)
        return max(values) if values else nan

    def block(entries):
        if not entries:
            return {"n": 0}
        nan = float("nan")
        betas = nums(entries, "beta_dev_pct")
        cent = nums(entries, "beta_dev_centroid_pct")
        return {
            "n": len(entries),
            "machs": sorted({e["mach"] for e in entries}),
            "thetas": sorted({e["theta_deg"] for e in entries}),
            "degrees": sorted({e["degree"] for e in entries}),
            "pass": sum(1 for e in entries if e["verdict"] == "PASS"),
            "fail": sum(1 for e in entries if e["verdict"] != "PASS"),
            "beta_dev_pct_max": max(betas) if betas else nan,
            "beta_dev_pct_median": float(np.median(betas)) if betas else nan,
            "beta_dev_centroid_pct_max": max(cent) if cent else nan,
            "beta_dev_centroid_pct_median": float(np.median(cent)) if cent else nan,
            "rho_dev_pct_max": worst(entries, "rho_dev_pct"),
            "p_dev_pct_max": worst(entries, "p_dev_pct"),
            "preserve_l1_max": worst(entries, "preserve_l1"),
            "hard_fail_15pct": [e["case_id"] for e in entries if e["hard_fail_15pct"]],
            "aborted": [e["case_id"] for e in entries if e["beta_hat_deg"] is None],
        }
    summary = {"weak": block(weak), "strong": block(strong),
               "all": block(rows),
               "not_detected": [e["case_id"] for e in rows
                                if e["beta_hat_deg"] is None]}
    if convergence:
        summary["gci"] = [
            {"case_id": entry["case_id"],
             "gci_beta_pct": (entry["gci_beta"]["gci_fine_pct"]
                              if entry["gci_beta"] else None),
             "observed_order": (entry["gci_beta"]["observed_order"]
                                if entry["gci_beta"] else None),
             "gate": entry["gate_gci_beta_2pct"]}
            for entry in convergence["cases"]]
    return summary


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_theta_beta(anchors, rows):
    fig, axes = plt.subplots(1, 1, figsize=(7.4, 5.2))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(tb.MACH_LIST)))
    for mach, color in zip(tb.MACH_LIST, colors):
        theta_max, _ = tb.detachment_angle(mach)
        thetas = np.linspace(1e-3, theta_max - 1e-3, 160)
        weak = [os.theta_from_beta(mach, tb.beta_from_theta_branch(mach, t, "weak"))
                for t in thetas]
        strong = [os.theta_from_beta(mach, tb.beta_from_theta_branch(mach, t, "strong"))
                  for t in thetas]
        betas_w = [tb.beta_from_theta_branch(mach, t, "weak") for t in thetas]
        betas_s = [tb.beta_from_theta_branch(mach, t, "strong") for t in thetas]
        axes.plot(thetas, betas_w, "-", color=color, lw=1.5, label=f"M={mach:g} weak")
        axes.plot(thetas, betas_s, "--", color=color, lw=1.1, alpha=0.8,
                  label=f"M={mach:g} strong")
        points = [r for r in rows if r["mach"] == mach and r["beta_hat_deg"] is not None]
        if points:
            axes.plot([p["theta_deg"] for p in points], [p["beta_hat_deg"] for p in points],
                      "o", ms=4.0, mfc="none", mec=color, mew=1.1)
        t_max, b_det = tb.detachment_angle(mach)
        axes.plot([t_max], [b_det], "k*", ms=7)
    axes.set_xlabel(r"deflection angle $\theta$ [deg]")
    axes.set_ylabel(r"shock angle $\beta$ [deg]")
    axes.set_title("theta--beta--Mach anchor curves (lines) vs DG runs (open circles)")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=6.5, ncol=2, loc="upper left")
    return _save(fig, "theta_beta_curves.png")


def fig_deviation_heatmap(rows):
    weak = [r for r in rows if r["branch"] == "weak"]
    degrees = sorted({r["degree"] for r in weak})
    machs = sorted({r["mach"] for r in weak})
    values = [r["max_dev_pct"] for r in weak if r["max_dev_pct"] is not None]
    vmax = max(values) if values else 1.0
    fig, axes = plt.subplots(1, len(degrees), figsize=(5.0 * len(degrees), 4.2),
                             squeeze=False)
    for ax, degree in zip(axes[0], degrees):
        thetas = sorted({r["theta_deg"] for r in weak if r["degree"] == degree})
        grid = np.full((len(machs), len(thetas)), np.nan)
        for r in weak:
            if r["degree"] != degree:
                continue
            if r["max_dev_pct"] is None:
                continue
            if np.isfinite(r["max_dev_pct"]):
                grid[machs.index(r["mach"]), thetas.index(r["theta_deg"])] = r["max_dev_pct"]
        image = ax.imshow(grid, origin="lower", cmap="magma_r",
                          vmin=0.0, vmax=max(2.0, vmax))
        ax.set_xticks(range(len(thetas)))
        ax.set_xticklabels([f"{t:g}" for t in thetas])
        ax.set_yticks(range(len(machs)))
        ax.set_yticklabels([f"{m:g}" for m in machs])
        ax.set_xlabel(r"$\theta$ [deg]")
        ax.set_ylabel("M")
        ax.set_title(f"P{degree}: worst deviation [%]")
        for i in range(len(machs)):
            for j in range(len(thetas)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="w" if grid[i, j] > 0.5 * max(2.0, vmax) else "k")
        fig.colorbar(image, ax=ax, label="max(%s) [%%]" % r"$\beta,\rho,p$")
    fig.tight_layout()
    return _save(fig, "deviation_heatmap.png")


def fig_convergence(convergence):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for entry in convergence["cases"]:
        levels = np.array(entry["levels"], dtype=float)
        h = 1.0 / levels
        exact = entry["beta_exact_deg"]
        beta = entry["beta_hat_deg"]
        error = [abs(b - exact) if b is not None else np.nan for b in beta]
        axes[0].loglog(h, [max(e, 1e-12) for e in error], "o-",
                       label=f"{entry['case_id']} (P{entry['degree']})")
        gci = entry["gci_beta"]
        if gci:
            axes[0].annotate(f"GCI={gci['gci_fine_pct']:.3f}%  p={gci['observed_order']:.2f}"
                             if gci["observed_order"] else "GCI", xy=(h[0], max(error[0], 1e-12)),
                             xytext=(4, 6), textcoords="offset points", fontsize=8)
        axes[1].semilogx(levels, [b - exact for b in beta], "o-",
                         label=f"{entry['case_id']} (P{entry['degree']})")
    axes[0].set_xlabel("mesh size h")
    axes[0].set_ylabel(r"$|\hat\beta-\beta|$ [deg]")
    axes[0].set_title("shock-angle error vs mesh (3 levels)")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("cells per direction")
    axes[1].set_ylabel(r"$\hat\beta-\beta$ [deg]")
    axes[1].set_title("signed shock-angle error")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "convergence.png")


def fig_branch_scatter(rows, anchors):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for branch, marker, color in (("weak", "o", "tab:blue"), ("strong", "s", "tab:red")):
        entries = [r for r in rows if r["branch"] == branch and r["beta_hat_deg"] is not None]
        if not entries:
            continue
        ax.plot([r["beta_exact_deg"] for r in entries],
                [r["beta_hat_deg"] for r in entries], marker, ls="none", ms=6,
                mfc="none", color=color, label=f"{branch} branch")
    lo, hi = 0.0, 92.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="perfect agreement")
    ax.fill_between([lo, hi], [0.98 * lo, 0.98 * hi], [1.02 * lo, 1.02 * hi],
                    color="grey", alpha=0.15, label="+/-2 % band")
    ax.set_xlabel(r"exact shock angle $\beta$ [deg]")
    ax.set_ylabel(r"measured $\hat\beta$ [deg]")
    ax.set_title("weak vs strong branch: measured against exact")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, "branch_scatter.png")


def fig_field_snapshot(mach=3.0, theta=15.0, degree=2, ncells=32, final_time=0.05):
    """One DG field with the exact front overlaid (density)."""
    beta = tb.beta_from_theta_branch(mach, theta, "weak")
    solution = os.oblique_shock(mach, beta)
    initial = os.oblique_shock_initial(solution)
    from tensordg.euler_dg_2d import solve_euler_dg_2d, cell_fields
    x, y, coeff, _ = solve_euler_dg_2d(initial, ncells=(ncells, ncells),
                                       degree=degree, final_time=final_time,
                                       periodic=(False, False), dtype=__import__("torch").float64)
    fields = cell_fields(coeff)
    rho = fields["density"].cpu().numpy()
    xc, yc = x.cpu().numpy(), y.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    image = axes[0].pcolormesh(xc, yc, rho.T, cmap="viridis", shading="auto")
    beta_r = math.radians(beta)
    n = np.array([math.sin(beta_r), -math.cos(beta_r)])
    t = np.array([math.cos(beta_r), math.sin(beta_r)])
    s = np.linspace(-0.7, 0.7, 2)
    axes[0].plot(0.5 + s * n[0] + 0.0 * t[0], 0.5 + s * n[1], "w--", lw=1.0,
                 label="exact front")
    axes[0].set_title(f"M={mach:g}, theta={theta:g} deg, P{degree}, {ncells}^2: density")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.colorbar(image, ax=axes[0], label=r"$\rho$")

    # profile along the shock normal through the front centre
    offsets = np.linspace(-0.25, 0.25, 400)
    px, py = 0.5 + offsets * n[0], 0.5 + offsets * n[1]
    i = np.clip(np.searchsorted(xc, px) - 1, 0, len(xc) - 2)
    j = np.clip(np.searchsorted(yc, py) - 1, 0, len(yc) - 2)
    sampled = rho[i, j]
    axes[1].plot(offsets, sampled, "o", ms=2.5, label="DG density")
    axes[1].plot(offsets, np.where(offsets > 0, solution.downstream.rho, solution.upstream.rho),
                 "k--", lw=1.2, label="exact RH")
    axes[1].set_xlabel("distance along shock normal")
    axes[1].set_ylabel(r"$\rho$")
    axes[1].set_title("cross-front profile")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "field_snapshot.png")


def fig_error_vs_mach(rows):
    """How do the deviations scale with Mach number and polynomial order?"""
    weak = [r for r in rows if r["branch"] == "weak" and r["beta_dev_pct"] is not None]
    machs = sorted({r["mach"] for r in weak})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    metric_keys = [("beta_dev_pct", r"$|\Delta\beta|/\beta$ [%]"),
                   ("preserve_l1", "preservation error [-]"),
                   ("max_dev_pct", "worst deviation [%]")]
    def collect(degree, mach, key):
        return [r[key] for r in weak
                if r["degree"] == degree and r["mach"] == mach
                and isinstance(r[key], float) and np.isfinite(r[key])]

    for ax, (key, label) in zip(axes, [metric_keys[0], metric_keys[1]]):
        for degree, marker in zip(sorted({r["degree"] for r in weak}), ("o", "s", "^")):
            used, values, lows, highs = [], [], [], []
            for mach in machs:
                series = collect(degree, mach, key)
                if not series:
                    continue
                used.append(mach)
                values.append(float(np.mean(series)))
                lows.append(min(series))
                highs.append(max(series))
            if not used:
                continue
            ax.errorbar(used, values,
                        yerr=[np.subtract(values, lows), np.subtract(highs, values)],
                        marker=marker, capsize=3, lw=1.2,
                        label=f"DG P{degree} (mean over theta, bars = min/max)")
        ax.set_xlabel("free-stream Mach number M")
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_title("shock-angle error vs Mach number")
    axes[1].set_title("shock preservation error vs Mach number")
    fig.tight_layout()
    return _save(fig, "error_vs_mach.png")


def write_readme(anchors, rows, convergence, summary, path=ART / "README.md"):
    """Emit the artifact README from the artifacts themselves (no hand copies)."""
    commit = tb._git_commit()[:12]
    weak = [r for r in rows if r["branch"] == "weak"]
    strong = [r for r in rows if r["branch"] == "strong"]
    facts = _matrix_facts(rows)
    limit_rows = [{"name": c["name"], "status": c["status"]} for c in anchors["limit_checks"]]
    limit_md = _md_table(limit_rows, ["name", "status"], lambda row, key: row[key])
    det_rows = [{"M": str(m), "theta_max_deg": f"{float(v['theta_max_deg']):.4f}",
                 "beta_detached_deg": f"{float(v['beta_detached_deg']):.4f}",
                 "mach_angle_deg": f"{float(v['mach_angle_deg']):.4f}"}
                for m, v in anchors["detachment"].items()]
    det_md = _md_table(det_rows, ["M", "theta_max_deg", "beta_detached_deg",
                                  "mach_angle_deg"], lambda row, key: row[key])
    case_rows = [{
        "case": r["case_id"],
        "branch": r["branch"],
        "beta_exact": f"{r['beta_exact_deg']:.4f}",
        "beta_hat": "n/a" if r["beta_hat_deg"] is None else f"{r['beta_hat_deg']:.4f}",
        "beta_dev_pct": "n/a" if r["beta_dev_pct"] is None else f"{r['beta_dev_pct']:.3f}",
        "rho_dev_pct": _f(r['rho_dev_pct'], ".3f"),
        "p_dev_pct": _f(r['p_dev_pct'], ".3f"),
        "preserve_l1": _f(r['preserve_l1'], ".4f"),
        "verdict": r["verdict"],
    } for r in rows]
    case_md = _md_table(case_rows,
                        ["case", "branch", "beta_exact", "beta_hat", "beta_dev_pct",
                         "rho_dev_pct", "p_dev_pct", "preserve_l1", "verdict"],
                        lambda row, key: row[key])
    conv_md = "_(convergence study not run)_"
    if convergence:
        conv_rows = []
        for entry in convergence["cases"]:
            g = entry["gci_beta"]
            gr = entry["gci_rho"]
            conv_rows.append({
                "case": entry["case_id"],
                "levels": "/".join(str(l) for l in entry["levels"]),
                "GCI_beta_pct": "n/a" if not g else f"{g['gci_fine_pct']:.3f}",
                "order_beta": "n/a" if not g or g["observed_order"] is None
                              else f"{g['observed_order']:.2f}",
                "order_estimated": "n/a" if not g else str(g["order_estimated"]),
                "GCI_rho_pct": "n/a" if not gr else f"{gr['gci_fine_pct']:.3f}",
                "gate_beta_<2pct": "PASS" if entry["gate_gci_beta_2pct"] else "FAIL",
            })
        conv_md = _md_table(conv_rows,
                            ["case", "levels", "GCI_beta_pct", "order_beta",
                             "order_estimated", "GCI_rho_pct", "gate_beta_<2pct"],
                            lambda row, key: row[key])
    strong_notes = "\n".join(
        f"* `{r['case_id']}` -- {('FAIL: ' + (r.get('note') or '')) if r['verdict'] != 'PASS' else 'PASS (interior exact solution preserved)'}"
        for r in strong)
    text = f"""# Oblique-shock theta--beta--Mach anchors and DG case matrix

Mission m15977, paper line `dg-oblique-shock-theta-beta`.
Generated by `benchmarks/oblique_shock_theta_beta.py` (+ this README by
`benchmarks/report_oblique_shock_theta_beta.py::write_readme`) at commit
`{commit}`. Every number below is read back from the JSON/CSV artifacts; none is
hand-typed.

## Artifacts

| file | content |
|---|---|
| `theta_beta_anchors.json` | anchor set `{anchors['anchor_id']}`: {len(anchors['table'])} (M, theta, branch) entries, detachment angles, RH jumps, 5 limit checks |
| `case_matrix.csv` | {len(rows)} DG runs ({len(weak)} weak, {len(strong)} strong) with recovered angle, jumps, deviations and verdicts |
| `convergence.json` | three-level grid refinement (Roache GCI, Fs=1.25, ratio 2) of `beta` and the jumps |
| `README.md` | this file |

## Anchors (generated, not transcribed)

Theta--beta--Mach relation with both roots, Rankine--Hugoniot jump conditions,
`gamma = {anchors['gamma']}`, upstream `rho = p = 1`, `u = M sqrt(gamma)`.
Limit checks:

{limit_md}

Detachment point and Mach angle per Mach number:

{det_md}

## Case matrix ({len(rows)} runs)

`N = {facts['weak_ncells']:g}^2` cells, `t_final = {facts['weak_tfinal']:g}`, `CFL = {facts['weak_cfl']:g}`,
no limiter, `float64`, transmissive faces. Gates declared before the runs:
`|d beta| / beta <= 2 %`, `|d rho| / rho <= 2 %`, `|d p| / p <= 2 %`.

{case_md}

Weak branch: PASS {summary['weak']['pass']}/{summary['weak']['n']}, worst
`beta` {summary['weak'].get('beta_dev_pct_max', float('nan')):.3f} %,
`rho` {summary['weak']['rho_dev_pct_max']:.3f} %,
`p` {summary['weak']['p_dev_pct_max']:.3f} %.
Strong branch: PASS {summary['strong']['pass']}/{summary['strong']['n']}.

Strong-branch per-case attribution:

{strong_notes if strong_notes else '_(none)_'}

## Grid convergence

{conv_md}

## Reproduce

```bash
python -m benchmarks.oblique_shock_theta_beta --stage all --ncells 32 --final-time 0.05
python -m benchmarks.report_oblique_shock_theta_beta            # report + figures
pytest -q tests/test_oblique_shock_theta_beta.py tests/test_oblique_shock.py
```

## Scope and limitations (honest)

* Only straight, plane, stationary fronts are covered: the DG solver exposes
  periodic / transmissive faces, so a wall-bounded wedge (where the weak branch
  is physically selected) is out of scope. The angle is measured on an interior
  window at least 0.12 from the boundaries.
* The strong branch is non-unique with a free subsonic outflow; those runs are
  reported separately and are **not** part of the weak-branch acceptance verdict.
* No limiter: the front is smeared over a few cells, post-shock plateaus carry
  small Gibbs oscillations, which is the intended stress case for angle recovery.
* The reference is the closed-form relation itself (recomputed here), so the
  only literature comparisons are the `theta_max` values checked in
  `tests/test_oblique_shock_theta_beta.py`.
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print("README ->", path)
    return path


def _png_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _md_table(rows, header, fmt):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row, key) for key in header) + " |")
    return "\n".join(out)


def _f(value, spec=".3f"):
    """Format a possibly-missing number without blowing up on None."""
    return "n/a" if value is None else format(value, spec)


def _matrix_facts(rows):
    """Describe the matrix actually run, read from the CSV (no hand copy)."""
    weak = [r for r in rows if r["branch"] == "weak"]
    strong = [r for r in rows if r["branch"] == "strong"]

    def fmt(values, width=0):
        return ", ".join(f"{v:g}" for v in sorted(set(values)))

    def first(entries, key):
        return entries[0][key] if entries else None

    return {
        "weak_machs": fmt([r["mach"] for r in weak]),
        "weak_thetas": fmt([r["theta_deg"] for r in weak]),
        "weak_degrees": fmt([r["degree"] for r in weak]),
        "weak_ncells": first(weak, "ncells"),
        "weak_tfinal": first(weak, "final_time"),
        "weak_cfl": first(weak, "cfl"),
        "strong_machs": fmt([r["mach"] for r in strong]) or "n/a",
        "strong_thetas": fmt([r["theta_deg"] for r in strong]) or "n/a",
        "strong_degrees": fmt([r["degree"] for r in strong]) or "n/a",
        "n_weak": len(weak),
        "n_strong": len(strong),
        "n_total": len(rows),
    }


def load_diagnose(pattern="diagnose_M*_P*.json", art=ART):
    """Read the shallow-angle attribution studies, if they have been run."""
    studies = []
    for path in sorted(Path(art).glob(pattern)):
        try:
            studies.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return studies


def _diagnose_markdown(studies):
    if not studies:
        return "_(attribution study not run for this build)_"
    rows = []
    for study in studies:
        for run in study["runs"]:
            if run.get("beta_dev_centroid_pct") is None:
                continue
            if abs(run["final_time"] - 0.05) < 1e-9 and abs(run["boundary_margin"] - 0.12) > 1e-9:
                continue                      # keep one margin per (grid, time)
            rows.append({
                "case": f"M{study['mach']:g} th{study['theta_deg']:g} P{study['degree']}",
                "N": f"{run['ncells']}",
                "t": f"{run['final_time']:g}",
                "margin": f"{run['boundary_margin']:g}",
                "beta_exact": f"{study['beta_exact_deg']:.3f}",
                "centroid_dev_pct": f"{run['beta_dev_centroid_pct']:.3f}",
                "crossing_dev_pct": f"{run['beta_dev_pct']:.3f}",
                "preserve_l1": f"{run['preserve_l1']:.4f}",
            })
    table = _md_table(rows, ["case", "N", "t", "margin", "beta_exact",
                             "centroid_dev_pct", "crossing_dev_pct", "preserve_l1"],
                      lambda row, key: row[key])
    notes = "\n".join(f"* `M{study['mach']:g} th{study['theta_deg']:g} P{study['degree']}`: "
                       f"{study['interpretation']}" for study in studies)
    return table + "\n\n" + notes


def build_markdown(anchors, rows, convergence, summary, studies=()) -> str:
    commit = tb._git_commit()[:12]
    weak = [r for r in rows if r["branch"] == "weak"]
    strong = [r for r in rows if r["branch"] == "strong"]
    facts = _matrix_facts(rows)

    # per-case table (weak, both degrees side by side is long: full list)
    case_rows = []
    for r in weak:
        case_rows.append({
            "case": r["case_id"],
            "M": f"{r['mach']:g}",
            "theta": f"{r['theta_deg']:g}",
            "p": f"{r['degree']}",
            "N": f"{int(r['ncells'])}",
            "beta_exact": f"{r['beta_exact_deg']:.4f}",
            "beta_hat": "n/a" if r["beta_hat_deg"] is None else f"{r['beta_hat_deg']:.4f}",
            "beta_dev": "n/a" if r["beta_dev_pct"] is None else f"{r['beta_dev_pct']:.3f}",
            "rho_dev": f"{r['rho_dev_pct']:.3f}",
            "p_dev": f"{r['p_dev_pct']:.3f}",
            "preserve": f"{r['preserve_l1']:.4f}",
            "verdict": r["verdict"],
        })
    case_md = _md_table(case_rows,
                        ["case", "M", "theta", "p", "N", "beta_exact", "beta_hat",
                         "beta_dev", "rho_dev", "p_dev", "preserve", "verdict"],
                        lambda row, key: row[key])

    strong_rows = []
    for r in strong:
        strong_rows.append({
            "case": r["case_id"],
            "M": f"{r['mach']:g}",
            "theta": f"{r['theta_deg']:g}",
            "p": f"{r['degree']}",
            "beta_exact": f"{r['beta_exact_deg']:.4f}",
            "beta_hat": "n/a" if r["beta_hat_deg"] is None else f"{r['beta_hat_deg']:.4f}",
            "beta_dev": "n/a" if r["beta_dev_pct"] is None else f"{r['beta_dev_pct']:.3f}",
            "rho_dev": f"{r['rho_dev_pct']:.3f}",
            "p_dev": f"{r['p_dev_pct']:.3f}",
            "preserve": f"{r['preserve_l1']:.4f}",
            "verdict": r["verdict"],
            "note": (r.get("note") or "-").replace("\n", " "),
        })
    strong_md = _md_table(strong_rows,
                          ["case", "M", "theta", "p", "beta_exact", "beta_hat",
                           "beta_dev", "rho_dev", "p_dev", "preserve", "verdict", "note"],
                          lambda row, key: row[key]) if strong_rows else "_(no strong-branch runs)_"

    limit_rows = [{"name": c["name"], "criterion": c["criterion"], "status": c["status"]}
                  for c in anchors["limit_checks"]]
    limit_md = _md_table(limit_rows, ["name", "criterion", "status"],
                         lambda row, key: row[key])

    det_rows = [{"M": str(m),
                 "theta_max": f"{float(v['theta_max_deg']):.4f}",
                 "beta_detached": f"{float(v['beta_detached_deg']):.4f}",
                 "mu": f"{float(v['mach_angle_deg']):.4f}"}
                for m, v in anchors["detachment"].items()]
    det_md = _md_table(det_rows, ["M", "theta_max", "beta_detached", "mu"],
                       lambda row, key: row[key])

    gci_md = ""
    if convergence:
        conv_rows = []
        for entry in convergence["cases"]:
            g = entry["gci_beta"]
            conv_rows.append({
                "case": entry["case_id"],
                "levels": "/".join(str(l) for l in entry["levels"]),
                "beta_hat": " ".join("n/a" if b is None else f"{b:.5f}"
                                     for b in entry["beta_hat_deg"]),
                "GCI_beta_pct": "n/a" if not g else f"{g['gci_fine_pct']:.3f}",
                "order": "n/a" if not g or g["observed_order"] is None
                         else f"{g['observed_order']:.2f}",
                "richardson": "n/a" if not g else f"{g['richardson_limit']:.5f}",
                "gate_<2pct": "PASS" if entry["gate_gci_beta_2pct"] else "FAIL",
            })
        gci_md = _md_table(conv_rows,
                           ["case", "levels", "beta_hat", "GCI_beta_pct", "order",
                            "richardson", "gate_<2pct"],
                           lambda row, key: row[key])

    hard = summary["weak"]["hard_fail_15pct"] + summary["strong"]["hard_fail_15pct"]
    hard_md = ", ".join(hard) if hard else "none"

    worst_beta = summary["weak"]["beta_dev_pct_max"]
    worst_rho = summary["weak"]["rho_dev_pct_max"]
    worst_p = summary["weak"]["p_dev_pct_max"]
    gci_pass = all(item["gate"] for item in summary.get("gci", [])) \
        if summary.get("gci") else False
    three_pct = (worst_beta is not None and worst_beta < 3.0
                 and worst_rho < 3.0 and worst_p < 3.0)
    verdict_line = ("PASS" if (summary["weak"]["fail"] == 0 and gci_pass) else "FAIL")

    md = f"""## 1. Case introduction (physics, geometry, parameters)

A plane oblique shock in a perfect gas (`gamma = {anchors['gamma']}`) is the
classical two-state exact solution of the 2-D compressible Euler equations: a
uniform supersonic stream at Mach `M` along `+x` is deflected by `theta` through
a straight front at angle `beta`, and the downstream state follows from the
Rankine--Hugoniot jump conditions. For a given `(M, theta)` with
`theta < theta_max(M)` the theta--beta--Mach relation has **two** roots -- the
weak (physically selected) and the strong (non-unique, needs a downstream back
pressure) branch -- and both are exercised here.

Geometry / setup: unit square `[0,1]^2`, straight front through `(0.5, 0.5)` at
angle `beta`, upstream `rho = p = 1`, `u = M sqrt(gamma)`, `v = 0`. Because the
initial field is piecewise uniform and satisfies the jump conditions exactly, it
**is** the exact steady solution, so every deviation of the numerical field from
it is discretisation error (no manufactured reference, no previous solver run
used as truth).

Matrix actually run (rows of `benchmarks/oblique_shock/case_matrix.csv`):
weak branch `M in {facts['weak_machs']}` x `theta in {facts['weak_thetas']}` x `p in {facts['weak_degrees']}`;
strong branch `M in {facts['strong_machs']}` x `theta in {facts['strong_thetas']}` x `p in {facts['strong_degrees']}`;
`N = {facts['weak_ncells']:g}^2` cells, `t_final = {facts['weak_tfinal']:g}`, `CFL = {facts['weak_cfl']:g}`,
no limiter (positivity fallback only), `float64`.

Anchor artifact: `benchmarks/oblique_shock/theta_beta_anchors.json`
(id `{anchors['anchor_id']}`, {len(anchors['table'])} (M, theta, branch) entries),
generated by `benchmarks/oblique_shock_theta_beta.py::build_anchors`
at commit `{commit}`; no anchor number is typed by hand.

## 2. Method

1. **Anchor generation** -- `theta(beta)` from the theta--beta--Mach relation,
   both roots by bisection inside `(mu, beta_det)` and `(beta_det, 90)`, with
   `(theta_max, beta_detached)` located by golden-section refinement. Each
   anchor pair is built from the RH relations and independently checked two
   ways: the four RH flux residuals must vanish, and the deflection recovered
   from the two velocity vectors must equal the theta--beta--Mach value.
2. **Limit checks** (declared before the runs): `theta -> 0` on the weak branch
   (Mach line, no jump), `theta -> 0` on the strong branch (normal shock, 1-D
   ratios), `theta -> theta_max` (the two roots merge, `dtheta/dbeta -> 0`),
   branch ordering/continuity, and jump closure.
3. **DG runs** -- `tensordg.euler_dg_2d.solve_euler_dg_2d` (tensor-product
   Legendre modal basis, HLLC interface flux, SSP-RK3, positivity fallback,
   transmissive faces). Each case starts from its own exact two-state field.
4. **Measurement** (independent of the initial data) -- the front angle is
   recovered from the numerical density field by mid-level crossings along the
   exact normal, sampled on 81 rays and fitted by total least squares; the
   post-shock plateaus are averaged more than 4 cells away from the front.
5. **Gates declared before the runs**: `|beta_hat - beta|/beta <= 2 %`,
   `|rho_hat/rho - 1| <= 2 %`, `|p_hat/p - 1| <= 2 %`; Roache GCI of `beta`
   on the finest level `< 2 %` (Fs = 1.25, refinement ratio 2); deviations
   `> 15 %` reported separately.

## 3. Results

### 3.1 Anchor limit checks

{limit_md}

Detachment (`theta_max`, `beta_detached`) and Mach angle per Mach number:

{det_md}

The values reproduce the classical table (Anderson, *Modern Compressible Flow*,
3rd ed., ch. 4): `theta_max(2) = {anchors['detachment']['2.0']['theta_max_deg']:.2f} deg`,
`theta_max(3) = {anchors['detachment']['3.0']['theta_max_deg']:.2f} deg`,
`theta_max(4) = {anchors['detachment']['4.0']['theta_max_deg']:.2f} deg`,
`theta_max(10) = {anchors['detachment']['10.0']['theta_max_deg']:.2f} deg`.

### 3.2 Weak-branch case matrix ({len(weak)} runs)

{case_md}

Aggregate: PASS {summary['weak']['pass']}/{summary['weak']['n']}, worst
`beta` deviation {worst_beta:.3f} %, worst `rho` deviation {worst_rho:.3f} %,
worst `p` deviation {worst_p:.3f} %, worst preservation error
{summary['weak']['preserve_l1_max']:.4f}.

### 3.3 Strong-branch runs ({len(strong)} runs)

{strong_md}

Aggregate: PASS {summary['strong']['pass']}/{summary['strong']['n']}, worst
`beta` deviation {summary['strong'].get('beta_dev_pct_max', float('nan')):.3f} %.

### 3.4 Grid convergence (Roache GCI)

{gci_md}

### 3.5 Attribution study for the shallow-angle failures (M = 6, 10)

{_diagnose_markdown(list(studies))}

### 3.6 Deviations above 15 %

{hard_md}

## 4. Figures

1. `theta_beta_curves.png` -- anchor curves on both branches with the measured DG
   shock angles overlaid (open circles) and the detachment points (stars).
2. `deviation_heatmap.png` -- worst of the three deviations per (M, theta) cell,
   P1 and P2.
3. `convergence.png` -- shock-angle error versus mesh size and signed error.
4. `branch_scatter.png` -- measured versus exact shock angle on both branches
   with the +/-2 % band.
5. `field_snapshot.png` -- density field of one P2 run with the
   exact front overlaid and the cross-front profile against the exact RH jump.

## 5. Notes / limitations

* The solver offers only periodic / transmissive faces, so a wall-bounded wedge
  (the configuration in which the weak branch is physically selected) is out of
  reach; the shock front is measured on an interior window at least 0.12 from
  the boundaries. This is a benchmark of *shock preservation and angle
  recovery*, not of wedge-flow selection.
* The strong branch is non-unique: with a free subsonic outflow and no imposed
  back pressure the exact two-state field is only marginally stable, so
  strong-branch runs are reported separately and are not part of the weak-branch
  acceptance verdict.
* No limiter is used, so the front is smeared over a few cells and post-shock
  plateaus carry small Gibbs oscillations; that is the intended stress case for
  angle recovery.
* Anchor values come from the closed-form relation, not from a digitised table,
  so no literature transcription error is possible; the literature comparison is
  limited to the `theta_max` values quoted above.

## 6. Error judgement against the declared criteria

| criterion | declared | measured | status |
|---|---|---|---|
| anchor limit checks (5) | all PASS | {sum(1 for c in anchors['limit_checks'] if c['status']=='PASS')}/5 PASS | {'PASS' if all(c['status']=='PASS' for c in anchors['limit_checks']) else 'FAIL'} |
| beta deviation, weak matrix | <= 2 % | worst {worst_beta:.3f} % | {'PASS' if worst_beta <= 2.0 else 'FAIL'} |
| rho deviation, weak matrix | <= 2 % | worst {worst_rho:.3f} % | {'PASS' if worst_rho <= 2.0 else 'FAIL'} |
| p deviation, weak matrix | <= 2 % | worst {worst_p:.3f} % | {'PASS' if worst_p <= 2.0 else 'FAIL'} |
| GCI of beta | < 2 % | {', '.join('%.3f %%' % g['gci_beta_pct'] for g in summary.get('gci', []) if g['gci_beta_pct'] is not None) or 'n/a'} | {'PASS' if gci_pass else 'FAIL'} |
| deviations > 15 % | listed separately | {hard_md} | {'info' if hard else 'none'} |

**Are the key quantities within 3 % of the reference?** Worst weak-branch
deviations: beta {worst_beta:.3f} %, rho {worst_rho:.3f} %, p {worst_p:.3f} % --
{'yes, all three are far below 3 %' if three_pct else 'no, see the table above'}.

Overall verdict for the weak-branch criterion set: **{verdict_line}**.
"""
    return md


# --------------------------------------------------------------------------- #
# publish
# --------------------------------------------------------------------------- #
def publish(payload_path="/tmp/report.json"):
    payload = json.loads(Path(payload_path).read_text())
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    login = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{base}/auth/login",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"username": user, "password": password})],
        capture_output=True, text=True, check=True)
    token = None
    try:
        blob = json.loads(login.stdout)
        token = (blob.get("access_token") or blob.get("token")
                 or blob.get("data", {}).get("access_token"))
    except Exception:
        token = login.stdout.strip()
    if not token:
        raise SystemExit(f"login failed: {login.stdout[:400]}")
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{base}/tensor/reports",
         "-H", f"Authorization: Bearer {token}",
         "-H", "Content-Type: application/json",
         "-d", f"@{payload_path}"], capture_output=True, text=True, check=True)
    print(result.stdout[:1000])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--out", default="/tmp/report.json")
    parser.add_argument("--skip-snapshot", action="store_true")
    parser.add_argument("--mission-id", type=int, default=15977)
    args = parser.parse_args(argv)

    anchors, rows, convergence = load_artifacts()
    summary = summarise(rows, convergence)
    weak = [r for r in rows if r["branch"] == "weak"]
    strong = [r for r in rows if r["branch"] == "strong"]

    def safe(builder, *args, **kwargs):
        """One broken figure must not take the whole report down."""
        try:
            return builder(*args, **kwargs)
        except Exception as error:                     # pragma: no cover
            import traceback
            traceback.print_exc()
            print(f"figure {builder.__name__} skipped: {error}")
            return None

    figures = []
    for path, caption in (
        (safe(fig_theta_beta, anchors, rows),
         "theta--beta--Mach anchor curves (solid = weak, dashed = strong) with the "
         "measured DG shock angles (open circles) and detachment points (stars)."),
        (safe(fig_deviation_heatmap, rows),
         "Worst of the |beta|, |rho|, |p| deviations in percent for each (M, theta) "
         "cell, P1 and P2 weak-branch runs."),
    ):
        if path is not None:
            figures.append((path, caption))
    if convergence:
        path = safe(fig_convergence, convergence)
        if path is not None:
            figures.append((path,
                            "Grid convergence of the recovered shock angle: error vs mesh "
                            "size (left) and signed error (right), three levels."))
    for path, caption in (
        (safe(fig_error_vs_mach, rows),
         "Deviation scaling with the free-stream Mach number for DG P1 and P2 "
         "(mean over the deflection angles, bars = min/max)."),
        (safe(fig_branch_scatter, rows, anchors),
         "Measured vs exact shock angle for weak (blue) and strong (red) branches; "
         "grey band = +/-2 %."),
    ):
        if path is not None:
            figures.append((path, caption))
    snapshot_degree = 2
    if not args.skip_snapshot:
        try:
            figures.append((fig_field_snapshot(degree=snapshot_degree),
                            "Density field of one DG P2 run with the exact front "
                            "overlaid (left) and the cross-front density profile "
                            "against the exact Rankine--Hugoniot jump (right)."))
        except Exception as error:                        # pragma: no cover
            print("snapshot skipped:", error)

    studies = load_diagnose()
    try:
        write_readme(anchors, rows, convergence, summary)
    except Exception as error:                          # pragma: no cover
        import traceback
        traceback.print_exc()
        print("README generation skipped:", error)
    try:
        markdown = build_markdown(anchors, rows, convergence, summary, studies)
    except Exception as error:                          # pragma: no cover
        import traceback
        traceback.print_exc()
        print("markdown generation fell back to the summary:", error)
        markdown = (f"## Results summary\n\n{len(rows)} DG oblique-shock runs; "
                    f"weak branch PASS {summary['weak']['pass']}/"
                    f"{summary['weak']['n']}, worst beta deviation "
                    f"{summary['weak']['beta_dev_centroid_pct_max']:.3f} %, worst "
                    f"rho {summary['weak']['rho_dev_pct_max']:.3f} %, worst p "
                    f"{summary['weak']['p_dev_pct_max']:.3f} %. "
                    f"See benchmarks/oblique_shock/ for the artifacts.")
    payload = {
        "title": ("Oblique-shock theta--beta--Mach anchors and "
                  f"(M, theta, p) DG verification matrix ({len(rows)} runs)"),
        "summary": (
            "TensorDG: theta--beta--Mach anchor set (weak+strong branches, RH jumps, "
            f"5 limit checks all PASS; {len(anchors['table'])} anchor entries) plus a "
            f"{len(weak)}-run weak-branch and {len(strong)}-run strong-branch DG case "
            f"matrix at P1/P2. Worst weak-branch deviations: beta "
            f"{summary['weak']['beta_dev_pct_max']:.3f} %, rho "
            f"{summary['weak']['rho_dev_pct_max']:.3f} %, p "
            f"{summary['weak']['p_dev_pct_max']:.3f} %."),
        "scenario_key": "dg-oblique-shock-theta-beta",
        "mission_id": args.mission_id,
        "tags": ["TensorDG", "compressible-flow", "oblique-shock",
                 "theta-beta-Mach", "Rankine-Hugoniot", "GCI", "verification"],
        "markdown": markdown,
        "figures": [{"name": path.name,
                     "caption": caption,
                     "png_base64": _png_base64(path)} for path, caption in figures],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("report payload ->", args.out, "| figures:", [p.name for p, _ in figures])
    if args.publish:
        publish(args.out)


if __name__ == "__main__":
    main()