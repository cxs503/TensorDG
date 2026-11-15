"""Set-level feature gate on the round-2 certified reference (Shu--Osher case).

The round-2 practice certifies a *profile* (global L1 against a Richardson
reference) and, in a second step, individual *functionals* of the wave train.
``docs/XVAL-DG-ROUND2-FEATURE-CERTIFICATION.md`` closed with a recommendation
that is still unimplemented as a machine-checkable rule:

    "在 CSV/平台验收中补一条功能量级门槛（>=1 个积分型 + 1 个极值型，均带宽感知）"

This driver implements exactly that rule (:mod:`tensordg.feature_gate`) and
re-runs it on the **already committed** cell-average profiles -- no new
simulation, so every number below is reproducible from
``validation/repro_shu_osher/artifacts/raw/*.npz`` alone.

Two questions are answered, both with the estimator's own construction as the
only new ingredient:

Q1  Does the extremum estimator change the certified functional?
    The cell-average extremum is biased by ``h^2/24 f''``.  On this ladder the
    bias decays with a measured exponent (expect 2) and is *comparable to the
    finest-level solution increment*, so the raw ``crest`` ladder's apparent
    monotone convergence is partly a property of the measurement.  Both
    estimators are reported; neither is silently "fixed".

Q2  Does the gate separate L1-passing legs from feature-passing legs?
    Every DG leg of the round-2 ladder is judged by the profile gate
    (published) and by the feature gate (this driver).  A leg that passes L1
    and fails the feature gate is the counterexample the recommendation was
    written for, and it is counted.

Usage::

    PYTHONPATH=. python -m benchmarks.xval_dg_round2_feature_gate            # JSON to stdout
    PYTHONPATH=. python -m benchmarks.xval_dg_round2_feature_gate --figures  # + PNGs
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensordg.certified_reference import CERTIFIED, certify_functional, judge_functional
from tensordg.feature_gate import (
    DEFAULT_REQUIRED,
    GATE_PASS,
    GATE_PASS_MARGINAL,
    cell_average_extremum,
    gate_teeth_check,
    judge_feature_set,
)
from benchmarks.xval_dg_round2_feature_certification import (
    DG_LEGS,
    FEATURES,
    GATE_PERCENT,
    ROUND2_AUDIT,
    U_SHARE_MAX,
    leg_path,
    load_profile,
    round2_profile_verdicts,
    wave_window,
)

ANCHOR = "XVAL-DG-ROUND2-CERTIFIED-REFERENCE"
CASE = "CASE-SHU-OSHER (1-D Euler, M=3 shock + entropy wave train, t = 1.8)"
FV_LEVELS = (800, 1600, 3200, 6400)
OUTDIR = Path("validation/xval_dg_round2/artifacts/feature_gate")

#: functional -> kind, as used by the set-level gate
KINDS = {
    "rms": "integral",
    "l1": "integral",
    "amplitude": "extremum",
    "crest": "extremum",
    "trough": "extremum",
    "peak_x": "phase",
}
#: the recommendation under test: >= 1 integral + >= 1 extremum
RECOMMENDED = {"integral": 1, "extremum": 1}
#: a stricter variant that also demands the extremum *location*
PHASE_ALSO = {"integral": 1, "extremum": 1, "phase": 1}


# --------------------------------------------------------------- measurements
def window_slice(x, rho, window):
    x_lo, x_hi, _plateau = window
    mask = (x >= x_lo) & (x <= x_hi)
    if mask.sum() < 8:
        raise ValueError(f"only {int(mask.sum())} cells inside the wave window")
    return np.asarray(x, float)[mask], np.asarray(rho, float)[mask]


def functionals(x, rho, *, window=None, estimator: str = "raw") -> dict:
    """The gate's functional set, measured with one extremum estimator.

    ``estimator='raw'`` reproduces the published measurement (cell-average
    argmax / plain max); ``'deconvolved'`` uses
    :func:`tensordg.feature_gate.cell_average_extremum`.  Integral functionals
    are identical under both estimators by construction -- that is the point of
    the split.
    """
    x_lo, x_hi, plateau = window if window is not None else wave_window()
    xr, rr = window_slice(x, rho, (x_lo, x_hi, plateau))
    dev = rr - plateau
    if estimator == "raw":
        crest, trough = float(rr.max()), float(rr.min())
        peak_x = float(xr[int(np.argmax(rr))])
        status = ("raw_cell_average", "raw_cell_average")
    elif estimator == "deconvolved":
        hi = cell_average_extremum(rr, x=xr, mode="max")
        lo = cell_average_extremum(rr, x=xr, mode="min")
        crest, trough = hi["value"], lo["value"]
        if hi["status"] == "ok":
            peak_x = hi["position"]
        else:                        # degraded estimate, reported as such
            peak_x = hi["raw_position"]
        status = (hi["status"], lo["status"])
    else:
        raise ValueError(f"estimator must be 'raw' or 'deconvolved', got {estimator!r}")
    return {
        "crest": float(crest),
        "trough": float(trough),
        "amplitude": float(0.5 * (crest - trough)),
        "peak_x": float(peak_x),
        "rms": float(math.sqrt(float(np.mean(dev ** 2)))),
        "l1": float(np.mean(np.abs(dev))),
        "estimator_status": list(status),
        "n_cells_in_window": int(rr.size),
    }


def bias_study(window) -> dict:
    """Measured ``h^2/24 f''`` bias of the crest estimator per FV level."""
    rows = []
    for n in FV_LEVELS:
        x, rho = load_profile(Path("validation/repro_shu_osher/artifacts/raw")
                              / f"fv_cfl0p3_n{n}.npz")
        xr, rr = window_slice(x, rho, window)
        est = cell_average_extremum(rr, x=xr, mode="max")
        rows.append({"ncells": n, "h": float(xr[1] - xr[0]),
                     "bias": float(est["bias_correction"]),
                     "curvature": float(est["curvature"]),
                     "status": est["status"]})
    exps = []
    for a, b in zip(rows[:-1], rows[1:]):
        if a["bias"] != 0.0 and b["bias"] != 0.0:
            exps.append(math.log(abs(a["bias"] / b["bias"])) / math.log(a["h"] / b["h"]))
    return {"levels": rows, "bias_decay_exponents": exps,
            "expected_exponent": 2.0,
            "law_verified": bool(exps and all(abs(e - 2.0) < 0.25 for e in exps))}


# ------------------------------------------------------------------- ladders
def fv_ladder(estimator: str, window) -> list[dict]:
    out = []
    for n in FV_LEVELS:
        x, rho = load_profile(Path("validation/repro_shu_osher/artifacts/raw")
                              / f"fv_cfl0p3_n{n}.npz")
        out.append({"ncells": n, **functionals(x, rho, window=window,
                                               estimator=estimator)})
    return out


def certify_ladder(levels: list[dict], *, tolerance_percent: float) -> dict:
    out = {}
    names = sorted({n for lv in levels for n in lv if n in KINDS})
    for name in names:
        cert = certify_functional([lv[name] for lv in levels],
                                  tolerance_percent=tolerance_percent,
                                  u_share_max=U_SHARE_MAX)
        cert["kind"] = KINDS[name]
        out[name] = cert
    return out


def dg_legs(estimator: str, window) -> list[dict]:
    rows = []
    for degree, ncells, cfl in DG_LEGS:
        path = leg_path(ncells, degree, cfl)
        x, rho = load_profile(path)
        rows.append({
            "configuration": f"DG P{degree} N={ncells} CFL={cfl}",
            "file": path.name, "degree": degree, "ncells": ncells, "cfl": cfl,
            "functionals": functionals(x, rho, window=window, estimator=estimator),
        })
    return rows


# --------------------------------------------------------------------- audit
def run_audit(*, gate_percent: float = GATE_PERCENT) -> dict:
    window = wave_window()
    bias = bias_study(window)
    references, ladders = {}, {}
    for estimator in ("raw", "deconvolved"):
        ladders[estimator] = fv_ladder(estimator, window)
        references[estimator] = certify_ladder(ladders[estimator],
                                               tolerance_percent=gate_percent)

    teeth = gate_teeth_check(references["deconvolved"], tolerance_percent=gate_percent,
                             kinds=KINDS, required=RECOMMENDED)

    profile_verdicts = round2_profile_verdicts()
    legs = {}
    for estimator in ("raw", "deconvolved"):
        rows = dg_legs(estimator, window)
        for row in rows:
            refs = references[estimator]
            candidates = {n: row["functionals"][n] for n in refs}
            gate = judge_feature_set(candidates, references=refs,
                                     tolerance_percent=gate_percent, kinds=KINDS,
                                     required=RECOMMENDED)
            gate_phase = judge_feature_set(candidates, references=refs,
                                           tolerance_percent=gate_percent, kinds=KINDS,
                                           required=PHASE_ALSO)
            prof = profile_verdicts.get(row["file"], {})
            legs.setdefault(row["file"], {"configuration": row["configuration"],
                                          "ncells": row["ncells"],
                                          "degree": row["degree"], "cfl": row["cfl"]})
            entry = legs[row["file"]]
            entry["profile_rel_l1_percent"] = prof.get("rel_l1_percent")
            entry["profile_decision"] = prof.get("decision")
            entry[f"gate_{estimator}"] = gate["verdict"]
            entry[f"gate_{estimator}_phase"] = gate_phase["verdict"]
            entry[f"failures_{estimator}"] = {
                n: {"decision": e["decision"], "error_percent": e["error_percent"],
                    "reference": e["reference"]}
                for n, e in gate["functionals"].items()
                if e["decision"] not in ("PASS", "PASS_MARGINAL")}
            if estimator == "deconvolved":
                entry["functionals"] = {
                    n: row["functionals"][n] for n in
                    ("crest", "trough", "amplitude", "peak_x", "rms", "l1")}

    # --- the counterexample count the recommendation was written for
    def _passes(decision):
        return decision in (GATE_PASS, GATE_PASS_MARGINAL)

    profile_pass = [k for k, v in legs.items() if v.get("profile_decision") == "PASS"]
    contradiction = [k for k in profile_pass
                     if not _passes(legs[k]["gate_deconvolved"])]
    summary = {
        "n_legs": len(legs),
        "n_profile_pass": len(profile_pass),
        "n_feature_pass_deconvolved": sum(_passes(v["gate_deconvolved"])
                                          for v in legs.values()),
        "n_feature_pass_raw": sum(_passes(v["gate_raw"]) for v in legs.values()),
        "n_profile_pass_feature_fail": len(contradiction),
        "contradiction_files": sorted(contradiction),
        "profile_gate_sufficient": not contradiction,
    }

    # --- functional certification under both estimators (Q1)
    feature_cert = {}
    for name in sorted(KINDS):
        entry = {"kind": KINDS[name]}
        for estimator in ("raw", "deconvolved"):
            cert = references[estimator].get(name, {})
            entry[estimator] = {
                "verdict": cert.get("verdict"),
                "observed_order": cert.get("observed_order"),
                "classification": cert.get("classification"),
                "values": cert.get("values"),
                "certified_value": cert.get("certified_value"),
                "band_percent": cert.get("band_percent"),
                "band_share_of_gate": None if cert.get("band_percent") is None
                else cert["band_percent"] / gate_percent,
                "gate_usable": bool(cert.get("verdict") == CERTIFIED
                                    and cert.get("band_percent") is not None
                                    and cert["band_percent"] / gate_percent <= U_SHARE_MAX),
            }
        flips = (entry["raw"]["verdict"] != entry["deconvolved"]["verdict"])
        entry["verdict_changed_by_estimator"] = bool(flips)
        feature_cert[name] = entry

    gate_inputs = {
        estimator: sorted(n for n, c in refs.items()
                          if c.get("verdict") == CERTIFIED
                          and c.get("band_percent") is not None
                          and c["band_percent"] / gate_percent <= U_SHARE_MAX)
        for estimator, refs in references.items()}

    return {
        "anchor": ANCHOR,
        "case": CASE,
        "gate_percent": gate_percent,
        "u_share_max": U_SHARE_MAX,
        "window": list(wave_window()),
        "estimator_bias": bias,
        "reference_ladders": ladders,
        "feature_certification": feature_cert,
        "gate_inputs": gate_inputs,
        "required_recommended": RECOMMENDED,
        "required_phase_also": PHASE_ALSO,
        "gate_teeth": teeth,
        "legs": legs,
        "summary": summary,
        "profile_source": str(ROUND2_AUDIT),
        "data_source": "validation/repro_shu_osher/artifacts/raw/*.npz (committed)",
    }


# -------------------------------------------------------------------- report
def _fmt(value, digits=6):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def to_markdown(audit: dict) -> str:
    gate_percent = audit["gate_percent"]
    lines = [
        f"# Set-level feature gate — {audit['anchor']}",
        "",
        f"Case: {audit['case']}  ·  gate {gate_percent:g} %  ·  "
        f"band ceiling {100 * audit['u_share_max']:.1f} % of gate",
        "",
        "## 1. Cell-average extremum bias on the committed profiles",
        "",
        "| N | h | bias `h²/24 f''` | f'' | status |",
        "|---|---|---|---|---|",
    ]
    for row in audit["estimator_bias"]["levels"]:
        lines.append(f"| {row['ncells']} | {row['h']:.3e} | {row['bias']:.6e} | "
                     f"{row['curvature']:.6e} | {row['status']} |")
    exps = ", ".join(f"{e:.3f}" for e in audit["estimator_bias"]["bias_decay_exponents"])
    lines += [
        "",
        f"Measured decay exponent(s): {exps} (expected "
        f"{audit['estimator_bias']['expected_exponent']:g}) → "
        f"**law verified = {audit['estimator_bias']['law_verified']}**",
        "",
        "## 2. Functional certification under both estimators (Q1)",
        "",
        "| functional | kind | raw verdict | raw p | raw band % | deconvolved verdict "
        "| deconv. p | deconv. band % | verdict changed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in sorted(audit["feature_certification"].items()):
        r, d = entry["raw"], entry["deconvolved"]
        lines.append(
            f"| {name} | {entry['kind']} | {r['verdict']} | {_fmt(r['observed_order'], 4)} "
            f"| {_fmt(r['band_percent'], 4)} | {d['verdict']} | {_fmt(d['observed_order'], 4)} "
            f"| {_fmt(d['band_percent'], 4)} | **{entry['verdict_changed_by_estimator']}** |")
    lines += [
        "",
        f"Gate inputs (usable certified references) — raw: "
        f"{', '.join(audit['gate_inputs']['raw']) or 'none'}; deconvolved: "
        f"{', '.join(audit['gate_inputs']['deconvolved']) or 'none'}",
        "",
        f"Gate teeth check: {audit['gate_teeth']}",
        "",
        "## 3. Feature gate vs profile gate on the 11 DG legs (Q2)",
        "",
        f"Required quotas: {audit['required_recommended']}",
        "",
        "| leg | profile rel L1 % | profile gate | feature gate (raw) | "
        "feature gate (deconvolved) |",
        "|---|---|---|---|---|",
    ]
    for fname in sorted(audit["legs"], key=lambda k: (audit["legs"][k]["degree"],
                                                      audit["legs"][k]["ncells"],
                                                      audit["legs"][k]["cfl"])):
        leg = audit["legs"][fname]
        lines.append(f"| {leg['configuration']} | {_fmt(leg.get('profile_rel_l1_percent'), 4)} "
                     f"| {leg.get('profile_decision')} | {leg['gate_raw']} | "
                     f"{leg['gate_deconvolved']} |")
    s = audit["summary"]
    lines += [
        "",
        f"**Summary**: {s['n_profile_pass']}/{s['n_legs']} legs pass the profile gate; "
        f"{s['n_feature_pass_deconvolved']}/{s['n_legs']} pass the feature gate "
        f"(deconvolved reference). Legs that pass the profile gate and fail the feature "
        f"gate: **{s['n_profile_pass_feature_fail']}** → profile gate sufficient = "
        f"**{s['profile_gate_sufficient']}**.",
        "",
    ]
    return "\n".join(lines)


def make_figures(audit: dict, outdir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    made = []

    # fig1: crest/amplitude ladder, raw vs deconvolved, and the bias law
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ladders = audit["reference_ladders"]
    ns = [lv["ncells"] for lv in ladders["raw"]]
    for name, ax in (("crest", axes[0]), ("amplitude", axes[1])):
        for est, style in (("raw", "o--"), ("deconvolved", "s-")):
            vals = [lv[name] for lv in ladders[est]]
            ref = [lv["ncells"] for lv in ladders[est]]
            ax.plot(ref, vals, style, label=est)
        ax.set_xscale("log")
        ax.set_xlabel("N (cells)")
        ax.set_ylabel(name)
        ax.set_title(f"{name}: raw vs deconvolved")
        ax.grid(alpha=0.3)
        ax.legend()
    bias_ns = [r["ncells"] for r in audit["estimator_bias"]["levels"]]
    bias = [abs(r["bias"]) for r in audit["estimator_bias"]["levels"]]
    axb = axes[0].twinx()
    axb.loglog(bias_ns, bias, "k^:")
    axb.set_ylabel("|bias| = h²/24 f''")
    path = outdir / "fig1_ladder_estimators.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)

    # fig2: per-leg relative error of each gate functional (deconvolved reference)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    names = ["rms", "amplitude", "crest", "peak_x"]
    files = sorted(audit["legs"], key=lambda k: (audit["legs"][k]["degree"],
                                                 audit["legs"][k]["ncells"]))
    labels = [audit["legs"][f]["configuration"].replace("DG ", "") for f in files]
    width = 0.2
    for i, name in enumerate(names):
        errs = []
        for f in files:
            entry = audit["legs"][f].get(f"failures_deconvolved", {})
            if name in entry and entry[name]["error_percent"] is not None:
                errs.append(entry[name]["error_percent"])
            else:
                errs.append(0.05)          # passed: plotted at a floor value
        x = np.arange(len(files)) + (i - 1.5) * width
        ax.bar(x, errs, width, label=name)
    ax.set_yscale("log")
    ax.axhline(audit["gate_percent"], color="r", ls="--", label=f"gate {audit['gate_percent']:g} %")
    ax.set_xticks(np.arange(len(files)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("relative error % (0.05 = passed)")
    ax.set_title("Feature errors per DG leg (deconvolved reference); absent bars = pass")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = outdir / "fig2_feature_errors.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)

    # fig3: two profiles with the wave-window extremum zoomed
    fig, ax = plt.subplots(figsize=(9, 4.2))
    window = audit["window"]
    for name, style in (("fv_cfl0p3_n800", "C0-"), ("fv_cfl0p3_n6400", "C1-")):
        x, rho = load_profile(Path("validation/repro_shu_osher/artifacts/raw")
                              / f"{name}.npz")
        mask = (x >= window[0]) & (x <= window[1])
        ax.plot(x[mask], rho[mask], style, lw=1.2, label=name)
        est = cell_average_extremum(rho[mask], x=x[mask], mode="max")
        ax.plot(est["position"], est["value"], "k*", ms=12,
                label=f"{name}: deconvolved crest")
        ax.plot(est["raw_position"], est["raw_value"], "kx", ms=7,
                label=f"{name}: raw argmax")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("Wave train extremum: raw cell-average argmax vs deconvolved vertex")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / "fig3_extremum_zoom.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)
    return made


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-stdout", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--gate-percent", type=float, default=GATE_PERCENT)
    args = parser.parse_args(argv)

    audit = run_audit(gate_percent=args.gate_percent)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "feature_gate_audit.json").write_text(json.dumps(audit, indent=1, default=float))
    (outdir / "feature_gate_audit.md").write_text(to_markdown(audit))
    if args.figures:
        for path in make_figures(audit, outdir):
            print(f"figure: {path}")
    if args.json_stdout:
        print(json.dumps(audit, indent=1, default=float))
    else:
        print(to_markdown(audit))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())