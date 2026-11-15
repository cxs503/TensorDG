"""Feature-level certification of the CASE-SHU-OSHER round-2 reference.

What gap this closes
--------------------
Round 2 of CASE-SHU-OSHER certified the *whole density profile*: a Richardson
extrapolation over a four-level FV-WENO5 ladder, a GCI band ``U``, and a 3 %
gate applied to the relative L1 norm of eleven DG legs.  Every DG leg passes
(``rel_l1`` <= 2.45 %).  That is a **global** statement, and a global L1 norm is
dominated by the shock, where the solution varies by O(1) over a few cells; the
wave train behind the shock -- the actual physics content of the case, the
amplified entropy wave that schemes either resolve or dissipate -- carries a
small fraction of the L1 budget and can be badly wrong while the profile gate
still reads PASS.

So the question this audit answers is not "is the profile right" (that was round
2) but: **are the reference and the DG legs right *feature by feature*, with the
reference's own feature-level uncertainty quantified?**  The engine side of the
answer is :func:`tensordg.certified_reference.certify_functional`, Roache's
method applied to a scalar functional instead of a profile.

Measured functionals (region ``R`` = the wave train, bounded by the case's own
closed forms: head ``x_head`` and shock position ``x_shock`` at ``t = 1.8``,
inset by 0.05 / 0.10 so the shock's own smearing cells do not enter the
statistics)::

    amplitude  = 0.5 * (max rho - min rho)              over R   (extremum)
    crest      = max rho                                over R   (extremum)
    rms        = sqrt(mean (rho - rho_plateau)**2)      over R   (integral)
    l1         = mean |rho - rho_plateau|               over R   (integral)
    peak_x     = x at max rho                           over R   (phase)

``rho_plateau`` is the Rankine-Hugoniot plateau from the case module, i.e. an
analytic constant, so the two integral functionals are well defined without
this script inventing a mean.

Predeclared decision rules (declared before the numbers were looked at)
----------------------------------------------------------------------
``G0`` certifiability   the WENO5 functional ladder is CERTIFIED (asymptotic,
                        p >= 0.5).  Otherwise the feature is WITHHELD -- there is
                        no reference value to gate against.
``G1`` gate validity    the feature band consumes at most 1/3 of the 3 % gate
                        (``U_F / tau <= 1/3``), otherwise WITHHELD: the feature
                        gate would be vacuous.
``G2`` family honesty   |F_weno5 - F_weno7| <= U5 + U7 (sum, conservative).  A
                        feature both families certify but disagree on beyond
                        their bands is *not* a certified feature.
``G3`` leg fidelity     each DG leg is judged per feature at ``tau = 3 %`` with
                        the band-aware decision (PASS / PASS_MARGINAL / FAIL).
``G4`` gate honesty     the round-2 profile gate is called *sufficient* only if
                        no leg that PASSes the 3 % profile gate FAILs a
                        feature gate of G1-passing features.  Any such leg is a
                        concrete counterexample and is reported as one.

Everything is computed from committed ``.npz`` profiles; no solver is run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tensordg.cases.shock_entropy import rankine_hugoniot_anchor  # noqa: E402
from tensordg.certified_reference import (  # noqa: E402
    CERTIFIED,
    FAIL,
    PASS,
    PASS_MARGINAL,
    WITHHELD,
    certify_functional,
    judge_functional,
)

FV_LEVELS = (800, 1600, 3200, 6400)
FV_WENO5_DIR = "validation/repro_shu_osher/artifacts/raw"
FV_WENO7_DIR = "validation/xval_dg_round2/artifacts/band_honesty/raw_weno7"
DG_RAW_DIR = "validation/repro_shu_osher/artifacts/raw"
ROUND2_AUDIT = "validation/xval_dg_round2/artifacts/xval_round2_audit.json"
OUTDIR = Path("validation/xval_dg_round2/artifacts/feature_certification")

DOMAIN = (-5.0, 5.0)
GATE_PERCENT = 3.0
U_SHARE_MAX = 1.0 / 3.0
#: inset of the statistics window away from the analytic head / shock positions
HEAD_INSET = 0.05
SHOCK_INSET = 0.10

DG_LEGS = (
    (1, 200, 0.4), (1, 400, 0.4), (1, 800, 0.4),
    (2, 200, 0.4), (2, 400, 0.4), (2, 800, 0.4),
    (3, 200, 0.4), (3, 400, 0.4), (3, 800, 0.4),
    (2, 400, 0.1), (2, 400, 0.2),
)

FEATURES = ("amplitude", "crest", "rms", "l1", "peak_x")
#: extremum-type functionals carry an O(h) sampling bias (see module docstring of
#: ``certify_functional``); integral-type ones do not
FEATURE_KIND = {"amplitude": "extremum", "crest": "extremum", "rms": "integral",
                "l1": "integral", "peak_x": "phase"}


# ---------------------------------------------------------------- data access
def load_profile(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as d:
        return np.asarray(d["x"], float), np.asarray(d["rho"], float)


def leg_path(ncells: int, degree: int, cfl: float, rawdir: str = DG_RAW_DIR) -> Path:
    tag = f"{cfl:g}".replace(".", "p")
    return Path(rawdir) / f"dg_cfl{tag}_limweno_n{ncells}_p{degree}.npz"


def wave_window() -> tuple[float, float, float]:
    """(x_lo, x_hi, rho_plateau) of the wave-train statistics window."""
    rh = rankine_hugoniot_anchor()
    x_head = float(rh["quiet_zone_head_analytic"]) + HEAD_INSET
    x_shock = float(rh["shock_position_analytic"]) - SHOCK_INSET
    plateau = float(rh["post_shock_plateau"]["rho"])
    if not x_head < x_shock:
        raise ValueError("wave window collapsed: check the analytic anchors")
    return x_head, x_shock, plateau


# ---------------------------------------------------------------- features
def features(x, rho, *, window: tuple[float, float, float] | None = None) -> dict:
    """The five wave-train functionals of one density profile."""
    x = np.asarray(x, float)
    rho = np.asarray(rho, float)
    if x.shape != rho.shape:
        raise ValueError(f"x {x.shape} and rho {rho.shape} differ")
    x_lo, x_hi, plateau = window if window is not None else wave_window()
    mask = (x >= x_lo) & (x <= x_hi)
    if mask.sum() < 8:
        raise ValueError(f"only {int(mask.sum())} cells inside the wave window "
                         f"[{x_lo:.3f}, {x_hi:.3f}] -- grid too coarse for this audit")
    r = rho[mask]
    xr = x[mask]
    dev = r - plateau
    j = int(np.argmax(r))
    return {
        "amplitude": float(0.5 * (r.max() - r.min())),
        "crest": float(r.max()),
        "rms": float(math.sqrt(float(np.mean(dev ** 2)))),
        "l1": float(np.mean(np.abs(dev))),
        "peak_x": float(xr[j]),
        "n_cells_in_window": int(mask.sum()),
        "window": [float(x_lo), float(x_hi)],
        "rho_plateau": float(plateau),
    }


# ---------------------------------------------------------------- ladders
def fv_ladder(directory: str, levels=FV_LEVELS,
              stem: str = "fv_cfl0p3_n") -> list[dict]:
    out = []
    for n in levels:
        x, rho = load_profile(Path(directory) / f"{stem}{n}.npz")
        out.append({"ncells": n, "x": x, "rho": rho})
    return out


def certify_features(ladder: list[dict], *, window=None,
                     tolerance_percent: float = GATE_PERCENT) -> dict:
    """Certify every functional on one nesting ladder."""
    per_level = [features(lv["x"], lv["rho"], window=window) for lv in ladder]
    out = {}
    for name in FEATURES:
        values = [lv[name] for lv in per_level]
        cert = certify_functional(values, tolerance_percent=tolerance_percent,
                                  u_share_max=U_SHARE_MAX)
        cert["ncells"] = [lv["ncells"] for lv in ladder]
        cert["kind"] = FEATURE_KIND[name]
        out[name] = cert
    return {"features": out, "per_level": per_level,
            "ncells": [lv["ncells"] for lv in ladder]}


def round2_profile_verdicts(path: Path | str = ROUND2_AUDIT) -> dict:
    """The shipped round-2 global-L1 verdict of each DG leg, keyed by file name."""
    p = Path(path)
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {leg["file"]: {"rel_l1_percent": leg.get("rel_l1_percent_3pct_gate"),
                          "decision": leg.get("decision_3pct")}
            for leg in d.get("legs", [])}


# ---------------------------------------------------------------- decisions
def run_audit(*, weno5_dir: str = FV_WENO5_DIR, weno7_dir: str = FV_WENO7_DIR,
          dg_dir: str = DG_RAW_DIR, gate_percent: float = GATE_PERCENT) -> dict:
    window = wave_window()
    w5 = certify_features(fv_ladder(weno5_dir), window=window,
                          tolerance_percent=gate_percent)
    w7 = certify_features(fv_ladder(weno7_dir, stem="fv_weno7_cfl0p3_n"),
                          window=window,
                          tolerance_percent=gate_percent)

    # --- G0/G1: which features are certifiable and usable at the gate
    feature_status = {}
    for name in FEATURES:
        c5 = w5["features"][name]
        c7 = w7["features"][name]
        g0 = c5["verdict"] == CERTIFIED and c7["verdict"] == CERTIFIED
        g1 = bool(g0 and c5["band_percent"] is not None
                  and c5["band_percent"] / gate_percent <= U_SHARE_MAX)
        # --- G2 cross-family honesty
        cross = None
        if g0:
            delta = abs(c5["certified_value"] - c7["certified_value"])
            band_sum = c5["band_absolute"] + c7["band_absolute"]
            cross = {"delta": delta,
                     "delta_percent_of_value": 100.0 * delta / abs(c5["certified_value"]),
                     "band_sum_absolute": band_sum,
                     "ratio_to_band_sum": None if band_sum <= 0 else delta / band_sum,
                     "covered": bool(band_sum > 0 and delta <= band_sum)}
        feature_status[name] = {
            "kind": FEATURE_KIND[name],
            "weno5": {k: c5[k] for k in ("verdict", "classification", "values",
                                          "certified_value", "band_absolute",
                                          "band_percent", "observed_order",
                                          "band_share_of_gate", "gate_usable")},
            "weno7": {k: c7[k] for k in ("verdict", "classification", "values",
                                          "certified_value", "band_absolute",
                                          "band_percent", "observed_order")},
            "G0_certifiable": bool(g0),
            "G1_gate_usable": g1,
            "G2_cross_family": cross,
        }

    usable = [n for n in FEATURES if feature_status[n]["G1_gate_usable"]]

    # --- G3: judge every DG leg on every usable feature
    profile_verdicts = round2_profile_verdicts()
    legs = []
    for degree, ncells, cfl in DG_LEGS:
        path = leg_path(ncells, degree, cfl, dg_dir)
        x, rho = load_profile(path)
        f = features(x, rho, window=window)
        per_feature = {}
        for name in usable:
            per_feature[name] = judge_functional(f[name], w5["features"][name],
                                                 tolerance_percent=gate_percent)
        prof = profile_verdicts.get(path.name, {})
        decisions = [d["decision"] for d in per_feature.values()]
        legs.append({
            "configuration": f"DG P{degree} N={ncells} CFL={cfl}",
            "file": path.name,
            "degree": degree, "ncells": ncells, "cfl": cfl,
            "features": f,
            "feature_judgements": per_feature,
            "round2_profile_rel_l1_percent": prof.get("rel_l1_percent"),
            "round2_profile_decision": prof.get("decision"),
            "worst_error_percent": max(
                [d["error_percent"] for d in per_feature.values()], default=None),
            "worst_feature": max(per_feature, key=lambda n: per_feature[n]["error_percent"],
                                 default=None) if per_feature else None,
            "n_feature_pass": decisions.count(PASS),
            "n_feature_marginal": decisions.count(PASS_MARGINAL),
            "n_feature_fail": decisions.count(FAIL),
            "n_feature_withheld": decisions.count(WITHHELD),
        })

    # --- supplementary: features certified by ONE family only.  G0 refuses them
    # as *gated* features (one family's tight band is not evidence), but the
    # numbers are reported so the refusal is auditable and so the amplitude
    # finding can be checked for metric dependence.
    supplementary = {}
    for name in FEATURES:
        if name in usable:
            continue
        fam = None
        for tag, ladder in (("weno5", w5), ("weno7", w7)):
            if ladder["features"][name]["verdict"] == CERTIFIED:
                fam = tag
        if fam is None:
            supplementary[name] = {"certifiable_by": None,
                                   "note": "no family certifies this functional"}
            continue
        cert = (w5 if fam == "weno5" else w7)["features"][name]
        entries = []
        for degree, ncells, cfl in DG_LEGS:
            path = leg_path(ncells, degree, cfl, dg_dir)
            x, rho = load_profile(path)
            f = features(x, rho, window=window)
            j = judge_functional(f[name], cert, tolerance_percent=gate_percent)
            entries.append({"configuration": f"DG P{degree} N={ncells} CFL={cfl}",
                            "error_percent": j["error_percent"],
                            "decision": j["decision"]})
        supplementary[name] = {
            "certifiable_by": fam,
            "certified_value": cert["certified_value"],
            "band_percent": cert["band_percent"],
            "observed_order": cert["observed_order"],
            "informational_only": True,
            "legs": entries,
            "worst_error_percent": max(e["error_percent"] for e in entries),
            "median_error_percent": float(np.median([e["error_percent"] for e in entries])),
            "n_fail": sum(1 for e in entries if e["decision"] == FAIL),
        }

    # --- same-grid cross-scheme check: at equal N, is the DG amplitude deficit a
    # resolution effect or a scheme (limiter) effect?
    same_grid = []
    for name in usable:
        for n in (800, 1600, 3200, 6400):
            row = {"feature": name, "ncells": n,
                   "reference_certified": w5["features"][name]["certified_value"]}
            for tag, directory, stem in (("weno5", weno5_dir, "fv_cfl0p3_n"),
                                         ("weno7", weno7_dir, "fv_weno7_cfl0p3_n")):
                try:
                    x, rho = load_profile(Path(directory) / f"{stem}{n}.npz")
                except FileNotFoundError:
                    continue
                row[f"{tag}_error_percent"] = 100.0 * abs(
                    features(x, rho, window=window)[name]
                    - row["reference_certified"]) / abs(row["reference_certified"])
            for degree, ncells, cfl in DG_LEGS:
                if ncells == n and cfl == 0.4:
                    path = leg_path(ncells, degree, cfl, dg_dir)
                    if not path.exists():
                        continue
                    x, rho = load_profile(path)
                    row[f"dg_p{degree}_error_percent"] = 100.0 * abs(
                        features(x, rho, window=window)[name]
                        - row["reference_certified"]) / abs(row["reference_certified"])
            if len(row) > 3:
                same_grid.append(row)

    # --- G4: counterexamples to "the profile gate is sufficient"
    counterexamples = [
        {"configuration": leg["configuration"], "file": leg["file"],
         "profile_decision": leg["round2_profile_decision"],
         "profile_rel_l1_percent": leg["round2_profile_rel_l1_percent"],
         "failed_features": {n: {"error_percent": d["error_percent"],
                                 "decision": d["decision"]}
                             for n, d in leg["feature_judgements"].items()
                             if d["decision"] == FAIL}}
        for leg in legs
        if leg["round2_profile_decision"] == PASS and leg["n_feature_fail"] > 0
    ]

    # feature-wise summary across legs (dg worst / median)
    summary = {}
    for name in usable:
        errs = [leg["feature_judgements"][name]["error_percent"] for leg in legs]
        summary[name] = {
            "reference": w5["features"][name]["certified_value"],
            "band_percent": w5["features"][name]["band_percent"],
            "worst_error_percent": max(errs),
            "median_error_percent": float(np.median(errs)),
            "n_fail": sum(1 for leg in legs
                          if leg["feature_judgements"][name]["decision"] == FAIL),
            "n_marginal": sum(1 for leg in legs
                              if leg["feature_judgements"][name]["decision"] == PASS_MARGINAL),
        }

    return {
        "audit": "XVAL-DG-ROUND2-CERTIFIED-REFERENCE/feature-level",
        "case": "CASE-SHU-OSHER (Shu & Osher 1988), t=1.8, gamma=1.4",
        "anchor": "XVAL-DG-ROUND2-CERTIFIED-REFERENCE",
        "gate_percent": gate_percent,
        "u_share_max": U_SHARE_MAX,
        "window": {"x_lo": window[0], "x_hi": window[1],
                   "rho_plateau": window[2], "head_inset": HEAD_INSET,
                   "shock_inset": SHOCK_INSET},
        "ladders": {"weno5": w5["ncells"], "weno7": w7["ncells"]},
        "features": feature_status,
        "usable_features": usable,
        "legs": legs,
        "feature_summary": summary,
        "supplementary_single_family": supplementary,
        "same_grid_cross_scheme": same_grid,
        "G4_counterexamples": counterexamples,
        "G4_profile_gate_sufficient": len(counterexamples) == 0,
    }


# ---------------------------------------------------------------- reporting
def markdown(audit: dict) -> str:
    w = audit["window"]
    lines = [
        "# Feature-level certification — CASE-SHU-OSHER round-2 reference",
        "",
        f"Anchor: `{audit['anchor']}` · gate τ = {audit['gate_percent']} % · "
        f"U/τ ≤ {audit['u_share_max']:.3f}",
        "",
        f"Wave-train window: x ∈ [{w['x_lo']:.4f}, {w['x_hi']:.4f}] "
        f"(analytic head +{w['head_inset']}, analytic shock −{w['shock_inset']}), "
        f"plateau ρ = {w['rho_plateau']:.6f}.",
        "",
        "## 1. Functional certification (WENO5 / WENO7 ladders)",
        "",
        "| feature | kind | value (WENO5) | U (%) | p | WENO7 value | Δ family (%) | Δ/(U5+U7) | G0 | G1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, st in audit["features"].items():
        c5, c7 = st["weno5"], st["weno7"]
        cross = st["G2_cross_family"]
        lines.append(
            f"| {name} | {st['kind']} | "
            f"{_fmt(c5['certified_value'])} | {_fmt(c5['band_percent'], 4)} | "
            f"{_fmt(c5['observed_order'], 3)} | {_fmt(c7['certified_value'])} | "
            f"{_fmt(cross['delta_percent_of_value'], 3) if cross else '—'} | "
            f"{_fmt(cross['ratio_to_band_sum'], 2) if cross else '—'} | "
            f"{'yes' if st['G0_certifiable'] else 'NO'} | "
            f"{'yes' if st['G1_gate_usable'] else 'NO'} |")
    lines += [
        "",
        "## 2. DG legs vs the certified features",
        "",
        "| leg | profile L1 % (round 2) | " + " | ".join(
            f"{n} err %" for n in audit["usable_features"]) + " | PASS/MARG/FAIL |",
        "|---" * (len(audit["usable_features"]) + 3) + "|",
    ]
    for leg in audit["legs"]:
        cells = " | ".join(
            f"{leg['feature_judgements'][n]['error_percent']:.3f}"
            if n in leg["feature_judgements"] else "—"
            for n in audit["usable_features"])
        lines.append(
            f"| {leg['configuration']} | {_fmt(leg['round2_profile_rel_l1_percent'], 3)} | "
            f"{cells} | {leg['n_feature_pass']}/{leg['n_feature_marginal']}/"
            f"{leg['n_feature_fail']} |")
    supp = audit.get("supplementary_single_family", {})
    if supp:
        lines += ["", "## 3. Supplementary — functionals certified by one family only "
                      "(G0 refuses them; informational)", "",
                  "| feature | certified by | value | U (%) | p | worst DG err % | median % | FAIL |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, st in supp.items():
            lines.append(f"| {name} | {st.get('certifiable_by')} | "
                         f"{_fmt(st.get('certified_value'))} | "
                         f"{_fmt(st.get('band_percent'), 4)} | "
                         f"{_fmt(st.get('observed_order'), 3)} | "
                         f"{_fmt(st.get('worst_error_percent'), 4)} | "
                         f"{_fmt(st.get('median_error_percent'), 4)} | "
                         f"{st.get('n_fail')} |")
    sg = audit.get("same_grid_cross_scheme", [])
    if sg:
        lines += ["", "## 4. Same-grid cross-scheme comparison", "",
                  "| feature | N | WENO5 err % | WENO7 err % | DG P1 % | DG P2 % | DG P3 % |",
                  "|---|---|---|---|---|---|---|"]
        for row in sg:
            lines.append(
                f"| {row['feature']} | {row['ncells']} | "
                f"{_fmt(row.get('weno5_error_percent'), 4)} | "
                f"{_fmt(row.get('weno7_error_percent'), 4)} | "
                f"{_fmt(row.get('dg_p1_error_percent'), 4)} | "
                f"{_fmt(row.get('dg_p2_error_percent'), 4)} | "
                f"{_fmt(row.get('dg_p3_error_percent'), 4)} |")
    lines += ["", "## 5. Decisions", ""]
    for name, s in audit["feature_summary"].items():
        lines.append(f"- **{name}**: reference {_fmt(s['reference'])}, "
                     f"U = {_fmt(s['band_percent'], 4)} %; worst DG leg "
                     f"{s['worst_error_percent']:.3f} %, median "
                     f"{s['median_error_percent']:.3f} %, FAIL {s['n_fail']}/"
                     f"{len(audit['legs'])}, PASS_MARGINAL {s['n_marginal']}.")
    lines += ["", f"- **G4 profile gate sufficient: "
                  f"{'YES' if audit['G4_profile_gate_sufficient'] else 'NO'}** "
                  f"({len(audit['G4_counterexamples'])} counterexample(s))"]
    for ce in audit["G4_counterexamples"]:
        lines.append(f"  - `{ce['configuration']}`: profile L1 "
                     f"{_fmt(ce['profile_rel_l1_percent'], 3)} % PASS, but "
                     + ", ".join(f"{k} {v['error_percent']:.3f} %"
                                 for k, v in ce["failed_features"].items()) + " FAIL")
    return "\n".join(lines) + "\n"


def _fmt(v, digits: int = 6) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "—"
    return f"{v:.{digits}g}" if abs(v) < 1e3 else f"{v:.1f}"


def figures(audit: dict, outdir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    made = []

    # fig 1: reference profile + DG legs in the wave window, zoomed
    w = audit["window"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x, rho = load_profile(Path(FV_WENO5_DIR) / "fv_cfl0p3_n6400.npz")
    ax.plot(x, rho, "-", lw=1.0, color="0.35", label="FV-WENO5 N=6400")
    for leg in audit["legs"]:
        if leg["n_feature_fail"] > 0:
            lx, lr = load_profile(Path(DG_RAW_DIR) / leg["file"])
            ax.plot(lx, lr, "o", ms=1.6, alpha=0.6, label=leg["configuration"])
            break
    ax.axvline(w["x_lo"], ls=":", color="k", lw=0.8)
    ax.axvline(w["x_hi"], ls=":", color="k", lw=0.8)
    ax.set_xlabel("x"), ax.set_ylabel(r"$\rho$")
    ax.set_title("wave-train window and the worst feature-level leg")
    ax.legend(fontsize=7)
    p1 = outdir / "fig1_wave_window.png"
    fig.tight_layout(), fig.savefig(p1, dpi=140), plt.close(fig)
    made.append(p1)

    # fig 2: per-feature DG error vs the 3 % gate, log bar chart
    names = audit["usable_features"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.8 / len(names)
    idx = np.arange(len(audit["legs"]))
    for k, name in enumerate(names):
        errs = [leg["feature_judgements"][name]["error_percent"] for leg in audit["legs"]]
        ax.bar(idx + k * width, errs, width, label=f"{name} (U="
               f"{audit['features'][name]['weno5']['band_percent']:.3f} %)")
    ax.axhline(audit["gate_percent"], color="r", ls="--", lw=1,
               label=f"gate {audit['gate_percent']} %")
    ax.set_yscale("log"), ax.set_xticks(idx + 0.4 - width / 2)
    ax.set_xticklabels([leg["configuration"].replace("DG ", "")
                        for leg in audit["legs"]], rotation=60, fontsize=6)
    ax.set_ylabel("|error| vs certified feature  [%]")
    ax.set_title("DG legs, feature level (base-10 log)")
    ax.legend(fontsize=6, ncol=2)
    p2 = outdir / "fig2_feature_errors.png"
    fig.tight_layout(), fig.savefig(p2, dpi=140), plt.close(fig)
    made.append(p2)

    # fig 3: profile L1 error vs worst feature error (the G4 claim in one plot)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for leg in audit["legs"]:
        ax.scatter(leg["round2_profile_rel_l1_percent"], leg["worst_error_percent"],
                   s=26, label=leg["configuration"] if leg["n_feature_fail"] else None)
    ax.axhline(audit["gate_percent"], color="r", ls="--", lw=1, label="gate 3 %")
    ax.axvline(audit["gate_percent"], color="r", ls="--", lw=1)
    ax.set_xlabel("round-2 global L1 error [%]")
    ax.set_ylabel("worst feature error [%]")
    ax.set_title("global gate vs feature fidelity")
    ax.legend(fontsize=6)
    p3 = outdir / "fig3_profile_vs_feature.png"
    fig.tight_layout(), fig.savefig(p3, dpi=140), plt.close(fig)
    made.append(p3)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weno5-dir", default=FV_WENO5_DIR)
    ap.add_argument("--weno7-dir", default=FV_WENO7_DIR)
    ap.add_argument("--dg-dir", default=DG_RAW_DIR)
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    report = run_audit(weno5_dir=args.weno5_dir, weno7_dir=args.weno7_dir,
                       dg_dir=args.dg_dir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature_certification_audit.json").write_text(
        json.dumps(report, indent=1, default=float))
    (out / "feature_certification_audit.md").write_text(markdown(report))
    if not args.no_figures:
        for p in figures(report, out):
            print(f"figure: {p}")
    print(f"usable features: {report['usable_features']}")
    print(f"counterexamples (profile PASS but feature FAIL): "
          f"{len(report['G4_counterexamples'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())